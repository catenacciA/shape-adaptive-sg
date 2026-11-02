import copy
import math
import os
from time import perf_counter

import torch
import torch.nn as nn

from fit.densification import (
    DensificationSchedule,
    build_densification_event,
    build_residual_proposal_seeds,
    append_selected_asg_batch,
    select_asg_insertions,
    write_densification_events,
)
from fit.asg_shape import SphericalCapIndex
from envmap.quadrature import (
    resolve_geodesic_subdivision,
)
from fit.evaluation import evaluate_directions_batched, evaluate_representation
from fit.metrics import radiometric_diagnostics
from fit.problem import (
    build_initial_allocation,
    build_loss_setup,
    build_training_dataset,
    load_target,
)
from fit.rowwise_adam import RowwiseAdam
from codec.native import SKC_METHODS, encode_native
from model.serialization import save_native_inference_artifact, load_native_inference_artifact
from utils.provenance import collect_run_provenance, artifact_reference
from fit.artifacts import format_summary, write_metrics, save_densification_proposals
from fit.kernel_visuals import save_kernel_plot
from model import build_kernel_bank
from utils.config import save_cfg, set_random_seed
from utils.logging import create_worklog, method_name, prepare_run_directories


class EnvmapTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.run_start = perf_counter()
        self.args = args
        self.debug_artifacts = bool(args.debug_artifacts)
        self.debug_dir = os.path.join(args.log_dir, "debug")
        set_random_seed(seed=args.seed)
        self.device = args.device
        self.dtype = torch.float32

        self._init_logging(args)
        self._init_target(args)
        self._init_training_cfg(args)
        self._init_quadrature_cfg(args)
        self._init_training_dataset(args)
        self._init_optimization_target_statistics(args)
        self._init_kernel_allocation(args)
        self._init_kernel_bank(args)
        self._init_optimizer()
        self.step = 0
        self._log_setup()

    def _init_logging(self, args):
        self.log_dir = args.log_dir
        self.log_every = int(args.log_every)

        prepare_run_directories(
            self.log_dir,
            overwrite=bool(args.overwrite_output),
        )

        self.worklog = create_worklog(self.log_dir, args.log_level, debug=self.debug_artifacts)
        save_cfg(path=f"{self.log_dir}/config.yaml", args=args)

    def _init_target(self, args):
        self.field = load_target(args, dtype=self.dtype, device=self.device)
        self.envmap = self.field.values
        self.projection = self.field.projection
        self.feat_dim = self.field.feat_dim
        self.envmap_height = self.field.height
        self.envmap_width = self.field.width
        self.num_pixels = self.field.num_pixels
        self.input_full_path = os.path.join(args.data_root, args.input_path)

    def _init_training_cfg(self, args):
        self.start_step = 1
        self.max_steps = int(args.num_steps)
        self.completed_step = 0
        self.batch_size = int(args.batch_size)
        self.grad_accum_steps = int(args.grad_accum_steps)
        self.grad_clip_norm = float(args.grad_clip_norm)
        self.lr_decay_start = float(args.lr_decay_start)
        self.lr_decay_end = float(args.lr_decay_end)
        self.initial_kernel_budget = int(args.num_gaussians)
        self.final_kernel_budget = int(
            args.densify_max_gaussians
            if bool(args.densify_enabled)
            else args.num_gaussians
        )
        self.requested_hdr_scale = float(args.hdr_scale)
        self.residual_densification_events = []
        self.densification_cap_index = None  # built at the first insertion event
        self._init_densification_cfg(args)

    def _init_optimization_target_statistics(self, args):
        dataset = self.training_dataset
        self._set_optimization_target_statistics(dataset.target)
        # Validate the fixed samples once; training batches skip these checks.
        self.loss_fn._validate_inputs(dataset.target, dataset.target, dataset.quadrature_weights)

    def _set_optimization_target_statistics(self, target, solid_angles=None):
        if solid_angles is None:
            solid_angles = self.training_dataset.solid_angles
        (
            self.training_target_radiance_integral,
            self.optimization_hdr_scale,
            self.loss_fn,
        ) = build_loss_setup(
            self.args, target, solid_angles, self.requested_hdr_scale
        )

    def _init_quadrature_cfg(self, args):
        self.resolved_geodesic_subdivision = resolve_geodesic_subdivision(
            args.geodesic_subdivision, self.envmap_height, self.envmap_width,
        )
        self.evaluation_direction_chunk_size = int(
            args.evaluation_direction_chunk_size
        )

    def _init_densification_cfg(self, args):
        self.densification_schedule = DensificationSchedule.from_args(
            args,
            max_steps=self.max_steps,
            final_kernel_budget=self.final_kernel_budget,
        )

    def _init_kernel_allocation(self, args):
        allocation = build_initial_allocation(
            args,
            field=self.field,
            dataset=self.training_dataset,
            densification_schedule=self.densification_schedule,
        )
        self.init_allocation = allocation

    def _init_training_dataset(self, args):
        self.training_dataset = build_training_dataset(
            args,
            field=self.field,
            subdivision=self.resolved_geodesic_subdivision,
        )

    def _init_kernel_bank(self, args):
        kernel_args = copy.copy(args)
        kernel_args.num_gaussians = self.densification_schedule.initial_kernel_budget
        bank_kwargs = dict(
            init_allocation=self.init_allocation,
            target_radiance_integral=self.training_target_radiance_integral,
            feat_dim=self.feat_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.kernel_bank = build_kernel_bank(kernel_args, **bank_kwargs)
        self._refresh_kernel_accounting()

    def _refresh_kernel_accounting(self):
        self.num_gaussians = int(self.kernel_bank.num_kernels)

    def _init_optimizer(self):
        """Adam with per-group learning rates and per-row bias correction.

        The rates are constant for the first lr_decay_start fraction of the
        steps, then follow a cosine decay to lr_decay_end times their initial
        values. Each update clips the global gradient norm to grad_clip_norm.
        """
        self.optimizer_param_groups = self.kernel_bank.get_optimizer_param_groups()
        self.optimizer_base_lrs = [group["lr"] for group in self.optimizer_param_groups]
        self.optimizer = RowwiseAdam(self.optimizer_param_groups)

        decay_start_step = int(self.max_steps * self.lr_decay_start)
        decay_steps = max(1, self.max_steps - decay_start_step)

        def lr_lambda(step):
            if step < decay_start_step:
                return 1.0
            progress = (step - decay_start_step) / decay_steps
            return self.lr_decay_end + (1.0 - self.lr_decay_end) * 0.5 * (
                1.0 + math.cos(math.pi * min(progress, 1.0))
            )

        self.lr_lambda = lr_lambda
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _log_setup(self):
        """Summarize the fitting problem; the full configuration is in config.yaml."""
        args, quadrature = self.args, self.training_dataset.quadrature
        schedule = self.densification_schedule
        if schedule.enabled:
            primitives = (f"{schedule.initial_kernel_budget:,} -> {schedule.final_kernel_budget:,}, "
                          f"densified every {schedule.interval:,} steps from step "
                          f"{schedule.start_step:,} to {schedule.stop_step:,}")
        elif schedule.disabled_reason == "no_event_window":
            primitives = f"{self.num_gaussians:,} (densification disabled: no event fits in the step budget)"
        else:
            primitives = f"{self.num_gaussians:,} (no densification)"
        for line in (
            f"Fitting {args.input_path} ({self.projection.name} {self.envmap_width}x{self.envmap_height}) "
            f"with method '{method_name(args)}' on {self.device}",
            f"Quadrature: {self.training_dataset.num_samples:,} directions (geodesic level "
            f"{quadrature.base_subdivision}, {quadrature.refined_parent_fraction:.1%} of cells refined)",
            f"Objective: quadrature-weighted PQ Charbonnier loss + concentration regularizer, "
            f"{self.max_steps:,} steps of {self.grad_accum_steps * self.batch_size:,} directions",
            f"Primitives: {primitives}",
        ):
            self.worklog.info(line)

    def evaluate_directions(self, dirs):
        return self.kernel_bank.render_directions_chunked(dirs, int(self.args.renderer_chunk_size))

    def _evaluate_directions_batched(self, dirs, chunk_size):
        return evaluate_directions_batched(
            self.evaluate_directions,
            dirs,
            chunk_size=chunk_size,
            feat_dim=self.feat_dim,
        )

    def _evaluate_training_batch(self, batch):
        return self.evaluate_directions(batch.dirs)

    def _sample_batch(self, num_samples):
        return self.training_dataset.sample_batch(num_samples)

    def _synchronize_for_timing(self):
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _should_densify(self, step):
        return self.densification_schedule.should_densify(
            step,
            current_kernel_count=self.kernel_bank.num_kernels,
            max_steps=self.max_steps,
        )

    def _densify_count(self):
        return self.densification_schedule.addition_count(
            current_kernel_count=self.kernel_bank.num_kernels,
        )

    def _densify_from_spherical_residual(self, step):
        num_new = self._densify_count()
        if num_new <= 0:
            return
        begin = perf_counter()  # the preceding step ended synchronized
        dataset = self.training_dataset
        dirs, target, areas = dataset.dirs, dataset.target, dataset.solid_angles
        chunk = self.densification_schedule.direction_chunk_size
        prediction = self._evaluate_directions_batched(dirs, chunk).clamp_min(0)
        if self.densification_cap_index is None:
            # The fitting directions are fixed: sort them for cap queries once.
            self.densification_cap_index = SphericalCapIndex(dirs)
        event_id = f"densify_step_{int(step)}"
        diagnostics = {} if self.debug_artifacts else None
        with torch.no_grad():
            seeds = build_residual_proposal_seeds(
                dirs,
                areas,
                target,
                prediction,
                num_new=num_new,
                current_kernel_count=self.kernel_bank.num_kernels,
                optimization_hdr_scale=self.optimization_hdr_scale,
                tonemap_exposures=self.args.tonemap_exposures,
                tonemap_gamma=self.args.tonemap_gamma,
                asg_lambda_min=self.kernel_bank.asg_lambda_min,
                asg_lambda_max=self.kernel_bank.asg_lambda_max,
                seed=int(self.args.seed) + int(step),
                min_linear_residual=self.args.densify_min_linear_residual,
                min_relative_residual=self.args.densify_min_relative_residual,
                candidate_oversample=self.args.densify_candidate_oversample,
            )
            selected, stats = select_asg_insertions(
                seeds,
                dirs,
                areas,
                target,
                prediction,
                num_new=num_new,
                quadrature_weights=dataset.quadrature_weights,
                loss_fn=self.loss_fn,
                kernel_bank=self.kernel_bank,
                step=step,
                candidate_nms_radius_scale=self.args.densify_candidate_nms_radius_scale,
                sample_chunk_size=chunk,
                gaussian_chunk_size=int(self.args.renderer_chunk_size),
                diagnostics=diagnostics,
                spatial_index=self.densification_cap_index,
            )
        event = build_densification_event(
            step, num_new, self.kernel_bank.num_kernels, selected, stats
        )
        if self.debug_artifacts:
            event["radiometry_before"] = radiometric_diagnostics(prediction, target, areas, dirs)
        committed_start = None
        if selected is not None:
            committed_start = self.kernel_bank.num_kernels
            append_selected_asg_batch(
                self.kernel_bank, self.optimizer, selected,
                final_kernel_budget=self.final_kernel_budget,
            )
            self._refresh_kernel_accounting()
            event.update(
                status="inserted",
                optimizer_state="incumbents_preserved_new_rows_fresh",
                resulting_count=self.num_gaussians,
            )
        if self.debug_artifacts and selected is not None:
            event["radiometry_after"] = radiometric_diagnostics(selected.prediction_after, target, areas, dirs)
        if diagnostics is not None:
            event["proposal_diagnostics"] = save_densification_proposals(
                self.debug_dir,
                event_id,
                diagnostics,
                committed_start=committed_start,
            )
        self._synchronize_for_timing()
        event["runtime_sec"] = perf_counter() - begin
        self.residual_densification_events.append(event)
        if self.debug_artifacts:
            write_densification_events(self.debug_dir, self.residual_densification_events)
        reason = f" ({event['reason'].replace('_', ' ')})" if event["reason"] else ""
        self.worklog.info(
            f"Densification at step {step}: +{event['accepted_count']:,}/{num_new:,} "
            f"primitives{reason} -> {self.num_gaussians:,} ({event['runtime_sec']:.1f} s)"
        )

    def _compute_loss(self, pred, target, quadrature_weights):
        reconstruction_loss = self.loss_fn(pred, target, quadrature_weights, validate=False)
        return reconstruction_loss + self.kernel_bank.regularization_loss(self.step)

    def _optimization_step(self, step):
        """Advance the sample stream, optimizer and scheduler by one update."""
        self.step = step
        losses, finite = [], []

        for _ in range(self.grad_accum_steps):
            batch = self._sample_batch(self.batch_size)
            pred = self._evaluate_training_batch(batch)
            loss = self._compute_loss(
                pred, batch.target, batch.quadrature_weights
            )
            (loss / self.grad_accum_steps).backward()
            losses.append(loss.detach())
            finite.append(torch.isfinite(pred.detach()).all())

        torch.nn.utils.clip_grad_norm_(
            self.kernel_bank.parameters(), self.grad_clip_norm
        )
        previous_centers = self.kernel_bank.centers().detach().clone()
        self.optimizer.step()
        self.kernel_bank.project_parameters_()
        self.kernel_bank.preserve_tangent_chart_(previous_centers)
        self.optimizer.zero_grad()
        self.scheduler.step()
        # Device reads happen only here, after the update is queued. The
        # loss cannot detect NaN predictions (PQ encodes them like zero), so
        # they are flagged separately. Raising aborts the run before anything
        # is saved, so the discarded update is never used.
        accum_loss = 0.0
        for value in torch.stack(losses).tolist():
            accum_loss += value / self.grad_accum_steps
        if not bool(torch.stack(finite).all()):
            raise ValueError(f"prediction samples contain non-finite values at step {step}")
        self.completed_step = step
        return accum_loss

    def optimize(self):
        """Fit, then export and evaluate the representation.

        Wall-clock trainer phases recorded in metrics.json under
        trainer_runtime_sec, each bounded by a device synchronization so
        queued CUDA work is counted:
          setup: trainer construction to the first optimization step;
          optimization: the step loop (first-call compilation included),
            excluding densification;
          densification: the densification events;
          export_and_evaluation: saving the native and SKC representations,
            evaluation, reconstruction.exr and kernels.png;
          total: their sum, from trainer construction to the final results.
        This is not process wall time: Python startup, imports and train.py's
        pre-flight checks precede the trainer and are excluded.
        """
        self.step = 0
        self.completed_step = 0
        self.optimizer.zero_grad()
        self._synchronize_for_timing()
        training_start = perf_counter()
        width = len(str(self.max_steps))

        for step in range(self.start_step, self.max_steps + 1):
            # Each step ends by reading its loss on the host, so the clock
            # reads below see completed device work without extra syncs.
            loss = self._optimization_step(step)
            should_densify = self._should_densify(step)
            if step % self.log_every == 0 or step == self.max_steps:
                self.worklog.info(
                    f"Step {step:>{width}d}/{self.max_steps} | loss {loss:.4g} | "
                    f"{self.num_gaussians:,} primitives | elapsed {perf_counter() - self.run_start:.1f} s"
                )
            if should_densify:
                self._densify_from_spherical_residual(step)

        if self.completed_step != self.max_steps:
            raise RuntimeError(
                "optimization did not complete the configured step budget: "
                f"{self.completed_step} vs {self.max_steps}"
            )
        self._synchronize_for_timing()
        training_end = perf_counter()
        with torch.no_grad():
            save_native_inference_artifact(
                self.log_dir, kernel_bank=self.kernel_bank,
                provenance=collect_run_provenance(self.args, input_path=self.input_full_path),
            )
            binding = artifact_reference(os.path.join(self.log_dir, "inference.json"))
            representation = load_native_inference_artifact(
                os.path.join(self.log_dir, "inference.json"), device=self.device, dtype=self.dtype,
            )
            codec = self._save_bitstream(representation)
            self.evaluation_context, evaluation = evaluate_representation(
                representation, self.field, device=self.device,
                subdivision=self.args.evaluation_subdivision,
                chunk_size=self.evaluation_direction_chunk_size,
                image_path=os.path.join(self.log_dir, "reconstruction.exr"),
                latlong_reference=self.args.latlong_reference,
            )
            save_kernel_plot(
                os.path.join(self.log_dir, "kernels.png"),
                field=self.field, representation=representation,
                hdr_scale=self.evaluation_context.hdr_scale,
            )
            self._synchronize_for_timing()
            end = perf_counter()
            densification = sum(event["runtime_sec"] for event in self.residual_densification_events)
            runtime = {
                "setup": training_start - self.run_start,
                "optimization": training_end - training_start - densification,
                "densification": densification,
                "export_and_evaluation": end - training_end,
                "total": end - self.run_start,
            }
            self.final_metric_data = write_metrics(self.log_dir, {
                "artifact": binding,
                **evaluation,
                "completed_steps": self.completed_step,
                "num_kernels": self.num_gaussians,
                "trainer_runtime_sec": {name: round(value, 3) for name, value in runtime.items()},
                **({"codec": codec} if codec else {}),
            })

        for line in format_summary(self.final_metric_data).splitlines():
            self.worklog.info(line)
        self.worklog.info(
            "Trainer runtime: " + " | ".join(f"{name.replace('_and_', ' + ')} {value:.1f} s"
                                     for name, value in runtime.items())
        )
        self.worklog.info(f"Results in {self.log_dir}")
        return None

    def _save_bitstream(self, representation):
        """Write the fixed-rate SKC bitstream of the fit; None for variants SKC cannot code."""
        method = SKC_METHODS.get(method_name(self.args))
        if method is None:
            self.worklog.info("No SKC code for this variant; representation.skc not written")
            return None
        data = encode_native(representation, method=method)
        with open(os.path.join(self.log_dir, "representation.skc"), "wb") as handle:
            handle.write(data)
        return {"file": "representation.skc", "bytes": len(data)}
