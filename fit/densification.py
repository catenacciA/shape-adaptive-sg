import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from envmap.statistics import (
    compute_luminance,
    robust_unit_component,
    sanitize_nonnegative,
    weighted_quantile,
)
from fit.allocation import local_width_from_density
from fit.evaluation import evaluate_directions_batched
from fit.rowwise_adam import RowwiseAdam
from fit.asg_shape import SphericalCapIndex, estimate_proposal_shapes
from model.asg import (
    ASGProposalParameters,
    asg_kernel_weights,
    clamp_asg_local_width,
    replace_asg_proposal_coefficients,
    replace_asg_proposal_geometry,
    render_asg_proposal_aggregate,
)

RESIDUAL_PERCEPTUAL_CLIP_QUANTILE = 0.995
DENSIFICATION_EVENTS_SCHEMA_VERSION = 15


@dataclass(frozen=True)
class DensificationSchedule:
    enabled: bool
    initial_kernel_budget: int
    final_kernel_budget: int
    interval: int
    start_step: int
    stop_step: int
    add_fraction: float
    direction_chunk_size: int
    disabled_reason: str | None = None

    @classmethod
    def from_args(cls, args, *, max_steps, final_kernel_budget):
        final_kernel_budget = int(final_kernel_budget)
        initial_kernel_budget = int(args.num_gaussians)
        max_steps = int(max_steps)
        if not bool(args.densify_enabled):
            return cls(
                enabled=False,
                initial_kernel_budget=initial_kernel_budget,
                final_kernel_budget=final_kernel_budget,
                interval=0,
                start_step=0,
                stop_step=0,
                add_fraction=0.0,
                direction_chunk_size=0,
            )

        interval = int(args.densify_interval)
        start_step = int(args.densify_start_step)
        stop_fraction = float(args.densify_stop_fraction)
        stop_step = int(round(max_steps * stop_fraction))
        if max_steps > 1:
            stop_step = min(max(stop_step, 1), max_steps - 1)
        else:
            stop_step = 0
        add_fraction = float(args.densify_add_fraction)
        direction_chunk_size = int(args.densify_direction_chunk_size)
        schedule_stop = min(stop_step, max_steps - 1)
        has_event_window = (
            max_steps > 1
            and initial_kernel_budget < final_kernel_budget
            and start_step <= schedule_stop
        )
        if not has_event_window:
            return cls(
                enabled=False,
                initial_kernel_budget=initial_kernel_budget,
                final_kernel_budget=final_kernel_budget,
                interval=0,
                start_step=0,
                stop_step=0,
                add_fraction=add_fraction,
                direction_chunk_size=direction_chunk_size,
                disabled_reason="no_event_window",
            )
        return cls(
            enabled=True,
            initial_kernel_budget=initial_kernel_budget,
            final_kernel_budget=final_kernel_budget,
            interval=interval,
            start_step=start_step,
            stop_step=stop_step,
            add_fraction=add_fraction,
            direction_chunk_size=direction_chunk_size,
        )

    def should_densify(self, step, *, current_kernel_count, max_steps):
        if not self.enabled:
            return False
        if step >= int(max_steps):
            return False
        if int(current_kernel_count) >= self.final_kernel_budget:
            return False
        if step < self.start_step or step > self.stop_step:
            return False
        return (step - self.start_step) % self.interval == 0

    def addition_count(self, *, current_kernel_count):
        remaining = self.final_kernel_budget - int(current_kernel_count)
        if remaining <= 0:
            return 0
        configured = int(
            math.ceil(self.final_kernel_budget * self.add_fraction)
        )
        return min(remaining, max(configured, 0))


@dataclass(frozen=True)
class SafeguardedResidualDensity:
    """Perceptual proposal measure with an absolute linear-energy safeguard."""

    probability: torch.Tensor
    density: torch.Tensor
    perceptual_probability: torch.Tensor
    perceptual_density: torch.Tensor
    perceptual_mass: torch.Tensor
    linear_residual_mass: torch.Tensor
    safeguard: torch.Tensor
    perceptual_total_mass: torch.Tensor
    linear_residual_luminance_integral: torch.Tensor
    target_luminance_integral: torch.Tensor
    eta: torch.Tensor
    q_sum: torch.Tensor
    skipped_reason: str | None = None


@dataclass(frozen=True)
class ResidualProposalSeeds:
    """Residual allocation geometry and nominal integrated RGB seeds."""

    directions: torch.Tensor
    local_width: torch.Tensor
    support_angular_radius: torch.Tensor
    radiance_coeff_seed: torch.Tensor
    skipped_reason: str | None = None
    source_sample_indices: torch.Tensor | None = None

    @property
    def num_seeds(self):
        return int(self.directions.shape[0])

    def index_select(self, indices):
        indices = torch.as_tensor(
            indices, device=self.directions.device, dtype=torch.long
        )
        return ResidualProposalSeeds(
            directions=self.directions.index_select(0, indices),
            local_width=self.local_width.index_select(0, indices),
            support_angular_radius=self.support_angular_radius.index_select(0, indices),
            radiance_coeff_seed=self.radiance_coeff_seed.index_select(0, indices),
            skipped_reason=self.skipped_reason,
            source_sample_indices=(
                None
                if self.source_sample_indices is None
                else self.source_sample_indices.index_select(0, indices)
            ),
        )


@dataclass(frozen=True)
class SelectedASGBatch:
    """Positive append batch accepted on the complete fitting quadrature."""

    proposals: ASGProposalParameters
    prediction_after: torch.Tensor
    reconstruction_objective_before: torch.Tensor
    reconstruction_objective_after: torch.Tensor
    total_objective_before: torch.Tensor
    total_objective_after: torch.Tensor
    acceptance_tolerance: float
    source_sample_indices: torch.Tensor

    @property
    def count(self):
        return self.proposals.num_proposals


@dataclass(frozen=True)
class GlobalScalarFitResult:
    scalar: torch.Tensor
    objective_value: torch.Tensor


@dataclass(frozen=True)
class StreamingNMSResult:
    accepted_positions: tuple[int, ...]
    rejected_candidate_nms: int


def streaming_candidate_nms(
    directions,
    angular_radii,
    *,
    requested_count,
    candidate_radius_scale,
    comparison_chunk_size=512,
):
    """Greedy spherical NMS with bounded accepted-set comparisons.

    Candidate order is the input order. Workspace is bounded by
    comparison_chunk_size and never depends on the candidate-pool square.
    """
    directions = torch.as_tensor(directions)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("NMS directions must have shape [N, 3]")
    angular_radii = torch.as_tensor(
        angular_radii, device=directions.device, dtype=directions.dtype
    ).reshape(-1)
    count = int(directions.shape[0])
    if angular_radii.shape != (count,):
        raise ValueError("angular radii must match NMS directions")
    requested_count = max(int(requested_count), 0)
    chunk = max(int(comparison_chunk_size), 1)
    accepted = []
    rejected_candidate = 0
    for position in range(count):
        if len(accepted) >= requested_count:
            break
        conflicting_position = None
        for start in range(0, len(accepted), chunk):
            accepted_chunk = accepted[start : start + chunk]
            accepted_indices = torch.tensor(
                accepted_chunk, device=directions.device, dtype=torch.long
            )
            dots = (
                directions.index_select(0, accepted_indices)
                @ directions[position]
            ).clamp(-1.0, 1.0)
            spatial = (
                float(candidate_radius_scale)
                * torch.maximum(
                    angular_radii[position],
                    angular_radii.index_select(0, accepted_indices),
                )
            ).clamp(0.0, math.pi)
            conflicts = dots > torch.cos(spatial)
            conflict_indices = torch.nonzero(
                conflicts, as_tuple=False
            ).reshape(-1)
            if conflict_indices.numel() > 0:
                conflicting_position = accepted_chunk[
                    int(conflict_indices[0].item())
                ]
                break
        if conflicting_position is not None:
            rejected_candidate += 1
            continue
        accepted.append(position)
    return StreamingNMSResult(
        accepted_positions=tuple(accepted),
        rejected_candidate_nms=rejected_candidate,
    )


def build_densification_event(
    step, requested_count, current_count, selected, selection_stats
):
    """Build the stable event record before the trainer mutates its kernel bank."""
    actual_count = 0 if selected is None else selected.count
    rejection_reason = selection_stats.get("final_rejection_reason")
    objective_before = None
    objective_after = None
    objective_gain = None
    tolerance = selection_stats.get("acceptance_tolerance")
    if selected is not None:
        objective_before = float(selected.total_objective_before.item())
        objective_after = float(selected.total_objective_after.item())
        objective_gain = objective_before - objective_after
        tolerance = float(selected.acceptance_tolerance)
    return {
        "event_id": f"densify_step_{int(step)}",
        "step": int(step),
        "requested_count": int(requested_count),
        "proposed_count": int(selection_stats.get("proposed_count", actual_count)),
        "accepted_count": int(actual_count),
        "resulting_count": int(current_count),
        "reason": rejection_reason,
        "rejected_candidate_nms": int(
            selection_stats.get("rejected_candidate_nms", 0)
        ),
        "objective_before": objective_before,
        "objective_after": objective_after,
        "objective_gain": objective_gain,
        "effective_tolerance": tolerance,
        "optimizer_state": "unchanged",
        "status": "candidate_shortfall",
        "selection": dict(selection_stats),
    }


def write_densification_events(log_dir, events):
    path = Path(log_dir) / "densification_events.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"schema_version": DENSIFICATION_EVENTS_SCHEMA_VERSION, "events": events},
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")


def _weighted_upper_clip(values, solid_angles, quantile):
    """Apply the residual sampling density's final weighted upper clip."""
    values = sanitize_nonnegative(values)
    positive = values > 0.0
    if not bool(positive.any().item()):
        return torch.zeros_like(values)
    clip_value = weighted_quantile(
        values[positive],
        solid_angles.reshape(-1)[positive],
        quantile,
    ).clamp(min=1e-12)
    return values.clamp(max=clip_value)


def _parse_exposures(value):
    if value is None:
        return (-2.0, 0.0, 2.0)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(float(part) for part in parts)
    if isinstance(value, (list, tuple)):
        return tuple(float(part) for part in value)
    return (float(value),)


def _tonemap(values, ev, hdr_scale, gamma):
    """Exposed Reinhard tone map followed by display gamma."""
    exposed = values.clamp(min=0.0) * (2.0 ** float(ev)) / float(hdr_scale)
    mapped = exposed / (1.0 + exposed)
    return mapped.clamp(min=0.0).pow(1.0 / max(float(gamma), 1e-6))


def transformed_underfit_importance(
    target,
    prediction,
    solid_angles,
    optimization_hdr_scale,
    clip_quantile=RESIDUAL_PERCEPTUAL_CLIP_QUANTILE,
    log_error_weight=0.65,
    tonemap_error_weight=0.35,
    tonemap_exposures=(-2.0, 0.0, 2.0),
    tonemap_gamma=2.2,
):
    """Perceptual underfit score from positive log and tone-mapped deficits."""
    target = torch.as_tensor(target).clamp(min=0.0)
    prediction = torch.as_tensor(
        prediction, device=target.device, dtype=target.dtype
    ).clamp(min=0.0)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("perceptual residual inputs must match as [N, C]")
    optimization_hdr_scale = float(optimization_hdr_scale)
    if (
        not math.isfinite(optimization_hdr_scale)
        or optimization_hdr_scale <= 0.0
    ):
        raise ValueError(
            "optimization HDR scale must be finite and positive for densification"
        )
    solid_angles = torch.as_tensor(
        solid_angles, device=target.device, dtype=target.dtype
    ).reshape(-1)
    if solid_angles.shape != (target.shape[0],):
        raise ValueError("perceptual residual solid angles must match the samples")

    log_underfit = (
        torch.log1p(target / optimization_hdr_scale)
        - torch.log1p(prediction / optimization_hdr_scale)
    ).clamp(min=0.0).mean(dim=-1)
    log_component = robust_unit_component(
        log_underfit, solid_angles, clip_quantile
    )

    tonemap_components = []
    for ev in _parse_exposures(tonemap_exposures):
        target_tm = _tonemap(target, ev, optimization_hdr_scale, tonemap_gamma)
        pred_tm = _tonemap(prediction, ev, optimization_hdr_scale, tonemap_gamma)
        tonemap_components.append(
            (target_tm - pred_tm).clamp(min=0.0).mean(dim=-1)
        )
    tonemap_underfit = torch.stack(tonemap_components, dim=0).mean(dim=0)
    tonemap_component = robust_unit_component(
        tonemap_underfit, solid_angles, clip_quantile
    )

    log_error_weight = max(float(log_error_weight), 0.0)
    tonemap_error_weight = max(float(tonemap_error_weight), 0.0)
    total_weight = log_error_weight + tonemap_error_weight
    if total_weight <= 0.0:
        return log_component
    return (
        log_error_weight * log_component
        + tonemap_error_weight * tonemap_component
    ) / total_weight


def build_safeguarded_residual_density(
    perceptual_importance,
    positive_residual_luminance,
    target_luminance,
    solid_angles,
):
    """Build the normalized perceptual measure plus absolute linear safeguard."""
    perceptual_importance = sanitize_nonnegative(
        torch.as_tensor(perceptual_importance).reshape(-1)
    )
    positive_residual_luminance = sanitize_nonnegative(
        torch.as_tensor(
            positive_residual_luminance,
            device=perceptual_importance.device,
            dtype=perceptual_importance.dtype,
        ).reshape(-1)
    )
    target_luminance = sanitize_nonnegative(
        torch.as_tensor(
            target_luminance,
            device=perceptual_importance.device,
            dtype=perceptual_importance.dtype,
        ).reshape(-1)
    )
    solid_angles = torch.as_tensor(
        solid_angles,
        device=perceptual_importance.device,
        dtype=perceptual_importance.dtype,
    ).reshape(-1)
    if not (
        perceptual_importance.shape
        == positive_residual_luminance.shape
        == target_luminance.shape
        == solid_angles.shape
    ):
        raise ValueError(
            "residual proposal importance, luminance, and solid angles must match"
        )
    solid_angles = torch.where(
        torch.isfinite(solid_angles),
        solid_angles,
        torch.zeros_like(solid_angles),
    ).clamp(min=0.0)
    zero = torch.zeros_like(perceptual_importance)
    scalar_zero = perceptual_importance.new_zeros(())
    if not bool((solid_angles > 0.0).any().item()):
        return SafeguardedResidualDensity(
            probability=zero,
            density=zero,
            perceptual_probability=zero,
            perceptual_density=zero,
            perceptual_mass=zero,
            linear_residual_mass=zero,
            safeguard=zero,
            perceptual_total_mass=scalar_zero,
            linear_residual_luminance_integral=scalar_zero,
            target_luminance_integral=scalar_zero,
            eta=scalar_zero,
            q_sum=scalar_zero,
            skipped_reason="zero_solid_angle",
        )

    perceptual_mass = perceptual_importance * solid_angles
    linear_residual_mass = positive_residual_luminance * solid_angles
    target_luminance_mass = target_luminance * solid_angles
    perceptual_total = perceptual_mass.sum()
    linear_residual_total = linear_residual_mass.sum()
    target_luminance_integral = target_luminance_mass.sum()
    perceptual_probability = (
        perceptual_mass / perceptual_total
        if bool((perceptual_total > 0.0).item())
        else zero
    )
    safeguard = (
        linear_residual_mass / target_luminance_integral
        if bool((target_luminance_integral > 0.0).item())
        else zero
    )
    eta = safeguard.sum()
    q = torch.maximum(perceptual_probability, safeguard)
    q_sum = q.sum()
    skipped_reason = (
        None
        if bool((q_sum > 0.0).item())
        else "zero_proposal_probability"
    )

    probability = (
        q / q_sum
        if skipped_reason is None
        else zero
    )
    minimum_area = torch.finfo(solid_angles.dtype).tiny
    density = probability / solid_angles.clamp(min=minimum_area)
    perceptual_density = perceptual_probability / solid_angles.clamp(
        min=minimum_area
    )
    return SafeguardedResidualDensity(
        probability=probability,
        density=density,
        perceptual_probability=perceptual_probability,
        perceptual_density=perceptual_density,
        perceptual_mass=perceptual_mass,
        linear_residual_mass=linear_residual_mass,
        safeguard=safeguard,
        perceptual_total_mass=perceptual_total,
        linear_residual_luminance_integral=linear_residual_total,
        target_luminance_integral=target_luminance_integral,
        eta=eta,
        q_sum=q_sum,
        skipped_reason=skipped_reason,
    )


def _linear_underfit_mask(
    positive_residual,
    target,
    min_linear_residual,
    min_relative_residual,
):
    residual_max = positive_residual.max(dim=-1).values
    target_max = target.clamp(min=0.0).max(dim=-1).values
    relative = residual_max / target_max.clamp(min=1e-12)
    return (
        (residual_max >= float(min_linear_residual))
        & (relative >= float(min_relative_residual))
    )


def _sample_candidate_indices(probability, count, seed=None):
    probability_cpu = probability.detach().to(device="cpu", dtype=torch.float64)
    nonzero = int((probability_cpu > 0.0).sum().item())
    count = min(max(int(count), 0), nonzero)
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    return torch.multinomial(
        probability_cpu,
        count,
        replacement=False,
        generator=generator,
    )


def _objective_values_for_batch_scalars(
    prediction,
    target,
    quadrature_weights,
    aggregate_contribution,
    scalars,
    loss_fn,
    *,
    sample_chunk_size=8192,
    scalar_chunk_size=4,
):
    """Evaluate exact objectives with bounded sample and scalar workspaces.

    The largest candidate-prediction tensor is
    [sample_chunk_size, scalar_chunk_size, channels].  In particular, this
    function never constructs [num_samples, num_scalars, channels].
    """
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(
        target, device=prediction.device, dtype=prediction.dtype
    )
    weights = torch.as_tensor(
        quadrature_weights, device=prediction.device, dtype=prediction.dtype
    ).reshape(-1)
    aggregate_contribution = torch.as_tensor(
        aggregate_contribution,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    scalars = torch.as_tensor(
        scalars, device=prediction.device, dtype=prediction.dtype
    ).reshape(-1)
    if aggregate_contribution.shape != prediction.shape:
        raise ValueError(
            "aggregate contribution must match prediction [samples, channels]"
        )
    if target.shape != prediction.shape or weights.shape != (prediction.shape[0],):
        raise ValueError("target and quadrature weights must match prediction")
    if not bool(torch.isfinite(scalars).all().item()) or bool(
        (scalars < 0.0).any().item()
    ):
        raise ValueError("batch scalars must be finite and non-negative")
    values = torch.zeros(
        scalars.numel(), device=prediction.device, dtype=torch.float64
    )
    sample_chunk = max(int(sample_chunk_size), 1)
    scalar_chunk = max(int(scalar_chunk_size), 1)
    for scalar_start in range(0, scalars.numel(), scalar_chunk):
        scalar_end = min(scalar_start + scalar_chunk, scalars.numel())
        scalar_batch = scalars[scalar_start:scalar_end]
        batch_values = values.new_zeros(scalar_end - scalar_start)
        for sample_start in range(0, prediction.shape[0], sample_chunk):
            sample_end = min(sample_start + sample_chunk, prediction.shape[0])
            candidate_prediction = (
                prediction[sample_start:sample_end, None, :]
                + aggregate_contribution[sample_start:sample_end, None, :]
                * scalar_batch[None, :, None]
            ).clamp_min(0.0)
            per_sample = loss_fn.per_sample(
                candidate_prediction.to(torch.float64),
                target[sample_start:sample_end, None, :].to(torch.float64),
            )
            batch_values += (
                per_sample
                * weights[sample_start:sample_end, None].to(torch.float64)
            ).sum(dim=0)
        values[scalar_start:scalar_end] = batch_values
    return values / prediction.shape[0]


def exact_reconstruction_objective(
    prediction,
    target,
    quadrature_weights,
    loss_fn,
    *,
    sample_chunk_size=8192,
):
    """Evaluate the reconstruction objective with float64 loss accumulation."""
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(
        target, device=prediction.device, dtype=prediction.dtype
    )
    weights = torch.as_tensor(
        quadrature_weights, device=prediction.device
    ).reshape(-1)
    total = torch.zeros((), device=prediction.device, dtype=torch.float64)
    chunk = max(int(sample_chunk_size), 1)
    for start in range(0, prediction.shape[0], chunk):
        end = min(start + chunk, prediction.shape[0])
        chunk_value = loss_fn(
            prediction[start:end].to(torch.float64),
            target[start:end].to(torch.float64),
            weights[start:end],
        )
        total += chunk_value.to(torch.float64) * (end - start)
    return total / prediction.shape[0]


def objective_gain_tolerance(objective_scale, repeatability, *, multiplier):
    """Scale a gain tolerance from measured repeatability and float64 ULPs."""
    scale = abs(float(objective_scale))
    repeatability = abs(float(repeatability))
    eps64 = torch.finfo(torch.float64).eps
    ulp_scale = eps64 * max(scale, torch.finfo(torch.float64).tiny)
    return max(float(multiplier) * repeatability, float(multiplier) * ulp_scale)


def fit_nonnegative_global_batch_scalar(
    prediction,
    target,
    quadrature_weights,
    aggregate_contribution,
    loss_fn,
    *,
    max_expansions=6,
    max_downward_expansions=12,
    refinement_points=9,
    sample_chunk_size=8192,
    scalar_chunk_size=4,
):
    """Fit one nonnegative scalar for an aggregate proposal contribution.

    The search is a deterministic bounded grid followed by local refinement.
    Every point uses the configured reconstruction objective, accumulated in
    float64 by :func:`_objective_values_for_batch_scalars`.
    """
    device = prediction.device
    dtype = prediction.dtype
    downward = [
        math.ldexp(1.0, -exponent)
        for exponent in range(max(int(max_downward_expansions), 0), 0, -1)
    ]
    upward = [
        math.ldexp(1.0, exponent)
        for exponent in range(0, max(int(max_expansions), 0) + 1)
    ]
    coarse_grid = torch.tensor(
        [0.0, *downward, *upward], device=device, dtype=dtype
    )
    coarse_objectives = _objective_values_for_batch_scalars(
        prediction,
        target,
        quadrature_weights,
        aggregate_contribution,
        coarse_grid,
        loss_fn,
        sample_chunk_size=sample_chunk_size,
        scalar_chunk_size=scalar_chunk_size,
    )
    coarse_best_index = int(torch.argmin(coarse_objectives).item())
    coarse_best_scalar = coarse_grid[coarse_best_index]
    coarse_best_objective = coarse_objectives[coarse_best_index]
    lower_index = max(coarse_best_index - 1, 0)
    upper_index = min(coarse_best_index + 1, coarse_grid.numel() - 1)
    lower = coarse_grid[lower_index]
    upper = coarse_grid[upper_index]
    fraction = torch.linspace(
        0.0,
        1.0,
        max(int(refinement_points), 2),
        device=device,
        dtype=dtype,
    )
    if float(lower.item()) > 0.0:
        refined_grid = torch.exp(
            torch.log(lower)
            + (torch.log(upper) - torch.log(lower)) * fraction
        )
    else:
        refined_grid = lower + (upper - lower) * fraction
    refined_objectives = _objective_values_for_batch_scalars(
        prediction,
        target,
        quadrature_weights,
        aggregate_contribution,
        refined_grid,
        loss_fn,
        sample_chunk_size=sample_chunk_size,
        scalar_chunk_size=scalar_chunk_size,
    )
    refined_best_index = int(torch.argmin(refined_objectives).item())
    refined_scalar = refined_grid[refined_best_index]
    refined_objective = refined_objectives[refined_best_index]
    if bool(refined_objective < coarse_best_objective):
        scalar = refined_scalar
        objective_value = refined_objective
    else:
        scalar = coarse_best_scalar
        objective_value = coarse_best_objective
    return GlobalScalarFitResult(
        scalar=scalar.detach(),
        objective_value=objective_value.detach(),
    )


def build_residual_importance(
    target,
    prediction,
    solid_angles,
    *,
    optimization_hdr_scale,
    tonemap_exposures=(-2.0, 0.0, 2.0),
    tonemap_gamma=2.2,
) -> tuple[SafeguardedResidualDensity, torch.Tensor]:
    """Return normalized allocation importance and positive physical RGB residual.

    The density field is per unit solid angle. Candidate validity filtering
    belongs to proposal selection, after this shared measure is constructed.
    """
    positive_residual = (target - prediction).clamp(min=0.0)
    perceptual_importance = _weighted_upper_clip(
        transformed_underfit_importance(
            target,
            prediction,
            solid_angles,
            optimization_hdr_scale=optimization_hdr_scale,
            clip_quantile=RESIDUAL_PERCEPTUAL_CLIP_QUANTILE,
            tonemap_exposures=tonemap_exposures,
            tonemap_gamma=tonemap_gamma,
        ),
        solid_angles,
        RESIDUAL_PERCEPTUAL_CLIP_QUANTILE,
    )
    positive_residual_luminance = compute_luminance(positive_residual)
    target_luminance = compute_luminance(target.clamp(min=0.0))
    density = build_safeguarded_residual_density(
        perceptual_importance,
        positive_residual_luminance,
        target_luminance,
        solid_angles,
    )
    return density, positive_residual


def build_residual_proposal_seeds(
    directions,
    solid_angles,
    target,
    prediction,
    *,
    num_new,
    current_kernel_count,
    optimization_hdr_scale,
    min_linear_residual=1e-6,
    min_relative_residual=1e-4,
    candidate_oversample=4,
    tonemap_exposures=(-2.0, 0.0, 2.0),
    tonemap_gamma=2.2,
    asg_lambda_min=5.0,
    asg_lambda_max=1.0e6,
    seed=None,
):
    """Generate safeguarded residual geometry and integral RGB seeds."""
    density, positive_residual = build_residual_importance(
        target,
        prediction,
        solid_angles,
        optimization_hdr_scale=optimization_hdr_scale,
        tonemap_exposures=tonemap_exposures,
        tonemap_gamma=tonemap_gamma,
    )
    valid_underfit = _linear_underfit_mask(
        positive_residual,
        target,
        min_linear_residual=min_linear_residual,
        min_relative_residual=min_relative_residual,
    )
    empty_indices = torch.empty(0, device=directions.device, dtype=torch.long)
    empty_values = solid_angles.new_empty((0,))
    skipped_reason = density.skipped_reason
    all_widths = empty_values
    sampled = empty_indices
    pool_count = max(
        int(num_new),
        int(math.ceil(int(num_new) * float(candidate_oversample))),
    )
    if skipped_reason is None and not bool(valid_underfit.any().item()):
        skipped_reason = "linear_underfit_below_threshold"
    if skipped_reason is None:
        all_widths = clamp_asg_local_width(
            local_width_from_density(
                density.perceptual_density, int(current_kernel_count) + int(num_new)
            ),
            asg_lambda_min, asg_lambda_max,
        )
        # The safeguard controls discovery and local ranking only. Width remains
        # tied to perceptual density, independent of absolute radiometric energy.
        residual_score = torch.where(
            valid_underfit, density.density, torch.zeros_like(density.density)
        )
        eligible_probability = torch.where(
            valid_underfit, density.probability, torch.zeros_like(density.probability)
        )
        eligible_total = eligible_probability.sum()
        if bool((eligible_total > 0.0).item()):
            eligible_probability = eligible_probability / eligible_total
            sampled = _sample_candidate_indices(
                eligible_probability, pool_count, seed=seed
            ).to(device=directions.device)
            sampled = sampled[
                torch.argsort(residual_score[sampled], descending=True, stable=True)
            ]
        else:
            skipped_reason = "zero_eligible_proposal_probability"
    widths = all_widths[sampled] if sampled.numel() else empty_values
    radii = torch.asin(widths.clamp(min=0.0, max=1.0))
    equivalent_cap_area = 2.0 * math.pi * (1.0 - torch.cos(radii))
    radiance_coeff_seed = (
        equivalent_cap_area[:, None] * positive_residual[sampled]
    )
    return ResidualProposalSeeds(
        directions=directions[sampled],
        local_width=widths,
        support_angular_radius=radii,
        radiance_coeff_seed=radiance_coeff_seed,
        skipped_reason=skipped_reason,
        source_sample_indices=sampled,
    )


def exact_linear_squared_error(prediction, target, solid_angles, *, sample_chunk_size=8192):
    total = prediction.new_zeros((), dtype=torch.float64)
    for start in range(0, len(prediction), sample_chunk_size):
        error = (
            prediction[start : start + sample_chunk_size].double()
            - target[start : start + sample_chunk_size].double()
        )
        total += (
            solid_angles[start : start + sample_chunk_size].double() * error.square().sum(-1)
        ).sum()
    return total


@torch.no_grad()
def reconstruction_objective_gradient(prediction, target, weights, loss_fn, chunk_size=8192):
    """Derivative of the configured finite-quadrature reconstruction objective."""
    result = torch.empty_like(prediction)
    for start in range(0, len(prediction), chunk_size):
        end = min(start + chunk_size, len(prediction))
        gradient = loss_fn.prediction_gradient(
            prediction[start:end].double(), target[start:end].double(), weights[start:end]
        )
        result[start:end] = (gradient * ((end - start) / len(prediction))).to(prediction.dtype)
    return result


def _shape_proposals(bank, seeds, directions, areas, positive_residual, spatial_index=None):
    proposals = bank.initialize_densification_proposals(
        seeds.directions,
        seeds.local_width,
        seeds.support_angular_radius,
        seeds.radiance_coeff_seed,
    )
    estimates = estimate_proposal_shapes(
        proposals, directions, areas, compute_luminance(positive_residual),
        spatial_index=spatial_index,
    )
    proposals, _ = replace_asg_proposal_geometry(
        proposals,
        estimates.precision,
        estimates.eligible,
        lambda_min=bank.asg_lambda_min,
        lambda_max=bank.asg_lambda_max,
        width_ratio_max=bank.width_ratio_max,
    )
    return proposals


def calibrate_insertion_batch(
    proposals, directions, prediction, target, weights, loss_fn, chunk_size,
):
    """Fit one nonnegative scalar to the complete positive proposal field."""
    aggregate = render_asg_proposal_aggregate(
        directions, proposals, sample_chunk_size=chunk_size
    )
    fit = fit_nonnegative_global_batch_scalar(
        prediction, target, weights, aggregate, loss_fn,
        sample_chunk_size=chunk_size,
    )
    coefficients = proposals.radiance_coeff * fit.scalar
    if float(fit.scalar) <= 0 or not bool(torch.isfinite(coefficients).all()):
        return None
    # Acceptance renders these rounded raw coefficients through the bank renderer.
    return replace_asg_proposal_coefficients(proposals, coefficients)


def acceptance_metrics(
    prediction,
    proposed,
    target,
    areas,
    weights,
    loss_fn,
    reg_before,
    reg_after,
    *,
    chunk_size=8192,
):
    """Gate prospective appends on reconstruction and total loss, including render precision."""
    before = exact_reconstruction_objective(
        prediction, target, weights, loss_fn, sample_chunk_size=chunk_size
    )
    after = exact_reconstruction_objective(
        proposed, target, weights, loss_fn, sample_chunk_size=chunk_size
    )
    error = 8 * torch.finfo(prediction.dtype).eps * (prediction.abs() + proposed.abs())
    # Charbonnier is 1-Lipschitz in transformed radiance. Transform the forward
    # error envelope, rather than using loss-summation ULPs as render tolerance.
    tolerance = prediction.new_zeros((), dtype=torch.float64)
    for start in range(0, len(prediction), chunk_size):
        end = start + chunk_size
        for value in (prediction[start:end], proposed[start:end]):
            value = value.double().clamp_min(0)
            radius = error[start:end].double()
            center = loss_fn.transform_radiance(value)
            upper = (loss_fn.transform_radiance(value + radius) - center).abs()
            lower = (loss_fn.transform_radiance((value - radius).clamp_min(0)) - center).abs()
            tolerance += (
                torch.maximum(upper, lower).mean(-1) * weights[start:end].double()
            ).sum() / len(prediction)
    tolerance = max(float(tolerance), objective_gain_tolerance(float(before), 0, multiplier=64.0))
    total_before, total_after = before + reg_before.double(), after + reg_after.double()
    reg_tolerance = (
        8 * torch.finfo(prediction.dtype).eps * float(reg_before.abs() + reg_after.abs())
    )
    linear_before = exact_linear_squared_error(
        prediction, target, areas, sample_chunk_size=chunk_size
    )
    linear_after = exact_linear_squared_error(proposed, target, areas, sample_chunk_size=chunk_size)
    accepted = bool(
        before - after > tolerance
        and total_before - total_after > tolerance + reg_tolerance
    )
    return accepted, {
        "reconstruction_before": float(before),
        "reconstruction_after": float(after),
        "total_before": float(total_before),
        "total_after": float(total_after),
        "linear_squared_error_before": float(linear_before),
        "linear_squared_error_after": float(linear_after),
        "acceptance_tolerance": tolerance,
    }


def proposal_gradient_products(
    directions, gradient, proposals, chunk_size=8192, proposal_chunk_size=4096,
):
    """Objective directional derivatives; their negatives are first-order insertion gains.

    For proposal k with weights W[:, k] and coefficients a_k, the derivative is
    sum_j W[j, k] (a_k . g_j), evaluated as ((W^T G) * a).sum(-1).
    """
    footprint = proposals.footprint()
    result = directions.new_zeros(proposals.num_proposals, dtype=torch.float64)
    for row in range(0, proposals.num_proposals, proposal_chunk_size):
        rows = slice(row, row + proposal_chunk_size)
        projected = 0
        for start in range(0, len(directions), chunk_size):
            weights = asg_kernel_weights(
                F.normalize(directions[start:start + chunk_size], dim=1),
                proposals.z[rows], proposals.x[rows], proposals.y[rows],
                proposals.lambda_[rows], proposals.mu_bandwidth[rows],
                None if footprint is None else footprint[rows],
            )
            chunk_gradient = gradient[start:start + chunk_size].double()
            projected = projected + weights.t().double() @ chunk_gradient
        result[rows] = (projected * proposals.radiance_coeff[rows].double()).sum(-1)
    return result


def _render_prospective_bank(bank, directions, sample_chunk, gaussian_chunk, proposals=None):
    """Use exactly the committed raw parameters, row order and renderer."""
    parameters = bank.current_parameters(proposals)
    return evaluate_directions_batched(
        lambda chunk: bank.render_directions_from_parameters(
            F.normalize(chunk, dim=1), *parameters, chunk_size=gaussian_chunk
        ),
        directions, chunk_size=sample_chunk, feat_dim=bank.feat_dim,
    )


@torch.no_grad()
def select_asg_insertions(
    seeds, directions, solid_angles, target, prediction, *, num_new,
    quadrature_weights, loss_fn, kernel_bank, step,
    candidate_nms_radius_scale=0.75, sample_chunk_size=8192,
    gaussian_chunk_size=4000, diagnostics=None, spatial_index=None,
):
    """Score and calibrate a positive batch, then verify its canonical append render.

    Try the ordinary residual batch first. If it fails, retry its non-adverse
    objective-slope subset once; no incumbent row is removed or edited.
    spatial_index, a SphericalCapIndex of directions, may be reused across calls.
    """
    bank, weights = kernel_bank, quadrature_weights
    empty = torch.empty(0, device=directions.device, dtype=torch.long)
    stats = dict(selector="positive_residual", requested_maximum=num_new,
                 proposed_count=0, accepted_count=0, exact_batch_attempts=0)
    if diagnostics is not None:
        diagnostics.update(candidate_sample_indices=seeds.source_sample_indices,
                           selected_sample_indices=empty)
    if num_new <= 0 or seeds.num_seeds == 0:
        stats["final_rejection_reason"] = (
            "zero_requested_count" if num_new <= 0
            else seeds.skipped_reason or "empty_candidate_pool"
        )
        return None, stats
    gradient = reconstruction_objective_gradient(
        prediction, target, weights, loss_fn, sample_chunk_size
    )
    reg_before = bank.regularization_loss(step)
    nms = streaming_candidate_nms(
        seeds.directions.cpu(), seeds.support_angular_radius.cpu(), requested_count=num_new,
        candidate_radius_scale=candidate_nms_radius_scale,
    )
    stats["rejected_candidate_nms"] = nms.rejected_candidate_nms
    insertion_seeds = seeds.index_select(nms.accepted_positions)
    insertions = _shape_proposals(
        bank, insertion_seeds, directions, solid_angles, (target - prediction).clamp_min(0),
        SphericalCapIndex(directions) if spatial_index is None else spatial_index,
    )
    gains = -proposal_gradient_products(directions, gradient, insertions, sample_chunk_size)
    order = torch.argsort(gains, descending=True, stable=True)
    insertions = insertions.index_select(order)
    sources = insertion_seeds.source_sample_indices[order]
    descent = torch.where(gains[order] >= 0)[0]
    stats["proposed_count"] = insertions.num_proposals
    attempts = [("full_batch", torch.arange(insertions.num_proposals, device=directions.device))]
    if 0 < len(descent) < insertions.num_proposals:
        attempts.append(("descent_subset", descent))
    for label, available in attempts:
        stats["exact_batch_attempts"] += 1
        rows = calibrate_insertion_batch(
            insertions.index_select(available), directions, prediction, target,
            weights, loss_fn, sample_chunk_size,
        )
        if rows is None:
            stats[f"{label}_status"] = "no_calibrated_batch"
            continue
        proposed = _render_prospective_bank(
            bank, directions, sample_chunk_size, gaussian_chunk_size, rows,
        )
        reg = bank.prospective_regularization_loss(rows, step)
        accepted, metrics = acceptance_metrics(
            prediction, proposed, target, solid_angles, weights, loss_fn, reg_before, reg,
            chunk_size=sample_chunk_size,
        )
        stats.update(metrics)
        stats[f"{label}_status"] = "accepted" if accepted else "rejected"
        if not accepted:
            continue
        bank.validate_densification_proposals(rows)
        selected = SelectedASGBatch(
            rows, proposed,
            *[prediction.new_tensor(metrics[key], dtype=torch.float64) for key in (
                "reconstruction_before", "reconstruction_after", "total_before", "total_after"
            )],
            metrics["acceptance_tolerance"], sources[available],
        )
        stats.update(accepted_count=selected.count, final_rejection_reason=None)
        if diagnostics is not None:
            diagnostics.update(
                selected_sample_indices=selected.source_sample_indices,
                centers=rows.z, final_lambda=rows.lambda_, final_mu=rows.mu_bandwidth,
                final_theta=rows.theta, calibrated_coefficients=rows.radiance_coeff,
            )
        return selected, stats
    stats["final_rejection_reason"] = "no_beneficial_joint_batch"
    return None, stats


@torch.no_grad()
def append_selected_asg_batch(kernel_bank, optimizer, selected, *, final_kernel_budget):
    """Append an already scored batch and extend Adam history with fresh rows.

    Selection and append are consecutive: the trainer must not optimize the bank
    between them. Numerical acceptance uses the canonical prospective renderer
    before mutation, so no transaction or post-append rollback is needed.
    """
    if not isinstance(selected, SelectedASGBatch) or not isinstance(optimizer, RowwiseAdam):
        raise TypeError("append requires SelectedASGBatch and RowwiseAdam")
    if selected.count <= 0 or kernel_bank.num_kernels + selected.count > final_kernel_budget:
        raise ValueError("selected ASG insertions exceed the hard kernel budget")
    previous = dict(kernel_bank.named_parameters(recurse=False))
    kernel_bank.append_densification_proposals(selected.proposals)
    for name, parameter in previous.items():
        optimizer.append_parameter_rows(parameter, getattr(kernel_bank, name))
