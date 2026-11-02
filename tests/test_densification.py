"""Residual-driven insertion: schedule, density, NMS, line search, acceptance, append."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from envmap.field import EnvmapField
from envmap.projection import build_projection
from envmap.quadrature import build_training_quadrature
from envmap.statistics import spherical_channel_integral
from envmap.structure import SphericalStructureField
from fit.allocation import KernelAllocation
from fit.densification import DensificationSchedule, ResidualProposalSeeds, acceptance_metrics, append_selected_asg_batch, build_residual_importance, build_residual_proposal_seeds, build_safeguarded_residual_density, exact_reconstruction_objective, fit_nonnegative_global_batch_scalar, local_width_from_density, select_asg_insertions, streaming_candidate_nms, transformed_underfit_importance
from fit.evaluation import evaluate_directions_batched
from fit.loss import SphericalHDRLoss
from fit.rowwise_adam import RowwiseAdam
from model import build_kernel_bank
from model.asg import render_asg_proposal_aggregate, render_asg_proposal_contributions
from model.serialization import load_native_inference_artifact, save_native_inference_artifact
from tests.helpers import uniform_allocation, asg_bank
from utils.config import training_config_defaults
from utils.provenance import sha256_file


def _args():
    values = training_config_defaults()
    values.update(num_gaussians=1, seed=23)
    return SimpleNamespace(**values)


def _uniform_allocation(projection, height, width, dtype):
    envmap = torch.ones(height, width, 3, dtype=dtype)
    field = EnvmapField(envmap, projection)
    quadrature = build_training_quadrature(1, dtype=dtype)
    directions, solid_angle = quadrature.directions, quadrature.solid_angles
    target_values = field.sample(directions)
    probability = solid_angle / solid_angle.sum()
    density = probability / solid_angle.clamp(min=1e-12)
    local_width = torch.rsqrt(math.pi * density)
    zeros = torch.zeros_like(probability)
    ones = torch.ones_like(probability)
    return KernelAllocation(
        directions=directions,
        solid_angles=solid_angle,
        target_values=target_values,
        probability_flat=probability,
        density_flat=density,
        local_width_flat=local_width,
        structure=SphericalStructureField(
            theta_flat=zeros,
            tangent_axis=torch.zeros_like(directions),
            detail_flat=zeros,
            coherence_flat=zeros,
            anisotropy_ratio_flat=ones,
            energy_flat=zeros,
            score_flat=zeros,
        ),
    )


def _append_candidates(bank, directions, local_width, support_radius, coefficients):
    proposals = bank.initialize_densification_proposals(
        directions, local_width, support_radius, coefficients
    )
    return bank.append_densification_proposals(proposals)


def _reference_continuous_nms(
    directions,
    radii,
    *,
    requested_count,
    candidate_radius_scale,

):
    accepted = []
    rejected_candidate = 0
    for position in range(directions.shape[0]):
        if len(accepted) >= requested_count:
            break
        conflict = False
        for accepted_position in accepted:
            spatial_radius = (
                candidate_radius_scale
                * torch.maximum(radii[position], radii[accepted_position])
            ).clamp(0.0, math.pi)
            if bool(
                (
                    directions[position] * directions[accepted_position]
                ).sum()
                > torch.cos(spatial_radius)
            ):
                conflict = True
                break
        if conflict:
            rejected_candidate += 1
        else:
            accepted.append(position)
    return tuple(accepted), rejected_candidate


class ScheduleTests(unittest.TestCase):
    def test_canonical_screen_budgets_treat_final_count_as_a_cap(self):
        args = SimpleNamespace(
            densify_enabled=True,
            num_gaussians=350,
            densify_interval=1000,
            densify_start_step=1000,
            densify_stop_fraction=0.65,
            densify_add_fraction=0.15,
            densify_direction_chunk_size=8192,
        )
        for budget in (1000, 2500, 5000, 10000):
            with self.subTest(budget=budget):
                schedule = DensificationSchedule.from_args(
                    args, max_steps=10000, final_kernel_budget=budget
                )
                count = schedule.initial_kernel_budget
                event_counts = []
                for step in range(1, 10000):
                    if schedule.should_densify(
                        step,
                        current_kernel_count=count,
                        max_steps=10000,
                    ):
                        addition = schedule.addition_count(current_kernel_count=count)
                        self.assertGreater(addition, 0)
                        count += addition
                        event_counts.append(addition)
                configured = int(math.ceil(budget * args.densify_add_fraction))
                expected = min(
                    budget,
                    schedule.initial_kernel_budget
                    + len(event_counts) * configured,
                )
                self.assertEqual(count, expected)
                self.assertLessEqual(count, budget)
                self.assertTrue(
                    all(value <= configured for value in event_counts)
                )

    def test_rejected_events_do_not_inflate_later_requests(self):
        schedule = DensificationSchedule.from_args(
            SimpleNamespace(
                densify_enabled=True,
                num_gaussians=350,
                densify_interval=1000,
                densify_start_step=1000,
                densify_stop_fraction=0.5,
                densify_add_fraction=0.15,
                densify_direction_chunk_size=8192,
            ),
            max_steps=10000,
            final_kernel_budget=10000,
        )

        requests = [schedule.addition_count(current_kernel_count=350) for _ in range(5)]

        self.assertEqual(requests, [1500] * 5)


class ResidualDensityTests(unittest.TestCase):
    def test_linear_safeguard_is_absolute_target_energy_fraction(self):
        perceptual = torch.tensor([2.0, 1.0], dtype=torch.float64)
        residual_luminance = torch.tensor([4.0, 2.0], dtype=torch.float64)
        target_luminance = torch.tensor([10.0, 20.0], dtype=torch.float64)
        solid_angle = torch.tensor([0.25, 0.75], dtype=torch.float64)
        density = build_safeguarded_residual_density(
            perceptual,
            residual_luminance,
            target_luminance,
            solid_angle,
        )

        expected_residual_mass = solid_angle * residual_luminance
        expected_target_integral = (solid_angle * target_luminance).sum()
        expected_safeguard = expected_residual_mass / expected_target_integral
        self.assertTrue(
            torch.equal(density.linear_residual_mass, expected_residual_mass)
        )
        self.assertEqual(
            density.target_luminance_integral, expected_target_integral
        )
        self.assertTrue(torch.equal(density.safeguard, expected_safeguard))
        self.assertEqual(density.eta, expected_safeguard.sum())

    def test_max_combination_has_required_mass_bounds_and_normalization(self):
        density = build_safeguarded_residual_density(
            torch.tensor([3.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 6.0], dtype=torch.float64),
            torch.tensor([10.0, 10.0], dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )
        q = torch.maximum(density.perceptual_probability, density.safeguard)
        self.assertTrue(torch.equal(density.probability, q / q.sum()))
        self.assertGreaterEqual(float(density.q_sum), 1.0)
        self.assertLessEqual(
            float(density.q_sum), 1.0 + float(density.eta) + 1.0e-12
        )
        self.assertAlmostEqual(float(density.probability.sum()), 1.0, places=12)

    def test_safeguard_converges_to_perceptual_baseline_as_residual_falls(self):
        perceptual = torch.ones(3, dtype=torch.float64)
        target_luminance = torch.ones(3, dtype=torch.float64)
        solid_angle = torch.ones(3, dtype=torch.float64)

        def build(scale):
            return build_safeguarded_residual_density(
                perceptual,
                scale * torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64),
                target_luminance,
                solid_angle,
            )

        strong = build(1.0)
        weaker = build(0.5)
        zero = build(0.0)
        self.assertGreater(float(strong.eta), float(weaker.eta))
        self.assertGreater(float(weaker.eta), float(zero.eta))
        baseline = zero.perceptual_probability
        self.assertTrue(torch.equal(zero.probability, baseline))
        self.assertLess(
            torch.linalg.vector_norm(weaker.probability - baseline),
            torch.linalg.vector_norm(strong.probability - baseline),
        )

    def test_width_density_is_independent_of_safeguard_strength(self):
        perceptual = torch.tensor([1.0, 3.0, 0.5], dtype=torch.float64)
        target_luminance = torch.ones(3, dtype=torch.float64)
        solid_angle = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
        weak = build_safeguarded_residual_density(
            perceptual,
            torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64),
            target_luminance,
            solid_angle,
        )
        strong = build_safeguarded_residual_density(
            perceptual,
            torch.tensor([0.0, 0.0, 100.0], dtype=torch.float64),
            target_luminance,
            solid_angle,
        )
        self.assertFalse(torch.allclose(weak.probability, strong.probability))
        self.assertTrue(
            torch.equal(weak.perceptual_density, strong.perceptual_density)
        )
        weak_width = local_width_from_density(
            weak.perceptual_density, active_count_after=10
        )
        strong_width = local_width_from_density(
            strong.perceptual_density, active_count_after=10
        )
        self.assertTrue(torch.equal(weak_width, strong_width))

    def test_parent_child_refinement_preserves_safeguard_and_width_semantics(self):
        original = build_safeguarded_residual_density(
            torch.tensor([2.0, 1.0]),
            torch.tensor([4.0, 1.0]),
            torch.tensor([6.0, 2.0]),
            torch.tensor([0.2, 0.8]),
        )
        split = build_safeguarded_residual_density(
            torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0]),
            torch.tensor([4.0, 4.0, 4.0, 4.0, 1.0]),
            torch.tensor([6.0, 6.0, 6.0, 6.0, 2.0]),
            torch.tensor([0.05, 0.05, 0.05, 0.05, 0.8]),
        )

        for name in ("probability", "perceptual_probability", "safeguard"):
            original_value = getattr(original, name)[0]
            split_values = getattr(split, name)[:4]
            self.assertAlmostEqual(
                float(original_value), float(split_values.sum()), places=7
            )
        self.assertTrue(
            torch.allclose(split.density[:4], original.density[0].expand(4))
        )
        self.assertTrue(
            torch.allclose(
                split.perceptual_density[:4],
                original.perceptual_density[0].expand(4),
            )
        )
        original_width = local_width_from_density(
            original.perceptual_density, active_count_after=10
        )
        split_width = local_width_from_density(
            split.perceptual_density, active_count_after=10
        )
        self.assertTrue(
            torch.allclose(split_width[:4], original_width[0].expand(4))
        )

    def test_compact_fixed_flux_retains_probability_mass(self):
        for area in (1e-2, 1e-4, 1e-6):
            areas = torch.tensor([1 - area, area], dtype=torch.float64)
            target_luma = torch.tensor([1.0, 1.0 / area], dtype=torch.float64)
            measure = build_safeguarded_residual_density(
                torch.ones_like(areas), 0.5 * target_luma, target_luma, areas
            )
            self.assertGreater(
                float(measure.safeguard[1]), float(measure.perceptual_probability[1])
            )
            probability_mass = areas[1] * measure.density[1]
            self.assertAlmostEqual(
                float(probability_mass), float(measure.probability[1]), places=14
            )
            self.assertGreater(float(probability_mass), 0.19)


class CandidateTests(unittest.TestCase):
    def test_safeguarded_seed_generation_is_scale_equivariant_and_seeded(self):
        quadrature = build_training_quadrature(0, dtype=torch.float64)
        directions = quadrature.directions
        solid_angles = quadrature.solid_angles
        index = torch.arange(directions.shape[0], dtype=torch.float64)
        prediction = torch.stack(
            (0.2 + 0.01 * index, 0.3 + 0.02 * index, 0.1 + 0.01 * index),
            dim=-1,
        )
        target = prediction + torch.stack(
            (1.0 + index, 0.5 + index.remainder(3), 0.25 + index.remainder(5)),
            dim=-1,
        )

        def build(candidate_target, candidate_prediction, scale):
            return build_residual_proposal_seeds(
                directions,
                solid_angles,
                candidate_target,
                candidate_prediction,
                num_new=2,
                current_kernel_count=3,
                optimization_hdr_scale=scale,
                candidate_oversample=4,
                seed=19,
            )

        baseline = build(target, prediction, 2.0)
        repeated = build(target, prediction, 2.0)
        scaled = build(100.0 * target, 100.0 * prediction, 200.0)

        self.assertTrue(torch.equal(repeated.directions, baseline.directions))
        self.assertTrue(torch.equal(scaled.directions, baseline.directions))
        self.assertTrue(torch.allclose(scaled.local_width, baseline.local_width))
        self.assertTrue(
            torch.allclose(
                scaled.radiance_coeff_seed,
                100.0 * baseline.radiance_coeff_seed,
            )
        )

    def test_compact_linear_fixture_enters_the_safeguarded_candidate_pool(self):
        quadrature = build_training_quadrature(0, dtype=torch.float64)
        directions = quadrature.directions[:12]
        solid_angles = torch.ones(12, dtype=torch.float64)
        solid_angles[-1] = 1.0e-6
        prediction = torch.zeros(12, 3, dtype=torch.float64)
        target = torch.ones_like(prediction)
        target[-1] = 1.0e8
        perceptual = transformed_underfit_importance(
            target,
            prediction,
            solid_angles,
            optimization_hdr_scale=1.0,
        )
        perceptual_mass = perceptual * solid_angles
        perceptual_probability = perceptual_mass / perceptual_mass.sum()
        self.assertLess(float(perceptual_probability[-1]), 1.0e-4)

        compact_index = directions.shape[0] - 1
        density, _ = build_residual_importance(target, prediction, solid_angles, optimization_hdr_scale=1.0)
        self.assertGreater(float(density.probability[compact_index]), 0.4)  # safeguard, not perceptual
        sampled = 0
        for seed in range(17, 37):  # the draw depends on the torch RNG stream; check the property
            seeds = build_residual_proposal_seeds(
                directions, solid_angles, target, prediction, num_new=1, current_kernel_count=3,
                optimization_hdr_scale=1.0, candidate_oversample=4, seed=seed,
            )
            if compact_index in seeds.source_sample_indices.tolist():
                sampled += 1
                self.assertTrue(torch.equal(seeds.directions[0], directions[compact_index]))  # ranked first
                self.assertAlmostEqual(float(seeds.local_width[0]), 1.0 / math.sqrt(5.0), places=12)
        self.assertGreaterEqual(sampled, 10)

    def test_safeguarded_sampling_does_not_accept_zero_underfit_candidates(self):
        quadrature = build_training_quadrature(1, dtype=torch.float32)
        directions, solid_angles = quadrature.directions, quadrature.solid_angles
        target = torch.zeros(directions.shape[0], 3)
        prediction = torch.zeros_like(target)
        target[0] = torch.tensor([8.0, 0.0, 0.0])

        candidates = build_residual_proposal_seeds(
            directions,
            solid_angles,
            target,
            prediction,
            num_new=4,
            current_kernel_count=4,
            optimization_hdr_scale=1.0,
            candidate_oversample=100,
            seed=7,
        )

        self.assertEqual(candidates.num_seeds, 1)
        self.assertTrue(torch.equal(candidates.directions[0], directions[0]))

    def test_residual_asg_center_stays_inside_width_relative_trust_region(self):
        projection = build_projection("latlong")
        height, width = 8, 16
        envmap = torch.ones(height, width, 3)
        allocation = _uniform_allocation(projection, height, width, envmap.dtype)
        target_integral = spherical_channel_integral(
            allocation.target_values, allocation.solid_angles
        )
        args = _args()
        args.densify_center_max_width_scale = 0.5
        bank = build_kernel_bank(
            args,
            init_allocation=allocation,
            target_radiance_integral=target_integral,
            feat_dim=3,
            device=envmap.device,
            dtype=envmap.dtype,
        )
        initial_count = bank.num_kernels
        self.assertTrue(
            torch.isinf(bank.residual_center_max_angle[:initial_count]).all()
        )
        initial_anchor = bank.centers()[0].detach().clone()
        reference = torch.tensor([1.0, 0.0, 0.0], dtype=envmap.dtype)
        if float(initial_anchor[0].abs()) > 0.9:
            reference = torch.tensor([0.0, 1.0, 0.0], dtype=envmap.dtype)
        tangent = torch.linalg.cross(initial_anchor, reference)
        tangent = torch.nn.functional.normalize(tangent, dim=0)
        unconstrained_angle = 0.2
        with torch.no_grad():
            bank.lobe_axis[0] = (
                math.cos(unconstrained_angle) * initial_anchor
                + math.sin(unconstrained_angle) * tangent
            )
            bank.project_parameters_()
        initial_distance = torch.acos(
            (bank.centers()[0].detach() * initial_anchor).sum().clamp(-1.0, 1.0)
        )
        self.assertAlmostEqual(float(initial_distance), unconstrained_angle, places=5)
        direction = torch.tensor([[1.0, 0.0, 0.0]], dtype=envmap.dtype)
        local_width = 0.02
        _append_candidates(
            bank,
            direction,
            torch.tensor([local_width]),
            torch.tensor([local_width]),
            torch.ones(1, 3),
        )
        inserted = bank.num_kernels - 1
        anchor = bank.centers()[inserted].detach().clone()
        self.assertTrue(
            torch.isinf(bank.residual_center_max_angle[:initial_count]).all()
        )
        self.assertTrue(torch.isfinite(bank.residual_center_max_angle[inserted]))

        optimizer = torch.optim.Adam(bank.get_optimizer_param_groups())
        distractor_angle = 0.2
        distractor = torch.tensor(
            [math.cos(distractor_angle), math.sin(distractor_angle), 0.0],
            dtype=envmap.dtype,
        )
        for _ in range(20):
            optimizer.zero_grad()
            loss = -(bank.lobe_axis[inserted] * distractor).sum()
            loss.backward()
            optimizer.step()
            bank.project_parameters_()

        distance = torch.acos(
            (bank.centers()[inserted].detach() * anchor).sum().clamp(-1.0, 1.0)
        )
        self.assertGreaterEqual(float(distance), 0.45 * local_width)
        self.assertLessEqual(float(distance), 0.5 * local_width + 1e-5)
        self.assertAlmostEqual(
            float(bank.residual_center_max_angle[inserted]),
            0.5 * local_width,
            places=6,
        )

    def test_residual_width_clamping_matches_initialized_asg_geometry(self):
        quadrature = build_training_quadrature(1, dtype=torch.float32)
        target = torch.ones(quadrature.sample_count, 3)
        prediction = torch.zeros_like(target)
        lambda_min = 5.0
        lambda_max = 1.0e6
        width_min = 1.0 / math.sqrt(lambda_max)
        width_max = 1.0 / math.sqrt(lambda_min)
        seeds_by_case = {}
        for name, current_count in (("above", 1), ("below", 100_000_000)):
            seeds = build_residual_proposal_seeds(
                quadrature.directions,
                quadrature.solid_angles,
                target,
                prediction,
                num_new=1,
                current_kernel_count=current_count,
                optimization_hdr_scale=1.0,
                candidate_oversample=1,
                asg_lambda_min=lambda_min,
                asg_lambda_max=lambda_max,
                seed=7,
            )
            seeds_by_case[name] = seeds

        self.assertAlmostEqual(
            float(seeds_by_case["above"].local_width[0]), width_max, places=6
        )
        self.assertAlmostEqual(
            float(seeds_by_case["below"].local_width[0]), width_min, places=7
        )

        allocation = _uniform_allocation(
            build_projection("latlong"), 8, 16, torch.float32
        )
        bank = build_kernel_bank(
            _args(),
            init_allocation=allocation,
            target_radiance_integral=spherical_channel_integral(
                allocation.target_values, allocation.solid_angles
            ),
            feat_dim=3,
            device=allocation.directions.device,
            dtype=allocation.directions.dtype,
        )
        initial_count = bank.num_kernels
        for seeds in seeds_by_case.values():
            proposal = bank.initialize_densification_proposals(
                seeds.directions,
                seeds.local_width,
                seeds.support_angular_radius,
                seeds.radiance_coeff_seed,
            )
            initialized_width = torch.pow(
                proposal.lambda_ * proposal.mu_bandwidth, -0.25
            )
            self.assertTrue(
                torch.allclose(initialized_width, seeds.local_width, atol=1e-7)
            )
            self.assertTrue(
                torch.allclose(
                    proposal.support_angular_radius,
                    torch.asin(initialized_width),
                    atol=1e-7,
                )
            )
            bank.append_densification_proposals(proposal)

        inserted = slice(initial_count, bank.num_kernels)
        expected_trust = 0.5 * torch.cat(
            [
                seeds_by_case["above"].support_angular_radius,
                seeds_by_case["below"].support_angular_radius,
            ]
        )
        self.assertTrue(
            torch.allclose(
                bank.residual_center_max_angle[inserted],
                expected_trust,
                atol=1e-7,
            )
        )

    def test_shared_asg_proposal_matches_inserted_rows_and_rendering(self):
        projection = build_projection("latlong")
        envmap = torch.ones(8, 16, 3)
        allocation = _uniform_allocation(
            projection, 8, 16, envmap.dtype
        )
        bank = build_kernel_bank(
            _args(),
            init_allocation=allocation,
            target_radiance_integral=spherical_channel_integral(
                allocation.target_values, allocation.solid_angles
            ),
            feat_dim=3,
            device=envmap.device,
            dtype=envmap.dtype,
        )
        directions = allocation.directions[3:5]
        local_width = torch.tensor([0.18, 0.12])
        support_radius = torch.tensor([0.2, 0.13])
        coefficient_seed = torch.tensor(
            [[0.5, 0.75, 1.0], [0.25, 1.25, 0.5]]
        )
        proposal = bank.initialize_densification_proposals(
            directions, local_width, support_radius, coefficient_seed
        )
        queries = allocation.directions[:17]
        before = bank.render_directions(queries)
        expected_difference = render_asg_proposal_contributions(
            queries, proposal
        ).sum(dim=1)
        initial_count = bank.num_kernels
        bank.append_densification_proposals(proposal)
        actual_difference = bank.render_directions(queries) - before

        self.assertTrue(
            torch.allclose(
                expected_difference,
                actual_difference,
                rtol=2e-5,
                atol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                bank.centers()[initial_count:], proposal.z, atol=1e-7
            )
        )
        self.assertTrue(
            torch.allclose(
                bank.log_lambda[initial_count:],
                proposal.log_lambda,
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                bank.log_mu[initial_count:], proposal.log_mu, atol=1e-7
            )
        )
        self.assertTrue(
            torch.allclose(
                bank.radiance_coeff[initial_count:],
                proposal.radiance_coeff,
                rtol=1e-6,
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                bank.residual_center_max_angle[initial_count:],
                0.5 * support_radius,
                atol=1e-7,
            )
        )
        self.assertTrue(torch.allclose(proposal.radiance_coeff, coefficient_seed))
        self.assertTrue(torch.equal(proposal.lambda_, proposal.mu_bandwidth))
        self.assertTrue(torch.equal(proposal.theta, torch.zeros_like(proposal.theta)))

    def test_residual_seed_is_equivalent_cap_area_times_sampled_residual(self):
        allocation = uniform_allocation(subdivision=1)
        prediction = torch.full_like(allocation.target_values, 0.25)
        target = prediction + torch.stack(
            (
                1.0 + allocation.directions[:, 0],
                2.0 + allocation.directions[:, 1],
                3.0 + allocation.directions[:, 2],
            ),
            dim=1,
        ).clamp_min(0.1)
        seeds = build_residual_proposal_seeds(
            allocation.directions,
            allocation.solid_angles,
            target,
            prediction,
            num_new=8,
            current_kernel_count=2,
            optimization_hdr_scale=1.0,
            candidate_oversample=2,
            seed=7,
        )
        source = torch.argmax(
            seeds.directions @ allocation.directions.t(), dim=1
        )
        area = 2.0 * math.pi * (
            1.0 - torch.cos(seeds.support_angular_radius)
        )
        expected = area[:, None] * (target - prediction).clamp_min(0.0)[source]
        self.assertTrue(torch.allclose(seeds.radiance_coeff_seed, expected))


class LineSearchTests(unittest.TestCase):
    def test_global_scalar_line_search_uses_exact_objective(self):
        sample_count = 96
        prediction = torch.full((sample_count, 3), 0.2)
        aggregate = torch.linspace(0.1, 1.0, sample_count).unsqueeze(1).expand(-1, 3)
        target_scalar = 0.25
        target = prediction + target_scalar * aggregate
        loss = SphericalHDRLoss()
        result = fit_nonnegative_global_batch_scalar(
            prediction,
            target,
            torch.ones(sample_count),
            aggregate,
            loss,
            sample_chunk_size=17,
            scalar_chunk_size=2,
        )
        self.assertAlmostEqual(float(result.scalar), target_scalar, delta=0.02)
        unit_objective = loss(prediction + aggregate, target, torch.ones(sample_count))
        self.assertLess(result.objective_value, unit_objective)


class NMSTests(unittest.TestCase):
    def _assert_reference_equivalent(
        self,
        directions,
        radii,
        *,
        requested_count,
        candidate_scale,

    ):
        expected = _reference_continuous_nms(
            directions,
            radii,
            requested_count=requested_count,
            candidate_radius_scale=candidate_scale,

        )
        observed = streaming_candidate_nms(
            directions,
            radii,
            requested_count=requested_count,
            candidate_radius_scale=candidate_scale,

            comparison_chunk_size=7,
        )
        self.assertEqual(observed.accepted_positions, expected[0])
        self.assertEqual(observed.rejected_candidate_nms, expected[1])

    def test_randomized_and_boundary_pools_match_reference(self):
        generator = torch.Generator().manual_seed(19)
        directions = torch.nn.functional.normalize(
            torch.randn(53, 3, generator=generator, dtype=torch.float64),
            dim=1,
        )
        radii = torch.exp(
            torch.empty(53, dtype=torch.float64).uniform_(
                math.log(0.02), math.log(0.5), generator=generator
            )
        )
        self._assert_reference_equivalent(
            directions,
            radii,
            requested_count=31,
            candidate_scale=0.75,

        )

        angles = torch.tensor(
            [0.0, 0.05, 0.05, 0.2, 0.2, 0.55],
            dtype=torch.float64,
        )
        directions = torch.stack(
            (
                torch.cos(angles),
                torch.sin(angles),
                torch.zeros_like(angles),
            ),
            dim=1,
        )
        radii = torch.tensor(
            [0.3, 0.3, 0.03, 0.15, 0.075, 0.4],
            dtype=torch.float64,
        )
        self._assert_reference_equivalent(
            directions,
            radii,
            requested_count=6,
            candidate_scale=0.75,

        )

    def test_same_center_similar_scales_suppress_with_consistent_count(self):
        directions = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        radii = torch.tensor([0.2, 0.15, 0.14], dtype=torch.float64)
        observed = streaming_candidate_nms(
            directions,
            radii,
            requested_count=3,
            candidate_radius_scale=0.75,

        )
        self.assertEqual(observed.accepted_positions, (0,))
        self.assertEqual(observed.rejected_candidate_nms, 2)


class SelectionTests(unittest.TestCase):
    def test_selector_and_trainer_use_normalized_training_quadrature(self):
        allocation = uniform_allocation()
        directions = allocation.directions
        solid_angles = allocation.solid_angles
        normalized = solid_angles * (directions.shape[0] / (4.0 * math.pi))
        target = (
            1.5
            + 0.4 * directions[:, :1]
            + torch.tensor([0.0, 0.2, 0.4], dtype=directions.dtype)
        ).clamp(min=0.0)
        prediction = torch.full_like(target, 0.2)
        base_loss = SphericalHDRLoss(charbonnier_eps=1e-3)

        class RecordingLoss:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.observed_weights = []

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def __call__(self, pred, reference, weights):
                self.observed_weights.append(weights.detach().clone())
                return self.wrapped(pred, reference, weights)

        recording = RecordingLoss(base_loss)
        bank = asg_bank(allocation)
        seeds = build_residual_proposal_seeds(
            directions,
            solid_angles,
            target,
            prediction,
            num_new=2,
            current_kernel_count=bank.num_kernels,
            optimization_hdr_scale=2.0,
            candidate_oversample=3,
            seed=123,
        )
        select_asg_insertions(
            seeds,
            directions,
            solid_angles,
            target,
            prediction,
            num_new=2,
            quadrature_weights=normalized,
            loss_fn=recording,
            kernel_bank=bank,
            step=0,
            candidate_nms_radius_scale=0.0,
            sample_chunk_size=37,
        )
        self.assertTrue(recording.observed_weights)
        self.assertTrue(
            all(
                any(
                    torch.equal(
                        value,
                        normalized[start : start + value.numel()],
                    )
                    for start in range(
                        0, normalized.numel() - value.numel() + 1
                    )
                )
                for value in recording.observed_weights
            )
        )

        expected = float(base_loss(prediction, target, normalized))
        raw_area_value = float(base_loss(prediction, target, solid_angles))
        self.assertTrue(math.isfinite(expected))
        self.assertNotAlmostEqual(expected, raw_area_value, places=6)


class PositiveInsertionTests(unittest.TestCase):
    def fixture(self, dtype=torch.float32):
        allocation = uniform_allocation(subdivision=2, dtype=dtype)
        bank = asg_bank(allocation)
        optimizer = RowwiseAdam(bank.get_optimizer_param_groups())
        for parameter in bank.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
        bank.project_parameters_()
        directions = allocation.directions
        prediction = bank.render_directions_chunked(directions, 2).detach()
        lobe = bank.initialize_densification_proposals(
            directions[:1], directions.new_tensor([0.3]),
            directions.new_tensor([math.asin(0.3)]), directions.new_tensor([[0.5, 0.4, 0.3]]),
        )
        target = prediction + render_asg_proposal_aggregate(directions, lobe)
        weights = allocation.solid_angles * (len(directions) / (4 * math.pi))
        loss = SphericalHDRLoss()
        return allocation, bank, optimizer, prediction, target, weights, loss

    def select(self, fixture, **overrides):
        allocation, bank, _, prediction, target, weights, loss = fixture
        seeds = build_residual_proposal_seeds(
            allocation.directions, allocation.solid_angles, target, prediction,
            num_new=3, current_kernel_count=bank.num_kernels,
            optimization_hdr_scale=1.0, candidate_oversample=3, seed=123,
        )
        return select_asg_insertions(
            seeds, allocation.directions, allocation.solid_angles, target, prediction,
            num_new=3, quadrature_weights=weights, loss_fn=loss, kernel_bank=bank,
            step=1, sample_chunk_size=37, gaussian_chunk_size=2,
            candidate_nms_radius_scale=0.0, **overrides,
        )

    def test_positive_append_preserves_bank_and_adam_and_matches_scored_render(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                fixture = self.fixture(dtype)
                allocation, bank, optimizer, prediction, target, weights, loss = fixture
                before = {name: value.detach().clone() for name, value in bank.state_dict().items()}
                moments = {name: {k: v.clone() for k, v in optimizer.state[p].items()}
                           for name, p in bank.named_parameters()}
                diagnostics = {}
                selected, stats = self.select(fixture, diagnostics=diagnostics)
                self.assertIsNotNone(selected)
                self.assertTrue((selected.proposals.radiance_coeff > 0).all())
                self.assertEqual(stats['selector'], 'positive_residual')
                self.assertLess(selected.total_objective_after, selected.total_objective_before)
                for name, value in bank.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]))
                old_count = bank.num_kernels
                append_selected_asg_batch(bank, optimizer, selected, final_kernel_budget=5)
                self.assertEqual(bank.num_kernels, old_count + selected.count)
                for name, value in bank.state_dict().items():
                    self.assertTrue(torch.equal(value[:old_count], before[name]), name)
                for name, p in bank.named_parameters():
                    state = optimizer.state[p]
                    for key, value in moments[name].items():
                        self.assertTrue(torch.equal(state[key][:old_count], value))
                        self.assertEqual(torch.count_nonzero(state[key][old_count:]), 0)
                committed = evaluate_directions_batched(
                    lambda dirs: bank.render_directions_chunked(dirs, 2),
                    allocation.directions, chunk_size=37, feat_dim=3,
                )
                torch.testing.assert_close(
                    committed, bank.render_directions_chunked(allocation.directions, 2),
                    rtol=32 * torch.finfo(dtype).eps, atol=1e-14,
                )
                torch.testing.assert_close(committed, selected.prediction_after, rtol=0, atol=0)
                envelope = 8 * torch.finfo(dtype).eps * (committed.abs() + prediction.abs())
                self.assertTrue((committed - prediction >= -envelope).all())
                objective = exact_reconstruction_objective(committed, target, weights, loss, sample_chunk_size=37)
                torch.testing.assert_close(objective, selected.reconstruction_objective_after, rtol=0, atol=0)
                torch.testing.assert_close(
                    bank.regularization_loss(1).double() + objective,
                    selected.total_objective_after, rtol=0, atol=0,
                )
                self.assertFalse(any('parent' in key or 'replacement' in key for key in diagnostics))
                bank.zero_grad()
                loss(bank.render_directions(allocation.directions), target, weights).backward()
                optimizer.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in bank.parameters()))

    def test_hard_budget_rejects_before_mutation(self):
        fixture = self.fixture()
        _, bank, optimizer, *_ = fixture
        selected, _ = self.select(fixture)
        previous = dict(bank.named_parameters())
        with self.assertRaisesRegex(ValueError, 'hard kernel budget'):
            append_selected_asg_batch(bank, optimizer, selected, final_kernel_budget=bank.num_kernels)
        self.assertTrue(all(getattr(bank, name) is value for name, value in previous.items()))

    def test_exact_fit_and_overfit_have_no_positive_candidates(self):
        fixture = self.fixture()
        allocation, bank, optimizer, prediction, _, weights, loss = fixture
        for target in (prediction, prediction * 0.5):
            with patch('fit.densification.reconstruction_objective_gradient') as gradient:
                selected, stats = self.select((allocation, bank, optimizer, prediction, target, weights, loss))
            self.assertIsNone(selected)
            self.assertEqual(stats['accepted_count'], 0)
            gradient.assert_not_called()

    def test_selection_is_deterministic_and_rejects_without_mutation(self):
        fixture = self.fixture()
        _, bank, _, *_ = fixture
        first, _ = self.select(fixture)
        second, _ = self.select(fixture)
        self.assertTrue(torch.equal(first.proposals.radiance_coeff_raw, second.proposals.radiance_coeff_raw))
        parameters = dict(bank.named_parameters())
        with patch("fit.densification.acceptance_metrics", return_value=(False, {})):
            rejected, stats = self.select(fixture)
        self.assertIsNone(rejected)
        self.assertEqual(stats['accepted_count'], 0)
        self.assertTrue(all(getattr(bank, name) is value for name, value in parameters.items()))

    def test_shared_render_and_serialization_account_for_appended_rows(self):
        fixture = self.fixture()
        allocation, bank, optimizer, *_ = fixture
        selected, _ = self.select(fixture)
        proposals = selected.proposals
        contribution = render_asg_proposal_contributions(allocation.directions, proposals).sum(1)
        aggregate = render_asg_proposal_aggregate(allocation.directions, proposals, sample_chunk_size=37, proposal_chunk_size=1)
        torch.testing.assert_close(contribution, aggregate)
        append_selected_asg_batch(bank, optimizer, selected, final_kernel_budget=5)
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_native_inference_artifact(directory, kernel_bank=bank)
            self.assertNotIn('logical_encoded_payload', manifest)
            self.assertNotIn('arrays_file_bytes', manifest)
            native = load_native_inference_artifact(Path(directory) / 'inference.json')
            torch.testing.assert_close(native.evaluate_directions(allocation.directions), bank.render_directions(allocation.directions))

    def test_failed_aggregate_recovers_beneficial_insertion_subset(self):
        allocation, bank, optimizer, prediction, _, weights, loss = self.fixture()
        dirs = allocation.directions
        ids = torch.stack((dirs[:, 2].argmax(), dirs[:, 2].argmin()))
        widths = dirs.new_full((2,), 0.3)
        coefficients = dirs.new_tensor([[100., 100., 100.], [0.01, 0.01, 0.01]])
        seeds = ResidualProposalSeeds(
            dirs[ids], widths, torch.asin(widths), coefficients,
            source_sample_indices=ids,
        )
        good = bank.initialize_densification_proposals(
            seeds.directions[1:], widths[1:], seeds.support_angular_radius[1:], coefficients[1:]
        )
        target = prediction.clone()
        target[dirs[:, 2] > 0] *= 0.5
        target += render_asg_proposal_aggregate(dirs, good)
        selected, stats = select_asg_insertions(
            seeds, dirs, allocation.solid_angles, target, prediction, num_new=2,
            quadrature_weights=weights, loss_fn=loss, kernel_bank=bank,
            step=1, sample_chunk_size=37, gaussian_chunk_size=2,
        )
        self.assertEqual(stats['exact_batch_attempts'], 2)
        self.assertEqual(selected.count, 1)
        self.assertEqual(selected.source_sample_indices.tolist(), ids[1:].tolist())
        append_selected_asg_batch(bank, optimizer, selected, final_kernel_budget=3)

    def test_render_roundoff_gain_and_regularization_increase_are_rejected(self):
        allocation, _, _, prediction, target, weights, loss = self.fixture()
        zero = prediction.new_zeros(())
        for proposed, reference, reg_after in (
            (prediction + torch.finfo(prediction.dtype).eps * prediction, target, zero),
            (target, target, prediction.new_tensor(100.)),
        ):
            accepted, _ = acceptance_metrics(
                prediction, proposed, reference, allocation.solid_angles, weights,
                loss, zero, reg_after,
            )
            self.assertFalse(accepted)

    def test_pq_improvement_does_not_impose_a_linear_error_gate(self):
        # Improve two dim channels at the expense of one bright channel.
        pred = torch.tensor([[0.01, 0.01, 100.]])
        target = torch.tensor([[1., 1., 100.]])
        proposed = torch.tensor([[1., 1., 105.]])
        weights, areas = torch.ones(1), torch.full((1,), 4 * math.pi)
        accepted, metrics = acceptance_metrics(
            pred, proposed, target, areas, weights, SphericalHDRLoss(),
            torch.tensor(0.), torch.tensor(0.),
        )
        self.assertTrue(accepted)
        self.assertGreater(metrics['linear_squared_error_after'], metrics['linear_squared_error_before'])

    def test_native_loader_rejects_false_counts_shapes_and_negative_coefficients(self):
        _, bank, *_ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inference.json'
            save_native_inference_artifact(directory, kernel_bank=bank)
            original = json.loads(path.read_text())
            for key, value in [('num_local_kernels', 99), ('feature_dimension', 4)]:
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaisesRegex(ValueError, 'shape'):
                    load_native_inference_artifact(path)
            arrays_path = path.with_name('inference.npz')
            with np.load(arrays_path) as archive:
                arrays = dict(archive)
            arrays['local_coefficients'][0, 0] = -1
            np.savez(arrays_path, **arrays)
            original['arrays_sha256'] = sha256_file(arrays_path)
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                load_native_inference_artifact(path)

    def test_zero_pq_derivative_still_allows_beneficial_finite_insertions(self):
        allocation, bank, optimizer, _, _, weights, loss = self.fixture()
        with torch.no_grad():
            bank.radiance_coeff_raw.fill_(-1000.)
        prediction = bank.render_directions(allocation.directions).detach()
        self.assertEqual(torch.count_nonzero(prediction), 0)
        target = torch.ones_like(prediction)
        self.assertEqual(torch.count_nonzero(loss.prediction_gradient(prediction, target, weights)), 0)
        selected, _ = self.select((allocation, bank, optimizer, prediction, target, weights, loss))
        self.assertIsNotNone(selected)
        self.assertEqual(selected.count, 3)
        self.assertLess(selected.reconstruction_objective_after, selected.reconstruction_objective_before)


class RowwiseAdamTests(unittest.TestCase):
    def test_matches_standard_adam_before_any_append(self):
        initial = torch.tensor(
            [[0.2, -0.3], [1.1, 0.4], [-0.7, 0.8]],
            dtype=torch.float64,
        )
        rowwise_parameter = torch.nn.Parameter(initial.clone())
        reference_parameter = torch.nn.Parameter(initial.clone())
        rowwise = RowwiseAdam([rowwise_parameter], lr=2e-3)
        reference = torch.optim.Adam([reference_parameter], lr=2e-3)

        for step in range(5):
            gradient = torch.arange(6, dtype=torch.float64).reshape(3, 2)
            gradient = 0.01 * gradient + 0.03 * (step + 1)
            rowwise_parameter.grad = gradient.clone()
            reference_parameter.grad = gradient.clone()
            rowwise.step()
            reference.step()

        self.assertTrue(
            torch.equal(rowwise_parameter, reference_parameter)
        )
        state = rowwise.state[rowwise_parameter]
        self.assertTrue(
            torch.equal(
                state["row_age"], torch.full((3,), 5, dtype=torch.int64)
            )
        )

    def test_append_preserves_incumbents_and_freshens_new_rows(self):
        initial = torch.tensor(
            [[0.2, -0.3], [1.1, 0.4], [-0.7, 0.8]],
            dtype=torch.float64,
        )
        rowwise_parameter = torch.nn.Parameter(initial.clone())
        incumbent_reference = torch.nn.Parameter(initial.clone())
        rowwise = RowwiseAdam([rowwise_parameter], lr=2e-3)
        incumbent_adam = torch.optim.Adam(
            [incumbent_reference], lr=2e-3
        )
        for step in range(4):
            gradient = torch.full_like(initial, 0.02 * (step + 1))
            rowwise_parameter.grad = gradient.clone()
            incumbent_reference.grad = gradient.clone()
            rowwise.step()
            incumbent_adam.step()

        old_state = {
            name: value.clone()
            for name, value in rowwise.state[rowwise_parameter].items()
        }
        appended_values = torch.tensor(
            [[0.6, -0.2], [-0.4, 0.9]], dtype=torch.float64
        )
        combined = torch.nn.Parameter(
            torch.cat((rowwise_parameter.detach(), appended_values), dim=0)
        )
        rowwise.append_parameter_rows(rowwise_parameter, combined)
        migrated = rowwise.state[combined]

        self.assertTrue(
            torch.equal(migrated["row_age"][:3], old_state["row_age"])
        )
        self.assertTrue(
            torch.equal(migrated["exp_avg"][:3], old_state["exp_avg"])
        )
        self.assertTrue(
            torch.equal(migrated["exp_avg_sq"][:3], old_state["exp_avg_sq"])
        )
        self.assertTrue(
            torch.equal(
                migrated["row_age"][3:],
                torch.zeros(2, dtype=torch.int64),
            )
        )
        self.assertEqual(
            torch.count_nonzero(migrated["exp_avg"][3:]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(migrated["exp_avg_sq"][3:]).item(), 0
        )

        fresh_reference = torch.nn.Parameter(appended_values.clone())
        fresh_adam = torch.optim.Adam([fresh_reference], lr=2e-3)
        old_gradient = torch.tensor(
            [[0.03, -0.01], [0.07, 0.02], [-0.04, 0.05]],
            dtype=torch.float64,
        )
        new_gradient = torch.tensor(
            [[0.11, -0.08], [0.06, 0.09]], dtype=torch.float64
        )
        combined.grad = torch.cat((old_gradient, new_gradient), dim=0)
        incumbent_reference.grad = old_gradient.clone()
        fresh_reference.grad = new_gradient.clone()
        rowwise.step()
        incumbent_adam.step()
        fresh_adam.step()

        self.assertTrue(torch.equal(combined[:3], incumbent_reference))
        self.assertTrue(torch.equal(combined[3:], fresh_reference))
        self.assertTrue(
            torch.equal(
                migrated["row_age"],
                torch.tensor([5, 5, 5, 1, 1], dtype=torch.int64),
            )
        )


if __name__ == "__main__":
    unittest.main()
