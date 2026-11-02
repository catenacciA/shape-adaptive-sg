"""ASG and shape-adaptive kernel: normalization, gradients, rendering, initialization."""
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from envmap.field import EnvmapField
from envmap.projection import build_projection
from envmap.quadrature import build_training_quadrature
from envmap.statistics import spherical_channel_integral, spherical_channel_mean
from envmap.structure import SphericalStructureField
from fit.allocation import KernelAllocation, build_kernel_allocation
from fit.densification import append_selected_asg_batch, build_residual_proposal_seeds, select_asg_insertions
from fit.kernel_visuals import asg_footprint_contours
from fit.loss import SphericalHDRLoss
from fit.rowwise_adam import RowwiseAdam
from model import build_kernel_bank
from model.asg import (
    ASGFootprint, ASGKernelBank, asg_integral_norm, asg_kernel_weights, render_asg_proposal_aggregate, render_asg_proposal_contributions, replace_asg_proposal_geometry, superelliptic_exponent,
)
from model.primitives import (
    COEFFICIENT_ACTIVATION_FLOOR, build_asg_tangent_frame, initialize_radiance_coefficients,
)
from model.serialization import load_native_inference_artifact, save_native_inference_artifact
from tests.helpers import uniform_allocation, asg_bank, superelliptic_bank
from utils.config import resolve_training_config, training_config_defaults


def footprint(p, q, dtype=torch.float64):
    return ASGFootprint.from_exponents(torch.as_tensor(p, dtype=dtype), torch.as_tensor(q, dtype=dtype))


def plane_density(u, v, p, q):
    shape = ASGFootprint.from_exponents(p, q)
    return (shape.log_c - superelliptic_exponent(u, v, shape)).exp()


def _args(footprint="gaussian", count=16):
    values = training_config_defaults()
    values.update(asg_footprint=footprint, num_gaussians=count, seed=19)
    return SimpleNamespace(**values)


def _uniform_allocation(
    projection,
    height,
    width,
    num_kernels,
    device,
    dtype,
    envmap=None,
):
    if envmap is None:
        envmap = torch.ones(height, width, 3, device=device, dtype=dtype)
    field = EnvmapField(envmap, projection)
    quadrature = build_training_quadrature(2, device=device, dtype=dtype)
    directions, solid_angle = quadrature.directions, quadrature.solid_angles
    target_values = field.sample(directions)
    probability = solid_angle / solid_angle.sum()
    density = probability / solid_angle.clamp(min=1e-12)
    local_width = torch.rsqrt(math.pi * num_kernels * density)
    zeros = torch.zeros_like(probability)
    ones = torch.ones_like(probability)
    structure = SphericalStructureField(
        theta_flat=zeros,
        tangent_axis=torch.zeros_like(directions),
        detail_flat=zeros,
        coherence_flat=zeros,
        anisotropy_ratio_flat=ones,
        energy_flat=zeros,
        score_flat=zeros,
    )
    return KernelAllocation(
        directions=directions,
        solid_angles=solid_angle,
        target_values=target_values,
        probability_flat=probability,
        density_flat=density,
        local_width_flat=local_width,
        structure=structure,
    )


def _target_integral(allocation):
    return spherical_channel_integral(
        allocation.target_values, allocation.solid_angles
    )


def _args_asg_initialization():
    values = training_config_defaults()
    values.update(
        num_gaussians=1,
        lr_theta=1.25e-2,
        asg_lambda_min=1.0,
        seed=11,
    )
    return SimpleNamespace(**values)


def _single_allocation(direction, local_width, theta=0.0, ratio=1.0):
    directions = torch.as_tensor(direction, dtype=torch.float32).reshape(1, 3)
    directions = torch.nn.functional.normalize(directions, dim=1)
    probability = torch.ones(1, dtype=torch.float32)
    solid_angles = torch.full((1,), 4.0 * math.pi, dtype=torch.float32)
    target_values = torch.ones(1, 3, dtype=torch.float32)
    structure = SphericalStructureField(
        theta_flat=torch.tensor([float(theta)], dtype=torch.float32),
        tangent_axis=torch.zeros_like(directions),
        detail_flat=torch.tensor([1.0 if ratio > 1.0 else 0.0], dtype=torch.float32),
        coherence_flat=torch.tensor([1.0 if ratio > 1.0 else 0.0], dtype=torch.float32),
        anisotropy_ratio_flat=torch.tensor([float(ratio)], dtype=torch.float32),
        energy_flat=torch.ones(1, dtype=torch.float32),
        score_flat=torch.ones(1, dtype=torch.float32),
    )
    density = probability / solid_angles
    return KernelAllocation(
        directions=directions,
        solid_angles=solid_angles,
        target_values=target_values,
        probability_flat=probability,
        density_flat=density,
        local_width_flat=torch.full_like(probability, float(local_width)),
        structure=structure,
    )


def _make_bank(theta=0.0, ratio=1.0):
    envmap = torch.ones(8, 16, 3)
    footprint = math.sqrt(4.0 * math.pi / (8 * 16))
    sigma_init = max(0.05 * footprint, min(0.5, 2.0 * footprint))
    allocation = _single_allocation(
        [1.0, 0.0, 0.0], sigma_init, theta=theta, ratio=ratio
    )
    return ASGKernelBank(
        _args_asg_initialization(),
        init_allocation=allocation,
        target_radiance_integral=spherical_channel_integral(
            allocation.target_values, allocation.solid_angles
        ),
        feat_dim=3,
        device=envmap.device,
        dtype=envmap.dtype,
    )


def _width_ratio(bank, index=-1):
    lambda_, mu = bank.concentrations()
    return float(torch.sqrt(mu[index] / lambda_[index]).detach())


def _cell_directions(projection, height, width, dtype=torch.float32):
    y = torch.arange(height, dtype=dtype)
    x = torch.arange(width, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return projection.pixel_to_dir(grid_x + 0.5, grid_y + 0.5, width, height)


def _analytic_envmap(projection, height, width):
    directions = _cell_directions(projection, height, width)
    axis = torch.tensor([0.8, -0.3, 0.5])
    axis = axis / axis.norm()
    bump = torch.exp(32.0 * ((directions * axis).sum(dim=-1) - 1.0))
    detail = torch.sin(9.0 * directions[..., 0] + 7.0 * directions[..., 1]).square()
    value = 0.08 + 0.35 * detail + 8.0 * bump
    return torch.stack((value, 0.6 * value + 0.04, 0.3 * value + 0.08), dim=-1)


def _allocation_kernel_allocation(field, count, subdivision=4):
    quadrature = build_training_quadrature(
        subdivision, device=field.device, dtype=field.dtype
    )
    directions, solid_angles = quadrature.directions, quadrature.solid_angles
    target_values = field.sample(directions)
    return build_kernel_allocation(
        field,
        directions,
        solid_angles,
        count,
        target_values=target_values,
        anisotropy_ratio_max=4.0,
    )


class ShapeAdaptiveKernelMathTests(unittest.TestCase):
    def test_gaussian_member_values_and_geometry_gradients(self):
        for dtype in (torch.float32, torch.float64):
            gen = torch.Generator().manual_seed(71)
            dirs = F.normalize(torch.randn(37, 3, generator=gen, dtype=dtype), dim=-1)
            axis = torch.randn(3, 3, generator=gen, dtype=dtype, requires_grad=True)
            theta = torch.tensor([0.2, 0.7, -0.1], dtype=dtype, requires_grad=True)
            log_lam = torch.tensor([1.7, 2.5, 4.0], dtype=dtype, requires_grad=True)
            log_mu = torch.tensor([2.5, 3.3, 2.7], dtype=dtype, requires_grad=True)
            x, y, z = build_asg_tangent_frame(axis, theta)
            args = (dirs, z, x, y, log_lam.exp(), log_mu.exp())
            old = asg_kernel_weights(*args)
            new = asg_kernel_weights(*args, footprint(torch.ones(3), torch.full((3,), 2.), dtype))
            tol = 3e-5 if dtype == torch.float32 else 2e-12
            torch.testing.assert_close(new, old, atol=tol, rtol=tol)
            parameters = (axis, theta, log_lam, log_mu)
            old_grad = torch.autograd.grad(old.sum(), parameters, retain_graph=True)
            new_grad = torch.autograd.grad(new.sum(), parameters)
            for actual, expected in zip(new_grad, old_grad):
                torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    def test_shape_scores_at_gaussian_member(self):
        u, v = torch.tensor(.8, dtype=torch.float64), torch.tensor(-.3, dtype=torch.float64)
        p, q = (torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (1., 2.))
        dp, dq = torch.autograd.grad(plane_density(u, v, p, q).log(), (p, q))
        r2 = u.square() + v.square()
        t = u.square() / r2
        entropy = -t * t.log() - (1 - t) * (1 - t).log()
        torch.testing.assert_close(dp, -1 - r2 * (r2.log() + 0.5772156649015329 - 2))
        torch.testing.assert_close(dq, r2 * (entropy / 2 - math.log(2) + .5))
        self.assertGreater(abs(float(dp)), .01)
        self.assertGreater(abs(float(dq)), .001)

    def test_gradcheck_axes_diagonal_and_center(self):
        for uv in ((.7, -.3), (0., .8), (.4, .4), (0., 0.)):
            args = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                         for value in (*uv, 1.7, 4.3))
            self.assertTrue(torch.autograd.gradcheck(plane_density, args, atol=1e-6, rtol=1e-4))

    def test_extreme_bounds_remain_finite_in_forward_and_backward(self):
        for dtype in (torch.float32, torch.float64):
            p = torch.tensor([1., 4., 1., 4.], dtype=dtype, requires_grad=True)
            q = torch.tensor([1.25, 1.25, 16., 16.], dtype=dtype, requires_grad=True)
            u = torch.tensor([[0., 0., 0., 0.], [.5, -.5, .5, -.5],
                              [1000., -1000., 0., 1000.], [1e-20] * 4],
                             dtype=dtype, requires_grad=True)
            v = torch.tensor([[0.] * 4, [.5] * 4, [1000.] * 4, [0.] * 4],
                             dtype=dtype, requires_grad=True)
            values = plane_density(u, v, p, q)
            grads = torch.autograd.grad(values.sum(), (u, v, p, q))
            self.assertTrue(bool((values >= 0).all()))
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in (values, *grads)))
            torch.testing.assert_close(grads[0][0], torch.zeros(4, dtype=dtype))
            torch.testing.assert_close(grads[1][0], torch.zeros(4, dtype=dtype))

    def test_plane_integrals_and_covariance_by_independent_quadrature(self):
        # Tensor product polar quadrature, directly integrating the density:
        # no reuse of its analytic radial integral or moment formulas.
        nodes, weights = np.polynomial.legendre.leggauss(180)
        radius = torch.from_numpy((nodes + 1) * 4)
        dr = torch.from_numpy(weights * 4)
        angle = (torch.arange(1024, dtype=torch.float64) + .5) * (2 * math.pi / 1024)
        u, v = radius[:, None] * angle.cos(), radius[:, None] * angle.sin()
        measure = (radius * dr)[:, None] * (2 * math.pi / 1024)
        for p, q in ((1, 2), (1, 1.25), (1, 16), (2, 4), (4, 16)):
            shape = footprint(p, q)
            density = (shape.log_c - superelliptic_exponent(u, v, shape)).exp()
            mass = density * measure
            self.assertAlmostEqual(float(mass.sum()), 1, delta=2e-5)
            for coord in (u, v):
                self.assertAlmostEqual(float((mass * coord.square()).sum()), .5, delta=2e-5)
                self.assertAlmostEqual(float((mass * coord).sum()), 0, delta=1e-10)
            self.assertAlmostEqual(float((mass * u * v).sum()), 0, delta=1e-10)

    def test_spherical_truncation_and_concentration_monotonicity(self):
        nodes, weights = np.polynomial.legendre.leggauss(120)
        radius = torch.from_numpy((nodes + 1) / 2)[:, None]
        angle = (torch.arange(768, dtype=torch.float64) + .5) * (2 * math.pi / 768)
        xx, yy = radius * angle.cos(), radius * angle.sin()
        zz = (1 - radius.square()).sqrt().expand_as(xx)
        dirs = torch.stack((xx, yy, zz), -1).reshape(-1, 3)
        measure = (torch.from_numpy(weights / 2)[:, None] * radius / zz * (2 * math.pi / 768)).reshape(-1)
        frame = torch.eye(3, dtype=torch.float64)
        for p, q, expected in ((1, 2, .993262053), (1, 16, .991098126), (4, 16, .999999770)):
            masses = []
            for lam, mu in ((5., 5.), (10., 5.), (10., 10.)):
                values = asg_kernel_weights(dirs, frame[2:3], frame[:1], frame[1:2],
                                            torch.tensor([lam]), torch.tensor([mu]), footprint([p], [q]))
                masses.append(float((values[:, 0] * measure).sum()))
            self.assertAlmostEqual(masses[0], expected, delta=2e-6)
            self.assertLessEqual(masses[0], masses[1] + 1e-9)
            self.assertLessEqual(masses[1], masses[2] + 1e-9)

    def test_rotation_axis_exchange_and_horizon(self):
        bank, allocation = superelliptic_bank()
        with torch.no_grad():
            bank.log_p.fill_(math.log(2))
            bank.log_q_ratio.fill_(math.log(3))
            z, x, y, lam, mu, _, shape = bank.current_parameters()
        dirs = allocation.directions
        expected = asg_kernel_weights(dirs, z, x, y, lam, mu, shape)
        rotation, _ = torch.linalg.qr(torch.tensor([[.2, .3, .7], [.7, -.4, .8], [.3, .8, .1]], dtype=dirs.dtype))
        actual = asg_kernel_weights(dirs @ rotation.T, z @ rotation.T, x @ rotation.T,
                                    y @ rotation.T, lam, mu, shape)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(asg_kernel_weights(dirs, z, y, -x, mu, lam, shape), expected)
        for eps in (1e-3, 1e-6, 0.):
            near = F.normalize(x[:1] + eps * z[:1], dim=-1)
            value = asg_kernel_weights(near, z[:1], x[:1], y[:1], lam[:1], mu[:1], shape[:1])
            self.assertLessEqual(float(value), 2 * eps * float(asg_integral_norm(lam[:1], mu[:1], footprint=shape[:1])) + 1e-12)


class ShapeAdaptiveBankTests(unittest.TestCase):
    def test_config_freezing_projection_and_disabled_baseline(self):
        baseline, allocation = superelliptic_bank(asg_footprint="gaussian")
        frozen, _ = superelliptic_bank(asg_learn_p=False, asg_learn_q=False)
        self.assertEqual(len(baseline.current_parameters()), 6)
        self.assertEqual(len(baseline.get_optimizer_param_groups()), 4)
        self.assertNotIn("log_p", dict(baseline.named_parameters()))
        torch.testing.assert_close(baseline.render_directions(allocation.directions), frozen.render_directions(allocation.directions))
        self.assertFalse(frozen.log_p.requires_grad or frozen.log_q_ratio.requires_grad)
        with torch.no_grad():
            frozen.log_p.copy_(torch.tensor([-1., 5., 0.]))
            frozen.log_q_ratio.copy_(torch.tensor([-5., 5., 0.]))
        frozen.project_parameters_()
        shape = frozen.current_parameters()[6]
        torch.testing.assert_close(shape.p, torch.tensor([1., 4., 1.], dtype=shape.p.dtype))
        torch.testing.assert_close(shape.q, torch.tensor([1.25, 16., 2.], dtype=shape.q.dtype))
        for update in ({"asg_footprint": "invalid"}, {"asg_learn_p": 1}, {"asg_learn_q": "false"},
                       {"lr_asg_shape": float("nan")}, {"lr_asg_shape": 0}):
            cfg = training_config_defaults()
            cfg.update(update)
            with self.assertRaises(ValueError):
                resolve_training_config(cfg)

    def test_proposals_geometry_and_all_render_paths(self):
        for dtype in (torch.float32, torch.float64):
            bank, allocation = superelliptic_bank(dtype=dtype)
            dirs = allocation.directions
            proposals = bank.initialize_densification_proposals(dirs[:3], dirs.new_full((3,), .25),
                                                                dirs.new_full((3,), .3), dirs.new_ones((3, 3)))
            self.assertTrue(torch.equal(proposals.log_p, torch.zeros_like(proposals.log_p)))
            proposals = replace(proposals, log_p=dirs.new_tensor([0., .2, .8]),
                                log_q_ratio=dirs.new_tensor([-.2, .5, 1.]))
            precision = torch.diag(dirs.new_tensor([1., 3.])).expand(3, 2, 2)
            proposals, changed = replace_asg_proposal_geometry(
                proposals, precision, torch.ones(3, dtype=torch.bool), lambda_min=5.,
                lambda_max=1e6, width_ratio_max=8,
            )
            self.assertTrue(bool(changed.all()))
            bank.validate_densification_proposals(proposals)
            contributions = render_asg_proposal_contributions(dirs, proposals)
            aggregate = render_asg_proposal_aggregate(dirs, proposals, sample_chunk_size=17, proposal_chunk_size=2)
            torch.testing.assert_close(aggregate, contributions.sum(1))
            prediction = bank.render_directions_from_parameters(dirs, *bank.current_parameters(proposals), chunk_size=2)
            bank.append_densification_proposals(proposals)
            torch.testing.assert_close(bank.render_directions(dirs), prediction)
            torch.testing.assert_close(bank.render_directions_chunked(dirs, 2), prediction)
            with tempfile.TemporaryDirectory() as root:
                save_native_inference_artifact(root, kernel_bank=bank)
                native = load_native_inference_artifact(Path(root) / "inference.json", dtype=dtype)
                torch.testing.assert_close(native.evaluate_directions(dirs), prediction)

    def test_actual_contours_and_tangent_chart_crossing(self):
        bank, _ = superelliptic_bank(count=1)
        with torch.no_grad():
            bank.log_p.fill_(math.log(2))
            bank.log_q_ratio.fill_(math.log(4))
            bank.theta.fill_(.37)
            bank.lobe_axis.copy_(torch.tensor([[.6, .8, 1e-9]], dtype=bank.dtype))
        with tempfile.TemporaryDirectory() as root:
            save_native_inference_artifact(root, kernel_bank=bank)
            native = load_native_inference_artifact(Path(root) / "inference.json", dtype=bank.dtype)
            contour = asg_footprint_contours(native, torch.tensor([0]), samples=32)[0]
        peak = bank.render_directions(bank.centers())[0]
        torch.testing.assert_close(bank.render_directions(contour), peak.expand(len(contour), -1) / math.e,
                                   atol=1e-8, rtol=1e-7)
        expected = bank.render_directions(contour).detach()
        previous = bank.centers().detach().clone()
        with torch.no_grad():
            bank.lobe_axis[:, 2].neg_()
        bank.project_parameters_()
        bank.preserve_tangent_chart_(previous)
        torch.testing.assert_close(bank.render_directions(contour), expected, atol=1e-8, rtol=1e-7)

    def test_selector_acceptance_append_and_frozen_optimizer_state(self):
        for learn_p, learn_q in ((True, True), (True, False), (False, True), (False, False)):
            bank, allocation = superelliptic_bank(asg_learn_p=learn_p, asg_learn_q=learn_q)
            dirs, areas = allocation.directions, allocation.solid_angles
            optimizer = RowwiseAdam(bank.get_optimizer_param_groups())
            bank.render_directions(dirs).square().mean().backward()
            optimizer.step()
            bank.project_parameters_()
            optimizer.zero_grad()
            old = {name: value.detach().clone() for name, value in bank.named_parameters()}
            state = {name: {key: value.clone() for key, value in optimizer.state[param].items()}
                     for name, param in bank.named_parameters()}
            before = bank.render_directions(dirs).detach()
            target = before + 2
            seeds = build_residual_proposal_seeds(dirs, areas, target, before, num_new=2,
                                                 current_kernel_count=bank.num_kernels,
                                                 optimization_hdr_scale=1., seed=123)
            selected, stats = select_asg_insertions(
                seeds, dirs, areas, target, before, num_new=2,
                quadrature_weights=areas * (len(dirs) / (4 * math.pi)),
                loss_fn=SphericalHDRLoss(), kernel_bank=bank, step=1000,
                sample_chunk_size=80, gaussian_chunk_size=2,
            )
            self.assertIsNotNone(selected, stats)
            self.assertLess(float(selected.reconstruction_objective_after), float(selected.reconstruction_objective_before))
            append_selected_asg_batch(bank, optimizer, selected, final_kernel_budget=5)
            torch.testing.assert_close(bank.render_directions_chunked(dirs, 2), selected.prediction_after)
            for name, param in bank.named_parameters():
                self.assertTrue(torch.equal(param[:3], old[name]))
                for key, value in state[name].items():
                    self.assertTrue(torch.equal(optimizer.state[param][key][:3], value))
                self.assertTrue(bool((optimizer.state[param]["row_age"][3:] == 0).all()))
            self.assertEqual(bank.log_p.requires_grad, learn_p)
            self.assertEqual(bank.log_q_ratio.requires_grad, learn_q)
            bank.render_directions(dirs).square().mean().backward()
            optimizer.step()
            bank.project_parameters_()
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in bank.parameters()))

    def test_empty_and_invalid_shape_proposals(self):
        bank, allocation = superelliptic_bank()
        empty = bank.initialize_densification_proposals(allocation.directions[:0],
                                                        torch.empty(0), torch.empty(0), torch.empty(0, 3))
        self.assertTrue(bank.validate_densification_proposals(empty))
        self.assertEqual(render_asg_proposal_contributions(allocation.directions, empty).shape,
                         (len(allocation.directions), 0, 3))
        proposal = bank.initialize_densification_proposals(allocation.directions[:1],
                                                          torch.tensor([.2]), torch.tensor([.3]), torch.ones(1, 3))
        for malformed in (replace(proposal, log_p=None),
                          replace(proposal, log_p=torch.tensor([float("nan")], dtype=bank.dtype)),
                          replace(proposal, log_q_ratio=torch.tensor([4.], dtype=bank.dtype))):
            with self.assertRaises(ValueError):
                bank.validate_densification_proposals(malformed)
        before = {name: value.detach().clone() for name, value in bank.named_parameters()}
        with self.assertRaises(ValueError):
            bank.append_densification_proposals(replace(proposal, log_p=None))
        for name, value in bank.named_parameters():
            self.assertTrue(torch.equal(value, before[name]))


class KernelNormalizationTests(unittest.TestCase):
    """Coefficients are integrated radiance: initialization and rendering agree with it."""

    def _bank(self, allocation, footprint="gaussian", count=16):
        return build_kernel_bank(
            _args(footprint, count),
            init_allocation=allocation,
            target_radiance_integral=_target_integral(allocation),
            feat_dim=3,
            device=allocation.directions.device,
            dtype=allocation.directions.dtype,
        )

    def test_nominal_coefficients_sum_to_target_spherical_integral(self):
        projection = build_projection("latlong")
        envmap = torch.rand(12, 24, 3) * torch.tensor([1.0, 3.0, 7.0])
        allocation = _uniform_allocation(projection, 12, 24, 16, envmap.device, envmap.dtype, envmap=envmap)
        target_integral = _target_integral(allocation)
        target_mean = spherical_channel_mean(allocation.target_values, allocation.solid_angles)
        torch.testing.assert_close(target_integral, target_mean * allocation.solid_angles.sum(), rtol=1e-6, atol=1e-6)
        for footprint in ("gaussian", "superelliptic"):
            with self.subTest(footprint=footprint):
                bank = self._bank(allocation, footprint)
                torch.testing.assert_close(bank.radiance_coeff.sum(dim=0), target_integral, rtol=2e-5, atol=1e-6)

    def test_isotropic_asg_mass_matches_documented_nominal_deficit(self):
        concentration = torch.tensor(5.0, dtype=torch.float64)
        norm = asg_integral_norm(concentration, concentration)
        unit_disk_gaussian_integral = math.pi * (1.0 - torch.exp(-concentration)) / concentration
        expected = 1.0 - math.exp(-5.0)
        self.assertAlmostEqual(float(norm * unit_disk_gaussian_integral), expected, places=14)
        self.assertLess(1.0 - expected, 0.00675)

    def test_zero_target_channel_has_only_the_activation_floor(self):
        projection = build_projection("latlong")
        envmap = torch.rand(12, 24, 3)
        envmap[..., 1] = 0.0
        allocation = _uniform_allocation(projection, 12, 24, 16, envmap.device, envmap.dtype, envmap=envmap)
        target_integral = _target_integral(allocation)
        self.assertEqual(float(target_integral[1]), 0.0)
        activated_sum = self._bank(allocation).radiance_coeff.detach().sum(dim=0)
        torch.testing.assert_close(activated_sum[[0, 2]], target_integral[[0, 2]], rtol=2e-6, atol=2e-6)
        self.assertAlmostEqual(float(activated_sum[1]), 16 * COEFFICIENT_ACTIVATION_FLOOR, places=12)

    def test_positive_integral_uses_uniform_fallback_for_zero_samples(self):
        sampled = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
        target_integral = torch.tensor([2.0, 0.0, 8.0], dtype=torch.float64)
        activated = torch.nn.functional.softplus(initialize_radiance_coefficients(sampled, target_integral))
        torch.testing.assert_close(activated.sum(dim=0)[[0, 2]], target_integral[[0, 2]], rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(activated[:, 0], torch.tensor([1.0, 1.0], dtype=torch.float64)))
        self.assertAlmostEqual(float(activated[:, 1].sum()), 2 * COEFFICIENT_ACTIVATION_FLOOR, places=15)
        tiny_integral = torch.tensor([1.0e-12], dtype=torch.float64)
        tiny = torch.nn.functional.softplus(initialize_radiance_coefficients(torch.zeros(2, 1, dtype=torch.float64), tiny_integral))
        torch.testing.assert_close(tiny.sum(dim=0), tiny_integral, rtol=1e-12, atol=0.0)

    def test_rendered_kernel_integrals_match_coefficients(self):
        projection = build_projection("latlong")
        allocation = _uniform_allocation(projection, 8, 16, 4, "cpu", torch.float32)
        quadrature = build_training_quadrature(5, dtype=torch.float32)
        directions, areas = quadrature.directions, quadrature.solid_angles
        for footprint in ("gaussian", "superelliptic"):
            with self.subTest(footprint=footprint):
                torch.manual_seed(13)
                bank = self._bank(allocation, footprint, count=4)
                rendered = bank.render_directions_chunked(directions, chunk_size=16)
                integral = (rendered * areas.unsqueeze(1)).sum(dim=0)
                torch.testing.assert_close(integral, bank.radiance_coeff.sum(dim=0), rtol=0.012, atol=2e-3)

    def test_parameter_projection_enforces_feasible_state(self):
        allocation = _uniform_allocation(build_projection("latlong"), 8, 16, 16, "cpu", torch.float32)
        bank = self._bank(allocation)
        bank.lobe_axis.data.mul_(3.0)
        bank.log_lambda.data.fill_(math.log(bank.asg_lambda_max) + 2.0)
        bank.log_mu.data.fill_(math.log(bank.asg_lambda_min) - 2.0)
        bank.project_parameters_()
        torch.testing.assert_close(bank.lobe_axis.norm(dim=1), torch.ones(bank.num_kernels), atol=1e-6, rtol=0)
        lambda_, mu = bank.concentrations()
        self.assertTrue(bool(((lambda_ >= bank.asg_lambda_min) & (lambda_ <= bank.asg_lambda_max)).all()))
        self.assertTrue(bool((mu / lambda_ <= bank.asg_ratio_max * (1 + 1e-5)).all()))
        for value in bank.current_parameters():
            self.assertTrue(torch.isfinite(value).all())

    def test_tangent_frame_is_orthonormal_near_poles(self):
        centers = torch.tensor([[1e-7, 0.0, 1.0], [1e-7, 0.0, -1.0], [0.3, -0.4, 0.5]], dtype=torch.float64)
        theta = torch.tensor([0.0, 0.7, -1.2], dtype=torch.float64)
        x, y, z = build_asg_tangent_frame(centers, theta)
        self.assertTrue(torch.allclose(x.norm(dim=1), torch.ones(3, dtype=x.dtype)))
        self.assertTrue(torch.allclose(y.norm(dim=1), torch.ones(3, dtype=y.dtype)))
        self.assertTrue(torch.allclose((x * y).sum(dim=1), torch.zeros(3, dtype=x.dtype), atol=1e-12))
        self.assertTrue(torch.allclose((x * z).sum(dim=1), torch.zeros(3, dtype=x.dtype), atol=1e-12))

    def test_projected_boundary_parameters_keep_finite_gradients(self):
        allocation = _uniform_allocation(build_projection("latlong"), 8, 16, 16, "cpu", torch.float32)
        bank = self._bank(allocation)
        bank.log_lambda.data.fill_(math.log(bank.asg_lambda_max))
        bank.concentrations()[0].clamp(min=1e-12).rsqrt().sum().backward()
        self.assertTrue(torch.isfinite(bank.log_lambda.grad).all())
        self.assertGreater(float(bank.log_lambda.grad.abs().sum()), 0.0)

    def test_chunked_render_matches_direct_render_including_empty_bank(self):
        allocation = _uniform_allocation(build_projection("latlong"), 8, 16, 7, "cpu", torch.float32)
        directions = build_training_quadrature(2, dtype=torch.float32).directions
        probe = torch.randn(len(directions), 3, generator=torch.Generator().manual_seed(5))
        for footprint in ("gaussian", "superelliptic"):
            for count in (7, 0):
                with self.subTest(footprint=footprint, count=count):
                    bank = self._bank(allocation, footprint, count=7)
                    with torch.no_grad():  # anisotropic, so the theta gradient is not just roundoff
                        bank.log_mu.add_(math.log(4.0))
                    if count == 0:
                        for name, parameter in tuple(bank.named_parameters(recurse=False)):
                            setattr(bank, name, torch.nn.Parameter(parameter.detach()[:0].clone(),
                                                                   requires_grad=parameter.requires_grad))
                        bank.num_kernels = 0
                    results = []
                    # Uneven chunks of 3, 3 and 1 kernels against one direct render.
                    for render in (lambda: bank.render_directions_chunked(directions, 3),
                                   lambda: bank.render_directions(directions)):
                        bank.zero_grad(set_to_none=True)
                        values = render()
                        if count:
                            (values * probe).sum().backward()
                        results.append([values.detach()] + [p.grad for p in bank.parameters() if p.grad is not None])
                    self.assertEqual(len(results[0]), len(results[1]))
                    for actual, expected in zip(*results):
                        scale = float(expected.abs().max()) if expected.numel() else 0.0
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5 * scale)


class TangentChartTests(unittest.TestCase):
    def test_both_crossing_directions_preserve_the_continued_chart(self):
        bank = asg_bank(uniform_allocation(), num_gaussians=1)
        for sign in (1.0, -1.0):
            previous = torch.tensor([[0.6, 0.8, sign * 1e-8]], dtype=bank.dtype)
            with torch.no_grad():
                bank.lobe_axis.copy_(previous)
                bank.theta.fill_(0.37)
            old_x, old_y, _ = build_asg_tangent_frame(previous, bank.theta.detach())
            with torch.no_grad():
                bank.lobe_axis[:, 2] *= -1
            bank.project_parameters_()
            bank.preserve_tangent_chart_(previous)
            new_x, new_y, _ = build_asg_tangent_frame(bank.centers(), bank.theta.detach())
            torch.testing.assert_close(new_x, old_x, atol=3e-8, rtol=0)
            torch.testing.assert_close(new_y, old_y, atol=3e-8, rtol=0)
            before = bank.theta.detach().clone()
            bank.preserve_tangent_chart_(bank.centers().detach().clone())
            self.assertTrue(torch.equal(bank.theta, before))

    def test_chart_compensation_uses_projected_centers_and_keeps_adam_history(self):
        bank = asg_bank(uniform_allocation(), num_gaussians=1)
        optimizer = RowwiseAdam([bank.lobe_axis, bank.theta], lr=0.02)
        with torch.no_grad():
            bank.lobe_axis.copy_(torch.tensor([[0.6, 0.8, 1e-4]], dtype=bank.dtype))
            bank.theta.fill_(0.37)
        previous = bank.centers().detach().clone()
        old_x, old_y, _ = build_asg_tangent_frame(previous, bank.theta.detach())
        bank.lobe_axis.grad = torch.tensor([[0.0, 0.0, 1.0]], dtype=bank.dtype)
        bank.theta.grad = torch.zeros_like(bank.theta)
        optimizer.step()
        bank.project_parameters_()
        bank.preserve_tangent_chart_(previous)
        new_x, new_y, _ = build_asg_tangent_frame(bank.centers().detach(), bank.theta.detach())
        self.assertLess(float((old_x - new_x).norm()), 0.03)
        self.assertLess(float((old_y - new_y).norm()), 0.03)
        self.assertEqual(optimizer.state[bank.theta]["row_age"].tolist(), [1])
        previous = bank.centers().detach().clone()
        optimizer.step()
        bank.project_parameters_()
        bank.preserve_tangent_chart_(previous)
        self.assertEqual(optimizer.state[bank.theta]["row_age"].tolist(), [2])
        # A trust projection that returns to the old chart must not rotate theta.
        with torch.no_grad():
            bank.residual_center_anchor.copy_(bank.centers())
            bank.residual_center_max_angle.zero_()
            previous = bank.centers().clone()
            theta = bank.theta.clone()
            bank.lobe_axis[:, 2].abs_()
        bank.project_parameters_()
        bank.preserve_tangent_chart_(previous)
        self.assertTrue(torch.equal(theta, bank.theta))


class InitializationTests(unittest.TestCase):
    def test_spherical_structure_controls_anisotropy_and_geometric_mean(self):
        flat_bank = _make_bank(ratio=1.0)
        edge_bank = _make_bank(ratio=3.0)

        self.assertAlmostEqual(_width_ratio(flat_bank), 1.0, places=5)
        self.assertAlmostEqual(_width_ratio(edge_bank), 3.0, places=4)

        lambda_, mu = edge_bank.concentrations()
        base_width = lambda_.rsqrt() / math.sqrt(_width_ratio(edge_bank))
        expected = base_width.pow(-4)
        self.assertTrue(torch.allclose(lambda_ * mu, expected, rtol=1e-5, atol=1e-5))


class AllocationTests(unittest.TestCase):
    def test_constant_signal_is_finite_normalized_and_uniform(self):
        projection = build_projection("latlong")
        envmap = torch.ones(16, 32, 3)
        field = EnvmapField(envmap, projection)
        allocation = _allocation_kernel_allocation(field, 64, subdivision=3)
        solid_angle = allocation.solid_angles

        for value in (
            allocation.probability_flat,
            allocation.density_flat,
            allocation.local_width_flat,
            allocation.structure.anisotropy_ratio_flat,
        ):
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue((value >= 0.0).all())
        self.assertAlmostEqual(float(allocation.probability_flat.sum()), 1.0, places=6)
        self.assertAlmostEqual(
            float((allocation.density_flat * solid_angle).sum()), 1.0, places=6
        )
        self.assertTrue(
            torch.allclose(
                allocation.density_flat,
                torch.full_like(allocation.density_flat, 1.0 / (4.0 * math.pi)),
                rtol=1e-5,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                allocation.structure.anisotropy_ratio_flat,
                torch.ones_like(allocation.structure.anisotropy_ratio_flat),
                rtol=1e-5,
                atol=1e-6,
            )
        )

    def test_projection_representations_agree_on_common_spherical_candidates(self):
        latlong = build_projection("latlong")
        octahedral = build_projection("octahedral")
        lat_field = EnvmapField(_analytic_envmap(latlong, 32, 64), latlong)
        oct_field = EnvmapField(_analytic_envmap(octahedral, 45, 45), octahedral)
        common = build_training_quadrature(4, dtype=torch.float32)
        common_dirs, common_areas = common.directions, common.solid_angles
        lat_allocation = build_kernel_allocation(
            lat_field,
            common_dirs,
            common_areas,
            128,
            target_values=lat_field.sample(common_dirs),
        )
        oct_allocation = build_kernel_allocation(
            oct_field,
            common_dirs,
            common_areas,
            128,
            target_values=oct_field.sample(common_dirs),
        )

        density_similarity = torch.nn.functional.cosine_similarity(
            lat_allocation.density_flat, oct_allocation.density_flat, dim=0
        )
        detail_similarity = torch.nn.functional.cosine_similarity(
            lat_allocation.structure.detail_flat,
            oct_allocation.structure.detail_flat,
            dim=0,
        )
        self.assertGreater(float(density_similarity), 0.90)
        self.assertGreater(float(detail_similarity), 0.85)


if __name__ == "__main__":
    unittest.main()
