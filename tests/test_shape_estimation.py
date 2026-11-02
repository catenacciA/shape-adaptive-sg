"""Local shape estimation for inserted primitives (orientation and anisotropy)."""
import math
import unittest

import torch

from envmap.quadrature import build_training_quadrature
from fit.asg_shape import SphericalCapIndex, estimate_proposal_shapes, fit_local_asg_shape
from fit.loss import SphericalHDRLoss
from model.asg import (
    asg_integral_norm, render_asg_proposal_aggregate,
    render_asg_proposal_contributions, replace_asg_proposal_geometry,
)
from model.primitives import build_asg_tangent_frame
from tests.helpers import uniform_allocation, asg_bank


def patch_samples(index=500, subdivision=5):
    q = build_training_quadrature(subdivision, dtype=torch.float64)
    center = q.directions[index:index + 1]
    x, y, _ = build_asg_tangent_frame(center, torch.zeros(1, dtype=torch.float64))
    ids = torch.arange(q.sample_count)
    h = q.directions[ids] @ center[0]
    ids = ids[h >= math.sqrt(1 - .15**2)]
    h = q.directions[ids] @ center[0]
    v = torch.stack((q.directions[ids] @ x[0], q.directions[ids] @ y[0]), -1) / .15
    return v, q.solid_angles[ids], h


def local_signal(v, h, B, b):
    return h * torch.exp(v @ b - (v * (v @ B)).sum(-1))


class ConditionalShapeTests(unittest.TestCase):
    def test_cap_index_matches_full_scan_including_duplicates_and_poles(self):
        for dtype in (torch.float32, torch.float64):
            q = build_training_quadrature(3, dtype=dtype)
            directions = torch.cat((q.directions, q.directions[:7])).double()
            index = SphericalCapIndex(directions)
            centers = torch.cat((torch.eye(3).double(), -torch.eye(3).double(), directions[::91]))
            for center in centers:
                for radius in (0.001, 0.07, 0.44, math.pi / 2):
                    cosine = (directions * center).sum(-1)
                    expected = torch.where((cosine >= math.cos(radius)) & (cosine > 0))[0]
                    actual = torch.from_numpy(index.query(center.numpy(), radius))
                    self.assertTrue(torch.equal(actual, expected))

    def test_cap_uses_all_nodes_independent_of_order_and_outside_signal(self):
        q = build_training_quadrature(4, dtype=torch.float64)
        bank = asg_bank(uniform_allocation())
        proposals = bank.initialize_densification_proposals(
            q.directions[500:501], torch.tensor([.3]),
            torch.tensor([math.asin(.3)]), torch.ones(1, 3),
        )
        x, y, z = build_asg_tangent_frame(proposals.lobe_axis, proposals.theta)
        h = q.directions @ z[0]
        v = torch.stack((q.directions @ x[0], q.directions @ y[0]), -1) / proposals.local_width[0]
        truth = v.new_tensor([[1.1, -.4], [-.4, 2.]])
        luma = local_signal(v, h.clamp_min(0), truth, v.new_tensor([.3, -.2]))
        mask = h >= proposals.support_angular_radius[0].cos()
        first = estimate_proposal_shapes(proposals, q.directions, q.solid_angles, luma)
        self.assertEqual(int(first.sample_count[0]), int(mask.sum()))
        self.assertTrue(bool(first.eligible[0]))
        torch.testing.assert_close(first.precision[0], truth, atol=2e-7, rtol=2e-7)
        luma[~mask] = 1e20
        order = torch.randperm(q.sample_count, generator=torch.Generator().manual_seed(17))
        second = estimate_proposal_shapes(proposals, q.directions[order], q.solid_angles[order], luma[order])
        torch.testing.assert_close(second.precision, first.precision, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(second.tilt, first.tilt, atol=2e-7, rtol=2e-7)
        # Subdividing a quadrature weight at the same observation is measure
        # equivalent, including when packed adaptive nodes have unequal areas.
        split = estimate_proposal_shapes(
            proposals, q.directions.repeat_interleave(2, 0),
            q.solid_angles.repeat_interleave(2) / 2, luma.repeat_interleave(2),
        )
        torch.testing.assert_close(split.precision, first.precision, atol=2e-7, rtol=2e-7)

    def test_joint_spherical_rotation_transports_the_shape_tensor(self):
        q = build_training_quadrature(4, dtype=torch.float64)
        bank = asg_bank(uniform_allocation())
        proposals = bank.initialize_densification_proposals(
            q.directions[500:501], torch.tensor([.3]), torch.tensor([math.asin(.3)]), torch.ones(1, 3),
        )
        x, y, z = build_asg_tangent_frame(proposals.lobe_axis, proposals.theta)
        v = torch.stack((q.directions @ x[0], q.directions @ y[0]), -1) / .3
        h = q.directions @ z[0]
        luma = local_signal(v, h.clamp_min(0), torch.diag(v.new_tensor([1., 3.])), v.new_tensor([.3, -.1]))
        first = estimate_proposal_shapes(proposals, q.directions, q.solid_angles, luma)
        angle = .87
        rotation = v.new_tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        rotated = bank.initialize_densification_proposals(
            proposals.lobe_axis @ rotation.T, proposals.local_width,
            proposals.support_angular_radius, proposals.radiance_coeff,
        )
        second = estimate_proposal_shapes(rotated, q.directions @ rotation.T, q.solid_angles, luma)
        self.assertTrue(bool(first.eligible[0] and second.eligible[0]))
        rx, ry, _ = build_asg_tangent_frame(rotated.lobe_axis, rotated.theta)
        E = torch.stack((x[0], y[0]), -1)
        rotated_E = torch.stack((rx[0], ry[0]), -1)
        torch.testing.assert_close(
            rotated_E @ second.precision[0] @ rotated_E.T,
            rotation @ E @ first.precision[0] @ E.T @ rotation.T,
            atol=2e-6, rtol=2e-6,
        )

    def test_recovers_curvature_and_nuisance_tilt_on_allocation_caps(self):
        B = torch.tensor([[1.1, -.6], [-.6, 2.4]], dtype=torch.float64)
        b = torch.tensor([.8, -.4], dtype=torch.float64)
        for index in (500, 517, 4096):
            v, a, h = patch_samples(index)
            for precision in (B, torch.eye(2, dtype=torch.float64) * 2):
                signal = local_signal(v, h, precision, b)
                fit = fit_local_asg_shape(v, a, h, signal)
                self.assertEqual(fit.reason, "eligible")
                torch.testing.assert_close(fit.precision, precision, atol=2e-7, rtol=2e-7)
                torch.testing.assert_close(fit.tilt, b, atol=2e-7, rtol=2e-7)
                amplified = fit_local_asg_shape(v, a, h, signal * 1e8)
                torch.testing.assert_close(amplified.precision, fit.precision, atol=2e-7, rtol=2e-7)

    def test_basis_rotation_covariance(self):
        v, a, h = patch_samples()
        B = torch.tensor([[1.1, -.6], [-.6, 2.4]], dtype=torch.float64)
        b = torch.tensor([.8, -.4], dtype=torch.float64)
        signal = local_signal(v, h, B, b)
        angle = .73
        rotation = v.new_tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        fit = fit_local_asg_shape(v @ rotation, a, h, signal)
        torch.testing.assert_close(fit.precision, rotation.T @ B @ rotation, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(fit.tilt, rotation.T @ b, atol=2e-7, rtol=2e-7)

    def test_degenerate_and_non_lobe_fields_fall_back(self):
        v, a, h = patch_samples()
        cases = [
            (v[:4], a[:4], h[:4], h[:4], "unidentifiable_geometry"),
            (v, a, h, torch.zeros_like(h), "zero_mass"),
            (v, a, h, torch.nn.functional.one_hot(torch.tensor(0), len(h)).double(), "singular_mass"),
            (v, a, h, h * torch.exp(v[:, 0].square() - v[:, 1].square()), "nonpositive_curvature"),
            (v, a, h, torch.ones_like(h), "nonpositive_curvature"),
        ]
        for coords, areas, cosine, signal, reason in cases:
            self.assertEqual(fit_local_asg_shape(coords, areas, cosine, signal).reason, reason)
        line = torch.stack((v[:, 0], v[:, 0]), -1)
        self.assertEqual(fit_local_asg_shape(line, a, h, h).reason, "unidentifiable_geometry")

    def test_boundary_mle_does_not_become_a_finite_shape(self):
        # Mass on a circle with additional interior observation nodes has no
        # finite quadratic-family solution, despite full spatial covariance.
        phi = torch.arange(16, dtype=torch.float64) * (2 * math.pi / 16)
        ring = torch.stack((phi.cos(), phi.sin()), -1)
        v = torch.cat((ring, .5 * ring, ring[:1] * 0))
        luma = torch.cat((torch.ones(16), torch.zeros(17))).double()
        fit = fit_local_asg_shape(v, torch.ones(33), torch.ones(33), luma)
        self.assertEqual(fit.reason, "nonconverged")


class ShapeGeometryAndGateTests(unittest.TestCase):
    def setUp(self):
        self.a = uniform_allocation(subdivision=4)
        self.bank = asg_bank(self.a)
        self.dirs = self.a.directions
        self.areas = self.a.solid_angles
        self.loss = SphericalHDRLoss()
        self.base = self.bank.initialize_densification_proposals(
            self.dirs[500:501], torch.tensor([.15]), torch.tensor([math.asin(.15)]),
            torch.tensor([[.04, .04, .04]]),
        )
        self.shape, self.changed = replace_asg_proposal_geometry(
            self.base, torch.diag(torch.tensor([.25, 4.])).unsqueeze(0), torch.tensor([True]),
            lambda_min=5., lambda_max=1e6, width_ratio_max=8.,
        )
        self.prediction = self.bank.render_directions(self.dirs).detach()
        self.r = render_asg_proposal_aggregate(self.dirs, self.base)
        self.h = render_asg_proposal_aggregate(self.dirs, self.shape)

    def test_shape_preserves_coefficients_footprint_peak_and_bounds(self):
        for name in ("lobe_axis", "local_width", "support_angular_radius", "radiance_coeff", "radiance_coeff_raw"):
            self.assertIs(getattr(self.base, name), getattr(self.shape, name))
        torch.testing.assert_close(self.shape.lambda_ * self.shape.mu_bandwidth, self.base.lambda_ * self.base.mu_bandwidth)
        torch.testing.assert_close(asg_integral_norm(self.shape.lambda_, self.shape.mu_bandwidth), asg_integral_norm(self.base.lambda_, self.base.mu_bandwidth))
        torch.testing.assert_close(
            render_asg_proposal_contributions(self.base.lobe_axis, self.base),
            render_asg_proposal_contributions(self.base.lobe_axis, self.shape),
        )
        for width in (1/math.sqrt(5), 1/math.sqrt(1e6), .002, .3):
            base = self.bank.initialize_densification_proposals(self.base.lobe_axis, torch.tensor([width]), torch.tensor([.4]), self.base.radiance_coeff)
            shaped, _ = replace_asg_proposal_geometry(base, torch.diag(torch.tensor([1e-12, 1e12])).unsqueeze(0), torch.tensor([True]), lambda_min=5., lambda_max=1e6, width_ratio_max=8.)
            torch.testing.assert_close(asg_integral_norm(shaped.lambda_, shaped.mu_bandwidth), asg_integral_norm(base.lambda_, base.mu_bandwidth))
            self.assertGreaterEqual(float(shaped.lambda_[0]), 5 - 1e-10)
            self.assertLessEqual(float(shaped.mu_bandwidth[0]), 1e6 + 1e-8)
            self.assertLessEqual(float(torch.sqrt(shaped.mu_bandwidth / shaped.lambda_)[0]), 8 + 1e-10)

    def test_isotropic_geometry_is_an_exact_noop(self):
        result, changed = replace_asg_proposal_geometry(self.base, torch.eye(2).unsqueeze(0), torch.tensor([True]), lambda_min=5., lambda_max=1e6, width_ratio_max=8.)
        self.assertIs(result, self.base)
        self.assertFalse(bool(changed.any()))

    def test_float32_geometry_preserves_determinant_peak_and_width(self):
        bank = asg_bank(uniform_allocation(dtype=torch.float32))
        widths = torch.tensor([1 / math.sqrt(5), .3, .02, .002, .001])
        base = bank.initialize_densification_proposals(
            self.base.lobe_axis.float().repeat(len(widths), 1), widths,
            torch.asin(widths), torch.ones(len(widths), 3),
        )
        shaped, changed = replace_asg_proposal_geometry(
            base, torch.diag(torch.tensor([1e-12, 1e12])).repeat(len(widths), 1, 1),
            torch.ones(len(widths), dtype=torch.bool),
            lambda_min=5., lambda_max=1e6, width_ratio_max=8.,
        )
        self.assertTrue(bool(changed.any()))
        self.assertIs(shaped.local_width, base.local_width)
        self.assertIs(shaped.radiance_coeff, base.radiance_coeff)
        torch.testing.assert_close(asg_integral_norm(shaped.lambda_, shaped.mu_bandwidth), asg_integral_norm(base.lambda_, base.mu_bandwidth), rtol=2e-6, atol=0)
        torch.testing.assert_close(
            shaped.lambda_ * shaped.mu_bandwidth,
            base.lambda_ * base.mu_bandwidth, rtol=2e-6, atol=0,
        )
        self.assertTrue(bank.validate_densification_proposals(shaped))
        self.assertTrue(bool((torch.sqrt(shaped.mu_bandwidth / shaped.lambda_) <= 8 * (1 + 2e-6)).all()))

    def test_nominal_normalization_and_spherical_tail_bound(self):
        for proposals in (self.base, self.shape):
            rendered = render_asg_proposal_contributions(self.dirs, proposals)[:, 0]
            mass = (self.areas[:, None] * rendered).sum(0)
            self.assertTrue(bool((mass <= proposals.radiance_coeff[0] * 1.002).all()))
            self.assertTrue(bool((mass >= proposals.radiance_coeff[0] * (1 - math.exp(-5) - .002)).all()))
            torch.testing.assert_close(asg_integral_norm(proposals.lambda_, proposals.mu_bandwidth), asg_integral_norm(proposals.lambda_, proposals.mu_bandwidth))
