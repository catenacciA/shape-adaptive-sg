"""Projections, geodesic quadrature and spherical statistics."""
import math
import unittest

import torch

from envmap.dataset import GeodesicSphereDataset
from envmap.field import EnvmapField, validate_projection_shape
from envmap.projection import build_projection
from envmap.quadrature import EVALUATION_ROTATION_ID, build_adaptive_fitting_quadrature, build_evaluation_quadrature, build_training_quadrature, resolve_geodesic_subdivision
from envmap.statistics import compute_hdr_scale, spherical_channel_integral, spherical_channel_mean, weighted_quantile


def _field(projection_name, height=8, width=16):
    values = torch.linspace(0.0, 2.0, height * width * 3, dtype=torch.float32)
    values = values.reshape(height, width, 3)
    return EnvmapField(values, build_projection(projection_name))


def _constant_field(projection_name, height=8, width=16, value=1.0):
    values = torch.full((height, width, 3), float(value), dtype=torch.float32)
    return EnvmapField(values, build_projection(projection_name))


def _octa_field(values, projection="octahedral"):
    return EnvmapField(values, build_projection(projection))


def _constant_octa_field(value=1.0, *, dtype=torch.float64):
    return _octa_field(torch.full((8, 8, 3), float(value), dtype=dtype))


def _spike_field(*, dtype=torch.float64):
    values = torch.full((16, 16, 3), 0.05, dtype=dtype)
    values[8, 8] = torch.tensor((1000.0, 700.0, 300.0), dtype=dtype)
    return _octa_field(values)


def _analytic_signal(directions):
    base = directions.new_tensor((1.0, 2.0, 4.0))
    linear = torch.stack(
        (
            0.20 * directions[..., 0] - 0.10 * directions[..., 2],
            -0.15 * directions[..., 1] + 0.05 * directions[..., 2],
            0.25 * directions[..., 0] + 0.10 * directions[..., 1],
        ),
        dim=-1,
    )
    return base + linear


class ProjectionShapeTests(unittest.TestCase):
    def test_canonical_shapes_are_accepted(self):
        validate_projection_shape(16, 32, build_projection("latlong"))
        validate_projection_shape(16, 16, build_projection("octahedral"))

    def test_noncanonical_shapes_are_rejected_with_actionable_message(self):
        cases = (("latlong", 16, 16, "2:1"), ("octahedral", 16, 32, "square"))
        for name, height, width, expected in cases:
            with self.subTest(projection=name):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_projection_shape(
                        height, width, build_projection(name)
                    )


class ProjectionMathTests(unittest.TestCase):
    def test_roundtrip_and_solid_angle_sum(self):
        torch.manual_seed(4)
        directions = torch.randn(4096, 3, dtype=torch.float64)
        directions = directions / directions.norm(dim=1, keepdim=True)
        for name, height, width in (("latlong", 32, 64), ("octahedral", 32, 32)):
            with self.subTest(projection=name):
                projection = build_projection(name)
                u, v = projection.dir_to_uv(directions)
                restored = projection.pixel_to_dir(
                    u * width, v * height, width, height
                )
                self.assertTrue(
                    torch.allclose(restored, directions, rtol=1e-10, atol=1e-10)
                )
                area = projection.solid_angle_map(
                    height, width, torch.device("cpu"), torch.float64
                )
                self.assertAlmostEqual(float(area.sum()), 4.0 * math.pi, places=10)

    def test_latlong_poles_use_border_rows(self):
        projection = build_projection("latlong")
        envmap = torch.zeros(4, 8, 1)
        envmap[0] = 2.0
        envmap[1] = 10.0
        north = torch.tensor([[0.0, 0.0, 1.0]])
        south = torch.tensor([[0.0, 0.0, -1.0]])
        self.assertEqual(float(projection.sample_bilinear(envmap, north)), 2.0)
        self.assertEqual(float(projection.sample_bilinear(envmap, south)), 0.0)

    def test_octahedral_filter_is_continuous_across_square_seams(self):
        projection = build_projection("octahedral")
        size = 64
        coords = torch.arange(size, dtype=torch.float64)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center_dirs = projection.pixel_to_dir(xx + 0.5, yy + 0.5, size, size)
        envmap = (0.5 + 0.25 * center_dirs).contiguous()
        y = torch.linspace(4.0, size - 4.0, 32, dtype=torch.float64)
        left_outside = projection.pixel_to_dir(
            torch.full_like(y, -0.25), y, size, size
        )
        left_inside = projection.pixel_to_dir(
            torch.full_like(y, 0.25), size - y, size, size
        )
        sampled_outside = projection.sample_bilinear(envmap, left_outside)
        sampled_inside = projection.sample_bilinear(envmap, left_inside)
        expected_outside = 0.5 + 0.25 * left_outside
        expected_inside = 0.5 + 0.25 * left_inside
        self.assertLess(float((sampled_outside - expected_outside).abs().max()), 0.02)
        self.assertLess(float((sampled_inside - expected_inside).abs().max()), 0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_device_field_samples_host_directions(self):
        # Evaluation samples a device-resident reference at CPU float64 directions.
        directions = torch.nn.functional.normalize(torch.randn(64, 3, dtype=torch.float64), dim=1)
        for name, width in (("latlong", 16), ("octahedral", 8)):
            host = _field(name, 8, width)
            field = EnvmapField(host.values.cuda(), host.projection)
            torch.testing.assert_close(field.sample(directions).cpu(), host.sample(directions))


class GeodesicGridTests(unittest.TestCase):
    def test_level_five_geometry(self):
        quadrature = build_training_quadrature(5, dtype=torch.float64)
        directions = quadrature.directions
        solid_angles = quadrature.solid_angles

        self.assertEqual(tuple(directions.shape), (20480, 3))
        self.assertEqual(tuple(solid_angles.shape), (20480,))
        self.assertTrue(torch.isfinite(directions).all())
        self.assertTrue(torch.isfinite(solid_angles).all())
        self.assertTrue((solid_angles > 0.0).all())
        self.assertTrue(
            torch.allclose(
                directions.norm(dim=1),
                torch.ones(20480, dtype=directions.dtype),
                rtol=1e-12,
                atol=1e-12,
            )
        )
        self.assertAlmostEqual(float(solid_angles.sum()), 4.0 * math.pi, places=10)
        self.assertEqual(torch.unique(directions, dim=0).shape[0], 20480)

    def test_geometry_is_deterministic(self):
        quadrature_a = build_training_quadrature(2, dtype=torch.float64)
        quadrature_b = build_training_quadrature(2, dtype=torch.float64)

        self.assertTrue(torch.equal(quadrature_a.directions, quadrature_b.directions))
        self.assertTrue(
            torch.equal(quadrature_a.solid_angles, quadrature_b.solid_angles)
        )
        self.assertEqual(quadrature_a.grid_id, quadrature_b.grid_id)

        quadrature_a.directions.zero_()
        rebuilt = build_training_quadrature(2, dtype=torch.float64)
        self.assertFalse(torch.equal(quadrature_a.directions, rebuilt.directions))
        self.assertEqual(rebuilt.grid_id, quadrature_b.grid_id)

    def test_evaluation_rotation_preserves_areas_and_changes_identity(self):
        training = build_training_quadrature(2, dtype=torch.float64)
        evaluation = build_evaluation_quadrature(2, dtype=torch.float64)
        self.assertTrue(torch.equal(training.solid_angles, evaluation.solid_angles))
        self.assertFalse(torch.equal(training.directions, evaluation.directions))
        maximum_training_dot = (
            evaluation.directions @ training.directions.transpose(0, 1)
        ).max(dim=1).values
        self.assertTrue(bool((maximum_training_dot < 1.0 - 1e-12).all().item()))
        self.assertNotEqual(training.grid_id, evaluation.grid_id)
        self.assertEqual(evaluation.rotation_id, EVALUATION_ROTATION_ID)
        self.assertTrue(
            torch.allclose(
                evaluation.directions.norm(dim=1),
                torch.ones(evaluation.sample_count, dtype=torch.float64),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        matrix = torch.tensor(evaluation.rotation_matrix, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(matrix.T @ matrix, torch.eye(3, dtype=torch.float64), atol=1e-12)
        )
        self.assertAlmostEqual(float(torch.linalg.det(matrix)), 1.0, places=12)

    def test_auto_subdivision_matches_multiplicative_pixel_budget(self):
        self.assertEqual(resolve_geodesic_subdivision("auto", 1024, 1024), 8)
        self.assertEqual(resolve_geodesic_subdivision("auto", 512, 1024), 7)
        self.assertEqual(resolve_geodesic_subdivision("auto", 4, 8), 0)
        self.assertEqual(resolve_geodesic_subdivision("auto", 5, 8), 0)
        self.assertEqual(resolve_geodesic_subdivision(3, 1, 1), 3)
        self.assertEqual(resolve_geodesic_subdivision("3", 1, 1), 3)
        for value in (-1, "invalid", 1.5, 3.0, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_geodesic_subdivision(value, 8, 8)

    def test_full_geodesic_quadrature_integrates_low_order_functions(self):
        quadrature = build_training_quadrature(4, dtype=torch.float64)
        directions = quadrature.directions
        solid_angles = quadrature.solid_angles
        weights = solid_angles * directions.shape[0] / (4.0 * math.pi)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=10)
        self.assertTrue(
            torch.allclose((weights[:, None] * directions).mean(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
        )
        quadratic = (weights[:, None] * directions.square()).mean(dim=0)
        self.assertTrue(
            torch.allclose(quadratic, torch.full((3,), 1.0 / 3.0, dtype=torch.float64), atol=2e-4)
        )


class SampleStreamTests(unittest.TestCase):
    def test_batches_are_reproducible_and_roll_over_by_epoch(self):
        field = _constant_field("latlong")
        dataset_a = GeodesicSphereDataset(field, subdivision=0, seed=123)
        dataset_b = GeodesicSphereDataset(field, subdivision=0, seed=123)

        batch_a = dataset_a.sample_batch(27)
        batch_b = dataset_b.sample_batch(27)
        self.assertTrue(torch.equal(batch_a.sample_indices, batch_b.sample_indices))
        self.assertEqual(torch.unique(batch_a.sample_indices[:20]).numel(), 20)
        self.assertEqual(torch.unique(batch_a.sample_indices[20:]).numel(), 7)
        self.assertTrue(
            torch.equal(batch_a.dirs, dataset_a.dirs[batch_a.sample_indices])
        )
        self.assertTrue(
            torch.equal(batch_a.target, dataset_a.target[batch_a.sample_indices])
        )
        self.assertTrue(
            torch.equal(
                batch_a.quadrature_weights,
                dataset_a.quadrature_weights[batch_a.sample_indices],
            )
        )

    def test_quadrature_weights_encode_face_solid_angle(self):
        dataset = GeodesicSphereDataset(_field("octahedral"), subdivision=1, seed=1)
        expected = dataset.solid_angles * (
            dataset.num_samples / (4.0 * math.pi)
        )
        self.assertTrue(torch.allclose(dataset.quadrature_weights, expected))
        self.assertAlmostEqual(
            float(dataset.quadrature_weights.sum()),
            float(dataset.num_samples),
            places=4,
        )


class AdaptiveQuadratureTests(unittest.TestCase):
    def test_constant_and_gentle_fields_retain_the_base_grid(self):
        constant = build_adaptive_fitting_quadrature(
            _constant_octa_field(2.0), 2, parent_chunk_size=3
        )
        self.assertEqual(constant.refined_parent_count, 0)
        self.assertEqual(constant.sample_count, constant.base_count)
        expected_integral = torch.full(
            (3,), 8.0 * math.pi, dtype=constant.target_values.dtype
        )
        self.assertTrue(
            torch.allclose(
                (constant.solid_angles[:, None] * constant.target_values).sum(0),
                expected_integral,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        gentle_values = torch.ones((8, 8, 3), dtype=torch.float64)
        gentle_values[..., 0] += torch.linspace(
            -1.0e-4, 1.0e-4, gentle_values.shape[1], dtype=torch.float64
        ).unsqueeze(0)
        gentle_values[..., 1] += torch.linspace(
            -1.0e-4, 1.0e-4, gentle_values.shape[0], dtype=torch.float64
        ).unsqueeze(1)
        gentle = build_adaptive_fitting_quadrature(
            _octa_field(gentle_values), 2, parent_chunk_size=11
        )
        self.assertEqual(gentle.refined_parent_count, 0)
        self.assertEqual(gentle.sample_count, gentle.base_count)

    def test_child_localized_hdr_spike_refines_only_local_parents(self):
        quadrature = build_adaptive_fitting_quadrature(
            _spike_field(), 2, parent_chunk_size=7
        )
        self.assertGreater(quadrature.refined_parent_count, 0)
        self.assertLess(quadrature.refined_parent_count, quadrature.base_count)
        self.assertEqual(
            quadrature.sample_count,
            quadrature.base_count + 3 * quadrature.refined_parent_count,
        )
        self.assertGreater(quadrature.refined_luminance_energy_fraction, 0.0)

    def test_layout_is_exclusive_canonical_and_integral_preserving(self):
        field = _spike_field()
        base_subdivision = 2
        adaptive = build_adaptive_fitting_quadrature(
            field, base_subdivision, parent_chunk_size=5
        )
        base = build_training_quadrature(base_subdivision, dtype=field.dtype)
        fine = build_training_quadrature(base_subdivision + 1, dtype=field.dtype)
        fine_target = field.sample(fine.directions)

        counts = adaptive.layout.sample_counts
        self.assertTrue(bool(((counts == 1) | (counts == 4)).all().item()))
        self.assertTrue(
            torch.equal(counts == 4, adaptive.layout.refined_parent_mask)
        )
        self.assertEqual(int(adaptive.layout.sample_offsets[0]), 0)
        self.assertEqual(
            int(adaptive.layout.sample_offsets[-1]), adaptive.sample_count
        )
        self.assertAlmostEqual(
            float(adaptive.solid_angles.double().sum()), 4.0 * math.pi, places=11
        )

        adaptive_integral = (
            adaptive.solid_angles[:, None] * adaptive.target_values
        ).double().sum(dim=0)
        fine_integral = (
            fine.solid_angles[:, None] * fine_target
        ).double().sum(dim=0)
        self.assertTrue(
            torch.allclose(adaptive_integral, fine_integral, rtol=1e-12, atol=1e-11)
        )

        offsets = adaptive.layout.sample_offsets.tolist()
        refined = adaptive.layout.refined_parent_mask.tolist()
        for parent_index, is_refined in enumerate(refined):
            packed = slice(offsets[parent_index], offsets[parent_index + 1])
            children = slice(4 * parent_index, 4 * parent_index + 4)
            child_area = fine.solid_angles[children]
            child_target = fine_target[children]
            self.assertTrue(
                torch.allclose(
                    adaptive.solid_angles[packed].sum(),
                    child_area.sum(),
                    rtol=1e-13,
                    atol=1e-13,
                )
            )
            if is_refined:
                self.assertTrue(
                    torch.equal(adaptive.directions[packed], fine.directions[children])
                )
                self.assertTrue(
                    torch.equal(adaptive.solid_angles[packed], child_area)
                )
                self.assertTrue(
                    torch.equal(adaptive.target_values[packed], child_target)
                )
            else:
                self.assertTrue(
                    torch.equal(adaptive.directions[packed][0], base.directions[parent_index])
                )
                pooled = (
                    child_area[:, None] * child_target
                ).sum(dim=0) / child_area.sum()
                self.assertTrue(
                    torch.allclose(
                        adaptive.target_values[packed][0],
                        pooled,
                        rtol=1e-13,
                        atol=1e-13,
                    )
                )

    def test_construction_is_deterministic_and_chunk_size_invariant(self):
        field = _spike_field()
        first = build_adaptive_fitting_quadrature(
            field, 2, parent_chunk_size=1
        )
        second = build_adaptive_fitting_quadrature(
            field, 2, parent_chunk_size=13
        )
        third = build_adaptive_fitting_quadrature(
            field, 2, parent_chunk_size=13
        )
        for other in (second, third):
            self.assertTrue(
                torch.equal(
                    first.layout.refined_parent_mask,
                    other.layout.refined_parent_mask,
                )
            )
            self.assertTrue(
                torch.equal(first.layout.sample_offsets, other.layout.sample_offsets)
            )
            self.assertTrue(torch.equal(first.directions, other.directions))
            self.assertTrue(torch.equal(first.solid_angles, other.solid_angles))
            self.assertTrue(torch.equal(first.target_values, other.target_values))
            self.assertEqual(first.grid_id, other.grid_id)
            self.assertEqual(first.sha256, other.sha256)


class SphericalStatisticsTests(unittest.TestCase):
    def test_weighted_quantile_preserves_left_threshold_semantics(self):
        values = torch.tensor([100.0, 0.0, 3.0, 2.0], dtype=torch.float64)
        weights = torch.tensor([0.01, 0.89, 0.05, 0.05], dtype=torch.float64)
        self.assertEqual(float(weighted_quantile(values, weights, 0.0)), 0.0)
        self.assertEqual(float(weighted_quantile(values, weights, 0.89)), 0.0)
        self.assertEqual(float(weighted_quantile(values, weights, 0.90)), 2.0)
        self.assertEqual(float(weighted_quantile(values, weights, 0.95)), 3.0)
        self.assertEqual(float(weighted_quantile(values, weights, 1.0)), 100.0)

        order = torch.tensor([2, 0, 3, 1])
        original = weighted_quantile(values, weights, 0.95)
        permuted = weighted_quantile(values[order], 17.0 * weights[order], 0.95)
        self.assertTrue(torch.equal(original, permuted))

    def test_spherical_mean_and_integral_match_analytic_linear_signal(self):
        quadrature = build_training_quadrature(4, dtype=torch.float64)
        samples = _analytic_signal(quadrature.directions)
        integral = spherical_channel_integral(samples, quadrature.solid_angles)
        mean = spherical_channel_mean(samples, quadrature.solid_angles)
        expected_mean = torch.tensor((1.0, 2.0, 4.0), dtype=torch.float64)

        self.assertTrue(torch.allclose(mean, expected_mean, atol=1e-12, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                integral,
                mean * quadrature.solid_angles.sum(),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                integral,
                4.0 * math.pi * expected_mean,
                atol=1e-11,
                rtol=1e-12,
            )
        )

    def test_hdr_scale_uses_unbounded_q95_and_positive_support_fallback(self):
        samples = torch.tensor(
            [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]], dtype=torch.float64
        )
        dominant_dark_area = torch.tensor([0.99, 0.01], dtype=torch.float64)
        dominant_bright_area = dominant_dark_area.flip(0)
        self.assertEqual(compute_hdr_scale(samples, dominant_dark_area), 100.0)
        self.assertEqual(compute_hdr_scale(samples, dominant_bright_area), 100.0)
        self.assertEqual(
            compute_hdr_scale(torch.zeros_like(samples), dominant_dark_area),
            1.0,
        )
        for factor in (1.0e-8, 1.0e8):
            self.assertEqual(
                compute_hdr_scale(factor * samples, dominant_dark_area),
                factor * 100.0,
            )

    def test_finer_rotated_grid_improves_compact_emitter_integral(self):
        center = torch.tensor(
            [0.123, 0.456, 0.881], dtype=torch.float64
        )
        center = center / center.norm()
        concentration = 500.0
        exact = (
            2.0
            * math.pi
            * (-math.expm1(-2.0 * concentration))
            / concentration
        )
        relative_errors = []
        for subdivision in (3, 4):
            quadrature = build_evaluation_quadrature(
                subdivision, dtype=torch.float64
            )
            samples = torch.exp(
                concentration
                * (quadrature.directions @ center - 1.0)
            )
            estimate = float(
                (samples * quadrature.solid_angles).sum().item()
            )
            relative_errors.append(abs(estimate - exact) / exact)
        self.assertLess(relative_errors[1], relative_errors[0])
        self.assertGreater(relative_errors[1], 0.0)


if __name__ == "__main__":
    unittest.main()
