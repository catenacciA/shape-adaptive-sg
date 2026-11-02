"""Reconstruction metrics and the evaluation protocol."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from envmap.field import EnvmapField
from envmap.projection import build_projection
from envmap.quadrature import build_evaluation_quadrature
from fit.evaluation import SphericalEvaluationContext, build_spherical_evaluation_context, evaluate_reconstruction
from fit.metrics import compute_image_metrics, compute_spherical_metrics, metrics_to_json
from fit.pu21 import (PU21_MAX_NITS, PU21_MIN_NITS, PU21_PEAK, PU21_REFERENCE_WHITE_NITS,
                       pu21_encode_exr, pu21_encode_nits)


class MetricTests(unittest.TestCase):
    def test_spherical_relative_l2_keeps_solid_angles_and_chunking(self):
        p = torch.tensor([[0., 2., 4.], [1., 3., 5.]], dtype=torch.float64)
        t = torch.tensor([[1., 1., 2.], [2., 4., 8.]], dtype=torch.float64)
        w = torch.tensor([3., 1.], dtype=torch.float64)
        expected = (((p-t).square().mean(-1)*w).sum()/(t.square().mean(-1)*w).sum()).sqrt()
        for chunk in (1, 2):
            scores = compute_spherical_metrics(p, t, w*17, chunk_size=chunk)
            self.assertEqual(set(scores), {'relative_l2'})
            torch.testing.assert_close(scores['relative_l2'], expected)

    def test_pu21_is_uniform_raster_mse_and_same_encoding_for_ssim(self):
        t = torch.ones(32, 64, 3, dtype=torch.float64)
        p = t.clone(); p[:4] *= 3
        lpips_inputs = []
        def score(a, b):
            lpips_inputs.append((a.copy(), b.copy()))
            return .25
        with patch('fit.metrics._evaluate_lpips', side_effect=score):
            scores = compute_image_metrics(p, t)
            other = compute_image_metrics(torch.roll(p, 8, 0), t)
        expected = -10*math.log10(float((pu21_encode_exr(p)-pu21_encode_exr(t)).square().mean()))
        self.assertAlmostEqual(scores['pu21_psnr'], expected, places=12)
        self.assertAlmostEqual(scores['pu21_psnr'], other['pu21_psnr'], places=12)
        self.assertLess(scores['pu21_ssim'], 1.)
        self.assertEqual(scores['pu21_lpips'], .25)
        np.testing.assert_array_equal(lpips_inputs[0][0], pu21_encode_exr(p).numpy())
        np.testing.assert_array_equal(lpips_inputs[0][1], pu21_encode_exr(t).numpy())
        # Constant-image SSIM has zero variance: only the luminance factor remains.
        with patch('fit.metrics._evaluate_lpips', return_value=.25):
            constant = compute_image_metrics(t*2, t)
        a, b = float(pu21_encode_exr(2.)), float(pu21_encode_exr(1.))
        self.assertAlmostEqual(constant['pu21_ssim'], (2*a*b+.01**2)/(a*a+b*b+.01**2), places=10)

    def test_pu21_encoding_is_fixed_and_clamped(self):
        self.assertEqual(float(pu21_encode_exr(PU21_MAX_NITS / PU21_REFERENCE_WHITE_NITS)), 1.)
        self.assertEqual(float(pu21_encode_exr(0.)), float(pu21_encode_nits(PU21_MIN_NITS)) / PU21_PEAK)
        values = torch.logspace(-6, 3, 50, dtype=torch.float64)
        self.assertTrue(bool((pu21_encode_exr(values).diff() >= 0).all()))

    def test_pu21_metrics_clip_highlights_but_relative_l2_does_not(self):
        p, t = np.full((32, 64, 3), 100., np.float32), np.full((32, 64, 3), 200., np.float32)
        with patch('fit.metrics._evaluate_lpips', return_value=0.):
            scores = compute_image_metrics(p, t)
        self.assertEqual(scores['pu21_psnr'], math.inf)  # both above 10,000 cd/m^2
        self.assertAlmostEqual(scores['pu21_ssim'], 1.)
        self.assertEqual(float(compute_spherical_metrics(torch.from_numpy(p.reshape(-1, 3)),
            torch.from_numpy(t.reshape(-1, 3)), torch.ones(p.shape[0] * p.shape[1]))['relative_l2']), .5)

    def test_black_spherical_target_and_invalid_inputs(self):
        t = torch.zeros(2, 3)
        self.assertEqual(metrics_to_json(compute_spherical_metrics(t, t, [1., 2.])), {'relative_l2': 0.})
        self.assertIsNone(compute_spherical_metrics(t+1, t, [1., 2.])['relative_l2'])
        for value in (-1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                compute_spherical_metrics(t+value, t, [1., 2.])
            with self.assertRaises(ValueError):
                compute_image_metrics(torch.full((32, 64, 3), value), torch.ones(32, 64, 3))
        for w in ([0., 1.], [-1., 1.], [math.inf, 1.], [1.]):
            with self.assertRaises(ValueError):
                compute_spherical_metrics(t, t, w)
        with self.assertRaises(ValueError):
            compute_image_metrics(t, t)
        with self.assertRaises(ValueError):
            metrics_to_json({'pu21_ssim': math.nan})


class EvaluationContextTests(unittest.TestCase):
    def test_scale_is_fixed_across_quadrature_and_reusable_for_other_sources(self):
        from envmap.statistics import canonical_hdr_scale_reference
        values = torch.arange(1, 193).float().reshape(8, 8, 3)
        field = EnvmapField(values, build_projection('octahedral'))
        reference = canonical_hdr_scale_reference(field)
        a = build_spherical_evaluation_context(field, device='cpu', subdivision=0)
        b = build_spherical_evaluation_context(field, device='cpu', subdivision=3)
        self.assertEqual(a.protocol_metadata['scale_reference'], b.protocol_metadata['scale_reference'])
        other = EnvmapField(values * 10, field.projection)
        c = build_spherical_evaluation_context(other, device='cpu', subdivision=1, scale_reference=reference)
        self.assertEqual(c.hdr_scale, a.hdr_scale)
        self.assertNotEqual(canonical_hdr_scale_reference(other)['value'], c.hdr_scale)

    def test_negative_prediction_is_an_error(self):
        grid=build_evaluation_quadrature(0)
        context=SphericalEvaluationContext(grid,torch.ones(20,3),1.,{})
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            evaluate_reconstruction(context,render_directions=lambda d: -torch.ones(len(d),3),direction_chunk_size=7,feat_dim=3)


if __name__ == "__main__":
    unittest.main()
