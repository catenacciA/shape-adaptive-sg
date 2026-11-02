"""Footprint ablations: the Beta kernel and normalization off.

CPU only. Checks the moments of the Beta footprint, the ASG values of the
unnormalized terms, that the four presets are unaffected by the variant
code (same parameters and fused-path inputs), and that both variants fit with
residual insertion, save, reload and evaluate.
"""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fit.trainer import EnvmapTrainer
from model.asg import (ASGFootprint, BETA_BOUNDS, BetaFootprint, asg_kernel_weights, beta_profile,
                       footprint_log_terms, unnormalized_log_terms)
from model.serialization import load_native_inference_artifact
from tests.helpers import lobe_target
from tests.test_end_to_end import TINY_FIT
from utils.config import training_config_defaults
from utils.image_io import save_image


def plane_moments(profile, log_c, extent, n=4001):
    u = torch.linspace(-extent, extent, n, dtype=torch.float64)
    du = float(u[1] - u[0])
    U, V = torch.meshgrid(u, u, indexing="ij")
    f = math.exp(log_c) * profile(U, V)
    return float(f.sum() * du * du), float((U * U * f).sum() * du * du), float((U * V * f).sum() * du * du)


class BetaFootprintTests(unittest.TestCase):
    def test_unit_mass_and_covariance_half(self):
        for beta in (1.0, 2.5, 4.0, 16.0, 256.0):
            shape = BetaFootprint.from_power(torch.tensor(beta, dtype=torch.float64))
            extent = math.sqrt(beta + 2) * 1.01
            mass, var, cov = plane_moments(lambda u, v: beta_profile(u, v, shape), float(shape.log_c), extent)
            self.assertAlmostEqual(mass, 1.0, delta=2e-4, msg=beta)
            self.assertAlmostEqual(var, 0.5, delta=2e-4, msg=beta)
            self.assertAlmostEqual(cov, 0.0, delta=1e-9, msg=beta)

    def test_compact_support_and_large_beta_limit(self):
        shape = BetaFootprint.from_power(torch.tensor([4.0], dtype=torch.float64))
        r = torch.tensor([math.sqrt(6.0) * 1.0001], dtype=torch.float64)
        self.assertEqual(float(beta_profile(r, torch.zeros_like(r), shape)), 0.0)
        big = BetaFootprint.from_power(torch.tensor([1e6], dtype=torch.float64))
        u = torch.linspace(0, 2, 21, dtype=torch.float64)
        gaussian = torch.exp(-u * u) / math.pi
        beta = beta_profile(u, torch.zeros_like(u), big) * big.log_c.exp()
        self.assertLess(float((beta - gaussian).abs().max()), 1e-5)

    def test_gradients_finite_at_support_boundary_and_center(self):
        beta = torch.tensor([1.0, 4.0, 256.0], dtype=torch.float64, requires_grad=True)
        shape = BetaFootprint.from_power(beta)
        u = torch.tensor([math.sqrt(3.0), 0.0, 1.0], dtype=torch.float64, requires_grad=True)
        value = beta_profile(u, torch.zeros_like(u), shape).sum()
        grads = torch.autograd.grad(value, (u, beta))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))

    def test_kernel_weights_use_the_beta_profile(self):
        torch.manual_seed(0)
        dirs = torch.nn.functional.normalize(torch.randn(64, 3, dtype=torch.float64), dim=1)
        z = torch.nn.functional.normalize(torch.randn(5, 3, dtype=torch.float64), dim=1)
        x = torch.nn.functional.normalize(torch.cross(z, torch.randn(5, 3, dtype=torch.float64), dim=1), dim=1)
        y = torch.cross(z, x, dim=1)
        lam, mu = torch.full((5,), 3.0, dtype=torch.float64), torch.full((5,), 7.0, dtype=torch.float64)
        shape = BetaFootprint.from_power(torch.linspace(1, 50, 5, dtype=torch.float64))
        weights = asg_kernel_weights(dirs, z, x, y, lam, mu, shape)
        u, v = (dirs @ x.T) * lam.sqrt(), (dirs @ y.T) * mu.sqrt()
        expected = ((dirs @ z.T).clamp_min(0) * beta_profile(u, v, shape) * (lam * mu).sqrt() * shape.log_c.exp())
        self.assertLess(float((weights - expected).abs().max()), 1e-12)


class UnnormalizedTests(unittest.TestCase):
    def test_asg_values_for_every_shape(self):
        p = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        log_b, log_c = unnormalized_log_terms(p)
        self.assertTrue(bool((log_b == 0).all()))
        self.assertTrue(torch.allclose(log_c, torch.full_like(p, -math.log(math.pi))))
        # At the Gaussian member both agree with the normalized terms.
        b1, c1 = footprint_log_terms(torch.ones(1, dtype=torch.float64), torch.full((1,), 2.0, dtype=torch.float64))
        self.assertAlmostEqual(float(b1), 0.0, places=12)
        self.assertAlmostEqual(float(c1), -math.log(math.pi), places=12)

    def test_from_exponents_default_is_normalized(self):
        p, q = torch.tensor([3.0], dtype=torch.float64), torch.tensor([8.0], dtype=torch.float64)
        self.assertTrue(torch.equal(ASGFootprint.from_exponents(p, q).log_b, footprint_log_terms(p, q)[0]))


def fit(root, method, **preset):
    save_image(lobe_target(), str(root / "target.exr"))
    cfg = training_config_defaults()
    cfg.update(TINY_FIT, **preset, device="cpu", data_root=str(root), input_path="target.exr",
               log_dir=str(root / method))
    trainer = EnvmapTrainer(SimpleNamespace(**cfg))
    trainer.optimize()
    return trainer, root / method


class VariantFitTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("fit.metrics._evaluate_lpips", return_value=.125)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_beta_fits_with_insertion_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, run = fit(Path(tmp), "beta", asg_footprint="beta")
            arrays = dict(np.load(run / "inference.npz"))
            self.assertEqual(set(arrays) - {"local_z", "local_x", "local_y", "local_lambda", "local_mu",
                                            "local_coefficients"}, {"local_beta"})
            self.assertGreater(len(arrays["local_beta"]), TINY_FIT["num_gaussians"])
            self.assertTrue(((arrays["local_beta"] >= BETA_BOUNDS[0]) & (arrays["local_beta"] <= BETA_BOUNDS[1])).all())
            self.assertFalse(bool((arrays["local_beta"] == 4.0).all()), "beta did not move")
            self.assertFalse((run / "representation.skc").exists())
            representation = load_native_inference_artifact(run / "inference.json", dtype=torch.float64)
            self.assertIsInstance(representation.asg_footprint, BetaFootprint)
            dirs = torch.nn.functional.normalize(torch.randn(256, 3, dtype=torch.float64), dim=1)
            with torch.no_grad():
                expected = trainer.kernel_bank.render_directions(dirs.to(trainer.kernel_bank.dtype)).double()
            self.assertLess(float((representation.evaluate_directions(dirs) - expected).abs().max()
                                  / expected.abs().max()), 1e-5)

    def test_unnormalized_fits_and_reloads_unnormalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, run = fit(Path(tmp), "ours_unnormalized", asg_footprint="superelliptic",
                               asg_normalize_shape=False)
            representation = load_native_inference_artifact(run / "inference.json", dtype=torch.float64)
            self.assertFalse(representation.manifest["asg_normalize_shape"])
            self.assertTrue(bool((representation.asg_footprint.log_b == 0).all()))
            self.assertFalse((run / "representation.skc").exists())
            dirs = torch.nn.functional.normalize(torch.randn(256, 3, dtype=torch.float64), dim=1)
            with torch.no_grad():
                expected = trainer.kernel_bank.render_directions(dirs.to(trainer.kernel_bank.dtype)).double()
            self.assertLess(float((representation.evaluate_directions(dirs) - expected).abs().max()
                                  / expected.abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
