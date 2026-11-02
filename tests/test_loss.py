"""PQ transform and the quadrature-weighted Charbonnier objective (Eq. 7)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fit.densification import (
    _objective_values_for_batch_scalars, proposal_gradient_products, reconstruction_objective_gradient,
)
from fit.loss import SphericalHDRLoss
from fit.pq import PQ_EXR_NOMINAL_CEILING, pq_encode_exr
from fit.trainer import EnvmapTrainer
from model.asg import render_asg_proposal_aggregate
from tests.helpers import uniform_allocation, asg_bank, tiny_config


class _RegularizedKernelBank:
    def __init__(self, value):
        self.value = float(value)
        self.calls = []

    def regularization_loss(self, step):
        self.calls.append(int(step))
        return torch.tensor(self.value)


def _weights_for(samples):
    return torch.ones(samples.shape[0], device=samples.device, dtype=samples.dtype)


class PQTransformTests(unittest.TestCase):
    def test_pq_reference_values_and_monotonic_extension(self):
        values = torch.tensor([0.0, 1.0, 10.0, PQ_EXR_NOMINAL_CEILING, 100.0], dtype=torch.float64)
        encoded = pq_encode_exr(values)
        self.assertAlmostEqual(float(encoded[0]), 7.309559025783966e-7, places=15)
        self.assertAlmostEqual(float(encoded[1]), 0.5806888810416109, places=12)  # 203 cd/m^2
        self.assertAlmostEqual(float(encoded[3]), 1.0, places=12)  # 10,000 cd/m^2
        self.assertTrue(bool((encoded[1:] > encoded[:-1]).all()))
        self.assertGreater(float(encoded[-1]), 1.0)  # training extends beyond the nominal peak
        clipped = pq_encode_exr(values, clip_to_nominal=True)
        self.assertAlmostEqual(float(clipped[-1]), 1.0, places=12)

    def test_metric_clipping_has_zero_gradient_above_nominal_peak(self):
        values = torch.tensor([2.0 * PQ_EXR_NOMINAL_CEILING], requires_grad=True)
        gradient, = torch.autograd.grad(pq_encode_exr(values, clip_to_nominal=True).sum(), values)
        self.assertEqual(float(gradient), 0.0)

    def test_analytic_pq_derivative_matches_autograd(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                tiny = torch.finfo(dtype).tiny * PQ_EXR_NOMINAL_CEILING
                values = torch.tensor([0., tiny / 2, tiny, 2 * tiny, 1e-6, 1., PQ_EXR_NOMINAL_CEILING,
                                       2 * PQ_EXR_NOMINAL_CEILING], dtype=dtype, requires_grad=True)
                encoded, derivative = pq_encode_exr(values, return_derivative=True)
                expected, = torch.autograd.grad(encoded.sum(), values)
                self.assertTrue(bool(torch.isfinite(derivative).all()))
                # Exactly at the smallest normal float, autograd's clamp convention
                # differs across torch releases; compare everywhere else.
                compare = torch.ones(len(values), dtype=torch.bool)
                compare[2] = False
                torch.testing.assert_close(derivative[compare], expected[compare],
                                           rtol=4e-5 if dtype == torch.float32 else 2e-12, atol=1e-12)


class SphericalHDRLossTests(unittest.TestCase):
    def test_quadrature_weighted_per_channel_charbonnier(self):
        pred = torch.tensor([[1.0, 8.0, 27.0], [0.5, 2.0, 16.0]], dtype=torch.float64)
        target = torch.tensor([[0.0, 1.0, 8.0], [0.25, 1.0, 4.0]], dtype=torch.float64)
        weights = torch.tensor([0.25, 2.0], dtype=torch.float64)
        eps = 1e-3
        actual = SphericalHDRLoss(charbonnier_eps=eps)(pred, target, weights)
        residual = pq_encode_exr(pred) - pq_encode_exr(target)
        per_sample = (torch.sqrt(residual.square() + eps ** 2) - eps).mean(dim=-1)
        torch.testing.assert_close(actual, (per_sample * weights).mean(), rtol=1e-12, atol=0)

    def test_minibatch_weights_are_not_renormalized(self):
        pred = torch.tensor([[1.0], [8.0]])
        target = torch.zeros_like(pred)
        weights = torch.tensor([0.1, 0.2])
        loss_fn = SphericalHDRLoss()
        actual = loss_fn(pred, target, weights)
        residual = loss_fn._transform(pred) - loss_fn._transform(target)
        per_sample = (torch.sqrt(residual.pow(2) + loss_fn.charbonnier_eps**2) - loss_fn.charbonnier_eps).mean(dim=-1)
        self.assertTrue(torch.allclose(actual, (per_sample * weights).mean()))
        self.assertFalse(torch.allclose(actual, (per_sample * weights).sum() / weights.sum()))

    def test_zero_near_zero_and_above_ceiling_inputs_have_finite_gradients(self):
        exact = torch.zeros(2, 3, requires_grad=True)
        exact_loss = SphericalHDRLoss()(exact, torch.zeros_like(exact), _weights_for(exact))
        exact_loss.backward()
        self.assertEqual(float(exact_loss.detach()), 0.0)
        self.assertTrue(torch.equal(exact.grad, torch.zeros_like(exact.grad)))

        values = torch.tensor([[0.0, 1e-12, 1e-6, 2.0 * PQ_EXR_NOMINAL_CEILING]], requires_grad=True)
        target = torch.tensor([[1e-6, 0.0, 2e-6, PQ_EXR_NOMINAL_CEILING]])
        loss = SphericalHDRLoss()(values, target, _weights_for(values))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertNotEqual(float(values.grad[0, 3]), 0.0)  # analytic extension keeps a gradient

    def test_trainer_adds_kernel_regularization_exactly_once(self):
        trainer = EnvmapTrainer.__new__(EnvmapTrainer)
        torch.nn.Module.__init__(trainer)
        trainer.loss_fn = SphericalHDRLoss()
        trainer.kernel_bank = _RegularizedKernelBank(0.125)
        trainer.step = 3

        pred = torch.tensor([[0.5, 1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[0.25, 1.5, 1.0]])
        weights = torch.tensor([1.0])
        expected_reconstruction = trainer.loss_fn(pred, target, weights)
        total = trainer._compute_loss(pred, target, weights)

        self.assertTrue(torch.allclose(total, expected_reconstruction + 0.125))
        self.assertEqual(trainer.kernel_bank.calls, [3])
        expected_grad = torch.autograd.grad(expected_reconstruction, pred, retain_graph=True)[0]
        total.backward()
        self.assertTrue(torch.allclose(pred.grad, expected_grad))

    def test_non_finite_prediction_aborts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = tiny_config(Path(directory), "cfgs/ours.yaml", "nan", num_gaussians=2,
                              densify_enabled=False, densify_max_gaussians=None, num_steps=1,
                              batch_size=8, log_level="ERROR")
            trainer = EnvmapTrainer(SimpleNamespace(**cfg))
            with torch.no_grad():
                trainer.kernel_bank.radiance_coeff_raw[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                trainer._optimization_step(1)
            self.assertEqual(trainer.completed_step, 0)

    def test_invalid_sample_inputs_are_rejected(self):
        loss_fn = SphericalHDRLoss()
        pred = torch.ones(2, 3)
        target = torch.zeros_like(pred)
        with self.assertRaises(TypeError):
            loss_fn(pred, target)
        with self.assertRaisesRegex(ValueError, r"\[M, C\]"):
            loss_fn(pred.unsqueeze(0), target.unsqueeze(0), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            loss_fn(pred, target[:1], torch.ones(2))
        with self.assertRaisesRegex(ValueError, r"shape \[M\]"):
            loss_fn(pred, target, torch.ones(2, 1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            invalid_pred = pred.clone()
            invalid_pred[0, 0] = float("nan")
            loss_fn(invalid_pred, target, torch.ones(2))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            loss_fn(pred, target, torch.tensor([1.0, -0.5]))
        with self.assertRaisesRegex(ValueError, "positive"):
            loss_fn(pred, target, torch.zeros(2))


class ObjectiveDerivativeTests(unittest.TestCase):
    """The analytic dJ/dprediction used to rank insertion candidates."""

    def cases(self):
        for dtype in (torch.float32, torch.float64):
            for eps in (1e-3, 0.007):
                yield dtype, SphericalHDRLoss(charbonnier_eps=eps)

    def test_analytic_derivative_matches_autograd_across_hdr_range_and_clamps(self):
        for dtype, loss in self.cases():
            with self.subTest(dtype=dtype, eps=loss.charbonnier_eps):
                values = torch.tensor([-1., 0., 1e-20, 1e-9, 0.001, 0.1, 1., 5., PQ_EXR_NOMINAL_CEILING,
                                       100., 1e4, 1e8], dtype=dtype)
                pred = values[:, None].expand(-1, 3).clone().requires_grad_(True)
                target = pred.detach().abs().roll(3, dims=0) * torch.tensor([0.4, 1.0, 2.0], dtype=dtype)
                weights = torch.linspace(0., 2., len(pred), dtype=dtype)
                expected, = torch.autograd.grad(loss(pred, target, weights), pred)
                actual = loss.prediction_gradient(pred, target, weights)
                torch.testing.assert_close(actual, expected, rtol=4e-5 if dtype == torch.float32 else 2e-12, atol=1e-12)
                self.assertFalse(actual.requires_grad)

    def test_chunked_double_derivative_matches_full_objective_without_autograd(self):
        for dtype, loss in self.cases():
            generator = torch.Generator().manual_seed(123)
            pred = torch.randn(19, 3, generator=generator, dtype=dtype).exp()
            target = torch.randn(19, 3, generator=generator, dtype=dtype).exp()
            weights = torch.rand(19, generator=generator, dtype=dtype)
            reference = pred.double().requires_grad_(True)
            expected, = torch.autograd.grad(loss(reference, target.double(), weights), reference)
            for chunk in (1, 7, 32):
                with patch('torch.autograd.grad', side_effect=AssertionError('no autograd for proposal ranking')):
                    actual = reconstruction_objective_gradient(pred, target, weights, loss, chunk)
                torch.testing.assert_close(actual, expected.to(dtype), rtol=1e-6 if dtype == torch.float32 else 2e-12,
                                           atol=1e-14)

    def test_exact_fit_has_zero_derivative(self):
        for dtype, loss in self.cases():
            values = torch.tensor([[0., 1., 100.]], dtype=dtype)
            self.assertEqual(torch.count_nonzero(loss.prediction_gradient(values, values, values.new_ones(1))), 0)

    def test_broadcast_scalar_scoring_uses_same_loss_terms(self):
        for dtype, loss in self.cases():
            pred = torch.tensor([[-1., 0., 1.], [10., 50., 100.], [0.1, 0.3, 5.]], dtype=dtype)
            target = pred.abs().flip(0)
            weights = pred.new_tensor([0.3, 2., 0.8])
            delta = torch.ones_like(pred)
            scalars = pred.new_tensor([0., 0.1, 1., 10.])
            values = _objective_values_for_batch_scalars(pred, target, weights, delta, scalars, loss,
                                                         sample_chunk_size=2, scalar_chunk_size=2)
            expected = torch.stack([loss((pred + scalar * delta).double(), target.double(), weights) for scalar in scalars])
            torch.testing.assert_close(values, expected, rtol=2e-12, atol=1e-14)

    def test_proposal_scores_match_objective_directional_derivatives(self):
        alloc = uniform_allocation()
        dirs = alloc.directions
        bank = asg_bank(alloc)
        proposal = bank.initialize_densification_proposals(
            dirs[:1], dirs.new_tensor([0.3]), dirs.new_tensor([0.31]), dirs.new_tensor([[0.1, 0.3, 0.5]]),
        )
        contribution = render_asg_proposal_aggregate(dirs, proposal)
        pred = dirs.new_full((len(dirs), 3), 1.)
        weights = dirs.new_ones(len(dirs))
        loss = SphericalHDRLoss()
        for sign in (-1, 1):
            target = pred + sign * 0.1
            gradient = reconstruction_objective_gradient(pred, target, weights, loss)
            score = proposal_gradient_products(dirs, gradient, proposal)[0]
            eps = 1e-5
            finite_difference = (loss(pred + eps * contribution, target, weights)
                                 - loss(pred - eps * contribution, target, weights)) / (2 * eps)
            torch.testing.assert_close(score, finite_difference, rtol=1e-7, atol=1e-10)
            self.assertLess(float(score) * sign, 0.)


if __name__ == "__main__":
    unittest.main()
