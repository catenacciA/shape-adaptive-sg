"""Fused CUDA paths must reproduce the eager CPU reference (skipped without CUDA).

The CPU path is the unmodified reference implementation; on CUDA the same
functions run through torch.compile. These tests compare both on tiny inputs.
"""
import copy
import math
import unittest

import torch

from fit.evaluation import evaluate_directions_batched
from fit.loss import SphericalHDRLoss
from fit.rowwise_adam import RowwiseAdam
from model.asg import ASGFootprint, asg_kernel_weights, render_mixture, render_parameters
from tests import test_end_to_end
from tests.helpers import superelliptic_bank

CUDA = torch.device("cuda") if torch.cuda.is_available() else None


def raw_rows(dtype, count=29):
    """Raw kernel rows spanning the concentration and shape bounds."""
    gen = torch.Generator().manual_seed(11)
    uniform = lambda low, high: low + (high - low) * torch.rand(count, generator=gen, dtype=dtype)
    log_p = uniform(0.0, math.log(4.0))
    log_q_ratio = uniform(math.log(1.25 / 2), math.log(8.0))
    log_p[0] = log_q_ratio[0] = 0.0  # the Gaussian member (p, q) = (1, 2)
    return (torch.randn(count, 3, generator=gen, dtype=dtype), uniform(-math.pi, math.pi),
            uniform(math.log(5.0), math.log(2e4)), uniform(math.log(5.0), math.log(2e4)),
            torch.randn(count, 3, generator=gen, dtype=dtype), log_p, log_q_ratio)


def query_directions(raw, axes=True):
    """Random directions and kernel centers, optionally with tangent-axis points.

    On a tangent axis, v = dot(w, y) is a rounding-level number. For q < 2 the
    exact gradient of |v|^q is only Hoelder continuous there, ~|v|^(q-1), so
    projection roundoff (matmul vs pointwise) shifts gradients by ~1e-5 even
    in float64. Gradients are therefore compared tightly without these points.
    """
    gen = torch.Generator().manual_seed(12)
    z, x, y = render_parameters(*raw[:5])[:3]
    dirs = (torch.randn(256, 3, generator=gen, dtype=z.dtype), z)
    if axes:
        dirs += (z + 0.03 * x, z + 0.03 * y, z - 0.2 * x)
    return torch.nn.functional.normalize(torch.cat(dirs), dim=1)


def render(dirs, raw, superelliptic, weights_only=False):
    z, x, y, lambda_, mu, coefficients, *shape = render_parameters(*raw[: 7 if superelliptic else 5])
    footprint = ASGFootprint(*shape) if superelliptic else None
    if weights_only:
        return asg_kernel_weights(dirs, z, x, y, lambda_, mu, footprint)
    return render_mixture(dirs, z, x, y, lambda_, mu, coefficients, footprint)


def assert_close_scaled(test, actual, expected, rtol, name):
    """Elementwise closeness relative to the largest reference magnitude."""
    error = float((actual.cpu() - expected).abs().max() / expected.abs().max())
    test.assertLessEqual(error, rtol, f"{name}: scaled error {error:.2e}")


class FusionTestCase(unittest.TestCase):
    def setUp(self):
        # Compiled variants accumulate per process; a training run needs few,
        # but this module exercises many and must not hit the recompile limit.
        torch._dynamo.reset()


@unittest.skipUnless(CUDA, "requires CUDA")
class FusedRendererTests(FusionTestCase):
    def test_values_and_gradients_match_eager_reference(self):
        cases = ((torch.float64, False, 1e-9), (torch.float64, True, 1e-4), (torch.float32, True, 2e-4))
        for dtype, axes, rtol in cases:
            for superelliptic in (False, True):
                with self.subTest(dtype=dtype, axes=axes, superelliptic=superelliptic):
                    raw = [value.requires_grad_() for value in raw_rows(dtype)]
                    raw_cuda = [value.detach().to(CUDA).requires_grad_() for value in raw]
                    dirs = query_directions([value.detach() for value in raw], axes)
                    probe = torch.randn(len(dirs), 3, generator=torch.Generator().manual_seed(3), dtype=dtype)
                    expected = render(dirs, raw, superelliptic)
                    actual = render(dirs.to(CUDA), raw_cuda, superelliptic)
                    assert_close_scaled(self, actual.detach(), expected.detach(), rtol, "render")
                    used = 7 if superelliptic else 5
                    grads = torch.autograd.grad((expected * probe).sum(), raw[:used])
                    grads_cuda = torch.autograd.grad((actual * probe.to(CUDA)).sum(), raw_cuda[:used])
                    for index, (value, reference) in enumerate(zip(grads_cuda, grads)):
                        assert_close_scaled(self, value, reference, rtol, f"gradient {index}")

    def test_inference_mode_batched_render(self):
        raw = raw_rows(torch.float32)
        dirs = query_directions(raw)
        expected = render(dirs, raw, True)
        raw_cuda = [value.to(CUDA) for value in raw]
        actual = evaluate_directions_batched(
            lambda chunk: render(chunk, raw_cuda, True), dirs.to(CUDA), chunk_size=97, feat_dim=3
        )
        assert_close_scaled(self, actual, expected, 2e-4, "inference render")

    def test_proposal_weights_match_eager_reference(self):
        raw = raw_rows(torch.float32)
        dirs = query_directions(raw)
        for superelliptic in (False, True):
            with self.subTest(superelliptic=superelliptic), torch.no_grad():
                expected = render(dirs, raw, superelliptic, weights_only=True)
                actual = render(dirs.to(CUDA), [value.to(CUDA) for value in raw], superelliptic, weights_only=True)
                assert_close_scaled(self, actual, expected, 2e-4, "kernel weights")

    def test_loss_terms_and_gradient_match_eager_reference(self):
        gen = torch.Generator().manual_seed(4)
        loss = SphericalHDRLoss()
        for dtype, rtol in ((torch.float64, 1e-9), (torch.float32, 5e-5)):
            with self.subTest(dtype=dtype):
                pred = torch.rand(64, 3, generator=gen, dtype=dtype).pow(4) * 500
                pred[0] = 0.0
                pred[1, 0] = -0.5
                target = torch.rand(64, 3, generator=gen, dtype=dtype) * 30
                weights = torch.rand(64, generator=gen, dtype=dtype)
                pred.requires_grad_()
                pred_cuda = pred.detach().to(CUDA).requires_grad_()
                expected = loss(pred, target, weights)
                actual = loss(pred_cuda, target.to(CUDA), weights.to(CUDA), validate=False)
                assert_close_scaled(self, actual.detach(), expected.detach(), rtol, "loss")
                gradient, = torch.autograd.grad(expected, pred)
                gradient_cuda, = torch.autograd.grad(actual, pred_cuda)
                assert_close_scaled(self, gradient_cuda, gradient, rtol, "loss gradient")
                # Broadcast candidate scoring as used by densification.
                candidates = pred.detach()[:, None, :] * torch.tensor([0.5, 1.0, 2.0], dtype=dtype)[:, None]
                assert_close_scaled(
                    self, loss.per_sample(candidates.to(CUDA), target.to(CUDA)[:, None, :]),
                    loss.per_sample(candidates, target[:, None, :]), rtol, "broadcast terms",
                )


@unittest.skipUnless(CUDA, "requires CUDA")
class FusedUpdateTests(FusionTestCase):
    def test_update_matches_eager_reference_without_recompiling(self):
        # In float32, Adam's bias correction 1 - 0.999**t cancels: a one-ulp
        # difference between CPU and CUDA powf moves early steps by up to ~3e-5.
        for dtype, rtol in ((torch.float64, 1e-9), (torch.float32, 1e-4)):
            with self.subTest(dtype=dtype):
                torch._dynamo.reset()
                cpu, cuda = self._run_updates(dtype)
                for (name, expected), actual in zip(cpu.named_parameters(), cuda.parameters()):
                    assert_close_scaled(self, actual.detach(), expected.detach(), rtol, name)

    @staticmethod
    def _run_updates(dtype):
        cpu, _ = superelliptic_bank(dtype=dtype, count=40)
        with torch.no_grad():
            cpu.residual_center_max_angle[:20] = 0.02  # inserted rows: active trust region
        cuda = copy.deepcopy(cpu).to(CUDA)
        banks = [(bank, RowwiseAdam(bank.get_optimizer_param_groups())) for bank in (cpu, cuda)]
        gen = torch.Generator().manual_seed(7)
        for step in range(6):
            grads = [0.1 * torch.randn(p.shape, generator=gen, dtype=dtype) for p in cpu.parameters()]
            # The first step compiles every variant; a decaying rate must not recompile.
            with torch._dynamo.config.patch(error_on_recompile=step > 0):
                for bank, optimizer in banks:
                    for parameter, gradient in zip(bank.parameters(), grads):
                        parameter.grad = gradient.to(parameter.device)
                    for group in optimizer.param_groups:
                        group["lr"] = 0.02 / (step + 1)
                    previous = bank.centers().detach().clone()
                    optimizer.step()
                    bank.project_parameters_()
                    bank.preserve_tangent_chart_(previous)
        return cpu, cuda


@unittest.skipUnless(CUDA, "requires CUDA")
class CudaPresetFitTests(FusionTestCase, test_end_to_end.PresetFitTests):
    """The preset fits on CUDA; the unindexed device is the one typed on the CLI."""

    device = "cuda"


if __name__ == "__main__":
    unittest.main()
