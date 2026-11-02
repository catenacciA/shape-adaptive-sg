import math

import torch

from fit.pq import pq_encode_exr
from utils.fusion import fused_on_cuda


def charbonnier(values, eps):
    return torch.sqrt(values * values + eps * eps)


@fused_on_cuda(training_only=True)
def pq_charbonnier_terms(pred, target, eps):
    """Channel-mean Charbonnier terms of the clamped PQ residual."""
    residual = pq_encode_exr(pred.clamp_min(0.0)) - pq_encode_exr(target.clamp_min(0.0))
    return (charbonnier(residual, eps) - eps).mean(dim=-1)


class SphericalHDRLoss:
    """Quadrature-weighted PQ-domain Charbonnier reconstruction loss (Eq. 7)."""

    def __init__(self, charbonnier_eps=1e-3):
        self.charbonnier_eps = float(charbonnier_eps)
        if not math.isfinite(self.charbonnier_eps) or self.charbonnier_eps <= 0.0:
            raise ValueError("charbonnier_eps must be finite and positive.")

    def _transform(self, values, *, return_derivative=False):
        return pq_encode_exr(values, return_derivative=return_derivative)

    def transform_radiance(self, values):
        """PQ-encode nonnegative physical radiance (analytic extension above peak)."""
        values = torch.as_tensor(values)
        if bool((values < 0.0).any().item()):
            raise ValueError("HDR transform requires nonnegative radiance")
        return self._transform(values)

    def _validate_inputs(self, pred, target, quadrature_weights):
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "spherical HDR loss expects [M, C] sample tensors, got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}"
            )
        if pred.shape != target.shape:
            raise ValueError(
                "prediction and target samples must have matching shapes, got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}"
            )
        if pred.shape[0] <= 0 or pred.shape[1] <= 0:
            raise ValueError(
                "spherical HDR loss requires at least one sample and channel"
            )
        if pred.device != target.device:
            raise ValueError("prediction and target samples must be on the same device")
        if not bool(torch.isfinite(pred).all().item()):
            raise ValueError("prediction samples contain non-finite values")
        if not bool(torch.isfinite(target).all().item()):
            raise ValueError("target samples contain non-finite values")

        weights = torch.as_tensor(
            quadrature_weights,
            device=pred.device,
            dtype=pred.dtype,
        )
        if weights.ndim != 1 or weights.shape[0] != pred.shape[0]:
            raise ValueError(
                "quadrature weights must have shape [M] matching the samples, got "
                f"{tuple(weights.shape)} for {pred.shape[0]} samples"
            )
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("quadrature weights contain non-finite values")
        if bool((weights < 0.0).any().item()):
            raise ValueError("quadrature weights must be non-negative")
        if not bool((weights > 0.0).any().item()):
            raise ValueError(
                "quadrature weights must contain at least one positive value"
            )
        return weights

    def _residual(self, pred, target, *, return_derivative=False):
        transformed = self._transform(pred.clamp_min(0.0), return_derivative=return_derivative)
        reference = self._transform(target.clamp_min(0.0))
        if return_derivative:
            value, derivative = transformed
            return value - reference, torch.where(pred >= 0.0, derivative, 0.0)
        return transformed - reference

    def per_sample(self, pred, target):
        """Channel-mean reconstruction terms, allowing broadcast batch axes."""
        return pq_charbonnier_terms(pred, target, self.charbonnier_eps)

    def __call__(self, pred, target, quadrature_weights, *, validate=True):
        """validate=False skips the synchronizing input checks for callers
        that validate fixed targets once and check predictions themselves."""
        weights = (self._validate_inputs(pred, target, quadrature_weights)
                   if validate else quadrature_weights)
        return (self.per_sample(pred, target) * weights).mean()

    @torch.no_grad()
    def prediction_gradient(self, pred, target, quadrature_weights):
        """Analytic dJ/dpred, including weights, channel/sample means and clamps.

        For r = T(pred) - T(target), dJ/dpred =
        weights / (M*C) * r / sqrt(r*r + eps*eps) * T'(pred).
        Clamp derivatives follow the same boundary conventions as autograd.
        """
        weights = self._validate_inputs(pred, target, quadrature_weights)
        residual, derivative = self._residual(pred, target, return_derivative=True)
        return (
            residual / charbonnier(residual, self.charbonnier_eps) * derivative
            * weights[:, None] / pred.numel()
        )
