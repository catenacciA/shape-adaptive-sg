import math
import hashlib

import torch


def compute_luminance(values):
    """Compute scalar luminance from tensor samples with channels last."""
    values = torch.as_tensor(values)
    if values.ndim < 1 or values.shape[-1] <= 0:
        raise ValueError("luminance expects a non-empty channel dimension")
    channels = values.shape[-1]
    if channels >= 3:
        return (
            0.2126 * values[..., 0]
            + 0.7152 * values[..., 1]
            + 0.0722 * values[..., 2]
        )
    if channels == 1:
        return values[..., 0]
    return values.mean(dim=-1)


def weighted_quantile(values, weights, quantile):
    """Smallest value whose cumulative weight reaches the quantile fraction."""
    values = torch.as_tensor(values).detach().reshape(-1)
    weights = torch.as_tensor(
        weights, device=values.device, dtype=values.dtype
    ).detach().reshape(-1)
    if values.numel() != weights.numel():
        raise ValueError(
            f"weighted quantile size mismatch: {values.numel()} values vs "
            f"{weights.numel()} weights"
        )
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any().item()):
        return values.new_tensor(0.0)
    values = values[valid]
    weights = weights[valid]
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = torch.cumsum(weights, dim=0)
    q = float(quantile)
    if not math.isfinite(q):
        raise ValueError("weighted quantile must be finite")
    threshold = cumulative[-1] * min(max(q, 0.0), 1.0)
    index = torch.searchsorted(cumulative, threshold, right=False).clamp(
        max=values.numel() - 1
    )
    return values[index]


def sanitize_nonnegative(values):
    """Replace non-finite values by zero and clamp to nonnegative values."""
    values = torch.as_tensor(values)
    return torch.where(
        torch.isfinite(values), values, torch.zeros_like(values)
    ).clamp_min(0.0)


def robust_unit_component(values, solid_angles, quantile):
    """Normalize a nonnegative signal by its positive weighted quantile."""
    values = sanitize_nonnegative(values)
    positive = values > 0.0
    if not bool(positive.any().item()):
        return torch.zeros_like(values)
    scale = weighted_quantile(
        values[positive],
        torch.as_tensor(solid_angles, device=values.device, dtype=values.dtype)
        .reshape(-1)[positive],
        quantile,
    ).clamp(min=1e-12)
    return (values / scale).clamp(0.0, 1.0)


def _validate_spherical_samples(samples, solid_angles):
    samples = torch.as_tensor(samples)
    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError(
            "spherical statistics expect samples with shape [N, C], got "
            f"{tuple(samples.shape)}"
        )
    if not torch.is_floating_point(samples):
        raise ValueError("spherical statistic samples must be floating point")
    solid_angles = torch.as_tensor(
        solid_angles, device=samples.device, dtype=samples.dtype
    )
    if solid_angles.ndim != 1 or solid_angles.shape[0] != samples.shape[0]:
        raise ValueError(
            "spherical solid angles must have shape [N] matching the samples, got "
            f"{tuple(solid_angles.shape)} for {samples.shape[0]} samples"
        )
    if not bool(torch.isfinite(samples).all().item()):
        raise ValueError("spherical statistic samples contain non-finite values")
    if not bool(torch.isfinite(solid_angles).all().item()):
        raise ValueError("spherical solid angles contain non-finite values")
    if bool((solid_angles < 0.0).any().item()):
        raise ValueError("spherical solid angles must be non-negative")
    if not bool((solid_angles > 0.0).any().item()):
        raise ValueError("spherical solid angles must contain a positive value")
    return samples, solid_angles


def spherical_channel_integral(samples, solid_angles):
    """Integrate channels from aligned spherical samples and raw areas."""
    samples, solid_angles = _validate_spherical_samples(samples, solid_angles)
    return (samples * solid_angles.unsqueeze(-1)).sum(dim=0)


def spherical_channel_mean(samples, solid_angles):
    """Compute the area-normalized channel mean of spherical samples."""
    samples, solid_angles = _validate_spherical_samples(samples, solid_angles)
    integral = (samples * solid_angles.unsqueeze(-1)).sum(dim=0)
    return integral / solid_angles.sum()


def compute_hdr_scale(
    samples,
    solid_angles,
    quantile=0.95,
):
    """Compute a radiometrically scale-equivariant HDR scale."""
    samples, solid_angles = _validate_spherical_samples(samples, solid_angles)
    luminance = compute_luminance(samples).clamp(min=0.0)
    scale = weighted_quantile(luminance, solid_angles, quantile)
    scale = float(scale.detach().cpu().item())
    if math.isfinite(scale) and scale > 0.0:
        return scale

    positive = luminance > 0.0
    if bool(positive.any().item()):
        scale = weighted_quantile(
            luminance[positive],
            solid_angles[positive],
            quantile,
        )
        scale = float(scale.detach().cpu().item())
        if math.isfinite(scale) and scale > 0.0:
            return scale
        raise RuntimeError("positive luminance did not produce a positive HDR scale")
    return 1.0


@torch.no_grad()
def canonical_hdr_scale_reference(field):
    """A fixed raster statistic, independent of fitting/evaluation quadrature."""
    values = field.values.detach().to(device="cpu", dtype=torch.float64).contiguous()
    areas = field.projection.solid_angle_map(field.height, field.width, "cpu", torch.float64)
    return {
        "definition": "reference_raster_solid_angle_luminance_q95_v1",
        "value": compute_hdr_scale(values.reshape(-1, values.shape[-1]), areas.reshape(-1)),
        "projection": field.projection.name,
        "height": field.height,
        "width": field.width,
        "values_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }


