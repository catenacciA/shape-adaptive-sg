import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from envmap.statistics import (
    compute_luminance,
    robust_unit_component,
    weighted_quantile,
)


@dataclass(frozen=True)
class SphericalStructureField:
    """Intrinsic tangent-plane structure sampled on spherical candidates."""

    theta_flat: torch.Tensor
    tangent_axis: torch.Tensor
    detail_flat: torch.Tensor
    coherence_flat: torch.Tensor
    anisotropy_ratio_flat: torch.Tensor
    energy_flat: torch.Tensor
    score_flat: torch.Tensor

def _robust_scale(values, quantile=0.75, fallback=1.0, measure=None):
    flat = values.detach().reshape(-1)
    finite = torch.isfinite(flat)
    positive = finite & (flat > 0.0)
    if not bool(positive.any().item()):
        return values.new_tensor(float(fallback))
    q = min(max(float(quantile), 0.0), 1.0)
    if measure is None:
        result = torch.quantile(flat[positive], q)
    else:
        positive_measure = torch.as_tensor(
            measure, device=values.device, dtype=values.dtype
        ).reshape(-1)[positive]
        result = weighted_quantile(flat[positive], positive_measure, q)
    return result.clamp(min=1e-6)


def _spherical_tangent_frame(directions, eps=1e-12):
    directions = F.normalize(directions, dim=-1, eps=eps)
    x, y, z = directions.unbind(dim=-1)
    sign = torch.where(z >= 0.0, torch.ones_like(z), -torch.ones_like(z))
    a = -1.0 / (sign + z)
    b = x * y * a
    tangent_x = torch.stack(
        (1.0 + sign * x.square() * a, sign * b, -sign * x), dim=-1
    )
    tangent_y = torch.stack((b, sign + y.square() * a, -y), dim=-1)
    return F.normalize(tangent_x, dim=-1), F.normalize(tangent_y, dim=-1)


def _log_signal(values, scale):
    return torch.log1p(values.clamp(min=0.0) / float(scale))


def build_spherical_structure_field(
    field,
    directions,
    solid_angles,
    target_values=None,
    ratio_max=4.0,
    ring_radius=None,
    scale=None,
    scale_quantile=0.75,
    detail_quantile=0.90,
    chunk_size=65536,
):
    """Estimate tangent-plane target structure on spherical candidates."""
    with torch.no_grad():
        directions = F.normalize(torch.as_tensor(directions), dim=-1)
        solid_angles = torch.as_tensor(
            solid_angles, device=directions.device, dtype=directions.dtype
        ).reshape(-1)
        if target_values is None:
            target_values = field.sample(directions)
        target_values = torch.as_tensor(
            target_values, device=directions.device, dtype=directions.dtype
        )
        if target_values.shape[0] != directions.shape[0]:
            raise ValueError(
                "target values must match direction count: "
                f"{tuple(target_values.shape)} vs {tuple(directions.shape)}"
            )

        luma = compute_luminance(target_values).clamp(min=0.0)
        if scale is None or float(scale) <= 0.0:
            scale = _robust_scale(luma, quantile=scale_quantile, measure=solid_angles)
        else:
            scale = luma.new_tensor(float(scale)).clamp(min=1e-6)

        if ring_radius is None:
            mean_area = solid_angles.mean().detach().clamp(min=1e-12)
            ring_radius = math.sqrt(float(mean_area))
        ring_radius = max(float(ring_radius), 1e-5)
        radii = directions.new_tensor((ring_radius, 2.0 * ring_radius))

        jxx = torch.empty(directions.shape[0], device=directions.device, dtype=directions.dtype)
        jyy = torch.empty_like(jxx)
        jxy = torch.empty_like(jxx)

        for start in range(0, directions.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), directions.shape[0])
            center = directions[start:end]
            tangent_x, tangent_y = _spherical_tangent_frame(center)
            accum_xx = torch.zeros(end - start, device=directions.device, dtype=directions.dtype)
            accum_yy = torch.zeros_like(accum_xx)
            accum_xy = torch.zeros_like(accum_xx)

            for radius in radii:
                cos_r = torch.cos(radius)
                sin_r = torch.sin(radius)
                plus_x = F.normalize(cos_r * center + sin_r * tangent_x, dim=-1)
                minus_x = F.normalize(cos_r * center - sin_r * tangent_x, dim=-1)
                plus_y = F.normalize(cos_r * center + sin_r * tangent_y, dim=-1)
                minus_y = F.normalize(cos_r * center - sin_r * tangent_y, dim=-1)
                samples = torch.cat((plus_x, minus_x, plus_y, minus_y), dim=0)
                signal = _log_signal(field.sample(samples), scale)
                count = end - start
                sxp, sxm, syp, sym = signal.split(count, dim=0)
                gx = (sxp - sxm) / (2.0 * radius).clamp(min=1e-6)
                gy = (syp - sym) / (2.0 * radius).clamp(min=1e-6)
                accum_xx += gx.square().mean(dim=-1)
                accum_yy += gy.square().mean(dim=-1)
                accum_xy += (gx * gy).mean(dim=-1)

            inv_count = 1.0 / float(radii.numel())
            jxx[start:end] = accum_xx * inv_count
            jyy[start:end] = accum_yy * inv_count
            jxy[start:end] = accum_xy * inv_count

        trace = (jxx + jyy).clamp(min=0.0)
        delta = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square())
        high_gradient_angle = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
        theta = high_gradient_angle + 0.5 * math.pi
        tangent_x, tangent_y = _spherical_tangent_frame(directions)
        tangent_axis = F.normalize(
            torch.cos(theta).unsqueeze(1) * tangent_x
            + torch.sin(theta).unsqueeze(1) * tangent_y,
            dim=-1,
        )

        detail = torch.sqrt(trace)
        detail_unit = robust_unit_component(
            detail, solid_angles, quantile=detail_quantile
        )
        coherence = torch.where(
            trace > 1e-12,
            (delta / trace.clamp(min=1e-12)).clamp(0.0, 1.0),
            torch.zeros_like(trace),
        )
        luma_energy = robust_unit_component(
            _log_signal(luma, scale), solid_angles, quantile=0.95
        )
        ratio_max = max(float(ratio_max), 1.0)
        anisotropy_ratio = 1.0 + (ratio_max - 1.0) * (detail_unit * coherence)
        anisotropy_ratio = anisotropy_ratio.clamp(1.0, ratio_max)
        score = (
            0.60 * detail_unit
            + 0.35 * torch.sqrt((luma_energy * detail_unit).clamp(min=0.0))
            + 0.05 * luma_energy
        )

        return SphericalStructureField(
            theta_flat=torch.where(torch.isfinite(theta), theta, torch.zeros_like(theta)),
            tangent_axis=tangent_axis,
            detail_flat=detail_unit,
            coherence_flat=coherence,
            anisotropy_ratio_flat=anisotropy_ratio,
            energy_flat=luma_energy,
            score_flat=torch.where(torch.isfinite(score), score, torch.zeros_like(score)),
        )



