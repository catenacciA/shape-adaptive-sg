import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from envmap.statistics import weighted_quantile
from envmap.structure import SphericalStructureField, build_spherical_structure_field


def local_width_from_density(density, active_count_after):
    """Footprint radius at which active_count_after kernels tile the density."""
    active_count_after = max(int(active_count_after), 1)
    density = torch.as_tensor(density)
    return torch.rsqrt(math.pi * float(active_count_after) * density.clamp(min=1e-12))


@dataclass(frozen=True)
class KernelAllocation:
    """Spherical kernel allocation density on a shared S2 candidate set."""

    directions: torch.Tensor
    solid_angles: torch.Tensor
    target_values: torch.Tensor
    probability_flat: torch.Tensor
    density_flat: torch.Tensor
    local_width_flat: torch.Tensor
    structure: SphericalStructureField


def build_kernel_allocation(
    field,
    directions,
    solid_angles,
    num_kernels,
    target_values=None,
    uniform_fraction=0.10,
    clip_quantile=0.995,
    anisotropy_ratio_max=4.0,
    structure_chunk_size=65536,
    structure=None,
):
    """Build a projection-independent spherical allocation density."""
    with torch.no_grad():
        num_kernels = int(num_kernels)
        if num_kernels <= 0:
            raise ValueError("kernel allocation requires a positive kernel count")
        directions = F.normalize(torch.as_tensor(directions), dim=-1)
        solid_angles = torch.as_tensor(
            solid_angles, device=directions.device, dtype=directions.dtype
        ).reshape(-1)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"directions must be [N, 3], got {tuple(directions.shape)}")
        if solid_angles.shape[0] != directions.shape[0]:
            raise ValueError(
                "solid angle count must match directions: "
                f"{solid_angles.shape[0]} vs {directions.shape[0]}"
            )
        if target_values is None:
            target_values = field.sample(directions).detach()
        target_values = torch.as_tensor(
            target_values, device=directions.device, dtype=directions.dtype
        )
        if structure is None:
            structure = build_spherical_structure_field(
                field,
                directions,
                solid_angles,
                target_values=target_values,
                ratio_max=anisotropy_ratio_max,
                chunk_size=structure_chunk_size,
            )
        elif structure.score_flat.shape[0] != directions.shape[0]:
            raise ValueError("precomputed structure must match allocation directions")

        score = structure.score_flat
        positive = score > 0.0
        if bool(positive.any().item()):
            clip_value = weighted_quantile(
                score[positive], solid_angles[positive], clip_quantile
            )
            score = score.clamp(max=clip_value.clamp(min=1e-12))

        sphere_area = solid_angles.sum().clamp(min=1e-12)
        uniform_density = torch.full_like(score, 1.0 / sphere_area)
        score_integral = (score * solid_angles).sum()
        if bool((score_integral > 1e-12).item()):
            guided_density = score / score_integral
        else:
            guided_density = uniform_density
        uniform_fraction = min(max(float(uniform_fraction), 0.0), 1.0)
        density = (
            uniform_fraction * uniform_density
            + (1.0 - uniform_fraction) * guided_density
        )
        probability = density * solid_angles
        probability = probability / probability.sum().clamp(min=1e-12)
        density = probability / solid_angles.clamp(min=1e-12)
        local_width = local_width_from_density(density, num_kernels)
        return KernelAllocation(
            directions=directions,
            solid_angles=solid_angles,
            target_values=target_values.detach(),
            probability_flat=probability,
            density_flat=density,
            local_width_flat=local_width,
            structure=structure,
        )


