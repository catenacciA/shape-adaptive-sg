"""Shared helpers for positive spherical kernel banks."""

import torch


COEFFICIENT_ACTIVATION_FLOOR = 1e-8


def inverse_softplus(y, eps=COEFFICIENT_ACTIVATION_FLOOR):
    """Stable finite inverse with eps used only for exact nonpositive input."""
    floor = torch.as_tensor(float(eps), device=y.device, dtype=y.dtype)
    safe = torch.where(y > 0.0, y, floor)
    return safe + torch.log(-torch.expm1(-safe))


def normalize_vectors_3d(values, eps=1e-12):
    """Normalize 3-vectors with a row-local, shape-independent reduction."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("3-vector normalization expects shape [N, 3]")
    norm_squared = (
        values[:, 0].square()
        + values[:, 1].square()
        + values[:, 2].square()
    )
    inverse_norm = torch.rsqrt(norm_squared.clamp_min(float(eps) ** 2))
    return values * inverse_norm.unsqueeze(1)


def build_asg_tangent_frame(z, theta, eps=1e-12):
    z = normalize_vectors_3d(z, eps=eps)
    # Duff's stable two-chart basis switches at the equator. Training compensates
    # theta on chart crossings; inference uses the same frame convention.
    sign = torch.where(z[:, 2] >= 0.0, torch.ones_like(z[:, 2]), -torch.ones_like(z[:, 2]))
    a = -1.0 / (sign + z[:, 2])
    b = z[:, 0] * z[:, 1] * a
    t1 = torch.stack(
        (1.0 + sign * z[:, 0].square() * a, sign * b, -sign * z[:, 0]),
        dim=1,
    )
    t2 = torch.stack(
        (b, sign + z[:, 1].square() * a, -z[:, 1]),
        dim=1,
    )
    t1 = normalize_vectors_3d(t1, eps=eps)
    t2 = normalize_vectors_3d(t2, eps=eps)

    cos_t = torch.cos(theta).unsqueeze(1)
    sin_t = torch.sin(theta).unsqueeze(1)
    x = cos_t * t1 + sin_t * t2
    y = -sin_t * t1 + cos_t * t2
    return x, y, z


def _empty_init_candidates(init_allocation):
    return (
        torch.empty(
            0,
            3,
            device=init_allocation.directions.device,
            dtype=init_allocation.directions.dtype,
        ),
        torch.empty(
            0,
            init_allocation.target_values.shape[-1],
            device=init_allocation.target_values.device,
            dtype=init_allocation.target_values.dtype,
        ),
    )


def sample_kernel_allocation_candidates(
    init_allocation,
    num_kernels,
    return_indices=False,
    replacement=True,
    seed=None,
):
    with torch.no_grad():
        if num_kernels == 0:
            dirs, colors = _empty_init_candidates(init_allocation)
            if return_indices:
                indices = torch.empty(
                    0, device=init_allocation.directions.device, dtype=torch.long
                )
                return dirs, colors, indices
            return dirs, colors
        sampling_device = "cpu" if seed is not None else init_allocation.directions.device
        weights_flat = init_allocation.probability_flat.to(
            device=sampling_device, dtype=init_allocation.directions.dtype
        )
        weights_flat = torch.where(
            torch.isfinite(weights_flat), weights_flat, torch.zeros_like(weights_flat)
        ).clamp(min=0.0)
        if not bool((weights_flat > 0.0).any().item()):
            weights_flat = torch.ones_like(weights_flat)
        weights_flat = weights_flat / weights_flat.sum().clamp(min=1e-12)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        replacement = bool(replacement)
        if not replacement and num_kernels > weights_flat.numel():
            unique = torch.multinomial(
                weights_flat,
                weights_flat.numel(),
                replacement=False,
                generator=generator,
            )
            overflow = torch.multinomial(
                weights_flat,
                num_kernels - weights_flat.numel(),
                replacement=True,
                generator=generator,
            )
            indices = torch.cat((unique, overflow))
        else:
            indices = torch.multinomial(
                weights_flat,
                num_kernels,
                replacement=replacement,
                generator=generator,
            )
        indices = indices.to(device=init_allocation.directions.device)
        dirs = init_allocation.directions[indices]
        colors = init_allocation.target_values[indices].clamp(min=0.001)
        if return_indices:
            return dirs, colors, indices
        return dirs, colors


def initialize_radiance_coefficients(
    sampled_values,
    target_radiance_integral,
    eps=COEFFICIENT_ACTIVATION_FLOOR,
):
    """Distribute a target spherical integral over positive coefficients.

    Sampled channel proportions are floored and normalized before applying the
    requested integral. A positive channel with no sampled mass falls back to a
    deterministic uniform distribution. Exactly zero target channels activate to
    the small finite eps floor because softplus cannot represent exact zero
    with a finite raw parameter.
    """
    with torch.no_grad():
        sampled_values = torch.as_tensor(sampled_values)
        if sampled_values.ndim != 2 or sampled_values.shape[0] <= 0:
            raise ValueError(
                "coefficient initialization expects sampled values with shape "
                f"[K, C], got {tuple(sampled_values.shape)}"
            )
        target_radiance_integral = torch.as_tensor(
            target_radiance_integral,
            device=sampled_values.device,
            dtype=sampled_values.dtype,
        )
        if (
            target_radiance_integral.ndim != 1
            or target_radiance_integral.shape[0] != sampled_values.shape[1]
        ):
            raise ValueError(
                "target radiance integral must have shape [C] matching sampled "
                f"values, got {tuple(target_radiance_integral.shape)}"
            )
        if not bool(torch.isfinite(sampled_values).all().item()):
            raise ValueError("sampled coefficient proportions must be finite")
        if not bool(torch.isfinite(target_radiance_integral).all().item()):
            raise ValueError("target radiance integral must be finite")
        if bool((target_radiance_integral < 0.0).any().item()):
            raise ValueError("target radiance integral must be non-negative")

        proportions = sampled_values.clamp(min=0.0)
        sampled_sum = proportions.sum(dim=0)
        positive_target = target_radiance_integral > 0.0
        use_uniform = positive_target & (sampled_sum <= 0.0)
        if bool(use_uniform.any().item()):
            proportions[:, use_uniform] = 1.0

        proportions = proportions.clamp(min=float(eps))
        proportions = proportions / proportions.sum(dim=0).clamp(min=float(eps))
        coefficients = proportions * target_radiance_integral.unsqueeze(0)
        return inverse_softplus(coefficients, eps=eps)

