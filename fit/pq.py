"""Fixed-normalization SMPTE ST 2084 (PQ) encoding for EXR radiance."""

import torch


PQ_REFERENCE_WHITE_NITS = 203.0
PQ_NOMINAL_PEAK_NITS = 10_000.0
PQ_EXR_NOMINAL_CEILING = PQ_NOMINAL_PEAK_NITS / PQ_REFERENCE_WHITE_NITS

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0


def pq_encode_exr(values, *, clip_to_nominal=False, return_derivative=False):
    """Encode nonnegative EXR values after mapping EXR 1 to 203 cd/m^2.

    By default (training) the ST 2084 analytic expression is extended beyond
    its nominal 10,000 cd/m^2 domain. clip_to_nominal (metrics) clamps to
    that domain before encoding. Callers are responsible for enforcing finite,
    nonnegative inputs.
    """
    values = torch.as_tensor(values)
    if not torch.is_floating_point(values):
        raise ValueError("PQ encoding requires floating-point values")

    normalized_luminance = values / PQ_EXR_NOMINAL_CEILING
    if clip_to_nominal:
        normalized_luminance = normalized_luminance.clamp(max=1.0)

    # Direct y**m1 has an infinite derivative at y=0. Clamping only the
    # inactive power branch defines a finite zero gradient without perturbing
    # any positive representable input or the ST 2084 value at zero.
    safe_luminance = normalized_luminance.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    powered = torch.where(
        normalized_luminance > 0.0,
        safe_luminance.pow(PQ_M1),
        torch.zeros_like(normalized_luminance),
    )
    numerator = PQ_C1 + PQ_C2 * powered
    denominator = 1.0 + PQ_C3 * powered
    ratio = numerator / denominator
    encoded = ratio.pow(PQ_M2)
    if not return_derivative:
        return encoded
    # Chain rule through the same intermediates as encoding. The quotient
    # derivative simplifies to (C2 - C1*C3) / denominator**2.
    derivative = (
        PQ_M2 * ratio.pow(PQ_M2 - 1.0)
        * (PQ_C2 - PQ_C1 * PQ_C3) / denominator.square()
        * PQ_M1 * safe_luminance.pow(PQ_M1 - 1.0)
        / PQ_EXR_NOMINAL_CEILING
    )
    active = normalized_luminance >= torch.finfo(values.dtype).tiny
    if clip_to_nominal:
        active = active & (values / PQ_EXR_NOMINAL_CEILING <= 1.0)
    return encoded, torch.where(active, derivative, torch.zeros_like(derivative))
