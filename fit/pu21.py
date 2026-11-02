"""Fixed-normalization PU21 (banding_glare) encoding for EXR radiance, for the image metrics.

Formula and coefficients of gfxdisp/pu21 (matlab/pu21_encoder.m). Training uses PQ (fit/pq.py).
"""

import torch


PU21_REFERENCE_WHITE_NITS = 203.0
PU21_MIN_NITS = 0.005
PU21_MAX_NITS = 10_000.0
PU21_COEFFICIENTS = (0.353487901, 0.3734658629, 8.277049286e-05,
                     0.9062562627, 0.09150303166, 0.9099517204, 596.3148142)


def pu21_encode_nits(luminance):
    """PU21 of absolute luminance in cd/m^2, clamped to [0.005, 10000], in float64."""
    y = torch.as_tensor(luminance, dtype=torch.float64).clamp(PU21_MIN_NITS, PU21_MAX_NITS)
    a, b, c, d, e, f, g = PU21_COEFFICIENTS
    return (g * (((a + b * y.pow(d)) / (1 + c * y.pow(d))).pow(e) - f)).clamp_min(0)


PU21_PEAK = float(pu21_encode_nits(PU21_MAX_NITS))


def pu21_encode_exr(values):
    """Encode nonnegative EXR values per channel, EXR 1 = 203 cd/m^2, normalized to [0, 1] by PU21(10000).

    The map is fixed: no image-wise scaling. Callers enforce finite, nonnegative inputs.
    """
    return pu21_encode_nits(torch.as_tensor(values, dtype=torch.float64) * PU21_REFERENCE_WHITE_NITS) / PU21_PEAK
