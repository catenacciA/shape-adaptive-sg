"""Separate spherical radiometry and ordinary 2D lat-long image metrics."""
from functools import lru_cache
import math

import torch

from fit.pu21 import pu21_encode_exr

METRIC_API = "reconstruction_domains_v7"
SPHERICAL_METRIC_API = "spherical_relative_l2_v1"
IMAGE_PROTOCOL = dict(
    version="canonical_latlong_1024x512_pu21_v1",
    domain="latlong_2d_raster", weighting="uniform_pixels_and_RGB_channels",
    pu21="banding_glare per RGB channel; EXR 1 = 203 nits; clamp to [0.005, 10000] nits; divided by PU21(10000)",
    ssim="RGB mean; Gaussian 11x11 sigma 1.5; population covariance; valid windows; K1=.01 K2=.03",
    lpips="same PU21 RGB; [0,1] to [-1,1]; pretrained AlexNet; LPIPS v0.1",
)


def _validate_inputs(prediction_samples, target_samples, solid_angles):
    if prediction_samples.ndim != 2 or target_samples.ndim != 2:
        raise ValueError(
            "spherical HDR metrics expect [M, C] tensors, got "
            f"{tuple(prediction_samples.shape)} and {tuple(target_samples.shape)}"
        )
    if prediction_samples.shape != target_samples.shape:
        raise ValueError(
            "spherical HDR metric inputs must have matching shapes, got "
            f"{tuple(prediction_samples.shape)} and {tuple(target_samples.shape)}"
        )
    if prediction_samples.shape[0] <= 0 or prediction_samples.shape[1] <= 0:
        raise ValueError("spherical HDR metrics require at least one sample and channel")
    if not torch.is_floating_point(prediction_samples) or not torch.is_floating_point(
        target_samples
    ):
        raise ValueError("spherical HDR metric samples must be floating-point tensors")
    if prediction_samples.device != target_samples.device:
        raise ValueError("prediction and target samples must be on the same device")
    if not bool(torch.isfinite(prediction_samples).all().item()):
        raise ValueError("prediction samples contain non-finite values")
    if not bool(torch.isfinite(target_samples).all().item()):
        raise ValueError("target samples contain non-finite values")
    if bool((prediction_samples < 0.0).any().item()):
        raise ValueError("prediction samples must be nonnegative physical radiance")
    if bool((target_samples < 0.0).any().item()):
        raise ValueError("target samples must be nonnegative physical radiance")

    areas = torch.as_tensor(
        solid_angles,
        device=prediction_samples.device,
        dtype=torch.float64,
    )
    if areas.ndim != 1 or areas.shape[0] != prediction_samples.shape[0]:
        raise ValueError(
            "solid angles must have shape [M] matching the samples, got "
            f"{tuple(areas.shape)} for {prediction_samples.shape[0]} samples"
        )
    if not bool(torch.isfinite(areas).all().item()):
        raise ValueError("solid angles contain non-finite values")
    if not bool((areas > 0.0).all().item()):
        raise ValueError("solid angles must be strictly positive")
    total = areas.sum()
    if not bool(torch.isfinite(total)):
        raise ValueError("solid angle sum must be finite")
    return areas / total


def compute_spherical_metrics(prediction_samples, target_samples, solid_angles, *, chunk_size=None):
    """Intrinsic linear radiance relative L2 on S², with supplied solid angles."""
    weights = _validate_inputs(prediction_samples, target_samples, solid_angles)
    count = len(weights)
    chunk_size = count if chunk_size is None else int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    linear_mse, energy = weights.new_zeros(2).unbind()
    for start in range(0, count, chunk_size):
        prediction = prediction_samples[start:start + chunk_size].double()
        target = target_samples[start:start + chunk_size].double()
        area = weights[start:start + chunk_size]
        linear_mse += ((prediction - target).square().mean(-1) * area).sum()
        energy += (target.square().mean(-1) * area).sum()
    if not bool(torch.isfinite(torch.stack((linear_mse, energy))).all()):
        raise ValueError("metric arithmetic overflow")
    relative = (linear_mse / energy).sqrt() if energy > 0 else (energy if linear_mse == 0 else None)
    return {"relative_l2": relative}


@lru_cache(maxsize=1)
def load_lpips():
    """One shared CPU inference model for the PU21-LPIPS metric."""
    import lpips
    return lpips.LPIPS(
        net="alex", version="0.1", pretrained=True, verbose=False,
    ).eval()


@torch.inference_mode()
def _evaluate_lpips(prediction, target):
    """LPIPS on the already encoded HWC PU21 arrays in [0, 1]."""
    inputs = tuple(
        torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float().mul(2).sub(1)
        for value in (prediction, target)
    )
    value = float(load_lpips()(*inputs).item())
    if not math.isfinite(value):
        raise ValueError("nonfinite PU21-LPIPS result")
    return value


def compute_image_metrics(prediction, target):
    """Unweighted 2D metrics on the same saved linear lat-long RGB images."""
    import cv2
    import numpy as np
    prediction, target = (torch.as_tensor(v).detach().cpu().double() for v in (prediction, target))
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3 or min(prediction.shape[:2]) < 32:
        raise ValueError("image metrics require matching HWC RGB rasters at least 32x32")
    for value in (prediction, target):
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("image metrics require finite nonnegative radiance")
    p, t = (pu21_encode_exr(v).numpy() for v in (prediction, target))
    mse = np.mean((p-t)**2)
    def blur(v):
        return cv2.GaussianBlur(v, (11, 11), 1.5)[5:-5, 5:-5]
    mp, mt = blur(p), blur(t)
    vp, vt, covariance = blur(p*p)-mp*mp, blur(t*t)-mt*mt, blur(p*t)-mp*mt
    ssim = ((2*mp*mt+.01**2)*(2*covariance+.03**2) /
            ((mp*mp+mt*mt+.01**2)*(vp+vt+.03**2))).mean()
    lpips = _evaluate_lpips(p, t)
    return dict(pu21_psnr=-10*math.log10(mse) if mse else math.inf, pu21_ssim=float(ssim), pu21_lpips=lpips)


def metrics_to_json(metrics):
    """Strict JSON: undefined relative error is null; perfect PU21 PSNR is 'inf'."""
    result = {}
    for name, value in metrics.items():
        if value is None and name == "relative_l2":
            result[name] = None
            continue
        value = float(value)
        if name == "pu21_psnr" and value == math.inf:
            result[name] = "inf"
        elif math.isfinite(value):
            result[name] = value
        else:
            raise ValueError(f"invalid metric {name}: {value}")
    return result


@torch.no_grad()
def radiometric_diagnostics(prediction, target, solid_angles, directions, chunk_size=8192):
    """RGB radiance and first directional moments under the supplied solid angles."""
    totals = torch.zeros(2, prediction.shape[1], device=prediction.device, dtype=torch.float64)
    moments = torch.zeros(2, prediction.shape[1], 3, device=prediction.device, dtype=torch.float64)
    for start in range(0, len(prediction), chunk_size):
        end = start + chunk_size
        area = solid_angles[start:end].double()[:, None]
        for i, values in enumerate((target, prediction)):
            mass = values[start:end].double() * area
            totals[i] += mass.sum(0)
            moments[i] += mass.T @ directions[start:end].double()
    difference = totals[1] - totals[0]
    denominator = float(totals[0].norm())
    relative = (
        float(difference.norm()) / denominator
        if denominator
        else (0.0 if not bool(difference.any()) else None)
    )
    return {
        "measure": "supplied_spherical_solid_angles",
        "solid_angle_sum": float(solid_angles.double().sum()),
        "target_integrated_rgb": totals[0].tolist(),
        "prediction_integrated_rgb": totals[1].tolist(),
        "integrated_rgb_difference": difference.tolist(),
        "integrated_rgb_relative_error": relative,
        "zero_target_integral": denominator == 0,
        "target_first_directional_moment": moments[0].tolist(),
        "prediction_first_directional_moment": moments[1].tolist(),
    }
