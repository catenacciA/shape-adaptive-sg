"""Reconstruction evaluation, independent of training."""
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
import math
import tempfile

import torch

from envmap.field import EnvmapField
from envmap.projection import build_projection
from envmap.quadrature import build_evaluation_quadrature, resolve_geodesic_subdivision
from envmap.statistics import canonical_hdr_scale_reference
from fit.metrics import (IMAGE_PROTOCOL, METRIC_API, SPHERICAL_METRIC_API, compute_image_metrics,
                         compute_spherical_metrics, metrics_to_json)
from fit.pq import PQ_REFERENCE_WHITE_NITS, PQ_NOMINAL_PEAK_NITS
from utils.image_io import load_rgb_hdr, save_image
from utils.provenance import sha256_file


@dataclass(frozen=True)
class SphericalEvaluationContext:
    quadrature: object
    target_samples: torch.Tensor
    hdr_scale: float
    protocol_metadata: dict


@dataclass(frozen=True)
class SphericalEvaluationResult:
    prediction_samples: torch.Tensor
    metrics: dict
    protocol_metadata: dict


@torch.no_grad()
def build_spherical_evaluation_context(field, *, device, dtype=torch.float32, subdivision=None,
                                       scale_reference=None):
    subdivision = (resolve_geodesic_subdivision("auto", field.height, field.width) + 1
                   if subdivision is None else subdivision)
    grid = build_evaluation_quadrature(subdivision, device="cpu", dtype=torch.float64)
    # Target sampling and scale are reproducible CPU FP64 operations. Query the
    # same rounded directions that the representation receives.
    directions = grid.directions.to(dtype=dtype)
    target_field = EnvmapField(field.values.detach().to(device="cpu", dtype=torch.float64), field.projection)
    target = target_field.sample(directions.double())
    reference = canonical_hdr_scale_reference(field) if scale_reference is None else dict(scale_reference)
    scale = float(reference["value"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("reference HDR scale must be finite and positive")
    quadrature = replace(grid, directions=directions.to(device), solid_angles=grid.solid_angles.to(device))
    return SphericalEvaluationContext(
        quadrature, target.to(device), float(scale),
        {
            "version": SPHERICAL_METRIC_API,
            "metrics": {"relative_l2": "linear radiance on S2; supplied solid angles"},
            "grid_id": grid.grid_id,
            "subdivision": int(subdivision),
            "sample_count": grid.sample_count,
            "log_scale": float(scale),
            "scale_reference": reference,
            "pq_reference_white_nits": PQ_REFERENCE_WHITE_NITS,
            "pq_nominal_peak_nits": PQ_NOMINAL_PEAK_NITS,
        },
    )


def evaluate_directions_batched(
    render_directions,
    directions,
    *,
    chunk_size,
    feat_dim,
):
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or len(directions) == 0:
        raise ValueError("evaluation needs directions and a positive chunk size")
    with torch.inference_mode():
        result = None
        for start in range(0, directions.shape[0], chunk_size):
            end = min(start + chunk_size, directions.shape[0])
            values = render_directions(directions[start:end])
            if values.shape != (end - start, feat_dim):
                raise ValueError("direction query returned an unexpected shape")
            if result is None:
                result = torch.empty(len(directions), feat_dim, device=directions.device, dtype=values.dtype)
            result[start:end] = values
    return result


def evaluate_reconstruction(context, *, render_directions, direction_chunk_size, feat_dim):
    prediction = evaluate_directions_batched(
        render_directions, context.quadrature.directions,
        chunk_size=direction_chunk_size, feat_dim=feat_dim,
    )
    metrics = compute_spherical_metrics(
        prediction, context.target_samples, context.quadrature.solid_angles,
        chunk_size=direction_chunk_size,
    )
    return SphericalEvaluationResult(prediction.detach(), metrics, context.protocol_metadata)


@torch.inference_mode()
def project(query, projection_name, height, chunk_size):
    if height <= 0 or chunk_size <= 0:
        raise ValueError("projection needs positive height and chunk size")
    projection = build_projection(projection_name)
    width = height * 2 if projection_name == 'latlong' else height
    out = torch.empty(height * width, 3, dtype=torch.float32)
    for start in range(0, len(out), chunk_size):
        index = torch.arange(start, min(start + chunk_size, len(out)))
        dirs = projection.pixel_to_dir((index % width).double()+.5,
                                       (index // width).double()+.5, width, height)
        values = query(dirs)
        if values.shape != (len(index), 3) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("query must return finite nonnegative RGB radiance")
        out[start:start+len(index)] = values.cpu().float()
    return out.reshape(height, width, 3)


LATLONG_HEIGHT = 512  # the image metrics compare 1024x512 lat-long rasters


def load_latlong_raster(path, height=LATLONG_HEIGHT):
    """Load an RGB HDR lat-long raster of the image-metric size."""
    image = load_rgb_hdr(path)
    if image.shape != (height, height * 2, 3):
        raise ValueError(f"{path} must be a {height * 2}x{height} lat-long map, "
                         f"got {image.shape[1]}x{image.shape[0]}")
    return image


def evaluate_latlong(query, target_query, *, image_path=None, height=LATLONG_HEIGHT, chunk_size=8192,
                     target_path=None):
    """PU21 image metrics on FP32 lat-long EXRs, saved and reloaded so they describe the actual rasters.

    The reconstruction is written to image_path (default: a temporary file). The
    target is target_query resampled to the same raster, or the prepared lat-long
    map at target_path.
    """
    if height < 32:
        raise ValueError("image evaluation height must be at least 32")
    images, hashes = {}, {}
    with tempfile.TemporaryDirectory(prefix="latlong-evaluation-") as scratch:
        for name, render in (("reconstruction", query), ("target", target_query)):
            if name == "target" and target_path is not None:
                path = Path(target_path)
            else:
                path = Path(image_path) if name == "reconstruction" and image_path else Path(scratch, name + ".exr")
                save_image(project(render, "latlong", height, chunk_size), str(path))
            images[name] = load_latlong_raster(path, height)
            hashes[name + "_sha256"] = sha256_file(path)
    metrics = compute_image_metrics(images["reconstruction"], images["target"])
    return metrics, dict(IMAGE_PROTOCOL, height=height, width=height*2,
                         lpips_package_version=version("lpips"), **hashes)


def evaluate_representation(representation, field, *, device, subdivision=None, chunk_size=8192,
                            image_height=LATLONG_HEIGHT, image_path=None, latlong_reference=None):
    """All reconstruction metrics of a representation against the reference map in field.

    Relative L2 of linear radiance on the sphere, and PU21-PSNR, PU21-SSIM and PU21-LPIPS
    on lat-long rasters of the reconstruction and of the reference: resampled from
    field, or the prepared lat-long map latlong_reference. Returns the spherical
    evaluation context and a JSON-ready record.
    """
    context = build_spherical_evaluation_context(field, device=device, subdivision=subdivision)
    spherical = evaluate_reconstruction(
        context, render_directions=representation.evaluate_directions,
        direction_chunk_size=chunk_size, feat_dim=representation.feat_dim,
    )
    image_metrics, image_protocol = evaluate_latlong(
        representation.evaluate_directions, field.sample, image_path=image_path,
        height=image_height, chunk_size=chunk_size, target_path=latlong_reference)
    image_protocol["reference"] = "prepared_latlong" if latlong_reference else "reprojected_reference"
    if latlong_reference:
        image_protocol["reference_file"] = Path(latlong_reference).name
    return context, {**metrics_to_json(dict(spherical.metrics, **image_metrics)),
                     "evaluation": dict(version=METRIC_API, relative_l2=spherical.protocol_metadata,
                                        image=image_protocol)}
