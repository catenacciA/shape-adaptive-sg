"""JSON/NPZ serialization for native spherical inference artifacts."""

import json
from pathlib import Path

import numpy as np
import torch

from model.inference import NativeInferenceRepresentation
from model.asg import ASG_P_BOUNDS, ASG_Q_BOUNDS, BETA_BOUNDS
from codec.transforms import validate_frames
from utils.provenance import sha256_file


INFERENCE_SCHEMA_VERSION = 8


def _numpy(tensor):
    return tensor.detach().cpu().contiguous().numpy()


def _local_arrays(kernel_bank):
    params = kernel_bank.current_parameters()
    z, x, y, lambda_, mu_bandwidth, coefficient = params[:6]
    arrays = {
        "local_z": _numpy(z),
        "local_x": _numpy(x),
        "local_y": _numpy(y),
        "local_lambda": _numpy(lambda_),
        "local_mu": _numpy(mu_bandwidth),
        "local_coefficients": _numpy(coefficient),
    }
    if kernel_bank.asg_footprint == "superelliptic":
        arrays.update(local_p=_numpy(params[6].p), local_q=_numpy(params[6].q))
    elif kernel_bank.asg_footprint == "beta":
        arrays.update(local_beta=_numpy(params[6].beta))
    return arrays


def save_native_inference_artifact(
    log_dir,
    *,
    kernel_bank,
    provenance=None,
):
    log_dir = Path(log_dir)
    arrays_path = log_dir / "inference.npz"
    manifest_path = log_dir / "inference.json"
    arrays = _local_arrays(kernel_bank)

    np.savez(arrays_path, **arrays)
    manifest = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "artifact_type": "native_spherical_inference",
        "arrays_file": arrays_path.name,
        "kernel_type": "asg",
        "feature_dimension": int(kernel_bank.feat_dim),
        "num_local_kernels": int(kernel_bank.num_kernels),
        "arrays_sha256": sha256_file(arrays_path),
        "asg_footprint": kernel_bank.asg_footprint,
    }
    if not getattr(kernel_bank, "asg_normalize_shape", True):
        manifest["asg_normalize_shape"] = False
    if provenance is not None:
        manifest["provenance"] = provenance
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    return manifest


def load_native_inference_artifact(manifest_path, *, device="cpu", dtype=torch.float32):
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported inference artifact schema")
    if manifest.get("artifact_type") != "native_spherical_inference" or manifest.get("kernel_type") != "asg":
        raise ValueError("unsupported inference artifact type")
    arrays_name = manifest.get("arrays_file")
    if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
        raise ValueError("invalid inference array filename")
    arrays_path = manifest_path.with_name(arrays_name)
    if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
        raise ValueError("inference array hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    count, channels = manifest.get("num_local_kernels"), manifest.get("feature_dimension")
    if type(count) is not int or count < 0 or type(channels) is not int or channels <= 0:
        raise ValueError("invalid inference kernel count or feature dimension")
    footprint = manifest.get("asg_footprint")
    if footprint not in {"gaussian", "superelliptic", "beta"}:
        raise ValueError("invalid ASG footprint")
    scalars = ("lambda", "mu") + {"superelliptic": ("p", "q"), "beta": ("beta",)}.get(footprint, ())
    expected = {"local_coefficients": (count, channels)}
    expected.update({f"local_{name}": (count, 3) for name in ("z", "x", "y")})
    expected.update({f"local_{name}": (count,) for name in scalars})
    if set(arrays) != set(expected):
        raise ValueError("inference arrays do not match the kernel representation")
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"invalid inference array shape or values: {name}")
    if (arrays["local_coefficients"] < 0).any() or any(
        (arrays[f"local_{name}"] <= 0).any() for name in scalars
    ):
        raise ValueError("inference coefficients must be nonnegative and kernel scales positive")
    if footprint == "superelliptic":
        for name, bounds in (("p", ASG_P_BOUNDS), ("q", ASG_Q_BOUNDS)):
            if ((arrays[f"local_{name}"] < bounds[0]) |
                (arrays[f"local_{name}"] > bounds[1])).any():
                raise ValueError(f"inference {name} is outside ASG shape bounds")
    if footprint == "beta" and ((arrays["local_beta"] < BETA_BOUNDS[0] * (1 - 1e-6)) |
                                (arrays["local_beta"] > BETA_BOUNDS[1] * (1 + 1e-6))).any():
        raise ValueError("inference beta is outside its bounds")
    frames = validate_frames(arrays["local_z"], arrays["local_x"], arrays["local_y"])
    for name, value in zip(("local_z", "local_x", "local_y"), frames):
        arrays[name] = value.astype(arrays[name].dtype)
    return NativeInferenceRepresentation(manifest, arrays, device=device, dtype=dtype)
