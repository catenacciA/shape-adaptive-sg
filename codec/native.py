"""Adapter between the standalone SKC codec and the native renderer."""

from pathlib import Path

import numpy as np
import torch

from codec.spherical import decode_arrays, encode_arrays
from model.inference import NativeInferenceRepresentation
from model.serialization import load_native_inference_artifact

# SKC method of each fitted variant, keyed by utils.logging.method_name.
SKC_METHODS = {"ours": "full", "asg": "asg", "p_only": "radial", "q_only": "contour"}


def encode_native(representation, *, method):
    manifest = representation.manifest
    if manifest["kernel_type"] != "asg" or representation.feat_dim != 3:
        raise ValueError("SKC v1 requires native RGB ASG kernels")
    footprint = manifest["asg_footprint"]
    if (method == "asg") != (footprint == "gaussian"):
        raise ValueError("SKC method does not match native footprint")
    arrays = {name: value.detach().cpu().numpy() for name, value in representation.arrays.items()}
    return encode_arrays(arrays, method=method)


def decode_native(data, *, device="cpu", dtype=torch.float32):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("SKC rendering requires FP32 or FP64")
    header, arrays = decode_arrays(data)
    manifest = {
        "artifact_type": "native_spherical_inference",
        "kernel_type": "asg", "feature_dimension": 3,
        "num_local_kernels": header.count,
        "asg_footprint": "gaussian" if header.method == "asg" else "superelliptic",
        "packed_codec": {"name": "skc", "version": 1, "method": header.method,
                         "file_bytes": len(data),
                         "record_bits": header.record_bits},
    }
    representation = NativeInferenceRepresentation(manifest, arrays, device=device, dtype=dtype)
    # Guard the native FP32 product and positive sum against representational
    # overflow. Underflow is permitted, as in the native renderer.
    l = arrays["local_lambda"].astype(np.float64)
    m = arrays["local_mu"].astype(np.float64)
    limit = torch.finfo(dtype).max
    if np.any(l * m > limit):
        raise ValueError("decoded concentrations overflow native normalization")
    norm = np.sqrt(np.maximum(l, 1e-12) * np.maximum(m, 1e-12))
    if representation.asg_footprint is None:
        norm /= np.pi
    else:
        norm *= representation.asg_footprint.log_c.detach().cpu().double().exp().numpy()
    bound = (norm[:, None] * arrays["local_coefficients"].astype(np.float64)).sum(axis=0)
    if not np.isfinite(bound).all() or np.any(bound > limit / 2):
        raise ValueError("decoded radiance can overflow native accumulation")
    return representation


def load_packed_artifact(path, *, device="cpu", dtype=torch.float32):
    return decode_native(Path(path).read_bytes(), device=device, dtype=dtype)


def load_representation(path, *, device="cpu"):
    """A fit (its run directory or inference.json) or its .skc bitstream."""
    path = Path(path)
    if path.suffix == ".skc":
        return load_packed_artifact(path, device=device)
    return load_native_inference_artifact(path / "inference.json" if path.is_dir() else path, device=device)
