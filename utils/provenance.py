"""Compact provenance recorded with every fit: source image, config, code revision, runtime."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("*.py", "cfgs")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_reference(path):
    """Bind a result to a file; native manifests transitively bind their NPZ."""
    path = Path(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        if manifest.get("artifact_type") == "native_spherical_inference":
            name = manifest["arrays_file"]
            if Path(name).name != name or sha256_file(path.with_name(name)) != manifest["arrays_sha256"]:
                raise ValueError("inference array hash mismatch")
    return {"path": path.name, "sha256": sha256_file(path)}


def validate_artifact_reference(path, reference):
    if artifact_reference(path)["sha256"] != reference["sha256"]:
        raise ValueError("result artifact hash mismatch")


def _git(*args):
    result = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.decode().strip() if result.returncode == 0 else None


def collect_source_state():
    """Code revision; git_dirty flags uncommitted changes to tracked source files."""
    status = _git("status", "--porcelain", "--untracked-files=no", "--", *SOURCE_PATHS)
    return {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def collect_run_provenance(args, *, input_path):
    device = torch.device(args.device)
    return {
        "source": {"file": Path(input_path).name, "sha256": sha256_file(input_path),
                   "projection": args.envmap_projection},
        "config_sha256": sha256_file(Path(args.log_dir) / "config.yaml"),
        "code": collect_source_state(),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "numpy": np.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)},
    }
