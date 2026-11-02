#!/usr/bin/env python3
"""Evaluate a fit, or its SKC bitstream, against a reference HDR map.

    python evaluate.py RUN_DIR/representation.skc            # the decoded bitstream
    python evaluate.py RUN_DIR --latlong-reference LL.exr    # original lat-long reference
    python evaluate.py RUN_DIR --reference OTHER.exr --output result.json

Every fit already evaluates itself and writes the result to RUN_DIR/metrics.json.
By default this script repeats that evaluation: against the fit's input map and,
if the fit was given one, its --latlong-reference, each of which must still have
the SHA-256 recorded by the fit. An explicit --reference or --latlong-reference
replaces them; --reference alone resamples that map for the image metrics.

* relative L2: linear RGB radiance on a rotated icosahedral grid (level 9:
  5,242,880 directions) weighted by solid angle;
* PU21-PSNR / PU21-SSIM / PU21-LPIPS: lat-long rasters (1024x512), EXR 1 = 203 cd/m^2,
  clamped to [0.005, 10,000] cd/m^2 and PU21-encoded (banding_glare) per RGB channel.

The image metrics compare against a lat-long resampling of the reference, or
against --latlong-reference when one is given (it must already match the raster
size).
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from codec.native import load_representation
from envmap.field import infer_projection, load_envmap_field
from fit.artifacts import format_summary
from fit.evaluation import LATLONG_HEIGHT, evaluate_representation
from utils.provenance import artifact_reference, sha256_file, validate_artifact_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fit", type=Path, help="run directory, its inference.json, or an .skc bitstream")
    parser.add_argument("--reference", type=Path,
                        help="reference HDR map, square (octahedral) or 2:1 (lat-long); "
                             "default: the map the fit was trained on")
    parser.add_argument("--latlong-reference", type=Path,
                        help="original 1024x512 lat-long map for the image metrics "
                             "(default: the fit's, or resample the reference)")
    parser.add_argument("--output", type=Path, help="also write the full result as JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subdivision", type=int, default=9, help="relative L2 grid level")
    parser.add_argument("--image-height", type=int, default=LATLONG_HEIGHT,
                        help="height of the lat-long metric rasters")
    parser.add_argument("--chunk-size", type=int, default=8192)
    return parser.parse_args(argv)


def fit_references(run_dir):
    """The input map and lat-long reference of a fit, each as (path, SHA-256 recorded by the fit) or None."""
    try:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        manifest = json.loads((run_dir / "inference.json").read_text())
        source = Path(config["data_root"], config["input_path"]), manifest["provenance"]["source"]["sha256"]
    except (OSError, KeyError, TypeError):
        return None, None
    if config.get("latlong_reference") is None:
        return source, None
    try:
        recorded = json.loads((run_dir / "metrics.json").read_text())["evaluation"]["image"]["target_sha256"]
    except (OSError, KeyError, TypeError):
        recorded = None
    return source, (Path(config["latlong_reference"]), recorded)


def unchanged(path, sha256, option):
    if not path.is_file() or sha256_file(path) != sha256:
        raise ValueError(f"{path} is missing or has changed since the fit; pass {option}")
    return path


def evaluate_artifact(arguments):
    fit = Path(arguments.fit)
    artifact = fit / "inference.json" if fit.is_dir() else fit
    source, latlong = fit_references(artifact.parent)
    reference, latlong_reference = arguments.reference, arguments.latlong_reference
    if reference is None:  # repeat the fit's own evaluation
        if source is None:
            raise FileNotFoundError(f"cannot find the map {fit} was fitted to; pass --reference")
        reference = unchanged(*source, "--reference")
        if latlong_reference is None and latlong is not None:
            latlong_reference = unchanged(*latlong, "--latlong-reference")
    reference_sha256 = sha256_file(reference)
    binding = artifact_reference(artifact)
    representation = load_representation(fit, device=arguments.device)
    field = load_envmap_field(str(reference.parent), reference.name, dtype=torch.float32,
                              device=arguments.device, projection_name=infer_projection(reference))
    _, record = evaluate_representation(
        representation, field, device=arguments.device, subdivision=arguments.subdivision,
        chunk_size=arguments.chunk_size, image_height=arguments.image_height,
        latlong_reference=latlong_reference,
    )
    validate_artifact_reference(artifact, binding)
    record.update(artifact=binding, num_kernels=representation.manifest["num_local_kernels"],
                  reference_sha256=reference_sha256,
                  reference_is_fit_source=reference_sha256 == source[1] if source else None)
    if artifact.suffix == ".skc":
        record["codec"] = {"file": artifact.name, "bytes": artifact.stat().st_size}
    return record


def main(argv=None):
    arguments = parse_args(argv)
    record = evaluate_artifact(arguments)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(format_summary(record))


if __name__ == "__main__":
    main()
