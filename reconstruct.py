#!/usr/bin/env python3
"""Render a fitted representation to an HDR environment map.

    python reconstruct.py RUN_DIR output.exr [--projection latlong|octahedral] [--height 1024]
    python reconstruct.py RUN_DIR/representation.skc output.exr

Every fit already writes a 1024x512 lat-long render, RUN_DIR/reconstruction.exr.
The representation is continuous on the sphere: every output pixel evaluates the
mixture at its center direction, at any resolution and in either layout.
"""
import argparse
from pathlib import Path

import torch

from codec.native import load_representation
from fit.evaluation import project
from utils.image_io import save_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fit", type=Path, help="run directory, its inference.json, or an .skc bitstream")
    parser.add_argument("output", type=Path, help="output .exr")
    parser.add_argument("--projection", choices=("latlong", "octahedral"), default="latlong")
    parser.add_argument("--height", type=int, default=1024,
                        help="output height; lat-long maps are twice as wide")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".exr":
        parser.error("the output must be an .exr file")
    representation = load_representation(args.fit, device=args.device)
    image = project(representation.evaluate_directions, args.projection, args.height, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, str(args.output))
    print(f"{args.output}: {image.shape[1]}x{image.shape[0]} {args.projection}")


if __name__ == "__main__":
    main()
