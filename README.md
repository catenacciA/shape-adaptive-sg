<div align="center">

# Shape-Adaptive Spherical Gaussians<br>for HDR Environment Map Reconstruction

**Alessandro Catenacci** · **Fabio Pellacini**  
University of Modena and Reggio Emilia

*STAG: Smart Tools and Applications in Graphics, 2026*

</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8cc66703-0ce1-47af-bde7-c9a4aeea06f1" alt="Shape-adaptive spherical Gaussian representation and HDR environment reconstruction" width="100%">
</p>

We reconstruct HDR environment maps as mixtures of anisotropic primitives fitted directly on the sphere. Each primitive extends the anisotropic spherical Gaussian (ASG) with learnable radial falloff *p* and contour shape *q*, capturing smooth lobes as well as sharp, rectangular emitters. Fitting combines adaptive geodesic quadrature, a solid-angle-weighted PQ objective, and progressive residual-driven densification. The result is a continuous, projection-independent representation stored in **15 bytes per primitive**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/180345dd-c6f6-4ebb-926d-f45da1c81db4" alt="Artist Workshop reconstruction comparison with Image-GS and ASG" width="100%">
  <br>
  <sub>Artist Workshop at 10,000 primitives, with FLIP error insets.</sub>
</p>

At 10,000 primitives, on the 12 environments of our benchmark that are not sky-only, our method lowers the mean relative radiance error by **19%** compared with ASG and **20%** compared with Image-GS, and improves PU21-PSNR, PU21-SSIM and PU21-LPIPS over ASG on each of them.

## Installation

```bash
conda env create -f environment.yml
conda activate spherical-kernels
```

Tested on Linux with NVIDIA CUDA GPUs. `torch.compile` requires `gcc` on the `PATH`. On an RTX 4090, fitting a 1K envmap at 10,000 primitives takes approximately **2 minutes**, including evaluation and export.

## Usage

Run commands from the repository root. The input can be a lat-long (2:1) or octahedral (square) RGB `.exr` or `.hdr` environment map.

```bash
# Download an example HDRI from Poly Haven
mkdir -p data
curl -L -o data/venice_sunset_2k.exr \
  https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/2k/venice_sunset_2k.exr

# Fit 10,000 shape-adaptive primitives
python train.py data/venice_sunset_2k.exr
```

The fit automatically produces a compressed `representation.skc`, a lat-long `reconstruction.exr`, `metrics.json` (spherical relative L2, PU21 image metrics and trainer runtimes), `kernels.png`, and the native fitted parameters under `results/fits/`.

Use `--budget` to change the number of primitives, or `--config cfgs/asg.yaml` for the ASG baseline. Ablations are available in `cfgs/p_only.yaml` and `cfgs/q_only.yaml` (one shape exponent), `cfgs/beta.yaml` (a compactly supported Beta kernel) and `cfgs/ours_unnormalized.yaml` (shape normalization off); the last two write no `.skc` bitstream. See `python train.py --help` for other options.

For octahedral inputs, you can supply an original lat-long HDRI for the image metrics, as in the paper:

```bash
python train.py data/octa/artist_workshop_octa_1024.exr \
  --latlong-reference data/latlong/artist_workshop_latlong_512.exr
```

Without this option, the input is resampled to lat-long for the image metrics.

The image metrics are PSNR, SSIM and LPIPS on 1024x512 lat-long rasters encoded with PU21 (banding_glare, EXR 1 = 203 cd/m², clamped to [0.005, 10,000] cd/m²). Relative L2 is measured on linear radiance over the sphere. The fit itself minimizes a PQ-encoded loss.

### Reconstruction and evaluation

```bash
RUN=results/fits/venice_sunset_2k_ours_10000_s123
python reconstruct.py "$RUN" recon_4k.exr --height 2048
python evaluate.py "$RUN"
```

Both scripts also accept a `.skc` bitstream. Use `--help` for additional options.

Run the CPU test suite with `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -t .`. On a CUDA machine, the same command without `CUDA_VISIBLE_DEVICES=''` also checks the fused GPU code paths against the CPU reference.

## Citation

```bibtex
@inproceedings{catenacci2026shapeadaptive,
  title     = {Shape-Adaptive Spherical Gaussians for HDR Environment Map Reconstruction},
  author    = {Catenacci, Alessandro and Pellacini, Fabio},
  booktitle = {STAG: Smart Tools and Applications in Graphics},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE). Some utility functions are adapted from [Image-GS](https://github.com/NYU-ICL/image-gs) (MIT). The HDRIs are from [Poly Haven](https://polyhaven.com) (CC0).
