import os
from dataclasses import dataclass

import torch

from envmap.projection import build_projection
from utils.image_io import load_rgb_hdr


@dataclass
class EnvmapField:
    values: torch.Tensor
    projection: object

    def __post_init__(self):
        if self.values.ndim != 3:
            raise ValueError(f"EnvmapField values must be [H, W, C], got {tuple(self.values.shape)}")
        self.height = int(self.values.shape[0])
        self.width = int(self.values.shape[1])
        self.feat_dim = int(self.values.shape[2])
        self.num_pixels = self.height * self.width
        self.device = self.values.device
        self.dtype = self.values.dtype
        self.solid_angle = self.projection.solid_angle_map(
            self.height, self.width, self.device, self.dtype
        )

    def sample(self, dirs):
        """Bilinear radiance at dirs, which may live on any device."""
        return self.projection.sample_bilinear(self.values, dirs.to(self.device))


def validate_projection_shape(
    height,
    width,
    projection,
):
    height = int(height)
    width = int(width)
    if projection.name == "latlong":
        valid = width == 2 * height
        expected = "width == 2 * height (2:1)"
    elif projection.name == "octahedral":
        valid = width == height
        expected = "width == height (square)"
    else:
        return

    if not valid:
        raise ValueError(
            f"Projection '{projection.name}' received non-standard envmap shape "
            f"{height}x{width}; expected {expected}."
        )


def detect_projection(height, width):
    """Projection implied by a raster shape: square is octahedral, 2:1 is lat-long."""
    if width == height:
        return "octahedral"
    if width == 2 * height:
        return "latlong"
    raise ValueError(
        f"cannot infer the projection of a {height}x{width} map: expected a square "
        "(octahedral) or 2:1 (lat-long) image"
    )


def infer_projection(path):
    """Projection of an HDR map file, from its shape."""
    height, width = load_rgb_hdr(path).shape[:2]
    return detect_projection(height, width)


def load_envmap_field(
    data_root,
    input_path,
    dtype,
    device,
    projection_name="latlong",
):
    envmap_path = os.path.join(data_root, input_path)
    envmap_data = load_rgb_hdr(envmap_path)
    values = torch.from_numpy(envmap_data).to(dtype=dtype, device=device)
    values = values.contiguous().clamp(min=0.0)
    projection = build_projection(projection_name)
    validate_projection_shape(
        values.shape[0],
        values.shape[1],
        projection,
    )
    return EnvmapField(values=values, projection=projection)
