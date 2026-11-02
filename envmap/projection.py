import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _safe_sign(values):
    return torch.where(values >= 0.0, torch.ones_like(values), -torch.ones_like(values))


def spherical_triangle_area(a, b, c):
    triple = torch.sum(a * torch.cross(b, c, dim=-1), dim=-1).abs()
    denom = (
        1.0
        + torch.sum(a * b, dim=-1)
        + torch.sum(b * c, dim=-1)
        + torch.sum(c * a, dim=-1)
    )
    return 2.0 * torch.atan2(triple, denom.clamp(min=1e-12)).clamp(min=0.0)


@dataclass(frozen=True)
class EnvmapProjection:
    name: str

    def pixel_to_dir(self, x, y, width, height):
        raise NotImplementedError

    def dir_to_uv(self, dirs):
        raise NotImplementedError

    def pixel_solid_angle(self, x, y, width, height):
        raise NotImplementedError

    def solid_angle_map(self, height, width, device, dtype):
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return self.pixel_solid_angle(grid_x, grid_y, width, height)

    @staticmethod
    def _gather_bilinear(envmap, x0, x1, y0, y1, wx, wy):
        c00 = envmap[y0, x0]
        c10 = envmap[y0, x1]
        c01 = envmap[y1, x0]
        c11 = envmap[y1, x1]

        wx = wx.unsqueeze(-1)
        wy = wy.unsqueeze(-1)
        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        return c0 * (1.0 - wy) + c1 * wy

    def sample_bilinear(self, envmap, dirs):
        raise NotImplementedError

class LatLongProjection(EnvmapProjection):
    def __init__(self):
        super().__init__("latlong")

    def pixel_to_dir(self, x, y, width, height):
        theta = (y / height) * math.pi
        phi = (x / width) * (2.0 * math.pi)
        sin_theta = torch.sin(theta)
        dirs = torch.stack(
            [
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=-1,
        )
        return F.normalize(dirs, dim=-1)

    def dir_to_uv(self, dirs):
        dirs = F.normalize(dirs, dim=-1)
        z = torch.clamp(dirs[..., 2], -1.0, 1.0)
        theta = torch.acos(z)
        phi = torch.atan2(dirs[..., 1], dirs[..., 0])
        phi = torch.remainder(phi, 2.0 * math.pi)
        return phi / (2.0 * math.pi), theta / math.pi

    def solid_angle_rows(self, height, device, dtype):
        theta0 = torch.arange(height, device=device, dtype=dtype) / height * math.pi
        theta1 = (
            torch.arange(1, height + 1, device=device, dtype=dtype)
            / height
            * math.pi
        )
        return torch.cos(theta0) - torch.cos(theta1)

    def solid_angle_map(self, height, width, device, dtype):
        rows = self.solid_angle_rows(height, device, dtype).unsqueeze(1).expand(
            height, width
        )
        return rows * (2.0 * math.pi / width)

    def pixel_solid_angle(self, x, y, width, height):
        del x
        dtype = y.dtype if torch.is_floating_point(y) else torch.float32
        y = y.to(dtype)
        theta0 = y / height * math.pi
        theta1 = (y + 1.0) / height * math.pi
        return (2.0 * math.pi / width) * (torch.cos(theta0) - torch.cos(theta1))

    def sample_bilinear(self, envmap, dirs):
        height, width = envmap.shape[:2]
        u, v = self.dir_to_uv(dirs)
        x = u * width - 0.5
        y = v * height - 0.5
        x0_raw = torch.floor(x).to(torch.long)
        y0_raw = torch.floor(y).to(torch.long)
        wx = (x - x0_raw.to(x.dtype)).clamp(0.0, 1.0)
        wy = (y - y0_raw.to(y.dtype)).clamp(0.0, 1.0)

        x0 = torch.remainder(x0_raw, width)
        x1 = torch.remainder(x0_raw + 1, width)
        y0 = y0_raw.clamp(0, height - 1)
        y1 = (y0_raw + 1).clamp(0, height - 1)
        return self._gather_bilinear(envmap, x0, x1, y0, y1, wx, wy)


class OctahedralProjection(EnvmapProjection):
    def __init__(self):
        super().__init__("octahedral")

    def pixel_to_dir(self, x, y, width, height):
        u = x / width
        v = y / height
        px = u * 2.0 - 1.0
        py = v * 2.0 - 1.0
        pz = 1.0 - px.abs() - py.abs()

        folded_x = (1.0 - py.abs()) * _safe_sign(px)
        folded_y = (1.0 - px.abs()) * _safe_sign(py)
        mask = pz < 0.0
        nx = torch.where(mask, folded_x, px)
        ny = torch.where(mask, folded_y, py)
        dirs = torch.stack([nx, ny, pz], dim=-1)
        return F.normalize(dirs, dim=-1)

    def dir_to_uv(self, dirs):
        dirs = F.normalize(dirs, dim=-1)
        denom = dirs.abs().sum(dim=-1).clamp(min=1e-12)
        px = dirs[..., 0] / denom
        py = dirs[..., 1] / denom
        pz = dirs[..., 2]

        folded_x = (1.0 - py.abs()) * _safe_sign(px)
        folded_y = (1.0 - px.abs()) * _safe_sign(py)
        mask = pz < 0.0
        px = torch.where(mask, folded_x, px)
        py = torch.where(mask, folded_y, py)
        return (px + 1.0) * 0.5, (py + 1.0) * 0.5

    def pixel_solid_angle(self, x, y, width, height):
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        x = x.to(dtype)
        y = y.to(dtype)
        corners = (
            self.pixel_to_dir(x, y, width, height),
            self.pixel_to_dir(x + 1.0, y, width, height),
            self.pixel_to_dir(x + 1.0, y + 1.0, width, height),
            self.pixel_to_dir(x, y + 1.0, width, height),
        )
        return spherical_triangle_area(corners[0], corners[1], corners[2]) + (
            spherical_triangle_area(corners[0], corners[2], corners[3])
        )

    def _canonical_texel_indices(self, x_index, y_index, width, height):
        in_bounds = (
            (x_index >= 0)
            & (x_index < width)
            & (y_index >= 0)
            & (y_index < height)
        )
        safe_x = x_index.clamp(0, width - 1)
        safe_y = y_index.clamp(0, height - 1)
        if bool(in_bounds.all().item()):
            return safe_x, safe_y

        dirs = self.pixel_to_dir(
            x_index.to(torch.float32) + 0.5,
            y_index.to(torch.float32) + 0.5,
            width,
            height,
        )
        u, v = self.dir_to_uv(dirs)
        wrapped_x = torch.floor(u * width).to(torch.long).clamp(0, width - 1)
        wrapped_y = torch.floor(v * height).to(torch.long).clamp(0, height - 1)
        return (
            torch.where(in_bounds, safe_x, wrapped_x),
            torch.where(in_bounds, safe_y, wrapped_y),
        )

    def sample_bilinear(self, envmap, dirs):
        height, width = envmap.shape[:2]
        u, v = self.dir_to_uv(dirs)
        x = u * width - 0.5
        y = v * height - 0.5
        x0_raw = torch.floor(x).to(torch.long)
        y0_raw = torch.floor(y).to(torch.long)
        x1_raw = x0_raw + 1
        y1_raw = y0_raw + 1
        wx = (x - x0_raw.to(x.dtype)).clamp(0.0, 1.0)
        wy = (y - y0_raw.to(y.dtype)).clamp(0.0, 1.0)

        x00, y00 = self._canonical_texel_indices(x0_raw, y0_raw, width, height)
        x10, y10 = self._canonical_texel_indices(x1_raw, y0_raw, width, height)
        x01, y01 = self._canonical_texel_indices(x0_raw, y1_raw, width, height)
        x11, y11 = self._canonical_texel_indices(x1_raw, y1_raw, width, height)
        c00 = envmap[y00, x00]
        c10 = envmap[y10, x10]
        c01 = envmap[y01, x01]
        c11 = envmap[y11, x11]
        wx = wx.unsqueeze(-1)
        wy = wy.unsqueeze(-1)
        c0 = c00 * (1.0 - wx) + c10 * wx
        c1 = c01 * (1.0 - wx) + c11 * wx
        return c0 * (1.0 - wy) + c1 * wy


def build_projection(name=None):
    normalized = str(name or "latlong").strip().lower().replace("-", "_")
    if normalized in {"latlong", "equirect", "equirectangular", "lat_long"}:
        return LatLongProjection()
    if normalized in {"octahedral", "octahedron", "oct", "octa"}:
        return OctahedralProjection()
    raise ValueError(
        f"Unsupported envmap projection '{name}'. Use 'latlong' or 'octahedral'."
    )
