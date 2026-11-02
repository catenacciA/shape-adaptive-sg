"""Shared fixtures for the test suite."""
import math
from types import SimpleNamespace

import torch

from envmap.field import EnvmapField
from envmap.projection import build_projection
from envmap.quadrature import build_training_quadrature
from envmap.statistics import spherical_channel_integral
from envmap.structure import SphericalStructureField
from fit.allocation import KernelAllocation
from model import build_kernel_bank
from model.primitives import build_asg_tangent_frame
from utils.config import load_training_profile, training_config_defaults
from utils.image_io import save_image


def _interval(t, edge, extent=1.0):
    """A unit interval [-extent, extent] convolved with a Gaussian of width edge."""
    return .5 * (torch.erf((t + extent) / (math.sqrt(2) * edge))
                 - torch.erf((t - extent) / (math.sqrt(2) * edge)))


def analytic_lobe(directions, kind, center, theta, width, aspect, edge):
    """Analytic HDR lobe on the sphere: 0.02 + 100 * cos * shape * rgb."""
    center = directions.new_tensor([center])
    x, y, z = build_asg_tangent_frame(center, directions.new_tensor([theta]))
    u = (directions @ x[0]) / (width * math.sqrt(aspect))
    v = (directions @ y[0]) / (width / math.sqrt(aspect))
    h = (directions @ z[0]).clamp_min(0)
    if kind == "gaussian":
        value = (-(u.square() + v.square())).exp()
    elif kind == "rectangle":
        value = _interval(u, edge) * _interval(v, edge)
    elif kind == "strip":
        value = _interval(u, edge, 3) * _interval(v, edge, .3)
    else:
        raise ValueError(f"unknown analytic lobe: {kind}")
    rgb = directions.new_tensor([1., .7, .3]).expand(*value.shape, 3)
    return .02 + 100 * (h * value)[..., None] * rgb


def lobe_target(resolution=32):
    """Octahedral FP32 target with a flat-topped, an elongated and a round lobe."""
    grid = torch.arange(resolution, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    dirs = build_projection("octahedral").pixel_to_dir(xx + .5, yy + .5, resolution, resolution)
    value = (
        analytic_lobe(dirs, "rectangle", [.6, .8, 0.], .37, .04, 3., .06)
        + analytic_lobe(dirs, "strip", [.3, -.4, -.866025403784], .7, .24, 2., .25)
        + analytic_lobe(dirs, "gaussian", [0., 0., 1.], 0., .12, 1., .12)
    )
    return value.float()


def asg_args(num_gaussians=2, **overrides):
    values = training_config_defaults()
    values.update(num_gaussians=num_gaussians, **overrides)
    return SimpleNamespace(**values)


def uniform_allocation(subdivision=1, dtype=torch.float64):
    """Uniform kernel allocation on a constant field over a geodesic grid."""
    projection = build_projection("latlong")
    envmap = torch.ones(8, 16, 3, dtype=dtype)
    field = EnvmapField(envmap, projection)
    quadrature = build_training_quadrature(subdivision, dtype=dtype)
    directions = quadrature.directions
    solid_angles = quadrature.solid_angles
    target = field.sample(directions)
    probability = solid_angles / solid_angles.sum()
    density = probability / solid_angles.clamp(min=1e-12)
    width = torch.rsqrt(math.pi * density)
    zeros = torch.zeros_like(probability)
    ones = torch.ones_like(probability)
    return KernelAllocation(
        directions=directions,
        solid_angles=solid_angles,
        target_values=target,
        probability_flat=probability,
        density_flat=density,
        local_width_flat=width,
        structure=SphericalStructureField(
            theta_flat=zeros,
            tangent_axis=torch.zeros_like(directions),
            detail_flat=zeros,
            coherence_flat=zeros,
            anisotropy_ratio_flat=ones,
            energy_flat=zeros,
            score_flat=zeros,
        ),
    )


def asg_bank(allocation, num_gaussians=2, **overrides):
    """Gaussian-footprint ASG bank initialized from an allocation."""
    return build_kernel_bank(
        asg_args(num_gaussians, **overrides),
        init_allocation=allocation,
        target_radiance_integral=spherical_channel_integral(
            allocation.target_values, allocation.solid_angles
        ),
        feat_dim=3,
        device=allocation.directions.device,
        dtype=allocation.directions.dtype,
    )


def superelliptic_bank(*, dtype=torch.float64, count=3, **overrides):
    """Shape-adaptive bank on a level-2 allocation; returns (bank, allocation)."""
    alloc = uniform_allocation(subdivision=2, dtype=dtype)
    cfg = training_config_defaults()
    cfg.update(num_gaussians=count, asg_footprint="superelliptic", device="cpu")
    cfg.update(overrides)
    bank = build_kernel_bank(
        SimpleNamespace(**cfg), init_allocation=alloc,
        target_radiance_integral=torch.ones(3, dtype=dtype),
        feat_dim=3, device="cpu", dtype=dtype,
    )
    return bank, alloc


def tiny_envmap():
    y = torch.linspace(0.0, 1.0, 4).view(4, 1, 1)
    x = torch.linspace(0.0, 2.0, 8).view(1, 8, 1)
    return torch.cat(
        [
            (0.1 + x + y).expand(4, 8, 1),
            (0.2 + 0.5 * x + y).expand(4, 8, 1),
            (0.3 + x + 0.25 * y).expand(4, 8, 1),
        ],
        dim=-1,
    )


def tiny_config(root, profile, run_name, **overrides):
    """Resolved-config dict for a 3-step CPU fit of a 4x8 latlong map."""
    input_path = root / "tiny.exr"
    if not input_path.exists():
        save_image(tiny_envmap(), str(input_path))
    cfg = load_training_profile(profile)
    cfg.update(
        {
            "data_root": str(root),
            "input_path": input_path.name,
            "output_dir": str(root),
            "log_dir": str(root / run_name),
            "device": "cpu",
            "envmap_projection": "latlong",
            "num_steps": 3,
            "geodesic_subdivision": 1,
            "log_every": 3,
            "evaluation_subdivision": None,
            "renderer_chunk_size": 16,
            **overrides,
        }
    )
    return cfg
