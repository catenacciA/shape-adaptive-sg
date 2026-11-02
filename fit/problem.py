"""Construction of the spherical fitting problem."""

from envmap.dataset import GeodesicSphereDataset
from envmap.field import load_envmap_field
from envmap.statistics import compute_hdr_scale, spherical_channel_integral
from fit.allocation import build_kernel_allocation
from fit.loss import SphericalHDRLoss


def load_target(args, *, dtype, device):
    return load_envmap_field(
        args.data_root,
        args.input_path,
        dtype=dtype,
        device=device,
        projection_name=args.envmap_projection,
    )


def build_training_dataset(args, *, field, subdivision):
    return GeodesicSphereDataset(field, subdivision=subdivision, seed=int(args.seed))


def build_loss_setup(args, target, solid_angles, requested_hdr_scale):
    target_integral = spherical_channel_integral(target, solid_angles).detach()
    scale = compute_hdr_scale(target, solid_angles) if requested_hdr_scale == 0.0 else requested_hdr_scale
    loss = SphericalHDRLoss(charbonnier_eps=float(args.charbonnier_eps))
    return target_integral, scale, loss


def build_initial_allocation(
    args,
    *,
    field,
    dataset,
    densification_schedule,
):
    allocation = build_kernel_allocation(
        field,
        dataset.dirs,
        dataset.solid_angles,
        densification_schedule.initial_kernel_budget,
        target_values=dataset.target,
        uniform_fraction=float(args.allocation_uniform_fraction),
        clip_quantile=float(args.allocation_clip_quantile),
        anisotropy_ratio_max=float(args.anisotropy_init_ratio_max),
        structure=None,
    )
    return allocation
