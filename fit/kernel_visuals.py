"""Diagnostic plot of the fitted kernel footprints over the target map."""
import math

import numpy as np
import torch

from envmap.field import EnvmapField
from envmap.projection import build_projection
from fit.artifacts import rgb_for_display


@torch.no_grad()
def asg_footprint_contours(representation, indices, *, relative_level=math.exp(-1), samples=96):
    """Actual spherical radiance contours, including the cosine factor.

    Each orthographic tangent ray is monotone from peak to zero at the horizon.
    Bisection is a plotting operation only; the primitive remains continuous.
    """
    from model.asg import BetaFootprint, beta_profile, superelliptic_exponent

    if not 0 < relative_level < 1 or samples < 4:
        raise ValueError("contours require a level in (0, 1) and at least four samples")
    arrays = representation.arrays
    z, x, y, lam, mu = (arrays[f"local_{name}"][indices] for name in ("z", "x", "y", "lambda", "mu"))
    footprint = representation.asg_footprint
    footprint = footprint[indices] if footprint is not None else None
    angle = torch.linspace(0, 2 * math.pi, samples + 1, device=z.device, dtype=z.dtype)
    cx, cy = angle.cos()[:, None], angle.sin()[:, None]
    u, v = cx * lam.sqrt(), cy * mu.sqrt()
    low, high = torch.zeros_like(u), torch.ones_like(u)
    for _ in range(32):
        radius = (low + high) / 2
        if isinstance(footprint, BetaFootprint):
            profile = beta_profile(radius * u, radius * v, footprint)
        else:
            t = ((radius * u).square() + (radius * v).square()).clamp(max=80) if footprint is None else (
                superelliptic_exponent(radius * u, radius * v, footprint)
            )
            profile = (-t).exp()
        level = (1 - radius.square()).clamp_min(0).sqrt() * profile
        inside = level > relative_level
        low, high = torch.where(inside, radius, low), torch.where(inside, high, radius)
    radius = ((low + high) / 2).T
    return (radius[..., None] * (cx.T[..., None] * x[:, None] + cy.T[..., None] * y[:, None])
            + (1 - radius.square()).clamp_min(0).sqrt()[..., None] * z[:, None])


def contour_indices(representation, limit=512):
    """Stable coefficient-luminance ordering: a nominal contribution proxy."""
    coefficients = representation.arrays["local_coefficients"].detach().double().cpu()
    importance = (coefficients[:, :3] @ coefficients.new_tensor([0.2126, 0.7152, 0.0722])
                  if coefficients.shape[1] >= 3 else coefficients.mean(-1))
    order = torch.argsort(importance, descending=True, stable=True)
    return order[importance[order] > 0][:limit].to(representation.device)


@torch.no_grad()
def save_kernel_plot(path, *, field, representation, hdr_scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width, height = 1024, 512
    projection = build_projection("latlong")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    directions = projection.pixel_to_dir(x.float() + 0.5, y.float() + 0.5, width, height)
    target = EnvmapField(field.values.detach().cpu(), field.projection).sample(directions.reshape(-1, 3))
    background = rgb_for_display(target.reshape(height, width, -1) / hdr_scale).numpy()
    centers = representation.arrays["local_z"].detach().cpu()
    u, v = projection.dir_to_uv(centers)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(background, extent=(0, 1, 1, 0))
    selected = contour_indices(representation)
    if selected.numel():
        contours = asg_footprint_contours(representation, selected).cpu()
        cu, cv = projection.dir_to_uv(contours)
        for xs, ys in zip(cu.numpy(), cv.numpy()):
            points = np.column_stack((xs, ys))
            # Duplicate neither the contour nor its seam-spanning segment.
            breaks = np.flatnonzero(np.abs(np.diff(xs)) > 0.5) + 1
            for segment in np.split(points, breaks):
                ax.plot(segment[:, 0], segment[:, 1], color="#00d8f0", linewidth=0.55, alpha=0.8)
    ax.scatter(u.numpy(), v.numpy(), s=3, c="#ffe244", linewidths=0)
    ax.set(xlim=(0, 1), ylim=(1, 0), aspect="auto")
    ax.set_axis_off()
    caption = (f"All {len(centers)} centers; {len(selected)} contours at 1/e peak, "
               "ranked by coefficient luminance\nNominal contribution proxy, not exact spherical energy")
    ax.set_title(caption, fontsize=10)
    fig.tight_layout(pad=0.5)
    try:
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
