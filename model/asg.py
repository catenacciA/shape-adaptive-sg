import math
from dataclasses import dataclass, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.primitives import (
    build_asg_tangent_frame,
    inverse_softplus,
    normalize_vectors_3d,
    initialize_radiance_coefficients,
    sample_kernel_allocation_candidates,
)
from utils.fusion import fused_on_cuda


ASG_P_BOUNDS = (1.0, 4.0)
ASG_Q_BOUNDS = (1.25, 16.0)
# Beta footprint: the range of its learned power and the initial power.
BETA_BOUNDS = (1.0, 256.0)
BETA_INIT = 4.0


@dataclass(frozen=True)
class ASGFootprint:
    """Continuous shape and differentiable, per-row plane normalization.

    The normalized standardized footprint has covariance I/2 for every p, q.
    Prepare once per parameter conversion, then reuse across direction chunks.
    """

    p: torch.Tensor
    q: torch.Tensor
    log_b: torch.Tensor
    log_c: torch.Tensor

    @classmethod
    def from_exponents(cls, p, q, normalize=True):
        return cls(p, q, *(footprint_log_terms(p, q) if normalize else unnormalized_log_terms(p)))

    def __getitem__(self, rows):
        return ASGFootprint(*(value[rows] for value in (self.p, self.q, self.log_b, self.log_c)))

    def tensors(self):
        return (self.p, self.q, self.log_b, self.log_c)


@dataclass(frozen=True)
class BetaFootprint:
    """Beta profile (1 - r^2 / (beta + 2))_+^beta, the kernel ablation of the paper.

    r^2 = u^2 + v^2 in the standardized plane coordinates of every footprint.
    The support radius sqrt(beta + 2) matches the moments: like every member
    of the family the footprint has unit mass and covariance I/2, so lambda
    and mu keep their meaning, and beta -> infinity gives the ASG.
    """

    beta: torch.Tensor
    log_c: torch.Tensor

    @classmethod
    def from_power(cls, beta):
        return cls(beta, beta_log_norm(beta))

    def __getitem__(self, rows):
        return BetaFootprint(self.beta[rows], self.log_c[rows])

    def tensors(self):
        return (self.beta, self.log_c)


def beta_log_norm(beta):
    """log c of the Beta footprint: mass pi R^2 / (beta + 1) with R^2 = beta + 2."""
    return torch.log((beta + 1) / (math.pi * (beta + 2)))


def beta_profile(u, v, footprint):
    """(1 - (u^2 + v^2) / (beta + 2))_+^beta, zero outside the support, finite gradients."""
    base = 1 - (u.square() + v.square()) / (footprint.beta + 2)
    inside = base > 0
    safe = torch.where(inside, base, torch.ones_like(base))
    return torch.where(inside, torch.exp(footprint.beta * safe.log()), torch.zeros_like(base))


def unnormalized_log_terms(p):
    """Normalization off: b = 1 and c = 1/pi, the ASG values, for every (p, q)."""
    return torch.zeros_like(p), torch.full_like(p, -math.log(math.pi))


def footprint_log_terms(p, q):
    """Return (log b_pq, log c_pq), the plane normalization of Eq. 5."""
    # Only O(kernels) scalar work uses double precision. This avoids gamma
    # cancellation near the Gaussian member without widening sample tensors.
    pd, qd = p.double(), q.double()
    log_a = math.log(4) + 2 * torch.lgamma(1 + 1 / qd) - torch.lgamma(1 + 2 / qd)
    log_m = (torch.lgamma(3 / qd) + torch.lgamma(1 + 2 / qd)
             - torch.lgamma(1 / qd) - torch.lgamma(1 + 4 / qd))
    log_b = (math.log(2) + log_m + torch.lgamma(1 + 2 / pd)
             - torch.lgamma(1 + 1 / pd))
    log_c = log_b - log_a - torch.lgamma(1 + 1 / pd)
    return log_b.to(p.dtype), log_c.to(p.dtype)


@fused_on_cuda(training_only=True)
def render_parameters(lobe_axis, theta, log_lambda, log_mu, radiance_coeff_raw,
                      log_p=None, log_q_ratio=None, log_beta=None, normalize=True):
    """Canonical raw-to-render conversion of kernel rows.

    Returns (z, x, y, lambda, mu, coefficients), followed by
    (p, q, log_b, log_c) for the superelliptic footprint, or by
    (beta, log_c) for the Beta footprint.
    """
    x, y, z = build_asg_tangent_frame(normalize_vectors_3d(lobe_axis), theta)
    parameters = (z, x, y, log_lambda.exp(), log_mu.exp(), F.softplus(radiance_coeff_raw))
    if log_beta is not None:
        beta = log_beta.exp()
        return parameters + (beta, beta_log_norm(beta))
    if log_p is None:
        return parameters
    p, q = log_p.exp(), 2 * log_q_ratio.exp()
    return parameters + (p, q, *(footprint_log_terms(p, q) if normalize else unnormalized_log_terms(p)))


def superelliptic_exponent(u, v, footprint):
    """Return min((b ||(u,v)||_q^2)^p, 80), with finite center gradients.

    Maximum-coordinate scaling avoids large powers. Safe operands, rather than
    only masking the result, prevent undefined backward values at the origin.
    The footprint tensors broadcast with u and v.
    """
    magnitude = torch.maximum(u.abs(), v.abs())
    nonzero = magnitude > 0
    scale = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
    s = (u.abs() / scale).pow(footprint.q) + (v.abs() / scale).pow(footprint.q)
    s = torch.where(nonzero, s, torch.ones_like(s))
    log_t = footprint.p * (footprint.log_b + 2 * scale.log() + (2 / footprint.q) * s.log())
    t = log_t.clamp(max=math.log(80)).exp()
    return torch.where(nonzero, t, torch.zeros_like(t))


def asg_integral_norm(lambda_, mu_bandwidth, eps=1e-12, *, footprint=None):
    """Kernel normalization sqrt(lambda mu) c_pq, with c = 1/pi for the ASG.

    The clamped cosine cancels the Jacobian of the tangent-plane projection,
    so the kernel has unit mass over the full plane. Its spherical mass is
    the restriction to the unit disk (Eq. 6), which is not corrected for.
    """
    scale = torch.sqrt(lambda_.clamp(min=eps) * mu_bandwidth.clamp(min=eps))
    return scale / math.pi if footprint is None else scale * footprint.log_c.exp()


def clamp_asg_local_width(local_width, lambda_min, lambda_max):
    """Apply the production ASG concentration bounds in width space."""
    local_width = torch.as_tensor(local_width)
    return local_width.clamp(
        min=1.0 / math.sqrt(float(lambda_max)),
        max=1.0 / math.sqrt(float(lambda_min)),
    )


def bound_asg_concentrations(
    log_lambda,
    log_mu,
    lambda_min,
    lambda_max,
    ratio_max,
):
    """Return positive ASG concentrations with hard min/max and ratio bounds."""
    log_min = math.log(lambda_min)
    log_max = math.log(lambda_max)
    log_lambda = log_lambda.clamp(log_min, log_max)
    log_mu = log_mu.clamp(log_min, log_max)

    if ratio_max >= 1.0:
        log_center = 0.5 * (log_lambda + log_mu)
        log_aniso = 0.5 * (log_lambda - log_mu)
        max_aniso = min(
            0.5 * math.log(ratio_max), 0.5 * (log_max - log_min)
        )
        log_aniso = log_aniso.clamp(-max_aniso, max_aniso)
        abs_aniso = log_aniso.abs()
        log_center = torch.maximum(log_center, log_min + abs_aniso)
        log_center = torch.minimum(log_center, log_max - abs_aniso)
        log_lambda = log_center + log_aniso
        log_mu = log_center - log_aniso

    return (
        torch.exp(log_lambda).clamp(min=lambda_min, max=lambda_max),
        torch.exp(log_mu).clamp(min=lambda_min, max=lambda_max),
    )


def _trust_region_axes(axes, anchors, max_angle):
    """Move unit axes outside their angle around an anchor onto its boundary."""
    # Evaluate every row and keep unconstrained rows unchanged: masked
    # indexing and an any() test would synchronize with the device.
    rows = torch.isfinite(max_angle)
    centers = F.normalize(axes, dim=1)
    limits = max_angle.clamp(0.0, math.pi)
    dots = (centers * anchors).sum(dim=1).clamp(-1.0, 1.0)
    angles = torch.acos(dots)
    outside = angles > limits
    tangent = centers - dots.unsqueeze(1) * anchors
    tangent_norm = tangent.norm(dim=1, keepdim=True)
    fallback, _, _ = build_asg_tangent_frame(
        anchors,
        torch.zeros(
            anchors.shape[0],
            device=anchors.device,
            dtype=anchors.dtype,
        ),
    )
    tangent = torch.where(
        tangent_norm > 1e-12,
        tangent / tangent_norm.clamp(min=1e-12),
        fallback,
    )
    projected = (
        torch.cos(limits).unsqueeze(1) * anchors
        + torch.sin(limits).unsqueeze(1) * tangent
    )
    projected = torch.where(outside.unsqueeze(1), projected, centers)
    return torch.where(rows.unsqueeze(1), projected, axes)


@fused_on_cuda
def project_rows_(lobe_axis, log_lambda, log_mu, anchors, max_angle,
                  lambda_min, lambda_max, ratio_max):
    """Project kernel rows in place: unit axes, the center trust region (none
    when anchors is None) and the concentration bounds. Fused on CUDA, where
    the per-step update is otherwise bound by small kernel launches."""
    axes = F.normalize(lobe_axis, dim=1)
    if anchors is not None:
        axes = _trust_region_axes(axes, anchors, max_angle)
    lobe_axis.copy_(axes)
    lambda_, mu_bandwidth = bound_asg_concentrations(
        log_lambda, log_mu, lambda_min, lambda_max, ratio_max
    )
    log_lambda.copy_(torch.log(lambda_))
    log_mu.copy_(torch.log(mu_bandwidth))


@fused_on_cuda
def tangent_chart_correction(previous_centers, lobe_axis):
    """Theta increments that continue the tangent chart of previous_centers."""
    centers = normalize_vectors_3d(lobe_axis)
    old_north = previous_centers[:, 2] >= 0
    crossed = old_north != (centers[:, 2] >= 0)
    xy = centers[:, :2]
    xy = torch.where((xy.square().sum(-1) > 0)[:, None], xy, previous_centers[:, :2])
    x, y = xy.unbind(-1)
    rotation = torch.atan2(2 * x * y, y.square() - x.square())
    correction = torch.where(old_north, rotation, -rotation)
    return torch.where(crossed, correction, torch.zeros_like(correction))


@dataclass(frozen=True)
class ASGProposalParameters:
    """Fully initialized, insertion-ready positive-ASG proposal rows."""

    lobe_axis: torch.Tensor
    theta: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    lambda_: torch.Tensor
    mu_bandwidth: torch.Tensor
    log_lambda: torch.Tensor
    log_mu: torch.Tensor
    radiance_coeff_raw: torch.Tensor
    radiance_coeff: torch.Tensor
    local_width: torch.Tensor
    support_angular_radius: torch.Tensor | None
    log_p: torch.Tensor | None = None
    log_q_ratio: torch.Tensor | None = None
    log_beta: torch.Tensor | None = None
    normalize_shape: bool = True

    def footprint(self):
        if self.log_beta is not None:
            return BetaFootprint.from_power(self.log_beta.exp())
        if self.log_p is None:
            return None
        return ASGFootprint.from_exponents(self.log_p.exp(), 2 * self.log_q_ratio.exp(), self.normalize_shape)

    @property
    def num_proposals(self):
        return int(self.lobe_axis.shape[0])

    def index_select(self, indices):
        indices = torch.as_tensor(indices, device=self.lobe_axis.device, dtype=torch.long)
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        return replace(self, **{
            name: value.index_select(0, indices)
            for name, value in values.items() if isinstance(value, torch.Tensor)
        })


def render_asg_proposal_contributions(dirs, proposals):
    """Render each proposal separately as [samples, proposals, channels]."""
    dirs = F.normalize(
        torch.as_tensor(
            dirs,
            device=proposals.lobe_axis.device,
            dtype=proposals.lobe_axis.dtype,
        ),
        dim=1,
    )
    if proposals.num_proposals == 0:
        return dirs.new_empty((dirs.shape[0], 0, proposals.radiance_coeff.shape[-1]))
    weights = asg_kernel_weights(
        dirs, proposals.z, proposals.x, proposals.y, proposals.lambda_, proposals.mu_bandwidth,
        proposals.footprint(),
    )
    return weights.unsqueeze(-1) * proposals.radiance_coeff.unsqueeze(0)


def _project(dirs, axes):
    """Return the [kernels, samples] products of kernel axes and directions.

    Compiled code uses the equal pointwise sum, which fuses into the weight
    kernel; eager code transposes the [samples, kernels] matrix product.
    """
    if torch.compiler.is_compiling():
        return (axes[:, 0, None] * dirs[:, 0] + axes[:, 1, None] * dirs[:, 1]
                + axes[:, 2, None] * dirs[:, 2])
    return (dirs @ axes.t()).t()


@fused_on_cuda
def asg_kernel_weights(dirs, z, x, y, lambda_, mu_bandwidth, footprint=None):
    """Normalized kernel values K_i(omega_j) as a [samples, kernels] tensor (Eq. 3).

    Directions and tangent frames are unit vectors supplied by the caller.
    The values are computed kernel-major and returned transposed: the backward
    pass then sums per-kernel gradients along contiguous memory, which under
    torch.compile is several times faster than reducing across samples.
    """
    dx, dy, dz = _project(dirs, x), _project(dirs, y), _project(dirs, z)
    lambda_, mu_bandwidth = lambda_[:, None], mu_bandwidth[:, None]
    if isinstance(footprint, BetaFootprint):
        footprint = footprint[:, None]
        profile = beta_profile(dx * lambda_.sqrt(), dy * mu_bandwidth.sqrt(), footprint)
        return (dz.clamp_min(0) * profile
                * asg_integral_norm(lambda_, mu_bandwidth, footprint=footprint)).t()
    if footprint is not None:
        footprint = footprint[:, None]
        exponent = -superelliptic_exponent(dx * lambda_.sqrt(), dy * mu_bandwidth.sqrt(), footprint)
        return (dz.clamp_min(0) * exponent.exp()
                * asg_integral_norm(lambda_, mu_bandwidth, footprint=footprint)).t()
    exponent = (-lambda_ * dx.square() - mu_bandwidth * dy.square()).clamp(min=-80.0)
    return (dz.clamp_min(0.0) * torch.exp(exponent) * asg_integral_norm(lambda_, mu_bandwidth)).t()


@fused_on_cuda
def render_mixture(dirs, z, x, y, lambda_, mu_bandwidth, coefficients, footprint=None):
    """Mixture radiance sum_i a_i K_i(omega_j) as a [samples, channels] tensor (Eq. 1).

    Compiled code forms the channel sums in fused reduction kernels instead
    of a matrix product with only three columns; eager code multiplies the
    weight matrix.
    """
    # The undecorated weights, traced into the same compiled graph.
    weights = asg_kernel_weights.__wrapped__(dirs, z, x, y, lambda_, mu_bandwidth, footprint)
    if torch.compiler.is_compiling():
        return torch.stack([(weights * a).sum(1) for a in coefficients.unbind(1)], 1)
    return weights @ coefficients


def render_asg_proposal_aggregate(
    dirs,
    proposals,
    *,
    sample_chunk_size=8192,
    proposal_chunk_size=4096,
):
    """Render a proposal batch into [samples, channels] with bounded work."""
    dirs = torch.as_tensor(
        dirs,
        device=proposals.lobe_axis.device,
        dtype=proposals.lobe_axis.dtype,
    )
    sample_chunk = max(int(sample_chunk_size), 1)
    proposal_chunk = max(int(proposal_chunk_size), 1)
    result = dirs.new_zeros((dirs.shape[0], proposals.radiance_coeff.shape[-1]))
    footprint = proposals.footprint()
    for sample_start in range(0, dirs.shape[0], sample_chunk):
        sample_end = min(sample_start + sample_chunk, dirs.shape[0])
        total = result.new_zeros(
            (sample_end - sample_start, result.shape[-1])
        )
        sample_dirs = F.normalize(dirs[sample_start:sample_end], dim=1)
        for proposal_start in range(
            0, proposals.num_proposals, proposal_chunk
        ):
            proposal_end = min(
                proposal_start + proposal_chunk,
                proposals.num_proposals,
            )
            rows = slice(proposal_start, proposal_end)
            total += render_mixture(
                sample_dirs, proposals.z[rows], proposals.x[rows], proposals.y[rows],
                proposals.lambda_[rows], proposals.mu_bandwidth[rows],
                proposals.radiance_coeff[rows], None if footprint is None else footprint[rows],
            )
        result[sample_start:sample_end] = total
    return result


def replace_asg_proposal_coefficients(proposals, coefficients):
    """Return the same proposal geometry with explicit positive RGB weights."""
    coefficients = torch.as_tensor(
        coefficients,
        device=proposals.radiance_coeff.device,
        dtype=proposals.radiance_coeff.dtype,
    )
    if coefficients.shape != proposals.radiance_coeff.shape:
        raise ValueError("proposal coefficients must match [proposals, channels]")
    if not bool(torch.isfinite(coefficients).all().item()) or bool(
        (coefficients < 0.0).any().item()
    ):
        raise ValueError("proposal coefficients must be finite and non-negative")
    raw = inverse_softplus(coefficients)
    return replace(
        proposals,
        radiance_coeff_raw=raw,
        radiance_coeff=F.softplus(raw),
    )


def replace_asg_proposal_geometry(
    proposals, precision, eligible, *, lambda_min, lambda_max, width_ratio_max,
):
    """Set orientation and anisotropy from a tangent-plane precision matrix.

    Precision tensors are expressed in the zero-theta tangent frame at each
    proposal center. Centers, coefficients and the geometric mean of the two
    concentrations (the scale) are unchanged.
    """
    device, dtype = proposals.lobe_axis.device, proposals.lobe_axis.dtype
    precision = torch.as_tensor(precision, device=device, dtype=torch.float64)
    eligible = torch.as_tensor(eligible, device=device, dtype=torch.bool)
    count = proposals.num_proposals
    if precision.shape != (count, 2, 2) or eligible.shape != (count,):
        raise ValueError("shape precision and eligibility must match proposal rows")
    if not (0 < lambda_min <= lambda_max and width_ratio_max >= 1):
        raise ValueError("shape concentration and width-ratio bounds are invalid")
    if not all(math.isfinite(float(v)) for v in (lambda_min, lambda_max, width_ratio_max)):
        raise ValueError("shape bounds must be finite")
    if not bool(torch.isfinite(precision).all()) or not torch.allclose(
        precision, precision.transpose(-1, -2)
    ):
        raise ValueError("shape precision must be finite and symmetric")
    eigenvalues, axes = torch.linalg.eigh(precision)
    if bool((eligible & (eigenvalues[:, 0] <= 0)).any()):
        raise ValueError("eligible shape precision must be positive definite")
    safe = torch.where(eligible[:, None], eigenvalues, torch.ones_like(eigenvalues))
    anisotropy = 0.5 * (safe[:, 1].log() - safe[:, 0].log())
    log_base = 0.5 * (proposals.log_lambda.double() + proposals.log_mu.double())
    cap = torch.minimum(
        log_base - math.log(lambda_min), math.log(lambda_max) - log_base
    ).clamp(min=0, max=math.log(width_ratio_max))
    anisotropy = torch.minimum(anisotropy, cap)
    # Changes below the output representation's precision are exact no-ops.
    changed = eligible & (anisotropy > 64 * torch.finfo(dtype).eps)
    if not bool(changed.any()):
        return proposals, changed
    theta = torch.atan2(axes[:, 1, 0], axes[:, 0, 0]).remainder(math.pi).to(dtype)
    theta = torch.where(changed, theta, proposals.theta)
    log_lambda = torch.where(changed, (log_base - anisotropy).to(dtype), proposals.log_lambda)
    log_mu = torch.where(changed, (log_base + anisotropy).to(dtype), proposals.log_mu)
    # A cap bound at lambda_max rounds above it in float32; clamp as bound_asg_concentrations does.
    lambda_ = torch.where(changed, log_lambda.exp().clamp(lambda_min, lambda_max), proposals.lambda_)
    mu = torch.where(changed, log_mu.exp().clamp(lambda_min, lambda_max), proposals.mu_bandwidth)
    x, y, _ = build_asg_tangent_frame(proposals.lobe_axis, theta)
    return replace(
        proposals, theta=theta,
        x=torch.where(changed[:, None], x, proposals.x),
        y=torch.where(changed[:, None], y, proposals.y),
        z=proposals.z,
        lambda_=lambda_, mu_bandwidth=mu, log_lambda=log_lambda, log_mu=log_mu,
    ), changed


class ASGKernelBank(nn.Module):
    """Positive mixture of ASG or shape-adaptive kernels (Eq. 1).

    Each RGB coefficient is the full-plane integral of its lobe, which is
    close to its spherical integral for concentrated lobes (Eq. 6).
    """

    def __init__(
        self,
        args,
        init_allocation,
        target_radiance_integral,
        feat_dim,
        device,
        dtype,
    ):
        super().__init__()
        self.num_kernels = args.num_gaussians
        self.feat_dim = feat_dim
        self.device = device
        self.dtype = dtype
        self.seed = int(args.seed)
        self.asg_footprint = args.asg_footprint
        self.asg_normalize_shape = bool(args.asg_normalize_shape)
        self.lr_asg_shape = float(args.lr_asg_shape)

        self.asg_lambda_min = float(args.asg_lambda_min)
        self.asg_lambda_max = float(args.asg_lambda_max)
        self.width_ratio_max = max(
            float(args.kernel_width_ratio_max), 1.0
        )
        self.asg_ratio_max = self.width_ratio_max**2
        max_init_ratio = self.width_ratio_max
        self.asg_init_anisotropy_max = min(
            max(
                float(args.anisotropy_init_ratio_max),
                1.0,
            ),
            max_init_ratio,
        )
        self.lambda_asg_concentration = float(
            args.lambda_asg_concentration
        )
        self.asg_lambda_warmup_steps = int(args.asg_lambda_warmup_steps)
        self.asg_lambda_max_start = float(
            args.asg_lambda_max_start
        )
        self.asg_lambda_max_start = max(
            self.asg_lambda_min, min(self.asg_lambda_max, self.asg_lambda_max_start)
        )
        self.densify_center_max_width_scale = float(
            args.densify_center_max_width_scale
        )
        if (
            not math.isfinite(self.densify_center_max_width_scale)
            or self.densify_center_max_width_scale < 0.0
        ):
            raise ValueError(
                "densify_center_max_width_scale must be finite and non-negative"
            )

        self.lr_mu = float(args.lr_mu)
        self.lr_sigma = float(args.lr_sigma)
        self.lr_coeff = float(args.lr_coeff)
        self.lr_theta = float(args.lr_theta)

        (
            lobe_axis_init,
            sampled_values,
            theta_init,
            log_lambda_init,
            log_mu_init,
        ) = self._init_from_allocation(
            init_allocation,
            self.num_kernels,
            replacement=False,
            seed=self.seed,
        )

        self.lobe_axis = nn.Parameter(lobe_axis_init, requires_grad=True)
        self.theta = nn.Parameter(theta_init, requires_grad=True)
        self.log_lambda = nn.Parameter(log_lambda_init, requires_grad=True)
        self.log_mu = nn.Parameter(log_mu_init, requires_grad=True)
        if self.asg_footprint == "superelliptic":
            self.log_p = nn.Parameter(torch.zeros_like(log_lambda_init), requires_grad=args.asg_learn_p)
            self.log_q_ratio = nn.Parameter(torch.zeros_like(log_lambda_init), requires_grad=args.asg_learn_q)
        elif self.asg_footprint == "beta":
            self.log_beta = nn.Parameter(torch.full_like(log_lambda_init, math.log(BETA_INIT)), requires_grad=True)

        coeff_init = initialize_radiance_coefficients(
            sampled_values, target_radiance_integral
        )
        self.radiance_coeff_raw = nn.Parameter(coeff_init, requires_grad=True)
        self.project_parameters_()
        self.register_buffer(
            "residual_center_anchor",
            self.centers().detach().clone(),
        )
        self.register_buffer(
            "residual_center_max_angle",
            torch.full(
                (self.num_kernels,),
                float("inf"),
                device=self.device,
                dtype=self.dtype,
            ),
        )

    @property
    def lambda_(self):
        return self.concentrations()[0]

    @property
    def mu_bandwidth(self):
        return self.concentrations()[1]

    def concentrations(self):
        return torch.exp(self.log_lambda), torch.exp(self.log_mu)

    @property
    def radiance_coeff(self):
        """Positive nominal full-plane integrated-radiance weights."""
        return F.softplus(self.radiance_coeff_raw)

    def _init_from_allocation(
        self,
        init_allocation,
        count,
        replacement=True,
        seed=None,
    ):
        lobe_axis, sampled_values, indices = sample_kernel_allocation_candidates(
            init_allocation,
            count,
            return_indices=True,
            replacement=replacement,
            seed=seed,
        )
        theta = init_allocation.structure.theta_flat[indices].to(
            device=self.device, dtype=self.dtype
        )
        ratio = init_allocation.structure.anisotropy_ratio_flat[indices].to(
            device=self.device, dtype=self.dtype
        )
        ratio = ratio.clamp(1.0, self.asg_init_anisotropy_max)
        base_width = init_allocation.local_width_flat[indices].to(
            device=self.device, dtype=self.dtype
        )
        base_width = clamp_asg_local_width(
            base_width, self.asg_lambda_min, self.asg_lambda_max
        )
        log_base = -2.0 * torch.log(base_width)
        log_ratio = torch.log(ratio.to(device=self.device, dtype=self.dtype))
        lambda_init, mu_init = bound_asg_concentrations(
            log_base - log_ratio,
            log_base + log_ratio,
            self.asg_lambda_min,
            self.asg_lambda_max,
            self.asg_ratio_max,
        )
        return (
            lobe_axis,
            sampled_values,
            theta.to(device=self.device, dtype=self.dtype),
            torch.log(lambda_init),
            torch.log(mu_init),
        )

    def centers(self):
        return normalize_vectors_3d(self.lobe_axis)

    def current_parameters(self, proposals=None):
        """Canonical raw-to-render conversion, also for a prospective append."""
        axis, theta, log_lambda, log_mu, raw = (
            self.lobe_axis, self.theta, self.log_lambda, self.log_mu, self.radiance_coeff_raw
        )
        if proposals is not None:
            axis, theta, log_lambda, log_mu, raw = (
                torch.cat((value, getattr(proposals, name)))
                for name, value in zip(
                    ("lobe_axis", "theta", "log_lambda", "log_mu", "radiance_coeff_raw"),
                    (axis, theta, log_lambda, log_mu, raw),
                )
            )
        if self.asg_footprint == "beta":
            log_beta = self.log_beta if proposals is None else torch.cat((self.log_beta, proposals.log_beta))
            # Positional, so that the parameter is passed as a plain view (utils/fusion.py).
            parameters = render_parameters(axis, theta, log_lambda, log_mu, raw, None, None, log_beta)
            return parameters[:6] + (BetaFootprint(*parameters[6:]),)
        if self.asg_footprint != "superelliptic":
            return render_parameters(axis, theta, log_lambda, log_mu, raw)
        log_p, log_q_ratio = self.log_p, self.log_q_ratio
        if proposals is not None:
            log_p = torch.cat((log_p, proposals.log_p))
            log_q_ratio = torch.cat((log_q_ratio, proposals.log_q_ratio))
        if self.asg_normalize_shape:
            parameters = render_parameters(axis, theta, log_lambda, log_mu, raw, log_p, log_q_ratio)
        else:
            parameters = render_parameters(axis, theta, log_lambda, log_mu, raw, log_p, log_q_ratio, normalize=False)
        return parameters[:6] + (ASGFootprint(*parameters[6:]),)

    def render_directions_core(
        self, dirs, z, x, y, lambda_, mu_bandwidth, radiance_coeff, footprint=None
    ):
        return render_mixture(dirs, z, x, y, lambda_, mu_bandwidth, radiance_coeff, footprint)

    def render_directions_from_parameters(
        self,
        dirs,
        z,
        x,
        y,
        lambda_,
        mu_bandwidth,
        radiance_coeff,
        footprint=None,
        *,
        chunk_size,
    ):
        num_kernels = z.shape[0]
        if num_kernels == 0:
            return dirs.new_zeros((dirs.shape[0], self.feat_dim))
        if num_kernels <= chunk_size:
            return self.render_directions_core(
                dirs, z, x, y, lambda_, mu_bandwidth, radiance_coeff, footprint
            )

        result = torch.zeros(
            dirs.shape[0], self.feat_dim, device=dirs.device, dtype=dirs.dtype
        )
        # Split views back-propagate through a single concatenation per row
        # tensor, rather than a zero-filled gradient per slice and chunk.
        rows = (z, x, y, lambda_, mu_bandwidth, radiance_coeff)
        if footprint is not None:
            rows += footprint.tensors()
        for chunk in zip(*(value.split(chunk_size) for value in rows)):
            result += self.render_directions_core(
                dirs, *chunk[:6], type(footprint)(*chunk[6:]) if footprint is not None else None
            )
        return result

    def render_directions(self, dirs):
        dirs = F.normalize(dirs, dim=1)
        return self.render_directions_core(dirs, *self.current_parameters())

    def render_directions_chunked(self, dirs, chunk_size):
        dirs = F.normalize(dirs, dim=1)
        return self.render_directions_from_parameters(
            dirs, *self.current_parameters(), chunk_size=chunk_size
        )

    def get_optimizer_param_groups(self):
        groups = [
            {"params": [self.lobe_axis], "lr": self.lr_mu},
            {"params": [self.theta], "lr": self.lr_theta},
            {"params": [self.log_lambda, self.log_mu], "lr": self.lr_sigma},
            {"params": [self.radiance_coeff_raw], "lr": self.lr_coeff},
        ]
        if self.asg_footprint == "superelliptic":
            # Frozen ablation exponents stay in the optimizer so that
            # insertion extends every row tensor alike.
            groups.append({"params": [self.log_p, self.log_q_ratio], "lr": self.lr_asg_shape})
        elif self.asg_footprint == "beta":
            groups.append({"params": [self.log_beta], "lr": self.lr_asg_shape})
        return groups

    def scheduled_lambda_max(self, step):
        """Concentration cap, rising log-linearly to asg_lambda_max during warm-up."""
        if self.asg_lambda_warmup_steps <= 0:
            return self.asg_lambda_max
        progress = min(max(float(step) / max(1, self.asg_lambda_warmup_steps), 0.0), 1.0)
        log_start = math.log(self.asg_lambda_max_start)
        log_end = math.log(self.asg_lambda_max)
        return math.exp(log_start + (log_end - log_start) * progress)

    def _concentration_penalty(self, step, log_concentrations):
        """Mean squared excess of log concentrations over the scheduled cap.

        This is the penalty on narrow lobes. Once warm-up ends the cap equals
        asg_lambda_max, which the projection already enforces, so the penalty
        vanishes; with the defaults this happens at step 1000, the first
        insertion step.
        """
        if self.lambda_asg_concentration <= 0.0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        log_cap = math.log(max(self.asg_lambda_min, self.scheduled_lambda_max(step)))
        violation = torch.cat([F.relu(value - log_cap) for value in log_concentrations])
        return self.lambda_asg_concentration * violation.square().mean()

    def regularization_loss(self, step):
        return self._concentration_penalty(step, (self.log_lambda, self.log_mu))

    def prospective_regularization_loss(self, proposals, step):
        """Evaluate the mean penalty over incumbent and appended rows."""
        return self._concentration_penalty(step, (
            self.log_lambda, self.log_mu, proposals.log_lambda, proposals.log_mu
        ))

    @torch.no_grad()
    def preserve_tangent_chart_(self, previous_centers):
        """Compensate the coordinate rotation on a north/south chart crossing.

        The relative chart rotation depends only on longitude. This continues
        the previous chart at the new center without changing saved-model axes.
        At an exact pole use the incoming longitude (longitude is undefined).
        """
        self.theta.add_(tangent_chart_correction(previous_centers, self.lobe_axis))

    @torch.no_grad()
    def project_parameters_(self):
        # Trust region of inserted rows: each center stays within its stored
        # angle (densify_center_max_width_scale times the insertion radius) of
        # its insertion center. Initial rows have an infinite angle and move
        # freely. The constructor projects before these buffers exist.
        trust_region = (
            hasattr(self, "residual_center_max_angle")
            and self.densify_center_max_width_scale > 0.0
        )
        project_rows_(
            self.lobe_axis,
            self.log_lambda,
            self.log_mu,
            self.residual_center_anchor if trust_region else None,
            self.residual_center_max_angle if trust_region else None,
            self.asg_lambda_min,
            self.asg_lambda_max,
            self.asg_ratio_max,
        )
        if self.asg_footprint == "superelliptic":
            self.log_p.clamp_(math.log(ASG_P_BOUNDS[0]), math.log(ASG_P_BOUNDS[1]))
            self.log_q_ratio.clamp_(math.log(ASG_Q_BOUNDS[0] / 2), math.log(ASG_Q_BOUNDS[1] / 2))
        elif self.asg_footprint == "beta":
            self.log_beta.clamp_(math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1]))

    @torch.no_grad()
    def validate_densification_proposals(self, proposals):
        """Validate every insertion tensor before mutating the kernel bank."""
        count = int(proposals.num_proposals)
        vector_fields = ("lobe_axis", "x", "y", "z")
        scalar_fields = (
            "theta",
            "lambda_",
            "mu_bandwidth",
            "log_lambda",
            "log_mu",
            "local_width",
        )
        shape_fields = {"superelliptic": ("log_p", "log_q_ratio"), "beta": ("log_beta",)}
        own = shape_fields.get(self.asg_footprint, ())
        for name in own:
            if not isinstance(getattr(proposals, name), torch.Tensor):
                raise ValueError(f"{self.asg_footprint} proposal requires {name}")
        scalar_fields += own
        if any(getattr(proposals, name) is not None
               for names in shape_fields.values() for name in names if name not in own):
            raise ValueError("proposal rows carry another footprint's shape parameters")
        for name in vector_fields:
            value = getattr(proposals, name)
            if value.shape != (count, 3):
                raise ValueError(f"proposal {name} must have shape [count, 3]")
        for name in scalar_fields:
            value = getattr(proposals, name)
            if value.shape != (count,):
                raise ValueError(f"proposal {name} must have shape [count]")
        for name in ("radiance_coeff", "radiance_coeff_raw"):
            value = getattr(proposals, name)
            if value.shape != (count, self.feat_dim):
                raise ValueError(
                    f"proposal {name} must have shape [count, channels]"
                )
        for name in vector_fields + scalar_fields + (
            "radiance_coeff",
            "radiance_coeff_raw",
        ):
            value = getattr(proposals, name)
            if value.device != self.lobe_axis.device or value.dtype != self.dtype:
                raise ValueError(f"proposal {name} device or dtype mismatch")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"proposal {name} contains non-finite values")
        if bool((proposals.radiance_coeff < 0.0).any().item()):
            raise ValueError("proposal coefficients must be non-negative")
        if self.asg_footprint == "superelliptic":
            for name, bounds, offset in (("log_p", ASG_P_BOUNDS, 1),
                                          ("log_q_ratio", ASG_Q_BOUNDS, 2)):
                values = getattr(proposals, name)
                if bool(((values < math.log(bounds[0] / offset)) |
                         (values > math.log(bounds[1] / offset))).any()):
                    raise ValueError(f"proposal {name} is outside shape bounds")
        if self.asg_footprint == "beta":
            values = proposals.log_beta
            if bool(((values < math.log(BETA_BOUNDS[0])) | (values > math.log(BETA_BOUNDS[1]))).any()):
                raise ValueError("proposal log_beta is outside shape bounds")
        if bool((proposals.lambda_ < self.asg_lambda_min).any().item()) or bool(
            (proposals.lambda_ > self.asg_lambda_max).any().item()
        ) or bool((proposals.mu_bandwidth < self.asg_lambda_min).any().item()) or bool(
            (proposals.mu_bandwidth > self.asg_lambda_max).any().item()
        ):
            raise ValueError("proposal concentrations are outside ASG bounds")
        if proposals.support_angular_radius is not None and (
            proposals.support_angular_radius.shape != (count,)
            or not bool(torch.isfinite(proposals.support_angular_radius).all().item())
        ):
            raise ValueError("proposal support radii must be finite [count]")
        return True

    @torch.no_grad()
    def initialize_densification_proposals(
        self,
        directions,
        local_width,
        support_angular_radius,
        radiance_coeff_seed,
    ):
        """Create insertion-ready isotropic ASG rows from integrated seed mass."""
        directions = torch.as_tensor(
            directions, device=self.device, dtype=self.dtype
        )
        local_width = torch.as_tensor(
            local_width, device=self.device, dtype=self.dtype
        ).reshape(-1)
        support_angular_radius = torch.as_tensor(
            support_angular_radius, device=self.device, dtype=self.dtype
        ).reshape(-1)
        radiance_coeff_seed = torch.as_tensor(
            radiance_coeff_seed, device=self.device, dtype=self.dtype
        )
        num_new = int(directions.shape[0])
        if directions.shape != (num_new, 3):
            raise ValueError("proposal directions must have shape [P, 3]")
        if local_width.shape != (num_new,):
            raise ValueError("proposal widths must have shape [P]")
        if support_angular_radius.shape != (num_new,):
            raise ValueError("proposal support radii must have shape [P]")
        if radiance_coeff_seed.shape != (num_new, self.feat_dim):
            raise ValueError("proposal coefficient seeds must have shape [P, C]")
        for name, values in (
            ("directions", directions),
            ("widths", local_width),
            ("support radii", support_angular_radius),
            ("coefficient seeds", radiance_coeff_seed),
        ):
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"proposal {name} must be finite")
        if bool((local_width <= 0.0).any().item()):
            raise ValueError("proposal widths must be positive")
        if bool((support_angular_radius < 0.0).any().item()):
            raise ValueError("proposal support radii must be non-negative")
        if bool((radiance_coeff_seed < 0.0).any().item()):
            raise ValueError("proposal coefficient seeds must be non-negative")

        lobe_axis_new = normalize_vectors_3d(directions)
        theta_new = torch.zeros(num_new, dtype=self.dtype, device=self.device)
        base_width = local_width
        base_width = clamp_asg_local_width(
            base_width, self.asg_lambda_min, self.asg_lambda_max
        )
        log_base = -2.0 * torch.log(base_width)
        lambda_new, mu_new = bound_asg_concentrations(
            log_base,
            log_base,
            self.asg_lambda_min,
            self.asg_lambda_max,
            self.asg_ratio_max,
        )
        log_lambda_new = torch.log(lambda_new)
        log_mu_new = torch.log(mu_new)
        log_p = log_q_ratio = log_beta = None
        if self.asg_footprint == "superelliptic":
            log_p, log_q_ratio = torch.zeros_like(log_base), torch.zeros_like(log_base)
        elif self.asg_footprint == "beta":
            log_beta = torch.full_like(log_base, math.log(BETA_INIT))
        coeff_raw_new = inverse_softplus(radiance_coeff_seed)
        coeff_new = F.softplus(coeff_raw_new)
        x_new, y_new, z_new = build_asg_tangent_frame(
            lobe_axis_new, theta_new
        )
        return ASGProposalParameters(
            lobe_axis=lobe_axis_new,
            theta=theta_new,
            x=x_new,
            y=y_new,
            z=z_new,
            lambda_=lambda_new,
            mu_bandwidth=mu_new,
            log_lambda=log_lambda_new,
            log_mu=log_mu_new,
            radiance_coeff_raw=coeff_raw_new,
            radiance_coeff=coeff_new,
            local_width=base_width,
            support_angular_radius=support_angular_radius,
            log_p=log_p,
            log_q_ratio=log_q_ratio,
            log_beta=log_beta,
            normalize_shape=self.asg_normalize_shape,
        )

    @torch.no_grad()
    def append_densification_proposals(self, proposals):
        """Append positive rows, preserving every incumbent parameter and buffer."""
        self.validate_densification_proposals(proposals)
        for name, previous in list(self.named_parameters(recurse=False)):
            value = torch.cat((previous.detach(), getattr(proposals, name)), dim=0)
            current = nn.Parameter(value, requires_grad=previous.requires_grad)
            if previous.grad is not None:
                current.grad = torch.cat(
                    (previous.grad, torch.zeros_like(value[self.num_kernels :]))
                )
            self._parameters[name] = current
        self.residual_center_anchor = torch.cat(
            (
                self.residual_center_anchor,
                normalize_vectors_3d(proposals.lobe_axis.detach()),
            )
        )
        radius = proposals.support_angular_radius
        radius = proposals.local_width if radius is None else radius
        limits = (
            radius * self.densify_center_max_width_scale
            if self.densify_center_max_width_scale > 0
            else torch.full_like(radius, float("inf"))
        )
        self.residual_center_max_angle = torch.cat((self.residual_center_max_angle, limits))
        self.num_kernels += proposals.num_proposals
