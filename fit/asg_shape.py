"""Shape estimation for insertion proposals from the residual in their caps.

A log-quadratic density is fitted to the residual mass around each proposal
center. Only its quadratic term is used: it sets the orientation and
anisotropy of the proposal, while the scale stays the insertion width derived
from the residual density, and the center is not moved.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from model.primitives import build_asg_tangent_frame


# Numeric fit codes recorded in debug proposal arrays.
SHAPE_STATUS = {
    "eligible": 0,
    "unidentifiable_geometry": 2,
    "zero_mass": 3,
    "singular_mass": 4,
    "nonconverged": 5,
    "nonpositive_curvature": 6,
}


class SphericalCapIndex:
    """Exact cap queries through sorted coordinate bounds; no spatial dependency."""

    def __init__(self, directions):
        self.directions = directions.detach().cpu().double().numpy()
        self.order = np.argsort(self.directions, axis=0, kind="stable")
        self.sorted = np.take_along_axis(self.directions, self.order, axis=0)

    def query(self, center, radius):
        center = np.asarray(center, dtype=np.float64)
        # The chord box contains the cap. Allow for unit-vector FP32 rounding;
        # the final cosine predicate is exactly the original estimator's test.
        chord = 2 * math.sin(float(radius) / 2) + 1e-6
        bounds = [
            np.searchsorted(self.sorted[:, axis], [center[axis] - chord, center[axis] + chord],
                            side="left")
            for axis in range(3)
        ]
        axis = min(range(3), key=lambda a: bounds[a][1] - bounds[a][0])
        lo, hi = bounds[axis]
        ids = self.order[lo:hi, axis]
        cosine = np.einsum("ij,j->i", self.directions[ids], center)
        return np.sort(ids[(cosine >= math.cos(float(radius))) & (cosine > 0)])


@dataclass(frozen=True)
class LocalASGShapeFit:
    precision: torch.Tensor | None
    tilt: torch.Tensor | None
    reason: str
    iterations: int = 0
    moment_error: float = 0.0


@dataclass(frozen=True)
class ASGShapeEstimates:
    precision: torch.Tensor
    eligible: torch.Tensor
    sample_count: torch.Tensor
    status: torch.Tensor
    iterations: torch.Tensor
    moment_error: torch.Tensor
    tilt: torch.Tensor


def _design(v, areas, cosine):
    """Whiten the geometric sufficient statistics without a ridge prior."""
    if len(v) < 6:  # Five independent centered statistics require six nodes.
        return None
    x, y = v.unbind(-1)
    features = torch.stack((x, y, x.square(), 2 * x * y, y.square()), -1)
    log_base = torch.log(areas) + torch.log(cosine)
    base = torch.softmax(log_base, 0)
    centered = features - (base[:, None] * features).sum(0)
    gram = centered.T @ (base[:, None] * centered)
    eigenvalues, axes = torch.linalg.eigh(gram)
    tolerance = torch.finfo(v.dtype).eps * max(len(v), 5) * eigenvalues[-1]
    if not bool(eigenvalues[0] > tolerance):
        return None
    whitening = axes * eigenvalues.rsqrt()[None, :]
    return centered @ whitening, log_base, whitening


@torch.no_grad()
def fit_local_asg_shape(v, solid_angles, cosine, residual_luminance):
    """Fit a log-quadratic density to the residual mass by moment matching.

    The density is proportional to a_j h_j exp(eta . f(v_j)), with node areas
    a_j, cosines h_j and f = (x, y, x^2, 2xy, y^2); its first and second
    moments equal those of the residual mass a_j L_j. Coordinates are
    v = E^T omega / w in the zero-theta tangent frame E, scaled by the insertion
    width w. The quadratic coefficients give the precision matrix. Computation
    is in float64; a failed fit returns no precision and the proposal stays
    isotropic.
    """
    v = torch.as_tensor(v, dtype=torch.float64)
    area, h, luma = [
        torch.as_tensor(value, device=v.device, dtype=v.dtype).reshape(-1)
        for value in (solid_angles, cosine, residual_luminance)
    ]
    if v.ndim != 2 or v.shape[1] != 2 or any(
        value.shape != (len(v),) for value in (area, h, luma)
    ):
        raise ValueError("shape samples must match [N, 2] coordinates")
    if any(not bool(torch.isfinite(value).all()) for value in (v, area, h, luma)):
        raise ValueError("shape samples must be finite")
    if bool((area <= 0).any() or (h <= 0).any() or (luma < 0).any()):
        raise ValueError("shape areas/cosines must be positive and residual nonnegative")
    design = _design(v, area, h)
    if design is None:
        return LocalASGShapeFit(None, None, "unidentifiable_geometry")
    features, log_base, whitening = design
    mass = area * luma
    if not bool(mass.sum() > 0):
        return LocalASGShapeFit(None, None, "zero_mass")
    p = mass / mass.sum()
    centered_v = v - (p[:, None] * v).sum(0)
    covariance = centered_v.T @ (p[:, None] * centered_v)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    eps = torch.finfo(v.dtype).eps
    if not bool(covariance_eigenvalues[0] > eps * covariance_eigenvalues[-1]):
        return LocalASGShapeFit(None, None, "singular_mass")
    moment = (p[:, None] * features).sum(0)
    natural = v.new_zeros(5)
    numerical_tolerance = math.sqrt(eps)

    def evaluate(parameters):
        logits = log_base + features @ parameters
        log_z = torch.logsumexp(logits, 0)
        q = torch.softmax(logits, 0)
        mean = (q[:, None] * features).sum(0)
        centered = features - mean
        return (
            log_z - parameters @ moment,
            mean - moment,
            centered.T @ (q[:, None] * centered),
        )

    error = 0.0
    for iteration in range(64):
        objective, gradient, hessian = evaluate(natural)
        error = float(gradient.abs().max())
        eigenvalues = torch.linalg.eigvalsh(hessian)
        if not bool(eigenvalues[0] > eps * 5 * eigenvalues[-1]):
            break
        delta = torch.linalg.solve(hessian, -gradient)
        # A vanishing gradient alone also occurs at an infinite boundary MLE.
        # Require the parameter step to vanish as well.
        if error <= numerical_tolerance and bool(
            delta.norm() <= numerical_tolerance * (1 + natural.norm())
        ):
            parameters = whitening @ natural
            precision = -torch.stack((
                torch.stack((parameters[2], parameters[3])),
                torch.stack((parameters[3], parameters[4])),
            ))
            curvature = torch.linalg.eigvalsh(precision)
            curvature_tolerance = eps * 64 * max(1.0, float(curvature.abs().max()))
            if not bool(curvature[0] > curvature_tolerance):
                return LocalASGShapeFit(
                    None, None, "nonpositive_curvature", iteration, error
                )
            return LocalASGShapeFit(
                precision, parameters[:2], "eligible", iteration, error
            )
        step = 1.0
        for _ in range(40):
            candidate = natural + step * delta
            candidate_objective = evaluate(candidate)[0]
            rounding = 8 * eps * max(1.0, abs(float(objective)))
            if bool(candidate_objective <= objective + 1e-4 * step * (gradient @ delta) + rounding):
                break
            step *= 0.5
        else:
            break
        if not bool(torch.isfinite(candidate).all()):
            break
        natural = candidate
    return LocalASGShapeFit(None, None, "nonconverged", iteration + 1, error)


@torch.no_grad()
def estimate_proposal_shapes(
    proposals, directions, solid_angles, residual_luminance, *, spatial_index=None,
):
    """Fit each proposal's shape on the quadrature nodes inside its insertion cap.

    The small, variable-size fits run on the CPU after a single transfer of
    the field, avoiding many tiny GPU launches. Rows whose fit fails are
    marked ineligible and keep their isotropic shape.
    """
    count = proposals.num_proposals
    if proposals.support_angular_radius is None:
        raise ValueError("shape estimation requires residual allocation radii")
    dirs, areas, luma, centers, widths, radii = [
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in (
            directions, solid_angles, residual_luminance, proposals.lobe_axis,
            proposals.local_width, proposals.support_angular_radius,
        )
    ]
    areas, luma = areas.reshape(-1), luma.reshape(-1)
    if dirs.shape != (len(areas), 3) or luma.shape != areas.shape:
        raise ValueError("shape field must match spherical quadrature")
    x, y, centers = build_asg_tangent_frame(centers, widths.new_zeros(count))
    # NumPy's explicit contraction avoids a CPU thread-pool launch for each
    # allocation cap mask. Tiny local moment fits remain in float64 torch.
    direction_array, center_array = dirs.numpy(), centers.numpy()
    spatial_index = spatial_index or SphericalCapIndex(dirs)
    area_array, luminance_array = areas.numpy(), luma.numpy()
    precision = torch.eye(2, dtype=torch.float64).repeat(count, 1, 1)
    eligible = torch.zeros(count, dtype=torch.bool)
    sample_count = torch.zeros(count, dtype=torch.long)
    status = torch.full_like(sample_count, SHAPE_STATUS["unidentifiable_geometry"])
    iterations = torch.zeros_like(sample_count)
    error = widths.new_zeros(count)
    tilt = widths.new_zeros((count, 2))
    for row in range(count):
        ids = spatial_index.query(center_array[row], radii[row])
        local_dirs = torch.from_numpy(direction_array[ids])
        h = torch.from_numpy(np.einsum("ij,j->i", direction_array[ids], center_array[row]))
        v = torch.stack((local_dirs @ x[row], local_dirs @ y[row]), -1) / widths[row]
        a = torch.from_numpy(area_array[ids])
        residual = torch.from_numpy(luminance_array[ids])
        fit = fit_local_asg_shape(v, a, h, residual)
        sample_count[row] = len(v)
        status[row] = SHAPE_STATUS[fit.reason]
        iterations[row], error[row] = fit.iterations, fit.moment_error
        if fit.precision is not None:
            precision[row], tilt[row] = fit.precision, fit.tilt
            eligible[row] = True
    device = proposals.lobe_axis.device
    return ASGShapeEstimates(*(
        value.to(device=device) for value in (
            precision, eligible, sample_count,
            status, iterations, error, tilt,
        )
    ))
