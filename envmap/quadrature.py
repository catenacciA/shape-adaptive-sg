import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import torch
import torch.nn.functional as F

from envmap.projection import spherical_triangle_area
from envmap.statistics import compute_luminance, spherical_channel_integral


GRID_VERSION = "icosahedral_face_centers_vectorized_v2"
ADAPTIVE_FITTING_GRID_VERSION = "adaptive_icosahedral_children_v1"
ADAPTIVE_REFINEMENT_CRITERION = "rgb_l2_variation_luma_peak_v1"
ADAPTIVE_RGB_VARIATION_RELATIVE_TOLERANCE = 0.10
ADAPTIVE_RGB_VARIATION_GLOBAL_FLOOR = 0.01
ADAPTIVE_LUMINANCE_PEAK_GLOBAL_THRESHOLD = 1.0
DEFAULT_ADAPTIVE_PARENT_CHUNK_SIZE = 32768
TRAINING_ROTATION_ID = "none"
EVALUATION_ROTATION_ID = "axis_1_2_3_angle_pi_over_7_v1"


@dataclass(frozen=True)
class GeodesicQuadrature:
    directions: torch.Tensor
    solid_angles: torch.Tensor
    subdivision: int
    sample_count: int
    solid_angle_sum: float
    grid_id: str
    sha256: str
    grid_kind: str
    rotation_id: str
    rotation_axis: tuple[float, float, float]
    rotation_angle_radians: float
    rotation_matrix: tuple[tuple[float, float, float], ...]

    def metadata(self):
        return {
            "grid_version": GRID_VERSION,
            "grid_kind": self.grid_kind,
            "subdivision": int(self.subdivision),
            "sample_count": int(self.sample_count),
            "solid_angle_sum": float(self.solid_angle_sum),
            "grid_id": self.grid_id,
            "sha256": self.sha256,
            "rotation_id": self.rotation_id,
            "rotation_axis": list(self.rotation_axis),
            "rotation_angle_radians": float(self.rotation_angle_radians),
            "rotation_matrix": [list(row) for row in self.rotation_matrix],
        }


@dataclass(frozen=True)
class AdaptiveGeodesicLayout:
    """Packed one-level refinement of a complete base grid."""

    base_subdivision: int
    inspected_subdivision: int
    refined_parent_mask: torch.Tensor
    sample_offsets: torch.Tensor

    @property
    def base_count(self):
        return int(self.refined_parent_mask.numel())

    @property
    def refined_parent_count(self):
        return int(self.refined_parent_mask.sum().item())

    @property
    def sample_count(self):
        return int(self.sample_offsets[-1].item())

    @property
    def sample_counts(self):
        return self.sample_offsets[1:] - self.sample_offsets[:-1]


@dataclass(frozen=True)
class AdaptiveFittingQuadrature:
    """Target-dependent adaptive quadrature on which the mixture is fitted."""

    directions: torch.Tensor
    solid_angles: torch.Tensor
    target_values: torch.Tensor
    layout: AdaptiveGeodesicLayout
    sample_count: int
    solid_angle_sum: float
    grid_id: str
    sha256: str
    refinement_mask_sha256: str
    refined_luminance_energy_fraction: float

    @property
    def subdivision(self):
        """Base subdivision level, as recorded in the fitting metadata."""
        return int(self.layout.base_subdivision)

    @property
    def base_subdivision(self):
        return int(self.layout.base_subdivision)

    @property
    def inspected_subdivision(self):
        return int(self.layout.inspected_subdivision)

    @property
    def base_count(self):
        return self.layout.base_count

    @property
    def refined_parent_count(self):
        return self.layout.refined_parent_count

    @property
    def refined_parent_fraction(self):
        return self.refined_parent_count / max(self.base_count, 1)

    def to(self, *, device=None, dtype=None):
        """Move the quadrature to another device or dtype without rebuilding it."""
        if device is None:
            device = self.directions.device
        if dtype is None:
            dtype = self.directions.dtype
        layout = AdaptiveGeodesicLayout(
            base_subdivision=self.layout.base_subdivision,
            inspected_subdivision=self.layout.inspected_subdivision,
            refined_parent_mask=self.layout.refined_parent_mask.to(device=device),
            sample_offsets=self.layout.sample_offsets.to(device=device),
        )
        return AdaptiveFittingQuadrature(
            directions=self.directions.to(device=device, dtype=dtype),
            solid_angles=self.solid_angles.to(device=device, dtype=dtype),
            target_values=self.target_values.to(device=device, dtype=dtype),
            layout=layout,
            sample_count=self.sample_count,
            solid_angle_sum=self.solid_angle_sum,
            grid_id=self.grid_id,
            sha256=self.sha256,
            refinement_mask_sha256=self.refinement_mask_sha256,
            refined_luminance_energy_fraction=(
                self.refined_luminance_energy_fraction
            ),
        )

    def metadata(self):
        return {
            "grid_version": ADAPTIVE_FITTING_GRID_VERSION,
            "grid_kind": "fitting",
            "subdivision": self.base_subdivision,
            "base_subdivision": self.base_subdivision,
            "inspected_subdivision": self.inspected_subdivision,
            "base_sample_count": self.base_count,
            "refined_parent_count": self.refined_parent_count,
            "refined_parent_fraction": self.refined_parent_fraction,
            "sample_count": int(self.sample_count),
            "solid_angle_sum": float(self.solid_angle_sum),
            "grid_id": self.grid_id,
            "sha256": self.sha256,
            "refinement_mask_sha256": self.refinement_mask_sha256,
            "rotation_id": TRAINING_ROTATION_ID,
            "rotation_axis": [0.0, 0.0, 1.0],
            "rotation_angle_radians": 0.0,
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "refinement": {
                "criterion": ADAPTIVE_REFINEMENT_CRITERION,
                "rgb_variation_relative_tolerance": (
                    ADAPTIVE_RGB_VARIATION_RELATIVE_TOLERANCE
                ),
                "rgb_variation_global_floor": (
                    ADAPTIVE_RGB_VARIATION_GLOBAL_FLOOR
                ),
                "luminance_peak_global_threshold": (
                    ADAPTIVE_LUMINANCE_PEAK_GLOBAL_THRESHOLD
                ),
                "refined_luminance_energy_fraction": float(
                    self.refined_luminance_energy_fraction
                ),
            },
        }


def resolve_geodesic_subdivision(value, height, width):
    """Resolve an explicit or pixel-budget-matched geodesic subdivision."""
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(
            f"geodesic subdivision resolution must be positive, got {height}x{width}"
        )

    if isinstance(value, bool):
        raise ValueError("geodesic subdivision must be 'auto' or a non-negative integer")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            target_count = height * width
            raw = math.log(target_count / 20.0, 4.0)
            lower = max(int(math.floor(raw)), 0)
            upper = max(int(math.ceil(raw)), 0)
            lower_error = abs(math.log((20 * (4**lower)) / target_count))
            upper_error = abs(math.log((20 * (4**upper)) / target_count))
            if lower_error < upper_error or math.isclose(
                lower_error, upper_error, rel_tol=0.0, abs_tol=1e-15
            ):
                return lower
            return upper
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "geodesic subdivision must be 'auto' or a non-negative integer"
            ) from exc

    return validate_resolved_geodesic_subdivision(value)


def validate_resolved_geodesic_subdivision(value):
    """Validate a subdivision after any input-dependent resolution."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("resolved geodesic subdivision must be a non-negative integer")
    subdivision = int(value)
    if subdivision < 0:
        raise ValueError("resolved geodesic subdivision must be non-negative")
    return subdivision


def _base_icosahedron():
    t = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = torch.tensor(
        [
            (-1.0, t, 0.0),
            (1.0, t, 0.0),
            (-1.0, -t, 0.0),
            (1.0, -t, 0.0),
            (0.0, -1.0, t),
            (0.0, 1.0, t),
            (0.0, -1.0, -t),
            (0.0, 1.0, -t),
            (t, 0.0, -1.0),
            (t, 0.0, 1.0),
            (-t, 0.0, -1.0),
            (-t, 0.0, 1.0),
        ],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=torch.long,
    )
    return F.normalize(vertices, dim=1), faces


def _subdivide_triangles(triangles):
    """Vectorize one recursive spherical midpoint refinement.

    Children retain the canonical per-parent order: A, B, C, then the central face.
    Shared-edge midpoints are evaluated from the same ordered endpoint values, so
    duplicate triangle storage does not change the mathematical grid.
    """
    a, b, c = triangles.unbind(dim=1)
    ab = F.normalize(a + b, dim=1)
    bc = F.normalize(b + c, dim=1)
    ca = F.normalize(c + a, dim=1)

    children = triangles.new_empty((triangles.shape[0], 4, 3, 3))
    children[:, 0, 0] = a
    children[:, 0, 1] = ab
    children[:, 0, 2] = ca
    children[:, 1, 0] = b
    children[:, 1, 1] = bc
    children[:, 1, 2] = ab
    children[:, 2, 0] = c
    children[:, 2, 1] = ca
    children[:, 2, 2] = bc
    children[:, 3, 0] = ab
    children[:, 3, 1] = bc
    children[:, 3, 2] = ca
    return children.reshape(-1, 3, 3).contiguous()


def _solid_angle_pool(values, solid_angles):
    """Pool a final child axis with its exact solid-angle measure."""
    values = torch.as_tensor(values)
    solid_angles = torch.as_tensor(
        solid_angles, device=values.device, dtype=values.dtype
    )
    if values.ndim < 2 or solid_angles.shape != values.shape[:-1]:
        raise ValueError(
            "pooled values must end in channels and match solid-angle groups"
        )
    area_sum = solid_angles.sum(dim=-1)
    if not bool(torch.isfinite(area_sum).all().item()) or bool(
        (area_sum <= 0.0).any().item()
    ):
        raise ValueError("pooled solid-angle groups must be finite and positive")
    pooled = (
        (values * solid_angles.unsqueeze(-1)).sum(dim=-2)
        / area_sum.unsqueeze(-1)
    )
    return pooled, area_sum


def _adaptive_grid_hash(base_subdivision, inspected_subdivision, refined_mask):
    mask_bytes = (
        refined_mask.detach().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
    )
    mask_digest = hashlib.sha256(mask_bytes).hexdigest()
    identity = {
        "grid_version": ADAPTIVE_FITTING_GRID_VERSION,
        "grid_kind": "fitting",
        "base_subdivision": int(base_subdivision),
        "inspected_subdivision": int(inspected_subdivision),
        "criterion": ADAPTIVE_REFINEMENT_CRITERION,
        "rgb_variation_relative_tolerance": (
            ADAPTIVE_RGB_VARIATION_RELATIVE_TOLERANCE
        ),
        "rgb_variation_global_floor": ADAPTIVE_RGB_VARIATION_GLOBAL_FLOOR,
        "luminance_peak_global_threshold": (
            ADAPTIVE_LUMINANCE_PEAK_GLOBAL_THRESHOLD
        ),
        "refinement_mask_sha256": mask_digest,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, mask_digest


def _root_base_triangles(root_triangle, subdivision):
    triangles = root_triangle.reshape(1, 3, 3)
    for _ in range(int(subdivision)):
        triangles = _subdivide_triangles(triangles)
    return triangles


def build_adaptive_fitting_quadrature(
    field,
    base_subdivision,
    *,
    parent_chunk_size=DEFAULT_ADAPTIVE_PARENT_CHUNK_SIZE,
):
    """Build the adaptive fitting quadrature from base cells or their children.

    Each base cell is kept or replaced by its four children. Children are
    sampled, classified and packed one chunk of parents at a time, so the
    fully refined grid is never held in memory.
    """
    base_subdivision = validate_resolved_geodesic_subdivision(base_subdivision)
    inspected_subdivision = base_subdivision + 1
    parent_chunk_size = int(parent_chunk_size)
    if parent_chunk_size <= 0:
        raise ValueError("adaptive parent chunk size must be positive")
    base_count = 20 * (4**base_subdivision)

    source_values = field.values.reshape(-1, field.feat_dim)
    source_areas = field.solid_angle.reshape(-1)
    source_integral = spherical_channel_integral(source_values, source_areas)
    rgb_channel_count = min(int(source_integral.shape[0]), 3)
    source_rgb = source_integral[:rgb_channel_count]
    mean_base_rgb_energy = torch.linalg.vector_norm(source_rgb) / float(base_count)
    source_luminance_integral = compute_luminance(source_integral)
    mean_base_luminance_energy = source_luminance_integral / float(base_count)

    direction_chunks = []
    area_chunks = []
    target_chunks = []
    refinement_chunks = []
    base_luminance_energy_chunks = []
    vertices, base_faces = _base_icosahedron()
    root_triangles = vertices[base_faces]

    with torch.no_grad():
        for root_triangle in root_triangles:
            parent_triangles = _root_base_triangles(
                root_triangle, base_subdivision
            )
            for start in range(0, parent_triangles.shape[0], parent_chunk_size):
                parent = parent_triangles[start : start + parent_chunk_size]
                parent_directions = F.normalize(parent.sum(dim=1), dim=1).to(
                    device=field.device, dtype=field.dtype
                )
                child_triangles = _subdivide_triangles(parent)
                child_directions_flat = F.normalize(
                    child_triangles.sum(dim=1), dim=1
                ).to(device=field.device, dtype=field.dtype)
                child_areas = spherical_triangle_area(
                    child_triangles[:, 0],
                    child_triangles[:, 1],
                    child_triangles[:, 2],
                ).to(device=field.device, dtype=field.dtype).reshape(-1, 4)
                child_directions = child_directions_flat.reshape(-1, 4, 3)
                child_targets = field.sample(child_directions_flat).reshape(
                    -1, 4, field.feat_dim
                )
                if not bool(torch.isfinite(child_targets).all().item()):
                    raise ValueError("adaptive child target contains non-finite values")
                if bool((child_targets < 0.0).any().item()):
                    raise ValueError("adaptive child target must be non-negative")

                parent_targets, parent_areas = _solid_angle_pool(
                    child_targets, child_areas
                )
                child_rgb = child_targets[..., :rgb_channel_count]
                parent_rgb = parent_targets[..., :rgb_channel_count]
                rgb_variation = (
                    child_areas
                    * torch.linalg.vector_norm(
                        child_rgb - parent_rgb.unsqueeze(1), dim=-1
                    )
                ).sum(dim=1)
                cell_rgb_energy = parent_areas * torch.linalg.vector_norm(
                    parent_rgb, dim=-1
                )
                variation_refine = rgb_variation > (
                    ADAPTIVE_RGB_VARIATION_RELATIVE_TOLERANCE * cell_rgb_energy
                    + ADAPTIVE_RGB_VARIATION_GLOBAL_FLOOR * mean_base_rgb_energy
                )

                child_luminance = compute_luminance(child_targets)
                parent_luminance = (
                    (child_luminance * child_areas).sum(dim=1) / parent_areas
                )
                peak_excess = (
                    child_areas
                    * (child_luminance - parent_luminance.unsqueeze(1)).clamp_min(
                        0.0
                    )
                ).max(dim=1).values
                peak_refine = peak_excess > (
                    ADAPTIVE_LUMINANCE_PEAK_GLOBAL_THRESHOLD
                    * mean_base_luminance_energy
                )
                refined = variation_refine | peak_refine

                candidate_directions = torch.cat(
                    (parent_directions.unsqueeze(1), child_directions), dim=1
                )
                candidate_targets = torch.cat(
                    (parent_targets.unsqueeze(1), child_targets), dim=1
                )
                candidate_areas = torch.cat(
                    (parent_areas.unsqueeze(1), child_areas), dim=1
                )
                keep = torch.cat(
                    (
                        (~refined).unsqueeze(1),
                        refined.unsqueeze(1).expand(-1, 4),
                    ),
                    dim=1,
                )
                direction_chunks.append(candidate_directions[keep])
                target_chunks.append(candidate_targets[keep])
                area_chunks.append(candidate_areas[keep])
                refinement_chunks.append(refined)
                base_luminance_energy_chunks.append(
                    parent_areas * parent_luminance
                )

    directions = torch.cat(direction_chunks, dim=0).contiguous()
    solid_angles = torch.cat(area_chunks, dim=0).contiguous()
    target_values = torch.cat(target_chunks, dim=0).contiguous().detach()
    refined_parent_mask = torch.cat(refinement_chunks, dim=0).contiguous()
    base_luminance_energy = torch.cat(
        base_luminance_energy_chunks, dim=0
    ).contiguous()
    if refined_parent_mask.shape != (base_count,):
        raise RuntimeError("adaptive refinement mask does not match the base grid")
    sample_counts = torch.where(
        refined_parent_mask,
        torch.full_like(refined_parent_mask, 4, dtype=torch.long),
        torch.ones_like(refined_parent_mask, dtype=torch.long),
    )
    sample_offsets = torch.cat(
        (
            torch.zeros(1, device=field.device, dtype=torch.long),
            torch.cumsum(sample_counts, dim=0),
        )
    )
    layout = AdaptiveGeodesicLayout(
        base_subdivision=base_subdivision,
        inspected_subdivision=inspected_subdivision,
        refined_parent_mask=refined_parent_mask,
        sample_offsets=sample_offsets,
    )
    if directions.shape[0] != layout.sample_count:
        raise RuntimeError("adaptive packed layout does not match its sample tensors")
    if not bool(torch.isfinite(solid_angles).all().item()) or bool(
        (solid_angles <= 0.0).any().item()
    ):
        raise ValueError("adaptive solid angles must be finite and positive")
    solid_angle_sum = float(solid_angles.to(torch.float64).sum().item())
    if not math.isclose(solid_angle_sum, 4.0 * math.pi, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("adaptive quadrature solid angles must sum to 4*pi")
    expected_count = base_count + 3 * layout.refined_parent_count
    if layout.sample_count != expected_count:
        raise RuntimeError("adaptive sample count violates parent-child exclusivity")

    refined_energy = base_luminance_energy[refined_parent_mask].to(
        torch.float64
    ).sum()
    total_inspected_energy = base_luminance_energy.to(torch.float64).sum()
    refined_energy_fraction = float(
        (refined_energy / total_inspected_energy.clamp_min(1e-20)).item()
        if bool((total_inspected_energy > 0.0).item())
        else 0.0
    )
    digest, mask_digest = _adaptive_grid_hash(
        base_subdivision, inspected_subdivision, refined_parent_mask
    )
    grid_id = (
        f"{ADAPTIVE_FITTING_GRID_VERSION}/fitting/"
        f"base={base_subdivision}/inspected={inspected_subdivision}/"
        f"mask={mask_digest}"
    )
    return AdaptiveFittingQuadrature(
        directions=directions,
        solid_angles=solid_angles,
        target_values=target_values,
        layout=layout,
        sample_count=layout.sample_count,
        solid_angle_sum=solid_angle_sum,
        grid_id=grid_id,
        sha256=digest,
        refinement_mask_sha256=mask_digest,
        refined_luminance_energy_fraction=refined_energy_fraction,
    )


@lru_cache(maxsize=None)
def _canonical_geometry(subdivision):
    subdivision = validate_resolved_geodesic_subdivision(subdivision)

    vertices, faces = _base_icosahedron()
    triangles = vertices[faces]
    for _ in range(subdivision):
        triangles = _subdivide_triangles(triangles)

    directions = F.normalize(triangles.sum(dim=1), dim=1).contiguous()
    solid_angles = spherical_triangle_area(
        triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ).contiguous()
    expected_count = 20 * (4**subdivision)
    if directions.shape[0] != expected_count:
        raise RuntimeError(
            f"geodesic face count {directions.shape[0]} does not match {expected_count}"
        )
    if not torch.isfinite(directions).all() or not torch.isfinite(solid_angles).all():
        raise FloatingPointError("geodesic quadrature contains non-finite values")
    if not bool((solid_angles > 0.0).all().item()):
        raise ValueError("geodesic quadrature solid angles must be positive")
    if not torch.isclose(
        solid_angles.sum(),
        solid_angles.new_tensor(4.0 * math.pi),
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError("geodesic quadrature solid angles must sum to 4*pi")
    return directions, solid_angles


@lru_cache(maxsize=1)
def _evaluation_rotation():
    axis = torch.tensor((1.0, 2.0, 3.0), dtype=torch.float64)
    axis = axis / axis.norm()
    angle = math.pi / 7.0
    x, y, z = axis.unbind()
    skew = torch.stack(
        (
            torch.stack((x.new_tensor(0.0), -z, y)),
            torch.stack((z, x.new_tensor(0.0), -x)),
            torch.stack((-y, x, x.new_tensor(0.0))),
        )
    )
    identity = torch.eye(3, dtype=torch.float64)
    matrix = (
        math.cos(angle) * identity
        + (1.0 - math.cos(angle)) * torch.outer(axis, axis)
        + math.sin(angle) * skew
    )
    return axis, angle, matrix.contiguous()


def _hash_geometry(subdivision, grid_kind, rotation_id, axis, angle, matrix, dirs, areas):
    identity = {
        "grid_version": GRID_VERSION,
        "grid_kind": grid_kind,
        "subdivision": int(subdivision),
        "rotation_id": rotation_id,
        "rotation_axis": list(axis),
        "rotation_angle_radians": float(angle),
        "rotation_matrix": [list(row) for row in matrix],
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    digest.update(dirs.numpy().astype("<f8", copy=False).tobytes(order="C"))
    digest.update(areas.numpy().astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _cached_quadrature(subdivision, grid_kind):
    directions, solid_angles = _canonical_geometry(int(subdivision))
    if grid_kind == "training":
        axis = (0.0, 0.0, 1.0)
        angle = 0.0
        matrix_tensor = torch.eye(3, dtype=torch.float64)
        rotation_id = TRAINING_ROTATION_ID
        output_directions = directions
    elif grid_kind == "evaluation":
        axis_tensor, angle, matrix_tensor = _evaluation_rotation()
        axis = tuple(float(value) for value in axis_tensor.tolist())
        rotation_id = EVALUATION_ROTATION_ID
        output_directions = F.normalize(
            directions @ matrix_tensor.transpose(0, 1), dim=1
        ).contiguous()
    else:
        raise ValueError(f"unknown geodesic grid kind: {grid_kind}")

    if grid_kind == "training":
        axis = tuple(float(value) for value in axis)
    matrix = tuple(tuple(float(value) for value in row) for row in matrix_tensor.tolist())
    digest = _hash_geometry(
        subdivision,
        grid_kind,
        rotation_id,
        axis,
        angle,
        matrix,
        output_directions,
        solid_angles,
    )
    return output_directions, solid_angles, digest, rotation_id, axis, angle, matrix


def _build_quadrature(subdivision, grid_kind, device, dtype):
    subdivision = validate_resolved_geodesic_subdivision(subdivision)
    directions, solid_angles, digest, rotation_id, axis, angle, matrix = (
        _cached_quadrature(subdivision, grid_kind)
    )
    output_directions = directions.to(device=device, dtype=dtype).clone()
    output_areas = solid_angles.to(device=device, dtype=dtype).clone()
    grid_id = (
        f"{GRID_VERSION}/{grid_kind}/subdivision={subdivision}/"
        f"{rotation_id}"
    )
    return GeodesicQuadrature(
        directions=output_directions,
        solid_angles=output_areas,
        subdivision=subdivision,
        sample_count=int(output_directions.shape[0]),
        solid_angle_sum=float(solid_angles.sum().item()),
        grid_id=grid_id,
        sha256=digest,
        grid_kind=grid_kind,
        rotation_id=rotation_id,
        rotation_axis=axis,
        rotation_angle_radians=float(angle),
        rotation_matrix=matrix,
    )


def build_training_quadrature(subdivision, *, device=None, dtype=torch.float32):
    return _build_quadrature(subdivision, "training", device, dtype)


def build_evaluation_quadrature(subdivision, *, device=None, dtype=torch.float32):
    return _build_quadrature(subdivision, "evaluation", device, dtype)
