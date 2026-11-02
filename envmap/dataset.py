import math
from dataclasses import dataclass

import torch

from envmap.quadrature import (
    AdaptiveFittingQuadrature,
    build_adaptive_fitting_quadrature,
    validate_resolved_geodesic_subdivision,
)


@dataclass
class SampleBatch:
    dirs: torch.Tensor
    target: torch.Tensor
    quadrature_weights: torch.Tensor
    sample_indices: torch.Tensor


class _ShuffledIndexStream:
    def __init__(self, size, seed, device="cpu"):
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("shuffled index stream size must be positive")
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.permutation = torch.empty(0, dtype=torch.long)
        self.cursor = 0
        self._reshuffle()

    def _reshuffle(self):
        # Seeded on the CPU, then copied once per epoch: batches are slices
        # on the device, without a host-to-device copy per batch.
        self.permutation = torch.randperm(
            self.size, generator=self.generator, device="cpu"
        ).to(self.device)
        self.cursor = 0

    def take(self, count):
        parts = []
        remaining = int(count)
        while remaining > 0:
            available = self.size - self.cursor
            take = min(remaining, available)
            parts.append(self.permutation[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == self.size:
                self._reshuffle()
        if not parts:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return torch.cat(parts)


class GeodesicSphereDataset:
    def __init__(
        self,
        field,
        subdivision=5,
        seed=0,
        quadrature=None,
    ):
        self.field = field
        self.subdivision = validate_resolved_geodesic_subdivision(subdivision)
        self.seed = int(seed)
        if quadrature is None:
            self.quadrature = build_adaptive_fitting_quadrature(
                field, self.subdivision
            )
        else:
            if not isinstance(quadrature, AdaptiveFittingQuadrature):
                raise TypeError("provided fitting quadrature must be adaptive")
            if quadrature.base_subdivision != self.subdivision:
                raise ValueError(
                    "provided fitting quadrature subdivision does not match "
                    "the training subdivision"
                )
            if quadrature.target_values.shape[-1] != field.feat_dim:
                raise ValueError(
                    "provided fitting quadrature channel count does not match "
                    "the training target"
                )
            self.quadrature = quadrature.to(
                device=field.device,
                dtype=field.dtype,
            )
        self.dirs = self.quadrature.directions
        self.solid_angles = self.quadrature.solid_angles
        self.num_samples = int(self.dirs.shape[0])
        self.target = self.quadrature.target_values
        self.quadrature_weights = (
            self.solid_angles * (self.num_samples / (4.0 * math.pi))
        )
        self._uniform_stream = _ShuffledIndexStream(
            self.num_samples, self.seed, self.field.device
        )

    def _uniform_batch(self, count):
        indices = self._uniform_stream.take(count)
        return SampleBatch(
            dirs=self.dirs[indices],
            target=self.target[indices],
            quadrature_weights=self.quadrature_weights[indices],
            sample_indices=indices,
        )

    def sample_batch(self, num_samples):
        num_samples = int(num_samples)
        if num_samples < 0:
            raise ValueError("number of geodesic samples must be non-negative")
        return self._uniform_batch(num_samples)
