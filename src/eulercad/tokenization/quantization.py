"""Coordinate and distance quantizers: the only place where geometry becomes integers.

Coordinates of the model frame, `[-1, 1]`, fall into equal bins. Segment lengths fall into
logarithmic bins, so short segments, the most common ones, keep their detail. The bin counts
and the distance range come from `TokenizerConfig`.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CoordinateQuantizer:
    """Equal bins over `[-1, 1]`: `bin(z) = min(bins - 1, floor((z + 1) * bins / 2))`.

    Attributes:
        bins: the number of bins per axis.
    """

    bins: int

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("the coordinate bins must be at least 1")

    @property
    def max_distance_error(self) -> float:
        """The most the distance between two bin centers can differ from that between the points.

        Each center is within `sqrt(2) / bins` of its point, and both endpoints can be off.
        """
        return 2 * math.sqrt(2) / self.bins

    def to_bins(self, z: torch.Tensor) -> torch.Tensor:
        """Map frame coordinates to bins.

        Returns:
            int64 bins, the same shape as `z`.

        Raises:
            ValueError: If a coordinate is not finite or lies outside `[-1, 1]`.
        """
        if not bool(z.isfinite().all()) or bool((z.abs() > 1).any()):
            raise ValueError("coordinates must be finite and within [-1, 1]")
        return ((z + 1) * (self.bins / 2)).floor().long().clamp_max(self.bins - 1)

    def centers(self, index: torch.Tensor) -> torch.Tensor:
        """Return the float64 frame coordinate at the middle of each bin: `-1 + (k + 0.5) * 2 / bins`.

        Returns:
            float64 coordinates, the same shape as `index`.

        Raises:
            ValueError: If a coordinate bin is out of range.
        """
        if index.numel() and (index.min() < 0 or index.max() >= self.bins):
            raise ValueError(f"every coordinate bin must be in [0, {self.bins - 1}]")
        return -1 + (index.double() + 0.5) / (self.bins / 2)


@dataclass(frozen=True, slots=True)
class DistanceQuantizer:
    """Logarithmic bins over segment lengths.

    `0` below `d_min`; otherwise `min(bins - 1, 1 + floor((bins - 1) * ln(d / d_min) / ln(d_max / d_min)))`.

    Attributes:
        bins: the number of bins, bin 0 included.
        d_min: the shortest length outside bin 0.
        d_max: the length where the last bin starts to absorb everything longer.
    """

    bins: int
    d_min: float
    d_max: float

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ValueError("the distance bins must be at least 2")
        if not (0 < self.d_min < self.d_max < math.inf):
            raise ValueError("the distance range must satisfy 0 < d_min < d_max, both finite")

    @property
    def ratio(self) -> float:
        """How many times wider each logarithmic bin is than the one before."""
        return (self.d_max / self.d_min) ** (1 / (self.bins - 1))

    def to_bins(self, d: torch.Tensor) -> torch.Tensor:
        """Map segment lengths to bins.

        Returns:
            int64 bins, the same shape as `d`.

        Raises:
            ValueError: If a distance is negative or not finite.
        """
        if not bool(d.isfinite().all()) or bool((d < 0).any()):
            raise ValueError("every distance must be finite and non-negative")
        scaled = torch.log(d.double().clamp_min(self.d_min) / self.d_min) / math.log(self.ratio)
        bins = 1 + scaled.floor().long()
        return torch.where(d < self.d_min, 0, bins).clamp_max(self.bins - 1)

    def bounds(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the float64 interval `[low, high)` of lengths in each bin.

        Bin 0 is `[0, d_min)`; the last bin has no upper end.

        Returns:
            `low` and `high`, each the same shape as `index`.

        Raises:
            ValueError: If a distance bin is out of range.
        """
        if index.numel() and (index.min() < 0 or index.max() >= self.bins):
            raise ValueError(f"every distance bin must be in [0, {self.bins - 1}]")
        k = index.double()
        low = torch.where(index == 0, 0.0, self.d_min * self.ratio ** (k - 1))
        high = torch.where(index == self.bins - 1, math.inf, self.d_min * self.ratio**k)
        return low, high
