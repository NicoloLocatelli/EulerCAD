"""Tests for the coordinate and distance quantizers of the GraphGPT-based architecture."""

import math
from collections.abc import Callable

import pytest
import torch

from eulercad.tokenization.quantization import CoordinateQuantizer, DistanceQuantizer

COORDINATES = [CoordinateQuantizer(256), CoordinateQuantizer(64), CoordinateQuantizer(1)]
DISTANCES = [
    DistanceQuantizer(256, 0.001, 2 * math.sqrt(2)),
    DistanceQuantizer(32, 0.01, 2.0),
    DistanceQuantizer(2, 0.1, 1.0),
]


def test_256_coordinate_bins_split_the_frame_into_equal_bins() -> None:
    z = torch.tensor(
        [-1.0, -1.0 + 1 / 128 - 1e-12, -1.0 + 1 / 128, -0.5, 0.0, 0.5, 1.0 - 1e-12, 1.0], dtype=torch.float64
    )

    bins = CoordinateQuantizer(256).to_bins(z)

    assert bins.tolist() == [0, 0, 1, 64, 128, 192, 255, 255]
    assert bins.dtype == torch.int64


@pytest.mark.parametrize("quantizer", COORDINATES, ids=lambda quantizer: f"{quantizer.bins}")
def test_every_coordinate_is_within_half_a_bin_of_its_center(quantizer: CoordinateQuantizer) -> None:
    z = torch.cat(
        [torch.linspace(-1, 1, 100_001, dtype=torch.float64), torch.rand(10_000, dtype=torch.float64) * 2 - 1]
    )

    bins = quantizer.to_bins(z)
    error = (quantizer.centers(bins) - z).abs()

    assert torch.equal(bins.unique(), torch.arange(quantizer.bins))
    assert float(error.max()) <= 1 / quantizer.bins + 1e-15
    assert quantizer.max_distance_error == pytest.approx(2 * math.sqrt(2) / quantizer.bins)


@pytest.mark.parametrize("value", [-1.0000001, 1.0000001, float("nan"), float("inf")])
def test_a_coordinate_outside_the_frame_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        CoordinateQuantizer(256).to_bins(torch.tensor([0.0, value], dtype=torch.float64))


@pytest.mark.parametrize("index", [-1, 64])
def test_a_center_outside_the_bins_is_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="coordinate bin"):
        CoordinateQuantizer(64).centers(torch.tensor([index]))


def test_the_default_distance_bins_follow_the_logarithmic_scale() -> None:
    quantizer = DISTANCES[0]
    d = torch.tensor([0.0, 0.000999, 0.001, 0.5, 1.0, 2 * math.sqrt(2), 3.0], dtype=torch.float64)

    assert quantizer.to_bins(d).tolist() == [0, 0, 1, 200, 222, 255, 255]
    assert quantizer.ratio == pytest.approx(1.0317, abs=1e-4)


@pytest.mark.parametrize("quantizer", DISTANCES, ids=lambda quantizer: f"{quantizer.bins}")
def test_distance_bins_grow_with_distance_and_use_every_bin(quantizer: DistanceQuantizer) -> None:
    low, high = math.log10(quantizer.d_min) - 0.5, math.log10(quantizer.d_max)
    d = torch.cat([torch.zeros(1, dtype=torch.float64), torch.logspace(low, high, 100_000, dtype=torch.float64)])

    bins = quantizer.to_bins(d)

    assert bool((bins[1:] >= bins[:-1]).all())
    assert torch.equal(bins.unique(), torch.arange(quantizer.bins))


@pytest.mark.parametrize("quantizer", DISTANCES, ids=lambda quantizer: f"{quantizer.bins}")
def test_every_distance_lies_inside_its_bin_bounds(quantizer: DistanceQuantizer) -> None:
    low, high = math.log10(quantizer.d_min) - 1, math.log10(quantizer.d_max)
    d = torch.cat(
        [
            torch.logspace(low, high, 100_000, dtype=torch.float64),
            torch.rand(10_000, dtype=torch.float64) * quantizer.d_max,
            torch.tensor([0.0, quantizer.d_min, quantizer.d_max, 2 * quantizer.d_max], dtype=torch.float64),
        ]
    )

    lower, upper = quantizer.bounds(quantizer.to_bins(d))

    assert bool((lower <= d * (1 + 1e-12)).all())
    assert bool((d < upper * (1 + 1e-12)).all())


@pytest.mark.parametrize("quantizer", DISTANCES, ids=lambda quantizer: f"{quantizer.bins}")
def test_distance_bins_are_contiguous_with_a_constant_ratio(quantizer: DistanceQuantizer) -> None:
    low, high = quantizer.bounds(torch.arange(quantizer.bins))

    assert torch.equal(low[:1], torch.zeros(1, dtype=torch.float64))
    assert high[0].item() == pytest.approx(quantizer.d_min)
    assert math.isinf(high[-1].item())
    assert torch.allclose(low[1:], high[:-1])
    assert torch.allclose(high[1:-1] / low[1:-1], torch.full_like(low[1:-1], quantizer.ratio))


@pytest.mark.parametrize("value", [-1e-9, float("nan"), float("inf")])
def test_a_negative_or_non_finite_distance_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="distance"):
        DISTANCES[0].to_bins(torch.tensor([value], dtype=torch.float64))


@pytest.mark.parametrize("index", [-1, 32])
def test_bounds_outside_the_bins_are_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="distance bin"):
        DISTANCES[1].bounds(torch.tensor([index]))


@pytest.mark.parametrize(
    "build",
    [
        lambda: CoordinateQuantizer(0),
        lambda: DistanceQuantizer(1, 0.001, 1.0),
        lambda: DistanceQuantizer(8, 0.0, 1.0),
        lambda: DistanceQuantizer(8, 1.0, 1.0),
        lambda: DistanceQuantizer(8, 0.001, math.inf),
    ],
)
def test_invalid_settings_are_rejected(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="must"):
        build()
