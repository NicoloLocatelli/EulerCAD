"""Tests for the layer-by-layer measures of `eulercad/analysis/measures.py`, and for `joint_dime`,
the DiME of the Layer by Layer paper, in `eulercad/analysis/metrics/relational.py`."""

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from eulercad.analysis.layers import hidden_states
from eulercad.analysis.measures import (
    METRICS,
    drawing_metrics,
    layer_metrics,
    mean_rows,
    normalize_rows,
    plot_metrics,
)
from eulercad.analysis.metrics.geometric import curvature
from eulercad.analysis.metrics.relational import joint_dime
from eulercad.analysis.metrics.spectral import matrix_based_entropy
from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.representation.euler import euler_walk
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.tokenization.vocabulary import PAD

SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))


def _cloud(rows: int, width: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, width))


def test_joint_dime_is_large_for_matching_views_and_small_for_unrelated_ones() -> None:
    views = _cloud(60, 8, 0)
    matching = joint_dime(views, views + 0.01 * _cloud(60, 8, 1), rng=np.random.default_rng(0))
    unrelated = joint_dime(views, _cloud(60, 8, 2), rng=np.random.default_rng(0))
    assert matching > 0.3
    assert abs(unrelated) < 0.2 * matching


def test_joint_dime_is_reproducible_with_one_seed() -> None:
    first, second = _cloud(30, 5, 3), _cloud(30, 5, 4)
    assert joint_dime(first, second, rng=np.random.default_rng(7)) == joint_dime(
        first, second, rng=np.random.default_rng(7)
    )


def test_joint_dime_refuses_views_of_different_sizes() -> None:
    with pytest.raises(ValueError, match="same drawings"):
        joint_dime(_cloud(10, 4, 0), _cloud(11, 4, 1))


def test_normalize_rows_centers_and_gives_unit_rows() -> None:
    rows = normalize_rows(_cloud(20, 6, 5) + 3.0)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)
    unit = _cloud(20, 6, 5) + 3.0
    unit -= unit.mean(axis=0)
    assert np.allclose(rows, unit / np.linalg.norm(unit, axis=1, keepdims=True))


def test_drawing_metrics_use_the_shared_metrics_on_normalized_rows() -> None:
    rows = _cloud(12, 6, 6)
    entropy, angle = drawing_metrics(rows)
    unit = normalize_rows(rows)
    assert entropy == pytest.approx(matrix_based_entropy(unit) / math.log(6))
    assert angle == pytest.approx(curvature(unit).mean())


def test_drawing_metrics_leave_curvature_undefined_under_three_rows() -> None:
    _, angle = drawing_metrics(_cloud(2, 6, 7))
    assert math.isnan(angle)


def test_layer_metrics_reward_views_that_agree() -> None:
    pooled = _cloud(80, 10, 8)
    agree = layer_metrics(pooled, pooled, pooled + 0.01 * _cloud(80, 10, 9), np.random.default_rng(0))
    disagree = layer_metrics(pooled, pooled, _cloud(80, 10, 10), np.random.default_rng(0))
    assert set(agree) == set(METRICS)
    assert agree["infonce"] < disagree["infonce"]
    assert agree["dime"] > disagree["dime"]
    assert agree["lidar"] > disagree["lidar"]
    assert 0 < agree["dataset_entropy"] <= 1


def test_mean_rows_averages_only_real_rows() -> None:
    states = torch.tensor([[[1.0, 3.0], [3.0, 5.0]], [[2.0, 2.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, True], [True, False]])
    assert torch.allclose(mean_rows(states, mask), torch.tensor([[2.0, 4.0], [2.0, 2.0]]))


def test_plot_metrics_draws_one_figure_per_metric_and_an_overview(tmp_path: Path) -> None:
    names = ["rows", "layer_1", "layer_2", "final_norm"]
    layers = [{"layer": name} | {metric: float(index) for metric in METRICS} for index, name in enumerate(names)]
    written = plot_metrics(layers, tmp_path)
    assert sorted(path.name for path in written) == sorted([f"{metric}.png" for metric in METRICS] + ["overview.png"])
    assert all(path.stat().st_size > 0 for path in written)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    graphs = [
        ([[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]], [(0, 1), (1, 2), (2, 3), (0, 3)]),
        ([[-0.5, 0.0], [0.5, 0.0], [0.0, 0.8]], [(0, 1), (1, 2), (0, 2)]),
    ]
    samples = []
    for seed, (positions, pairs) in enumerate(graphs):
        edges = torch.tensor(sorted(pairs), dtype=torch.int64).T
        graph = DrawingGraph(0, torch.tensor(positions, dtype=torch.float64), edges, (0.0, 0.0), 1.0)
        rng = random.Random(seed)  # noqa: S311  # nosec B311
        samples.append(SMALL.tokenize(graph, euler_walk(graph, rng), rng).rows)
    longest = max(len(sample) for sample in samples)
    input_ids = torch.full((len(samples), longest, SMALL.row_width), PAD, dtype=torch.int64)
    mask = torch.zeros((len(samples), longest), dtype=torch.bool)
    for index, sample in enumerate(samples):
        input_ids[index, : len(sample)] = sample
        mask[index, : len(sample)] = True
    return input_ids, mask


def test_hidden_states_give_the_rows_every_layer_and_the_final_norm() -> None:
    torch.manual_seed(0)
    config = ModelConfig(hidden_size=32, hidden_layers=3, attention_heads=4, qk_norm=True, value_residual=True)
    model = GraphGPTModel(SMALL, config).eval()
    input_ids, mask = _batch()
    with torch.no_grad():
        states = hidden_states(model, input_ids, mask)
        encoded = model.encode(input_ids, mask)
    assert list(states) == ["rows", "layer_1", "layer_2", "layer_3", "final_norm"]
    assert all(state.shape == encoded.shape for state in states.values())
    assert torch.allclose(states["final_norm"], encoded)
    kept = dict(states)
    with torch.no_grad():
        model.encode(input_ids[:1], mask[:1])
    assert all(states[name] is kept[name] for name in kept)  # the hooks are gone: nothing was overwritten
