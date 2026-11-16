"""Tests for the geometric input features of the GraphGPT-based architecture."""

import json
import math
import random

import pytest
import torch

from eulercad.model.config import ModelConfig
from eulercad.model.geometry import (
    FREQUENCIES,
    VISITS,
    GeometryFeatures,
    geometry_features,
)
from eulercad.model.model import GraphGPTModel
from eulercad.representation.euler import euler_walk
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.tokenizer import GraphTokenizer, PlaceTokenizer
from eulercad.tokenization.vocabulary import D_NONE, EDGE, JUMP, MASK, PAD

TOKENIZER = GraphTokenizer()
VOCABULARY = TOKENIZER.vocabulary
BINS = TOKENIZER.config.coordinate_bins
# Where each group of features starts in the last dimension.
FOURIER = 0
STEP = 4 * FREQUENCIES
DIRECTION = STEP + 2
VISIT = DIRECTION + 2
VALID = VISIT + VISITS


def _center(index: int) -> float:
    return -1 + (index + 0.5) * 2 / BINS


def _row(node: int, x: int, y: int, kind: int = EDGE) -> list[int]:
    return [VOCABULARY.node.start + node, VOCABULARY.x.start + x, VOCABULARY.y.start + y, kind, D_NONE]


def _rows(*rows: list[int]) -> torch.Tensor:
    return torch.tensor([rows], dtype=torch.int64)


def _features(input_ids: torch.Tensor) -> torch.Tensor:
    return geometry_features(input_ids, VOCABULARY, BINS)


def test_there_is_one_feature_vector_per_row() -> None:
    features = _features(_rows(_row(0, 10, 20), _row(1, 30, 40)))

    assert tuple(features.shape) == (1, 2, GeometryFeatures.width())
    assert features.dtype == torch.float32


def test_coordinates_are_sine_and_cosine_waves_of_the_bin_centers() -> None:
    features = _features(_rows(_row(0, 10, 200)))[0, 0]
    x, y = _center(10), _center(200)
    expected = []
    for value in (x, y):
        for k in range(FREQUENCIES):
            angle = math.pi * 2**k * value
            expected += [math.sin(angle), math.cos(angle)]

    torch.testing.assert_close(features[FOURIER:STEP], torch.tensor(expected), rtol=0, atol=1e-5)


def test_the_step_and_direction_come_from_the_previous_row() -> None:
    features = _features(_rows(_row(0, 10, 10), _row(1, 50, 10), _row(2, 50, 13, JUMP)))[0]
    step = _center(50) - _center(10)

    torch.testing.assert_close(features[0, STEP:VISIT], torch.zeros(4))
    torch.testing.assert_close(features[1, STEP:DIRECTION], torch.tensor([step, 0.0]), rtol=0, atol=1e-6)
    torch.testing.assert_close(features[1, DIRECTION:VISIT], torch.tensor([1.0, 0.0]), rtol=0, atol=1e-6)
    torch.testing.assert_close(
        features[2, STEP:DIRECTION], torch.tensor([0.0, _center(13) - _center(10)]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(features[2, DIRECTION:VISIT], torch.tensor([0.0, 1.0]), rtol=0, atol=1e-6)


def test_visits_count_the_rows_so_far_with_the_same_identity() -> None:
    rows = [_row(0, 1, 1), _row(1, 2, 2), _row(0, 1, 1), _row(1, 2, 2), _row(0, 1, 1), _row(0, 1, 1), _row(0, 1, 1)]
    visits = _features(_rows(*rows))[0, :, VISIT:VALID]

    counts = [1, 1, 2, 2, 3, 4, 5]
    expected = torch.zeros(len(counts), VISITS)
    for row, count in enumerate(counts):
        expected[row, min(count, VISITS) - 1] = 1.0
    torch.testing.assert_close(visits, expected)


@pytest.mark.parametrize("hidden", ["pad", "mask"])
def test_rows_without_coordinates_have_no_features_and_break_the_next_step(hidden: str) -> None:
    input_ids = _rows(_row(0, 10, 10), _row(1, 20, 20), _row(2, 30, 30))
    if hidden == "pad":
        input_ids[0, 1] = PAD
    else:
        input_ids[0, 1, 1:3] = MASK
    features = _features(input_ids)[0]

    torch.testing.assert_close(features[:, VALID], torch.tensor([1.0, 0.0, 1.0]))
    assert not features[1, :VISIT].any()
    torch.testing.assert_close(features[2, STEP:VISIT], torch.zeros(4))
    if hidden == "pad":
        assert not features[1].any()


def test_a_row_depends_only_on_itself_and_the_rows_before_it() -> None:
    input_ids = _rows(_row(0, 10, 10), _row(1, 50, 10), _row(0, 10, 10), _row(2, 70, 90))
    changed = input_ids.clone()
    changed[0, 2] = torch.tensor(_row(3, 99, 99))

    before, after = _features(input_ids)[0], _features(changed)[0]

    torch.testing.assert_close(after[:2], before[:2])
    assert not torch.equal(after[2], before[2])
    assert not torch.equal(after[3], before[3])


def test_the_features_match_the_tokenizer_bin_centers() -> None:
    positions = torch.tensor([[-0.9, -0.9], [0.3, -0.9], [0.3, 0.7], [-0.9, 0.7]], dtype=torch.float64)
    edges = torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]], dtype=torch.int64)
    graph = DrawingGraph(0, positions, edges, (0.0, 0.0), 1.0)
    rng = random.Random(0)  # noqa: S311  # nosec B311
    walk = euler_walk(graph, rng)
    rows = TOKENIZER.tokenize(graph, walk, rng).rows
    features = _features(rows[None])[0]

    centers = TOKENIZER.coordinates.centers(TOKENIZER.coordinates.to_bins(positions))[walk.nodes].float()
    steps = centers[1:] - centers[:-1]
    torch.testing.assert_close(features[1:, STEP:DIRECTION], steps, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        features[1:, DIRECTION:VISIT], steps / steps.norm(dim=1, keepdim=True), rtol=0, atol=1e-6
    )


# In the model


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full((2, 4, TOKENIZER.row_width), PAD, dtype=torch.int64)
    input_ids[0] = _rows(_row(0, 10, 10), _row(1, 50, 10), _row(0, 10, 10), _row(2, 70, 90))[0]
    input_ids[1, :2] = _rows(_row(0, 100, 30), _row(1, 120, 40))[0]
    attention_mask = input_ids[..., 0] != PAD
    return input_ids, attention_mask


def _model(geometry: bool, tokenizer: GraphTokenizer = TOKENIZER) -> GraphGPTModel:
    torch.manual_seed(0)
    return GraphGPTModel(tokenizer, ModelConfig(geometry_features=geometry)).eval()


def test_the_setting_is_off_by_default_and_round_trips_through_json() -> None:
    config = ModelConfig(geometry_features=True)

    assert ModelConfig().geometry_features is False
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_a_config_saved_before_the_features_loads_without_them() -> None:
    saved = ModelConfig.medium().to_dict()
    del saved["geometry_features"]

    assert ModelConfig.from_dict(saved) == ModelConfig.medium()


def test_the_model_starts_where_the_model_without_features_starts() -> None:
    input_ids, attention_mask = _batch()
    plain, featured = _model(geometry=False), _model(geometry=True)
    with torch.no_grad():
        expected = plain(input_ids, attention_mask)
        actual = featured(input_ids, attention_mask)

    assert plain.geometry is None
    assert isinstance(featured.geometry, GeometryFeatures)
    assert not featured.geometry.projection.weight.any()
    torch.testing.assert_close(actual, expected)


def test_the_features_add_one_projection_without_bias() -> None:
    plain, featured = _model(geometry=False), _model(geometry=True)
    extra = sum(p.numel() for p in featured.parameters()) - sum(p.numel() for p in plain.parameters())

    assert extra == GeometryFeatures.width() * ModelConfig().hidden_size


def test_the_projection_learns_and_changes_the_output() -> None:
    input_ids, attention_mask = _batch()
    model = _model(geometry=True).train()
    logits = model(input_ids, attention_mask)
    logits[attention_mask].float().square().mean().backward()
    geometry = model.geometry
    assert geometry is not None
    gradient = geometry.projection.weight.grad
    assert gradient is not None
    assert bool(gradient.any())

    model.eval()
    with torch.no_grad():
        before = model(input_ids, attention_mask)
        geometry.projection.weight.normal_(std=0.02)
        after = model(input_ids, attention_mask)
    assert not torch.allclose(before, after)


def test_the_model_with_features_is_still_causal() -> None:
    input_ids, attention_mask = _batch()
    model = _model(geometry=True)
    geometry = model.geometry
    assert geometry is not None
    changed = input_ids.clone()
    changed[0, 2] = torch.tensor(_row(3, 99, 99))
    with torch.no_grad():
        geometry.projection.weight.normal_(std=0.02)
        before = model(input_ids, attention_mask)
        after = model(changed, attention_mask)

    torch.testing.assert_close(after[0, :2], before[0, :2], rtol=0, atol=1e-5)
    assert not torch.allclose(after[0, 2], before[0, 2])


def test_the_place_layout_gets_the_same_features() -> None:
    place = PlaceTokenizer()
    input_ids, attention_mask = _batch()
    model = _model(geometry=True, tokenizer=place)
    with torch.no_grad():
        logits = model(input_ids[..., : place.row_width], attention_mask)

    assert tuple(logits.shape) == (2, 4, place.row_width, place.vocab_size)
    torch.testing.assert_close(
        geometry_features(input_ids[..., : place.row_width], place.vocabulary, BINS), _features(input_ids)
    )


def test_largest_attention_scores_see_the_features() -> None:
    input_ids, attention_mask = _batch()
    model = _model(geometry=True)
    geometry = model.geometry
    assert geometry is not None
    with torch.no_grad():
        before = model.largest_attention_scores(input_ids, attention_mask)
        geometry.projection.weight.normal_(std=0.5)
        after = model.largest_attention_scores(input_ids, attention_mask)

    assert not torch.allclose(before, after)
