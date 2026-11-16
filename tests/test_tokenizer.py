"""Tests for the tokenizer of the GraphGPT-based architecture: settings, rows, and decoding."""

import json
import math
import random

import pytest
import torch

from eulercad.representation.euler import Arrival, EulerWalk, euler_walk
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.rows import DecodingError, DecodingFailure
from eulercad.tokenization.tokenizer import (
    GraphTokenizer,
    PlaceTokenizer,
    TokenizerConfig,
    tokenizer_from_dict,
)
from eulercad.tokenization.vocabulary import D_NONE, EDGE, EOS, JUMP, PAD

DEFAULT = GraphTokenizer()
SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
PLACE = PlaceTokenizer()
TOKENIZERS = pytest.mark.parametrize("tokenizer", [DEFAULT, SMALL, PLACE], ids=["default", "small", "place"])
# Only a full row has a distance field, so rules about it are tested on these alone.
FULL_TOKENIZERS = pytest.mark.parametrize("tokenizer", [DEFAULT, SMALL], ids=["default", "small"])


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  # nosec B311


class _FixedOffset(random.Random):  # noqa: S311
    """Draws a chosen identity offset."""

    def __init__(self, offset: int) -> None:
        super().__init__(0)
        self.offset = offset

    def randrange(self, *_: object, **__: object) -> int:
        return self.offset


def _graph(positions: list[list[float]], pairs: list[tuple[int, int]]) -> DrawingGraph:
    edges = torch.tensor(sorted(pairs), dtype=torch.int64).reshape(-1, 2).T
    return DrawingGraph(0, torch.tensor(positions, dtype=torch.float64), edges, (0.0, 0.0), 1.0)


def _walk(nodes: list[int], arrivals: list[Arrival]) -> EulerWalk:
    return EulerWalk(torch.tensor(nodes), torch.tensor(arrivals, dtype=torch.int8))


def _example_rows(tokenizer: GraphTokenizer, offset: int) -> torch.Tensor:
    # tokenization/CONTEXT.md, section 4: a -> b -> c, jump, d -> e.
    graph = _graph([[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [0.5, 0.0], [1.0, 0.0]], [(0, 1), (1, 2), (3, 4)])
    arrivals = [Arrival.START, Arrival.ORIGINAL, Arrival.ORIGINAL, Arrival.JUMP, Arrival.ORIGINAL]
    return tokenizer.tokenize(graph, _walk([0, 1, 2, 3, 4], arrivals), _FixedOffset(offset)).rows


def _names(tokenizer: GraphTokenizer, rows: torch.Tensor) -> list[list[str]]:
    return [[tokenizer.vocabulary.token_name(token) for token in row] for row in rows.tolist()]


def _random_graph(seed: int) -> DrawingGraph:
    generator = _rng(seed)
    num_nodes = generator.randint(1, 40)
    spread = generator.choice([1.0, 0.05, 0.002])
    positions = [[generator.uniform(-spread, spread), generator.uniform(-spread, spread)] for _ in range(num_nodes)]
    pairs: set[tuple[int, int]] = set()
    for _ in range(num_nodes if num_nodes > 1 else 0):
        a, b = generator.sample(range(num_nodes), 2)
        pairs.add((min(a, b), max(a, b)))
    return _graph(positions, sorted(pairs))


def test_the_default_settings_are_the_designed_ones() -> None:
    assert TokenizerConfig() == TokenizerConfig(512, 256, 256, 0.001, 2 * math.sqrt(2))
    assert (DEFAULT.vocab_size, DEFAULT.row_width) == (1287, 5)


def test_the_components_follow_the_settings() -> None:
    assert SMALL.coordinates.bins == 64
    assert (SMALL.distances.bins, SMALL.distances.d_min, SMALL.distances.d_max) == (32, 0.01, 2.0)
    assert (SMALL.vocabulary.max_nodes, SMALL.vocabulary.coordinate_bins, SMALL.vocabulary.distance_bins) == (
        64,
        64,
        32,
    )
    assert SMALL.vocab_size == 7 + 64 + 64 + 64 + 32


def test_invalid_settings_fail_when_the_tokenizer_is_built() -> None:
    with pytest.raises(ValueError, match="must"):
        GraphTokenizer(TokenizerConfig(coordinate_bins=0))
    with pytest.raises(ValueError, match="must"):
        GraphTokenizer(TokenizerConfig(d_min=1.0, d_max=0.5))


def test_the_settings_survive_a_json_round_trip() -> None:
    restored = GraphTokenizer.from_dict(json.loads(json.dumps(SMALL.to_dict())))

    assert restored == SMALL
    assert restored != DEFAULT
    assert SMALL.to_dict()["tokenizer"] == "GraphTokenizer"


def test_settings_of_another_tokenizer_are_rejected() -> None:
    data = DEFAULT.to_dict() | {"tokenizer": "OtherTokenizer"}

    with pytest.raises(ValueError, match="OtherTokenizer"):
        GraphTokenizer.from_dict(data)


def test_the_tokenizer_cannot_be_changed_after_it_is_built() -> None:
    with pytest.raises(AttributeError):
        DEFAULT.config = TokenizerConfig(coordinate_bins=8)  # ty: ignore[invalid-assignment]


def test_the_default_worked_example_gives_the_rows_in_context() -> None:
    assert _names(DEFAULT, _example_rows(DEFAULT, 300)) == [
        ["ID_300", "X_000", "Y_064", "EDGE", "D_NONE"],
        ["ID_301", "X_128", "Y_064", "EDGE", "D_222"],
        ["ID_302", "X_128", "Y_192", "EDGE", "D_222"],
        ["ID_303", "X_192", "Y_128", "JUMP", "D_NONE"],
        ["ID_304", "X_255", "Y_128", "EDGE", "D_200"],
    ]


def test_the_worked_example_follows_other_settings() -> None:
    # 64 coordinate bins: -1, -0.5, 0, 0.5, 1 fall in bins 0, 16, 32, 48, 63. Identities wrap at 64.
    # 32 distance bins from 0.01 to 2: 1.0 -> 1 + floor(31 ln(100) / ln(200)) = 27; 0.5 -> 23.
    assert _names(SMALL, _example_rows(SMALL, 62)) == [
        ["ID_062", "X_000", "Y_016", "EDGE", "D_NONE"],
        ["ID_063", "X_032", "Y_016", "EDGE", "D_027"],
        ["ID_000", "X_032", "Y_048", "EDGE", "D_027"],
        ["ID_001", "X_048", "Y_032", "JUMP", "D_NONE"],
        ["ID_002", "X_063", "Y_032", "EDGE", "D_023"],
    ]


def test_a_revisited_node_repeats_its_identity_and_bins() -> None:
    graph = _graph([[0.0, 0.0], [0.3, 0.0], [0.3, 0.4]], [(0, 1), (1, 2)])
    walk = _walk([1, 0, 1, 2], [Arrival.START, Arrival.ORIGINAL, Arrival.RETRACE, Arrival.ORIGINAL])

    tokenized = DEFAULT.tokenize(graph, walk, _rng(0))

    assert torch.equal(tokenized.rows[0, :3], tokenized.rows[2, :3])
    assert torch.equal(tokenized.nodes, walk.nodes)
    assert tokenized.rows[:, 3].tolist() == [EDGE] * 4


def test_the_distance_is_measured_between_frame_coordinates_not_bin_centers() -> None:
    # Both points sit in the same X and Y bins, yet the segment keeps its length.
    graph = _graph([[0.0, 0.0], [0.005, 0.0]], [(0, 1)])

    rows = DEFAULT.tokenize(graph, _walk([0, 1], [Arrival.START, Arrival.ORIGINAL]), _rng(0)).rows

    low, high = DEFAULT.distances.bounds(DEFAULT.vocabulary.distance.index(rows[1, 4:]))
    assert torch.equal(rows[0, 1:3], rows[1, 1:3])
    assert float(low) <= 0.005 < float(high)


def test_a_drawing_above_the_node_capacity_is_rejected() -> None:
    tokenizer = GraphTokenizer(TokenizerConfig(max_nodes=4))
    graph = _graph([[0.1 * node, 0.0] for node in range(5)], [(0, 1), (1, 2), (2, 3), (3, 4)])

    with pytest.raises(ValueError, match="do not fit in 4"):
        tokenizer.tokenize(graph, euler_walk(graph, _rng(0)), _rng(0))


@TOKENIZERS
@pytest.mark.parametrize("seed", range(20))
def test_random_drawings_decode_to_their_bins_and_edges(tokenizer: GraphTokenizer, seed: int) -> None:
    graph = _random_graph(seed)
    tokenized = tokenizer.tokenize(graph, euler_walk(graph, _rng(seed)), _rng(seed))

    decoded = tokenizer.decode(tokenized.rows)

    first_visits = list(dict.fromkeys(tokenized.nodes.tolist()))
    renumber = {node: new for new, node in enumerate(first_visits)}
    expected = sorted((min(renumber[a], renumber[b]), max(renumber[a], renumber[b])) for a, b in graph.edges.T.tolist())
    coordinates = tokenizer.coordinates
    assert tokenized.rows.shape == (tokenized.nodes.numel(), tokenizer.row_width)
    assert torch.equal(decoded.positions, coordinates.centers(coordinates.to_bins(graph.positions[first_visits])))
    assert [tuple(edge) for edge in decoded.edges.T.tolist()] == expected


def test_a_trailing_end_row_is_ignored() -> None:
    rows = _example_rows(DEFAULT, 0)
    with_end = torch.cat([rows, torch.full((1, 5), EOS)])

    assert torch.equal(DEFAULT.decode(with_end).edges, DEFAULT.decode(rows).edges)


def _worst_rounding_gap(tokenizer: GraphTokenizer) -> float:
    # A segment from the lower corner of one cell to the upper corner of another, whose length
    # sits just above a distance bin edge: the bin centers are as short as they can be.
    bins = tokenizer.coordinates.bins
    tolerance = tokenizer.coordinates.max_distance_error
    lows = tokenizer.distances.bounds(torch.arange(tokenizer.distances.bins))[0].tolist()
    widest = 0.0
    for cells in range(1, bins // 2):
        centers = math.sqrt(2) * (cells - 1) * 2 / bins
        edges = [low for low in lows if centers < low < centers + tolerance]
        if not edges:
            continue
        side = max(edges) * (1 + 1e-12) / math.sqrt(2)
        graph = _graph([[0.0, 0.0], [side, side]], [(0, 1)])
        tokenizer.decode(tokenizer.tokenize(graph, _walk([0, 1], [Arrival.START, Arrival.ORIGINAL]), _rng(0)).rows)
        widest = max(widest, max(edges) - centers)
    return widest


@TOKENIZERS
def test_segments_at_the_worst_rounding_still_decode(tokenizer: GraphTokenizer) -> None:
    assert _worst_rounding_gap(tokenizer) > 0


def test_the_default_worst_rounding_needs_more_than_0_011() -> None:
    assert _worst_rounding_gap(DEFAULT) > 0.011


def _failure(tokenizer: GraphTokenizer, rows: torch.Tensor) -> DecodingFailure:
    with pytest.raises(DecodingError) as caught:
        tokenizer.decode(rows)
    return caught.value.reason


def test_no_rows_is_an_empty_drawing() -> None:
    assert _failure(DEFAULT, torch.empty((0, 5), dtype=torch.int64)) == DecodingFailure.EMPTY
    assert _failure(DEFAULT, torch.full((1, 5), EOS)) == DecodingFailure.EMPTY


@pytest.mark.parametrize(
    ("row", "field", "token"),
    [
        (0, 0, DEFAULT.vocabulary.x.start),  # a coordinate in the identity field
        (1, 1, DEFAULT.vocabulary.y.start),  # Y in the X field
        (2, 3, D_NONE),  # not an edge type
        (0, 3, JUMP),  # the first row cannot jump
        (0, 4, DEFAULT.vocabulary.distance.start + 5),  # the first row has no distance
        (1, 4, D_NONE),  # an edge row needs a distance
        (3, 4, DEFAULT.vocabulary.distance.start + 5),  # a jump row has no distance
        (2, 0, PAD),
        (2, 0, EOS),
    ],
)
def test_a_token_from_the_wrong_family_is_rejected(row: int, field: int, token: int) -> None:
    rows = _example_rows(DEFAULT, 0)
    rows[row, field] = token

    assert _failure(DEFAULT, rows) == DecodingFailure.WRONG_FAMILY


def test_a_token_of_a_larger_vocabulary_is_rejected() -> None:
    rows = _example_rows(DEFAULT, 0)

    assert _failure(SMALL, rows) == DecodingFailure.WRONG_FAMILY


def test_a_known_identity_with_other_bins_is_rejected() -> None:
    rows = _example_rows(DEFAULT, 0)
    rows[2, 0] = rows[0, 0]

    assert _failure(DEFAULT, rows) == DecodingFailure.IDENTITY_CONFLICT


def test_an_edge_back_to_the_same_node_is_rejected() -> None:
    rows = _example_rows(DEFAULT, 0)
    rows[2] = rows[1]

    assert _failure(DEFAULT, rows) == DecodingFailure.SELF_LOOP


def _with_matching_distances(tokenizer: GraphTokenizer, rows: torch.Tensor) -> torch.Tensor:
    # Rewrite every distance token from the bin centers, so only the rule under test is broken.
    coordinates, vocabulary = tokenizer.coordinates, tokenizer.vocabulary
    centers = torch.stack(
        [coordinates.centers(vocabulary.x.index(rows[:, 1])), coordinates.centers(vocabulary.y.index(rows[:, 2]))], 1
    )
    lengths = (centers[1:] - centers[:-1]).norm(dim=1)
    crossed = rows[1:, 3] == EDGE
    rows[1:, 4][crossed] = vocabulary.distance.ids(tokenizer.distances.to_bins(lengths[crossed]))
    return rows


def test_a_jump_back_to_a_known_identity_is_rejected() -> None:
    # The mutation in the review: the jump returns to a, and e is joined to a.
    rows = _example_rows(DEFAULT, 0)
    rows[3, :3] = rows[0, :3]
    rows[4, 0] = rows[2, 0] + 1
    rows = _with_matching_distances(DEFAULT, rows)

    assert _failure(DEFAULT, rows) == DecodingFailure.COMPONENT_CONFLICT


def test_an_edge_into_an_earlier_component_is_rejected() -> None:
    # After the jump to d, the walk steps to c, a node of the first component.
    rows = _example_rows(DEFAULT, 0)
    rows[4, :3] = rows[2, :3]
    rows = _with_matching_distances(DEFAULT, rows)

    assert _failure(DEFAULT, rows) == DecodingFailure.COMPONENT_CONFLICT


@TOKENIZERS
def test_a_new_identity_must_follow_the_previous_one(tokenizer: GraphTokenizer) -> None:
    # Offset 62 wraps from the last identity to the first on the small tokenizer.
    rows = _example_rows(tokenizer, 62)
    assert tokenizer.decode(rows).positions.shape[0] == 5

    for row in (1, 3, 4):
        skipped = rows.clone()
        skipped[row:, 0] = tokenizer.vocabulary.node.ids(
            (tokenizer.vocabulary.node.index(skipped[row:, 0]) + 1) % tokenizer.config.max_nodes
        )
        assert _failure(tokenizer, skipped) == DecodingFailure.IDENTITY_ORDER


@FULL_TOKENIZERS
def test_a_distance_that_cannot_match_the_bins_is_rejected(tokenizer: GraphTokenizer) -> None:
    rows = _example_rows(tokenizer, 0)
    rows[1, 4] = tokenizer.vocabulary.distance.start

    assert _failure(tokenizer, rows) == DecodingFailure.DISTANCE_MISMATCH


def test_rows_of_the_wrong_shape_are_a_programming_error() -> None:
    with pytest.raises(ValueError, match=r"\(L, 5\)") as caught:
        DEFAULT.decode(torch.zeros((2, 4), dtype=torch.int64))
    assert not isinstance(caught.value, DecodingError)


# The place layout: four-field rows, without the distance field.


def test_the_place_tokenizer_writes_four_field_rows() -> None:
    assert PLACE.row_width == 4
    assert PLACE.fields == ("node_id", "x_bin", "y_bin", "edge_type")
    assert DEFAULT.row_width == 5
    assert DEFAULT.fields == ("node_id", "x_bin", "y_bin", "edge_type", "distance_bin")


@pytest.mark.parametrize("seed", range(5))
def test_a_place_row_is_the_full_row_without_its_distance(seed: int) -> None:
    graph = _random_graph(seed)

    full = DEFAULT.tokenize(graph, euler_walk(graph, _rng(seed)), _rng(seed)).rows
    place = PLACE.tokenize(graph, euler_walk(graph, _rng(seed)), _rng(seed)).rows

    assert torch.equal(place, full[:, :4])


def test_the_place_tokenizer_rejects_full_rows() -> None:
    graph = _random_graph(0)
    rows = DEFAULT.tokenize(graph, euler_walk(graph, _rng(0)), _rng(0)).rows

    with pytest.raises(ValueError, match=r"\(L, 4\)"):
        PLACE.decode(rows)


def test_the_full_tokenizer_rejects_place_rows() -> None:
    graph = _random_graph(0)
    rows = PLACE.tokenize(graph, euler_walk(graph, _rng(0)), _rng(0)).rows

    with pytest.raises(ValueError, match=r"\(L, 5\)"):
        DEFAULT.decode(rows)


def test_a_place_row_cannot_carry_a_distance_that_disagrees_with_its_bins() -> None:
    # The full layout can: it is the failure the place layout removes by construction.
    graph = _graph([[0.0, 0.0], [0.5, 0.0]], [(0, 1)])
    walk = _walk([0, 1], [Arrival.START, Arrival.ORIGINAL])
    rows = DEFAULT.tokenize(graph, walk, _FixedOffset(0)).rows
    rows[1, 4] = DEFAULT.vocabulary.distance.ids(torch.tensor(0))

    with pytest.raises(DecodingError) as caught:
        DEFAULT.decode(rows)

    assert caught.value.reason is DecodingFailure.DISTANCE_MISMATCH
    assert PLACE.decode(rows[:, :4]).positions.shape == (2, 2)


def test_the_place_tokenizer_is_read_back_by_its_class_name() -> None:
    saved = json.loads(json.dumps(PLACE.to_dict()))

    assert saved["tokenizer"] == "PlaceTokenizer"
    assert tokenizer_from_dict(saved) == PLACE
    assert tokenizer_from_dict(json.loads(json.dumps(DEFAULT.to_dict()))) == DEFAULT


def test_the_two_layouts_are_not_each_other() -> None:
    assert PLACE != DEFAULT
    with pytest.raises(ValueError, match="PlaceTokenizer"):
        GraphTokenizer.from_dict(PLACE.to_dict())
    with pytest.raises(ValueError, match="GraphTokenizer"):
        PlaceTokenizer.from_dict(DEFAULT.to_dict())


def test_an_unknown_tokenizer_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a known tokenizer"):
        tokenizer_from_dict({"tokenizer": "Nonsense", "config": {}})
