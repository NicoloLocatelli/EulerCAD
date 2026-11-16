"""Tests for the checked drawing graph of the GraphGPT-based architecture."""

import pytest
import torch

from eulercad.representation.graph import DrawingGraph, GraphCapacityError

# An L of two segments plus one separate segment; 4 units wide, 2 units tall.
POS = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 1.0]], dtype=torch.float64)
DOUBLED = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])


def _graph(pos: torch.Tensor = POS, edge_index: torch.Tensor = DOUBLED, **kwargs) -> DrawingGraph:
    return DrawingGraph.from_point_graph(7, pos, edge_index, **kwargs)


def test_doubled_edges_become_one_sorted_column_per_segment() -> None:
    graph = _graph()

    assert graph.sketch_id == 7
    assert graph.edges.dtype == torch.int64
    assert graph.edges.tolist() == [[0, 1, 3], [1, 2, 4]]
    assert (graph.num_nodes, graph.num_edges) == (5, 3)


def test_edge_storage_order_and_direction_do_not_change_the_graph() -> None:
    shuffled = DOUBLED[:, torch.tensor([5, 2, 0, 4, 3, 1])]

    assert torch.equal(_graph(edge_index=shuffled).edges, _graph().edges)


def test_nodes_keep_their_stored_order() -> None:
    graph = _graph()

    assert torch.equal(graph.positions * graph.scale + torch.tensor(graph.center, dtype=torch.float64), POS)


def test_the_longer_axis_spans_minus_one_to_one_and_the_aspect_ratio_is_kept() -> None:
    graph = _graph()

    assert graph.positions.dtype == torch.float64
    assert graph.center == pytest.approx((2.0, 1.0))
    assert graph.scale == pytest.approx(2.0)
    assert torch.equal(graph.positions.amin(dim=0), torch.tensor([-1.0, -0.5], dtype=torch.float64))
    assert torch.equal(graph.positions.amax(dim=0), torch.tensor([1.0, 0.5], dtype=torch.float64))


def test_every_coordinate_stays_within_minus_one_and_one_despite_rounding() -> None:
    # The x extremes of validation drawing 446: centering and dividing give 1.0000000000000002.
    pos = torch.tensor([[-7.917503260446901, 0.0], [27.263250044850935, 0.0]], dtype=torch.float64)
    graph = _graph(pos=pos, edge_index=torch.tensor([[0, 1], [1, 0]]))

    assert graph.positions.abs().max() <= 1.0


def test_input_tensors_are_not_shared_with_the_graph() -> None:
    pos, edge_index = POS.clone(), DOUBLED.clone()
    graph = _graph(pos=pos, edge_index=edge_index)
    pos.zero_()
    edge_index.zero_()

    assert torch.equal(graph.positions, _graph().positions)
    assert torch.equal(graph.edges, _graph().edges)


def test_a_drawing_above_the_node_capacity_raises_its_own_error() -> None:
    pos = torch.rand((6, 2), dtype=torch.float64)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])

    assert _graph(pos=pos, edge_index=edge_index, max_nodes=6).num_nodes == 6
    with pytest.raises(GraphCapacityError):
        _graph(pos=pos, edge_index=edge_index, max_nodes=5)


def test_the_default_capacity_is_512_nodes() -> None:
    pos = torch.rand((513, 2), dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 0]])

    with pytest.raises(GraphCapacityError):
        _graph(pos=pos, edge_index=edge_index)
    assert _graph(pos=pos[:512], edge_index=edge_index).num_nodes == 512


@pytest.mark.parametrize(
    ("edge_index", "message"),
    [
        (torch.tensor([[0, 1, 1], [1, 0, 1]]), "self-loop"),
        (torch.tensor([[0, 1, 1], [1, 0, 2]]), "both directions"),
        (torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]), "repeated"),
        (torch.tensor([[0, 5], [5, 0]]), "out of range"),
        (torch.tensor([[0, -1], [-1, 0]]), "out of range"),
    ],
)
def test_malformed_edges_are_rejected(edge_index: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _graph(edge_index=edge_index)


def test_non_finite_coordinates_are_rejected() -> None:
    pos = POS.clone()
    pos[2, 1] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        _graph(pos=pos)


@pytest.mark.parametrize(
    ("pos", "edge_index", "error"),
    [
        (POS.float(), DOUBLED, TypeError),
        (POS, DOUBLED.int(), TypeError),
        (POS[:, :1], DOUBLED, ValueError),
        (POS, DOUBLED[0], ValueError),
        (POS[:0], torch.empty((2, 0), dtype=torch.int64), ValueError),
    ],
)
def test_wrong_shapes_and_types_are_rejected(
    pos: torch.Tensor, edge_index: torch.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _graph(pos=pos, edge_index=edge_index)


def test_a_drawing_with_one_point_keeps_it_at_the_origin() -> None:
    graph = _graph(pos=torch.tensor([[5.0, -3.0]], dtype=torch.float64), edge_index=torch.empty((2, 0)).long())

    assert graph.positions.tolist() == [[0.0, 0.0]]
    assert graph.edges.shape == (2, 0)


def test_direct_construction_checks_the_canonical_rules() -> None:
    graph = _graph()

    with pytest.raises(ValueError, match="source < target"):
        DrawingGraph(7, graph.positions, graph.edges.flip(0), graph.center, graph.scale)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        DrawingGraph(7, graph.positions * 2, graph.edges, graph.center, graph.scale)
    with pytest.raises(ValueError, match="sorted"):
        DrawingGraph(7, graph.positions, graph.edges[:, torch.tensor([2, 0, 1])], graph.center, graph.scale)
