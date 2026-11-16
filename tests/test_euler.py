"""Tests for the Eulerization and the randomized Euler walk of the GraphGPT-based architecture."""

import itertools
import random
from collections import Counter

import pytest
import torch

from eulercad.representation.euler import Arrival, EulerWalk, euler_walk, eulerize
from eulercad.representation.graph import DrawingGraph


def _graph(num_nodes: int, pairs: list[tuple[int, int]]) -> DrawingGraph:
    angles = torch.arange(num_nodes, dtype=torch.float64) * 0.7
    pos = torch.stack([angles.cos(), angles.sin()], dim=1) * (1 + torch.arange(num_nodes, dtype=torch.float64))[:, None]
    doubled = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    edge_index = torch.tensor(doubled, dtype=torch.int64).T.reshape(2, -1)
    return DrawingGraph.from_point_graph(0, pos, edge_index)


def _degrees(graph: DrawingGraph, duplicates: torch.Tensor) -> torch.Tensor:
    return torch.bincount(torch.cat([graph.edges, duplicates], dim=1).flatten(), minlength=graph.num_nodes)


def _rng(seed: int | str) -> random.Random:
    return random.Random(seed)  # noqa: S311  # nosec B311


def _random_graph(seed: int, num_nodes: int, num_edges: int) -> DrawingGraph:
    generator = _rng(seed)
    pairs = generator.sample(list(itertools.combinations(range(num_nodes), 2)), num_edges)
    return _graph(num_nodes, pairs)


def _minimum_duplicates(graph: DrawingGraph) -> int:
    # Brute force: the fewest existing edges whose copies make every degree even.
    odd = torch.bincount(graph.edges.flatten(), minlength=graph.num_nodes) % 2
    columns = graph.edges.T.tolist()
    for size in range(len(columns) + 1):
        for chosen in itertools.combinations(columns, size):
            parity = torch.zeros(graph.num_nodes, dtype=torch.int64)
            for a, b in chosen:
                parity[a] += 1
                parity[b] += 1
            if torch.equal(parity % 2, odd):
                return size
    raise AssertionError("no duplicate set makes the degrees even")


def _check_walk(graph: DrawingGraph, duplicates: torch.Tensor, walk: EulerWalk) -> None:
    nodes, arrivals = walk.nodes.tolist(), walk.arrivals.tolist()
    segments = {tuple(edge) for edge in graph.edges.T.tolist()}
    assert arrivals[0] == Arrival.START
    assert set(nodes) == set(range(graph.num_nodes))
    crossed: set[tuple[int, int]] = set()
    for previous, node, arrival in zip(nodes[:-1], nodes[1:], arrivals[1:], strict=True):
        pair = (min(previous, node), max(previous, node))
        if arrival == Arrival.JUMP:
            assert pair not in segments
            continue
        assert pair in segments
        assert (arrival == Arrival.RETRACE) == (pair in crossed)
        crossed.add(pair)
    assert crossed == segments
    components = arrivals.count(Arrival.JUMP) + 1
    assert len(nodes) <= 1 + graph.num_edges + duplicates.shape[1] + components - 1


def test_an_eulerian_graph_needs_no_duplicates() -> None:
    assert eulerize(_graph(3, [(0, 1), (1, 2), (0, 2)])).shape == (2, 0)


def test_a_path_duplicates_every_edge_between_its_two_ends() -> None:
    assert eulerize(_graph(3, [(1, 2), (0, 1)])).tolist() == [[0, 1], [1, 2]]


@pytest.mark.parametrize("seed", range(12))
def test_duplicates_make_every_degree_even_with_the_fewest_edges(seed: int) -> None:
    graph = _random_graph(seed, num_nodes=7, num_edges=8)
    duplicates = eulerize(graph)

    assert bool((_degrees(graph, duplicates) % 2 == 0).all())
    assert duplicates.shape[1] == _minimum_duplicates(graph)
    assert {tuple(edge) for edge in duplicates.T.tolist()} <= {tuple(edge) for edge in graph.edges.T.tolist()}


def test_duplicates_stay_inside_each_component() -> None:
    graph = _graph(6, [(0, 1), (1, 2), (3, 4), (4, 5)])

    assert eulerize(graph).tolist() == [[0, 1, 3, 4], [1, 2, 4, 5]]


@pytest.mark.parametrize("seed", range(20))
def test_a_walk_crosses_every_segment_and_visits_every_node(seed: int) -> None:
    graph = _random_graph(seed, num_nodes=9, num_edges=9)
    duplicates = eulerize(graph)

    _check_walk(graph, duplicates, euler_walk(graph, _rng(seed), duplicates))


def test_components_are_contiguous_and_each_ends_on_a_new_segment() -> None:
    graph = _graph(9, [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 4), (7, 8)])

    for seed in range(30):
        walk = euler_walk(graph, _rng(seed))
        arrivals = walk.arrivals.tolist()
        blocks: list[list[int]] = []
        for node, arrival in zip(walk.nodes.tolist(), arrivals, strict=True):
            if arrival in (Arrival.START, Arrival.JUMP):
                blocks.append([])
            blocks[-1].append(node)
        assert sorted(sorted(set(block)) for block in blocks) == [[0, 1, 2, 3], [4, 5, 6], [7, 8]]
        ends = [position - 1 for position, arrival in enumerate(arrivals) if arrival == Arrival.JUMP] + [
            len(arrivals) - 1
        ]
        assert all(arrivals[end] == Arrival.ORIGINAL for end in ends)


def test_an_isolated_node_is_one_row() -> None:
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=torch.float64)
    graph = DrawingGraph.from_point_graph(0, pos, torch.tensor([[0, 1], [1, 0]]))

    for seed in range(10):
        walk = euler_walk(graph, _rng(seed))
        assert sorted(walk.nodes.tolist()) == [0, 1, 2]
        assert Counter(walk.arrivals.tolist())[Arrival.ORIGINAL] == 1


def test_the_same_generator_state_gives_the_same_walk_and_supplied_duplicates_change_nothing() -> None:
    graph = _random_graph(3, num_nodes=10, num_edges=12)

    first = euler_walk(graph, _rng(42))
    second = euler_walk(graph, _rng(42), eulerize(graph))

    assert torch.equal(first.nodes, second.nodes) and torch.equal(first.arrivals, second.arrivals)


def test_different_draws_give_different_walks() -> None:
    graph = _graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)])

    walks = {tuple(euler_walk(graph, _rng(seed)).nodes.tolist()) for seed in range(40)}

    assert len(walks) > 10


def test_the_walk_does_not_touch_global_random_state() -> None:
    graph = _random_graph(5, num_nodes=8, num_edges=10)
    random.seed(1)
    expected = random.random()  # noqa: S311  # nosec B311

    random.seed(1)
    euler_walk(graph, _rng(0))

    assert random.random() == expected  # noqa: S311  # nosec B311


def test_duplicates_that_leave_an_odd_degree_are_rejected() -> None:
    graph = _graph(3, [(0, 1), (1, 2)])

    with pytest.raises(ValueError, match="even"):
        euler_walk(graph, _rng(0), torch.tensor([[0], [1]]))


def test_a_walk_record_checks_its_shape() -> None:
    with pytest.raises(ValueError, match="START"):
        EulerWalk(torch.tensor([0, 1]), torch.tensor([Arrival.ORIGINAL, Arrival.ORIGINAL], dtype=torch.int8))
    with pytest.raises(ValueError, match="same length"):
        EulerWalk(torch.tensor([0, 1]), torch.tensor([Arrival.START], dtype=torch.int8))
