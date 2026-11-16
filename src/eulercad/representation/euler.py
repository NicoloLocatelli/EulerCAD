"""Eulerization and the randomized Euler walk over a `DrawingGraph`.

The policy follows the GraphGPT baseline (`core/graph_gpt_baseline/representation/euler.py`):
components in random order, each Eulerized by duplicating existing edges, walked from a
random start, and cut once every segment has been crossed; components are joined by jumps.

The work is split in two. `eulerize` is deterministic and does the expensive part, a
minimum-weight matching of the odd-degree nodes, with networkx; its result can be cached per
drawing. `euler_walk` is random and cheap: Hierholzer's algorithm on plain lists, so a new
walk can be drawn every epoch.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import itertools
import random
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

import networkx as nx
import torch

from eulercad.representation.graph import DrawingGraph


class Arrival(IntEnum):
    """How the walk reached a node."""

    START = 0  # the first node; nothing precedes it
    ORIGINAL = 1  # along a segment crossed for the first time
    RETRACE = 2  # along a segment already crossed
    JUMP = 3  # to the first node of the next component, along no segment


@dataclass(frozen=True, slots=True)
class EulerWalk:
    """A walk aligned with the token rows: row `t` is `nodes[t]`, reached as `arrivals[t]`.

    Attributes:
        nodes: `(L,)` int64 node indices of the `DrawingGraph`.
        arrivals: `(L,)` int8 `Arrival` values; only the first is `START`.
    """

    nodes: torch.Tensor
    arrivals: torch.Tensor

    def __post_init__(self) -> None:
        if self.nodes.dtype != torch.int64 or self.arrivals.dtype != torch.int8:
            raise TypeError("nodes must be int64 and arrivals int8")
        if self.nodes.ndim != 1 or self.nodes.shape != self.arrivals.shape or self.nodes.numel() < 1:
            raise ValueError("nodes and arrivals must be one-dimensional, non-empty, and the same length")
        if int(self.arrivals[0]) != Arrival.START or bool((self.arrivals[1:] == Arrival.START).any()):
            raise ValueError("only the first arrival may be START, and it must be")


def eulerize(graph: DrawingGraph) -> torch.Tensor:
    """Return a minimum-cardinality T-join for the graph's odd-degree nodes.

    This is the Eulerization step of the undirected Chinese Postman
    (Route Inspection) algorithm. In each connected component, construct
    the shortest-path metric closure of the odd-degree nodes, pair them
    with a minimum-weight perfect matching, and duplicate the matched
    shortest paths.

    Original edges have unit cost, so the result minimizes the number of
    duplicated edges. Duplicate pairs cancel because only edge parity
    affects the T-join.

    Returns:
        `(2, D)` int64 duplicate edges, `source < target`, sorted.
    """
    degree = torch.bincount(graph.edges.flatten(), minlength=graph.num_nodes)
    odd = set((degree % 2).nonzero().flatten().tolist())
    if not odd:
        return torch.empty((2, 0), dtype=torch.int64)
    network = nx.Graph()
    network.add_nodes_from(range(graph.num_nodes))
    network.add_edges_from(graph.edges.T.tolist())
    copies: Counter[tuple[int, int]] = Counter()
    for component in nx.connected_components(network):
        ends = sorted(odd & component)
        if not ends:
            continue
        pairs = nx.Graph()
        for position, source in enumerate(ends):
            hops = nx.single_source_shortest_path_length(network, source)
            pairs.add_weighted_edges_from((source, target, hops[target]) for target in ends[position + 1 :])
        for source, target in nx.min_weight_matching(pairs):
            for a, b in itertools.pairwise(nx.shortest_path(network, source, target)):
                copies[min(a, b), max(a, b)] += 1
    kept = sorted(edge for edge, count in copies.items() if count % 2)
    return torch.tensor(kept, dtype=torch.int64).T.reshape(2, -1)


def _components(adjacency: list[list[tuple[int, int]]]) -> list[list[int]]:
    seen = [False] * len(adjacency)
    components = []
    for root in range(len(adjacency)):
        if seen[root]:
            continue
        seen[root] = True
        members, frontier = [root], [root]
        while frontier:
            for neighbour, _ in adjacency[frontier.pop()]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    members.append(neighbour)
                    frontier.append(neighbour)
        components.append(sorted(members))
    return components


def _circuit(adjacency: list[list[tuple[int, int]]], start: int, used: bytearray) -> list[tuple[int, int]]:
    """Hierholzer's algorithm, taking each node's unused edges from the end of its list.

    Returns:
        The closed walk as `(node, edge id it was reached by)`, the start first with id -1.
    """
    stack, reversed_walk = [(start, -1)], []
    while stack:
        node, _ = stack[-1]
        edges = adjacency[node]
        while edges and used[edges[-1][1]]:
            edges.pop()
        if edges:
            neighbour, edge = edges.pop()
            used[edge] = 1
            stack.append((neighbour, edge))
        else:
            reversed_walk.append(stack.pop())
    # Popped order is the walk backwards; the edge joining two neighbouring entries is the one
    # stored with the earlier-popped entry, which is exactly the edge that entry was reached by.
    reversed_walk.reverse()
    return reversed_walk


def euler_walk(graph: DrawingGraph, rng: random.Random, duplicates: torch.Tensor | None = None) -> EulerWalk:
    """Draw a randomized Hierholzer walk and keep its shortest edge-covering prefix.

    `duplicates` supplies the minimum-cardinality T-join from `eulerize`, making every degree
    even. Hierholzer's algorithm then constructs an Euler circuit in each connected component.
    The circuit is cut immediately after its last first crossing because the remaining
    retraces add no source segment. Components are joined by `JUMP` arrivals.

    Random draws, all from `rng`, in this order: the component order; then, per component, the
    start node and the order in which each node tries its edges. For fixed duplicates, the
    chance of each walk does not depend on how the nodes are numbered; one walk from one seed
    does. `eulerize` can pick other duplicates under another numbering when shortest paths tie.

    Args:
        graph: the checked drawing.
        rng: the only source of randomness.
        duplicates: `eulerize(graph)`, when already computed; supplying it changes nothing.

    Returns:
        The walk, one entry per token row.

    Raises:
        ValueError: If a duplicate is not an edge of the graph or leaves a degree odd.
    """
    if duplicates is None:
        duplicates = eulerize(graph)
    original = [tuple(edge) for edge in graph.edges.T.tolist()]
    index = {edge: position for position, edge in enumerate(original)}
    try:
        segment = list(range(len(original))) + [index[tuple(edge)] for edge in duplicates.T.tolist()]
    except KeyError as error:
        raise ValueError(f"duplicate {error.args[0]} is not an edge of the graph") from error
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(graph.num_nodes)]
    for edge, (a, b) in enumerate(original + [tuple(edge) for edge in duplicates.T.tolist()]):
        adjacency[a].append((b, edge))
        adjacency[b].append((a, edge))
    if any(len(edges) % 2 for edges in adjacency):
        raise ValueError("the duplicates must make every degree even")

    components = _components(adjacency)
    rng.shuffle(components)
    used = bytearray(len(segment))
    nodes: list[int] = []
    arrivals: list[int] = []
    for position, component in enumerate(components):
        start = rng.choice(component)
        for node in component:
            rng.shuffle(adjacency[node])
        circuit = _circuit(adjacency, start, used)
        remaining = len({segment[edge] for _, edge in circuit[1:]})
        nodes.append(start)
        arrivals.append(Arrival.JUMP if position else Arrival.START)
        crossed: set[int] = set()
        for node, edge in circuit[1:]:
            if len(crossed) == remaining:
                break
            arrivals.append(Arrival.RETRACE if segment[edge] in crossed else Arrival.ORIGINAL)
            crossed.add(segment[edge])
            nodes.append(node)
    return EulerWalk(torch.tensor(nodes, dtype=torch.int64), torch.tensor(arrivals, dtype=torch.int8))
