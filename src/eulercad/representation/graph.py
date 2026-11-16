"""The checked drawing graph: canonical segments and coordinates in the model frame.

Cleaned from `core/graph_gpt_baseline/representation/graph.py` (edge checks) and
`core/datasets/transforms.py` (`CenterAndScale`). Unlike the baseline graph it stores no
attribute tensors: every token attribute is computed later from the coordinates.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

from dataclasses import dataclass

import torch

MAX_NODES = 512
# Lower bound on the half-extent, as in CenterAndScale, so a one-point drawing does not divide by zero.
MIN_HALF_EXTENT = 1e-8


class GraphCapacityError(ValueError):
    """The drawing has more nodes than there are identity tokens."""


def _check_edges_in_range(edges: torch.Tensor, num_nodes: int) -> None:
    if edges.numel() and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError(f"edge indices are out of range for {num_nodes} nodes")


def _canonical_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Collapse doubled storage into one sorted `source < target` column per segment.

    Returns:
        `(2, E)` int64 edges.

    Raises:
        ValueError: On a self-loop, a repeated column, or a segment stored in one direction only.
    """
    source, target = edge_index
    if bool((source == target).any()):
        raise ValueError("a self-loop is not a segment")
    _, directed_counts = torch.unique(source * num_nodes + target, return_counts=True)
    if bool((directed_counts > 1).any()):
        raise ValueError("a segment direction is repeated")
    low, high = torch.minimum(source, target), torch.maximum(source, target)
    keys, counts = torch.unique(low * num_nodes + high, return_counts=True)
    if bool((counts != 2).any()):
        raise ValueError("every segment must be stored in both directions")
    return torch.stack([keys // num_nodes, keys % num_nodes])


@dataclass(frozen=True, slots=True)
class DrawingGraph:
    """One drawing as a checked undirected graph in the model frame.

    Build it with `from_point_graph`. Direct construction is for already canonical data and
    checks only the rules below.

    Attributes:
        sketch_id: the stored drawing identifier; never an input to the model.
        positions: `(N, 2)` float64, node `i` is stored point `i`; the longer axis spans
            `[-1, 1]`, so every coordinate lies within it.
        edges: `(2, E)` int64, one column per segment, `source < target`, sorted by
            source then target.
        center: the raw bounding-box midpoint, `(x, y)`.
        scale: the raw larger half-extent; `raw = positions * scale + center`.
    """

    sketch_id: int
    positions: torch.Tensor
    edges: torch.Tensor
    center: tuple[float, float]
    scale: float

    def __post_init__(self) -> None:
        if self.positions.dtype != torch.float64 or self.edges.dtype != torch.int64:
            raise TypeError("positions must be float64 and edges int64")
        if self.positions.ndim != 2 or self.positions.shape[1] != 2 or self.positions.shape[0] < 1:
            raise ValueError("positions must have shape (N, 2) with N >= 1")
        if self.edges.ndim != 2 or self.edges.shape[0] != 2:
            raise ValueError("edges must have shape (2, E)")
        if not bool(self.positions.isfinite().all()) or bool((self.positions.abs() > 1).any()):
            raise ValueError("positions must be finite and within [-1, 1]")
        if not (self.scale > 0 and all(abs(value) < float("inf") for value in (*self.center, self.scale))):
            raise ValueError("center must be finite and scale finite and positive")
        _check_edges_in_range(self.edges, self.num_nodes)
        source, target = self.edges
        if bool((source >= target).any()):
            raise ValueError("every edge must have source < target")
        keys = source * self.num_nodes + target
        if bool((keys[1:] <= keys[:-1]).any()):
            raise ValueError("edges must be unique and sorted by source, then target")

    @classmethod
    def from_point_graph(
        cls, sketch_id: int, pos: torch.Tensor, edge_index: torch.Tensor, *, max_nodes: int = MAX_NODES
    ) -> "DrawingGraph":
        """Check one stored drawing, canonicalize its segments, and move it to the model frame.

        Args:
            sketch_id: the stored drawing identifier.
            pos: `(N, 2)` float64 raw coordinates, as `PointGraphDataset` stores them.
            edge_index: `(2, 2E)` int64, every segment once in each direction.
            max_nodes: the node capacity, the number of identity tokens.

        Returns:
            The checked graph. The inputs are not modified and not shared.

        Raises:
            TypeError: If `pos` is not float64 or `edge_index` not int64.
            ValueError: On a wrong shape, a non-finite coordinate, an out-of-range index, a
                self-loop, a repeated direction, or a segment stored in one direction only.
            GraphCapacityError: If the drawing has more than `max_nodes` nodes.
        """
        if pos.dtype != torch.float64 or edge_index.dtype != torch.int64:
            raise TypeError("pos must be float64 and edge_index int64")
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] < 1:
            raise ValueError("pos must have shape (N, 2) with N >= 1")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, 2E)")
        num_nodes = pos.shape[0]
        if num_nodes > max_nodes:
            raise GraphCapacityError(f"drawing {sketch_id} has {num_nodes} nodes, above the capacity of {max_nodes}")
        if not bool(pos.isfinite().all()):
            raise ValueError("coordinates must be finite")
        _check_edges_in_range(edge_index, num_nodes)
        edges = _canonical_edges(edge_index, num_nodes)

        low, high = pos.min(dim=0).values, pos.max(dim=0).values
        center = (low + high) / 2
        half = float(((high - low) / 2).max().clamp_min(MIN_HALF_EXTENT))
        # Rounding can put an extreme point a hair past 1 (1.0000000000000002 on 4,206 of 52,820
        # validation drawings); the clamp removes only that.
        positions = ((pos - center) / half).clamp(-1.0, 1.0)
        return cls(int(sketch_id), positions, edges, (float(center[0]), float(center[1])), half)

    @property
    def num_nodes(self) -> int:
        return self.positions.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edges.shape[1]
