"""Token rows: one drawing and its walk to `(L, F)` identifiers, and back.

Cleaned from `core/graph_gpt_baseline/representation/tokenizer.py`, short layout only. A full
row is `[node_id, x_bin, y_bin, edge_type, distance_bin]`: the node, then how the walk reached
it. A place row is the same without its last field, because the distance is already implied by
the two endpoints' bins; `ROW_FIELDS` holds the order and `PLACE_FIELDS` the shorter layout.

Which layout a run uses is the identity of its tokenizer class, not a setting: `GraphTokenizer`
writes full rows and `PlaceTokenizer` place rows. These functions take the layout as an
argument so the two classes share one decoder. The bins and the vocabulary come in as
arguments too, and the decoder reads the rows alone, so it also works on generated sequences.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

from dataclasses import dataclass
from enum import StrEnum

import torch

from eulercad.representation.euler import Arrival, EulerWalk
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.identity import NodeIdentities
from eulercad.tokenization.quantization import CoordinateQuantizer, DistanceQuantizer
from eulercad.tokenization.vocabulary import D_NONE, EDGE, EOS, JUMP, Vocabulary

ROW_FIELDS = ("node_id", "x_bin", "y_bin", "edge_type", "distance_bin")
ROW_WIDTH = len(ROW_FIELDS)
# The place layout drops the last field, so the shorter row is a prefix of the longer one.
PLACE_FIELDS = ROW_FIELDS[:-1]
PLACE_WIDTH = len(PLACE_FIELDS)
DISTANCE_FIELD = ROW_WIDTH - 1
# Added to the distance tolerance for float rounding in the logarithm and the bin bounds.
FLOAT_SLACK = 1e-9


def row_fields(width: int) -> tuple[str, ...]:
    """Name the fields of a row `width` tokens wide.

    The place layout is a prefix of the full one, so the names are `ROW_FIELDS` cut to `width`.

    Returns:
        The field names, in row order.

    Raises:
        ValueError: If no layout has that width.
    """
    if width not in (PLACE_WIDTH, ROW_WIDTH):
        raise ValueError(f"no row layout is {width} fields wide; expected {PLACE_WIDTH} or {ROW_WIDTH}")
    return ROW_FIELDS[:width]


@dataclass(frozen=True, slots=True)
class DecodedDrawing:
    """A drawing read back from rows; node `i` is the `i`-th identity to appear.

    Attributes:
        positions: `(N, 2)` float64 bin centers in the model frame.
        edges: `(2, E)` int64, one column per segment, `source < target`, sorted.
    """

    positions: torch.Tensor
    edges: torch.Tensor


class DecodingFailure(StrEnum):
    """Why rows do not describe a drawing."""

    EMPTY = "empty"
    WRONG_FAMILY = "wrong family"
    IDENTITY_CONFLICT = "identity conflict"
    IDENTITY_ORDER = "identity order"
    COMPONENT_CONFLICT = "component conflict"
    SELF_LOOP = "self-loop"
    DISTANCE_MISMATCH = "distance mismatch"


class DecodingError(ValueError):
    """Rows that do not describe a drawing, with the reason."""

    def __init__(self, reason: DecodingFailure, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def encode_rows(
    graph: DrawingGraph,
    walk: EulerWalk,
    identities: NodeIdentities,
    coordinates: CoordinateQuantizer,
    distances: DistanceQuantizer,
    vocabulary: Vocabulary,
    *,
    with_distance: bool = True,
) -> torch.Tensor:
    """Write one row per walk entry. Distances are measured between the float64 frame coordinates.

    Args:
        graph: the drawing.
        walk: one of its Euler walks.
        identities: the identity of every node.
        coordinates: the coordinate bins.
        distances: the distance bins; unused when `with_distance` is False.
        vocabulary: the token families.
        with_distance: whether to write the fifth field.

    Returns:
        `(L, F)` int64 identifiers, without an end row: `F` is `ROW_WIDTH` with the distance
        field and `PLACE_WIDTH` without it.
    """
    nodes = walk.nodes
    bins = coordinates.to_bins(graph.positions)[nodes]
    jump = walk.arrivals == Arrival.JUMP
    lengths = (graph.positions[nodes[1:]] - graph.positions[nodes[:-1]]).norm(dim=1)
    distance = torch.full_like(nodes, D_NONE)
    distance[1:] = torch.where(jump[1:], D_NONE, vocabulary.distance.ids(distances.to_bins(lengths)))
    edge_type = torch.where(jump, JUMP, EDGE).long()
    node_ids = vocabulary.node.ids(identities.of_node[nodes])
    fields = [node_ids, vocabulary.x.ids(bins[:, 0]), vocabulary.y.ids(bins[:, 1]), edge_type]
    return torch.stack([*fields, distance] if with_distance else fields, 1)


def _crossed_rows(rows: torch.Tensor, vocabulary: Vocabulary) -> torch.Tensor:
    """Check that every token belongs to its field's family.

    Returns:
        `(L,)` bool, the rows that cross a segment: every `EDGE` row after the first.

    Raises:
        DecodingError: If a token is outside its field's family.
    """
    identity, x, y, edge_type = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    first = torch.arange(rows.shape[0]) == 0
    crossed = ~first & (edge_type == EDGE)
    legal = vocabulary.node.contains(identity) & vocabulary.x.contains(x) & vocabulary.y.contains(y)
    legal &= (edge_type == EDGE) | ((edge_type == JUMP) & ~first)
    if rows.shape[1] > DISTANCE_FIELD:
        distance = rows[:, DISTANCE_FIELD]
        legal &= torch.where(crossed, vocabulary.distance.contains(distance), distance == D_NONE)
    if not bool(legal.all()):
        row = int((~legal).nonzero()[0])
        raise DecodingError(DecodingFailure.WRONG_FAMILY, f"row {row} has a token outside its field's family")
    return crossed


def _walk_rows(rows: torch.Tensor, capacity: int) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]]:
    """Register the nodes and segments the rows describe.

    A new identity must be the previous new one plus 1, modulo `capacity`. A `JUMP` starts the
    next component, so it must bring a new identity, and no row may return to an earlier one.

    Returns:
        The X and Y identifiers of each node, the node of each row, and the sorted segments.

    Raises:
        DecodingError: If an identity is out of order or changes bins, a row returns to an
            earlier component, or a segment joins a node to itself.
    """
    node_of: dict[int, int] = {}
    cells: list[tuple[int, int]] = []
    component_of: list[int] = []
    row_nodes: list[int] = []
    edges: set[tuple[int, int]] = set()
    first_token = int(rows[0, 0])
    component = 0
    for row, (token, x_id, y_id, kind) in enumerate(rows[:, :PLACE_WIDTH].tolist()):
        component += row > 0 and kind == JUMP
        node = node_of.setdefault(token, len(cells))
        if node == len(cells):
            if (token - first_token) % capacity != node:
                raise DecodingError(DecodingFailure.IDENTITY_ORDER, f"row {row} skips an identity")
            cells.append((x_id, y_id))
            component_of.append(component)
        elif cells[node] != (x_id, y_id):
            raise DecodingError(DecodingFailure.IDENTITY_CONFLICT, f"row {row} repeats an identity with other bins")
        elif component_of[node] != component:
            raise DecodingError(DecodingFailure.COMPONENT_CONFLICT, f"row {row} returns to an earlier component")
        if row and kind == EDGE:
            if row_nodes[-1] == node:
                raise DecodingError(DecodingFailure.SELF_LOOP, f"row {row} joins a node to itself")
            edges.add((min(row_nodes[-1], node), max(row_nodes[-1], node)))
        row_nodes.append(node)
    return cells, row_nodes, sorted(edges)


def decode_rows(
    rows: torch.Tensor,
    coordinates: CoordinateQuantizer,
    distances: DistanceQuantizer,
    vocabulary: Vocabulary,
    *,
    with_distance: bool = True,
) -> DecodedDrawing:
    """Read a drawing back from rows, checking every rule of the layout.

    A trailing all-`EOS` row is ignored. A new identity registers a node at its bin centers and
    must follow the previous new one; a known identity must repeat its bins. `EDGE` adds the
    segment from the previous row's node. `JUMP` adds nothing and starts the next component:
    it brings a new identity, and later rows never return to an earlier component. With
    `with_distance`, the distance field places nothing but must agree: its bin must overlap the
    distance between the two bin centers, widened by `coordinates.max_distance_error` plus
    `FLOAT_SLACK`. A place row carries no such field, so nothing can disagree.

    Returns:
        The drawing at bin centers.

    Raises:
        ValueError: If `rows` is not an `(L, F)` int64 tensor of this layout's width.
        DecodingError: If the rows break a rule; `reason` says which.
    """
    width = ROW_WIDTH if with_distance else PLACE_WIDTH
    if rows.dtype != torch.int64 or rows.ndim != 2 or rows.shape[1] != width:
        raise ValueError(f"rows must be an (L, {width}) int64 tensor")
    if rows.shape[0] and bool((rows[-1] == EOS).all()):
        rows = rows[:-1]
    if not rows.shape[0]:
        raise DecodingError(DecodingFailure.EMPTY, "there are no walk rows")
    crossed = _crossed_rows(rows, vocabulary)
    cells, row_nodes, edges = _walk_rows(rows, vocabulary.node.size)

    cell_ids = torch.tensor(cells, dtype=torch.int64)
    positions = torch.stack(
        [
            coordinates.centers(vocabulary.x.index(cell_ids[:, 0])),
            coordinates.centers(vocabulary.y.index(cell_ids[:, 1])),
        ],
        dim=1,
    )
    if with_distance:
        walked = positions[row_nodes]
        lengths = (walked[1:] - walked[:-1]).norm(dim=1)[crossed[1:]]
        low, high = distances.bounds(vocabulary.distance.index(rows[crossed, DISTANCE_FIELD]))
        tolerance = coordinates.max_distance_error + FLOAT_SLACK
        mismatch = (low > lengths + tolerance) | (high < lengths - tolerance)
        if bool(mismatch.any()):
            row = int(crossed.nonzero()[int(mismatch.nonzero()[0])])
            raise DecodingError(DecodingFailure.DISTANCE_MISMATCH, f"row {row} has a distance its bins cannot have")
    return DecodedDrawing(positions, torch.tensor(edges, dtype=torch.int64).reshape(-1, 2).T.contiguous())
