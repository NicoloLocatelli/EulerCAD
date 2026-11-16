"""`GraphTokenizer`: the settings and components that turn a walked drawing into token rows.

Build one before training, pass it to the model, and store `to_dict()` with the checkpoint.
The special tokens are fixed, and a layout is a class rather than a setting, which is why
`to_dict()` records the class name and `tokenizer_from_dict` reads it back:

- `GraphTokenizer` writes the full five-field row.
- `PlaceTokenizer` writes the four-field place row, leaving the distance to the two endpoints.

Both share `TokenizerConfig`, so a place run keeps the full run's token identifiers and the two
can be compared field by field. A place run therefore carries the distance family in its
vocabulary without ever writing one of its tokens, which costs 256 embedding rows.
"""

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from eulercad.representation.euler import EulerWalk
from eulercad.representation.graph import MAX_NODES, DrawingGraph
from eulercad.tokenization.identity import NodeIdentities, assign_identities
from eulercad.tokenization.quantization import CoordinateQuantizer, DistanceQuantizer
from eulercad.tokenization.rows import (
    PLACE_FIELDS,
    PLACE_WIDTH,
    ROW_FIELDS,
    ROW_WIDTH,
    DecodedDrawing,
    decode_rows,
    encode_rows,
)
from eulercad.tokenization.vocabulary import Vocabulary


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """The tokenizer settings, plain values that can be written to JSON.

    Attributes:
        max_nodes: node identity tokens, and so the most nodes a drawing may have.
        coordinate_bins: bins per axis over `[-1, 1]`.
        distance_bins: logarithmic distance bins, bin 0 included.
        d_min: the shortest length outside distance bin 0.
        d_max: the length where the last distance bin starts; the frame diagonal by default.
    """

    max_nodes: int = MAX_NODES
    coordinate_bins: int = 256
    distance_bins: int = 256
    d_min: float = 0.001
    d_max: float = 2 * math.sqrt(2)


@dataclass(frozen=True, slots=True)
class TokenizedDrawing:
    """One sample's rows.

    Attributes:
        rows: `(L, F)` int64 vocabulary identifiers, one row per walk entry, no end row.
        nodes: `(L,)` int64 graph node of each row.
        identities: the identity of every graph node.
    """

    rows: torch.Tensor
    nodes: torch.Tensor
    identities: NodeIdentities


@dataclass(frozen=True, slots=True)
class GraphTokenizer:
    """Five-field rows `[node_id, x_bin, y_bin, edge_type, distance_bin]` for drawing graphs.

    Immutable: the components are built once from `config`, which also checks the settings.
    Two tokenizers are equal when their settings are.
    """

    config: TokenizerConfig = field(default_factory=TokenizerConfig)
    coordinates: CoordinateQuantizer = field(init=False)
    distances: DistanceQuantizer = field(init=False)
    vocabulary: Vocabulary = field(init=False)

    def __post_init__(self) -> None:
        config = self.config
        object.__setattr__(self, "coordinates", CoordinateQuantizer(config.coordinate_bins))
        object.__setattr__(self, "distances", DistanceQuantizer(config.distance_bins, config.d_min, config.d_max))
        vocabulary = Vocabulary(config.max_nodes, config.coordinate_bins, config.distance_bins)
        object.__setattr__(self, "vocabulary", vocabulary)

    @property
    def vocab_size(self) -> int:
        """The number of embedding rows."""
        return self.vocabulary.size

    @property
    def fields(self) -> tuple[str, ...]:
        """The name of each field of a row, in row order."""
        return ROW_FIELDS

    @property
    def row_width(self) -> int:
        """The number of tokens per row."""
        return ROW_WIDTH

    def tokenize(self, graph: DrawingGraph, walk: EulerWalk, rng: random.Random) -> TokenizedDrawing:
        """Turn a drawing and one of its walks into token rows.

        Draws one value from `rng`, the identity offset; the caller draws the walk first, from
        the same generator. The walk must come from `euler_walk` on the same graph.

        Returns:
            The rows, without an end row.
        """
        identities = assign_identities(walk, graph.num_nodes, rng, self.config.max_nodes)
        rows = encode_rows(graph, walk, identities, self.coordinates, self.distances, self.vocabulary)
        return TokenizedDrawing(rows, walk.nodes.clone(), identities)

    def decode(self, rows: torch.Tensor) -> DecodedDrawing:
        """Read a drawing back from rows; see `decode_rows`.

        Returns:
            The drawing at bin centers.
        """
        return decode_rows(rows, self.coordinates, self.distances, self.vocabulary)

    def to_dict(self) -> dict[str, Any]:
        """Return the class name and the settings, ready for JSON."""
        return {"tokenizer": type(self).__name__, "config": asdict(self.config)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphTokenizer":
        """Rebuild a tokenizer from `to_dict()`.

        Returns:
            A tokenizer equal to the one that was saved.

        Raises:
            ValueError: If the data belongs to another tokenizer class.
        """
        if data.get("tokenizer") != cls.__name__:
            raise ValueError(f"the settings belong to {data.get('tokenizer')!r}, not {cls.__name__}")
        return cls(TokenizerConfig(**data["config"]))


@dataclass(frozen=True, slots=True)
class PlaceTokenizer(GraphTokenizer):
    """Four-field rows `[node_id, x_bin, y_bin, edge_type]` for drawing graphs.

    The distance field is gone. It never decided anything: `evaluation/constrained.py` picks it
    last, among the bins the two endpoints already allow, so the model's opinion about it was
    overwritten by geometry at every step. Training it still cost 39% of the next-row loss,
    spent telling apart a median of 23 bins that all describe the same length.

    Everything else, including the vocabulary, is `GraphTokenizer`'s, so the two layouts share
    their token identifiers and their per-field losses can be read side by side.
    """

    @property
    def fields(self) -> tuple[str, ...]:
        """The name of each field of a row, in row order."""
        return PLACE_FIELDS

    @property
    def row_width(self) -> int:
        """The number of tokens per row."""
        return PLACE_WIDTH

    def tokenize(self, graph: DrawingGraph, walk: EulerWalk, rng: random.Random) -> TokenizedDrawing:
        """Turn a drawing and one of its walks into place rows; see `GraphTokenizer.tokenize`.

        Returns:
            The rows, without an end row.
        """
        identities = assign_identities(walk, graph.num_nodes, rng, self.config.max_nodes)
        rows = encode_rows(
            graph, walk, identities, self.coordinates, self.distances, self.vocabulary, with_distance=False
        )
        return TokenizedDrawing(rows, walk.nodes.clone(), identities)

    def decode(self, rows: torch.Tensor) -> DecodedDrawing:
        """Read a drawing back from place rows; see `decode_rows`.

        Returns:
            The drawing at bin centers.
        """
        return decode_rows(rows, self.coordinates, self.distances, self.vocabulary, with_distance=False)


TOKENIZERS: dict[str, type[GraphTokenizer]] = {cls.__name__: cls for cls in (GraphTokenizer, PlaceTokenizer)}


def tokenizer_from_dict(data: dict[str, Any]) -> GraphTokenizer:
    """Rebuild whichever tokenizer wrote `data`, reading its class name.

    Returns:
        The tokenizer, of the class named in `data`.

    Raises:
        ValueError: If the class name is not one of `TOKENIZERS`.
    """
    name = data.get("tokenizer")
    if name not in TOKENIZERS:
        raise ValueError(f"{name!r} is not a known tokenizer; expected one of {sorted(TOKENIZERS)}")
    return TOKENIZERS[name].from_dict(data)
