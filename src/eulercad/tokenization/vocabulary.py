"""The token table shared by every row field.

Seven special tokens come first, with fixed identifiers. Four families of consecutive
identifiers follow, sized by `TokenizerConfig`: node identities, X bins, Y bins, distance bins.
The table is built from the sizes alone, never from data.
"""

from dataclasses import dataclass

import torch

SPECIAL_TOKENS = ("PAD", "MASK", "SEP", "EOS", "EDGE", "JUMP", "D_NONE")
PAD, MASK, SEP, EOS, EDGE, JUMP, D_NONE = range(len(SPECIAL_TOKENS))


@dataclass(frozen=True, slots=True)
class TokenFamily:
    """`size` consecutive tokens, `NAME_000` onwards, with identifiers `start` to `stop - 1`."""

    name: str
    start: int
    size: int

    @property
    def stop(self) -> int:
        return self.start + self.size

    def contains(self, ids: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of the identifiers in this family."""
        return (ids >= self.start) & (ids < self.stop)

    def ids(self, index: torch.Tensor) -> torch.Tensor:
        """Map indices in `[0, size)` to identifiers.

        Returns:
            The identifiers, the same shape as `index`.

        Raises:
            ValueError: If an index is out of range.
        """
        if index.numel() and (index.min() < 0 or index.max() >= self.size):
            raise ValueError(f"every {self.name} index must be in [0, {self.size - 1}]")
        return index + self.start

    def index(self, ids: torch.Tensor) -> torch.Tensor:
        """Map identifiers of this family back to indices.

        Returns:
            The indices, the same shape as `ids`.

        Raises:
            ValueError: If an identifier belongs to another token.
        """
        if not bool(self.contains(ids).all()):
            raise ValueError(f"every identifier must be a {self.name} token")
        return ids - self.start


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The identifiers of every token, laid out from three sizes.

    Attributes:
        max_nodes: the number of node identity tokens.
        coordinate_bins: the number of X tokens, and of Y tokens.
        distance_bins: the number of distance tokens.
    """

    max_nodes: int
    coordinate_bins: int
    distance_bins: int

    def __post_init__(self) -> None:
        if min(self.max_nodes, self.coordinate_bins, self.distance_bins) < 1:
            raise ValueError("every family size must be at least 1")

    @property
    def node(self) -> TokenFamily:
        return TokenFamily("ID", len(SPECIAL_TOKENS), self.max_nodes)

    @property
    def x(self) -> TokenFamily:
        return TokenFamily("X", self.node.stop, self.coordinate_bins)

    @property
    def y(self) -> TokenFamily:
        return TokenFamily("Y", self.x.stop, self.coordinate_bins)

    @property
    def distance(self) -> TokenFamily:
        return TokenFamily("D", self.y.stop, self.distance_bins)

    @property
    def families(self) -> tuple[TokenFamily, TokenFamily, TokenFamily, TokenFamily]:
        return self.node, self.x, self.y, self.distance

    @property
    def size(self) -> int:
        """The number of embedding rows."""
        return self.distance.stop

    def token_name(self, token_id: int) -> str:
        """Return the readable name of one identifier, such as `X_128`.

        Returns:
            A special token's name, or the family name and the index.

        Raises:
            ValueError: If the identifier is outside the vocabulary.
        """
        if 0 <= token_id < len(SPECIAL_TOKENS):
            return SPECIAL_TOKENS[token_id]
        for family in self.families:
            if family.start <= token_id < family.stop:
                return f"{family.name}_{token_id - family.start:03d}"
        raise ValueError(f"identifier {token_id} is outside the vocabulary of {self.size} tokens")
