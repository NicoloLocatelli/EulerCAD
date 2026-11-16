"""Cyclic node identities: first-visit order, shifted by a random offset.

Cleaned from `core/graph_gpt_baseline/representation/identity.py`. There is one identity token
per possible node, `capacity` of them, so one token per identity is enough and no two nodes of
a drawing share one.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import random
from dataclasses import dataclass

import torch

from eulercad.representation.euler import EulerWalk


@dataclass(frozen=True, slots=True)
class NodeIdentities:
    """The identity of every graph node in one sample.

    Attributes:
        offset: the identity of the first node visited, in `[0, capacity)`.
        of_node: `(N,)` int64, identity index of graph node `i`, unique in `[0, capacity)`.
        capacity: the number of identity tokens.
    """

    offset: int
    of_node: torch.Tensor
    capacity: int

    def __post_init__(self) -> None:
        if self.of_node.dtype != torch.int64:
            raise TypeError("of_node must be int64")
        if not 0 <= self.offset < self.capacity:
            raise ValueError(f"the offset must be in [0, {self.capacity - 1}]")
        if self.of_node.ndim != 1 or self.of_node.unique().numel() != self.of_node.numel():
            raise ValueError("of_node must be one-dimensional and unique")
        if self.of_node.numel() and (self.of_node.min() < 0 or self.of_node.max() >= self.capacity):
            raise ValueError(f"every identity must be in [0, {self.capacity - 1}]")


def assign_identities(walk: EulerWalk, num_nodes: int, rng: random.Random, capacity: int) -> NodeIdentities:
    """Number the nodes in the order the walk first visits them, from a random offset, modulo `capacity`.

    Draws one value from `rng`: the offset.

    Returns:
        The identities, one per graph node.

    Raises:
        ValueError: If the graph has more nodes than `capacity`, or the walk does not visit
            every node of the graph and only those.
    """
    if num_nodes > capacity:
        raise ValueError(f"{num_nodes} nodes do not fit in {capacity} identities")
    first_visits = list(dict.fromkeys(walk.nodes.tolist()))
    if sorted(first_visits) != list(range(num_nodes)):
        raise ValueError("the walk must visit every node of the graph, and only those")
    offset = rng.randrange(capacity)
    of_node = torch.empty(num_nodes, dtype=torch.int64)
    of_node[first_visits] = (offset + torch.arange(num_nodes)) % capacity
    return NodeIdentities(offset, of_node, capacity)
