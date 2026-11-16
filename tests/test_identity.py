"""Tests for the cyclic node identities of the GraphGPT-based architecture."""

import random

import pytest
import torch

from eulercad.representation.euler import Arrival, EulerWalk
from eulercad.tokenization.identity import NodeIdentities, assign_identities


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  # nosec B311


def _walk(nodes: list[int]) -> EulerWalk:
    arrivals = [Arrival.START] + [Arrival.ORIGINAL] * (len(nodes) - 1)
    return EulerWalk(torch.tensor(nodes), torch.tensor(arrivals, dtype=torch.int8))


@pytest.mark.parametrize("capacity", [512, 8])
def test_identities_follow_the_first_visit_order_from_the_offset(capacity: int) -> None:
    identities = assign_identities(_walk([2, 0, 2, 3, 1, 3]), 4, _rng(7), capacity)

    offset = _rng(7).randrange(capacity)
    assert (identities.offset, identities.capacity) == (offset, capacity)
    assert identities.of_node.tolist() == [(offset + k) % capacity for k in (1, 3, 0, 2)]


@pytest.mark.parametrize("capacity", [512, 16])
def test_identities_wrap_around_and_stay_unique_at_full_capacity(capacity: int) -> None:
    order = torch.randperm(capacity, generator=torch.Generator().manual_seed(0)).tolist()

    for seed in range(5):
        identities = assign_identities(_walk(order), capacity, _rng(seed), capacity)
        assert sorted(identities.of_node.tolist()) == list(range(capacity))
        assert identities.of_node[order[0]] == identities.offset


def test_the_offset_changes_between_draws_and_repeats_for_the_same_seed() -> None:
    walk = _walk([0, 1, 2])

    offsets = {assign_identities(walk, 3, _rng(seed), 512).offset for seed in range(50)}

    assert len(offsets) > 40
    first, second = (assign_identities(walk, 3, _rng(3), 512) for _ in range(2))
    assert torch.equal(first.of_node, second.of_node)


@pytest.mark.parametrize(("nodes", "num_nodes"), [([0, 1], 3), ([0, 3], 3)])
def test_a_walk_that_does_not_visit_exactly_the_graph_nodes_is_rejected(nodes: list[int], num_nodes: int) -> None:
    with pytest.raises(ValueError, match="every node"):
        assign_identities(_walk(nodes), num_nodes, _rng(0), 512)


def test_more_nodes_than_identities_are_rejected() -> None:
    with pytest.raises(ValueError, match="do not fit in 8"):
        assign_identities(_walk(list(range(9))), 9, _rng(0), 8)


def test_an_identity_record_checks_its_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        NodeIdentities(0, torch.tensor([0, 0]), 8)
    with pytest.raises(ValueError, match="offset"):
        NodeIdentities(8, torch.tensor([0]), 8)
    with pytest.raises(ValueError, match="identity"):
        NodeIdentities(0, torch.tensor([0, 8]), 8)
    with pytest.raises(TypeError, match="int64"):
        NodeIdentities(0, torch.tensor([0], dtype=torch.int32), 8)
