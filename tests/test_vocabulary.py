"""Tests for the token table of the GraphGPT-based architecture."""

import pytest
import torch

from eulercad.tokenization.vocabulary import (
    D_NONE,
    EDGE,
    EOS,
    JUMP,
    MASK,
    PAD,
    SEP,
    SPECIAL_TOKENS,
    Vocabulary,
)

DEFAULT = Vocabulary(max_nodes=512, coordinate_bins=256, distance_bins=256)
SMALL = Vocabulary(max_nodes=16, coordinate_bins=8, distance_bins=4)


def test_special_tokens_come_first_with_fixed_identifiers() -> None:
    assert tuple(range(7)) == (PAD, MASK, SEP, EOS, EDGE, JUMP, D_NONE)
    assert SPECIAL_TOKENS == ("PAD", "MASK", "SEP", "EOS", "EDGE", "JUMP", "D_NONE")


def test_the_default_layout_matches_the_table_in_context() -> None:
    assert [(family.name, family.start, family.stop) for family in DEFAULT.families] == [
        ("ID", 7, 519),
        ("X", 519, 775),
        ("Y", 775, 1031),
        ("D", 1031, 1287),
    ]
    assert DEFAULT.size == 1287


def test_the_layout_follows_the_sizes() -> None:
    assert [(family.name, family.start, family.stop) for family in SMALL.families] == [
        ("ID", 7, 23),
        ("X", 23, 31),
        ("Y", 31, 39),
        ("D", 39, 43),
    ]
    assert (SMALL.node, SMALL.x, SMALL.y, SMALL.distance) == SMALL.families
    assert SMALL.size == 43


@pytest.mark.parametrize("vocabulary", [DEFAULT, SMALL], ids=["default", "small"])
def test_every_identifier_belongs_to_exactly_one_token(vocabulary: Vocabulary) -> None:
    ids = torch.arange(vocabulary.size)
    owners = (ids < len(SPECIAL_TOKENS)).long() + sum(family.contains(ids).long() for family in vocabulary.families)

    assert bool((owners == 1).all())


@pytest.mark.parametrize("vocabulary", [DEFAULT, SMALL], ids=["default", "small"])
def test_token_names_are_unique(vocabulary: Vocabulary) -> None:
    assert len({vocabulary.token_name(index) for index in range(vocabulary.size)}) == vocabulary.size


def test_token_names_are_readable() -> None:
    names = [DEFAULT.token_name(index) for index in (0, 6, 7, 518, 519, 1030, 1286)]

    assert names == ["PAD", "D_NONE", "ID_000", "ID_511", "X_000", "Y_255", "D_255"]
    assert [SMALL.token_name(index) for index in (22, 23, 42)] == ["ID_015", "X_000", "D_003"]


@pytest.mark.parametrize("index", [-1, 43])
def test_a_name_outside_the_vocabulary_is_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        SMALL.token_name(index)


def test_a_family_maps_between_indices_and_identifiers() -> None:
    assert DEFAULT.x.ids(torch.tensor([0, 128, 255])).tolist() == [519, 647, 774]
    assert DEFAULT.x.index(torch.tensor([519, 647, 774])).tolist() == [0, 128, 255]
    assert DEFAULT.x.contains(torch.tensor([518, 519, 774, 775])).tolist() == [False, True, True, False]


def test_a_family_rejects_what_is_not_its_own() -> None:
    with pytest.raises(ValueError, match="Y index"):
        SMALL.y.ids(torch.tensor([8]))
    with pytest.raises(ValueError, match="D token"):
        SMALL.distance.index(torch.tensor([D_NONE]))


def test_sizes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must"):
        Vocabulary(max_nodes=0, coordinate_bins=8, distance_bins=4)
