"""Tests for the multi-member generalisation of the relational metrics.

Two things have to hold at once. A group of exactly two must still produce the
number the old exactly-two implementation produced, or every result already
recorded against these metrics silently changes meaning. And a larger group must
actually use all of its members, or the generalisation bought nothing.

The reference implementations below are the old formulas, written out so the
first property is checked against something rather than asserted.

Plain asserts and a __main__ runner, matching tests/test_gnn04_wiring.py -
pytest is not a project dependency. Everything is synthetic.
"""

import sys

import numpy as np
from scipy.special import logsumexp

from eulercad.analysis.metrics.relational import (
    _decouple,
    _group_members,
    _within_group_pairs,
    dime,
    infonce,
    whitened_dirichlet_energy,
)


def _normalised(z: np.ndarray) -> np.ndarray:
    """
    Returns:
        z with unit-norm rows.
    """
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def _old_infonce(z: np.ndarray, labels: np.ndarray, temperature: float = 0.5) -> float:
    """The exactly-two-members implementation, kept as the compatibility reference.

    Returns:
        mean NT-Xent loss, each row scored against its single partner.
    """
    labels = np.asarray(labels)
    partner = np.empty(len(labels), dtype=int)
    for group in np.unique(labels):
        i, j = np.where(labels == group)[0]
        partner[i], partner[j] = j, i

    sim = (_normalised(z) @ _normalised(z).T) / temperature
    np.fill_diagonal(sim, -np.inf)
    return float(np.mean(logsumexp(sim, axis=1) - sim[np.arange(len(labels)), partner]))


def _reference_infonce(z: np.ndarray, labels: np.ndarray, temperature: float = 0.5) -> float:
    """Multi-positive InfoNCE written the slow, obvious way.

    Returns:
        mean over anchors of the mean over that anchor's positives.
    """
    labels = np.asarray(labels)
    sim = (_normalised(z) @ _normalised(z).T) / temperature
    np.fill_diagonal(sim, -np.inf)

    losses = []
    for i in range(len(labels)):
        positives = [j for j in range(len(labels)) if j != i and labels[j] == labels[i]]
        if not positives:
            continue
        losses.append(np.mean([logsumexp(sim[i]) - sim[i, j] for j in positives]))
    return float(np.mean(losses))


def _corpus(sizes: list[int], dim: int = 6, spread: float = 0.4, seed: int = 0):
    """Groups of the requested sizes, each a jittered cloud around its own centre.

    Returns:
        (z, labels).
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim))
    z = np.concatenate([centres[i] + spread * rng.normal(size=(s, dim)) for i, s in enumerate(sizes)])
    labels = np.array([i for i, s in enumerate(sizes) for _ in range(s)])
    return z, labels


def test_infonce_on_pairs_reproduces_the_old_number():
    """The compatibility gate. Every infonce already reported was computed on a
    one-random-pair-per-part subsample; if the two-member case moved, none of
    those numbers would mean what the notebooks say they mean."""
    z, labels = _corpus([2] * 15, seed=1)
    got, expected = infonce(z, labels, block=4), _old_infonce(z, labels)
    assert abs(got - expected) < 1e-10, f"infonce on pairs: {got} vs old {expected}"


def test_infonce_averages_over_every_positive():
    """The point of the change. With four members an anchor has three positives,
    and scoring only one of them gives a different answer."""
    z, labels = _corpus([4, 4, 4, 4], seed=2)
    got, expected = infonce(z, labels, block=3), _reference_infonce(z, labels)
    assert abs(got - expected) < 1e-10, f"infonce: {got} vs reference {expected}"

    # and it is genuinely not the one-positive answer
    first_only = np.concatenate([np.where(labels == g)[0][:2] for g in np.unique(labels)])
    assert abs(got - _old_infonce(z[first_only], labels[first_only])) > 1e-3


def test_infonce_block_size_cannot_change_the_answer():
    z, labels = _corpus([3, 5, 2, 4], seed=3)
    one, many = infonce(z, labels, block=1), infonce(z, labels, block=10_000)
    assert abs(one - many) < 1e-10, f"block size moved infonce: {one} vs {many}"


def test_infonce_of_a_random_embedding_sits_at_log_n_minus_one():
    """Chance stays log(N-1) whatever the group sizes, because the average is per
    positive. If it drifted with group size, runs on differently sized groups
    could not be put in one table."""
    rng = np.random.default_rng(4)
    for sizes in ([2] * 80, [8] * 20):
        z = rng.normal(size=(sum(sizes), 32))
        labels = np.array([i for i, s in enumerate(sizes) for _ in range(s)])
        chance = np.log(sum(sizes) - 1)
        got = infonce(z, labels)
        assert abs(got - chance) < 0.25, f"sizes {sizes[0]}: {got} vs chance {chance}"


def test_infonce_skips_rows_with_no_positive():
    """A singleton group has nothing to be scored against. Scoring it anyway would
    charge the model for a question that was never asked."""
    z, labels = _corpus([3, 3], seed=5)
    z_extra = np.concatenate([z, np.array([[9.0, 9.0, 9.0, 9.0, 9.0, 9.0]])])
    labels_extra = np.concatenate([labels, [99]])
    assert abs(infonce(z_extra, labels_extra) - _reference_infonce(z_extra, labels_extra)) < 1e-10


def test_dime_uses_every_within_group_pair():
    members, sizes, _ = _group_members(np.array([0, 0, 0, 1, 1, 2, 2, 2, 2]))
    left, right = _within_group_pairs(members, sizes, np.random.default_rng(0), max_pairs=10_000)
    assert len(left) == 3 + 1 + 6, f"{len(left)} pairs, expected 3 + 1 + 6"
    # no pair may join two different groups, and no row may pair with itself
    codes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert (codes[left] == codes[right]).all()
    assert (left != right).all()


def test_dime_pair_cap_is_respected_and_reproducible():
    members, sizes, _ = _group_members(np.repeat(np.arange(20), 6))
    first = _within_group_pairs(members, sizes, np.random.default_rng(7), max_pairs=50)
    second = _within_group_pairs(members, sizes, np.random.default_rng(7), max_pairs=50)
    assert len(first[0]) == 50, f"cap ignored: {len(first[0])} pairs"
    assert (first[0] == second[0]).all() and (first[1] == second[1]).all(), "same seed gave different pairs"


def test_decouple_keeps_the_multiset_and_breaks_the_groups():
    """The random baseline must reuse exactly the same embeddings as the true one,
    or the two entropies are not comparable. A permutation does that; resampling
    would use some rows twice and others never."""
    codes = np.repeat(np.arange(30), 4)
    members, sizes, group_of = _group_members(codes)
    left, right = _within_group_pairs(members, sizes, np.random.default_rng(0), max_pairs=10_000)

    shuffled = _decouple(left, right, group_of, np.random.default_rng(1))
    assert sorted(shuffled.tolist()) == sorted(right.tolist()), "the multiset changed"
    collisions = int((group_of[left] == group_of[shuffled]).sum())
    assert collisions == 0, f"{collisions} pairs still share a group after decoupling"


def test_dime_is_positive_when_the_within_group_variation_is_low_rank():
    """The direction check, and a correction to the obvious reading of DiME.

    matrix_based_entropy is the entropy of the *normalised* eigenspectrum, so it
    measures how many directions a set of residuals occupies, not how large they
    are. Tight-but-isotropic groups therefore score DiME *negative*: shrinking a
    residual cloud uniformly leaves it full-rank, which is maximal entropy.

    What DiME rewards is residuals confined to a subspace - which is exactly what
    view-invariance looks like, the nuisance varying along few directions while
    the semantic content stays put. So the positive case here varies its members
    along two of six dimensions, and the null case varies along all six."""
    rng = np.random.default_rng(8)
    centres = rng.normal(size=(40, 6))
    low_rank = np.concatenate([centres[i] + np.pad(0.4 * rng.normal(size=(6, 2)), ((0, 0), (0, 4))) for i in range(40)])
    labels = np.repeat(np.arange(40), 6)

    structured = dime(low_rank, labels, rng=np.random.default_rng(0))
    assert structured > 0.1, f"low-rank within-group variation scored {structured}"

    full_rank = np.concatenate([centres[i] + 0.4 * rng.normal(size=(6, 6)) for i in range(40)])
    isotropic = dime(full_rank, labels, rng=np.random.default_rng(0))
    assert isotropic < structured, f"isotropic {isotropic} should score below low-rank {structured}"


def test_dime_on_pairs_still_matches_a_hand_computed_true_residual():
    """The stochastic half of DiME is the baseline; the true-residual half is fully
    determined by the pair set, so that is what pins the two-member case."""
    from eulercad.analysis.metrics.spectral import matrix_based_entropy

    z, labels = _corpus([2] * 12, seed=10)
    members, sizes, _ = _group_members(labels)
    left, right = _within_group_pairs(members, sizes, np.random.default_rng(0), max_pairs=10_000)
    assert len(left) == 12, f"{len(left)} pairs from 12 two-member groups"

    expected_true = matrix_based_entropy(z[left] - z[right], alpha=1.0)
    pairs = [np.where(labels == g)[0] for g in np.unique(labels)]
    hand = matrix_based_entropy(np.array([z[a] - z[b] for a, b in pairs]), alpha=1.0)
    assert abs(expected_true - hand) < 1e-10


def test_group_members_agrees_with_a_direct_grouping():
    labels = np.array(["b", "a", "c", "a", "b", "b"])
    members, sizes, group_of = _group_members(labels)
    for code, name in enumerate(["a", "b", "c"]):
        rows = [int(r) for r in members[code] if r >= 0]
        assert sorted(rows) == np.where(labels == name)[0].tolist(), members
        assert int(sizes[code]) == len(rows)
        assert (group_of[rows] == code).all()


def test_both_metrics_reject_a_corpus_with_no_pairs():
    """
    Raises:
        AssertionError: if either metric scored a corpus in which no group has a
            second member, instead of saying the question is not well posed.
    """
    z = np.random.default_rng(0).normal(size=(5, 4))
    for metric in (dime, infonce):
        try:
            metric(z, np.arange(5))
        except ValueError:
            continue
        raise AssertionError(f"{metric.__name__} scored a corpus with no positive pairs")


def test_whitened_dirichlet_energy_was_already_general_and_still_is():
    """It always handled any group size >= 2. This guards it against being caught
    up in the change to its neighbours."""
    z, labels = _corpus([2, 5, 3], seed=11)
    energy = whitened_dirichlet_energy(z, labels)
    assert np.isfinite(energy) and energy > 0.0, energy


if __name__ == "__main__":
    # A minimal runner so this file works without pytest, which is not a project
    # dependency. T201 is suppressed because reporting results IS this block's job.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")  # noqa: T201
        except Exception as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")  # noqa: T201
    print(f"\n{len(tests) - failed}/{len(tests)} passed")  # noqa: T201
    sys.exit(1 if failed else 0)
