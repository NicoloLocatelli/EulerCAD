"""Tests for the scalable retrieval benchmark.

The point of core/metrics/retrieval.py is to compute exactly what
relational.recall_at_k and relational.mean_average_precision compute, without
the (N, N) allocation. So the load-bearing tests here are the two that check it
against those originals on inputs small enough for both to run: an
implementation that is fast and subtly different would be worse than no
implementation at all, because the DINOv2 baseline and every model after it get
compared through this number.

Plain asserts and a __main__ runner, matching tests/test_gnn04_wiring.py -
pytest is not a project dependency. Everything is synthetic.
"""

import sys

import numpy as np
import torch

from eulercad.analysis.metrics.relational import mean_average_precision, recall_at_k
from eulercad.analysis.metrics.retrieval import _padded_members, chance_recall_at_k, retrieval_metrics


def _corpus(n_groups: int = 12, per_group: int = 4, dim: int = 8, spread: float = 0.6, seed: int = 0):
    """A corpus with real group structure: a random centre per group, jittered.

    Returns:
        (z, labels) with labels as strings, so the string path gets exercised too.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_groups, dim))
    z = np.repeat(centres, per_group, axis=0) + spread * rng.normal(size=(n_groups * per_group, dim))
    labels = np.array([f"part_{i:03d}" for i in range(n_groups) for _ in range(per_group)])
    return z, labels


def test_recall_matches_the_numpy_original():
    z, labels = _corpus()
    scores = retrieval_metrics(z, labels, ks=(1, 5, 10), block=7)
    for k in (1, 5, 10):
        expected = recall_at_k(z, labels, k=k)
        assert abs(scores[f"recall@{k}"] - expected) < 1e-9, f"recall@{k}: {scores[f'recall@{k}']} vs {expected}"


def test_map_matches_the_numpy_original():
    """Tolerance is 1e-6, not exact: the originals build the similarity matrix in
    float64 while this runs in float32, which is the precision the embeddings
    actually arrive in. A gap at the ninth decimal is that cast, not a
    disagreement about what the metric means."""
    z, labels = _corpus()
    expected = mean_average_precision(z, labels)
    got = retrieval_metrics(z, labels, block=7)["map"]
    assert abs(got - expected) < 1e-6, f"map: {got} vs {expected}"


def test_it_still_matches_when_the_groups_are_ragged():
    """Uneven group sizes are the real case: a part whose views mostly fell out of
    the size band contributes two, not twenty. The padded-members table has to
    carry that without letting the padding leak into a rank."""
    rng = np.random.default_rng(3)
    sizes = [2, 5, 3, 2, 7, 4]
    z = np.concatenate([rng.normal(loc=i, scale=0.5, size=(s, 6)) for i, s in enumerate(sizes)])
    labels = np.array([i for i, s in enumerate(sizes) for _ in range(s)])

    scores = retrieval_metrics(z, labels, ks=(1, 3), block=4)
    expected_map = mean_average_precision(z, labels)
    assert abs(scores["map"] - expected_map) < 1e-6, f"map: {scores['map']} vs {expected_map}"
    for k in (1, 3):
        expected = recall_at_k(z, labels, k=k)
        assert abs(scores[f"recall@{k}"] - expected) < 1e-9, f"recall@{k}: {scores[f'recall@{k}']} vs {expected}"


def test_singletons_are_dropped_from_the_query_set():
    """A part with one surviving view has nothing to retrieve. Scoring it as a
    miss would make a corpus look worse the more it was filtered."""
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.normal(loc=0.0, size=(2, 5)), rng.normal(loc=9.0, size=(1, 5))])
    labels = np.array(["pair", "pair", "lonely"])

    scores = retrieval_metrics(z, labels, ks=(1,))
    assert int(scores["num_queries"]) == 2, f"queried {scores['num_queries']} rows, expected the 2 with a partner"
    assert int(scores["num_groups"]) == 1


def test_a_block_size_cannot_change_the_answer():
    """Blocking is the whole point of this module; if the answer moved with the
    block size every reported number would depend on the machine it ran on."""
    z, labels = _corpus(n_groups=9, per_group=3, seed=5)
    one = retrieval_metrics(z, labels, ks=(1, 5), block=1)
    many = retrieval_metrics(z, labels, ks=(1, 5), block=1000)
    for key in one:
        assert abs(one[key] - many[key]) < 1e-9, f"{key} moved with block size: {one[key]} vs {many[key]}"


def test_a_perfect_embedding_scores_one():
    """Identical embeddings within a group and orthogonal across: every relevant
    item outranks every irrelevant one, so both metrics must be exactly 1.

    This is also the tie case. All three views of a group sit at cosine 1.0 from
    each other, so a rank rule that counts only strictly-greater rows calls them
    all rank 1 and average precision comes out at 1.5 - a score no metric
    bounded by 1 should ever return."""
    z = np.repeat(np.eye(6), 3, axis=0)
    labels = np.repeat(np.arange(6), 3)
    scores = retrieval_metrics(z, labels, ks=(1, 2))
    assert abs(scores["recall@1"] - 1.0) < 1e-12, scores
    assert abs(scores["map"] - 1.0) < 1e-9, scores


def test_an_uninformative_embedding_lands_near_chance():
    """Random embeddings carry no group information, so recall@1 should sit near
    the chance value - which is what chance_recall_at_k is for. This also guards
    the direction of that helper: an off-by-one there would make every real
    result look either heroic or worthless."""
    rng = np.random.default_rng(11)
    n_groups, per_group = 60, 3
    z = rng.normal(size=(n_groups * per_group, 16))
    labels = np.repeat(np.arange(n_groups), per_group)

    got = retrieval_metrics(z, labels, ks=(1,))["recall@1"]
    chance = chance_recall_at_k([per_group] * n_groups, total=n_groups * per_group, k=1)
    assert abs(got - chance) < 0.05, f"random embeddings scored {got}, chance is {chance}"


def test_chance_grows_with_k_and_shrinks_with_corpus_size():
    assert chance_recall_at_k([2] * 100, total=200, k=1) < chance_recall_at_k([2] * 100, total=200, k=10)
    assert chance_recall_at_k([2] * 100, total=200, k=1) > chance_recall_at_k([2] * 1000, total=2000, k=1)
    # one relevant row out of 199 candidates
    assert abs(chance_recall_at_k([2] * 100, total=200, k=1) - 1.0 / 199.0) < 1e-12


def test_padded_members_never_reports_a_row_in_two_groups():
    codes = torch.tensor([2, 0, 1, 0, 2, 2, 1])
    members, sizes = _padded_members(codes)
    assert sizes.tolist() == [2, 2, 3]
    for group in range(3):
        rows = [int(r) for r in members[group] if r >= 0]
        assert sorted(rows) == sorted(torch.nonzero(codes == group).reshape(-1).tolist()), members
        assert len(rows) == int(sizes[group])


def test_a_corpus_with_no_pairs_raises_instead_of_scoring_zero():
    """Every group a singleton means the question is not well posed. Returning 0
    would read as a terrible model rather than an unusable split.

    Raises:
        AssertionError: if the corpus was scored instead of rejected.
    """
    z = np.random.default_rng(0).normal(size=(5, 4))
    try:
        retrieval_metrics(z, np.arange(5))
    except ValueError:
        return
    raise AssertionError("scored a corpus in which nothing is retrievable")


def test_labels_of_the_wrong_length_raise():
    """
    Raises:
        AssertionError: if mismatched labels were accepted instead of rejected.
    """
    z = np.random.default_rng(0).normal(size=(6, 4))
    try:
        retrieval_metrics(z, np.array([0, 0, 1]))
    except ValueError:
        return
    raise AssertionError("accepted labels that do not line up with the embeddings")


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
