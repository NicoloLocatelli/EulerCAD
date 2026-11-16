"""Multi-view retrieval scoring at corpus scale.

`relational.recall_at_k` and `relational.mean_average_precision` define the
semantics used here, but they materialise the full (N, N) similarity matrix and
then run a Python loop that argsorts one row per query. On the ABC test split
that is a 21 GB allocation and a loop of tens of thousands of full sorts, which
is why the DINOv2 baseline notebook computes every other metric and skips these
two.

This module computes the same two numbers in blocks of queries, so peak memory
is (block, N) rather than (N, N), and optionally on the GPU. `retrieval_metrics`
is checked against the numpy originals in tests/test_retrieval.py.

Ranks come from `torch.searchsorted` over the sorted similarity row rather than
from an argsort, which keeps the per-block cost at one sort and turns rank
lookup into a binary search. Tied scores are then spread over consecutive ranks
rather than all being called rank 1, which is what keeps average precision at or
below 1 when a model embeds several views onto the same point.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def _as_codes(labels: np.ndarray | Sequence) -> torch.Tensor:
    """Map arbitrary group labels onto contiguous integer codes.

    Args:
        labels: (N,) group id per row. Strings, ints, anything np.unique accepts.

    Returns:
        (N,) int64 tensor of codes in [0, G).
    """
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    return torch.from_numpy(codes.reshape(-1).astype(np.int64))


def _padded_members(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group membership as a padded table.

    Args:
        codes: (N,) int64 group code per row.

    Returns:
        (members, sizes) where members is (G, max_size) holding the row indices of
        each group padded with -1, and sizes is (G,) the true member count. The
        padding value is -1 rather than 0 because 0 is a valid row index.
    """
    order = torch.argsort(codes, stable=True)
    sorted_codes = codes[order]
    sizes = torch.bincount(sorted_codes)
    max_size = int(sizes.max()) if sizes.numel() else 0

    members = torch.full((sizes.numel(), max_size), -1, dtype=torch.long)
    # position of each row within its own group, from a running count restarted
    # at every group boundary: arange minus the offset of the group it fell into
    starts = torch.cumsum(sizes, 0) - sizes
    within = torch.arange(codes.numel()) - starts[sorted_codes]
    members[sorted_codes, within] = order
    return members, sizes


def retrieval_metrics(
    z: np.ndarray | torch.Tensor,
    labels: np.ndarray | Sequence,
    ks: Sequence[int] = (1, 5, 10),
    block: int = 256,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Recall@k and mean average precision over cosine similarity.

    Every row is used as a query in turn; its relevant set is the other rows
    carrying the same label. Rows whose group has no other member are dropped
    from the query set, matching `relational._rank_relevant`.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row, e.g. part_id.
        ks: cutoffs to report recall at. Values above N-1 are clamped.
        block: queries scored per pass. Peak memory is about
            block * N * 8 bytes; 256 queries against a 150k corpus is ~300 MB.
        device: where to run. None keeps the tensors where they already are,
            which for a numpy input means CPU.

    Returns:
        {"recall@1": ..., "map": ..., "num_queries": ..., "num_groups": ...}.
        Scores are in [0, 1]; num_queries and num_groups are counts, returned so
        a caller can tell a hard corpus from a nearly empty one.

    Raises:
        ValueError: if z and labels disagree on N, or no group has two members -
            either means the retrieval question is not well posed on this input,
            and a zero score would misreport that as a bad model.
    """
    embeddings = torch.as_tensor(z, dtype=torch.float32)
    codes = _as_codes(labels)
    if embeddings.shape[0] != codes.shape[0]:
        raise ValueError(f"z has {embeddings.shape[0]} rows but labels has {codes.shape[0]}")

    members, sizes = _padded_members(codes)
    if not bool((sizes >= 2).any()):
        raise ValueError("no label group has two or more members, so nothing is retrievable")

    if device is not None:
        embeddings, codes, members, sizes = (t.to(device) for t in (embeddings, codes, members, sizes))

    total = embeddings.shape[0]
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    queries = torch.nonzero(sizes[codes] >= 2, as_tuple=False).reshape(-1)
    cutoffs = sorted({min(int(k), total - 1) for k in ks})

    hits = dict.fromkeys(cutoffs, 0)
    average_precision = torch.zeros((), dtype=torch.float64, device=embeddings.device)

    for start in range(0, queries.numel(), block):
        rows = queries[start : start + block]
        similarity = embeddings[rows] @ embeddings.T  # (B, N)
        # a row must not retrieve itself; -inf also parks it below every real
        # score in the ascending sort, so it never counts as outranking anything
        similarity[torch.arange(rows.numel(), device=rows.device), rows] = -torch.inf

        top = similarity.topk(max(cutoffs), dim=1).indices
        same_group = codes[top] == codes[rows].unsqueeze(1)
        for k in cutoffs:
            hits[k] += int(same_group[:, :k].any(dim=1).sum())

        relevant = members[codes[rows]]  # (B, R), padded with -1
        relevant = torch.where(relevant == rows.unsqueeze(1), -1, relevant)  # drop self
        keep = relevant >= 0
        relevant_scores = torch.gather(similarity, 1, relevant.clamp_min(0))
        relevant_scores = torch.where(keep, relevant_scores, torch.full_like(relevant_scores, -torch.inf))

        ascending, _ = similarity.sort(dim=1)
        # rank = 1 + how many rows scored strictly higher. searchsorted on the
        # ascending row gives the count of entries <= the score, so N minus that
        # is the count above it - one binary search instead of a second argsort.
        above = total - torch.searchsorted(ascending.contiguous(), relevant_scores.contiguous(), right=True)
        ranks = torch.where(keep, above + 1, torch.full_like(above, total + 1))

        # sorting ranks ascending puts the real ones first, since the padded
        # slots were parked at total + 1; sorting `keep` descending lines its
        # mask up with them without a second gather
        ranks, _ = ranks.sort(dim=1)
        position = torch.arange(1, ranks.shape[1] + 1, device=ranks.device).unsqueeze(0)
        # Two relevant rows with an identical score get an identical count of
        # rows above them, so both would be called rank 1 and average precision
        # would come out at 1.5 on a model whose views embed to the same point.
        # The i-th relevant row cannot rank above i, and imposing that is enough:
        # with distinct scores the ranks already satisfy it, so this is a no-op
        # everywhere except the ties it exists to resolve.
        ranks = torch.maximum(ranks, position)
        precision = (position / ranks).float()
        precision = torch.where(keep.sort(dim=1, descending=True).values, precision, torch.zeros_like(precision))
        average_precision += (precision.sum(dim=1) / keep.sum(dim=1).clamp_min(1)).double().sum()

    n_queries = int(queries.numel())
    scores = {f"recall@{k}": hits[k] / n_queries for k in cutoffs}
    scores["map"] = float(average_precision) / n_queries
    scores["num_queries"] = float(n_queries)
    scores["num_groups"] = float(int((sizes >= 2).sum()))
    return scores


def chance_recall_at_k(sizes: np.ndarray | Sequence[int], total: int, k: int = 1) -> float:
    """Recall@k a model that ranks at random would reach, for reporting alongside.

    Retrieval numbers are meaningless without this: on a corpus of 150k sketches
    with 19 relevant each, chance Recall@1 is about 0.013%, so 1% looks poor and
    is in fact seventy times chance.

    Args:
        sizes: member count per group; groups smaller than 2 are ignored.
        total: corpus size, i.e. how many rows a query is ranked against.
        k: cutoff.

    Returns:
        expected Recall@k under a uniformly random ranking, in [0, 1].
    """
    counts = np.asarray(list(sizes), dtype=np.float64)
    counts = counts[counts >= 2]
    if counts.size == 0 or total < 2:
        return 0.0
    # P(at least one of the (size-1) relevant rows lands in a random k-subset of
    # the other total-1 rows), weighted by how many queries each group supplies
    misses = np.ones_like(counts)
    for step in range(int(min(k, total - 1))):
        misses *= np.clip((total - 1 - (counts - 1) - step) / (total - 1 - step), 0.0, 1.0)
    return float(np.average(1.0 - misses, weights=counts))
