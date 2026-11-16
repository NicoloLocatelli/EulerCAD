"""Invariance / relational embedding-quality metrics (uses the augmentation graph).

Groups may be any size from two upwards. `dime` and `infonce` originally demanded
exactly two members, which on ABC - where a CAD model contributes ten to fourteen
surviving views - forced every caller to draw one random pair per part and throw
the rest away. Both now use the whole group: `dime` over every within-group pair,
`infonce` averaged over every positive of each anchor. A group of exactly two
reduces to the old formula in both cases, so numbers already reported stay
comparable.

`infonce` is also blocked over rows now. The (N, N) similarity matrix it used to
build is 21 GB on the ABC test split, which is the other half of why callers
subsampled.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from eulercad.analysis.metrics.spectral import matrix_based_entropy


def _sym_inv_sqrt(mat: np.ndarray, delta: float) -> np.ndarray:
    """Symmetric inverse square root of (mat + delta*I), via eigendecomposition.

    Returns:
        the (mat + delta*I)^-1/2 matrix.
    """
    d = mat.shape[0]
    w, v = np.linalg.eigh(mat + delta * np.eye(d))
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def _group_members(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group membership as a padded table.

    Replaces the exactly-two-members rule these metrics used to impose. ABC gives
    ten to fourteen views per CAD model, so that rule forced every caller to pick
    one random pair per part and discard the rest - roughly nine tenths of the
    positive-pair evidence, and a different nine tenths on every seed.

    Args:
        labels: (N,) group id per row.

    Returns:
        (members, sizes, group_of): members is (G, max_size) holding each group's
        row indices padded with -1, sizes is (G,) the true counts, and group_of is
        (N,) the group index of each row. Padding is -1 because 0 is a valid row.
    """
    labels = np.asarray(labels).reshape(-1)
    unique, group_of = np.unique(labels, return_inverse=True)
    group_of = group_of.reshape(-1)

    order = np.argsort(group_of, kind="stable")
    sizes = np.bincount(group_of, minlength=len(unique))
    members = np.full((len(unique), int(sizes.max()) if sizes.size else 0), -1, dtype=int)
    # position of each row inside its own group: a running count restarted at
    # every group boundary, i.e. arange minus the offset of the group it fell into
    starts = np.cumsum(sizes) - sizes
    members[group_of[order], np.arange(len(labels)) - starts[group_of[order]]] = order
    return members, sizes, group_of


def _within_group_pairs(
    members: np.ndarray,
    sizes: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Every unordered pair of rows sharing a label, subsampled if there are too many.

    Args:
        members: (G, max_size) padded membership table from _group_members.
        sizes: (G,) true member count per group.
        rng: generator used for the subsample, so a capped run is reproducible.
        max_pairs: cap on the number of pairs returned. A group of m members
            contributes m(m-1)/2 pairs, so twelve views per part is sixty-six
            pairs and the total grows quadratically in group size.

    Returns:
        (left, right) row indices, each (P,), with P <= max_pairs.
    """
    left, right = [], []
    for group in np.flatnonzero(sizes >= 2):
        rows = members[group, : sizes[group]]
        a, b = np.triu_indices(len(rows), k=1)
        left.append(rows[a])
        right.append(rows[b])

    left, right = np.concatenate(left), np.concatenate(right)
    if len(left) > max_pairs:
        chosen = rng.choice(len(left), size=max_pairs, replace=False)
        left, right = left[chosen], right[chosen]
    return left, right


def _decouple(left: np.ndarray, right: np.ndarray, group_of: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute `right` so no entry keeps its partner's group, without changing the multiset.

    The entropies being compared are only comparable if the two residual sets draw
    on the same embeddings; a permutation preserves that exactly, where resampling
    would use some rows twice and others never. Colliding positions are re-permuted
    among themselves so the multiset survives the fix as well.

    Args:
        left: (P,) row index of each pair's first member.
        right: (P,) row index of each pair's second member.
        group_of: (N,) group index per row.
        rng: generator for the permutation.

    Returns:
        (P,) a permutation of `right` in which few or no entries share a group
        with the corresponding entry of `left`.
    """
    shuffled = right[rng.permutation(len(right))]
    for _ in range(20):
        collided = np.flatnonzero(group_of[left] == group_of[shuffled])
        if collided.size == 0:
            break
        if collided.size == 1:
            # a lone leftover cannot be fixed by permuting it with itself, so it
            # has to be swapped into a slot where neither side then collides
            i = int(collided[0])
            for j in rng.permutation(len(shuffled)):
                if group_of[left[i]] != group_of[shuffled[j]] and group_of[left[j]] != group_of[shuffled[i]]:
                    shuffled[[i, j]] = shuffled[[j, i]]
                    break
            else:  # pragma: no cover - only reachable when every row shares one group
                break
        else:
            shuffled[collided] = shuffled[collided[rng.permutation(collided.size)]]
    return shuffled


def whitened_dirichlet_energy(z: np.ndarray, labels: np.ndarray, delta: float = 1e-3) -> float:
    """Whitened Dirichlet energy of Z on the augmentation graph G_PV (Balestriero-LeCun, Thm 2).

    Energy of C^-1/2 Z summed over edges of the positive-view graph (all pairs within an
    augmentation group), averaged per edge. Whitening makes this collapse-safe: a degenerate
    C blows up C^-1/2, so mapping everything to one point does not trivially minimize it.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row (augmentation groups; any group size >= 2).
        delta: ridge term added to the covariance before whitening.

    Returns:
        mean squared whitened distance over all within-group pairs.
    """
    n = z.shape[0]
    cov = (z.T @ z) / n
    zw = z @ _sym_inv_sqrt(cov, delta)

    total, n_pairs = 0.0, 0
    for group in np.unique(labels):
        members = zw[labels == group]
        k = len(members)
        if k < 2:
            continue
        group_mean = members.mean(axis=0)
        # sum_{i<j} ||xi-xj||^2 == k * sum_i ||xi-mean||^2 (pairwise/deviation identity)
        total += k * np.sum((members - group_mean) ** 2)
        n_pairs += k * (k - 1) // 2
    return float(total / n_pairs)


def dime(
    z: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
    max_pairs: int = 200_000,
) -> float:
    """DiME: difference of matrix-based entropies between true and random pairings.

    Entropy of the true positive-pair residuals (z_a - z_b) vs. the same residuals under a
    random re-pairing of the b-views. A model that actually encodes the augmentation
    invariance produces tightly-spread true residuals (low entropy) relative to the random
    baseline (high entropy).

    Every unordered within-group pair is used, not one pair per group. On a group of
    exactly two members that is the single pair this used to require, so a caller that
    still subsamples gets the same number it always did.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row. Any group size >= 2; smaller groups contribute
            nothing and are skipped rather than rejected.
        alpha: Renyi order passed through to matrix_based_entropy.
        rng: random generator for the shuffled baseline and for the pair subsample
            (default: fresh Generator).
        max_pairs: cap on how many pairs are scored, since pairs grow quadratically
            in group size and the entropy costs O(P * d^2).

    Returns:
        S_random - S_true, in nats. Positive when the representation encodes real
        pair structure beyond chance.

    Raises:
        ValueError: if no group has two or more members, so no residual exists.
    """
    rng = rng or np.random.default_rng()
    members, sizes, group_of = _group_members(labels)
    if not (sizes >= 2).any():
        raise ValueError("no label group has two or more members, so there are no positive pairs")

    left, right = _within_group_pairs(members, sizes, rng, max_pairs)
    residual_true = z[left] - z[right]
    residual_random = z[left] - z[_decouple(left, right, group_of, rng)]

    s_true = matrix_based_entropy(residual_true, alpha=alpha)
    s_random = matrix_based_entropy(residual_random, alpha=alpha)
    return float(s_random - s_true)


def _gram_entropy(gram: np.ndarray, alpha: float) -> float:
    """Renyi entropy of the eigenvalues of a Gram matrix, scaled to sum to one; non-positive ones are dropped.

    Returns:
        entropy in nats.
    """
    eigvals = np.linalg.eigvalsh(gram)
    p = eigvals[eigvals > 0] / eigvals[eigvals > 0].sum()
    if alpha == 1:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p**alpha)) / (1 - alpha))


def joint_dime(
    z_a: np.ndarray,
    z_b: np.ndarray,
    alpha: float = 1.0,
    permutations: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """DiME as Skean et al. define it (arXiv 2301.08164) and the Layer-by-Layer code computes it.

    The joint entropy of two views is the matrix entropy of the elementwise product of their
    (N, N) linear Gram matrices. DiME is that joint entropy with the second view's rows randomly
    re-paired, averaged over `permutations` draws, minus the joint entropy of the true pairing
    (`repitl.difference_of_entropies.doe`). This differs from `dime` above, which compares the
    entropies of the pair differences z_a - z_b.

    Args:
        z_a: (N, d) first view, row i from sample i.
        z_b: (N, d) second view, row i from the same sample i.
        alpha: Renyi order.
        permutations: random re-pairings averaged.
        rng: generator of the re-pairings (default: fresh Generator).

    Returns:
        mean H(random pairing) - H(true pairing), in nats. Positive when the views agree on
        which samples are alike.

    Raises:
        ValueError: if the two views do not hold the same number of rows.
    """
    if z_a.shape[0] != z_b.shape[0]:
        raise ValueError("the two views must hold the same drawings, one row each")
    rng = rng or np.random.default_rng()
    gram_a, gram_b = z_a @ z_a.T, z_b @ z_b.T
    true = _gram_entropy(gram_a * gram_b, alpha)
    shuffled = []
    for _ in range(permutations):
        order = rng.permutation(len(gram_b))
        shuffled.append(_gram_entropy(gram_a * gram_b[np.ix_(order, order)], alpha))
    return float(np.mean(shuffled) - true)


def infonce(z: np.ndarray, labels: np.ndarray, temperature: float = 0.5, block: int = 256) -> float:
    """InfoNCE / NT-Xent, evaluated (no training) on precomputed embeddings.

    log N - InfoNCE lower-bounds the mutual information between augmented views:
    log(N) - InfoNCE <= I(X; Z) <= H(Z) (Skean et al., "Layer by Layer", Theorem 3 /
    Appendix G Theorem 7).

    A row with several positives is scored against all of them and the results
    averaged, which is the standard supervised-contrastive generalisation. On a
    group of exactly two members it collapses to the single-positive formula this
    used to require, so previously reported numbers stay comparable. Chance is
    log(N - 1) whatever the group sizes, because the average is taken per
    positive, so runs on differently sized groups remain on one scale.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row. Any group size >= 2; rows in a singleton
            group have nothing to be scored against and are skipped.
        temperature: softmax temperature.
        block: rows scored per pass. Peak memory is about block * N * 8 bytes,
            rather than the (N, N) matrix this used to build - which is what made
            the metric unrunnable on a full split without subsampling first.

    Returns:
        mean NT-Xent loss over the rows that have at least one positive.

    Raises:
        ValueError: if no group has two or more members, so nothing is scorable.
    """
    members, sizes, group_of = _group_members(labels)
    if not (sizes >= 2).any():
        raise ValueError("no label group has two or more members, so there are no positives to score")

    n = z.shape[0]
    z_norm = z / np.linalg.norm(z, axis=1, keepdims=True)
    total, counted = 0.0, 0

    for start in range(0, n, block):
        rows = np.arange(start, min(start + block, n))
        sim = (z_norm[rows] @ z_norm.T) / temperature
        sim[np.arange(len(rows)), rows] = -np.inf  # a row is not its own negative

        positives = members[group_of[rows]]  # (B, R), padded with -1
        keep = (positives >= 0) & (positives != rows[:, None])
        gathered = np.take_along_axis(sim, np.maximum(positives, 0), axis=1)
        # mean over p of (logsumexp - s_p) == logsumexp - mean over p of s_p
        positive_mean = np.where(keep, gathered, 0.0).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)

        scorable = keep.any(axis=1)
        total += float((logsumexp(sim, axis=1)[scorable] - positive_mean[scorable]).sum())
        counted += int(scorable.sum())

    return total / counted


def _rank_relevant(z: np.ndarray, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each query row, other rows ranked by descending cosine similarity + a relevance mask.

    Returns:
        one (ranked_indices, is_relevant) pair per query that has >= 1 relevant item
        (i.e. belongs to a group with other members).
    """
    labels = np.asarray(labels)
    n = z.shape[0]
    z_norm = z / np.linalg.norm(z, axis=1, keepdims=True)
    sim = z_norm @ z_norm.T

    out = []
    for i in range(n):
        others = np.delete(np.arange(n), i)  # exclude self: same label as itself is not a hit
        if np.sum(labels[others] == labels[i]) < 1:
            continue
        order = others[np.argsort(-sim[i, others])]
        out.append((order, labels[order] == labels[i]))
    return out


def recall_at_k(z: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Recall@k: fraction of queries with >= 1 relevant item (same group) in the top-k retrieved.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row (relevant set = other rows with the same label).
        k: number of top retrieved neighbors considered.

    Returns:
        hit rate over all queries that have at least one relevant item, in [0, 1].
    """
    ranked = _rank_relevant(z, labels)
    hits = [bool(np.any(is_relevant[:k])) for _, is_relevant in ranked]
    return float(np.mean(hits))


def mean_average_precision(z: np.ndarray, labels: np.ndarray) -> float:
    """mAP over the full ranking: mean, across queries, of the average precision at each relevant hit.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row (relevant set = other rows with the same label).

    Returns:
        mean average precision, in [0, 1].
    """
    ranked = _rank_relevant(z, labels)
    ap_scores = []
    for _, is_relevant in ranked:
        hit_ranks = np.flatnonzero(is_relevant) + 1  # 1-indexed rank of each relevant hit
        precisions_at_hits = np.arange(1, len(hit_ranks) + 1) / hit_ranks
        ap_scores.append(precisions_at_hits.mean())
    return float(np.mean(ap_scores))
