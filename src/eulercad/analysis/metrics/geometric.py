"""Geometric embedding-quality metrics."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def two_nn(z: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN: intrinsic dimension from nearest-neighbor distance ratios (Facco et al., 2017).

    Args:
        z: (N, d) embeddings.
        discard_fraction: largest-ratio points dropped before the MLE fit (finite-size
            effects make them the least reliable).

    Returns:
        estimated intrinsic dimension.
    """
    nn = NearestNeighbors(n_neighbors=3).fit(z)
    dist, _ = nn.kneighbors(z)
    r1, r2 = dist[:, 1], dist[:, 2]
    mu = r2[r1 > 0] / r1[r1 > 0]

    # MLE of d is N / sum(log(mu)) under the full Pareto(d) model, but that formula
    # assumes untruncated data: naively re-summing over the smallest (1 - discard_fraction)
    # of mu after dropping the noisy upper tail biases the estimate (truncated-Pareto MLE
    # != untruncated-Pareto MLE). Facco et al.'s actual discard-robust estimator instead
    # regresses -log(1 - F_i) on log(mu_i) (origin-forced OLS) over the kept points, where
    # F_i = i/(N+1) is each point's empirical CDF rank in the FULL (untruncated) sample.
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    empirical_cdf = np.arange(1, n + 1) / (n + 1)
    keep = int(n * (1 - discard_fraction))
    x = np.log(mu_sorted[:keep])
    y = -np.log(1 - empirical_cdf[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


def curvature(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Turning angle between consecutive embedding steps of one ordered sequence.

    Args:
        z: (L, d) embeddings already ordered (e.g. one chain's walk order). L >= 3
            for any angle to be defined.
        eps: floor on step norms to avoid dividing by a zero-length step.

    Returns:
        (L-2,) turning angles in radians, one per interior position i:
        angle between (z[i]-z[i-1]) and (z[i+1]-z[i]). Empty array if L < 3.
    """
    if z.shape[0] < 3:
        return np.array([])
    steps = np.diff(z, axis=0)
    v1, v2 = steps[:-1], steps[1:]
    cos_angle = (v1 * v2).sum(axis=1) / np.clip(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), eps, None)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def anisotropy(z: np.ndarray) -> float:
    """Average pairwise cosine similarity between all embeddings (Ethayarajh, 2019).

    Closed form via the resultant of unit-normalized vectors, O(N*d) instead of O(N^2).

    Args:
        z: (N, d) embeddings.

    Returns:
        average cosine similarity over all i != j pairs, in [-1, 1]. Near 0 for an
        isotropic cloud, near 1 for a narrow cone (all embeddings point the same way).
    """
    n = z.shape[0]
    u = z / np.linalg.norm(z, axis=1, keepdims=True)
    resultant = u.sum(axis=0)
    pairwise_sum = np.dot(resultant, resultant) - n  # self-pairs (u_i . u_i = 1) removed
    return float(pairwise_sum / (n * (n - 1)))
