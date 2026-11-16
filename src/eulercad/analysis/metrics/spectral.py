"""Spectral / information-theoretic embedding-quality metrics."""

from __future__ import annotations

import numpy as np


def _covariance_eigvals(z: np.ndarray) -> np.ndarray:
    """
    Eigenvalues of the d x d covariance C = Z^T Z / N, descending, clipped to >= 0.
    Returns:
        eigvals: (d,) array of eigenvalues.
    """
    n = z.shape[0]
    cov = (z.T @ z) / n
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    return np.clip(eigvals, 0.0, None)


def matrix_based_entropy(z: np.ndarray, alpha: float = 1.0, eps: float = 1e-12) -> float:
    """Renyi entropy S_alpha of the normalized eigenspectrum of C = Z^T Z / N.

    Args:
        z: (N, d) embeddings.
        alpha: Renyi order. alpha=1 is the von Neumann (Shannon) limit, alpha=2 is
            collision entropy.
        eps: eigenvalues below this fraction of the trace are dropped before taking logs.

    Returns:
        entropy in nats.
    """
    eigvals = _covariance_eigvals(z)
    p = eigvals / eigvals.sum()
    p = p[p > eps]
    if alpha == 1:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p**alpha)) / (1 - alpha))


def rankme(z: np.ndarray, eps: float = 1e-7) -> float:
    """RankMe: smooth effective rank of the embedding spectrum (Garrido et al., 2023).

    Args:
        z: (N, d) embeddings.
        eps: additive smoothing on the normalized singular values before taking logs.

    Returns:
        effective rank, in [1, d].
    """
    eigvals = _covariance_eigvals(z)
    sigma = np.sqrt(eigvals)  # singular values of Z, up to a shared sqrt(N) factor that cancels below
    p = sigma / sigma.sum() + eps
    return float(np.exp(-np.sum(p * np.log(p))))


def alpha_req(z: np.ndarray) -> float:
    """alpha-ReQ: power-law decay exponent of the eigenspectrum, lambda_k ~ k^-alpha.

    Args:
        z: (N, d) embeddings.

    Returns:
        alpha (positive for a decaying spectrum).
    """
    eigvals = _covariance_eigvals(z)
    eigvals = eigvals[eigvals > 0]
    k = np.arange(1, len(eigvals) + 1)
    slope, _ = np.polyfit(np.log(k), np.log(eigvals), 1)
    return float(-slope)


def lidar(z: np.ndarray, labels: np.ndarray, delta: float = 1e-3, eps: float = 1e-7) -> float:
    """LiDAR: effective rank of the between-class scatter, using augmentation groups as classes.

    Args:
        z: (N, d) embeddings.
        labels: (N,) group id per row (e.g. which original drawing an augmented view came from).
            Groups with a single member contribute no within-class scatter and are still valid.
        delta: ridge term added to the within-class scatter before whitening (collapse-safety).
        eps: additive smoothing on the normalized eigenvalues before taking logs.

    Returns:
        effective rank of the discriminative subspace, in [1, d].
    """
    n, d = z.shape
    global_mean = z.mean(axis=0)
    between = np.zeros((d, d))
    within = np.zeros((d, d))
    for group in np.unique(labels):
        members = z[labels == group]
        group_mean = members.mean(axis=0)
        diff = group_mean - global_mean
        between += len(members) * np.outer(diff, diff)
        centered = members - group_mean
        within += centered.T @ centered
    between /= n
    within /= n

    within_reg = within + delta * np.eye(d)
    w, v = np.linalg.eigh(within_reg)
    within_inv_sqrt = v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    discriminant = within_inv_sqrt @ between @ within_inv_sqrt

    eigvals = np.clip(np.linalg.eigvalsh(discriminant)[::-1], 0.0, None)
    p = eigvals / eigvals.sum() + eps
    return float(np.exp(-np.sum(p * np.log(p))))
