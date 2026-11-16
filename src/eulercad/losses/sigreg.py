"""`sigreg`: Sketched Isotropic Gaussian Regularization, the anti-collapse term of LeJEPA.

From "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
(Balestriero and LeCun, arXiv 2511.08544), Definition 2 and Algorithm 1. It pushes a batch of
embeddings towards the isotropic Gaussian N(0, I), which the paper proves minimizes the
worst-case risk of linear and nonlinear probes on tasks not known in advance (Section 3).

1. Project the embeddings on `num_slices` random unit directions, drawn from a generator
   seeded with the step, so every step tests new directions (Section 4.3).
2. On each direction, compare the empirical characteristic function of the projections,
   mean(exp(i t z)), with N(0, 1)'s, exp(-t^2 / 2), at `knots` points of [-5, 5]: the
   Epps-Pulley statistic N * integral |ecf(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt, by the
   trapezoid rule (Section 4.2.3).
3. Average over the directions (Definition 2).

The characteristic function is written as its cosine and sine means, which equal the paper's
complex form without complex tensors. Every sample's gradient is bounded whatever its value
(Theorem 4), so an outlier cannot blow up training. The statistic has a floor above zero, about
1.06 for perfectly Gaussian embeddings, from sampling (Theorem 6). Single device: the paper's
`all_reduce` of the characteristic function across devices is left out.
"""

import torch


def standardize(embeddings: torch.Tensor) -> torch.Tensor:
    """Center each dimension and divide by one standard deviation over every entry.

    Taken before `sigreg` in the fine-tunes (chosen 2026-10-13): the per-drawing means of
    long_gated's final norm spread about 0.16 per dimension, so on the raw vectors most of the
    statistic (62 of it, against 11 after this) asks only for bigger vectors, which the final
    norm's gain could give without changing their shape. One scale for all dimensions keeps
    uneven spreads and correlations visible to the test, unlike a per-dimension z-score.

    Returns:
        `(N, K)` float32, mean zero per dimension and standard deviation one overall.
    """
    centered = embeddings.float() - embeddings.float().mean(0)
    return centered / centered.std()


def sigreg(
    embeddings: torch.Tensor, global_step: int, *, num_slices: int = 256, knots: int = 17, bound: float = 5.0
) -> torch.Tensor:
    """Score how far a batch of embeddings is from N(0, I), along random directions.

    Args:
        embeddings: `(N, K)` one embedding per row; any float dtype, scored in float32.
        global_step: seeds the directions, so they change every step and repeat on a rerun.
        num_slices: random directions averaged.
        knots: points of the trapezoid rule on [-bound, bound].
        bound: half the width of the integration range.

    Returns:
        `()` float32, the mean Epps-Pulley statistic over the directions.

    Raises:
        ValueError: If `embeddings` is not a matrix of two rows or more.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be an (N, K) matrix, got {tuple(embeddings.shape)}")
    if embeddings.shape[0] < 2:
        raise ValueError("a characteristic function needs at least two rows")
    device = embeddings.device
    generator = torch.Generator(device=device)
    generator.manual_seed(global_step)
    directions = torch.randn(embeddings.shape[1], num_slices, generator=generator, device=device)
    directions /= directions.norm(p=2, dim=0)
    t = torch.linspace(-bound, bound, knots, device=device)
    gaussian = torch.exp(-0.5 * t**2)  # N(0, 1)'s characteristic function, also the weight
    phases = (embeddings.float() @ directions).unsqueeze(2) * t  # (N, slices, knots)
    error = (phases.cos().mean(0) - gaussian).square() + phases.sin().mean(0).square()
    return (torch.trapezoid(error * gaussian, t, dim=1) * embeddings.shape[0]).mean()
