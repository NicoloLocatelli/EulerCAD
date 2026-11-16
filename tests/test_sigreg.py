"""Tests for SIGReg (`losses/sigreg.py`), the regularizer of LeJEPA (Balestriero and LeCun, 2025)."""

import pytest
import torch

from eulercad.losses.sigreg import sigreg, standardize


def _paper(x: torch.Tensor, global_step: int, num_slices: int = 256) -> torch.Tensor:
    """Algorithm 1 of the paper, line for line, on one device, averaged over the slices as LeJEPA does.

    Returns:
        `()` the mean Epps-Pulley statistic over the slices.
    """
    dev = x.device
    g = torch.Generator(device=dev)
    g.manual_seed(global_step)
    proj_shape = (x.size(1), num_slices)
    a = torch.randn(proj_shape, generator=g, device=dev)
    a /= a.norm(p=2, dim=0)
    t = torch.linspace(-5, 5, 17, device=dev)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = (x @ a).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)
    err = (ecf - exp_f).abs().square().mul(exp_f)
    n = x.size(0)
    return (torch.trapz(err, t, dim=1) * n).mean()


def _gaussian(rows: int, width: int, seed: int) -> torch.Tensor:
    return torch.randn(rows, width, generator=torch.Generator().manual_seed(seed))


def test_sigreg_matches_algorithm_1_of_the_paper() -> None:
    x = _gaussian(300, 24, 0) * 1.7 + 0.3
    assert sigreg(x, global_step=11).item() == pytest.approx(_paper(x, 11).item(), rel=1e-5)


def test_standard_gaussian_embeddings_sit_near_the_sampling_floor() -> None:
    # Theorem 6: even a perfect Gaussian scores about the integral of w(t)(1 - exp(-t^2)), 1.06.
    assert sigreg(_gaussian(4096, 32, 1), global_step=0).item() == pytest.approx(1.06, abs=0.25)


def test_collapse_and_wrong_scale_score_far_higher() -> None:
    gaussian = sigreg(_gaussian(512, 16, 2), global_step=0)
    assert sigreg(torch.zeros(512, 16), global_step=0) > 50 * gaussian
    assert sigreg(3 * _gaussian(512, 16, 2), global_step=0) > 5 * gaussian


def test_the_directions_change_with_the_step_and_repeat_with_it() -> None:
    x = _gaussian(256, 8, 3) ** 2  # far from Gaussian, so the directions matter
    assert sigreg(x, global_step=5) == sigreg(x, global_step=5)
    assert sigreg(x, global_step=5) != sigreg(x, global_step=6)


def test_gradient_descent_on_sigreg_spreads_a_collapsed_batch() -> None:
    x = (0.01 * _gaussian(256, 8, 4)).requires_grad_()
    optimizer = torch.optim.Adam([x], lr=0.05)
    first = sigreg(x, global_step=0).item()
    for step in range(200):
        loss = sigreg(x, global_step=step)
        optimizer.zero_grad()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        optimizer.step()
    assert sigreg(x, global_step=999).item() < first / 20


def test_low_precision_inputs_are_scored_in_float32() -> None:
    x = _gaussian(128, 8, 5)
    assert sigreg(x.bfloat16(), global_step=0).dtype == torch.float32


def test_sigreg_refuses_anything_but_a_matrix_of_two_rows_or_more() -> None:
    with pytest.raises(ValueError, match="matrix"):
        sigreg(torch.zeros(4, 3, 2), global_step=0)
    with pytest.raises(ValueError, match="two rows"):
        sigreg(torch.zeros(1, 3), global_step=0)


def test_standardize_centers_and_divides_by_one_overall_spread() -> None:
    x = _gaussian(512, 6, 6) * torch.tensor([0.1, 0.1, 0.2, 0.2, 0.4, 0.8]) + 3.0
    z = standardize(x)

    torch.testing.assert_close(z.mean(0), torch.zeros(6), atol=1e-5, rtol=0)
    assert z.std().item() == pytest.approx(1.0, rel=1e-5)
    # One scale for every dimension: their spreads keep their ratios, so an uneven spread still shows.
    torch.testing.assert_close(z.std(0) / z.std(0)[0], x.std(0) / x.std(0)[0], rtol=1e-4, atol=0)
