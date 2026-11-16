"""Tests for the Muon and AdamW optimizer of the GraphGPT-based architecture."""

import copy
import io

import pytest
import torch

from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.training.config import TrainingConfig
from eulercad.training.optimizer import MuonAdamW, make_optimizer
from eulercad.training.schedule import WarmupDecay

SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
TINY = ModelConfig(qk_norm=True)


def _config(optimizer: str) -> TrainingConfig:
    return TrainingConfig(total_steps=10, warmup_steps=2, learning_rate=1e-3, optimizer=optimizer)


def _model() -> GraphGPTModel:
    torch.manual_seed(0)
    return GraphGPTModel(SMALL, TINY)


def _gradients(model: GraphGPTModel, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for parameter in model.parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator)


def _matrices(model: GraphGPTModel) -> set[int]:
    return {id(value) for value in model.backbone.layers.parameters() if value.ndim == 2}


def test_adamw_stays_the_default_optimizer() -> None:
    optimizer = make_optimizer(_model(), _config("adamw"))

    assert type(optimizer) is torch.optim.AdamW


def test_muon_takes_the_layer_matrices_and_adamw_everything_else() -> None:
    model = _model()

    optimizer = make_optimizer(model, _config("muon"))

    assert isinstance(optimizer, MuonAdamW)
    muon = {id(value) for group in optimizer.muon.param_groups for value in group["params"]}
    adamw = {id(value) for group in optimizer.adamw.param_groups for value in group["params"]}
    assert muon == _matrices(model)
    assert len(muon) == TINY.hidden_layers * 7
    assert muon | adamw == {id(value) for value in model.parameters()}
    assert not muon & adamw
    assert [group["params"] for group in optimizer.param_groups] == [
        group["params"] for group in (*optimizer.muon.param_groups, *optimizer.adamw.param_groups)
    ]


def test_a_step_equals_muon_and_adamw_run_apart() -> None:
    model = _model()
    apart = copy.deepcopy(model)
    optimizer = make_optimizer(model, _config("muon"))
    matrices = _matrices(apart)
    muon = torch.optim.Muon(
        [value for value in apart.parameters() if id(value) in matrices],
        lr=1e-3,
        weight_decay=0.1,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        [value for value in apart.parameters() if id(value) not in matrices],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    for seed in range(3):
        _gradients(model, seed)
        _gradients(apart, seed)
        optimizer.step()
        muon.step()
        adamw.step()

    for (name, value), other in zip(model.named_parameters(), apart.parameters(), strict=True):
        assert torch.equal(value, other), name


def test_a_restored_optimizer_continues_like_the_original() -> None:
    model = _model()
    optimizer = make_optimizer(model, _config("muon"))
    _gradients(model, 0)
    optimizer.step()
    saved = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, saved)
    saved.seek(0)
    restored = torch.load(saved, weights_only=True)
    other = _model()
    other.load_state_dict(restored["model"])
    other_optimizer = make_optimizer(other, _config("muon"))
    assert isinstance(other_optimizer, MuonAdamW)
    other_optimizer.load_state_dict(restored["optimizer"])

    for seed in (1, 2):
        _gradients(model, seed)
        _gradients(other, seed)
        optimizer.step()
        other_optimizer.step()

    for (name, value), copied in zip(model.named_parameters(), other.parameters(), strict=True):
        assert torch.equal(value, copied), name
    assert other_optimizer.muon.state is other_optimizer.state
    assert other_optimizer.adamw.state is other_optimizer.state


def test_the_schedule_sets_the_rate_of_both_optimizers() -> None:
    model = _model()
    optimizer = make_optimizer(model, _config("muon"))
    assert isinstance(optimizer, MuonAdamW)
    scheduler = WarmupDecay(optimizer, peak=1e-3, total_steps=10, warmup_steps=2)

    _gradients(model, 0)
    optimizer.step()
    scheduler.step()

    rate = scheduler.get_last_lr()[0]
    assert rate != pytest.approx(1e-3)
    for group in (*optimizer.muon.param_groups, *optimizer.adamw.param_groups):
        assert group["lr"] == pytest.approx(rate)


def test_zero_grad_clears_both_optimizers() -> None:
    model = _model()
    optimizer = make_optimizer(model, _config("muon"))
    _gradients(model, 0)

    optimizer.zero_grad()

    assert all(value.grad is None for value in model.parameters())
