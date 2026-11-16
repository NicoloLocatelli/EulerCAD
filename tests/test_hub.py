"""Tests for the published weight format (`eulercad/hub.py`) and its reading by `training/artifacts.py`."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from eulercad.analysis.loading import backbone
from eulercad.hub import CONFIG_FILE, WEIGHTS_FILE, export_checkpoint
from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.training.artifacts import checkpoint_step, load_model, load_weights

SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
CONFIG = replace(ModelConfig.tiny(), attention_dropout=0.0, gated_attention=True, qk_norm=True)


def _checkpoint(path: Path, step: int = 7) -> GraphGPTModel:
    """Save a training checkpoint shaped like the ones Lightning writes, with optimizer state.

    Returns:
        The saved model.
    """
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, CONFIG).eval()
    stored = {
        "state_dict": {f"model.{name}": value for name, value in model.state_dict().items()},
        "tokenizer": SMALL.to_dict(),
        "model_config": CONFIG.to_dict(),
        "global_step": step,
        "optimizer_states": [{"state": {0: {"exp_avg": torch.ones(3)}}}],
    }
    torch.save(stored, path)
    return model


def _outputs(model: GraphGPTModel) -> torch.Tensor:
    input_ids = torch.zeros((1, 5, SMALL.row_width), dtype=torch.int64)
    mask = torch.ones((1, 5), dtype=torch.bool)
    with torch.no_grad():
        return model.encode(input_ids, mask)


def test_export_writes_the_weights_and_a_config_without_the_training_state(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "best.ckpt")
    weights = export_checkpoint(tmp_path / "best.ckpt", tmp_path / "deep")
    assert weights == tmp_path / "deep" / WEIGHTS_FILE
    config = json.loads((tmp_path / "deep" / CONFIG_FILE).read_text())
    assert config["model_config"] == CONFIG.to_dict()
    assert config["tokenizer"] == SMALL.to_dict()
    assert config["global_step"] == 7
    assert "optimizer_states" not in config


def test_an_exported_model_computes_exactly_what_the_checkpoint_computes(tmp_path: Path) -> None:
    original = _checkpoint(tmp_path / "best.ckpt")
    weights = export_checkpoint(tmp_path / "best.ckpt", tmp_path / "deep")
    assert torch.equal(_outputs(load_model(weights).eval()), _outputs(original))
    assert torch.equal(_outputs(backbone(weights, "pretrained", 0).eval()), _outputs(original))
    assert checkpoint_step(weights) == 7


def test_an_exported_model_can_start_a_fine_tune(tmp_path: Path) -> None:
    original = _checkpoint(tmp_path / "best.ckpt")
    weights = export_checkpoint(tmp_path / "best.ckpt", tmp_path / "deep")
    torch.manual_seed(1)
    fresh = GraphGPTModel(SMALL, CONFIG).eval()
    load_weights(fresh, weights)
    assert torch.equal(_outputs(fresh), _outputs(original))
