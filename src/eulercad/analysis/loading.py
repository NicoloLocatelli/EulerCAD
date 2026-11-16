"""Build a checkpoint's network for the analyses, with its weights or freshly drawn, and the precision it runs in."""

from pathlib import Path

import torch

from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.tokenizer import tokenizer_from_dict
from eulercad.training.artifacts import read_checkpoint

INITIALIZATIONS = ("pretrained", "scratch")
PRECISIONS = ("32-true", "bf16-mixed")


def backbone(checkpoint: Path, initialization: str, seed: int) -> GraphGPTModel:
    """Build the checkpoint's network, with or without its weights.

    Both initializations read the checkpoint, because it names the tokenizer and the model
    configuration; only `pretrained` keeps what was in it.

    Returns:
        The network on the CPU.

    Raises:
        ValueError: If `initialization` is not one of `INITIALIZATIONS`.
    """
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {INITIALIZATIONS}")
    stored = read_checkpoint(checkpoint)
    tokenizer = tokenizer_from_dict(stored["tokenizer"])
    torch.manual_seed(seed)
    model = GraphGPTModel(tokenizer, ModelConfig.from_dict(stored["model_config"]))
    if initialization == "pretrained":
        prefix = "model."
        weights = {
            name.removeprefix(prefix): value for name, value in stored["state_dict"].items() if name.startswith(prefix)
        }
        model.load_state_dict(weights)
    return model


def precision_autocast(device: torch.device, precision: str) -> torch.autocast:
    """The autocast context of `precision`: bfloat16 for `bf16-mixed`, off for `32-true`.

    Returns:
        The context manager.
    """
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16-mixed")
