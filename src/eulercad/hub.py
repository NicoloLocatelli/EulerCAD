"""The published form of a model, and its download from Hugging Face.

A training checkpoint (`.ckpt`) also holds the optimizer state, which only resuming needs.
`export_checkpoint` keeps what using the model needs: the weights, in safetensors, and
`config.json` with the tokenizer, the model configuration, and the update it was saved at.
`training/artifacts.py` reads both forms. `download_model` and `download_dataset` fetch the published
model and data from Hugging Face.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_file

from eulercad.training.artifacts import read_checkpoint

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
PREFIX = "model."


def export_checkpoint(checkpoint: Path, folder: Path) -> Path:
    """Write the weights of a training checkpoint and its `config.json` into `folder`.

    Returns:
        The path of the weights.
    """
    stored = read_checkpoint(checkpoint)
    weights = {
        name.removeprefix(PREFIX): value.contiguous()
        for name, value in stored["state_dict"].items()
        if name.startswith(PREFIX)
    }
    folder.mkdir(parents=True, exist_ok=True)
    save_file(weights, folder / WEIGHTS_FILE)
    config = {key: stored[key] for key in ("tokenizer", "model_config", "global_step")}
    (folder / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return folder / WEIGHTS_FILE


MODEL_REPOSITORY = "NicoloLocatelli/EulerCAD-Deep"
MODELS = {"deep": MODEL_REPOSITORY, "deep-sigreg": "NicoloLocatelli/EulerCAD-Deep-SIGReg"}
DATASET_REPOSITORY = "NicoloLocatelli/EulerCAD-ABC-Preprocessed"


def download_model(folder: Path, repository: str = MODEL_REPOSITORY) -> Path:
    """Download a published model's weights and `config.json` into `folder`; Deep by default.

    Returns:
        The path of the weights.
    """
    for filename in (WEIGHTS_FILE, CONFIG_FILE):
        hf_hub_download(repo_id=repository, filename=filename, local_dir=folder)
    return folder / WEIGHTS_FILE


def download_dataset(folder: Path) -> Path:
    """Download the preprocessed point graphs and their duplicate tables into `folder`.

    Returns:
        `folder`, which then holds `point_graph_dataset/` and `duplicates/`.
    """
    snapshot_download(repo_id=DATASET_REPOSITORY, repo_type="dataset", local_dir=folder)
    return folder
