"""Measure the hidden states of a GraphGPT-based checkpoint layer by layer, and plot them.

The metrics are the seven of "Layer by Layer: Uncovering Hidden Representations in Language
Models" (Skean et al., ICML 2025), computed as its code computes them
(github.com/OFSkean/information_flow, `experiments/utils/metrics/`). A drawing plays the part
of a prompt and its token rows the part of the tokens. As in that code, every set of vectors is
centered and its rows scaled to unit length before a metric reads it: the rows of one drawing
for the per-drawing metrics, the mean rows of all drawings for the others.

| Metric | Reads | Function |
|---|---|---|
| prompt entropy | one drawing's rows | `spectral.matrix_based_entropy` / log(min(rows, width)), averaged |
| curvature | one drawing's rows, in walk order | `geometric.curvature`, averaged |
| dataset entropy | one mean row per drawing | `spectral.matrix_based_entropy` / log(min(drawings, width)) |
| effective rank | one mean row per drawing | `spectral.rankme` (Roy and Vetterli) |
| InfoNCE | the mean rows of two views | `relational.infonce`, temperature 0.1 as in the paper's code |
| LiDAR | the mean rows of two views | `spectral.lidar`, ridge 1e-4 as in the paper's code |
| DiME | the mean rows of two views | `relational.joint_dime`, ten re-pairings |

A view of a drawing is another Euler walk of it, with node numbers drawn again: the drawing is
the same, its serialization is not. The paper's views are prompts with typos. The views come
from the split's fixed draw under `--view-seeds`; the per-drawing metrics use `--seed`.

Where the paper's code departs from the paper, this follows the paper: curvature is the plain
angle between steps (the code takes the absolute cosine, which folds angles past 90 degrees);
LiDAR is the effective rank of the LiDAR paper (the code reports the entropy, its logarithm);
the entropies read the Gram matrix as it is (the code clamps its negative entries to zero).

Output: `outputs/<run>/layers_<checkpoint>_<split>/`, with `settings.json`, `metrics.json`
(one entry per layer), one `<metric>.png` per metric, and `overview.png`.

Usage:
    eulercad analyze runs/deep/checkpoints/model.safetensors --split val
"""

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eulercad.analysis.layers import hidden_states
from eulercad.analysis.metrics.geometric import curvature
from eulercad.analysis.metrics.relational import infonce, joint_dime
from eulercad.analysis.metrics.spectral import lidar, matrix_based_entropy, rankme
from eulercad.commands.pretrain import DUPLICATES, validation_positions
from eulercad.data.collate import collate
from eulercad.data.dataset import DrawingDataset
from eulercad.data.duplicates import DuplicateTable
from eulercad.model.model import GraphGPTModel
from eulercad.paths import REPOSITORY, get_absolute_path
from eulercad.training.artifacts import checkpoint_step, git_state, load_model

log = logging.getLogger(__name__)

# Deep, as `eulercad model download` saves it.
CHECKPOINT = "runs/deep/checkpoints/model.safetensors"
DATASET = "data/ABC/point_graph_dataset"
OUTPUTS = "outputs"
# The paper's code: InfoNCE temperature and LiDAR ridge.
TEMPERATURE = 0.1
LIDAR_RIDGE = 1e-4
# Each metric's plot title and y label.
METRICS = {
    "prompt_entropy": ("Prompt entropy", "entropy / log(min(rows, width))"),
    "curvature": ("Curvature", "mean angle between steps (rad)"),
    "dataset_entropy": ("Dataset entropy", "entropy / log(min(drawings, width))"),
    "effective_rank": ("Effective rank", "directions in use"),
    "infonce": ("InfoNCE (lower: views agree)", "loss (nats)"),
    "lidar": ("LiDAR", "effective rank of the LDA matrix"),
    "dime": ("DiME", "nats"),
}


def normalize_rows(z: np.ndarray) -> np.ndarray:
    """Center the rows and scale each to unit length, as the paper's code does before every metric.

    Returns:
        `(N, d)` float64.
    """
    z = z.astype(np.float64) - z.mean(axis=0)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def drawing_metrics(rows: np.ndarray) -> tuple[float, float]:
    """Prompt entropy and curvature of one drawing's rows, in walk order.

    Returns:
        The entropy divided by log(min(rows, width)), and the mean angle between consecutive
        steps; NaN for a curvature under three rows.
    """
    unit = normalize_rows(rows)
    entropy = matrix_based_entropy(unit) / math.log(min(unit.shape))
    angles = curvature(unit)
    return entropy, float(angles.mean()) if angles.size else math.nan


def layer_metrics(
    pooled: np.ndarray, view_a: np.ndarray, view_b: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """The metrics of one layer that read one mean row per drawing; the per-drawing ones are averaged elsewhere.

    Args:
        pooled: `(N, d)` mean rows of the drawings under `--seed`.
        view_a: `(N, d)` mean rows of the same drawings, first view.
        view_b: `(N, d)` mean rows of the same drawings, second view.
        rng: generator of DiME's re-pairings.

    Returns:
        Every metric of `METRICS` but the per-drawing ones, which are set to NaN.
    """
    unit = normalize_rows(pooled)
    views = np.concatenate([normalize_rows(view_a), normalize_rows(view_b)])
    drawings = np.tile(np.arange(len(view_a)), 2)
    return {
        "prompt_entropy": math.nan,
        "curvature": math.nan,
        "dataset_entropy": matrix_based_entropy(unit) / math.log(min(unit.shape)),
        "effective_rank": rankme(unit),
        "infonce": infonce(views, drawings, temperature=TEMPERATURE),
        "lidar": lidar(views, drawings, delta=LIDAR_RIDGE),
        "dime": joint_dime(views[: len(view_a)], views[len(view_a) :], rng=rng),
    }


def mean_rows(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The mean of each drawing's real rows.

    Returns:
        `(B, H)` in the dtype of `states`.
    """
    return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)


def _decorate(target: Any, depth: range, ticks: list[str], title: str, label: str, size: int) -> None:
    target.set_xticks(list(depth), ticks, fontsize=size, rotation=45)
    target.set_title(title)
    target.set_xlabel("depth (rows = before layer 1, norm = after the final norm)", fontsize=size)
    target.set_ylabel(label, fontsize=size)
    target.grid(alpha=0.3)


def plot_metrics(layers: Sequence[dict[str, Any]], output: Path) -> list[Path]:
    """Draw each metric against depth, one figure each, and all of them on one page.

    Returns:
        The files written.
    """
    ticks = [str(row["layer"]).removeprefix("layer_").replace("final_norm", "norm") for row in layers]
    depth = range(len(layers))
    written = []
    overview, grid = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for (metric, (title, label)), axes in zip(METRICS.items(), grid.flat, strict=False):
        values = [row[metric] for row in layers]
        figure, single = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for target, size in ((single, 9), (axes, 7)):
            target.plot(depth, values, marker="o", markersize=4, linewidth=2, color="#2a6fb0")
            _decorate(target, depth, ticks, title, label, size)
        path = output / f"{metric}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written.append(path)
    spare = grid.flat[len(METRICS) :]
    for axes in spare:
        axes.axis("off")
    path = output / "overview.png"
    overview.savefig(path, dpi=120)
    plt.close(overview)
    return [*written, path]


def _split(dataset: Path, model: GraphGPTModel, seed: int, duplicates: DuplicateTable) -> DrawingDataset:
    return DrawingDataset(dataset, model.tokenizer, seed=seed, resample_each_epoch=False, duplicates=duplicates)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure a checkpoint's hidden states layer by layer.")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=Path(CHECKPOINT))
    parser.add_argument("--dataset", type=Path, default=Path(DATASET), help="folder holding the split")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0, help="the run seed of the split's fixed draw")
    parser.add_argument("--view-seeds", type=int, nargs=2, default=[1, 2], help="the draws of the two views")
    parser.add_argument("--drawings", type=int, default=1000, help="evenly spaced drawings of the split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help=f"default: {OUTPUTS}/<run>/layers_<checkpoint>_<split>")
    return parser


def _measure(
    model: GraphGPTModel, draws: list[DrawingDataset], positions: list[int], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[int]]:
    """Run the drawings of `positions` under every draw and compute the seven metrics of every layer.

    Returns:
        One entry per layer, and the sketch identifiers in order.

    Raises:
        RuntimeError: If a view drew another drawing at some position than the main draw.
    """
    per_drawing: dict[str, list[tuple[float, float]]] = {}
    pooled: list[dict[str, list[torch.Tensor]]] = [{} for _ in draws]
    sketch_ids = []
    for start in range(0, len(positions), args.batch_size):
        chosen = positions[start : start + args.batch_size]
        for draw, means in zip(draws, pooled, strict=True):
            batch = collate([draw[position] for position in chosen])
            if draw is draws[0]:
                sketch_ids += batch.sketch_id.tolist()
            elif batch.sketch_id.tolist() != sketch_ids[start:]:
                raise RuntimeError("the views did not draw the same drawings at the same positions")
            input_ids, mask = batch.input_ids.to(args.device), batch.attention_mask.to(args.device)
            with torch.no_grad():
                states = hidden_states(model, input_ids, mask)
            lengths = mask.sum(1).tolist()
            for name, state in states.items():
                means.setdefault(name, []).append(mean_rows(state.float(), mask).cpu())
                if draw is draws[0]:
                    rows = state.float().cpu().numpy()
                    per_drawing.setdefault(name, []).extend(
                        drawing_metrics(rows[index, :length]) for index, length in enumerate(lengths)
                    )

    rng = np.random.default_rng(args.seed)
    layers: list[dict[str, Any]] = []
    for name, values in per_drawing.items():
        base, view_a, view_b = (torch.cat(means[name]).numpy() for means in pooled)
        entropy, angle = np.array(values).T
        layers.append(
            {"layer": name}
            | layer_metrics(base, view_a, view_b, rng)
            | {
                "prompt_entropy": float(np.nanmean(entropy)),
                "prompt_entropy_std": float(np.nanstd(entropy)),
                "curvature": float(np.nanmean(angle)),
                "curvature_std": float(np.nanstd(angle)),
            }
        )
    return layers, sketch_ids


def main(argv: list[str] | None = None) -> int:
    """Measure every layer, write the results, and plot them.

    Returns:
        The exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if len({args.seed, *args.view_seeds}) < 3:
        _parser().error("--seed and the two --view-seeds must differ")
    checkpoint = get_absolute_path(str(args.checkpoint))
    dataset = get_absolute_path(str(args.dataset))
    duplicates = get_absolute_path(DUPLICATES) / dataset.name / f"{args.split}.pt"
    default = Path(OUTPUTS) / checkpoint.parent.parent.name / f"layers_{checkpoint.stem}_{args.split}"
    output = get_absolute_path(str(args.output or default))

    model = load_model(checkpoint).to(args.device).eval()
    table = DuplicateTable.load(duplicates)
    draws = [_split(dataset / args.split, model, seed, table) for seed in (args.seed, *args.view_seeds)]
    positions = validation_positions(len(draws[0]), args.drawings)
    layers, sketch_ids = _measure(model, draws, positions, args)

    revision, dirty = git_state(REPOSITORY)
    settings = {
        "checkpoint": str(checkpoint),
        "step": checkpoint_step(checkpoint),
        "dataset": str(dataset / args.split),
        "duplicates": str(duplicates),
        "seed": args.seed,
        "view_seeds": args.view_seeds,
        "drawings": len(positions),
        "sketch_ids": sketch_ids,
        "infonce_temperature": TEMPERATURE,
        "lidar_ridge": LIDAR_RIDGE,
        "device": args.device,
        "revision": revision,
        "uncommitted_changes": dirty,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(layers, indent=2) + "\n", encoding="utf-8")
    plot_metrics(layers, output)
    columns = list(METRICS)
    sys.stdout.write(f"{'layer':<11}" + "".join(f" {column[:9]:>9}" for column in columns) + "\n")
    for row in layers:
        sys.stdout.write(f"{row['layer']:<11}" + "".join(f" {row[column]:9.3f}" for column in columns) + "\n")
    sys.stdout.write(f"written to {output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
