"""Retrieval of the other views of the same 3D object, per layer, from the files of `eulercad embed`.

Every drawing of a split is a query against all the others of the same split. The right answers
are the drawings of the same 3D object (`part_id`). The vectors are centred over the split, and
`retrieval_metrics` ranks by cosine similarity and reports mAP and recall@k. A random ranking
scores `chance_map`.

Output: `retrieval.json` in the embeddings folder, by split: the number of drawings, the chance
mAP, and the scores of every layer.

Usage:
    eulercad retrieval outputs/deep/embeddings
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from eulercad.analysis.metrics.retrieval import retrieval_metrics
from eulercad.paths import get_absolute_path

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
OUTPUT = "retrieval.json"


def centred(z: np.ndarray) -> np.ndarray:
    """Remove the mean vector, which every drawing shares.

    Returns:
        `z` minus its mean over rows.
    """
    return z - z.mean(0, keepdims=True)


def chance_map(labels: np.ndarray) -> float:
    """The mAP of a random ranking: each query's share of right drawings among the others, averaged over the queries.

    Returns:
        The mean over drawings whose part has another drawing of (part size - 1) / (drawings - 1).
    """
    _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    sizes = counts[codes]
    return float(np.mean((sizes[sizes >= 2] - 1) / (len(labels) - 1)))


def split_scores(folder: Path, layers: list[str], device: str | None = None) -> dict[str, dict[str, float]]:
    """Score retrieval at every layer of one split folder, every drawing against all the others of the split.

    A layer that is the same for every drawing is left out: centred it is zero up to the
    rounding of the mean, and `retrieval_metrics` breaks the ties in favour of the right drawings.

    Returns:
        Per layer, the scores of `retrieval_metrics` on the vectors centred over the split.
    """
    labels = np.load(folder / "index.npz")["part_id"]
    scores = {}
    for layer in layers:
        z = np.load(folder / f"{layer}.npy").astype(np.float32)
        if (z == z[0]).all():
            log.info("%s is the same for every drawing; skipped", layer)
            continue
        scores[layer] = retrieval_metrics(centred(z), labels, device=device)
    return scores


def main(argv: list[str] | None = None) -> int:
    """Score every split found in the embeddings folder and write `retrieval.json` there.

    Returns:
        The exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Score retrieval per layer from the files of `eulercad embed`.")
    parser.add_argument("embeddings", type=Path, help="the --output folder of `eulercad embed`")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    folder = get_absolute_path(args.embeddings)
    layers = json.loads((folder / "settings.json").read_text(encoding="utf-8"))["layers"]
    result = {}
    for split in SPLITS:
        if not (folder / split / "index.npz").exists():
            continue
        labels = np.load(folder / split / "index.npz")["part_id"]
        scores = split_scores(folder / split, layers, args.device)
        result[split] = {"drawings": len(labels), "chance_map": chance_map(labels), "scores": scores}
        best = max(scores, key=lambda layer: scores[layer]["map"])
        log.info("%s: best layer %s, mAP %.3f (chance %.4f)", split, best, scores[best]["map"], chance_map(labels))
    (folder / OUTPUT).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", folder / OUTPUT)
    return 0
