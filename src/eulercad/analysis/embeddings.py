"""Save every drawing's embedding at every layer of a GraphGPT-based checkpoint, one split at a time.

The embedding of a drawing at a layer is the mean of that layer's hidden states over the
drawing's real rows, as in retrieval (`analysis/retrieval.py`). Layers are
those of `hidden_states`: the input rows, each transformer layer, and the final norm. Each
drawing is read once, with the walk of the split's fixed draw (seed 0), causally, as it was
pretrained. Vectors are stored raw, not centred: retrieval centres them when it scores.
`--init scratch` embeds the checkpoint's network with fresh weights instead, a baseline in which
pretraining adds nothing. A later run into the same `--output` keeps the splits already written.

Layout under `--output`:

    settings.json                 checkpoint, step, revision, splits, layers, type of every file
    <split>/<layer>.npy           (drawings, width), float16 when every value fits, else float32
    <split>/index.npz             position, sketch_id, part_id, rows: one entry per file row
    <split>/row_<layer>.npy       with --row-seed: one real row per drawing, its place in `row` of the index

Part ids restart in every split, so a part is (split, part_id). A file is written in float32,
then rewritten in float16 if its largest value fits (65,504); the largest value of each layer is
in settings.json.

Usage:

    eulercad embed runs/deep/checkpoints/model.safetensors --output outputs/deep/embeddings
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from eulercad.analysis.layers import hidden_states, layer_names
from eulercad.analysis.loading import INITIALIZATIONS, PRECISIONS, backbone, precision_autocast
from eulercad.commands.pretrain import DUPLICATES
from eulercad.data.collate import collate
from eulercad.data.dataset import DrawingDataset
from eulercad.data.duplicates import DuplicateTable
from eulercad.model.model import GraphGPTModel
from eulercad.paths import REPOSITORY, get_absolute_path
from eulercad.training.artifacts import checkpoint_step, git_state, load_model

log = logging.getLogger(__name__)

DATASET = "data/ABC/point_graph_dataset"
SPLITS = ("train", "val", "test")
INDEX = "index.npz"
SETTINGS = "settings.json"
HALF_MAX = float(np.finfo(np.float16).max)


def half_if_safe(path: Path, limit: float = HALF_MAX) -> str:
    """Rewrite a float32 `.npy` file in float16 if every value's size is under `limit`.

    Returns:
        The type the file holds afterwards: `float16` or `float32`.
    """
    values = np.load(path, mmap_mode="r")
    if float(np.abs(values).max()) >= limit:
        return "float32"
    half = values.astype(np.float16)
    del values
    np.save(path, half)
    return "float16"


def embed_split(
    model: GraphGPTModel,
    dataset: DrawingDataset,
    output: Path,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    precision: str,
    row_seed: int | None = None,
) -> dict[str, Any]:
    """Write every kept drawing's per-layer embedding of one split, and its index.

    With `row_seed`, also write `row_<layer>.npy`: the hidden state of one real row per drawing, picked at random in
    loader order, so every checkpoint embedded with the same seed gets the same rows. The index then stores the pick
    as `row`, counted over the drawing's real rows.

    Returns:
        The number of drawings and, per layer, the stored type and the largest value's size.

    Raises:
        RuntimeError: If the loader gave another number of drawings, or another order, than the
            split stores.
    """
    output.mkdir(parents=True, exist_ok=True)
    names = layer_names(model)
    count, width = len(dataset), model.config.hidden_size
    files = {
        name: open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float32, shape=(count, width)) for name in names
    }
    sketch_id, rows = np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)
    if row_seed is not None:
        files |= {
            f"row_{name}": open_memmap(output / f"row_{name}.npy", mode="w+", dtype=np.float32, shape=(count, width))
            for name in names
        }
        picks, generator = np.empty(count, dtype=np.int64), np.random.default_rng(row_seed)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate)
    start = 0
    for batch in loader:
        ids, mask = batch.input_ids.to(device), batch.attention_mask.to(device)
        with torch.no_grad(), precision_autocast(device, precision):
            states = hidden_states(model, ids, mask)
        real = mask[..., None].float()
        end = start + len(ids)
        for name, state in states.items():
            files[name][start:end] = ((state.float() * real).sum(1) / real.sum(1)).cpu().numpy()
        sketch_id[start:end] = batch.sketch_id.numpy()
        rows[start:end] = batch.attention_mask.sum(1).numpy()
        if row_seed is not None:
            picks[start:end] = generator.integers(0, rows[start:end])
            at = torch.stack([batch.attention_mask[i].nonzero()[picks[start + i], 0] for i in range(len(ids))])
            for name, state in states.items():
                files[f"row_{name}"][start:end] = state[torch.arange(len(ids)), at.to(device)].float().cpu().numpy()
        start = end
    if start != count:
        raise RuntimeError(f"embedded {start} drawings of {count}")
    layers = {}
    for name, values in files.items():
        values.flush()
        largest = float(np.abs(values).max())
        del values
        layers[name] = {"dtype": half_if_safe(output / f"{name}.npy"), "max_abs": largest}
    position = dataset.kept.numpy()
    stored = torch.cat([chunk["sketch_id"] for chunk in dataset.source.chunks]).numpy()[position]
    if not np.array_equal(stored, sketch_id):
        raise RuntimeError("the loader's drawings are not the split's stored order")
    chunks = dataset.source.chunks
    part = torch.cat([chunk["part_id"] for chunk in chunks]).numpy()[position] if "part_id" in chunks[0] else None
    np.savez(output / INDEX, position=position, sketch_id=sketch_id, rows=rows,
             **({"part_id": part} if part is not None else {}),
             **({"row": picks} if row_seed is not None else {}))  # fmt: skip
    return {"drawings": count, "layers": layers}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save every drawing's per-layer embedding of a checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="folder that receives one subfolder per split")
    parser.add_argument("--dataset", type=Path, default=Path(DATASET), help="folder holding the splits")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--seed", type=int, default=0, help="the run seed of each split's fixed draw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8, help="loader processes building the drawings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=PRECISIONS, help="default: bf16-mixed on cuda, 32-true elsewhere")
    parser.add_argument("--row-seed", type=int, help="also save one random real row per drawing, picked with this seed")
    parser.add_argument("--init", choices=INITIALIZATIONS, default="pretrained",
                        help="scratch: the same network with fresh weights, as the scratch repairs")  # fmt: skip
    parser.add_argument("--init-seed", type=int, default=0, help="the seed of the fresh weights; scratch only")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Embed every requested split and write the settings.

    Returns:
        The exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    checkpoint = get_absolute_path(str(args.checkpoint))
    dataset = get_absolute_path(str(args.dataset))
    output = get_absolute_path(str(args.output))
    precision = args.precision or ("bf16-mixed" if args.device.startswith("cuda") else "32-true")
    device = torch.device(args.device)
    model = load_model(checkpoint) if args.init == "pretrained" else backbone(checkpoint, "scratch", args.init_seed)
    model = model.to(device).eval()
    revision, dirty = git_state(REPOSITORY)
    earlier = json.loads((output / SETTINGS).read_text(encoding="utf-8")) if (output / SETTINGS).exists() else {}
    settings: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "step": checkpoint_step(checkpoint),
        "init": args.init,
        "init_seed": args.init_seed if args.init == "scratch" else None,
        "dataset": str(dataset),
        "seed": args.seed,
        "precision": precision,
        "layers": layer_names(model),
        "embedding": "mean of the layer's hidden states over the drawing's real rows, causal, not centred",
        "revision": revision,
        "uncommitted_changes": dirty,
        "splits": earlier.get("splits", {}),  # a later run adds its splits to those already written
    }
    for split in args.splits:
        table = get_absolute_path(DUPLICATES) / dataset.name / f"{split}.pt"
        draw = DrawingDataset(dataset / split, model.tokenizer, seed=args.seed, resample_each_epoch=False,
                              duplicates=DuplicateTable.load(table) if table.exists() else None)  # fmt: skip
        log.info("%s: embedding %d drawings", split, len(draw))
        settings["splits"][split] = embed_split(
            model, draw, output / split, batch_size=args.batch_size, workers=args.workers, device=device,
            precision=precision, row_seed=args.row_seed,
        )  # fmt: skip
        settings["splits"][split]["duplicates_table"] = str(table) if table.exists() else None
        (output / SETTINGS).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        log.info("%s done", split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
