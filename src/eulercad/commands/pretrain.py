"""`pretrain`: next-row pretraining from the command line.

Parses the arguments, builds the tokenizer, the network, the datasets, and the loaders, and
hands them to `training/`. Run through `eulercad train`.
Cleaned from `core/graph_gpt_baseline/commands/cli.py`.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from eulercad.data.collate import collate
from eulercad.data.dataset import DrawingDataset
from eulercad.data.duplicates import DuplicateTable
from eulercad.data.stream import TrainingStream
from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.paths import REPOSITORY, get_absolute_path
from eulercad.tokenization.tokenizer import (
    TOKENIZERS,
    GraphTokenizer,
    tokenizer_from_dict,
)
from eulercad.training.artifacts import (
    KEPT_CHECKPOINTS,
    SOURCES,
    RunConflictError,
    RunRecorder,
    checkpoint_step,
    load_weights,
    open_run,
    run_record,
)
from eulercad.training.config import (
    FLOAT8_RECIPES,
    OPTIMIZERS,
    SIGREG_LAYERS,
    TrainingConfig,
)
from eulercad.training.module import PretrainingModule, make_trainer
from eulercad.training.schedule import (
    MIN_LEARNING_RATE_RATIO,
    SCHEDULES,
    WSD_DECAY_FRACTION,
)

log = logging.getLogger(__name__)

DATASET = "data/ABC/point_graph_dataset"
DUPLICATES = "data/ABC/duplicates"
RUNS = "runs"
# The baseline warms up over 1e8 of its 8e9 tokens.
WARMUP_DIVISOR = 80
SPLITS = ("train", "val")
SCALES = {"tiny": ModelConfig.tiny, "mini": ModelConfig.mini, "medium": ModelConfig.medium, "deep": ModelConfig.deep}


class CommandError(Exception):
    """A coordination problem reported as a message, without a traceback."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph_gpt_based_05_pretrain", description="Pretrain the GraphGPT-based model on next-row prediction."
    )
    parser.add_argument("--experiment", required=True, help="run folder name under --output-root")
    parser.add_argument(
        "--total-steps", type=int, required=True, help="updates of the whole run; the schedule ends here"
    )
    parser.add_argument("--warmup-steps", type=int, help=f"default: total steps // {WARMUP_DIVISOR}, as the baseline")
    parser.add_argument("--max-steps", type=int, help="stop after this many updates; the schedule keeps --total-steps")
    parser.add_argument("--dataset", type=Path, default=Path(DATASET), help="folder holding train/ and val/")
    parser.add_argument(
        "--duplicates", type=Path, help=f"folder holding train.pt and val.pt; default: {DUPLICATES}/<dataset name>"
    )
    parser.add_argument("--no-duplicates", action="store_true", help="run eulerize on access instead")
    parser.add_argument("--output-root", type=Path, default=Path(RUNS))
    parser.add_argument("--scale", choices=tuple(SCALES), default="mini")
    parser.add_argument("--qk-norm", action="store_true", help="normalize each head's queries and keys")
    parser.add_argument(
        "--geometry-features", action="store_true", help="add each row's coordinates, step, and visits as numbers"
    )
    parser.add_argument("--hidden-layers", type=int, help="layers of the model; default: the scale's")
    parser.add_argument("--hidden-size", type=int, help="width of the model; default: the scale's")
    parser.add_argument("--attention-heads", type=int, help="attention heads per layer; default: the scale's")
    parser.add_argument("--attention-dropout", type=float, help="dropout on the attention weights; default: 0.1")
    parser.add_argument(
        "--value-residual", action="store_true", help="every later layer also attends over the first layer's values"
    )
    parser.add_argument(
        "--zero-init-outputs", action="store_true", help="start the attention and MLP output projections at zero"
    )
    parser.add_argument(
        "--gated-attention", action="store_true", help="scale the attention output by a sigmoid gate per channel"
    )
    parser.add_argument(
        "--attention-residuals",
        action="store_true",
        help="each sublayer reads a learned softmax average of the earlier outputs instead of their sum",
    )
    parser.add_argument(
        "--attention-residual-block",
        type=int,
        default=1,
        help="with --attention-residuals: sum the outputs in groups of this many sublayers; 1 keeps every output",
    )
    parser.add_argument(
        "--layout",
        choices=tuple(TOKENIZERS),
        default="GraphTokenizer",
        help="the row layout: GraphTokenizer writes five fields, PlaceTokenizer four",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, help="the peak rate of the schedule; default: the config's")
    parser.add_argument(
        "--weight-decay", type=float, help="decoupled weight decay of every parameter; default: the config's"
    )
    parser.add_argument("--schedule", choices=SCHEDULES, default="linear", help="the decay after the warmup")
    parser.add_argument(
        "--final-learning-rate-ratio",
        type=float,
        default=MIN_LEARNING_RATE_RATIO,
        help="the rate at --total-steps, as a fraction of the peak",
    )
    parser.add_argument(
        "--wsd-decay-fraction",
        type=float,
        default=WSD_DECAY_FRACTION,
        help="the last share of the updates over which wsd and wsd_sqrt decay",
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        help="train on (1 - w) * next_row + w * SIGReg; 0 records SIGReg only; default: no SIGReg",
    )
    parser.add_argument(
        "--sigreg-layers",
        choices=SIGREG_LAYERS,
        default="final",
        help="all: the mean SIGReg of every layer's output and the final norm",
    )
    parser.add_argument(
        "--sigreg-raw", action="store_true", help="read SIGReg on the raw drawing means, not standardized ones"
    )
    parser.add_argument(
        "--init-from", type=Path, help="start from this checkpoint's weights; a fresh run, not a resume"
    )
    parser.add_argument(
        "--cuts",
        type=int,
        nargs="+",
        help="score only the last rows of each walk, one of these drawn per sample; default scores all of them",
    )
    parser.add_argument("--batch-size", type=int, default=160, help="drawings per update")
    parser.add_argument("--cell-budget", type=int, default=8192, help="rows times drawings per group; 0 runs one pass")
    parser.add_argument("--compile", action="store_true", help="compile the forward with the inductor backend")
    default_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default=default_accelerator)
    parser.add_argument(
        "--precision", choices=("bf16-mixed", "32-true"), help="default: bf16-mixed on gpu, 32-true on cpu"
    )
    parser.add_argument(
        "--float8", choices=FLOAT8_RECIPES, help="run the layer products in FP8 with this torchao recipe"
    )
    parser.add_argument(
        "--optimizer", choices=OPTIMIZERS, default="adamw", help="muon: Muon for the layer matrices, AdamW for the rest"
    )
    parser.add_argument("--workers", type=int, default=14, help="loader processes; batches do not change")
    parser.add_argument("--log-every", type=int, default=100, help="updates per training.jsonl line")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="updates per step checkpoint")
    parser.add_argument(
        "--keep-checkpoints", type=int, default=KEPT_CHECKPOINTS, help="step checkpoints kept, newest first"
    )
    parser.add_argument("--validate-every", type=int, default=2000, help="updates per validation; 0 never validates")
    parser.add_argument(
        "--validation-drawings", type=int, default=5000, help="evenly spaced validation drawings scored each time"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-config", action="store_true", help="print the resolved settings and exit")
    mode.add_argument("--dry-run", action="store_true", help="build data, model, and schedule; print; write nothing")
    mode.add_argument("--resume", action="store_true", help="continue the run from its last checkpoint")
    return parser


def _resolve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Check the arguments and build every setting the run uses.

    Returns:
        `tokenizer`, `model`, and `training` as their dictionaries, and `settings`: paths and
        the values a resume may change.
    """
    for name in ("log_every", "checkpoint_every", "validation_drawings", "keep_checkpoints"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.workers, args.validate_every, args.cell_budget, args.seed) < 0:
        parser.error("--workers, --validate-every, --cell-budget, and --seed must not be negative")
    total = args.total_steps
    warmup = total // WARMUP_DIVISOR if args.warmup_steps is None else args.warmup_steps
    rate = {} if args.learning_rate is None else {"learning_rate": args.learning_rate}
    if args.weight_decay is not None:
        rate["weight_decay"] = args.weight_decay
    try:
        training = TrainingConfig(
            total_steps=total,
            warmup_steps=warmup,
            batch_size=args.batch_size,
            precision=args.precision or ("bf16-mixed" if args.accelerator == "gpu" else "32-true"),
            seed=args.seed,
            cell_budget=args.cell_budget or None,
            compile_backend="inductor" if args.compile else None,
            float8=args.float8,
            optimizer=args.optimizer,
            schedule=args.schedule,
            final_learning_rate_ratio=args.final_learning_rate_ratio,
            wsd_decay_fraction=args.wsd_decay_fraction,
            sigreg_weight=args.sigreg_weight,
            sigreg_layers=args.sigreg_layers,
            sigreg_standardize=not args.sigreg_raw,
            **rate,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.max_steps is not None and not 1 <= args.max_steps <= total:
        parser.error("--max-steps must be between 1 and --total-steps")
    if args.cuts is not None and min(args.cuts) < 1:
        parser.error("--cuts must be positive")
    initial = None if args.init_from is None else str(get_absolute_path(str(args.init_from)))
    dataset = get_absolute_path(str(args.dataset))
    duplicates = None
    if not args.no_duplicates:
        duplicates = get_absolute_path(str(args.duplicates or Path(DUPLICATES) / dataset.name))
    settings = {
        "dataset": str(dataset),
        "duplicates": None if duplicates is None else str(duplicates),
        "run_directory": str(get_absolute_path(str(args.output_root)) / args.experiment),
        "accelerator": args.accelerator,
        "workers": args.workers,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "validate_every": args.validate_every,
        "validation_drawings": args.validation_drawings,
        "max_steps": args.max_steps,
        "keep_checkpoints": args.keep_checkpoints,
        "init_from": initial,
        "cuts": None if args.cuts is None else sorted(args.cuts),
    }
    overrides = {
        name: value
        for name, value in (
            ("hidden_layers", args.hidden_layers),
            ("hidden_size", args.hidden_size),
            ("attention_heads", args.attention_heads),
            ("attention_dropout", args.attention_dropout),
        )
        if value is not None
    }
    try:
        model = replace(
            SCALES[args.scale](),
            qk_norm=args.qk_norm,
            geometry_features=args.geometry_features,
            value_residual=args.value_residual,
            zero_init_outputs=args.zero_init_outputs,
            gated_attention=args.gated_attention,
            attention_residuals=args.attention_residuals,
            attention_residual_block=args.attention_residual_block,
            **overrides,
        )
    except ValueError as error:
        parser.error(str(error))
    return {
        "tokenizer": TOKENIZERS[args.layout]().to_dict(),
        "model": model.to_dict(),
        "training": training.to_dict(),
        "settings": settings,
    }


def validation_positions(drawings: int, count: int) -> list[int]:
    """Pick `count` evenly spaced drawings of a split, or all of them if it holds fewer.

    The split is stored in chunks, so even spacing samples every chunk; no random draw.

    Returns:
        Increasing dataset indices.
    """
    count = min(count, drawings)
    return [index * drawings // count for index in range(count)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _data(
    settings: dict[str, Any], tokenizer: GraphTokenizer, seed: int
) -> tuple[DrawingDataset, Subset, dict[str, Any]]:
    """Open both splits and their tables, and describe them for `run.json`.

    Returns:
        The training split, the validation drawings scored at each validation, and the data
        record: each split's path, drawing counts, and file hashes, the validation positions,
        and each table's path and hash.

    Raises:
        CommandError: If a table does not belong to its split.
    """
    root = Path(settings["dataset"])
    folder = None if settings["duplicates"] is None else Path(settings["duplicates"])
    datasets, record = {}, {}
    for split in SPLITS:
        table = None if folder is None else DuplicateTable.load(folder / f"{split}.pt")
        try:
            dataset = DrawingDataset(
                root / split,
                tokenizer,
                seed=seed,
                resample_each_epoch=split == "train",
                duplicates=table,
                # The validation loss stays the whole-walk one, so it compares with pretraining.
                cuts=settings["cuts"] if split == "train" else None,
            )
        except ValueError as error:
            raise CommandError(f"{split}: {error}") from error
        datasets[split] = dataset
        files = sorted((root / split).glob("*.pt"))
        record[split] = {
            "path": str(root / split),
            "drawings": len(dataset),
            "dropped_above_capacity": dataset.dropped,
            "files": {path.name: _sha256(path) for path in files},
        }
    positions = validation_positions(len(datasets["val"]), settings["validation_drawings"])
    record["val"]["validation_positions"] = positions
    record["duplicates"] = (
        None
        if folder is None
        else {
            split: {"path": str(folder / f"{split}.pt"), "sha256": _sha256(folder / f"{split}.pt")} for split in SPLITS
        }
    )
    return datasets["train"], Subset(datasets["val"], positions), record


def _emit(value: object) -> None:
    sys.stdout.write(json.dumps(value, indent=2, default=str) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pretraining command.

    Returns:
        Zero on success, one when coordination fails. Invalid arguments exit with two.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _parser()
    args = parser.parse_args(argv)
    resolved = _resolve(args, parser)
    if args.print_config:
        _emit(resolved)
        return 0
    try:
        return _pretrain(args, resolved)
    except (CommandError, RunConflictError, FileNotFoundError) as error:
        log.error("%s", error)
        return 1


def _pretrain(args: argparse.Namespace, resolved: dict[str, Any]) -> int:
    settings = resolved["settings"]
    tokenizer = tokenizer_from_dict(resolved["tokenizer"])
    model_config = ModelConfig.from_dict(resolved["model"])
    config = TrainingConfig.from_dict(resolved["training"])
    train, val, data = _data(settings, tokenizer, config.seed)
    torch.manual_seed(config.seed)
    model = GraphGPTModel(tokenizer, model_config)
    if settings["init_from"]:
        try:
            load_weights(model, Path(settings["init_from"]))
        except (ValueError, FileNotFoundError) as error:
            raise CommandError(f"--init-from: {error}") from error
    if args.dry_run:
        _emit({**resolved, "data": data, "parameters": sum(value.numel() for value in model.parameters())})
        return 0

    directory = Path(settings["run_directory"])
    record = run_record(
        tokenizer=tokenizer,
        model_config=model_config,
        config=config,
        data=data,
        package=REPOSITORY,
        sources=SOURCES,
        settings=settings,
    )
    checkpoint = open_run(directory, record, resume=args.resume)
    start = checkpoint_step(checkpoint) if checkpoint is not None else 0
    workers = settings["workers"]
    # Own generators: a loader without one draws its base seed from the global state, which drives dropout.
    loading: dict[str, Any] = {"num_workers": workers}
    if workers:
        loading |= {"persistent_workers": True, "multiprocessing_context": "forkserver"}
    loader = DataLoader(
        TrainingStream(train, config.batch_size, start=start),
        batch_size=None,
        generator=torch.Generator().manual_seed(config.seed),
        **loading,
    )
    val_loader = None
    if settings["validate_every"]:
        generator = torch.Generator().manual_seed(config.seed)
        val_loader = DataLoader(val, batch_size=config.batch_size, collate_fn=collate, generator=generator, **loading)
    recorder = RunRecorder(
        directory,
        log_every=settings["log_every"],
        checkpoint_every=settings["checkpoint_every"],
        keep_checkpoints=settings["keep_checkpoints"],
    )
    trainer = make_trainer(
        config,
        accelerator=settings["accelerator"],
        callbacks=[recorder],
        max_steps=settings["max_steps"],
        validate_every=settings["validate_every"] or None,
    )
    stop = settings["max_steps"] or config.total_steps
    log.info("training from update %d to %d of %d in %s", start, stop, config.total_steps, directory)
    trainer.fit(
        PretrainingModule(model, config), train_dataloaders=loader, val_dataloaders=val_loader, ckpt_path=checkpoint
    )
    return 0
