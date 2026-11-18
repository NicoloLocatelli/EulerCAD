"""The `eulercad` command: data, model, training and analysis, one subcommand each.

    eulercad data download                  the preprocessed point graphs, from Hugging Face
    eulercad data build-abc --chunks 0-4    or rebuild them from the raw ABC STEP files (needs the abc extra)
    eulercad model download [deep-sigreg]   Deep or its SIGReg fine-tune, from Hugging Face, into runs/<name>/checkpoints/
    eulercad model export <ckpt> <folder>   a training checkpoint in the published form
    eulercad train --preset deep            pretrain with the recipe of Deep; any other option is passed on
    eulercad train --preset deep-sigreg     fine-tune the downloaded Deep with SIGReg
    eulercad analyze <checkpoint>           Layer by Layer measures, embeddings and retrieval of one split
    eulercad embed | retrieval | measures   one step of `analyze` alone, with all its options

Every relative path resolves under `$EULERCAD_HOME`, or the current folder.
"""

import argparse
import importlib
import logging
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

# Two deprecation notices about library code, not ours, silenced before any library loads:
# torch_geometric 2.8 still calls `torch.jit.script`, and torchao (which transformers imports
# whenever it is installed) still registers its enums with torch's pytree.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
logging.getLogger("torch.utils._pytree").addFilter(
    lambda record: "is an Enum subclass and is now natively supported" not in record.getMessage()
)

from eulercad import hub  # noqa: E402  (after the filters above, on purpose)
from eulercad.paths import get_absolute_path  # noqa: E402

DATASET = "data/ABC/point_graph_dataset"
OUTPUTS = "outputs"


def _step(module: str) -> Callable[[list[str]], int]:
    """A step's `main`, imported only when it runs: the ABC steps need the Open CASCADE kernel.

    Returns:
        A function of the step's arguments that returns its exit status.
    """

    def run(argv: list[str]) -> int:
        return importlib.import_module(module).main(argv) or 0

    return run


STEPS: dict[str, Callable[[list[str]], int]] = {
    "pretrain": _step("eulercad.commands.pretrain"),
    "measures": _step("eulercad.analysis.measures"),
    "embed": _step("eulercad.analysis.embeddings"),
    "retrieval": _step("eulercad.analysis.retrieval"),
    "download": _step("eulercad.abc.download"),
    "views": _step("eulercad.abc.views"),
    "preprocess": _step("eulercad.abc.preprocess"),
    "split": _step("eulercad.abc.split"),
    "point_graphs": _step("eulercad.abc.point_graphs"),
    "duplicates": _step("eulercad.data.duplicate_tables"),
}

# The recipe of Deep (thesis, Chapter 6): width 384, 14 layers, gated attention, Muon, 208,200 updates
# of 160 drawings, WSD schedule with a square-root decay from the peak rate 1.2e-3.
DEEP = [
    "--scale", "deep", "--qk-norm", "--geometry-features", "--optimizer", "muon", "--attention-dropout", "0",
    "--value-residual", "--zero-init-outputs", "--gated-attention", "--dataset", DATASET, "--batch-size", "160",
    "--cell-budget", "4096", "--compile", "--accelerator", "gpu", "--validation-drawings", "5000",
    "--validate-every", "2000", "--log-every", "100", "--checkpoint-every", "2000", "--total-steps", "208200",
    "--warmup-steps", "1000", "--keep-checkpoints", "3", "--learning-rate", "1.2e-3", "--weight-decay", "0.1",
    "--schedule", "wsd_sqrt", "--final-learning-rate-ratio", "0",
]  # fmt: skip


def _changed(options: list[str], changes: dict[str, str]) -> list[str]:
    """Return `options` with the value after each option of `changes` replaced, and new options at the end.

    Returns:
        The new option list.
    """
    result = list(options)
    for option, value in changes.items():
        if option in result:
            result[result.index(option) + 1] = value
        else:
            result += [option, value]
    return result


# The SIGReg fine-tune of Deep (thesis, Section 6.6): 20,000 more updates from Deep's weights, peak rate 3e-4
# flat for half the run then linear to zero, loss (1 - 0.018) next-row + 0.018 SIGReg, standardized, on the
# drawing means of the 14 layers and the final norm. Needs `eulercad model download` first.
DEEP_SIGREG = _changed(
    DEEP,
    {
        "--total-steps": "20000",
        "--warmup-steps": "250",
        "--learning-rate": "3e-4",
        "--schedule": "wsd",
        "--wsd-decay-fraction": "0.5",
        "--init-from": "runs/deep/checkpoints/model.safetensors",
        "--sigreg-weight": "0.018",
        "--sigreg-layers": "all",
    },
)
PRESETS = {"deep": DEEP, "deep-sigreg": DEEP_SIGREG}
# The 20 projection directions of the thesis: four standard views and 16 random ones of seed 0.
VIEWS = ["--views", "iso,front,top,left,rand16", "--seed", "0"]
SPLITS = ("train", "val", "test")


def _train(args: argparse.Namespace, rest: list[str]) -> int:
    if args.preset is None:
        return STEPS["pretrain"](rest)
    named = "--experiment" in rest
    return STEPS["pretrain"]([*PRESETS[args.preset], *([] if named else ["--experiment", args.preset]), *rest])


def _analyze(args: argparse.Namespace, rest: list[str]) -> int:
    checkpoint = Path(args.checkpoint)
    status = STEPS["measures"]([args.checkpoint, "--split", args.split, *rest])
    if status or not args.retrieval:
        return status
    output = str(Path(OUTPUTS) / checkpoint.parent.parent.name / f"embeddings_{checkpoint.stem}")
    return STEPS["embed"]([args.checkpoint, "--output", output, "--splits", args.split]) or STEPS["retrieval"]([output])


def _build_abc(args: argparse.Namespace, rest: list[str]) -> int:
    steps = [
        ("download", ["--chunks", args.chunks, "--formats", "step"]),
        ("views", [*VIEWS, *rest]),
        ("preprocess", []),
        ("split", []),
        ("point_graphs", []),
        *(("duplicates", ["--dataset", f"{DATASET}/{split}"]) for split in SPLITS),
    ]
    for name, argv in steps:
        status = STEPS[name](argv)
        if status:
            return status
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eulercad", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="get the point-graph dataset").add_subparsers(dest="action", required=True)
    data.add_parser("download", help="download the preprocessed dataset from Hugging Face into data/ABC/")
    build = data.add_parser("build-abc", help="rebuild the dataset from raw ABC; other options go to the views step")
    build.add_argument("--chunks", required=True, help="ABC archives of 10,000 objects; the thesis used 0-4")

    model = commands.add_parser("model", help="get or export a model").add_subparsers(dest="action", required=True)
    download = model.add_parser("download", help="download a published model into runs/<name>/checkpoints/")
    download.add_argument("name", nargs="?", default="deep", choices=sorted(hub.MODELS))
    export = model.add_parser("export", help="write a training checkpoint as safetensors and config.json")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("folder", type=Path)

    # No help of its own: `eulercad train --help` reaches the trainer, which lists every option.
    train = commands.add_parser("train", help="pretrain; options other than --preset go to the trainer", add_help=False)
    train.add_argument("--preset", choices=sorted(PRESETS), help="start from the recipe of a published model")

    analyze = commands.add_parser("analyze", help="measure a checkpoint layer by layer; other options go to measures")
    analyze.add_argument("checkpoint", help="a .ckpt of a run, or a downloaded model.safetensors")
    analyze.add_argument("--split", default="val", choices=SPLITS)
    analyze.add_argument("--retrieval", action=argparse.BooleanOptionalAction, default=True,
                         help="also embed the whole split and score retrieval per layer")  # fmt: skip

    for name in ("measures", "embed", "retrieval"):
        commands.add_parser(name, help=f"run the {name} step alone; see `eulercad {name} -- --help`", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand.

    Returns:
        The exit status.
    """
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("measures", "embed", "retrieval"):
        return STEPS[argv[0]](argv[1:])
    args, rest = _parser().parse_known_args(argv)
    if args.command == "data" and args.action == "download":
        hub.download_dataset(get_absolute_path("data/ABC"))
        return 0
    if args.command == "data":
        return _build_abc(args, rest)
    if args.command == "model" and args.action == "download":
        hub.download_model(get_absolute_path(f"runs/{args.name}/checkpoints"), hub.MODELS[args.name])
        return 0
    if args.command == "model":
        hub.export_checkpoint(get_absolute_path(args.checkpoint), get_absolute_path(args.folder))
        return 0
    if args.command == "train":
        return _train(args, rest)
    return _analyze(args, rest)
