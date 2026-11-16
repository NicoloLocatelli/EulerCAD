"""Run records, checkpoints, and measurements of a pretraining run.

A run lives in one folder under `runs/`: `run.json` (what the run is),
`training.jsonl` (one line per record interval, validation, start, resume, and end), and
`checkpoints/`. Cleaned from `core/graph_gpt_baseline/training/artifacts.py`.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess  # noqa: S404  # nosec B404
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import pynvml
import torch
from safetensors.torch import load_file as load_safetensors

from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.tokenizer import GraphTokenizer, tokenizer_from_dict
from eulercad.training.config import TrainingConfig

RUN_FILE = "run.json"
LOG_FILE = "training.jsonl"
CHECKPOINTS = "checkpoints"
LAST_CHECKPOINT = "last.ckpt"
BEST_CHECKPOINT = "best.ckpt"
# The baseline keeps the last five step checkpoints.
KEPT_CHECKPOINTS = 5
PROBE_DRAWINGS = 8
# The code a run executes, relative to the repository: the package and the dependency pins.
# `analysis/` is excluded because it only ever reads a finished checkpoint: editing an analysis
# cannot change what a pretraining run computes, and a long run must stay resumable meanwhile.
SOURCES = (
    "src/eulercad",
    ":(exclude)src/eulercad/analysis",
    # Prose cannot change what a run computes, so editing a CONTEXT.md while a run is going
    # must not cost it its `--resume`. The git revision still records what the docs said.
    ":(exclude)src/eulercad/**/*.md",
    ":(exclude)src/eulercad/*.md",
    "pyproject.toml",
    "uv.lock",
)
PACKAGES = ("torch", "lightning", "transformers", "torch-geometric", "nvidia-ml-py")


class RunConflictError(Exception):
    """The run folder does not allow the requested start or resume."""


def digest(value: object) -> str:
    """Return the SHA-256 of `value` written as canonical JSON."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    """Run one git command in `repository`.

    Returns:
        Its standard output.

    Raises:
        RunConflictError: If git is not installed.
    """
    git = shutil.which("git")
    if git is None:
        raise RunConflictError("git is needed to record the source")
    command = [git, "-C", str(repository), *arguments]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603  # nosec B603


def git_state(repository: Path) -> tuple[str | None, bool | None]:
    """Return the repository revision and whether it has uncommitted changes; needs git.

    Returns:
        No revision before the first commit, and neither outside a git repository.
    """
    try:
        changed = bool(_git(repository, "status", "--porcelain").strip())
    except subprocess.CalledProcessError:
        return None, None
    try:
        revision = _git(repository, "rev-parse", "--verify", "--quiet", "HEAD").strip()
    except subprocess.CalledProcessError:
        return None, changed
    return revision, changed


def source_digest(package: Path, paths: Sequence[str] = SOURCES) -> str:
    """Return the SHA-256 of the code under `paths`, as it is on disk.

    The files are the ones git tracks or would track (ignored files left out), committed or
    not, so committing unchanged code keeps the digest and any edit changes it.

    Returns:
        The digest of every file's path and content. Needs git to list the files.
    """
    listed = _git(package, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *paths)
    digest = hashlib.sha256()
    for name in sorted(set(filter(None, listed.split("\0")))):
        path = package / name
        if path.is_file():  # a tracked file deleted from disk is left out
            digest.update(f"{name}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    """Return the Python, package, and graphics card versions of this process."""
    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": gpu.name if gpu is not None else None,
        "gpu_memory_bytes": gpu.total_memory if gpu is not None else None,
    }


def run_record(
    *,
    tokenizer: GraphTokenizer,
    model_config: ModelConfig,
    config: TrainingConfig,
    data: Mapping[str, Any],
    package: Path,
    sources: Sequence[str] = SOURCES,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a run for `run.json`.

    `tokenizer`, `model`, `training`, `data`, and the source code decide the result, so each
    gets a digest that a resume must match. The source digest covers the files under
    `sources` in `package` as they are on disk (`source_digest`); the git revision is recorded
    beside it. `settings` holds what may change on a resume (workers, record intervals).

    Returns:
        The record.
    """
    revision, dirty = git_state(package)
    fixed = {
        "tokenizer": tokenizer.to_dict(),
        "model": model_config.to_dict(),
        "training": config.to_dict(),
        "data": dict(data),
    }
    code = source_digest(package, sources)
    return {
        **fixed,
        "settings": dict(settings or {}),
        "digests": {name: digest(value) for name, value in fixed.items()} | {"source": code},
        "source": {"revision": revision, "uncommitted_changes": dirty, "digest": code, "paths": list(sources)},
        "environment": environment(),
        "created": time.time(),
    }


def open_run(directory: Path, record: Mapping[str, Any], *, resume: bool) -> Path | None:
    """Write `run.json` for a new run, or check a resume against it.

    A new run needs a missing or empty folder, so no earlier log or checkpoint can mix in.

    Returns:
        The checkpoint to resume from, or None for a new run.

    Raises:
        RunConflictError: If a new run's folder holds anything, or a resume has no run, no
            checkpoint, or different digests.
    """
    run_file = directory / RUN_FILE
    if not resume:
        if directory.exists() and any(directory.iterdir()):
            raise RunConflictError(f"{directory} is not empty; resume it or choose another experiment")
        directory.mkdir(parents=True, exist_ok=True)
        run_file.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        return None
    checkpoint = directory / CHECKPOINTS / LAST_CHECKPOINT
    if not run_file.exists() or not checkpoint.exists():
        raise RunConflictError(f"no run to resume in {directory}")
    saved = json.loads(run_file.read_text(encoding="utf-8"))["digests"]
    changed = sorted(
        name for name in saved.keys() | record["digests"].keys() if saved.get(name) != record["digests"].get(name)
    )
    if changed:
        raise RunConflictError(f"refusing to resume {directory}: {', '.join(changed)} changed since {RUN_FILE}")
    return checkpoint


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read a training checkpoint (`.ckpt`) or published weights (`.safetensors` beside `config.json`).

    Returns:
        `tokenizer`, `model_config`, `global_step`, and `state_dict` with the `model.` prefix of
        a training checkpoint, whichever the format.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        return torch.load(path, map_location="cpu", weights_only=True)
    config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
    weights = load_safetensors(path)
    return config | {"state_dict": {f"model.{name}": value for name, value in weights.items()}}


def checkpoint_step(path: Path) -> int:
    """Return the number of updates completed when a checkpoint was saved."""
    return int(read_checkpoint(path)["global_step"])


def load_model(path: Path) -> GraphGPTModel:
    """Rebuild the tokenizer and the model stored in a checkpoint, with its weights.

    Returns:
        The model on the CPU, in training mode.
    """
    checkpoint = read_checkpoint(path)
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    model = GraphGPTModel(tokenizer, ModelConfig.from_dict(checkpoint["model_config"]))
    prefix = "model."
    model.load_state_dict(
        {
            name.removeprefix(prefix): value
            for name, value in checkpoint["state_dict"].items()
            if name.startswith(prefix)
        }
    )
    return model


def load_weights(model: GraphGPTModel, path: Path) -> None:
    """Copy a checkpoint's weights into `model`, which must be the same network.

    Raises:
        ValueError: If the checkpoint holds another model or another tokenizer.
    """
    checkpoint = read_checkpoint(path)
    stored = ModelConfig.from_dict(checkpoint["model_config"])
    if stored != model.config:
        raise ValueError(f"the checkpoint holds {stored}, not {model.config}")
    if checkpoint["tokenizer"] != model.tokenizer.to_dict():
        raise ValueError("the checkpoint was written with another tokenizer")
    prefix = "model."
    model.load_state_dict(
        {
            name.removeprefix(prefix): value
            for name, value in checkpoint["state_dict"].items()
            if name.startswith(prefix)
        }
    )


def _block(name: str) -> str:
    """Name the block of a parameter: `embedding`, `merge`, `backbone.layers.<n>`, `backbone.norm`, `head`.

    Returns:
        The block name.
    """
    parts = name.split(".")
    if parts[:2] == ["backbone", "layers"]:
        return ".".join(parts[:3])
    return ".".join(parts[:2]) if parts[0] == "backbone" else parts[0]


class RunRecorder(pl.Callback):
    """Append measurements to `training.jsonl` and keep the checkpoints.

    Every `log_every` updates, one line covers the updates since the previous line: the loss
    and each of its parts at the last update, the learning rate that update used, the tokens
    seen so far, rows and samples per second, the padding share, and where the time went.
    `tokens` counts every real input row since the run began, the baseline's token unit: one
    position of the sequence, which stacks the five token identifiers of a row. It and the best
    validation loss are saved in every checkpoint and restored on a resume. `active_seconds` spans the
    training hooks of each batch, `data_wait_seconds` the gaps between batches; validation
    time is left out of both and recorded on its own line.

    Each log line also holds the gradient norm of every parameter, read before clipping, their
    total (`gradient_norm`, the value clipping compares with `max_gradient_norm`), the share
    clipping keeps (`clip_ratio`), the norms of each block (`gradient_norms_by_block`: the
    embedding, the field merge, each decoder layer, the final norm, the head), the largest
    attention score of each layer on the first `PROBE_DRAWINGS` drawings of the batch, measured
    after the update in eval mode, so no dropout draws from the generator, and each layer's
    query and key weight records (`attention_weights`).

    Every `checkpoint_every` updates, and when training stops, a step checkpoint is written,
    `last.ckpt` points to it, and only the last `keep_checkpoints` are kept. A run whose
    checkpoints will be compared later, not just resumed, wants enough of them to span it. A validation at the same step
    runs after the checkpoint, so the checkpoint is written again to hold its result. After each validation,
    `best.ckpt` is replaced when the validation loss is the lowest so far.
    """

    def __init__(
        self, directory: Path, *, log_every: int, checkpoint_every: int, keep_checkpoints: int = KEPT_CHECKPOINTS
    ) -> None:
        self.directory = directory
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.keep_checkpoints = keep_checkpoints
        self.tokens = 0
        self.best: float | None = None
        self._gpu: Any = None
        self._saved_step = -1
        self._rate = 0.0
        self._batch_start = self._validation_start = 0.0
        self._gradient_norms: dict[str, float] = {}
        self._clip: float | int | None = None
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self._interval_start = self._previous_end = now
        self._active = self._wait = 0.0
        self._rows = self._padded_rows = self._samples = 0

    def state_dict(self) -> dict[str, Any]:
        """Return what a resume needs: the tokens seen and the best validation loss."""
        return {"tokens": self.tokens, "best": self.best}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the tokens seen and the best validation loss of a checkpoint."""
        self.tokens = state_dict["tokens"]
        self.best = state_dict["best"]

    def _write(self, row: Mapping[str, Any]) -> None:
        with (self.directory / LOG_FILE).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Open the interval clock, the memory counter, and the utilization handle."""
        device = trainer.strategy.root_device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            pynvml.nvmlInit()
            uuid = torch.cuda.get_device_properties(device).uuid
            self._gpu = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        self._saved_step = trainer.global_step
        event = "resume" if trainer.global_step else "start"
        self._write(self._event(event, trainer))
        self._reset(time.monotonic())

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Close the data wait and note the rate this update uses."""
        now = time.monotonic()
        self._wait += now - self._previous_end
        self._batch_start = now
        self._rate = trainer.optimizers[0].param_groups[0]["lr"]

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer
    ) -> None:
        """Read every gradient norm of an update that will be logged; Lightning clips after this hook."""
        if (trainer.global_step + 1) % self.log_every:
            return
        named = [
            (name, parameter.grad) for name, parameter in pl_module.named_parameters() if parameter.grad is not None
        ]
        norms = torch.stack([torch.linalg.vector_norm(grad.detach().float()) for _, grad in named]).tolist()
        self._gradient_norms = {name.removeprefix("model."): norm for (name, _), norm in zip(named, norms, strict=True)}
        self._clip = trainer.gradient_clip_val

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Log and checkpoint on their step intervals."""
        self._active += time.monotonic() - self._batch_start
        rows = int(outputs["rows"])
        self._rows += rows
        self.tokens += rows
        self._padded_rows += int(outputs["padded_rows"])
        self._samples += int(batch.input_ids.shape[0])
        step = trainer.global_step
        if step % self.log_every == 0:
            self._write(self._measurements(trainer, outputs) | self._diagnostics(pl_module, batch))
        if step % self.checkpoint_every == 0:
            self._save(trainer)
        self._previous_end = time.monotonic()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start the validation clock."""
        self._validation_start = time.monotonic()

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Record the validation losses, keep the best weights, and leave the time out of the interval."""
        result = getattr(pl_module, "validation", None)
        if result is None or trainer.sanity_checking:
            return
        seconds = time.monotonic() - self._validation_start
        best = self.best is None or result["loss"] < self.best
        self._write(
            {"event": "validation", "step": trainer.global_step, "tokens": self.tokens, **result, "seconds": seconds}
            | {"best": best}
        )
        if best:
            self.best = result["loss"]
            (self.directory / CHECKPOINTS).mkdir(exist_ok=True)
            trainer.save_checkpoint(self.directory / CHECKPOINTS / BEST_CHECKPOINT)
        if trainer.global_step == self._saved_step:
            # Lightning validates after the batch hooks, so this step's checkpoint misses the result.
            self._save(trainer)
        elapsed = time.monotonic() - self._validation_start
        self._interval_start += elapsed
        self._previous_end += elapsed

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save the final state and close the log."""
        if trainer.global_step != self._saved_step:
            self._save(trainer)
        self._write(self._event("end", trainer))

    def _event(self, event: str, trainer: pl.Trainer) -> dict[str, Any]:
        return {
            "event": event,
            "step": trainer.global_step,
            "tokens": self.tokens,
            "best_validation_loss": self.best,
            "time": time.time(),
        }

    def _measurements(self, trainer: pl.Trainer, outputs: Any) -> dict[str, Any]:
        device = trainer.strategy.root_device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.monotonic()
        interval = now - self._interval_start
        row = {
            "step": trainer.global_step,
            "learning_rate": self._rate,
            "tokens": self.tokens,
            "loss": float(outputs["loss"]),
            "loss_parts": {name: float(value) for name, value in outputs["parts"].items()},
            "rows": self._rows,
            "padding_share": 1 - self._rows / self._padded_rows,
            "interval_seconds": interval,
            "active_seconds": self._active,
            "data_wait_seconds": self._wait,
            "samples_per_second": self._samples / interval,
            "rows_per_second": self._rows / interval,
            "gpu_utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu if self._gpu else None,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "time": time.time(),
        }
        self._reset(now)
        return row

    def _diagnostics(self, pl_module: pl.LightningModule, batch: Any) -> dict[str, Any]:
        norms, self._gradient_norms = self._gradient_norms, {}
        model = cast("GraphGPTModel", pl_module.model)
        training = model.training
        model.eval()
        with torch.no_grad():
            scores = model.largest_attention_scores(
                batch.input_ids[:PROBE_DRAWINGS], batch.attention_mask[:PROBE_DRAWINGS]
            )
        model.train(training)
        total = math.sqrt(sum(norm**2 for norm in norms.values())) if norms else None
        blocks: dict[str, float] = {}
        for name, norm in norms.items():
            blocks[_block(name)] = blocks.get(_block(name), 0.0) + norm**2
        clip = None
        if total is not None and self._clip is not None:
            clip = min(1.0, float(self._clip) / total) if total > 0 else 1.0
        return {
            "gradient_norm": total,
            "clip_ratio": clip,
            "gradient_norms_by_block": {name: math.sqrt(square) for name, square in blocks.items()},
            "gradient_norms": norms,
            "largest_attention_scores": scores.tolist(),
            "attention_weights": model.attention_weight_records(),
        }

    def _save(self, trainer: pl.Trainer) -> None:
        folder = self.directory / CHECKPOINTS
        folder.mkdir(exist_ok=True)
        path = folder / f"step_{trainer.global_step:09d}.ckpt"
        trainer.save_checkpoint(path)
        last = folder / LAST_CHECKPOINT
        last.unlink(missing_ok=True)
        last.symlink_to(path.name)
        for old in sorted(folder.glob("step_*.ckpt"))[: -self.keep_checkpoints]:
            old.unlink()
        self._saved_step = trainer.global_step
