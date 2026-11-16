"""`PretrainingModule`: next-row pretraining with the baseline's optimizer and schedule, in Lightning.

AdamW over every parameter (or Muon for the layer matrices, `optimizer.py`), the warmup-decay
schedule stepped once per update, and global-norm clipping (set on the trainer). The baseline's float16 loss scaling is replaced by bfloat16
autocast, which needs no scaling and never skips an update. With `float8` set, torchao runs the
products of every linear layer whose sizes are multiples of 16 in FP8 (8-bit floats).

Cleaned from `core/graph_gpt_baseline/training/module.py`.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from eulercad.data.collate import IGNORED_LABEL, Batch
from eulercad.losses.next_row import LossOutput, next_row_loss
from eulercad.losses.sigreg import sigreg, standardize
from eulercad.model.model import GraphGPTModel
from eulercad.training.config import TrainingConfig
from eulercad.training.microbatch import (
    PAD_TO_MULTIPLE_OF,
    grouped_loss,
    grouped_loss_and_drawings,
)
from eulercad.training.optimizer import make_optimizer
from eulercad.training.schedule import WarmupDecay


def compiled_forward(model: GraphGPTModel, *, backend: str = "inductor") -> Callable[..., torch.Tensor]:
    """Compile `model` and hand back a plain callable.

    `torch.compile` returns a module that holds the original one. Assigning it to the
    Lightning module would register the same parameters twice and write every checkpoint key
    twice, so the closure keeps it out of the module tree; the uncompiled model is the one
    that is saved.

    Returns:
        A callable with the model's own signature.
    """
    compiled = torch.compile(model, backend=backend)

    def forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        return compiled(*args, **kwargs)

    return forward


def _with_states(
    model: GraphGPTModel, all_layers: bool = False
) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """A forward that also hands back the final-norm states the logits are read from, for SIGReg,
    or with `all_layers` every layer's states before them.

    Returns:
        `(input_ids, attention_mask) -> (logits, states)`, the states `(B, T, H)` or `(layers + 1, B, T, H)`.
    """

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if all_layers:
            states = model.encode_layers(input_ids, attention_mask)
            return model.head(states[-1]), states
        states = model.encode(input_ids, attention_mask)
        return model.head(states), states

    return forward


def sigreg_parts(drawings: torch.Tensor, global_step: int, *, standardized: bool) -> dict[str, torch.Tensor]:
    """SIGReg of each drawing's mean state, raw or standardized, averaged over the layers if there are several.

    Args:
        drawings: `(B, H)` final-norm means, or `(layers + 1, B, H)`: layer 1 to the last, then the final norm.
        global_step: seeds the random slices, the same for every layer.
        standardized: whether to read `standardize(drawings)` instead of the raw means.

    Returns:
        `sigreg`, the value trained on, and with several layers `sigreg_layer_1`, ...,
        `sigreg_final_norm`, each layer's own.
    """
    prepare = standardize if standardized else torch.Tensor.float
    if drawings.ndim == 2:
        return {"sigreg": sigreg(prepare(drawings), global_step)}
    names = [*(f"sigreg_layer_{index}" for index in range(1, len(drawings))), "sigreg_final_norm"]
    each = {name: sigreg(prepare(layer), global_step) for name, layer in zip(names, drawings, strict=True)}
    return {"sigreg": torch.stack(list(each.values())).mean(), **each}


def float8_fits(module: nn.Module, name: str) -> bool:
    """Whether torchao may convert a layer: FP8 products need both sizes divisible by 16.

    Returns:
        True for a linear layer whose input and output sizes are multiples of 16.
    """
    return isinstance(module, nn.Linear) and module.in_features % 16 == 0 and module.out_features % 16 == 0


class PretrainingModule(pl.LightningModule):
    """Train a `GraphGPTModel` on next-row batches.

    Attributes:
        model: the network; the only thing with weights.
        config: the optimization settings, the cell budget and compilation included.
        validation: the last validation result, `{"loss": float, "loss_parts": {field: float}}`,
            or None before the first one.
    """

    def __init__(self, model: GraphGPTModel, config: TrainingConfig) -> None:
        super().__init__()
        if config.float8 is not None:
            # In place and before compiling. The converted layers keep their weights and names,
            # so checkpoints load into a plain model.
            convert_to_float8_training(
                model, config=Float8LinearConfig.from_recipe_name(config.float8), module_filter_fn=float8_fits
            )
        self.model = model
        self.config = config
        backend = config.compile_backend
        # A bound method, not the module: assigning the module again would register it twice.
        self._forward = compiled_forward(model, backend=backend) if backend else model.__call__
        self._forward_states = None
        if config.sigreg_weight is not None:
            states = _with_states(model, all_layers=config.sigreg_layers == "all")
            self._forward_states = torch.compile(states, backend=backend) if backend else states
        self.validation: dict[str, Any] | None = None
        self._fields = model.tokenizer.fields
        self._part_sums = torch.zeros(len(self._fields))
        self._cells = torch.zeros(len(self._fields))

    def batch_loss(self, batch: Batch) -> LossOutput:
        """Score one batch, in length-sorted groups when a cell budget is set.

        Returns:
            The loss. Without dropout both ways give the same value; with dropout, grouping
            draws different masks (`microbatch.py`).
        """
        if self.config.cell_budget is None:
            return next_row_loss(self._forward(batch.input_ids, batch.attention_mask), batch.labels)
        # FP8 products need the rows of a group, drawings times rows, to be a multiple of 16.
        multiple = 16 if self.config.float8 else PAD_TO_MULTIPLE_OF
        return grouped_loss(self._forward, batch, cell_budget=self.config.cell_budget, pad_to_multiple_of=multiple)

    def _sigreg_loss(self, batch: Batch) -> LossOutput:
        """The next-row loss mixed with SIGReg on the mean state of each drawing (`sigreg_parts`).

        Returns:
            `(1 - w) * next_row + w * SIGReg` for `w = config.sigreg_weight`, next-row alone
            when `w` is 0; the parts add `next_row` and those of `sigreg_parts` to the fields.

        Raises:
            RuntimeError: If the module was built without a SIGReg weight.
        """
        weight = self.config.sigreg_weight
        if self._forward_states is None or weight is None:
            raise RuntimeError("the module was built without a SIGReg weight")
        if self.config.cell_budget is None:
            logits, states = self._forward_states(batch.input_ids, batch.attention_mask)
            output = next_row_loss(logits, batch.labels)
            real = batch.attention_mask[..., None].float()
            drawings = (states.float() * real).sum(-2) / real.sum(-2)
        else:
            multiple = 16 if self.config.float8 else PAD_TO_MULTIPLE_OF
            output, drawings = grouped_loss_and_drawings(
                self._forward_states, batch, cell_budget=self.config.cell_budget, pad_to_multiple_of=multiple
            )
        regularizer = sigreg_parts(drawings, self.global_step, standardized=self.config.sigreg_standardize)
        total = (1 - weight) * output.total + weight * regularizer["sigreg"] if weight else output.total
        return LossOutput(total, {**output.parts, "next_row": output.total, **regularizer})

    def training_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Compute the loss of one batch, with SIGReg when the config sets a weight.

        Returns:
            The optimized `loss`, detached `parts`, the real `rows`, and the `padded_rows`.
        """
        output = self.batch_loss(batch) if self.config.sigreg_weight is None else self._sigreg_loss(batch)
        return {
            "loss": output.total,
            "parts": {name: value.detach() for name, value in output.parts.items()},
            "rows": batch.attention_mask.sum(),
            "padded_rows": batch.attention_mask.numel(),
        }

    def on_validation_epoch_start(self) -> None:
        """Clear the sums of the previous validation."""
        self._part_sums = torch.zeros(len(self._fields), device=self.device)
        self._cells = torch.zeros(len(self._fields), device=self.device)

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Add one batch's per-field loss sums and labelled cell counts."""
        output = self.batch_loss(batch)
        cells = (batch.labels != IGNORED_LABEL).flatten(0, 1).sum(dim=0)
        self._part_sums += torch.stack([output.parts[name] for name in self._fields]).float() * cells
        self._cells += cells

    def on_validation_epoch_end(self) -> None:
        """Store the means over every labelled cell of the validation set, total and per field."""
        parts = (self._part_sums / self._cells).tolist()
        total = float(self._part_sums.sum() / self._cells.sum())
        self.validation = {"loss": total, "loss_parts": dict(zip(self._fields, parts, strict=True))}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """The config's optimizer (`optimizer.py`), with the per-update warmup-decay schedule.

        Returns:
            The optimizer and its step-interval scheduler.
        """
        config = self.config
        optimizer = make_optimizer(self.model, config)
        scheduler = WarmupDecay(
            optimizer,
            peak=config.learning_rate,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            schedule=config.schedule,
            final_ratio=config.final_learning_rate_ratio,
            decay_fraction=config.wsd_decay_fraction,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def transfer_batch_to_device(self, batch: Batch, device: torch.device, dataloader_idx: int) -> Batch:
        """Move every tensor of a batch.

        Returns:
            The same batch on `device`.
        """
        return Batch(**{field.name: getattr(batch, field.name).to(device) for field in fields(batch)})

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store what rebuilds the model, the settings, and the generator states behind dropout."""
        checkpoint["tokenizer"] = self.model.tokenizer.to_dict()
        checkpoint["model_config"] = self.model.config.to_dict()
        checkpoint["training_config"] = self.config.to_dict()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        checkpoint["rng_states"] = {"torch": torch.get_rng_state(), "cuda": cuda}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Refuse a checkpoint of another tokenizer or network, then restore the generator states.

        Raises:
            ValueError: If the checkpoint's tokenizer or model settings differ from this model's.
        """
        if checkpoint["tokenizer"] != self.model.tokenizer.to_dict():
            raise ValueError("the checkpoint was trained with another tokenizer")
        if checkpoint["model_config"] != self.model.config.to_dict():
            raise ValueError("the checkpoint holds another network configuration")
        states = checkpoint["rng_states"]
        torch.set_rng_state(states["torch"])
        if states["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])


def make_trainer(
    config: TrainingConfig,
    *,
    accelerator: str,
    callbacks: Sequence[pl.Callback] = (),
    max_steps: int | None = None,
    validate_every: int | None = None,
) -> pl.Trainer:
    """Build a one-device trainer with the config's precision and gradient clipping.

    Args:
        config: the optimization settings.
        accelerator: `gpu` or `cpu`.
        callbacks: typically a `RunRecorder`.
        max_steps: stop after this many updates; the default, `config.total_steps`, ends
            the run. A smaller value stops early without changing the schedule.
        validate_every: validate after every this many updates; None never validates.

    Returns:
        A trainer without Lightning's own logger, progress bar, or checkpointing.

    Raises:
        ValueError: If `max_steps` is not in `[1, config.total_steps]`.
    """
    steps = config.total_steps if max_steps is None else max_steps
    if not 1 <= steps <= config.total_steps:
        raise ValueError("max_steps must be between 1 and config.total_steps")
    validation: dict[str, Any] = (
        {"val_check_interval": validate_every, "check_val_every_n_epoch": None}
        if validate_every
        else {"limit_val_batches": 0}
    )
    return pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=config.precision,  # ty: ignore[invalid-argument-type]
        gradient_clip_val=config.max_gradient_norm,
        gradient_clip_algorithm="norm",
        callbacks=list(callbacks),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        max_steps=steps,
        **validation,
    )
