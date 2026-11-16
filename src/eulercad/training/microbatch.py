"""Compute one batch in length-sorted groups, so the model never runs on padding rows.

A batch is padded to its longest drawing, and 36 to 39% of its rows are padding
(`data/CONTEXT.md`). Each group is weighted by the labelled cells it holds, so the combined
loss is the mean over exactly the cells the single pass averages. Without dropout the loss and
gradient are the single pass's. With dropout they are not: the groups call dropout on other
shapes and in another order, so they draw other masks (on a Tiny model at attention dropout
0.1, gradients differ by up to 7e-3). The cell budget is therefore fixed for a whole run
(`training/CONTEXT.md`). The optimizer still takes one step over the whole batch.

Cleaned from `core/graph_gpt_baseline/training/microbatch.py`.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import math
from collections.abc import Callable
from typing import Any, cast

import torch
from torch.nn import functional

from eulercad.data.collate import IGNORED_LABEL, Batch
from eulercad.losses.next_row import LossOutput, next_row_loss
from eulercad.tokenization.rows import row_fields
from eulercad.tokenization.vocabulary import PAD

# `collate` rounds its batch length the same way.
PAD_TO_MULTIPLE_OF = 8


def _padded(rows: int, multiple: int) -> int:
    return math.ceil(rows / multiple) * multiple


def _groups(lengths: list[int], cell_budget: int, pad_to_multiple_of: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    # Ascending and stable, so the sample being added sets the group's padded length.
    for index in sorted(range(len(lengths)), key=lambda position: lengths[position]):
        if current and (len(current) + 1) * _padded(lengths[index], pad_to_multiple_of) > cell_budget:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


def length_groups(
    attention_mask: torch.Tensor, *, cell_budget: int, pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF
) -> list[torch.Tensor]:
    """Sort samples by their real row count and cut them into groups of bounded cost.

    A group costs its sample count times its padded row count. Groups grow while that stays
    within `cell_budget`; a sample longer than the whole budget forms a group of its own.

    Returns:
        Index tensors, in increasing row count, that partition every sample.

    Raises:
        ValueError: If either bound is not positive.
    """
    if cell_budget < 1 or pad_to_multiple_of < 1:
        raise ValueError("cell_budget and pad_to_multiple_of must be positive")
    lengths: list[int] = attention_mask.sum(dim=1).tolist()
    return [torch.tensor(group) for group in _groups(lengths, cell_budget, pad_to_multiple_of)]


def grouped_loss(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch: Batch,
    *,
    cell_budget: int,
    pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF,
) -> LossOutput:
    """Score one batch in length-sorted groups, each cut to its own padded length.

    Args:
        forward: `(input_ids, attention_mask) -> logits`, the model or its compiled forward.
        batch: the batch; real rows come before padding, as `collate` returns them.
        cell_budget: the most rows times samples one group may hold.
        pad_to_multiple_of: each group's row count is rounded up to a multiple of this. Above
            `collate`'s 8, a group can need more rows than the batch holds; it gets padding
            rows, which no real row sees and no loss scores. FP8 needs 16 (`module.py`).

    Returns:
        Every group's total and parts weighted by the labelled cells behind them: the single
        pass's loss when the model has no dropout.

    Raises:
        ValueError: If `cell_budget` or `pad_to_multiple_of` is not positive.
    """
    if cell_budget < 1 or pad_to_multiple_of < 1:
        raise ValueError("cell_budget and pad_to_multiple_of must be positive")
    return _grouped(forward, batch, cell_budget, pad_to_multiple_of, drawings=False)[0]


def grouped_loss_and_drawings(
    forward: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    batch: Batch,
    *,
    cell_budget: int,
    pad_to_multiple_of: int = PAD_TO_MULTIPLE_OF,
) -> tuple[LossOutput, torch.Tensor]:
    """`grouped_loss`, plus the mean state of each drawing's real rows, for SIGReg.

    Args:
        forward: `(input_ids, attention_mask) -> (logits, states)`, the states `(B, T, H)` or, one per
            layer, `(S, B, T, H)`.
        batch: as in `grouped_loss`.
        cell_budget: as in `grouped_loss`.
        pad_to_multiple_of: as in `grouped_loss`.

    Returns:
        The loss of `grouped_loss`, and `(B, H)` or `(S, B, H)` float32 means, in batch order, with
        their gradients.

    Raises:
        ValueError: If `cell_budget` or `pad_to_multiple_of` is not positive.
    """
    if cell_budget < 1 or pad_to_multiple_of < 1:
        raise ValueError("cell_budget and pad_to_multiple_of must be positive")
    loss, drawings = _grouped(forward, batch, cell_budget, pad_to_multiple_of, drawings=True)
    return loss, cast("torch.Tensor", drawings)


def _grouped(
    forward: Callable[..., Any], batch: Batch, cell_budget: int, pad_to_multiple_of: int, *, drawings: bool
) -> tuple[LossOutput, torch.Tensor | None]:
    # One transfer for every length; reading them one at a time stalls the device once per sample.
    lengths: list[int] = batch.attention_mask.sum(dim=1).tolist()
    results: list[tuple[LossOutput, torch.Tensor]] = []
    means: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group in _groups(lengths, cell_budget, pad_to_multiple_of):
        rows = _padded(max(lengths[index] for index in group), pad_to_multiple_of)
        indices = torch.tensor(group, device=batch.input_ids.device)
        input_ids, attention_mask = batch.input_ids[indices, :rows], batch.attention_mask[indices, :rows]
        labels = batch.labels[indices, :rows]
        extra = rows - input_ids.shape[1]
        if extra:
            input_ids = functional.pad(input_ids, (0, 0, 0, extra), value=PAD)
            attention_mask = functional.pad(attention_mask, (0, extra), value=False)
            labels = functional.pad(labels, (0, 0, 0, extra), value=IGNORED_LABEL)
        logits = forward(input_ids, attention_mask)
        if drawings:
            logits, states = logits
            real = attention_mask[..., None].float()
            means.append((indices, (states.float() * real).sum(-2) / real.sum(-2)))
        results.append((next_row_loss(logits, labels), (labels != IGNORED_LABEL).flatten(0, 1).sum(dim=0)))
    counts = torch.stack([count for _, count in results]).sum(dim=0)
    total = sum(loss.total * count.sum() for loss, count in results) / counts.sum()
    parts = {
        name: sum(loss.parts[name] * count[field] for loss, count in results) / counts[field]
        for field, name in enumerate(row_fields(batch.labels.shape[-1]))
    }
    if not means:
        return LossOutput(total, parts), None
    order = torch.cat([indices for indices, _ in means]).argsort()
    return LossOutput(total, parts), torch.cat([mean for _, mean in means], dim=-2)[..., order, :]
