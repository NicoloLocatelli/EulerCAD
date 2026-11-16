"""`next_row_loss`: cross-entropy of the next row, in total and per field.

Cleaned from `core/graph_gpt_baseline/model/losses.py` (`token_cross_entropy`, GraphGPT's
`_get_ce_loss`): the total is the mean over every labelled cell, in float32. Unlike the
baseline, a batch in which a field has no labelled cell raises instead of giving NaN.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

from dataclasses import dataclass

import torch
from torch.nn.functional import cross_entropy

from eulercad.data.collate import IGNORED_LABEL
from eulercad.tokenization.rows import row_fields


@dataclass(frozen=True, slots=True)
class LossOutput:
    """The scalar training minimizes, and its named parts for the run record.

    Attributes:
        total: the scalar to minimize.
        parts: every component by name; each keeps its gradient, and `training/` records all.
    """

    total: torch.Tensor
    parts: dict[str, torch.Tensor]


def next_row_loss(logits: torch.Tensor, labels: torch.Tensor) -> LossOutput:
    """Score the next-row logits against the labels; cells labelled -100 are ignored.

    Args:
        logits: `(B, T, F, V)` from the model; label `t` is scored with the logits at row `t`.
        labels: `(B, T, F)` int64 from `data/`.

    Returns:
        `total`: mean cross-entropy over every labelled cell. `parts`: one mean cross-entropy
        per field of the layout the labels are in, over that field's labelled cells.

    Raises:
        ValueError: If the shapes do not match, the rows fit no layout, or a field has no
            labelled cell, which would make its part NaN.
    """
    if logits.ndim != 4 or labels.dtype != torch.int64 or labels.shape != logits.shape[:3]:
        raise ValueError(f"logits must be (B, T, F, V) and labels int64 (B, T, F); got {logits.shape}, {labels.shape}")
    names = row_fields(labels.shape[-1])
    scored = labels != IGNORED_LABEL
    counts = scored.flatten(0, 1).sum(dim=0)
    empty = (counts == 0).nonzero().flatten().tolist()  # one device sync per step
    if empty:
        raise ValueError(f"field {names[empty[0]]} has no labelled cell")
    cells = cross_entropy(
        logits.flatten(0, -2).float(), labels.flatten(), ignore_index=IGNORED_LABEL, reduction="none"
    ).view_as(labels)
    total = cells.sum() / counts.sum()
    parts = {name: cells[..., field].sum() / counts[field] for field, name in enumerate(names)}
    return LossOutput(total, parts)
