"""One sample: the token rows of one drawing, their next-row labels, and the generator behind them.

Cleaned from `core/graph_gpt_baseline/data/next_token.py` and the seeding of `examples.py`.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import random
from dataclasses import dataclass

import torch

from eulercad.tokenization.vocabulary import EOS

STREAM = "next_row"
# The label of a cell training ignores; `collate.py` re-exports it, where most callers read it.
IGNORED_LABEL = -100


def sample_rng(run_seed: int, epoch: int | None, sketch_id: int, stream: str = STREAM) -> random.Random:
    """Return the only generator of one sample; `epoch=None` gives the fixed validation draw.

    The key is a string, which `random` hashes with SHA-512, so the stream does not depend on
    the process, the loader worker, or `PYTHONHASHSEED`.

    Returns:
        A fresh generator for `(run_seed, epoch, sketch_id, stream)`.
    """
    key = ":".join(map(str, (run_seed, "fixed" if epoch is None else epoch, sketch_id, stream)))
    return random.Random(key)  # noqa: S311  # nosec B311


def next_row_labels(rows: torch.Tensor) -> torch.Tensor:
    """Label row `t` with row `t + 1`, and the last row with the all-`EOS` end row.

    The model does not shift again: it scores label `t` with its logits at row `t`.

    Returns:
        `(L, F)` labels, a new tensor; `rows` is unchanged.
    """
    return torch.cat([rows[1:], torch.full_like(rows[:1], EOS)])


def suffix_labels(rows: torch.Tensor, cut: int) -> torch.Tensor:
    """Label only the last `cut` rows of the walk, and the end row after them.

    Every earlier cell is ignored, so the walk before the cut is context the model reads and
    is never asked to predict. A cut at least as long as the walk scores all of it.

    Returns:
        `(L, F)` labels, a new tensor; `rows` is unchanged.
    """
    labels = next_row_labels(rows)
    labels[: max(len(rows) - cut - 1, 0)] = IGNORED_LABEL
    return labels


@dataclass(frozen=True, slots=True)
class Sample:
    """One drawing, ready for batching.

    Attributes:
        input_ids: `(L, F)` int64 token rows, as `GraphTokenizer.tokenize` returns them.
        labels: `(L, F)` int64 next-row labels.
        sketch_id: bookkeeping only; never passed to the model.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    sketch_id: int

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.labels.shape != self.input_ids.shape:
            raise ValueError("input_ids must be (L, F) and labels the same shape")
