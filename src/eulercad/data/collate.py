"""`collate`: right-padded batches of samples.

Cleaned from `core/graph_gpt_baseline/data/collate.py` (GraphGPT's `DataCollatorForGSTCausal`).

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from eulercad.data.sample import IGNORED_LABEL, Sample
from eulercad.tokenization.vocabulary import PAD

MAX_ROWS = 1024


@dataclass(frozen=True, slots=True)
class Batch:
    """Samples padded to one length `T`.

    Attributes:
        input_ids: `(B, T, F)` int64 token rows, right-padded with `PAD`.
        attention_mask: `(B, T)` bool, true on real rows.
        labels: `(B, T, F)` int64 next-row labels, `IGNORED_LABEL` on padding.
        sketch_id: `(B,)` int64, bookkeeping only; never passed to the model.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sketch_id: torch.Tensor


def collate(samples: Sequence[Sample], *, max_rows: int = MAX_ROWS, pad_to_multiple_of: int = 8) -> Batch:
    """Right-pad samples to the longest one, rounded up to `pad_to_multiple_of`, at most `max_rows`.

    A sample longer than `max_rows` keeps its leading rows, as in the baseline. At the default
    1,024 no drawing within 512 nodes can be cut (`data/CONTEXT.md`).

    Returns:
        The batch.

    Raises:
        ValueError: If there is no sample, a length setting is not positive, or the samples
            have different row widths.
    """
    if not samples:
        raise ValueError("a batch needs at least one sample")
    if max_rows < 1 or pad_to_multiple_of < 1:
        raise ValueError("max_rows and pad_to_multiple_of must be positive")
    width = samples[0].input_ids.shape[1]
    if any(sample.input_ids.shape[1] != width for sample in samples):
        raise ValueError("every sample must have the same row width")
    longest = max(sample.input_ids.shape[0] for sample in samples)
    rows = min(math.ceil(longest / pad_to_multiple_of) * pad_to_multiple_of, max_rows)

    input_ids = torch.full((len(samples), rows, width), PAD, dtype=torch.int64)
    labels = torch.full_like(input_ids, IGNORED_LABEL)
    attention_mask = torch.zeros((len(samples), rows), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = min(sample.input_ids.shape[0], rows)
        input_ids[index, :count] = sample.input_ids[:count]
        labels[index, :count] = sample.labels[:count]
        attention_mask[index, :count] = True
    sketch_id = torch.tensor([sample.sketch_id for sample in samples], dtype=torch.int64)
    return Batch(input_ids, attention_mask, labels, sketch_id)
