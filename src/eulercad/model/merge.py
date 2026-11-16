"""`FieldMerge`: one vector per row from the vectors of its fields.

Cleaned from `core/graph_gpt_baseline/model/aggregation.py` (GraphGPT's gated
`StackedFeatAggregation`).

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import math

import torch
from torch import nn

from eulercad.model.config import ModelConfig


class FieldMerge(nn.Module):
    """Reduce `(B, T, F, H)` field vectors to `(B, T, H)` rows: `sum_f gate[f] * vector[f]`.

    Each field `f` has a learned gate `weight[f]` of hidden width.
    """

    def __init__(self, row_width: int, config: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(row_width, config.hidden_size))
        # Uniform in [-1/sqrt(H), 1/sqrt(H)], as upstream.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return the gated sum of each row's field vectors.

        Returns:
            `(B, T, H)` row vectors.
        """
        return torch.einsum("btfh,fh->bth", embeddings, self.weight)
