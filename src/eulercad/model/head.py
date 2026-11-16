"""`RowHead`: vocabulary logits for every field of the next row.

Cleaned from `core/graph_gpt_baseline/model/heads.py` (`PretrainingHead`, GraphGPT's
`n_token_proj` and `lm_head`).

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import torch
from torch import nn

from eulercad.model.config import ModelConfig


class RowHead(nn.Module):
    """Map `(..., H)` rows to `(..., F, V)` logits.

    The field projection gives one hidden block per field; block `f` goes through the
    vocabulary projection, shared by all fields, to give the logits of field `f`. Every field
    scores the whole vocabulary, as in the baseline.
    """

    def __init__(self, row_width: int, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.row_width = row_width
        self.field_projection = nn.Linear(config.hidden_size, config.hidden_size * row_width, bias=False)
        self.vocabulary_projection = nn.Linear(config.hidden_size, vocab_size, bias=False)
        for layer in (self.field_projection, self.vocabulary_projection):
            nn.init.normal_(layer.weight, std=config.initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score every token for every field.

        Returns:
            `(..., F, V)` logits.
        """
        blocks = self.field_projection(hidden_states).unflatten(-1, (self.row_width, hidden_states.shape[-1]))
        return self.vocabulary_projection(blocks)
