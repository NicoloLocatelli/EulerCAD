"""`TokenEmbedding`: one vector per token of every row field.

Cleaned from `core/graph_gpt_baseline/model/backbone.py` (`embed_tokens`, `embedding_dropout`).

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import torch
from torch import nn

from eulercad.model.config import ModelConfig
from eulercad.tokenization.vocabulary import PAD


class TokenEmbedding(nn.Module):
    """Look up `(B, T, F)` token identifiers as `(B, T, F, H)` vectors, then apply dropout.

    One table serves every field. The `PAD` row is zero and gets no gradient.
    """

    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, config.hidden_size, padding_idx=PAD)
        self.dropout = nn.Dropout(config.embedding_dropout)
        nn.init.normal_(self.table.weight, std=config.initializer_range)
        with torch.no_grad():
            self.table.weight[PAD].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the embedding of every token.

        Returns:
            `(B, T, F, H)` vectors.
        """
        return self.dropout(self.table(input_ids))
