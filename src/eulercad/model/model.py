"""`GraphGPTModel`: token rows in, logits for every field of the next row out.

The model reads two sizes from the tokenizer it is given, `vocab_size` and `row_width`, and
keeps the tokenizer so a checkpoint can store it. It never tokenizes: `data/` does that.
"""

import torch
from torch import nn

from eulercad.model.backbone import Backbone
from eulercad.model.config import ModelConfig
from eulercad.model.embedding import TokenEmbedding
from eulercad.model.geometry import GeometryFeatures
from eulercad.model.head import RowHead
from eulercad.model.merge import FieldMerge
from eulercad.tokenization.tokenizer import GraphTokenizer


class GraphGPTModel(nn.Module):
    """Token embedding, field merge, optional geometry features, backbone, and row head.

    Attributes:
        tokenizer: the tokenizer the inputs come from.
        config: the network settings.
    """

    def __init__(self, tokenizer: GraphTokenizer, config: ModelConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.embedding = TokenEmbedding(tokenizer.vocab_size, config)
        self.merge = FieldMerge(tokenizer.row_width, config)
        self.backbone = Backbone(config)
        self.head = RowHead(tokenizer.row_width, tokenizer.vocab_size, config)
        # Built last, so the parts above draw the same initial weights with or without it.
        self.geometry = (
            GeometryFeatures(tokenizer.vocabulary, tokenizer.config.coordinate_bins, config.hidden_size)
            if config.geometry_features
            else None
        )

    def _rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return one vector per row: the merged field vectors, plus the geometry if enabled.

        Returns:
            `(B, T, H)` row vectors.
        """
        rows = self.merge(self.embedding(input_ids))
        return rows if self.geometry is None else rows + self.geometry(input_ids)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Encode right-padded token rows.

        Args:
            input_ids: `(B, T, F)` int64, real rows first, padding cells `PAD`.
            attention_mask: `(B, T)` bool, true on real rows.
            causal: whether a row may read only the rows before it, as pretraining requires.

        Returns:
            `(B, T, H)` hidden states.
        """
        self._check(input_ids, attention_mask)
        return self.backbone(self._rows(input_ids), attention_mask, causal)

    def encode_layers(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """`encode`, keeping every layer's output; see `encode` for the inputs.

        Returns:
            `(layers + 1, B, T, H)`: layer 1 to the last, then the final norm, which is `encode`'s output.
        """
        self._check(input_ids, attention_mask)
        return self.backbone.layer_states(self._rows(input_ids), attention_mask, causal)

    def _check(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """Refuse inputs of the wrong type or shape.

        Raises:
            ValueError: If an input has the wrong type or shape.
        """
        width = self.tokenizer.row_width
        if input_ids.dtype != torch.int64 or input_ids.ndim != 3 or input_ids.shape[-1] != width:
            raise ValueError(f"input_ids must be int64 (B, T, {width}), got {input_ids.dtype} {tuple(input_ids.shape)}")
        if attention_mask.dtype != torch.bool or attention_mask.shape != input_ids.shape[:2]:
            raise ValueError(
                f"attention_mask must be bool {tuple(input_ids.shape[:2])}, "
                f"got {attention_mask.dtype} {tuple(attention_mask.shape)}"
            )

    def largest_attention_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the largest absolute attention score of each layer, over pairs of real rows.

        A diagnostic for the run records: scores that keep growing make the softmax pick one
        row by rounding, and preceded the divergence of the first long run. Call it under
        `torch.no_grad()` and in eval mode, so no dropout draws from the generator.

        Returns:
            `(layers,)` float32.
        """
        self._check(input_ids, attention_mask)
        return self.backbone.largest_attention_scores(self._rows(input_ids), attention_mask)

    def attention_weight_records(self) -> list[dict[str, float]]:
        """Describe each layer's query and key weights, for the run records; see `Backbone`.

        Returns:
            One `{name: value}` per layer.
        """
        return self.backbone.attention_weight_records()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Score every field of the next row, at every row; see `encode` for the inputs.

        Returns:
            `(B, T, F, V)` logits.
        """
        return self.head(self.encode(input_ids, attention_mask))
