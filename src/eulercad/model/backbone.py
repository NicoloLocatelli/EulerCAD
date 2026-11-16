"""`Backbone`: Llama decoder layers over row vectors, causal or bidirectional.

Cleaned from `core/graph_gpt_baseline/model/backbone.py`, itself derived from GraphGPT v0.5.0:
pre-normalized Llama layers with stochastic depth and MLP dropout, rotary positions, and an
explicit attention mask. The baseline's layer scale is left out; its bidirectional mode is
kept, because `downstream/` encodes a whole drawing at once and a repair reads both sides of
the hole. Pretraining stays causal, which is the default. Added here: optional QK-norm through
Qwen3's attention, and records of the attention scores and weights of each layer, for the run
records.

Since 24 September the layer runs the attention itself, from the module's own projections, so
two options fit in: the value residual and a causal pass without a mask. Every caller puts the
real rows first, so under causal attention a real row never sees a padding row; the mask adds
nothing, and without it PyTorch may pick its FlashAttention kernel. Bidirectional attention
keeps the mask.

With `attention_residuals` (added 3 October), the layers no longer add their updates to the rows:
each update is a source of `DepthAttention`, which gives every sublayer and the final
normalization its learned average of the sources (`depth_attention.py`).

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.nn import functional
from transformers import LlamaConfig, Qwen3Config
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from eulercad.model.config import ModelConfig
from eulercad.model.depth_attention import DepthAttention

ATTENTION_IMPLEMENTATION = "sdpa"


def drop_path(update: torch.Tensor, probability: float, *, training: bool) -> torch.Tensor:
    """Zero the update of whole samples with `probability` and rescale the rest (stochastic depth).

    Returns:
        The rescaled update, or `update` itself outside training.
    """
    if probability <= 0 or not training:
        return update
    keep = 1 - probability
    shape = (update.shape[0],) + (1,) * (update.ndim - 1)
    kept = torch.floor(keep + torch.rand(shape, dtype=update.dtype, device=update.device))
    return update / keep * kept


def attention_scores(
    attention: LlamaAttention | Qwen3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return an attention module's scores before masking and softmax, as its forward computes them.

    Returns:
        `(B, heads, T, T)` scores `q . k / sqrt(head_dim)`.
    """
    shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
    query, key = attention.q_proj(hidden_states).view(shape), attention.k_proj(hidden_states).view(shape)
    if isinstance(attention, Qwen3Attention):
        query, key = attention.q_norm(query), attention.k_norm(key)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query.transpose(1, 2), key.transpose(1, 2), cos, sin)
    return query @ key.transpose(-1, -2) * attention.scaling


class DecoderLayer(nn.Module):
    """Pre-normalized Llama layer: an attention update, then a GELU MLP update.

    The attention is `LlamaAttention`, or, given a `Qwen3Config`, `Qwen3Attention`: the same
    projections plus QK-norm, each head's queries and keys RMS-normalized with learned gains
    before the rotary positions. With `gated`, `attention_gate` computes a sigmoid gate from the
    normalized input that scales each channel of the attention output before `o_proj`.
    """

    def __init__(
        self,
        llama: LlamaConfig,
        index: int,
        *,
        path_dropout: float,
        mlp_dropout: float,
        qwen: Qwen3Config | None = None,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(llama.hidden_size, eps=llama.rms_norm_eps)
        self.self_attn: LlamaAttention | Qwen3Attention = (
            LlamaAttention(llama, index) if qwen is None else Qwen3Attention(qwen, index)
        )
        self.attention_gate = nn.Linear(llama.hidden_size, llama.hidden_size, bias=False) if gated else None
        self.post_attention_layernorm = LlamaRMSNorm(llama.hidden_size, eps=llama.rms_norm_eps)
        self.mlp = LlamaMLP(llama)
        self.mlp_dropout = nn.Dropout(mlp_dropout)
        self.path_dropout = path_dropout

    def _attend(
        self,
        normed: torch.Tensor,
        visible: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        first_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the attention update as `self_attn`'s forward does, with the value residual and the gate.

        Returns:
            The update `(B, T, H)` and this layer's own values `(B, heads, T, head_dim)`.
        """
        attention = self.self_attn
        shape = (*normed.shape[:-1], -1, attention.head_dim)
        query, key = attention.q_proj(normed).view(shape), attention.k_proj(normed).view(shape)
        if isinstance(attention, Qwen3Attention):
            query, key = attention.q_norm(query), attention.k_norm(key)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query.transpose(1, 2), key.transpose(1, 2), cos, sin)
        values = attention.v_proj(normed).view(shape).transpose(1, 2)
        mixed = values if first_values is None else 0.5 * (values + first_values)
        output = functional.scaled_dot_product_attention(
            query,
            key,
            mixed,
            attn_mask=visible,
            dropout_p=cast("float", attention.attention_dropout) if self.training else 0.0,
            scale=attention.scaling,
            is_causal=visible is None,
        )
        output = output.transpose(1, 2).reshape(*normed.shape[:-1], -1)
        if self.attention_gate is not None:
            output = output * torch.sigmoid(self.attention_gate(normed))
        return attention.o_proj(output), values

    def attention_update(
        self,
        hidden_states: torch.Tensor,
        visible: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        first_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize the rows and compute the attention update, after stochastic depth; see `forward`.

        Returns:
            The update `(B, T, H)` and this layer's own values.
        """
        update, values = self._attend(self.input_layernorm(hidden_states), visible, position_embeddings, first_values)
        return drop_path(update, self.path_dropout, training=self.training), values

    def mlp_update(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the rows and compute the MLP update, after dropout and stochastic depth.

        Returns:
            The update `(B, T, H)`.
        """
        mlp, normed = self.mlp, self.post_attention_layernorm(hidden_states)
        update = self.mlp_dropout(mlp.act_fn(mlp.gate_proj(normed)) * mlp.up_proj(normed))
        update = self.mlp_dropout(mlp.down_proj(update))
        return drop_path(update, self.path_dropout, training=self.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        visible: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        first_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the attention and MLP updates to `(B, T, H)` rows.

        Args:
            hidden_states: the rows.
            visible: `(B, 1, T, T)` bool, which rows each row reads; None for plain causal
                attention, which is exact when the real rows come first.
            position_embeddings: the rotary cosines and sines.
            first_values: the first layer's values, for the value residual; None otherwise.

        Returns:
            Updated rows of the same shape, and this layer's own values.
        """
        update, values = self.attention_update(hidden_states, visible, position_embeddings, first_values)
        hidden_states = hidden_states + update
        return hidden_states + self.mlp_update(hidden_states), values


class Backbone(nn.Module):
    """Encode `(B, T, H)` rows; padding rows are never seen, whichever way the rows are read."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        llama = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.hidden_layers,
            num_attention_heads=config.attention_heads,
            num_key_value_heads=config.attention_heads,
            hidden_act="gelu",
            rms_norm_eps=config.rms_norm_epsilon,
            attention_dropout=config.attention_dropout,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            # Consumed by PreTrainedConfig.__init__; the dataclass signature does not list it.
            attn_implementation=ATTENTION_IMPLEMENTATION,  # ty: ignore[unknown-argument]
        )
        qwen = (
            Qwen3Config(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.hidden_layers,
                num_attention_heads=config.attention_heads,
                num_key_value_heads=config.attention_heads,
                head_dim=config.hidden_size // config.attention_heads,
                rms_norm_eps=config.rms_norm_epsilon,
                attention_dropout=config.attention_dropout,
                attention_bias=False,
                use_sliding_window=False,
                attn_implementation=ATTENTION_IMPLEMENTATION,  # ty: ignore[unknown-argument]
            )
            if config.qk_norm
            else None
        )
        layers = config.hidden_layers
        self.layers = nn.ModuleList(
            DecoderLayer(
                llama,
                index,
                path_dropout=config.path_dropout * index / (layers - 1) if layers > 1 else 0.0,
                mlp_dropout=config.mlp_dropout,
                qwen=qwen,
                gated=config.gated_attention,
            )
            for index in range(layers)
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.rotary_embedding = LlamaRotaryEmbedding(llama)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        if config.zero_init_outputs:
            for layer in self._decoder_layers():
                nn.init.zeros_(layer.self_attn.o_proj.weight)
                nn.init.zeros_(layer.mlp.down_proj.weight)
        self.value_residual = config.value_residual
        # Outside `layers`, so the optimizer gives the vectors to AdamW with the norms.
        self.depth_attention = (
            DepthAttention(
                config.hidden_size,
                sublayers=2 * layers,
                block=config.attention_residual_block,
                eps=config.rms_norm_epsilon,
            )
            if config.attention_residuals
            else None
        )

    def _decoder_layers(self) -> list[DecoderLayer]:
        return [cast("DecoderLayer", module) for module in self.layers]

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        visible: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        inspect: Callable[[DecoderLayer, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Run the layers; `inspect`, if given, sees each layer with the rows its attention reads.

        Returns:
            The rows after the last layer, before the final normalization.
        """
        first = None
        stream = None if self.depth_attention is None else self.depth_attention.stream(hidden_states)
        for layer in self._decoder_layers():
            if stream is not None:
                hidden_states = stream.read()
            if inspect is not None:
                inspect(layer, hidden_states)
            if stream is None:
                hidden_states, values = layer(hidden_states, visible, position_embeddings, first)
            else:
                update, values = layer.attention_update(hidden_states, visible, position_embeddings, first)
                stream.write(update)
                stream.write(layer.mlp_update(stream.read()))
            if self.value_residual and first is None:
                first = values
        return hidden_states if stream is None else stream.read()

    def positions_and_mask(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal: bool = True
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        rows = hidden_states.shape[1]
        device = hidden_states.device
        position_embeddings = self.rotary_embedding(hidden_states, torch.arange(rows, device=device)[None])
        visible = attention_mask[:, None, None, :]
        if causal:
            visible = visible & torch.ones(rows, rows, dtype=torch.bool, device=device).tril()
        return position_embeddings, visible

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Encode rows; row `t` sees row `k` when `attention_mask[k]` is true and, if `causal`, `k <= t`.

        Real rows must come first: row `t` has rotary position `t`. Without `causal` every real
        row reads every other, which is what a downstream task wants and what pretraining must
        never do, since its labels are the next row.

        Returns:
            `(B, T, H)` hidden states after the final normalization.
        """
        position_embeddings, visible = self.positions_and_mask(hidden_states, attention_mask, causal)
        # Real rows come first, so plain causal attention never lets a real row read padding.
        hidden_states = self._run_layers(hidden_states, None if causal else visible, position_embeddings)
        return self.norm(hidden_states)

    def layer_states(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        """Encode rows as `forward` does, keeping the output of every layer and of the final norm.

        Returns:
            `(layers + 1, B, T, H)`: layer 1 to the last, then the final normalization.

        Raises:
            ValueError: If the layers mix with attention residuals, whose outputs are not the next layer's input.
        """
        if self.depth_attention is not None:
            raise ValueError("models with attention residuals are not supported")
        position_embeddings, visible = self.positions_and_mask(hidden_states, attention_mask, causal)
        inputs: list[torch.Tensor] = []
        last = self._run_layers(
            hidden_states, None if causal else visible, position_embeddings, lambda _, rows: inputs.append(rows)
        )
        return torch.stack([*inputs[1:], last, self.norm(last)])

    def largest_attention_scores(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the layers and return the largest absolute score of each, over pairs of real rows it attends.

        Returns:
            `(layers,)` float32, one value per layer.
        """
        position_embeddings, visible = self.positions_and_mask(hidden_states, attention_mask)
        pairs = visible & attention_mask[:, None, :, None]
        largest = []

        def record(layer: DecoderLayer, rows: torch.Tensor) -> None:
            scores = attention_scores(layer.self_attn, layer.input_layernorm(rows), position_embeddings)
            largest.append(scores.float().abs().masked_fill(~pairs, 0).amax())

        self._run_layers(hidden_states, visible, position_embeddings, record)
        return torch.stack(largest)

    def attention_weight_records(self) -> list[dict[str, float]]:
        """Describe each layer's query and key weights: Frobenius norm and largest singular value,
        and with QK-norm the largest absolute and the RMS gain.

        Returns:
            One `{name: value}` per layer.
        """
        names, values = [], []
        for index, module in enumerate(self.layers):
            attention = cast("DecoderLayer", module).self_attn
            for field, projection in (("query", attention.q_proj), ("key", attention.k_proj)):
                weight = projection.weight.detach().float()
                names += [(index, f"{field}_norm"), (index, f"{field}_top_singular_value")]
                values += [torch.linalg.matrix_norm(weight), torch.linalg.matrix_norm(weight, ord=2)]
            if isinstance(attention, Qwen3Attention):
                for field, norm in (("query", attention.q_norm), ("key", attention.k_norm)):
                    gain = norm.weight.detach().float()
                    names += [(index, f"{field}_gain_max"), (index, f"{field}_gain_rms")]
                    values += [gain.abs().amax(), gain.square().mean().sqrt()]
        records: list[dict[str, float]] = [{} for _ in self.layers]
        for (index, name), value in zip(names, torch.stack(values).tolist(), strict=True):
            records[index][name] = value
        return records
