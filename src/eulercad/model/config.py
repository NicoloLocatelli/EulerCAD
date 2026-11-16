"""`ModelConfig`: the network settings, plain values that can be written to JSON.

The sizes are GraphGPT's Tiny and Mini (`core/graph_gpt_baseline/contracts/config.py`); the
dropouts are the baseline's pretraining defaults.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The network settings. The default is Tiny.

    Attributes:
        hidden_size: the width of every row vector.
        hidden_layers: the number of decoder layers.
        attention_heads: attention heads per layer; must divide `hidden_size`.
        attention_dropout: dropout on the attention weights.
        embedding_dropout: dropout on the token embeddings.
        path_dropout: stochastic depth at the last layer; earlier layers get less, linearly.
        mlp_dropout: dropout inside and after each MLP.
        rope_theta: the base of the rotary positions.
        rms_norm_epsilon: added to the variance in every RMS normalization.
        initializer_range: standard deviation of the normal weight initialization.
        qk_norm: RMS-normalize each head's queries and keys, with learned gains, before the
            rotary positions. It bounds the attention scores; off in the baseline.
        geometry_features: add each row's coordinates, step, direction, and visit count as
            numbers (`model/geometry.py`); off in the baseline.
        value_residual: every layer after the first attends over the mean of its own values and
            the first layer's (Identity-ResFormer, Zhou et al. 2025); no new parameters.
        zero_init_outputs: start each layer's attention output and MLP down projections at
            zero, so every layer starts as the identity (modded-nanogpt).
        gated_attention: multiply each channel of the attention output by a sigmoid of the
            layer's normalized input, before the output projection (Qiu et al. 2025, used in
            Qwen3-Next); one `hidden_size` x `hidden_size` matrix per layer.
        attention_residuals: each sublayer reads a learned softmax average of the earlier
            sublayer outputs instead of their sum (Attention Residuals, Kimi Team 2026;
            `model/depth_attention.py`); one `hidden_size` vector per sublayer, plus one.
        attention_residual_block: with `attention_residuals`, sum the outputs in groups of this
            many sublayers and average over the groups (Block Attention Residuals); 1 averages
            over every output.
    """

    hidden_size: int = 128
    hidden_layers: int = 2
    attention_heads: int = 2
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.0
    path_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rope_theta: float = 10_000.0
    rms_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    qk_norm: bool = False
    geometry_features: bool = False
    value_residual: bool = False
    zero_init_outputs: bool = False
    gated_attention: bool = False
    attention_residuals: bool = False
    attention_residual_block: int = 1

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.hidden_layers, self.attention_heads) < 1:
            raise ValueError("hidden_size, hidden_layers, and attention_heads must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("attention_heads must divide hidden_size")
        for dropout in (self.attention_dropout, self.embedding_dropout, self.path_dropout, self.mlp_dropout):
            if not 0 <= dropout < 1:
                raise ValueError("every dropout must be in [0, 1)")
        if min(self.rope_theta, self.rms_norm_epsilon, self.initializer_range) <= 0:
            raise ValueError("rope_theta, rms_norm_epsilon, and initializer_range must be positive")
        if self.attention_residual_block < 1:
            raise ValueError("attention_residual_block must be positive")
        if self.attention_residual_block > 1 and not self.attention_residuals:
            raise ValueError("attention_residual_block needs attention_residuals")

    @property
    def intermediate_size(self) -> int:
        """The MLP width, four times the hidden width as in the baseline."""
        return 4 * self.hidden_size

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """GraphGPT Tiny: width 128, 2 layers, 2 heads. For tests.

        Returns:
            The Tiny configuration.
        """
        return cls()

    @classmethod
    def mini(cls) -> "ModelConfig":
        """GraphGPT Mini: width 256, 4 layers, 4 heads. For pretraining.

        Returns:
            The Mini configuration.
        """
        return cls(hidden_size=256, hidden_layers=4, attention_heads=4)

    @classmethod
    def medium(cls) -> "ModelConfig":
        """GraphGPT Medium: width 512, 8 layers, 8 heads. Eight times Mini's layer weights.

        Returns:
            The Medium configuration.
        """
        return cls(hidden_size=512, hidden_layers=8, attention_heads=8)

    @classmethod
    def deep(cls) -> "ModelConfig":
        """Deeper and narrower than Medium: width 384, 14 layers, 6 heads of Medium's width 64.

        96% of Medium's parameters, so a comparison with Medium weighs depth against width.

        Returns:
            The Deep configuration.
        """
        return cls(hidden_size=384, hidden_layers=14, attention_heads=6)

    def to_dict(self) -> dict[str, Any]:
        """Return the settings, ready for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Rebuild a configuration from `to_dict()`.

        Returns:
            A configuration equal to the one that was saved. A configuration saved before
            `qk_norm`, `geometry_features`, `value_residual`, `zero_init_outputs`,
            `gated_attention`, or `attention_residuals` existed loads with them off.
        """
        return cls(**data)
