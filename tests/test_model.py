"""Tests for the network of the GraphGPT-based architecture: settings, parts, and the full model."""

import json
import random
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from eulercad.analysis.layers import hidden_states
from eulercad.model.backbone import Backbone, DecoderLayer, attention_scores
from eulercad.model.config import ModelConfig
from eulercad.model.embedding import TokenEmbedding
from eulercad.model.head import RowHead
from eulercad.model.merge import FieldMerge
from eulercad.model.model import GraphGPTModel
from eulercad.representation.euler import euler_walk
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.tokenization.vocabulary import JUMP, PAD

DEFAULT = GraphTokenizer()
SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")),
]
# A square with a tail, then a triangle and a separate segment: different row counts, so padding.
DRAWINGS = [
    ([[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0], [1.0, 1.0]], [(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)]),
    ([[-0.5, 0.0], [0.5, 0.0], [0.0, 0.8], [0.9, -0.9], [0.9, 0.9]], [(0, 1), (1, 2), (0, 2), (3, 4)]),
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  # nosec B311


def _graph(positions: list[list[float]], pairs: list[tuple[int, int]]) -> DrawingGraph:
    edges = torch.tensor(sorted(pairs), dtype=torch.int64).T
    return DrawingGraph(0, torch.tensor(positions, dtype=torch.float64), edges, (0.0, 0.0), 1.0)


def _batch(tokenizer: GraphTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize the drawings and right-pad them with `PAD`, as `data/` will.

    Returns:
        `input_ids` `(2, T, F)` and `attention_mask` `(2, T)`.
    """
    samples = []
    for seed, (positions, pairs) in enumerate(DRAWINGS):
        graph, rng = _graph(positions, pairs), _rng(seed)
        samples.append(tokenizer.tokenize(graph, euler_walk(graph, rng), rng).rows)
    rows = max(len(sample) for sample in samples)
    input_ids = torch.full((len(samples), rows, tokenizer.row_width), PAD, dtype=torch.int64)
    attention_mask = torch.zeros((len(samples), rows), dtype=torch.bool)
    for index, sample in enumerate(samples):
        input_ids[index, : len(sample)] = sample
        attention_mask[index, : len(sample)] = True
    return input_ids, attention_mask


def _model(tokenizer: GraphTokenizer = DEFAULT, config: ModelConfig | None = None) -> GraphGPTModel:
    torch.manual_seed(0)
    return GraphGPTModel(tokenizer, config or ModelConfig.tiny()).eval()


def _replace_row(input_ids: torch.Tensor, sample: int, row: int, vocab_size: int) -> torch.Tensor:
    changed = input_ids.clone()
    changed[sample, row] = changed[sample, row] % (vocab_size - 1) + 1
    return changed


# ModelConfig


def test_presets_have_the_baseline_tiny_mini_and_medium_shapes() -> None:
    tiny, mini, medium = ModelConfig.tiny(), ModelConfig.mini(), ModelConfig.medium()

    assert ModelConfig() == tiny
    assert (tiny.hidden_size, tiny.hidden_layers, tiny.attention_heads, tiny.intermediate_size) == (128, 2, 2, 512)
    assert (mini.hidden_size, mini.hidden_layers, mini.attention_heads, mini.intermediate_size) == (256, 4, 4, 1024)
    assert (medium.hidden_size, medium.hidden_layers, medium.attention_heads, medium.intermediate_size) == (
        512,
        8,
        8,
        2048,
    )
    for config in (tiny, mini, medium):
        assert (config.attention_dropout, config.embedding_dropout, config.path_dropout, config.mlp_dropout) == (
            0.1,
            0.0,
            0.0,
            0.0,
        )
        assert (config.rope_theta, config.rms_norm_epsilon, config.initializer_range) == (10_000.0, 1e-6, 0.02)
        assert config.qk_norm is False


def test_deep_is_narrower_and_deeper_than_medium_with_the_same_head_width() -> None:
    deep, medium = ModelConfig.deep(), ModelConfig.medium()

    assert (deep.hidden_size, deep.hidden_layers, deep.attention_heads, deep.intermediate_size) == (384, 14, 6, 1536)
    assert deep.hidden_size // deep.attention_heads == medium.hidden_size // medium.attention_heads
    assert (deep.qk_norm, deep.geometry_features) == (False, False)


def test_config_round_trips_through_json() -> None:
    config = ModelConfig(hidden_size=64, hidden_layers=3, attention_heads=4, path_dropout=0.2, qk_norm=True)

    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_a_config_saved_before_qk_norm_loads_without_it() -> None:
    saved = ModelConfig.mini().to_dict()
    del saved["qk_norm"]

    assert ModelConfig.from_dict(saved) == ModelConfig.mini()


@pytest.mark.parametrize(
    "settings",
    [
        {"hidden_size": 0},
        {"hidden_layers": 0},
        {"attention_heads": 0},
        {"hidden_size": 100, "attention_heads": 3},
        {"attention_dropout": 1.0},
        {"embedding_dropout": -0.1},
        {"path_dropout": 1.0},
        {"mlp_dropout": 1.5},
        {"rope_theta": 0.0},
        {"rms_norm_epsilon": 0.0},
        {"initializer_range": -0.02},
    ],
)
def test_bad_settings_are_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**settings)


# Parts


def test_token_embedding_looks_up_every_field_and_keeps_pad_at_zero() -> None:
    torch.manual_seed(0)
    embedding = TokenEmbedding(DEFAULT.vocab_size, ModelConfig.tiny())
    input_ids, _ = _batch(DEFAULT)
    output = embedding(input_ids)
    output.sum().backward()
    weight = embedding.table.weight

    assert tuple(output.shape) == (*input_ids.shape, 128)
    assert tuple(weight.shape) == (DEFAULT.vocab_size, 128)
    torch.testing.assert_close(output, weight[input_ids])
    assert not bool(weight[PAD].any())
    assert weight.grad is not None
    assert not bool(weight.grad[PAD].any())
    assert float(weight[PAD + 1 :].detach().std()) == pytest.approx(0.02, rel=0.05)


def test_field_merge_is_a_gated_sum_over_fields() -> None:
    torch.manual_seed(0)
    merge = FieldMerge(5, ModelConfig.tiny())
    embeddings = torch.randn(2, 3, 5, 128)
    bound = 1 / 128**0.5

    assert tuple(merge.weight.shape) == (5, 128)
    assert float(merge.weight.detach().abs().max()) <= bound
    torch.testing.assert_close(merge(embeddings), (embeddings * merge.weight).sum(dim=2))


def test_row_head_projects_each_field_block_onto_the_shared_vocabulary() -> None:
    torch.manual_seed(0)
    head = RowHead(5, DEFAULT.vocab_size, ModelConfig.tiny())
    hidden = torch.randn(2, 3, 128)
    logits = head(hidden)
    blocks = head.field_projection(hidden)

    assert tuple(logits.shape) == (2, 3, 5, DEFAULT.vocab_size)
    for field in range(5):
        expected = head.vocabulary_projection(blocks[..., field * 128 : (field + 1) * 128])
        torch.testing.assert_close(logits[..., field, :], expected)
    assert head.field_projection.bias is None
    assert head.vocabulary_projection.bias is None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (ModelConfig.tiny(), [0.0, 0.0]),
        (ModelConfig(hidden_layers=1, path_dropout=0.1), [0.0]),
        (
            ModelConfig(hidden_size=256, hidden_layers=4, attention_heads=4, path_dropout=0.1),
            [0, 0.1 / 3, 0.2 / 3, 0.1],
        ),
    ],
)
def test_path_dropout_rises_linearly_across_layers(config: ModelConfig, expected: list[float]) -> None:
    backbone = Backbone(config)

    assert [layer.path_dropout for layer in backbone.layers] == pytest.approx(expected)


def test_settings_reach_the_transformer_modules() -> None:
    backbone = Backbone(ModelConfig.mini())
    first, last = backbone.layers[0], backbone.layers[-1]
    assert isinstance(first, DecoderLayer)
    assert isinstance(last, DecoderLayer)
    rotary = 1.0 / (10_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))

    assert len(backbone.layers) == 4
    assert (first.self_attn.head_dim, first.self_attn.attention_dropout) == (64, 0.1)
    assert first.mlp.gate_proj.out_features == 1024
    assert type(first.mlp.act_fn).__name__ == "GELUActivation"
    assert backbone.norm.variance_epsilon == pytest.approx(1e-6)
    assert last.input_layernorm.variance_epsilon == pytest.approx(1e-6)
    torch.testing.assert_close(backbone.rotary_embedding.inv_freq, rotary)


def test_linear_weights_are_drawn_with_the_initializer_range() -> None:
    torch.manual_seed(0)
    model = GraphGPTModel(DEFAULT, ModelConfig.mini())
    linear = [module.weight for module in model.modules() if isinstance(module, torch.nn.Linear)]

    assert len(linear) == 4 * 7 + 2
    for weight in linear:
        assert float(weight.detach().std()) == pytest.approx(0.02, rel=0.1)


# GraphGPTModel


@pytest.mark.parametrize(
    ("config", "parameters"),
    [
        (ModelConfig.tiny(), 936_960),
        (ModelConfig.mini(), 5_184_512),
        (ModelConfig(hidden_size=256, hidden_layers=4, attention_heads=4, qk_norm=True), 5_185_024),
        (ModelConfig.medium(), 36_194_304),
        (replace(ModelConfig.medium(), qk_norm=True), 36_195_328),
        (replace(ModelConfig.deep(), qk_norm=True), 34_770_688),
    ],
)
def test_parameter_count(config: ModelConfig, parameters: int) -> None:
    width, layers, vocab, fields = config.hidden_size, config.hidden_layers, DEFAULT.vocab_size, DEFAULT.row_width
    head = width // config.attention_heads
    # Embedding, field gates, per layer four attention and three MLP matrices plus two norms
    # (and the query and key gains with QK-norm), final norm, field projection, vocabulary projection.
    formula = (
        vocab * width
        + fields * width
        + layers * (4 * width**2 + 3 * width * 4 * width + 2 * width + 2 * head * config.qk_norm)
        + width
        + width * fields * width
        + width * vocab
    )

    assert formula == parameters == sum(value.numel() for value in _model(config=config).parameters())


@pytest.mark.parametrize("tokenizer", [DEFAULT, SMALL], ids=["default", "small"])
def test_sizes_come_from_the_tokenizer(tokenizer: GraphTokenizer) -> None:
    model = _model(tokenizer)
    input_ids, attention_mask = _batch(tokenizer)
    with torch.no_grad():
        hidden = model.encode(input_ids, attention_mask)
        logits = model(input_ids, attention_mask)

    assert model.tokenizer is tokenizer
    assert model.config == ModelConfig.tiny()
    assert tuple(hidden.shape) == (*attention_mask.shape, 128)
    assert tuple(logits.shape) == (*input_ids.shape, tokenizer.vocab_size)
    torch.testing.assert_close(logits, model.head(hidden))


def test_attention_is_causal() -> None:
    model = _model()
    input_ids, attention_mask = _batch(DEFAULT)
    with torch.no_grad():
        before = model(input_ids, attention_mask)
        after = model(_replace_row(input_ids, 0, 2, DEFAULT.vocab_size), attention_mask)

    torch.testing.assert_close(after[0, :2], before[0, :2], rtol=0, atol=1e-5)
    assert not torch.allclose(after[0, 2], before[0, 2])
    torch.testing.assert_close(after[1:], before[1:], rtol=0, atol=1e-5)


def test_padding_rows_are_invisible() -> None:
    model = _model()
    input_ids, real = _batch(DEFAULT)
    noisy = input_ids.clone()
    noisy[~real] = JUMP
    longer = torch.cat([input_ids, torch.full_like(input_ids[:, :8], PAD)], dim=1)
    longer_mask = torch.cat([real, torch.zeros_like(real[:, :8])], dim=1)
    with torch.no_grad():
        base = model(input_ids, real)
        with_noise = model(noisy, real)
        with_more_padding = model(longer, longer_mask)

    assert bool((~real).any())
    torch.testing.assert_close(with_noise[real], base[real], rtol=0, atol=1e-5)
    torch.testing.assert_close(with_more_padding[:, : real.shape[1]][real], base[real], rtol=1e-5, atol=1e-5)


def test_eval_forward_is_deterministic_and_training_forward_uses_dropout() -> None:
    model = _model()
    input_ids, attention_mask = _batch(DEFAULT)
    with torch.no_grad():
        first = model(input_ids, attention_mask)
        second = model(input_ids, attention_mask)
        model.train()
        trained = model(input_ids, attention_mask)

    assert torch.equal(first, second)
    assert not torch.allclose(first, trained)


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk_norm"])
@pytest.mark.parametrize("device", DEVICES)
def test_every_parameter_gets_a_gradient_except_the_pad_row(device: str, qk_norm: bool) -> None:
    model = _model(config=ModelConfig(qk_norm=qk_norm)).train().to(device)
    input_ids, attention_mask = (tensor.to(device) for tensor in _batch(DEFAULT))
    logits = model(input_ids, attention_mask)
    logits[attention_mask].float().square().mean().backward()

    assert logits.device.type == device
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.any()), name
    pad_gradient = model.embedding.table.weight.grad
    assert pad_gradient is not None
    assert not bool(pad_gradient[PAD].any())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ids, mask: (ids.float(), mask), "input_ids"),
        (lambda ids, mask: (ids[..., :4], mask), "input_ids"),
        (lambda ids, mask: (ids[0], mask), "input_ids"),
        (lambda ids, mask: (ids, mask.long()), "attention_mask"),
        (lambda ids, mask: (ids, mask[:, :-1]), "attention_mask"),
    ],
)
def test_wrong_inputs_are_rejected(change, message: str) -> None:
    model = _model()
    input_ids, attention_mask = change(*_batch(DEFAULT))

    with pytest.raises(ValueError, match=message):
        model(input_ids, attention_mask)


# Attention scores and QK-norm


def _layers_of(backbone: Backbone) -> list[DecoderLayer]:
    layers = list(backbone.layers)
    assert all(isinstance(layer, DecoderLayer) for layer in layers)
    return cast("list[DecoderLayer]", layers)


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk_norm"])
def test_the_scores_are_the_ones_the_attention_uses(qk_norm: bool) -> None:
    config = ModelConfig(qk_norm=qk_norm)
    backbone = Backbone(config)
    attention = _layers_of(backbone)[0].self_attn.eval()
    torch.manual_seed(1)
    hidden = torch.randn(2, 6, config.hidden_size)
    position_embeddings = backbone.rotary_embedding(hidden, torch.arange(6)[None])
    visible = torch.ones(6, 6, dtype=torch.bool).tril()[None, None].expand(2, 1, 6, 6)
    with torch.no_grad():
        output, _ = attention(hidden, position_embeddings=position_embeddings, attention_mask=visible)
        scores = attention_scores(attention, hidden, position_embeddings)
        weights = scores.masked_fill(~visible, float("-inf")).softmax(-1)
        values = attention.v_proj(hidden).view(2, 6, -1, attention.head_dim).transpose(1, 2)
        expected = attention.o_proj((weights @ values).transpose(1, 2).reshape(2, 6, -1))

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_qk_norm_bounds_the_attention_scores() -> None:
    input_ids, attention_mask = _batch(DEFAULT)
    largest = {}
    for qk_norm in (False, True):
        model = _model(config=ModelConfig(qk_norm=qk_norm))
        with torch.no_grad():
            for layer in _layers_of(model.backbone):
                layer.self_attn.q_proj.weight.mul_(100)
                layer.self_attn.k_proj.weight.mul_(100)
            largest[qk_norm] = model.largest_attention_scores(input_ids, attention_mask)

    head = ModelConfig().hidden_size // ModelConfig().attention_heads
    # Normalized queries and keys with unit gains have length sqrt(head), so a score is at most sqrt(head).
    assert tuple(largest[True].shape) == (ModelConfig().hidden_layers,)
    assert float(largest[True].max()) <= head**0.5 + 1e-3
    assert float(largest[False].min()) > 1_000


def test_largest_attention_scores_cover_only_visible_pairs() -> None:
    model = _model()
    input_ids, attention_mask = _batch(DEFAULT)
    noisy = input_ids.clone()
    noisy[~attention_mask] = JUMP
    with torch.no_grad():
        base = model.largest_attention_scores(input_ids, attention_mask)
        with_noise = model.largest_attention_scores(noisy, attention_mask)

    assert bool((~attention_mask).any())
    torch.testing.assert_close(with_noise, base)


def test_qk_norm_uses_qwen3_attention_and_plain_uses_llama_attention() -> None:
    plain = _layers_of(Backbone(ModelConfig.mini()))[0].self_attn
    normed = _layers_of(Backbone(replace(ModelConfig.mini(), qk_norm=True)))[0].self_attn

    assert type(plain) is LlamaAttention
    assert isinstance(normed, Qwen3Attention)
    assert (normed.head_dim, normed.attention_dropout, normed.sliding_window) == (64, 0.1, None)
    assert tuple(normed.q_norm.weight.shape) == tuple(normed.k_norm.weight.shape) == (64,)
    assert normed.q_norm.variance_epsilon == pytest.approx(1e-6)
    assert {name for name, _ in plain.named_parameters()} | {"q_norm.weight", "k_norm.weight"} == {
        name for name, _ in normed.named_parameters()
    }


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk_norm"])
def test_attention_weight_records_describe_every_layer(qk_norm: bool) -> None:
    model = _model(config=ModelConfig(qk_norm=qk_norm))
    if qk_norm:
        with torch.no_grad():
            cast("Qwen3Attention", _layers_of(model.backbone)[0].self_attn).q_norm.weight.mul_(3)

    records = model.attention_weight_records()

    assert len(records) == ModelConfig().hidden_layers
    for layer, record in zip(_layers_of(model.backbone), records, strict=True):
        attention = layer.self_attn
        for name, projection in (("query", attention.q_proj), ("key", attention.k_proj)):
            weight = projection.weight.detach()
            assert record[f"{name}_norm"] == pytest.approx(float(torch.linalg.matrix_norm(weight)), rel=1e-5)
            assert record[f"{name}_top_singular_value"] == pytest.approx(
                float(torch.linalg.matrix_norm(weight, ord=2)), rel=1e-4
            )
        gains = {"query_gain_max", "query_gain_rms", "key_gain_max", "key_gain_rms"}
        assert gains <= set(record) if qk_norm else not gains & set(record)
    if qk_norm:
        assert records[0]["query_gain_max"] == pytest.approx(3.0)
        assert records[0]["query_gain_rms"] == pytest.approx(3.0)
        assert records[1]["key_gain_max"] == pytest.approx(1.0)


def test_qk_norm_builds_more_layers_than_qwen3s_default_of_32() -> None:
    backbone = Backbone(ModelConfig(hidden_size=16, hidden_layers=33, attention_heads=2, qk_norm=True))

    assert len(backbone.layers) == 33
    assert all(isinstance(layer.self_attn, Qwen3Attention) for layer in _layers_of(backbone))


# Bidirectional encoding, for the downstream tasks.


def test_encoding_is_causal_by_default() -> None:
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, replace(ModelConfig.tiny(), attention_dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, SMALL.vocab_size, (2, 6, SMALL.row_width), generator=generator)
    mask = torch.ones(2, 6, dtype=torch.bool)

    with torch.no_grad():
        default, asked = model.encode(input_ids, mask), model.encode(input_ids, mask, causal=True)

    torch.testing.assert_close(default, asked)


def test_a_later_row_cannot_change_an_earlier_one_under_causal_attention() -> None:
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, replace(ModelConfig.tiny(), attention_dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, SMALL.vocab_size, (1, 6, SMALL.row_width), generator=generator)
    changed = input_ids.clone()
    changed[0, 5] = (changed[0, 5] + 1) % SMALL.vocab_size
    mask = torch.ones(1, 6, dtype=torch.bool)

    with torch.no_grad():
        before = model.encode(input_ids, mask, causal=True)
        after = model.encode(changed, mask, causal=True)

    torch.testing.assert_close(before[:, :5], after[:, :5])


def test_a_later_row_does_change_an_earlier_one_under_bidirectional_attention() -> None:
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, replace(ModelConfig.tiny(), attention_dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, SMALL.vocab_size, (1, 6, SMALL.row_width), generator=generator)
    changed = input_ids.clone()
    changed[0, 5] = (changed[0, 5] + 1) % SMALL.vocab_size
    mask = torch.ones(1, 6, dtype=torch.bool)

    with torch.no_grad():
        before = model.encode(input_ids, mask, causal=False)
        after = model.encode(changed, mask, causal=False)

    assert not torch.allclose(before[:, :5], after[:, :5])


def test_bidirectional_attention_still_never_reads_padding() -> None:
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, replace(ModelConfig.tiny(), attention_dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, SMALL.vocab_size, (1, 6, SMALL.row_width), generator=generator)
    padded = input_ids.clone()
    padded[0, 4:] = (padded[0, 4:] + 1) % SMALL.vocab_size
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4:] = False

    with torch.no_grad():
        before = model.encode(input_ids, mask, causal=False)
        after = model.encode(padded, mask, causal=False)

    torch.testing.assert_close(before[:, :4], after[:, :4])


# Causal attention without an explicit mask, value residual, zero-initialized outputs.


def _explicit_reference(backbone: Backbone, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run the layers with the explicit causal mask, as every run before 24 September did.

    Returns:
        `(B, T, H)` hidden states after the final normalization.
    """
    position_embeddings, visible = backbone.positions_and_mask(hidden, mask, causal=True)
    first = None
    for layer in _layers_of(backbone):
        hidden, values = layer(hidden, visible, position_embeddings, first)
        if backbone.value_residual and first is None:
            first = values
    return backbone.norm(hidden)


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk_norm"])
def test_a_layer_computes_what_the_transformers_attention_computes(qk_norm: bool) -> None:
    config = ModelConfig(qk_norm=qk_norm)
    backbone = Backbone(config).eval()
    layer = _layers_of(backbone)[0]
    torch.manual_seed(1)
    hidden = torch.randn(2, 6, config.hidden_size)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    position_embeddings, visible = backbone.positions_and_mask(hidden, mask, causal=False)
    with torch.no_grad():
        output, _ = layer(hidden, visible, position_embeddings)
        update, _ = layer.self_attn(
            layer.input_layernorm(hidden), position_embeddings=position_embeddings, attention_mask=visible
        )
        expected = hidden + update
        expected = expected + layer.mlp(layer.post_attention_layernorm(expected))

    torch.testing.assert_close(output[mask], expected[mask], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("qk_norm", [False, True], ids=["plain", "qk_norm"])
@pytest.mark.parametrize("value_residual", [False, True], ids=["plain_values", "value_residual"])
def test_causal_attention_without_a_mask_matches_the_explicit_mask_on_real_rows(
    qk_norm: bool, value_residual: bool
) -> None:
    config = ModelConfig(hidden_layers=3, qk_norm=qk_norm, value_residual=value_residual)
    torch.manual_seed(0)
    backbone = Backbone(config).eval()
    hidden = torch.randn(3, 7, config.hidden_size)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2, [True] * 2 + [False] * 5])
    with torch.no_grad():
        fast = backbone(hidden, mask)
        explicit = _explicit_reference(backbone, hidden, mask)

    torch.testing.assert_close(fast[mask], explicit[mask], rtol=1e-5, atol=1e-5)


def test_causal_encoding_asks_for_the_causal_kernel_and_bidirectional_keeps_the_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = Backbone(ModelConfig(qk_norm=True)).eval()
    hidden, mask = torch.randn(2, 5, 128), torch.ones(2, 5, dtype=torch.bool)
    calls: list[tuple[bool, bool]] = []
    original = torch.nn.functional.scaled_dot_product_attention

    def recording(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((kwargs.get("attn_mask") is None, bool(kwargs.get("is_causal"))))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
    with torch.no_grad():
        backbone(hidden, mask)
        causal = calls[:]
        calls.clear()
        backbone(hidden, mask, causal=False)

    assert causal == [(True, True)] * 2
    assert calls == [(False, False)] * 2


def test_value_residual_is_off_by_default_and_adds_no_parameters() -> None:
    plain = ModelConfig.deep()
    mixed = replace(plain, value_residual=True)
    saved = plain.to_dict()
    del saved["value_residual"], saved["zero_init_outputs"]

    assert (plain.value_residual, plain.zero_init_outputs) == (False, False)
    assert ModelConfig.from_dict(saved) == plain
    assert ModelConfig.from_dict(json.loads(json.dumps(mixed.to_dict()))) == mixed
    counts = [sum(value.numel() for value in Backbone(config).parameters()) for config in (plain, mixed)]
    assert counts[0] == counts[1]


def test_value_residual_leaves_the_first_layer_alone_and_changes_the_later_ones() -> None:
    hidden, mask = torch.randn(2, 6, 128), torch.ones(2, 6, dtype=torch.bool)
    outputs = {}
    for layers in (1, 2):
        for value_residual in (False, True):
            torch.manual_seed(0)
            backbone = Backbone(ModelConfig(hidden_layers=layers, value_residual=value_residual)).eval()
            with torch.no_grad():
                outputs[layers, value_residual] = backbone(hidden, mask)

    torch.testing.assert_close(outputs[1, True], outputs[1, False])
    assert not torch.allclose(outputs[2, True], outputs[2, False])


def test_value_residual_averages_each_layers_values_with_the_first_layers() -> None:
    config = ModelConfig(hidden_layers=2, attention_dropout=0.0, value_residual=True)
    torch.manual_seed(0)
    backbone = Backbone(config).eval()
    first, second = _layers_of(backbone)
    hidden, mask = torch.randn(1, 5, 128), torch.ones(1, 5, dtype=torch.bool)
    position_embeddings, visible = backbone.positions_and_mask(hidden, mask, causal=True)
    with torch.no_grad():
        after_first, values = first(hidden, visible, position_embeddings)
        normed = second.input_layernorm(after_first)
        attention = second.self_attn
        scores = attention_scores(attention, normed, position_embeddings)
        weights = scores.masked_fill(~visible, float("-inf")).softmax(-1)
        own = attention.v_proj(normed).view(1, 5, -1, attention.head_dim).transpose(1, 2)
        update = attention.o_proj((weights @ (0.5 * (own + values))).transpose(1, 2).reshape(1, 5, -1))
        expected = after_first + update
        expected = expected + second.mlp(second.post_attention_layernorm(expected))
        output, _ = second(after_first, visible, position_embeddings, values)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_zero_initialized_outputs_make_every_layer_start_as_the_identity() -> None:
    config = ModelConfig(hidden_layers=3, zero_init_outputs=True)
    torch.manual_seed(0)
    backbone = Backbone(config).eval()
    hidden, mask = torch.randn(2, 6, 128), torch.ones(2, 6, dtype=torch.bool)

    for layer in _layers_of(backbone):
        assert not bool(layer.self_attn.o_proj.weight.any())
        assert not bool(layer.mlp.down_proj.weight.any())
        assert float(layer.self_attn.q_proj.weight.std()) == pytest.approx(0.02, rel=0.1)
        assert float(layer.mlp.up_proj.weight.std()) == pytest.approx(0.02, rel=0.1)
    with torch.no_grad():
        torch.testing.assert_close(backbone(hidden, mask), backbone.norm(hidden))


# Gated attention.


def test_gated_attention_is_off_by_default_and_adds_one_square_matrix_per_layer() -> None:
    plain = ModelConfig.deep()
    gated = replace(plain, gated_attention=True)
    saved = plain.to_dict()
    del saved["gated_attention"]

    assert not plain.gated_attention
    assert ModelConfig.from_dict(saved) == plain
    assert ModelConfig.from_dict(json.loads(json.dumps(gated.to_dict()))) == gated
    counts = [sum(value.numel() for value in Backbone(config).parameters()) for config in (plain, gated)]
    assert counts[1] - counts[0] == plain.hidden_layers * plain.hidden_size**2
    assert all(layer.attention_gate is None for layer in _layers_of(Backbone(plain)))


def test_gated_attention_scales_each_channel_of_the_attention_output_before_the_output_projection() -> None:
    config = ModelConfig(hidden_layers=1, attention_dropout=0.0, gated_attention=True)
    torch.manual_seed(0)
    backbone = Backbone(config).eval()
    layer = _layers_of(backbone)[0]
    assert layer.attention_gate is not None
    hidden, mask = torch.randn(2, 5, 128), torch.ones(2, 5, dtype=torch.bool)
    position_embeddings, visible = backbone.positions_and_mask(hidden, mask, causal=True)
    with torch.no_grad():
        normed = layer.input_layernorm(hidden)
        attention = layer.self_attn
        scores = attention_scores(attention, normed, position_embeddings)
        weights = scores.masked_fill(~visible, float("-inf")).softmax(-1)
        values = attention.v_proj(normed).view(2, 5, -1, attention.head_dim).transpose(1, 2)
        mixed = (weights @ values).transpose(1, 2).reshape(2, 5, -1)
        expected = hidden + attention.o_proj(mixed * torch.sigmoid(layer.attention_gate(normed)))
        expected = expected + layer.mlp(layer.post_attention_layernorm(expected))
        output, _ = layer(hidden, visible, position_embeddings)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_the_attention_gate_gets_a_gradient() -> None:
    torch.manual_seed(0)
    backbone = Backbone(ModelConfig(gated_attention=True, zero_init_outputs=False))
    hidden, mask = torch.randn(2, 5, 128), torch.ones(2, 5, dtype=torch.bool)

    backbone(hidden, mask).square().sum().backward()

    for layer in _layers_of(backbone):
        assert layer.attention_gate is not None
        assert layer.attention_gate.weight.grad is not None
        assert bool(layer.attention_gate.weight.grad.any())


def test_encode_layers_gives_every_layer_then_the_final_norm_as_the_hooks_see_them() -> None:
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, replace(ModelConfig.tiny(), attention_dropout=0.0, value_residual=True)).eval()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, SMALL.vocab_size, (2, 6, SMALL.row_width), generator=generator)
    mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])

    with torch.no_grad():
        layers = model.encode_layers(input_ids, mask)
        hooked = hidden_states(model, input_ids, mask)

    names = [f"layer_{index}" for index in range(1, model.config.hidden_layers + 1)] + ["final_norm"]
    assert layers.shape == (len(names), 2, 6, model.config.hidden_size)
    torch.testing.assert_close(layers, torch.stack([hooked[name] for name in names]))
