"""Tests for Attention Residuals in the GraphGPT-based architecture (`model/depth_attention.py`)."""

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest
import torch

from eulercad.model.backbone import Backbone, DecoderLayer
from eulercad.model.config import ModelConfig
from eulercad.model.depth_attention import DepthAttention
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.tokenization.vocabulary import JUMP, PAD
from eulercad.training.config import TrainingConfig
from eulercad.training.optimizer import MuonAdamW, make_optimizer

SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
EPS = 1e-6


def _reference_read(
    query: torch.Tensor, embedding: torch.Tensor, outputs: list[torch.Tensor], block: int
) -> torch.Tensor:
    """The input after `outputs`, from the paper's equations, with `torch.stack` as in its pseudocode.

    The sources are the embedding, then the sums of the outputs in groups of `block`; the last
    group may be unfinished (the partial block). With `block` 1, every output is its own source.

    Returns:
        `sum_i softmax_i(query . RMSNorm(source_i)) source_i`.
    """
    sources = torch.stack(
        [embedding] + [torch.stack(outputs[i : i + block]).sum(0) for i in range(0, len(outputs), block)]
    )
    keys = sources * torch.rsqrt(sources.square().mean(-1, keepdim=True) + EPS)
    weights = torch.einsum("d,n...d->n...", query, keys).softmax(0)
    return torch.einsum("n...,n...d->...d", weights, sources)


def _reads(attention: DepthAttention, embedding: torch.Tensor, outputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    stream = attention.stream(embedding)
    reads = []
    for output in outputs:
        reads.append(stream.read())
        stream.write(output)
    return [*reads, stream.read()]


def _layers_of(backbone: Backbone) -> list[DecoderLayer]:
    return [cast("DecoderLayer", layer) for layer in backbone.layers]


def _depth(backbone: Backbone) -> DepthAttention:
    assert backbone.depth_attention is not None
    return backbone.depth_attention


# DepthAttention alone.


def test_the_queries_start_at_zero_one_per_sublayer_and_one_for_the_output() -> None:
    attention = DepthAttention(16, sublayers=6, block=1, eps=EPS)

    assert attention.queries.shape == (7, 16)
    assert not bool(attention.queries.any())


@pytest.mark.parametrize("block", [1, 2, 3])
def test_zero_queries_read_the_plain_average_of_the_sources(block: int) -> None:
    torch.manual_seed(0)
    embedding, outputs = torch.randn(2, 5, 16), list(torch.randn(6, 2, 5, 16))

    reads = _reads(DepthAttention(16, sublayers=6, block=block, eps=EPS), embedding, outputs)

    for count, read in enumerate(reads):
        done = outputs[:count]
        sources = [embedding] + [torch.stack(done[i : i + block]).sum(0) for i in range(0, count, block)]
        torch.testing.assert_close(read, torch.stack(sources).mean(0))


@pytest.mark.parametrize("block", [1, 2, 4])
def test_every_read_matches_the_papers_equations(block: int) -> None:
    torch.manual_seed(0)
    attention = DepthAttention(16, sublayers=6, block=block, eps=EPS)
    with torch.no_grad():
        attention.queries.normal_()
    embedding, outputs = torch.randn(2, 5, 16), list(torch.randn(6, 2, 5, 16) * 3)

    reads = _reads(attention, embedding, outputs)

    assert len(reads) == 7
    for count, read in enumerate(reads):
        expected = _reference_read(attention.queries[count].detach(), embedding, outputs[:count], block)
        torch.testing.assert_close(read, expected, rtol=1e-5, atol=1e-5)


def test_a_zero_output_is_a_source_with_a_zero_score() -> None:
    attention = DepthAttention(4, sublayers=1, block=1, eps=EPS)
    with torch.no_grad():
        attention.queries.fill_(5.0)
    embedding = torch.ones(1, 1, 4)

    stream = attention.stream(embedding)
    stream.read()
    stream.write(torch.zeros(1, 1, 4))
    read = stream.read()

    assert bool(torch.isfinite(read).all())
    weight = torch.sigmoid(torch.tensor(20.0))  # scores 5 * 4 (four ones of RMS 1) and 0
    torch.testing.assert_close(read, embedding * weight)


def test_the_queries_get_a_gradient() -> None:
    torch.manual_seed(0)
    attention = DepthAttention(16, sublayers=4, block=1, eps=EPS)
    embedding, outputs = torch.randn(2, 5, 16), list(torch.randn(4, 2, 5, 16))

    torch.stack(_reads(attention, embedding, outputs)).square().sum().backward()

    assert attention.queries.grad is not None
    assert bool(attention.queries.grad[1:].any(dim=1).all())


# Settings.


def test_attention_residuals_are_off_by_default_and_old_settings_load() -> None:
    plain = ModelConfig.deep()
    saved = plain.to_dict()
    del saved["attention_residuals"], saved["attention_residual_block"]
    on = replace(plain, attention_residuals=True, attention_residual_block=4)

    assert (plain.attention_residuals, plain.attention_residual_block) == (False, 1)
    assert ModelConfig.from_dict(saved) == plain
    assert ModelConfig.from_dict(json.loads(json.dumps(on.to_dict()))) == on
    assert Backbone(plain).depth_attention is None


@pytest.mark.parametrize(
    "settings",
    [
        {"attention_residuals": True, "attention_residual_block": 0},
        {"attention_residual_block": 2},
    ],
)
def test_bad_attention_residual_settings_are_rejected(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="attention_residual"):
        ModelConfig(**settings)  # ty: ignore[invalid-argument-type]


def test_attention_residuals_add_one_vector_per_sublayer_and_one_for_the_output() -> None:
    plain = ModelConfig.deep()
    counts = [
        sum(value.numel() for value in Backbone(config).parameters())
        for config in (plain, replace(plain, attention_residuals=True))
    ]

    assert counts[1] - counts[0] == (2 * plain.hidden_layers + 1) * plain.hidden_size


# The backbone.


def test_the_backbone_reads_every_sublayer_input_from_the_depth_attention() -> None:
    config = ModelConfig(hidden_layers=3, attention_dropout=0.0, attention_residuals=True, attention_residual_block=2)
    torch.manual_seed(0)
    backbone = Backbone(config).eval()
    depth = _depth(backbone)
    with torch.no_grad():
        depth.queries.normal_()
    hidden, mask = torch.randn(2, 5, 128), torch.ones(2, 5, dtype=torch.bool)
    position_embeddings, _ = backbone.positions_and_mask(hidden, mask, causal=True)

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for layer in _layers_of(backbone):
            read = _reference_read(depth.queries[len(outputs)], hidden, outputs, 2)
            outputs.append(layer.attention_update(read, None, position_embeddings, None)[0])
            outputs.append(layer.mlp_update(_reference_read(depth.queries[len(outputs)], hidden, outputs, 2)))
        expected = backbone.norm(_reference_read(depth.queries[len(outputs)], hidden, outputs, 2))
        output = backbone(hidden, mask)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_with_zero_outputs_and_zero_queries_the_backbone_starts_as_the_identity() -> None:
    config = ModelConfig(hidden_layers=3, zero_init_outputs=True, attention_residuals=True)
    torch.manual_seed(0)
    backbone = Backbone(config).eval()
    hidden, mask = torch.randn(2, 6, 128), torch.ones(2, 6, dtype=torch.bool)

    with torch.no_grad():
        torch.testing.assert_close(backbone(hidden, mask), backbone.norm(hidden), rtol=1e-4, atol=1e-4)


def _model(config: ModelConfig) -> GraphGPTModel:
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, config).eval()
    with torch.no_grad():
        _depth(model.backbone).queries.normal_()
    return model


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, SMALL.vocab_size, (2, 7, SMALL.row_width), generator=generator)
    attention_mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    input_ids[~attention_mask] = PAD
    return input_ids, attention_mask


@pytest.mark.parametrize("block", [1, 2])
def test_attention_residuals_keep_attention_causal(block: int) -> None:
    model = _model(ModelConfig(attention_residuals=True, attention_residual_block=block))
    input_ids, attention_mask = _batch()
    changed = input_ids.clone()
    changed[0, 3] = changed[0, 3] % (SMALL.vocab_size - 1) + 1
    with torch.no_grad():
        before, after = model(input_ids, attention_mask), model(changed, attention_mask)

    torch.testing.assert_close(after[0, :3], before[0, :3], rtol=0, atol=1e-5)
    assert not torch.allclose(after[0, 3], before[0, 3])


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "bidirectional"])
def test_attention_residuals_never_read_padding(causal: bool) -> None:
    model = _model(ModelConfig(attention_residuals=True, attention_residual_block=2))
    input_ids, attention_mask = _batch()
    noisy = input_ids.clone()
    noisy[~attention_mask] = JUMP
    with torch.no_grad():
        base = model.encode(input_ids, attention_mask, causal=causal)
        with_noise = model.encode(noisy, attention_mask, causal=causal)
        scores = model.largest_attention_scores(input_ids, attention_mask)
        noisy_scores = model.largest_attention_scores(noisy, attention_mask)

    torch.testing.assert_close(with_noise[attention_mask], base[attention_mask], rtol=0, atol=1e-5)
    torch.testing.assert_close(noisy_scores, scores)
    assert scores.shape == (model.config.hidden_layers,)


def test_every_parameter_gets_a_gradient_with_attention_residuals() -> None:
    model = _model(ModelConfig(attention_residuals=True, gated_attention=True, value_residual=True)).train()
    input_ids, attention_mask = _batch()

    model(input_ids, attention_mask)[attention_mask].float().square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.any()), name


def test_the_depth_queries_go_to_adamw_not_muon() -> None:
    model = _model(ModelConfig(qk_norm=True, attention_residuals=True))
    config = TrainingConfig(total_steps=10, warmup_steps=2, learning_rate=1e-3, optimizer="muon")

    optimizer = make_optimizer(model, config)

    assert isinstance(optimizer, MuonAdamW)
    adamw = {id(value) for group in optimizer.adamw.param_groups for value in group["params"]}
    assert id(_depth(model.backbone).queries) in adamw
