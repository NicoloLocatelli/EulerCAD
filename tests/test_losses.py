"""Tests for the losses of the GraphGPT-based architecture."""

import math
import random

import pytest
import torch
from torch.nn.functional import cross_entropy

from eulercad.data.collate import IGNORED_LABEL, collate
from eulercad.data.sample import Sample, next_row_labels
from eulercad.losses.next_row import next_row_loss
from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.representation.euler import euler_walk
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.rows import ROW_FIELDS, ROW_WIDTH
from eulercad.tokenization.tokenizer import GraphTokenizer

VOCAB = 40


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random logits and labels with the second sample's last three rows as padding.

    Returns:
        `logits` `(2, 6, 5, VOCAB)` and `labels` `(2, 6, 5)`.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 6, ROW_WIDTH, VOCAB, generator=generator)
    labels = torch.randint(0, VOCAB, (2, 6, ROW_WIDTH), generator=generator)
    labels[1, 3:] = IGNORED_LABEL
    return logits, labels


def test_the_row_fields_name_the_row_layout() -> None:
    assert ROW_FIELDS == ("node_id", "x_bin", "y_bin", "edge_type", "distance_bin")
    assert len(ROW_FIELDS) == ROW_WIDTH


def test_the_total_is_the_mean_cross_entropy_over_labelled_cells() -> None:
    logits, labels = _inputs()
    scored = labels != IGNORED_LABEL
    expected = cross_entropy(logits[scored], labels[scored])

    torch.testing.assert_close(next_row_loss(logits, labels).total, expected)


def test_each_field_part_is_that_field_s_mean_cross_entropy() -> None:
    logits, labels = _inputs()
    loss = next_row_loss(logits, labels)

    assert tuple(loss.parts) == ROW_FIELDS
    for field, name in enumerate(ROW_FIELDS):
        expected = cross_entropy(logits[..., field, :].flatten(0, 1), labels[..., field].flatten())
        torch.testing.assert_close(loss.parts[name], expected)
    # Every labelled row labels all five fields, so the total is the mean of the parts.
    torch.testing.assert_close(loss.total, torch.stack(list(loss.parts.values())).mean())


def test_ignored_cells_change_nothing_and_get_no_gradient() -> None:
    logits, labels = _inputs()
    changed = logits.clone()
    changed[1, 3:] = 100.0
    logits.requires_grad_(True)
    loss = next_row_loss(logits, labels)
    loss.total.backward()

    torch.testing.assert_close(next_row_loss(changed, labels).total, loss.total.detach())
    assert logits.grad is not None
    assert not bool(logits.grad[1, 3:].any())
    assert bool(logits.grad[0].any())


def test_the_loss_is_float32_for_low_precision_logits() -> None:
    logits, labels = _inputs()
    loss = next_row_loss(logits.bfloat16(), labels)

    assert loss.total.dtype == torch.float32
    assert all(part.dtype == torch.float32 for part in loss.parts.values())
    torch.testing.assert_close(loss.total, next_row_loss(logits.bfloat16().float(), labels).total)


def test_every_part_carries_a_gradient() -> None:
    logits, labels = _inputs()
    logits.requires_grad_(True)
    loss = next_row_loss(logits, labels)

    for field, name in enumerate(ROW_FIELDS):
        (gradient,) = torch.autograd.grad(loss.parts[name], logits, retain_graph=True)
        assert bool(gradient[..., field, :].any())
        others = [index for index in range(ROW_WIDTH) if index != field]
        assert not bool(gradient[..., others, :].any())


@pytest.mark.parametrize(
    ("logits", "labels"),
    [
        (torch.zeros(2, 6, ROW_WIDTH, VOCAB), torch.zeros(2, 6, ROW_WIDTH - 1, dtype=torch.int64)),
        (torch.zeros(2, 6, ROW_WIDTH, VOCAB), torch.zeros(2, 5, ROW_WIDTH, dtype=torch.int64)),
        (torch.zeros(6, ROW_WIDTH, VOCAB), torch.zeros(6, ROW_WIDTH, dtype=torch.int64)),
        (torch.zeros(2, 6, ROW_WIDTH, VOCAB), torch.zeros(2, 6, ROW_WIDTH)),
    ],
)
def test_wrong_shapes_are_rejected(logits: torch.Tensor, labels: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        next_row_loss(logits, labels)


def test_a_batch_without_labelled_cells_is_rejected() -> None:
    logits, labels = _inputs()
    labels[:] = IGNORED_LABEL

    with pytest.raises(ValueError, match="no labelled cell"):
        next_row_loss(logits, labels)


@pytest.mark.parametrize("field", range(ROW_WIDTH))
def test_a_field_without_labelled_cells_is_rejected(field: int) -> None:
    logits, labels = _inputs()
    labels[..., field] = IGNORED_LABEL

    with pytest.raises(ValueError, match=f"{ROW_FIELDS[field]} has no labelled cell"):
        next_row_loss(logits, labels)


def test_an_untrained_model_scores_near_the_uniform_loss() -> None:
    tokenizer = GraphTokenizer()
    positions = torch.tensor([[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0, 1, 2, 3], [1, 3, 2, 3, 4]])
    graph = DrawingGraph(0, positions, edges, (0.0, 0.0), 1.0)
    samples = []
    for seed in range(3):
        rng = random.Random(seed)  # noqa: S311  # nosec B311
        rows = tokenizer.tokenize(graph, euler_walk(graph, rng), rng).rows
        samples.append(Sample(rows, next_row_labels(rows), seed))
    batch = collate(samples)
    torch.manual_seed(0)
    model = GraphGPTModel(tokenizer, ModelConfig.tiny()).eval()
    with torch.no_grad():
        loss = next_row_loss(model(batch.input_ids, batch.attention_mask), batch.labels)

    # Initial weights are small, so the logits are nearly equal and the loss is near log(V).
    assert loss.total.item() == pytest.approx(math.log(tokenizer.vocab_size), abs=0.05)
