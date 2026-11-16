"""Tests for the training stage of the GraphGPT-based architecture."""

import json
import math
import random
import shutil
import subprocess  # noqa: S404  # nosec B404
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchao.float8.float8_linear import Float8Linear

from eulercad.abc.point_graph import write_chunk
from eulercad.data.collate import collate
from eulercad.data.dataset import DrawingDataset
from eulercad.data.sample import Sample
from eulercad.data.stream import TrainingStream
from eulercad.losses.next_row import next_row_loss
from eulercad.losses.sigreg import sigreg, standardize
from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.rows import ROW_FIELDS
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.tokenization.vocabulary import PAD
from eulercad.training.artifacts import (
    CHECKPOINTS,
    LOG_FILE,
    SOURCES,
    RunConflictError,
    RunRecorder,
    checkpoint_step,
    load_model,
    open_run,
    run_record,
    source_digest,
)
from eulercad.training.config import FLOAT8_RECIPES, TrainingConfig
from eulercad.training.microbatch import (
    grouped_loss,
    grouped_loss_and_drawings,
    length_groups,
)
from eulercad.training.module import PretrainingModule, make_trainer, sigreg_parts
from eulercad.training.optimizer import MuonAdamW
from eulercad.training.schedule import WarmupDecay, learning_rate

SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
OTHER = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=32, distance_bins=32, d_min=0.01, d_max=2.0))
PACKAGE = Path(__file__).resolve().parents[1]
TINY = ModelConfig.tiny()


def _drawing(sketch_id: int, num_nodes: int) -> Data:
    generator = random.Random(sketch_id)  # noqa: S311  # nosec B311
    pos = [[generator.uniform(-50, 50), generator.uniform(-50, 50)] for _ in range(num_nodes)]
    pairs = {tuple(sorted(generator.sample(range(num_nodes), 2))) for _ in range(num_nodes + 3)}
    edges = torch.tensor(sorted(pairs), dtype=torch.int64).T
    return Data(
        pos=torch.tensor(pos, dtype=torch.float64),
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
        sketch_id=torch.tensor([sketch_id]),
    )


def _split(root: Path, sizes: list[int]) -> Path:
    root.mkdir(parents=True)
    write_chunk([_drawing(100 + index, nodes) for index, nodes in enumerate(sizes)], root / "chunk_000.pt")
    return root


@pytest.fixture(scope="module")
def splits(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("drawings")
    return _split(root / "train", [4, 9, 6, 14, 5, 11, 7, 3]), _split(root / "val", [5, 12, 8])


def _config(**settings: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "batch_size": 4,
        "total_steps": 4,
        "warmup_steps": 2,
        "precision": "32-true",
        "cell_budget": 64,
    }
    return TrainingConfig(**defaults | settings)


def _batch(sizes: list[int], seed: int = 0) -> Any:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for index, rows in enumerate(sizes):
        input_ids = torch.randint(PAD + 1, SMALL.vocab_size, (rows, 5), generator=generator)
        labels = torch.randint(PAD + 1, SMALL.vocab_size, (rows, 5), generator=generator)
        samples.append(Sample(input_ids, labels, index))
    return collate(samples)


# Configuration


def test_training_config_round_trips() -> None:
    config = _config(learning_rate=1e-3, adam_betas=(0.8, 0.9))

    assert TrainingConfig.from_dict(config.to_dict()) == config
    assert TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


def test_training_config_defaults_are_the_baseline_s() -> None:
    config = TrainingConfig(total_steps=10, warmup_steps=1)

    assert (config.batch_size, config.learning_rate, config.weight_decay) == (160, 3e-4, 0.1)
    assert (config.adam_betas, config.adam_epsilon, config.max_gradient_norm) == ((0.9, 0.95), 1e-8, 5.0)
    assert (config.precision, config.seed) == ("bf16-mixed", 0)
    assert (config.cell_budget, config.compile_backend, config.float8) == (8192, None, None)
    assert config.optimizer == "adamw"
    assert (config.schedule, config.final_learning_rate_ratio) == ("linear", 0.1)


def test_training_config_written_before_the_schedule_fields_still_loads() -> None:
    saved = _config().to_dict()
    del saved["schedule"], saved["final_learning_rate_ratio"]

    assert TrainingConfig.from_dict(saved) == _config()


def test_training_config_keeps_the_float8_recipe() -> None:
    config = _config(float8="rowwise")

    assert TrainingConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 0},
        {"total_steps": 0},
        {"warmup_steps": -1},
        {"warmup_steps": 4},
        {"learning_rate": 0.0},
        {"weight_decay": -0.1},
        {"adam_betas": (0.9, 1.0)},
        {"adam_epsilon": 0.0},
        {"max_gradient_norm": 0.0},
        {"precision": "16-mixed"},
        {"cell_budget": 0},
        {"compile_backend": ""},
        {"float8": "fp8"},
        {"optimizer": "sgd"},
        {"schedule": "step"},
        {"final_learning_rate_ratio": -0.1},
        {"final_learning_rate_ratio": 1.5},
    ],
)
def test_training_config_rejects_bad_values(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**settings)


# FP8


def test_float8_converts_the_layer_weights_and_keeps_the_rest() -> None:
    model = GraphGPTModel(SMALL, replace(TINY, geometry_features=True))
    names = set(model.state_dict())

    PretrainingModule(model, _config(float8="tensorwise"))

    converted = {name for name, module in model.named_modules() if isinstance(module, Float8Linear)}
    layer = "backbone.layers.0"
    assert {f"{layer}.self_attn.q_proj", f"{layer}.mlp.down_proj", "head.field_projection"} <= converted
    # 41 inputs and SMALL's vocabulary are not multiples of 16, which FP8 products need.
    assert type(model.head.vocabulary_projection) is torch.nn.Linear
    assert model.geometry is not None
    assert type(model.geometry.projection) is torch.nn.Linear
    assert set(model.state_dict()) == names


def test_without_float8_every_linear_layer_stays_plain() -> None:
    model = GraphGPTModel(SMALL, TINY)

    PretrainingModule(model, _config())

    assert not any(isinstance(module, Float8Linear) for module in model.modules())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 needs compute capability 8.9",
)
@pytest.mark.parametrize("recipe", FLOAT8_RECIPES)
def test_float8_loss_stays_close_to_the_bfloat16_loss(recipe: str) -> None:
    torch.manual_seed(0)
    plain = GraphGPTModel(SMALL, TINY).cuda()
    converted = GraphGPTModel(SMALL, TINY).cuda()
    converted.load_state_dict(plain.state_dict())
    PretrainingModule(converted, _config(float8=recipe))
    batch = _batch([40, 64, 23, 57])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = [
            next_row_loss(model(batch.input_ids.cuda(), batch.attention_mask.cuda()), batch.labels.cuda()).total
            for model in (plain, converted)
        ]
    losses[1].backward()

    assert losses[1].item() == pytest.approx(losses[0].item(), rel=0.02)
    assert all(parameter.grad is not None for parameter in converted.parameters() if parameter.requires_grad)


# Schedule


def test_learning_rate_follows_the_baseline_warmup_and_decay() -> None:
    def rate(step: int) -> float:
        return learning_rate(step, peak=1.0, total_steps=100, warmup_steps=10)

    # The first update uses the peak; then a logarithmic warmup from a tenth of it.
    assert rate(0) == pytest.approx(1.0)
    assert rate(1) == pytest.approx(0.1)
    assert rate(5) == pytest.approx(0.1 + 0.9 * math.log(5) / math.log(10))
    assert rate(11) == pytest.approx(1.0)
    assert rate(56) == pytest.approx(0.1 + 0.9 * 45 / 90)
    assert rate(101) == pytest.approx(0.1)
    assert rate(500) == pytest.approx(0.1)


def test_a_linear_decay_can_end_at_zero_and_keeps_its_warmup() -> None:
    def rate(step: int) -> float:
        return learning_rate(step, peak=1.0, total_steps=100, warmup_steps=10, final_ratio=0.0)

    assert rate(1) == pytest.approx(0.1)
    assert rate(11) == pytest.approx(1.0)
    assert rate(56) == pytest.approx(45 / 90)
    assert rate(101) == pytest.approx(0.0)
    assert rate(500) == pytest.approx(0.0)


def test_a_cosine_decay_follows_half_a_cosine_after_the_warmup() -> None:
    def rate(step: int) -> float:
        return learning_rate(step, peak=1.0, total_steps=100, warmup_steps=10, schedule="cosine", final_ratio=0.0)

    assert rate(1) == pytest.approx(0.1)
    assert rate(11) == pytest.approx(1.0)
    assert rate(33) == pytest.approx(0.5 * (1 + math.cos(math.pi * 22 / 90)))
    assert rate(56) == pytest.approx(0.5)
    assert rate(101) == pytest.approx(0.0)
    assert rate(500) == pytest.approx(0.0)


@pytest.mark.parametrize(("schedule", "halfway"), [("wsd", 0.5), ("wsd_sqrt", 1 - math.sqrt(0.5))])
def test_wsd_holds_the_peak_then_decays_over_the_last_fifth(schedule: str, halfway: float) -> None:
    def rate(step: int) -> float:
        return learning_rate(step, peak=1.0, total_steps=100, warmup_steps=10, schedule=schedule, final_ratio=0.0)

    assert rate(1) == pytest.approx(0.1)
    assert rate(11) == pytest.approx(1.0)
    assert rate(50) == pytest.approx(1.0)
    assert rate(81) == pytest.approx(1.0)
    assert rate(91) == pytest.approx(halfway)
    assert rate(101) == pytest.approx(0.0)
    assert rate(500) == pytest.approx(0.0)


def test_an_unknown_schedule_is_refused() -> None:
    with pytest.raises(ValueError):
        learning_rate(5, peak=1.0, total_steps=100, warmup_steps=10, schedule="step")


def test_warmup_decay_follows_the_chosen_schedule() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    scheduler = WarmupDecay(optimizer, peak=2.0, total_steps=20, warmup_steps=5, schedule="wsd", final_ratio=0.0)

    rates = []
    for _ in range(25):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    expected = [
        learning_rate(step, peak=2.0, total_steps=20, warmup_steps=5, schedule="wsd", final_ratio=0.0)
        for step in range(25)
    ]
    assert rates == pytest.approx(expected)


def test_warmup_decay_sets_the_rate_of_each_update() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    scheduler = WarmupDecay(optimizer, peak=2.0, total_steps=20, warmup_steps=5)

    rates = []
    for _ in range(25):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates == pytest.approx([learning_rate(step, peak=2.0, total_steps=20, warmup_steps=5) for step in range(25)])


# Length groups


def test_length_groups_partition_the_batch_within_the_budget() -> None:
    batch = _batch([3, 17, 9, 2, 30, 8])

    groups = length_groups(batch.attention_mask, cell_budget=48)

    assert sorted(torch.cat(groups).tolist()) == list(range(6))
    lengths = batch.attention_mask.sum(dim=1)
    for group in groups:
        padded = math.ceil(int(lengths[group].max()) / 8) * 8
        assert len(group) == 1 or len(group) * padded <= 48
    assert [lengths[group].tolist() for group in groups] == [[2, 3, 8], [9, 17], [30]]


def test_a_sample_longer_than_the_budget_gets_its_own_group() -> None:
    batch = _batch([40, 2])

    groups = length_groups(batch.attention_mask, cell_budget=16)

    assert [group.tolist() for group in groups] == [[1], [0]]


def test_grouped_loss_equals_the_single_pass_loss_and_gradients() -> None:
    batch = _batch([3, 17, 9, 2, 30, 8])
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, ModelConfig(attention_dropout=0.0))

    single = next_row_loss(model(batch.input_ids, batch.attention_mask), batch.labels)
    single.total.backward()
    expected = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    model.zero_grad()
    grouped = grouped_loss(model, batch, cell_budget=48)
    grouped.total.backward()

    assert grouped.total.item() == pytest.approx(single.total.item(), rel=1e-5)
    for name in ROW_FIELDS:
        assert grouped.parts[name].item() == pytest.approx(single.parts[name].item(), rel=1e-5)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=1e-7)


def test_groups_padded_to_16_rows_keep_the_single_pass_loss() -> None:
    # The longest drawing has 21 rows, so the batch is 24 wide: a multiple of 8, not of 16.
    batch = _batch([3, 17, 9, 2, 21, 8])
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, ModelConfig(attention_dropout=0.0))
    shapes = []

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        shapes.append(input_ids.shape[1])
        return model(input_ids, attention_mask)

    single = next_row_loss(model(batch.input_ids, batch.attention_mask), batch.labels)
    grouped = grouped_loss(forward, batch, cell_budget=48, pad_to_multiple_of=16)

    assert batch.input_ids.shape[1] == 24
    assert shapes
    assert all(rows % 16 == 0 for rows in shapes)
    assert grouped.total.item() == pytest.approx(single.total.item(), rel=1e-5)


def test_float8_pads_every_group_to_16_rows() -> None:
    model = GraphGPTModel(SMALL, TINY)
    module = PretrainingModule(model, _config(float8="tensorwise", cell_budget=48))
    shapes = []
    model.register_forward_pre_hook(lambda _, inputs: shapes.append(inputs[0].shape[1]))
    batch = _batch([3, 17, 9, 2, 21, 8])
    with torch.no_grad():
        module.batch_loss(batch)

    assert shapes
    assert all(rows % 16 == 0 for rows in shapes)


def test_float8_needs_a_cell_budget() -> None:
    with pytest.raises(ValueError, match="cell_budget"):
        _config(float8="tensorwise", cell_budget=None)


# Runs


def _fit(
    directory: Path,
    splits: tuple[Path, Path],
    *,
    stop: int = 4,
    resume: bool = False,
    tokenizer: GraphTokenizer = SMALL,
    config: TrainingConfig | None = None,
    validate_every: int | None = None,
    log_every: int = 1,
) -> PretrainingModule:
    config = config or _config()
    record = run_record(
        tokenizer=tokenizer,
        model_config=TINY,
        config=config,
        data={"train": str(splits[0]), "val": str(splits[1])},
        package=PACKAGE,
    )
    checkpoint = open_run(directory, record, resume=resume)
    torch.manual_seed(config.seed)
    module = PretrainingModule(GraphGPTModel(tokenizer, TINY), config)
    train = DrawingDataset(splits[0], tokenizer, seed=config.seed, resample_each_epoch=True)
    start = checkpoint_step(checkpoint) if checkpoint is not None else 0
    # Own generators: a loader without one draws its base seed from the global state, which drives dropout.
    stream = TrainingStream(train, config.batch_size, start=start)
    loader = DataLoader(stream, batch_size=None, generator=torch.Generator().manual_seed(config.seed))
    val = DrawingDataset(splits[1], tokenizer, seed=config.seed, resample_each_epoch=False)
    generator = torch.Generator().manual_seed(config.seed)
    val_loader = DataLoader(val, batch_size=2, collate_fn=collate, generator=generator) if validate_every else None
    recorder = RunRecorder(directory, log_every=log_every, checkpoint_every=2)
    trainer = make_trainer(
        config, accelerator="cpu", callbacks=[recorder], max_steps=stop, validate_every=validate_every
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=val_loader, ckpt_path=checkpoint)
    return module


def _rows(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / LOG_FILE).read_text(encoding="utf-8").splitlines()]


def _checkpoint(directory: Path, name: str = "last.ckpt") -> dict[str, Any]:
    return torch.load(directory / CHECKPOINTS / name, map_location="cpu", weights_only=True)


def test_a_short_fit_updates_every_parameter_and_records_every_loss_part(
    tmp_path: Path, splits: tuple[Path, Path]
) -> None:
    torch.manual_seed(0)
    initial = GraphGPTModel(SMALL, TINY).state_dict()

    module = _fit(tmp_path, splits)

    for name, value in module.model.state_dict().items():
        changed = ~torch.isclose(value, initial[name])
        if name == "embedding.table.weight":
            assert not changed[PAD].any()
            changed = changed[PAD + 1 :]
        assert changed.any(), name
    steps = [row for row in _rows(tmp_path) if "event" not in row]
    assert [row["step"] for row in steps] == [1, 2, 3, 4]
    assert all(set(row["loss_parts"]) == set(ROW_FIELDS) for row in steps)
    assert all(0 <= row["padding_share"] < 1 for row in steps)
    assert [row["learning_rate"] for row in steps] == pytest.approx(
        [learning_rate(step, peak=3e-4, total_steps=4, warmup_steps=2) for step in range(4)]
    )
    assert TrainingConfig.from_dict(json.loads((tmp_path / "run.json").read_text())["training"]) == _config()


def test_log_lines_record_the_gradient_norms_before_clipping(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    module = _fit(tmp_path, splits, config=_config(max_gradient_norm=1e-6))

    names = [name for name, _ in module.model.named_parameters()]
    steps = [row for row in _rows(tmp_path) if "event" not in row]
    assert len(steps) == 4
    blocks = {"embedding", "merge", "backbone.layers.0", "backbone.layers.1", "backbone.norm", "head"}
    for row in steps:
        assert list(row["gradient_norms"]) == names
        total = math.sqrt(sum(norm**2 for norm in row["gradient_norms"].values()))
        assert row["gradient_norm"] == pytest.approx(total, rel=1e-5)
        # The clip is 1e-6: a larger norm was read before clipping.
        assert row["gradient_norm"] > 1e-3
        assert row["clip_ratio"] == pytest.approx(1e-6 / row["gradient_norm"])
        assert set(row["gradient_norms_by_block"]) == blocks
        layer = [norm for name, norm in row["gradient_norms"].items() if name.startswith("backbone.layers.1.")]
        assert row["gradient_norms_by_block"]["backbone.layers.1"] == pytest.approx(
            math.sqrt(sum(norm**2 for norm in layer)), rel=1e-5
        )


def test_the_clip_ratio_is_one_below_the_threshold(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits, config=_config(max_gradient_norm=1e6))

    assert [row["clip_ratio"] for row in _rows(tmp_path) if "event" not in row] == pytest.approx([1.0] * 4)


def test_log_lines_record_the_attention_weights_of_each_layer(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    module = _fit(tmp_path, splits)

    row = [row for row in _rows(tmp_path) if "event" not in row][-1]
    assert row["attention_weights"] == [
        pytest.approx(record, rel=1e-5) for record in module.model.attention_weight_records()
    ]


def test_log_lines_record_the_largest_attention_score_of_each_layer(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    module = _fit(tmp_path, splits)

    stream = TrainingStream(DrawingDataset(splits[0], SMALL, seed=0, resample_each_epoch=True), batch_size=4)
    last = stream.batch(3)
    with torch.no_grad():
        expected = module.model.eval().largest_attention_scores(last.input_ids[:8], last.attention_mask[:8])
    row = [row for row in _rows(tmp_path) if "event" not in row][-1]
    assert row["largest_attention_scores"] == pytest.approx(expected.tolist(), rel=1e-5)
    assert len(row["largest_attention_scores"]) == TINY.hidden_layers


def test_the_measurements_change_no_update(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path / "every", splits, log_every=1)
    _fit(tmp_path / "rarely", splits, log_every=3)

    every, rarely = _checkpoint(tmp_path / "every"), _checkpoint(tmp_path / "rarely")
    for name, value in every["state_dict"].items():
        assert torch.equal(rarely["state_dict"][name], value), name


def test_tokens_count_every_real_row_the_model_has_seen(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits)

    stream = TrainingStream(DrawingDataset(splits[0], SMALL, seed=0, resample_each_epoch=True), batch_size=4)
    expected = torch.stack([stream.batch(index).attention_mask.sum() for index in range(4)]).cumsum(0).tolist()
    rows = _rows(tmp_path)
    assert [row["tokens"] for row in rows if "event" not in row] == expected
    assert rows[0]["tokens"] == 0
    assert rows[-1]["event"] == "end"
    assert rows[-1]["tokens"] == expected[-1]


def test_resumed_run_matches_an_uninterrupted_run(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path / "straight", splits)
    _fit(tmp_path / "split", splits, stop=2)
    _fit(tmp_path / "split", splits, resume=True)

    straight, split = _checkpoint(tmp_path / "straight"), _checkpoint(tmp_path / "split")
    assert straight["global_step"] == split["global_step"] == 4
    for name, value in straight["state_dict"].items():
        assert torch.equal(split["state_dict"][name], value), name
    losses = [
        [row["loss"] for row in _rows(tmp_path / name) if row.get("step", 0) > 2 and "event" not in row]
        for name in ("straight", "split")
    ]
    assert len(losses[0]) == 2
    assert losses[0] == losses[1]
    tokens = [[row["tokens"] for row in _rows(tmp_path / name) if "event" not in row] for name in ("straight", "split")]
    assert tokens[0] == tokens[1]
    events = [row.get("event") for row in _rows(tmp_path / "split")]
    assert events == ["start", None, None, "end", "resume", None, None, "end"]


def test_a_muon_run_resumes_like_an_uninterrupted_one(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    config = _config(optimizer="muon")
    module = _fit(tmp_path / "straight", splits, config=config)
    _fit(tmp_path / "split", splits, stop=2, config=config)
    _fit(tmp_path / "split", splits, resume=True, config=config)

    assert isinstance(module.trainer.optimizers[0], MuonAdamW)
    straight, split = _checkpoint(tmp_path / "straight"), _checkpoint(tmp_path / "split")
    for name, value in straight["state_dict"].items():
        assert torch.equal(split["state_dict"][name], value), name
    steps = [row for row in _rows(tmp_path / "straight") if "event" not in row]
    assert all(math.isfinite(row["loss"]) for row in steps)


def test_a_resume_keeps_the_token_count_and_the_best_validation_loss(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits, stop=2, validate_every=2)
    _fit(tmp_path, splits, resume=True, validate_every=2)

    rows = _rows(tmp_path)
    first = next(row for row in rows if row.get("event") == "validation")
    resumed = next(row for row in rows if row.get("event") == "resume")
    assert resumed["tokens"] == first["tokens"] > 0
    assert resumed["best_validation_loss"] == first["loss"]
    second = [row for row in rows if row.get("event") == "validation"][1]
    assert second["best"] == (second["loss"] < first["loss"])


def test_validation_is_the_mean_over_every_labelled_cell(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    module = _fit(tmp_path, splits, stop=2, validate_every=2)

    val = DrawingDataset(splits[1], SMALL, seed=0, resample_each_epoch=False)
    batch = collate([val[index] for index in range(len(val))])
    model = module.model.eval()
    with torch.no_grad():
        expected = next_row_loss(model(batch.input_ids, batch.attention_mask), batch.labels)
    rows = [row for row in _rows(tmp_path) if row.get("event") == "validation"]
    assert [row["step"] for row in rows] == [2]
    assert rows[0]["loss"] == pytest.approx(expected.total.item(), rel=1e-5)
    for name in ROW_FIELDS:
        assert rows[0]["loss_parts"][name] == pytest.approx(expected.parts[name].item(), rel=1e-5)
    assert (tmp_path / CHECKPOINTS / "best.ckpt").exists()


def test_a_checkpoint_rebuilds_its_tokenizer_and_model(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    module = _fit(tmp_path, splits, stop=2)

    model = load_model(tmp_path / CHECKPOINTS / "last.ckpt")

    assert model.tokenizer == SMALL
    assert model.config == TINY
    for name, value in module.model.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    assert _checkpoint(tmp_path)["training_config"] == _config().to_dict()


def test_a_checkpoint_of_another_tokenizer_is_refused(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits, stop=2)
    module = PretrainingModule(GraphGPTModel(OTHER, TINY), _config())
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="tokenizer"):
        module.on_load_checkpoint(checkpoint)


def test_resume_refuses_a_changed_setting(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits, stop=1)

    with pytest.raises(RunConflictError, match="training"):
        _fit(tmp_path, splits, resume=True, config=_config(learning_rate=1e-3))
    with pytest.raises(RunConflictError, match="training"):
        _fit(tmp_path, splits, resume=True, config=_config(cell_budget=None))
    with pytest.raises(RunConflictError, match="tokenizer"):
        _fit(tmp_path, splits, resume=True, tokenizer=OTHER)
    assert _checkpoint(tmp_path)["global_step"] == 1


def test_runs_are_never_overwritten_or_resumed_from_nothing(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    with pytest.raises(RunConflictError, match="no run to resume"):
        _fit(tmp_path, splits, stop=1, resume=True)
    _fit(tmp_path, splits, stop=1)
    with pytest.raises(RunConflictError, match="not empty"):
        _fit(tmp_path, splits, stop=1)


def test_a_new_run_refuses_a_folder_with_files_but_accepts_an_empty_one(
    tmp_path: Path, splits: tuple[Path, Path]
) -> None:
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / LOG_FILE).write_text('{"step": 7}\n', encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RunConflictError, match="not empty"):
        _fit(stale, splits, stop=1)
    assert (stale / LOG_FILE).read_text(encoding="utf-8") == '{"step": 7}\n'
    _fit(empty, splits, stop=1)
    assert _rows(empty)[0]["event"] == "start"


# Source identity


def _git(repository: Path, *arguments: str) -> None:
    git = shutil.which("git")
    assert git is not None
    identity = ("-c", "user.name=test", "-c", "user.email=test@example.com")
    subprocess.run([git, "-C", str(repository), *identity, *arguments], check=True, capture_output=True)  # noqa: S603  # nosec B603


def _repository(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")
    (root / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "first")
    return root


def test_git_state_has_no_revision_before_the_first_commit(tmp_path: Path) -> None:
    from eulercad.training.artifacts import git_state

    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    assert git_state(tmp_path) == (None, True)


def test_git_state_has_no_revision_outside_a_repository(tmp_path: Path) -> None:
    from eulercad.training.artifacts import git_state

    assert git_state(tmp_path) == (None, None)


def test_the_source_digest_ignores_the_paths_sources_excludes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "src" / "downstream").mkdir()
    paths = ("src", ":(exclude)src/downstream")
    before = source_digest(repository, paths)

    (repository / "src" / "downstream" / "task.py").write_text("z = 1\n", encoding="utf-8")

    assert source_digest(repository, paths) == before
    assert source_digest(repository, ("src",)) != before
    (repository / "src" / "code.py").write_text("x = 9\n", encoding="utf-8")
    assert source_digest(repository, paths) != before


def test_analysis_code_cannot_invalidate_a_pretraining_run() -> None:
    assert ":(exclude)src/eulercad/analysis" in SOURCES


def test_the_source_digest_ignores_documentation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "src" / "notes").mkdir()
    paths = ("src", ":(exclude)src/**/*.md", ":(exclude)src/*.md")
    (repository / "src" / "README.md").write_text("first\n", encoding="utf-8")
    (repository / "src" / "notes" / "CONTEXT.md").write_text("first\n", encoding="utf-8")
    before = source_digest(repository, paths)

    (repository / "src" / "README.md").write_text("rewritten\n", encoding="utf-8")
    (repository / "src" / "notes" / "CONTEXT.md").write_text("rewritten\n", encoding="utf-8")
    (repository / "src" / "notes" / "NEW.md").write_text("added\n", encoding="utf-8")

    assert source_digest(repository, paths) == before
    (repository / "src" / "code.py").write_text("x = 9\n", encoding="utf-8")
    assert source_digest(repository, paths) != before


def test_prose_cannot_invalidate_a_pretraining_run() -> None:
    assert ":(exclude)src/eulercad/**/*.md" in SOURCES
    assert ":(exclude)src/eulercad/*.md" in SOURCES


def test_the_source_digest_follows_the_code_not_the_commits(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = source_digest(repository, ("src",))
    (repository / "src" / "code.py").write_text("x = 3\n", encoding="utf-8")
    uncommitted = source_digest(repository, ("src",))

    assert uncommitted != first
    _git(repository, "commit", "-qam", "code")
    assert source_digest(repository, ("src",)) == uncommitted
    (repository / "notes.md").write_text("other notes\n", encoding="utf-8")
    assert source_digest(repository, ("src",)) == uncommitted
    (repository / "src" / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert source_digest(repository, ("src",)) != uncommitted
    (repository / "src" / "new.py").unlink()
    (repository / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")
    assert source_digest(repository, ("src",)) != uncommitted
    (repository / "src" / "code.py").unlink()
    assert source_digest(repository, ("src",)) != uncommitted


def test_resume_refuses_changed_source_code(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    directory = tmp_path / "run"

    def record() -> dict[str, Any]:
        return run_record(
            tokenizer=SMALL, model_config=TINY, config=_config(), data={}, package=repository, sources=("src",)
        )

    open_run(directory, record(), resume=False)
    (directory / CHECKPOINTS).mkdir()
    (directory / CHECKPOINTS / "last.ckpt").touch()
    assert open_run(directory, record(), resume=True) == directory / CHECKPOINTS / "last.ckpt"
    (repository / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")

    with pytest.raises(RunConflictError, match="source"):
        open_run(directory, record(), resume=True)
    saved = json.loads((directory / "run.json").read_text(encoding="utf-8"))["source"]
    assert set(saved) == {"revision", "uncommitted_changes", "digest", "paths"}


def test_only_the_last_checkpoints_are_kept(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits, config=_config(total_steps=14), stop=14)

    names = sorted(path.name for path in (tmp_path / CHECKPOINTS).glob("step_*.ckpt"))
    assert names == [f"step_{step:09d}.ckpt" for step in (6, 8, 10, 12, 14)]
    assert (tmp_path / CHECKPOINTS / "last.ckpt").resolve().name == names[-1]


# SIGReg fine-tunes and the WSD decay length (added 2026-10-13)


def test_training_config_has_no_sigreg_and_the_last_fifth_wsd_by_default() -> None:
    config = TrainingConfig(total_steps=10, warmup_steps=1)

    assert (config.sigreg_weight, config.wsd_decay_fraction) == (None, 0.2)


def test_training_config_written_before_sigreg_still_loads() -> None:
    saved = _config().to_dict()
    del saved["sigreg_weight"], saved["wsd_decay_fraction"]

    assert TrainingConfig.from_dict(saved) == _config()


@pytest.mark.parametrize(
    "settings",
    [{"sigreg_weight": -0.1}, {"sigreg_weight": 1.0}, {"wsd_decay_fraction": 0.0}, {"wsd_decay_fraction": 1.5}],
)
def test_training_config_rejects_a_bad_sigreg_weight_or_decay_fraction(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _config(**settings)


def test_wsd_can_decay_over_the_second_half() -> None:
    def rate(step: int) -> float:
        return learning_rate(
            step, peak=1.0, total_steps=100, warmup_steps=10, schedule="wsd", final_ratio=0.0, decay_fraction=0.5
        )

    assert rate(50) == pytest.approx(1.0)
    assert rate(76) == pytest.approx(0.5)
    assert rate(101) == pytest.approx(0.0)


def test_warmup_decay_passes_the_decay_fraction_on() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = WarmupDecay(
        optimizer, peak=1.0, total_steps=100, warmup_steps=10, schedule="wsd", final_ratio=0.0, decay_fraction=0.5
    )
    for _ in range(76):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_grouped_loss_also_returns_each_drawing_s_mean_final_state_in_batch_order() -> None:
    batch = _batch([3, 17, 9, 2, 30, 8])
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, ModelConfig(attention_dropout=0.0)).eval()

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = model.encode(input_ids, attention_mask)
        return model.head(states), states

    loss, drawings = grouped_loss_and_drawings(forward, batch, cell_budget=48)
    states = model.encode(batch.input_ids, batch.attention_mask)
    mask = batch.attention_mask[..., None]
    expected = (states * mask).sum(1) / mask.sum(1)

    assert loss.total.item() == pytest.approx(grouped_loss(model, batch, cell_budget=48).total.item(), rel=1e-5)
    torch.testing.assert_close(drawings, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_a_sigreg_fit_mixes_the_two_losses_and_records_both(
    tmp_path: Path, splits: tuple[Path, Path], weight: float
) -> None:
    _fit(tmp_path, splits, config=_config(sigreg_weight=weight))

    steps = [row for row in _rows(tmp_path) if "event" not in row]
    assert len(steps) == 4
    for row in steps:
        parts = row["loss_parts"]
        assert set(parts) == {*ROW_FIELDS, "next_row", "sigreg"}
        assert parts["sigreg"] > 0
        assert row["loss"] == pytest.approx((1 - weight) * parts["next_row"] + weight * parts["sigreg"], rel=1e-5)


# SIGReg on every layer, raw or standardized (added 2026-10-14)


def test_training_config_reads_sigreg_on_the_standardized_final_norm_by_default() -> None:
    config = _config()

    assert (config.sigreg_layers, config.sigreg_standardize) == ("final", True)


def test_training_config_written_before_all_layer_sigreg_still_loads() -> None:
    saved = _config().to_dict()
    del saved["sigreg_layers"], saved["sigreg_standardize"]

    assert TrainingConfig.from_dict(saved) == _config()


def test_training_config_rejects_unknown_sigreg_layers() -> None:
    with pytest.raises(ValueError, match="sigreg_layers"):
        _config(sigreg_layers="middle")


def test_grouped_loss_returns_every_layer_s_drawing_means_in_batch_order() -> None:
    batch = _batch([3, 17, 9, 2, 30, 8])
    torch.manual_seed(0)
    model = GraphGPTModel(SMALL, ModelConfig(attention_dropout=0.0)).eval()

    def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layers = model.encode_layers(input_ids, attention_mask)
        return model.head(layers[-1]), layers

    _, drawings = grouped_loss_and_drawings(forward, batch, cell_budget=48)
    layers = model.encode_layers(batch.input_ids, batch.attention_mask)
    mask = batch.attention_mask[..., None]
    expected = (layers * mask).sum(-2) / mask.sum(-2)

    assert drawings.shape == (model.config.hidden_layers + 1, 6, model.config.hidden_size)
    torch.testing.assert_close(drawings, expected, rtol=1e-4, atol=1e-6)


def test_sigreg_parts_average_the_layers_and_name_each_one() -> None:
    drawings = torch.randn(3, 32, 8, generator=torch.Generator().manual_seed(0)) * 0.1 + 0.2
    raw = sigreg_parts(drawings, 5, standardized=False)
    scaled = sigreg_parts(drawings, 5, standardized=True)

    each = [sigreg(layer, 5) for layer in drawings]
    assert list(raw) == ["sigreg", "sigreg_layer_1", "sigreg_layer_2", "sigreg_final_norm"]
    torch.testing.assert_close(raw["sigreg"], torch.stack(each).mean())
    torch.testing.assert_close(raw["sigreg_layer_2"], each[1])
    torch.testing.assert_close(scaled["sigreg_final_norm"], sigreg(standardize(drawings[2]), 5))
    assert sigreg_parts(drawings[2], 5, standardized=True) == {"sigreg": scaled["sigreg_final_norm"]}


def test_an_all_layer_raw_sigreg_fit_records_every_layer(tmp_path: Path, splits: tuple[Path, Path]) -> None:
    _fit(tmp_path, splits, config=_config(sigreg_weight=0.5, sigreg_layers="all", sigreg_standardize=False))

    steps = [row for row in _rows(tmp_path) if "event" not in row]
    layers = [f"sigreg_layer_{index}" for index in range(1, TINY.hidden_layers + 1)]
    for row in steps:
        parts = row["loss_parts"]
        assert set(parts) == {*ROW_FIELDS, "next_row", "sigreg", *layers, "sigreg_final_norm"}
        assert parts["sigreg"] == pytest.approx(
            sum(parts[name] for name in [*layers, "sigreg_final_norm"]) / (TINY.hidden_layers + 1)
        )
        assert row["loss"] == pytest.approx(0.5 * parts["next_row"] + 0.5 * parts["sigreg"], rel=1e-5)
