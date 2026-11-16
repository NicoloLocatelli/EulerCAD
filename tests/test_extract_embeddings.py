"""Tests for saving every drawing's per-layer embedding (`scripts/abc_03_embeddings_analysis/extract_embeddings.py`)."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from eulercad.abc.point_graph import write_chunk
from eulercad.analysis.embeddings import INDEX, SETTINGS, embed_split, half_if_safe, main
from eulercad.analysis.layers import hidden_states, layer_names
from eulercad.analysis.loading import backbone
from eulercad.data.collate import collate
from eulercad.data.dataset import DrawingDataset
from eulercad.model.config import ModelConfig
from eulercad.model.model import GraphGPTModel
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig

SMALL = GraphTokenizer(TokenizerConfig(max_nodes=24, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
CONFIG = ModelConfig(hidden_size=32, hidden_layers=2, attention_heads=4, qk_norm=True)
CPU = torch.device("cpu")
SIZES = [9, 14, 30, 11, 12, 10, 13, 8]  # 30 is over the tokenizer's 24 nodes, so it is not kept


def _drawing(sketch_id: int, num_nodes: int) -> Data:
    generator = random.Random(sketch_id)  # noqa: S311  # nosec B311
    pos = [[generator.uniform(-50, 50), generator.uniform(-50, 50)] for _ in range(num_nodes)]
    pairs = {tuple(sorted(generator.sample(range(num_nodes), 2))) for _ in range(num_nodes + 3)}
    edges = torch.tensor(sorted(pairs), dtype=torch.int64).T
    return Data(
        pos=torch.tensor(pos, dtype=torch.float64),
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
        sketch_id=torch.tensor([100 + sketch_id]),
        part_id=torch.tensor([sketch_id // 3]),
    )


@pytest.fixture(scope="module")
def split(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write one split of drawings, one of them too big for the tokenizer.

    Returns:
        The split folder.
    """
    root = tmp_path_factory.mktemp("embeddings") / "val"
    root.mkdir()
    write_chunk([_drawing(index, nodes) for index, nodes in enumerate(SIZES)], root / "chunk_000.pt")
    return root


def _model() -> GraphGPTModel:
    torch.manual_seed(0)
    return GraphGPTModel(SMALL, CONFIG).eval()


def _dataset(split: Path) -> DrawingDataset:
    return DrawingDataset(split, SMALL, seed=0, resample_each_epoch=False)


def _embed(split: Path, out: Path, workers: int = 0) -> dict:
    return embed_split(_model(), _dataset(split), out, batch_size=3, workers=workers, device=CPU, precision="32-true")


def test_one_file_per_layer_with_one_row_per_kept_drawing(split: Path, tmp_path: Path) -> None:
    _embed(split, tmp_path)

    for name in layer_names(_model()):
        assert np.load(tmp_path / f"{name}.npy").shape == (len(SIZES) - 1, CONFIG.hidden_size)


def test_each_row_is_the_mean_of_the_layer_over_the_drawing_s_real_rows(split: Path, tmp_path: Path) -> None:
    _embed(split, tmp_path)
    model, dataset = _model(), _dataset(split)

    for position in (0, 4):
        batch = collate([dataset[position]])
        with torch.no_grad():
            states = hidden_states(model, batch.input_ids, batch.attention_mask)
        for name, state in states.items():
            expected = state[0, batch.attention_mask[0]].mean(0).numpy()
            np.testing.assert_allclose(np.load(tmp_path / f"{name}.npy")[position], expected, rtol=1e-3, atol=1e-4)


def test_the_index_names_every_row_in_file_order(split: Path, tmp_path: Path) -> None:
    _embed(split, tmp_path)
    dataset = _dataset(split)

    index = np.load(tmp_path / INDEX)

    kept = [i for i, nodes in enumerate(SIZES) if nodes <= 24]
    assert index["position"].tolist() == dataset.kept.tolist() == kept
    assert index["sketch_id"].tolist() == [100 + i for i in kept]
    assert index["part_id"].tolist() == [i // 3 for i in kept]
    assert index["rows"].tolist() == [int(dataset[p].input_ids.shape[0]) for p in range(len(dataset))]


def test_loader_workers_write_the_same_files(split: Path, tmp_path: Path) -> None:
    _embed(split, tmp_path / "in_place")
    _embed(split, tmp_path / "workers", workers=2)

    for name in layer_names(_model()):
        np.testing.assert_array_equal(
            np.load(tmp_path / "workers" / f"{name}.npy"), np.load(tmp_path / "in_place" / f"{name}.npy")
        )


def test_the_report_gives_each_layer_s_type_and_largest_value(split: Path, tmp_path: Path) -> None:
    report = _embed(split, tmp_path)

    for name in layer_names(_model()):
        values = np.load(tmp_path / f"{name}.npy")
        assert report["layers"][name]["dtype"] == str(values.dtype)
        assert report["layers"][name]["max_abs"] == pytest.approx(
            float(np.abs(values.astype(np.float32)).max()), rel=1e-3
        )
    assert report["drawings"] == len(SIZES) - 1


def test_half_precision_is_kept_only_when_every_value_fits(tmp_path: Path) -> None:
    small, large = tmp_path / "small.npy", tmp_path / "large.npy"
    np.save(small, np.array([[1.5, -2.25], [3.0, 0.125]], dtype=np.float32))
    np.save(large, np.array([[1.5, -70000.0]], dtype=np.float32))

    assert half_if_safe(small) == "float16"
    assert half_if_safe(large) == "float32"

    assert np.load(small).dtype == np.float16
    np.testing.assert_array_equal(np.load(small), np.array([[1.5, -2.25], [3.0, 0.125]], dtype=np.float16))
    assert np.load(large).dtype == np.float32


def test_a_row_seed_also_saves_the_hidden_state_of_one_real_row_per_drawing(split: Path, tmp_path: Path) -> None:
    embed_split(_model(), _dataset(split), tmp_path, batch_size=3, workers=0, device=CPU, precision="32-true",
                row_seed=0)  # fmt: skip
    model, dataset = _model(), _dataset(split)
    index = np.load(tmp_path / INDEX)

    assert ((index["row"] >= 0) & (index["row"] < index["rows"])).all()
    for position in (0, 4):
        batch = collate([dataset[position]])
        with torch.no_grad():
            states = hidden_states(model, batch.input_ids, batch.attention_mask)
        row = batch.attention_mask[0].nonzero()[index["row"][position], 0]
        for name, state in states.items():
            np.testing.assert_allclose(np.load(tmp_path / f"row_{name}.npy")[position], state[0, row].numpy(),
                                       rtol=1e-3, atol=1e-4)  # fmt: skip


def test_the_row_seed_picks_the_same_rows_whatever_the_loader(split: Path, tmp_path: Path) -> None:
    for folder, workers in (("in_place", 0), ("workers", 2)):
        embed_split(_model(), _dataset(split), tmp_path / folder, batch_size=3, workers=workers, device=CPU,
                    precision="32-true", row_seed=0)  # fmt: skip

    workers, in_place = (np.load(tmp_path / folder / INDEX)["row"].tolist() for folder in ("workers", "in_place"))
    assert workers == in_place
    assert len(set(in_place)) > 1  # not always the same row


def test_without_a_row_seed_no_row_files_are_written(split: Path, tmp_path: Path) -> None:
    _embed(split, tmp_path)

    assert not list(tmp_path.glob("row_*.npy"))
    assert "row" not in np.load(tmp_path / INDEX)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a checkpoint of the small model with weights no fresh network has.

    Returns:
        The checkpoint file.
    """
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.01)
    path = tmp_path_factory.mktemp("checkpoints") / "best.ckpt"
    torch.save({"tokenizer": SMALL.to_dict(), "model_config": CONFIG.to_dict(), "global_step": 5,
                "state_dict": {f"model.{name}": value for name, value in model.state_dict().items()}}, path)  # fmt: skip
    return path


def _main(checkpoint: Path, split: Path, out: Path, *arguments: str) -> dict:
    assert main([str(checkpoint), "--output", str(out), "--dataset", str(split.parent), "--splits", split.name,
                 "--device", "cpu", "--workers", "0", "--batch-size", "3", *arguments]) == 0  # fmt: skip
    return json.loads((out / SETTINGS).read_text(encoding="utf-8"))


def test_random_weights_embed_the_network_the_scratch_repairs_start_from(
    split: Path, checkpoint: Path, tmp_path: Path
) -> None:
    settings = _main(checkpoint, split, tmp_path / "scratch", "--init", "scratch", "--init-seed", "3")
    pretrained = _main(checkpoint, split, tmp_path / "pretrained")
    embed_split(backbone(checkpoint, "scratch", 3).eval(), _dataset(split), tmp_path / "expected", batch_size=3,
                workers=0, device=CPU, precision="32-true")  # fmt: skip

    assert (settings["init"], settings["init_seed"]) == ("scratch", 3)
    assert pretrained["init"] == "pretrained"
    scratch = np.load(tmp_path / "scratch" / split.name / "layer_1.npy")
    assert np.array_equal(scratch, np.load(tmp_path / "expected" / "layer_1.npy"))
    assert not np.allclose(scratch, np.load(tmp_path / "pretrained" / split.name / "layer_1.npy"))


def test_a_later_run_keeps_the_splits_an_earlier_run_wrote(split: Path, checkpoint: Path, tmp_path: Path) -> None:
    other = tmp_path / "data" / "test"
    other.mkdir(parents=True)
    write_chunk([_drawing(index, nodes) for index, nodes in enumerate(SIZES[:4])], other / "chunk_000.pt")
    _main(checkpoint, split, tmp_path / "out")
    settings = _main(checkpoint, other, tmp_path / "out")

    assert set(settings["splits"]) == {split.name, "test"}
    assert settings["splits"]["test"]["drawings"] == 3  # 30 nodes is over the tokenizer's 24
