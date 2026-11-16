"""Tests for `eulercad/analysis/retrieval.py`: retrieval of the other views of a 3D object, per layer and split."""

import json
import math
from pathlib import Path

import numpy as np

from eulercad.analysis.metrics.retrieval import retrieval_metrics
from eulercad.analysis.retrieval import centred, chance_map, main, split_scores


def _folder(root: Path, labels: list[int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "index.npz", part_id=np.array(labels), position=np.arange(len(labels)))
    layers = {
        name: (rng.standard_normal((len(labels), 4)) + 5).astype(np.float16) for name in ("layer_1", "final_norm")
    }
    for name, values in layers.items():
        np.save(root / f"{name}.npy", values)
    return layers


def test_centred_removes_the_shared_offset() -> None:
    z = np.random.default_rng(0).normal(size=(50, 4)) + 100.0
    assert np.allclose(centred(z).mean(0), 0.0, atol=1e-9)


def test_split_scores_score_every_layer_centred_over_the_split(tmp_path: Path) -> None:
    labels = [1, 1, 2, 2, 3, 3, 4]
    layers = _folder(tmp_path, labels)
    scores = split_scores(tmp_path, list(layers))
    assert list(scores) == ["layer_1", "final_norm"]
    for name, values in layers.items():
        expected = retrieval_metrics(centred(values.astype(np.float32)), np.array(labels))
        assert math.isclose(scores[name]["map"], expected["map"])
        assert scores[name]["recall@1"] == expected["recall@1"]


def test_split_scores_skip_a_layer_that_is_the_same_for_every_drawing(tmp_path: Path) -> None:
    # centred, a constant layer is zero, and the ties would score a perfect mAP
    labels = [1, 1, 2, 2, 3, 3, 4]
    _folder(tmp_path, labels)
    row = np.random.default_rng(0).standard_normal(16).astype(np.float32)
    np.save(tmp_path / "constant.npy", np.tile(row, (len(labels), 1)))
    assert list(split_scores(tmp_path, ["constant", "layer_1"])) == ["layer_1"]


def test_chance_map_averages_the_share_of_right_drawings_over_the_queries() -> None:
    # parts of 3, 2 and 1 drawings in 6: the lone drawing is no query
    labels = np.array([1, 1, 1, 2, 2, 3])
    assert math.isclose(chance_map(labels), (3 * 2 / 5 + 2 * 1 / 5) / 5)


def test_main_scores_every_split_of_an_embeddings_folder(tmp_path: Path) -> None:
    embeddings = tmp_path / "embeddings"
    for split in ("val", "test"):
        _folder(embeddings / split, [1, 1, 2, 2, 3])
    (embeddings / "settings.json").write_text(json.dumps({"layers": ["layer_1", "final_norm"]}))
    assert main([str(embeddings), "--device", "cpu"]) == 0
    result = json.loads((embeddings / "retrieval.json").read_text())
    assert set(result) == {"val", "test"}
    assert set(result["val"]["scores"]) == {"layer_1", "final_norm"}
    assert result["val"]["drawings"] == 5
    assert math.isclose(result["val"]["chance_map"], chance_map(np.array([1, 1, 2, 2, 3])))
