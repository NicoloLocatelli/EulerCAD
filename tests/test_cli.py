"""Tests for the `eulercad` command: each subcommand calls the right steps with the right arguments.

The steps themselves need the dataset, a GPU or the network, so they are replaced by recorders here;
their own tests cover what they compute.
"""

from pathlib import Path

import pytest

from eulercad import cli, hub


@pytest.fixture
def calls(monkeypatch) -> list[tuple[str, list[str]]]:
    """Replace every step the CLI may run with a recorder of its name and arguments.

    Returns:
        The calls, in order, as (step, arguments).
    """
    seen: list[tuple[str, list[str]]] = []
    for name in cli.STEPS:
        monkeypatch.setitem(cli.STEPS, name, lambda argv, name=name: seen.append((name, list(argv))) or 0)
    return seen


def test_train_with_the_deep_preset_runs_the_recipe_of_deep_then_the_user_arguments(calls) -> None:
    assert cli.main(["train", "--preset", "deep", "--workers", "4"]) == 0
    [(name, argv)] = calls
    assert name == "pretrain"
    assert argv[: len(cli.DEEP)] == cli.DEEP
    assert argv[len(cli.DEEP) :] == ["--experiment", "deep", "--workers", "4"]
    for setting in ("--gated-attention", "--total-steps", "208200", "--learning-rate", "1.2e-3", "wsd_sqrt"):
        assert setting in cli.DEEP


def test_train_keeps_the_experiment_the_user_names(calls) -> None:
    cli.main(["train", "--preset", "deep", "--experiment", "mine"])
    [(_, argv)] = calls
    assert argv.count("--experiment") == 1
    assert argv[argv.index("--experiment") + 1] == "mine"


def test_analyze_measures_the_layers_then_embeds_the_split_and_scores_retrieval(calls) -> None:
    checkpoint = "runs/deep/checkpoints/model.safetensors"
    assert cli.main(["analyze", checkpoint, "--split", "test"]) == 0
    assert [name for name, _ in calls] == ["measures", "embed", "retrieval"]
    measures, embed, retrieval = (argv for _, argv in calls)
    assert measures[:3] == [checkpoint, "--split", "test"]
    output = str(Path("outputs/deep/embeddings_model"))
    assert embed == [checkpoint, "--output", output, "--splits", "test"]
    assert retrieval == [output]


def test_analyze_can_skip_retrieval(calls) -> None:
    cli.main(["analyze", "runs/x/checkpoints/best.ckpt", "--no-retrieval"])
    assert [name for name, _ in calls] == ["measures"]


def test_build_abc_runs_the_thesis_pipeline_in_order(calls) -> None:
    assert cli.main(["data", "build-abc", "--chunks", "0-4"]) == 0
    names = [name for name, _ in calls]
    assert names == [
        "download",
        "views",
        "preprocess",
        "split",
        "point_graphs",
        "duplicates",
        "duplicates",
        "duplicates",
    ]
    assert calls[0][1] == ["--chunks", "0-4", "--formats", "step"]
    assert calls[1][1] == ["--views", "iso,front,top,left,rand16", "--seed", "0"]
    assert [argv[1] for _, argv in calls[5:]] == [f"data/ABC/point_graph_dataset/{s}" for s in ("train", "val", "test")]


def test_data_download_fetches_the_preprocessed_dataset(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: seen.update(kwargs) or "")
    assert cli.main(["data", "download"]) == 0
    assert seen["repo_id"] == "NicoloLocatelli/EulerCAD-ABC-Preprocessed"
    assert seen["repo_type"] == "dataset"
    assert Path(seen["local_dir"]).parts[-2:] == ("data", "ABC")


def test_model_download_fetches_the_weights_and_the_config_of_deep(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(hub, "hf_hub_download", lambda **kwargs: seen.append(kwargs) or "")
    assert cli.main(["model", "download"]) == 0
    assert {call["filename"] for call in seen} == {hub.WEIGHTS_FILE, hub.CONFIG_FILE}
    assert {call["repo_id"] for call in seen} == {"NicoloLocatelli/EulerCAD-Deep"}
    assert all(Path(call["local_dir"]).parts[-3:] == ("runs", "deep", "checkpoints") for call in seen)


def test_train_help_reaches_the_trainer(calls) -> None:
    cli.main(["train", "--help"])
    assert calls == [("pretrain", ["--help"])]
