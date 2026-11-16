from pathlib import Path

from eulercad.paths import get_absolute_path


def test_relative_paths_resolve_under_eulercad_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EULERCAD_HOME", str(tmp_path))
    assert get_absolute_path("data/ABC") == tmp_path / "data/ABC"


def test_relative_paths_resolve_under_the_current_folder_without_eulercad_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EULERCAD_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_absolute_path("runs") == tmp_path / "runs"


def test_absolute_paths_are_kept(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EULERCAD_HOME", "/elsewhere")
    assert get_absolute_path(tmp_path / "x") == tmp_path / "x"


def test_repository_is_the_folder_holding_pyproject() -> None:
    from eulercad.paths import REPOSITORY

    assert (REPOSITORY / "pyproject.toml").is_file()
    assert (REPOSITORY / "src" / "eulercad").is_dir()
