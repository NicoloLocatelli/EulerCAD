"""Where EulerCAD reads and writes: relative paths resolve under `$EULERCAD_HOME`, or the current folder."""

import os
from pathlib import Path

_ROOT_ENV_VAR = "EULERCAD_HOME"
# The cloned repository: its git state is recorded with every run and measurement.
REPOSITORY = Path(__file__).resolve().parents[2]


def get_absolute_path(relative: str | Path, override_root_folder: str | Path | None = None) -> Path:
    """Resolve `relative` under the root; an absolute path is returned as it is.

    Returns:
        The path under `override_root_folder`, else `$EULERCAD_HOME`, else the current folder.
    """
    root = override_root_folder or os.environ.get(_ROOT_ENV_VAR) or Path.cwd()
    return Path(root) / relative
