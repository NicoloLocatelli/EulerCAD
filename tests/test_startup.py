"""Tests for what the `eulercad` command prints before it does anything: no warnings from the libraries.

The test imports in a fresh interpreter, since this one has already imported everything.
"""

import subprocess  # noqa: S404  # nosec B404
import sys


def test_the_command_and_its_heaviest_step_load_without_library_warnings() -> None:
    done = subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, "-c", "import eulercad.cli, eulercad.analysis.measures"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stderr == ""
