"""The cleaning stages that run in parallel must start their workers without forking.

A fork copies the parent's locks but not the threads that hold them: after polars or torch
have started threads, a forked worker can wait forever. The full test suite hung exactly there.
"""

from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pytest

from eulercad.abc.cleaning import simplify, weld
from eulercad.abc.cleaning.schema import POLARS_SCHEMA


def _lines() -> pl.DataFrame:
    rows = [(1, 0.0, 0.0, 1.0, 0.0), (1, 1.0, 0.0, 2.0, 0.0), (2, 0.0, 0.0, 0.0, 1.0)]
    return pl.DataFrame(rows, schema=POLARS_SCHEMA, orient="row")


@pytest.mark.parametrize(("module", "stage"), [(weld, "weld_collinear"), (simplify, "visvalingam_simplify")])
def test_parallel_cleaning_stages_spawn_their_workers(monkeypatch, module, stage) -> None:
    methods = []

    def recorded(*args, **kwargs) -> ProcessPoolExecutor:
        context = kwargs.get("mp_context")
        methods.append(None if context is None else context.get_start_method())
        return ProcessPoolExecutor(*args, **kwargs)

    monkeypatch.setattr(module, "ProcessPoolExecutor", recorded)
    getattr(module, stage)(_lines())
    assert methods == ["spawn"]
