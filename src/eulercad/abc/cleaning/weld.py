import math
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import polars as pl

from eulercad.abc.cleaning.schema import POLARS_SCHEMA

DEFAULT_ANGLE_TOL_DEG = 2.0


def _is_collinear_at(p, a_other, b_other, angle_tol_deg: float) -> bool:
    ax, ay = a_other[0] - p[0], a_other[1] - p[1]
    bx, by = b_other[0] - p[0], b_other[1] - p[1]
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    if la == 0 or lb == 0:
        return False
    dot = (ax * bx + ay * by) / (la * lb)
    return dot < -math.cos(math.radians(angle_tol_deg))


def _weld_one_sketch(lines: list[dict], angle_tol_deg: float) -> list[dict]:
    lines = [dict(ln) for ln in lines]
    alive = [True] * len(lines)
    while True:
        graph: dict[tuple[float, float], list[int]] = defaultdict(list)
        for i, ln in enumerate(lines):
            if not alive[i]:
                continue
            graph[ln["start_x"], ln["start_y"]].append(i)
            graph[ln["end_x"], ln["end_y"]].append(i)
        merged = False
        for p, incident in graph.items():
            if len(incident) != 2:
                continue
            i, j = incident
            if not (alive[i] and alive[j]) or i == j:
                continue
            li, lj = lines[i], lines[j]
            a_other = (
                (li["end_x"], li["end_y"]) if (li["start_x"], li["start_y"]) == p else (li["start_x"], li["start_y"])
            )
            b_other = (
                (lj["end_x"], lj["end_y"]) if (lj["start_x"], lj["start_y"]) == p else (lj["start_x"], lj["start_y"])
            )
            if not _is_collinear_at(p, a_other, b_other, angle_tol_deg):
                continue
            li["start_x"], li["start_y"] = a_other
            li["end_x"], li["end_y"] = b_other
            alive[j] = False
            merged = True
        if not merged:
            break
    return [ln for ln, a in zip(lines, alive) if a]


def weld_collinear(
    df: pl.DataFrame, angle_tol_deg: float = DEFAULT_ANGLE_TOL_DEG, progress: bool = False
) -> pl.DataFrame:
    """
    Iteratively merge degree-2 collinear chains. Per-sketch, parallel across processes.
    Returns:
        a new DataFrame with collinear lines welded together.
    """
    # partition_by does one grouped pass over df instead of one filter() scan per sketch.
    subs = [g.to_dicts() for g in df.partition_by("sketch_id", maintain_order=True)]
    out: list[dict] = []
    # spawn, not fork: a forked worker can deadlock on a lock held by a thread of polars or torch
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as ex:
        results = ex.map(partial(_weld_one_sketch, angle_tol_deg=angle_tol_deg), subs)

        if progress:
            from tqdm.auto import tqdm

            results = tqdm(results, total=len(subs), desc="weld")
        for welded in results:
            out.extend(welded)
    return pl.DataFrame(out, schema=POLARS_SCHEMA) if out else df.clear()
