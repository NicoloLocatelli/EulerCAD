import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import pairwise

import polars as pl
from simplification.cutil import simplify_coords_vw
from tqdm.auto import tqdm

from eulercad.abc.cleaning.chains import walk_chains
from eulercad.abc.cleaning.schema import POLARS_SCHEMA

DEFAULT_EPSILON_VW = 0.05  # triangle area threshold (coord²)


def _vw_one_sketch(lines: list[dict], epsilon: float) -> list[dict]:
    if not lines:
        return []
    sid = lines[0]["sketch_id"]
    out: list[dict] = []
    for chain in walk_chains(lines):
        if len(chain) < 2:
            continue
        pts = chain if len(chain) == 2 else simplify_coords_vw([list(p) for p in chain], epsilon)
        if len(pts) < 2:
            continue
        for a, b in pairwise(pts):
            out.append(
                {
                    "sketch_id": sid,
                    "start_x": a[0],
                    "start_y": a[1],
                    "end_x": b[0],
                    "end_y": b[1],
                }
            )
    return out


def visvalingam_simplify(df: pl.DataFrame, epsilon: float = DEFAULT_EPSILON_VW, progress: bool = False) -> pl.DataFrame:
    """VW simplification on Line chains. Per-sketch, parallel across processes.

    Threshold = triangle area in coord units squared.

    Returns:
        a new DataFrame with a reduced number of lines.
    """
    # partition_by does one grouped pass over df instead of one filter() scan per sketch.
    subs = [g.to_dicts() for g in df.partition_by("sketch_id", maintain_order=True)]
    out: list[dict] = []
    # spawn, not fork: a forked worker can deadlock on a lock held by a thread of polars or torch
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as ex:
        results = ex.map(partial(_vw_one_sketch, epsilon=epsilon), subs)
        if progress:
            results = tqdm(results, total=len(subs), desc="VW")
        for simplified in results:
            out.extend(simplified)
    return pl.DataFrame(out, schema=POLARS_SCHEMA) if out else df.clear()
