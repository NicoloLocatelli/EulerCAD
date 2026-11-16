import itertools
import logging
import math
from collections import Counter
from pathlib import Path

import ezdxf
import polars as pl
from ezdxf import disassemble

from eulercad.abc.cleaning.schema import POLARS_SCHEMA, line_row

log = logging.getLogger(__name__)

DEFAULT_FLATTEN_DIST = 0.01  # max curve→polyline approximation distance (mm)


def list_dxfs(dxf_dir: Path, view: str | None = None, lod: str | None = None) -> list[Path]:
    files = sorted(Path(dxf_dir).glob("*.dxf"))
    if view:
        files = [f for f in files if f.stem.endswith(f"_{view}")]
    if lod:
        files = [f for f in files if f"_LOD_{lod}_" in f.stem]
    return files


def _entity_to_rows(sid: int, e, flatten_dist: float, dropped: Counter, zero: Counter) -> list[dict]:
    """
    Uniform ezdxf primitive flattening → consecutive Line rows.
    Returns:

        empty list if the entity is dropped or has zero length.
    """
    try:
        prim = disassemble.make_primitive(e, max_flattening_distance=flatten_dist)
    except Exception:
        dropped[e.dxftype()] += 1
        return []
    if prim.is_empty:
        dropped[e.dxftype()] += 1
        return []
    try:
        pts = list(prim.vertices())
    except (ZeroDivisionError, ArithmeticError, ValueError) as exc:
        log.debug("flattening failed for %s: %s", e.dxftype(), exc)
        dropped[e.dxftype()] += 1
        return []
    if len(pts) < 2:
        zero[e.dxftype()] += 1
        return []
    out = []
    for a, b in itertools.pairwise(pts):
        if math.hypot(b.x - a.x, b.y - a.y) < 1e-9:
            continue
        out.append(line_row(sid, a.x, a.y, b.x, b.y))
    return out


def parse_file(
    path: Path,
    sid: int,
    flatten_dist: float = DEFAULT_FLATTEN_DIST,
    dropped: Counter | None = None,
    zero: Counter | None = None,
) -> list[dict]:
    if dropped is None:
        dropped = Counter()
    if zero is None:
        zero = Counter()
    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:
        log.warning("skip %s: %s", path.name, exc)
        return []
    rows = []
    for e in disassemble.recursive_decompose(doc.modelspace()):
        rows.extend(_entity_to_rows(sid, e, flatten_dist, dropped, zero))
    return rows


def load_sample(paths: list[Path], flatten_dist: float = DEFAULT_FLATTEN_DIST, progress: bool = False) -> pl.DataFrame:
    dropped: Counter = Counter()
    zero: Counter = Counter()
    rows: list[dict] = []
    it = enumerate(paths)
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(list(it), desc="parse dxf")
    for sid, p in it:
        rows.extend(parse_file(p, sid, flatten_dist, dropped, zero))
    df = pl.DataFrame(rows, schema=POLARS_SCHEMA)
    if dropped:
        log.info("dropped types: %s", dict(dropped))
    if zero:
        log.info("zero-size dropped: %s", dict(zero))
    return df
