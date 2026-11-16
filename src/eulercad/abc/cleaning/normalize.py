import logging

import polars as pl

log = logging.getLogger(__name__)

COORD_COLS = ("start_x", "start_y", "end_x", "end_y")

SCALES = ("diag", "geom", "minside")
DEFAULT_SCALE = "geom"

# Floor on the short bounding-box side, as a fraction of the diagonal, so a
# perfectly flat drawing cannot produce a zero or vanishing scale.
DEFAULT_MIN_SIDE_FLOOR = 1e-3

# Re-anchoring factor per scale, relative to the bounding-box diagonal. For a square
# drawing (w = h = s) the diagonal is s*sqrt(2) while geom and minside are both s, so
# a threshold written per-diagonal has to grow by sqrt(2) to mean the same physical
# length in the new frame. An *area* threshold grows by the square of that.
_SCALE_RATIO = {"diag": 1.0, "geom": 2.0**0.5, "minside": 2.0**0.5}


def sketch_frames(
    df: pl.DataFrame,
    scale: str = DEFAULT_SCALE,
    min_side_floor: float = DEFAULT_MIN_SIDE_FLOOR,
) -> pl.DataFrame:
    """Per-sketch normalization frame: bounding-box midpoint plus one size scale.

    The three scales differ only once a drawing is elongated - for a square they are
    all proportional:

    - ``diag``    = sqrt(w^2 + h^2), the corner-to-corner distance.
    - ``geom``    = sqrt(w * h), the side of the square with the same bounding-box
      area. This is the default: ``vw_epsilon`` is a triangle *area*, and dividing
      coordinates by ``geom`` gives every drawing unit bounding-box area, so an area
      threshold means the same thing everywhere.
    - ``minside`` = min(w, h), the short side.

    ``diag`` is dominated by the long axis, so on an elongated drawing an area
    threshold expressed as a fraction of it can exceed the entire transverse extent
    and erase every feature across the short side. ``minside`` fails the other way,
    becoming vanishingly small as the short side goes to zero.

    Args:
        df: Line table with `sketch_id` and the four coordinate columns.
        scale: One of `SCALES`.
        min_side_floor: Short-side floor, as a fraction of the diagonal.

    Returns:
        One row per sketch with columns `sketch_id`, `cx`, `cy`, `s`.

    Raises:
        ValueError: If `scale` is not one of `SCALES`.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")

    bounds = df.group_by("sketch_id").agg(
        pl.min_horizontal(pl.col("start_x").min(), pl.col("end_x").min()).alias("min_x"),
        pl.max_horizontal(pl.col("start_x").max(), pl.col("end_x").max()).alias("max_x"),
        pl.min_horizontal(pl.col("start_y").min(), pl.col("end_y").min()).alias("min_y"),
        pl.max_horizontal(pl.col("start_y").max(), pl.col("end_y").max()).alias("max_y"),
    )

    w = pl.col("max_x") - pl.col("min_x")
    h = pl.col("max_y") - pl.col("min_y")
    diag = (w.pow(2) + h.pow(2)).sqrt()
    floored = pl.max_horizontal(pl.min_horizontal(w, h), diag * min_side_floor)
    raw_scale = {"diag": diag, "geom": (pl.max_horizontal(w, h) * floored).sqrt(), "minside": floored}[scale]

    frames = bounds.select(
        "sketch_id",
        ((pl.col("max_x") + pl.col("min_x")) / 2).alias("cx"),
        ((pl.col("max_y") + pl.col("min_y")) / 2).alias("cy"),
        raw_scale.alias("s"),
    )

    # A sketch collapsed to a single point has no size to normalize by. Leave it in
    # millimetres (s = 1) rather than dividing by zero; it is broken input and the
    # min_length filter will drop it anyway.
    degenerate = frames.filter(pl.col("s") <= 0).height
    if degenerate:
        log.warning("%d sketch(es) have a zero-size bounding box; left un-normalized", degenerate)
        frames = frames.with_columns(pl.when(pl.col("s") > 0).then(pl.col("s")).otherwise(1.0).alias("s"))
    return frames


def _join_frames(df: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Attach `cx`, `cy`, `s`, failing loudly if any sketch is unaccounted for.

    Returns:
        `df` with the three frame columns appended.

    Raises:
        ValueError: If any line belongs to a sketch missing from `frames`.
    """
    out = df.join(frames.select("sketch_id", "cx", "cy", "s"), on="sketch_id", how="left")
    missing = out["s"].null_count()
    if missing:
        raise ValueError(
            f"{missing} line(s) belong to sketches absent from `frames`; compute the "
            "frames from the same DataFrame you are transforming"
        )
    return out


def to_normalized(df: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Center each sketch on its bounding-box midpoint and divide by its scale.

    Afterwards one fixed threshold means the same thing on a 1 mm part and a 262 m
    one, which is the whole point: the cleaning stages can then run with constants
    that do not have to be re-derived per drawing size.

    Returns:
        The same table with the four coordinate columns replaced.
    """
    return (
        _join_frames(df, frames)
        .with_columns(
            [((pl.col(c) - pl.col("cx" if c.endswith("_x") else "cy")) / pl.col("s")).alias(c) for c in COORD_COLS]
        )
        .drop("cx", "cy", "s")
    )


def to_absolute(df: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Exact inverse of `to_normalized`, restoring millimetres.

    Joins on `sketch_id` rather than row position, which matters because the cleaning
    stages add and remove rows in between.

    Returns:
        The same table with the four coordinate columns replaced.
    """
    return (
        _join_frames(df, frames)
        .with_columns(
            [(pl.col(c) * pl.col("s") + pl.col("cx" if c.endswith("_x") else "cy")).alias(c) for c in COORD_COLS]
        )
        .drop("cx", "cy", "s")
    )


def anchor_thresholds(
    min_length: float,
    snap_delta: float,
    vw_epsilon: float,
    scale: str = DEFAULT_SCALE,
) -> tuple[float, float, float]:
    """Re-express per-diagonal thresholds in the frame `scale` defines.

    Without this, changing the scale would silently change how hard the pipeline
    simplifies, confounding "which scale handles elongation better" with "which scale
    is more aggressive". After it, a square drawing is treated identically by every
    scale and only elongation can produce a difference.

    `min_length` and `snap_delta` are lengths and scale linearly; `vw_epsilon` is a
    triangle area and scales with the square.

    Returns:
        `(min_length, snap_delta, vw_epsilon)` in the new frame.

    Raises:
        ValueError: If `scale` is not one of `SCALES`.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    r = _SCALE_RATIO[scale]
    return min_length * r, snap_delta * r, vw_epsilon * r * r
