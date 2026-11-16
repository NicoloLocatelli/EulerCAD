import argparse
import logging
import math
from collections.abc import Callable

import polars as pl
from tqdm.auto import tqdm

from eulercad.abc.cleaning import (
    DEFAULT_MIN_SIDE_FLOOR,
    DEFAULT_SCALE,
    SCALES,
    anchor_thresholds,
    deduplicate,
    filter_tiny_entities,
    sketch_frames,
    snap_endpoints,
    to_absolute,
    to_normalized,
    visvalingam_simplify,
    weld_collinear,
)
from eulercad.paths import get_absolute_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

LINES_PATH = get_absolute_path("data/ABC/abc_lines.parquet")
META_PATH = get_absolute_path("data/ABC/abc_meta.parquet")
OUT_PATH = get_absolute_path("data/ABC/parquet/full_dataset.parquet")
OUT_META_PATH = get_absolute_path("data/ABC/parquet/sketch_meta.parquet")
CHUNK_PARTS_DIR = get_absolute_path("data/ABC/parquet/_preprocess_chunks")


# Thresholds for --scale none, in absolute millimetres. The historical behaviour.
ABS_DEFAULTS_MM = {"min_length": 0.01, "snap_delta": 1e-4, "vw_epsilon": 0.05}

# Thresholds for every other --scale, written in units of the bounding-box DIAGONAL
# and converted to the chosen frame by anchor_thresholds. Chosen so the MEDIAN sketch
# (diagonal 103 mm) behaves roughly as it did under ABS_DEFAULTS_MM, which makes the
# change a correction of the SIZE BIAS rather than a blanket change in aggressiveness:
#     min_length  0.01  / 103    = 9.7e-5   -> 1e-4
#     snap_delta  1e-4  / 103    = 9.7e-7   -> 1e-5
#     vw_epsilon  0.05  / 103**2 = 4.7e-6   -> 5e-6   (an AREA: diagonal SQUARED)
REL_DEFAULTS_DIAG = {"min_length": 1e-4, "snap_delta": 1e-5, "vw_epsilon": 5e-6}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ABC line sketches into cleaned parquet.")
    parser.add_argument(
        "--scale",
        choices=(*SCALES, "none"),
        default=DEFAULT_SCALE,
        help=(
            "per-sketch size each drawing is divided by before cleaning, so that fixed thresholds mean the "
            "same thing at every drawing size; 'none' restores the old absolute-millimetre behaviour"
        ),
    )
    parser.add_argument(
        "--min-side-floor",
        type=float,
        default=DEFAULT_MIN_SIDE_FLOOR,
        help="floor on the short bbox side, as a fraction of the diagonal (guards degenerate drawings)",
    )
    parser.add_argument("--min-length", type=float, default=None, help="minimum entity length to keep")
    parser.add_argument("--snap-delta", type=float, default=None, help="endpoint snap rounding delta")
    parser.add_argument("--weld-angle-deg", type=float, default=2.0, help="collinear weld angle tolerance (deg)")
    parser.add_argument("--vw-epsilon", type=float, default=None, help="Visvalingam-Whyatt simplification epsilon")
    parser.add_argument(
        "--chunk-sketches",
        type=int,
        default=1_000,
        help="sketch_id range size processed at a time (bounds peak RAM; dedup/weld/VW never need cross-sketch state)",
    )
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help=(
            "drop models left with fewer than this many non-empty views; counted AFTER cleaning, so it "
            "catches both models that arrived degenerate and views this pipeline emptied. 1 disables it"
        ),
    )
    return parser.parse_args(argv)


def resolve_thresholds(args: argparse.Namespace) -> tuple[float, float, float]:
    """Fill in whichever thresholds were not given on the command line.

    Defaults are supplied in the frame `--scale` selects: millimetres when the scale
    is 'none', otherwise units of the chosen per-sketch scale. A threshold passed
    explicitly is used verbatim, so it too is read in that frame.

    Returns:
        `(min_length, snap_delta, vw_epsilon)`.
    """
    if args.scale == "none":
        d = ABS_DEFAULTS_MM
    else:
        min_length, snap_delta, vw_epsilon = anchor_thresholds(**REL_DEFAULTS_DIAG, scale=args.scale)
        d = {"min_length": min_length, "snap_delta": snap_delta, "vw_epsilon": vw_epsilon}
    return (
        d["min_length"] if args.min_length is None else args.min_length,
        d["snap_delta"] if args.snap_delta is None else args.snap_delta,
        d["vw_epsilon"] if args.vw_epsilon is None else args.vw_epsilon,
    )


def run_pipeline(
    df: pl.DataFrame,
    args: argparse.Namespace,
    stage_rows: dict[str, list[int]] | None = None,
) -> pl.DataFrame:
    """Run stages 1-5 on `df`. Safe to call on any subset of sketches - every
    stage is per-sketch (or per-row), none need to see sketches outside `df`.

    Unless `--scale` is 'none', the coordinates are normalized per sketch first and
    mapped back to millimetres at the end, so a fixed threshold means the same thing
    on a 1 mm part and on a 262 m one. The output schema and units are unchanged
    either way. The normalization frames are derived from `df` itself, which is why
    a chunk must contain whole sketches - main() slices on sketch_id ranges, so it
    does.

    Args:
        df: the lines to clean.
        args: parsed command line, supplying the thresholds.
        stage_rows: if given, `{stage: [rows_in, rows_out]}` accumulated across calls
            so main() can print one corpus-wide table instead of only per-chunk lines.

    Returns:
        pl.DataFrame: the fully processed lines, in millimetres.
    """
    frames = None
    if args.scale != "none":
        frames = sketch_frames(df, scale=args.scale, min_side_floor=args.min_side_floor)
        df = to_normalized(df, frames)

    for name, stage in cleaning_stages(args):
        before = df.height
        df = stage(df)
        after = df.height
        log.debug(
            "    %-22s %10d -> %10d  (%+d, %+.2f%%)",
            name,
            before,
            after,
            after - before,
            100.0 * (after - before) / before if before else 0.0,
        )
        if stage_rows is not None:
            rows = stage_rows.setdefault(name, [0, 0])
            rows[0] += before
            rows[1] += after

    return df if frames is None else to_absolute(df, frames)


def cleaning_stages(args: argparse.Namespace) -> tuple[tuple[str, Callable[[pl.DataFrame], pl.DataFrame]], ...]:
    """The cleaning stages in pipeline order, as `(name, function)` pairs.

    Shared by run_pipeline and scripts/thesis_plots/segment_lengths.py, which reads
    the segments between stages.

    Returns:
        One `(name, function)` pair per stage; each function maps a line table to the
        cleaned line table, in whatever frame it was given.
    """
    min_length, snap_delta, vw_epsilon = resolve_thresholds(args)

    # Driven from a list so the row count can be read either side of every stage
    # without repeating the bookkeeping five times. Order is the pipeline order and
    # must stay that way: snapping before dedup is what makes near-duplicates exact.
    #
    # filter_tiny_entities runs twice, and the second pass is not redundant. VW keeps
    # the first and last point of every chain it is given, and walk_chains emits closed
    # loops, whose first and last point are the same one. A small circle or fillet whose
    # every interior vertex falls under `epsilon` therefore simplifies to [p0, p0] - two
    # points, so simplify.py's `len(pts) < 2` guard passes it - and pairwise turns that
    # into one zero-length segment. Measured on the first full corpus run: 352,211 of
    # them across 2.99% of sketches. Downstream that is a self-loop edge in any
    # point-as-node graph, so it is cleaned here rather than left for every consumer.
    return (
        ("filter_tiny_entities", lambda d: filter_tiny_entities(d, min_length=min_length)),
        ("snap_endpoints", lambda d: snap_endpoints(d, delta=snap_delta)),
        ("deduplicate", deduplicate),
        ("weld_collinear", lambda d: weld_collinear(d, angle_tol_deg=args.weld_angle_deg, progress=False)),
        ("visvalingam_simplify", lambda d: visvalingam_simplify(d, epsilon=vw_epsilon, progress=False)),
        ("filter_tiny_post_vw", lambda d: filter_tiny_entities(d, min_length=min_length)),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    chunk_sketches = args.chunk_sketches

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_df = pl.read_parquet(META_PATH)
    # ty misresolves LazyFrame.collect()'s return type here (reports an
    # internal InProcessQuery instead of DataFrame) - a polars stub gap in
    # ty 0.0.51, not an actual type issue: this is always a real DataFrame.
    lo, hi = (
        pl.scan_parquet(LINES_PATH)
        .select(pl.col("sketch_id").min().alias("lo"), pl.col("sketch_id").max().alias("hi"))
        .collect()
        .row(0)  # ty: ignore[unresolved-attribute]
    )
    log.info("  meta: %d sketches, sketch_id range [%d, %d]", meta_df.height, lo, hi)

    min_length, snap_delta, vw_epsilon = resolve_thresholds(args)
    unit = "mm" if args.scale == "none" else f"x {args.scale}"
    log.info(
        "  scale=%s  min_length=%.4g (%s)  snap_delta=%.4g (%s)  vw_epsilon=%.4g (%s^2)  weld_angle=%.4g deg",
        args.scale,
        min_length,
        unit,
        snap_delta,
        unit,
        vw_epsilon,
        "mm" if args.scale == "none" else args.scale,
        args.weld_angle_deg,
    )

    n_chunks = math.ceil((hi - lo + 1) / chunk_sketches)
    lines_in = 0
    total_lines = 0
    part_idx = 0
    stage_rows: dict[str, list[int]] = {}
    for start in tqdm(range(lo, hi + 1, chunk_sketches), total=n_chunks, desc="chunks"):
        end = start + chunk_sketches - 1
        # sketch_id is monotonic non-decreasing across the merged file (see
        # step_to_views.py), so this range predicate lets parquet row-group
        # min/max stats prune most of the file per chunk instead of a full scan.
        # same ty/polars stub gap as above: collect() here is always a DataFrame.
        chunk = pl.scan_parquet(LINES_PATH).filter(pl.col("sketch_id").is_between(start, end)).collect()
        if chunk.is_empty():  # ty: ignore[unresolved-attribute]
            continue
        lines_in += chunk.height  # ty: ignore[unresolved-attribute]
        chunk = run_pipeline(chunk, args, stage_rows)  # ty: ignore[invalid-argument-type]
        total_lines += chunk.height
        chunk.write_parquet(CHUNK_PARTS_DIR / f"part_{part_idx:06d}.parquet")
        part_idx += 1

    log.info("Lines per stage, whole corpus:")
    log.info("  %-22s %14s %14s %14s %9s", "stage", "in", "out", "removed", "kept")
    for name, (rows_in, rows_out) in stage_rows.items():
        log.info(
            "  %-22s %14d %14d %14d %8.2f%%",
            name,
            rows_in,
            rows_out,
            rows_in - rows_out,
            100.0 * rows_out / rows_in if rows_in else 0.0,
        )
    log.info(
        "  %-22s %14d %14d %14d %8.2f%%",
        "TOTAL",
        lines_in,
        total_lines,
        lines_in - total_lines,
        100.0 * total_lines / lines_in if lines_in else 0.0,
    )

    # label -> part_id: mirrors the WUERTH convention so the existing grouped
    # train/val/test splitter (which keys on part_id) works unchanged.
    meta_df = meta_df.rename({"label": "part_id"})

    # A sketch that cleaning emptied has no rows left in the parts, so it is dropped
    # from meta here regardless of --min-views: keeping it would leave a meta row
    # pointing at geometry that no longer exists.
    surviving = pl.scan_parquet(CHUNK_PARTS_DIR / "*.parquet").select("sketch_id").unique().collect()
    sketches_before, models_before = meta_df.height, meta_df["part_id"].n_unique()
    # same ty/polars stub gap as above: `surviving` is always a real DataFrame.
    meta_df = meta_df.join(surviving, on="sketch_id", how="semi")  # ty: ignore[invalid-argument-type]
    log.info(
        "Emptied by cleaning: %d sketch(es), %d model(s)",
        sketches_before - meta_df.height,
        models_before - meta_df["part_id"].n_unique(),
    )

    # Counted on what survived, not on the input: a model whose views this pipeline
    # emptied is as unusable for a multi-view objective as one that arrived degenerate.
    if args.min_views > 1:
        keep = (
            meta_df.group_by("part_id")
            .agg(pl.len().alias("n_views"))
            .filter(pl.col("n_views") >= args.min_views)
            .select("part_id")
        )
        sketches_before, models_before = meta_df.height, meta_df["part_id"].n_unique()
        meta_df = meta_df.join(keep, on="part_id", how="semi")
        log.info(
            "Below --min-views %d: dropped %d model(s), %d sketch(es)",
            args.min_views,
            models_before - meta_df["part_id"].n_unique(),
            sketches_before - meta_df.height,
        )

    log.info("Merging %d chunk part(s) -> %s", part_idx, OUT_PATH)
    # Semi-join, not is_in: the keep set is ~1M sketch_ids and a join streams it
    # rather than materialising one huge predicate.
    pl.scan_parquet(CHUNK_PARTS_DIR / "*.parquet").join(
        meta_df.lazy().select("sketch_id"), on="sketch_id", how="semi"
    ).sink_parquet(OUT_PATH)
    for part_file in CHUNK_PARTS_DIR.glob("*.parquet"):
        part_file.unlink()
    CHUNK_PARTS_DIR.rmdir()

    # same ty/polars stub gap as above: collect() here is always a DataFrame.
    written = pl.scan_parquet(OUT_PATH).select(pl.len()).collect().item()  # ty: ignore[unresolved-attribute]
    log.info(
        "  %-22s %14d %14d %14d %8.2f%%",
        "min_views filter",
        total_lines,
        written,
        total_lines - written,
        100.0 * written / total_lines if total_lines else 0.0,
    )

    log.info("Writing meta: %s", OUT_META_PATH)
    meta_df.write_parquet(OUT_META_PATH)
    log.info(
        "Done. %d lines across %d sketches (%d models) -> %s / %s",
        written,
        meta_df.height,
        meta_df["part_id"].n_unique(),
        OUT_PATH,
        OUT_META_PATH,
    )


if __name__ == "__main__":
    main()
