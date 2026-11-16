import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

# Trivial 3-vector numpy ops here gain nothing from BLAS multithreading; keeping
# it single-threaded avoids pointless thread churn across many worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# If this OCCT build links TBB, HLRBRep/BRepMesh calls may try to parallelize
# internally by default. With one heavy OCC call already running per worker
# process, that's N worker processes x TBB's own N threads fighting over N
# cores - the same oversubscription problem OMP/BLAS/MKL are capped for
# above, just from OCCT instead of numpy. No-op if this build has no TBB.
os.environ.setdefault("TBB_NUM_THREADS", "1")

import polars as pl
from tqdm.auto import tqdm

from eulercad.abc.cad import (
    VIEW_PRESETS,
    cap_address_space,
    current_stage,
    load_step,
    meminfo_gb,
    memory_gate,
    model_diagonal,
    project_step_to_lines,
    project_step_to_lines_poly,
    random_views,
    render_lines_to_image,
    run_pool,
    set_stage_hook,
    stage_writer,
)
from eulercad.abc.cad.views import View
from eulercad.abc.cleaning.schema import POLARS_SCHEMA
from eulercad.paths import get_absolute_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

STEP_DIR = get_absolute_path("data/ABC/step")
IMG_DIR = get_absolute_path("data/ABC/images")
LINES_PATH = get_absolute_path("data/ABC/abc_lines.parquet")
META_PATH = get_absolute_path("data/ABC/abc_meta.parquet")
LINES_PARTS_DIR = get_absolute_path("data/ABC/_abc_lines_parts")
META_PARTS_DIR = get_absolute_path("data/ABC/_abc_meta_parts")

# One worker process per in-flight model. Saturating every core with
# os.cpu_count() workers means each one gets a fraction of its solo speed
# under contention - a model that takes 1min alone can take much longer,
# which looks like a hang and trips the timeout. Leaving headroom keeps
# individual models close to solo speed.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 2)

# Wall-clock budget in seconds for one model: its STEP load plus every
# requested view's projection. A worker that overruns this is force-killed
# at the OS level (SIGKILL), which works even against a single unyielding
# native OCC call - unlike a Python-level signal, an OS-level kill doesn't
# need the interpreter to regain control first. Empirically, solo models
# (load + a handful of views) complete in well under a minute; this leaves
# generous headroom before assuming one is stuck.
DEFAULT_MODEL_TIMEOUT = 600.0

# Tessellation tolerance as a fraction of each model's own 3D bounding-box diagonal,
# rather than a fixed length. An absolute tolerance means a different *relative*
# accuracy on every model, because the corpus spans a huge size range: measured, it
# varied by a factor of 2.3 million across the corpus, leaving small parts coarsely
# approximated (a 0.19 mm part came back as 4 segments where its outline is a curve)
# and large ones tessellated far finer than anything downstream can use.
DEFAULT_DEFLECTION_K = 2e-4

# Angular tessellation tolerance, radians. Being an angle this is inherently
# scale-free, so it enforces a minimum number of chords per curve whatever the
# model's size - which is what actually fixes small-part undersampling. 0.5 was
# chosen over the tighter 0.2 because, measured end-to-end through the cleaning
# pipeline, the two produce the same final line counts (2,758 vs 2,765 on a trial)
# while 0.5 emits a third fewer intermediate segments and keeps the heavy tail at
# parity with the old pipeline.
DEFAULT_ANGULAR_DEFLECTION = 0.5

# Legacy absolute tolerance, used only with --hlr exact.
DEFAULT_DEFLECTION = 0.05

# Dispatch is held while memory is tight, so a heavy model cannot drive the host
# into swap. Projection is memory-bound on complex models, not CPU-bound.
#
# The floor is deliberately low: it is a backstop against genuine exhaustion, not
# the throttle. Swap growth is what actually bounds this run (see memory_gate). An
# earlier 4 GB floor read against MemFree throttled 12 workers to 1 over a 9-hour
# run while swap never rose above its starting value.
DEFAULT_MIN_FREE_GB = 2.0
DEFAULT_SWAP_GROWTH_GB = 0.5

# Hard per-worker address-space ceiling. The dispatch gate above is reactive: it can
# refuse to start new work, but nothing stops a worker that is already running from
# allocating without bound - measured, a model kept allocating for 300 s inside the
# mesher before it was killed, and single workers were seen holding 6+ GB.
#
# Deliberately loose. RLIMIT_AS caps *virtual* address space, not resident memory,
# and numpy/OpenMP/pybind11 reserve far more address space than they touch, so a
# tight value rejects ordinary models: calibrated on this corpus, 4 GB turned away
# 9 of 12 models, 8 GB one of 12, 12 GB none. At 16 GB this is a runaway killer
# rather than a filter - it converts "the host swaps for an hour" into "one worker
# dies and its model is reported in the skip list".
DEFAULT_MEMORY_CAP_GB = 16.0

META_SCHEMA = {
    "sketch_id": pl.Int64,
    "label": pl.Utf8,
    "view": pl.Utf8,
    "image": pl.Utf8,
}


def resolve_views(spec: str, seed: int) -> dict[str, View]:
    """Parse a comma-separated view spec: preset names from VIEW_PRESETS, plus at most one 'randK' for K random views.

    Returns:
        dict[str, View]: Mapping from view name to a View (direction, up) tuple.

    Raises:
        ValueError: if the spec has more than one 'randK' token.
    """
    views: dict[str, View] = {}
    seen_rand = False
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.startswith("rand"):
            if seen_rand:
                raise ValueError(f"'--views' spec {spec!r} has more than one 'randK' token; only one is allowed")
            seen_rand = True
            views.update(random_views(int(tok[len("rand") :]), seed))
        else:
            views[tok] = VIEW_PRESETS[tok]
    return views


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project ABC STEP models to 2D line sketches + raster images.")
    parser.add_argument(
        "--views",
        default="iso,front,top,left",
        help="comma-separated view names: preset(s) from iso,front,top,left and/or 'randK' for K random views",
    )
    parser.add_argument("--img-size", type=int, default=224, help="output image size in pixels")
    parser.add_argument("--seed", type=int, default=0, help="seed for random views")
    parser.add_argument("--limit", type=int, default=None, help="max number of STEP models to process (debug)")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="parallel worker processes, one per in-flight model (default: cpu_count() - 2, leaves headroom to avoid contention)",
    )
    parser.add_argument(
        "--hlr",
        choices=("poly", "exact"),
        default="poly",
        help=(
            "hidden-line-removal algorithm. 'poly' triangulates then removes hidden lines on the mesh: "
            "measured 17x faster in total and 85x better worst-case than 'exact', and it recovers geometry "
            "the exact algorithm silently drops on awkward shapes. 'exact' is the analytic algorithm used "
            "before, kept for comparison"
        ),
    )
    parser.add_argument(
        "--deflection-k",
        type=float,
        default=DEFAULT_DEFLECTION_K,
        help="tessellation tolerance as a fraction of the model's 3D bounding-box diagonal (--hlr poly)",
    )
    parser.add_argument(
        "--angular-deflection",
        type=float,
        default=DEFAULT_ANGULAR_DEFLECTION,
        help="angular tessellation tolerance in radians; scale-free, so it sets the floor on curve sampling",
    )
    parser.add_argument(
        "--deflection",
        type=float,
        default=DEFAULT_DEFLECTION,
        help="absolute curve -> polyline tolerance in model units (mm); used only with --hlr exact",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help="hold dispatching new models while AVAILABLE memory is below this (a backstop; swap growth is the throttle)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard per-model results already on disk and reproject everything (default resumes from them)",
    )
    parser.add_argument(
        "--swap-growth-gb",
        type=float,
        default=DEFAULT_SWAP_GROWTH_GB,
        help="hold dispatching once swap grows this much within a 5-minute window; keeps the host out of swap",
    )
    parser.add_argument(
        "--memory-cap-gb",
        type=float,
        default=DEFAULT_MEMORY_CAP_GB,
        help=(
            "hard address-space ceiling per worker; a runaway model fails that worker alone instead of "
            "the host. Set to 0 to disable"
        ),
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=DEFAULT_MODEL_TIMEOUT,
        help="wall-clock seconds budget for one model's full load + all views before its worker is force-killed",
    )
    return parser.parse_args(argv)


def _process_model(
    model_idx: int,
    model_path: Path,
    views_items: list[tuple[str, View]],
    img_size: int,
    hlr: str,
    deflection_k: float,
    angular_deflection: float,
    deflection: float,
    memory_cap_gb: float,
    lines_part_path: Path,
    meta_part_path: Path,
    stage_buf,
) -> None:
    """Load one STEP model, project every requested view, write results to disk.

    Runs in its own process (see main): a genuinely stuck OCC call blocks
    this process forever, but never the parent, which force-kills it on its
    own schedule. Writes nothing until everything below has run, so the
    parent can tell success from failure purely from a zero exit code plus
    both part files existing - no data is returned through IPC.

    With `--hlr poly` the tolerance is `deflection_k x` the model's own 3D diagonal,
    so every model is tessellated to the same relative accuracy; `--hlr exact` uses
    the absolute `deflection` the pipeline used before.

    A model that exhausts `memory_cap_gb` dies here rather than dragging the host
    into swap; the parent reports it in the skip list like any other failure.

    Raises:
        SystemExit: with code 1 if the STEP file itself fails to load, or if a
            polygonal run finds no usable extent to scale its tolerance by;
            per-view projection failures are logged and skipped instead.
    """
    if memory_cap_gb > 0:
        cap_address_space(memory_cap_gb)
    set_stage_hook(stage_writer(stage_buf))
    label = model_path.stem

    try:
        shape = load_step(model_path)
    except Exception as e:
        log.error(
            "Failed to load %s: %s: %s (stuck at: %s)",
            model_path.name,
            type(e).__name__,
            e,
            current_stage(),
        )
        raise SystemExit(1) from None

    tolerance = deflection
    if hlr == "poly":
        diag3d = model_diagonal(shape)
        if diag3d <= 0.0:
            log.error("%s: degenerate 3D bounding box, cannot scale the tolerance", label)
            raise SystemExit(1)
        tolerance = deflection_k * diag3d

    line_rows: list[dict] = []
    meta_rows: list[dict] = []
    views_ok = 0
    segment_count = 0

    for view_idx, (view_name, (direction, up)) in enumerate(views_items):
        try:
            if hlr == "poly":
                segments = project_step_to_lines_poly(
                    shape,
                    direction,
                    up,
                    tolerance,
                    mesh_deflection=tolerance,
                    angular_deflection=angular_deflection,
                    label=f"{label}/{view_name}",
                )
            else:
                segments = project_step_to_lines(shape, direction, up, tolerance, label=f"{label}/{view_name}")
        except Exception as e:
            log.error(
                "Failed to project %s / %s: %s: %s (stuck at: %s)",
                label,
                view_name,
                type(e).__name__,
                e,
                current_stage(),
            )
            continue
        if not segments:
            continue

        # global, collision-free across models: fixed view order + global model index
        sid = model_idx * len(views_items) + view_idx
        for x1, y1, x2, y2 in segments:
            line_rows.append({"sketch_id": sid, "start_x": x1, "start_y": y1, "end_x": x2, "end_y": y2})
        img_path = IMG_DIR / f"{label}__{view_name}.png"
        render_lines_to_image(segments, img_path, size=img_size)
        meta_rows.append({"sketch_id": sid, "label": label, "view": view_name, "image": img_path.name})
        views_ok += 1
        segment_count += len(segments)

    log.info("%s: done (%d/%d views, %d segments)", label, views_ok, len(views_items), segment_count)
    pl.DataFrame(line_rows, schema=POLARS_SCHEMA).write_parquet(lines_part_path)
    pl.DataFrame(meta_rows, schema=META_SCHEMA).write_parquet(meta_part_path)


def _pending_models(
    step_files: list[Path],
    part_paths: Callable[[Path], tuple[Path, Path]],
    restart: bool,
) -> list[Path]:
    """Drop models that already have both parts on disk, so a run resumes.

    Returns:
        The models still needing projection.
    """
    if restart:
        for parts_dir in (LINES_PARTS_DIR, META_PARTS_DIR):
            for stale in parts_dir.glob("*.parquet"):
                stale.unlink()

    todo = [p for p in step_files if not all(x.exists() for x in part_paths(p))]
    if len(todo) < len(step_files):
        log.info(
            "resuming: %d model(s) already projected, %d to go (use --restart to redo)",
            len(step_files) - len(todo),
            len(todo),
        )
    if not todo:
        log.info("every model already has results; merging what is on disk")
    return todo


def _run_pipeline(
    step_files: list[Path],
    views_items: list[tuple[str, View]],
    args: argparse.Namespace,
) -> list[str]:
    """Schedule one process per model via eulercad.abc.cad.pool.run_pool.

    A model whose worker times out, crashes, or doesn't leave both part files
    behind is skipped - see run_pool for retry semantics (none).

    Returns:
        list[str]: filenames skipped.
    """

    # Parts are named by model, not by dispatch position, so a part stays valid when
    # the work list changes - which is what makes resuming possible at all.
    def part_paths(model_path: Path) -> tuple[Path, Path]:
        return (
            LINES_PARTS_DIR / f"{model_path.stem}.parquet",
            META_PARTS_DIR / f"{model_path.stem}.parquet",
        )

    # sketch_id is derived from the model's position, so that position must come from
    # the FULL sorted list. Numbering by dispatch position instead would renumber
    # every model on a resumed run and collide with the ids already written.
    index_of = {p: i for i, p in enumerate(step_files)}

    def make_args(key: int, model_path: Path, stage_buf) -> tuple:
        lines_part, meta_part = part_paths(model_path)
        return (
            index_of[model_path],
            model_path,
            views_items,
            args.img_size,
            args.hlr,
            args.deflection_k,
            args.angular_deflection,
            args.deflection,
            args.memory_cap_gb,
            lines_part,
            meta_part,
            stage_buf,
        )

    def is_success(key: int, model_path: Path) -> bool:
        return all(p.exists() for p in part_paths(model_path))

    todo = _pending_models(step_files, part_paths, restart=args.restart)
    if not todo:
        return []

    with tqdm(total=len(todo), desc="Models") as pbar:

        def on_finish(key: int, model_path: Path, ok: bool) -> None:
            pbar.update(1)
            if ok:
                return
            for part in part_paths(model_path):
                part.unlink(missing_ok=True)
            log.error("%s: failed, giving up on it", model_path.name)

        skipped_paths = run_pool(
            todo,
            target=_process_model,
            make_args=make_args,
            n_workers=args.workers,
            timeout=args.model_timeout,
            is_success=is_success,
            on_finish=on_finish,
            label=lambda p: p.name,
            can_dispatch=memory_gate(args.min_free_gb, args.swap_growth_gb),
        )

    return [p.name for p in skipped_paths]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    views = resolve_views(args.views, args.seed)
    views_items = list(views.items())  # fixed order, shared identically across all workers
    log.info("Views: %s", list(views))

    if args.hlr == "poly":
        log.info(
            "HLR: polygonal | tolerance %.3g x 3D diagonal | angular %.3g rad",
            args.deflection_k,
            args.angular_deflection,
        )
    else:
        log.info("HLR: exact | absolute tolerance %.4g mm", args.deflection)
    info = meminfo_gb()
    log.info(
        "memory gate: hold below %.1f GB free or after %.1f GB swap growth (now %.1f GB free); per-worker cap %s",
        args.min_free_gb,
        args.swap_growth_gb,
        info.get("MemFree", float("nan")),
        f"{args.memory_cap_gb:.0f} GB" if args.memory_cap_gb > 0 else "disabled",
    )

    step_files = sorted(STEP_DIR.rglob("*.step")) + sorted(STEP_DIR.rglob("*.stp"))
    if args.limit:
        step_files = step_files[: args.limit]
    # Cheapest first, by file size - a free proxy for complexity. The light bulk then
    # runs at full width, and the heavy tail arrives when there is room for it.
    step_files.sort(key=lambda p: p.stat().st_size)
    log.info("Found %d STEP models", len(step_files))

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LINES_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    META_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    skipped = _run_pipeline(step_files, views_items, args)

    if skipped:
        log.warning("Skipped %d model(s): %s", len(skipped), ", ".join(skipped))

    if not any(LINES_PARTS_DIR.glob("*.parquet")):
        log.error("No model succeeded, nothing to merge.")
        return

    # Parts are kept, not deleted: they are what a resumed run reads to skip work it
    # has already done. Re-running merges whatever is on disk; --restart clears them.
    log.info("Merging line parts -> %s", LINES_PATH)
    pl.scan_parquet(LINES_PARTS_DIR / "*.parquet").sink_parquet(LINES_PATH)

    log.info("Merging meta parts -> %s", META_PATH)
    pl.scan_parquet(META_PARTS_DIR / "*.parquet").sink_parquet(META_PATH)

    meta = pl.scan_parquet(META_PATH).select(pl.len().alias("n"), pl.col("label").n_unique().alias("models")).collect()
    # ty misresolves LazyFrame.collect()'s return type here (reports an
    # internal InProcessQuery instead of DataFrame) - a polars stub gap in
    # ty 0.0.51, not an actual type issue: `meta` is a real DataFrame.
    log.info(
        "Done. %d sketches from %d models -> %s / %s",
        meta["n"][0],  # ty: ignore[not-subscriptable]
        meta["models"][0],  # ty: ignore[not-subscriptable]
        LINES_PATH,
        META_PATH,
    )


if __name__ == "__main__":
    main()
