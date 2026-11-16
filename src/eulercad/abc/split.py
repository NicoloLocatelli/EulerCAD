"""Split ABC sketches into train/val/test by part_id (all views of one part stay together)."""

import argparse
import logging

import numpy as np
import polars as pl

from eulercad.paths import get_absolute_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

PARQUET_DIR = get_absolute_path("data/ABC/parquet")
FULL_PATH = PARQUET_DIR / "full_dataset.parquet"
META_PATH = PARQUET_DIR / "sketch_meta.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split ABC sketches into train/val/test by part_id.")
    parser.add_argument("--train-frac", type=float, default=0.70, help="fraction of parts assigned to train")
    parser.add_argument("--val-frac", type=float, default=0.15, help="fraction of parts assigned to val")
    parser.add_argument("--test-frac", type=float, default=0.15, help="fraction of parts assigned to test")
    parser.add_argument("--seed", type=int, default=42, help="seed for shuffling part_id before splitting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) >= 1e-9:
        raise ValueError(
            f"--train-frac + --val-frac + --test-frac must sum to 1.0, got "
            f"{args.train_frac + args.val_frac + args.test_frac}"
        )

    meta = pl.read_parquet(META_PATH)
    log.info("loaded %d sketches", meta.height)

    part_ids = meta["part_id"].drop_nulls().unique().sort().to_list()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(part_ids)
    n = len(part_ids)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_parts = set(part_ids[:n_train])
    val_parts = set(part_ids[n_train : n_train + n_val])
    test_parts = set(part_ids[n_train + n_val :])
    log.info("parts: %d train | %d val | %d test", len(train_parts), len(val_parts), len(test_parts))

    split_col = (
        pl.when(pl.col("part_id").is_in(list(train_parts)))
        .then(pl.lit("train"))
        .when(pl.col("part_id").is_in(list(val_parts)))
        .then(pl.lit("val"))
        .when(pl.col("part_id").is_in(list(test_parts)))
        .then(pl.lit("test"))
        .otherwise(None)
    )
    meta = meta.with_columns(split=split_col)

    # full_dataset.parquet is too big to read eagerly (100M+ lines) - scan +
    # sink so each split streams straight from disk to disk.
    full = pl.scan_parquet(FULL_PATH)
    for name in ("train", "val", "test"):
        sids = meta.filter(pl.col("split") == name)["sketch_id"]
        out = PARQUET_DIR / f"{name}.parquet"
        full.filter(pl.col("sketch_id").is_in(sids.to_list())).sink_parquet(out)
        # ty misresolves LazyFrame.collect()'s return type here (reports an
        # internal InProcessQuery instead of DataFrame) - a polars stub gap in
        # ty 0.0.51, not an actual type issue: this is always a real DataFrame.
        n_lines = pl.scan_parquet(out).select(pl.len()).collect().item()  # ty: ignore[unresolved-attribute]
        log.info("  %s: %d sketches, %d lines -> %s", name, sids.len(), n_lines, out)

    meta.write_parquet(PARQUET_DIR / "split_meta.parquet")
    log.info("wrote split_meta.parquet")


if __name__ == "__main__":
    main()
