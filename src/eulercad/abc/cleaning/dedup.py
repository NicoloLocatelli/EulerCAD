import polars as pl


def _canonical_line_key() -> pl.Expr:
    """
    Direction-agnostic line key: sort endpoints lexicographically.
    Returns:
        a string like "x1|y1|x2|y2" where (x1, y1) ≤ (x2, y2).
    """
    p1_smaller = (pl.col("start_x") < pl.col("end_x")) | (
        (pl.col("start_x") == pl.col("end_x")) & (pl.col("start_y") <= pl.col("end_y"))
    )
    a_x = pl.when(p1_smaller).then(pl.col("start_x")).otherwise(pl.col("end_x"))
    a_y = pl.when(p1_smaller).then(pl.col("start_y")).otherwise(pl.col("end_y"))
    b_x = pl.when(p1_smaller).then(pl.col("end_x")).otherwise(pl.col("start_x"))
    b_y = pl.when(p1_smaller).then(pl.col("end_y")).otherwise(pl.col("start_y"))
    return pl.format("{}|{}|{}|{}", a_x, a_y, b_x, b_y)


def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Spatial-hash dedup. Per-sketch. First occurrence wins.
    Returns:
        a new DataFrame with duplicate lines removed.
    """
    return (
        df.with_columns(_canonical_line_key().alias("_key"))
        .unique(subset=["sketch_id", "_key"], keep="first", maintain_order=True)
        .drop("_key")
    )
