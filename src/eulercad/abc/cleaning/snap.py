import polars as pl


def snap_endpoints(df: pl.DataFrame, delta: float = 1e-4) -> pl.DataFrame:
    """
    Round every coord to nearest multiple of `delta`.
    Returns:
        a new DataFrame with snapped coordinates.
    """
    cols = ("start_x", "start_y", "end_x", "end_y")
    return df.with_columns([(pl.col(c) / delta).round() * delta for c in cols if c in df.columns])
