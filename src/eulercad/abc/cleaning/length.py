import polars as pl


def line_length_expr() -> pl.Expr:
    return ((pl.col("end_x") - pl.col("start_x")).pow(2) + (pl.col("end_y") - pl.col("start_y")).pow(2)).sqrt()


def filter_tiny_entities(df: pl.DataFrame, min_length: float) -> pl.DataFrame:
    return df.filter(line_length_expr() >= min_length)
