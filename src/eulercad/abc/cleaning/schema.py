import polars as pl

POLARS_SCHEMA = {
    "sketch_id": pl.Int64,
    "start_x": pl.Float64,
    "start_y": pl.Float64,
    "end_x": pl.Float64,
    "end_y": pl.Float64,
}


def line_row(sid: int, ax: float, ay: float, bx: float, by: float) -> dict:
    return {
        "sketch_id": sid,
        "start_x": ax,
        "start_y": ay,
        "end_x": bx,
        "end_y": by,
    }
