"""Tests for eulercad.abc.cleaning.normalize.

Plain asserts and a __main__ runner, so this works either under pytest or as
`python tests/test_normalize.py` (pytest is not currently a project dependency).
"""

import math
import sys

import polars as pl

from eulercad.abc.cleaning import (
    SCALES,
    anchor_thresholds,
    sketch_frames,
    to_absolute,
    to_normalized,
)
from eulercad.abc.cleaning.normalize import COORD_COLS


def _f(value) -> float:
    """Read a polars scalar as a float.

    polars types its element and aggregate accessors as a wide union that includes
    `Decimal`, dates and more, which ty cannot narrow - a stub gap, not a real type
    issue. Funnelling every numeric read through this one untyped boundary keeps the
    tests readable instead of scattering suppressions over every call site.

    Returns:
        The value as a plain float.
    """
    return float(value)


def _assert_raises(exc_type: type[Exception], fn, *args, match: str | None = None) -> None:
    """Assert `fn(*args)` raises `exc_type`, optionally containing `match`.

    Raises:
        AssertionError: If nothing was raised, or the message did not match.
    """
    try:
        fn(*args)
    except exc_type as exc:
        assert match is None or match in str(exc), f"expected {match!r} in {exc!s}"
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc_type.__name__}")


def _lines(rows: list[tuple[int, float, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sketch_id": [r[0] for r in rows],
            "start_x": [r[1] for r in rows],
            "start_y": [r[2] for r in rows],
            "end_x": [r[3] for r in rows],
            "end_y": [r[4] for r in rows],
        }
    )


# a 100 x 100 square centred on (150, 250), and a 1000 x 10 bar centred on (0, 0)
SQUARE = [(1, 100.0, 200.0, 200.0, 200.0), (1, 200.0, 200.0, 200.0, 300.0), (1, 200.0, 300.0, 100.0, 300.0)]
BAR = [(2, -500.0, -5.0, 500.0, -5.0), (2, 500.0, -5.0, 500.0, 5.0), (2, 500.0, 5.0, -500.0, 5.0)]


def test_frames_geometry() -> None:
    """Midpoint and each scale come out at their textbook values."""
    f = sketch_frames(_lines(SQUARE + BAR), scale="diag").sort("sketch_id")
    assert abs(f["cx"][0] - 150.0) < 1e-9
    assert abs(f["cy"][0] - 250.0) < 1e-9
    assert abs(f["s"][0] - math.hypot(100.0, 100.0)) < 1e-9
    assert abs(f["cx"][1] - 0.0) < 1e-9
    assert abs(f["s"][1] - math.hypot(1000.0, 10.0)) < 1e-9

    expected = {"diag": math.hypot(1000.0, 10.0), "geom": math.sqrt(1000.0 * 10.0), "minside": 10.0}
    for scale, want in expected.items():
        got = sketch_frames(_lines(BAR), scale=scale)["s"][0]
        assert abs(got - want) < 1e-9, f"{scale}: {got} != {want}"


def test_square_scales_are_proportional() -> None:
    """On a square, diag = geom * sqrt(2) = minside * sqrt(2).

    This is what makes anchor_thresholds correct: only elongation may separate the
    three scales.
    """
    df = _lines(SQUARE)
    s = {sc: sketch_frames(df, scale=sc)["s"][0] for sc in SCALES}
    assert abs(s["geom"] - s["minside"]) < 1e-9
    assert abs(s["diag"] - s["geom"] * math.sqrt(2)) < 1e-9


def test_round_trip_is_identity() -> None:
    """to_absolute(to_normalized(x)) == x, for every scale."""
    df = _lines(SQUARE + BAR)
    for scale in SCALES:
        frames = sketch_frames(df, scale=scale)
        back = to_absolute(to_normalized(df, frames), frames)
        for c in COORD_COLS:
            err = _f((back[c] - df[c]).abs().max())
            assert err < 1e-9, f"{scale}/{c}: round trip off by {err}"


def test_normalized_frame_is_unit_sized_and_centred() -> None:
    """Normalizing by `scale` makes that scale exactly 1, and centres the drawing."""
    df = _lines(SQUARE + BAR)
    for scale in SCALES:
        frames = sketch_frames(df, scale=scale)
        after = sketch_frames(to_normalized(df, frames), scale=scale)
        assert abs(_f(after["s"].max()) - 1.0) < 1e-9, scale
        assert abs(_f(after["s"].min()) - 1.0) < 1e-9, scale
        assert _f(after["cx"].abs().max()) < 1e-9, scale
        assert _f(after["cy"].abs().max()) < 1e-9, scale


def test_anchor_thresholds_exponents() -> None:
    """Lengths scale linearly, the VW epsilon as an area."""
    ml, sd, eps = 1e-4, 1e-5, 5e-6
    assert anchor_thresholds(ml, sd, eps, "diag") == (ml, sd, eps)
    for scale in ("geom", "minside"):
        a, b, c = anchor_thresholds(ml, sd, eps, scale)
        assert abs(a - ml * math.sqrt(2)) < 1e-18
        assert abs(b - sd * math.sqrt(2)) < 1e-18
        assert abs(c - eps * 2.0) < 1e-18, "vw_epsilon is an AREA and must scale with the square"


def test_anchored_thresholds_agree_on_a_square() -> None:
    """A square drawing sees the same physical thresholds under every scale.

    Converting an anchored threshold back to millimetres must land on one value,
    otherwise a scale change would silently alter how hard the pipeline simplifies.
    """
    df = _lines(SQUARE)
    physical = {}
    for scale in SCALES:
        s = sketch_frames(df, scale=scale)["s"][0]
        ml, _, eps = anchor_thresholds(1e-4, 1e-5, 5e-6, scale)
        physical[scale] = (ml * s, eps * s * s)
    ref = physical["diag"]
    for scale, got in physical.items():
        assert abs(got[0] - ref[0]) < 1e-9, f"{scale} length threshold differs"
        assert abs(got[1] - ref[1]) < 1e-9, f"{scale} area threshold differs"


def test_degenerate_sketch_is_left_unnormalized() -> None:
    """A sketch collapsed to a point gets s = 1 rather than a division by zero."""
    df = _lines([(3, 7.0, 7.0, 7.0, 7.0)])
    frames = sketch_frames(df)
    assert abs(_f(frames["s"][0]) - 1.0) < 1e-12
    back = to_absolute(to_normalized(df, frames), frames)
    for c in COORD_COLS:
        assert abs(_f(back[c][0]) - 7.0) < 1e-9


def test_missing_frame_raises() -> None:
    """A sketch with no frame must fail loudly, not silently produce nulls.

    This is the mistake that matters in practice: computing frames on one chunk and
    applying them to another.
    """
    df = _lines(SQUARE + BAR)
    partial = sketch_frames(_lines(SQUARE))
    for fn in (to_normalized, to_absolute):
        _assert_raises(ValueError, fn, df, partial, match="absent from")


def test_rejects_unknown_scale() -> None:
    """An unrecognised scale name is rejected rather than silently defaulted."""
    for bad in ("DIAG", "area", ""):
        _assert_raises(ValueError, sketch_frames, _lines(SQUARE), bad)


def test_pipeline_end_to_end_preserves_schema() -> None:
    """run_pipeline keeps the parquet schema and returns millimetres."""
    import argparse

    from eulercad.abc.preprocess import resolve_thresholds, run_pipeline

    df = _lines(SQUARE + BAR)
    for scale in (*SCALES, "none"):
        args = argparse.Namespace(
            scale=scale,
            min_side_floor=1e-3,
            min_length=None,
            snap_delta=None,
            vw_epsilon=None,
            weld_angle_deg=2.0,
        )
        assert all(t > 0 for t in resolve_thresholds(args))
        out = run_pipeline(df, args)
        assert out.columns == df.columns, f"{scale}: schema changed"
        assert out.height > 0, f"{scale}: everything was removed"

        # Output must land back in millimetres, inside the original bounding box.
        # Pool all four columns: weld_collinear may move an original end_* into a
        # start_*, so comparing a column against itself is not meaningful. Allow a
        # slack of 1e-3 of the extent, since snap_endpoints can round a coordinate
        # outward by half a grid cell - still far tighter than the ~100x error a
        # missing to_absolute would produce.
        got = pl.concat([out[c] for c in COORD_COLS]).to_numpy()
        ref = pl.concat([df[c] for c in COORD_COLS]).to_numpy()
        slack = (ref.max() - ref.min()) * 1e-3
        assert got.max() <= ref.max() + slack, f"{scale}: output above the input bbox"
        assert got.min() >= ref.min() - slack, f"{scale}: output below the input bbox"


if __name__ == "__main__":
    # A minimal runner so this file works without pytest, which is not a project
    # dependency. T201 is suppressed because reporting results IS this block's job.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")  # noqa: T201
        except Exception as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")  # noqa: T201
    print(f"\n{len(tests) - failed}/{len(tests)} passed")  # noqa: T201
    sys.exit(1 if failed else 0)
