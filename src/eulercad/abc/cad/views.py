"""Camera view presets and random view sampling for STEP -> 2D line projection."""

from __future__ import annotations

import numpy as np

View = tuple[tuple[float, float, float], tuple[float, float, float]]

# name -> (direction, up). `direction` points from the model toward the camera
# (i.e. the HLR projection looks along -direction); `up` is the desired
# on-screen up vector, used to build the projection plane's local X axis.
VIEW_PRESETS: dict[str, View] = {
    "iso": ((1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}


def random_views(k: int, seed: int) -> dict[str, View]:
    """
    Sample k camera directions uniform on the sphere, each with a consistent up vector.

    Returns:
        a dict of view name -> (direction, up) tuples.
    """
    rng = np.random.default_rng(seed)
    views: dict[str, View] = {}
    for i in range(k):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        up = ref - d * np.dot(ref, d)
        up /= np.linalg.norm(up)
        views[f"rand{i}"] = (tuple(d.tolist()), tuple(up.tolist()))
    return views
