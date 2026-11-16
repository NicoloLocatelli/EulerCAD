"""Rasterize projected 2D line segments to a fixed-size image, for image-backbone input."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def render_lines_to_image(
    segments: list[tuple[float, float, float, float]],
    out_path: Path,
    size: int = 224,
    linewidth: float = 1.2,
) -> None:
    """Center + unit-scale the segments and save a black-on-white line drawing."""
    arr = np.array(segments, dtype=np.float64).reshape(-1, 2, 2)
    pts = arr.reshape(-1, 2)
    mn, mx = pts.min(0), pts.max(0)
    center = (mn + mx) / 2
    half = max((mx - mn).max() / 2, 1e-8)
    arr = (arr - center) / half

    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.add_collection(LineCollection(arr, colors="black", linewidths=linewidth))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
