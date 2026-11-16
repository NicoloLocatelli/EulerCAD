"""Build points-as-nodes graph datasets from the cleaned ABC parquet splits.

The lines-as-nodes datalists (create_graph_datalists.py) make one node per line
segment and connect lines that share an endpoint. Here the representation is
inverted: one node per distinct endpoint, one edge per line segment. Node
features are the raw 2D positions; nothing task-specific is baked in, so
masking / link-prediction objectives stay runtime transforms.

Both ends are streamed: the parquet is read in row slices and the graphs are
flushed chunk by chunk, so neither a whole split nor a whole datalist is ever
held in RAM. Output is one directory per split - see core/datasets/point_graph.py.

Each sketch is one projected view of a 3D CAD model, and every model contributes
twenty of them. That grouping is carried through as `part_id`, because it is the
only supervision in this corpus that is neither synthetic nor hand-made, and the
retrieval benchmark is built on it.

Carrying it forces the size filter to be decided per part rather than per sketch.
A band applied sketch by sketch keeps whichever views happen to fall inside it,
so a model contributes two views instead of twenty and the groups are shredded -
which does not fail loudly. Contrastive training on shredded groups still runs;
it just has almost nothing to contrast, and retrieval scores near zero for a
reason that has nothing to do with the model. `--min-views` is the guard.
"""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from eulercad.abc.point_graph import PointGraphDataset, write_chunk
from eulercad.paths import get_absolute_path

PARQUET_DIR = get_absolute_path("data/ABC/parquet")
SPLIT_META_PATH = get_absolute_path("data/ABC/parquet/split_meta.parquet")
POINT_GRAPH_DATASET_DIR = get_absolute_path("data/ABC/point_graph_dataset")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def iter_sketches(path: Path, rows_per_read: int) -> Iterator[tuple[int, np.ndarray]]:
    """Stream one sketch at a time out of a split parquet.

    The file is read in row slices rather than collected whole - train.parquet is
    249M rows. This relies on each sketch's rows being contiguous, which the
    preprocessing pipeline guarantees (they are not in ascending sketch_id order,
    but they are never interleaved). A sketch straddling a slice boundary is
    carried over to the next slice instead of being emitted twice.

    Args:
        path: split parquet, e.g. data/ABC/parquet/train.parquet.
        rows_per_read: rows pulled into RAM per slice.

    Yields:
        (sketch_id, coords) where coords is (m, 4) of [start_x, start_y, end_x, end_y].
    """
    lf = pl.scan_parquet(path)
    n_rows: pl.DataFrame = lf.select(pl.len()).collect()  # ty: ignore[invalid-assignment]
    total = n_rows.item()

    carry_id, carry = 0, None  # trailing sketch of a slice, may continue in the next one
    offset = 0

    with tqdm(total=total, unit="line", unit_scale=True, desc=path.stem) as bar:
        while offset < total:
            df: pl.DataFrame = lf.slice(offset, rows_per_read).collect()  # ty: ignore[invalid-assignment]
            bar.update(df.height)
            offset += df.height

            sketch_ids = df["sketch_id"].to_numpy()
            coords = df.select("start_x", "start_y", "end_x", "end_y").to_numpy()
            if carry is not None:
                sketch_ids = np.concatenate([np.full(len(carry), carry_id), sketch_ids])
                coords = np.concatenate([carry, coords])

            # split positions = where sketch_id changes, so each slice is one sketch
            cuts = np.flatnonzero(np.diff(sketch_ids)) + 1
            starts = np.concatenate([[0], cuts])
            ends = np.concatenate([cuts, [len(sketch_ids)]])

            if offset < total:  # the last sketch may continue past this slice
                carry_id, carry = int(sketch_ids[starts[-1]]), coords[starts[-1] : ends[-1]]
                starts, ends = starts[:-1], ends[:-1]
            else:
                carry = None

            for s, e in zip(starts, ends):
                yield int(sketch_ids[s]), coords[s:e]


def select_sketches(path: Path, split: str, args: argparse.Namespace) -> tuple[dict[int, int], pl.DataFrame]:
    """Decide which sketches survive, and give each one its part's integer code.

    Runs before any graph is built, because the `--min-views` rule is a property
    of a whole part and cannot be evaluated one streamed sketch at a time.

    Args:
        path: split parquet, e.g. data/ABC/parquet/train.parquet.
        split: split name, used to slice the shared split_meta table.
        args: parsed command line, for the line band and --min-views.

    Returns:
        (codes, index) where codes maps sketch_id to a part code dense in
        [0, n_parts), and index is the (part_code, part_id) table that lets a
        code be traced back to the CAD model it stands for. Codes are assigned
        over part_id sorted, so a rebuild of the same split reproduces them.
    """
    # same ty/polars stub gap as preprocess_drawings.py: collect() is reported as
    # an internal InProcessQuery, but it is always a real DataFrame here.
    counts: pl.DataFrame = pl.scan_parquet(path).group_by("sketch_id").len().collect()  # ty: ignore[invalid-assignment]
    in_band = counts.filter(pl.col("len").is_between(args.min_lines, args.max_lines))

    meta = pl.read_parquet(SPLIT_META_PATH).filter(pl.col("split") == split)
    kept = in_band.join(meta.select("sketch_id", "part_id"), on="sketch_id", how="inner")

    parts_before = kept["part_id"].n_unique()
    if args.min_views > 1:
        survivors = kept.group_by("part_id").len().filter(pl.col("len") >= args.min_views).select("part_id")
        kept = kept.join(survivors, on="part_id", how="semi")

    index = kept.select("part_id").unique().sort("part_id").with_row_index("part_code")
    kept = kept.join(index, on="part_id")

    per_part = kept.group_by("part_id").len()["len"]
    log.info(
        "%s: %d/%d sketches in [%d, %d] lines; parts %d -> %d after --min-views %d; "
        "views per part min/median/max %d/%d/%d",
        split,
        kept.height,
        counts.height,
        args.min_lines,
        args.max_lines,
        parts_before,
        index.height,
        args.min_views,
        per_part.min() if index.height else 0,
        per_part.median() if index.height else 0,
        per_part.max() if index.height else 0,
    )
    return dict(zip(kept["sketch_id"].to_list(), kept["part_code"].to_list(), strict=True)), index


def sketch_to_data(coords: np.ndarray, sketch_id: int, part_code: int) -> Data:
    """Turn one sketch's lines into a point graph.

    Endpoints are merged by exact float equality. That is enough because
    snap_endpoints already rounded every coordinate onto a fixed grid, so two
    points at the same grid cell are bit-identical, and every later stage
    (weld_collinear, visvalingam_simplify, filter_tiny_post_vw) only *selects*
    existing points instead of computing new ones. A stage that averaged merged
    endpoints would break that guarantee and reintroduce the need to quantize.

    Args:
        coords: (m, 4) array of [start_x, start_y, end_x, end_y] per line.
        sketch_id: kept on the Data object so a graph can be traced back to the parquet.
        part_code: which CAD model this sketch is a view of, as the dense code
            from select_sketches.

    Returns:
        Data with pos (P, 2) float64 and edge_index (2, 2E) int64, undirected.

        pos stays float64 because ABC coordinates are unnormalized and reach 3.6e5,
        where float32 resolves only to 0.03 while genuinely distinct points sit 1e-5
        apart. Casting belongs after the per-sketch normalization transform, not here.
    """
    endpoints = coords.reshape(-1, 2)  # (2m, 2), start/end interleaved

    # identical endpoints collapse into one node; `inverse` maps every line
    # endpoint to the node it became
    pos, inverse = np.unique(endpoints, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    # even positions are line starts, odd positions are line ends
    pairs = np.stack([inverse[0::2], inverse[1::2]], axis=1)
    pairs = np.sort(pairs, axis=1)  # canonical (low, high) order so A-B and B-A dedup together
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # a line whose endpoints merged is a self-loop
    pairs = np.unique(pairs, axis=0)

    edge_index = np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)

    return Data(
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index),
        sketch_id=torch.tensor([sketch_id], dtype=torch.long),
        part_id=torch.tensor([part_code], dtype=torch.long),
    )


def build_split(path: Path, split: str, out_dir: Path, args: argparse.Namespace) -> None:
    """Convert one split parquet into chunk files.

    Args:
        path: split parquet to read.
        split: split name, for the metadata join.
        out_dir: directory to write chunk files into.
        args: parsed command line, for the line-count filter and chunk sizes.
    """
    codes, index = select_sketches(path, split, args)
    index.write_parquet(out_dir / "part_index.parquet")

    graphs: list[Data] = []
    chunk_i = 0

    def flush() -> None:
        nonlocal graphs, chunk_i
        write_chunk(graphs, out_dir / f"chunk_{chunk_i:04d}.pt")
        graphs = []
        chunk_i += 1

    for sketch_id, coords in iter_sketches(path, args.rows_per_read):
        part_code = codes.get(sketch_id)
        if part_code is None:
            continue
        graphs.append(sketch_to_data(coords, sketch_id, part_code))
        if len(graphs) == args.chunk_size:
            flush()

    if graphs:
        flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-lines", type=int, default=100, help="drop sketches with fewer lines than this")
    parser.add_argument("--max-lines", type=int, default=300, help="drop sketches with more lines than this")
    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
        help=(
            "drop a whole CAD model unless this many of its twenty views survived the line band. "
            "2 is the floor at which a part still contributes one positive pair; 1 disables it and "
            "leaves singleton parts that no multi-view objective or retrieval query can use"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=20_000, help="graphs per chunk file")
    parser.add_argument(
        "--rows-per-read",
        type=int,
        default=5_000_000,
        help="parquet rows pulled into RAM per slice; the main RAM knob (~1 GB per 5M rows)",
    )
    parser.add_argument("--out-dir", type=Path, default=POINT_GRAPH_DATASET_DIR, help="where the chunks are written")
    args = parser.parse_args(argv)

    for split in ("train", "val", "test"):
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in split_dir.glob("chunk_*.pt"):
            stale.unlink()

        build_split(Path(PARQUET_DIR) / f"{split}.parquet", split, split_dir, args)

        ds = PointGraphDataset(split_dir)
        # .get() rather than ds[i]: indexing a PyG Dataset also accepts slices, so
        # its return type is Data-or-Dataset, while .get() is always one graph
        graphs = [ds.get(i) for i in range(len(ds))]
        nodes = np.array([g.num_nodes for g in graphs])
        edges = np.array([g.num_edges // 2 for g in graphs])
        parts = np.array([int(g.part_id) for g in graphs])
        log.info(
            "%s: %d graphs in %d chunk(s), nodes min/median/max %d/%d/%d, edges min/median/max %d/%d/%d, "
            "%d parts (%.2f views each)",
            split,
            len(ds),
            len(ds.paths),
            nodes.min(),
            np.median(nodes),
            nodes.max(),
            edges.min(),
            np.median(edges),
            edges.max(),
            len(np.unique(parts)),
            len(parts) / max(len(np.unique(parts)), 1),
        )


if __name__ == "__main__":
    main()
