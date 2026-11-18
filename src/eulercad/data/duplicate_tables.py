"""Build the `eulerize` table of one point-graph split, so training does not recompute it every epoch.

The table holds every drawing of the split in stored order, above-capacity ones included, so
one table serves any tokenizer. See `src/eulercad/data/duplicates.py`.

Output: `data/ABC/duplicates/<dataset folder>/<split>.pt` by default, for
example `data/ABC/duplicates/point_graph_dataset/train.pt`.
"""
# ruff: noqa: T201

import argparse
import time
from pathlib import Path

from eulercad.data.duplicates import DuplicateTable
from eulercad.paths import get_absolute_path

OUTPUT_ROOT = "data/ABC/duplicates"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the eulerize table of one point-graph split.")
    parser.add_argument("--dataset", default="data/ABC/point_graph_dataset/train", help="split directory")
    parser.add_argument("--output", default=None, help=f"default: {OUTPUT_ROOT}/<dataset folder>/<split>.pt")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing table")
    args = parser.parse_args(argv)

    root: Path = get_absolute_path(args.dataset)
    output = get_absolute_path(args.output or f"{OUTPUT_ROOT}/{root.parent.name}/{root.name}.pt")
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")
    start = time.perf_counter()
    table = DuplicateTable.build(root, workers=args.workers)
    table.save(output)
    print(
        f"{args.dataset}: {len(table)} drawings, {table.edges.shape[1]} duplicate edges, "
        f"{time.perf_counter() - start:.0f} s -> {output}"
    )


if __name__ == "__main__":
    main()
