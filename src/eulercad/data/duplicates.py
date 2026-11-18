"""`DuplicateTable`: the output of `eulerize` for every drawing of a split, computed once.

`eulerize` is deterministic and costs about ten times the walk (`representation/CONTEXT.md`),
so a run reads it from this table instead of recomputing it every epoch. Entries follow the
stored order of the split, above-capacity drawings included, so the table does not depend on
the tokenizer. The table changes no sample: `euler_walk` receives what it would compute.

Each entry carries a digest of the topology it was computed from, the node count and the
canonical edges, which are all `eulerize` reads. `get` refuses an entry whose digest differs
from the graph at hand. `euler_walk` alone cannot catch that: another graph's duplicates can
still make every degree even while being more than the minimum.
"""

import hashlib
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import torch

from eulercad.abc.point_graph import PointGraphDataset
from eulercad.data.drawings import read_drawing
from eulercad.representation.euler import eulerize
from eulercad.representation.graph import DrawingGraph

# Raise when `eulerize` or the stored layout changes; older tables then refuse to load.
FORMAT_VERSION = 2
_TASK = 256

_source: PointGraphDataset | None = None


def _open(root: str) -> None:
    global _source
    torch.set_num_threads(1)
    _source = PointGraphDataset(root)


def topology_digest(graph: DrawingGraph) -> int:
    """Return a stable 64-bit digest of the node count and the canonical edges.

    Coordinates and `sketch_id` do not enter it: `eulerize` does not read them.

    Returns:
        The digest as a signed integer, so it fits an int64 tensor.
    """
    digest = hashlib.blake2b(graph.num_nodes.to_bytes(8, "little"), digest_size=8)
    digest.update(graph.edges.contiguous().numpy().astype("<i8").tobytes())
    return int.from_bytes(digest.digest(), "little", signed=True)


def _eulerize_range(source: PointGraphDataset, bounds: tuple[int, int]) -> list[tuple[torch.Tensor, int]]:
    graphs = (read_drawing(source, position) for position in range(*bounds))
    return [(eulerize(graph), topology_digest(graph)) for graph in graphs]


def _work(bounds: tuple[int, int]) -> list[tuple[torch.Tensor, int]]:
    assert _source is not None
    return _eulerize_range(_source, bounds)


@dataclass(frozen=True, slots=True)
class DuplicateTable:
    """The duplicate edges of every drawing of a split, in stored order.

    Attributes:
        sketch_id: `(N,)` int64, the split's drawings in stored order; checks that a table
            belongs to the split it is used with.
        digest: `(N,)` int64, `topology_digest` of each drawing when the table was built.
        offsets: `(N + 1,)` int64; drawing `i` owns columns `offsets[i]` to `offsets[i + 1]`.
        edges: `(2, M)` int64, every drawing's `eulerize` output, one after the other.
    """

    sketch_id: torch.Tensor
    digest: torch.Tensor
    offsets: torch.Tensor
    edges: torch.Tensor

    def __post_init__(self) -> None:
        count = self.sketch_id.shape[0]
        if self.sketch_id.ndim != 1 or self.digest.shape != (count,) or self.offsets.shape != (count + 1,):
            raise ValueError("sketch_id and digest must be (N,) and offsets (N + 1,)")
        if self.edges.ndim != 2 or self.edges.shape[0] != 2:
            raise ValueError("edges must be (2, M)")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != self.edges.shape[1]:
            raise ValueError("offsets must start at 0 and end at the number of edges")
        if bool((self.offsets.diff() < 0).any()):
            raise ValueError("offsets must not decrease")

    def __len__(self) -> int:
        return self.sketch_id.shape[0]

    def get(self, position: int, graph: DrawingGraph) -> torch.Tensor:
        """Return the duplicates of the drawing stored at `position`, whose graph is `graph`.

        Returns:
            `(2, D)` int64, as `eulerize` returns them.

        Raises:
            ValueError: If the entry was built from another topology; rebuild the table.
        """
        if topology_digest(graph) != int(self.digest[position]):
            raise ValueError(f"the entry of drawing {graph.sketch_id} was built from another topology; rebuild it")
        return self.edges[:, int(self.offsets[position]) : int(self.offsets[position + 1])]

    @classmethod
    def build(cls, root: Path | str, *, workers: int = 0) -> "DuplicateTable":
        """Eulerize every drawing of the split at `root`.

        Args:
            root: a split directory of `PointGraphDataset` chunks.
            workers: worker processes; 0 builds in this process.

        Returns:
            The table, in stored order.
        """
        source = PointGraphDataset(root)
        sketch_id = torch.cat([chunk["sketch_id"] for chunk in source.chunks])
        tasks = [(start, min(start + _TASK, len(source))) for start in range(0, len(source), _TASK)]
        if workers:
            # forkserver: forking this process, which runs torch threads, can deadlock the children.
            context = multiprocessing.get_context("forkserver")
            with context.Pool(workers, initializer=_open, initargs=(str(root),)) as pool:
                parts = pool.map(_work, tasks)
        else:
            parts = [_eulerize_range(source, task) for task in tasks]
        entries = [entry for part in parts for entry in part]
        duplicates = [edges for edges, _ in entries]
        digest = torch.tensor([value for _, value in entries], dtype=torch.int64)
        counts = torch.tensor([0] + [edges.shape[1] for edges in duplicates])
        edges = torch.cat(duplicates, dim=1) if duplicates else torch.empty((2, 0), dtype=torch.int64)
        return cls(sketch_id.clone(), digest, counts.cumsum(0), edges)

    def save(self, path: Path | str) -> None:
        """Write the table, with its format version."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = {"format_version": FORMAT_VERSION, "sketch_id": self.sketch_id, "digest": self.digest}
        torch.save(stored | {"offsets": self.offsets, "edges": self.edges}, path)

    @classmethod
    def load(cls, path: Path | str) -> "DuplicateTable":
        """Read a table written by `save`.

        Returns:
            The stored table.

        Raises:
            ValueError: If the table was written in another format; rebuild it.
        """
        stored = torch.load(path, weights_only=True)
        if stored.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path} has format {stored.get('format_version')}, not {FORMAT_VERSION}; rebuild it")
        return cls(stored["sketch_id"], stored["digest"], stored["offsets"], stored["edges"])
