"""Chunked on-disk storage for points-as-nodes sketch graphs.

A whole split as one `torch.save`d list of Data does not scale: the list has to
be built in RAM before writing and rebuilt in RAM on every load. Here a split is
many chunk files, each holding its graphs flattened into a handful of big
tensors plus offset tables. Reading uses `torch.load(mmap=True)`, so the split
never has to fit in RAM - the OS pages in only the graphs actually touched.

Node indices inside a chunk's `edge_index` stay local to their own graph, so
slicing one graph back out needs no offset arithmetic.
"""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

CHUNK_GLOB = "chunk_*.pt"


def write_chunk(graphs: list[Data], path: Path) -> None:
    """Flatten graphs into one chunk file.

    Args:
        graphs: the graphs to store, each with pos, edge_index and sketch_id, and
            optionally part_id - the CAD model a sketch is a projected view of.
            It is written only when every graph carries it, so a chunk can never
            end up with a part_id column that is right for some rows and
            meaningless for the rest.
        path: destination file.
    """
    # PyG types pos/edge_index as optional; sketch_to_data always sets both
    pos: list[torch.Tensor] = [g.pos for g in graphs]  # ty: ignore[invalid-assignment]
    edge_index: list[torch.Tensor] = [g.edge_index for g in graphs]  # ty: ignore[invalid-assignment]
    chunk = {
        "pos": torch.cat(pos),
        "edge_index": torch.cat(edge_index, dim=1),
        "pos_ptr": torch.tensor([0] + [p.shape[0] for p in pos]).cumsum(0),
        "edge_ptr": torch.tensor([0] + [e.shape[1] for e in edge_index]).cumsum(0),
        "sketch_id": torch.cat([g.sketch_id for g in graphs]),
    }
    if graphs and all("part_id" in g for g in graphs):
        chunk["part_id"] = torch.cat([g.part_id for g in graphs])
    torch.save(chunk, path)


class PointGraphDataset(Dataset):
    """Read-only view over a directory of chunk files.

    Subclasses torch_geometric's Dataset rather than torch's: its DataLoader only
    accepts that type. Nothing of the download/process lifecycle runs, because
    `download` and `process` are left unoverridden - the base class checks for
    that. In exchange the base handles `transform` and provides shuffle() and
    slicing, so `len`/`get` are the only two methods to supply.
    """

    def __init__(self, root: Path | str, transform=None) -> None:
        """
        Args:
            root: directory holding the chunk files, e.g. data/ABC/point_graph_dataset/train.
            transform: optional callable applied to each Data on access. This is
                where masking / link-prediction objectives belong - nothing
                task-specific is stored on disk.

        Raises:
            FileNotFoundError: if `root` holds no chunk files.
        """
        self.paths = sorted(Path(root).glob(CHUNK_GLOB))
        if not self.paths:
            raise FileNotFoundError(f"no {CHUNK_GLOB} in {root}")
        self.chunks = [torch.load(p, mmap=True, weights_only=True) for p in self.paths]
        self.offsets = np.cumsum([0] + [len(c["sketch_id"]) for c in self.chunks])
        super().__init__(transform=transform)

    def len(self) -> int:
        """
        Returns:
            Total number of graphs across all chunks.
        """
        return int(self.offsets[-1])

    def get(self, idx: int) -> Data:
        """
        Args:
            idx: global graph index across all chunks.

        Returns:
            One graph. Tensors are cloned out of the memory map, so transforms
            may write to them freely.

            part_id is present only if the chunk carries it, which is how chunk
            directories written before it existed keep loading. Anything that
            groups sketches by their CAD model - the multi-view objectives, the
            retrieval benchmark - should say so rather than silently scoring a
            corpus in which every part looks like a singleton.
        """
        c = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        chunk = self.chunks[c]
        i = idx - self.offsets[c]

        data = Data(
            pos=chunk["pos"][chunk["pos_ptr"][i] : chunk["pos_ptr"][i + 1]].clone(),
            edge_index=chunk["edge_index"][:, chunk["edge_ptr"][i] : chunk["edge_ptr"][i + 1]].clone(),
            sketch_id=chunk["sketch_id"][i : i + 1].clone(),
        )
        if "part_id" in chunk:
            data.part_id = chunk["part_id"][i : i + 1].clone()
        return data

    @property
    def has_part_id(self) -> bool:
        """
        Returns:
            Whether every chunk carries the part_id column. False means this
            directory predates it and needs rebuilding before any objective or
            metric that groups views of the same CAD model can be used.
        """
        return all("part_id" in c for c in self.chunks)
