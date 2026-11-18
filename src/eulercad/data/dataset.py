"""`DrawingDataset`: the drawings of one split as next-row samples, tokenized on access.

Each `__getitem__` builds the graph, draws the walk and the identities from the sample's own
generator, and adds the labels. DataLoader workers run it on the CPU while the GPU trains.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from eulercad.abc.point_graph import PointGraphDataset
from eulercad.data.drawings import read_drawing
from eulercad.data.duplicates import DuplicateTable
from eulercad.data.sample import Sample, next_row_labels, sample_rng, suffix_labels
from eulercad.representation.euler import euler_walk
from eulercad.tokenization.tokenizer import GraphTokenizer
from torch.utils.data import Dataset


class DrawingDataset(Dataset[Sample]):
    """The drawings of one split within the tokenizer's node capacity.

    A sample depends only on `seed`, the epoch, and the drawing: not on the loader workers, the
    access order, or the global random state. With `resample_each_epoch`, each epoch draws a
    new walk and new identities (training); without it, every epoch gets the same fixed draw
    (validation and test).

    Attributes:
        root: the split directory.
        tokenizer: turns each walk into rows; its `max_nodes` sets the capacity.
        seed: the run seed.
        resample_each_epoch: whether the epoch enters the sample's generator.
        duplicates: the split's `DuplicateTable`, or None to run `eulerize` on access.
        cuts: rows a sample may be asked to continue; None scores the whole walk.
        kept: `(len,)` int64 stored positions of the drawings within capacity.
        dropped: the number of drawings above capacity.
    """

    def __init__(
        self,
        root: Path | str,
        tokenizer: GraphTokenizer,
        *,
        seed: int,
        resample_each_epoch: bool,
        duplicates: DuplicateTable | None = None,
        cuts: Sequence[int] | None = None,
    ) -> None:
        """
        Raises:
            ValueError: If `duplicates` was built from another split. `__getitem__` also raises
                it for an entry built from another topology.
        """
        self.root = Path(root)
        self.source = PointGraphDataset(root)
        self.tokenizer = tokenizer
        self.seed = seed
        self.resample_each_epoch = resample_each_epoch
        self.duplicates = duplicates
        self.cuts = tuple(cuts) if cuts else None
        nodes = torch.cat([chunk["pos_ptr"].diff() for chunk in self.source.chunks])
        self.kept = (nodes <= tokenizer.config.max_nodes).nonzero().flatten()
        self.dropped = nodes.shape[0] - self.kept.shape[0]
        if duplicates is not None:
            sketch_id = torch.cat([chunk["sketch_id"] for chunk in self.source.chunks])
            if not torch.equal(duplicates.sketch_id, sketch_id):
                raise ValueError("the duplicate table was built from another split")
        # Shared memory, so loader workers, persistent ones included, see `set_epoch`.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __getstate__(self) -> dict[str, object]:
        # Loader workers started by spawn or forkserver reopen the files instead of copying them.
        return {name: value for name, value in self.__dict__.items() if name != "source"}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.source = PointGraphDataset(self.root)

    @property
    def epoch(self) -> int:
        return int(self._epoch)

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch; call it before iterating the loader for that epoch."""
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return self.kept.shape[0]

    def __getitem__(self, index: int) -> Sample:
        """Build the sample of the `index`-th drawing within capacity, at the selected epoch.

        Returns:
            Its rows, labels, and sketch identifier.
        """
        return self.sample(index, self.epoch)

    def sample(self, index: int, epoch: int) -> Sample:
        """Build the sample of the `index`-th drawing within capacity, at `epoch`.

        Without `resample_each_epoch`, `epoch` is ignored and the fixed draw is returned.

        Returns:
            Its rows, labels, and sketch identifier.
        """
        position = int(self.kept[index])
        graph = read_drawing(self.source, position, self.tokenizer.config.max_nodes)
        sketch_id = graph.sketch_id
        rng = sample_rng(self.seed, epoch if self.resample_each_epoch else None, sketch_id)
        duplicates = None if self.duplicates is None else self.duplicates.get(position, graph)
        rows = self.tokenizer.tokenize(graph, euler_walk(graph, rng, duplicates), rng).rows
        if self.cuts is None:
            return Sample(rows, next_row_labels(rows), sketch_id)
        # Drawn after the walk, so asking for a cut never changes the walk itself.
        fitting = [cut for cut in self.cuts if cut < len(rows)]
        return Sample(rows, suffix_labels(rows, rng.choice(fitting) if fitting else len(rows)), sketch_id)
