"""`TrainingStream`: the endless stream of training batches, in which batch `k` depends only on the seed and `k`.

Training can therefore resume at any batch, and the number of loader workers cannot change the
order. Cleaned from `core/graph_gpt_baseline/data/examples.py` (`PretrainingBatches`).

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import itertools
import random
from collections.abc import Iterator

from eulercad.data.collate import MAX_ROWS, Batch, collate
from eulercad.data.dataset import DrawingDataset
from torch.utils.data import IterableDataset, get_worker_info


class TrainingStream(IterableDataset[Batch]):
    """Endless training batches, beginning at batch `start`.

    Batch `k` holds stream positions `k * B` to `k * B + B - 1`. Position `p` is drawing
    `order(e)[p mod N]` of epoch `e = p // N`, sampled at epoch `e`, where each epoch is a
    seeded permutation of the `N` drawings. A batch can therefore span two epochs. Worker `w`
    of `W` builds batches `start + w`, `start + w + W`, and so on; the loader returns them in
    index order. Use it with `DataLoader(stream, batch_size=None, ...)`.

    Attributes:
        dataset: the training split; its `seed` seeds the order and every sample.
        batch_size: samples per batch.
        start: the first batch yielded.
        max_rows: passed to `collate`.
    """

    def __init__(self, dataset: DrawingDataset, batch_size: int, *, start: int = 0, max_rows: int = MAX_ROWS) -> None:
        """
        Raises:
            ValueError: If `batch_size` is not positive, `start` is negative, or the split is empty.
        """
        super().__init__()
        if batch_size < 1 or start < 0:
            raise ValueError("batch_size must be positive and start non-negative")
        if not len(dataset):
            raise ValueError("the training split has no drawing within capacity")
        self.dataset = dataset
        self.batch_size = batch_size
        self.start = start
        self.max_rows = max_rows
        self._epoch: int | None = None
        self._order: list[int] = []

    def order(self, epoch: int) -> list[int]:
        """Return the drawing indices of `epoch` in stream order.

        Returns:
            A permutation of `range(len(dataset))`, drawn from its own string-keyed generator.
        """
        order = list(range(len(self.dataset)))
        key = f"{self.dataset.seed}:{epoch}:order"
        random.Random(key).shuffle(order)  # noqa: S311  # nosec B311
        return order

    def position(self, position: int) -> tuple[int, int]:
        """Locate one stream position.

        Returns:
            The drawing index and the epoch.
        """
        epoch, offset = divmod(position, len(self.dataset))
        if epoch != self._epoch:
            self._order = self.order(epoch)
            self._epoch = epoch
        return self._order[offset], epoch

    def batch(self, index: int) -> Batch:
        """Build batch `index`.

        Returns:
            The padded batch of its `batch_size` positions.
        """
        first = index * self.batch_size
        positions = map(self.position, range(first, first + self.batch_size))
        return collate([self.dataset.sample(drawing, epoch) for drawing, epoch in positions], max_rows=self.max_rows)

    def __iter__(self) -> Iterator[Batch]:
        """Yield this worker's share of the stream, from `start` on.

        Yields:
            Batches in increasing index order.
        """
        worker = get_worker_info()
        first, step = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        for index in itertools.count(self.start + first, step):
            yield self.batch(index)
