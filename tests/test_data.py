"""Tests for the data stage of the GraphGPT-based architecture: samples, duplicates, dataset, batches."""

import itertools
import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from eulercad.abc.point_graph import write_chunk
from eulercad.data.collate import IGNORED_LABEL, Batch, collate
from eulercad.data.dataset import DrawingDataset
from eulercad.data.duplicates import FORMAT_VERSION, DuplicateTable, topology_digest
from eulercad.data.sample import Sample, next_row_labels, sample_rng, suffix_labels
from eulercad.data.stream import TrainingStream
from eulercad.representation.euler import euler_walk, eulerize
from eulercad.representation.graph import DrawingGraph
from eulercad.tokenization.rows import ROW_WIDTH
from eulercad.tokenization.tokenizer import GraphTokenizer, TokenizerConfig
from eulercad.tokenization.vocabulary import EOS, PAD

# Capacity 64 nodes, so the 70-node drawing below is dropped.
SMALL = GraphTokenizer(TokenizerConfig(max_nodes=64, coordinate_bins=64, distance_bins=32, d_min=0.01, d_max=2.0))
# (sketch_id, nodes) per chunk: two chunks, one drawing above capacity, one isolated node.
CHUNKS = [[(10, 5), (11, 70), (12, 1)], [(13, 20), (14, 8)]]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  # nosec B311


def _drawing(sketch_id: int, num_nodes: int) -> Data:
    generator = _rng(sketch_id)
    pos = [[generator.uniform(-50, 50), generator.uniform(-50, 50)] for _ in range(num_nodes)]
    pairs: set[tuple[int, int]] = set()
    for _ in range(num_nodes + generator.randint(0, num_nodes) if num_nodes > 1 else 0):
        a, b = generator.sample(range(num_nodes), 2)
        pairs.add((min(a, b), max(a, b)))
    edges = torch.tensor(sorted(pairs), dtype=torch.int64).reshape(-1, 2).T
    return Data(
        pos=torch.tensor(pos, dtype=torch.float64),
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
        sketch_id=torch.tensor([sketch_id]),
    )


def _split(root: Path, chunks: list[list[tuple[int, int]]] = CHUNKS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index, chunk in enumerate(chunks):
        write_chunk([_drawing(sketch_id, nodes) for sketch_id, nodes in chunk], root / f"chunk_{index:03d}.pt")
    return root


def _graph(data: Data, max_nodes: int) -> DrawingGraph:
    assert data.pos is not None
    assert data.edge_index is not None
    return DrawingGraph.from_point_graph(int(data.sketch_id), data.pos, data.edge_index, max_nodes=max_nodes)


@pytest.fixture
def split(tmp_path: Path) -> Path:
    return _split(tmp_path / "train")


def _samples(dataset: DrawingDataset) -> list[Sample]:
    return [dataset[index] for index in range(len(dataset))]


def _same(first: list[Sample], second: list[Sample]) -> bool:
    return all(
        a.sketch_id == b.sketch_id and torch.equal(a.input_ids, b.input_ids) and torch.equal(a.labels, b.labels)
        for a, b in zip(first, second, strict=True)
    )


# Samples


def test_sample_rng_depends_on_every_part_of_its_key() -> None:
    def draws(rng: random.Random) -> list[float]:
        return [rng.random() for _ in range(4)]

    reference = draws(sample_rng(0, 3, 10))

    assert draws(sample_rng(0, 3, 10)) == reference
    for other in (sample_rng(1, 3, 10), sample_rng(0, 4, 10), sample_rng(0, None, 10), sample_rng(0, 3, 11)):
        assert draws(other) != reference
    assert draws(sample_rng(0, 3, 10, "completion")) != reference


def test_next_row_labels_shift_the_rows_and_end_with_the_eos_row() -> None:
    rows = torch.arange(15, dtype=torch.int64).reshape(3, 5) + 10
    original = rows.clone()
    labels = next_row_labels(rows)

    assert torch.equal(labels[:2], rows[1:])
    assert torch.equal(labels[2], torch.full((5,), EOS))
    assert torch.equal(rows, original)


# Duplicate table


def test_duplicate_table_holds_eulerize_for_every_drawing(split: Path) -> None:
    table = DuplicateTable.build(split)
    drawings = [_drawing(sketch_id, nodes) for chunk in CHUNKS for sketch_id, nodes in chunk]

    assert table.sketch_id.tolist() == [10, 11, 12, 13, 14]
    for position, data in enumerate(drawings):
        graph = _graph(data, max_nodes=10**6)
        assert torch.equal(table.get(position, graph), eulerize(graph))
        assert int(table.digest[position]) == topology_digest(graph)
    assert bool(table.offsets.diff().any())


def test_duplicate_table_is_the_same_with_worker_processes(split: Path) -> None:
    serial, parallel = DuplicateTable.build(split), DuplicateTable.build(split, workers=2)

    for name in ("sketch_id", "digest", "offsets", "edges"):
        assert torch.equal(getattr(serial, name), getattr(parallel, name))


def test_duplicate_table_round_trips_and_checks_its_format(split: Path, tmp_path: Path) -> None:
    table = DuplicateTable.build(split)
    path = tmp_path / "tables" / "train.pt"
    table.save(path)
    loaded = DuplicateTable.load(path)

    for name in ("sketch_id", "digest", "offsets", "edges"):
        assert torch.equal(getattr(loaded, name), getattr(table, name))
    stored = torch.load(path, weights_only=True)
    torch.save(stored | {"format_version": FORMAT_VERSION + 1}, path)
    with pytest.raises(ValueError, match="format"):
        DuplicateTable.load(path)


@pytest.mark.parametrize(
    "offsets",
    [torch.tensor([0, 2, 1, 3]), torch.tensor([1, 2, 3, 3]), torch.tensor([0, 1, 2, 4]), torch.tensor([0, 1, 3])],
)
def test_duplicate_table_rejects_inconsistent_offsets(offsets: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        DuplicateTable(
            torch.tensor([1, 2, 3]), torch.zeros(3, dtype=torch.int64), offsets, torch.zeros((2, 3), dtype=torch.int64)
        )


def test_duplicate_table_needs_one_digest_per_drawing() -> None:
    with pytest.raises(ValueError):
        DuplicateTable(
            torch.tensor([1, 2]),
            torch.zeros(3, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
            torch.zeros((2, 0), dtype=torch.int64),
        )


def test_topology_digest_follows_the_node_count_and_the_edges_only() -> None:
    edges = torch.tensor([[0, 1], [1, 2]])
    square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    graph = DrawingGraph(0, square, edges, (0.0, 0.0), 1.0)
    moved = DrawingGraph(5, square / 2, edges.clone(), (3.0, 1.0), 2.0)
    more_edges = DrawingGraph(0, square, torch.tensor([[0, 0, 1], [1, 2, 2]]), (0.0, 0.0), 1.0)
    more_nodes = DrawingGraph(0, torch.cat([square, square[:1]]), edges, (0.0, 0.0), 1.0)

    assert topology_digest(moved) == topology_digest(graph)
    assert topology_digest(more_edges) != topology_digest(graph)
    assert topology_digest(more_nodes) != topology_digest(graph)


# Dataset


def test_drawings_above_capacity_are_dropped_and_counted(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)

    assert (len(dataset), dataset.dropped) == (4, 1)
    assert [sample.sketch_id for sample in _samples(dataset)] == [10, 12, 13, 14]


def test_a_sample_draws_the_walk_then_the_identities_from_one_generator(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=7, resample_each_epoch=True)
    dataset.set_epoch(3)
    kept = [(10, 5), (12, 1), (13, 20), (14, 8)]

    for index, (sketch_id, nodes) in enumerate(kept):
        data = _drawing(sketch_id, nodes)
        graph = _graph(data, max_nodes=64)
        rng = sample_rng(7, 3, sketch_id)
        rows = SMALL.tokenize(graph, euler_walk(graph, rng), rng).rows
        sample = dataset[index]
        assert sample.sketch_id == sketch_id
        assert torch.equal(sample.input_ids, rows)
        assert torch.equal(sample.labels, next_row_labels(rows))


def test_training_samples_repeat_within_an_epoch_and_change_across_epochs(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    first = _samples(dataset)
    again = _samples(dataset)
    dataset.set_epoch(1)

    assert dataset.epoch == 1
    assert _same(first, again)
    assert not _same(first, _samples(dataset))


def test_fixed_samples_ignore_the_epoch(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=False)
    first = _samples(dataset)
    dataset.set_epoch(5)

    assert _same(first, _samples(dataset))
    data = _drawing(10, 5)
    graph = _graph(data, max_nodes=64)
    rng = sample_rng(0, None, 10)
    assert torch.equal(first[0].input_ids, SMALL.tokenize(graph, euler_walk(graph, rng), rng).rows)


def test_sample_builds_the_epoch_it_is_given(split: Path) -> None:
    training = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    fixed = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=False)
    training.set_epoch(3)

    assert _same([training.sample(index, 3) for index in range(len(training))], _samples(training))
    assert _same([fixed.sample(index, 3) for index in range(len(fixed))], _samples(fixed))


def test_the_duplicate_table_changes_no_sample(split: Path) -> None:
    table = DuplicateTable.build(split)
    plain = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    cached = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True, duplicates=table)

    assert _same(_samples(plain), _samples(cached))


def test_a_duplicate_table_of_another_split_is_rejected(split: Path, tmp_path: Path) -> None:
    other = DuplicateTable.build(_split(tmp_path / "val", [[(20, 5), (21, 6)]]))

    with pytest.raises(ValueError, match="another split"):
        DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True, duplicates=other)


def _write(root: Path, sketch_id: int, pairs: list[tuple[int, int]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    edges = torch.tensor(pairs).T
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    data = Data(pos=pos, edge_index=torch.cat([edges, edges.flip(0)], 1), sketch_id=torch.tensor([sketch_id]))
    write_chunk([data], root / "chunk_000.pt")
    return root


def test_a_table_entry_of_another_topology_is_rejected(tmp_path: Path) -> None:
    # A star cached three duplicates; the complete graph on the same nodes needs two. Both
    # sets make every degree even, so only the topology digest tells them apart.
    star = DuplicateTable.build(_write(tmp_path / "old", 30, [(0, 1), (0, 2), (0, 3)]))
    split = _write(tmp_path / "new", 30, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    complete = DuplicateTable.build(split)
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True, duplicates=star)

    assert (star.edges.shape[1], complete.edges.shape[1]) == (3, 2)
    with pytest.raises(ValueError, match="topology"):
        dataset[0]


def test_loader_workers_follow_the_epoch(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    loader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate,
        multiprocessing_context="forkserver",
    )

    for epoch in (0, 1):
        dataset.set_epoch(epoch)
        batches = list(loader)
        expected = [collate(_samples(dataset)[:3]), collate(_samples(dataset)[3:])]
        for batch, reference in zip(batches, expected, strict=True):
            assert torch.equal(batch.input_ids, reference.input_ids)
            assert torch.equal(batch.labels, reference.labels)


# Training stream


def _same_batch(first: Batch, second: Batch) -> bool:
    return all(
        torch.equal(getattr(first, name), getattr(second, name))
        for name in ("input_ids", "attention_mask", "labels", "sketch_id")
    )


def test_the_stream_visits_every_drawing_once_per_epoch(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    stream = TrainingStream(dataset, batch_size=3)
    # Four batches of three hold three epochs of the four drawings.
    sketch_ids = torch.cat([stream.batch(index).sketch_id for index in range(4)]).tolist()

    epochs = [sketch_ids[start : start + 4] for start in (0, 4, 8)]
    assert all(sorted(epoch) == [10, 12, 13, 14] for epoch in epochs)
    assert len({tuple(epoch) for epoch in epochs}) > 1


def test_each_stream_position_samples_its_own_epoch(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    stream = TrainingStream(dataset, batch_size=3)

    for index in range(4):
        positions = [stream.position(position) for position in range(3 * index, 3 * index + 3)]
        expected = collate([dataset.sample(drawing, epoch) for drawing, epoch in positions])
        assert _same_batch(stream.batch(index), expected)
    assert [epoch for _, epoch in map(stream.position, range(9))] == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    assert sorted(stream.order(1)) == [0, 1, 2, 3]


def test_the_stream_order_depends_on_the_seed(split: Path) -> None:
    orders = [
        TrainingStream(DrawingDataset(split, SMALL, seed=seed, resample_each_epoch=True), batch_size=3).order(epoch)
        for seed in range(4)
        for epoch in range(4)
    ]

    assert len({tuple(order) for order in orders}) > 1
    first = TrainingStream(DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True), batch_size=3)
    assert first.order(2) == orders[2]


def test_a_stream_started_at_k_begins_with_batch_k(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)

    resumed = list(itertools.islice(TrainingStream(dataset, batch_size=3, start=2), 2))

    reference = TrainingStream(dataset, batch_size=3)
    assert all(_same_batch(batch, reference.batch(2 + index)) for index, batch in enumerate(resumed))


def test_loader_workers_keep_the_stream_order(split: Path) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)
    stream = TrainingStream(dataset, batch_size=3, start=1)
    loader = DataLoader(stream, batch_size=None, num_workers=2, multiprocessing_context="forkserver")

    batches = list(itertools.islice(loader, 5))

    assert all(_same_batch(batch, stream.batch(1 + index)) for index, batch in enumerate(batches))


@pytest.mark.parametrize("settings", [{"batch_size": 0}, {"batch_size": 2, "start": -1}])
def test_the_stream_rejects_bad_settings(split: Path, settings: dict[str, int]) -> None:
    dataset = DrawingDataset(split, SMALL, seed=0, resample_each_epoch=True)

    with pytest.raises(ValueError):
        TrainingStream(dataset, **settings)


# Batches


def _sample(rows: int, sketch_id: int = 0, start: int = 10) -> Sample:
    input_ids = torch.arange(start, start + rows * 5, dtype=torch.int64).reshape(rows, 5)
    return Sample(input_ids, next_row_labels(input_ids), sketch_id)


def test_collate_right_pads_to_a_multiple_of_eight() -> None:
    samples = [_sample(3, 1), _sample(10, 2)]
    batch = collate(samples)

    assert isinstance(batch, Batch)
    assert tuple(batch.input_ids.shape) == (2, 16, 5)
    assert batch.attention_mask.dtype == torch.bool
    assert batch.attention_mask.sum(dim=1).tolist() == [3, 10]
    for index, sample in enumerate(samples):
        rows = len(sample.input_ids)
        assert torch.equal(batch.input_ids[index, :rows], sample.input_ids)
        assert torch.equal(batch.labels[index, :rows], sample.labels)
        assert bool((batch.input_ids[index, rows:] == PAD).all())
        assert bool((batch.labels[index, rows:] == IGNORED_LABEL).all())
    assert batch.sketch_id.tolist() == [1, 2]


def test_collate_caps_the_length_and_keeps_the_leading_rows() -> None:
    long = _sample(12)
    batch = collate([long, _sample(2)], max_rows=9)

    assert tuple(batch.input_ids.shape) == (2, 9, 5)
    assert torch.equal(batch.input_ids[0], long.input_ids[:9])
    assert torch.equal(batch.labels[0], long.labels[:9])
    assert bool(batch.attention_mask[0].all())


@pytest.mark.parametrize(
    ("samples", "settings"),
    [
        ([], {}),
        ([_sample(2), Sample(torch.zeros((2, 4), dtype=torch.int64), torch.zeros((2, 4), dtype=torch.int64), 0)], {}),
        ([_sample(2)], {"max_rows": 0}),
        ([_sample(2)], {"pad_to_multiple_of": 0}),
    ],
)
def test_collate_rejects_bad_input(samples: list[Sample], settings: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        collate(samples, **settings)


def test_a_sample_needs_matching_rows_and_labels() -> None:
    with pytest.raises(ValueError):
        Sample(torch.zeros((3, 5), dtype=torch.int64), torch.zeros((2, 5), dtype=torch.int64), 0)


def test_the_suffix_labels_score_only_the_cut_and_the_end_row() -> None:
    rows = torch.arange(7 * ROW_WIDTH).view(7, ROW_WIDTH)

    labels = suffix_labels(rows, 3)

    assert torch.equal(labels[:3], torch.full((3, ROW_WIDTH), IGNORED_LABEL))
    assert torch.equal(labels[3:6], rows[4:7])  # the three rows the cut removed
    assert torch.equal(labels[6], torch.full((ROW_WIDTH,), EOS))  # and the end row after them
    assert torch.equal(suffix_labels(rows, 99), next_row_labels(rows))


def test_a_cut_dataset_scores_a_suffix_and_draws_the_same_cut_again(split: Path) -> None:
    tokenizer = GraphTokenizer()
    whole = DrawingDataset(split, tokenizer, seed=0, resample_each_epoch=True)
    cut = DrawingDataset(split, tokenizer, seed=0, resample_each_epoch=True, cuts=(2, 3))

    for index in range(len(whole)):
        plain, short = whole.sample(index, 4), cut.sample(index, 4)
        assert torch.equal(plain.input_ids, short.input_ids)  # the walk itself is untouched
        scored = (short.labels != IGNORED_LABEL).all(-1)
        assert torch.equal(short.labels[scored], plain.labels[scored])
        assert int(scored.sum()) in {2 + 1, 3 + 1, len(plain.labels)}
        assert bool(scored[-1]) and torch.equal(short.labels, cut.sample(index, 4).labels)
