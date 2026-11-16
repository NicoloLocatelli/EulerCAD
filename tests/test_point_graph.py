"""Tests for the chunked point-graph store, focused on the part_id column.

part_id is what makes the multi-view objectives and the retrieval benchmark
possible: it says which CAD model a sketch is a projected view of. It is also
the kind of column whose absence is silent - a dataset without it still loads,
still trains, and only shows up as a retrieval score near chance, which reads as
a bad model rather than a missing field. Hence `has_part_id`, and hence these.

Plain asserts and a __main__ runner, matching tests/test_gnn04_wiring.py -
pytest is not a project dependency. Chunks are written to a temp directory, so
none of this needs the ABC dataset to be present.
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data

from eulercad.abc.point_graph import PointGraphDataset, write_chunk


def _graph(n_nodes: int, sketch_id: int, part_code: int | None = None) -> Data:
    """One small graph, a path through n_nodes points.

    Returns:
        Data with pos, edge_index, sketch_id, and part_id when part_code is given.
    """
    pos = torch.arange(n_nodes * 2, dtype=torch.float64).reshape(n_nodes, 2)
    line = torch.stack([torch.arange(n_nodes - 1), torch.arange(1, n_nodes)])
    data = Data(
        pos=pos,
        edge_index=torch.cat([line, line.flip(0)], dim=1),
        sketch_id=torch.tensor([sketch_id], dtype=torch.long),
    )
    if part_code is not None:
        data.part_id = torch.tensor([part_code], dtype=torch.long)
    return data


def _written(graphs: list[Data], per_chunk: int = 2) -> PointGraphDataset:
    """Write graphs out across several chunk files and read them back.

    Returns:
        A dataset over a temp directory. The directory is intentionally not
        cleaned up: it lives for the test process and the OS reclaims it.
    """
    root = Path(tempfile.mkdtemp())
    for i in range(0, len(graphs), per_chunk):
        write_chunk(graphs[i : i + per_chunk], root / f"chunk_{i // per_chunk:04d}.pt")
    return PointGraphDataset(root)


def test_part_id_survives_the_round_trip_in_the_right_order():
    """Chunking flattens every graph into shared tensors, so a wrong offset would
    hand back another sketch's part. Distinct part codes per sketch catch that."""
    graphs = [_graph(4 + i, sketch_id=100 + i, part_code=i // 2) for i in range(6)]
    ds = _written(graphs)

    assert ds.has_part_id
    for i in range(len(ds)):
        got = ds.get(i)
        assert int(got.sketch_id) == 100 + i, f"sketch {i} came back as {int(got.sketch_id)}"
        assert int(got.part_id) == i // 2, f"sketch {i} came back with part {int(got.part_id)}"
        assert got.num_nodes == 4 + i


def test_a_chunk_without_part_id_still_loads():
    """gnn_04 through gnn_08 all ran against chunk directories written before
    part_id existed. Those must keep loading, or every earlier study becomes
    unreproducible the moment this column is added."""
    ds = _written([_graph(4, sketch_id=i) for i in range(4)])

    assert not ds.has_part_id
    got = ds.get(0)
    assert "part_id" not in got, "a part_id appeared on a dataset that has none"
    assert got.num_nodes == 4


def test_a_half_labelled_chunk_refuses_to_write_the_column():
    """A chunk where only some graphs carry part_id would produce a column that is
    correct for some rows and silently wrong for the rest - worse than absent."""
    mixed = [_graph(4, sketch_id=0, part_code=7), _graph(4, sketch_id=1)]
    ds = _written(mixed, per_chunk=2)

    assert not ds.has_part_id
    assert "part_id" not in ds.get(0)


def test_part_id_batches_as_one_value_per_graph():
    """The retrieval and contrastive objectives read part_id off a batch, so it has
    to concatenate one entry per graph. A key PyG decided to treat as an index
    would be offset per graph instead, quietly renumbering the parts."""
    from torch_geometric.loader import DataLoader

    graphs = [_graph(4, sketch_id=i, part_code=i % 3) for i in range(6)]
    batch = next(iter(DataLoader(_written(graphs), batch_size=6, shuffle=False)))

    assert batch.part_id.tolist() == [0, 1, 2, 0, 1, 2], batch.part_id
    assert batch.num_graphs == 6


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
