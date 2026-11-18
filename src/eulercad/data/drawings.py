"""Read one stored drawing as a checked `DrawingGraph`."""

from eulercad.abc.point_graph import PointGraphDataset
from eulercad.representation.graph import DrawingGraph


def read_drawing(source: PointGraphDataset, position: int, max_nodes: int | None = None) -> DrawingGraph:
    """Build the graph of the drawing stored at `position`; `max_nodes=None` accepts any size.

    `DrawingGraph.from_point_graph` raises `GraphCapacityError` above `max_nodes`.

    Returns:
        The checked graph in the model frame.

    Raises:
        ValueError: If the stored drawing has no coordinates or no edges.
    """
    data = source.get(position)
    pos, edge_index = data.pos, data.edge_index
    if pos is None or edge_index is None:
        raise ValueError(f"the drawing stored at {position} has no pos or edge_index")
    capacity = pos.shape[0] if max_nodes is None else max_nodes
    return DrawingGraph.from_point_graph(int(data.sketch_id), pos, edge_index, max_nodes=capacity)
