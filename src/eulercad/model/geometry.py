"""`GeometryFeatures`: what a row's bins say about geometry, as numbers instead of token names.

The embedding table gives every bin its own vector, so the model has to learn from data that
`X_100` lies next to `X_101`; it learned that only over about 20 bins
(`models/graph_gpt_based/diagnostics/embedding_order/`). These features hand it the geometry
directly. They are computed from the input rows alone, row `t` from rows `t - 1` and earlier,
so they add no information the rows lack and keep generation causal. Tokens and outputs stay
as they are.

Per row, in this order:

- the X and Y bin centers as sine and cosine waves, `FREQUENCIES` scales each;
- the step from the previous row's point, `(dx, dy)`, and its direction as a unit vector;
- how many rows so far, this one included, carry the same node identity, one-hot up to `VISITS`;
- a flag, 1 when the row has both coordinate bins.

A row without both coordinate bins (padding, `MASK`) gets zero coordinates and flag, and the row
after it gets no step.
"""

import math

import torch
from torch import nn

from eulercad.tokenization.vocabulary import Vocabulary

FREQUENCIES = 8
VISITS = 4


def geometry_features(input_ids: torch.Tensor, vocabulary: Vocabulary, coordinate_bins: int) -> torch.Tensor:
    """Compute the features of every row from its tokens and the tokens of the rows before it.

    Args:
        input_ids: `(B, T, F)` int64 token rows, `F` at least 3: node, X, and Y come first.
        vocabulary: the token families the rows use.
        coordinate_bins: bins per axis over `[-1, 1]`.

    Returns:
        `(B, T, GeometryFeatures.width())` float32.
    """
    node, x, y = input_ids[..., 0], input_ids[..., 1], input_ids[..., 2]
    placed = vocabulary.x.contains(x) & vocabulary.y.contains(y)
    index = torch.stack([x - vocabulary.x.start, y - vocabulary.y.start], dim=-1).float()
    point = torch.where(placed[..., None], -1 + (index + 0.5) * (2 / coordinate_bins), 0.0)

    scales = math.pi * 2 ** torch.arange(FREQUENCIES, device=input_ids.device, dtype=torch.float32)
    angles = point[..., None] * scales
    waves = torch.stack([angles.sin(), angles.cos()], dim=-1) * placed[..., None, None, None]

    previous = torch.cat([torch.zeros_like(point[:, :1]), point[:, :-1]], dim=1)
    stepped = placed & torch.cat([torch.zeros_like(placed[:, :1]), placed[:, :-1]], dim=1)
    step = torch.where(stepped[..., None], point - previous, 0.0)
    length = step.norm(dim=-1, keepdim=True)
    direction = torch.where(length > 0, step / length.clamp_min(1e-12), 0.0)

    identified = vocabulary.node.contains(node)
    rows = input_ids.shape[1]
    earlier = torch.ones(rows, rows, dtype=torch.bool, device=input_ids.device).tril()
    same = (node[:, :, None] == node[:, None, :]) & identified[:, None, :] & earlier
    visits = torch.where(identified, same.sum(dim=-1), 0).clamp_max(VISITS)
    visited = nn.functional.one_hot(visits, VISITS + 1)[..., 1:].float()

    return torch.cat([waves.flatten(-3), step, direction, visited, placed[..., None].float()], dim=-1)


class GeometryFeatures(nn.Module):
    """Project the features of every row to one `(B, T, H)` vector, added to the merged row.

    The projection starts at zero, so a new model computes exactly what it would without it.
    """

    def __init__(self, vocabulary: Vocabulary, coordinate_bins: int, hidden_size: int) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.coordinate_bins = coordinate_bins
        self.projection = nn.Linear(self.width(), hidden_size, bias=False)
        nn.init.zeros_(self.projection.weight)

    @staticmethod
    def width() -> int:
        """Count the features of one row.

        Returns:
            The last dimension of `geometry_features`.
        """
        return 4 * FREQUENCIES + 2 + 2 + VISITS + 1

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the projected features.

        Returns:
            `(B, T, H)` vectors.
        """
        return self.projection(geometry_features(input_ids, self.vocabulary, self.coordinate_bins))
