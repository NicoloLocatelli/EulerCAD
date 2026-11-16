"""`DepthAttention`: Attention Residuals (Kimi Team, arXiv 2603.15031, March 2026).

A plain residual stream gives every sublayer (each attention and each MLP) the sum of all
earlier outputs. Here each sublayer reads a learned softmax average of them instead, and the
token rows count as output 0:

    input_l = sum_i softmax_i(w_l . RMSNorm(source_i)) source_i

with one vector `w_l` per sublayer, plus one for the final output. The vectors start at zero,
so every input starts as the plain average of its sources; the paper says this "prevents
training volatility". With `block` above 1 (Block Attention Residuals), the outputs are summed
in groups of `block` sublayers, and the sources are the token rows, the finished groups, and the
unfinished group so far. `block` 1 is the full version.

The paper's RMSNorm on the sources has a learned gain; here it has none, since the gain and
`w_l` multiply each other in the score, so they can only express the same scores. Each score is
computed source by source and the average is summed source by source, instead of stacking the
sources as the paper's pseudocode does: a stack would copy every source again at every sublayer,
and autograd would keep each copy until the backward pass.
"""

import torch
from torch import nn

Source = tuple[torch.Tensor, torch.Tensor]  # a source and its inverse RMS, (..., H) and (...)


class DepthAttention(nn.Module):
    """The vectors `w_l` of `sublayers` sublayers and of the final output, all starting at zero."""

    def __init__(self, hidden_size: int, *, sublayers: int, block: int, eps: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(sublayers + 1, hidden_size))
        self.block = block
        self.eps = eps

    def stream(self, rows: torch.Tensor) -> "DepthStream":
        """Start the sources of one forward pass from the token rows.

        Returns:
            A `DepthStream` whose first `read` is for the first sublayer.
        """
        return DepthStream(self, rows)


class DepthStream:
    """The sources of one forward pass: alternate `read` (a sublayer's input) and `write` (its output).

    The last `read`, after every sublayer has written, is the final output.
    """

    def __init__(self, attention: DepthAttention, rows: torch.Tensor) -> None:
        self._attention = attention
        self._finished = [self._source(rows)]
        self._partial: Source | None = None
        self._written = 0
        self._read = 0

    def _source(self, value: torch.Tensor) -> Source:
        squares = torch.linalg.vector_norm(value, dim=-1, dtype=torch.float32).square()
        return value, torch.rsqrt(squares / value.shape[-1] + self._attention.eps)

    def read(self) -> torch.Tensor:
        """Return the next sublayer's input: the softmax average of the sources so far.

        Returns:
            `(..., H)`, float32 under autocast, like the plain residual stream.
        """
        sources = self._finished if self._partial is None else [*self._finished, self._partial]
        query = self._attention.queries[self._read]
        self._read += 1
        # An elementwise product, not a matmul: autocast would keep a bfloat16 copy of each
        # float32 source for the backward pass of every read.
        scores = torch.stack([(value * query).sum(-1) * inverse for value, inverse in sources])
        weights = scores.softmax(0).unbind(0)
        mixed = weights[0][..., None] * sources[0][0]
        for weight, (value, _) in zip(weights[1:], sources[1:], strict=True):
            mixed = mixed + weight[..., None] * value
        return mixed

    def write(self, output: torch.Tensor) -> None:
        """Add a sublayer's output to the unfinished group, and close the group after `block` outputs."""
        total = output if self._partial is None else self._partial[0] + output
        self._partial = self._source(total)
        self._written += 1
        if self._written % self._attention.block == 0:
            self._finished.append(self._partial)
            self._partial = None
