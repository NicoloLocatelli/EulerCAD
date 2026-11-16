"""`make_optimizer`: AdamW over every parameter, or Muon for the layer matrices and AdamW for the rest.

Muon (MomentUm Orthogonalized by Newton-Schulz) takes each matrix's momentum and replaces it by
the nearest matrix with all singular values near one, so every direction of the matrix moves by
a similar amount. It is defined only for matrices, so embeddings, the field gates, the norms'
gains, the head, and the geometry projection keep AdamW, as Keller Jordan's and Moonshot AI's
recipes do. `adjust_lr_fn="match_rms_adamw"` scales each matrix's step to the size an AdamW step
would have (Moonshot AI, "Muon is Scalable for LLM Training", 2025), so both parts share the
learning rate, its schedule, and the weight decay.

Lightning's automatic optimization takes one optimizer. `MuonAdamW` is one optimizer that holds
both, so gradient clipping, the scheduler, the step count, and checkpoints work as with AdamW.
"""

from typing import Any

import torch
from torch import nn

from eulercad.model.model import GraphGPTModel
from eulercad.training.config import TrainingConfig

MUON_MOMENTUM = 0.95


class MuonAdamW(torch.optim.Optimizer):
    """Muon over `matrices`, AdamW over `others`, stepped together.

    `param_groups` is Muon's groups followed by AdamW's, the same dictionaries, so a scheduler
    that sets `lr` sets both. The two share one `state`, keyed by parameter, so the standard
    `state_dict` and `load_state_dict` cover both.

    Attributes:
        muon: the Muon part.
        adamw: the AdamW part.
    """

    def __init__(
        self,
        matrices: list[nn.Parameter],
        others: list[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        self.muon = torch.optim.Muon(
            matrices, lr=lr, weight_decay=weight_decay, momentum=MUON_MOMENTUM, adjust_lr_fn="match_rms_adamw"
        )
        self.adamw = torch.optim.AdamW(others, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__([*self.muon.param_groups, *self.adamw.param_groups], {})
        self._link()

    def _link(self) -> None:
        """Point both parts at this optimizer's groups and state, also after a load replaced them."""
        count = len(self.muon.param_groups)
        self.muon.param_groups = self.param_groups[:count]
        self.adamw.param_groups = self.param_groups[count:]
        self.muon.state = self.adamw.state = self.state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the groups and state, then link both parts to them again."""
        super().load_state_dict(state_dict)
        self._link()

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Step Muon, then AdamW.

        Returns:
            The closure's loss, or None without a closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss


def make_optimizer(model: GraphGPTModel, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build the optimizer `config.optimizer` names over every parameter of `model`.

    Returns:
        AdamW, or `MuonAdamW` with Muon on the matrices of the backbone's layers.
    """
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
    matrices = {id(value) for value in model.backbone.layers.parameters() if value.ndim == 2}
    return MuonAdamW(
        [value for value in model.parameters() if id(value) in matrices],
        [value for value in model.parameters() if id(value) not in matrices],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
    )
