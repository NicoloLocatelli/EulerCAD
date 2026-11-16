"""The learning-rate schedule: logarithmic warmup, then a decay to a fraction of the peak.

The default, `linear` to a tenth of the peak, is cleaned from
`core/graph_gpt_baseline/training/module.py`, which reproduces the DeepSpeed 0.15.1
`WarmupDecayLR` schedule that GraphGPT v0.5.0 pretraining runs, with `min_lr` a tenth of the
peak as in `examples/train_pretrain.py`.

The other decays keep the same warmup (added on 2026-09-25 to compare schedules):
- `cosine`: half a cosine from the peak to the final rate.
- `wsd` (warmup, stable, decay; Hu et al. 2024, MiniCPM): the peak until the last
  `WSD_DECAY_FRACTION` of the updates (or `decay_fraction`, added 2026-10-13 for the
  half-and-half fine-tunes), then a linear decay to the final rate.
- `wsd_sqrt`: the same, with the `1 - sqrt` decay shape of Hägele et al. 2024.

Derived from GraphGPT (MIT License); see `THIRD_PARTY_NOTICES.md` at the repository root.
"""

import math

import torch
from torch.optim.lr_scheduler import LRScheduler

MIN_LEARNING_RATE_RATIO = 0.1
SCHEDULES = ("linear", "cosine", "wsd", "wsd_sqrt")
WSD_DECAY_FRACTION = 0.2


def learning_rate(
    step: int,
    *,
    peak: float,
    total_steps: int,
    warmup_steps: int,
    schedule: str = "linear",
    final_ratio: float = MIN_LEARNING_RATE_RATIO,
    decay_fraction: float = WSD_DECAY_FRACTION,
) -> float:
    """Return the rate of the update that follows `step` applied updates.

    DeepSpeed builds the schedule without stepping it, so the first update uses the
    optimizer's own rate, the peak. After each update the schedule advances once: a
    logarithmic warmup from `peak / 10` over at least two steps, then the `schedule` decay to
    `peak * final_ratio` at `total_steps`, where it stays. The two WSD schedules decay over the
    last `decay_fraction` of the updates.

    Returns:
        The learning rate; `linear` to a tenth is computed in DeepSpeed's floating-point order.

    Raises:
        ValueError: if `schedule` is not one of `SCHEDULES`.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if step == 0:
        return peak
    iteration = step - 1
    warmup = max(2, warmup_steps)
    if iteration < warmup:
        start = peak * MIN_LEARNING_RATE_RATIO
        return start + (peak - start) * (1.0 / math.log(warmup) * math.log(iteration + 1))
    final = peak * final_ratio
    if schedule == "linear":
        gamma = max(0.0, float(total_steps - iteration) / float(max(1.0, total_steps - warmup)))
    elif schedule == "cosine":
        progress = min(1.0, (iteration - warmup) / max(1, total_steps - warmup))
        gamma = 0.5 * (1 + math.cos(math.pi * progress))
    else:
        decay_start = max(warmup, total_steps - round(decay_fraction * total_steps))
        progress = min(1.0, max(0, iteration - decay_start) / max(1, total_steps - decay_start))
        gamma = 1 - progress if schedule == "wsd" else 1 - math.sqrt(progress)
    return final + (peak - final) * gamma


class WarmupDecay(LRScheduler):
    """PyTorch scheduler for `learning_rate`; `last_epoch` counts applied updates."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak: float,
        total_steps: int,
        warmup_steps: int,
        schedule: str = "linear",
        final_ratio: float = MIN_LEARNING_RATE_RATIO,
        decay_fraction: float = WSD_DECAY_FRACTION,
    ) -> None:
        self.peak = peak
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.final_ratio = final_ratio
        self.decay_fraction = decay_fraction
        super().__init__(optimizer)

    def get_lr(self) -> list[float | torch.Tensor]:
        """Return the same rate for every parameter group.

        Returns:
            One learning rate per parameter group.
        """
        rate = learning_rate(
            self.last_epoch,
            peak=self.peak,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            schedule=self.schedule,
            final_ratio=self.final_ratio,
            decay_fraction=self.decay_fraction,
        )
        return [rate for _ in self.optimizer.param_groups]
