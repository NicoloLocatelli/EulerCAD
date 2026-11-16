"""`TrainingConfig`: the settings that decide what training computes, plain values that can be written to JSON.

The optimizer values are the baseline's pretraining defaults
(`core/graph_gpt_baseline/contracts/training_config.py`, `PretrainingConfig`). The cell budget
and compilation are here too: with dropout, they change the random draws and the arithmetic of
each update, so a run keeps them from start to end. Settings that change no computed value
(workers, record intervals) are not here.
"""

from dataclasses import asdict, dataclass
from typing import Any

from eulercad.training.schedule import (
    MIN_LEARNING_RATE_RATIO,
    SCHEDULES,
    WSD_DECAY_FRACTION,
)

PRECISIONS = ("bf16-mixed", "32-true")
# torchao's FP8 training recipes: one scale per tensor (fastest) or one per row (handles outliers).
FLOAT8_RECIPES = ("tensorwise", "rowwise")
OPTIMIZERS = ("adamw", "muon")
SIGREG_LAYERS = ("final", "all")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The optimization settings of one pretraining run.

    Attributes:
        total_steps: optimizer updates of the whole run; the schedule ends here.
        warmup_steps: updates of the logarithmic warmup.
        batch_size: drawings per update.
        learning_rate: the peak rate.
        weight_decay: AdamW's decoupled weight decay, applied to every parameter.
        adam_betas: AdamW's moment coefficients.
        adam_epsilon: AdamW's denominator term.
        max_gradient_norm: the global gradient norm is clipped to this value.
        precision: `bf16-mixed` (bfloat16 autocast; the loss stays float32) or `32-true`.
        seed: the run seed: weights, stream order, samples, and dropout.
        cell_budget: each batch runs in length-sorted groups of at most this many rows times
            drawings (`microbatch.py`); None runs it in one pass.
        compile_backend: the `torch.compile` backend of the forward pass, or None for eager.
        float8: the torchao recipe that runs the layer products in FP8 (8-bit floats), or None
            for bfloat16. Weights, optimizer, and loss keep their precision (`module.py`).
        optimizer: `adamw` over every parameter, or `muon`: Muon for the matrices of the layers
            and AdamW for the rest, at the same learning rate and weight decay (`optimizer.py`).
        schedule: the decay after the warmup: `linear` (the baseline's), `cosine`, `wsd`, or
            `wsd_sqrt` (`schedule.py`).
        final_learning_rate_ratio: the rate at `total_steps`, as a fraction of the peak.
        wsd_decay_fraction: the last share of the updates over which `wsd` and `wsd_sqrt` decay.
        sigreg_weight: None trains on the next-row loss alone. A weight in [0, 1) trains on
            `(1 - weight) * next_row + weight * SIGReg`, SIGReg read on the standardized mean
            final-norm state of each drawing (`losses/sigreg.py`); 0 records SIGReg without
            training on it. Added 2026-10-13 for the SIGReg fine-tunes.
        sigreg_layers: `final`, SIGReg on the final norm alone, or `all`, the mean of SIGReg on every
            layer's output and the final norm (added 2026-10-14).
        sigreg_standardize: whether SIGReg reads the standardized means (`losses/sigreg.py`
            `standardize`) or the raw ones (added 2026-10-14).
    """

    total_steps: int
    warmup_steps: int
    batch_size: int = 160
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    max_gradient_norm: float = 5.0
    precision: str = "bf16-mixed"
    seed: int = 0
    cell_budget: int | None = 8192
    compile_backend: str | None = None
    float8: str | None = None
    optimizer: str = "adamw"
    schedule: str = "linear"
    final_learning_rate_ratio: float = MIN_LEARNING_RATE_RATIO
    wsd_decay_fraction: float = WSD_DECAY_FRACTION
    sigreg_weight: float | None = None
    sigreg_layers: str = "final"
    sigreg_standardize: bool = True

    def __post_init__(self) -> None:
        if self.total_steps < 1 or not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("total_steps must be positive and exceed warmup_steps")
        if self.batch_size < 1 or self.learning_rate <= 0 or self.adam_epsilon <= 0:
            raise ValueError("batch_size, learning_rate, and adam_epsilon must be positive")
        if self.weight_decay < 0 or self.max_gradient_norm <= 0:
            raise ValueError("weight_decay must be non-negative and max_gradient_norm positive")
        if len(self.adam_betas) != 2 or not all(0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("adam_betas must be two values in [0, 1)")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if self.cell_budget is not None and self.cell_budget < 1:
            raise ValueError("cell_budget must be positive or None")
        if self.compile_backend is not None and not self.compile_backend:
            raise ValueError("compile_backend must be a backend name or None")
        self._check_float8_and_optimizer()

    def _check_float8_and_optimizer(self) -> None:
        if self.float8 is not None and self.float8 not in FLOAT8_RECIPES:
            raise ValueError(f"float8 must be one of {FLOAT8_RECIPES} or None")
        if self.float8 is not None and self.cell_budget is None:
            raise ValueError("float8 needs a cell_budget: only groups can be padded to 16 rows")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if not 0 <= self.final_learning_rate_ratio <= 1:
            raise ValueError("final_learning_rate_ratio must be in [0, 1]")
        if not 0 < self.wsd_decay_fraction <= 1:
            raise ValueError("wsd_decay_fraction must be in (0, 1]")
        if self.sigreg_weight is not None and not 0 <= self.sigreg_weight < 1:
            raise ValueError("sigreg_weight must be None or in [0, 1)")
        if self.sigreg_layers not in SIGREG_LAYERS:
            raise ValueError(f"sigreg_layers must be one of {SIGREG_LAYERS}")

    def to_dict(self) -> dict[str, Any]:
        """Return the settings, ready for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Rebuild a configuration from `to_dict()`, also after a JSON round trip.

        Returns:
            A configuration equal to the one that was saved.
        """
        settings: dict[str, Any] = {**data, "adam_betas": tuple(data["adam_betas"])}
        return cls(**settings)
