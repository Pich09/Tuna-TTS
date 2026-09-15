"""
WSD (Warmup -> Stable -> Decay -> Stable) learning-rate schedule
(PLAN.md section 16), implemented as a custom scheduler rather than a
standard cosine schedule, per the plan's explicit instruction.

Two independent stages are configured (one per optimizer param group:
LoRA, Khmer embedding -- PLAN.md section 15), each with its own
warmup-start / peak / decay-end values but sharing the same step
boundaries (0-2k warmup, 2k-10k stable, 10k-35k decay, 35k-40k stable).

The pure math (`wsd_lr_at_step`) has no torch dependency and is
independently testable; `WSDScheduler` is a thin torch-optimizer-facing
wrapper around it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WSDStageConfig:
    warmup_end: int
    stable_end: int
    decay_end: int
    start_lr: float
    peak_lr: float
    end_lr: float

    def __post_init__(self):
        if not (0 <= self.warmup_end <= self.stable_end <= self.decay_end):
            raise ValueError("stage boundaries must satisfy 0 <= warmup_end <= stable_end <= decay_end")


# PLAN.md section 17 experiment presets.
LORA_STAGE_EXP001 = WSDStageConfig(2000, 10000, 35000, start_lr=5e-5, peak_lr=1e-4, end_lr=1e-5)
EMBED_STAGE_EXP001 = WSDStageConfig(2000, 10000, 35000, start_lr=1e-4, peak_lr=2e-4, end_lr=2e-5)

LORA_STAGE_EXP002 = WSDStageConfig(2000, 10000, 35000, start_lr=2.5e-5, peak_lr=5e-5, end_lr=5e-6)
EMBED_STAGE_EXP002 = WSDStageConfig(2000, 10000, 35000, start_lr=5e-5, peak_lr=1e-4, end_lr=1e-5)

LORA_STAGE_EXP003 = WSDStageConfig(2000, 10000, 35000, start_lr=1e-4, peak_lr=2e-4, end_lr=2e-5)
EMBED_STAGE_EXP003 = WSDStageConfig(2000, 10000, 35000, start_lr=1e-4, peak_lr=2e-4, end_lr=2e-5)

EXPERIMENT_STAGES = {
    "EXP001": {"lora": LORA_STAGE_EXP001, "khmer_embedding": EMBED_STAGE_EXP001},
    "EXP002": {"lora": LORA_STAGE_EXP002, "khmer_embedding": EMBED_STAGE_EXP002},
    "EXP003": {"lora": LORA_STAGE_EXP003, "khmer_embedding": EMBED_STAGE_EXP003},
}


def wsd_lr_at_step(step: int, cfg: WSDStageConfig) -> float:
    """Absolute LR at `step` (not a multiplier -- warmup starts from a nonzero
    value per PLAN.md section 16, so this cannot be expressed as a plain
    LambdaLR multiplier without carrying the base LR through separately)."""
    if step <= 0:
        return cfg.start_lr if cfg.warmup_end > 0 else cfg.peak_lr

    if step <= cfg.warmup_end:
        if cfg.warmup_end == 0:
            return cfg.peak_lr
        frac = step / cfg.warmup_end
        return cfg.start_lr + frac * (cfg.peak_lr - cfg.start_lr)

    if step <= cfg.stable_end:
        return cfg.peak_lr

    if step <= cfg.decay_end:
        span = cfg.decay_end - cfg.stable_end
        frac = (step - cfg.stable_end) / span if span > 0 else 1.0
        return cfg.peak_lr + frac * (cfg.end_lr - cfg.peak_lr)

    # Final stable stage (35k-40k and beyond, section 16).
    return cfg.end_lr


class WSDScheduler:
    """
    Torch-optimizer-facing wrapper. Expects each `optimizer.param_groups[i]`
    to carry a `"name"` key matching a key in `group_configs`
    (e.g. "lora", "khmer_embedding" -- see training/trainer.py's optimizer
    group setup, PLAN.md section 15).
    """

    def __init__(self, optimizer, group_configs: dict):
        self.optimizer = optimizer
        self.group_configs = group_configs
        for group in optimizer.param_groups:
            if group.get("name") not in group_configs:
                raise ValueError(
                    f"param_group missing/unknown 'name' key: {group.get('name')!r}; "
                    f"expected one of {list(group_configs)}"
                )
        self.last_step = 0
        self.step(0)

    def step(self, step: int = None) -> None:
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        for group in self.optimizer.param_groups:
            cfg = self.group_configs[group["name"]]
            group["lr"] = wsd_lr_at_step(step, cfg)

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_step = state_dict["last_step"]
        self.step(self.last_step)


if __name__ == "__main__":
    # Pure-python self-test -- verifies the schedule against PLAN.md section 16's
    # table exactly, with no torch dependency.
    checks = [
        (LORA_STAGE_EXP001, 0, 5e-5),
        (LORA_STAGE_EXP001, 2000, 1e-4),
        (LORA_STAGE_EXP001, 10000, 1e-4),
        (LORA_STAGE_EXP001, 35000, 1e-5),
        (LORA_STAGE_EXP001, 40000, 1e-5),
        (EMBED_STAGE_EXP001, 0, 1e-4),
        (EMBED_STAGE_EXP001, 2000, 2e-4),
        (EMBED_STAGE_EXP001, 10000, 2e-4),
        (EMBED_STAGE_EXP001, 35000, 2e-5),
        (EMBED_STAGE_EXP001, 40000, 2e-5),
    ]
    for cfg, step, expected in checks:
        actual = wsd_lr_at_step(step, cfg)
        assert abs(actual - expected) < 1e-12, f"step={step} expected={expected} actual={actual}"
    # Midpoint sanity: warmup and decay should be strictly monotonic.
    assert wsd_lr_at_step(1000, LORA_STAGE_EXP001) < wsd_lr_at_step(2000, LORA_STAGE_EXP001)
    assert wsd_lr_at_step(20000, LORA_STAGE_EXP001) < wsd_lr_at_step(10000, LORA_STAGE_EXP001)
    print("scheduler.py self-test OK")
