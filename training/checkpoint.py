"""
Checkpoint manager (PLAN.md sections 29-39): latest.pt / best.pt /
metadata.json, atomic writes, and resume-compatibility verification.

The compatibility-check logic (`check_resume_compatibility`,
`ARCHITECTURE_VERSION`) is pure Python/JSON and has no torch dependency,
so it is independently testable. Save/load of the actual checkpoint
tensors requires torch.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ARCHITECTURE_VERSION = "tuna-v1"  # PLAN.md section 39

# PLAN.md section 38: these must match exactly before resuming, or training stops.
CRITICAL_METADATA_KEYS = [
    "architecture_version",
    "base_model",
    "tokenizer",
    "dataset",
    "vocab_size",
    "hidden_dim",
]


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint's metadata does not match the current run's
    architecture -- PLAN.md section 38: stop instead of silently loading."""


def check_resume_compatibility(saved_metadata: dict, current_metadata: dict) -> None:
    mismatches = []
    for key in CRITICAL_METADATA_KEYS:
        saved_val = saved_metadata.get(key)
        current_val = current_metadata.get(key)
        if saved_val != current_val:
            mismatches.append((key, saved_val, current_val))
    if mismatches:
        details = "; ".join(f"{k}: saved={s!r} current={c!r}" for k, s, c in mismatches)
        raise IncompatibleCheckpointError(f"Checkpoint incompatible with current run: {details}")


@dataclass
class CheckpointPaths:
    directory: Path
    latest_name: str = "latest.pt"
    best_name: str = "best.pt"
    metadata_name: str = "metadata.json"

    @property
    def latest(self) -> Path:
        return self.directory / self.latest_name

    @property
    def best(self) -> Path:
        return self.directory / self.best_name

    @property
    def metadata(self) -> Path:
        return self.directory / self.metadata_name


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)  # atomic on the same filesystem, section 34


def _atomic_torch_save(obj, path: Path, required_keys: tuple = ("model_state_dict", "global_step")) -> None:
    import torch

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    _verify_torch_checkpoint(tmp_path, required_keys=required_keys)  # section 34: verify before replacing
    os.replace(tmp_path, path)


def _verify_torch_checkpoint(path: Path, required_keys: tuple = ("model_state_dict", "global_step")) -> dict:
    """
    `required_keys` defaults to latest.pt's full-resumable-state shape, but
    is NOT universal: best.pt (see maybe_save_best) is deliberately a
    lighter-weight {model_state_dict, val_loss} snapshot for evaluation/
    export, not resuming -- it never carries global_step, so verifying it
    against the resume-shaped default would always fail. Pass the actual
    keys the caller's checkpoint dict is meant to have instead.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu")
    for key in required_keys:
        if key not in ckpt:
            raise ValueError(f"checkpoint at {path} missing required key {key!r}")
    return ckpt


@dataclass
class TrainingState:
    """What `latest.pt` must contain to resume without losing state (section 30)."""

    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    global_step: int
    epoch: int
    best_val_loss: float
    scaler_state_dict: Optional[dict] = None
    rng_state: Optional[bytes] = None
    cuda_rng_state: Optional[bytes] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }
        if self.scaler_state_dict is not None:
            d["scaler_state_dict"] = self.scaler_state_dict
        if self.rng_state is not None:
            d["rng_state"] = self.rng_state
        if self.cuda_rng_state is not None:
            d["cuda_rng_state"] = self.cuda_rng_state
        d.update(self.extra)
        return d


class CheckpointManager:
    def __init__(
        self,
        directory: str,
        checkpoint_interval: int = 200,
        validation_interval: int = 1000,
        architecture_info: Optional[dict] = None,
    ):
        self.paths = CheckpointPaths(Path(directory))
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.architecture_info = architecture_info or {}
        self.best_val_loss = float("inf")

    def should_checkpoint(self, step: int) -> bool:
        return step > 0 and step % self.checkpoint_interval == 0

    def should_validate(self, step: int) -> bool:
        return step > 0 and step % self.validation_interval == 0

    def save_latest(self, state: TrainingState) -> None:
        """PLAN.md section 33: save/verify/replace latest.pt every N steps."""
        _atomic_torch_save(state.to_dict(), self.paths.latest)

    def maybe_save_best(self, val_loss: float, model_state_dict: dict) -> bool:
        """PLAN.md section 31/33: overwrite best.pt only if val_loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            _atomic_torch_save(
                {"model_state_dict": model_state_dict, "val_loss": val_loss},
                self.paths.best,
                required_keys=("model_state_dict", "val_loss"),
            )
            return True
        return False

    def write_metadata(self, global_step: int, epoch: int, experiment_id: str, extra: Optional[dict] = None) -> None:
        """PLAN.md section 32."""
        metadata = {
            "project": "Tuna-TTS",
            "architecture_version": ARCHITECTURE_VERSION,
            "global_step": global_step,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "latest_checkpoint": self.paths.latest_name,
            "best_checkpoint": self.paths.best_name,
            "checkpoint_interval": self.checkpoint_interval,
            "validation_interval": self.validation_interval,
            "experiment_id": experiment_id,
            "last_checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self.architecture_info,
        }
        if extra:
            metadata.update(extra)
        _atomic_write_json(self.paths.metadata, metadata)

    def load_for_resume(self, current_metadata: dict) -> Optional[dict]:
        """
        PLAN.md section 35: download/load latest.pt + metadata.json, verify
        compatibility (section 38), and return the training state dict, or
        None if there is nothing to resume from.
        """
        if not self.paths.latest.exists() or not self.paths.metadata.exists():
            return None

        with open(self.paths.metadata, "r", encoding="utf-8") as f:
            saved_metadata = json.load(f)

        check_resume_compatibility(saved_metadata, current_metadata)

        import torch

        state = torch.load(self.paths.latest, map_location="cpu")
        self.best_val_loss = saved_metadata.get("best_val_loss", float("inf"))
        return state


if __name__ == "__main__":
    # Pure-python self-test of the compatibility check (no torch needed).
    current = {
        "architecture_version": "tuna-v1",
        "base_model": "fishaudio/openaudio-s1-mini",
        "tokenizer": "Panhapich/khmer-sp-8k",
        "dataset": "Panhapich/khmer-tts-processed",
        "vocab_size": 40000,
        "hidden_dim": 1024,
    }
    check_resume_compatibility(current, current)  # should not raise

    stale = dict(current, architecture_version="tuna-v0")
    try:
        check_resume_compatibility(stale, current)
        raise AssertionError("expected IncompatibleCheckpointError")
    except IncompatibleCheckpointError:
        pass

    print("checkpoint.py self-test OK")
