"""
Multi-GPU (DistributedDataParallel) support (PLAN.md section 26).

Launched via `torchrun --nproc_per_node=<N> scripts/train.py --config ...`,
which sets RANK / LOCAL_RANK / WORLD_SIZE env vars before the script runs.
This module reads those env vars -- it does not itself decide how many
processes to launch (that's torchrun's/the shell wrapper's job, see
scripts/train_ddp.sh).

Important (PLAN.md section 26): N GPUs do not pool into one larger memory
space. Each GPU holds a full model replica and processes its own batch
shard -- the benefit is throughput (~Nx), not more usable VRAM per step.
Section 25's `effective_batch = per_gpu_batch * gradient_accumulation *
num_gpus` formula already accounts for this; nothing here changes it.

The env-var parsing here has no torch dependency and is independently
testable; actual process-group init requires torch + (for the GPU
backend) NCCL, and can only be meaningfully exercised with >1 process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DistributedEnv:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def read_distributed_env() -> DistributedEnv:
    """Reads torchrun's RANK/LOCAL_RANK/WORLD_SIZE. Defaults to a single,
    non-distributed process (rank 0 of 1) when unset -- i.e. running
    `python scripts/train.py` directly still works, unchanged, without
    torchrun (PLAN.md section 26: start single-GPU, this is that path)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not (0 <= rank < world_size) or not (0 <= local_rank):
        raise ValueError(f"inconsistent distributed env: rank={rank} local_rank={local_rank} world_size={world_size}")
    return DistributedEnv(rank=rank, local_rank=local_rank, world_size=world_size)


def setup_distributed(env: DistributedEnv, backend: str = "nccl") -> None:
    """No-op when not distributed. Otherwise sets the CUDA device for this
    process and initializes the process group. Must be called before
    building the model (so `.to(device)` targets the right GPU) and before
    wrapping it in DistributedDataParallel."""
    if not env.is_distributed:
        return

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(env.local_rank)
    dist.init_process_group(backend=backend, rank=env.rank, world_size=env.world_size)


def cleanup_distributed(env: DistributedEnv) -> None:
    if not env.is_distributed:
        return
    import torch.distributed as dist

    dist.destroy_process_group()


def device_for(env: DistributedEnv) -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{env.local_rank}" if env.is_distributed else "cuda"


def wrap_model_for_distributed(model, env: DistributedEnv):
    """Wraps `model` in DistributedDataParallel when running under torchrun
    with >1 process; returns `model` unchanged otherwise. Only parameters
    with requires_grad=True are gradient-synced across ranks, so the
    frozen base weights (PLAN.md section 14) are unaffected -- only the
    Khmer embedding rows and LoRA params actually get synced."""
    if not env.is_distributed:
        return model

    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(model, device_ids=[env.local_rank], output_device=env.local_rank)


def unwrap_model(model):
    """Get back the underlying model to access custom attributes (`.config`,
    `.tokenizer`, `.embeddings`, ...) that DDP's wrapper does not proxy."""
    return model.module if hasattr(model, "module") else model


def make_sampler(dataset, env: DistributedEnv, shuffle: bool):
    """Returns a DistributedSampler when running distributed (each rank
    sees a disjoint shard of the dataset every epoch), or None otherwise
    (DataLoader's own `shuffle=` argument is used instead)."""
    if not env.is_distributed:
        return None

    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(dataset, num_replicas=env.world_size, rank=env.rank, shuffle=shuffle)


def reduce_mean(value: float, env: DistributedEnv) -> float:
    """Averages a scalar (e.g. validation loss) across all ranks, so every
    process reports the same number instead of just its own shard's."""
    if not env.is_distributed:
        return value

    import torch
    import torch.distributed as dist

    tensor = torch.tensor([value], dtype=torch.float64, device=f"cuda:{env.local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor.item()) / env.world_size


if __name__ == "__main__":
    # Pure env-var-parsing self-test, no torch/process-group needed.
    import os as _os

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        _os.environ.pop(key, None)
    env = read_distributed_env()
    assert env == DistributedEnv(rank=0, local_rank=0, world_size=1)
    assert not env.is_distributed
    assert env.is_main_process

    _os.environ.update({"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"})
    env2 = read_distributed_env()
    assert env2 == DistributedEnv(rank=1, local_rank=1, world_size=2)
    assert env2.is_distributed
    assert not env2.is_main_process

    _os.environ.update({"RANK": "5", "LOCAL_RANK": "0", "WORLD_SIZE": "2"})
    try:
        read_distributed_env()
        raise AssertionError("expected ValueError for rank >= world_size")
    except ValueError:
        pass

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        _os.environ.pop(key, None)

    print("distributed.py self-test OK")
