#!/usr/bin/env bash
# Multi-GPU launch wrapper (PLAN.md section 26).
#
# Usage:
#   ./scripts/train_ddp.sh 2 configs/experiments/exp001.yaml
#
# torchrun sets RANK/LOCAL_RANK/WORLD_SIZE per process; training/distributed.py
# reads them and enables DistributedDataParallel automatically. Running
# `python scripts/train.py --config ...` directly (no torchrun) still works
# unchanged -- single GPU, no DDP wrapping (section 26: verify single-GPU
# first).
#
# Reminder (section 26): N GPUs give ~Nx throughput, not pooled memory --
# each GPU still needs enough VRAM for the model + its own per-GPU batch.

set -euo pipefail

NUM_GPUS="${1:?Usage: $0 <num_gpus> <config_path>}"
CONFIG_PATH="${2:?Usage: $0 <num_gpus> <config_path>}"

exec torchrun --standalone --nproc_per_node="${NUM_GPUS}" scripts/train.py --config "${CONFIG_PATH}"
