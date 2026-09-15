#!/usr/bin/env bash
# Watches checkpoints/metadata.json's global_step and runs
# scripts/run_inference_checkpoint.py once per new checkpoint (i.e. every
# checkpoint_interval steps, matching training/checkpoint.py's
# save_latest cadence), on cuda:1 so it never competes with training's
# GPU 0 for memory. Meant to run for the lifetime of a training run,
# started alongside scripts/train.py (nohup ... &).
set -uo pipefail

REPO_DIR="/home/helpdesk/Desktop/Khmer-TTS/Tuna-tts"
CHECKPOINT="$REPO_DIR/checkpoints/latest.pt"
METADATA="$REPO_DIR/checkpoints/metadata.json"
PROGRESS_LOG="$REPO_DIR/progress.log"
OUT_DIR="$REPO_DIR/samples/EXP001"

cd "$REPO_DIR" || exit 1

last_step=-1

while true; do
  if [ -f "$METADATA" ]; then
    step=$(python3 -c "import json; print(json.load(open('$METADATA')).get('global_step', -1))" 2>/dev/null || echo -1)
    if [ "$step" != "$last_step" ] && [ "$step" -gt 0 ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  watch_and_infer: new checkpoint at step $step, running inference on cuda:1" >> "$PROGRESS_LOG"
      .venv-smoketest/bin/python scripts/run_inference_checkpoint.py \
        --checkpoint "$CHECKPOINT" \
        --base-model-path checkpoints/openaudio-s1-mini \
        --khmer-sp-model data/khmer-sp-8k.model \
        --codec-checkpoint checkpoints/openaudio-s1-mini/codec.pth \
        --out-dir "$OUT_DIR" --device cuda:1 --step "$step" \
        >> /tmp/watch_and_infer.log 2>&1
      result_line=$(tail -1 /tmp/watch_and_infer.log)
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  watch_and_infer: $result_line" >> "$PROGRESS_LOG"
      last_step="$step"
    fi
  fi
  sleep 15
done
