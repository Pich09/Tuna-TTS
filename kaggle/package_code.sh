#!/usr/bin/env bash
# Packages just the Tuna-TTS CODE (no venv, no checkpoints, no dataset, no
# samples/logs) into a small zip you upload as a Kaggle Dataset. The
# notebooks in this folder download the base model + training data
# themselves from Hugging Face at runtime (Kaggle's network is much faster
# than uploading multi-GB files from a slow home connection) -- this zip
# only needs to carry source code + small config/data files.
#
# Usage:
#   bash kaggle/package_code.sh
# Produces: kaggle/tuna-tts-code.zip in the repo root's kaggle/ folder.
#
# Then on kaggle.com: Datasets -> New Dataset -> upload tuna-tts-code.zip
# (Kaggle unzips it automatically into /kaggle/input/<dataset-slug>/).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ZIP="$REPO_DIR/kaggle/tuna-tts-code.zip"

cd "$REPO_DIR"
rm -f "$OUT_ZIP"

zip -r "$OUT_ZIP" . \
  -x ".venv-smoketest/*" \
  -x "checkpoints/*" \
  -x "data/protos/*" \
  -x "data/manifest.json" \
  -x "data/manifest_omni.json" \
  -x "samples/*" \
  -x "*.pt" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "progress.log" \
  -x "kaggle/tuna-tts-code.zip"

echo "Wrote $OUT_ZIP ($(du -sh "$OUT_ZIP" | cut -f1))"
echo "Upload this as a Kaggle Dataset, then set CODE_DATASET_SLUG in the notebooks to its slug."
