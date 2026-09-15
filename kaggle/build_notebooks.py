"""
Generates the two Kaggle notebooks (.ipynb JSON) in this folder. Run once
locally with `python kaggle/build_notebooks.py` after editing this file --
kept as a .py source instead of hand-editing raw notebook JSON so the cell
content stays reviewable/diffable in normal code review.
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).parent


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


CODE_DATASET_SLUG_CELL = '''\
# EDIT THIS if you fork/rename the repo.
GITHUB_REPO_URL = "https://github.com/Pich09/Tuna-TTS.git"

REPO_DIR = "/kaggle/working/Tuna-tts"
'''

SETUP_CELLS_MD = md('''\
## 1. Environment check

Confirms the accelerator is on and (for the training notebook) that both
GPUs are visible. If this prints 0 GPUs, go to
**Notebook settings (right sidebar) -> Accelerator -> GPU T4 x2** and
re-run.
''')

ENV_CHECK_CODE = '''\
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
'''

COPY_CODE_CELL_MD = md('''\
## 2. Clone the project code

Cloned straight from GitHub into `/kaggle/working/Tuna-tts` (a writable
directory -- training writes checkpoints/logs next to the code). No
dataset upload needed for code changes; just push to GitHub and re-run
this cell to pick them up.

If the repo is private, add a Kaggle Secret named `GITHUB_TOKEN` (a
fine-grained PAT with read-only access to just this repo) -- the cell
below picks it up automatically and never prints or hardcodes it. Public
repo: leave the secret unset, nothing else to do.
''')

COPY_CODE_CELL = '''\
import shutil, subprocess, os

try:
    from kaggle_secrets import UserSecretsClient
    github_token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    github_token = None  # fine for a public repo

clone_url = GITHUB_REPO_URL
if github_token:
    clone_url = GITHUB_REPO_URL.replace("https://", f"https://{github_token}@")

if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(["git", "clone", "--depth", "1", clone_url, REPO_DIR], check=True)
print(f"Cloned {GITHUB_REPO_URL} -> {REPO_DIR}")
'''

INSTALL_CELL_MD = md('''\
## 3. Install dependencies

Kaggle ships a recent torch+CUDA already (kept as-is -- reinstalling torch
here would be slow and is unnecessary). Everything else installs from
`requirements.txt`, which pins fish-speech to the exact commit
(`d3df505`) this codebase's tokenizer/model code was verified against --
main HEAD is NOT compatible with openaudio-s1-mini (see requirements.txt's
comments). Requires this notebook's **Internet** toggle to be ON.
''')

INSTALL_CELL = '''\
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt"], check=True)
# protobuf resolver conflict (descript-audiotools wants <3.20, fish-speech's
# generated _pb2.py needs >=3.20) -- upgrade separately after the main install,
# same fix used in local development (see requirements.txt).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "protobuf>=3.20,<6"], check=True)
print("Dependencies installed.")
'''

DOWNLOAD_CELL_MD = md('''\
## 4. Download the base model, Khmer tokenizer, and training data

Downloaded here (on Kaggle's fast network) rather than uploaded from a slow
home connection. Add an `HF_TOKEN` secret via the notebook's **Add-ons ->
Secrets** menu -- needed if `fishaudio/openaudio-s1-mini` or
`Panhapich/khmer-tts-processed` are gated/private, AND (training notebook
only) for uploading checkpoints to `Panhapich/Tuna-TTS` -- that token needs
**write** access to that repo specifically. Never paste a token directly
into a cell; this reads it from the secret and also exports it as the
`HF_TOKEN` environment variable so the training subprocess launched later
inherits it the same way.
''')

DOWNLOAD_CELL = '''\
import os
from huggingface_hub import snapshot_download, hf_hub_download

try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = None  # fine if the repos are public and no secret is set

if hf_token:
    os.environ["HF_TOKEN"] = hf_token  # inherited by the torchrun subprocess later

ckpt_dir = f"{REPO_DIR}/checkpoints/openaudio-s1-mini"
snapshot_download(repo_id="fishaudio/openaudio-s1-mini", local_dir=ckpt_dir, token=hf_token)
print(f"Base model -> {ckpt_dir}")

hf_hub_download(repo_id="Panhapich/khmer-sp-8k", filename="khmer-sp-8k.model",
                 local_dir=f"{REPO_DIR}/data", token=hf_token)
print("Khmer tokenizer -> data/khmer-sp-8k.model")

snapshot_download(repo_id="Panhapich/khmer-tts-processed", repo_type="dataset",
                   local_dir=f"{REPO_DIR}/data/protos", token=hf_token)
print(f"Training data -> {REPO_DIR}/data/protos")

# fish_speech.models.dac.inference does pyrootutils.setup_root(indicator=".project-root")
# at import time, which fails when fish-speech is pip-installed rather than a
# repo checkout -- same fix used in local development.
import fish_speech.models.dac as dac_pkg
marker_path = os.path.join(os.path.dirname(dac_pkg.__file__), ".project-root")
open(marker_path, "a").close()
print(f"Touched {marker_path}")
'''

FIND_PROTOS_CELL_MD = md('''\
## 5. Locate the actual train/validation .protos subdirectories

`Panhapich/khmer-tts-processed`'s exact folder layout inside the downloaded
snapshot can vary; this finds the train/validation dirs so the config
doesn't need hand-editing.
''')

FIND_PROTOS_CELL = '''\
import glob, os

protos_root = f"{REPO_DIR}/data/protos"
train_dir = next((d for d in glob.glob(f"{protos_root}/**/train", recursive=True) if os.path.isdir(d)), None) \\
    or next((d for d in glob.glob(f"{protos_root}/**/*train*", recursive=True) if os.path.isdir(d)), protos_root)
val_dir = next((d for d in glob.glob(f"{protos_root}/**/val*", recursive=True) if os.path.isdir(d)), protos_root)

print(f"proto_train_dir -> {train_dir}")
print(f"proto_val_dir   -> {val_dir}")

# Patch the experiment config in place so scripts/train.py picks these paths
# up as-is (avoids hand-editing YAML for a path that may differ per run).
import re
for cfg_name in ["exp001.yaml", "exp001_kaggle.yaml"]:
    cfg_path = f"{REPO_DIR}/configs/experiments/{cfg_name}"
    text = open(cfg_path).read()
    text = re.sub(r"proto_train_dir: .*", f"proto_train_dir: {os.path.relpath(train_dir, REPO_DIR)}", text)
    text = re.sub(r"proto_val_dir: .*", f"proto_val_dir: {os.path.relpath(val_dir, REPO_DIR)}", text)
    open(cfg_path, "w").write(text)
print("Patched proto_train_dir / proto_val_dir in exp001*.yaml")
'''

CODEC_CHECK_MD = md('''\
## 6. Codec spot-check (PLAN.md section 21.5)

Decodes a handful of the dataset's precomputed semantic codes straight
through the S1-mini codec, with no model involved -- confirms the
downloaded checkpoint/data are actually compatible with each other before
spending any GPU time on training.
''')

CODEC_CHECK_CELL = '''\
import subprocess, sys

result = subprocess.run(
    [sys.executable, "scripts/codec_spotcheck.py",
     "--proto-dir", "data/protos",
     "--codec-checkpoint", "checkpoints/openaudio-s1-mini/codec.pth",
     "--num-samples", "5",
     "--out-dir", "/kaggle/working/codec_spotcheck"],
    cwd=REPO_DIR, capture_output=True, text=True,
)
print(result.stdout)
print(result.stderr)
assert result.returncode == 0, "Codec spot-check FAILED -- do not proceed to training, see output above."
print("[PASS] codec spot-check")
'''

FORWARD_BACKWARD_MD = md('''\
## 7. Forward/backward smoke test (PLAN.md section 43)

Builds the real model (S1-mini + Khmer embedding extension + LoRA), runs a
handful of real training batches through forward + backward + optimizer
step on a single GPU, and checks the loss and every gradient are finite.
This is the same check as `scripts/test_model_protos.py`, run inline here
so its assertion stops the notebook before touching the expensive
multi-GPU training cell below.
''')

FORWARD_BACKWARD_CELL = '''\
import subprocess, sys

result = subprocess.run(
    [sys.executable, "scripts/test_model_protos.py",
     "--base-model-path", "checkpoints/openaudio-s1-mini",
     "--khmer-sp-model", "data/khmer-sp-8k.model",
     "--proto-train-dir", "data/protos/train",
     "--num-batches", "5", "--batch-size", "1"],
    cwd=REPO_DIR, capture_output=True, text=True,
)
print(result.stdout)
print(result.stderr)
assert result.returncode == 0, "Forward/backward smoke test FAILED -- see output above. Do not proceed to training."
print("\\n[PASS] forward/backward smoke test -- safe to proceed to the training notebook.")
'''

smoke_test_nb = nb([
    md('''\
# Tuna-TTS -- Kaggle setup + smoke test

Run this notebook FIRST, top to bottom, before the training notebook. It:
1. Verifies the GPU(s) and installs dependencies.
2. Downloads the base model, Khmer tokenizer, and dataset from Hugging Face.
3. Runs the PLAN.md section 21.5 codec spot-check and a real
   forward+backward smoke test on the actual model/data/checkpoint path.

**Before running:** turn **Internet** ON in notebook settings (needed to
clone the repo and download from Hugging Face). A single T4 GPU is enough
for this notebook (the 2-GPU run is in the training notebook).

If every cell below prints PASS, go run `Tuna_TTS_Kaggle_Train.ipynb`.
'''),
    md("## 0. Config"),
    code(CODE_DATASET_SLUG_CELL),
    SETUP_CELLS_MD, code(ENV_CHECK_CODE),
    COPY_CODE_CELL_MD, code(COPY_CODE_CELL),
    INSTALL_CELL_MD, code(INSTALL_CELL),
    DOWNLOAD_CELL_MD, code(DOWNLOAD_CELL),
    FIND_PROTOS_CELL_MD, code(FIND_PROTOS_CELL),
    CODEC_CHECK_MD, code(CODEC_CHECK_CELL),
    FORWARD_BACKWARD_MD, code(FORWARD_BACKWARD_CELL),
    md("## All checks passed. Proceed to `Tuna_TTS_Kaggle_Train.ipynb`."),
])

train_nb = nb([
    md('''\
# Tuna-TTS -- Kaggle training (2x GPU via DDP)

Runs `scripts/train.py` under `torchrun --nproc_per_node=2` so both of
Kaggle's GPUs are used (training/distributed.py already implements DDP --
see PLAN.md section 26). Each GPU still needs the same per-GPU memory
budget as a single-GPU run; 2 GPUs buys ~2x throughput, not pooled VRAM.

**Run the smoke-test notebook in a separate session first.** This
notebook also re-runs a compact version of that smoke test itself before
launching the expensive multi-GPU job, so a stale/broken environment stops
here instead of burning GPU-hours -- but that inline check is not a
substitute for actually reading the smoke-test notebook's full output at
least once.

**Before running:** set the accelerator to **GPU T4 x2** and turn
**Internet** ON. To have checkpoints uploaded to `Panhapich/Tuna-TTS`
automatically (recommended -- see step 8), add an `HF_TOKEN` secret with
write access to that repo via **Add-ons -> Secrets**.

**Session limits:** Kaggle notebook sessions are capped (~9-12h) and GPU
quota is weekly (~30h). This will NOT finish EXP001's full 40,000 steps in
one session. Checkpoints save under `/kaggle/working/Tuna-tts/checkpoints/`
every 100 steps; when the session ends, commit the notebook (Save Version)
so `/kaggle/working` is preserved as that version's Output, then in the
next session's setup, copy `latest.pt`/`metadata.json` from the previous
version's output into `checkpoints/` before re-running this notebook --
`training/trainer.py`'s existing resume logic picks up from there
automatically (same mechanism verified locally after a host reboot).
'''),
    md("## 0. Config"),
    code(CODE_DATASET_SLUG_CELL),
    SETUP_CELLS_MD, code(ENV_CHECK_CODE + '''
import torch
n = torch.cuda.device_count()
print(f"torch sees {n} GPU(s)")
assert n >= 2, "Expected 2 GPUs -- check Notebook settings -> Accelerator -> GPU T4 x2"
'''),
    COPY_CODE_CELL_MD, code(COPY_CODE_CELL),
    INSTALL_CELL_MD, code(INSTALL_CELL),
    DOWNLOAD_CELL_MD, code(DOWNLOAD_CELL),
    FIND_PROTOS_CELL_MD, code(FIND_PROTOS_CELL),
    md('''\
## 6. Resuming from a previous session (skip if this is a fresh run)

Two ways to get a previous checkpoint back before launching training --
pick whichever is available:

- **From the Hub** (works even if you forgot to Save Version last time --
  checkpoints upload to `Panhapich/Tuna-TTS` automatically per step 8):
  uncomment the `hf_hub_download` cell below.
- **From a Kaggle output dataset**: attach the previous session's Output
  as an additional input dataset, then uncomment the `shutil.copy` cell.
'''),
    code('''\
# from huggingface_hub import hf_hub_download
# import os
# os.makedirs(f"{REPO_DIR}/checkpoints", exist_ok=True)
# for fname in ["latest.pt", "metadata.json"]:
#     hf_hub_download(repo_id="Panhapich/Tuna-TTS", filename=fname,
#                      local_dir=f"{REPO_DIR}/checkpoints", token=hf_token)
# print("Downloaded previous checkpoint from the Hub -- training will resume from its global_step.")
'''),
    code('''\
# import shutil, os
# PREV_OUTPUT_DIR = "/kaggle/input/<previous-session-output-slug>/Tuna-tts/checkpoints"
# os.makedirs(f"{REPO_DIR}/checkpoints", exist_ok=True)
# for fname in ["latest.pt", "metadata.json"]:
#     shutil.copy(f"{PREV_OUTPUT_DIR}/{fname}", f"{REPO_DIR}/checkpoints/{fname}")
# print("Copied previous checkpoint -- training will resume from its global_step.")
'''),
    md('''\
## 7. Mandatory smoke test -- do not skip

Same checks as the setup notebook (codec spot-check + forward/backward
pass), run again here because this is a separate Kaggle session/container
with its own fresh environment. Both cells must print PASS before the
training cell below is allowed to run.
'''),
    code(CODEC_CHECK_CELL),
    code(FORWARD_BACKWARD_CELL),
    md('''\
## 8. Launch training on both GPUs

Runs in the background (so this cell returns immediately) via
`torchrun --standalone --nproc_per_node=2`, matching
`scripts/train_ddp.sh`. Output goes to `/kaggle/working/train.log`.

`configs/experiments/exp001_kaggle.yaml` has `checkpoint.upload_to_hub:
true`, so every checkpoint saved locally (every 100 steps) is also pushed
to `Panhapich/Tuna-TTS` on the Hub in a background thread inside the
training process -- this survives even if you forget to Save Version
before a Kaggle session ends. Uploads run one at a time; if a checkpoint's
upload is still in flight when the next one is due, that one is skipped
(logged as `[hub_upload] skipped ...`) rather than queued -- the next
checkpoint supersedes it anyway. Needs the `HF_TOKEN` secret set up in
step 4 with **write** access to `Panhapich/Tuna-TTS`; without it, uploads
fail loudly in `train.log` but training itself keeps running unaffected.
'''),
    code('''\
import subprocess, os

log_path = "/kaggle/working/train.log"
log_file = open(log_path, "w")
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

proc = subprocess.Popen(
    ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/train.py",
     "--config", "configs/experiments/exp001_kaggle.yaml"],
    cwd=REPO_DIR, stdout=log_file, stderr=subprocess.STDOUT, env=env,
)
print(f"Launched training, pid={proc.pid}. Logging to {log_path}")
'''),
    md('''\
## 9. Monitor progress

Re-run this cell any time (or leave it running) to see the latest lines.
Interrupting this cell does NOT stop training -- it keeps running in the
background per the previous cell; use the next cell to actually stop it.
'''),
    code('''\
import time

try:
    while True:
        with open(log_path) as f:
            lines = f.readlines()
        print("".join(lines[-20:]))
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopped watching (training keeps running in the background).")
'''),
    md("## 10. Stop training (only run this cell when you actually want to stop)"),
    code('''\
proc.terminate()
proc.wait(timeout=30)
print(f"Training process {proc.pid} stopped. Checkpoints remain under {REPO_DIR}/checkpoints/.")
'''),
    md('''\
## 11. Before ending the session

Click **Save Version** (top right) so `/kaggle/working` (including
`Tuna-tts/checkpoints/latest.pt` and `metadata.json`) is preserved as this
version's Output -- otherwise everything is lost when the session ends.
'''),
])

(OUT_DIR / "Tuna_TTS_Kaggle_Smoke_Test.ipynb").write_text(json.dumps(smoke_test_nb, indent=1))
(OUT_DIR / "Tuna_TTS_Kaggle_Train.ipynb").write_text(json.dumps(train_nb, indent=1))
print("Wrote Tuna_TTS_Kaggle_Smoke_Test.ipynb and Tuna_TTS_Kaggle_Train.ipynb")
