# Running Tuna-TTS training on Kaggle (2x GPU)

Two notebooks, meant to be run in this order:

1. **`Tuna_TTS_Kaggle_Smoke_Test.ipynb`** -- environment setup + verification
   only (single GPU is enough). Clones the repo, downloads the base
   model/tokenizer/dataset, runs the codec spot-check (PLAN.md 21.5) and a
   real forward/backward pass on the actual model. Every cell must print
   `[PASS]` before moving on.
2. **`Tuna_TTS_Kaggle_Train.ipynb`** -- re-does the same setup + a compact
   version of the same smoke test (it's a fresh Kaggle session/container,
   so nothing from notebook 1 carries over), then launches the real EXP001
   run under `torchrun --nproc_per_node=2` across both GPUs, with every
   checkpoint uploaded to `Panhapich/Tuna-TTS` on the Hub as training runs.

Both notebooks clone straight from `https://github.com/Pich09/Tuna-TTS.git`
-- no dataset upload needed for code. Push a change to GitHub and re-run
the clone cell to pick it up.

## Per-notebook setup on kaggle.com

For each of the two notebooks:
1. Create a new Notebook, paste in (or upload) the `.ipynb` file.
2. **Notebook Options** (right sidebar): **Internet -> On**, **Accelerator
   -> GPU T4 x2** (the training notebook asserts 2 GPUs; the smoke test
   only needs 1 but T4 x2 is fine too).
3. If the GitHub repo is private, add a Kaggle Secret named `GITHUB_TOKEN`
   (a fine-grained PAT with read-only access to just this repo) via
   **Add-ons -> Secrets**. Public repo: nothing to do here.
4. If `fishaudio/openaudio-s1-mini` or `Panhapich/khmer-tts-processed` ever
   require a Hugging Face token (gated/private), or to enable checkpoint
   uploads (see below), add an `HF_TOKEN` secret the same way. Neither
   notebook ever asks you to paste a token into a cell.

## Checkpoints upload to the Hub automatically

`configs/experiments/exp001_kaggle.yaml` sets `checkpoint.upload_to_hub:
true`, so the training notebook pushes every checkpoint (every 100 steps,
same cadence as local disk saves) to `Panhapich/Tuna-TTS` on Hugging Face,
in a background thread inside the training process
(`training/hub_upload.py`) -- a slow/failed upload never blocks or kills
training. This needs the `HF_TOKEN` secret above to have **write** access
to `Panhapich/Tuna-TTS` specifically; without it, uploads fail loudly in
`train.log` but training itself keeps running unaffected (checkpoints are
always safe on local disk regardless).

This is the same `checkpoint.hf_repo` field `configs/tuna_v1.yaml` already
declares for every experiment -- only `upload_to_hub` is the opt-in
switch, and it's off by default (including in the local `exp001.yaml`), so
local runs never start uploading unless you explicitly turn it on the same
way.

## Session limits

Kaggle notebook sessions are capped at roughly 9-12 hours, and GPU quota is
weekly (~30h/week as of this writing). EXP001's full 40,000-step run will
NOT finish in one session. Two ways to resume in a new session (see the
training notebook's "Resuming from a previous session" cell):
- **From the Hub** -- works even if you forgot to click Save Version last
  time, since checkpoints are already on `Panhapich/Tuna-TTS`.
- **From a Kaggle output dataset** -- click **Save Version** before the
  session ends, then attach that version's Output as an input dataset next
  time.

## Why 2 GPUs needs a different config

`configs/experiments/exp001_kaggle.yaml` is `exp001.yaml` with
`gradient_accumulation` halved (16 -> 8), so the effective batch size
(`per_gpu_batch_size * gradient_accumulation * num_gpus`) stays at 16 across
1 GPU locally vs. 2 GPUs on Kaggle -- keeping the learning-rate schedule
tuned for EXP001 valid unchanged. See that file's own comments for the
full explanation.

Two GPUs give roughly 2x training throughput; they do **not** pool VRAM --
each T4 still needs the same per-GPU memory budget `exp001.yaml` was
already tuned to fit (16GB, `per_gpu_batch_size=1`, `proto_max_length=150`).
