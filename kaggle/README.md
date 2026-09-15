# Running Tuna-TTS training on Kaggle (2x GPU)

Two notebooks, meant to be run in this order:

1. **`Tuna_TTS_Kaggle_Smoke_Test.ipynb`** -- environment setup + verification
   only (single GPU is enough). Downloads the base model/tokenizer/dataset,
   runs the codec spot-check (PLAN.md 21.5) and a real forward/backward pass
   on the actual model. Every cell must print `[PASS]` before moving on.
2. **`Tuna_TTS_Kaggle_Train.ipynb`** -- re-does the same setup + a compact
   version of the same smoke test (it's a fresh Kaggle session/container, so
   nothing from notebook 1 carries over), then launches the real EXP001 run
   under `torchrun --nproc_per_node=2` across both GPUs.

## One-time setup

```bash
bash kaggle/package_code.sh
```

This produces `kaggle/tuna-tts-code.zip` (source code + small config/data
files only -- a few hundred KB, no checkpoints or dataset). Upload it once
on kaggle.com: **Datasets -> New Dataset -> upload the zip**. Kaggle unzips
it automatically; note the dataset's slug (shown in its "usage" snippet,
e.g. `your-username/tuna-tts-code`).

The base model (`fishaudio/openaudio-s1-mini`, ~3.4GB) and the training
data (`Panhapich/khmer-tts-processed`) are deliberately **not** part of
this zip -- both notebooks download them directly from Hugging Face at
runtime, since Kaggle's network is far faster than uploading multi-GB files
from a slow connection.

## Per-notebook setup on kaggle.com

For each of the two notebooks:
1. Create a new Notebook, paste in (or upload) the `.ipynb` file.
2. **Add Input** (right sidebar) -> attach the `tuna-tts-code` dataset you
   uploaded above.
3. **Notebook Options** (right sidebar): **Internet -> On**, **Accelerator
   -> GPU T4 x2** (the training notebook asserts 2 GPUs; the smoke test
   only needs 1 but T4 x2 is fine too).
4. If `fishaudio/openaudio-s1-mini` or `Panhapich/khmer-tts-processed` ever
   require a Hugging Face token (gated/private), add it as a Kaggle Secret
   named `HF_TOKEN` (**Add-ons -> Secrets**) rather than pasting it into a
   cell -- both notebooks read it that way automatically if present.
5. Edit the `CODE_DATASET_SLUG` variable in the first code cell if your
   dataset's slug differs from `tuna-tts-code`.

## Session limits (read before launching the real run)

Kaggle notebook sessions are capped at roughly 9-12 hours, and GPU quota is
weekly (~30h/week as of this writing). EXP001's full 40,000-step run will
NOT finish in one session. The training notebook:
- Saves checkpoints every 100 steps under
  `/kaggle/working/Tuna-tts/checkpoints/` (same cadence/format as local
  training -- `training/trainer.py`'s existing resume logic applies
  unchanged).
- Requires you to click **Save Version** before the session ends, so
  `/kaggle/working` is preserved as that version's Output.
- To resume in a new session: attach the previous version's Output as an
  additional input dataset, then use the "Resuming from a previous
  session" cell (near the top of the training notebook) to copy
  `latest.pt`/`metadata.json` into `checkpoints/` before re-running -- the
  trainer will pick up from that `global_step` automatically.

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
