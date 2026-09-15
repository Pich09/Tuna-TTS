# Tuna-TTS

Khmer adaptation of `fishaudio/openaudio-s1-mini` via extended vocabulary +
LoRA. Full design in [PLAN.md](PLAN.md); this README maps the plan's
phases onto the code in this repo and what's been verified so far.

## Status

Everything below has been implemented and self-tested **without** the
gated S1-mini checkpoint or a GPU (this environment has neither). Model-
dependent pieces are wired against the *real* fish-speech source (cloned
and inspected directly, see PLAN.md section 61) rather than guessed, but
have not been run against real weights. Phases 3 onward require an actual
Kaggle/GPU session with the checkpoint downloaded.

| Phase (PLAN.md) | Code | Status |
|---|---|---|
| Text normalization (21.1) | `data/text_normalize.py` | ✅ implemented + tested (`scripts/test_normalize.py`) |
| Vocabulary extension math (5) | `tokenizer/vocabulary.py` | ✅ implemented + tested |
| Tokenizer adapter (6) | `tokenizer/tuna_tokenizer.py` | ✅ implemented + tested with fakes (`scripts/test_tokenizer.py`) |
| 1. Inspect S1-mini architecture | `scripts/inspect_model.py` | ✅ script ready; needs real checkpoint to run |
| 7-9. Embedding extension + weight tying | `model/embedding_extension.py` | ✅ implemented against verified real architecture (section 61.1) |
| 11-13. LoRA | `model/lora.py` | ✅ thin wrapper around fish-speech's own `setup_lora` (section 61.4) |
| Model assembly (55) | `model/model_builder.py` | ✅ implemented; `verify_architecture()` is Phase 1's gate |
| 15-16. Optimizer + WSD schedule | `training/scheduler.py` | ✅ implemented + tested against the plan's exact LR table |
| 20-23. Dataset pipeline + splits | `data/dataset.py`, `scripts/make_splits.py` | ✅ implemented; delegates label packing to fish-speech's real `ContentSequence` (section 61.5) |
| 21.2. Semantic target cache | `data/semantic_cache.py` | ✅ index/bookkeeping implemented + tested; actual codec precompute script still needed |
| 22. Data validation | `data/validation.py` | ✅ implemented + tested |
| 29-39. Checkpointing | `training/checkpoint.py` | ✅ implemented + tested (compatibility check; atomic save needs torch) |
| Training loop | `training/trainer.py` | ✅ implemented; forward/loss matches fish-speech's real training step |
| Phase 3. Forward pass test | `scripts/test_model.py` | ⏳ ready to run once checkpoint + real manifest exist |
| 47-48. Evaluation | `evaluation/evaluate.py`, `generate_samples.py` | ✅ scoring harness implemented + tested; audio generation needs real model |

## Quick start (no GPU/checkpoint required)

Run every self-test that doesn't need torch or the gated checkpoint:

```bash
python3 data/text_normalize.py
python3 scripts/test_normalize.py
python3 tokenizer/vocabulary.py
python3 scripts/test_tokenizer.py
python3 training/scheduler.py
python3 training/checkpoint.py
python3 data/semantic_cache.py
python3 data/collator.py
python3 data/validation.py
python3 evaluation/evaluate.py --help
```

## Full pipeline (requires GPU + the gated checkpoint)

1. `pip install -r requirements.txt`
2. Accept the license and download `fishaudio/openaudio-s1-mini` locally.
3. Download `Panhapich/khmer-sp-8k` (SentencePiece model) and
   `Panhapich/khmer-tts-processed` (dataset).
4. `python scripts/inspect_model.py --model-path <path>` — confirm the
   architecture assumptions in PLAN.md section 61 still hold for the
   exact checkpoint you downloaded (config can drift between releases).
5. `python scripts/make_splits.py --manifest raw_manifest.json --out data/manifest.json`
6. Precompute the semantic cache (section 21.2) — script not yet written;
   wire `fish_speech`'s audio codec against `data/semantic_cache.py`'s index.
7. `python scripts/test_tokenizer.py --real --base-tokenizer-path <path> --khmer-sp-model <path>`
8. `python scripts/test_model.py --base-model-path <path> --manifest data/manifest.json --semantic-cache-dir data/semantic_cache --codec-revision v1`
9. Pilot run (500-3,000 steps, PLAN.md Phase 4), then
   `python scripts/train.py --config configs/experiments/exp001.yaml`

## What's genuinely still open

- The semantic-cache **precompute** script (running the real audio codec
  over raw audio) isn't written — it needs the actual checkpoint to test
  against, unlike everything else here.
- `scripts/inspect_model.py`'s output should be re-checked against
  whatever exact S1-mini revision you pull; section 61's findings come
  from fish-speech's public model code, not the (gated) weights repo.
