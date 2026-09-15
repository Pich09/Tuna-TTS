# Tuna-TTS — Implementation Plan

## 1. Project Overview

### Objective

Adapt **Fish Audio S1-mini** (`fishaudio/openaudio-s1-mini`) for Khmer text-to-speech using:

* A custom Khmer tokenizer: `Panhapich/khmer-sp-8k`
* A Khmer TTS dataset: `Panhapich/khmer-tts-processed`
* LoRA-based adaptation of the pretrained S1-mini model
* An extended text vocabulary for Khmer
* Multi-session training on Kaggle
* Hugging Face as persistent checkpoint storage

The goal is to preserve the pretrained capabilities of S1-mini while teaching the model to process Khmer text efficiently without retraining the entire model from scratch.

---

# 2. High-Level Architecture

The adapted model will consist of:

```text
                    Tuna-TTS
                       │
        ┌──────────────┴──────────────┐
        │                             │
 Original S1-mini              Khmer Adaptation
        │                             │
        │                    ┌────────┴────────┐
        │                    │                 │
 Original Text Tokens   Khmer Embeddings     LoRA
        │                    │                 │
        └────────────┬───────┴─────────────────┘
                     │
              S1 Text/Semantic Model
                     │
              Semantic Tokens
                     │
              Existing Codec
                     │
                  Audio
```

The important distinction is:

> Khmer tokenizer tokens are **text tokens**, not semantic/audio tokens.

Therefore, the existing S1 semantic-token vocabulary and codec must remain unchanged.

---

# 3. Base Model

Use:

```text
fishaudio/openaudio-s1-mini
```

The original S1-mini model should remain frozen initially.

Do not modify the original checkpoint directly.

Instead:

```text
Original S1-mini
       │
       ├── Original text vocabulary
       ├── Original model weights
       └── Original semantic/audio vocabulary
                    │
                    ▼
             Tuna-TTS adaptation
```

---

# 4. Custom Khmer Tokenizer

Use:

```text
Panhapich/khmer-sp-8k
```

The tokenizer is based on SentencePiece Unigram with:

```text
Vocabulary size = 8,000
```

The tokenizer includes Khmer and English/code-switching support.

Special tokens include:

```text
<PAD>  = 0
<UNK>  = 1
<BOS>  = 2
<EOS>  = 3
<MASK> = 4
```

## Important

Do not simply load:

```python
SentencePieceProcessor(...)
```

and assume the resulting IDs are the complete tokenizer interface.

The repository provides its own Khmer segmentation/tokenization pipeline. Use the provided tokenizer implementation so that Khmer segmentation and special-token handling remain consistent.

---

# 5. Vocabulary Integration

## 5.1 Do NOT replace the S1 vocabulary

The original S1 text vocabulary must remain unchanged.

Suppose:

```python
S1_VOCAB_SIZE = V
KHMER_VOCAB_SIZE = 8000
```

Create an extended vocabulary:

```text
0 ─────────────────── V-1
│
│ Original S1 text tokens
│
V ─────────────────── V+7999
│
│ Khmer tokenizer tokens
│
V+8000
```

Map Khmer token IDs using:

```python
extended_id = S1_VOCAB_SIZE + khmer_token_id
```

For example:

```text
Khmer ID 0  → S1_VOCAB_SIZE + 0
Khmer ID 1  → S1_VOCAB_SIZE + 1
Khmer ID 2  → S1_VOCAB_SIZE + 2
...
Khmer ID 7999 → S1_VOCAB_SIZE + 7999
```

---

# 6. Special Token Handling

Do not blindly offset all tokenizer special tokens.

The S1 model already has its own special/control tokens.

Therefore, establish an explicit mapping layer:

```text
Input text
   │
   ▼
Khmer tokenizer
   │
   ▼
Khmer token IDs
   │
   ▼
ID mapping
   │
   ├── S1 special/control tokens → existing S1 IDs
   │
   └── Khmer vocabulary IDs → S1_VOCAB_SIZE + Khmer ID
```

Create a dedicated tokenizer adapter:

```python
TunaTokenizer
```

Responsibilities:

1. Detect/segment Khmer text.
2. Tokenize Khmer using `Panhapich/khmer-sp-8k`.
3. Preserve supported English/S1 tokens where appropriate.
4. Convert Khmer IDs into extended S1 IDs.
5. Handle BOS/EOS/PAD/MASK/control tokens correctly.
6. Produce the exact input format expected by S1-mini.

---

# 7. Extend the Text Embedding

The original S1 text embedding has approximately:

```text
[V, H]
```

where:

* `V` = original text vocabulary size
* `H` = hidden dimension

Extend it to:

```text
[V + 8000, H]
```

Implementation concept:

```python
old_embedding = model.text_embedding.weight

new_embedding = torch.empty(
    old_embedding.shape[0] + 8000,
    old_embedding.shape[1],
    device=old_embedding.device,
    dtype=old_embedding.dtype,
)

new_embedding[:old_embedding.shape[0]] = old_embedding
```

The new rows correspond only to Khmer tokens.

---

# 8. Khmer Embedding Initialization

The new Khmer embedding rows have no pretrained representation.

Initialize:

```text
Khmer embedding rows
        ↓
Random initialization
        ↓
Train with higher learning rate
```

A reasonable initialization is the same distribution used by the original embedding layer, or a statistically compatible initialization based on the original embedding weights.

Do not initialize Khmer rows by copying arbitrary S1 token embeddings.

The initial goal is for the embedding layer to learn useful Khmer representations from the TTS dataset.

---

# 9. Weight Tying Check

Before implementing the vocabulary extension, inspect the S1-mini architecture to determine whether the text embedding is tied to another projection/output layer.

Check:

```python
model.named_parameters()
model.named_modules()
```

Specifically determine:

```text
Input text embedding
        │
        └── Is it tied to any output projection?
```

If tied:

* extend both sides consistently.

If not tied:

* only extend the text input embedding.

Do **not** extend the semantic/audio output vocabulary merely because the text vocabulary was extended.

---

# 10. Semantic Token Vocabulary

The existing S1 semantic/audio token vocabulary must remain unchanged.

Do not do:

```text
Original semantic vocab + 8000 Khmer tokens
```

The 8,000 Khmer tokens belong to the **text side**.

The semantic tokenizer/codec remains the original S1 system.

---

# 11. LoRA Adaptation

Initially freeze the pretrained S1 parameters.

Train:

```text
1. Khmer embedding rows
2. LoRA parameters
```

Conceptually:

```text
S1 original weights
        │
        ├── Frozen
        │
        └── LoRA adapters
                  │
                  └── Trainable

Khmer embedding
        │
        └── Trainable
```

This significantly reduces trainable parameters and GPU memory requirements.

---

# 12. LoRA Configuration

Initial configuration:

```yaml
lora:
  r: 8
  alpha: 16
  dropout: 0.01
```

Start conservatively.

Potential later experiment:

```yaml
r: 32
alpha: 16
```

Do not increase LoRA capacity until the baseline experiment is stable.

---

# 13. LoRA Target Modules

Inspect the official S1-mini implementation before selecting target modules.

Prefer transformer linear layers involved in text/semantic processing, rather than indiscriminately applying LoRA to every module.

Candidate categories:

```text
Attention projections
    q_proj
    k_proj
    v_proj
    output projection

Feed-forward projections
    up/gate/down projections
```

The exact names must be determined from the actual S1-mini model implementation.

Create a script that prints:

```python
for name, module in model.named_modules():
    ...
```

Then select targets explicitly.

---

# 14. Trainable Parameter Policy

Initial training:

```text
Original S1 parameters       → FROZEN
Original text embeddings     → FROZEN
Khmer embedding rows         → TRAINABLE
LoRA parameters              → TRAINABLE
Codec/vocoder                → FROZEN
Semantic vocabulary          → FROZEN
```

Verify this programmatically before training.

Example checks:

```python
trainable_params = [
    (name, p)
    for name, p in model.named_parameters()
    if p.requires_grad
]
```

Print:

```text
Total parameters
Trainable parameters
Trainable percentage
```

---

# 15. Optimizer

Use AdamW:

```yaml
optimizer:
  name: AdamW
  betas: [0.9, 0.95]
  eps: 1e-5
  weight_decay: 0.1
```

(Originally specified as `0.0`; changed to `0.1` per explicit decision —
both `configs/tuna_v1.yaml` and `configs/tuna_v1_omni.yaml`, and
`training/trainer.py`'s `TrainingConfig.weight_decay` default, were
updated to match. Revisit if EXP001's pilot run shows the small trainable
set — Khmer embedding rows + LoRA params, section 14 — needs weaker
regularization than 0.1 provides.)

Use two parameter groups:

```text
Group 1:
    Khmer embedding

Group 2:
    LoRA parameters
```

This allows independent learning rates.

---

# 16. Learning-Rate Schedule

Use a WSD-style schedule:

```text
Warmup → Stable → Decay → Stable
```

## LoRA

```text
0 ───────────── 2k
5e-5 → 1e-4
Warmup

2k ─────────── 10k
1e-4
Stable

10k ────────── 35k
1e-4 → 1e-5
Decay

35k ────────── 40k
1e-5
Stable
```

## Khmer Embeddings

```text
0 ───────────── 2k
1e-4 → 2e-4
Warmup

2k ─────────── 10k
2e-4
Stable

10k ────────── 35k
2e-4 → 2e-5
Decay

35k ────────── 40k
2e-5
Stable
```

Implement this as a custom scheduler rather than relying on a standard cosine scheduler.

---

# 17. Learning-Rate Experiments

Run small experiments before committing to the full 40k-step run.

## EXP001 — Baseline

```text
LoRA peak LR       = 1e-4
Khmer embedding LR = 2e-4
```

## EXP002 — Conservative

```text
LoRA peak LR       = 5e-5
Khmer embedding LR = 1e-4
```

## EXP003 — Aggressive

```text
LoRA peak LR       = 2e-4
Khmer embedding LR = 2e-4
```

Do not start with:

```text
LoRA = 5e-4
```

unless the lower-LR experiments demonstrate that a higher learning rate is necessary.

---

# 18. Gradient Management

Use gradient clipping:

```yaml
max_grad_norm: 1.0
```

Recommended sequence:

```text
forward
   ↓
loss
   ↓
backward
   ↓
gradient scaling if AMP
   ↓
unscale
   ↓
gradient clipping
   ↓
optimizer.step()
   ↓
scheduler.step()
```

---

# 19. Mixed Precision

Use BF16 if supported by the Kaggle GPU/runtime.

Otherwise use FP16 with GradScaler.

Recommended:

```text
Preferred:
    BF16

Fallback:
    FP16 + GradScaler
```

The checkpoint system must save scaler state when FP16 GradScaler is used.

---

# 20. Dataset

Dataset:

```text
Panhapich/khmer-tts-processed
```

~~The dataset contains approximately 700 hours of Khmer TTS data.~~

**Corrected (section 20.1): the actual dataset is ~128 hours across
~55,206 clips (54,083 train + 1,123 validation), 10 train / 6 validation
speakers, 48.6GB total, sourced from `DDD-Cambodia/khm-asr-cultural`.**
The 700-hour / 126k-252k-clip estimate below was never verified against
the real dataset page and was wrong by roughly 5x; see section 20.1 for
the full corrected numbers and their downstream consequences.

---

# 21. Dataset Pipeline

Each sample should contain the information required by S1 training, conceptually:

```text
audio
text
speaker information if applicable
metadata
```

Pipeline:

```text
Dataset
   │
   ▼
Load sample
   │
   ▼
Normalize text
   │
   ▼
Khmer tokenizer
   │
   ▼
Extended S1 text IDs
   │
   ▼
Audio preprocessing
   │
   ▼
S1 semantic target generation/loading
   │
   ▼
Training batch
```

Avoid repeatedly computing expensive preprocessing if semantic targets can safely be cached.

## 21.1 Text Normalization (implemented)

"Normalize text" is not a placeholder box — it is a required, versioned
component, implemented in:

```text
data/text_normalize.py
```

with a smoke test in:

```text
scripts/test_normalize.py
```

It runs identically at training, validation, and inference time (import it
from exactly one place on each side; never duplicate the logic). Scope:

```text
Khmer digit (០-៩) <-> Arabic digit conversion
Cardinal numbers   → Khmer words   (2026 → ពីរពាន់ម្ភៃប្រាំមួយ)
Currency           → Khmer words   ($10, ៛5000, 20000 រៀល)
Dates              → Khmer words   (ថ្ងៃទី5 ខែមករា ឆ្នាំ2026, 05/01/2026)
```

Known limitation (tracked, not yet fixed): digit groups that should be
read digit-by-digit — phone numbers, ID numbers — are currently read as
cardinal quantities instead. Do not include raw phone/ID numbers in
training text until this is addressed; strip or hand-normalize them first.

Any future extension (percentages, units, time-of-day, abbreviations)
must go into this module, not a second normalizer, so training and
inference never drift apart. Re-run `scripts/test_normalize.py` before
every dataset regeneration.

### 21.1.1 Training data is already normalized — do not re-run this module on it

`Panhapich/khmer-tts-processed` states its own processing pipeline
already verbalizes numbers/currency/dates/percentages (PLAN.md section
20.1). Re-running `data/text_normalize.py` on that text a second time is
redundant (idempotent in practice — there are no digits left to convert
— but wasted work and a second place a subtle mismatch could hide).

`data/dataset.py`'s `TunaTTSDataset` therefore defaults to
`text_is_pre_normalized=True` and uses manifest text as-is; set it to
`False` only for a dataset whose text has NOT already been normalized.
Wired through `training/trainer.py`'s `TrainingConfig` and
`configs/tuna_v1.yaml`'s `dataset.text_is_pre_normalized`.

**This does not remove the need for `data/text_normalize.py` — it moves
where it matters.** Raw user-typed input at inference time is never
pre-normalized, so `data/text_normalize.py` must still run there. This
means training text was normalized by whatever pipeline built
`Panhapich/khmer-tts-processed` (unverified against this repo's own
`data/text_normalize.py` — the exact wording conventions may differ),
while inference text is normalized by this repo's normalizer. If the two
diverge on some construct (a currency phrasing, a date format), the
model would see that construct spelled one way during training and
another way at inference. Section 47.1's evaluation set is exactly where
this would surface — include a few numbers/currency/date sentences in it
specifically to catch this, and revisit `text_is_pre_normalized` if a
mismatch shows up in generated audio.

## 21.2 Semantic Target Generation

Semantic targets (discrete audio-codec tokens) are the most expensive step
in the pipeline — do not generate them on the fly inside the training loop.

Precompute once, offline, before Phase 3 (Forward Pass Test):

```text
For each audio clip:
   audio
     │
     ▼
   S1 audio codec (frozen, same one S1-mini ships with)
     │
     ▼
   semantic token sequence
     │
     ▼
   write to disk as a compact per-clip file (e.g. .npy / .pt),
   keyed by a stable clip ID
```

Storage layout:

```text
semantic_cache/
├── <clip_id>.npy      (or .pt)
└── index.json         (clip_id → text, duration, sample_rate, split)
```

Rules:

```text
1. Cache is keyed by clip_id + codec revision, not by dataset split.
2. If the codec version changes, the cache is invalidated wholesale —
   record codec_revision in metadata.json (see section 32) and refuse
   to reuse a cache built with a different codec_revision.
3. The dataset loader reads the cached semantic tokens directly;
   it must never fall back to recomputing them silently.
4. Precompute on the highest-throughput compute session available
   (semantic generation is a one-time cost per clip; do not repeat
   it per epoch or per training session).
```

Practical sizing (corrected, see section 20.1): the actual dataset is
~128 hours / ~55k clips, not the ~700 hours originally assumed here.
Semantic tokens run at roughly 1/100th to 1/300th the byte rate of the
24kHz audio they're derived from (4 codebooks + text row, ~2 bytes/token,
vs. ~48KB/sec raw audio), so the resulting cache is small — see section
21.3's worked numbers. Small enough to persist on Kaggle's disk or
Hugging Face permanently; never needs regenerating or re-downloading raw
audio once built. Treat "precompute and upload the semantic cache" as its
own pipeline step, separate from and prior to the training script.

## 20.1 Corrected Dataset Numbers

The ~700 hours / 126k-252k clips estimate in section 20 was wrong.
Verified against the actual `Panhapich/khmer-tts-processed` dataset page:

```text
Total size:   48.6 GB
Duration:     125.2 hours train + 2.6 hours validation  (~128 hours total)
Clips:        54,083 train + 1,123 validation            (~55,206 total)
Speakers:     10 train, 6 validation
Source:       DDD-Cambodia/khm-asr-cultural (real recordings, not synthetic)
Format:       .wav, Fish Speech speaker-folder layout
License:      CC BY-SA 4.0 (share-alike -- the fine-tuned model likely
              inherits this obligation; check before any restrictive
              redistribution, see section 54)
```

This changes section 26.1's session-budget math (fewer total clips to
epoch over) and confirms section 21.2/21.3's disk-budget design is
comfortably achievable even on a constrained Kaggle disk.

## 21.3 Disk Budget for Kaggle (20GB working example)

The 20GB-class disk constraint only binds the ONE-TIME precompute phase
(section 21.2), not ordinary training sessions. Implemented in
`data/sharding.py` + `scripts/precompute_semantic_cache.py`:

```text
Kaggle disk budget                    20 GB
  - S1 checkpoint (model.pth + codec.pth, ~3.6GB, rounded up)   4 GB
  - working overhead (python env, temp files, margin)           2 GB
  = per-shard raw-audio budget                                14 GB
```

`data/sharding.py.make_precompute_shards` deterministically bins clips
(stable-hash ordered, same technique as section 23's split assignment)
into shards whose estimated raw-audio size stays under that per-shard
budget. For the corrected 128-hour / 55k-clip dataset, this produces
only 2-4 shards total (worked example in the module's self-test).

Precompute flow, one shard at a time, each session or within one session
back-to-back:

```text
download shard's audio (only)
        ↓
run S1's audio codec (tools/vqgan/extract_vq.py's own loading pattern,
reused as-is -- see scripts/precompute_semantic_cache.py)
        ↓
write semantic tokens into the cache
        ↓
DELETE the shard's raw audio immediately
        ↓
mark shard complete (data/sharding.py.PrecomputeProgress, resumable
across sessions the same way section 35 resumes training)
        ↓
optionally upload the (small) semantic cache so this step is durable
and never repeats even after local disk is wiped
```

Once precompute finishes across however many sessions it takes, EVERY
subsequent training session only needs: the S1 checkpoint (~4GB) + the
semantic cache (estimated well under 1GB for this dataset size, see
`data/sharding.py`'s self-test) + the text manifest (negligible). No
audio, no sharding, no rotation — training sessions are not disk-bound
at all once precompute is done.

The raw-audio byte-rate assumption (24kHz/16-bit PCM) `data/sharding.py`
uses for shard-size planning does not match the dataset's actual stated
total (48.6GB implies a higher effective rate than that assumption
predicts for 128 hours) — `scripts/precompute_semantic_cache.py` logs
the *measured* MB/s on first download so this can be recalibrated rather
than trusted blindly.

## 21.4 Alternate data source: local omni_asr_kh corpus

`/home/helpdesk/Desktop/omni_asr_kh` is a separate, already-local project
(Khmer ASR fine-tuning of Meta's Omnilingual ASR) whose training data can
be reused for Tuna-TTS instead of downloading `Panhapich/khmer-tts-processed`
from HF. Converted via `scripts/convert_omni_asr_manifest.py`, verified
against the real files (not assumed):

```text
Source manifests:  omni_asr_kh/data/manifest_v3/{train,dev}.tsv + .wrd
                    (fairseq2/wav2vec2 format -- authoritative; the raw
                    audio/ directory tree contains many more files than
                    these manifests reference, from a since-superseded
                    resplit -- never glob audio/ directly, always join
                    from the .tsv+.wrd pair)
Sources:           DDD-Cambodia/khmer-speech-dataset (same family as
                    Panhapich's dataset, CC BY-SA 4.0) +
                    actableai/data-khmer (license NOT verified -- check
                    before training/releasing a model on this data)
Audio format:      16kHz mono 16-bit FLAC (lower than Panhapich's 24kHz;
                    the codec resamples during precompute regardless,
                    section 21.2, but upsampling doesn't recover
                    frequency content never captured at 16kHz)
Speaker coverage:  ddd_speaker_ids.json covers ~1.4% of clips (DDD-Cambodia
                    portion only, confirmed by inspection) -- most clips
                    (actableai portion) have no speaker label
Text:              raw ASR transcripts, NOT normalized (contains literal
                    Khmer digits, e.g. "៨" not "ប្រាំបី") -- use
                    text_is_pre_normalized: false (configs/tuna_v1_omni.yaml)
```

After conversion + validation-based filtering (section 22; excludes
clips >30s and text >500 chars, both legitimate exclusions rather than
pipeline bugs):

```text
468,081 usable clips, ~1,088.5 hours
  train:      466,053
  validation: 2,028
```

This is ~8.5x more hours than `Panhapich/khmer-tts-processed` (~128
hours, section 20.1), at the cost of lower native audio sample rate and
almost no speaker labels. `configs/tuna_v1_omni.yaml` /
`configs/experiments/exp001_omni.yaml` point the pipeline at this
manifest (`data/manifest.json`, produced locally, no download needed) —
everything downstream (tokenizer, embedding extension, LoRA, semantic
cache, WSD schedule, checkpointing) is identical to the
`Panhapich/khmer-tts-processed` path; only the manifest and
`text_is_pre_normalized` flag differ.

Not yet re-derived for this data source and worth doing before a full
run: section 26.1's session-budget estimate (1,088.5 hours is
substantially more `t_step`s to precompute/train over than the
128-hour numbers that estimate was based on).

## 21.5 Panhapich/khmer-tts-processed actually ships pre-tokenized data — use it directly (chosen path)

Correction to section 21.4's framing: `Panhapich/khmer-tts-processed`'s
files are NOT the "Fish Speech speaker-folder .wav" layout implied by an
earlier page-summary read. Inspecting the real files (downloaded and
opened, not assumed) shows two file families:

```text
processed/khmer_base__khmer_base_v1__shard{0-5}of6.tar        ~3.6GB each  raw audio
processed/khmer_base__khmer_base_v1__ready_shard{0-5}of6.tar  ~34MB each   .protos (fish-speech native format)
processed/khmer_base__khmer_base_v1__ready_val.tar            4.2MB       .protos, validation split
```

The `ready_*` files are ~100x smaller than their raw-audio counterparts
and, verified against `fish_speech/datasets/protos/text-data.proto`
(cloned and read directly):

```protobuf
message Sentence {
    repeated string texts = 1;        // raw text -- confirmed: 4,425 real
                                       // Khmer strings found in ready_val.tar
    repeated Semantics semantics = 3; // PRECOMPUTED VQ codebook indices
}
```

This is exactly the `(sentences, semantics)` shape `pack_sentences()`
consumes (section 61.5) — Panhapich already ran the S1 audio codec.
Fish-speech ships a dataset class that reads `.protos` files natively:
`fish_speech.datasets.semantic.AutoTextSemanticInstructionIterableDataset`,
which tokenizes `Sentence.texts` at load time through whatever tokenizer
object it's given.

**Consequence — chosen path:** use this dataset directly, with
`TunaTokenizer` swapped in as its tokenizer, via the new
`data/proto_dataset.py`. This makes section 21.2's entire semantic-cache
precompute step (`scripts/precompute_semantic_cache.py`,
`data/sharding.py`) **unnecessary for this dataset** — no raw audio
download (only ~208MB of `.protos` shards vs. ~45GB of audio), no codec
to run, no disk-budget sharding. `data/dataset.py`'s manifest.json path
remains the right tool for a dataset that does NOT ship pre-packaged
`.protos` (e.g. `omni_asr_kh`, section 21.4) — do not run both pipelines
against the same data.

Downloaded and extracted locally:

```text
data/protos/train/shard{0-5}_protos/*.protos   (12 files, 198MB total)
data/protos/validation/*.protos                (1 file)
```

`configs/tuna_v1.yaml`'s `dataset.format: protos` /
`dataset.proto_train_dir` / `dataset.proto_val_dir` point the pipeline
here; `training/trainer.py`'s `build_dataloaders` branches on
`dataset.format` to choose `data/proto_dataset.py` vs `data/dataset.py`.

Not yet verified (needs `fish-speech` installed and the real S1-mini
checkpoint, both in progress): that `AutoTextSemanticInstructionIterableDataset`
actually loads these specific files end-to-end with `TunaTokenizer`
substituted in, and that the precomputed `Semantics` codes are compatible
with S1-mini's current codec revision (no `codec_revision` metadata is
stored in the `.protos` format itself, unlike section 21.2's cache index
— if Panhapich's codec run predates or differs from the checkpoint's
current codec, the codes would be silently wrong. Verify by decoding a
few `Semantics` back to audio and listening, before trusting a full run).

---

# 22. Data Validation

Before training, run a dataset validation script.

Check:

```text
Missing audio
Missing text
Corrupted audio
Empty text
Unexpected characters
Very long text
Very short text
Invalid tokenizer output
Invalid token IDs
Audio duration
Sample rate
```

Also inspect a random sample of:

```text
Original Khmer text
        ↓
Tokenized text
        ↓
Extended IDs
```

Make sure no token IDs accidentally collide with the original S1 vocabulary.

---

# 23. Data Splits

Create deterministic:

```text
train
validation
```

splits.

The validation set must remain fixed across experiments so that validation losses are comparable.

Do not randomly regenerate the validation set every session.

Record the split information in:

```text
metadata.json
```

---

# 24. Training Length

Target:

```text
40,000 optimizer steps
```

Acceptable range:

```text
30,000–50,000 steps
```

Use optimizer steps as the primary training unit.

Do not rely on epoch count for checkpoint continuation because Kaggle sessions may terminate at arbitrary points.

---

# 25. Effective Batch Size

Choose the largest batch that fits the available GPU memory.

Use gradient accumulation if necessary.

For example:

```text
Per-GPU batch size = 2
Gradient accumulation = 8

Effective batch size = 16
```

Formula:

```text
effective_batch =
    per_gpu_batch
    × gradient_accumulation
    × number_of_GPUs
```

For two GPUs:

```text
2 × 4 × 2 = 16
```

Tune this based on actual memory usage.

---

# 26. Kaggle GPU Strategy

Kaggle may provide:

```text
2 × NVIDIA T4 16 GB
```

Two GPUs do not automatically provide one 32 GB memory pool.

With standard distributed training:

```text
GPU 0 → model replica
GPU 1 → model replica
```

Therefore, each GPU still needs enough VRAM for the model and batch.

Start with:

```text
Single GPU
```

and verify that training works.

Then move to:

```text
DistributedDataParallel
```

if additional throughput is required.

## 26.0 DistributedDataParallel (implemented)

Implemented in `training/distributed.py`, wired into
`training/trainer.py`. Launch:

```text
python scripts/train.py --config <config>                  # single GPU/CPU, unchanged
./scripts/train_ddp.sh <num_gpus> <config>                  # multi-GPU via torchrun
```

`training/distributed.py` reads torchrun's `RANK`/`LOCAL_RANK`/`WORLD_SIZE`
env vars; when absent (plain `python scripts/train.py`), everything is a
no-op and training runs exactly as it did before this section existed.
When present:

```text
model wrapped in DistributedDataParallel (only Khmer embedding + LoRA
    params are gradient-synced -- frozen base weights have
    requires_grad=False and are excluded automatically)
each rank's DataLoader uses DistributedSampler (disjoint shard per rank,
    re-shuffled per epoch via set_epoch)
only rank 0 writes latest.pt / best.pt / metadata.json (avoids every
    rank racing on the same file)
validation loss is computed per-rank then averaged across ranks
    (all_reduce) before the best.pt comparison, since each rank only
    saw its own shard of the validation set
gradient accumulation uses model.no_sync() on non-final micro-steps, so
    DDP all-reduces gradients once per optimizer step, not once per
    micro-batch
```

Re-stating the plan's own caveat above since it doesn't change with this
implementation: N GPUs give ~Nx throughput, not pooled memory. Section
25's `effective_batch = per_gpu_batch * gradient_accumulation *
num_gpus` already accounts for this — `per_gpu_batch` in that formula is
literally `config.per_gpu_batch_size`, unchanged whether running on 1
GPU or N.

## 26.1 Session Budget Estimate

Kaggle GPU sessions are capped (historically ~12h/session, ~30h/week per
account on T4/P100). The 40,000-step target must be converted into a
session count before committing to the schedule in section 16, otherwise
"40k steps" is not a real plan, just a number.

Estimate before starting EXP001:

```text
1. Run 200 steps at the intended batch size / gradient accumulation.
2. Measure wall-clock seconds per optimizer step  →  t_step
3. sessions_required ≈ (40,000 × t_step) / (usable_seconds_per_session)
```

Record the measured `t_step` and resulting `sessions_required` estimate in
`metadata.json` alongside the experiment ID, and re-check it after Phase 4
(Pilot Training) once real throughput is known — the Phase 3 forward-pass
numbers are usually optimistic (no dataloader contention, no checkpoint
upload overhead). If `sessions_required` exceeds what the weekly quota
allows in a reasonable timeframe (e.g. more than ~6-8 weeks), reduce
`max_steps` toward the lower end of the 30k-50k acceptable range (section
24) rather than silently letting the project stall mid-training.

Also budget session time for:

```text
Downloading latest.pt / metadata.json at session start
Downloading the semantic cache (section 21.2) if not already local
Uploading latest.pt every 200 steps
```

These are fixed per-session overheads, not per-step — on short sessions
they can be a significant fraction of total time, which is another reason
to prefer fewer, longer sessions over many short ones where possible.

---

# 27. Gradient Accumulation

If the batch does not fit:

```text
micro-batch
    ↓
forward/backward
    ↓
accumulate gradients
    ↓
repeat
    ↓
optimizer step
```

Checkpointing must occur based on **optimizer steps**, not micro-batches.

---

# 28. Validation

Run validation periodically.

Recommended:

```text
Validation every 1,000 steps
```

During early experiments, validation can be more frequent.

Track:

```text
train_loss
validation_loss
learning_rate
gradient_norm
step
epoch
```

The primary model-selection metric should initially be:

```text
validation loss
```

---

# 29. Checkpoint Strategy

The checkpoint system must remain deliberately simple.

The Hugging Face repository should contain only:

```text
Tuna-TTS/
├── latest.pt
├── best.pt
└── metadata.json
```

Do not create:

```text
checkpoints/
step_00200/
step_00400/
step_00600/
...
```

Do not accumulate historical checkpoint files.

---

# 30. `latest.pt`

`latest.pt` is the **continuation checkpoint**.

It is overwritten every:

```text
200 optimizer steps
```

It should contain everything required to continue training.

Recommended structure:

```python
checkpoint = {
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "global_step": ...,
    "epoch": ...,
    "best_val_loss": ...,
    "rng_state": ...,
    "cuda_rng_state": ...,
}
```

The exact keys can be adjusted to the implementation.

The important requirement is:

> `latest.pt` must be sufficient to resume training without losing optimizer/scheduler state.

---

# 31. `best.pt`

`best.pt` stores the model corresponding to the best validation result.

If:

```text
current_val_loss < best_val_loss
```

then:

```text
overwrite best.pt
```

Otherwise:

```text
do nothing
```

`best.pt` does not need to contain the full training continuation state unless desired.

At minimum it must contain the complete model adaptation state required to reconstruct the best model:

```text
Khmer embedding
LoRA parameters
and any other modified model parameters
```

---

# 32. `metadata.json`

`metadata.json` is the small human-readable source of training information.

It should contain information such as:

```json
{
  "project": "Tuna-TTS",
  "architecture_version": "tuna-v1",
  "base_model": "fishaudio/openaudio-s1-mini",
  "tokenizer": "Panhapich/khmer-sp-8k",
  "dataset": "Panhapich/khmer-tts-processed",
  "global_step": 13800,
  "epoch": 1,
  "best_val_loss": 1.82,
  "latest_checkpoint": "latest.pt",
  "best_checkpoint": "best.pt",
  "checkpoint_interval": 200,
  "validation_interval": 1000,
  "experiment_id": "EXP001"
}
```

Also record:

```text
Base model revision
Tokenizer revision
Dataset revision
LoRA configuration
Optimizer
Learning-rate configuration
Batch size
Gradient accumulation
Number of GPUs
Random seed
PyTorch version
CUDA version
Training start time
Last checkpoint time
```

---

# 33. Checkpoint Lifecycle

Every 200 optimizer steps:

```text
Training
   │
   ▼
Save local temporary checkpoint
   │
   ▼
Verify checkpoint
   │
   ▼
Replace latest.pt
   │
   ▼
Update metadata.json
   │
   ▼
Upload to Hugging Face
```

For validation:

```text
Validation
   │
   ▼
Is val_loss better?
   │
   ├── YES → replace best.pt
   │
   └── NO  → keep existing best.pt
```

---

# 34. Safe Checkpoint Writing

Avoid directly overwriting a valid checkpoint with a partially written file.

Use:

```text
latest.pt.tmp
```

then:

```text
write
   ↓
flush
   ↓
verify
   ↓
replace latest.pt
```

Conceptually:

```python
save_checkpoint("latest.pt.tmp")
verify_checkpoint("latest.pt.tmp")
replace("latest.pt.tmp", "latest.pt")
```

This prevents a Kaggle interruption during upload/save from destroying the previous valid `latest.pt`.

---

# 35. Kaggle Session Recovery

Suppose a Kaggle session terminates at:

```text
13,847 steps
```

The latest completed checkpoint may be:

```text
latest.pt → step 13,800
```

The next session should:

```text
1. Download latest.pt
2. Download metadata.json
3. Load checkpoint
4. Restore model
5. Restore optimizer
6. Restore scheduler
7. Restore scaler
8. Restore RNG
9. Set global_step = 13,800
10. Continue from step 13,801
```

No historical checkpoint search is necessary.

---

# 36. Best vs Latest

These have different purposes.

### latest.pt

```text
Purpose:
Training continuation

Updated:
Every 200 steps

Contains:
Full training state
```

### best.pt

```text
Purpose:
Best model for evaluation/inference

Updated:
Only when validation improves

Contains:
Best model adaptation state
```

### metadata.json

```text
Purpose:
Experiment and continuation information

Updated:
Alongside checkpoints
```

Example:

```text
step 10,000
val_loss = 1.85

latest.pt → 10,000
best.pt   → 10,000
```

Later:

```text
step 12,000
val_loss = 1.79

latest.pt → 12,000
best.pt   → 12,000
```

Later:

```text
step 14,000
val_loss = 1.82

latest.pt → 14,000
best.pt   → 12,000
```

This is expected.

---

# 37. Hugging Face Repository

Repository:

```text
Panhapich/Tuna-TTS
```

Keep the repository minimal:

```text
Panhapich/Tuna-TTS/
│
├── latest.pt
├── best.pt
└── metadata.json
```

No periodic checkpoint directories.

No:

```text
step_1000/
step_2000/
step_3000/
```

---

# 38. Resume Compatibility

Before loading `latest.pt`, verify:

```text
architecture_version
base_model
base_model_revision
tokenizer
tokenizer_revision
dataset
LoRA configuration
vocabulary size
hidden dimension
```

If any architecture-critical value differs, stop instead of silently loading an incompatible checkpoint.

Example:

```python
if metadata["architecture_version"] != CURRENT_ARCHITECTURE_VERSION:
    raise RuntimeError(...)
```

---

# 39. Architecture Versioning

Define:

```text
architecture_version = tuna-v1
```

The version should change whenever the checkpoint structure or model architecture changes incompatibly.

For example:

```text
tuna-v1
```

for:

```text
S1-mini
+
8k Khmer vocabulary
+
LoRA
```

If the architecture changes substantially:

```text
tuna-v2
```

This prevents accidentally loading incompatible checkpoints.

---

# 40. Experiment Management

Use experiment IDs:

```text
EXP001
EXP002
EXP003
```

Suggested experiments:

```text
EXP001
LoRA LR = 1e-4
Embedding LR = 2e-4

EXP002
LoRA LR = 5e-5
Embedding LR = 1e-4

EXP003
LoRA LR = 2e-4
Embedding LR = 2e-4
```

Record the experiment ID in:

```text
metadata.json
```

Do not overwrite the meaning of an experiment ID.

---

# 41. Training Phases

## Phase 1 — Architecture Verification

Before training:

```text
Load S1-mini
        ↓
Load Khmer tokenizer
        ↓
Extend vocabulary
        ↓
Extend embedding
        ↓
Verify dimensions
        ↓
Verify token IDs
        ↓
Verify LoRA
        ↓
Verify trainable parameters
```

No expensive training yet.

---

# 42. Phase 2 — Tokenizer Test

Test representative examples:

```text
Khmer only
English only
Khmer + English
Numbers
Punctuation
Mixed text
Long Khmer sentences
```

Verify:

```text
text
 → tokenizer
 → extended IDs
 → model embedding
```

No out-of-range IDs.

---

# 43. Phase 3 — Forward Pass Test

Run a small number of batches.

Check:

```text
No embedding index errors
No shape errors
No NaN
No Inf
Loss is finite
Gradients are finite
```

Run:

```text
10–100 optimizer steps
```

before starting the real experiment.

---

# 44. Phase 4 — Pilot Training

Run:

```text
500–3,000 steps
```

Evaluate:

```text
training loss
validation loss
gradient norms
GPU memory
GPU utilization
training speed
audio quality
Khmer pronunciation
```

Use this phase to identify implementation errors and unstable learning rates.

---

# 45. Phase 5 — Baseline Training

Start:

```text
EXP001
```

Target:

```text
40,000 steps
```

Checkpoint:

```text
every 200 steps
```

Validation:

```text
every 1,000 steps
```

---

# 46. Phase 6 — Compare Experiments

Compare:

```text
Validation loss
Convergence speed
Khmer pronunciation
Naturalness
English preservation
Stability
```

Do not select an experiment purely from training loss.

Validation performance and actual generated speech matter.

---

# 47. Audio Evaluation

Create a fixed evaluation set containing representative Khmer sentences.

Keep the sentences unchanged between experiments.

Evaluate:

```text
Khmer pronunciation
Tone/prosody
Naturalness
Speech clarity
Punctuation handling
Numbers
English words
Code-switching
Long sentences
```

Generate the same evaluation sentences from:

```text
Base S1-mini
EXP001
EXP002
EXP003
```

where applicable.

## 47.1 Evaluation Set Sourcing and Scoring

The evaluation set must be built, not improvised per experiment.

Sourcing:

```text
1. Hold out ~30-50 sentences from Panhapich/khmer-tts-processed that are
   EXCLUDED from both train and validation splits (record their IDs in
   the same metadata.json split record described in section 23).
2. Deliberately cover, at minimum:
     - short plain Khmer sentences
     - long Khmer sentences (multi-clause)
     - Khmer + English code-switching
     - numbers, currency, and dates run through data/text_normalize.py
     - punctuation-heavy text (lists, questions, quotes)
3. Freeze this set once chosen. Do not add/remove sentences between
   experiments — comparability depends on identical inputs.
```

Scoring: use a fixed rubric so results are numeric and comparable, not
just subjective listening notes. Adapt the release-gate style used by
comparable Khmer TTS efforts:

```text
Pronunciation      ≥ 4.0 / 5
Naturalness        ≥ 3.8 / 5
English/code-switch preservation ≥ 3.8 / 5
No severe artifacts (clipping, dropped words, garbled audio)
```

Have at least two listeners score each generated sample independently on
this rubric (script: `evaluation/evaluate.py`); average the scores per
experiment. Store per-sentence scores, not just the average, so a
regression on one category (e.g. English preservation, per section 48)
is visible even if the aggregate score looks fine.

---

# 48. Prevent Catastrophic Forgetting

Because most S1 parameters remain frozen, the risk is reduced.

Nevertheless, monitor:

```text
English text
mixed Khmer/English
general pronunciation
speech naturalness
```

If English performance deteriorates significantly, investigate:

```text
LoRA capacity
LoRA learning rate
Khmer embedding learning rate
training duration
dataset composition
```

---

# 49. Memory Optimization

For T4 16 GB GPUs:

Use:

```text
BF16/FP16
Gradient accumulation
Frozen base parameters
LoRA
Efficient data loading
```

Avoid unnecessarily storing gradients for frozen parameters.

Use:

```python
requires_grad = False
```

for frozen parameters.

---

# 50. DataLoader Optimization

Tune:

```text
num_workers
pin_memory
persistent_workers
prefetch_factor
```

according to Kaggle runtime behavior.

Monitor:

```text
GPU utilization
CPU utilization
data loading time
step time
```

If GPU utilization is low, determine whether the bottleneck is:

```text
audio loading
tokenization
semantic target generation
DataLoader
CPU preprocessing
```

rather than immediately increasing model size.

---

# 51. Reproducibility

Set:

```text
Python seed
NumPy seed
PyTorch seed
CUDA seed
```

Record the seed in:

```text
metadata.json
```

Also save RNG state in:

```text
latest.pt
```

This makes multi-session continuation substantially more reproducible.

---

# 52. Final Model

Once training is complete:

```text
latest.pt
```

contains the latest trained state.

```text
best.pt
```

contains the best validation model.

For deployment/evaluation, prefer:

```text
best.pt
```

unless the latest model has clearly better evaluation results outside the validation metric.

---

# 53. LoRA Merge

After selecting the final model, optionally merge LoRA into the base model.

Conceptually:

```text
Base S1-mini
     +
LoRA
     +
Khmer embedding extension
     ↓
Merged Tuna-TTS
```

Do not merge during training.

Keep the LoRA representation available until the final model has been selected and evaluated.

---

# 54. Final Model Export

The final export should contain:

```text
Tuna-TTS model
Khmer tokenizer
Tokenizer configuration
Architecture configuration
Inference configuration
```

Make sure the exported model knows that:

```text
original IDs → original S1 vocabulary

extended IDs → Khmer vocabulary
```

---

# 55. Implementation Order

Implement in this order:

```text
1. Inspect S1-mini architecture
        ↓
2. Identify text embedding
        ↓
3. Identify text vocabulary size
        ↓
4. Identify special/control tokens
        ↓
5. Check weight tying
        ↓
6. Implement Khmer tokenizer adapter
        ↓
7. Implement extended vocabulary mapping
        ↓
8. Extend text embedding
        ↓
9. Verify embedding dimensions
        ↓
10. Add LoRA
        ↓
11. Freeze original parameters
        ↓
12. Create optimizer parameter groups
        ↓
13. Implement WSD scheduler
        ↓
14. Implement dataset pipeline
        ↓
15. Implement validation
        ↓
16. Implement checkpoint manager
        ↓
17. Run forward-pass test
        ↓
18. Run 500–3,000 step pilot
        ↓
19. Run EXP001
        ↓
20. Compare experiments
        ↓
21. Run full training
        ↓
22. Select best checkpoint
        ↓
23. Evaluate audio
        ↓
24. Merge/export final model
```

---

# 56. Suggested Project Structure

The training code can be organized as:

```text
Tuna-TTS/
│
├── configs/
│   ├── tuna_v1.yaml
│   └── experiments/
│       ├── exp001.yaml
│       ├── exp002.yaml
│       └── exp003.yaml
│
├── tokenizer/
│   ├── tuna_tokenizer.py
│   └── vocabulary.py
│
├── model/
│   ├── model_builder.py
│   ├── embedding_extension.py
│   └── lora.py
│
├── data/
│   ├── dataset.py
│   ├── collator.py
│   └── validation.py
│
├── training/
│   ├── trainer.py
│   ├── scheduler.py
│   ├── checkpoint.py
│   └── validation.py
│
├── evaluation/
│   ├── evaluate.py
│   └── generate_samples.py
│
├── scripts/
│   ├── inspect_model.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── train.py
│
└── PLAN.md
```

The Hugging Face output repository remains separate and minimal:

```text
Panhapich/Tuna-TTS/
├── latest.pt
├── best.pt
└── metadata.json
```

---

# 57. Initial Configuration

A starting configuration should approximately be:

```yaml
project:
  name: Tuna-TTS
  architecture_version: tuna-v1
  experiment_id: EXP001

model:
  base_model: fishaudio/openaudio-s1-mini

tokenizer:
  name: Panhapich/khmer-sp-8k
  vocab_size: 8000

dataset:
  name: Panhapich/khmer-tts-processed

training:
  max_steps: 40000
  validation_interval: 1000
  checkpoint_interval: 200
  gradient_clip_norm: 1.0
  seed: 42

optimizer:
  name: AdamW
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1e-5

lora:
  r: 8
  alpha: 16
  dropout: 0.01

learning_rate:
  lora:
    warmup_start: 5e-5
    peak: 1e-4
    decay_end: 1e-5

  khmer_embedding:
    warmup_start: 1e-4
    peak: 2e-4
    decay_end: 2e-5

checkpoint:
  latest: latest.pt
  best: best.pt
  metadata: metadata.json
```

---

# 58. Success Criteria

The project is successful when:

### Architecture

* [ ] S1-mini loads correctly.
* [ ] Khmer tokenizer works correctly.
* [ ] Khmer IDs do not collide with original S1 IDs.
* [ ] Extended embedding has the correct dimensions.
* [ ] Special/control tokens are handled correctly.
* [ ] Semantic/audio vocabulary remains unchanged.
* [ ] LoRA is correctly attached.
* [ ] Original S1 parameters remain frozen.

### Training

* [ ] Forward pass works.
* [ ] Loss is finite.
* [ ] Gradients are finite.
* [ ] WSD schedule works correctly.
* [ ] Training survives Kaggle session restarts.
* [ ] `latest.pt` is updated every 200 steps.
* [ ] `best.pt` updates only when validation improves.
* [ ] No historical checkpoint accumulation occurs.
* [ ] `metadata.json` accurately tracks training state.

### Model

* [ ] Khmer text can be processed.
* [ ] Khmer pronunciation is intelligible.
* [ ] Speech is reasonably natural.
* [ ] English/code-switching remains usable.
* [ ] Validation loss improves.
* [ ] Final model can be reconstructed from the stored checkpoint.

---

# 59. Final Training Workflow

The complete workflow should be:

```text
                ┌─────────────────────┐
                │ S1-mini Base Model  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Extend Text Vocab   │
                │ + 8k Khmer Tokens   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Add Khmer Embedding │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Add LoRA Adapters   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Freeze S1 Weights   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Khmer TTS Dataset   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Train + Validate    │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
          Every 200 steps       Validation
                │                     │
                ▼                     ▼
          latest.pt             Better?
                                      │
                              ┌───────┴───────┐
                              │               │
                             Yes              No
                              │               │
                              ▼               │
                           best.pt            │
                              │               │
                              └───────┬───────┘
                                      │
                                      ▼
                               metadata.json
                                      │
                                      ▼
                              Hugging Face
                                      │
                                      ▼
                              Next Kaggle Session
                                      │
                                      ▼
                               Load latest.pt
                                      │
                                      ▼
                                Continue
```

---

# 60. Core Design Decision

The central design of Tuna-TTS is therefore:

```text
                Fish S1-mini
                     │
        ┌────────────┴────────────┐
        │                         │
   Original text            Khmer text
   vocabulary               vocabulary
        │                         │
   Existing IDs           V + Khmer IDs
        │                         │
        └────────────┬────────────┘
                     │
              Extended embedding
                     │
              ┌──────┴──────┐
              │             │
          Frozen S1       Trainable
          parameters      Khmer + LoRA
              │             │
              └──────┬──────┘
                     │
               S1 Semantic Model
                     │
              Existing codec
                     │
                     ▼
                   Audio
```

And the persistent training state is intentionally kept to:

```text
Panhapich/Tuna-TTS
│
├── latest.pt      ← resume training
├── best.pt        ← best validation model
└── metadata.json  ← training/reproducibility information
```

This keeps the repository clean while still allowing training to continue safely across multiple Kaggle sessions.

---

# 61. Verified S1-mini / fish-speech Architecture Findings

`fishaudio/openaudio-s1-mini`'s weights are gated on Hugging Face, but the
model code that defines its architecture is public in
`fishaudio/fish-speech` (`fish_speech/models/text2semantic/llama.py`,
`lora.py`, `fish_speech/tokenizer.py`). Section 9's "inspect before
implementing" step has been done against that source; this section
records the result so the codebase does not guess at names.

## 61.1 Text embedding and weight tying

```python
class BaseTransformer(nn.Module):
    self.embeddings = nn.Embedding(config.vocab_size, config.dim)
    self.codebook_embeddings = nn.Embedding(
        config.codebook_size * config.num_codebooks, config.dim
    )
    if config.tie_word_embeddings is False:
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
```

`tie_word_embeddings` defaults to `True`. When tied, output logits are
computed as `F.linear(hidden, self.embeddings.weight)` — there is no
separate output projection at all in the default configuration.

**Consequence (resolves section 9):** extending `self.embeddings` by
8,000 rows extends both the input *and* output side in one step when
`tie_word_embeddings=True`, because both read the same weight tensor.
`model/embedding_extension.py` still handles the untied case (extend
`model.output` too) defensively, in case a future base checkpoint ships
with `tie_word_embeddings=False`.

## 61.2 Text vocabulary vs. audio codes are already separate

`self.embeddings` (size `vocab_size`) holds **both** ordinary text tokens
*and* special semantic-boundary placeholder tokens (`<|semantic:i|>`,
per `fish_speech/tokenizer.py`). The actual residual-VQ audio codebook
contents live in a wholly separate table, `codebook_embeddings` /
`fast_embeddings` (size `codebook_size * num_codebooks`). This confirms
section 10: the 8,000 Khmer tokens only ever touch `self.embeddings`;
`codebook_embeddings` must not be touched.

## 61.3 Base tokenizer is not SentencePiece

`fish_speech.tokenizer.FishTokenizer` wraps a `transformers.AutoTokenizer`
over a tiktoken-format vocab (`tokenizer.tiktoken` + `special_tokens.json`
in the model repo), with special tokens for EOS/PAD/`<|im_start|>` etc.
and 4,096 `<|semantic:i|>` placeholders. This is unrelated to, and does
not conflict with, `Panhapich/khmer-sp-8k` (a separate SentencePiece
model used only to tokenize the Khmer-script portions of input text
before mapping into extended IDs — see `tokenizer/tuna_tokenizer.py`).

## 61.4 LoRA: use fish-speech's own implementation, don't reimplement

`fish_speech/models/text2semantic/lora.py` already implements LoRA for
this exact architecture via the third-party `loralib` package, with
verified target-module names:

```text
attention:   wqkv, wo
mlp:         w1, w2, w3
embeddings:  model.embeddings, model.codebook_embeddings
output:      model.output          (only present if tie_word_embeddings=False)
```

and ships `fish_speech/configs/lora/r_8_alpha_16.yaml` —
`r=8, lora_alpha=16, lora_dropout=0.01` — matching section 12's initial
config exactly. `model/lora.py` in this codebase is a thin wrapper around
`fish_speech.models.text2semantic.lora.setup_lora`, configured to target
only `["attention", "mlp"]` — explicitly excluding `"embeddings"` and
`"output"`, because Tuna-TTS trains the (extended) embedding directly at
full rank per sections 7-8/11/14, not through a LoRA adapter.

This means section 13's "create a script that prints named_modules" step
is satisfied by `scripts/inspect_model.py`, and target-module selection
is no longer a guess.

## 61.5 Training examples are templated, not raw tokenized text

`fish_speech.datasets.semantic.AutoTextSemanticInstructionDataset
.pack_sentences` (verified from source) does not train on plain
`tokenizer.encode(text)` sequences. It builds a
`fish_speech.content_sequence.ContentSequence` out of `TextPart`/`VQPart`
objects -- an instruction preamble ("Speak out the provided text."),
`<|speaker:user|>` + the text, `<|speaker:assistant|>` + `<|voice|>` +
the VQ audio codes -- and encodes that whole scaffold into a
`[num_codebooks+1, T]` `(tokens, labels)` pair: row 0 is text/semantic-
placeholder ids, rows 1..num_codebooks are the codec's residual-VQ
codebook ids over the semantic span, and `labels` is -100 everywhere
loss should be ignored.

`ContentSequence.encode(tokenizer=...)` calls `tokenizer.encode(text,
add_special_tokens=False)`, `tokenizer.get_token_id(...)`, and reads
`tokenizer.semantic_begin_id` directly off whatever tokenizer object it's
given. Consequence for `tokenizer/tuna_tokenizer.py`: `TunaTokenizer`
must be a true drop-in for `FishTokenizer` -- match that `encode()`
signature exactly and delegate every other attribute
(`get_token_id`, `semantic_begin_id`, `semantic_end_id`, `pad_token_id`,
...) straight through to the wrapped base tokenizer. Then fish-speech's
own packer can be reused unmodified (`data/dataset.py` does exactly
this) instead of Tuna-TTS reimplementing the instruction-template/label
packing logic, which is not something to guess at either.


