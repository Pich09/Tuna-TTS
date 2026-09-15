"""
Training loop (PLAN.md sections 15-19, 24-28, 33): wires together
model_builder, the dataset/collator, WSDScheduler, and CheckpointManager.

Multi-GPU (PLAN.md section 26): launched under `torchrun --nproc_per_node=N`
(see scripts/train_ddp.sh), this runs DistributedDataParallel across N
GPUs automatically -- launched as plain `python scripts/train.py`
(no torchrun), it runs single-GPU/CPU exactly as before. See
training/distributed.py for the env-var-driven single/multi switch and
its own note on why N GPUs give ~Nx throughput, not pooled memory.

Requires torch + fish-speech + a real base-model checkpoint; this module
is the integration point, not something unit-testable in this sandbox
(see model/model_builder.py, training/scheduler.py, training/checkpoint.py
for the independently-tested pieces this file composes).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

from data.collator import TunaCollator
from data.dataset import TunaTTSDataset
from data.semantic_cache import SemanticCacheIndex
from model.model_builder import TunaModelConfig, build_tuna_model, verify_architecture
from training.checkpoint import ARCHITECTURE_VERSION, CheckpointManager, TrainingState
from training.distributed import (
    cleanup_distributed,
    device_for,
    make_sampler,
    read_distributed_env,
    reduce_mean,
    setup_distributed,
    unwrap_model,
    wrap_model_for_distributed,
)
from training.scheduler import EXPERIMENT_STAGES, WSDScheduler


@dataclass
class TrainingConfig:
    experiment_id: str
    base_model_path: str
    checkpoint_dir: str
    # Dataset: exactly one of the two shapes below is used, chosen by
    # dataset_format (PLAN.md section 21.5).
    #   "manifest": manifest_path + semantic_cache_dir + codec_revision
    #               (data/dataset.py -- raw audio, codec run by Tuna-TTS)
    #   "protos":   proto_train_dir + proto_val_dir
    #               (data/proto_dataset.py -- pre-packaged, e.g.
    #               Panhapich/khmer-tts-processed's ready_*.tar shards)
    dataset_format: str = "manifest"
    dataset_name: str = "Panhapich/khmer-tts-processed"
    manifest_path: Optional[str] = None
    semantic_cache_dir: Optional[str] = None
    codec_revision: Optional[str] = None
    proto_train_dir: Optional[str] = None
    proto_val_dir: Optional[str] = None
    max_steps: int = 40000
    checkpoint_interval: int = 200
    validation_interval: int = 1000
    max_grad_norm: float = 1.0
    per_gpu_batch_size: int = 2
    gradient_accumulation: int = 8
    seed: int = 42
    khmer_vocab_size: int = 8000
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.01
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "none" -- section 19
    text_is_pre_normalized: bool = True  # section 21.1: Panhapich/khmer-tts-processed ships pre-normalized
    weight_decay: float = 0.1  # section 15 originally specified 0.0; overridden per user request
    proto_max_length: Optional[int] = None  # protos path only; None -> build_proto_dataset's own default.
    max_seq_len: int = 8192  # backstop truncation in _protos_collate; matches S1-mini's own max_seq_len.
    max_val_batches: int = 50  # required cap: the protos val_loader is an infinite stream, see run_validation.
    log_interval: int = 1  # print a progress line every this many optimizer steps.
    hf_repo: Optional[str] = None  # e.g. "Panhapich/Tuna-TTS" -- configs/*.yaml's checkpoint.hf_repo.
    hf_upload_enabled: bool = False  # opt-in per config (checkpoint.upload_to_hub): see training/hub_upload.py.
    hf_token: Optional[str] = None  # prefer HF_TOKEN env var / cached login over this; see training/hub_upload.py.


def set_seed(seed: int) -> None:
    """PLAN.md section 51: reproducibility across Python/NumPy/PyTorch/CUDA."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _no_sync_context(model, env, is_sync_step: bool):
    """model.no_sync() during non-final gradient-accumulation micro-steps
    under DDP (see run_training); a no-op context everywhere else."""
    import contextlib

    if env.is_distributed and not is_sync_step and hasattr(model, "no_sync"):
        return model.no_sync()
    return contextlib.nullcontext()


def build_optimizer(model, config: TrainingConfig, stages: dict):
    """PLAN.md section 15: AdamW with two parameter groups (LoRA, Khmer embedding).

    Reads names off unwrap_model(model), not `model` directly: run_training
    calls this AFTER wrap_model_for_distributed, and under DDP every
    parameter name gets a "module." prefix, which silently breaks the
    startswith("embeddings.") match below (the "lora_" in n check survives
    since it's a substring match, not a prefix match) -- invisible on the
    single-GPU path (no DDP, no prefix) but breaks under torchrun --nproc_per_node>1.
    The underlying tensors are identical either way (DDP wraps, doesn't
    clone), so this doesn't change which parameters the optimizer updates.
    """
    import torch
    from training.distributed import unwrap_model

    named_params = list(unwrap_model(model).named_parameters())
    lora_params = [p for n, p in named_params if p.requires_grad and "lora_" in n]
    embedding_params = [p for n, p in named_params if p.requires_grad and n.startswith("embeddings.")]

    if not lora_params:
        raise RuntimeError("no trainable LoRA parameters found -- did apply_tuna_lora run?")
    if not embedding_params:
        raise RuntimeError("no trainable embedding parameters found -- did extend_text_embedding run?")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "name": "lora", "lr": stages["lora"].start_lr},
            {"params": embedding_params, "name": "khmer_embedding", "lr": stages["khmer_embedding"].start_lr},
        ],
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=config.weight_decay,
    )
    return optimizer


def _protos_collate(batch, multiple_of: int = 8, max_length: int = 8192):
    """
    Collates items from fish_speech's own AutoTextSemanticInstructionIterableDataset
    (each item: {"tokens": Tensor[num_codebooks+1, T], "labels": Tensor[same]},
    verified from source) into the same {"input_ids", "attention_mask", "labels"}
    shape data/collator.py's TunaCollator produces, so model_forward_loss and
    the rest of the training loop don't need to know which dataset format
    supplied the batch.

    Defensive hard truncation to `max_length` (default: S1-mini's own
    max_seq_len): verified from source that
    AutoTextSemanticInstructionIterableDataset.augment() does NOT enforce
    its own max_length -- sample_data() picks
    `num_samples = max_length // 20` sentences (a fixed 20-tokens/sentence
    estimate) and augment() concatenates all of them with no length check
    afterward. That estimate is wildly wrong for this dataset (measured:
    individual Khmer sentences pack to ~160-475 tokens each, not ~20), so
    build_proto_dataset's own max_length must be tuned low enough that
    `max_length // 20` sentences realistically stays in budget (see
    data/proto_dataset.py) -- but since that's a statistical estimate, not
    a guarantee, truncate here too as a hard backstop so an occasional
    long outlier truncates cleanly instead of crashing the whole training
    run with a mask-shape RuntimeError deep inside the model's forward
    pass (as it did before this was added).
    """
    import torch
    import torch.nn.functional as F

    from data.collator import IGNORE_LABEL_ID, CODEBOOK_PAD_TOKEN_ID, compute_padded_length

    batch = [
        {"tokens": item["tokens"][:, :max_length], "labels": item["labels"][:, :max_length]}
        for item in batch
    ]

    lengths = [item["tokens"].shape[1] for item in batch]
    pad_len = compute_padded_length(lengths, multiple_of=multiple_of)

    padded_tokens, padded_labels, attention_masks = [], [], []
    for item in batch:
        tokens, labels = item["tokens"], item["labels"]
        length = tokens.shape[1]
        pad_amount = pad_len - length
        padded_tokens.append(F.pad(tokens, (0, pad_amount), value=CODEBOOK_PAD_TOKEN_ID))
        padded_labels.append(F.pad(labels, (0, pad_amount), value=IGNORE_LABEL_ID))
        attention_masks.append(torch.tensor([1] * length + [0] * pad_amount, dtype=torch.long))

    return {
        "input_ids": torch.stack(padded_tokens, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "attention_mask": torch.stack(attention_masks, dim=0),
    }


def build_dataloaders(config: TrainingConfig, tokenizer, env):
    """
    `batch_size=config.per_gpu_batch_size` is per-GPU, not the effective
    batch size (PLAN.md section 25/26): under DDP each rank's DataLoader
    yields its own per_gpu_batch_size-sized batch from its own shard of
    the dataset (via DistributedSampler for the manifest path; the protos
    path shards internally, see below), so total work per optimizer step
    scales with world_size automatically.

    Branches on config.dataset_format (section 21.5):
        "manifest": data/dataset.py + data/semantic_cache.py (raw audio,
                    codec run by Tuna-TTS's own precompute script)
        "protos":   data/proto_dataset.py (pre-packaged .protos files,
                    e.g. Panhapich/khmer-tts-processed's ready_*.tar)
    """
    import torch

    if config.dataset_format == "protos":
        from data.proto_dataset import build_proto_dataset

        # AutoTextSemanticInstructionIterableDataset is an IterableDataset:
        # it shards by rank/worker and shuffles internally (verified from
        # source -- split_by_rank_worker + Random(seed).shuffle), so no
        # DistributedSampler here (map-style-only) and shuffle must be
        # left to the dataset itself.
        proto_kwargs = {} if config.proto_max_length is None else {"max_length": config.proto_max_length}
        train_ds = build_proto_dataset([config.proto_train_dir], tokenizer, **proto_kwargs)
        val_ds = build_proto_dataset([config.proto_val_dir], tokenizer, **proto_kwargs)

        collate = partial(_protos_collate, max_length=config.max_seq_len)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.per_gpu_batch_size, collate_fn=collate, num_workers=4,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.per_gpu_batch_size, collate_fn=collate, num_workers=2,
        )
        return train_loader, val_loader, None  # no sampler to call set_epoch on

    semantic_cache = SemanticCacheIndex(config.semantic_cache_dir, codec_revision=config.codec_revision)
    train_ds = TunaTTSDataset(
        config.manifest_path, "train", tokenizer, semantic_cache,
        text_is_pre_normalized=config.text_is_pre_normalized,
    )
    val_ds = TunaTTSDataset(
        config.manifest_path, "validation", tokenizer, semantic_cache,
        text_is_pre_normalized=config.text_is_pre_normalized,
    )

    collator = TunaCollator()

    train_sampler = make_sampler(train_ds, env, shuffle=True)
    val_sampler = make_sampler(val_ds, env, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.per_gpu_batch_size,
        shuffle=(train_sampler is None),  # DistributedSampler does its own shuffling
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.per_gpu_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=2,
    )
    return train_loader, val_loader, train_sampler


def run_training(config: TrainingConfig, tokenizer) -> None:
    import torch

    env = read_distributed_env()
    setup_distributed(env)  # no-op if not launched via torchrun (section 26: single-GPU path unchanged)

    # Distinct seed per rank keeps ranks from doing identical dropout/init
    # noise while still being fully reproducible (section 51).
    set_seed(config.seed + env.rank)

    stages = EXPERIMENT_STAGES.get(config.experiment_id)
    if stages is None:
        cleanup_distributed(env)
        raise ValueError(f"unknown experiment_id {config.experiment_id!r}; see training/scheduler.py EXPERIMENT_STAGES")

    model_config = TunaModelConfig(
        base_model_path=config.base_model_path,
        khmer_vocab_size=config.khmer_vocab_size,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    model, arch_info = build_tuna_model(model_config)
    verify_architecture(model, arch_info)  # Phase 1 gate -- stop here if this fails

    device = device_for(env)
    model.to(device)
    model = wrap_model_for_distributed(model, env)  # no-op if world_size == 1

    optimizer = build_optimizer(model, config, stages)
    scheduler = WSDScheduler(optimizer, stages)

    use_bf16 = config.mixed_precision == "bf16"
    use_fp16 = config.mixed_precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Only rank 0 touches the checkpoint/metadata files (section 33) --
    # every rank would otherwise race on the same latest.pt.tmp.
    checkpoint_manager = CheckpointManager(
        config.checkpoint_dir,
        checkpoint_interval=config.checkpoint_interval,
        validation_interval=config.validation_interval,
        architecture_info={
            "architecture_version": ARCHITECTURE_VERSION,
            "base_model": config.base_model_path,
            "tokenizer": "Panhapich/khmer-sp-8k",
            "dataset": config.dataset_name,
            "vocab_size": arch_info["extended_vocab_size"],
            "hidden_dim": arch_info["hidden_dim"],
            "lora": arch_info["lora"],
        },
    ) if env.is_main_process else None

    # Only rank 0 uploads (same reasoning as checkpoint_manager above); a
    # missing hf_repo or upload opt-out just leaves this None, and every
    # call site below is already a no-op guarded on that.
    hub_uploader = None
    if env.is_main_process and config.hf_upload_enabled and config.hf_repo:
        from training.hub_upload import HubUploader

        hub_uploader = HubUploader(config.hf_repo, token=config.hf_token)

    global_step = 0
    epoch = 0
    # All ranks must load the same resumed state, but only rank 0 reads
    # disk/HF; broadcast is implicit here because every rank runs the same
    # deterministic CheckpointManager.load_for_resume against the same
    # shared checkpoint directory (already-synced via HF/shared storage,
    # not re-downloaded per rank).
    resume_source = checkpoint_manager or CheckpointManager(
        config.checkpoint_dir, architecture_info={}
    )
    resumed = resume_source.load_for_resume({
        "architecture_version": ARCHITECTURE_VERSION,
        "base_model": config.base_model_path,
        "tokenizer": "Panhapich/khmer-sp-8k",
        "dataset": "Panhapich/khmer-tts-processed",
        "vocab_size": arch_info["extended_vocab_size"],
        "hidden_dim": arch_info["hidden_dim"],
    })
    if resumed is not None:
        unwrap_model(model).load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        if use_fp16 and "scaler_state_dict" in resumed:
            scaler.load_state_dict(resumed["scaler_state_dict"])
        global_step = resumed["global_step"]
        epoch = resumed["epoch"]
        if env.is_main_process:
            print(f"Resumed from step {global_step} (section 35)")

    train_loader, val_loader, train_sampler = build_dataloaders(config, tokenizer, env)

    if env.is_main_process:
        print(f"Dataloaders ready -- starting training loop (max_steps={config.max_steps}).", flush=True)

    model.train()
    accum_step = 0
    accum_loss_sum = 0.0
    optimizer.zero_grad()

    import time

    step_start_time = time.monotonic()

    while global_step < config.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # required for correct shuffling under DDP

        for batch in train_loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            accum_step += 1
            is_sync_step = accum_step % config.gradient_accumulation == 0
            # Under DDP, backward() all-reduces gradients by default on every
            # call. During gradient accumulation that means a wasteful
            # all-reduce per micro-batch instead of once per optimizer step;
            # model.no_sync() suppresses it for every micro-batch except the
            # last, where the sync must happen. No-op (contextlib.nullcontext)
            # when not distributed.
            sync_context = _no_sync_context(model, env, is_sync_step)

            with sync_context:
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                    loss = model_forward_loss(model, batch)  # see note below
                    loss = loss / config.gradient_accumulation

                accum_loss_sum += loss.item() * config.gradient_accumulation  # undo the division, for logging

                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if not is_sync_step:
                continue
            accum_step = 0

            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), config.max_grad_norm
            )
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            scheduler.step(global_step)

            # Progress line every `log_interval` optimizer steps (main
            # process only, to avoid duplicate prints under DDP), plus
            # always at global_step==1 so a fresh run confirms it's alive
            # immediately instead of going silent for log_interval *
            # gradient_accumulation micro-batches (e.g. 20*8=160, which on
            # Kaggle's T4s can be several minutes -- easy to mistake for a
            # hang otherwise). This was previously silent between
            # checkpoint_interval/validation_interval boundaries (every
            # 200 / 1000 steps), which at gradient_accumulation=16 and
            # this environment's per-step time left no way to tell a
            # slow-but-healthy run apart from a hung one from the log alone.
            now = time.monotonic()
            step_seconds = now - step_start_time
            step_start_time = now
            if env.is_main_process and (global_step == 1 or global_step % config.log_interval == 0):
                avg_micro_loss = accum_loss_sum / config.gradient_accumulation
                lr_lora = optimizer.param_groups[0]["lr"]
                print(
                    f"step {global_step}/{config.max_steps}: loss={avg_micro_loss:.4f} "
                    f"lr_lora={lr_lora:.2e} {step_seconds:.1f}s/step",
                    flush=True,
                )
            accum_loss_sum = 0.0

            # Computed identically on every rank from config alone (not via
            # checkpoint_manager, which only exists on rank 0) -- global_step
            # is kept in lockstep across ranks by DDP's gradient sync, so
            # every rank reaches these branches on the same step.
            should_checkpoint_step = global_step > 0 and global_step % config.checkpoint_interval == 0
            should_validate_step = global_step > 0 and global_step % config.validation_interval == 0

            if should_checkpoint_step and checkpoint_manager is not None:
                state = TrainingState(
                    model_state_dict=unwrap_model(model).state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    global_step=global_step,
                    epoch=epoch,
                    best_val_loss=checkpoint_manager.best_val_loss,
                    scaler_state_dict=scaler.state_dict() if use_fp16 else None,
                )
                checkpoint_manager.save_latest(state)
                checkpoint_manager.write_metadata(global_step, epoch, config.experiment_id)
                if hub_uploader is not None:
                    started = hub_uploader.maybe_upload_async(
                        {
                            checkpoint_manager.paths.latest_name: checkpoint_manager.paths.latest,
                            checkpoint_manager.paths.metadata_name: checkpoint_manager.paths.metadata,
                        },
                        commit_message=f"step {global_step}",
                    )
                    if not started:
                        print(f"[hub_upload] skipped step {global_step}: previous upload still in progress", flush=True)

            if should_validate_step:
                # Every rank must call this: reduce_mean below is a collective
                # (all_reduce) -- skipping it on non-main ranks would deadlock
                # them waiting for a collective that rank 0 never joins.
                val_loss = run_validation(model, val_loader, device, use_bf16, max_batches=config.max_val_batches)
                val_loss = reduce_mean(val_loss, env)  # average across ranks, each of which saw a different shard
                if checkpoint_manager is not None:
                    improved = checkpoint_manager.maybe_save_best(val_loss, unwrap_model(model).state_dict())
                    checkpoint_manager.write_metadata(
                        global_step, epoch, config.experiment_id, extra={"last_val_loss": val_loss}
                    )
                    print(f"step {global_step}: val_loss={val_loss:.4f} improved={improved}")
                    if improved and hub_uploader is not None:
                        started = hub_uploader.maybe_upload_async(
                            {checkpoint_manager.paths.best_name: checkpoint_manager.paths.best},
                            commit_message=f"step {global_step}: val_loss={val_loss:.4f} (best)",
                        )
                        if not started:
                            print(f"[hub_upload] skipped best.pt at step {global_step}: previous upload still in progress", flush=True)
                model.train()

            if global_step >= config.max_steps:
                break
        epoch += 1

    cleanup_distributed(env)


def model_forward_loss(model, batch):
    """
    Forward + loss, matching fish_speech's own training step exactly
    (fish_speech/models/text2semantic/lit_module.py TextToSemantic._step,
    verified from source -- PLAN.md section 61). Reused as-is rather than
    reinvented: base_loss is text-token cross-entropy (this is where the
    extended Khmer vocabulary's new rows actually get gradient signal),
    semantic_loss is the codebook cross-entropy over the *existing*,
    untouched semantic vocabulary (section 10).

    `batch["labels"]` must be shaped [B, 1 + num_codebooks, T]: row 0 is
    text/semantic-placeholder token labels, rows 1..num_codebooks are the
    audio codec's residual-VQ codebook labels for the semantic span, with
    -100 everywhere loss should be ignored (standard HF/PyTorch convention).
    Building `labels` in that shape from the collator's raw
    `semantic_targets` is dataset-specific glue left for
    data/collator.py once real semantic-cache tensors are available.

    Called with `model` possibly a DistributedDataParallel wrapper: the
    forward call itself must go through `model(...)` (not the unwrapped
    module) so DDP's gradient-sync hooks fire, but custom attributes like
    `.tokenizer`/`.config` aren't proxied by DDP and must be read off the
    unwrapped module instead (training/distributed.py's unwrap_model).
    """
    import torch.nn.functional as F

    from training.distributed import unwrap_model

    raw_model = unwrap_model(model)

    outputs = model(
        inp=batch["input_ids"],
        key_padding_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    token_logits = outputs.token_logits
    codebook_logits = outputs.codebook_logits
    labels = batch["labels"]

    base_loss = F.cross_entropy(
        token_logits.view(-1, token_logits.size(-1)),
        labels[:, 0].reshape(-1),
        ignore_index=-100,
    )

    token_ids = labels[:, 0]
    semantic_mask = (token_ids >= raw_model.tokenizer.semantic_begin_id) & (
        token_ids <= raw_model.tokenizer.semantic_end_id
    )
    all_codebook_labels = labels[:, 1 : 1 + raw_model.config.num_codebooks]
    all_codebook_labels_permuted = all_codebook_labels.permute(0, 2, 1)
    filtered_codebook_labels = all_codebook_labels_permuted[semantic_mask]
    semantic_loss = F.cross_entropy(
        codebook_logits.reshape(-1, codebook_logits.size(-1)),
        filtered_codebook_labels.reshape(-1),
        ignore_index=-100,
    )

    return base_loss + semantic_loss


def run_validation(model, val_loader, device: str, use_bf16: bool, max_batches: int = 50) -> float:
    """
    `max_batches` is required, not optional (PLAN.md section 21.5 finding):
    the .protos path's val_loader wraps
    fish_speech.datasets.semantic.AutoTextSemanticInstructionIterableDataset,
    whose __iter__ is `while True: yield self.augment()` (verified from
    source) -- an infinite generator by design, for streaming training.
    `for batch in val_loader` with no cap hangs forever on that path (it
    did, during this project's own pilot run). The manifest.json path's
    val_loader is a finite map-style Dataset and would stop on its own,
    but capping here too keeps validation duration bounded and predictable
    regardless of dataset format or size.
    """
    import torch

    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= max_batches:
                break
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model_forward_loss(model, batch)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)
