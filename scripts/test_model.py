"""
Phase 3 - Forward Pass Test (PLAN.md section 43).

Runs a small number of batches through the fully-assembled Tuna-TTS model
and checks: no embedding index errors, no shape errors, no NaN/Inf, loss
finite, gradients finite. Must pass before Phase 4 (Pilot Training).

Requires the real S1-mini checkpoint, fish-speech, torch, and a small
real (or synthetic-but-correctly-shaped) manifest + semantic cache; not
runnable in this sandbox. This script is the executable version of
section 43's checklist -- run it on Kaggle/GPU before spending any real
training budget.

Run:
    python scripts/test_model.py \
        --base-model-path /path/to/openaudio-s1-mini \
        --manifest manifest.json --semantic-cache-dir semantic_cache \
        --codec-revision v1 --num-batches 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--semantic-cache-dir", required=True)
    parser.add_argument("--codec-revision", required=True)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=50, help="optimizer steps, section 43")
    args = parser.parse_args()

    import torch

    from data.collator import TunaCollator
    from data.dataset import TunaTTSDataset
    from data.semantic_cache import SemanticCacheIndex
    from model.model_builder import TunaModelConfig, build_tuna_model, verify_architecture
    from training.trainer import build_optimizer, model_forward_loss
    from training.scheduler import EXPERIMENT_STAGES

    print("Building model...")
    model_config = TunaModelConfig(base_model_path=args.base_model_path)
    model, info = build_tuna_model(model_config)
    verify_architecture(model, info)
    print(f"Trainable: {info['trainable_percent']:.3f}% ({info['trainable_parameters']:,} / {info['total_parameters']:,})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    semantic_cache = SemanticCacheIndex(args.semantic_cache_dir, codec_revision=args.codec_revision)
    dataset = TunaTTSDataset(args.manifest, "train", model.tokenizer, semantic_cache)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=TunaCollator())

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.experiment_id = "EXP001"
    cfg.weight_decay = 0.1
    optimizer = build_optimizer(model, cfg, EXPERIMENT_STAGES["EXP001"])

    step = 0
    for batch in loader:
        if step >= args.num_steps:
            break
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

        loss = model_forward_loss(model, batch)
        if not torch.isfinite(loss):
            print(f"FAIL at step {step}: loss is not finite ({loss.item()})")
            return 1

        optimizer.zero_grad()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"FAIL at step {step}: non-finite gradient in {name}")
                return 1

        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()

        print(f"step {step}: loss={loss.item():.4f}")
        step += 1

    print(f"\nPhase 3 forward-pass test passed: {step} steps, all losses/gradients finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
