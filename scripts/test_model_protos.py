"""
Phase 3 - Forward Pass Test, .protos dataset variant (PLAN.md section 43,
section 21.5).

Same checks as scripts/test_model.py (no embedding index errors, no shape
errors, no NaN/Inf, loss finite, gradients finite) but wired for the
pre-packaged .protos dataset path (data/proto_dataset.py) used by
Panhapich/khmer-tts-processed, instead of manifest.json + semantic_cache.

Run:
    python scripts/test_model_protos.py \
        --base-model-path checkpoints/openaudio-s1-mini \
        --khmer-sp-model data/khmer-sp-8k.model \
        --proto-train-dir data/protos/train \
        --num-batches 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--khmer-sp-model", required=True)
    parser.add_argument("--khmer-vocab-size", type=int, default=8000)
    parser.add_argument("--proto-train-dir", required=True)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    import torch
    import sentencepiece as spm

    from fish_speech.tokenizer import FishTokenizer
    from tokenizer.tuna_tokenizer import TunaTokenizer
    from tokenizer.vocabulary import ExtendedVocabulary
    from data.proto_dataset import build_proto_dataset
    from training.trainer import _protos_collate, model_forward_loss, build_optimizer, TrainingConfig
    from training.scheduler import EXPERIMENT_STAGES
    from model.model_builder import TunaModelConfig, build_tuna_model, verify_architecture

    print("Building tokenizer...")
    base_tokenizer = FishTokenizer(f"{args.base_model_path}/tokenizer.tiktoken")
    khmer_sp = spm.SentencePieceProcessor(model_file=args.khmer_sp_model)
    vocabulary = ExtendedVocabulary(
        s1_vocab_size=base_tokenizer.vocab_size,
        khmer_vocab_size=args.khmer_vocab_size,
    )
    tokenizer = TunaTokenizer(base_tokenizer, khmer_sp, vocabulary)

    print("Building model...")
    model_config = TunaModelConfig(base_model_path=args.base_model_path)
    model, info = build_tuna_model(model_config)
    verify_architecture(model, info)
    print(f"Trainable: {info['trainable_percent']:.3f}% ({info['trainable_parameters']:,} / {info['total_parameters']:,})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model.to(device)
    model.tokenizer = tokenizer  # model_forward_loss reads semantic_begin_id/end_id off this

    print(f"Building .protos dataset from {args.proto_train_dir} ...")
    train_ds = build_proto_dataset([args.proto_train_dir], tokenizer)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=_protos_collate,
    )

    train_cfg = TrainingConfig(
        experiment_id="smoketest",
        base_model_path=args.base_model_path,
        checkpoint_dir="/tmp/smoketest-ckpt",
        weight_decay=0.1,
    )
    optimizer = build_optimizer(model, train_cfg, EXPERIMENT_STAGES["EXP001"])

    model.train()
    it = iter(loader)
    for step in range(args.num_batches):
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        loss = model_forward_loss(model, batch)

        if not torch.isfinite(loss):
            print(f"[FAIL] step {step}: loss is not finite: {loss.item()}")
            return 1

        loss.backward()

        bad_grad = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"[FAIL] step {step}: non-finite gradient in {name}")
                bad_grad = True
        if bad_grad:
            return 1

        optimizer.step()
        print(f"step {step}: loss={loss.item():.4f}")

    print("\n[PASS] all batches produced finite loss and finite gradients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
