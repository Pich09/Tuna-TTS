"""
Main training entry point (PLAN.md sections 24-28, Phase 5/45).

Run (after Phases 1-4 have passed -- scripts/inspect_model.py,
scripts/test_tokenizer.py, scripts/test_model.py, a pilot run):

    python scripts/train.py --config configs/experiments/exp001.yaml
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(path: str) -> dict:
    import yaml

    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    extends = config.pop("extends", None)
    if extends:
        base = load_config(str((config_path.parent / extends).resolve()))
        merged = {**base, **config}
        # shallow-merge nested dicts one level deep (sufficient for this config shape)
        for key in ("learning_rate",):
            if key in base and key in config:
                merged[key] = {**base.get(key, {}), **config[key]}
        return merged
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    from training.trainer import TrainingConfig, run_training
    from tokenizer.tuna_tokenizer import TunaTokenizer
    from tokenizer.vocabulary import ExtendedVocabulary
    from fish_speech.tokenizer import FishTokenizer
    import sentencepiece as spm

    base_model_path = config["model"]["base_model"]
    base_tokenizer = FishTokenizer(f"{base_model_path}/tokenizer.tiktoken")
    khmer_sp = spm.SentencePieceProcessor(model_file=config["tokenizer"]["khmer_sp_model_path"])
    vocabulary = ExtendedVocabulary(
        s1_vocab_size=base_tokenizer.vocab_size,
        khmer_vocab_size=config["tokenizer"]["vocab_size"],
    )
    tokenizer = TunaTokenizer(base_tokenizer, khmer_sp, vocabulary)

    dataset_format = config["dataset"].get("format", "manifest")

    training_config = TrainingConfig(
        experiment_id=config["experiment_id"],
        base_model_path=base_model_path,
        checkpoint_dir=config["checkpoint"]["directory"],
        dataset_format=dataset_format,
        dataset_name=config["dataset"]["name"],
        manifest_path=config["dataset"].get("manifest_path"),
        semantic_cache_dir=config["dataset"].get("semantic_cache_dir"),
        codec_revision=config["dataset"].get("codec_revision"),
        proto_train_dir=config["dataset"].get("proto_train_dir"),
        proto_val_dir=config["dataset"].get("proto_val_dir"),
        max_steps=config["training"]["max_steps"],
        checkpoint_interval=config["training"]["checkpoint_interval"],
        validation_interval=config["training"]["validation_interval"],
        max_grad_norm=config["training"]["gradient_clip_norm"],
        per_gpu_batch_size=config["training"]["per_gpu_batch_size"],
        gradient_accumulation=config["training"]["gradient_accumulation"],
        seed=config["training"]["seed"],
        khmer_vocab_size=config["tokenizer"]["vocab_size"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        mixed_precision=config["training"]["mixed_precision"],
        text_is_pre_normalized=config["dataset"].get("text_is_pre_normalized", True),
        weight_decay=config["optimizer"]["weight_decay"],
        proto_max_length=config["dataset"].get("proto_max_length"),
        max_seq_len=config["model"].get("max_seq_len", 8192),
        max_val_batches=config["training"].get("max_val_batches", 50),
        log_interval=config["training"].get("log_interval", 1),
        hf_repo=config["checkpoint"].get("hf_repo"),
        hf_upload_enabled=config["checkpoint"].get("upload_to_hub", False),
        # Never put a real token in a config file -- read it from the
        # environment (or a secret store like a Kaggle Secret) instead.
        hf_token=os.environ.get("HF_TOKEN"),
    )

    run_training(training_config, tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
