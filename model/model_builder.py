"""
Model builder: assembles the full Tuna-TTS adaptation stack in the order
PLAN.md section 55 specifies:

    load S1-mini -> extend vocabulary/embedding -> add LoRA ->
    freeze base -> verify trainable parameters

This is Phase 1 (Architecture Verification, PLAN.md section 41) made
concrete and re-runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from model.embedding_extension import (
    extend_text_embedding,
    freeze_base_parameters,
    freeze_original_embedding_rows,
    is_tied,
    summarize_trainable_parameters,
)
from model.lora import apply_tuna_lora


@dataclass
class TunaModelConfig:
    base_model_path: str  # local path or HF repo id for fishaudio/openaudio-s1-mini
    khmer_vocab_size: int = 8000
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.01
    lora_target_modules: Optional[list] = None


def load_base_model(base_model_path: str):
    """
    Load the frozen S1-mini base model (PLAN.md section 3). Requires
    fish-speech + torch installed and the checkpoint downloaded locally
    (fish-speech's from_pretrained expects a local directory containing
    model.pth / config.json / tokenizer.tiktoken).
    """
    try:
        from fish_speech.models.text2semantic.llama import BaseTransformer
    except ImportError as e:
        raise ImportError(
            "model/model_builder.py requires the `fish-speech` package. "
            "Install it per requirements.txt."
        ) from e

    model = BaseTransformer.from_pretrained(base_model_path, load_weights=True)
    return model


def build_tuna_model(config: TunaModelConfig):
    """
    Full assembly pipeline. Returns (model, info) where `info` is a dict
    suitable for logging / writing into metadata.json (section 32).
    """
    model = load_base_model(config.base_model_path)

    original_vocab_size = model.config.vocab_size
    tied = is_tied(model)

    extend_text_embedding(model, khmer_vocab_size=config.khmer_vocab_size)
    freeze_original_embedding_rows(model.embeddings, num_original_rows=original_vocab_size)

    apply_tuna_lora(
        model,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )

    # Freeze everything else (base weights); embeddings and LoRA params are
    # left alone by freeze_base_parameters and were just made trainable above.
    freeze_base_parameters(model)

    param_summary = summarize_trainable_parameters(model)

    info = {
        "base_model": config.base_model_path,
        "original_vocab_size": original_vocab_size,
        "khmer_vocab_size": config.khmer_vocab_size,
        "extended_vocab_size": model.config.vocab_size,
        "tie_word_embeddings": tied,
        "hidden_dim": getattr(model.config, "dim", None),
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules or ["attention", "mlp"],
        },
        **param_summary,
    }
    return model, info


def verify_architecture(model, info: dict) -> None:
    """
    PLAN.md section 41 checklist, made executable:
        - dimensions match expectation
        - trainable set is exactly {new embedding rows, LoRA params}
        - no accidental full-model unfreeze
    """
    assert model.config.vocab_size == info["extended_vocab_size"], "vocab size mismatch after extension"
    assert model.embeddings.weight.shape[0] == info["extended_vocab_size"], "embedding row count mismatch"

    trainable_pct = info["trainable_percent"]
    if trainable_pct > 20.0:
        raise RuntimeError(
            f"trainable_percent={trainable_pct:.2f}% is implausibly high for "
            "LoRA + embedding-only training -- check that freeze_base_parameters "
            "actually ran and that no unintended parameters were left trainable."
        )
    if trainable_pct <= 0.0:
        raise RuntimeError("trainable_percent is 0 -- nothing would be trained.")

    for name, param in model.named_parameters():
        if param.requires_grad:
            if not (name.startswith("embeddings.") or "lora_" in name):
                raise RuntimeError(
                    f"unexpected trainable parameter outside embedding/LoRA scope: {name}"
                )
