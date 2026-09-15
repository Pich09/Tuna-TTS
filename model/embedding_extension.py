"""
Text embedding extension (PLAN.md sections 7, 8, 9) against the real
fish-speech BaseTransformer architecture (PLAN.md section 61).

Requires torch + a loaded fish_speech BaseTransformer. Import errors are
raised lazily (at call time) rather than at module import time, so that
tokenizer-only / scheduler-only code paths in this repo can still be
imported and tested without torch installed (see the rest of this repo's
self-tests, which all run without torch).
"""

from __future__ import annotations

from typing import Optional


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "model/embedding_extension.py requires torch. "
            "Install it (see requirements.txt) before calling into this module."
        ) from e
    return torch, nn


def is_tied(model) -> bool:
    """PLAN.md section 9: check weight tying before extending anything."""
    return bool(getattr(model.config, "tie_word_embeddings", True))


def get_text_embedding(model):
    if not hasattr(model, "embeddings"):
        raise AttributeError(
            "model has no `embeddings` attribute -- this does not look like a "
            "fish_speech BaseTransformer (PLAN.md section 61.1). Re-run "
            "scripts/inspect_model.py against the actual loaded model before "
            "assuming this code path applies."
        )
    return model.embeddings


def extend_text_embedding(model, khmer_vocab_size: int = 8000, init_std: Optional[float] = None):
    """
    Extend model.embeddings from [V, H] to [V + khmer_vocab_size, H]
    (PLAN.md section 7), initializing the new rows from a statistically
    compatible normal distribution (PLAN.md section 8: never copy
    arbitrary existing rows into the new ones).

    If tie_word_embeddings is False, also extends model.output
    (PLAN.md section 9's untied branch) since output logits then come
    from a separate Linear rather than reusing embeddings.weight.
    """
    torch, nn = _require_torch()

    old_embed = get_text_embedding(model)
    old_num, hidden_dim = old_embed.weight.shape
    new_num = old_num + khmer_vocab_size

    new_embed = nn.Embedding(new_num, hidden_dim, padding_idx=old_embed.padding_idx)
    with torch.no_grad():
        new_embed.weight[:old_num] = old_embed.weight
        std = init_std if init_std is not None else old_embed.weight.std().item()
        nn.init.normal_(new_embed.weight[old_num:], mean=0.0, std=std)

    new_embed = new_embed.to(device=old_embed.weight.device, dtype=old_embed.weight.dtype)
    model.embeddings = new_embed
    model.config.vocab_size = new_num

    if not is_tied(model):
        if hasattr(model, "output"):
            extend_output_projection(model, khmer_vocab_size)
        else:
            raise AttributeError(
                "tie_word_embeddings is False but model has no `output` attribute -- "
                "re-inspect the loaded model (scripts/inspect_model.py) before proceeding; "
                "section 9's untied branch does not apply cleanly here."
            )

    return model


def extend_output_projection(model, khmer_vocab_size: int):
    """Untied-output branch of PLAN.md section 9. Not exercised by the default
    S1-mini config (tie_word_embeddings=True, section 61.1), kept for safety."""
    torch, nn = _require_torch()

    old_linear = model.output
    old_out, in_dim = old_linear.weight.shape
    new_out = old_out + khmer_vocab_size

    new_linear = nn.Linear(in_dim, new_out, bias=old_linear.bias is not None)
    with torch.no_grad():
        new_linear.weight[:old_out] = old_linear.weight
        std = old_linear.weight.std().item()
        nn.init.normal_(new_linear.weight[old_out:], mean=0.0, std=std)
        if old_linear.bias is not None:
            new_linear.bias[:old_out] = old_linear.bias
            nn.init.zeros_(new_linear.bias[old_out:])

    new_linear = new_linear.to(device=old_linear.weight.device, dtype=old_linear.weight.dtype)
    model.output = new_linear


def freeze_original_embedding_rows(embedding, num_original_rows: int):
    """
    PLAN.md sections 11/14: the original S1 text embedding rows must stay
    frozen while the new Khmer rows train. requires_grad is per-Parameter,
    not per-row, so a single Embedding weight can't have `requires_grad`
    set differently for a subrange. Instead: keep requires_grad=True on the
    whole weight, and mask the gradient for original rows to zero via a
    backward hook, so the optimizer's update on those rows is always 0.
    """
    torch, _ = _require_torch()

    embedding.weight.requires_grad_(True)
    # Built once on whatever device the embedding lives on *right now*, but
    # the model is commonly moved to another device (e.g. model.to("cuda"))
    # after this hook is registered -- .to(grad.device) inside the closure
    # (cheap no-op once mask is already on the right device) keeps this
    # correct regardless of when that move happens, instead of silently
    # keeping a stale-device mask that only fails at backward() time.
    mask = torch.zeros_like(embedding.weight)
    mask[num_original_rows:] = 1.0

    def _zero_original_rows_grad(grad):
        return grad * mask.to(grad.device)

    embedding.weight.register_hook(_zero_original_rows_grad)
    return embedding


def freeze_base_parameters(model, exclude: Optional[list] = None):
    """
    PLAN.md section 14: freeze everything except Khmer embedding rows and
    LoRA parameters. `exclude` is a list of (name substrings) to leave alone
    (LoRA params are typically already handled by
    fish_speech's own mark_only_lora_as_trainable via model/lora.py, so the
    default here only needs to skip the embedding, which gets its own
    row-masked gradient via freeze_original_embedding_rows above).
    """
    exclude = exclude or []
    for name, param in model.named_parameters():
        if any(substr in name for substr in exclude):
            continue
        if "lora_" in name:
            continue  # left alone; fish_speech's setup_lora manages this
        if name.startswith("embeddings."):
            continue  # left alone; handled by freeze_original_embedding_rows
        param.requires_grad = False


def summarize_trainable_parameters(model) -> dict:
    """PLAN.md section 14: print/return total vs. trainable parameter counts."""
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_percent": pct}
