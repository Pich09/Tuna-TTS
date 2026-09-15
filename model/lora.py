"""
LoRA adaptation (PLAN.md sections 11-13) -- thin wrapper, not a
reimplementation.

fish_speech already ships a working, tested LoRA implementation for this
exact architecture: fish_speech/models/text2semantic/lora.py, using the
third-party `loralib` package. Verified target-module names
(PLAN.md section 61.4):

    attention:  wqkv, wo
    mlp:        w1, w2, w3
    embeddings: model.embeddings, model.codebook_embeddings
    output:     model.output (only present if tie_word_embeddings=False)

Tuna-TTS deliberately targets only ["attention", "mlp"]: the (extended)
text embedding is trained directly at full rank (sections 7-8), not
through a LoRA adapter, so "embeddings" and "output" are excluded here.
"""

from __future__ import annotations

from typing import List, Optional

# Excludes "embeddings" and "output" -- see module docstring and PLAN.md section 14.
DEFAULT_TARGET_MODULES = ["attention", "mlp"]


def apply_tuna_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.01,
    target_modules: Optional[List[str]] = None,
):
    """
    Attach LoRA adapters to `model` in place, using fish_speech's own
    `loralib`-based `lora.Linear` for the attention/mlp Linear layers --
    NOT fish_speech's `setup_lora` wholesale (verified from source: that
    function unconditionally re-wraps `model.embeddings`, `model.output`,
    and `model.codebook_embeddings` with freshly-initialized LoRA layers,
    which would discard the full-rank, carefully-initialized extended
    embedding this repo just built in extend_text_embedding(); it also
    calls `mark_only_lora_as_trainable`, which sets `requires_grad=False`
    on every non-"lora_" parameter including the just-unfrozen Khmer
    embedding rows -- since model_builder.py calls this *after*
    freeze_original_embedding_rows but *before* freeze_base_parameters,
    that would silently zero out the one thing this whole project trains).

    Tuna-TTS only ever targets ["attention", "mlp"] (see module docstring
    and PLAN.md section 14), so this replicates exactly the attention/mlp
    portion of fish_speech's own linears-wrapping loop and nothing else --
    embeddings/output are intentionally left as plain nn.Embedding/Linear,
    and their trainability is handled entirely by
    freeze_original_embedding_rows + freeze_base_parameters instead.
    """
    try:
        import loralib as lora
        import torch
    except ImportError as e:
        raise ImportError(
            "model/lora.py requires the `loralib` and `torch` packages. Install "
            "them per requirements.txt before calling apply_tuna_lora."
        ) from e

    modules = target_modules if target_modules is not None else list(DEFAULT_TARGET_MODULES)
    if set(modules) != set(DEFAULT_TARGET_MODULES):
        raise NotImplementedError(
            f"apply_tuna_lora only supports target_modules == {DEFAULT_TARGET_MODULES} "
            f"(got {modules}); embeddings/output are deliberately excluded, see "
            "module docstring and PLAN.md section 14."
        )

    linears = []
    for layer in model.layers:
        linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
        linears.extend(
            [(layer.feed_forward, "w1"), (layer.feed_forward, "w2"), (layer.feed_forward, "w3")]
        )
    if hasattr(model, "fast_layers"):
        for layer in model.fast_layers:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            linears.extend(
                [(layer.feed_forward, "w1"), (layer.feed_forward, "w2"), (layer.feed_forward, "w3")]
            )

    for module, name in linears:
        old_linear = getattr(module, name)
        new_linear = lora.Linear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            bias=old_linear.bias,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
        )
        with torch.no_grad():
            new_linear.weight.copy_(old_linear.weight)
            if old_linear.bias is not None:
                new_linear.bias.copy_(old_linear.bias)
        new_linear = new_linear.to(device=old_linear.weight.device, dtype=old_linear.weight.dtype)
        setattr(module, name, new_linear)

    return model


def get_merged_state_dict(model):
    """Delegates to fish_speech's own merge helper (used at export time, section 53)."""
    from fish_speech.models.text2semantic.lora import get_merged_state_dict as _merge

    return _merge(model)
