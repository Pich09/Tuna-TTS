"""
Dataset pipeline (PLAN.md section 21):

    Load sample -> normalize text -> Khmer tokenizer -> extended S1 IDs ->
    audio preprocessing -> load cached semantic targets -> training batch

`Panhapich/khmer-tts-processed` ships already text-normalized (its own
page states its processing pipeline verbalizes numbers/currency/dates/
percentages -- PLAN.md section 20.1). `TunaTTSDataset` therefore does NOT
re-run `data/text_normalize.py` on manifest text by default
(`text_is_pre_normalized=True`) -- see PLAN.md section 21.1 for the
consequence this has for inference-time text, which still needs
normalization since user-typed input isn't pre-processed.

Label packing (PLAN.md section 61.5): fish-speech does not train on plain
`tokenizer.encode(text)` sequences. Training examples go through
`fish_speech.content_sequence.ContentSequence` with `TextPart`/`VQPart`
scaffolding (a "Speak out the provided text." instruction, `<|speaker:user|>`
/ `<|speaker:assistant|>` / `<|voice|>` control tokens, then the VQ codes),
producing a `[num_codebooks+1, T]` `(tokens, labels)` pair per example
(text/semantic-placeholder ids in row 0, codebook ids in rows 1..N, -100
where loss should be ignored) -- see
`fish_speech.datasets.semantic.AutoTextSemanticInstructionDataset.pack_sentences`,
verified from source.

Tuna-TTS reuses that packer as-is rather than reimplementing it: because
`TunaTokenizer` is a drop-in for `FishTokenizer` (delegates every attribute
`ContentSequence` reads -- `get_token_id`, `semantic_begin_id`, etc. --
via `__getattr__`, and matches its `encode(text, add_special_tokens=...)`
signature), passing a `TunaTokenizer` instance wherever fish-speech's own
dataset code expects a tokenizer makes Khmer-script text route through the
extended vocabulary with zero changes to fish-speech's packing logic.
"""

from __future__ import annotations

import json
from typing import Optional

from data.semantic_cache import SemanticCacheIndex
from data.text_normalize import normalize
from tokenizer.tuna_tokenizer import TunaTokenizer


class TunaTTSDataset:
    """
    A torch.utils.data.Dataset-compatible class (duck-typed: __len__ +
    __getitem__) kept free of a hard torch.utils.data.Dataset base class so
    its manifest/normalization logic is testable without torch. Label
    packing is delegated to fish-speech's own packer (see module docstring)
    inside `_pack`, which requires fish-speech + torch installed.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        tokenizer: TunaTokenizer,
        semantic_cache: SemanticCacheIndex,
        text_is_pre_normalized: bool = True,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.semantic_cache = semantic_cache
        self.text_is_pre_normalized = text_is_pre_normalized
        self.samples = self._load_manifest(manifest_path, split)

    @staticmethod
    def _load_manifest(manifest_path: str, split: str) -> list:
        """
        manifest.json format (one entry per clip, produced by
        scripts/make_splits.py from Panhapich/khmer-tts-processed):

            [{"clip_id": "...", "text": "...", "split": "train"|"validation",
              "speaker": "...", "audio_path": "..."}, ...]
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return [s for s in manifest if s["split"] == split]

    def __len__(self) -> int:
        return len(self.samples)

    def _normalized_and_validated_text(self, sample: dict) -> str:
        text = sample["text"]
        normalized_text = text if self.text_is_pre_normalized else normalize(text)
        # Cheap pre-flight check on the text-only encoding path (section 22);
        # the authoritative check happens on the fully packed sequence too.
        self.tokenizer.validate_ids(self.tokenizer.encode(normalized_text))
        return normalized_text

    def _pack(self, normalized_text: str, semantic_codes) -> dict:
        """Delegates to fish-speech's own ContentSequence-based packer."""
        from fish_speech.datasets.semantic import AutoTextSemanticInstructionDataset

        # pack_sentences is an instance method but only touches
        # self.tokenizer / self.num_codebooks, so a minimally-constructed
        # instance is sufficient here rather than building a full dataset.
        packer = AutoTextSemanticInstructionDataset.__new__(AutoTextSemanticInstructionDataset)
        packer.tokenizer = self.tokenizer
        packer.num_codebooks = None
        tokens, labels = packer.pack_sentences(
            sentences=[normalized_text],
            semantics=[semantic_codes],
        )
        return {"tokens": tokens, "labels": labels}

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        clip_id = sample["clip_id"]

        normalized_text = self._normalized_and_validated_text(sample)

        record = self.semantic_cache.get(clip_id)
        if record is None:
            raise KeyError(
                f"clip_id {clip_id!r} has no cached semantic target. "
                "Run scripts/precompute_semantic_cache.py before training "
                "(section 21.2) -- semantic targets are never computed inline."
            )

        import numpy as np

        semantic_codes = np.load(self.semantic_cache.tensor_path(clip_id))
        packed = self._pack(normalized_text, semantic_codes)

        return {
            "clip_id": clip_id,
            "text": sample["text"],
            "normalized_text": normalized_text,
            "tokens": packed["tokens"],
            "labels": packed["labels"],
            "speaker": sample.get("speaker"),
        }


def load_splits_metadata(splits_path: str) -> dict:
    """PLAN.md section 23: split assignment must be recorded, not regenerated."""
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)
