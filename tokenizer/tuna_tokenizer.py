"""
TunaTokenizer (PLAN.md section 6): the adapter that sits between raw text
and the extended S1 vocabulary.

Responsibilities (PLAN.md section 6):
    1. Detect/segment Khmer text.
    2. Tokenize Khmer using Panhapich/khmer-sp-8k.
    3. Preserve supported English/S1 tokens where appropriate.
    4. Convert Khmer IDs into extended S1 IDs.
    5. Handle BOS/EOS/PAD/MASK/control tokens correctly.
    6. Produce the exact input format expected by S1-mini.

Architecture note (PLAN.md section 61.3): the S1-mini base tokenizer
(`fish_speech.tokenizer.FishTokenizer`) is tiktoken/HF-based, not
SentencePiece. `Panhapich/khmer-sp-8k` is a *separate* SentencePiece
model used only for the Khmer-script portions of the text. This module
segments text into Khmer-script runs vs. everything else, tokenizes each
run with the appropriate tokenizer, and stitches the resulting IDs back
together in original order.

Per section 6 item 5: the Khmer tokenizer's own reserved IDs
(PAD=0, UNK=1, BOS=2, EOS=3, MASK=4) are control tokens, not vocabulary
content. They are stripped out of any Khmer-encoded chunk rather than
blindly offset into the extended range -- BOS/EOS/PAD for the *sequence*
come from the base S1 tokenizer instead, so the model sees a single
consistent set of control tokens.
"""

from __future__ import annotations

import inspect
import re
from functools import lru_cache
from typing import List, Optional, Protocol, Sequence, Tuple

from tokenizer.vocabulary import ExtendedVocabulary


@lru_cache(maxsize=None)
def _base_encode_takes_allowed_special(base_tokenizer_type) -> bool:
    """
    fish_speech.tokenizer.FishTokenizer.encode's signature has changed
    across fish-speech versions (verified from source): the commit pinned
    for S1-mini compatibility (fishaudio/fish-speech@d3df505) takes
    `allowed_special`, while a later rewrite takes `add_special_tokens`
    instead. Detect which one is actually installed rather than hardcoding
    either, so this module keeps working across both.
    """
    try:
        params = inspect.signature(base_tokenizer_type.encode).parameters
    except (TypeError, ValueError):
        return False
    return "allowed_special" in params

# Khmer Unicode block (U+1780-U+17FF) + Khmer Symbols (U+19E0-U+19FF).
_KHMER_RUN_RE = re.compile(r"[ក-៿᧠-᧿]+")

# Panhapich/khmer-sp-8k reserved IDs (PLAN.md section 4) -- stripped, not offset.
KHMER_RESERVED_IDS = frozenset({0, 1, 2, 3, 4})  # PAD, UNK, BOS, EOS, MASK


class BaseTextTokenizer(Protocol):
    """Structural type for fish_speech.tokenizer.FishTokenizer (or a stand-in)."""

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]: ...
    def decode(self, tokens, **kwargs) -> str: ...

    @property
    def pad_token_id(self) -> Optional[int]: ...
    @property
    def eos_token_id(self) -> Optional[int]: ...
    @property
    def vocab_size(self) -> int: ...


class KhmerSentencePieceTokenizer(Protocol):
    """Structural type for sentencepiece.SentencePieceProcessor."""

    def encode(self, text: str, out_type=int) -> List[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


def segment_khmer_runs(text: str) -> List[Tuple[str, str]]:
    """
    Split text into ('khmer', chunk) / ('other', chunk) runs, preserving order
    and every character (whitespace and punctuation stay in 'other' chunks).
    """
    segments: List[Tuple[str, str]] = []
    last = 0
    for m in _KHMER_RUN_RE.finditer(text):
        if m.start() > last:
            segments.append(("other", text[last : m.start()]))
        segments.append(("khmer", m.group(0)))
        last = m.end()
    if last < len(text):
        segments.append(("other", text[last:]))
    return segments


class TunaTokenizer:
    """
    Drop-in replacement for fish_speech.tokenizer.FishTokenizer wherever
    fish-speech's own training code expects one -- notably
    fish_speech.content_sequence.ContentSequence.encode(tokenizer=...),
    which is what actually builds training examples (system/user/assistant
    text scaffolding + interleaved VQ audio codes, per PLAN.md section 61.5).
    That code calls tokenizer.encode(...), tokenizer.get_token_id(...), and
    reads tokenizer.semantic_begin_id directly. Rather than re-implement
    ContentSequence's packing here, TunaTokenizer only overrides encode()/
    decode() (to route Khmer-script runs through the extended vocabulary)
    and transparently delegates every other attribute (get_token_id,
    semantic_begin_id, semantic_end_id, pad_token_id, ...) to the wrapped
    base FishTokenizer via __getattr__, so existing fish-speech dataset code
    can use a TunaTokenizer instance without modification.
    """

    def __init__(
        self,
        base_tokenizer: BaseTextTokenizer,
        khmer_sp: KhmerSentencePieceTokenizer,
        vocabulary: ExtendedVocabulary,
    ):
        self.base = base_tokenizer
        self.khmer_sp = khmer_sp
        self.vocab = vocabulary

    def __getattr__(self, name: str):
        # Only reached for attributes not found on TunaTokenizer itself
        # (encode/decode/validate_ids/base/khmer_sp/vocab are all defined
        # above and never hit this). Delegates e.g. get_token_id,
        # semantic_begin_id, semantic_end_id, pad_token_id, eos_token_id.
        return getattr(self.base, name)

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        """
        Signature matches fish_speech.tokenizer.FishTokenizer.encode exactly
        (text, add_special_tokens=..., **kwargs) so this is call-compatible
        with fish_speech.content_sequence.ContentSequence.encode(tokenizer=...),
        which calls `tokenizer.encode(part.text, add_special_tokens=False)`
        for every TextPart -- including the ones this project's dataset code
        builds from Khmer text (PLAN.md section 61.5). Sequence-level BOS/EOS
        and speaker/modality control tokens are added by ContentSequence
        itself, not here; `add_special_tokens` is accepted for interface
        compatibility and currently has no effect since Khmer-run tokens
        never carry S1 special tokens (section 6 item 5) and base-tokenizer
        chunks are always encoded with add_special_tokens=False.
        """
        ids: List[int] = []

        for kind, chunk in segment_khmer_runs(text):
            if chunk == "":
                continue
            if kind == "khmer":
                khmer_ids = self.khmer_sp.encode(chunk, out_type=int)
                ids.extend(
                    self.vocab.khmer_to_extended(kid)
                    for kid in khmer_ids
                    if kid not in KHMER_RESERVED_IDS
                )
            else:
                # fish_speech.tokenizer.FishTokenizer.encode (as actually
                # installed, verified from source) takes `allowed_special`,
                # not `add_special_tokens` -- allowed_special=False matches
                # this method's documented intent (no special tokens parsed
                # out of ordinary text chunks) despite the kwarg name.
                if _base_encode_takes_allowed_special(type(self.base)):
                    ids.extend(self.base.encode(chunk, allowed_special=False))
                else:
                    ids.extend(self.base.encode(chunk, add_special_tokens=False))

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a mixed sequence of original-S1 and extended-Khmer IDs back to text."""
        pieces: List[str] = []
        run: List[int] = []
        run_is_khmer: Optional[bool] = None

        def flush():
            if not run:
                return
            if run_is_khmer:
                khmer_ids = [self.vocab.extended_to_khmer(i) for i in run]
                pieces.append(self.khmer_sp.decode(khmer_ids))
            else:
                pieces.append(self.base.decode(list(run)))

        for token_id in ids:
            is_khmer = self.vocab.is_khmer_id(token_id)
            if run_is_khmer is not None and is_khmer != run_is_khmer:
                flush()
                run = []
            run.append(token_id)
            run_is_khmer = is_khmer
        flush()

        return "".join(pieces)

    def validate_ids(self, ids: Sequence[int]) -> None:
        """Raise if any ID falls outside [0, total_vocab_size) -- PLAN.md section 22/42."""
        for token_id in ids:
            if not (0 <= token_id < self.vocab.total_vocab_size):
                raise ValueError(
                    f"token id {token_id} out of range [0, {self.vocab.total_vocab_size})"
                )
