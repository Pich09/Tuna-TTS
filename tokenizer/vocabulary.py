"""
Extended vocabulary bookkeeping (PLAN.md section 5).

    0 ─────────────── S1_VOCAB_SIZE - 1     original S1 text/control tokens
    S1_VOCAB_SIZE ─── S1_VOCAB_SIZE+7999     Khmer tokenizer tokens

This module owns exactly one thing: the arithmetic mapping between Khmer
tokenizer IDs and extended S1 IDs, plus range checks. It has no torch or
sentencepiece dependency, so it is fully unit-testable on its own
(see scripts/test_tokenizer.py).
"""

from __future__ import annotations

from dataclasses import dataclass


class VocabularyRangeError(ValueError):
    """Raised when an ID falls outside its expected range."""


@dataclass(frozen=True)
class ExtendedVocabulary:
    s1_vocab_size: int
    khmer_vocab_size: int = 8000

    @property
    def total_vocab_size(self) -> int:
        return self.s1_vocab_size + self.khmer_vocab_size

    def khmer_to_extended(self, khmer_id: int) -> int:
        """Map a raw Khmer-tokenizer ID (0..khmer_vocab_size-1) to its extended S1 ID."""
        if not (0 <= khmer_id < self.khmer_vocab_size):
            raise VocabularyRangeError(
                f"khmer_id {khmer_id} out of range [0, {self.khmer_vocab_size})"
            )
        return self.s1_vocab_size + khmer_id

    def extended_to_khmer(self, extended_id: int) -> int:
        """Map an extended S1 ID back to its raw Khmer-tokenizer ID. Raises if not a Khmer ID."""
        if not self.is_khmer_id(extended_id):
            raise VocabularyRangeError(
                f"extended_id {extended_id} is not a Khmer-range ID "
                f"(Khmer range starts at {self.s1_vocab_size})"
            )
        khmer_id = extended_id - self.s1_vocab_size
        if khmer_id >= self.khmer_vocab_size:
            raise VocabularyRangeError(
                f"extended_id {extended_id} exceeds total vocab size {self.total_vocab_size}"
            )
        return khmer_id

    def is_khmer_id(self, extended_id: int) -> bool:
        return extended_id >= self.s1_vocab_size

    def is_original_s1_id(self, extended_id: int) -> bool:
        return 0 <= extended_id < self.s1_vocab_size

    def validate_no_collision(self) -> None:
        """
        Sanity check referenced in PLAN.md section 22: the Khmer range must
        start exactly at s1_vocab_size and never overlap [0, s1_vocab_size).
        Trivially true given how khmer_to_extended is defined, but kept as
        an explicit, callable assertion so dataset validation scripts have
        something concrete to invoke and log.
        """
        first_khmer_extended_id = self.khmer_to_extended(0)
        if first_khmer_extended_id != self.s1_vocab_size:
            raise VocabularyRangeError("Khmer range does not start at s1_vocab_size")
        if self.is_original_s1_id(first_khmer_extended_id):
            raise VocabularyRangeError("Khmer ID collides with original S1 vocabulary")


if __name__ == "__main__":
    vocab = ExtendedVocabulary(s1_vocab_size=32000, khmer_vocab_size=8000)
    vocab.validate_no_collision()
    assert vocab.khmer_to_extended(0) == 32000
    assert vocab.khmer_to_extended(7999) == 39999
    assert vocab.extended_to_khmer(32000) == 0
    assert vocab.extended_to_khmer(39999) == 7999
    assert vocab.is_khmer_id(32000) and not vocab.is_khmer_id(31999)
    assert vocab.total_vocab_size == 40000
    try:
        vocab.khmer_to_extended(8000)
        raise AssertionError("expected VocabularyRangeError")
    except VocabularyRangeError:
        pass
    print("vocabulary.py self-test OK")
