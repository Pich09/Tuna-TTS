"""
Phase 2 - Tokenizer Test (PLAN.md section 42).

Verifies: text -> tokenizer -> extended IDs -> (implicitly) model embedding,
with no out-of-range IDs, across Khmer-only, English-only, mixed, numbers,
punctuation, and long-sentence examples.

Two modes:
    --fake   (default) uses an in-process fake base tokenizer and a fake
             Khmer "sentencepiece" (character-level) so this test runs with
             zero external dependencies -- useful in CI / this sandbox,
             and to validate TunaTokenizer's segmentation/mapping logic in
             isolation from any real model weights.
    --real   loads fish_speech.tokenizer.FishTokenizer and a real
             sentencepiece.SentencePieceProcessor for Panhapich/khmer-sp-8k.
             Requires `pip install -r requirements.txt` and a Kaggle/GPU-ish
             environment with internet access to Hugging Face.

Run:
    python scripts/test_tokenizer.py
    python scripts/test_tokenizer.py --real --base-tokenizer-path <path> --khmer-sp-model <path>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.text_normalize import normalize  # noqa: E402
from tokenizer.tuna_tokenizer import TunaTokenizer, segment_khmer_runs  # noqa: E402
from tokenizer.vocabulary import ExtendedVocabulary  # noqa: E402

EXAMPLES = [
    "ខ្ញុំសុខសប្បាយទេ។",                                   # Khmer only
    "Hello, how are you today?",                             # English only
    "ខ្ញុំចង់ទៅ school នៅថ្ងៃស្អែក។",                        # Khmer + English
    "ខ្ញុំមាន $10 នៅឆ្នាំ 2026។",                            # numbers/currency (needs normalize first)
    "តើអ្នកសុខសប្បាយទេ? បាទ/ចាស ខ្ញុំសុខសប្បាយ។",           # punctuation
    "នេះជាប្រយោគវែងមួយដែលមានពាក្យច្រើន " * 5,               # long sentence
]


class FakeBaseTokenizer:
    """Stand-in for fish_speech.tokenizer.FishTokenizer: byte-level, deterministic."""

    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size
        self._pad_id = 0
        self._eos_id = 1

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs):
        # Deterministic stand-in: one ID per UTF-8 byte, offset past reserved ids.
        return [min(3 + b, self._vocab_size - 1) for b in text.encode("utf-8")]

    def decode(self, tokens, **kwargs) -> str:
        byte_vals = bytes(max(0, t - 3) & 0xFF for t in tokens)
        return byte_vals.decode("utf-8", errors="ignore")

    @property
    def pad_token_id(self):
        return self._pad_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def vocab_size(self):
        return self._vocab_size


class FakeKhmerSentencePiece:
    """Stand-in for Panhapich/khmer-sp-8k: one ID per unique character, deterministic."""

    def __init__(self, vocab_size: int = 8000):
        self._vocab_size = vocab_size
        self._char_to_id = {}
        self._id_to_char = {}
        self._next_id = 5  # 0-4 reserved: PAD, UNK, BOS, EOS, MASK

    def _id_for(self, ch: str) -> int:
        if ch not in self._char_to_id:
            if self._next_id >= self._vocab_size:
                return 1  # UNK
            self._char_to_id[ch] = self._next_id
            self._id_to_char[self._next_id] = ch
            self._next_id += 1
        return self._char_to_id[ch]

    def encode(self, text: str, out_type=int):
        return [self._id_for(ch) for ch in text]

    def decode(self, ids):
        return "".join(self._id_to_char.get(i, "") for i in ids)


def build_fake_tokenizer() -> TunaTokenizer:
    vocab = ExtendedVocabulary(s1_vocab_size=32000, khmer_vocab_size=8000)
    return TunaTokenizer(FakeBaseTokenizer(), FakeKhmerSentencePiece(), vocab)


def build_real_tokenizer(base_tokenizer_path: str, khmer_sp_model: str) -> TunaTokenizer:
    from fish_speech.tokenizer import FishTokenizer  # type: ignore
    import sentencepiece as spm  # type: ignore

    base = FishTokenizer(base_tokenizer_path)
    khmer_sp = spm.SentencePieceProcessor(model_file=khmer_sp_model)
    vocab = ExtendedVocabulary(s1_vocab_size=base.vocab_size, khmer_vocab_size=khmer_sp.vocab_size())
    return TunaTokenizer(base, khmer_sp, vocab)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--base-tokenizer-path", default=None)
    parser.add_argument("--khmer-sp-model", default=None)
    args = parser.parse_args()

    if args.real:
        if not (args.base_tokenizer_path and args.khmer_sp_model):
            parser.error("--real requires --base-tokenizer-path and --khmer-sp-model")
        tok = build_real_tokenizer(args.base_tokenizer_path, args.khmer_sp_model)
    else:
        tok = build_fake_tokenizer()

    failures = 0
    for raw in EXAMPLES:
        normalized = normalize(raw)
        segments = segment_khmer_runs(normalized)
        ids = tok.encode(normalized)
        try:
            tok.validate_ids(ids)
            status = "OK"
        except ValueError as e:
            status = f"FAIL ({e})"
            failures += 1

        decoded = tok.decode(ids)
        print(f"[{status}] {raw[:40]!r}")
        print(f"    normalized: {normalized[:60]!r}")
        print(f"    segments:   {[k for k, _ in segments]}")
        print(f"    n_ids:      {len(ids)}  (max_id={max(ids) if ids else None})")
        print(f"    decoded:    {decoded[:60]!r}")
        print()

    print(f"{len(EXAMPLES) - failures}/{len(EXAMPLES)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
