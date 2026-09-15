"""
Dataset validation (PLAN.md section 22): run before training, check every
listed failure mode, and specifically confirm no Khmer token ID collides
with the original S1 vocabulary.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    # Allow `python3 data/validation.py` directly, not just `python3 -m data.validation`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.text_normalize import normalize
from tokenizer.tuna_tokenizer import TunaTokenizer

MIN_TEXT_CHARS = 1
MAX_TEXT_CHARS = 500
MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 30.0
EXPECTED_SAMPLE_RATE = 24000


@dataclass
class ValidationReport:
    total_samples: int = 0
    ok: int = 0
    issues: List[dict] = field(default_factory=list)

    def add_issue(self, clip_id: str, reason: str) -> None:
        self.issues.append({"clip_id": clip_id, "reason": reason})

    def summary(self) -> str:
        by_reason: dict = {}
        for issue in self.issues:
            by_reason[issue["reason"]] = by_reason.get(issue["reason"], 0) + 1
        lines = [f"{self.ok}/{self.total_samples} samples passed validation."]
        for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason}: {count}")
        return "\n".join(lines)


def validate_sample(
    sample: dict,
    tokenizer: TunaTokenizer,
    audio_root: Optional[str] = None,
    text_is_pre_normalized: bool = True,
    expected_sample_rate: Optional[int] = EXPECTED_SAMPLE_RATE,
) -> Optional[str]:
    """Returns a failure reason string, or None if the sample is valid."""
    text = sample.get("text")
    audio_path = sample.get("audio_path")

    if not audio_path:
        return "missing_audio_path"
    if audio_root and not os.path.isfile(os.path.join(audio_root, audio_path)):
        return "audio_file_not_found"

    if not text or not text.strip():
        return "empty_text"
    if len(text) > MAX_TEXT_CHARS:
        return "text_too_long"
    if len(text) < MIN_TEXT_CHARS:
        return "text_too_short"

    duration = sample.get("duration_seconds")
    if duration is not None:
        if duration < MIN_DURATION_SECONDS:
            return "audio_too_short"
        if duration > MAX_DURATION_SECONDS:
            return "audio_too_long"

    sample_rate = sample.get("sample_rate")
    # expected_sample_rate=None disables this check entirely -- multiple
    # legitimate sources exist at different native rates (e.g. 16kHz ASR
    # corpora vs. 24kHz TTS-processed corpora); the codec resamples during
    # precompute regardless (PLAN.md section 21.2), so this check only
    # exists to catch obviously wrong/corrupt metadata when a caller
    # explicitly cares about one specific rate.
    if sample_rate is not None and expected_sample_rate is not None and sample_rate != expected_sample_rate:
        return f"unexpected_sample_rate_{sample_rate}"

    try:
        normalized = text if text_is_pre_normalized else normalize(text)
    except Exception:
        return "normalize_failed"

    try:
        token_ids = tokenizer.encode(normalized)
    except Exception:
        return "tokenizer_failed"

    if not token_ids:
        return "empty_token_ids"

    try:
        tokenizer.validate_ids(token_ids)
    except ValueError:
        return "invalid_token_id_range"

    # Section 22: explicitly re-derive the Khmer/original split and confirm
    # no ID that should be Khmer-range lands in original-S1 range or vice versa.
    for token_id in token_ids:
        is_khmer = tokenizer.vocab.is_khmer_id(token_id)
        is_original = tokenizer.vocab.is_original_s1_id(token_id)
        if is_khmer == is_original:  # should always differ -- ranges are disjoint by construction
            return "vocabulary_collision"

    return None


def validate_dataset(
    samples: List[dict],
    tokenizer: TunaTokenizer,
    audio_root: Optional[str] = None,
    text_is_pre_normalized: bool = True,
    expected_sample_rate: Optional[int] = EXPECTED_SAMPLE_RATE,
) -> ValidationReport:
    report = ValidationReport(total_samples=len(samples))
    for sample in samples:
        reason = validate_sample(sample, tokenizer, audio_root, text_is_pre_normalized, expected_sample_rate)
        if reason is None:
            report.ok += 1
        else:
            report.add_issue(sample.get("clip_id", "<unknown>"), reason)
    return report


if __name__ == "__main__":
    from scripts.test_tokenizer import build_fake_tokenizer

    tok = build_fake_tokenizer()
    samples = [
        {"clip_id": "c1", "text": "ខ្ញុំសុខសប្បាយទេ។", "audio_path": "a.wav", "duration_seconds": 3.0, "sample_rate": 24000},
        {"clip_id": "c2", "text": "", "audio_path": "b.wav"},  # empty_text
        {"clip_id": "c3", "text": "hello", "audio_path": None},  # missing_audio_path
        {"clip_id": "c4", "text": "x" * 501, "audio_path": "d.wav"},  # text_too_long
        {"clip_id": "c5", "text": "ok", "audio_path": "e.wav", "duration_seconds": 0.05},  # audio_too_short
    ]
    report = validate_dataset(samples, tok)
    assert report.total_samples == 5
    assert report.ok == 1
    reasons = {i["clip_id"]: i["reason"] for i in report.issues}
    assert reasons["c2"] == "empty_text"
    assert reasons["c3"] == "missing_audio_path"
    assert reasons["c4"] == "text_too_long"
    assert reasons["c5"] == "audio_too_short"
    print(report.summary())
    print("validation.py self-test OK")
