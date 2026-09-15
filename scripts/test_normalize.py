"""
Phase-0 smoke test for data/text_normalize.py.

Run before Phase 2 (Tokenizer Test) in PLAN.md's implementation order —
normalization output is what the tokenizer actually sees, so it must be
verified independently first.

    python scripts/test_normalize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.text_normalize import normalize  # noqa: E402

CASES = [
    ("ខ្ញុំមាន $10 នៅឆ្នាំ 2026។", ["ដប់ដុល្លារ", "ពីរពាន់ម្ភៃប្រាំមួយ"]),
    ("ថ្ងៃទី 5 ខែមករា ឆ្នាំ 2026", ["ថ្ងៃទីប្រាំ", "ខែមករា", "ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ"]),
    ("05/01/2026", ["ថ្ងៃទីប្រាំ", "ខែមករា", "ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ"]),
    ("៛5000", ["ប្រាំពាន់រៀល"]),
    ("20000 រៀល", ["ពីរម៉ឺនរៀល"]),
    ("2026", ["ពីរពាន់ម្ភៃប្រាំមួយ"]),
    ("1000000", ["មួយលាន"]),
    ("0", ["សូន្យ"]),
]

KNOWN_LIMITATIONS = [
    # Digit groups that should be read digit-by-digit (phone numbers) are
    # currently read as cardinal quantities instead. Tracked, not fixed —
    # do not feed raw phone numbers through this normalizer yet.
    "phone numbers",
]


def main() -> int:
    failures = 0
    for raw, must_contain in CASES:
        result = normalize(raw)
        missing = [s for s in must_contain if s not in result]
        status = "OK" if not missing else "FAIL"
        if missing:
            failures += 1
        print(f"[{status}] {raw!r} -> {result!r}")
        if missing:
            print(f"        missing expected substrings: {missing}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed.")
    print(f"Known unhandled cases: {KNOWN_LIMITATIONS}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
