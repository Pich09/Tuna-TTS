"""
Deterministic train/validation split generation (PLAN.md section 23).

Splits are computed once from a stable hash of clip_id (not `random` with
a seed re-rolled per run), so re-running this script on an unchanged
manifest always reproduces the exact same split -- and the result is
still recorded in metadata.json as a belt-and-suspenders check.

Run:
    python scripts/make_splits.py --manifest raw_manifest.json --out manifest.json --val-fraction 0.02
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def clip_split(clip_id: str, val_fraction: float) -> str:
    """Deterministic split assignment: hash(clip_id) mod 10000 vs. threshold.
    Same clip_id always lands in the same split, independent of manifest
    ordering or how many times this is re-run (section 23)."""
    digest = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    threshold = int(val_fraction * 10000)
    return "validation" if bucket < threshold else "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input manifest: list of {clip_id, text, audio_path, ...}")
    parser.add_argument("--out", required=True, help="Output manifest with a 'split' field added")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        samples = json.load(f)

    counts = {"train": 0, "validation": 0}
    for sample in samples:
        split = clip_split(sample["clip_id"], args.val_fraction)
        sample["split"] = split
        counts[split] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples -> {args.out}")
    print(f"  train:      {counts['train']}")
    print(f"  validation: {counts['validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
