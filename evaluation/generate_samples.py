"""
Generate audio samples for the fixed evaluation set (PLAN.md section 47)
from a given checkpoint (base S1-mini, or an experiment's best.pt/latest.pt).

Requires the real model + fish-speech's inference pipeline; not runnable
in this sandbox. Kept here as the wiring point so `evaluation/evaluate.py`
has something concrete to point listeners at.

Run:
    python evaluation/generate_samples.py \
        --checkpoint best.pt --eval-sentences eval_set.json \
        --out-dir samples/EXP001
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-sentences", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-model-path", required=True)
    args = parser.parse_args()

    from data.text_normalize import normalize  # noqa: E402

    with open(args.eval_sentences, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Text normalization runs identically here as at training time
    # (PLAN.md section 21.1) -- this is the one step this script can run
    # without the real model, so it's done eagerly and logged for review.
    normalized = [{"id": s["id"], "text": s["text"], "normalized": normalize(s["text"])} for s in sentences]
    with open(out_dir / "normalized_inputs.json", "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(normalized)} normalized inputs -> {out_dir / 'normalized_inputs.json'}\n"
        "Actual audio generation requires the loaded Tuna-TTS model + "
        "fish-speech's inference/generation utilities (real checkpoint + "
        "GPU) and is intentionally not stubbed further here -- fill in "
        "once scripts/test_model.py's forward pass is verified working "
        "against the real checkpoint (Phase 3)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
