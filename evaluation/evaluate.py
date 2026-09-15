"""
Evaluation harness (PLAN.md sections 47, 47.1): generate audio for a fixed
evaluation set from base S1-mini and each experiment checkpoint, then
record per-sentence scores against a fixed rubric.

Scoring itself is done by human listeners (per section 47.1: at least two,
independently) -- this script produces the audio samples and a scoring
template; it does not auto-score audio quality.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

RUBRIC_CATEGORIES = [
    "pronunciation",           # target >= 4.0 / 5
    "naturalness",             # target >= 3.8 / 5
    "english_preservation",    # target >= 3.8 / 5 (section 48)
    "no_severe_artifacts",     # boolean: 1 = clean, 0 = clipping/dropped/garbled
]


def load_eval_sentences(path: str) -> List[dict]:
    """
    Fixed evaluation set (PLAN.md section 47.1): held-out sentence IDs,
    excluded from train/validation, covering short/long/code-switch/
    numbers-currency-dates/punctuation categories. Format:

        [{"id": "eval_001", "text": "...", "category": "short"}, ...]

    Frozen once created -- do not regenerate between experiments.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_scoring_template(eval_sentences: List[dict], experiment_ids: List[str], out_csv: str) -> None:
    """
    One row per (sentence, experiment, listener) so scores are traceable
    per-sentence, not just averaged (section 47.1). Listeners fill in the
    rubric columns after listening to the generated sample referenced by
    (sentence id, experiment id).
    """
    fieldnames = ["sentence_id", "category", "experiment_id", "listener"] + RUBRIC_CATEGORIES + ["notes"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sentence in eval_sentences:
            for experiment_id in experiment_ids:
                for listener in ("listener_1", "listener_2"):
                    writer.writerow(
                        {
                            "sentence_id": sentence["id"],
                            "category": sentence.get("category", ""),
                            "experiment_id": experiment_id,
                            "listener": listener,
                            **{c: "" for c in RUBRIC_CATEGORIES},
                            "notes": "",
                        }
                    )


def summarize_scores(scored_csv: str) -> dict:
    """
    Aggregate a filled-in scoring CSV into per-experiment averages, and
    flag any experiment failing a release gate (section 47.1 thresholds).
    """
    gates = {"pronunciation": 4.0, "naturalness": 3.8, "english_preservation": 3.8}

    sums: dict = {}
    counts: dict = {}
    artifact_flags: dict = {}

    with open(scored_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exp = row["experiment_id"]
            sums.setdefault(exp, {c: 0.0 for c in RUBRIC_CATEGORIES if c != "no_severe_artifacts"})
            counts.setdefault(exp, 0)
            artifact_flags.setdefault(exp, 0)

            if not row.get("pronunciation"):
                continue  # unscored row, skip

            for cat in RUBRIC_CATEGORIES:
                if cat == "no_severe_artifacts":
                    if row[cat] == "0":
                        artifact_flags[exp] += 1
                    continue
                sums[exp][cat] += float(row[cat])
            counts[exp] += 1

    results = {}
    for exp, cat_sums in sums.items():
        n = max(counts[exp], 1)
        averages = {cat: total / n for cat, total in cat_sums.items()}
        failed_gates = [cat for cat, threshold in gates.items() if averages.get(cat, 0) < threshold]
        results[exp] = {
            "n_scored": counts[exp],
            "averages": averages,
            "severe_artifact_count": artifact_flags[exp],
            "failed_gates": failed_gates,
            "passes_release_gate": not failed_gates and artifact_flags[exp] == 0,
        }
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_template = sub.add_parser("make-template")
    p_template.add_argument("--eval-sentences", required=True)
    p_template.add_argument("--experiments", nargs="+", required=True)
    p_template.add_argument("--out", required=True)

    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--scored-csv", required=True)

    args = parser.parse_args()

    if args.command == "make-template":
        sentences = load_eval_sentences(args.eval_sentences)
        generate_scoring_template(sentences, args.experiments, args.out)
        print(f"Wrote scoring template -> {args.out}")
    elif args.command == "summarize":
        results = summarize_scores(args.scored_csv)
        print(json.dumps(results, indent=2))
