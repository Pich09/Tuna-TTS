"""
Converts /home/helpdesk/Desktop/omni_asr_kh's fairseq2/wav2vec2-format ASR
manifest (manifest_v3/{split}.tsv + .wrd) into Tuna-TTS's manifest.json
schema (PLAN.md section 21/23).

Source format (verified against the real files, not guessed):
    manifest_v3/train.tsv:
        line 1: absolute root dir (e.g. .../data/manifest/audio)
        line N: "<relpath>\t<num_samples>"   -- relpath relative to that root
    manifest_v3/train.wrd:
        line N: transcript text for row N (same order as the .tsv, 1:1)

IMPORTANT (found during inspection, not assumed): the raw audio/ directory
tree contains many more files than the .tsv/.wrd manifests reference (a
resplit script moved rows between splits without deleting orphaned files).
This script joins strictly from the .tsv+.wrd pair -- never globs the
audio directories -- so only actually-labeled, actually-current clips end
up in the output manifest.

Text is NOT normalized here (source transcripts contain raw Khmer digits,
not spelled-out words) -- Tuna-TTS's data/dataset.py handles this via
text_is_pre_normalized=False for this manifest (see configs/tuna_v1_omni.yaml).

Speaker attribution: ddd_speaker_ids.json maps a per-row index (as a
string key) to a list of per-sample speaker-id strings, but only for the
DDD-Cambodia portion -- confirmed by checking a few keys, not assumed to
cover actableai__data-khmer rows too. Rows without a resolvable speaker
get speaker=null rather than a guessed value.

Run:
    python scripts/convert_omni_asr_manifest.py \
        --omni-root /home/helpdesk/Desktop/omni_asr_kh \
        --out data/manifest_omni.json \
        --sample-rate 16000
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# omni_asr_kh's split names -> Tuna-TTS's split names (PLAN.md section 23).
# "test" and "fleurs_ood" are ASR-eval-oriented splits, not used for TTS
# training here -- kept out of the training manifest entirely rather than
# silently folded into "validation", which would corrupt the fixed
# validation set's comparability across experiments (section 23).
SPLIT_MAP = {"train": "train", "dev": "validation"}
SKIPPED_SPLITS = ("test", "fleurs_ood")


def load_speaker_ids(omni_root: Path) -> dict:
    path = omni_root / "data" / "ddd_speaker_ids.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_tsv(tsv_path: Path):
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    root = Path(lines[0])
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        relpath, num_samples = line.split("\t")
        rows.append((relpath, int(num_samples)))
    return root, rows


def read_wrd(wrd_path: Path):
    with open(wrd_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def make_clip_id(split: str, relpath: str) -> str:
    # relpath looks like "train/DDD-Cambodia__khmer-speech-dataset__shard_01262__6.flac"
    stem = Path(relpath).stem
    return f"omni_{split}_{stem}"


def convert_split(
    split_name: str,
    audio_root: Path,
    tsv_path: Path,
    wrd_path: Path,
    speaker_ids: dict,
    sample_rate: int,
    out_split: str,
) -> list:
    header_root, rows = read_tsv(tsv_path)
    transcripts = read_wrd(wrd_path)
    if len(rows) != len(transcripts):
        raise ValueError(
            f"{split_name}: {len(rows)} tsv rows but {len(transcripts)} wrd lines -- "
            "these must be 1:1, refusing to guess an alignment"
        )

    entries = []
    missing_audio = 0
    for i, ((relpath, num_samples), text) in enumerate(zip(rows, transcripts)):
        audio_path = audio_root / relpath
        if not audio_path.is_file():
            missing_audio += 1
            continue

        speaker_list = speaker_ids.get(str(i))
        speaker = speaker_list[0] if speaker_list else None

        entries.append({
            "clip_id": make_clip_id(split_name, relpath),
            "text": text,
            "split": out_split,
            "audio_path": str(audio_path),
            "duration_seconds": round(num_samples / sample_rate, 3),
            "sample_rate": sample_rate,
            "speaker": speaker,
            "source": "omni_asr_kh",
        })

    if missing_audio:
        print(f"  [{split_name}] WARNING: {missing_audio} rows reference audio files that don't exist on disk, skipped")

    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    omni_root = Path(args.omni_root)
    manifest_dir = omni_root / "data" / "manifest_v3"
    audio_root = omni_root / "data" / "manifest" / "audio"
    speaker_ids = load_speaker_ids(omni_root)

    all_entries = []
    for split_name, out_split in SPLIT_MAP.items():
        tsv_path = manifest_dir / f"{split_name}.tsv"
        wrd_path = manifest_dir / f"{split_name}.wrd"
        print(f"Converting {split_name} -> {out_split}...")
        entries = convert_split(split_name, audio_root, tsv_path, wrd_path, speaker_ids, args.sample_rate, out_split)
        print(f"  {len(entries)} usable clips")
        all_entries.extend(entries)

    for skipped in SKIPPED_SPLITS:
        print(f"Skipping {skipped} (ASR-eval split, not used for TTS training -- see module docstring)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False)

    total_hours = sum(e["duration_seconds"] for e in all_entries) / 3600
    n_train = sum(1 for e in all_entries if e["split"] == "train")
    n_val = sum(1 for e in all_entries if e["split"] == "validation")
    n_with_speaker = sum(1 for e in all_entries if e["speaker"] is not None)
    print(f"\nWrote {len(all_entries)} clips ({total_hours:.1f} hours) -> {args.out}")
    print(f"  train:      {n_train}")
    print(f"  validation: {n_val}")
    print(f"  with known speaker: {n_with_speaker} ({100 * n_with_speaker / max(len(all_entries), 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
