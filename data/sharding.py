"""
Disk-bounded sharding for the ONE-TIME semantic-target precompute phase
(PLAN.md section 21.2), not for the training loop itself.

Training never reads raw audio -- it only reads cached semantic tokens,
which are orders of magnitude smaller than the audio they were derived
from (see estimate_semantic_bytes vs estimate_audio_bytes below). So the
only phase that needs to fit under a disk budget like Kaggle's ~20GB is
precompute: download a batch of raw audio under budget, run the codec,
write semantic tokens, delete the raw audio, move to the next shard.

This module only computes shard boundaries and tracks precompute
progress (both pure Python, no torch/audio deps) so they're testable
here. The actual download + codec run is in
scripts/precompute_semantic_cache.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Assumes 24kHz mono 16-bit PCM WAV (matches the dataset's stated processing
# pipeline -- loudness-normalized, resampled -- PLAN.md training-data note).
# A small safety margin (1.05x) covers WAV headers and minor deviations.
DEFAULT_AUDIO_BYTES_PER_SECOND = 24_000 * 2 * 1.05

# Semantic token size estimate: num_codebooks+1 streams (PLAN.md section
# 61.1/61.2: codebook_size=160, num_codebooks=4, verified from source),
# 2 bytes/token (int16 is enough for codebook_size=160), at an assumed
# ~75 Hz frame rate. This is a planning estimate, not a measured constant
# -- scripts/precompute_semantic_cache.py should log the *actual* observed
# bytes/second once real codec output exists, and callers should prefer
# that measured value once available.
DEFAULT_SEMANTIC_TOKENS_HZ = 75
DEFAULT_SEMANTIC_BYTES_PER_TOKEN = 2
DEFAULT_NUM_CODEBOOK_STREAMS = 5  # num_codebooks (4) + 1 text/placeholder row


def estimate_audio_bytes(duration_seconds: float, bytes_per_second: float = DEFAULT_AUDIO_BYTES_PER_SECOND) -> int:
    return int(duration_seconds * bytes_per_second)


def estimate_semantic_bytes(
    duration_seconds: float,
    tokens_hz: float = DEFAULT_SEMANTIC_TOKENS_HZ,
    bytes_per_token: int = DEFAULT_SEMANTIC_BYTES_PER_TOKEN,
    num_streams: int = DEFAULT_NUM_CODEBOOK_STREAMS,
) -> int:
    return int(duration_seconds * tokens_hz * bytes_per_token * num_streams)


@dataclass(frozen=True)
class Shard:
    index: int
    clip_ids: List[str]
    total_audio_bytes: int


def _stable_order(samples: List[dict]) -> List[dict]:
    """Deterministic ordering independent of manifest file ordering, so
    shard boundaries don't shift if the manifest is regenerated/reordered
    (mirrors scripts/make_splits.py's hash-based determinism)."""
    return sorted(samples, key=lambda s: hashlib.sha256(s["clip_id"].encode("utf-8")).hexdigest())


def make_precompute_shards(
    samples: List[dict],
    budget_bytes: int,
    bytes_per_second: float = DEFAULT_AUDIO_BYTES_PER_SECOND,
) -> List[Shard]:
    """
    Greedily bins samples into shards whose total estimated raw-audio size
    stays under `budget_bytes`. `samples` entries need `clip_id` and
    `duration_seconds`. Leaves headroom for the caller to reserve some of
    the budget for the S1 checkpoint/working files -- pass a budget that
    already excludes that (see reserve_bytes_for_fixed_assets below).
    """
    ordered = _stable_order(samples)
    shards: List[Shard] = []
    current_ids: List[str] = []
    current_bytes = 0
    index = 0

    for sample in ordered:
        size = estimate_audio_bytes(sample["duration_seconds"], bytes_per_second)
        if size > budget_bytes:
            raise ValueError(
                f"single clip {sample['clip_id']!r} ({size} bytes) exceeds the "
                f"entire shard budget ({budget_bytes} bytes) -- increase the budget "
                "or exclude this clip."
            )
        if current_bytes + size > budget_bytes and current_ids:
            shards.append(Shard(index=index, clip_ids=current_ids, total_audio_bytes=current_bytes))
            index += 1
            current_ids = []
            current_bytes = 0
        current_ids.append(sample["clip_id"])
        current_bytes += size

    if current_ids:
        shards.append(Shard(index=index, clip_ids=current_ids, total_audio_bytes=current_bytes))

    return shards


def reserve_bytes_for_fixed_assets(
    total_budget_bytes: int,
    s1_checkpoint_bytes: int = 4 * 1024**3,  # model.pth + codec.pth, ~3.6GB, rounded up
    working_overhead_bytes: int = 2 * 1024**3,  # python env, torch, temp files, margin
) -> int:
    """Bytes actually available for one shard's raw audio, after reserving
    space for things that sit on disk the whole session (PLAN.md section 26.1
    already budgets session *time* for downloads; this budgets *space*)."""
    available = total_budget_bytes - s1_checkpoint_bytes - working_overhead_bytes
    if available <= 0:
        raise ValueError(
            f"total_budget_bytes={total_budget_bytes} leaves no room after reserving "
            f"{s1_checkpoint_bytes} (checkpoint) + {working_overhead_bytes} (overhead) -- "
            "increase disk budget or reduce reservations."
        )
    return available


@dataclass
class PrecomputeProgress:
    """
    Tracks which shards have been fully processed, so precompute can
    resume across Kaggle sessions the same way training resume does
    (PLAN.md section 35) -- one shard at a time, never re-downloading a
    shard whose semantic tokens are already cached and uploaded.
    """

    path: Path
    completed_shard_indices: set

    @classmethod
    def load_or_new(cls, path: str) -> "PrecomputeProgress":
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(path=p, completed_shard_indices=set(data.get("completed_shard_indices", [])))
        return cls(path=p, completed_shard_indices=set())

    def mark_complete(self, shard_index: int) -> None:
        self.completed_shard_indices.add(shard_index)
        self._save()

    def is_complete(self, shard_index: int) -> bool:
        return shard_index in self.completed_shard_indices

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"completed_shard_indices": sorted(self.completed_shard_indices)}, f, indent=2)
        os.replace(tmp_path, self.path)


if __name__ == "__main__":
    import tempfile

    # 128 hours (section 20 corrected numbers) across ~55k clips, ~8.3s avg.
    samples = [{"clip_id": f"clip_{i:06d}", "duration_seconds": 8.3} for i in range(55206)]
    total_duration_hours = sum(s["duration_seconds"] for s in samples) / 3600
    assert 120 < total_duration_hours < 135, total_duration_hours

    total_audio_bytes = sum(estimate_audio_bytes(s["duration_seconds"]) for s in samples)
    total_semantic_bytes = sum(estimate_semantic_bytes(s["duration_seconds"]) for s in samples)
    print(f"Estimated total raw audio:      {total_audio_bytes / 1e9:.2f} GB")
    print(f"Estimated total semantic cache: {total_semantic_bytes / 1e9:.2f} GB")
    assert total_semantic_bytes < total_audio_bytes / 20, "semantic cache should be far smaller than raw audio"

    kaggle_budget = 20 * 1024**3
    shard_budget = reserve_bytes_for_fixed_assets(kaggle_budget)
    print(f"Kaggle budget: {kaggle_budget / 1e9:.1f} GB -> per-shard audio budget: {shard_budget / 1e9:.2f} GB")

    shards = make_precompute_shards(samples, shard_budget)
    print(f"Dataset splits into {len(shards)} precompute shard(s) under that budget")
    for s in shards[:3]:
        assert s.total_audio_bytes <= shard_budget
        print(f"  shard {s.index}: {len(s.clip_ids)} clips, {s.total_audio_bytes / 1e9:.2f} GB")
    assert sum(len(s.clip_ids) for s in shards) == len(samples), "no clip should be dropped"

    with tempfile.TemporaryDirectory() as tmp:
        progress_path = os.path.join(tmp, "precompute_progress.json")
        progress = PrecomputeProgress.load_or_new(progress_path)
        assert not progress.is_complete(0)
        progress.mark_complete(0)

        progress2 = PrecomputeProgress.load_or_new(progress_path)
        assert progress2.is_complete(0)
        assert not progress2.is_complete(1)

    print("\nsharding.py self-test OK")
