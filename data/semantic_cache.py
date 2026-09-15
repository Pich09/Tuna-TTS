"""
Semantic target cache (PLAN.md section 21.2): precompute audio-codec
semantic tokens once, offline, keyed by clip_id + codec_revision, rather
than recomputing them inside the training loop.

Layout on disk:

    semantic_cache/
    ├── <clip_id>.npy      (or .pt)
    └── index.json         (clip_id -> {text, duration, sample_rate, split, codec_revision})

This module only manages the cache's bookkeeping (index, revision
matching, read/write paths). Actually running the S1 audio codec over
raw audio is done by a separate precompute script
(scripts/precompute_semantic_cache.py) that imports the real codec from
fish-speech; keeping that heavy step out of this module means the index
logic here is testable without torch/fish-speech installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


class CodecRevisionMismatchError(RuntimeError):
    """Raised when the on-disk cache was built with a different codec than
    the one currently configured -- PLAN.md section 21.2 rule 2: invalidate
    wholesale rather than silently mixing revisions."""


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    text: str
    duration_seconds: float
    sample_rate: int
    split: str  # "train" | "validation"
    codec_revision: str


class SemanticCacheIndex:
    def __init__(self, cache_dir: str, codec_revision: str):
        self.cache_dir = Path(cache_dir)
        self.codec_revision = codec_revision
        self.index_path = self.cache_dir / "index.json"
        self._records: dict[str, ClipRecord] = {}
        if self.index_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.index_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for clip_id, fields in raw.items():
            self._records[clip_id] = ClipRecord(clip_id=clip_id, **fields)

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(".json.tmp")
        payload = {cid: {k: v for k, v in asdict(r).items() if k != "clip_id"} for cid, r in self._records.items()}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.index_path)

    def add(self, record: ClipRecord) -> None:
        if record.codec_revision != self.codec_revision:
            raise CodecRevisionMismatchError(
                f"record codec_revision={record.codec_revision!r} != "
                f"index codec_revision={self.codec_revision!r}"
            )
        self._records[record.clip_id] = record

    def get(self, clip_id: str) -> Optional[ClipRecord]:
        record = self._records.get(clip_id)
        if record is not None and record.codec_revision != self.codec_revision:
            raise CodecRevisionMismatchError(
                f"cached record for {clip_id!r} was built with codec_revision="
                f"{record.codec_revision!r}, but this index expects {self.codec_revision!r}. "
                "The cache must be regenerated wholesale (section 21.2 rule 2), not patched."
            )
        return record

    def has(self, clip_id: str) -> bool:
        return self.get(clip_id) is not None

    def tensor_path(self, clip_id: str) -> Path:
        return self.cache_dir / f"{clip_id}.npy"

    def split_ids(self, split: str) -> list:
        return sorted(cid for cid, r in self._records.items() if r.split == split)

    def __len__(self) -> int:
        return len(self._records)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        idx = SemanticCacheIndex(tmp, codec_revision="v1")
        idx.add(ClipRecord("clip_0001", "ខ្ញុំសុខសប្បាយទេ", 3.2, 24000, "train", "v1"))
        idx.save()

        idx2 = SemanticCacheIndex(tmp, codec_revision="v1")
        assert len(idx2) == 1
        assert idx2.has("clip_0001")
        assert idx2.split_ids("train") == ["clip_0001"]

        # Wrong-revision index should refuse to read the same clip.
        idx3 = SemanticCacheIndex(tmp, codec_revision="v2")
        try:
            idx3.get("clip_0001")
            raise AssertionError("expected CodecRevisionMismatchError")
        except CodecRevisionMismatchError:
            pass

        print("semantic_cache.py self-test OK")
