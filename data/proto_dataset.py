"""
Thin wrapper around fish-speech's own `.protos`-format dataset loader
(PLAN.md section 21.5), used instead of the manifest.json + semantic_cache
pipeline (data/dataset.py) when training data ships pre-packaged in
fish-speech's native format -- as `Panhapich/khmer-tts-processed` does.

Verified from source (fish_speech/datasets/protos/text-data.proto):

    message Sentence {
        repeated string texts = 1;       // raw text, tokenized at load time
        repeated Semantics semantics = 3; // precomputed VQ codebook indices
    }

`fish_speech.datasets.semantic.AutoTextSemanticInstructionIterableDataset`
already reads `.protos` files (via `read_pb_stream`, expanding directories
for `*.protos` recursively) and calls the same `pack_sentences` that
data/dataset.py's TunaTTSDataset._pack delegates to -- so passing a
`TunaTokenizer` instance as its `tokenizer=` makes Khmer text route
through the extended vocabulary with zero fish-speech code changes,
exactly like the manifest.json path.

This means for a `.protos`-packaged dataset, Tuna-TTS needs NO semantic
cache, NO codec precompute step (scripts/precompute_semantic_cache.py),
and NO manifest.json -- the audio codec was already run by whoever built
the `.protos` files. Use this module instead of data/dataset.py for such
a dataset; do not run both against the same data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from tokenizer.tuna_tokenizer import TunaTokenizer


def build_proto_dataset(
    proto_dirs: List[Union[str, Path]],
    tokenizer: TunaTokenizer,
    max_length: int = 400,
    causal: bool = True,
):
    """
    Returns a fish_speech.datasets.semantic.AutoTextSemanticInstructionIterableDataset
    reading from `proto_dirs` (directories containing *.protos files, or
    individual file paths), tokenizing text through `tokenizer`.

    `max_length` is NOT a token budget the underlying dataset enforces --
    verified from source (fish_speech/datasets/semantic.py): it only
    controls how many sentences get packed into one item, via
    `num_samples = max_length // 20` (a fixed 20-tokens/sentence estimate),
    and nothing downstream truncates the result if that estimate is wrong.
    Measured directly against this project's Khmer `.protos` shards:
    individual packed sentences run ~160-475 tokens (not ~20), so the
    library's own default (1024, i.e. ~51 sentences/item) produces
    50,000-60,000-token sequences -- 6-7x the S1-mini checkpoint's
    max_seq_len=8192, which crashes in the model's attention-mask code.
    max_length=400 (-> 20 sentences/item) was tuned against that same
    measurement to land at a ~5,000-6,500 token median/p90 with real data,
    comfortably under 8192. training/trainer.py's `_protos_collate` still
    hard-truncates to max_seq_len as a backstop for the occasional outlier,
    since this is a statistical tuning, not a guarantee.

    Requires the `fish-speech` package installed (see requirements.txt).
    """
    from fish_speech.datasets.semantic import AutoTextSemanticInstructionIterableDataset

    return AutoTextSemanticInstructionIterableDataset(
        proto_files=[str(p) for p in proto_dirs],
        tokenizer=tokenizer,
        causal=causal,
        max_length=max_length,
        use_speaker=False,
    )


def count_protos_sentences(proto_dir: Union[str, Path]) -> int:
    """
    Quick sanity-check utility (PLAN.md section 22 equivalent for this data
    path): counts Sentence entries across all .protos files under
    `proto_dir` without going through the full tokenizer/dataset pipeline.
    Requires the `fish-speech` package (for the generated protobuf classes
    and read_pb_stream) but not torch.
    """
    from fish_speech.datasets.protos.text_data_stream import read_pb_stream

    proto_dir = Path(proto_dir)
    files = sorted(proto_dir.rglob("*.protos")) + sorted(proto_dir.rglob("*.proto"))
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            for text_data in read_pb_stream(fh):
                total += len(text_data.sentences)
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-dir", required=True)
    args = parser.parse_args()

    n = count_protos_sentences(args.proto_dir)
    print(f"{args.proto_dir}: {n} sentences across all .protos files")
