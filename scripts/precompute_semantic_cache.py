"""
One-time, disk-bounded semantic-target precompute (PLAN.md section 21.2),
sharded to fit under a Kaggle-sized disk budget (data/sharding.py).

This is the ONLY phase of the pipeline that touches raw audio. It:
    1. Splits the manifest into shards that each fit under
       --disk-budget-gb (default 20GB, minus reserved space for the S1
       checkpoint + working overhead).
    2. For each not-yet-completed shard: downloads just that shard's
       audio files from the HF dataset repo, runs the real S1 audio
       codec to produce semantic tokens, writes them into the semantic
       cache, then DELETES the shard's raw audio immediately.
    3. Records progress (data/sharding.py's PrecomputeProgress) so a
       session that dies mid-shard resumes at the next shard rather than
       re-downloading completed ones.
    4. Optionally uploads the semantic cache (small -- see sharding.py's
       size estimate) to a HF dataset repo after each shard, so the
       *whole* precompute step never has to be repeated in a future
       session even if local disk is wiped between Kaggle sessions.

After this script completes, ordinary training sessions
(training/trainer.py) never need raw audio again -- only the semantic
cache + manifest + S1 checkpoint, all of which comfortably fit in 20GB.

Run:
    python scripts/precompute_semantic_cache.py \
        --manifest data/manifest.json \
        --semantic-cache-dir data/semantic_cache \
        --audio-repo Panhapich/khmer-tts-processed \
        --audio-local-dir data/_audio_tmp \
        --base-model-path /path/to/openaudio-s1-mini \
        --codec-revision v1 \
        --disk-budget-gb 20 \
        --hf-upload-repo Panhapich/Tuna-TTS-semantic-cache   # optional
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sharding import (  # noqa: E402
    PrecomputeProgress,
    estimate_audio_bytes,
    make_precompute_shards,
    reserve_bytes_for_fixed_assets,
)
from data.semantic_cache import ClipRecord, SemanticCacheIndex  # noqa: E402


def load_manifest(manifest_path: str) -> list:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_shard_audio(clip_ids, samples_by_id, audio_repo: str, local_dir: Path) -> float:
    """
    Downloads just this shard's audio files from the HF dataset repo.
    Returns the actual measured bytes/second observed, so later shards'
    budgeting can use a calibrated rate instead of the estimate in
    data/sharding.py (see that module's calibration note).
    """
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    total_seconds = 0.0

    for clip_id in clip_ids:
        sample = samples_by_id[clip_id]
        local_path = hf_hub_download(
            repo_id=audio_repo,
            repo_type="dataset",
            filename=sample["audio_path"],
            local_dir=str(local_dir),
        )
        total_bytes += Path(local_path).stat().st_size
        total_seconds += sample["duration_seconds"]

    measured_bytes_per_second = total_bytes / total_seconds if total_seconds else 0.0
    return measured_bytes_per_second


def load_codec(base_model_path: str, device: str, config_name: str = "modded_dac_vq"):
    """
    Loads the real S1 audio codec exactly as fish-speech's own
    tools/vqgan/extract_vq.py does (verified from source): compose the
    hydra config that ships in fish_speech/configs/modded_dac_vq.yaml,
    instantiate it, then load codec.pth's state dict -- stripping a
    "generator." prefix if present, matching a Lightning-style checkpoint.
    Reused as-is rather than reimplemented, per PLAN.md's own instruction
    to prefer the official implementation over guessing.
    """
    import torch
    from hydra import compose, initialize
    from hydra.utils import instantiate

    fish_speech_configs_path = _find_fish_speech_configs_dir()
    with initialize_config_dir_compat(fish_speech_configs_path):
        cfg = compose(config_name=config_name)
    model = instantiate(cfg)

    state_dict = torch.load(f"{base_model_path}/codec.pth", map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def _find_fish_speech_configs_dir() -> str:
    import fish_speech

    return str(Path(fish_speech.__file__).resolve().parent / "configs")


def initialize_config_dir_compat(config_dir: str):
    from hydra import initialize_config_dir

    return initialize_config_dir(version_base="1.3", config_dir=config_dir)


def run_codec_on_shard(
    clip_ids,
    samples_by_id,
    audio_local_dir: Path,
    codec,
    device: str,
    codec_revision: str,
    cache_index: SemanticCacheIndex,
    split_lookup: dict,
    batch_size: int = 16,
) -> None:
    """
    Runs `codec` over one shard's downloaded audio, batched, mirroring
    extract_vq.py's process_batch exactly: mono-mix, resample to the
    codec's native sample rate, pad to a common length, call
    codec.encode(audios, audio_lengths) -> (indices, feature_lengths),
    then slice each clip's output back down to its own length before
    saving -- padding must not leak into the cached tokens.
    """
    import numpy as np
    import torch
    import torchaudio

    for i in range(0, len(clip_ids), batch_size):
        batch_ids = clip_ids[i : i + batch_size]
        wavs, lengths = [], []

        for clip_id in batch_ids:
            sample = samples_by_id[clip_id]
            wav, sr = torchaudio.load(str(audio_local_dir / sample["audio_path"]))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]
            wavs.append(wav)
            lengths.append(len(wav))

        max_len = max(lengths)
        padded = [torch.nn.functional.pad(w, (0, max_len - len(w))) for w in wavs]
        audios = torch.stack(padded, dim=0)[:, None]
        audio_lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        with torch.inference_mode():
            indices, feature_lengths = codec.encode(audios, audio_lengths)

        outputs = indices.cpu().numpy()
        for clip_id, length, feature in zip(batch_ids, feature_lengths, outputs):
            sample = samples_by_id[clip_id]
            trimmed = feature[:, :length].astype(np.int16)  # (num_codebooks, T) -- see module docstring
            np.save(cache_index.tensor_path(clip_id), trimmed)

            cache_index.add(
                ClipRecord(
                    clip_id=clip_id,
                    text=sample["text"],
                    duration_seconds=sample["duration_seconds"],
                    sample_rate=codec.sample_rate,
                    split=split_lookup[clip_id],
                    codec_revision=codec_revision,
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--semantic-cache-dir", required=True)
    parser.add_argument("--audio-repo", required=True, help="HF dataset repo id, e.g. Panhapich/khmer-tts-processed")
    parser.add_argument("--audio-local-dir", required=True, help="Scratch dir for one shard's audio; deleted after each shard")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--codec-revision", required=True)
    parser.add_argument("--disk-budget-gb", type=float, default=20.0)
    parser.add_argument("--progress-file", default=None, help="Defaults to <semantic-cache-dir>/precompute_progress.json")
    parser.add_argument("--hf-upload-repo", default=None, help="Optional: push semantic cache here after each shard")
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    samples_by_id = {s["clip_id"]: s for s in samples}
    split_lookup = {s["clip_id"]: s["split"] for s in samples}

    disk_budget_bytes = int(args.disk_budget_gb * 1024**3)
    shard_budget_bytes = reserve_bytes_for_fixed_assets(disk_budget_bytes)
    shards = make_precompute_shards(samples, shard_budget_bytes)

    print(f"{len(samples)} clips -> {len(shards)} shard(s) under a {shard_budget_bytes / 1e9:.2f} GB per-shard budget")

    progress_path = args.progress_file or str(Path(args.semantic_cache_dir) / "precompute_progress.json")
    progress = PrecomputeProgress.load_or_new(progress_path)
    cache_index = SemanticCacheIndex(args.semantic_cache_dir, codec_revision=args.codec_revision)

    audio_local_dir = Path(args.audio_local_dir)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = load_codec(args.base_model_path, device)
    print(f"Loaded codec on {device}, sample_rate={codec.sample_rate}")

    for shard in shards:
        if progress.is_complete(shard.index):
            print(f"shard {shard.index}: already done, skipping")
            continue

        print(f"shard {shard.index}: downloading {len(shard.clip_ids)} clips "
              f"(~{shard.total_audio_bytes / 1e9:.2f} GB estimated)")
        measured_rate = download_shard_audio(shard.clip_ids, samples_by_id, args.audio_repo, audio_local_dir)
        print(f"shard {shard.index}: measured ~{measured_rate / 1e6:.2f} MB/s "
              "(compare against data/sharding.py's DEFAULT_AUDIO_BYTES_PER_SECOND estimate)")

        print(f"shard {shard.index}: running codec...")
        run_codec_on_shard(
            shard.clip_ids, samples_by_id, audio_local_dir,
            codec, device, args.codec_revision, cache_index, split_lookup,
        )
        cache_index.save()

        print(f"shard {shard.index}: deleting raw audio ({audio_local_dir})")
        shutil.rmtree(audio_local_dir, ignore_errors=True)

        if args.hf_upload_repo:
            print(f"shard {shard.index}: uploading semantic cache -> {args.hf_upload_repo}")
            from huggingface_hub import upload_folder

            upload_folder(
                repo_id=args.hf_upload_repo,
                repo_type="dataset",
                folder_path=args.semantic_cache_dir,
                commit_message=f"semantic cache: shard {shard.index} complete",
            )

        progress.mark_complete(shard.index)
        print(f"shard {shard.index}: done ({len(cache_index)} clips cached total)")

    print(f"\nPrecompute complete: {len(cache_index)} / {len(samples)} clips cached "
          f"in {args.semantic_cache_dir}. Raw audio no longer needed for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
