"""
PLAN.md section 21.5 flagged risk: verify the `.protos` files' precomputed
`Semantics` codes are actually compatible with the current S1-mini
checkpoint's codec revision, by decoding a few real samples back to audio
and checking they're sane (finite, non-silent, plausible duration) before
trusting a full training run on them.

Decodes directly via fish_speech's own DAC codec inference code
(fish_speech.models.dac.inference.load_model / model.decode), reading raw
Sentence.semantics straight out of the .protos shards (bypassing the
text/tokenizer pipeline entirely -- this checks codec compatibility only).

Run:
    python scripts/codec_spotcheck.py --proto-dir data/protos/train \
        --codec-checkpoint checkpoints/openaudio-s1-mini/codec.pth \
        --num-samples 5 --out-dir /tmp/codec_spotcheck
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto-dir", required=True)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--config-name", default="modded_dac_vq")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--out-dir", default="/tmp/codec_spotcheck")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from fish_speech.datasets.protos.text_data_stream import read_pb_stream
    from fish_speech.models.dac.inference import load_model

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading codec ({args.config_name}) from {args.codec_checkpoint} on {device} ...")
    model = load_model(args.config_name, args.codec_checkpoint, device=device)
    print(f"Codec sample_rate: {model.sample_rate}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proto_dir = Path(args.proto_dir)
    files = sorted(proto_dir.rglob("*.protos"))
    if not files:
        print(f"[FAIL] no .protos files found under {proto_dir}")
        return 1

    collected = 0
    failures = 0
    with torch.no_grad():
        for f in files:
            with open(f, "rb") as fh:
                for text_data in read_pb_stream(fh):
                    for sentence in text_data.sentences:
                        if collected >= args.num_samples:
                            break
                        if not sentence.semantics:
                            continue

                        vq_codes = [list(codebook.values) for codebook in sentence.semantics]
                        num_codebooks = len(vq_codes)
                        t = len(vq_codes[0])
                        if any(len(c) != t for c in vq_codes):
                            print(f"[FAIL] sample {collected}: ragged codebook lengths, skipping")
                            failures += 1
                            continue

                        indices = torch.tensor(vq_codes, dtype=torch.long, device=device)
                        indices_lens = torch.tensor([t], device=device, dtype=torch.long)

                        try:
                            fake_audios, audio_lengths = model.decode(indices, indices_lens)
                        except Exception as e:
                            print(f"[FAIL] sample {collected}: decode raised {type(e).__name__}: {e}")
                            failures += 1
                            collected += 1
                            continue

                        audio = fake_audios[0, 0].float().cpu().numpy()
                        duration = audio.shape[-1] / model.sample_rate
                        text_preview = (sentence.texts[0][:60] if sentence.texts else "<no text>")

                        finite = np.isfinite(audio).all()
                        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                        silent = rms < 1e-4

                        status = "PASS" if (finite and not silent) else "FAIL"
                        if status == "FAIL":
                            failures += 1
                        print(
                            f"[{status}] sample {collected}: codebooks={num_codebooks} frames={t} "
                            f"-> {duration:.2f}s audio, finite={finite}, rms={rms:.5f}, "
                            f"text={text_preview!r}"
                        )

                        out_path = out_dir / f"sample_{collected:02d}.wav"
                        sf.write(out_path, audio, model.sample_rate)
                        print(f"       saved: {out_path}")

                        collected += 1
                    if collected >= args.num_samples:
                        break
            if collected >= args.num_samples:
                break

    print(f"\n{collected - failures}/{collected} samples decoded cleanly (finite, non-silent).")
    print(f"Listen to the .wav files in {out_dir} to confirm they sound like real speech.")
    return 0 if failures == 0 and collected > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
