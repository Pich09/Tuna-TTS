"""
Generate one audio sample from a Tuna-TTS training checkpoint, for
qualitative progress tracking during a long run (PLAN.md section 47's
generate_samples.py, made concrete -- that file was a stub with actual
generation "intentionally not stubbed further").

Loads the exact same architecture training builds (model_builder.py:
S1-mini -> extend embedding -> apply LoRA), loads a checkpoint's
model_state_dict into it, generates semantic tokens for a fixed Khmer eval
sentence via fish_speech's own generate_long/generate (verified from
source: fish_speech/models/text2semantic/inference.py), then decodes those
codes to audio via the real S1-mini codec (same path as
scripts/codec_spotcheck.py).

Meant to be invoked automatically after each new checkpoint (see
scripts/watch_and_infer.sh) -- runs on its own device (default: cuda:1,
distinct from training's GPU 0) so it doesn't compete with an in-progress
training run for GPU memory.

Run:
    python scripts/run_inference_checkpoint.py \
        --checkpoint checkpoints/latest.pt \
        --base-model-path checkpoints/openaudio-s1-mini \
        --khmer-sp-model data/khmer_sp.model \
        --codec-checkpoint checkpoints/openaudio-s1-mini/codec.pth \
        --text "សួស្តី" --out-dir samples/EXP001 --device cuda:1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_EVAL_TEXT = "សួស្តី តើអ្នកសុខសប្បាយជាទេ?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a latest.pt/best.pt saved by training/checkpoint.py")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--khmer-sp-model", required=True)
    parser.add_argument("--khmer-vocab-size", type=int, default=8000)
    parser.add_argument("--codec-checkpoint", required=True)
    parser.add_argument("--codec-config-name", default="modded_dac_vq")
    parser.add_argument("--text", default=DEFAULT_EVAL_TEXT)
    parser.add_argument("--out-dir", default="samples")
    parser.add_argument("--step", type=int, default=None, help="label for the output filename; read from the checkpoint if omitted")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import sentencepiece as spm
    import soundfile as sf
    import torch

    from fish_speech.tokenizer import FishTokenizer
    from fish_speech.models.text2semantic.inference import decode_one_token_ar, generate_long
    from fish_speech.models.dac.inference import load_model as load_codec_model

    from tokenizer.tuna_tokenizer import TunaTokenizer
    from tokenizer.vocabulary import ExtendedVocabulary
    from model.model_builder import TunaModelConfig, build_tuna_model

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"Loading checkpoint metadata from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    step = args.step if args.step is not None else ckpt.get("global_step", "unknown")

    print("Building tokenizer...")
    base_tokenizer = FishTokenizer(f"{args.base_model_path}/tokenizer.tiktoken")
    khmer_sp = spm.SentencePieceProcessor(model_file=args.khmer_sp_model)
    vocabulary = ExtendedVocabulary(s1_vocab_size=base_tokenizer.vocab_size, khmer_vocab_size=args.khmer_vocab_size)
    tokenizer = TunaTokenizer(base_tokenizer, khmer_sp, vocabulary)

    print("Building model architecture (same as training) and loading checkpoint weights...")
    model_config = TunaModelConfig(base_model_path=args.base_model_path, khmer_vocab_size=args.khmer_vocab_size)
    model, _info = build_tuna_model(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.tokenizer = tokenizer
    model = model.to(device=device)
    model.eval()

    # Matches fish_speech.models.text2semantic.inference.init_model's
    # post-load setup (verified from source) -- decode_one_token_ar reads
    # these fixed tensors off the model rather than taking them as args.
    model.fixed_temperature = torch.tensor(args.temperature, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(args.top_p, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(args.repetition_penalty, device=device, dtype=torch.float)
    model._cache_setup_done = False

    print(f"Generating semantic tokens for: {args.text!r}")
    codes = None
    with torch.no_grad():
        for response in generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token_ar,
            text=args.text,
            num_samples=1,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
        ):
            if response.action == "sample":
                codes = response.codes

    if codes is None or codes.shape[1] == 0:
        print("[FAIL] no semantic codes were generated (empty output)")
        return 1
    print(f"Generated codes shape: {tuple(codes.shape)}")

    print(f"Loading codec ({args.codec_config_name}) from {args.codec_checkpoint} ...")
    codec = load_codec_model(args.codec_config_name, args.codec_checkpoint, device=device)

    indices = codes.to(device=device, dtype=torch.long)
    indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)
    with torch.no_grad():
        fake_audios, _audio_lengths = codec.decode(indices, indices_lens)
    audio = fake_audios[0, 0].float().cpu().numpy()

    finite = bool(np.isfinite(audio).all())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    duration = audio.shape[-1] / codec.sample_rate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{step}.wav"
    sf.write(out_path, audio, codec.sample_rate)

    status = "PASS" if (finite and rms > 1e-4) else "FAIL"
    print(
        f"[{status}] step={step}: {duration:.2f}s audio, finite={finite}, rms={rms:.5f} "
        f"-> saved {out_path}"
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
