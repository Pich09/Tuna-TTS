"""
Phase 1 step (PLAN.md sections 9, 13, 55 items 1-5): inspect the actual
loaded S1-mini model before writing any adaptation code against it.

Prints: config, named_modules, named_parameters, weight-tying status,
and text-embedding shape -- everything section 9/13 ask a human to check
by hand, as one command.

Run (requires the real checkpoint + fish-speech installed):
    python scripts/inspect_model.py --model-path /path/to/openaudio-s1-mini
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local path to the downloaded S1-mini checkpoint dir")
    parser.add_argument("--max-modules", type=int, default=200)
    args = parser.parse_args()

    from model.model_builder import load_base_model  # noqa: E402
    from model.embedding_extension import is_tied  # noqa: E402

    model = load_base_model(args.model_path)

    print("=== config ===")
    for k, v in vars(model.config).items():
        print(f"  {k}: {v}")

    print("\n=== weight tying ===")
    tied = is_tied(model)
    print(f"  tie_word_embeddings: {tied}")
    if tied:
        print("  -> extending model.embeddings extends the output side too (section 61.1)")
    else:
        print("  -> model.output must be extended separately (see model/embedding_extension.py)")

    print("\n=== text embedding ===")
    print(f"  model.embeddings.weight.shape: {tuple(model.embeddings.weight.shape)}")
    if hasattr(model, "codebook_embeddings"):
        print(f"  model.codebook_embeddings.weight.shape: {tuple(model.codebook_embeddings.weight.shape)}")
        print("  (must remain untouched -- section 10)")

    print(f"\n=== named_modules (first {args.max_modules}) ===")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= args.max_modules:
            print(f"  ... truncated at {args.max_modules}")
            break
        print(f"  {name}: {type(module).__name__}")

    print("\n=== named_parameters (shapes only) ===")
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        print(f"  {name}: {tuple(param.shape)}")
    print(f"\nTotal parameters: {total:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
