"""
Batch collation. Each dataset item already carries packed `tokens`/`labels`
tensors of shape [num_codebooks+1, T] (see data/dataset.py, which delegates
packing to fish-speech's own ContentSequence-based packer -- PLAN.md
section 61.5). This collator only pads the T dimension to a common length
across the batch and stacks, matching
fish_speech.datasets.semantic.InterleaveDataset.batchify's padding
convention (pad tokens with 0, pad labels with -100).

The length-computation helper has no torch dependency and is covered by
the __main__ self-test below; the actual padding/stacking requires torch.
"""

from __future__ import annotations

from typing import List

CODEBOOK_PAD_TOKEN_ID = 0
IGNORE_LABEL_ID = -100


def compute_padded_length(lengths: List[int], multiple_of: int = 1) -> int:
    max_len = max(lengths, default=0)
    if multiple_of > 1 and max_len % multiple_of != 0:
        max_len += multiple_of - (max_len % multiple_of)
    return max_len


class TunaCollator:
    """Callable passed as `collate_fn` to a torch DataLoader."""

    def __init__(self, multiple_of: int = 8):
        self.multiple_of = multiple_of

    def __call__(self, batch: List[dict]) -> dict:
        import torch
        import torch.nn.functional as F

        lengths = [item["tokens"].shape[1] for item in batch]
        pad_len = compute_padded_length(lengths, multiple_of=self.multiple_of)

        padded_tokens, padded_labels, attention_masks = [], [], []
        for item in batch:
            tokens, labels = item["tokens"], item["labels"]
            length = tokens.shape[1]
            pad_amount = pad_len - length

            padded_tokens.append(F.pad(tokens, (0, pad_amount), value=CODEBOOK_PAD_TOKEN_ID))
            padded_labels.append(F.pad(labels, (0, pad_amount), value=IGNORE_LABEL_ID))
            attention_masks.append(torch.tensor([1] * length + [0] * pad_amount, dtype=torch.long))

        return {
            "clip_ids": [item["clip_id"] for item in batch],
            "input_ids": torch.stack(padded_tokens, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "speakers": [item.get("speaker") for item in batch],
        }


if __name__ == "__main__":
    assert compute_padded_length([3, 2, 4], multiple_of=1) == 4
    assert compute_padded_length([3, 2, 4], multiple_of=8) == 8
    assert compute_padded_length([], multiple_of=1) == 0
    print("collator.py self-test OK")
