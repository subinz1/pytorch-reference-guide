"""
Batched inference utilities for variable-length sequences.

Demonstrates padding, attention masking, and efficient batched forward
passes for transformer models in a serving context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchedInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    original_lengths: list[int]


def pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: int = 0,
    max_length: Optional[int] = None,
) -> BatchedInput:
    """Pad variable-length sequences to uniform length with attention masks."""
    lengths = [seq.size(0) for seq in sequences]
    target_len = max_length or max(lengths)

    batch_size = len(sequences)
    padded = torch.full((batch_size, target_len), pad_value, dtype=sequences[0].dtype)
    mask = torch.zeros(batch_size, target_len, dtype=torch.bool)

    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        actual_len = min(length, target_len)
        padded[i, :actual_len] = seq[:actual_len]
        mask[i, :actual_len] = True

    return BatchedInput(
        input_ids=padded,
        attention_mask=mask,
        original_lengths=lengths,
    )


def unpad_outputs(
    batched_output: torch.Tensor,
    original_lengths: list[int],
) -> list[torch.Tensor]:
    """Extract per-sequence outputs from padded batch output."""
    results = []
    for i, length in enumerate(original_lengths):
        results.append(batched_output[i, :length])
    return results


class SimpleTransformerEncoder(nn.Module):
    """Minimal transformer encoder for demonstrating batched inference."""

    def __init__(self, vocab_size: int = 10000, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_encoding(positions)

        src_key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_proj(x)


def batched_forward(
    model: nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    max_batch_size: int = 32,
) -> list[torch.Tensor]:
    """Run batched inference over variable-length inputs with automatic chunking."""
    all_outputs = []

    for start in range(0, len(sequences), max_batch_size):
        chunk = sequences[start : start + max_batch_size]
        batch = pad_sequences(chunk)

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        output = output.cpu()
        all_outputs.extend(unpad_outputs(output, batch.original_lengths))

    return all_outputs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformerEncoder().to(device)
    model.eval()

    sequences = [
        torch.randint(0, 10000, (length,))
        for length in [12, 45, 8, 67, 23, 34, 5, 89, 16, 50]
    ]

    print(f"Input: {len(sequences)} sequences, lengths: {[s.size(0) for s in sequences]}")

    outputs = batched_forward(model, sequences, device, max_batch_size=4)

    print(f"Output: {len(outputs)} tensors")
    for i, out in enumerate(outputs):
        print(f"  seq[{i}]: input_len={sequences[i].size(0)}, output_shape={out.shape}")

    batch = pad_sequences(sequences)
    print(f"\nPadded batch shape: {batch.input_ids.shape}")
    print(f"Attention mask sum per row: {batch.attention_mask.sum(dim=1).tolist()}")


if __name__ == "__main__":
    main()
