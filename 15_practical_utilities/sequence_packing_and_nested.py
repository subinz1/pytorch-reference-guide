"""
Sequence Packing & Nested Tensors — Efficient Variable-Length Processing
=========================================================================
Learn RNN packing and the modern NestedTensor approach.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
    pack_sequence,
    unpad_sequence,
)

print("=" * 65)
print("1. THE PROBLEM: Variable-Length Sequences")
print("=" * 65)

# In NLP/audio, sequences have different lengths
seq1 = torch.randn(7, 16)   # 7 tokens, 16-dim features
seq2 = torch.randn(3, 16)   # 3 tokens
seq3 = torch.randn(5, 16)   # 5 tokens
sequences = [seq1, seq2, seq3]

print(f"Sequence lengths: {[s.size(0) for s in sequences]}")
print(f"Feature dim: {sequences[0].size(1)}")

print("\n" + "=" * 65)
print("2. pad_sequence — Pad to Equal Length")
print("=" * 65)

# Pad all sequences to the length of the longest
padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
print(f"Padded shape: {padded.shape}")  # (3, 7, 16)
print(f"  Batch size: {padded.shape[0]}")
print(f"  Max length: {padded.shape[1]}")
print(f"  Feature dim: {padded.shape[2]}")

# Problem: seq2 has 4 wasted timesteps of zeros!
# The RNN will process these pad tokens unnecessarily.

print("\n" + "=" * 65)
print("3. pack_padded_sequence — Efficient RNN Processing")
print("=" * 65)

lengths = torch.tensor([7, 3, 5])

# Pack the padded tensor
packed = pack_padded_sequence(
    padded,
    lengths,
    batch_first=True,
    enforce_sorted=False  # Don't require sorted lengths
)

print(f"PackedSequence:")
print(f"  data shape:       {packed.data.shape}")
print(f"  batch_sizes:      {packed.batch_sizes}")
print(f"  sorted_indices:   {packed.sorted_indices}")
print(f"  unsorted_indices: {packed.unsorted_indices}")

# batch_sizes tells the RNN how many sequences are active at each timestep
# e.g., [3, 3, 3, 2, 2, 1, 1] means:
#   timestep 0: 3 sequences active
#   timestep 3: 2 sequences active (seq2 ended)
#   timestep 5: 1 sequence active (seq3 ended)

total_without_packing = 3 * 7  # 21 steps
total_with_packing = 7 + 3 + 5  # 15 steps
print(f"\nComputation savings:")
print(f"  Without packing: {total_without_packing} RNN steps")
print(f"  With packing:    {total_with_packing} RNN steps")
print(f"  Saved:           {total_without_packing - total_with_packing} steps "
      f"({1 - total_with_packing/total_without_packing:.0%})")

print("\n" + "=" * 65)
print("4. COMPLETE RNN WORKFLOW WITH PACKING")
print("=" * 65)

# Full workflow: pad → pack → RNN → unpack

# Step 1: Create variable-length data
batch_sequences = [torch.randn(l, 32) for l in [10, 6, 8, 4, 9]]
batch_lengths = torch.tensor([10, 6, 8, 4, 9])

# Step 2: Pad
padded_batch = pad_sequence(batch_sequences, batch_first=True)
print(f"Padded: {padded_batch.shape}")

# Step 3: Pack
packed_batch = pack_padded_sequence(
    padded_batch, batch_lengths, batch_first=True, enforce_sorted=False
)

# Step 4: Feed to RNN
lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
packed_output, (h_n, c_n) = lstm(packed_batch)

print(f"LSTM hidden state: {h_n.shape}")   # (2, 5, 64) — 2 layers, 5 seqs, 64 hidden
print(f"LSTM cell state:   {c_n.shape}")

# Step 5: Unpack
output_padded, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
print(f"Unpacked output: {output_padded.shape}")  # (5, 10, 64)
print(f"Output lengths:  {output_lengths}")

# Step 6: Extract the LAST valid output for each sequence
# (useful for classification)
last_outputs = []
for i, length in enumerate(output_lengths):
    last_outputs.append(output_padded[i, length - 1, :])
last_outputs = torch.stack(last_outputs)
print(f"Last valid outputs: {last_outputs.shape}")  # (5, 64)

print("\n" + "=" * 65)
print("5. SHORTCUT: pack_sequence (Skip Padding)")
print("=" * 65)

# If you have a list of tensors, you can pack directly
seqs = [torch.randn(l, 16) for l in [5, 3, 7, 2]]
packed_direct = pack_sequence(seqs, enforce_sorted=False)
print(f"Directly packed from list: data shape = {packed_direct.data.shape}")

print("\n" + "=" * 65)
print("6. unpad_sequence — Get Individual Sequences Back")
print("=" * 65)

padded = pad_sequence(seqs, batch_first=True)
lengths = torch.tensor([5, 3, 7, 2])

# Unpad back to list of individual sequences
unpadded = unpad_sequence(padded, lengths, batch_first=True)
for i, s in enumerate(unpadded):
    print(f"  Sequence {i}: shape {s.shape}")

print("\n" + "=" * 65)
print("7. NESTED TENSORS (Modern Alternative)")
print("=" * 65)

# NestedTensor: a tensor that natively holds variable-length data
# No padding, no wasted computation

seqs = [torch.randn(3, 8), torch.randn(5, 8), torch.randn(2, 8)]

# Create nested tensor
nt = torch.nested.nested_tensor(seqs)
print(f"NestedTensor: {nt.size()}")
print(f"  Is nested: {nt.is_nested}")

# Convert to padded when needed
padded = torch.nested.to_padded_tensor(nt, padding=0.0)
print(f"  As padded: {padded.shape}")

# Convert regular tensor to nested
regular = torch.randn(3, 5, 8)
nt_from_regular = torch.nested.as_nested_tensor(regular)
print(f"  From regular tensor: {nt_from_regular.size()}")

print("\n" + "=" * 65)
print("8. NESTED TENSOR OPERATIONS")
print("=" * 65)

nt1 = torch.nested.nested_tensor([torch.randn(3, 4), torch.randn(5, 4)])
nt2 = torch.nested.nested_tensor([torch.randn(3, 4), torch.randn(5, 4)])

# Element-wise operations work
nt_sum = nt1 + nt2
nt_relu = torch.relu(nt1)
print(f"Addition: works")
print(f"ReLU:     works")

# Linear layer works on nested tensors
linear = nn.Linear(4, 8)
output = linear(nt1)
padded_out = torch.nested.to_padded_tensor(output, padding=0.0)
print(f"Linear on nested: output padded shape = {padded_out.shape}")

print("\n" + "=" * 65)
print("9. PERFORMANCE COMPARISON: Padded vs Packed")
print("=" * 65)

import time

# Create dataset with highly variable lengths
torch.manual_seed(42)
n_sequences = 64
lengths_list = [torch.randint(5, 100, (1,)).item() for _ in range(n_sequences)]
data = [torch.randn(l, 32) for l in lengths_list]

padded_data = pad_sequence(data, batch_first=True)
padded_lengths = torch.tensor(lengths_list)
packed_data = pack_padded_sequence(
    padded_data, padded_lengths, batch_first=True, enforce_sorted=False
)

rnn = nn.LSTM(32, 64, batch_first=True)

# Benchmark padded
N = 20
start = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        rnn(padded_data)
padded_time = (time.perf_counter() - start) / N * 1000

# Benchmark packed
start = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        rnn(packed_data)
packed_time = (time.perf_counter() - start) / N * 1000

max_len = max(lengths_list)
avg_len = sum(lengths_list) / len(lengths_list)
wasted = 1 - avg_len / max_len

print(f"Sequence stats: max_len={max_len}, avg_len={avg_len:.0f}, "
      f"wasted padding={wasted:.0%}")
print(f"Padded LSTM: {padded_time:.1f} ms")
print(f"Packed LSTM: {packed_time:.1f} ms")
print(f"Speedup:     {padded_time/packed_time:.2f}x")

print("\nDone!")
