"""
torch.masked — MaskedTensor Basics
===================================
Demonstrates masked reductions, masked softmax, padded sequence handling,
mask propagation, and the torch.masked.* function API.

Run: python masked_tensor_basics.py
"""

import torch

print("=" * 70)
print("torch.masked — MaskedTensor Basics")
print(f"PyTorch version: {torch.__version__}")
print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Creating MaskedTensors
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. Creating MaskedTensors")
print("=" * 70)

from torch.masked import MaskedTensor

data_1d = torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0])
mask_1d = torch.tensor([True, True, True, False, False])
mt_1d = MaskedTensor(data_1d, mask_1d)
print(f"\n1D MaskedTensor:\n{mt_1d}")

data_2d = torch.tensor([
    [1.0, 2.0, 3.0, 0.0, 0.0],
    [4.0, 5.0, 0.0, 0.0, 0.0],
    [6.0, 7.0, 8.0, 9.0, 0.0],
])
lengths = torch.tensor([3, 2, 4])
mask_2d = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
mt_2d = MaskedTensor(data_2d, mask_2d)
print(f"\n2D MaskedTensor (padded sequences):\n{mt_2d}")
print(f"\nMask:\n{mask_2d}")


# ---------------------------------------------------------------------------
# 2. Masked Reductions — Sum and Mean
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Masked Reductions — Sum and Mean")
print("=" * 70)

data = torch.tensor([
    [3.0, 1.0, 4.0, 0.0, 0.0],
    [2.0, 7.0, 0.0, 0.0, 0.0],
    [5.0, 3.0, 2.0, 8.0, 0.0],
])
lengths = torch.tensor([3, 2, 4])
mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)

print(f"\nData:\n{data}")
print(f"Lengths: {lengths}")
print(f"Mask:\n{mask}")

regular_sum = data.sum(dim=1)
regular_mean = data.mean(dim=1)
print(f"\nRegular sum  (includes padding): {regular_sum}")
print(f"Regular mean (includes padding): {regular_mean}")

masked_sum = (data * mask.float()).sum(dim=1)
valid_count = mask.float().sum(dim=1)
manual_mean = masked_sum / valid_count
print(f"\nManual masked sum:  {masked_sum}")
print(f"Manual masked mean: {manual_mean}")

api_mean = torch.masked._ops.mean(data, dim=1, mask=mask)
print(f"torch.masked mean:  {api_mean}")

expected_means = torch.tensor([
    (3.0 + 1.0 + 4.0) / 3,
    (2.0 + 7.0) / 2,
    (5.0 + 3.0 + 2.0 + 8.0) / 4,
])
print(f"\nExpected means:     {expected_means}")
print(f"Match: {torch.allclose(api_mean, expected_means)}")


# ---------------------------------------------------------------------------
# 3. Masked Reductions — amax and amin
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. Masked Reductions — amax and amin")
print("=" * 70)

data = torch.tensor([
    [3.0, 1.0, 4.0, 99.0, 99.0],
    [2.0, 7.0, -5.0, -5.0, -5.0],
])
mask = torch.tensor([
    [True, True, True, False, False],
    [True, True, False, False, False],
])

regular_max = data.amax(dim=1)
regular_min = data.amin(dim=1)
print(f"\nData:\n{data}")
print(f"Mask:\n{mask}")
print(f"\nRegular max: {regular_max}  (includes padding junk)")
print(f"Regular min: {regular_min}  (includes padding junk)")

masked_max = torch.masked._ops.amax(data, dim=1, mask=mask)
masked_min = torch.masked._ops.amin(data, dim=1, mask=mask)
print(f"\nMasked max:  {masked_max}  (correct: [4.0, 7.0])")
print(f"Masked min:  {masked_min}  (correct: [1.0, 2.0])")


# ---------------------------------------------------------------------------
# 4. Masked Softmax — vs Manual masked_fill
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Masked Softmax vs Manual masked_fill")
print("=" * 70)

scores = torch.tensor([
    [0.5, 1.2, 0.3, 0.0, 0.0],
    [2.0, 1.0, 3.0, 0.5, 0.0],
])
mask = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, False],
])

manual_filled = scores.masked_fill(~mask, float('-inf'))
manual_softmax = torch.softmax(manual_filled, dim=1)
print(f"\nScores:\n{scores}")
print(f"Mask:\n{mask}")
print(f"\nManual masked_fill + softmax:\n{manual_softmax}")
print(f"  Row 0 valid sum: {manual_softmax[0, :3].sum().item():.4f}")
print(f"  Row 1 valid sum: {manual_softmax[1, :4].sum().item():.4f}")

masked_softmax = torch.masked.softmax(scores, dim=1, mask=mask)
print(f"\ntorch.masked.softmax:\n{masked_softmax}")
print(f"  Row 0 valid sum: {masked_softmax[0, :3].sum().item():.4f}")
print(f"  Row 1 valid sum: {masked_softmax[1, :4].sum().item():.4f}")


# ---------------------------------------------------------------------------
# 5. Edge Case: All-Masked Row
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. Edge Case: All-Masked Row in Softmax")
print("=" * 70)

scores = torch.tensor([[1.0, 2.0, 3.0]])
all_masked = torch.tensor([[False, False, False]])

manual = torch.softmax(
    scores.masked_fill(~all_masked, float('-inf')), dim=1
)
print(f"\nAll positions masked:")
print(f"  Manual masked_fill + softmax: {manual}  (NaN!)")

safe = torch.masked.softmax(scores, dim=1, mask=all_masked)
print(f"  torch.masked.softmax:         {safe}  (zeros — safe)")


# ---------------------------------------------------------------------------
# 6. Padded Sequence Mean: Three Approaches
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. Padded Sequence Mean: Three Approaches")
print("=" * 70)

torch.manual_seed(42)
seq_a = torch.tensor([3.0, 1.0, 4.0])
seq_b = torch.tensor([2.0, 7.0])
seq_c = torch.tensor([5.0, 3.0, 2.0, 8.0])

padded = torch.nn.utils.rnn.pad_sequence(
    [seq_a, seq_b, seq_c], batch_first=True, padding_value=0.0
)
lengths = torch.tensor([3, 2, 4])
print(f"\nPadded data:\n{padded}")
print(f"Lengths: {lengths}")

naive_mean = padded.mean(dim=1)
print(f"\nApproach 1 — Naive mean (WRONG):  {naive_mean}")

mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
manual_mean = (padded * mask.float()).sum(dim=1) / mask.float().sum(dim=1)
print(f"Approach 2 — Manual masking:      {manual_mean}")

api_mean = torch.masked._ops.mean(padded, dim=1, mask=mask)
print(f"Approach 3 — torch.masked API:    {api_mean}")

true_means = torch.tensor([
    seq_a.mean().item(),
    seq_b.mean().item(),
    seq_c.mean().item(),
])
print(f"\nTrue means: {true_means}")
print(f"All correct: {torch.allclose(api_mean, true_means)}")


# ---------------------------------------------------------------------------
# 7. Masked Operations on 2D Batch Data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. Masked Operations on 2D Batch Data")
print("=" * 70)

torch.manual_seed(0)
batch_size, max_len = 4, 6
data = torch.randn(batch_size, max_len)
lengths = torch.tensor([6, 3, 5, 2])
mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

print(f"\nBatch data shape: {data.shape}")
print(f"Lengths: {lengths}")
print(f"Mask:\n{mask}")

masked_sum = torch.masked._ops.sum(data, dim=1, mask=mask)
masked_mean = torch.masked._ops.mean(data, dim=1, mask=mask)
masked_max = torch.masked._ops.amax(data, dim=1, mask=mask)
masked_min = torch.masked._ops.amin(data, dim=1, mask=mask)

print(f"\nMasked sum:  {masked_sum}")
print(f"Masked mean: {masked_mean}")
print(f"Masked max:  {masked_max}")
print(f"Masked min:  {masked_min}")

for i in range(batch_size):
    valid = data[i, :lengths[i]]
    print(f"\n  Seq {i} (len={lengths[i]}): {valid.tolist()}")
    print(f"    sum={valid.sum():.4f}, mean={valid.mean():.4f}, "
          f"max={valid.max():.4f}, min={valid.min():.4f}")


# ---------------------------------------------------------------------------
# 8. torch.masked.* Function API Overview
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("8. torch.masked.* Function API Overview")
print("=" * 70)

data = torch.tensor([[1.0, 2.0, 3.0, 0.0],
                      [4.0, 5.0, 0.0, 0.0]])
mask = torch.tensor([[True, True, True, False],
                      [True, True, False, False]])

print(f"\nData:\n{data}")
print(f"Mask:\n{mask}")

results = {
    "sum":       torch.masked._ops.sum(data, dim=1, mask=mask),
    "mean":      torch.masked._ops.mean(data, dim=1, mask=mask),
    "amax":      torch.masked._ops.amax(data, dim=1, mask=mask),
    "amin":      torch.masked._ops.amin(data, dim=1, mask=mask),
    "prod":      torch.masked._ops.prod(data, dim=1, mask=mask),
    "softmax":   torch.masked.softmax(data, dim=1, mask=mask),
}

for name, val in results.items():
    if val.dim() == 1:
        print(f"  {name:10s}: {val}")
    else:
        print(f"  {name:10s}:\n{val}")


# ---------------------------------------------------------------------------
# 9. Mask Propagation Through MaskedTensor Operations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("9. Mask Propagation Through MaskedTensor Operations")
print("=" * 70)

data_a = torch.tensor([1.0, 2.0, 3.0, 4.0])
mask_a = torch.tensor([True, True, True, False])
mt_a = MaskedTensor(data_a, mask_a)

print(f"\nOriginal MaskedTensor a:\n  {mt_a}")

mt_neg = -mt_a
print(f"\nUnary: -a (mask preserved):\n  {mt_neg}")

mt_abs = mt_a.abs()
print(f"\nUnary: abs(a) (mask preserved):\n  {mt_abs}")

data_b = torch.tensor([10.0, 20.0, 30.0, 40.0])
mask_b = torch.tensor([True, False, True, True])
mt_b = MaskedTensor(data_b, mask_b)

print(f"\nMaskedTensor b:\n  {mt_b}")

mt_add = mt_a + mt_b
print(f"\nBinary: a + b (masks ANDed):\n  {mt_add}")
print(f"  Mask a: {mask_a.tolist()}")
print(f"  Mask b: {mask_b.tolist()}")
print(f"  Result: {(mask_a & mask_b).tolist()}")

mt_sum = mt_a.sum()
print(f"\nReduction: a.sum() = {mt_sum}")


# ---------------------------------------------------------------------------
# 10. Masked Log Softmax and Normalize
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("10. Masked Log Softmax and Normalize")
print("=" * 70)

data = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]])
mask = torch.tensor([[True, True, True, False, False]])

log_sm = torch.masked.log_softmax(data, dim=1, mask=mask)
print(f"\nData: {data}")
print(f"Mask: {mask}")
print(f"Masked log_softmax: {log_sm}")

normed = torch.masked.normalize(data, ord=2.0, dim=1, mask=mask)
print(f"Masked normalize (L2): {normed}")

manual_norm = data[0, :3].norm(2)
manual_normed = data[0, :3] / manual_norm
print(f"\nManual L2 norm of valid elements: {manual_norm:.4f}")
print(f"Manual normalized: {manual_normed}")


# ---------------------------------------------------------------------------
# 11. Limitations Demo: Unsupported Operations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("11. Limitations — Unsupported Operations")
print("=" * 70)

mt = MaskedTensor(
    torch.tensor([1.0, 2.0, 3.0, 0.0]),
    torch.tensor([True, True, True, False])
)

supported_ops = ['abs', 'neg', 'sum', 'mean']
for op_name in supported_ops:
    try:
        op = getattr(mt, op_name)
        result = op()
        print(f"  {op_name:12s}: supported ✓")
    except Exception as e:
        print(f"  {op_name:12s}: NOT supported — {e}")

unsupported_attempts = [
    ("torch.sort", lambda: torch.sort(mt)),
    ("mt.reshape", lambda: mt.reshape(2, 2)),
]
for name, fn in unsupported_attempts:
    try:
        fn()
        print(f"  {name:12s}: supported ✓")
    except Exception as e:
        err_msg = str(e)[:80]
        print(f"  {name:12s}: NOT supported — {err_msg}")


# ---------------------------------------------------------------------------
# 12. Practical: Masked Attention Scores
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("12. Practical: Masked Attention Scores")
print("=" * 70)

torch.manual_seed(42)
batch, seq_len, d_k = 2, 5, 4
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)
lengths = torch.tensor([3, 5])

scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
print(f"\nAttention scores shape: {scores.shape}")

key_mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
attn_mask = key_mask.unsqueeze(1).expand(-1, seq_len, -1)
print(f"Key mask:\n{key_mask}")

attn_weights = torch.masked.softmax(scores, dim=-1, mask=attn_mask)
print(f"\nMasked attention weights (batch 0):\n{attn_weights[0]}")
print(f"Row sums (batch 0): {attn_weights[0].sum(dim=-1)}")

output = attn_weights @ V
print(f"\nAttention output shape: {output.shape}")
print(f"Batch 0, position 0 output: {output[0, 0]}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Key torch.masked concepts:
  1. MaskedTensor = data + boolean mask (True=valid, False=masked)
  2. torch.masked._ops.{sum,mean,amax,amin,prod} — masked reductions
  3. torch.masked.softmax — softmax ignoring masked positions
  4. torch.masked.{log_softmax,normalize} — other masked ops
  5. Unary ops preserve mask, binary ops AND masks
  6. Prototype feature — not all ops supported yet
""")
