"""
Module 04: Loss Functions — Complete Guide
============================================
Every major loss function in PyTorch with formulas, usage examples,
and practical guidance on when to use each one.

Run: python loss_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("PART 1: Classification Losses")
print("=" * 70)

# =============================================================================
# CrossEntropyLoss — Multi-class classification
# =============================================================================
print("\n--- nn.CrossEntropyLoss ---")
print("Formula: loss = -log(exp(x[class]) / sum(exp(x[j])))")
print("Combines LogSoftmax + NLLLoss internally")
print("Input: raw logits (NOT softmax!), Target: class indices\n")

criterion = nn.CrossEntropyLoss()

# Example: 4 samples, 5 classes
logits = torch.tensor([
    [2.0, 1.0, 0.1, -1.0, 0.5],   # Predicts class 0
    [-1.0, 3.0, 0.5, 0.2, 0.1],   # Predicts class 1
    [0.1, 0.2, 4.0, 0.5, -0.5],   # Predicts class 2
    [0.5, 0.3, 0.1, 0.8, 2.5],    # Predicts class 4
])
targets = torch.tensor([0, 1, 2, 3])  # True classes

loss = criterion(logits, targets)
print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets.tolist()}")
print(f"Loss: {loss.item():.4f}")

# Manual computation for verification
log_softmax = F.log_softmax(logits, dim=1)
manual_loss = -log_softmax[range(4), targets].mean()
print(f"Manual loss: {manual_loss.item():.4f}")
print(f"Match: {torch.allclose(loss, manual_loss)}")

# With class weights (for imbalanced datasets)
print("\n--- CrossEntropyLoss with class weights ---")
# Suppose class 3 is rare (weight it more)
weights = torch.tensor([1.0, 1.0, 1.0, 5.0, 1.0])
criterion_weighted = nn.CrossEntropyLoss(weight=weights)
loss_weighted = criterion_weighted(logits, targets)
print(f"Unweighted loss: {loss.item():.4f}")
print(f"Weighted loss:   {loss_weighted.item():.4f} (class 3 sample penalized more)")

# With label smoothing
print("\n--- CrossEntropyLoss with label smoothing ---")
criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_smooth = criterion_smooth(logits, targets)
print(f"No smoothing:    {loss.item():.4f}")
print(f"Smoothing=0.1:   {loss_smooth.item():.4f}")
print("Label smoothing prevents overconfident predictions")

# Ignore index (for padding in sequences)
print("\n--- CrossEntropyLoss with ignore_index ---")
criterion_ignore = nn.CrossEntropyLoss(ignore_index=-100)
targets_with_pad = torch.tensor([0, 1, -100, 3])  # -100 = ignore
loss_ignore = criterion_ignore(logits, targets_with_pad)
print(f"Targets with padding (-100): {targets_with_pad.tolist()}")
print(f"Loss (ignoring padded): {loss_ignore.item():.4f}")

# =============================================================================
# BCEWithLogitsLoss — Binary / Multi-label classification
# =============================================================================
print("\n\n--- nn.BCEWithLogitsLoss ---")
print("Formula: loss = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]")
print("Combines Sigmoid + BCELoss (more numerically stable)")
print("Input: raw logits, Target: 0.0 or 1.0\n")

criterion_bce = nn.BCEWithLogitsLoss()

# Binary classification (single output)
logits_binary = torch.tensor([0.5, -1.2, 2.3, -0.1])
targets_binary = torch.tensor([1.0, 0.0, 1.0, 0.0])
loss_binary = criterion_bce(logits_binary, targets_binary)
print(f"Binary classification:")
print(f"  Logits:  {logits_binary.tolist()}")
print(f"  Targets: {targets_binary.tolist()}")
print(f"  Loss: {loss_binary.item():.4f}")

# Multi-label classification (multiple labels per sample)
print("\n  Multi-label classification:")
logits_multi = torch.randn(4, 5)  # 4 samples, 5 possible labels
targets_multi = torch.tensor([
    [1, 0, 1, 0, 0],  # Has labels 0 and 2
    [0, 1, 0, 1, 1],  # Has labels 1, 3, 4
    [1, 1, 0, 0, 0],  # Has labels 0 and 1
    [0, 0, 0, 0, 1],  # Has label 4 only
], dtype=torch.float)
loss_multi = criterion_bce(logits_multi, targets_multi)
print(f"  Logits shape: {logits_multi.shape}")
print(f"  Loss: {loss_multi.item():.4f}")

# With pos_weight for imbalanced labels
pos_weight = torch.tensor([2.0, 1.0, 3.0, 1.0, 5.0])  # Weight positive examples more
criterion_pos = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_pos = criterion_pos(logits_multi, targets_multi)
print(f"  With pos_weight: {loss_pos.item():.4f}")

# =============================================================================
# NLLLoss — When you've already applied log_softmax
# =============================================================================
print("\n\n--- nn.NLLLoss ---")
print("Use when you've already computed log-probabilities yourself")
log_probs = F.log_softmax(logits, dim=1)
criterion_nll = nn.NLLLoss()
loss_nll = criterion_nll(log_probs, targets)
print(f"NLLLoss(log_softmax(logits)): {loss_nll.item():.4f}")
print(f"CrossEntropyLoss(logits):     {loss.item():.4f}")
print(f"They are identical: {torch.allclose(loss_nll, loss)}")

print("\n" + "=" * 70)
print("PART 2: Regression Losses")
print("=" * 70)

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets_reg = torch.tensor([3.0, -0.5, 2.0, 7.0])

# =============================================================================
# MSELoss — Mean Squared Error
# =============================================================================
print("\n--- nn.MSELoss ---")
print("Formula: loss = mean((y_pred - y_true)^2)")
criterion_mse = nn.MSELoss()
loss_mse = criterion_mse(predictions, targets_reg)
print(f"Predictions: {predictions.tolist()}")
print(f"Targets:     {targets_reg.tolist()}")
print(f"MSE Loss: {loss_mse.item():.4f}")

# Manual verification
manual_mse = ((predictions - targets_reg) ** 2).mean()
print(f"Manual MSE: {manual_mse.item():.4f}")

# Reduction options
print(f"\n  reduction='none': {nn.MSELoss(reduction='none')(predictions, targets_reg).tolist()}")
print(f"  reduction='sum':  {nn.MSELoss(reduction='sum')(predictions, targets_reg).item():.4f}")
print(f"  reduction='mean': {nn.MSELoss(reduction='mean')(predictions, targets_reg).item():.4f}")

# =============================================================================
# L1Loss — Mean Absolute Error
# =============================================================================
print("\n\n--- nn.L1Loss ---")
print("Formula: loss = mean(|y_pred - y_true|)")
criterion_l1 = nn.L1Loss()
loss_l1 = criterion_l1(predictions, targets_reg)
print(f"L1 Loss (MAE): {loss_l1.item():.4f}")
print("L1 is more robust to outliers than MSE")

# =============================================================================
# SmoothL1Loss / HuberLoss
# =============================================================================
print("\n\n--- nn.HuberLoss (Smooth L1) ---")
print("Formula: L2 when |error| < delta, L1 otherwise")
print("Best of both worlds: smooth near zero, robust to outliers")

for delta in [0.5, 1.0, 2.0]:
    criterion_huber = nn.HuberLoss(delta=delta)
    loss_huber = criterion_huber(predictions, targets_reg)
    print(f"  delta={delta}: loss={loss_huber.item():.4f}")

# Compare all three for different error magnitudes
print("\n  Comparison for different error magnitudes:")
print(f"  {'Error':>8} | {'MSE':>8} | {'L1':>8} | {'Huber':>8}")
print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
for error in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    pred = torch.tensor([error])
    target = torch.tensor([0.0])
    mse = nn.MSELoss()(pred, target).item()
    l1 = nn.L1Loss()(pred, target).item()
    huber = nn.HuberLoss(delta=1.0)(pred, target).item()
    print(f"  {error:>8.1f} | {mse:>8.4f} | {l1:>8.4f} | {huber:>8.4f}")

print("\n" + "=" * 70)
print("PART 3: Distribution Losses")
print("=" * 70)

# =============================================================================
# KLDivLoss — Kullback-Leibler Divergence
# =============================================================================
print("\n--- nn.KLDivLoss ---")
print("Formula: KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))")
print("Measures how distribution Q differs from reference distribution P")
print("IMPORTANT: Input must be LOG-probabilities!\n")

# Create two probability distributions
p = torch.tensor([0.4, 0.3, 0.2, 0.1])  # Target distribution
q_logits = torch.tensor([0.5, 0.3, 0.1, 0.1])  # Predicted (will be log'd)

q_log = torch.log(q_logits)  # Input: log-probabilities

criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=False)
loss_kl = criterion_kl(q_log, p)
print(f"Target distribution P: {p.tolist()}")
print(f"Predicted (log) Q:     {q_log.tolist()}")
print(f"KL Divergence: {loss_kl.item():.4f}")

# Common use: knowledge distillation
print("\n  Knowledge Distillation example:")
temperature = 4.0
teacher_logits = torch.randn(8, 10)
student_logits = torch.randn(8, 10)
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
print(f"  Distillation loss (T={temperature}): {distill_loss.item():.4f}")

print("\n" + "=" * 70)
print("PART 4: Metric Learning Losses")
print("=" * 70)

# =============================================================================
# TripletMarginLoss
# =============================================================================
print("\n--- nn.TripletMarginLoss ---")
print("Formula: loss = max(d(anchor, pos) - d(anchor, neg) + margin, 0)")
print("Pushes positive pairs together and negative pairs apart\n")

criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)

# Create embeddings
anchor = torch.randn(8, 128)
positive = anchor + torch.randn(8, 128) * 0.1  # Similar to anchor
negative = torch.randn(8, 128)  # Different from anchor

loss_triplet = criterion_triplet(anchor, positive, negative)
print(f"Anchor shape: {anchor.shape}")
print(f"Triplet loss: {loss_triplet.item():.4f}")

# Show effect of margin
for margin in [0.5, 1.0, 2.0]:
    loss = nn.TripletMarginLoss(margin=margin)(anchor, positive, negative)
    print(f"  margin={margin}: loss={loss.item():.4f}")

# =============================================================================
# CosineEmbeddingLoss
# =============================================================================
print("\n\n--- nn.CosineEmbeddingLoss ---")
print("Measures similarity using cosine distance")
print("target=+1: similar pair, target=-1: dissimilar pair\n")

criterion_cosine = nn.CosineEmbeddingLoss(margin=0.0)

x1 = torch.randn(8, 64)
x2_similar = x1 + torch.randn(8, 64) * 0.1
x2_different = torch.randn(8, 64)

# Similar pairs (target=1)
target_similar = torch.ones(8)
loss_similar = criterion_cosine(x1, x2_similar, target_similar)
print(f"Similar pairs loss: {loss_similar.item():.4f} (should be low)")

# Dissimilar pairs (target=-1)
target_dissimilar = -torch.ones(8)
loss_dissimilar = criterion_cosine(x1, x2_different, target_dissimilar)
print(f"Dissimilar pairs loss: {loss_dissimilar.item():.4f}")

# =============================================================================
# ContrastiveMarginLoss (using CosineEmbeddingLoss)
# =============================================================================
print("\n--- Contrastive Learning Pattern ---")
# In practice, you often combine similar and dissimilar pairs
all_x1 = torch.cat([x1, x1])
all_x2 = torch.cat([x2_similar, x2_different])
all_targets = torch.cat([torch.ones(8), -torch.ones(8)])
contrastive_loss = criterion_cosine(all_x1, all_x2, all_targets)
print(f"Combined contrastive loss: {contrastive_loss.item():.4f}")

print("\n" + "=" * 70)
print("PART 5: Other Useful Losses")
print("=" * 70)

# =============================================================================
# MarginRankingLoss
# =============================================================================
print("\n--- nn.MarginRankingLoss ---")
print("loss = max(0, -target * (x1 - x2) + margin)")
print("For learning to rank: x1 should be ranked higher when target=1\n")

criterion_rank = nn.MarginRankingLoss(margin=0.5)
x1_scores = torch.tensor([1.5, 2.0, 3.0, 0.5])
x2_scores = torch.tensor([1.0, 2.5, 1.0, 0.8])
# target=1: x1 should be > x2; target=-1: x2 should be > x1
target_rank = torch.tensor([1.0, -1.0, 1.0, -1.0])
loss_rank = criterion_rank(x1_scores, x2_scores, target_rank)
print(f"Ranking loss: {loss_rank.item():.4f}")

# =============================================================================
# MultiMarginLoss (SVM-like hinge loss)
# =============================================================================
print("\n--- nn.MultiMarginLoss ---")
criterion_mm = nn.MultiMarginLoss(margin=1.0)
logits_mm = torch.randn(4, 5)
targets_mm = torch.tensor([0, 2, 1, 4])
loss_mm = criterion_mm(logits_mm, targets_mm)
print(f"Multi-margin (hinge) loss: {loss_mm.item():.4f}")

# =============================================================================
# CTCLoss (for sequence-to-sequence without alignment)
# =============================================================================
print("\n--- nn.CTCLoss ---")
print("For sequence tasks where input/output lengths differ (speech recognition, OCR)")
ctc_loss = nn.CTCLoss(blank=0)
# (input_length, batch, num_classes) — note: NOT batch_first!
log_probs = F.log_softmax(torch.randn(50, 4, 20), dim=2)  # T=50, B=4, C=20
targets_ctc = torch.randint(1, 20, (4, 15))  # Target sequences (no blank=0)
input_lengths = torch.full((4,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 16, (4,), dtype=torch.long)
loss_ctc = ctc_loss(log_probs, targets_ctc, input_lengths, target_lengths)
print(f"CTC loss: {loss_ctc.item():.4f}")

print("\n" + "=" * 70)
print("PART 6: Custom Loss Functions")
print("=" * 70)

print("\n--- Creating Custom Losses ---")


# Method 1: Simple function
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal Loss: focuses on hard examples by down-weighting easy ones."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    loss = alpha * focal_weight * bce
    return loss.mean()


logits_focal = torch.randn(32, 1)
targets_focal = torch.randint(0, 2, (32, 1)).float()
loss_focal = focal_loss(logits_focal, targets_focal)
print(f"Focal loss: {loss_focal.item():.4f}")


# Method 2: nn.Module subclass (for learnable parameters in loss)
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)
        # NLL component for true class
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Smooth component (uniform over all classes)
        smooth = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


criterion_ls = LabelSmoothingCrossEntropy(smoothing=0.1)
logits_ls = torch.randn(16, 10)
targets_ls = torch.randint(0, 10, (16,))
loss_ls = criterion_ls(logits_ls, targets_ls)
print(f"Label smoothing CE loss: {loss_ls.item():.4f}")

print("\n" + "=" * 70)
print("PART 7: Loss Function Selection Guide")
print("=" * 70)

print("""
Task                          | Loss Function            | Input Format
------------------------------|--------------------------|---------------------------
Multi-class classification    | CrossEntropyLoss         | Raw logits + class indices
Binary classification         | BCEWithLogitsLoss        | Raw logits + 0/1 targets
Multi-label classification    | BCEWithLogitsLoss        | Raw logits + multi-hot
Regression                    | MSELoss                  | Predictions + targets
Robust regression             | HuberLoss / L1Loss       | Predictions + targets
Knowledge distillation        | KLDivLoss                | Log-probs + teacher probs
Metric learning (triplets)    | TripletMarginLoss        | Anchor/pos/neg embeddings
Metric learning (pairs)       | CosineEmbeddingLoss      | Pair embeddings + +1/-1
Sequence (no alignment)       | CTCLoss                  | Log-probs + target seq
Object detection              | Focal Loss (custom)      | Logits + targets
Ranking                       | MarginRankingLoss        | Score pairs + direction

Tips:
- Always pass RAW LOGITS to CrossEntropyLoss and BCEWithLogitsLoss
- Use 'reduction=none' when you need per-sample losses (for weighting/debugging)
- Use label_smoothing for better generalization
- Use class weights or pos_weight for imbalanced datasets
""")

print("=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
