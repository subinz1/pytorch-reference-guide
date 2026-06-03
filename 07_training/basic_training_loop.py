"""
Basic Training Loop — Complete Annotated Example
=================================================
This file demonstrates the fundamental training loop pattern in PyTorch.
We train a simple neural network on synthetic data to show every step clearly.

Run: python basic_training_loop.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# 1. Create a simple model
# =============================================================================

class SimpleClassifier(nn.Module):
    """A 3-layer MLP for binary classification."""

    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 2. Create synthetic data
# =============================================================================

torch.manual_seed(42)

num_samples = 1000
input_dim = 20

X = torch.randn(num_samples, input_dim)
true_weights = torch.randn(input_dim)
y = (X @ true_weights > 0).float().unsqueeze(1)  # Binary labels

# Split into train/validation
train_X, val_X = X[:800], X[800:]
train_y, val_y = y[:800], y[800:]

train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# =============================================================================
# 3. Initialize model, loss function, optimizer
# =============================================================================

model = SimpleClassifier(input_dim=input_dim)
loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy with sigmoid built-in
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =============================================================================
# 4. The Training Loop
# =============================================================================

num_epochs = 20

print("=" * 60)
print("BASIC TRAINING LOOP")
print("=" * 60)

for epoch in range(num_epochs):
    # -------------------------------------------------------------------------
    # TRAINING PHASE
    # -------------------------------------------------------------------------
    model.train()  # Enable dropout and batch norm training behavior

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_inputs, batch_targets in train_loader:
        # Step 1: Clear gradients from previous iteration
        optimizer.zero_grad(set_to_none=True)

        # Step 2: Forward pass — compute predictions
        predictions = model(batch_inputs)

        # Step 3: Compute loss — how far are we from targets?
        loss = loss_fn(predictions, batch_targets)

        # Step 4: Backward pass — compute gradients
        loss.backward()

        # Step 5: Optimizer step — update parameters using gradients
        optimizer.step()

        # Track metrics
        train_loss += loss.item() * batch_inputs.size(0)
        predicted_labels = (predictions > 0).float()
        train_correct += (predicted_labels == batch_targets).sum().item()
        train_total += batch_inputs.size(0)

    avg_train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total

    # -------------------------------------------------------------------------
    # VALIDATION PHASE
    # -------------------------------------------------------------------------
    model.eval()  # Disable dropout, use running stats for batch norm

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for batch_inputs, batch_targets in val_loader:
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)

            val_loss += loss.item() * batch_inputs.size(0)
            predicted_labels = (predictions > 0).float()
            val_correct += (predicted_labels == batch_targets).sum().item()
            val_total += batch_inputs.size(0)

    avg_val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.3f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.3f}")

# =============================================================================
# 5. Demonstrating train() vs eval() difference
# =============================================================================

print("\n" + "=" * 60)
print("TRAIN vs EVAL MODE DIFFERENCE")
print("=" * 60)

test_input = torch.randn(1, input_dim)

model.train()
outputs_train = []
for _ in range(10):
    outputs_train.append(model(test_input).item())

model.eval()
outputs_eval = []
for _ in range(10):
    outputs_eval.append(model(test_input).item())

print(f"\nSame input, model.train() mode (10 runs):")
print(f"  Outputs vary due to dropout: {[f'{x:.4f}' for x in outputs_train[:5]]}")
print(f"  Std dev: {torch.tensor(outputs_train).std().item():.6f}")

print(f"\nSame input, model.eval() mode (10 runs):")
print(f"  Outputs are identical (no dropout): {[f'{x:.4f}' for x in outputs_eval[:5]]}")
print(f"  Std dev: {torch.tensor(outputs_eval).std().item():.6f}")

# =============================================================================
# 6. Demonstrating gradient accumulation effect of NOT zeroing gradients
# =============================================================================

print("\n" + "=" * 60)
print("WHY zero_grad() MATTERS")
print("=" * 60)

model.train()
sample_input, sample_target = next(iter(train_loader))

# First backward pass
optimizer.zero_grad()
loss1 = loss_fn(model(sample_input), sample_target)
loss1.backward()

# Get gradient of first layer after one backward
first_param = next(model.parameters())
grad_after_1 = first_param.grad.clone()

# Second backward WITHOUT zeroing — gradients accumulate!
loss2 = loss_fn(model(sample_input), sample_target)
loss2.backward()
grad_after_2 = first_param.grad.clone()

print(f"\nGradient norm after 1st backward: {grad_after_1.norm().item():.6f}")
print(f"Gradient norm after 2nd backward (accumulated): {grad_after_2.norm().item():.6f}")
print(f"Ratio (should be ~2x): {grad_after_2.norm().item() / grad_after_1.norm().item():.2f}")

# =============================================================================
# 7. Saving and loading a checkpoint
# =============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT SAVE/LOAD")
print("=" * 60)

import tempfile
import os

checkpoint_path = os.path.join(tempfile.gettempdir(), "model_checkpoint.pt")

checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
}
torch.save(checkpoint, checkpoint_path)
print(f"\nCheckpoint saved to {checkpoint_path}")
print(f"  Contains: {list(checkpoint.keys())}")

# Loading a checkpoint
loaded = torch.load(checkpoint_path, weights_only=False)
new_model = SimpleClassifier(input_dim=input_dim)
new_model.load_state_dict(loaded['model_state_dict'])
print(f"  Loaded model from epoch {loaded['epoch']}")

# Verify loaded model produces same output
model.eval()
new_model.eval()
with torch.no_grad():
    orig_out = model(test_input)
    loaded_out = new_model(test_input)
    print(f"  Original output: {orig_out.item():.6f}")
    print(f"  Loaded output:   {loaded_out.item():.6f}")
    print(f"  Match: {torch.allclose(orig_out, loaded_out)}")

# Cleanup
os.remove(checkpoint_path)

print("\n✓ Basic training loop complete!")
