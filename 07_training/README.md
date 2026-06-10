<div align="center">

[← Previous Module](../06_data_loading/) | [🏠 Home](../README.md) | [Next Module →](../08_torch_compile/)

</div>

---

> **Module 07** of the PyTorch Complete Learning Guide
> **Prerequisites:** Modules 04, 05, 06
> **Time to complete:** ~4 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `basic_training_loop.py` | Basic training loop — complete annotated example |
| `mixed_precision.py` | Mixed precision training — AMP with float16 and bfloat16 |
| `gradient_techniques.py` | Gradient techniques — accumulation, checkpointing, and clipping |
| `transfer_learning.py` | Transfer learning — freeze/unfreeze and differential learning rates |
| `regularization.py` | Regularization techniques — EMA, label smoothing, weight decay |

---

# Module 07: The Complete Training Guide

## Overview

Training a neural network is where theory meets practice. This module covers
everything from the basic training loop to advanced techniques used by
state-of-the-art models. By the end, you'll understand not just *how* to train
models, but *why* each step exists and how to diagnose problems.

---

## 1. Basic Training Loop Anatomy

Every training loop in PyTorch follows the same fundamental pattern:

```
forward pass → compute loss → backward pass → optimizer step
```

Let's break each step down:

### Step 1: Forward Pass

```python
predictions = model(inputs)
```

Data flows through the model's layers. Each layer applies its transformation
(matrix multiply, activation, normalization, etc.) and PyTorch records the
operations in a computational graph for later backpropagation.

### Step 2: Compute Loss

```python
loss = loss_fn(predictions, targets)
```

The loss function measures how far the model's predictions are from the true
targets. Common losses:
- `nn.CrossEntropyLoss()` — classification (combines LogSoftmax + NLLLoss)
- `nn.MSELoss()` — regression (mean squared error)
- `nn.BCEWithLogitsLoss()` — binary classification (numerically stable)

### Step 3: Backward Pass

```python
loss.backward()
```

PyTorch walks backward through the computational graph, computing the gradient
of the loss with respect to every parameter that has `requires_grad=True`. These
gradients accumulate in each parameter's `.grad` attribute.

### Step 4: Optimizer Step

```python
optimizer.step()
```

The optimizer uses the computed gradients to update the model's parameters.
Different optimizers (SGD, Adam, AdamW) use different update rules, but all
read from `.grad` and modify `.data`.

### The Missing Step: zero_grad()

```python
optimizer.zero_grad()
```

This clears old gradients before computing new ones. Without it, gradients
accumulate across iterations (which is sometimes intentional — see gradient
accumulation below).

### Complete Minimal Loop

```python
model.train()
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()           # Clear old gradients
        predictions = model(batch_inputs)  # Forward pass
        loss = loss_fn(predictions, batch_targets)  # Compute loss
        loss.backward()                 # Compute gradients
        optimizer.step()                # Update parameters
```

---

## 2. train() vs eval() Mode

### What model.train() Does

Sets the model to training mode. This affects layers that behave differently
during training vs inference:

- **Dropout**: Randomly zeros elements during training. During eval, all
  elements pass through (scaled appropriately).
- **BatchNorm**: During training, uses batch statistics (mean/var of current
  batch) and updates running statistics. During eval, uses the accumulated
  running statistics.

### What model.eval() Does

```python
model.eval()
```

Sets the model to evaluation mode. Dropout is disabled, BatchNorm uses running
statistics instead of batch statistics.

### Common Mistake

```python
# WRONG: Forgetting to switch modes
def evaluate(model, test_loader):
    # model is still in train() mode!
    # Dropout is randomly zeroing activations
    # BatchNorm is using (and updating!) batch statistics
    total_correct = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        ...

# CORRECT:
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():  # Also disable gradient computation
        total_correct = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            ...
    model.train()  # Switch back after evaluation
```

### torch.no_grad() vs model.eval()

These are different things:
- `model.eval()` — changes layer behavior (dropout, batchnorm)
- `torch.no_grad()` — disables gradient computation (saves memory/compute)

For evaluation, you typically want BOTH.

---

## 3. zero_grad() — Why and How

### Why Gradients Accumulate

PyTorch accumulates gradients by default. After `loss.backward()`, the
`.grad` attribute of each parameter ADDS to whatever was already there:

```python
param.grad += new_gradient  # This is what happens internally
```

This design choice enables gradient accumulation (discussed later), but
it means you must manually clear gradients each iteration.

### set_to_none=True Optimization

```python
optimizer.zero_grad(set_to_none=True)
```

Instead of setting gradients to zero tensors, this sets them to `None`.
Benefits:
- Slightly less memory (no zero tensor allocated)
- Can be marginally faster
- The gradient will be lazily created on the next backward pass

This is now the default in modern PyTorch (>= 2.0). The only reason to
use `set_to_none=False` is if your code explicitly checks `if param.grad is not None`.

---

## 4. Mixed Precision Training (AMP)

### What It Is

Mixed precision training uses lower-precision floating point numbers (float16
or bfloat16) for most operations, while keeping critical operations in float32.
This is faster because:
- Lower precision operations use less memory bandwidth
- Hardware (GPUs, modern CPUs) has specialized units for half-precision math
- Smaller tensors mean more data fits in cache

### float16 vs float32

| Property      | float32          | float16          | bfloat16         |
|---------------|------------------|------------------|------------------|
| Total bits    | 32               | 16               | 16               |
| Exponent bits | 8                | 5                | 8                |
| Mantissa bits | 23               | 10               | 7                |
| Max value     | ~3.4 × 10³⁸     | 65504            | ~3.4 × 10³⁸     |
| Min positive  | ~1.2 × 10⁻³⁸    | ~6.0 × 10⁻⁸     | ~1.2 × 10⁻³⁸    |

### The torch.amp API (Modern, Device-Agnostic)

```python
# The modern way (PyTorch 2.0+)
with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
    output = model(input)
    loss = loss_fn(output, target)
```

The `autocast` context manager automatically casts operations to the specified
lower precision where safe, and keeps float32 where needed (e.g., loss
computation, softmax, layer norm).

### GradScaler (for float16 only)

float16 has a limited range. Small gradients can underflow to zero. GradScaler
solves this by:
1. Scaling the loss UP before backward (so gradients are larger)
2. Unscaling gradients before optimizer step
3. Skipping steps where gradients contain inf/nan (and reducing the scale)

```python
scaler = torch.amp.GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(inputs)
        loss = loss_fn(output, targets)

    scaler.scale(loss).backward()     # Scaled backward
    scaler.step(optimizer)            # Unscale + step (or skip)
    scaler.update()                   # Adjust scale factor
```

### BFloat16 Doesn't Need GradScaler

bfloat16 has the same exponent range as float32, so gradients don't underflow.
You can use it without GradScaler:

```python
with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
    output = model(input)
    loss = loss_fn(output, target)

loss.backward()  # No scaler needed
optimizer.step()
```

---

## 5. BFloat16 vs Float16 — When to Use Each

### Float16 (FP16)
- **Pros**: Widely supported, maximum speed on older GPUs (V100)
- **Cons**: Limited range (max 65504), requires GradScaler, can overflow/underflow
- **Use when**: Training on older NVIDIA GPUs, inference where range is known

### BFloat16 (BF16)
- **Pros**: Same range as float32, no GradScaler needed, more stable training
- **Cons**: Less precision (7 mantissa bits vs 10), requires Ampere+ GPU or modern CPU
- **Use when**: Training large models (LLMs), when numerical stability matters

### Practical Recommendation

- For training: prefer bfloat16 if your hardware supports it
- For inference: either works, float16 slightly more precise per value
- For CPU: bfloat16 is supported on modern x86 (AMX) and ARM

---

## 6. Gradient Accumulation

### The Problem

You want a batch size of 256 but your memory only fits 32 samples.

### The Solution

Accumulate gradients over multiple mini-batches before stepping:

```python
accumulation_steps = 8  # Effective batch = 32 * 8 = 256

for i, (inputs, targets) in enumerate(dataloader):
    # Forward + backward (gradients accumulate)
    output = model(inputs)
    loss = loss_fn(output, targets)
    loss = loss / accumulation_steps  # Scale loss!
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Why Divide the Loss?

Without division, accumulated gradients are `accumulation_steps` times larger
than a true large-batch gradient. Dividing the loss by `accumulation_steps`
makes the accumulated gradient equivalent to computing it over one large batch.

Mathematically: `grad(L/N) summed N times = grad(sum(L_i)/N)` which equals
the gradient of the mean loss over all N mini-batches.

---

## 7. Gradient Checkpointing (Activation Checkpointing)

### The Problem

During the forward pass, PyTorch saves all intermediate activations for use
in the backward pass. For deep models, this uses enormous amounts of memory.

### The Solution

Don't save activations — recompute them during the backward pass. This trades
compute time (~33% more) for memory (~60-80% savings).

### Usage

```python
from torch.utils.checkpoint import checkpoint

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = HeavyBlock()
        self.block2 = HeavyBlock()
        self.block3 = HeavyBlock()

    def forward(self, x):
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        x = checkpoint(self.block3, x, use_reentrant=False)
        return x
```

### use_reentrant Parameter

- `use_reentrant=False` (recommended): Uses a newer, more robust implementation.
  Supports all autograd features correctly.
- `use_reentrant=True` (legacy): The old implementation. Has subtle bugs with
  certain autograd features. Being phased out.

Always use `use_reentrant=False` for new code.

---

## 8. Gradient Clipping

### The Problem

Exploding gradients: when gradients become extremely large, the optimizer
takes huge steps that destabilize training. Common in RNNs and deep networks.

### clip_grad_norm_

Scales all gradients so their combined L2 norm doesn't exceed a threshold:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This preserves the direction of gradients but limits their magnitude. The
most common approach.

### clip_grad_value_

Clamps each gradient element independently to [-value, +value]:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

More aggressive — changes gradient direction. Rarely used in practice.

### Where to Place Clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # Step with clipped gradients
```

Always clip AFTER backward, BEFORE step.

---

## 9. Transfer Learning

### The Concept

Take a model pre-trained on a large dataset (e.g., ImageNet with 1M images)
and adapt it to your smaller dataset. The pre-trained model already knows
useful features (edges, textures, shapes).

### Strategy 1: Feature Extraction (Freeze Everything)

```python
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier
model.fc = nn.Linear(512, num_classes)
# Only model.fc parameters will be trained
```

### Strategy 2: Fine-tuning with Different Learning Rates

```python
# Lower LR for pretrained backbone, higher LR for new head
optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3},
])
```

### Strategy 3: Progressive Unfreezing

Start with everything frozen except the head. Gradually unfreeze layers
from top to bottom:

```python
# Epoch 1-5: Only train the head
# Epoch 5-10: Unfreeze last block + head
# Epoch 10+: Unfreeze everything with small LR
```

This prevents the pre-trained features from being destroyed by large
early gradients.

---

## 10. Fine-Tuning Strategies

### Full Fine-Tuning

Unfreeze everything, train with a small learning rate:
```python
for param in model.parameters():
    param.requires_grad = True
optimizer = Adam(model.parameters(), lr=1e-5)
```
Best when you have enough data and the domain differs from pre-training.

### Linear Probing

Only train a linear layer on top of frozen features:
```python
for param in model.parameters():
    param.requires_grad = False
probe = nn.Linear(feature_dim, num_classes)
```
Good baseline to check how useful the features are.

### Progressive Unfreezing

Unfreeze one layer group at a time, starting from the output end:
```python
# Phase 1: Just the head
# Phase 2: Head + last block
# Phase 3: Head + last 2 blocks
# Phase N: Everything
```

### Which to Choose?

| Strategy              | Data Amount | Domain Similarity | Risk      |
|-----------------------|-------------|-------------------|-----------|
| Linear probe          | Very small  | Any               | Low       |
| Freeze + new head     | Small       | Similar           | Low       |
| Progressive unfreeze  | Medium      | Different         | Medium    |
| Full fine-tune        | Large       | Different         | Higher    |

---

## 11. Knowledge Distillation

### The Concept

Train a small "student" model to mimic a large "teacher" model. The student
learns from the teacher's soft predictions (probability distributions) which
contain more information than hard labels.

### Why Soft Targets Help

Hard label: [0, 0, 1, 0] — "this is a cat"
Soft prediction: [0.01, 0.05, 0.85, 0.09] — "this is mostly cat, slightly dog"

The soft predictions encode relationships between classes that hard labels miss.

### Temperature Scaling

Higher temperature makes the distribution softer (more informative):

```python
soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
soft_student = F.log_softmax(student_logits / temperature, dim=-1)
distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
distill_loss = distill_loss * (temperature ** 2)  # Scale back
```

The `temperature ** 2` factor compensates for the reduced gradient magnitude
at higher temperatures.

### Combined Loss

```python
total_loss = alpha * distill_loss + (1 - alpha) * hard_loss
```

Typically alpha = 0.5-0.9 (emphasize soft targets).

---

## 12. EMA (Exponential Moving Average)

### What It Is

Maintain a running average of model parameters that smooths out training noise:

```
ema_param = decay * ema_param + (1 - decay) * current_param
```

Typical decay: 0.999 or 0.9999 (very slow moving average).

### Why It Helps

- Reduces variance in the final model
- Often achieves better generalization than the final checkpoint
- Used in many SOTA models (diffusion models, GANs, etc.)

### Implementation

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach()
                      for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data, alpha=1 - self.decay
            )

    def apply(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
```

---

## 13. SWA (Stochastic Weight Averaging)

### The Concept

Average model weights from multiple points in training (typically from a
cyclical or high constant LR schedule). This tends to find flatter minima
which generalize better.

### PyTorch Built-in Support

```python
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(swa_start, total_epochs):
    train_one_epoch(model)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# Update batch normalization statistics
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

### SWA vs EMA

- **EMA**: Continuous exponential average, gives more weight to recent params
- **SWA**: Equal-weight average of checkpoints, typically from later training

---

## 14. Label Smoothing

### What It Does

Instead of training against hard targets [0, 0, 1, 0], use soft targets
[0.033, 0.033, 0.9, 0.033]. This prevents the model from becoming overconfident.

### Formula

```
smooth_target = (1 - smoothing) * one_hot + smoothing / num_classes
```

With smoothing=0.1 and 4 classes:
- True class: 0.9 + 0.1/4 = 0.925
- Other classes: 0.1/4 = 0.025

### PyTorch Implementation

```python
# Built-in support in CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### When to Use

- Large models prone to overconfidence
- When calibrated probabilities matter (not just accuracy)
- Generally helps with 0.05-0.1 smoothing; higher can hurt

---

## 15. Early Stopping

### The Pattern

Stop training when validation loss stops improving to prevent overfitting:

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        self.counter += 1
        return self.counter >= self.patience  # Stop if patience exceeded
```

### Best Practice

Always save the model at the best validation loss, not the final epoch:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pt')
```

---

## 16. Logging and Monitoring

### What to Track

- **Training loss** (per batch and per epoch average)
- **Validation loss** (per epoch)
- **Learning rate** (especially with schedulers)
- **Gradient norm** (detect exploding/vanishing gradients)
- **Parameter statistics** (weight magnitudes per layer)

### Simple Logging Pattern

```python
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        loss = train_step(inputs, targets)
        running_loss += loss.item()

        if (i + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            print(f"Epoch {epoch}, Step {i+1}, Loss: {avg_loss:.4f}")
            running_loss = 0.0

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
```

### When to Save Checkpoints

- After each epoch (for resume capability)
- When validation metric improves (for best model)
- At fixed intervals for long training runs

---

## 17. Reproducibility

### The Full Reproducibility Recipe

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# For fully deterministic operations
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Caveats

- `torch.use_deterministic_algorithms(True)` may raise errors for ops without
  deterministic implementations
- Setting `benchmark = False` can slow down training
- DataLoader with `num_workers > 0` needs worker seeding:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dataloader = DataLoader(
    dataset,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(42),
)
```

---

## 18. Common Training Debugging

### Loss Not Decreasing

1. **Learning rate too high**: Loss oscillates wildly. Try 10x smaller.
2. **Learning rate too low**: Loss decreases extremely slowly. Try 10x larger.
3. **Bug in data pipeline**: Verify labels match inputs.
4. **Wrong loss function**: Ensure loss matches the task (e.g., CrossEntropy for
   classification needs raw logits, not softmax outputs).
5. **Model too small**: May lack capacity to fit even training data.

### NaN Loss

1. **Learning rate too high**: Gradients explode. Lower LR or add clipping.
2. **Division by zero**: Check for zero denominators in custom loss.
3. **Log of zero/negative**: Ensure inputs to `log()` are positive.
4. **Overflow in float16**: Use GradScaler or switch to bfloat16.

### Overfitting (Train Loss Low, Val Loss High)

1. Add regularization (dropout, weight decay)
2. Reduce model size
3. Add data augmentation
4. Use early stopping
5. Get more data

### Underfitting (Both Losses High)

1. Increase model capacity (more layers, wider layers)
2. Train longer
3. Reduce regularization
4. Check for bugs in the model architecture
5. Verify the task is learnable with this architecture

### Gradient Debugging

```python
# Check for vanishing/exploding gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm == 0:
            print(f"WARNING: Zero gradient in {name}")
        elif grad_norm > 100:
            print(f"WARNING: Large gradient in {name}: {grad_norm}")
```

---

## 19. Putting It All Together

A production-ready training loop combines many of these techniques:

```python
def train(model, train_loader, val_loader, config):
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    early_stop = EarlyStopping(patience=10)
    ema = EMA(model, decay=0.999)

    for epoch in range(config.epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss = loss / config.accumulation_steps

            loss.backward()

            if (i + 1) % config.accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

        scheduler.step()
        val_loss = evaluate(model, val_loader)

        if early_stop(val_loss):
            break
```

---

## Summary

| Technique              | Purpose                     | Memory Impact | Speed Impact |
|------------------------|-----------------------------|---------------|--------------|
| Mixed precision        | Faster math, less memory    | Reduces 50%   | 2-3x faster  |
| Gradient accumulation  | Simulate large batches      | No change     | Slight slow  |
| Gradient checkpointing | Reduce activation memory    | Saves 60-80%  | ~33% slower  |
| Gradient clipping      | Prevent exploding gradients | None          | Negligible   |
| EMA                    | Smoother final model        | 2x params     | Negligible   |
| Label smoothing        | Prevent overconfidence      | None          | None         |
| Early stopping         | Prevent overfitting         | None          | Saves time   |

---

<div align="center">

[← Previous Module](../06_data_loading/) | [🏠 Home](../README.md) | [Next Module →](../08_torch_compile/)

**[📓 Open Notebook](../notebooks/04_training_complete_guide.ipynb)** — Interactive version of this module

</div>
