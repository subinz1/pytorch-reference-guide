# Module 05: Optimizers and Learning Rate Schedulers

## Table of Contents
1. [Optimizer Fundamentals](#optimizer-fundamentals)
2. [SGD — Stochastic Gradient Descent](#sgd)
3. [Adam — Adaptive Moment Estimation](#adam)
4. [AdamW — Decoupled Weight Decay](#adamw)
5. [Other Optimizers](#other-optimizers)
6. [Learning Rate Schedulers](#learning-rate-schedulers)
7. [Practical Advice](#practical-advice)
8. [Gradient Clipping](#gradient-clipping)
9. [Compiled Optimizers](#compiled-optimizers)

---

## Optimizer Fundamentals

An optimizer updates model parameters to minimize the loss function. In PyTorch, all optimizers
inherit from `torch.optim.Optimizer` and share this interface:

```python
import torch.optim as optim

# Create optimizer — pass it the parameters to optimize
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()       # Clear old gradients
    output = model(batch)       # Forward pass
    loss = criterion(output, target)
    loss.backward()             # Compute gradients
    optimizer.step()            # Update parameters
```

### Parameter Groups

Optimizers support different settings for different parameter groups:

```python
optimizer = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 0.001},   # Lower LR for backbone
    {'params': model.head.parameters(), 'lr': 0.01},         # Higher LR for head
], momentum=0.9, weight_decay=1e-4)
```

### Optimizer State Dict

```python
# Save optimizer state (for training resumption)
torch.save(optimizer.state_dict(), 'optimizer.pth')

# Load optimizer state
optimizer.load_state_dict(torch.load('optimizer.pth'))
```

The state dict contains:
- `state`: per-parameter state (momentum buffers, Adam moments, step counts)
- `param_groups`: hyperparameters (lr, momentum, weight_decay, etc.)

---

## SGD

Stochastic Gradient Descent is the simplest optimizer but with momentum is still competitive.

### Vanilla SGD
Update rule: `theta = theta - lr * gradient`

### SGD with Momentum
Momentum accumulates past gradients to smooth updates and escape shallow local minima.

**Classical momentum:**
```
v_t = momentum * v_{t-1} + gradient_t
theta_t = theta_{t-1} - lr * v_t
```

**Nesterov momentum (look-ahead):**
```
v_t = momentum * v_{t-1} + gradient(theta - lr * momentum * v_{t-1})
theta_t = theta_{t-1} - lr * v_t
```

Nesterov is generally better — it evaluates the gradient at the "look-ahead" position,
which provides better correction.

### SGD with Weight Decay (L2 Regularization)
```
gradient_with_wd = gradient + weight_decay * theta
theta = theta - lr * gradient_with_wd
```

Weight decay adds a penalty proportional to parameter magnitude, pushing weights toward zero.

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

**When to use SGD:**
- Computer vision tasks (often gives better generalization than Adam)
- When you have a good learning rate schedule
- Large batch training
- When you want maximum control

---

## Adam

Adam (Adaptive Moment Estimation) adapts the learning rate for each parameter based
on first and second moments of the gradients.

### Algorithm Step-by-Step

```
Initialize: m_0 = 0, v_0 = 0, t = 0

For each step:
  t = t + 1
  g_t = gradient at step t
  
  # Update biased first moment estimate (mean of gradients)
  m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
  
  # Update biased second moment estimate (mean of squared gradients)
  v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
  
  # Bias correction (crucial in early steps)
  m_hat_t = m_t / (1 - beta1^t)
  v_hat_t = v_t / (1 - beta2^t)
  
  # Update parameters
  theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

**Why bias correction?** Since m_0 = 0 and v_0 = 0, the estimates are biased toward
zero in early training. Dividing by (1 - beta^t) corrects this — as t grows large,
the correction approaches 1 and has no effect.

**Default hyperparameters:**
- `lr = 0.001`
- `beta1 = 0.9` (momentum for first moment)
- `beta2 = 0.999` (momentum for second moment)
- `eps = 1e-8` (numerical stability)

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0  # This is L2 regularization, NOT decoupled
)
```

**When to use Adam:**
- Default choice for most tasks
- NLP, transformers
- When you don't want to tune the learning rate carefully
- Faster convergence early in training

---

## AdamW

AdamW fixes a subtle but important problem with Adam's weight decay implementation.

### The Problem with Adam + Weight Decay

In standard Adam with weight_decay, the weight decay is applied to the gradient:
```
g_t = gradient + weight_decay * theta   # L2 regularization added to gradient
```

But then Adam's adaptive learning rate scales this differently per parameter, effectively
applying DIFFERENT regularization strengths to different parameters. This breaks the
intended uniform regularization.

### Decoupled Weight Decay (AdamW)

AdamW applies weight decay directly to the parameters AFTER the Adam update:
```
theta_t = theta_{t-1} - lr * adam_update - lr * weight_decay * theta_{t-1}
```

This means every parameter gets the same relative shrinkage regardless of its gradient history.

**Impact:** AdamW generalizes better than Adam+L2, especially for transformers and
large models. It's now the default for most modern training.

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # Decoupled! Common value: 0.01 to 0.1
)
```

**When to use AdamW:**
- Transformers (GPT, BERT, ViT, etc.)
- When using weight decay (almost always)
- Default recommendation for most tasks today

---

## Other Optimizers

### Adagrad
Adapts learning rate based on accumulated squared gradients. Good for sparse data
but learning rate decays to zero over time (problematic for long training).

```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

### RMSprop
Fixes Adagrad's decaying learning rate by using exponential moving average of squared gradients.
Predecessor to Adam (Adam = RMSprop + momentum).

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
```

### Adadelta
Similar to RMSprop but eliminates the need to set an initial learning rate.

```python
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9)
```

### LBFGS
Limited-memory BFGS — a quasi-Newton method. Uses second-order information (Hessian approximation).
Much more expensive per step but converges in fewer steps. Requires a closure.

```python
optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)

def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

### RAdam (Rectified Adam)
Adam with a variance-rectification term that provides an automatic warmup effect.
Addresses the high variance of Adam in early training.

### Muon
A newer optimizer designed for large language models. Uses momentum and sign-based
updates for more stable training at scale.

### Adafactor
Memory-efficient alternative to Adam. Factorizes the second moment matrix to reduce
memory from O(mn) to O(m+n). Popular for training very large models.

---

## Learning Rate Schedulers

Schedulers adjust the learning rate during training. The general pattern:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # Update LR after each epoch
```

### StepLR
Decay by `gamma` every `step_size` epochs.
```python
# LR: 0.1 -> 0.01 (at epoch 30) -> 0.001 (at epoch 60)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### MultiStepLR
Decay at specific milestones.
```python
# LR: 0.1 -> 0.01 (at epoch 30) -> 0.001 (at epoch 80)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

### ExponentialLR
Multiply LR by `gamma` every epoch.
```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### CosineAnnealingLR
Smoothly decays LR following a cosine curve from initial LR to `eta_min`.
```python
# Decays from lr to eta_min over T_max epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

### OneCycleLR
Implements the 1-cycle policy: ramp up LR, then decay. Often achieves best results.
```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=1000,
    pct_start=0.3,  # 30% warmup, 70% decay
    anneal_strategy='cos'
)
# Note: step() after each BATCH, not each epoch!
```

### ReduceLROnPlateau
Reduce LR when a metric plateaus (most practical for validation loss).
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
# Must pass the metric value:
scheduler.step(val_loss)
```

### LinearLR
Linearly scale LR from `start_factor` to `end_factor` over `total_iters`.
```python
# Warmup: LR goes from 0.001*0.1 to 0.001 over 10 epochs
scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
)
```

### SequentialLR (Warmup + Cosine)
Chain multiple schedulers together:
```python
warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[10]
)
```

### CosineAnnealingWarmRestarts
Cosine annealing with periodic restarts (warm restarts increase exploration).
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2  # First restart at 10, then 20, 40, ...
)
```

---

## Practical Advice

### Which Optimizer for Which Task?

| Task | Recommended | LR Range | Notes |
|------|-------------|----------|-------|
| Vision (CNNs) | SGD+momentum or AdamW | 0.01-0.1 | SGD often generalizes better |
| NLP/Transformers | AdamW | 1e-5 to 5e-4 | With cosine schedule + warmup |
| Fine-tuning | AdamW | 1e-5 to 3e-5 | Lower LR for pretrained weights |
| GANs | Adam (beta1=0.0) | 1e-4 to 2e-4 | Two separate optimizers |
| RL | Adam | 3e-4 | Simpler schedules |
| Small datasets | SGD+momentum | 0.01 | Better generalization |

### Learning Rate Selection

1. **Learning Rate Finder:** Start very small, increase exponentially, plot loss vs LR.
   Choose LR where loss is decreasing steepest (typically 1/10 of the minimum).
2. **Rule of thumb:** If training is unstable, reduce LR by 3-10x.
3. **Linear scaling rule:** When increasing batch size by N, multiply LR by N (with warmup).

### Warmup Strategies

Warmup is crucial for:
- Large learning rates
- Large batch sizes
- Transformer training

Common approach: Linear warmup for 5-10% of total training, then cosine decay.

---

## Gradient Clipping

Prevents exploding gradients by limiting gradient magnitude.

### clip_grad_norm_ (recommended)
Clips the total norm of all gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This scales all gradients uniformly to ensure the total L2 norm doesn't exceed `max_norm`.
It preserves gradient direction.

### clip_grad_value_
Clips each gradient element independently:
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Each element is clamped to [-clip_value, clip_value]. Can change gradient direction.

### Usage in training loop:
```python
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    # Clip AFTER backward(), BEFORE step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## Compiled Optimizers

PyTorch 2.0+ can compile optimizers with `torch.compile` for significant speedups:

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001)
# The optimizer step is fused and optimized
# This happens automatically when the model is compiled

@torch.compile
def train_step(model, x, y):
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss
```

Benefits:
- Fused kernels: multiple optimizer ops become one kernel launch
- Reduced memory traffic
- Horizontal fusion across parameter groups
- Can give 10-20% speedup on optimizer step

---

## Summary

| Optimizer | Key Feature | Best For |
|-----------|------------|----------|
| SGD+momentum | Simple, good generalization | Vision, large-scale |
| Adam | Adaptive per-param LR | General, fast convergence |
| AdamW | Proper weight decay | Transformers, modern default |
| Adagrad | Adapts to sparse features | NLP with sparse embeddings |
| RMSprop | Fixes Adagrad decay | RNNs (historically) |
| LBFGS | Second-order | Small problems, fine-tuning |

| Scheduler | Pattern | Best For |
|-----------|---------|----------|
| CosineAnnealing | Smooth decay to 0 | Most tasks |
| OneCycleLR | Warmup + decay | Fastest convergence |
| ReduceLROnPlateau | Adaptive decay | When you have val metric |
| Sequential(Linear+Cosine) | Warmup + cosine | Transformers |
| CosineWarmRestarts | Periodic resets | Long training, exploration |
