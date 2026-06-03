# PyTorch Quick Reference Cheat Sheet

## Tensor Creation

```python
torch.tensor([1, 2, 3])           # From data
torch.zeros(3, 4)                 # Zeros
torch.ones(3, 4)                  # Ones
torch.randn(3, 4)                 # Normal(0,1)
torch.rand(3, 4)                  # Uniform(0,1)
torch.empty(3, 4)                 # Uninitialized
torch.eye(4)                      # Identity
torch.arange(0, 10, 2)            # Range
torch.linspace(0, 1, 100)         # Evenly spaced
torch.full((3, 4), 3.14)          # Fill value
torch.zeros_like(x)               # Same shape/device/dtype
```

## Tensor Operations

```python
x + y, x * y, x / y, x ** 2      # Arithmetic
x @ y, torch.matmul(x, y)         # Matrix multiply
torch.bmm(batch_x, batch_y)       # Batch matmul
x.sum(), x.mean(), x.max()        # Reductions
x.sum(dim=0), x.argmax(dim=-1)    # Along dimension
torch.cat([a, b], dim=0)          # Concatenate
torch.stack([a, b])               # Stack (new dim)
x.view(3, 4), x.reshape(3, 4)    # Reshape
x.transpose(0, 1), x.permute(...)# Rearrange
x.unsqueeze(0), x.squeeze()       # Add/remove dims
x.flatten(), x.flatten(1)         # Flatten
```

## Gradient / Autograd

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()                       # Compute gradients
x.grad                             # Access gradient

with torch.no_grad(): ...          # Disable tracking
with torch.inference_mode(): ...   # Faster inference
x.detach()                         # Detach from graph
```

## Neural Network (nn.Module)

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x)

model = Net()
model.to(device)
model.train() / model.eval()
model.parameters()
model.state_dict()
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', weights_only=True))
```

## Common Layers

```python
nn.Linear(in, out)                 # Fully connected
nn.Conv2d(C_in, C_out, K)         # Convolution
nn.BatchNorm2d(C)                  # Batch normalization
nn.LayerNorm(D)                    # Layer normalization
nn.RMSNorm(D)                     # RMS normalization
nn.Embedding(vocab, dim)           # Embedding lookup
nn.Dropout(p)                      # Dropout
nn.MultiheadAttention(dim, heads)  # Multi-head attention
```

## Loss Functions

```python
F.cross_entropy(logits, targets)                    # Classification
F.binary_cross_entropy_with_logits(logits, targets) # Binary
F.mse_loss(pred, target)                            # Regression
F.l1_loss(pred, target)                             # MAE
F.huber_loss(pred, target)                          # Smooth L1
```

## Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda')

for data, target in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        loss = F.cross_entropy(model(data), target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

## torch.compile

```python
model = torch.compile(model)                      # Default
model = torch.compile(model, mode="max-autotune") # Max perf
model = torch.compile(model, dynamic=True)        # Dynamic shapes
model = torch.compile(model, fullgraph=True)      # Error on breaks
```

## Distributed

```python
# DDP
model = DDP(model.to(rank), device_ids=[rank])

# FSDP2
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
fully_shard(model, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))

# DeviceMesh
mesh = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))

# Launch
# torchrun --nproc_per_node=4 train.py
```

## Export

```python
exported = torch.export.export(model, example_inputs)
torch.export.save(exported, "model.pt2")
loaded = torch.export.load("model.pt2")
```

## Device Management

```python
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.memory_allocated()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
```

## Reproducibility

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```
