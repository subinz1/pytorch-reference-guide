# Module 04: Neural Networks in PyTorch

## Table of Contents
1. [What is nn.Module?](#what-is-nnmodule)
2. [Module Lifecycle](#module-lifecycle)
3. [Parameters vs Buffers](#parameters-vs-buffers)
4. [Container Modules](#container-modules)
5. [Linear Layers](#linear-layers)
6. [Convolution Layers](#convolution-layers)
7. [Pooling Layers](#pooling-layers)
8. [Normalization Layers](#normalization-layers)
9. [Activation Functions](#activation-functions)
10. [Dropout](#dropout)
11. [Recurrent Layers](#recurrent-layers)
12. [Transformer Layers](#transformer-layers)
13. [Embedding Layers](#embedding-layers)
14. [Loss Functions](#loss-functions)
15. [Functional API](#functional-api)
16. [Weight Initialization](#weight-initialization)
17. [Hooks](#hooks)
18. [State Dict](#state-dict)

---

## What is nn.Module?

`torch.nn.Module` is the base class for ALL neural network components in PyTorch. Every layer,
every model, every building block inherits from this class. When you build a neural network in
PyTorch, you create a class that inherits from `nn.Module`.

Think of `nn.Module` as a container that:
- Holds learnable parameters (weights and biases)
- Defines how data flows through the network (the `forward` method)
- Provides utilities for moving to GPU, saving/loading, switching train/eval modes
- Builds a tree of sub-modules (layers inside your model)

```python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # MUST call parent __init__
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

Key insight: When you assign an `nn.Module` or `nn.Parameter` as an attribute of your module
(in `__init__`), PyTorch automatically registers it. This means it will show up in
`.parameters()`, be moved when you call `.to(device)`, and be saved in `.state_dict()`.

---

## Module Lifecycle

### `__init__`: Construction

In `__init__`, you define all the layers and parameters your network needs. You MUST call
`super().__init__()` first. Any `nn.Module` or `nn.Parameter` assigned as an attribute
gets automatically registered.

```python
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # These are automatically registered as sub-modules
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # This is NOT registered (plain Python attribute)
        self.activation_name = "relu"
```

### `forward`: The Forward Pass

The `forward` method defines how input data flows through your network. You NEVER call
`forward()` directly — instead, you call the module as a function: `model(x)`. This is
because `__call__` does extra work (hooks, checks) before calling `forward`.

```python
def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = self.layer2(x)
    return x

# Correct usage:
output = model(input_tensor)  # calls __call__ which calls forward

# WRONG — never do this:
# output = model.forward(input_tensor)
```

### Train vs Eval Mode

Modules have two modes that affect behavior of certain layers (Dropout, BatchNorm):

```python
model.train()   # Sets training mode (dropout active, batchnorm uses batch stats)
model.eval()    # Sets evaluation mode (dropout disabled, batchnorm uses running stats)

# Check current mode
print(model.training)  # True or False
```

This is critical: forgetting `model.eval()` during inference causes incorrect results
because Dropout still drops neurons and BatchNorm uses batch statistics instead of
learned running statistics.

---

## Parameters vs Buffers

### Parameters

Parameters are tensors that require gradients and are updated by the optimizer.
They represent the learnable weights of your model.

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Manual parameter creation
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight.t() + self.bias
```

### Buffers

Buffers are tensors that are part of the module's state but do NOT require gradients.
They are saved in `state_dict()` and moved with `.to(device)`, but the optimizer
ignores them.

Common use cases:
- Running mean/variance in BatchNorm
- Fixed positional encodings
- Binary masks that don't change during training

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Buffer: saved in state_dict, moved with .to(), but NOT optimized
        self.register_buffer('running_mean', torch.zeros(10))
        # Non-persistent buffer: moved with .to() but NOT saved
        self.register_buffer('temp_mask', torch.ones(10), persistent=False)
```

### register_parameter and register_buffer

```python
class ExplicitRegistration(nn.Module):
    def __init__(self):
        super().__init__()
        # Explicit parameter registration (equivalent to self.weight = nn.Parameter(...))
        self.register_parameter('weight', nn.Parameter(torch.randn(5, 3)))
        # Can register None — useful for optional parameters
        self.register_parameter('optional_bias', None)
        # Buffer registration
        self.register_buffer('counter', torch.tensor(0))
```

### Traversing the Module Tree

```python
model = MyNetwork()

# All parameters (recursively)
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# All sub-modules (recursively)
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")

# Direct children only
for name, module in model.named_children():
    print(f"{name}: {type(module).__name__}")

# All buffers
for name, buf in model.named_buffers():
    print(f"{name}: shape={buf.shape}")
```

---

## Container Modules

### nn.Sequential

Chains modules in order. Input flows through each module sequentially.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
# Equivalent to calling each in order: output = layer3(relu(layer2(relu(layer1(x)))))
output = model(input_tensor)
```

Use when: Your network is a simple chain of operations with no branching.

### nn.ModuleList

A list of modules. Does NOT define a forward pass — you iterate manually.

```python
class MultiHeadModel(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(256, 10) for _ in range(num_heads)])
    
    def forward(self, x):
        return [head(x) for head in self.heads]
```

Use when: You need a variable number of layers that you'll iterate over yourself.
WARNING: A plain Python list `[]` will NOT register the modules!

### nn.ModuleDict

A dictionary of modules, accessed by string keys.

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(784, 256)
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(256, 10),
            'regression': nn.Linear(256, 1),
        })
    
    def forward(self, x, task):
        features = torch.relu(self.backbone(x))
        return self.heads[task](features)
```

Use when: You need named access to different sub-modules (multi-task, configurable architectures).

### nn.ParameterList and nn.ParameterDict

Same idea but for raw parameters instead of modules:

```python
class CustomModel(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.expert_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(256, 256)) for _ in range(num_experts)]
        )
        self.config_params = nn.ParameterDict({
            'scale': nn.Parameter(torch.ones(1)),
            'shift': nn.Parameter(torch.zeros(1)),
        })
```

---

## Linear Layers

### nn.Linear

Applies a linear transformation: `y = xW^T + b`

```python
# Input:  (batch_size, in_features)   e.g., (32, 784)
# Output: (batch_size, out_features)  e.g., (32, 256)
linear = nn.Linear(in_features=784, out_features=256, bias=True)

# Weight shape: (out_features, in_features) = (256, 784)
# Bias shape:   (out_features,) = (256,)
```

The math: For input x of shape (*, in_features), output y of shape (*, out_features):
`y_i = sum_j(x_j * W_ij) + b_i`

### nn.Bilinear

Applies a bilinear transformation: `y = x1^T A x2 + b`

```python
# Two inputs of potentially different sizes
bilinear = nn.Bilinear(in1_features=20, in2_features=30, out_features=40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = bilinear(input1, input2)  # shape: (128, 40)
```

### nn.LazyLinear

Infers `in_features` from the first input — useful for prototyping:

```python
lazy = nn.LazyLinear(out_features=256)
# in_features is determined on first forward pass
output = lazy(torch.randn(32, 784))  # Now it knows in_features=784
```

---

## Convolution Layers

### Core Concepts

Convolutions slide a kernel (filter) across the input, computing dot products at each position.

**Key parameters:**
- `kernel_size`: Size of the sliding window (e.g., 3 means 3x3 for Conv2d)
- `stride`: How far the kernel moves each step (default=1)
- `padding`: Zero-padding added to input borders
- `dilation`: Spacing between kernel elements (dilated/atrous convolution)
- `groups`: Split input channels into groups for grouped convolution

**Output size formula (for each spatial dimension):**
```
output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

### nn.Conv1d

For sequential/temporal data (text, audio, time series).

```python
# Input:  (batch, in_channels, length)      e.g., (32, 1, 100)
# Output: (batch, out_channels, new_length) e.g., (32, 16, 98)
conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
```

### nn.Conv2d

For image data. The most commonly used convolution.

```python
# Input:  (batch, in_channels, height, width)       e.g., (32, 3, 224, 224)
# Output: (batch, out_channels, new_height, new_width) e.g., (32, 64, 112, 112)
conv2d = nn.Conv2d(
    in_channels=3,       # RGB input
    out_channels=64,     # 64 filters
    kernel_size=3,       # 3x3 kernel
    stride=2,            # Downsample by 2
    padding=1            # Same padding for stride=1
)

# Weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
# = (64, 3, 3, 3)
```

### nn.Conv3d

For volumetric data (video, 3D medical images).

```python
# Input:  (batch, channels, depth, height, width)
conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
```

### ConvTranspose2d (Transposed Convolution)

Used for upsampling — goes from smaller to larger spatial dimensions.
Often called "deconvolution" (technically incorrect name).

```python
# Input:  (batch, in_channels, H, W)
# Output: (batch, out_channels, H*2, W*2) with stride=2
upsample = nn.ConvTranspose2d(
    in_channels=64, out_channels=32,
    kernel_size=4, stride=2, padding=1
)
```

### Depthwise Separable Convolution

A two-step convolution that's much more efficient:
1. Depthwise: Apply one filter per input channel (groups=in_channels)
2. Pointwise: 1x1 convolution to mix channels

```python
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        # Depthwise: each input channel gets its own filter
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size,
                                   padding=padding, groups=in_ch)
        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

---

## Pooling Layers

Pooling reduces spatial dimensions while retaining important information.

### MaxPool2d

Takes the maximum value in each pooling window:

```python
# Input:  (batch, channels, 224, 224)
# Output: (batch, channels, 112, 112)  — halves spatial dims
pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

### AvgPool2d

Takes the average value in each pooling window:

```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

### AdaptiveAvgPool2d — Global Average Pooling

Outputs a fixed spatial size regardless of input size. Setting output to (1,1)
gives "global average pooling" — commonly used before the final classifier.

```python
# No matter what spatial size comes in, output is (batch, channels, 1, 1)
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
# Then flatten: (batch, channels, 1, 1) -> (batch, channels)
```

This is the modern replacement for large fully-connected layers at the end of CNNs.

---

## Normalization Layers

### BatchNorm (nn.BatchNorm1d, BatchNorm2d)

Normalizes across the batch dimension. For each feature/channel:

**Formula:**
```
y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
```

Where `gamma` (weight) and `beta` (bias) are learnable parameters.

**Training behavior:** Uses batch mean and variance, updates running statistics.
**Eval behavior:** Uses stored running mean and variance (fixed).

```python
bn = nn.BatchNorm2d(num_features=64)  # 64 channels
# Maintains: running_mean, running_var (buffers), weight, bias (parameters)
```

**When to use:** CNNs with large batch sizes. Not suitable for batch_size=1 or
variable batch sizes (use LayerNorm or GroupNorm instead).

### LayerNorm

Normalizes across the feature dimensions (not the batch). Each sample is normalized
independently.

```python
# For a transformer with hidden_size=512
ln = nn.LayerNorm(normalized_shape=512)
# For image data: normalize over (C, H, W)
ln_image = nn.LayerNorm([64, 32, 32])
```

**When to use:** Transformers, RNNs, any case where batch stats are unreliable.

### GroupNorm

Splits channels into groups and normalizes within each group. A middle ground between
BatchNorm (all channels) and InstanceNorm (each channel separately).

```python
gn = nn.GroupNorm(num_groups=32, num_channels=256)
```

**When to use:** When batch size is small, or when you want BatchNorm-like behavior
without batch dependency.

### InstanceNorm

Normalizes each channel of each sample independently. Equivalent to GroupNorm with
num_groups = num_channels.

```python
inst_norm = nn.InstanceNorm2d(num_features=64)
```

**When to use:** Style transfer, image generation tasks.

### RMSNorm

Root Mean Square Layer Normalization — simpler than LayerNorm (no mean subtraction):

```python
rms_norm = nn.RMSNorm(normalized_shape=512)
```

**Formula:** `y = x / RMS(x) * gamma` where `RMS(x) = sqrt(mean(x^2) + eps)`

**When to use:** Modern LLMs (LLaMA, etc.) — slightly faster than LayerNorm.

---

## Activation Functions

Activations introduce non-linearity. Without them, stacking linear layers is
equivalent to a single linear layer.

### ReLU: `f(x) = max(0, x)`
The default choice. Simple, fast, but can "die" (output 0 for all inputs).
```python
nn.ReLU(inplace=False)  # inplace=True saves memory but can cause issues
```

### LeakyReLU: `f(x) = max(alpha*x, x)` (default alpha=0.01)
Prevents dying ReLU by allowing small negative slope.
```python
nn.LeakyReLU(negative_slope=0.01)
```

### PReLU: `f(x) = max(alpha*x, x)` where alpha is LEARNED
```python
nn.PReLU(num_parameters=1)  # One alpha per channel if num_parameters=num_channels
```

### GELU: `f(x) = x * Phi(x)` where Phi is the CDF of standard normal
Used in Transformers (BERT, GPT). Smooth approximation of ReLU.
```python
nn.GELU(approximate='none')  # 'tanh' for faster approximation
```

### SiLU/Swish: `f(x) = x * sigmoid(x)`
Smooth, non-monotonic. Used in EfficientNet, many modern architectures.
```python
nn.SiLU()
```

### Mish: `f(x) = x * tanh(softplus(x))`
Similar to SiLU but slightly different properties.
```python
nn.Mish()
```

### Sigmoid: `f(x) = 1 / (1 + exp(-x))`
Squashes to [0, 1]. Used for binary classification output, gates.
```python
nn.Sigmoid()
```

### Tanh: `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
Squashes to [-1, 1]. Used in RNN gates.
```python
nn.Tanh()
```

### Softmax: `f(x_i) = exp(x_i) / sum(exp(x_j))`
Outputs a probability distribution (sums to 1). Used for multi-class classification.
```python
nn.Softmax(dim=-1)  # Usually along the last dimension
```

---

## Dropout

Dropout randomly zeros elements during training to prevent overfitting.
During evaluation, dropout is disabled and outputs are unchanged.

**Key insight:** During training, remaining elements are scaled by `1/(1-p)` so that
expected values remain the same at test time (inverted dropout).

### nn.Dropout
```python
dropout = nn.Dropout(p=0.5)  # 50% of elements zeroed during training
```

### nn.Dropout2d
Drops entire channels (feature maps) for Conv2d outputs:
```python
dropout2d = nn.Dropout2d(p=0.1)  # Drops entire channels
```

### nn.AlphaDropout
For use with SELU activation — maintains self-normalizing property:
```python
alpha_dropout = nn.AlphaDropout(p=0.1)
```

---

## Recurrent Layers

### nn.RNN

Basic recurrent layer: `h_t = tanh(x_t W_ih^T + h_{t-1} W_hh^T + b)`

```python
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2,
             batch_first=True, bidirectional=False, dropout=0.1)
# Input:  (batch, seq_len, input_size)
# Output: (batch, seq_len, hidden_size * num_directions)
# Hidden: (num_layers * num_directions, batch, hidden_size)
output, h_n = rnn(input_seq, h_0)
```

### nn.LSTM

Long Short-Term Memory — solves vanishing gradients with gates:

**Gate equations:**
```
f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)    # Input gate
g_t = tanh(W_g [h_{t-1}, x_t] + b_g)       # Cell candidate
o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)    # Output gate
c_t = f_t * c_{t-1} + i_t * g_t            # Cell state
h_t = o_t * tanh(c_t)                       # Hidden state
```

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               batch_first=True, bidirectional=True, dropout=0.1)
output, (h_n, c_n) = lstm(input_seq)
# output shape: (batch, seq_len, hidden_size * 2) for bidirectional
```

### nn.GRU

Gated Recurrent Unit — simpler than LSTM with fewer parameters:

**Gate equations:**
```
r_t = sigmoid(W_r [h_{t-1}, x_t])  # Reset gate
z_t = sigmoid(W_z [h_{t-1}, x_t])  # Update gate
n_t = tanh(W_n [r_t * h_{t-1}, x_t])  # New gate
h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # Hidden state
```

```python
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2,
             batch_first=True, bidirectional=True)
output, h_n = gru(input_seq)
```

---

## Transformer Layers

### nn.MultiheadAttention

Computes scaled dot-product attention across multiple heads:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

```python
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1,
                            batch_first=True)
# Self-attention: query = key = value
attn_output, attn_weights = mha(query, key, value, key_padding_mask=mask)
```

### nn.TransformerEncoderLayer

One layer of a Transformer encoder (self-attention + feedforward):

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048,
    dropout=0.1, activation='gelu', batch_first=True,
    norm_first=True  # Pre-norm (more stable training)
)
```

### nn.TransformerEncoder

Stack of TransformerEncoderLayers:

```python
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
output = encoder(src, src_key_padding_mask=padding_mask)
```

---

## Embedding Layers

### nn.Embedding

Lookup table that maps integer indices to dense vectors:

```python
# vocab_size=10000, embedding_dim=256
embed = nn.Embedding(num_embeddings=10000, embedding_dim=256, padding_idx=0)
# Input: (batch, seq_len) of integer indices
# Output: (batch, seq_len, embedding_dim)
token_ids = torch.tensor([[1, 45, 234, 0, 0]])  # 0 = padding
embeddings = embed(token_ids)  # shape: (1, 5, 256)
```

`padding_idx`: The embedding at this index is always zero and is not updated.

### nn.EmbeddingBag

More efficient when you need to sum/mean/max embeddings (e.g., bag-of-words):

```python
embed_bag = nn.EmbeddingBag(num_embeddings=10000, embedding_dim=256, mode='mean')
# Returns one vector per "bag" — no need to manually average
```

---

## Loss Functions

### nn.CrossEntropyLoss
For multi-class classification. Combines LogSoftmax + NLLLoss.

**Formula:** `loss = -log(exp(x_y) / sum(exp(x_j)))` where y is the true class.

```python
criterion = nn.CrossEntropyLoss()
# logits: (batch, num_classes) — RAW scores, NOT softmax
# target: (batch,) — class indices (integers)
loss = criterion(logits, targets)
```

### nn.BCEWithLogitsLoss
For binary or multi-label classification. Combines Sigmoid + BCELoss.

**Formula:** `loss = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]`

```python
criterion = nn.BCEWithLogitsLoss()
# logits: (batch, num_labels) — RAW scores
# target: (batch, num_labels) — 0.0 or 1.0
```

### nn.MSELoss
Mean Squared Error for regression.

**Formula:** `loss = mean((y_pred - y_true)^2)`

```python
criterion = nn.MSELoss()
```

### nn.L1Loss
Mean Absolute Error for regression.

**Formula:** `loss = mean(|y_pred - y_true|)`

### nn.HuberLoss (Smooth L1)
Combination of L1 and L2 — less sensitive to outliers than MSE.

**Formula:** L2 for |error| < delta, L1 for |error| >= delta.

```python
criterion = nn.HuberLoss(delta=1.0)
```

### nn.KLDivLoss
Kullback-Leibler Divergence — measures how one distribution differs from another.

**Formula:** `loss = y_true * (log(y_true) - x)`

```python
criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
# Input must be log-probabilities!
```

### nn.TripletMarginLoss
For metric learning with (anchor, positive, negative) triplets.

**Formula:** `loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)`

```python
criterion = nn.TripletMarginLoss(margin=1.0)
loss = criterion(anchor, positive, negative)
```

### nn.CosineEmbeddingLoss
Measures cosine similarity between pairs.

```python
criterion = nn.CosineEmbeddingLoss(margin=0.0)
# target: +1 (similar) or -1 (dissimilar)
loss = criterion(x1, x2, target)
```

---

## Functional API

`torch.nn.functional` (commonly imported as `F`) provides the same operations as
`nn.Module` layers but as pure functions without stored state.

```python
import torch.nn.functional as F

# Module version (has stored parameters):
relu_module = nn.ReLU()
output = relu_module(x)

# Functional version (stateless):
output = F.relu(x)
```

**When to use Module vs Functional:**
- Use **Module** when the operation has learnable parameters (Linear, Conv, BatchNorm)
- Use **Module** when the operation has different train/eval behavior (Dropout, BatchNorm)
- Use **Functional** for stateless operations (relu, softmax in forward pass)
- Use **Functional** when you need the operation in a custom forward pass without
  wanting to register it as a sub-module

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)  # Module: has parameters
        self.bn = nn.BatchNorm2d(64)     # Module: has state (running stats)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)         # Functional: no state needed
        x = F.dropout(x, p=0.5, training=self.training)  # Must pass training flag!
        return x
```

---

## Weight Initialization

Proper initialization prevents vanishing/exploding gradients at the start of training.

### Xavier (Glorot) Initialization
Designed for sigmoid/tanh activations. Keeps variance constant across layers.

**Formula:** `W ~ Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))`

```python
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

### Kaiming (He) Initialization
Designed for ReLU activations. Accounts for the fact that ReLU zeros out half the inputs.

**Formula:** `W ~ Normal(0, sqrt(2/fan_in))`

```python
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### When to use each:
- **Xavier**: sigmoid, tanh activations
- **Kaiming**: ReLU, LeakyReLU activations
- **Normal/Uniform**: When you want simple random initialization
- **Zeros**: For biases (common default)
- **Ones**: For normalization layer weights

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)  # Recursively apply to all modules
```

---

## Hooks

Hooks let you inspect or modify intermediate values during forward/backward passes
without changing the model code.

### Forward Hook
Called after `forward()` completes. Receives (module, input, output).

```python
def print_output_shape(module, input, output):
    print(f"{module.__class__.__name__}: output shape = {output.shape}")

hook_handle = model.layer1.register_forward_hook(print_output_shape)
# Later: hook_handle.remove()
```

### Forward Pre-Hook
Called before `forward()`. Receives (module, input). Can modify the input.

```python
def modify_input(module, args):
    # args is a tuple of inputs
    return (args[0] * 2,)  # Double the input

handle = model.layer1.register_forward_pre_hook(modify_input)
```

### Backward Hook
Called during backward pass. Can inspect or modify gradients.

```python
def print_grad(module, grad_input, grad_output):
    print(f"Grad output norm: {grad_output[0].norm()}")

handle = model.layer1.register_full_backward_hook(print_grad)
```

### Common Use Cases:
- Feature extraction from intermediate layers
- Gradient visualization/debugging
- Gradient clipping per layer
- Activation statistics for debugging training

---

## State Dict

The state dict is an OrderedDict mapping parameter/buffer names to tensors.
It's the standard way to save and load models.

### Saving and Loading

```python
# Save
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = MyModel()  # Create model with same architecture
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
```

### Partial Loading (strict=False)

When architectures don't match exactly:

```python
# Load only matching keys, ignore missing/unexpected keys
state_dict = torch.load('pretrained.pth', weights_only=True)
model.load_state_dict(state_dict, strict=False)
```

### Inspecting State Dict

```python
state_dict = model.state_dict()
for key, tensor in state_dict.items():
    print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
```

### Modifying Before Loading

```python
# Remove prefix from keys (e.g., from DataParallel)
state_dict = torch.load('model.pth', weights_only=True)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  # Remove DataParallel prefix
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| nn.Module | Base class; register layers in __init__, compute in forward |
| Parameters | Learnable tensors (weights), updated by optimizer |
| Buffers | State tensors without gradients (running stats, masks) |
| Sequential | Simple chain of layers |
| Conv2d | Spatial feature extraction with kernel sliding |
| BatchNorm | Normalize across batch; different train/eval behavior |
| LayerNorm | Normalize across features; batch-independent |
| Dropout | Random zeroing during training only |
| LSTM | Recurrent with forget/input/output gates |
| Transformer | Self-attention + feedforward |
| CrossEntropyLoss | Multi-class classification standard |
| Kaiming init | Default for ReLU networks |
| Hooks | Inspect/modify without changing model code |
| state_dict | Standard save/load mechanism |
