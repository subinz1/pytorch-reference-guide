"""
Module 04: Weight Initialization Strategies
=============================================
Proper initialization is crucial for training deep networks.
This file covers all major initialization methods, the math behind them,
and practical guidance.

Run: python weight_initialization.py
"""

import torch
import torch.nn as nn
import math

print("=" * 70)
print("PART 1: Why Initialization Matters")
print("=" * 70)

print("""
The Problem:
- If weights are too large: activations explode, gradients explode
- If weights are too small: activations vanish, gradients vanish
- Goal: keep variance of activations roughly constant across layers

Without proper init, a 50-layer network can have:
- Exploding: activations grow to 10^15 by the last layer
- Vanishing: activations shrink to 10^-15 by the last layer
""")

# Demonstrate the problem
print("--- Demonstrating Exploding/Vanishing Activations ---")
torch.manual_seed(42)

# Bad initialization: too large
print("\nToo large (std=2.0):")
x = torch.randn(256, 512)
for i in range(10):
    w = torch.randn(512, 512) * 2.0  # Too large!
    x = torch.relu(x @ w)
    if x.std().item() > 1e10 or x.std().item() == 0:
        print(f"  Layer {i+1}: std = {x.std().item():.2e} — EXPLODED!")
        break
    print(f"  Layer {i+1}: std = {x.std().item():.4e}")

# Bad initialization: too small
print("\nToo small (std=0.01):")
x = torch.randn(256, 512)
for i in range(10):
    w = torch.randn(512, 512) * 0.01  # Too small!
    x = torch.relu(x @ w)
    print(f"  Layer {i+1}: std = {x.std().item():.4e}")

# Kaiming initialization: just right
print("\nKaiming (std=sqrt(2/fan_in)):")
x = torch.randn(256, 512)
for i in range(10):
    w = torch.randn(512, 512) * math.sqrt(2.0 / 512)  # Kaiming for ReLU
    x = torch.relu(x @ w)
    print(f"  Layer {i+1}: std = {x.std().item():.4f}")

print("\n" + "=" * 70)
print("PART 2: Xavier (Glorot) Initialization")
print("=" * 70)

print("""
Designed for: sigmoid, tanh activations (symmetric, linear near 0)
Goal: Var(output) = Var(input) for both forward and backward pass

Derivation:
  For y = Wx (no activation), Var(y) = n_in * Var(w) * Var(x)
  To keep Var(y) = Var(x), we need Var(w) = 1/n_in
  For backward pass, we need Var(w) = 1/n_out
  Compromise: Var(w) = 2/(n_in + n_out)

Uniform: W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
Normal:  W ~ N(0, sqrt(2/(fan_in+fan_out)))
""")

# Xavier Uniform
print("\n--- Xavier Uniform ---")
layer = nn.Linear(512, 256)
nn.init.xavier_uniform_(layer.weight)
bound = math.sqrt(6.0 / (512 + 256))
print(f"fan_in={512}, fan_out={256}")
print(f"Theoretical bound: +/-{bound:.4f}")
print(f"Actual min: {layer.weight.min().item():.4f}, max: {layer.weight.max().item():.4f}")
print(f"Actual std: {layer.weight.std().item():.4f}")

# Xavier Normal
print("\n--- Xavier Normal ---")
nn.init.xavier_normal_(layer.weight)
expected_std = math.sqrt(2.0 / (512 + 256))
print(f"Expected std: {expected_std:.4f}")
print(f"Actual std:   {layer.weight.std().item():.4f}")

# Demonstrate Xavier with tanh network
print("\n--- Xavier + Tanh activation (10 layers) ---")
x = torch.randn(256, 512)
for i in range(10):
    w = torch.empty(512, 512)
    nn.init.xavier_normal_(w)
    x = torch.tanh(x @ w)
    print(f"  Layer {i+1}: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

print("\n" + "=" * 70)
print("PART 3: Kaiming (He) Initialization")
print("=" * 70)

print("""
Designed for: ReLU activations (and variants)
Key insight: ReLU zeros out ~half the values, so variance drops by factor 2

Derivation:
  After ReLU, Var(output) = (1/2) * n_in * Var(w) * Var(input)
  To keep Var(output) = Var(input): Var(w) = 2/n_in (fan_in mode)
  For backward: Var(w) = 2/n_out (fan_out mode)

Uniform: W ~ U(-sqrt(6/fan_in), sqrt(6/fan_in))  [for ReLU]
Normal:  W ~ N(0, sqrt(2/fan_in))  [for ReLU]

For LeakyReLU with slope a: multiply by 2/(1+a^2)
""")

# Kaiming Uniform
print("\n--- Kaiming Uniform (fan_in, relu) ---")
layer = nn.Linear(512, 256)
nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
bound = math.sqrt(6.0 / 512)
print(f"Theoretical bound: +/-{bound:.4f}")
print(f"Actual min: {layer.weight.min().item():.4f}, max: {layer.weight.max().item():.4f}")

# Kaiming Normal
print("\n--- Kaiming Normal (fan_in, relu) ---")
nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
expected_std = math.sqrt(2.0 / 512)
print(f"Expected std: {expected_std:.4f}")
print(f"Actual std:   {layer.weight.std().item():.4f}")

# fan_in vs fan_out
print("\n--- fan_in vs fan_out ---")
print("fan_in:  preserves variance in FORWARD pass (default, recommended)")
print("fan_out: preserves variance in BACKWARD pass")
layer_fi = nn.Linear(512, 256)
layer_fo = nn.Linear(512, 256)
nn.init.kaiming_normal_(layer_fi.weight, mode="fan_in", nonlinearity="relu")
nn.init.kaiming_normal_(layer_fo.weight, mode="fan_out", nonlinearity="relu")
print(f"fan_in std:  {layer_fi.weight.std().item():.4f} (based on 512)")
print(f"fan_out std: {layer_fo.weight.std().item():.4f} (based on 256)")

# For LeakyReLU
print("\n--- Kaiming for LeakyReLU ---")
nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.2)
expected_std = math.sqrt(2.0 / (1 + 0.2**2) / 512)
print(f"Expected std (a=0.2): {expected_std:.4f}")
print(f"Actual std: {layer.weight.std().item():.4f}")

# Demonstrate Kaiming with ReLU network
print("\n--- Kaiming + ReLU activation (10 layers) ---")
x = torch.randn(256, 512)
for i in range(10):
    w = torch.empty(512, 512)
    nn.init.kaiming_normal_(w, nonlinearity="relu")
    x = torch.relu(x @ w)
    print(f"  Layer {i+1}: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

print("\n" + "=" * 70)
print("PART 4: Other Initialization Methods")
print("=" * 70)

layer = nn.Linear(256, 128)

# --- Uniform ---
print("\n--- Uniform Distribution ---")
nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
print(f"Uniform[-0.1, 0.1]: min={layer.weight.min().item():.4f}, "
      f"max={layer.weight.max().item():.4f}")

# --- Normal ---
print("\n--- Normal Distribution ---")
nn.init.normal_(layer.weight, mean=0.0, std=0.02)
print(f"Normal(0, 0.02): mean={layer.weight.mean().item():.5f}, "
      f"std={layer.weight.std().item():.4f}")

# --- Constant ---
print("\n--- Constant ---")
nn.init.constant_(layer.bias, 0.0)
print(f"Constant(0): all zeros = {(layer.bias == 0).all().item()}")

# --- Zeros and Ones ---
print("\n--- Zeros and Ones ---")
nn.init.zeros_(layer.bias)
print(f"Zeros: {layer.bias[:5]}")

bn_layer = nn.BatchNorm1d(128)
nn.init.ones_(bn_layer.weight)
print(f"Ones (for BatchNorm gamma): {bn_layer.weight[:5]}")

# --- Orthogonal ---
print("\n--- Orthogonal Initialization ---")
print("Preserves norm of the input: ||Wx|| = ||x||")
print("Particularly good for RNNs")
rnn_weight = torch.empty(256, 256)
nn.init.orthogonal_(rnn_weight, gain=1.0)
# Verify orthogonality: W^T W should be identity
identity_approx = rnn_weight.t() @ rnn_weight
print(f"W^T * W diagonal mean: {identity_approx.diag().mean().item():.4f} (should be ~1.0)")
print(f"W^T * W off-diagonal mean: {(identity_approx - torch.eye(256)).abs().mean().item():.6f} "
      f"(should be ~0.0)")

# --- Sparse ---
print("\n--- Sparse Initialization ---")
sparse_weight = torch.empty(256, 256)
nn.init.sparse_(sparse_weight, sparsity=0.9, std=0.01)
zero_fraction = (sparse_weight == 0).float().mean().item()
print(f"Sparsity=0.9: actual zero fraction = {zero_fraction:.2f}")

# --- Eye (Identity) ---
print("\n--- Eye (Identity) Initialization ---")
print("For residual connections: starts as identity mapping")
identity_weight = torch.empty(128, 128)
nn.init.eye_(identity_weight)
print(f"Diagonal sum: {identity_weight.diag().sum().item():.0f} (should be 128)")

# --- Dirac ---
print("\n--- Dirac Initialization ---")
print("For convolutions: starts as identity (preserves input)")
conv = nn.Conv2d(3, 3, 3, padding=1)
nn.init.dirac_(conv.weight)
x = torch.randn(1, 3, 8, 8)
y = conv(x)
print(f"Dirac conv output ≈ input: {torch.allclose(x, y, atol=1e-5)}")

print("\n" + "=" * 70)
print("PART 5: Gain Values for Different Activations")
print("=" * 70)

print("\nGain adjusts the initialization based on the activation function:")
print(f"  {'Activation':<20} {'Gain':<10}")
print(f"  {'-'*20} {'-'*10}")
activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu']
for act in activations:
    try:
        gain = nn.init.calculate_gain(act)
        print(f"  {act:<20} {gain:<10.4f}")
    except ValueError:
        pass

# Using gain
print("\n--- Using gain with Xavier ---")
layer = nn.Linear(512, 256)
gain = nn.init.calculate_gain('relu')
nn.init.xavier_uniform_(layer.weight, gain=gain)
print(f"Xavier with ReLU gain ({gain:.4f}):")
print(f"  std = {layer.weight.std().item():.4f}")

print("\n" + "=" * 70)
print("PART 6: Practical Initialization Patterns")
print("=" * 70)


# Pattern 1: Initialize entire model with model.apply()
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 10)


def init_weights_cnn(module):
    """Kaiming init for conv/linear, ones/zeros for batchnorm."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


print("\n--- Pattern 1: model.apply(init_fn) ---")
model = CNN()
model.apply(init_weights_cnn)
print("Applied Kaiming init to all Conv2d and Linear layers")
print(f"conv1 weight std: {model.conv1.weight.std().item():.4f}")
print(f"fc weight std: {model.fc.weight.std().item():.4f}")
print(f"bn1 weight: all ones = {(model.bn1.weight == 1).all().item()}")


# Pattern 2: Initialize in __init__
class WellInitializedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Special init for last layer (smaller for stability)
        last_linear = self.layers[-1]
        nn.init.normal_(last_linear.weight, std=0.01)

    def forward(self, x):
        return self.layers(x)


print("\n--- Pattern 2: Initialize in __init__ ---")
model = WellInitializedModel(784, 256, 10)
print(f"Hidden layer std: {model.layers[0].weight.std().item():.4f}")
print(f"Output layer std: {model.layers[4].weight.std().item():.4f} (smaller for stability)")


# Pattern 3: Transformer initialization
class TransformerInit(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                       batch_first=True)
            for _ in range(num_layers)
        ])
        self._init_weights(num_layers)

    def _init_weights(self, num_layers):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


print("\n--- Pattern 3: Transformer initialization ---")
tf_model = TransformerInit(d_model=256, nhead=8, num_layers=6)
print("Transformer: Normal(0, 0.02) for Linear, ones/zeros for LayerNorm")

print("\n" + "=" * 70)
print("PART 7: Initialization Recommendations")
print("=" * 70)

print("""
Architecture     | Weight Init          | Bias Init   | Notes
-----------------|---------------------|-------------|---------------------------
CNN (ReLU)       | Kaiming Normal      | Zeros       | fan_out for conv layers
MLP (ReLU)       | Kaiming Normal      | Zeros       | fan_in for linear layers
MLP (tanh)       | Xavier Normal       | Zeros       | Symmetric activation
Transformer      | Normal(0, 0.02)     | Zeros       | GPT/BERT standard
BatchNorm        | Ones (gamma)        | Zeros (beta)| Already default
LayerNorm        | Ones (gamma)        | Zeros (beta)| Already default
Embedding        | Normal(0, 0.02)     | N/A         | Or uniform
RNN/LSTM         | Orthogonal          | Zeros       | Helps with gradients
Residual output  | Zeros               | Zeros       | Start as identity
Last classifier  | Normal(0, 0.01)     | Zeros       | Small for stability

General rules:
1. PyTorch defaults are usually good (Kaiming uniform for linear/conv)
2. For very deep networks, init matters more
3. With residual connections, init matters less
4. BatchNorm/LayerNorm reduce sensitivity to initialization
5. When in doubt, use Kaiming Normal for ReLU networks
""")

# Verify PyTorch defaults
print("--- PyTorch Default Initializations ---")
linear = nn.Linear(512, 256)
k = 1.0 / math.sqrt(512)
print(f"nn.Linear default: Uniform(-{k:.4f}, {k:.4f})")
print(f"  Actual range: [{linear.weight.min().item():.4f}, {linear.weight.max().item():.4f}]")

conv = nn.Conv2d(3, 64, 3)
k = 1.0 / math.sqrt(3 * 9)  # fan_in = in_channels * kernel_h * kernel_w
print(f"\nnn.Conv2d default: Kaiming Uniform (fan_in={3*9})")
print(f"  Actual std: {conv.weight.std().item():.4f}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
