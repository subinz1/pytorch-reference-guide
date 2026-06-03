"""
Neural Network Basics — nn.Module, Layers, and Loss Functions
==============================================================
Covers: nn.Module lifecycle, common layers, containers, loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("1. BASIC nn.Module")
print("=" * 60)

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Non-trainable state
        self.register_buffer('step_count', torch.tensor(0))

    def forward(self, x):
        x = self.dropout(self.act(self.bn(self.fc1(x))))
        self.step_count += 1
        return self.fc2(x)

model = SimpleNet(784, 256, 10)
print(f"Model:\n{model}")

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable:        {trainable:,}")

# Named parameters and buffers
print(f"\nParameter names: {[n for n, _ in model.named_parameters()]}")
print(f"Buffer names:    {[n for n, _ in model.named_buffers()]}")

# Forward pass
x = torch.randn(4, 784)
output = model(x)
print(f"\nInput:  {x.shape}")
print(f"Output: {output.shape}")
print(f"Steps:  {model.step_count}")

print("\n" + "=" * 60)
print("2. CONTAINERS")
print("=" * 60)

# Sequential
seq_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
print(f"Sequential output: {seq_model(torch.randn(4, 784)).shape}")

# ModuleList
class MultiLayerNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

ml_model = MultiLayerNet([784, 512, 256, 10])
print(f"ModuleList output: {ml_model(torch.randn(4, 784)).shape}")

# ModuleDict
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleDict({
            'small': nn.Linear(128, 64),
            'large': nn.Linear(128, 256),
        })

    def forward(self, x, expert_name):
        return self.experts[expert_name](x)

router = Router()
print(f"Router 'small': {router(torch.randn(4, 128), 'small').shape}")
print(f"Router 'large': {router(torch.randn(4, 128), 'large').shape}")

print("\n" + "=" * 60)
print("3. COMMON LAYERS")
print("=" * 60)

# Convolutional
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
x = torch.randn(1, 3, 32, 32)
print(f"Conv2d: {x.shape} -> {conv(x).shape}")

# Normalization
bn = nn.BatchNorm2d(64)
ln = nn.LayerNorm(512)
gn = nn.GroupNorm(8, 64)
rms = nn.RMSNorm(512)
print(f"BatchNorm2d: {bn(torch.randn(4, 64, 8, 8)).shape}")
print(f"LayerNorm:   {ln(torch.randn(4, 10, 512)).shape}")
print(f"RMSNorm:     {rms(torch.randn(4, 10, 512)).shape}")

# Pooling
pool = nn.AdaptiveAvgPool2d((1, 1))
print(f"AdaptiveAvgPool2d: {pool(torch.randn(4, 64, 8, 8)).shape}")

# Embedding
emb = nn.Embedding(50000, 512, padding_idx=0)
tokens = torch.randint(0, 50000, (4, 20))
print(f"Embedding: {tokens.shape} -> {emb(tokens).shape}")

print("\n" + "=" * 60)
print("4. ACTIVATION FUNCTIONS")
print("=" * 60)

x = torch.linspace(-3, 3, 7)
print(f"Input: {x}")
print(f"ReLU:    {F.relu(x)}")
print(f"GELU:    {F.gelu(x).round(decimals=3)}")
print(f"SiLU:    {F.silu(x).round(decimals=3)}")
print(f"Sigmoid: {torch.sigmoid(x).round(decimals=3)}")
print(f"Softmax: {F.softmax(x, dim=0).round(decimals=3)}")

print("\n" + "=" * 60)
print("5. LOSS FUNCTIONS")
print("=" * 60)

# Classification
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
ce_loss = F.cross_entropy(logits, targets)
print(f"Cross-entropy loss: {ce_loss:.4f}")

# Binary classification
binary_logits = torch.randn(4)
binary_targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
bce_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
print(f"BCE loss: {bce_loss:.4f}")

# Regression
pred = torch.randn(4, 1)
target = torch.randn(4, 1)
mse = F.mse_loss(pred, target)
l1 = F.l1_loss(pred, target)
huber = F.huber_loss(pred, target)
print(f"MSE loss:   {mse:.4f}")
print(f"L1 loss:    {l1:.4f}")
print(f"Huber loss: {huber:.4f}")

print("\n" + "=" * 60)
print("6. HOOKS")
print("=" * 60)

activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.shape
    return hook

model = SimpleNet(784, 256, 10)
model.fc1.register_forward_hook(save_activation('fc1'))
model.fc2.register_forward_hook(save_activation('fc2'))

model.eval()
with torch.no_grad():
    model(torch.randn(4, 784))

print("Captured activation shapes:")
for name, shape in activations.items():
    print(f"  {name}: {shape}")

print("\n" + "=" * 60)
print("7. SAVE & LOAD")
print("=" * 60)

import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'model.pt')

    # Save state dict
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

    # Load state dict
    model2 = SimpleNet(784, 256, 10)
    model2.load_state_dict(torch.load(path, weights_only=True))
    print("Loaded model successfully")

    # Verify
    x = torch.randn(4, 784)
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    print(f"Outputs match: {torch.allclose(out1, out2)}")

print("\nDone!")
