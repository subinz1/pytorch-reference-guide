"""
Module 04: Neural Network Basics — Complete nn.Module Tutorial
==============================================================
This file covers everything about nn.Module: creating custom modules,
parameters, buffers, module traversal, container modules, and the
complete module lifecycle.

Run: python module_basics.py
"""

import torch
import torch.nn as nn

print("=" * 70)
print("PART 1: Creating a Basic nn.Module")
print("=" * 70)

# The simplest possible neural network module


class SimpleLinearModel(nn.Module):
    """A basic two-layer neural network."""

    def __init__(self, input_size, hidden_size, output_size):
        # MUST call parent __init__ first
        super().__init__()
        # Assigning nn.Module instances as attributes auto-registers them
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Define the forward pass. Never call this directly — use model(x)."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


model = SimpleLinearModel(784, 256, 10)
print(f"\nModel architecture:\n{model}")
print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters())}")

# Using the model (calls __call__ which calls forward plus hooks)
x = torch.randn(32, 784)  # Batch of 32, 784 features
output = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")

print("\n" + "=" * 70)
print("PART 2: Parameters — Learnable Weights")
print("=" * 70)


class CustomLinear(nn.Module):
    """Implementing nn.Linear from scratch to understand Parameters."""

    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        # nn.Parameter wraps a tensor and marks it as learnable
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            # register_parameter with None makes it explicit that bias doesn't exist
            self.register_parameter("bias", None)

    def forward(self, x):
        # y = xW^T + b
        output = x @ self.weight.t()
        if self.bias is not None:
            output = output + self.bias
        return output


custom_layer = CustomLinear(10, 5)
print("\nCustom linear layer parameters:")
for name, param in custom_layer.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# Demonstrate that parameters are tracked
print(f"\nTotal parameters: {sum(p.numel() for p in custom_layer.parameters())}")

# Parameters vs regular tensors
print("\n--- What happens with regular tensors (NOT Parameters) ---")


class BrokenModule(nn.Module):
    """This module has a 'weight' that won't be tracked!"""

    def __init__(self):
        super().__init__()
        # This is NOT registered as a parameter — optimizer won't update it!
        self.weight = torch.randn(5, 3)  # Just a regular tensor
        # This IS registered
        self.proper_weight = nn.Parameter(torch.randn(5, 3))

    def forward(self, x):
        return x @ self.weight.t() + x @ self.proper_weight.t()


broken = BrokenModule()
print(f"Number of tracked parameters: {sum(1 for _ in broken.parameters())}")
print("Only 'proper_weight' is tracked — 'weight' is invisible to the optimizer!")

print("\n" + "=" * 70)
print("PART 3: Buffers — Non-learnable State")
print("=" * 70)


class RunningMeanModule(nn.Module):
    """Demonstrates buffers: state that's saved but not optimized."""

    def __init__(self, num_features):
        super().__init__()
        # Persistent buffer: included in state_dict, moved with .to(device)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("num_batches_seen", torch.tensor(0, dtype=torch.long))
        # Non-persistent buffer: moved with .to() but NOT saved in state_dict
        self.register_buffer("temp_storage", torch.zeros(num_features), persistent=False)
        # Learnable parameter for comparison
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            self.num_batches_seen += 1
            # Update running mean with exponential moving average
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
        return (x - self.running_mean) * self.scale


module = RunningMeanModule(5)
print("\nBuffers:")
for name, buf in module.named_buffers():
    print(f"  {name}: shape={buf.shape}, dtype={buf.dtype}")

print("\nParameters (should NOT include buffers):")
for name, param in module.named_parameters():
    print(f"  {name}: shape={param.shape}")

# Demonstrate buffer behavior
x = torch.randn(32, 5)
_ = module(x)
print(f"\nAfter one forward pass:")
print(f"  running_mean: {module.running_mean[:3]}...")
print(f"  num_batches_seen: {module.num_batches_seen}")

# State dict shows buffers (persistent ones)
print(f"\nState dict keys: {list(module.state_dict().keys())}")
print("Note: 'temp_storage' is NOT in state_dict (non-persistent)")

print("\n" + "=" * 70)
print("PART 4: Train vs Eval Mode")
print("=" * 70)


class TrainEvalDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


demo = TrainEvalDemo()
x = torch.randn(32, 10)

# Training mode
demo.train()
print(f"\nTraining mode: {demo.training}")
out_train1 = demo(x)
out_train2 = demo(x)
print(f"Same input, different outputs (due to dropout): "
      f"{torch.allclose(out_train1, out_train2)}")

# Eval mode
demo.eval()
print(f"\nEval mode: {demo.training}")
with torch.no_grad():
    out_eval1 = demo(x)
    out_eval2 = demo(x)
print(f"Same input, same outputs (dropout disabled): "
      f"{torch.allclose(out_eval1, out_eval2)}")

print("\n" + "=" * 70)
print("PART 5: Module Traversal — Named Parameters and Modules")
print("=" * 70)


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 32)
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        return self.norm(self.linear(x))


class ParentModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 64)
        self.block1 = SubModule()
        self.block2 = SubModule()
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


parent = ParentModule()

print("\n--- named_modules() — All modules in the tree ---")
for name, module in parent.named_modules():
    print(f"  '{name}': {type(module).__name__}")

print("\n--- named_children() — Direct children only ---")
for name, module in parent.named_children():
    print(f"  '{name}': {type(module).__name__}")

print("\n--- named_parameters() — All parameters with full path ---")
for name, param in parent.named_parameters():
    print(f"  {name}: {param.shape}")

print("\n" + "=" * 70)
print("PART 6: Container Modules")
print("=" * 70)

# --- nn.Sequential ---
print("\n--- nn.Sequential ---")
seq_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
print(f"Sequential model:\n{seq_model}")
output = seq_model(torch.randn(16, 784))
print(f"Output shape: {output.shape}")

# Access layers by index
print(f"First layer: {seq_model[0]}")

# Named Sequential using OrderedDict
from collections import OrderedDict

named_seq = nn.Sequential(
    OrderedDict(
        [
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(784, 256)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(256, 10)),
        ]
    )
)
print(f"\nNamed sequential: {named_seq}")

# --- nn.ModuleList ---
print("\n--- nn.ModuleList ---")


class ResidualStack(nn.Module):
    """Stack of residual blocks using ModuleList."""

    def __init__(self, num_blocks, hidden_dim):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        return x


res_stack = ResidualStack(3, 64)
out = res_stack(torch.randn(8, 64))
print(f"ResidualStack output shape: {out.shape}")
print(f"Number of blocks: {len(res_stack.blocks)}")

# --- nn.ModuleDict ---
print("\n--- nn.ModuleDict ---")


class ConfigurableModel(nn.Module):
    """Model where you can select which activation and normalization to use."""

    def __init__(self, input_dim, hidden_dim, activation="relu", norm="layer"):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.activations = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "silu": nn.SiLU(),
            }
        )
        self.norms = nn.ModuleDict(
            {
                "layer": nn.LayerNorm(hidden_dim),
                "batch": nn.BatchNorm1d(hidden_dim),
            }
        )
        self.act_name = activation
        self.norm_name = norm

    def forward(self, x):
        x = self.linear(x)
        x = self.norms[self.norm_name](x)
        x = self.activations[self.act_name](x)
        return x


config_model = ConfigurableModel(784, 256, activation="gelu", norm="layer")
out = config_model(torch.randn(16, 784))
print(f"ConfigurableModel output: {out.shape}")

# --- nn.ParameterList and nn.ParameterDict ---
print("\n--- nn.ParameterList and nn.ParameterDict ---")


class MixtureOfExperts(nn.Module):
    """Simple mixture of experts using ParameterList."""

    def __init__(self, input_dim, output_dim, num_experts):
        super().__init__()
        self.expert_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(output_dim, input_dim) * 0.01) for _ in range(num_experts)]
        )
        self.expert_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(output_dim)) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Compute gating weights
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        # Compute each expert's output
        expert_outputs = []
        for w, b in zip(self.expert_weights, self.expert_biases):
            expert_outputs.append(x @ w.t() + b)
        # Stack and weight by gate
        stacked = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, output_dim)
        output = (gate_scores.unsqueeze(-1) * stacked).sum(dim=1)
        return output


moe = MixtureOfExperts(64, 32, num_experts=4)
out = moe(torch.randn(8, 64))
print(f"MixtureOfExperts output: {out.shape}")
print(f"Total parameters: {sum(p.numel() for p in moe.parameters())}")

print("\n" + "=" * 70)
print("PART 7: Moving Models — .to(), .cpu(), .cuda()")
print("=" * 70)

model = SimpleLinearModel(784, 256, 10)

# Move to specific dtype
model_float16 = SimpleLinearModel(784, 256, 10).to(dtype=torch.float16)
print(f"\nFloat16 model weight dtype: {model_float16.layer1.weight.dtype}")

# Move to specific device (CPU example)
model_cpu = model.to("cpu")
print(f"Model device: {next(model_cpu.parameters()).device}")

# Check if all parameters are on the same device
devices = {p.device for p in model.parameters()}
print(f"All parameters on devices: {devices}")

print("\n" + "=" * 70)
print("PART 8: Freezing Parameters")
print("=" * 70)


class FineTuneModel(nn.Module):
    """Demonstrates freezing a pretrained backbone."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.head = nn.Linear(128, 10)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


ft_model = FineTuneModel()
print(f"\nBefore freezing — trainable params: "
      f"{sum(p.numel() for p in ft_model.parameters() if p.requires_grad)}")

ft_model.freeze_backbone()
print(f"After freezing backbone — trainable params: "
      f"{sum(p.numel() for p in ft_model.parameters() if p.requires_grad)}")
print("Only the classification head is trainable now!")

ft_model.unfreeze_backbone()
print(f"After unfreezing — trainable params: "
      f"{sum(p.numel() for p in ft_model.parameters() if p.requires_grad)}")

print("\n" + "=" * 70)
print("PART 9: Custom Forward with Multiple Inputs/Outputs")
print("=" * 70)


class MultiInputMultiOutput(nn.Module):
    """A model that takes multiple inputs and returns multiple outputs."""

    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Linear(100, 64)
        self.image_encoder = nn.Linear(2048, 64)
        self.fusion = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 10)
        self.regressor = nn.Linear(64, 1)

    def forward(self, text_features, image_features):
        text_emb = torch.relu(self.text_encoder(text_features))
        img_emb = torch.relu(self.image_encoder(image_features))
        # Concatenate embeddings
        fused = torch.cat([text_emb, img_emb], dim=-1)
        fused = torch.relu(self.fusion(fused))
        # Multiple outputs
        class_logits = self.classifier(fused)
        regression_output = self.regressor(fused)
        return class_logits, regression_output


multi_model = MultiInputMultiOutput()
text = torch.randn(8, 100)
image = torch.randn(8, 2048)
logits, reg = multi_model(text, image)
print(f"\nMulti-input/output model:")
print(f"  Classification output: {logits.shape}")
print(f"  Regression output: {reg.shape}")

print("\n" + "=" * 70)
print("PART 10: Printing Model Summary")
print("=" * 70)


def model_summary(model, input_size=None):
    """Print a summary of model parameters."""
    print(f"\nModel: {model.__class__.__name__}")
    print("-" * 60)
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"Model size (MB):      {total_params * 4 / 1024 / 1024:.2f}")  # float32


big_model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
model_summary(big_model)

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
