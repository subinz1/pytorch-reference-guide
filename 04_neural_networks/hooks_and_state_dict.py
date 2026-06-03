"""
Module 04: Hooks and State Dict — Advanced Module Features
============================================================
Covers forward hooks, backward hooks, forward pre-hooks for
inspecting/modifying intermediate values, plus saving and loading
models with state_dict.

Run: python hooks_and_state_dict.py
"""

import torch
import torch.nn as nn
import tempfile
import os

print("=" * 70)
print("PART 1: Forward Hooks — Inspect Outputs")
print("=" * 70)

print("""
Forward hooks are called AFTER forward() completes.
Signature: hook(module, input, output) -> None or modified output
Use cases: feature extraction, debugging shapes, activation logging
""")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


model = SimpleModel()

# Example 1: Print shapes during forward pass
print("\n--- Example 1: Shape Logging Hook ---")
handles = []


def shape_hook(module, input, output):
    print(f"  {module.__class__.__name__}: input={input[0].shape} -> output={output.shape}")


for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.ReLU)):
        handles.append(module.register_forward_hook(shape_hook))

x = torch.randn(8, 784)
print("Forward pass with shape hooks:")
output = model(x)

# Clean up hooks
for h in handles:
    h.remove()
print("Hooks removed.")

# Example 2: Feature Extraction (saving intermediate activations)
print("\n--- Example 2: Feature Extraction ---")
features = {}


def save_features(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook


# Register hooks on layers we want features from
h1 = model.layer1.register_forward_hook(save_features("layer1"))
h2 = model.layer2.register_forward_hook(save_features("layer2"))

x = torch.randn(4, 784)
output = model(x)

print("Extracted features:")
for name, feat in features.items():
    print(f"  {name}: shape={feat.shape}")

h1.remove()
h2.remove()

# Example 3: Modify output (clamping activations)
print("\n--- Example 3: Modify Output (Clamping) ---")


def clamp_hook(module, input, output):
    return output.clamp(min=-1.0, max=1.0)


handle = model.layer3.register_forward_hook(clamp_hook)
x = torch.randn(4, 784)
output = model(x)
print(f"Output range after clamping: [{output.min().item():.4f}, {output.max().item():.4f}]")
handle.remove()

print("\n" + "=" * 70)
print("PART 2: Forward Pre-Hooks — Inspect/Modify Inputs")
print("=" * 70)

print("""
Forward pre-hooks are called BEFORE forward() executes.
Signature: hook(module, args) -> None or modified args
Use cases: input preprocessing, normalization, debugging
""")

# Example: Normalize inputs before a layer
print("\n--- Example: Input Normalization Pre-Hook ---")


def normalize_input_hook(module, args):
    x = args[0]
    normalized = (x - x.mean()) / (x.std() + 1e-8)
    return (normalized,)


handle = model.layer1.register_forward_pre_hook(normalize_input_hook)
x = torch.randn(4, 784) * 100 + 50  # Large mean and std
output = model(x)
print(f"Input mean: {x.mean().item():.2f}, std: {x.std().item():.2f}")
print("Pre-hook normalized input before layer1 processed it")
handle.remove()

# Example: Logging which branch is taken in a conditional model
print("\n--- Example: Input Shape Validation ---")


def validate_input(module, args):
    x = args[0]
    if x.dim() != 2:
        raise ValueError(f"Expected 2D input, got {x.dim()}D")
    if x.shape[1] != 784:
        raise ValueError(f"Expected 784 features, got {x.shape[1]}")


handle = model.layer1.register_forward_pre_hook(validate_input)
# This works fine:
output = model(torch.randn(4, 784))
print("Correct input: passed validation")
# This would raise an error:
try:
    output = model(torch.randn(4, 100))
except ValueError as e:
    print(f"Wrong input: caught error — '{e}'")
handle.remove()

print("\n" + "=" * 70)
print("PART 3: Backward Hooks — Inspect/Modify Gradients")
print("=" * 70)

print("""
Backward hooks are called during the backward pass.
Signature: hook(module, grad_input, grad_output) -> None or modified grad_input
Use cases: gradient analysis, gradient clipping per layer, debugging
""")

model = SimpleModel()

# Example 1: Monitor gradient norms
print("\n--- Example 1: Gradient Norm Monitoring ---")
grad_norms = {}


def gradient_hook(name):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_norms[name] = grad_output[0].norm().item()
    return hook


for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_full_backward_hook(gradient_hook(name))

x = torch.randn(8, 784)
output = model(x)
loss = output.sum()
loss.backward()

print("Gradient norms at each layer (during backward):")
for name, norm in grad_norms.items():
    print(f"  {name}: grad_output norm = {norm:.4f}")

# Example 2: Gradient Clipping per Layer
print("\n--- Example 2: Per-Layer Gradient Clipping ---")

model2 = SimpleModel()


def clip_gradient_hook(module, grad_input, grad_output):
    clipped = tuple(
        g.clamp(min=-0.5, max=0.5) if g is not None else g
        for g in grad_input
    )
    return clipped


handle = model2.layer1.register_full_backward_hook(clip_gradient_hook)

x = torch.randn(8, 784, requires_grad=True)
output = model2(x)
loss = (output * 100).sum()  # Large loss to create large gradients
loss.backward()

print(f"Gradient at input after clipping: "
      f"min={x.grad.min().item():.4f}, max={x.grad.max().item():.4f}")
handle.remove()

# Example 3: Tensor-level hook (on specific tensors, not modules)
print("\n--- Example 3: Tensor-level Gradient Hook ---")
x = torch.randn(4, 784, requires_grad=True)
model3 = SimpleModel()

intermediate_grads = []


def tensor_grad_hook(grad):
    intermediate_grads.append(grad.norm().item())


output = model3(x)
# Register hook on the tensor itself
output.register_hook(tensor_grad_hook)
loss = output.sum()
loss.backward()
print(f"Gradient norm at output tensor: {intermediate_grads[0]:.4f}")

print("\n" + "=" * 70)
print("PART 4: Practical Hook Patterns")
print("=" * 70)

# Pattern 1: Activation Statistics
print("\n--- Pattern 1: Activation Statistics ---")


class ActivationStats:
    """Collects activation statistics for debugging training."""

    def __init__(self, model):
        self.stats = {}
        self.hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                "mean": output.mean().item(),
                "std": output.std().item(),
                "min": output.min().item(),
                "max": output.max().item(),
                "dead_fraction": (output == 0).float().mean().item(),
            }
        return hook

    def print_stats(self):
        for name, s in self.stats.items():
            print(f"  {name}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
                  f"dead={s['dead_fraction']:.1%}")

    def remove(self):
        for h in self.hooks:
            h.remove()


model = SimpleModel()
stats_collector = ActivationStats(model)

x = torch.randn(64, 784)
_ = model(x)
print("Activation statistics after forward pass:")
stats_collector.print_stats()
stats_collector.remove()


# Pattern 2: Feature Extraction for Transfer Learning
print("\n--- Pattern 2: Feature Extractor ---")


class FeatureExtractor:
    """Extract features from any intermediate layer."""

    def __init__(self, model, layer_names):
        self.model = model
        self.features = {}
        self._hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def __call__(self, x):
        self.features.clear()
        _ = self.model(x)
        return self.features

    def remove(self):
        for h in self._hooks:
            h.remove()


extractor = FeatureExtractor(model, ["layer1", "layer2"])
x = torch.randn(4, 784)
features = extractor(x)
print("Extracted features:")
for name, feat in features.items():
    print(f"  {name}: {feat.shape}")
extractor.remove()

print("\n" + "=" * 70)
print("PART 5: State Dict — Saving and Loading Models")
print("=" * 70)

print("""
state_dict() returns an OrderedDict of all parameters and persistent buffers.
This is the RECOMMENDED way to save/load PyTorch models.
""")

# Create a model with parameters and buffers
class ModelWithBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 10)
        self.register_buffer("version", torch.tensor(1))

    def forward(self, x):
        x = torch.relu(self.bn1(self.linear1(x)))
        return self.linear2(x)


model = ModelWithBuffers()

# Train for a bit to update BatchNorm running stats
model.train()
for _ in range(5):
    x = torch.randn(32, 784)
    _ = model(x)

# Inspect state dict
print("\n--- Inspecting State Dict ---")
state_dict = model.state_dict()
print(f"State dict keys ({len(state_dict)} entries):")
for key, tensor in state_dict.items():
    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

print("\n" + "=" * 70)
print("PART 6: Saving and Loading")
print("=" * 70)

# Create temp directory for saving
save_dir = tempfile.mkdtemp()

# Method 1: Save/Load state_dict (RECOMMENDED)
print("\n--- Method 1: Save/Load state_dict (Recommended) ---")
save_path = os.path.join(save_dir, "model_state.pth")
torch.save(model.state_dict(), save_path)
print(f"Saved state dict to: {save_path}")

# Load into a new model instance
model_loaded = ModelWithBuffers()
model_loaded.load_state_dict(torch.load(save_path, weights_only=True))
print("Loaded state dict into new model")

# Verify parameters match
for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_loaded.named_parameters()):
    assert torch.equal(p1, p2), f"Mismatch at {n1}"
print("All parameters match!")

# Verify buffers match too
print(f"  BatchNorm running_mean matches: "
      f"{torch.equal(model.bn1.running_mean, model_loaded.bn1.running_mean)}")

# Method 2: Save entire model (NOT recommended for production)
print("\n--- Method 2: Save Entire Model (Not Recommended) ---")
save_path2 = os.path.join(save_dir, "model_full.pth")
torch.save(model, save_path2)
model_full = torch.load(save_path2, weights_only=False)
print("This works but couples the save to the exact class definition")
print("If you rename/move the class, loading will fail!")

print("\n" + "=" * 70)
print("PART 7: Partial Loading and Strict Mode")
print("=" * 70)


# Scenario: You have a pretrained model but your new model has extra layers
class ExtendedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)  # Same as original
        self.bn1 = nn.BatchNorm1d(256)  # Same as original
        self.linear2 = nn.Linear(256, 128)  # DIFFERENT (was 256->10)
        self.linear3 = nn.Linear(128, 10)  # NEW layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.linear1(x)))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


print("\n--- Partial Loading with strict=False ---")
extended_model = ExtendedModel()

# Load old state dict into new model
old_state_dict = torch.load(save_path, weights_only=True)
missing, unexpected = extended_model.load_state_dict(old_state_dict, strict=False)
print(f"Missing keys (in new model but not in checkpoint): {missing}")
print(f"Unexpected keys (in checkpoint but not in new model): {unexpected}")
print("\nlinear1 and bn1 were loaded from checkpoint")
print("linear2 and linear3 keep their random initialization")

# Verify the loaded layers match
print(f"\nlinear1 weights match: "
      f"{torch.equal(model.linear1.weight, extended_model.linear1.weight)}")

# Selective loading (only load specific keys)
print("\n--- Selective Loading ---")
new_model = ExtendedModel()
pretrained_dict = torch.load(save_path, weights_only=True)

# Filter: only load keys that exist in both and have matching shapes
model_dict = new_model.state_dict()
filtered_dict = {}
for k, v in pretrained_dict.items():
    if k in model_dict and v.shape == model_dict[k].shape:
        filtered_dict[k] = v

print(f"Filtered keys to load: {list(filtered_dict.keys())}")
model_dict.update(filtered_dict)
new_model.load_state_dict(model_dict)
print("Selectively loaded matching parameters")

print("\n" + "=" * 70)
print("PART 8: Checkpoint Saving (for Training Resumption)")
print("=" * 70)

print("""
For training resumption, save more than just the model:
- Model state_dict
- Optimizer state_dict
- Current epoch
- Current loss
- Learning rate scheduler state
- Random state (for reproducibility)
""")

model = ModelWithBuffers()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# Simulate some training
model.train()
for epoch in range(3):
    x = torch.randn(32, 784)
    output = model(x)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Save checkpoint
checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
checkpoint = {
    "epoch": 3,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss.item(),
    "torch_rng_state": torch.random.get_rng_state(),
}
torch.save(checkpoint, checkpoint_path)
print(f"\nSaved checkpoint at epoch 3")
print(f"  Loss: {loss.item():.4f}")
print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

# Resume training from checkpoint
print("\n--- Resuming from checkpoint ---")
model_resume = ModelWithBuffers()
optimizer_resume = torch.optim.Adam(model_resume.parameters(), lr=0.001)
scheduler_resume = torch.optim.lr_scheduler.StepLR(optimizer_resume, step_size=10)

checkpoint = torch.load(checkpoint_path, weights_only=False)
model_resume.load_state_dict(checkpoint["model_state_dict"])
optimizer_resume.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler_resume.load_state_dict(checkpoint["scheduler_state_dict"])
start_epoch = checkpoint["epoch"]
torch.random.set_rng_state(checkpoint["torch_rng_state"])

print(f"  Resumed from epoch {start_epoch}")
print(f"  Optimizer LR: {optimizer_resume.param_groups[0]['lr']:.6f}")
print(f"  Scheduler last epoch: {scheduler_resume.last_epoch}")

print("\n" + "=" * 70)
print("PART 9: State Dict Key Manipulation")
print("=" * 70)

# Common scenario: model was saved with DataParallel (adds 'module.' prefix)
print("\n--- Removing 'module.' prefix (DataParallel) ---")
# Simulate a DataParallel state dict
dp_state_dict = {"module." + k: v for k, v in model.state_dict().items()}
print(f"DataParallel keys (first 3): {list(dp_state_dict.keys())[:3]}")

# Remove prefix
clean_state_dict = {k.replace("module.", ""): v for k, v in dp_state_dict.items()}
print(f"Cleaned keys (first 3): {list(clean_state_dict.keys())[:3]}")
model.load_state_dict(clean_state_dict)
print("Successfully loaded after removing 'module.' prefix")

# Adding a prefix (for loading into a sub-module)
print("\n--- Adding prefix (loading into sub-module) ---")


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ModelWithBuffers()
        self.head = nn.Linear(10, 5)

    def forward(self, x):
        return self.head(self.encoder(x))


wrapper = Wrapper()
# Load the model weights into the encoder sub-module
encoder_state = torch.load(save_path, weights_only=True)
prefixed_state = {"encoder." + k: v for k, v in encoder_state.items()}
wrapper.load_state_dict(prefixed_state, strict=False)
print("Loaded pretrained weights into wrapper.encoder")

print("\n" + "=" * 70)
print("PART 10: Comparing State Dicts")
print("=" * 70)


def compare_state_dicts(sd1, sd2):
    """Compare two state dicts and report differences."""
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common = keys1 & keys2

    print(f"  Keys only in dict1: {len(only_in_1)}")
    print(f"  Keys only in dict2: {len(only_in_2)}")
    print(f"  Common keys: {len(common)}")

    mismatched = []
    for key in common:
        if sd1[key].shape != sd2[key].shape:
            mismatched.append((key, sd1[key].shape, sd2[key].shape))
        elif not torch.equal(sd1[key], sd2[key]):
            diff = (sd1[key] - sd2[key]).abs().max().item()
            mismatched.append((key, "values differ", f"max_diff={diff:.2e}"))

    if mismatched:
        print(f"  Mismatched keys: {len(mismatched)}")
        for info in mismatched[:5]:
            print(f"    {info}")
    else:
        print("  All common keys match perfectly!")


print("\n--- Comparing original vs loaded model ---")
compare_state_dicts(model.state_dict(), model_loaded.state_dict())

# After some training, they should differ
model.train()
x = torch.randn(32, 784)
output = model(x)
loss = output.sum()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("\n--- After training one step ---")
compare_state_dicts(model.state_dict(), model_loaded.state_dict())

# Cleanup
import shutil
shutil.rmtree(save_dir)
print(f"\nCleaned up temporary files")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
