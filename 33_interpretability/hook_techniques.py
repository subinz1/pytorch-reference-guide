"""
Module 33 — Hook Techniques for Model Interpretability
=====================================================
Demonstrates forward hooks, backward hooks, pre-hooks, tensor hooks,
activation extraction, statistics collection, and cleanup patterns.

Run: python hook_techniques.py
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Simple MLP for demonstrations
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, in_features=64, hidden=128, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# ===================================================================
# 1. Forward Hooks — extract activations from every layer
# ===================================================================
def demo_forward_hooks():
    print("=" * 70)
    print("1. Forward Hooks — Extract Activations from Every Layer")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(4, 64)

    activations = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    handles = []
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(save_activation(name)))

    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nCaptured activations from {len(activations)} layers:")
    for name, act in activations.items():
        print(f"  {name:10s} -> shape={act.shape}, "
              f"mean={act.mean():.4f}, std={act.std():.4f}")

    for h in handles:
        h.remove()
    print()


# ===================================================================
# 2. Forward Pre-Hooks — log input shapes
# ===================================================================
def demo_forward_pre_hooks():
    print("=" * 70)
    print("2. Forward Pre-Hooks — Log Input Shapes Before Each Layer")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(4, 64)

    input_log = []

    def log_input(name):
        def hook(module, input):
            shape = input[0].shape if isinstance(input, tuple) else input.shape
            entry = f"{name}: input shape = {shape}"
            input_log.append(entry)
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(log_input(name)))

    _ = model(x)

    print("Input shapes logged via pre-hooks:")
    for entry in input_log:
        print(f"  {entry}")

    for h in handles:
        h.remove()
    print()


# ===================================================================
# 3. Backward Hooks — log gradient norms per layer
# ===================================================================
def demo_backward_hooks():
    print("=" * 70)
    print("3. Backward Hooks — Gradient Norms Per Layer")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(4, 64)
    target = torch.randint(0, 10, (4,))

    grad_norms = {}

    def log_grad_norm(name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0]
            grad_norms[name] = {
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'max_abs': grad.abs().max().item(),
            }
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU)):
            handles.append(
                module.register_full_backward_hook(log_grad_norm(name))
            )

    output = model(x)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"\nGradient statistics per layer:")
    for name, stats in grad_norms.items():
        print(f"  {name:10s} -> norm={stats['norm']:.6f}, "
              f"mean={stats['mean']:.6f}, max_abs={stats['max_abs']:.6f}")

    for h in handles:
        h.remove()
    print()


# ===================================================================
# 4. Tensor Hooks — per-parameter gradient logging
# ===================================================================
def demo_tensor_hooks():
    print("=" * 70)
    print("4. Tensor Hooks — Per-Parameter Gradient Logging")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(4, 64)
    target = torch.randint(0, 10, (4,))

    param_grads = {}

    handles = []
    for name, param in model.named_parameters():
        def make_hook(n):
            def hook(grad):
                param_grads[n] = {
                    'shape': tuple(grad.shape),
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                }
            return hook
        handles.append(param.register_hook(make_hook(name)))

    output = model(x)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()

    print("Per-parameter gradient statistics:")
    for name, stats in param_grads.items():
        print(f"  {name:15s} shape={str(stats['shape']):15s} "
              f"norm={stats['norm']:.6f}  mean={stats['mean']:.8f}")

    for h in handles:
        h.remove()
    print()


# ===================================================================
# 5. FeatureExtractor class — register, collect, clean up
# ===================================================================
class FeatureExtractor:
    """Register forward hooks on target layers, collect activations, clean up."""

    def __init__(self, model, target_layers):
        self.model = model
        self.features = {}
        self._handles = []

        layer_map = dict(model.named_modules())
        for layer_name in target_layers:
            if layer_name not in layer_map:
                raise ValueError(f"Layer '{layer_name}' not found in model")
            handle = layer_map[layer_name].register_forward_hook(
                self._make_hook(layer_name)
            )
            self._handles.append(handle)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def __call__(self, x):
        self.features.clear()
        output = self.model(x)
        return output, dict(self.features)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def demo_feature_extractor():
    print("=" * 70)
    print("5. FeatureExtractor Class — Clean Activation Collection")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(4, 64)

    with FeatureExtractor(model, ['fc1', 'relu1', 'fc2', 'fc3']) as extractor:
        output, features = extractor(x)

    print(f"Model output shape: {output.shape}")
    print(f"Extracted features from {len(features)} layers:")
    for name, feat in features.items():
        print(f"  {name:10s} -> shape={feat.shape}")

    # Verify hooks were removed
    hook_count = sum(len(m._forward_hooks) for m in model.modules())
    print(f"\nHooks remaining after close: {hook_count}")
    print()


# ===================================================================
# 6. Activation statistics — mean, std, dead neurons
# ===================================================================
class ActivationStatistics:
    """Track activation statistics: mean, std, fraction of dead neurons."""

    def __init__(self, model):
        self.stats = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                handle = module.register_forward_hook(self._stats_hook(name))
                self._handles.append(handle)

    def _stats_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                flat = output.flatten()
                self.stats[name] = {
                    'mean': flat.mean().item(),
                    'std': flat.std().item(),
                    'min': flat.min().item(),
                    'max': flat.max().item(),
                    'dead_fraction': (flat == 0).float().mean().item(),
                    'num_elements': flat.numel(),
                }
        return hook

    def report(self):
        print(f"{'Layer':<12} {'Mean':>8} {'Std':>8} {'Min':>8} "
              f"{'Max':>8} {'Dead%':>8}")
        print("-" * 60)
        for name, s in self.stats.items():
            print(f"{name:<12} {s['mean']:>8.4f} {s['std']:>8.4f} "
                  f"{s['min']:>8.4f} {s['max']:>8.4f} "
                  f"{s['dead_fraction']*100:>7.1f}%")

    def close(self):
        for h in self._handles:
            h.remove()


def demo_activation_stats():
    print("=" * 70)
    print("6. Activation Statistics — Dead Neurons and Distribution")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(32, 64)

    stats_tracker = ActivationStatistics(model)
    _ = model(x)
    stats_tracker.report()
    stats_tracker.close()

    # Show effect of bad initialization (many dead neurons)
    print("\nWith intentionally bad initialization (large negative bias):")
    model.fc1.bias.data.fill_(-5.0)
    stats_tracker2 = ActivationStatistics(model)
    _ = model(x)
    stats_tracker2.report()
    stats_tracker2.close()
    print()


# ===================================================================
# 7. Hook-based model profiling — count FLOPs per layer
# ===================================================================
class FLOPCounter:
    """Approximate FLOPs per layer using forward hooks."""

    def __init__(self, model):
        self.flops = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(self._linear_flops(name))
                self._handles.append(handle)
            elif isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(self._conv_flops(name))
                self._handles.append(handle)

    def _linear_flops(self, name):
        def hook(module, input, output):
            batch_size = input[0].shape[0]
            # multiply-add: 2 * in_features * out_features per sample
            flops = 2 * module.in_features * module.out_features * batch_size
            if module.bias is not None:
                flops += module.out_features * batch_size
            self.flops[name] = flops
        return hook

    def _conv_flops(self, name):
        def hook(module, input, output):
            batch_size = output.shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (
                module.in_channels // module.groups
            )
            flops = 2 * kernel_ops * module.out_channels * out_h * out_w * batch_size
            self.flops[name] = flops
        return hook

    def total_flops(self):
        return sum(self.flops.values())

    def report(self):
        total = self.total_flops()
        print(f"{'Layer':<12} {'FLOPs':>15} {'Fraction':>10}")
        print("-" * 40)
        for name, flops in self.flops.items():
            fraction = flops / total if total > 0 else 0
            print(f"{name:<12} {flops:>15,} {fraction:>9.1%}")
        print("-" * 40)
        print(f"{'Total':<12} {total:>15,}")

    def close(self):
        for h in self._handles:
            h.remove()


def demo_flop_counter():
    print("=" * 70)
    print("7. Hook-Based Model Profiling — FLOP Count Per Layer")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(8, 64)

    counter = FLOPCounter(model)
    _ = model(x)
    counter.report()
    counter.close()
    print()


# ===================================================================
# 8. RemovableHandle pattern and cleanup
# ===================================================================
@contextmanager
def hook_context(module, hook_fn, hook_type='forward'):
    """Context manager that auto-removes hooks on exit."""
    if hook_type == 'forward':
        handle = module.register_forward_hook(hook_fn)
    elif hook_type == 'backward':
        handle = module.register_full_backward_hook(hook_fn)
    elif hook_type == 'pre':
        handle = module.register_forward_pre_hook(hook_fn)
    else:
        raise ValueError(f"Unknown hook_type: {hook_type}")
    try:
        yield handle
    finally:
        handle.remove()


def demo_removable_handle():
    print("=" * 70)
    print("8. RemovableHandle Pattern and Cleanup")
    print("=" * 70)

    model = SimpleMLP()
    x = torch.randn(4, 64)

    # Pattern 1: Manual remove
    print("Pattern 1 — Manual handle.remove():")
    captured = []
    handle = model.fc1.register_forward_hook(
        lambda m, i, o: captured.append(o.shape)
    )
    _ = model(x)
    handle.remove()
    print(f"  Captured shape: {captured[0]}")

    # Pattern 2: Context manager
    print("\nPattern 2 — Context manager (auto-remove):")
    captured2 = []
    with hook_context(model.fc2, lambda m, i, o: captured2.append(o.shape)):
        _ = model(x)
    print(f"  Captured shape: {captured2[0]}")

    # Pattern 3: FeatureExtractor (class-based)
    print("\nPattern 3 — Class-based with __enter__/__exit__:")
    with FeatureExtractor(model, ['fc1']) as ext:
        _, feats = ext(x)
    print(f"  fc1 shape: {feats['fc1'].shape}")

    # Verify all hooks are cleaned up
    total_hooks = sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks)
        + len(m._backward_hooks)
        for m in model.modules()
    )
    print(f"\nTotal hooks remaining: {total_hooks}")
    print()


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print()
    print("Module 33: Hook Techniques for Model Interpretability")
    print("=" * 70)
    print()

    demo_forward_hooks()
    demo_forward_pre_hooks()
    demo_backward_hooks()
    demo_tensor_hooks()
    demo_feature_extractor()
    demo_activation_stats()
    demo_flop_counter()
    demo_removable_handle()

    print("=" * 70)
    print("All hook technique demos completed!")
    print("=" * 70)
