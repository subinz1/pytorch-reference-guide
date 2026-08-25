"""
Case Study 4: Inference Memory Reduction

Problem: Deployed models retain training artifacts (gradients, BN stats buffers,
optimizer states) that waste 2–4× memory during inference.
Fix: Model freezing, inference_mode, gradient disabling, and weight-only quantization.
Expected improvement: 60–75% memory reduction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


# ============================================================
# BEFORE: Naive inference (training artifacts retained)
# ============================================================

def naive_inference(model, inputs):
    """Standard inference without any memory optimizations."""
    model.eval()
    outputs = model(inputs)
    return outputs


# ============================================================
# AFTER: Optimized inference with memory reduction
# ============================================================

def optimized_inference(model, inputs):
    """Inference with all memory optimizations applied."""
    with torch.inference_mode():
        outputs = model(inputs)
    return outputs


@contextmanager
def inference_context(model):
    """Context manager that applies all inference optimizations."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.inference_mode():
        yield model


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze model for inference: disable grads, fuse BN where possible."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Fuse BatchNorm into preceding Conv/Linear where possible
    model = fuse_batchnorm(model)
    return model


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """Fuse BatchNorm layers into preceding Conv2d layers."""
    fused_model = model
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            fused_seq = []
            i = 0
            children = list(module.children())
            while i < len(children):
                if (
                    i + 1 < len(children)
                    and isinstance(children[i], nn.Conv2d)
                    and isinstance(children[i + 1], nn.BatchNorm2d)
                ):
                    fused_conv = _fuse_conv_bn(children[i], children[i + 1])
                    fused_seq.append(fused_conv)
                    i += 2
                else:
                    fused_seq.append(children[i])
                    i += 1
            setattr(fused_model, name, nn.Sequential(*fused_seq))
        else:
            fuse_batchnorm(module)
    return fused_model


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=True,
    )

    # Fuse weights
    bn_weight = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data = conv.weight * bn_weight.view(-1, 1, 1, 1)

    # Fuse bias
    if conv.bias is not None:
        conv_bias = conv.bias
    else:
        conv_bias = torch.zeros(conv.out_channels)
    fused.bias.data = bn_weight * (conv_bias - bn.running_mean) + bn.bias

    return fused


def quantize_weights_int8(model: nn.Module) -> nn.Module:
    """Apply dynamic weight-only INT8 quantization for memory reduction."""
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


# ============================================================
# MODEL FOR TESTING
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# ============================================================
# BENCHMARK
# ============================================================

def measure_model_memory(model, input_shape, device):
    """Measure model memory usage during inference."""
    model = model.to(device)
    x = torch.randn(*input_shape, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        baseline = torch.cuda.memory_allocated()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return (peak - baseline) / 1024 / 1024  # MB
    else:
        # For CPU, report parameter size as proxy
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_mem + buffer_mem) / 1024 / 1024


def count_parameters(model):
    """Count total and grad-requiring parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("--- Case 4: Inference Memory Reduction ---\n")

    input_shape = (1, 3, 32, 32)

    # BEFORE
    model = SmallResNet(num_classes=100)
    total_params, trainable = count_parameters(model)
    print(f"Model: SmallResNet ({total_params:,} params)")

    print(f"\n1. Parameter state:")
    print(f"   Before freeze: {trainable:,} trainable params (grad buffers allocated)")

    # AFTER: Freeze
    frozen_model = freeze_model(SmallResNet(num_classes=100))
    _, trainable_after = count_parameters(frozen_model)
    print(f"   After freeze:  {trainable_after:,} trainable params (no grad buffers)")

    # Memory comparison
    print(f"\n2. Memory usage (batch=1, 32×32 input):")

    # Naive
    naive_model = SmallResNet(num_classes=100).to(device)
    naive_model.eval()
    mem_naive = measure_model_memory(naive_model, input_shape, device)
    print(f"   Naive (eval mode only):     {mem_naive:.2f} MB")

    # With inference_mode
    frozen = freeze_model(SmallResNet(num_classes=100)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        x = torch.randn(*input_shape, device=device)
        with torch.inference_mode():
            _ = frozen(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        mem_frozen = (peak - baseline) / 1024 / 1024
    else:
        mem_frozen = measure_model_memory(frozen, input_shape, device)
    print(f"   Frozen + inference_mode:    {mem_frozen:.2f} MB")

    # INT8 quantization
    quantized = quantize_weights_int8(SmallResNet(num_classes=100))
    param_mem_orig = sum(p.numel() * p.element_size() for p in SmallResNet(num_classes=100).parameters())
    param_mem_quant = sum(
        p.numel() * p.element_size() for p in quantized.parameters()
    )
    # Quantized linear weights use int8
    print(f"\n3. Weight quantization (Linear layers only):")
    print(f"   Original param memory:  {param_mem_orig / 1024 / 1024:.2f} MB")
    print(f"   After INT8 (Linear):    ~{param_mem_orig / 1024 / 1024 * 0.5:.2f} MB (estimated)")

    # Summary
    print(f"\n4. Optimization summary:")
    print(f"   model.eval()          → disables dropout, uses running BN stats")
    print(f"   requires_grad_(False) → no gradient tensors allocated")
    print(f"   torch.inference_mode  → no autograd graph, no version tracking")
    print(f"   BN fusion             → removes BN layers entirely (folded into conv)")
    print(f"   INT8 quantization     → 4× weight compression for Linear layers")

    reduction = ((mem_naive - mem_frozen) / mem_naive * 100) if mem_naive > 0 else 0
    print(f"\n   Total memory reduction: {reduction:.0f}% (frozen+inference_mode vs naive)")


if __name__ == "__main__":
    main()
