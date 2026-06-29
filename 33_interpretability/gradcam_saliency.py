"""
Module 33 — Grad-CAM, Saliency Maps, and Attribution Methods
=============================================================
Implements gradient-based interpretability methods: saliency maps,
Grad-CAM, guided backpropagation, and attention weight extraction.
All examples run on CPU with synthetic data.

Run: python gradcam_saliency.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Simple CNN for demonstrations
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Small CNN with 3 conv layers for interpretability demos."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ===================================================================
# 1. Saliency Map — gradient of output w.r.t. input
# ===================================================================
def compute_saliency(model, input_tensor, target_class):
    """Compute saliency map: |d(score)/d(input)|, max across channels."""
    model.eval()
    inp = input_tensor.clone().requires_grad_(True)

    output = model(inp)
    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    saliency = inp.grad.abs().squeeze(0)
    if saliency.dim() == 3:
        saliency = saliency.max(dim=0).values

    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def demo_saliency():
    print("=" * 70)
    print("1. Saliency Map — Pixel-Level Sensitivity")
    print("=" * 70)

    model = SimpleCNN()
    image = torch.randn(1, 3, 32, 32)

    output = model(image)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted class: {pred_class}")

    saliency = compute_saliency(model, image, pred_class)
    print(f"Saliency map shape: {saliency.shape}")
    print(f"Saliency range: [{saliency.min():.4f}, {saliency.max():.4f}]")

    high_saliency = (saliency > 0.5).sum().item()
    total_pixels = saliency.numel()
    print(f"High-saliency pixels (>0.5): {high_saliency}/{total_pixels} "
          f"({high_saliency/total_pixels*100:.1f}%)")

    print("\nSaliency heatmap (8x8 downsampled, values 0-9):")
    downsampled = F.adaptive_avg_pool2d(saliency.unsqueeze(0).unsqueeze(0), 8).squeeze()
    for row in range(8):
        line = ""
        for col in range(8):
            val = int(downsampled[row, col].item() * 9)
            line += str(val) + " "
        print(f"  {line}")
    print()


# ===================================================================
# 2. Grad-CAM — class activation mapping via gradients
# ===================================================================
class GradCAM:
    """Grad-CAM: compute class-discriminative localization maps."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )
        return cam.squeeze()

    def close(self):
        for h in self._handles:
            h.remove()


def demo_gradcam():
    print("=" * 70)
    print("2. Grad-CAM — Where Does the CNN Focus?")
    print("=" * 70)

    model = SimpleCNN()
    image = torch.randn(1, 3, 32, 32)

    output = model(image)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    print(f"Predicted class: {pred_class}, confidence: {confidence:.4f}")

    cam = GradCAM(model, model.conv3)
    heatmap = cam.generate(image, pred_class)
    cam.close()

    print(f"Grad-CAM heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    hot_region = (heatmap > 0.5).sum().item()
    total = heatmap.numel()
    print(f"Hot region (>0.5): {hot_region}/{total} ({hot_region/total*100:.1f}%)")

    print("\nGrad-CAM heatmap (8x8 downsampled, values 0-9):")
    downsampled = F.adaptive_avg_pool2d(heatmap.unsqueeze(0).unsqueeze(0), 8).squeeze()
    for row in range(8):
        line = ""
        for col in range(8):
            val = int(downsampled[row, col].item() * 9)
            line += str(val) + " "
        print(f"  {line}")

    # Grad-CAM for a different class
    other_class = (pred_class + 1) % 10
    cam2 = GradCAM(model, model.conv3)
    heatmap2 = cam2.generate(image, other_class)
    cam2.close()

    diff = (heatmap - heatmap2).abs().mean().item()
    print(f"\nHeatmap difference (class {pred_class} vs {other_class}): {diff:.4f}")
    print("  -> Different classes produce different spatial focus")
    print()


# ===================================================================
# 3. Guided Backpropagation — sharper pixel attributions
# ===================================================================
class GuidedBackprop:
    """Guided backpropagation: only pass positive gradients through ReLU."""

    def __init__(self, model):
        self.model = model
        self._handles = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_full_backward_hook(self._guided_relu_hook)
                self._handles.append(handle)

    @staticmethod
    def _guided_relu_hook(module, grad_input, grad_output):
        return (torch.clamp(grad_output[0], min=0.0),)

    def generate(self, input_tensor, target_class):
        self.model.eval()
        inp = input_tensor.clone().requires_grad_(True)

        output = self.model(inp)
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        guided_grads = inp.grad.clone()
        return guided_grads

    def close(self):
        for h in self._handles:
            h.remove()


def demo_guided_backprop():
    print("=" * 70)
    print("3. Guided Backpropagation — Sharper Attribution")
    print("=" * 70)

    model = SimpleCNN()
    image = torch.randn(1, 3, 32, 32)

    output = model(image)
    pred_class = output.argmax(dim=1).item()

    # Vanilla gradient (saliency)
    saliency = compute_saliency(model, image, pred_class)

    # Guided backprop
    gbp = GuidedBackprop(model)
    guided_grads = gbp.generate(image, pred_class)
    gbp.close()

    guided_saliency = guided_grads.abs().squeeze(0).max(dim=0).values
    guided_saliency = (guided_saliency - guided_saliency.min()) / (
        guided_saliency.max() - guided_saliency.min() + 1e-8
    )

    print(f"Vanilla saliency — non-zero pixels: "
          f"{(saliency > 0.01).sum().item()}/{saliency.numel()}")
    print(f"Guided backprop  — non-zero pixels: "
          f"{(guided_saliency > 0.01).sum().item()}/{guided_saliency.numel()}")

    vanilla_sparsity = (saliency < 0.01).float().mean().item()
    guided_sparsity = (guided_saliency < 0.01).float().mean().item()
    print(f"\nSparsity (fraction near zero):")
    print(f"  Vanilla:  {vanilla_sparsity:.4f}")
    print(f"  Guided:   {guided_sparsity:.4f}")
    print("  -> Guided backprop produces sparser (sharper) attributions")
    print()


# ===================================================================
# 4. Attention Weight Extraction from a Transformer
# ===================================================================
class TinyTransformer(nn.Module):
    """Minimal transformer for attention extraction demo."""

    def __init__(self, d_model=64, nhead=4, num_layers=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class AttentionExtractor:
    """Extract attention weights from MultiheadAttention layers via hooks."""

    def __init__(self, model):
        self.attention_maps = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                handle = module.register_forward_hook(self._attn_hook(name))
                self._handles.append(handle)

    def _attn_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_maps[name] = attn_weights.detach()
        return hook

    def close(self):
        for h in self._handles:
            h.remove()


def demo_attention_extraction():
    print("=" * 70)
    print("4. Attention Weight Extraction from Transformer")
    print("=" * 70)

    model = TinyTransformer(d_model=64, nhead=4, num_layers=2)
    seq = torch.randn(2, 8, 64)  # batch=2, seq_len=8, d_model=64

    extractor = AttentionExtractor(model)
    output = model(seq)
    extractor.close()

    print(f"Input sequence shape: {seq.shape} (batch, seq_len, d_model)")
    print(f"Output shape: {output.shape}")
    print(f"\nCaptured attention from {len(extractor.attention_maps)} layers:")
    for name, attn in extractor.attention_maps.items():
        print(f"  {name}: shape={attn.shape}")
        if attn.dim() == 3:
            # (batch, seq_len, seq_len) — averaged over heads
            attn_sample = attn[0]
            print(f"    Row sums (should be ~1.0): "
                  f"{attn_sample.sum(dim=-1)[:4].tolist()}")
            print(f"    Attention matrix (first 4x4 of sample 0):")
            for r in range(min(4, attn_sample.shape[0])):
                vals = [f"{attn_sample[r, c]:.3f}" for c in range(min(4, attn_sample.shape[1]))]
                print(f"      [{', '.join(vals)}, ...]")
    print()


# ===================================================================
# 5. Compare all methods on the same input
# ===================================================================
def demo_compare_methods():
    print("=" * 70)
    print("5. Compare All Methods on Same Input")
    print("=" * 70)

    model = SimpleCNN()
    image = torch.randn(1, 3, 32, 32)

    output = model(image)
    pred_class = output.argmax(dim=1).item()
    print(f"Predicted class: {pred_class}")
    print(f"Logits: {output[0].detach().tolist()[:5]}...")

    # Method 1: Saliency
    saliency = compute_saliency(model, image, pred_class)

    # Method 2: Grad-CAM
    cam = GradCAM(model, model.conv3)
    gradcam_map = cam.generate(image, pred_class)
    cam.close()

    # Method 3: Guided backprop
    gbp = GuidedBackprop(model)
    guided = gbp.generate(image, pred_class)
    gbp.close()
    guided_map = guided.abs().squeeze(0).max(dim=0).values
    guided_map = (guided_map - guided_map.min()) / (guided_map.max() - guided_map.min() + 1e-8)

    # Method 4: Guided Grad-CAM (element-wise product)
    gradcam_upsampled = gradcam_map
    guided_gradcam = guided_map * gradcam_upsampled
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
        guided_gradcam.max() - guided_gradcam.min() + 1e-8
    )

    methods = {
        'Saliency': saliency,
        'Grad-CAM': gradcam_map,
        'Guided BP': guided_map,
        'Guided Grad-CAM': guided_gradcam,
    }

    print(f"\n{'Method':<20} {'Mean':>8} {'Std':>8} {'Sparsity':>10} {'Hot%':>8}")
    print("-" * 58)
    for name, m in methods.items():
        sparsity = (m < 0.01).float().mean().item()
        hot_pct = (m > 0.5).float().mean().item() * 100
        print(f"{name:<20} {m.mean():.4f}   {m.std():.4f}   "
              f"{sparsity:.4f}     {hot_pct:.1f}%")

    # Text-based visualization: 8x8 downsampled heatmaps side by side
    print(f"\n8x8 heatmaps (0-9 intensity scale):")
    print(f"{'Saliency':<18} {'Grad-CAM':<18} {'Guided BP':<18} {'Guided GC':<18}")
    print("-" * 72)

    downsampled = {}
    for name, m in methods.items():
        ds = F.adaptive_avg_pool2d(m.unsqueeze(0).unsqueeze(0), 8).squeeze()
        downsampled[name] = ds

    for row in range(8):
        line = ""
        for name in methods:
            ds = downsampled[name]
            for col in range(8):
                val = int(ds[row, col].item() * 9)
                line += str(val) + " "
            line += "  "
        print(f"  {line}")
    print()


# ===================================================================
# 6. Activation-based class comparison
# ===================================================================
def demo_class_comparison():
    print("=" * 70)
    print("6. Grad-CAM Class Comparison — Same Image, Different Targets")
    print("=" * 70)

    model = SimpleCNN()
    image = torch.randn(1, 3, 32, 32)

    output = model(image)
    probs = torch.softmax(output, dim=1)[0]
    top3 = probs.topk(3)

    print("Top-3 predictions:")
    for i, (prob, cls) in enumerate(zip(top3.values, top3.indices)):
        print(f"  #{i+1}: class {cls.item()} (confidence: {prob.item():.4f})")

    print(f"\nGrad-CAM comparison across top-3 classes:")
    heatmaps = {}
    for cls_idx in top3.indices:
        cls = cls_idx.item()
        cam = GradCAM(model, model.conv3)
        heatmap = cam.generate(image, cls)
        cam.close()
        heatmaps[cls] = heatmap
        hot_pct = (heatmap > 0.5).float().mean().item() * 100
        print(f"  Class {cls}: hot region = {hot_pct:.1f}%, "
              f"mean intensity = {heatmap.mean():.4f}")

    classes = list(heatmaps.keys())
    if len(classes) >= 2:
        overlap = ((heatmaps[classes[0]] > 0.5) & (heatmaps[classes[1]] > 0.5))
        overlap_pct = overlap.float().mean().item() * 100
        print(f"\n  Overlap between class {classes[0]} and {classes[1]} "
              f"hot regions: {overlap_pct:.1f}%")
    print()


# ===================================================================
# 7. Vanishing gradient detector
# ===================================================================
class VanishingGradientDetector:
    """Hook-based detector for vanishing gradients."""

    def __init__(self, model, threshold=1e-6):
        self.threshold = threshold
        self.grad_norms = {}
        self.warnings = []
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_full_backward_hook(self._check_grads(name))
                self._handles.append(handle)

    def _check_grads(self, name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0]
            norm = grad.norm().item()
            self.grad_norms[name] = norm
            if norm < self.threshold:
                self.warnings.append(
                    f"VANISHING: {name} gradient norm = {norm:.2e} "
                    f"(< {self.threshold:.0e})"
                )
        return hook

    def report(self):
        print(f"Gradient norms per layer:")
        for name, norm in self.grad_norms.items():
            flag = " *** VANISHING ***" if norm < self.threshold else ""
            print(f"  {name:15s}: {norm:.6e}{flag}")
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  {w}")
        else:
            print("\nNo vanishing gradients detected.")

    def close(self):
        for h in self._handles:
            h.remove()


def demo_vanishing_detector():
    print("=" * 70)
    print("7. Vanishing Gradient Detector")
    print("=" * 70)

    # Deep network that may exhibit vanishing gradients
    class DeepMLP(nn.Module):
        def __init__(self, depth=10):
            super().__init__()
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(32, 32), nn.Sigmoid()])
            layers.append(nn.Linear(32, 5))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = DeepMLP(depth=10)
    x = torch.randn(4, 32)
    target = torch.randint(0, 5, (4,))

    detector = VanishingGradientDetector(model, threshold=1e-6)
    output = model(x)
    loss = F.cross_entropy(output, target)
    loss.backward()

    detector.report()
    detector.close()
    print()


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print()
    print("Module 33: Grad-CAM, Saliency Maps, and Attribution Methods")
    print("=" * 70)
    print()

    demo_saliency()
    demo_gradcam()
    demo_guided_backprop()
    demo_attention_extraction()
    demo_compare_methods()
    demo_class_comparison()
    demo_vanishing_detector()

    print("=" * 70)
    print("All interpretability demos completed!")
    print("=" * 70)
