"""
Module 40 — Image Classifier: Evaluation & Inference
=====================================================

Comprehensive metrics, test-time augmentation (TTA), Grad-CAM visualization,
confusion matrix, per-class reports, and confidence analysis.

Usage:
    python evaluation.py

Requirements:
    pip install torch torchvision
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline import (
    NUM_CLASSES,
    CLASS_NAMES,
    SyntheticShapeDataset,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    build_dataloaders,
)
from model_and_training import SimpleCNN, train, LabelSmoothingCrossEntropy


# ---------------------------------------------------------------------------
# 1. Comprehensive Classification Metrics
# ---------------------------------------------------------------------------

class ClassificationMetrics:
    """Accumulates predictions and computes precision, recall, F1, confusion matrix."""

    def __init__(self, num_classes: int, class_names: list[str] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.all_probs: list[torch.Tensor] = []
        self.all_labels: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        self.all_probs.append(probs.cpu())
        self.all_labels.append(labels.cpu())
        for pred, true in zip(preds.cpu(), labels.cpu()):
            self.confusion[true, pred] += 1

    def accuracy(self) -> float:
        correct = self.confusion.diag().sum().item()
        total = self.confusion.sum().item()
        return correct / max(1, total)

    def per_class_metrics(self) -> dict[str, dict[str, float]]:
        metrics = {}
        for i, name in enumerate(self.class_names):
            tp = self.confusion[i, i].item()
            fp = self.confusion[:, i].sum().item() - tp
            fn = self.confusion[i, :].sum().item() - tp
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            support = self.confusion[i, :].sum().item()
            metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        return metrics

    def macro_f1(self) -> float:
        per_class = self.per_class_metrics()
        return sum(m["f1"] for m in per_class.values()) / len(per_class)

    def weighted_f1(self) -> float:
        per_class = self.per_class_metrics()
        total_support = sum(m["support"] for m in per_class.values())
        return sum(m["f1"] * m["support"] for m in per_class.values()) / max(1, total_support)

    def top_k_accuracy(self, k: int = 5) -> float:
        all_probs = torch.cat(self.all_probs, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)
        top_k_preds = all_probs.topk(k, dim=-1).indices
        correct = (top_k_preds == all_labels.unsqueeze(1)).any(dim=1).sum().item()
        return correct / len(all_labels)

    def print_report(self):
        print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        print("-" * 50)
        per_class = self.per_class_metrics()
        for name, m in per_class.items():
            print(f"{name:<12} {m['precision']:10.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['support']:9.0f}")
        print("-" * 50)
        print(f"{'Accuracy':<12} {self.accuracy():>38.4f}")
        print(f"{'Macro F1':<12} {self.macro_f1():>38.4f}")
        print(f"{'Weighted F1':<12} {self.weighted_f1():>38.4f}")
        if self.num_classes > 2:
            print(f"{'Top-3 Acc':<12} {self.top_k_accuracy(3):>38.4f}")

    def print_confusion_matrix(self):
        print(f"\nConfusion Matrix ({self.num_classes}x{self.num_classes}):")
        header = "True\\Pred".ljust(12) + " ".join(f"{n[:6]:>6}" for n in self.class_names)
        print(header)
        print("-" * len(header))
        for i, name in enumerate(self.class_names):
            row = name[:12].ljust(12)
            row += " ".join(f"{self.confusion[i, j].item():>6}" for j in range(self.num_classes))
            print(row)


# ---------------------------------------------------------------------------
# 2. Evaluate with Full Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> ClassificationMetrics:
    model.eval()
    metrics = ClassificationMetrics(NUM_CLASSES, CLASS_NAMES)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        metrics.update(logits, labels)
    return metrics


# ---------------------------------------------------------------------------
# 3. Test-Time Augmentation (TTA)
# ---------------------------------------------------------------------------

class TTAAugmentation:
    """Applies multiple augmentations and averages predictions."""

    def __init__(self, num_augmentations: int = 5):
        self.num_augmentations = num_augmentations
        self.augmentations = [
            lambda x: x,
            lambda x: x.flip(-1),
            lambda x: x.flip(-2),
            lambda x: torch.rot90(x, 1, [-2, -1]),
            lambda x: torch.rot90(x, 2, [-2, -1]),
            lambda x: torch.rot90(x, 3, [-2, -1]),
            lambda x: x.flip(-1).flip(-2),
        ]

    @torch.no_grad()
    def predict(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        model.eval()
        all_probs = []
        for i, aug_fn in enumerate(self.augmentations[:self.num_augmentations]):
            aug_images = aug_fn(images)
            logits = model(aug_images)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs


@torch.no_grad()
def evaluate_with_tta(model: nn.Module, loader: DataLoader, device: torch.device, num_augmentations: int = 5) -> ClassificationMetrics:
    model.eval()
    tta = TTAAugmentation(num_augmentations)
    metrics = ClassificationMetrics(NUM_CLASSES, CLASS_NAMES)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        avg_probs = tta.predict(model, images)
        pseudo_logits = torch.log(avg_probs + 1e-8)
        metrics.update(pseudo_logits, labels)

    return metrics


# ---------------------------------------------------------------------------
# 4. Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN visualization.

    Hooks into the target convolutional layer, captures activations and
    gradients, then produces a heatmap showing which spatial regions
    contributed most to a target class prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=-1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * input_tensor.size(0), device=input_tensor.device)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i in range(input_tensor.size(0)):
            one_hot[i, target_class[i]] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.squeeze(1)


# ---------------------------------------------------------------------------
# 5. Confidence Analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def confidence_analysis(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Analyze model confidence: calibration, correct vs incorrect confidence."""
    model.eval()
    correct_confs = []
    incorrect_confs = []
    all_confs = []
    all_correct = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)
        is_correct = (preds == labels)

        correct_confs.extend(max_probs[is_correct].cpu().tolist())
        incorrect_confs.extend(max_probs[~is_correct].cpu().tolist())
        all_confs.extend(max_probs.cpu().tolist())
        all_correct.extend(is_correct.cpu().tolist())

    num_bins = 10
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(all_confs)
    confs_t = torch.tensor(all_confs)
    correct_t = torch.tensor(all_correct, dtype=torch.float)

    for i in range(num_bins):
        mask = (confs_t >= bin_boundaries[i]) & (confs_t < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correct_t[mask].mean().item()
            bin_conf = confs_t[mask].mean().item()
            ece += mask.sum().item() / total_samples * abs(bin_acc - bin_conf)

    return {
        "mean_correct_confidence": sum(correct_confs) / max(1, len(correct_confs)),
        "mean_incorrect_confidence": sum(incorrect_confs) / max(1, len(incorrect_confs)),
        "ece": ece,
        "num_correct": len(correct_confs),
        "num_incorrect": len(incorrect_confs),
    }


# ---------------------------------------------------------------------------
# 6. Inference Pipeline
# ---------------------------------------------------------------------------

class ImageClassifier:
    """Production-ready inference wrapper."""

    def __init__(self, model: nn.Module, class_names: list[str], device: torch.device, use_tta: bool = False):
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.normalize = Normalize()
        self.tta = TTAAugmentation(num_augmentations=5) if use_tta else None

    @torch.no_grad()
    def predict(self, images: torch.Tensor, top_k: int = 3) -> list[dict]:
        images = self.normalize(images).to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(0)

        if self.tta is not None:
            probs = self.tta.predict(self.model, images)
        else:
            logits = self.model(images)
            probs = F.softmax(logits, dim=-1)

        results = []
        for i in range(images.size(0)):
            top_probs, top_indices = probs[i].topk(top_k)
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    "class": self.class_names[idx.item()],
                    "class_id": idx.item(),
                    "confidence": prob.item(),
                })
            results.append({"predictions": predictions, "top_class": predictions[0]["class"], "confidence": predictions[0]["confidence"]})
        return results

    @torch.no_grad()
    def predict_batch(self, loader: DataLoader) -> list[dict]:
        all_results = []
        for images, _ in loader:
            images = images.to(self.device)
            results = self.predict(images)
            all_results.extend(results)
        return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MODULE 40 — IMAGE CLASSIFIER: EVALUATION & INFERENCE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # --- Train a model first ---
    print("\n1. Training Model for Evaluation")
    print("-" * 40)
    train_loader, val_loader, test_loader = build_dataloaders(
        num_train=2000, num_val=300, num_test=300, batch_size=64,
    )
    model = SimpleCNN()
    train(model, train_loader, val_loader, epochs=8, lr=3e-3, use_mixup=False, use_ema=True, patience=10, device=device)

    # --- Full metrics ---
    print("\n2. Comprehensive Evaluation")
    print("-" * 40)
    metrics = evaluate_full(model, test_loader, device)
    metrics.print_report()
    metrics.print_confusion_matrix()

    # --- TTA ---
    print("\n3. Test-Time Augmentation (TTA)")
    print("-" * 40)
    print("Standard evaluation:")
    std_metrics = evaluate_full(model, test_loader, device)
    print(f"  Accuracy: {std_metrics.accuracy():.4f}")

    print("TTA evaluation (5 augmentations):")
    tta_metrics = evaluate_with_tta(model, test_loader, device, num_augmentations=5)
    print(f"  Accuracy: {tta_metrics.accuracy():.4f}")

    improvement = tta_metrics.accuracy() - std_metrics.accuracy()
    print(f"  Improvement: {improvement:+.4f}")

    # --- Grad-CAM ---
    print("\n4. Grad-CAM Visualization")
    print("-" * 40)
    target_layer = model.features[-3]
    grad_cam = GradCAM(model, target_layer)
    test_ds = SyntheticShapeDataset(num_samples=10, img_size=32, transform=Compose([Normalize()]), seed=789)
    sample_imgs = torch.stack([test_ds[i][0] for i in range(4)]).to(device)
    sample_labels = [test_ds[i][1] for i in range(4)]

    heatmaps = grad_cam.generate(sample_imgs)
    print(f"  Heatmap shape: {heatmaps.shape}")
    print(f"  Heatmap range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")
    for i in range(4):
        print(f"  Sample {i}: class={CLASS_NAMES[sample_labels[i]]}, "
              f"heatmap max={heatmaps[i].max():.3f}, "
              f"hot region={(heatmaps[i] > 0.5).float().mean():.1%}")

    # --- Confidence analysis ---
    print("\n5. Confidence Analysis")
    print("-" * 40)
    conf = confidence_analysis(model, test_loader, device)
    print(f"  Correct predictions:     {conf['num_correct']}")
    print(f"  Incorrect predictions:   {conf['num_incorrect']}")
    print(f"  Mean confidence (correct):   {conf['mean_correct_confidence']:.4f}")
    print(f"  Mean confidence (incorrect): {conf['mean_incorrect_confidence']:.4f}")
    print(f"  Expected Calibration Error:  {conf['ece']:.4f}")

    # --- Inference pipeline ---
    print("\n6. Inference Pipeline")
    print("-" * 40)
    classifier = ImageClassifier(model, CLASS_NAMES, device, use_tta=False)
    ds = SyntheticShapeDataset(num_samples=5, img_size=32, seed=999)
    for i in range(5):
        img, true_label = ds[i]
        results = classifier.predict(img)
        r = results[0]
        status = "correct" if r["top_class"] == CLASS_NAMES[true_label] else "wrong"
        print(f"  Sample {i}: true={CLASS_NAMES[true_label]:<10} pred={r['top_class']:<10} "
              f"conf={r['confidence']:.3f} [{status}]")

    # --- Inference with TTA ---
    print("\n7. Inference with TTA")
    print("-" * 40)
    classifier_tta = ImageClassifier(model, CLASS_NAMES, device, use_tta=True)
    for i in range(3):
        img, true_label = ds[i]
        results = classifier_tta.predict(img, top_k=3)
        r = results[0]
        top3 = ", ".join(f"{p['class']}({p['confidence']:.2f})" for p in r["predictions"])
        print(f"  Sample {i}: true={CLASS_NAMES[true_label]:<10} top-3: {top3}")

    print("\n" + "=" * 70)
    print("EVALUATION & INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
