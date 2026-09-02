"""
Static Quantization with Calibration
=====================================

Static quantization pre-computes activation ranges using calibration data,
then quantizes both weights and activations to INT8. This gives better
inference speed than dynamic quantization but requires a representative dataset.

Also demonstrates Quantization-Aware Training (QAT).
"""

import copy
import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping


class ConvClassifier(nn.Module):
    """Small CNN with QuantStub/DeQuantStub for eager-mode static quantization."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = quant.QuantStub()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


def generate_calibration_data(num_batches=20, batch_size=32, image_size=28):
    """Synthetic calibration data mimicking MNIST-like inputs."""
    for _ in range(num_batches):
        yield torch.randn(batch_size, 1, image_size, image_size)


def static_quantize_eager(model):
    """
    Eager-mode static quantization pipeline:
    1. Fuse Conv-BN-ReLU modules
    2. Attach observers via qconfig
    3. Calibrate with representative data
    4. Convert to quantized model
    """
    model.eval()
    model_prepared = copy.deepcopy(model)

    model_prepared.qconfig = quant.get_default_qconfig("x86")

    quant.fuse_modules(
        model_prepared.features,
        [["0", "1", "2"], ["4", "5", "6"]],
        inplace=True,
    )

    quant.prepare(model_prepared, inplace=True)

    print("=== Calibrating (running representative data) ===")
    with torch.no_grad():
        for batch in generate_calibration_data(num_batches=20):
            model_prepared(batch)

    quantized_model = quant.convert(model_prepared)

    print("=== Static Quantization Complete ===")
    for name, module in quantized_model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight(), "qscheme"):
            print(f"  {name}: quantized ({module.weight().qscheme()})")

    return quantized_model


def compare_accuracy(fp32_model, int8_model, num_batches=50):
    """Compare output agreement between FP32 and INT8 models."""
    fp32_model.eval()
    int8_model.eval()

    total, agree = 0, 0
    with torch.no_grad():
        for batch in generate_calibration_data(num_batches=num_batches):
            fp32_preds = fp32_model(batch).argmax(dim=1)
            int8_preds = int8_model(batch).argmax(dim=1)
            agree += (fp32_preds == int8_preds).sum().item()
            total += batch.size(0)

    agreement = agree / total * 100
    print(f"\n=== Accuracy Agreement ===")
    print(f"  FP32 vs INT8 prediction agreement: {agreement:.1f}%")
    return agreement


def quantization_aware_training(model, num_epochs=3, lr=1e-3):
    """
    QAT inserts fake-quantize ops during training so the model learns
    to compensate for quantization noise.
    """
    model.train()
    model.qconfig = quant.get_default_qat_qconfig("x86")

    quant.fuse_modules(
        model.features,
        [["0", "1", "2"], ["4", "5", "6"]],
        inplace=True,
    )

    model_qat = quant.prepare_qat(model)

    optimizer = torch.optim.Adam(model_qat.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print("\n=== Quantization-Aware Training ===")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in generate_calibration_data(num_batches=30):
            targets = torch.randint(0, 10, (batch.size(0),))
            optimizer.zero_grad()
            output = model_qat(batch)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        print(f"  Epoch {epoch + 1}/{num_epochs}  loss={epoch_loss / num_batches:.4f}")

    model_qat.eval()
    quantized = quant.convert(model_qat)
    print("  QAT conversion complete")
    return quantized


def print_model_size(model, label="Model"):
    """Compute serialized model size in KB."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_kb = buffer.tell() / 1024
    print(f"  {label}: {size_kb:.1f} KB")
    return size_kb


def full_pipeline_demo():
    """End-to-end: train-like setup -> static quant -> QAT -> compare."""
    model = ConvClassifier(num_classes=10)
    model.eval()

    print("=== Model Sizes ===")
    fp32_size = print_model_size(model, "FP32")

    static_model = static_quantize_eager(model)
    static_size = print_model_size(static_model, "Static INT8")

    qat_model = quantization_aware_training(
        copy.deepcopy(model), num_epochs=2
    )
    qat_size = print_model_size(qat_model, "QAT INT8")

    print(f"\n=== Size Reduction ===")
    print(f"  Static: {(1 - static_size / fp32_size) * 100:.1f}% smaller")
    print(f"  QAT:    {(1 - qat_size / fp32_size) * 100:.1f}% smaller")

    compare_accuracy(model, static_model)


if __name__ == "__main__":
    full_pipeline_demo()
