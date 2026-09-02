"""
Dynamic Quantization Examples
=============================

Dynamic quantization converts weights to INT8 at save time and quantizes
activations on-the-fly during inference. No calibration data is needed.
Best for Linear-heavy models (LSTMs, Transformers, MLPs).
"""

import time
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """Simple LSTM-based text classifier for quantization demos."""

    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])


class TransformerBlock(nn.Module):
    """Minimal Transformer encoder for quantization benchmarking."""

    def __init__(self, d_model=512, nhead=8, dim_ff=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.ff(self.norm2(x))
        return x


def quantize_lstm_model():
    """Dynamic quantization of an LSTM model targeting Linear and LSTM layers."""
    model = TextClassifier()
    model.eval()

    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )

    original_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.nelement() * p.element_size() for p in quantized.parameters())

    print("=== LSTM Dynamic Quantization ===")
    print(f"  Original size:  {original_size / 1024:.1f} KB")
    print(f"  Quantized size: {quantized_size / 1024:.1f} KB")
    print(f"  Reduction:      {(1 - quantized_size / original_size) * 100:.1f}%")

    return model, quantized


def quantize_linear_only():
    """Quantize only nn.Linear layers (common for Transformer models)."""
    model = TransformerBlock()
    model.eval()

    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    print("\n=== Transformer Dynamic Quantization (Linear only) ===")
    for name, module in quantized.named_modules():
        if "DynamicQuantizedLinear" in type(module).__name__:
            print(f"  Quantized: {name} -> {type(module).__name__}")

    return model, quantized


def benchmark_latency(model_fp32, model_int8, input_fn, num_runs=100):
    """Compare inference latency between FP32 and INT8 models."""
    model_fp32.eval()
    model_int8.eval()

    for _ in range(10):
        with torch.no_grad():
            model_fp32(input_fn())
            model_int8(input_fn())

    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            model_fp32(input_fn())
    fp32_time = (time.perf_counter() - start) / num_runs

    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            model_int8(input_fn())
    int8_time = (time.perf_counter() - start) / num_runs

    print(f"\n=== Latency Benchmark ({num_runs} runs) ===")
    print(f"  FP32: {fp32_time * 1000:.2f} ms/inference")
    print(f"  INT8: {int8_time * 1000:.2f} ms/inference")
    print(f"  Speedup: {fp32_time / int8_time:.2f}x")


def save_and_load_quantized():
    """Demonstrate serialization of dynamically quantized models."""
    model = TextClassifier()
    model.eval()

    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )

    path = "quantized_text_classifier.pt"
    torch.save(quantized.state_dict(), path)

    loaded = TextClassifier()
    loaded_quantized = torch.ao.quantization.quantize_dynamic(
        loaded, {nn.Linear, nn.LSTM}, dtype=torch.qint8
    )
    loaded_quantized.load_state_dict(torch.load(path, weights_only=True))

    test_input = torch.randint(0, 10000, (1, 50))
    with torch.no_grad():
        orig_out = quantized(test_input)
        loaded_out = loaded_quantized(test_input)

    print(f"\n=== Save/Load Verification ===")
    print(f"  Outputs match: {torch.allclose(orig_out, loaded_out)}")

    import os
    os.remove(path)


if __name__ == "__main__":
    fp32_lstm, int8_lstm = quantize_lstm_model()
    fp32_transformer, int8_transformer = quantize_linear_only()

    benchmark_latency(
        fp32_lstm, int8_lstm,
        input_fn=lambda: torch.randint(0, 10000, (1, 50)),
        num_runs=50,
    )

    benchmark_latency(
        fp32_transformer, int8_transformer,
        input_fn=lambda: torch.randn(1, 32, 512),
        num_runs=50,
    )

    save_and_load_quantized()
