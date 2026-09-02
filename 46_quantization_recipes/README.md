# Module 46: Quantization Recipes

## Overview

**Quantization** reduces model size and inference latency by representing weights
and activations with lower-precision data types (INT8, INT4, FP8). PyTorch provides
three main approaches: **dynamic quantization**, **static quantization**, and
**quantization-aware training (QAT)**.

## Key Concepts

### Dynamic Quantization

Weights are quantized ahead of time; activations are quantized on-the-fly during
inference. Best for models dominated by `nn.Linear` (e.g., LSTMs, Transformers).
No calibration data required.

### Static Quantization

Both weights and activations are quantized using calibration data to determine
activation ranges. Requires inserting `QuantStub`/`DeQuantStub` and running a
representative dataset through the model. Produces faster inference than dynamic.

### Quantization-Aware Training (QAT)

Simulates quantization during training using fake-quantize operators. The model
learns to compensate for quantization error, yielding higher accuracy than
post-training quantization—especially for aggressive quantization (INT4).

### PyTorch 2 Export Quantization (pt2e)

The modern path uses `torch.export` + `torchao` for quantization. Define a
`Quantizer` that annotates the FX graph, then lower to a backend (XNNPack,
Executorch, etc.). This replaces the legacy `torch.quantization` eager-mode API.

### Data Types

| Type | Bits | Use Case |
|------|------|----------|
| INT8 | 8 | General-purpose server/edge inference |
| INT4 | 4 | LLM weight-only quantization |
| FP8 (E4M3/E5M2) | 8 | Training and inference on Hopper+ GPUs |
| UINT4 | 4 | Asymmetric weight packing (torchao) |

## Files in This Module

| File | Description |
|------|-------------|
| `dynamic_quantization.py` | Dynamic quantization with `torch.ao.quantization` |
| `static_quantization.py` | Static quantization with calibration and QAT |

## Examples

```python
# Dynamic quantization (simplest path)
import torch.ao.quantization as quant
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## References

- [Quantization docs](https://pytorch.org/docs/stable/quantization.html)
- [PT2E Quantization tutorial](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)
- [torchao](https://github.com/pytorch/ao) — Modern quantization and sparsity
- [Quantization-Aware Training](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [LLM.int8() paper](https://arxiv.org/abs/2208.07339)
