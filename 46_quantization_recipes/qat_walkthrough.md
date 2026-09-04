# QAT Walkthrough — Checklist & Pitfalls

Companion guide for [Module 46 — Quantization Recipes](../46_quantization_recipes/).

**Quantization-Aware Training (QAT)** inserts fake-quantize ops during training so
the model learns to tolerate INT8 (or similar) noise *before* conversion. This
doc is a practical checklist — not a full API reference.

## When QAT vs PTQ

| Approach | Use when |
|----------|----------|
| **PTQ** (post-training) | Accuracy holds after calibration; fastest path |
| **QAT** | PTQ drops accuracy; you can afford fine-tuning |
| **Dynamic quant** | Mostly linear layers / LSTM on CPU; weights quantized, activations dynamic |

Start with PTQ; escalate to QAT only if metrics fail.

## End-to-End Checklist

1. **Pick a backend / flow**
   - Eager / FX QAT (classic `torch.ao.quantization`)
   - PT2E / `torchao` paths for newer stacks (see Module 31)
2. **Fuse modules first** — Conv+BN+ReLU (or Linear+ReLU) before inserting observers/fake-quant.
3. **Attach QConfig** — per-tensor vs per-channel weights; activation observer dtype/range.
4. **Prepare** — `prepare_qat` / equivalent inserts `FakeQuantize`.
5. **Fine-tune** — lower LR (often 1–10% of original), few epochs; keep BN in train or follow recipe freeze schedule.
6. **Validate fake-quant accuracy** — still floating graph with noise; should be close to target INT8.
7. **Convert** — `convert` swaps fake-quant for real quantized modules.
8. **Validate converted model** — real INT8 kernels / packed weights; measure task metrics + latency.
9. **Export / deploy** — ensure the runtime (XNNPACK, TensorRT, Inductor, mobile) accepts the produced graph.

## Recommended Fine-Tune Settings

```text
LR:              1e-5 – 1e-4 (or 0.01× FP32 LR)
Epochs:          1–5 (task dependent)
BN:              follow fusion guide; often freeze BN late in QAT
Distillation:    optional — FP32 teacher soft labels help hard tasks
Calibration:     not a substitute for QAT, but useful for init ranges
```

## Pitfalls

### 1. Skipping fusion
BN folding after quantization is wrong; fuse **before** `prepare_qat` or ranges / scales will be wrong.

### 2. Training with eval-mode BN the whole time
BN running stats / affine params interact with fake-quant. Follow the backend recipe for `train()` vs freeze.

### 3. Too-high learning rate
Fake-quant is noisy; large LRs diverge or oscillate scales. Prefer gentle fine-tuning.

### 4. Quantizing everything blindly
Leave sensitive layers in FP32 (first/last conv, softmax, LayerNorm, some attention projections) when accuracy collapses.

### 5. Per-tensor weights on large Linear/Conv
Per-channel weight quant usually recovers accuracy for CNNs/Transformers; per-tensor is harsher.

### 6. Forgetting to `convert` before deployment benchmarks
Benchmarking fake-quant FP32 graph ≠ INT8 speed. Always time the converted model on the target device.

### 7. Observer / fake-quant mismatch at export
If export strips observers incorrectly, you get silent accuracy loss. Compare outputs: FP32 vs QAT-prepared vs converted on a fixed batch.

### 8. Data pipeline differs from production
Calibration / QAT fine-tune must see production-like preprocessing (resize, mean/std, tokenizers).

## Debugging Accuracy Drops

```text
FP32 baseline  ────────────────────────────  reference
   │
   ├─ fused FP32                          ─ should match closely
   ├─ QAT prepared (fake-quant)           ─ should be near INT8 target
   └─ converted INT8                      ─ final metric

If prepared ≫ converted gap → convert/backend packing issue
If FP32 ≫ prepared gap     → QConfig / bit-width / excluded layers
```

Log per-layer activation histograms when one block destroys accuracy.

## Minimal Eager-Style Sketch

```python
import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

model.train()
model.qconfig = get_default_qat_qconfig("x86")  # or "qnnpack"
# model = fuse_modules(model, [...])  # required for CNNs
model = prepare_qat(model)
# ... fine-tune ...
model.eval()
int8_model = convert(model)
```

APIs evolve — prefer the flow matching your PyTorch version (FX / PT2E / torchao).

## References

- Module scripts: `dynamic_quantization.py`, `static_quantization.py`
- [Quantization propagation](https://pytorch.org/docs/stable/quantization.html)
- [QAT tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- Module 31: [torchao](../31_torchao/)
