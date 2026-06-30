# Module 33: Model Interpretability with Hooks

<div align="center">

[← Previous Module (Efficient Data Pipelines)](../32_data_pipelines/) | [🏠 Home](../README.md) | [Next Module (End-to-End: Fine-Tuning an LLM) →](../34_llm_finetuning/)

</div>

---

> **Prerequisites**: [Module 04 — Neural Networks](../04_neural_networks/), [Module 07 — Training Pipelines](../07_training/)
> **Time**: ~2 hours
> **Files**: `hook_techniques.py`, `gradcam_saliency.py`

---

## Table of Contents

1. [What Are Hooks?](#1-what-are-hooks)
2. [Forward Hooks](#2-forward-hooks)
3. [Forward Pre-Hooks](#3-forward-pre-hooks)
4. [Backward Hooks](#4-backward-hooks)
5. [Tensor Hooks](#5-tensor-hooks)
6. [Activation Extraction](#6-activation-extraction)
7. [Grad-CAM (Gradient-weighted Class Activation Mapping)](#7-grad-cam-gradient-weighted-class-activation-mapping)
8. [Saliency Maps](#8-saliency-maps)
9. [Attention Map Extraction](#9-attention-map-extraction)
10. [Guided Backpropagation](#10-guided-backpropagation)
11. [Practical Tips](#11-practical-tips)
12. [Upstream Updates (June 27–29, 2026)](#12-upstream-updates-june-2729-2026)

---

## 1. What Are Hooks?

Hooks are callbacks registered on modules or tensors that execute during forward or backward passes. They let you inspect or modify activations and gradients **without changing model code**.

Three types of module hooks:

| Hook Type | Registration | Signature | When It Runs |
|-----------|-------------|-----------|--------------|
| Forward hook | `module.register_forward_hook(fn)` | `fn(module, input, output)` | After `forward()` returns |
| Forward pre-hook | `module.register_forward_pre_hook(fn)` | `fn(module, input)` | Before `forward()` executes |
| Backward hook | `module.register_full_backward_hook(fn)` | `fn(module, grad_input, grad_output)` | During `backward()` |

Plus one tensor-level hook:

| Hook Type | Registration | Signature | When It Runs |
|-----------|-------------|-----------|--------------|
| Tensor hook | `tensor.register_hook(fn)` | `fn(grad)` | When gradient is computed for that tensor |

All registration methods return a `RemovableHandle`. Call `handle.remove()` to unregister.

### Why Hooks Matter

Without hooks, inspecting intermediate activations requires modifying the model's `forward()` method — breaking encapsulation, cluttering code, and requiring different code paths for inference vs. debugging. Hooks decouple observation from computation:

```
Model Code (unchanged)          Observer Code (hooks)
┌─────────────────────┐        ┌──────────────────────┐
│ class MyModel:      │        │ activations = {}     │
│   def forward(x):   │        │                      │
│     x = self.conv(x)│──hook──│ store conv output    │
│     x = self.relu(x)│──hook──│ store relu output    │
│     x = self.fc(x)  │──hook──│ store fc output      │
│     return x        │        │                      │
└─────────────────────┘        └──────────────────────┘
```

---

## 2. Forward Hooks

A forward hook runs **after** a module's `forward()` completes:

```python
def hook_fn(module, input, output):
    # module: the nn.Module instance
    # input:  tuple of input tensors
    # output: the module's return value
    print(f"{module.__class__.__name__}: output shape = {output.shape}")

handle = model.layer1.register_forward_hook(hook_fn)
output = model(x)  # hook_fn called when layer1 executes
handle.remove()    # always clean up
```

### Use Cases

**Extract intermediate activations:**

```python
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

model.conv1.register_forward_hook(save_activation('conv1'))
model.conv2.register_forward_hook(save_activation('conv2'))
model(x)
# activations['conv1'] and activations['conv2'] now populated
```

**Log shapes for debugging:**

```python
def shape_hook(module, input, output):
    in_shape = input[0].shape if isinstance(input, tuple) else input.shape
    out_shape = output.shape if hasattr(output, 'shape') else type(output)
    print(f"{module.__class__.__name__}: {in_shape} -> {out_shape}")
```

**Modify outputs** (use carefully — can break autograd during training):

```python
def clamp_hook(module, input, output):
    return torch.clamp(output, -10, 10)
```

### The `with_kwargs` Parameter

PyTorch 2.x added support for keyword arguments in hooks:

```python
def hook_with_kwargs(module, input, kwargs, output):
    # kwargs is a dict of keyword arguments passed to forward()
    pass

handle = model.register_forward_hook(hook_with_kwargs, with_kwargs=True)
```

---

## 3. Forward Pre-Hooks

Pre-hooks run **before** the module's `forward()`:

```python
def pre_hook_fn(module, input):
    # input is a tuple of positional arguments to forward()
    # Return None to leave input unchanged, or return modified input
    print(f"Input to {module.__class__.__name__}: shape={input[0].shape}")

handle = model.layer1.register_forward_pre_hook(pre_hook_fn)
```

### Use Cases

**Input validation:**

```python
def validate_input(module, input):
    x = input[0]
    if torch.isnan(x).any():
        raise ValueError(f"NaN detected in input to {module.__class__.__name__}")
    if torch.isinf(x).any():
        raise ValueError(f"Inf detected in input to {module.__class__.__name__}")
```

**Input normalization:**

```python
def normalize_input(module, input):
    x = input[0]
    return (x - x.mean()) / (x.std() + 1e-8),
```

**Shape modification:**

```python
def reshape_for_conv(module, input):
    x = input[0]
    if x.dim() == 3:
        return x.unsqueeze(1),  # Add channel dimension
```

---

## 4. Backward Hooks

Backward hooks run during the backward pass and provide access to gradients:

```python
def backward_hook(module, grad_input, grad_output):
    # grad_input:  tuple of gradients w.r.t. module inputs
    # grad_output: tuple of gradients w.r.t. module outputs
    print(f"{module.__class__.__name__}: grad_output norm = {grad_output[0].norm():.4f}")

handle = model.layer1.register_full_backward_hook(backward_hook)
loss = criterion(model(x), target)
loss.backward()  # backward_hook called during backward pass
handle.remove()
```

### `register_full_backward_hook` vs `register_backward_hook`

Always use `register_full_backward_hook`. The older `register_backward_hook` has known issues with modules that have multiple inputs and is deprecated.

### Gradient Monitoring

```python
def gradient_monitor(name):
    def hook(module, grad_input, grad_output):
        grad = grad_output[0]
        stats = {
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'norm': grad.norm().item(),
            'max': grad.abs().max().item(),
            'has_nan': torch.isnan(grad).any().item(),
        }
        print(f"[{name}] {stats}")
    return hook
```

### Gradient Modification

Backward hooks can modify gradients by returning new values:

```python
def clip_grad_hook(module, grad_input, grad_output):
    clipped = tuple(
        torch.clamp(g, -1.0, 1.0) if g is not None else None
        for g in grad_input
    )
    return clipped
```

---

## 5. Tensor Hooks

Tensor hooks operate on individual tensors rather than modules. They're called when the gradient for that specific tensor is computed:

```python
x = torch.randn(3, requires_grad=True)

def tensor_hook(grad):
    print(f"Gradient for x: {grad}")
    return grad * 2  # Optionally modify the gradient

handle = x.register_hook(tensor_hook)
y = (x ** 2).sum()
y.backward()  # prints gradient, then doubles it
handle.remove()
```

### Use Cases

**Per-parameter gradient logging:**

```python
for name, param in model.named_parameters():
    param.register_hook(
        lambda grad, n=name: print(f"{n}: grad norm = {grad.norm():.4f}")
    )
```

**Per-tensor gradient clipping:**

```python
for param in model.parameters():
    param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
```

**Freezing specific gradients:**

```python
# Zero out gradients for specific parameters
param.register_hook(lambda grad: torch.zeros_like(grad))
```

---

## 6. Activation Extraction

The most common hook pattern: extract activations from target layers without modifying the model.

### FeatureExtractor Class

```python
class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.features = {}
        self._handles = []

        for name, module in model.named_modules():
            if name in target_layers:
                handle = module.register_forward_hook(self._make_hook(name))
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
```

### Usage

```python
extractor = FeatureExtractor(model, ['layer1', 'layer2.conv1', 'layer3'])
output, features = extractor(input_tensor)
print(features['layer1'].shape)  # Intermediate activations
extractor.close()
```

### Activation Statistics

Beyond raw activations, hooks can compute statistics on-the-fly:

```python
class ActivationStats:
    def __init__(self, model):
        self.stats = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                handle = module.register_forward_hook(self._stats_hook(name))
                self._handles.append(handle)

    def _stats_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'dead_fraction': (output == 0).float().mean().item(),
                'max': output.max().item(),
            }
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
```

The `dead_fraction` metric (fraction of ReLU outputs that are exactly zero) is particularly useful — a high dead neuron fraction suggests the learning rate is too high or the initialization is poor.

---

## 7. Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM visualizes which spatial regions of an input a CNN focuses on for a particular class prediction.

### Algorithm

1. Run forward pass, hook the **last convolutional layer** to capture its output activations `A`
2. Compute the gradient of the target class score w.r.t. `A`
3. Global-average-pool these gradients across spatial dimensions to get weights `α`
4. Compute weighted combination: `L = ReLU(Σ αk · Ak)`
5. Upsample `L` to input resolution

```
Input Image ──▶ CNN ──▶ [Last Conv Layer] ──▶ FC ──▶ Class Score
                             │                        │
                        Activations A            Gradient ∂y/∂A
                             │                        │
                             ▼                        ▼
                     Weighted Sum ◀── GAP(gradient) = weights α
                             │
                         ReLU + Upsample
                             │
                         Heatmap
```

### Why It Works

The global-average-pooled gradients represent the importance of each feature map channel for the target class. Weighting the activations by these importance values and taking the ReLU (we only care about features that have a positive influence) produces a coarse localization map.

### Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Global average pool gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to input size
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        return cam.squeeze()
```

---

## 8. Saliency Maps

The simplest gradient-based attribution method. Shows which input pixels most affect the model's prediction.

### Algorithm

1. Set `input.requires_grad_(True)`
2. Forward pass → get class score for target class
3. Backward pass → compute `∂score/∂input`
4. Take the absolute value of the gradient
5. For RGB images: take the max across channels

```python
def saliency_map(model, input_tensor, target_class):
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)

    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()

    # Absolute gradient, max across channels
    saliency = input_tensor.grad.abs()
    if saliency.dim() == 4:
        saliency = saliency.squeeze(0).max(dim=0).values

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency
```

### Interpreting Saliency Maps

- Bright regions = pixels that strongly influence the predicted class
- A good model should highlight the object, not the background
- Saliency maps are noisy — they show pixel-level sensitivity, not necessarily semantic understanding

### Limitations

- Very noisy compared to Grad-CAM
- Sensitive to input perturbations
- Doesn't capture spatial coherence
- Shows sensitivity, not necessarily relevance

---

## 9. Attention Map Extraction

For Transformer models, attention weights show which tokens attend to which. Hook into `MultiheadAttention` to capture them:

```python
class AttentionExtractor:
    def __init__(self, model):
        self.attention_maps = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                handle = module.register_forward_hook(self._attn_hook(name))
                self._handles.append(handle)

    def _attn_hook(self, name):
        def hook(module, input, output):
            # MHA returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2:
                self.attention_maps[name] = output[1].detach()
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
```

When calling `MultiheadAttention`, pass `need_weights=True` (default) and `average_attn_weights=False` to get per-head attention weights with shape `(batch, num_heads, seq_len, seq_len)`.

### Visualizing Attention

For a sequence `["The", "cat", "sat", "on", "mat"]`, attention weights form a matrix showing how each token attends to every other token:

```
         The   cat   sat   on   mat
The    [ 0.1   0.3   0.2  0.1  0.3 ]
cat    [ 0.2   0.1   0.4  0.1  0.2 ]
sat    [ 0.1   0.4   0.1  0.3  0.1 ]
on     [ 0.1   0.1   0.3  0.1  0.4 ]
mat    [ 0.3   0.2   0.1  0.1  0.3 ]
```

Each row sums to 1.0 (softmax). High values indicate strong attention from the row token to the column token.

---

## 10. Guided Backpropagation

Standard backpropagation through ReLU gates gradients based on the **forward pass** mask (positive activations). Guided backpropagation additionally masks out **negative gradients**, producing sharper attribution maps.

### Algorithm

At each ReLU during backward:
- Standard backprop: pass gradient where forward activation > 0
- Guided backprop: pass gradient where forward activation > 0 **AND** gradient > 0

```python
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self._handles = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_full_backward_hook(self._relu_backward_hook)
                self._handles.append(handle)

    def _relu_backward_hook(self, module, grad_input, grad_output):
        # Only pass positive gradients
        return (torch.clamp(grad_output[0], min=0.0),)

    def generate(self, input_tensor, target_class):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        guided_grads = input_tensor.grad.clone()
        return guided_grads

    def close(self):
        for h in self._handles:
            h.remove()
```

### Comparison of Methods

| Method | Granularity | Sharpness | Speed | Complexity |
|--------|-------------|-----------|-------|------------|
| Saliency maps | Pixel | Low | Fast | Trivial |
| Grad-CAM | Region | Medium | Fast | Low |
| Guided backprop | Pixel | High | Fast | Low |
| Guided Grad-CAM | Pixel | High | Fast | Medium |

Guided Grad-CAM combines Grad-CAM with guided backprop by element-wise multiplication:

```python
guided_gradcam = guided_grads * F.interpolate(gradcam, size=input_size)
```

---

## 11. Practical Tips

### Always Remove Hooks

Hooks that are not removed cause memory leaks — the hook closure holds a reference to whatever it captures:

```python
# BAD: hook leaks if exception occurs
handle = model.layer.register_forward_hook(my_hook)
output = model(x)
handle.remove()

# GOOD: use try/finally
handle = model.layer.register_forward_hook(my_hook)
try:
    output = model(x)
finally:
    handle.remove()
```

### Context Manager Pattern

Wrap hook registration in a context manager for automatic cleanup:

```python
from contextlib import contextmanager

@contextmanager
def hook_context(module, hook_fn, hook_type='forward'):
    if hook_type == 'forward':
        handle = module.register_forward_hook(hook_fn)
    elif hook_type == 'backward':
        handle = module.register_full_backward_hook(hook_fn)
    elif hook_type == 'pre':
        handle = module.register_forward_pre_hook(hook_fn)
    try:
        yield handle
    finally:
        handle.remove()

with hook_context(model.layer1, my_hook):
    output = model(x)
# hook automatically removed
```

### Don't Modify Outputs During Training

Returning a new value from a forward hook replaces the module's output. This can break autograd graph construction during training:

```python
# DANGEROUS during training — breaks gradient computation
def bad_hook(module, input, output):
    return output.detach()  # Detaches from autograd graph!
```

For training, only **read** from hooks. Save activations with `.detach()` to avoid holding the entire graph in memory, but don't replace the module output.

### Hook Execution Order

Multiple hooks on the same module execute in registration order:

```python
model.layer.register_forward_hook(hook_a)  # runs first
model.layer.register_forward_hook(hook_b)  # runs second
```

If hook_a returns a modified output, hook_b receives that modified output.

### Performance Considerations

- Hooks add overhead per module per forward/backward call
- `.detach().cpu()` in hooks moves data off GPU — useful for memory but adds transfer cost
- For large-scale profiling, consider disabling hooks after collecting enough data
- Hooks are **not** compatible with `torch.compile` in all cases — test before deploying

---

## 12. Upstream Updates (June 27–29, 2026)

Recent PyTorch commits relevant to model interpretability, hooks, and inference:

### Inductor CompiledArtifact Binary Extraction ([#187850](https://github.com/pytorch/pytorch/pull/187850))

New API for extracting compiled artifact binaries from Inductor. This enables better introspection of compiled models — inspecting the actual generated code and binaries that `torch.compile` produces. Useful for understanding what Inductor does under the hood and debugging compilation issues.

### MPS CTC Loss Backward ([#188187](https://github.com/pytorch/pytorch/pull/188187))

Backward pass implementation for CTC loss on Apple MPS devices. Previously, CTC loss gradients had to be computed on CPU even when the forward pass ran on MPS. This enables full MPS training for speech recognition and OCR models.

### MPS BatchNorm channels_last Fix ([#188371](https://github.com/pytorch/pytorch/pull/188371))

Fixes BatchNorm computation on MPS for tensors in channels-last memory format. The previous implementation produced incorrect results when the input tensor used `torch.channels_last` memory layout, which is the preferred format for CNN inference.

### torch._check LiteralString Enforcement ([#188274](https://github.com/pytorch/pytorch/pull/188274))

`torch._check` now enforces `LiteralString` type for its message argument. This prevents accidental injection of dynamic strings into check messages and ensures that constraint messages are static, making them safer for graph export and compilation.

### Dynamo CPython str Semantics Fix ([#187775](https://github.com/pytorch/pytorch/pull/187775))

Fixes string operation semantics in Dynamo tracing to match CPython behavior. Previously, certain string operations during tracing could produce incorrect results or graph breaks. This improves model traceability for code that manipulates strings in control flow.

### AO control_deps Ordering Fixes

Fixes ordering issues in `control_deps` for the Architecture Optimization (AO) library. Ensures correct operation ordering when quantization and sparsity transforms interact with control flow, particularly important for models that use conditional computation patterns.

---

## Putting It All Together

A typical interpretability workflow:

```python
model = load_pretrained_model()
model.eval()

# 1. Feature extraction
extractor = FeatureExtractor(model, ['features.28', 'features.14'])
output, features = extractor(image)

# 2. Grad-CAM for spatial attribution
cam = GradCAM(model, model.features[28])
heatmap = cam.generate(image, target_class=predicted_class)

# 3. Saliency for pixel-level sensitivity
saliency = saliency_map(model, image, predicted_class)

# 4. Compare and analyze
print(f"Grad-CAM highlights: {(heatmap > 0.5).sum()} pixels")
print(f"High-saliency pixels: {(saliency > 0.5).sum()}")

extractor.close()
```

### When to Use Each Method

| Goal | Method |
|------|--------|
| "What features did the model extract?" | Activation extraction |
| "Where does the model look?" | Grad-CAM |
| "Which pixels matter most?" | Saliency maps |
| "What does the model see at each level?" | Guided backprop |
| "Which tokens attend to which?" | Attention extraction |
| "Is training stable?" | Gradient monitoring hooks |
| "Are neurons dying?" | Activation statistics |

---

### Further Resources

- [PyTorch Hooks Documentation](https://pytorch.org/docs/stable/nn.html#hooks) — official API reference
- [Module 04 — Neural Networks](../04_neural_networks/) — `nn.Module` fundamentals and hooks overview
- [Module 07 — Training Pipelines](../07_training/) — training loop patterns
- [Module 30 — Debugging](../30_debugging/) — anomaly detection and gradient debugging
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017) — original Grad-CAM paper
- Springenberg et al., "Striving for Simplicity" (2015) — guided backpropagation

---

<div align="center">

[← Previous Module (Efficient Data Pipelines)](../32_data_pipelines/) | [🏠 Home](../README.md) | [Next Module (End-to-End: Fine-Tuning an LLM) →](../34_llm_finetuning/)

**Notebook**: [`33_interpretability.ipynb`](../notebooks/33_interpretability.ipynb)

</div>
