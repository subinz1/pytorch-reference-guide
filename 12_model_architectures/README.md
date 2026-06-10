<div align="center">

[← Previous Module](../11_export_deploy/) | [🏠 Home](../README.md) | [Next Module →](../13_advanced/)

</div>

---

> **Module 12** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 04 — Neural Networks](../04_neural_networks/), [Module 09 — Attention Mechanisms](../09_attention/)
> **Time to complete:** ~4 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `resnet.py` | ResNet (Residual Network) — complete implementation with BasicBlock and Bottleneck |
| `transformer.py` | Transformer — complete encoder-decoder implementation |
| `gpt.py` | GPT (Generative Pre-trained Transformer) — decoder-only implementation |
| `vit.py` | Vision Transformer (ViT) — complete implementation for image classification |
| `vae.py` | Variational Autoencoder (VAE) — complete implementation with reparameterization trick |

---

# Module 12: Model Architectures — From Paper to PyTorch

Building real neural network architectures is the bridge between understanding
PyTorch basics and doing real deep learning research or engineering. This module
walks through several landmark architectures, explaining the *why* behind each
design choice and showing you how to translate ideas from papers into working
PyTorch code.

---

## How to Read an Architecture Paper and Translate It to Code

Most deep learning papers follow a predictable structure. Learning to read them
systematically will save you enormous amounts of time.

### Step 1: Understand the Problem Statement

Before looking at the architecture, understand what problem the paper solves.
For ResNet, it's the "degradation problem" — deeper networks were performing
*worse* than shallower ones, even on training data. For Transformers, it was
the sequential bottleneck of RNNs that prevented parallelization.

### Step 2: Identify the Core Innovation

Every architecture paper has one or two key ideas. Everything else is
engineering around those ideas:

| Paper       | Core Innovation                          |
|-------------|------------------------------------------|
| ResNet      | Skip (residual) connections              |
| Transformer | Scaled dot-product self-attention        |
| GPT         | Decoder-only Transformer + autoregressive pretraining |
| ViT         | Treat image patches as token sequences   |
| VAE         | Reparameterization trick for differentiable sampling |
| U-Net       | Encoder-decoder with skip connections for dense prediction |

### Step 3: Map the Architecture Diagram to `nn.Module`s

Papers usually have an architecture diagram. Each box becomes either:
- An `nn.Module` subclass (if it's a reusable block)
- A line of code inside `forward()` (if it's a simple operation)

### Step 4: Match Dimensions

Papers describe tensor shapes. Track them through the network. A common
approach: write comments with shapes at each step in your `forward()` method
during development, then remove them once the code is tested.

### Step 5: Implement, Test with Random Data, Then Train

Always verify your architecture with random tensors before training:

```python
model = MyModel()
x = torch.randn(2, 3, 224, 224)  # batch=2, channels=3, 224x224
out = model(x)
print(out.shape)  # should be (2, num_classes)
```

---

## ResNet (Residual Networks)

**Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2015)

### The Degradation Problem

Before ResNet, researchers observed a paradox: adding more layers to a neural
network made it *worse*, even on the training set. This wasn't overfitting —
it was a fundamental optimization problem. Deeper networks were harder to
optimize because gradients had to flow through many layers.

### The Key Insight: Skip Connections

Instead of learning `H(x)` directly, learn the *residual* `F(x) = H(x) - x`,
then compute `H(x) = F(x) + x`. If the optimal transformation is close to
identity, it's easier to learn a small residual than to learn identity from
scratch.

```
Input x ──────────────────────┐
   │                          │
   ▼                          │
┌──────────┐                  │
│  Conv-BN  │                  │
│   ReLU    │                  │
│  Conv-BN  │                  │
└──────────┘                  │
   │                          │
   ▼                          │
  F(x)  ─────── + ◄──────────┘
   │
   ▼
  ReLU
   │
   ▼
 Output = ReLU(F(x) + x)
```

### BasicBlock vs Bottleneck

ResNet uses two block types:

**BasicBlock** (for ResNet-18 and ResNet-34):
- Two 3x3 convolutions
- Each followed by BatchNorm
- A skip connection that adds the input to the output
- If dimensions don't match, a 1x1 conv "projection" shortcut is used

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)
```

**Bottleneck** (for ResNet-50, 101, 152):
- Three convolutions: 1x1 (reduce), 3x3 (process), 1x1 (expand)
- The 1x1 convolutions reduce and restore channel dimensions
- This "bottleneck" design is more parameter-efficient for deep networks

The `expansion` factor controls how much the Bottleneck expands channels:
- BasicBlock: expansion = 1 (output channels == internal channels)
- Bottleneck: expansion = 4 (output channels == 4 * internal channels)

### Pre-Activation ResNet

The original ResNet applies BN and ReLU *after* each convolution. A later paper
("Identity Mappings in Deep Residual Networks", He et al., 2016) showed that
applying BN and ReLU *before* the convolution (pre-activation) improves both
optimization and generalization:

```python
# Post-activation (original): Conv -> BN -> ReLU
# Pre-activation (improved):  BN -> ReLU -> Conv
```

The intuition: in the pre-activation design, the skip connection is a true
identity mapping (no BN or ReLU on the shortcut path), allowing gradients to
flow unimpeded.

### ResNet Family Configurations

| Model     | Block      | Layers per stage    | Total layers | Parameters |
|-----------|------------|---------------------|--------------|------------|
| ResNet-18 | BasicBlock | [2, 2, 2, 2]       | 18           | ~11M       |
| ResNet-34 | BasicBlock | [3, 4, 6, 3]       | 34           | ~21M       |
| ResNet-50 | Bottleneck | [3, 4, 6, 3]       | 50           | ~25M       |
| ResNet-101| Bottleneck | [3, 4, 23, 3]      | 101          | ~44M       |

See `resnet.py` for the complete implementation.

---

## Transformer

**Paper**: "Attention Is All You Need" (Vaswani et al., 2017)

### Motivation

RNNs process sequences one token at a time — you can't compute step t until
step t-1 is done. This sequential bottleneck limits parallelism and makes it
hard to learn long-range dependencies. The Transformer replaces recurrence
entirely with attention mechanisms.

### Scaled Dot-Product Attention

The core operation. Given queries Q, keys K, and values V:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

- Q @ K^T computes similarity scores between every query and every key
- Division by sqrt(d_k) prevents dot products from becoming too large (which
  would push softmax into saturation, giving near-zero gradients)
- Softmax normalizes scores to a probability distribution
- Multiplication with V produces a weighted combination of values

### Multi-Head Attention

Instead of one big attention computation, split Q, K, V into `h` heads:

```python
# Instead of d_model-dimensional attention:
# Split into h heads, each of dimension d_k = d_model // h
# Compute attention independently per head
# Concatenate and project back to d_model
```

Why multiple heads? Each head can attend to different aspects of the input —
one head might focus on syntactic relationships, another on semantic ones.

### Self-Attention vs Cross-Attention

- **Self-attention**: Q, K, V all come from the same sequence. Each token
  attends to all other tokens in the same sequence.
- **Cross-attention**: Q comes from one sequence (decoder), K and V come from
  another (encoder output). This is how the decoder "reads" the encoder.

### Positional Encoding

Attention is permutation-invariant — it has no notion of order. Positional
encodings inject position information. The original paper uses sinusoidal
encodings:

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

These have a nice property: the encoding of position `pos+k` can be expressed
as a linear function of the encoding of position `pos`, making it possible for
the model to learn relative positions.

### Encoder-Decoder Architecture

**Encoder**: Stack of N identical layers, each containing:
1. Multi-head self-attention (with residual + LayerNorm)
2. Position-wise feed-forward network (with residual + LayerNorm)

**Decoder**: Stack of N identical layers, each containing:
1. Masked multi-head self-attention (causal mask prevents attending to future)
2. Multi-head cross-attention (attends to encoder output)
3. Position-wise feed-forward network

Each sub-layer uses the pattern: `LayerNorm(x + Sublayer(x))` (post-norm) or
`x + Sublayer(LayerNorm(x))` (pre-norm, more common now).

See `transformer.py` for the complete implementation.

---

## GPT (Decoder-Only Transformer)

**Paper**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)

### Design Philosophy

GPT simplifies the Transformer by keeping only the decoder (with causal
attention), removing the encoder and cross-attention entirely. The insight:
a powerful enough language model trained to predict the next token can learn
to perform many tasks without explicit task-specific architecture.

### Causal (Autoregressive) Attention

In a decoder-only model, each token can only attend to itself and previous
tokens. This is enforced by a triangular mask:

```python
# For sequence length 4:
mask = [[1, 0, 0, 0],   # token 0 sees only itself
        [1, 1, 0, 0],   # token 1 sees tokens 0-1
        [1, 1, 1, 0],   # token 2 sees tokens 0-2
        [1, 1, 1, 1]]   # token 3 sees tokens 0-3
```

Positions with 0 are set to -infinity before softmax, effectively zeroing
those attention weights.

### Weight Tying

GPT ties the token embedding matrix with the output projection (language
model head). If embedding maps token IDs to vectors, the LM head maps vectors
back to logits over the vocabulary — these are inverse operations, so sharing
weights makes sense and reduces parameters significantly:

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight  # weight tying
```

### Pre-Norm vs Post-Norm

Original Transformer: `x + Sublayer(LayerNorm(x))` — "post-norm"
GPT-2 and later:      `x + Sublayer(LayerNorm(x))` — "pre-norm"

Wait, the formulas look the same? The difference is subtle:
- Post-norm: `LayerNorm(x + Sublayer(x))` — norm is *outside* the residual
- Pre-norm: `x + Sublayer(LayerNorm(x))` — norm is *inside* the residual

Pre-norm is more stable for training deep networks because the residual
connection carries un-normalized values, preserving gradient magnitude.

### Generation Strategies

Given a trained model, how do you generate text?

**Greedy**: Always pick the most probable next token. Fast but repetitive.

**Temperature**: Divide logits by temperature T before softmax:
- T < 1: sharper distribution, more confident (less random)
- T = 1: original distribution
- T > 1: flatter distribution, more random

**Top-k sampling**: Keep only the top k most probable tokens, zero out the
rest, renormalize, then sample. Prevents sampling very unlikely tokens.

**Top-p (nucleus) sampling**: Keep the smallest set of tokens whose cumulative
probability exceeds p. Adapts the number of candidates dynamically — when the
model is confident, few tokens are kept; when uncertain, more are kept.

```python
def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[mask] = float('-inf')
    # Sample from the filtered distribution
    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, idx)
```

### KV Cache

During autoregressive generation, each new token only needs to attend to all
previous tokens. Without caching, you'd recompute K and V for the entire
sequence at every step. The KV cache stores previously computed K and V tensors
and only computes the new token's K and V, then concatenates:

```python
# Step 1: compute K, V for all tokens
# Step 2: only compute K_new, V_new for the new token
#          K = cat(K_cached, K_new)
#          V = cat(V_cached, V_new)
```

This reduces generation from O(n^2) to O(n) per token.

See `gpt.py` for the complete implementation.

---

## Vision Transformer (ViT)

**Paper**: "An Image Is Worth 16x16 Words" (Dosovitskiy et al., 2020)

### Core Idea

Treat an image as a sequence of patches and process them with a standard
Transformer encoder. No convolutions needed.

### Patch Embedding

Split the image into non-overlapping patches (e.g., 16x16 pixels), flatten
each patch into a vector, then project to the model dimension:

```python
# Image: (B, 3, 224, 224)
# Patches: 224/16 = 14 patches per side, 14*14 = 196 patches
# Each patch: 16*16*3 = 768 pixels

# Efficient implementation using Conv2d:
self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
# Output: (B, d_model, 14, 14) -> reshape to (B, 196, d_model)
```

### CLS Token

A special learnable token prepended to the sequence. After processing through
the Transformer, the CLS token's representation is used for classification:

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
# Prepend to patch sequence: (B, 196, d_model) -> (B, 197, d_model)
```

Why not just average-pool all patch representations? The CLS token provides a
single, fixed-position summary token that the model can learn to aggregate
global information into.

### Position Embedding

Since patches have spatial relationships, learnable position embeddings are
added (ViT uses learned embeddings, not sinusoidal):

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
# +1 for the CLS token
```

### Classification Head

After the Transformer encoder, take the CLS token's output and pass it
through a simple MLP head:

```python
cls_output = transformer_output[:, 0]  # CLS token at position 0
logits = self.mlp_head(cls_output)
```

See `vit.py` for the complete implementation.

---

## VAE (Variational Autoencoder)

**Paper**: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)

### Autoencoders vs Variational Autoencoders

A regular autoencoder learns to compress data to a latent code and reconstruct
it. The latent space, however, has no structure — nearby points in latent space
may decode to very different outputs, making it useless for generation.

A VAE forces the latent space to be structured (approximately Gaussian) by:
1. Encoding to a *distribution* (mean and variance) instead of a point
2. Sampling from that distribution (the reparameterization trick)
3. Adding a KL divergence loss that pushes the distribution toward N(0, I)

### The Reparameterization Trick

We can't backpropagate through random sampling. The trick: instead of sampling
`z ~ N(mu, sigma^2)`, compute `z = mu + sigma * epsilon` where
`epsilon ~ N(0, 1)`. Now the randomness is in epsilon (which doesn't need
gradients), and z is a deterministic function of mu and sigma.

```python
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### The ELBO Loss

The VAE loss has two terms:

```
Loss = Reconstruction Loss + KL Divergence
     = E[||x - x_hat||^2]  + KL(q(z|x) || p(z))
```

- **Reconstruction loss**: How well the decoder reconstructs the input.
  Binary cross-entropy for binary data, MSE for continuous data.
- **KL divergence**: How far the encoder's distribution is from the prior
  N(0, I). For Gaussians, this has a closed-form solution:

```python
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

The KL term acts as a regularizer, preventing the encoder from collapsing
the latent space to a single point (which would defeat the purpose of having
a generative model).

See `vae.py` for the complete implementation.

---

## U-Net

**Paper**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
(Ronneberger et al., 2015)

### Architecture Overview

U-Net has a symmetric encoder-decoder structure with skip connections:

```
Encoder                    Decoder
(downsampling)             (upsampling)
                     
[Input] ──────────────────────► [Output]
   │                               ▲
   ▼                               │
[Down1] ──── skip connection ──► [Up1]
   │                               ▲
   ▼                               │
[Down2] ──── skip connection ──► [Up2]
   │                               ▲
   ▼                               │
[Down3] ──── skip connection ──► [Up3]
   │                               ▲
   ▼                               │
[Down4] ──── skip connection ──► [Up4]
   │                               ▲
   ▼                               │
        [Bottleneck] ──────────────┘
```

### Why Skip Connections?

The encoder captures "what" is in the image (semantic features) but loses
spatial precision. The decoder recovers spatial resolution but lacks semantic
context. Skip connections combine both: the decoder gets high-resolution
features from the encoder concatenated with upsampled semantic features.

### Encoder Path

Each encoder block: two 3x3 convolutions + ReLU, then 2x2 max pooling.
Channels double at each level: 64 -> 128 -> 256 -> 512 -> 1024.

### Decoder Path

Each decoder block: 2x2 transposed convolution (upsample), concatenate with
the corresponding encoder features, then two 3x3 convolutions + ReLU.

### Key Difference from ResNet Skip Connections

- ResNet: skip connections *add* the input (element-wise addition)
- U-Net: skip connections *concatenate* encoder features with decoder features

This is because U-Net needs to preserve both the high-resolution spatial
information from the encoder and the semantic information from the decoder,
while ResNet just needs to facilitate gradient flow.

---

## Common Architectural Patterns

### Residual Connections

Found in almost every modern architecture. The core pattern:

```python
output = x + f(x)  # residual connection
```

Benefits: easier optimization, better gradient flow, ability to train very
deep networks.

### Pre-Norm vs Post-Norm

**Post-norm** (original Transformer):
```python
x = layer_norm(x + sublayer(x))
```

**Pre-norm** (GPT-2, modern practice):
```python
x = x + sublayer(layer_norm(x))
```

Pre-norm is more stable for training. Post-norm can achieve slightly better
performance but requires careful learning rate warmup.

### Weight Tying

Sharing parameters between the input embedding and the output projection:

```python
self.embed = nn.Embedding(vocab_size, d_model)
self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
self.output_proj.weight = self.embed.weight  # shared!
```

Reduces parameters by `vocab_size * d_model` and acts as regularization. Used
in GPT, BERT, T5, and most modern language models.

### Layer Scaling

Introduced in CaiT (Going Deeper with Image Transformers). Scale the output of
each residual block by a learnable scalar, initialized to a small value (e.g.,
0.1):

```python
self.gamma = nn.Parameter(torch.ones(d_model) * 0.1)
# In forward:
x = x + self.gamma * sublayer(x)
```

This helps stabilize training of very deep Transformers by starting with
near-identity blocks.

### GELU Activation

Most modern Transformers use GELU (Gaussian Error Linear Unit) instead of ReLU:

```python
F.gelu(x)  # smooth approximation: x * Phi(x)
```

GELU is smoother than ReLU and has been empirically shown to work better for
Transformers. It's the default in GPT, BERT, ViT, and most modern architectures.

### Dropout Patterns in Transformers

Dropout is typically applied in three places:
1. After attention weights (before multiplying with V)
2. After the feed-forward network's output projection
3. After adding positional embeddings (in some architectures)

```python
attn_weights = F.softmax(scores, dim=-1)
attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
```

### Weight Initialization

Different architectures use different initialization strategies:

```python
# Xavier/Glorot (good for tanh/sigmoid): used in Transformer
nn.init.xavier_uniform_(self.weight)

# Kaiming/He (good for ReLU): used in ResNet
nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

# Normal with small std: used in GPT
nn.init.normal_(self.weight, mean=0.0, std=0.02)
```

The right initialization prevents vanishing/exploding gradients at the start
of training and can significantly affect convergence speed.

---

## Summary

| Architecture | Year | Innovation | Key Pattern |
|-------------|------|------------|-------------|
| ResNet | 2015 | Residual connections | Identity shortcut |
| U-Net | 2015 | Encoder-decoder + skip | Concatenation skip |
| VAE | 2013 | Reparameterization trick | Stochastic latent |
| Transformer | 2017 | Self-attention | Q/K/V attention |
| GPT | 2018 | Decoder-only + pretrain | Causal masking |
| ViT | 2020 | Patch tokenization | Image as sequence |

## Files in This Module

- `resnet.py` — Complete ResNet with BasicBlock, Bottleneck, configs for 18/34/50/101
- `transformer.py` — Full encoder-decoder Transformer from scratch
- `gpt.py` — GPT with generation: greedy, temperature, top-k, top-p sampling
- `vit.py` — Vision Transformer for image classification
- `vae.py` — Variational Autoencoder with reparameterization trick and ELBO loss

---

<div align="center">

[← Previous Module](../11_export_deploy/) | [🏠 Home](../README.md) | [Next Module →](../13_advanced/)

**[📓 Open Notebook](../notebooks/09_model_architectures.ipynb)** — Interactive version of this module

</div>
