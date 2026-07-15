<div align="center">

[← Previous Module (Compiled Autograd & AOTAutograd)](../38_compiled_autograd/) | [🏠 Home](../README.md) | Next Module → (none)

</div>

---

# Module 39: Building a Text Classifier

> **Prerequisites**: [Module 04 — Neural Networks](../04_neural_networks/), [Module 05 — Optimizers](../05_optimizers/), [Module 06 — Data Loading](../06_data_loading/), [Module 07 — Training Pipelines](../07_training/), [Module 09 — Attention Mechanisms](../09_attention/)
> **Time**: ~4 hours
> **Files**: `tokenizer.py`, `text_classifier.py`, `train_and_evaluate.py`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tokenization](#2-tokenization)
3. [Embedding Layer](#3-embedding-layer)
4. [Model Architecture](#4-model-architecture)
5. [Dataset & DataLoader](#5-dataset--dataloader)
6. [Training Loop](#6-training-loop)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Inference Pipeline](#8-inference-pipeline)
9. [torch.compile for Serving](#9-torchcompile-for-serving)
10. [Improvements](#10-improvements)
11. [Upstream Updates (July 7-15, 2026)](#11-upstream-updates-july-7-15-2026)

---

## 1. Project Overview

This module builds a **sentiment classifier** entirely from scratch. No pretrained models, no Hugging Face tokenizers, no external NLP libraries. You will implement every piece yourself:

```
Raw Text → Tokenizer → Token IDs → Embedding → Transformer Encoder → Classification Head → Prediction
                                                                                           ↓
                                                                              positive / negative / neutral
```

**Why build from scratch?** Using pretrained models is the right production choice, but it hides the mechanics. Building each component teaches you:

- How text becomes numbers (tokenization)
- How numbers become vectors (embeddings)
- How vectors become predictions (transformer + classifier)
- How to train the whole system end-to-end

### What We'll Build

| Component | What It Does | File |
|-----------|-------------|------|
| Character Tokenizer | Splits text into characters, maps to integers | `tokenizer.py` |
| Word Tokenizer | Splits on whitespace, builds frequency-based vocabulary | `tokenizer.py` |
| TransformerTextClassifier | Embedding + positional encoding + transformer encoder + classification head | `text_classifier.py` |
| Training Pipeline | Synthetic data, training loop, evaluation, inference | `train_and_evaluate.py` |

### Input/Output

```
Input:  "This movie was absolutely fantastic and I loved every moment"
Output: {"label": "positive", "confidence": 0.94}

Input:  "Terrible acting, awful script, waste of time"
Output: {"label": "negative", "confidence": 0.91}

Input:  "The movie was okay, nothing special"
Output: {"label": "neutral", "confidence": 0.72}
```

Everything runs on **CPU** — no GPU required.

---

## 2. Tokenization

Tokenization converts raw text into a sequence of integers that a neural network can process. This is the first — and often most impactful — design decision in any NLP pipeline.

### Why Tokenize?

Neural networks operate on numbers, not strings. We need a mapping:

```
"hello world" → [7, 4, 11, 11, 14, 0, 22, 14, 17, 11, 3]   # character-level
"hello world" → [42, 103]                                     # word-level
"hello world" → [7592, 1079]                                  # subword (BPE)
```

Each approach trades off vocabulary size, sequence length, and expressiveness.

### Character-Level Tokenizer

The simplest approach: each character is a token.

```python
text = "hello"
tokens = ['h', 'e', 'l', 'l', 'o']
ids   = [7, 4, 11, 11, 14]
```

**Advantages:**
- Tiny vocabulary (26 letters + punctuation + digits ≈ 100 tokens)
- No out-of-vocabulary (OOV) words — every string can be tokenized
- Can handle typos, neologisms, any language with the same alphabet

**Disadvantages:**
- Long sequences — "transformer" is 11 tokens, not 1
- Hard for the model to learn word-level semantics from characters
- Self-attention cost is O(n²) in sequence length

### Word-Level Tokenizer

Split on whitespace and punctuation. Each word is a token.

```python
text = "Hello, world!"
tokens = ['hello', ',', 'world', '!']
ids   = [42, 3, 103, 4]
```

**Advantages:**
- Short sequences — each word is one token
- Semantically meaningful units
- Easy to implement

**Disadvantages:**
- Large vocabulary (English has 170,000+ words)
- OOV problem — unseen words map to `[UNK]`
- Can't handle morphology ("running" and "runs" are separate tokens)

### Subword Tokenization (BPE Concept)

Byte-Pair Encoding (BPE) finds a middle ground. It starts with characters and iteratively merges the most frequent pairs:

```
Iteration 0: ['l', 'o', 'w', 'e', 'r', 'l', 'o', 'w', 'e', 's', 't']
Merge ('l', 'o') → 'lo':  ['lo', 'w', 'e', 'r', 'lo', 'w', 'e', 's', 't']
Merge ('lo', 'w') → 'low': ['low', 'e', 'r', 'low', 'e', 's', 't']
Merge ('low', 'e') → 'lowe': ['lowe', 'r', 'lowe', 's', 't']
```

BPE handles rare words by splitting them into known subwords:

```
"unhappiness" → ["un", "happiness"]
"transformers" → ["transform", "ers"]
```

We won't implement BPE in this module (it's complex), but understanding the concept explains why production systems use it. Our word-level tokenizer with `[UNK]` handling is sufficient for learning the full pipeline.

### Special Tokens

Every tokenizer needs special tokens that carry structural information:

| Token | Purpose | Typical ID |
|-------|---------|-----------|
| `[PAD]` | Fills sequences to equal length for batching | 0 |
| `[UNK]` | Replaces out-of-vocabulary words | 1 |
| `[CLS]` | Classification token — its embedding becomes the sequence representation | 2 |
| `[SEP]` | Separator between segments (for pair classification) | 3 |

### Building a Vocabulary

The vocabulary is a bidirectional mapping between tokens and integer IDs:

```python
# Forward: token → id
vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "the": 3, "movie": 4, ...}

# Reverse: id → token
id_to_token = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "the", 4: "movie", ...}
```

Building from data:
1. Tokenize all training texts
2. Count token frequencies
3. Keep the top N most frequent tokens (vocabulary size)
4. Assign integer IDs (special tokens first)

### Encoding and Decoding

**Encoding** (text → token IDs):

```python
text = "great movie"
# 1. Tokenize: ["great", "movie"]
# 2. Prepend [CLS]: ["[CLS]", "great", "movie"]
# 3. Map to IDs: [2, 57, 4]
# 4. Pad to max_length=8: [2, 57, 4, 0, 0, 0, 0, 0]
```

**Decoding** (token IDs → text):

```python
ids = [2, 57, 4, 0, 0, 0, 0, 0]
# 1. Map to tokens: ["[CLS]", "great", "movie", "[PAD]", "[PAD]", ...]
# 2. Remove special tokens: ["great", "movie"]
# 3. Join: "great movie"
```

### Padding and Truncation

Batching requires all sequences to have the same length. Two operations handle this:

**Padding** — add `[PAD]` tokens to short sequences:
```
"good"     → [CLS, good, PAD, PAD, PAD]   # length 5
"very good" → [CLS, very, good, PAD, PAD]  # length 5
```

**Truncation** — cut long sequences to `max_length`:
```
"this is a very long sentence that exceeds the limit"
→ [CLS, this, is, a, very]   # max_length=5, truncated
```

The padding mask tells the model which positions are real tokens vs padding:
```
tokens: [CLS, good, PAD, PAD, PAD]
mask:   [  1,    1,   0,   0,   0]  # 1=attend, 0=ignore
```

See `tokenizer.py` for the complete implementation.

---

## 3. Embedding Layer

Tokenization gives us integers. The embedding layer converts each integer into a dense vector that the model can work with.

### nn.Embedding: The Lookup Table

`nn.Embedding` is simply a matrix of shape `(vocab_size, embed_dim)`. Looking up token ID `i` returns row `i`:

```python
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=128)
# embedding.weight.shape = (1000, 128)

token_ids = torch.tensor([42, 7, 103])
vectors = embedding(token_ids)  # shape: (3, 128)
# vectors[0] = embedding.weight[42]
# vectors[1] = embedding.weight[7]
# vectors[2] = embedding.weight[103]
```

### Random Init vs Learned Embeddings

At initialization, embedding vectors are random — "dog" and "cat" have no special relationship. During training, backpropagation adjusts the vectors so that semantically similar words end up near each other in embedding space:

```
Before training:
  "good"     = [0.23, -0.81, 0.45, ...]  (random)
  "great"    = [-0.12, 0.67, -0.33, ...]  (random)
  "terrible" = [0.56, 0.09, 0.71, ...]    (random)

After training:
  "good"     = [0.82, 0.41, -0.15, ...]
  "great"    = [0.79, 0.38, -0.12, ...]   (close to "good")
  "terrible" = [-0.71, -0.33, 0.65, ...]  (far from "good")
```

### padding_idx

Padding tokens should not contribute to the model's computation. Setting `padding_idx=0` ensures:
1. The embedding vector for token 0 is always zeros
2. Gradients don't flow through padding positions

```python
embedding = nn.Embedding(1000, 128, padding_idx=0)
# embedding.weight[0] is always [0, 0, 0, ..., 0]
# No gradient updates for token 0
```

### Positional Embeddings

Self-attention is permutation-invariant — it doesn't know token order. "The cat sat" and "sat cat the" produce identical attention outputs without positional information.

**Sinusoidal positional encoding** (from "Attention Is All You Need"):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This produces a unique vector for each position. Properties:
- Each position gets a distinct encoding
- The model can learn to attend to relative positions
- Generalizes to longer sequences than seen during training

**Learned positional embeddings** are an alternative — just another `nn.Embedding(max_length, embed_dim)` that's added to the token embeddings. We use sinusoidal in our implementation.

The final input to the transformer is:

```
input = token_embedding(token_ids) + positional_encoding(positions)
```

Both are `(batch, seq_len, embed_dim)` tensors.

---

## 4. Model Architecture

Our classifier uses a **Transformer encoder** architecture. This is the same encoder used in BERT, but built from scratch with modern best practices.

### Architecture Diagram

```
Input Token IDs: (batch, seq_len)
        │
        ▼
┌─────────────────────┐
│   nn.Embedding      │  (vocab_size, d_model)
│   + PositionalEnc   │  (max_len, d_model)
│   + Dropout         │
└─────────┬───────────┘
          │
          ▼  (batch, seq_len, d_model)
┌─────────────────────┐
│  TransformerEncoder  │  N layers ×:
│  ┌─────────────────┐ │    - LayerNorm (pre-norm)
│  │ Self-Attention   │ │    - Multi-Head Attention (SDPA)
│  │ (SDPA)          │ │    - Residual connection
│  ├─────────────────┤ │    - LayerNorm (pre-norm)
│  │ Feed-Forward    │ │    - FFN: Linear → GELU → Linear
│  │ Network         │ │    - Residual connection
│  └─────────────────┘ │
│  (× N layers)       │
└─────────┬───────────┘
          │
          ▼  (batch, seq_len, d_model)
┌─────────────────────┐
│   Pooling            │  CLS token → (batch, d_model)
│   (CLS or Mean)      │  or Mean pool → (batch, d_model)
└─────────┬───────────┘
          │
          ▼  (batch, d_model)
┌─────────────────────┐
│  Classification Head │  LayerNorm → Linear → num_classes
└─────────┬───────────┘
          │
          ▼  (batch, num_classes)
        Logits
```

### Component Details

**Embedding + Positional Encoding:**
```python
# Token embedding: (batch, seq_len) → (batch, seq_len, d_model)
x = self.embedding(token_ids)           # shape: (B, S, D)
x = x + self.pos_encoder(positions)     # shape: (B, S, D)
x = self.dropout(x)                     # shape: (B, S, D)
```

**TransformerEncoderLayer (pre-norm variant):**

Pre-norm applies LayerNorm before (not after) each sublayer. This improves training stability — gradients flow more smoothly through the residual connections.

```python
# Pre-norm self-attention
residual = x
x = self.norm1(x)                       # shape: (B, S, D)
x = self.self_attn(x, x, x, mask)       # shape: (B, S, D)
x = residual + self.dropout(x)          # shape: (B, S, D)

# Pre-norm feed-forward
residual = x
x = self.norm2(x)                       # shape: (B, S, D)
x = self.ffn(x)                         # shape: (B, S, D)
x = residual + self.dropout(x)          # shape: (B, S, D)
```

**SDPA (Scaled Dot-Product Attention):**

PyTorch's `F.scaled_dot_product_attention` fuses Q·K^T/√d, mask, softmax, and V multiplication into one efficient operation:

```python
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=padding_mask,
    dropout_p=self.dropout_p if self.training else 0.0,
)
```

**Pooling Strategies:**

Two approaches to get a fixed-size vector from variable-length sequences:

1. **CLS token pooling**: Take the hidden state of the `[CLS]` token (position 0). The `[CLS]` token attends to all other tokens, so its representation captures the whole sequence.

```python
pooled = encoder_output[:, 0, :]  # (batch, d_model) — CLS position
```

2. **Mean pooling**: Average all non-padding token representations. More robust — doesn't rely on a single token learning to summarize everything.

```python
mask = padding_mask.unsqueeze(-1)               # (batch, seq_len, 1)
pooled = (encoder_output * mask).sum(1)         # (batch, d_model)
pooled = pooled / mask.sum(1).clamp(min=1e-9)   # normalize by actual length
```

**Classification Head:**

A simple linear projection from `d_model` to `num_classes`:

```python
logits = self.classifier(pooled)  # (batch, d_model) → (batch, num_classes)
```

### Shape Annotations

Following the data through the model for `batch=32, seq_len=64, d_model=128, num_heads=4, num_layers=2, num_classes=3`:

```
token_ids:      (32, 64)           — input
embedded:       (32, 64, 128)      — after embedding + positional
encoder_out:    (32, 64, 128)      — after transformer encoder
pooled:         (32, 128)          — after CLS/mean pooling
logits:         (32, 3)            — final output
```

See `text_classifier.py` for the complete implementation.

---

## 5. Dataset & DataLoader

### Custom TextDataset

Our dataset stores tokenized sequences and their labels:

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = [tokenizer.encode(t, max_length) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

### Custom collate_fn

Texts have different lengths. Our `collate_fn` pads each batch to the length of the longest sequence in that batch (dynamic padding):

```python
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    # Pad to max length in this batch
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    # Create attention mask: 1 for real tokens, 0 for padding
    attention_mask = (input_ids != 0).float()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

Dynamic padding is more efficient than padding all sequences to the global `max_length` — shorter batches waste less compute.

### Train/Val/Test Split

Standard practice: 80% train, 10% validation, 10% test.

```python
from torch.utils.data import random_split

dataset = TextDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
```

**Why three splits?**
- **Train**: model learns from this data
- **Validation**: tune hyperparameters, decide when to stop training
- **Test**: final evaluation — never used for any decisions during training

---

## 6. Training Loop

Our training loop includes every best practice for a production-quality training run.

### Optimizer: AdamW

AdamW decouples weight decay from the gradient update. This is the standard optimizer for transformer training:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)
```

### Learning Rate Schedule: OneCycleLR

OneCycleLR implements warmup + cosine decay in one scheduler:

```
LR
 ↑        ╱╲
 │      ╱    ╲
 │    ╱        ╲
 │  ╱            ╲
 │╱                ╲________
 └─────────────────────────→ step
   warmup    decay    final
```

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
)
```

### Loss Function: CrossEntropyLoss

For multi-class classification (positive/negative/neutral), CrossEntropyLoss combines LogSoftmax + NLLLoss:

```python
criterion = nn.CrossEntropyLoss()
# Input: logits (batch, num_classes) — NOT probabilities
# Target: class indices (batch,) — integers 0, 1, 2
loss = criterion(logits, labels)
```

### Mixed Precision (BF16)

BFloat16 reduces memory and speeds up training on modern hardware without the numerical instability of FP16:

```python
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)
```

We use `device_type="cpu"` since this module runs without GPU. On GPU, change to `"cuda"`.

### Gradient Clipping

Prevents exploding gradients that can destabilize training:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This clips the global gradient norm to 1.0. If the total gradient norm exceeds 1.0, all gradients are scaled down proportionally.

### Validation After Each Epoch

After each training epoch, evaluate on the validation set:

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        # Accumulate metrics...
```

### Early Stopping

Stop training when validation loss stops improving:

```python
patience = 3
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(max_epochs):
    # ... train and validate ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping!")
            break
```

### Best Model Checkpointing

Save the model whenever validation loss improves:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, "best_model.pt")
```

See `train_and_evaluate.py` for the complete training loop.

---

## 7. Evaluation Metrics

Accuracy alone is misleading for imbalanced datasets. If 90% of reviews are positive, a model that always predicts "positive" gets 90% accuracy while being useless.

### Precision, Recall, F1

**Precision**: Of all predictions for class C, how many were correct?

```
Precision(C) = True Positives / (True Positives + False Positives)
```

**Recall**: Of all actual instances of class C, how many did we find?

```
Recall(C) = True Positives / (True Positives + False Negatives)
```

**F1 Score**: Harmonic mean of precision and recall:

```
F1(C) = 2 × (Precision × Recall) / (Precision + Recall)
```

**Macro F1**: Average F1 across all classes (treats all classes equally):

```
Macro-F1 = mean(F1(positive), F1(negative), F1(neutral))
```

### Per-Class Metrics Example

```
              Precision  Recall  F1-Score  Support
  positive      0.91      0.93    0.92      120
  negative      0.88      0.85    0.86       95
  neutral       0.79      0.82    0.80       85

  macro avg     0.86      0.87    0.86      300
```

### Confusion Matrix

A confusion matrix shows where the model confuses classes:

```
                 Predicted
              pos   neg   neu
Actual pos  [ 112     3     5 ]
       neg  [   5    81     9 ]
       neu  [   6     9    70 ]
```

Reading: Row = actual class, Column = predicted class. Diagonal = correct predictions. Off-diagonal = errors.

Key insights from this matrix:
- The model sometimes confuses neutral for negative (9 cases)
- Positive is the easiest class to predict
- Neutral has the lowest recall (most often misclassified)

### Implementation

We compute these metrics without scikit-learn — just PyTorch:

```python
def compute_metrics(all_preds, all_labels, num_classes):
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1

    per_class = {}
    for c in range(num_classes):
        tp = confusion[c][c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

    return confusion, per_class
```

---

## 8. Inference Pipeline

### Single Text Inference

End-to-end: raw text → prediction with confidence score:

```python
def predict(text, model, tokenizer, label_names, max_length=128):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(text, max_length)])
    attention_mask = (input_ids != 0).float()

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=-1)
    confidence, predicted = probs.max(dim=-1)

    return {
        "text": text,
        "label": label_names[predicted.item()],
        "confidence": confidence.item(),
        "all_probs": {name: probs[0][i].item() for i, name in enumerate(label_names)},
    }
```

### Batch Inference

For multiple texts, batch them together for efficiency:

```python
def predict_batch(texts, model, tokenizer, label_names, max_length=128):
    model.eval()
    encoded = [tokenizer.encode(t, max_length) for t in texts]
    input_ids = nn.utils.rnn.pad_sequence(
        [torch.tensor(e) for e in encoded], batch_first=True, padding_value=0,
    )
    attention_mask = (input_ids != 0).float()

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    return [
        {"text": t, "label": label_names[p.item()], "confidence": c.item()}
        for t, p, c in zip(texts, predictions, confidences)
    ]
```

### Confidence Scores

Softmax converts logits to probabilities:

```
logits:  [2.1, -0.5, 0.3]
softmax: [0.72, 0.05, 0.23]  (sums to 1.0)
```

Use confidence scores to set a threshold:
```python
if result["confidence"] < 0.5:
    result["label"] = "uncertain"  # don't trust low-confidence predictions
```

---

## 9. torch.compile for Serving

Once trained, compile the model for faster inference:

```python
model.eval()
compiled_model = torch.compile(model, mode="reduce-overhead")

# Warmup (first call triggers compilation)
dummy = torch.randint(0, vocab_size, (1, 64))
mask = torch.ones(1, 64)
_ = compiled_model(dummy, mask)

# Now inference is faster
with torch.no_grad():
    logits = compiled_model(input_ids, attention_mask)
```

### Benchmarking Compiled vs Eager

```python
import time

def benchmark(model, input_ids, mask, n_runs=100):
    # Warmup
    for _ in range(10):
        model(input_ids, mask)

    start = time.perf_counter()
    for _ in range(n_runs):
        model(input_ids, mask)
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # ms per inference

eager_ms = benchmark(model, input_ids, mask)
compiled_ms = benchmark(compiled_model, input_ids, mask)
print(f"Eager:    {eager_ms:.2f} ms")
print(f"Compiled: {compiled_ms:.2f} ms")
print(f"Speedup:  {eager_ms / compiled_ms:.2f}x")
```

Typical results on CPU for a small model:
```
Eager:    3.45 ms
Compiled: 2.10 ms
Speedup:  1.64x
```

On GPU with `mode="reduce-overhead"`, speedups can reach 2-4x for small models due to CUDA graph capture.

### Compile Modes for Serving

| Mode | Compile Time | Inference Speed | Use Case |
|------|-------------|----------------|----------|
| `default` | Fast | Good | General purpose |
| `reduce-overhead` | Medium | Best for small models | Low-latency serving |
| `max-autotune` | Slow | Best for large models | Throughput-optimized |

---

## 10. Improvements

### Pretrained Embeddings

Instead of learning embeddings from scratch, initialize with pretrained vectors (GloVe, FastText):

```python
pretrained = load_glove("glove.6B.100d.txt")  # word → 100d vector
for word, idx in vocab.items():
    if word in pretrained:
        embedding.weight.data[idx] = torch.tensor(pretrained[word])
```

This gives the model a head start — it already knows that "good" and "great" are similar before seeing any training data.

### Data Augmentation

**Synonym replacement**: Replace words with synonyms to create new training examples:
```
"This movie was great" → "This film was excellent"
```

**Random deletion**: Randomly remove words (the model should still classify correctly):
```
"This movie was absolutely great" → "This movie great"
```

**Back-translation concept**: Translate to another language and back to get paraphrases:
```
"Great movie" → (French) "Super film" → (English) "Awesome film"
```

### Attention Visualization

Extract attention weights to see what the model focuses on:

```python
# Register hooks to capture attention weights
attention_weights = []
def hook_fn(module, input, output):
    if hasattr(module, 'attn_weights'):
        attention_weights.append(module.attn_weights)

for layer in model.encoder.layers:
    layer.self_attn.register_forward_hook(hook_fn)
```

High attention on "terrible" for a negative prediction confirms the model learned meaningful patterns.

### Model Distillation

Train a smaller, faster model (student) to mimic the larger model (teacher):

```python
teacher_logits = teacher_model(input_ids, mask)
student_logits = student_model(input_ids, mask)

# Soft label loss (KL divergence between teacher and student distributions)
T = 4.0  # temperature
soft_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction="batchmean",
) * (T * T)

# Hard label loss (standard cross-entropy)
hard_loss = F.cross_entropy(student_logits, labels)

# Combined
loss = 0.7 * soft_loss + 0.3 * hard_loss
```

The student learns from both the teacher's soft probability distributions and the hard labels. The temperature `T` softens the teacher's outputs, exposing more information about inter-class relationships ("this review is 60% positive, 30% neutral, 10% negative" is more informative than just "positive").

---

## 11. Upstream Updates (July 7-15, 2026)

Recent upstream changes relevant to the text classifier workflow covered in this module:

### Inductor Padding Fusion for Ragged Sequences (#189112)

The Inductor backend improved its handling of padded tensor operations common in NLP pipelines. When `torch.compile` encounters masked reductions over padded sequences (exactly the pattern used in mean pooling over variable-length texts), the fused kernel now avoids reading padding positions entirely rather than multiplying by zero. For short-sequence batches with heavy padding, this reduces unnecessary memory bandwidth by up to 30%. This directly benefits the compiled inference pipeline in Section 9.

### nn.TransformerEncoderLayer Pre-Norm Default (#189445)

`nn.TransformerEncoderLayer` now defaults to `norm_first=True` (pre-norm) instead of `norm_first=False` (post-norm). Pre-norm is the standard in modern transformers and improves training stability. Our model already uses `norm_first=True` explicitly, so no code changes are needed, but new code no longer needs to specify it. The post-norm default dated back to the original "Attention Is All You Need" paper but caused training instability without careful learning rate warmup.

### SDPA Nested Tensor Padding Mask (#190023)

`F.scaled_dot_product_attention` can now accept a `NestedTensor` as input and automatically handles the padding mask. Instead of constructing an explicit `(batch, 1, 1, seq_len)` mask and passing it as `attn_mask`, you can pack variable-length sequences into a `NestedTensor` and SDPA derives the mask internally. This eliminates the mask broadcasting overhead and enables Flash Attention on padded NLP batches where it previously fell back to the math kernel.

### CrossEntropyLoss label_smoothing BF16 Fix (#190187)

Fixed a numerical issue in `nn.CrossEntropyLoss` with `label_smoothing > 0` when running in BF16 autocast. The smoothing computation accumulated small probabilities in BF16, which underflowed for large vocabularies. The fix promotes the smoothing arithmetic to FP32 internally. This matters for our training loop if users enable label smoothing as an improvement (a common regularization technique for classification).

### OneCycleLR step() Warning Suppression (#190542)

`OneCycleLR` no longer emits a deprecation warning when `step()` is called per-batch (its intended usage pattern). Previously, calling `scheduler.step()` inside the batch loop printed a warning suggesting epoch-level stepping, which was incorrect for OneCycleLR. Our training loop calls `scheduler.step()` per-batch, which is the correct pattern, and this warning no longer appears.

### torch.compile Dynamic Sequence Length Cache (#191003)

`torch.compile` improved its recompilation behavior for models with dynamic sequence lengths. Previously, each new sequence length triggered a full recompilation. With this change, the compiler generates specialized code for a set of "buckets" (powers of 2, common lengths) and falls back to a generic dynamic-shape kernel for other lengths. For our text classifier with variable-length inputs, this means the compiled model handles different batch configurations without excessive recompilation after the initial warmup.

---

## Key Takeaways

1. **Tokenization is the foundation** — the choice of character-level, word-level, or subword tokenization fundamentally shapes model capacity and sequence length
2. **Embeddings are learned** — `nn.Embedding` starts random and learns semantic relationships during training; `padding_idx` prevents the padding token from contributing
3. **Positional encoding adds order** — sinusoidal or learned position embeddings tell the transformer where each token sits in the sequence
4. **Pre-norm transformers are stabler** — applying LayerNorm before (not after) attention and FFN sublayers improves gradient flow
5. **Dynamic padding saves compute** — pad to the longest sequence in each batch, not the global maximum, using a custom `collate_fn`
6. **OneCycleLR handles warmup + decay** — one scheduler replaces manual warmup + cosine decay configuration
7. **Metrics beyond accuracy matter** — precision, recall, and F1 per class reveal where the model struggles; the confusion matrix shows which classes are confused
8. **Confidence scores enable thresholding** — softmax probabilities let you reject uncertain predictions at inference time
9. **torch.compile speeds up serving** — compile the trained model for 1.5-4x faster inference with no accuracy change

---

### Further Resources

- [Module 04 — Neural Networks](../04_neural_networks/) — `nn.Module`, layers, losses
- [Module 05 — Optimizers](../05_optimizers/) — AdamW, learning rate schedulers
- [Module 06 — Data Loading](../06_data_loading/) — Dataset, DataLoader, custom collate
- [Module 07 — Training Pipelines](../07_training/) — full training loops, mixed precision
- [Module 09 — Attention Mechanisms](../09_attention/) — SDPA, multi-head attention, transformers
- [Module 08 — torch.compile](../08_torch_compile/) — compilation for inference
- [PyTorch Text Classification Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

---

<div align="center">

[← Previous Module (Compiled Autograd & AOTAutograd)](../38_compiled_autograd/) | [🏠 Home](../README.md) | Next Module → (none)

**Notebook**: [`39_text_classifier.ipynb`](../notebooks/39_text_classifier.ipynb)

</div>
