"""
Module 04: Common Neural Network Layers
========================================
Comprehensive demonstration of all major layer types in PyTorch:
linear, convolutional, pooling, normalization, activation, dropout,
recurrent, transformer, and embedding layers.

Run: python common_layers.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("PART 1: Linear Layers")
print("=" * 70)

# --- nn.Linear ---
print("\n--- nn.Linear: y = xW^T + b ---")
linear = nn.Linear(in_features=20, out_features=10, bias=True)
print(f"Weight shape: {linear.weight.shape}")  # (out_features, in_features)
print(f"Bias shape:   {linear.bias.shape}")  # (out_features,)

x = torch.randn(32, 20)  # (batch_size, in_features)
y = linear(x)
print(f"Input:  {x.shape} -> Output: {y.shape}")

# Verify the math manually
y_manual = x @ linear.weight.t() + linear.bias
print(f"Manual computation matches: {torch.allclose(y, y_manual, atol=1e-6)}")

# --- nn.Bilinear ---
print("\n--- nn.Bilinear: y = x1^T A x2 + b ---")
bilinear = nn.Bilinear(in1_features=20, in2_features=30, out_features=10)
x1 = torch.randn(32, 20)
x2 = torch.randn(32, 30)
y = bilinear(x1, x2)
print(f"Input1: {x1.shape}, Input2: {x2.shape} -> Output: {y.shape}")
print(f"Weight shape: {bilinear.weight.shape}")  # (out, in1, in2)

# --- nn.LazyLinear ---
print("\n--- nn.LazyLinear: defers in_features ---")
lazy = nn.LazyLinear(out_features=64)
print(f"Before first forward: weight = {lazy.weight.__class__.__name__}")
x = torch.randn(16, 128)
y = lazy(x)
print(f"After first forward: weight shape = {lazy.weight.shape}")
print(f"Input: {x.shape} -> Output: {y.shape}")

print("\n" + "=" * 70)
print("PART 2: Convolutional Layers")
print("=" * 70)

# --- Conv1d ---
print("\n--- nn.Conv1d (for sequences/time series/audio) ---")
conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
x = torch.randn(8, 1, 100)  # (batch, channels, length)
y = conv1d(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
print(f"Kernel weight shape: {conv1d.weight.shape}")  # (out_ch, in_ch, kernel_size)

# --- Conv2d ---
print("\n--- nn.Conv2d (for images) ---")
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 32, 32)  # (batch, channels, height, width)
y = conv2d(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
print(f"Kernel weight shape: {conv2d.weight.shape}")  # (out_ch, in_ch, kH, kW)

# Demonstrate stride and output size calculation
print("\n--- Output size formula: floor((input + 2*pad - dilation*(kernel-1) - 1) / stride + 1) ---")
for stride in [1, 2]:
    for padding in [0, 1]:
        conv = nn.Conv2d(3, 64, kernel_size=3, stride=stride, padding=padding)
        out = conv(torch.randn(1, 3, 32, 32))
        h_out = (32 + 2 * padding - 1 * (3 - 1) - 1) // stride + 1
        print(f"  stride={stride}, padding={padding} -> output: {out.shape[2]}x{out.shape[3]} "
              f"(formula predicts: {h_out}x{h_out})")

# --- Dilated Convolution ---
print("\n--- Dilated (Atrous) Convolution ---")
dilated_conv = nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2)
x = torch.randn(1, 3, 32, 32)
y = dilated_conv(x)
print(f"Dilation=2, Input: {x.shape} -> Output: {y.shape}")
print("Dilated convolutions expand the receptive field without increasing parameters")

# --- Grouped Convolution ---
print("\n--- Grouped Convolution ---")
grouped_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4)
x = torch.randn(1, 64, 32, 32)
y = grouped_conv(x)
print(f"Groups=4, Input: {x.shape} -> Output: {y.shape}")
print(f"Weight shape: {grouped_conv.weight.shape}")  # (64, 16, 3, 3) — 64/4=16 in_ch per group
regular_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=1)
print(f"Grouped conv params: {sum(p.numel() for p in grouped_conv.parameters()):,}")
print(f"Regular conv params: {sum(p.numel() for p in regular_conv.parameters()):,}")

# --- Depthwise Separable Convolution ---
print("\n--- Depthwise Separable Convolution ---")


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


dw_sep = DepthwiseSeparableConv(64, 128)
regular = nn.Conv2d(64, 128, kernel_size=3, padding=1)
x = torch.randn(1, 64, 32, 32)
print(f"Depthwise separable params: {sum(p.numel() for p in dw_sep.parameters()):,}")
print(f"Regular conv params:        {sum(p.numel() for p in regular.parameters()):,}")
print(f"Ratio: {sum(p.numel() for p in dw_sep.parameters()) / sum(p.numel() for p in regular.parameters()):.2f}x")

# --- ConvTranspose2d (Upsampling) ---
print("\n--- nn.ConvTranspose2d (Transposed/Upsampling Convolution) ---")
up_conv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
x = torch.randn(1, 64, 16, 16)
y = up_conv(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
print("Doubles spatial dimensions (16x16 -> 32x32)")

# --- Conv3d ---
print("\n--- nn.Conv3d (for video/volumetric data) ---")
conv3d = nn.Conv3d(3, 64, kernel_size=3, padding=1)
x = torch.randn(1, 3, 16, 32, 32)  # (batch, channels, depth, height, width)
y = conv3d(x)
print(f"Input: {x.shape} -> Output: {y.shape}")

print("\n" + "=" * 70)
print("PART 3: Pooling Layers")
print("=" * 70)

x = torch.randn(1, 64, 32, 32)

# --- MaxPool2d ---
print("\n--- nn.MaxPool2d ---")
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
y = maxpool(x)
print(f"Input: {x.shape} -> Output: {y.shape}")

# MaxPool with return_indices (useful for MaxUnpool)
maxpool_idx = nn.MaxPool2d(2, 2, return_indices=True)
y, indices = maxpool_idx(x)
print(f"Indices shape: {indices.shape} (for MaxUnpool)")

# --- AvgPool2d ---
print("\n--- nn.AvgPool2d ---")
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
y = avgpool(x)
print(f"Input: {x.shape} -> Output: {y.shape}")

# --- AdaptiveAvgPool2d (Global Average Pooling) ---
print("\n--- nn.AdaptiveAvgPool2d (Global Average Pooling) ---")
gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
y = gap(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
print("This gives one value per channel — used before final classifier in modern CNNs")

# Flatten after GAP
y_flat = y.flatten(1)
print(f"After flatten: {y_flat.shape}")

# Can also specify non-1x1 output
adaptive = nn.AdaptiveAvgPool2d(output_size=(7, 7))
y = adaptive(x)
print(f"Adaptive to 7x7: Input: {x.shape} -> Output: {y.shape}")

# --- MaxPool1d for sequences ---
print("\n--- nn.MaxPool1d (for sequences) ---")
x_seq = torch.randn(8, 64, 100)  # (batch, channels, length)
pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
y = pool1d(x_seq)
print(f"Input: {x_seq.shape} -> Output: {y.shape}")

print("\n" + "=" * 70)
print("PART 4: Normalization Layers")
print("=" * 70)

# --- BatchNorm2d ---
print("\n--- nn.BatchNorm2d ---")
bn = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True)
x = torch.randn(32, 64, 8, 8)

# Training mode: uses batch statistics, updates running stats
bn.train()
y_train = bn(x)
print(f"BatchNorm2d — Input: {x.shape} -> Output: {y_train.shape}")
print(f"  weight (gamma): {bn.weight.shape}")
print(f"  bias (beta): {bn.bias.shape}")
print(f"  running_mean: {bn.running_mean.shape}")
print(f"  running_var: {bn.running_var.shape}")
print(f"  num_batches_tracked: {bn.num_batches_tracked}")

# Eval mode: uses running statistics
bn.eval()
y_eval = bn(x)
print(f"  Train and eval outputs differ: {not torch.allclose(y_train, y_eval, atol=1e-5)}")

# --- LayerNorm ---
print("\n--- nn.LayerNorm ---")
ln = nn.LayerNorm(normalized_shape=256)
x = torch.randn(32, 10, 256)  # (batch, seq_len, hidden_dim)
y = ln(x)
print(f"LayerNorm — Input: {x.shape} -> Output: {y.shape}")
# Verify normalization along last dim
print(f"  Output mean ≈ 0: {y[0, 0].mean().item():.6f}")
print(f"  Output std ≈ 1:  {y[0, 0].std().item():.4f}")

# --- GroupNorm ---
print("\n--- nn.GroupNorm ---")
gn = nn.GroupNorm(num_groups=8, num_channels=64)
x = torch.randn(4, 64, 16, 16)  # Works with small batches!
y = gn(x)
print(f"GroupNorm (8 groups, 64 channels) — Input: {x.shape} -> Output: {y.shape}")

# --- InstanceNorm ---
print("\n--- nn.InstanceNorm2d ---")
inst_norm = nn.InstanceNorm2d(64, affine=True)
x = torch.randn(4, 64, 16, 16)
y = inst_norm(x)
print(f"InstanceNorm2d — Input: {x.shape} -> Output: {y.shape}")

# --- RMSNorm ---
print("\n--- nn.RMSNorm ---")
rms = nn.RMSNorm(normalized_shape=256)
x = torch.randn(32, 10, 256)
y = rms(x)
print(f"RMSNorm — Input: {x.shape} -> Output: {y.shape}")

# Comparison: when to use each
print("\n--- Summary: When to use each normalization ---")
print("  BatchNorm:    CNNs with large batches (>=32)")
print("  LayerNorm:    Transformers, RNNs, small batches")
print("  GroupNorm:    CNNs with small batches")
print("  InstanceNorm: Style transfer, image generation")
print("  RMSNorm:      Modern LLMs (faster than LayerNorm)")

print("\n" + "=" * 70)
print("PART 5: Activation Functions")
print("=" * 70)

x = torch.linspace(-3, 3, 7)
print(f"\nInput: {x.tolist()}")

activations = {
    "ReLU": nn.ReLU(),
    "LeakyReLU(0.1)": nn.LeakyReLU(0.1),
    "PReLU": nn.PReLU(),
    "GELU": nn.GELU(),
    "SiLU/Swish": nn.SiLU(),
    "Mish": nn.Mish(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
}

for name, act in activations.items():
    y = act(x)
    print(f"  {name:15s}: {[f'{v:.3f}' for v in y.tolist()]}")

# Softmax (operates on a dimension)
print(f"\n  Softmax (dim=-1):")
logits = torch.tensor([2.0, 1.0, 0.1])
probs = nn.Softmax(dim=-1)(logits)
print(f"    Input:  {logits.tolist()}")
print(f"    Output: {[f'{v:.4f}' for v in probs.tolist()]} (sum={probs.sum():.4f})")

print("\n" + "=" * 70)
print("PART 6: Dropout Layers")
print("=" * 70)

# --- Dropout ---
print("\n--- nn.Dropout ---")
dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 10)

dropout.train()
y_train = dropout(x)
print(f"Training (p=0.5): {y_train}")
print(f"  Note: non-zero values are scaled by 1/(1-p) = 2.0")

dropout.eval()
y_eval = dropout(x)
print(f"Eval mode:        {y_eval}")
print(f"  No dropout applied in eval mode")

# --- Dropout2d ---
print("\n--- nn.Dropout2d (drops entire channels) ---")
dropout2d = nn.Dropout2d(p=0.5)
x = torch.ones(1, 4, 3, 3)  # 4 channels, 3x3 spatial
dropout2d.train()
y = dropout2d(x)
print(f"Input: 4 channels of ones")
print(f"Channel means after Dropout2d: {y.mean(dim=(2, 3)).squeeze()}")
print("Notice: entire channels are either all 0 or all scaled up")

# --- AlphaDropout ---
print("\n--- nn.AlphaDropout (for SELU networks) ---")
alpha_drop = nn.AlphaDropout(p=0.1)
selu = nn.SELU()
x = torch.randn(1000, 100)
x = selu(x)
x_dropped = alpha_drop(x)
print(f"After AlphaDropout — mean: {x_dropped.mean():.4f}, std: {x_dropped.std():.4f}")
print("AlphaDropout maintains self-normalizing property")

print("\n" + "=" * 70)
print("PART 7: Recurrent Layers")
print("=" * 70)

batch_size = 8
seq_len = 20
input_size = 64
hidden_size = 128

# --- nn.RNN ---
print("\n--- nn.RNN ---")
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2,
             batch_first=True, bidirectional=False, dropout=0.1)
x = torch.randn(batch_size, seq_len, input_size)
h0 = torch.zeros(2, batch_size, hidden_size)  # (num_layers, batch, hidden)
output, h_n = rnn(x, h0)
print(f"Input: {x.shape}")
print(f"Output (all timesteps): {output.shape}")
print(f"Final hidden state: {h_n.shape}")

# --- nn.LSTM ---
print("\n--- nn.LSTM ---")
lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
               batch_first=True, bidirectional=True, dropout=0.1)
x = torch.randn(batch_size, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
print(f"Bidirectional LSTM:")
print(f"  Input: {x.shape}")
print(f"  Output: {output.shape}")  # hidden_size * 2 for bidirectional
print(f"  h_n: {h_n.shape}")  # (num_layers * 2, batch, hidden_size)
print(f"  c_n: {c_n.shape}")

# Extracting final hidden state from bidirectional LSTM
forward_final = h_n[-2]  # Last layer, forward direction
backward_final = h_n[-1]  # Last layer, backward direction
combined = torch.cat([forward_final, backward_final], dim=-1)
print(f"  Combined final hidden: {combined.shape}")

# --- nn.GRU ---
print("\n--- nn.GRU ---")
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1,
             batch_first=True, bidirectional=False)
x = torch.randn(batch_size, seq_len, input_size)
output, h_n = gru(x)
print(f"GRU Output: {output.shape}, Final hidden: {h_n.shape}")

# Practical example: sequence classification with LSTM
print("\n--- Practical: Sequence Classification with LSTM ---")


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, _) = self.lstm(embedded)
        # Use last hidden states from both directions
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


classifier = LSTMClassifier(vocab_size=5000, embed_dim=128, hidden_dim=256, num_classes=5)
tokens = torch.randint(0, 5000, (16, 50))  # (batch, seq_len)
logits = classifier(tokens)
print(f"Token input: {tokens.shape} -> Logits: {logits.shape}")

print("\n" + "=" * 70)
print("PART 8: Transformer Layers")
print("=" * 70)

# --- MultiheadAttention ---
print("\n--- nn.MultiheadAttention ---")
embed_dim = 256
num_heads = 8
mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                            dropout=0.1, batch_first=True)

x = torch.randn(8, 20, embed_dim)  # (batch, seq_len, embed_dim)
# Self-attention: query = key = value = x
attn_output, attn_weights = mha(x, x, x)
print(f"Self-attention input: {x.shape}")
print(f"Attention output: {attn_output.shape}")
print(f"Attention weights: {attn_weights.shape}")  # (batch, tgt_len, src_len)

# With padding mask
key_padding_mask = torch.zeros(8, 20, dtype=torch.bool)
key_padding_mask[:, 15:] = True  # Mask positions 15-19
attn_output, _ = mha(x, x, x, key_padding_mask=key_padding_mask)
print(f"With padding mask: output shape unchanged: {attn_output.shape}")

# --- TransformerEncoderLayer ---
print("\n--- nn.TransformerEncoderLayer ---")
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256, nhead=8, dim_feedforward=1024,
    dropout=0.1, activation="gelu", batch_first=True,
    norm_first=True
)
x = torch.randn(8, 20, 256)
y = encoder_layer(x)
print(f"TransformerEncoderLayer: {x.shape} -> {y.shape}")

# --- TransformerEncoder (stack of layers) ---
print("\n--- nn.TransformerEncoder ---")
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
y = encoder(x)
print(f"6-layer TransformerEncoder: {x.shape} -> {y.shape}")
print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")

# Full Transformer-based classifier
print("\n--- Practical: Transformer Classifier ---")


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x, padding_mask=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # Use [CLS] token (first position) for classification
        x = x[:, 0]
        return self.classifier(x)


tf_classifier = TransformerClassifier(
    vocab_size=10000, d_model=256, nhead=8, num_layers=4, num_classes=5
)
tokens = torch.randint(0, 10000, (8, 50))
output = tf_classifier(tokens)
print(f"Transformer Classifier: tokens {tokens.shape} -> logits {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in tf_classifier.parameters()):,}")

print("\n" + "=" * 70)
print("PART 9: Embedding Layers")
print("=" * 70)

# --- nn.Embedding ---
print("\n--- nn.Embedding ---")
embed = nn.Embedding(num_embeddings=1000, embedding_dim=64, padding_idx=0)
print(f"Embedding weight shape: {embed.weight.shape}")

# Token IDs -> dense vectors
token_ids = torch.tensor([[5, 23, 456, 0, 0]])  # 0 = padding
embedded = embed(token_ids)
print(f"Token IDs: {token_ids.shape} -> Embeddings: {embedded.shape}")
print(f"Padding embedding (idx=0) is zero: {embedded[0, 3].sum().item() == 0}")

# --- nn.EmbeddingBag ---
print("\n--- nn.EmbeddingBag ---")
embed_bag = nn.EmbeddingBag(num_embeddings=1000, embedding_dim=64, mode="mean")
# Input: flat list of indices + offsets telling where each "bag" starts
indices = torch.tensor([1, 5, 23, 100, 200, 7, 8])
offsets = torch.tensor([0, 3, 5])  # Bag 0: indices[0:3], Bag 1: indices[3:5], Bag 2: indices[5:7]
output = embed_bag(indices, offsets)
print(f"3 bags from 7 indices -> Output: {output.shape}")
print("EmbeddingBag is more efficient than Embedding + mean for variable-length inputs")

print("\n" + "=" * 70)
print("PART 10: Building a Complete CNN")
print("=" * 70)


class SimpleCNN(nn.Module):
    """A complete CNN for image classification demonstrating multiple layer types."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


cnn = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)
output = cnn(x)
print(f"\nSimpleCNN:")
print(f"  Input: {x.shape}")
print(f"  Output: {output.shape}")
print(f"  Total parameters: {sum(p.numel() for p in cnn.parameters()):,}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
