"""
Transformer — Complete Encoder-Decoder Implementation
=====================================================

Implements the full Transformer from "Attention Is All You Need" (Vaswani et
al., 2017), including multi-head attention, positional encoding, encoder and
decoder stacks, and the final sequence-to-sequence model.

This is the original encoder-decoder architecture designed for tasks like
machine translation, where you have a source sequence and a target sequence.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the original Transformer paper.

    Adds position-dependent signals to the embedding so the model can
    distinguish token order. The sinusoidal form allows the model to
    extrapolate to longer sequences than seen during training.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention.

    Splits Q, K, V into `num_heads` parallel heads, computes attention
    independently per head, concatenates, and projects back.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape to (batch, heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask shape: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted combination of values
        context = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: two linear layers with GELU."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer: self-attention + feed-forward,
    each with residual connections and layer normalization (pre-norm).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-norm: normalize before the sublayer, add residual after
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + self.dropout1(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_out)
        return x


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer: masked self-attention + cross-attention
    + feed-forward, each with residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked self-attention (causal: decoder can't see future tokens)
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout1(attn_out)

        # Cross-attention (decoder attends to encoder output)
        normed = self.norm2(x)
        cross_out = self.cross_attn(normed, enc_output, enc_output, memory_mask)
        x = x + self.dropout2(cross_out)

        # Feed-forward
        ff_out = self.ff(self.norm3(x))
        x = x + self.dropout3(ff_out)
        return x


class TransformerEncoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence tasks.

    Components:
        - Source and target token embeddings (optionally shared)
        - Sinusoidal positional encoding
        - Encoder stack
        - Decoder stack
        - Output linear projection to vocabulary
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = TransformerEncoder(
            d_model, num_heads, d_ff, num_encoder_layers, dropout,
        )
        self.decoder = TransformerDecoder(
            d_model, num_heads, d_ff, num_decoder_layers, dropout,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(size):
        """Create a causal mask: upper-triangular positions are False (masked)."""
        mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
        return mask  # (size, size), True = attend, False = mask

    def encode(self, src, src_mask=None):
        """Encode source sequence to memory representation."""
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decode target sequence given encoder memory."""
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(tgt_emb, memory, tgt_mask, memory_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src: source token IDs, (batch, src_len)
            tgt: target token IDs, (batch, tgt_len)
            src_mask: optional source padding mask
            tgt_mask: causal mask for target, (tgt_len, tgt_len)
            memory_mask: optional mask for cross-attention
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        memory = self.encode(src, src_mask)
        dec_output = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.output_proj(dec_output)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    # Hyperparameters matching "Transformer base" from the paper
    src_vocab = 1000
    tgt_vocab = 1200
    d_model = 256       # smaller for demo (paper: 512)
    num_heads = 8
    d_ff = 512          # paper: 2048
    num_layers = 4      # paper: 6
    batch_size = 2
    src_len = 20
    tgt_len = 15

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
    )
    model.eval()

    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    tgt_mask = Transformer.generate_square_subsequent_mask(tgt_len)
    # Expand mask for multi-head: (1, 1, tgt_len, tgt_len)
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(src, tgt, tgt_mask=tgt_mask)

    print(f"Transformer (base-like config)")
    print(f"  Parameters:   {count_params(model):,}")
    print(f"  Source input:  {list(src.shape)}")
    print(f"  Target input:  {list(tgt.shape)}")
    print(f"  Output logits: {list(logits.shape)}")
    print(f"  Expected:      [{batch_size}, {tgt_len}, {tgt_vocab}]")

    # Test individual components
    print("\nComponent shapes:")
    memory = model.encode(src)
    print(f"  Encoder output: {list(memory.shape)}")
    dec_out = model.decode(tgt, memory, tgt_mask=tgt_mask)
    print(f"  Decoder output: {list(dec_out.shape)}")

    # Test the causal mask
    mask = Transformer.generate_square_subsequent_mask(5)
    print(f"\nCausal mask (5x5):\n{mask.int()}")

    print("\nTransformer verified successfully!")
