"""
Module 39: Transformer Text Classifier — Built from Scratch

Runnable on CPU:
    python text_classifier.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Produces a (1, max_len, d_model) buffer that is added to the token
    embeddings so the transformer can distinguish token positions.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Text Classifier
# ---------------------------------------------------------------------------
class TransformerTextClassifier(nn.Module):
    """Transformer-encoder classifier.

    Architecture:
        Embedding(vocab, d_model)
        + PositionalEncoding
        + N x TransformerEncoderLayer (pre-norm, SDPA)
        -> Pooling (CLS or mean)
        -> LayerNorm -> Linear(d_model, num_classes)

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    d_model : int
        Embedding / hidden dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of TransformerEncoderLayer blocks.
    dim_feedforward : int
        Inner dimension of the position-wise FFN.
    num_classes : int
        Number of output classes.
    max_len : int
        Maximum sequence length (for positional encoding).
    dropout : float
        Dropout probability.
    padding_idx : int
        Token ID that represents padding (embedding fixed to zeros).
    pool : str
        Pooling strategy: ``"cls"`` (use [CLS] token) or ``"mean"``
        (average non-padding positions).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 3,
        max_len: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pool: str = "cls",
    ):
        super().__init__()
        assert pool in ("cls", "mean"), f"pool must be 'cls' or 'mean', got '{pool}'"

        self.pool = pool
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len)   LongTensor of token IDs.
        attention_mask : (batch, seq_len)  FloatTensor, 1.0 for real tokens,
                         0.0 for padding.  If *None* a mask is derived from
                         ``padding_idx``.

        Returns
        -------
        logits : (batch, num_classes)
        """
        # -- derive mask if not supplied --
        if attention_mask is None:
            attention_mask = (input_ids != self.padding_idx).float()

        # TransformerEncoder expects True = *ignore*, so invert
        src_key_padding_mask = attention_mask == 0  # (B, S)

        # -- embedding + positional --
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # (B, S, D)
        x = self.pos_encoder(x)                                  # (B, S, D)

        # -- transformer encoder --
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, S, D)

        # -- pooling --
        if self.pool == "cls":
            pooled = x[:, 0]  # (B, D) — first token is [CLS]
        else:
            mask_expanded = attention_mask.unsqueeze(-1)          # (B, S, 1)
            pooled = (x * mask_expanded).sum(dim=1)              # (B, D)
            pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # -- classify --
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------
def model_summary(model: nn.Module) -> str:
    lines = [f"{'Layer':<45s} {'Shape':>20s} {'Params':>12s}"]
    lines.append("-" * 80)
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            total += n
            lines.append(f"{name:<45s} {str(list(p.shape)):>20s} {n:>12,d}")
    lines.append("-" * 80)
    lines.append(f"{'Total trainable parameters':<45s} {'':>20s} {total:>12,d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(42)

    vocab_size = 500
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    num_classes = 3
    batch_size = 8
    seq_len = 32

    print("=" * 70)
    print("TRANSFORMER TEXT CLASSIFIER — DEMO")
    print("=" * 70)

    # --- CLS pooling model ---------------------------------------------------
    model_cls = TransformerTextClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        pool="cls",
    )
    print(f"\nModel (CLS pooling): {model_cls.count_parameters():,} parameters")
    print()
    print(model_summary(model_cls))

    # --- forward pass with dummy data ----------------------------------------
    print("\n" + "=" * 70)
    print("FORWARD PASS (dummy data)")
    print("=" * 70)

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    input_ids[:, 0] = 2  # [CLS] token at position 0
    # simulate variable-length sequences with padding at the end
    for i in range(batch_size):
        pad_start = seq_len - i * 3
        if pad_start > 2:
            input_ids[i, pad_start:] = 0

    attention_mask = (input_ids != 0).float()

    print(f"input_ids shape:      {list(input_ids.shape)}")
    print(f"attention_mask shape: {list(attention_mask.shape)}")

    with torch.no_grad():
        logits = model_cls(input_ids, attention_mask)

    print(f"logits shape:         {list(logits.shape)}")
    print(f"logits[0]:            {logits[0].tolist()}")
    probs = torch.softmax(logits, dim=-1)
    print(f"probs[0]:             {[f'{p:.4f}' for p in probs[0].tolist()]}")
    print(f"prediction[0]:        class {probs[0].argmax().item()}")

    # --- Mean pooling model ---------------------------------------------------
    print("\n" + "=" * 70)
    print("MEAN POOLING VARIANT")
    print("=" * 70)

    model_mean = TransformerTextClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
        pool="mean",
    )
    print(f"Model (mean pooling): {model_mean.count_parameters():,} parameters")

    with torch.no_grad():
        logits_mean = model_mean(input_ids, attention_mask)
    print(f"logits shape:         {list(logits_mean.shape)}")
    print(f"logits[0]:            {logits_mean[0].tolist()}")

    # --- Verify mask effect ---------------------------------------------------
    print("\n" + "=" * 70)
    print("PADDING MASK EFFECT")
    print("=" * 70)

    short_ids = torch.tensor([[2, 10, 20, 0, 0]])   # 3 real tokens + 2 pad
    mask_a = torch.tensor([[1, 1, 1, 0, 0]]).float()
    mask_b = torch.tensor([[1, 1, 1, 1, 1]]).float()  # pretend no padding

    with torch.no_grad():
        out_a = model_cls(short_ids, mask_a)
        out_b = model_cls(short_ids, mask_b)

    same = torch.allclose(out_a, out_b, atol=1e-5)
    print(f"Outputs with vs without mask differ: {not same}")
    print(f"  With mask:    {out_a[0].tolist()}")
    print(f"  Without mask: {out_b[0].tolist()}")

    # --- Shape walkthrough ----------------------------------------------------
    print("\n" + "=" * 70)
    print("SHAPE WALKTHROUGH")
    print("=" * 70)

    x = input_ids[:1]
    m = attention_mask[:1]
    print(f"1. input_ids:                {list(x.shape)}")

    emb = model_cls.embedding(x) * math.sqrt(d_model)
    print(f"2. After embedding:          {list(emb.shape)}")

    emb_pos = model_cls.pos_encoder(emb)
    print(f"3. After positional enc:     {list(emb_pos.shape)}")

    enc_out = model_cls.encoder(emb_pos, src_key_padding_mask=(m == 0))
    print(f"4. After transformer enc:    {list(enc_out.shape)}")

    pooled = enc_out[:, 0]
    print(f"5. After CLS pooling:        {list(pooled.shape)}")

    logit = model_cls.classifier(pooled)
    print(f"6. After classifier head:    {list(logit.shape)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
