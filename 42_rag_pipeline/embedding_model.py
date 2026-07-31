"""
Module 42: Embedding Model for RAG Pipeline

Implements:
- Simple transformer encoder for text embeddings
- Mean pooling over transformer outputs (with attention masking)
- L2 normalization for unit-vector embeddings
- Single text and batch encoding
- Pairwise and query-to-corpus similarity computation
- Demo with synthetic texts showing semantic similarity

Usage:
    python embedding_model.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Simple Tokenizer (character-level for demonstration)
# ============================================================================


class SimpleTokenizer:
    """Character-level tokenizer with padding and special tokens.

    Production RAG systems use BPE/WordPiece tokenizers (e.g., from
    HuggingFace). This simple tokenizer demonstrates the interface.
    """

    def __init__(self, vocab_size=256, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1

    def encode(self, text):
        token_ids = [min(ord(c), self.vocab_size - 1) for c in text]
        return token_ids[: self.max_length]

    def encode_batch(self, texts):
        encoded = [self.encode(t) for t in texts]
        max_len = min(max(len(e) for e in encoded), self.max_length)
        padded = []
        masks = []
        for enc in encoded:
            length = min(len(enc), max_len)
            pad_len = max_len - length
            padded.append(enc[:length] + [self.pad_token_id] * pad_len)
            masks.append([1] * length + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


# ============================================================================
# Embedding Model (Transformer Encoder)
# ============================================================================


class EmbeddingModel(nn.Module):
    """Transformer encoder that produces fixed-size text embeddings.

    Architecture:
        Token embedding + Positional embedding
        -> N transformer encoder layers (self-attention + FFN)
        -> Mean pooling (masked)
        -> L2 normalization
        -> Unit-vector embedding
    """

    def __init__(self, vocab_size=256, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass: tokens -> embedding vector.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) 1=real token, 0=padding

        Returns:
            embeddings: (batch_size, d_model) L2-normalized embeddings
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        hidden_states = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        hidden_states = self.layer_norm(hidden_states)
        embeddings = self.mean_pool(hidden_states, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    @staticmethod
    def mean_pool(hidden_states, attention_mask):
        """Mean pooling: average token embeddings, ignoring padding.

        Args:
            hidden_states: (B, T, d_model)
            attention_mask: (B, T) or None

        Returns:
            pooled: (B, d_model)
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


# ============================================================================
# Encoding Interface
# ============================================================================


class TextEncoder:
    """High-level interface for encoding texts into embeddings.

    Wraps the tokenizer and embedding model into a simple encode() API.
    """

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        """Encode a list of texts into embedding vectors.

        Args:
            texts: list of strings
            batch_size: number of texts to encode at once

        Returns:
            embeddings: (len(texts), d_model) tensor
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            input_ids, attention_mask = self.tokenizer.encode_batch(batch_texts)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            embeddings = self.model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_single(self, text):
        """Encode a single text into an embedding vector.

        Returns:
            embedding: (d_model,) tensor
        """
        return self.encode([text])[0]


# ============================================================================
# Similarity Computation
# ============================================================================


def cosine_similarity_matrix(embeddings_a, embeddings_b):
    """Compute pairwise cosine similarity between two sets of embeddings.

    Since embeddings are L2-normalized, cosine similarity = dot product.

    Args:
        embeddings_a: (M, d_model)
        embeddings_b: (N, d_model)

    Returns:
        similarity: (M, N)
    """
    return embeddings_a @ embeddings_b.T


def find_most_similar(query_embedding, corpus_embeddings, k=5):
    """Find the k most similar embeddings from a corpus.

    Args:
        query_embedding: (d_model,) single query
        corpus_embeddings: (N, d_model) corpus of documents
        k: number of results to return

    Returns:
        indices: (k,) indices of most similar documents
        scores: (k,) similarity scores
    """
    similarities = corpus_embeddings @ query_embedding
    k = min(k, len(similarities))
    top_k = torch.topk(similarities, k=k)
    return top_k.indices, top_k.values


# ============================================================================
# Demo
# ============================================================================


def main():
    print("=" * 70)
    print("Module 42: Embedding Model for RAG")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    tokenizer = SimpleTokenizer(vocab_size=256, max_length=64)
    model = EmbeddingModel(
        vocab_size=256, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, max_seq_len=64,
    )
    encoder = TextEncoder(model, tokenizer, device=device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Embedding dimension: {model.d_model}")

    # --- Encode sample texts ---
    print("\n" + "-" * 70)
    print("1. Encoding Sample Texts")
    print("-" * 70)

    texts = [
        "PyTorch autograd tracks operations on tensors",
        "The autograd engine computes gradients automatically",
        "CUDA enables GPU-accelerated tensor operations",
        "Neural networks learn by backpropagation",
        "Data loaders provide batched iteration over datasets",
        "torch.compile optimizes model execution with graph capture",
        "Distributed training scales models across multiple GPUs",
        "Attention mechanisms weight the importance of input elements",
    ]

    embeddings = encoder.encode(texts)
    print(f"Encoded {len(texts)} texts -> embeddings shape: {embeddings.shape}")
    print(f"Embedding norms (should be ~1.0): {embeddings.norm(dim=1).tolist()[:4]}")

    # --- Similarity matrix ---
    print("\n" + "-" * 70)
    print("2. Pairwise Similarity Matrix")
    print("-" * 70)

    sim_matrix = cosine_similarity_matrix(embeddings, embeddings)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print("\nSimilarity scores (first 4 texts):")
    short_labels = ["autograd", "gradients", "CUDA/GPU", "backprop"]
    header = "            " + "  ".join(f"{l:>10}" for l in short_labels)
    print(header)
    for i in range(4):
        row = f"{short_labels[i]:>10}  " + "  ".join(f"{sim_matrix[i, j]:10.4f}" for j in range(4))
        print(row)

    # --- Query search ---
    print("\n" + "-" * 70)
    print("3. Query Search (find most similar)")
    print("-" * 70)

    queries = [
        "How do gradients flow in PyTorch?",
        "How to use multiple GPUs for training?",
        "What is torch.compile?",
    ]

    for query in queries:
        query_emb = encoder.encode_single(query)
        indices, scores = find_most_similar(query_emb, embeddings, k=3)
        print(f"\nQuery: '{query}'")
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            print(f"  #{rank + 1} (sim={score:.4f}): '{texts[idx]}'")

    # --- Batch encoding benchmark ---
    print("\n" + "-" * 70)
    print("4. Batch Encoding Performance")
    print("-" * 70)

    large_corpus = texts * 50  # 400 texts
    import time
    start = time.perf_counter()
    batch_embeddings = encoder.encode(large_corpus, batch_size=64)
    elapsed = time.perf_counter() - start
    print(f"Encoded {len(large_corpus)} texts in {elapsed:.3f}s")
    print(f"Throughput: {len(large_corpus) / elapsed:.0f} texts/sec")
    print(f"Output shape: {batch_embeddings.shape}")

    # --- Verify L2 normalization ---
    print("\n" + "-" * 70)
    print("5. Embedding Properties")
    print("-" * 70)

    norms = batch_embeddings.norm(dim=1)
    print(f"L2 norms — min: {norms.min():.6f}, max: {norms.max():.6f}, mean: {norms.mean():.6f}")
    dot_product = (batch_embeddings[0] @ batch_embeddings[1]).item()
    cos_sim = F.cosine_similarity(batch_embeddings[0:1], batch_embeddings[1:2]).item()
    print(f"Dot product vs cosine sim (should match): {dot_product:.6f} vs {cos_sim:.6f}")

    print("\n" + "=" * 70)
    print("Embedding model demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
