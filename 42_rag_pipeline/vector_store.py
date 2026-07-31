"""
Module 42: Vector Store for RAG Pipeline

Implements:
- VectorStore class with add, search, and persistence
- Three chunking strategies: fixed-size, sentence-based, overlapping
- Cosine similarity search with top-K retrieval
- Batch document operations
- Save/load to disk (torch.save format)
- Document-to-chunk mapping for source attribution
- Demo with synthetic documents

Usage:
    python vector_store.py
"""

import os
import time

import torch
import torch.nn.functional as F

from embedding_model import EmbeddingModel, SimpleTokenizer, TextEncoder


# ============================================================================
# Document Chunking Strategies
# ============================================================================


def chunk_fixed_size(text, chunk_size=50, overlap=10):
    """Split text into fixed-size word chunks with overlap.

    Args:
        text: input text string
        chunk_size: number of words per chunk
        overlap: number of overlapping words between consecutive chunks

    Returns:
        list of chunk strings
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    stride = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += stride
    return chunks


def chunk_by_sentences(text, max_words=50):
    """Split at sentence boundaries, grouping sentences up to max_words.

    Args:
        text: input text string
        max_words: maximum words per chunk

    Returns:
        list of chunk strings
    """
    delimiters = [". ", "! ", "? "]
    sentences = []
    remaining = text
    for delim in delimiters:
        remaining = remaining.replace(delim, ". ")
    raw_sentences = remaining.split(". ")
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_word_count + sent_words > max_words and current_sentences:
            chunks.append(". ".join(current_sentences) + ".")
            current_sentences = []
            current_word_count = 0
        current_sentences.append(sent)
        current_word_count += sent_words

    if current_sentences:
        chunks.append(". ".join(current_sentences) + ".")
    return chunks


def chunk_overlapping(text, chunk_size=50, stride=25):
    """Overlapping window chunking with configurable stride.

    Args:
        text: input text string
        chunk_size: window size in words
        stride: step size in words (stride < chunk_size creates overlap)

    Returns:
        list of chunk strings
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += stride
    return chunks


CHUNKING_STRATEGIES = {
    "fixed": chunk_fixed_size,
    "sentence": chunk_by_sentences,
    "overlapping": chunk_overlapping,
}


# ============================================================================
# Vector Store
# ============================================================================


class VectorStore:
    """In-memory vector store with document chunking and similarity search.

    Stores documents as chunked text with their embeddings for fast
    cosine similarity retrieval.

    Attributes:
        encoder: TextEncoder for computing embeddings
        chunk_strategy: chunking function name ("fixed", "sentence", "overlapping")
        chunk_size: maximum words per chunk
        chunk_overlap: overlap between consecutive chunks
        documents: list of original document texts
        chunks: list of chunk texts
        embeddings: (num_chunks, d_model) tensor of chunk embeddings
        chunk_to_doc: list mapping chunk index -> document index
        chunk_metadata: list of dicts with chunk metadata
    """

    def __init__(self, encoder, chunk_strategy="fixed", chunk_size=50, chunk_overlap=10):
        self.encoder = encoder
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.chunk_to_doc = []
        self.chunk_metadata = []

    def _chunk_text(self, text):
        """Apply the configured chunking strategy to a text."""
        strategy_fn = CHUNKING_STRATEGIES[self.chunk_strategy]
        if self.chunk_strategy == "sentence":
            return strategy_fn(text, max_words=self.chunk_size)
        elif self.chunk_strategy == "overlapping":
            stride = max(1, self.chunk_size - self.chunk_overlap)
            return strategy_fn(text, chunk_size=self.chunk_size, stride=stride)
        else:
            return strategy_fn(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

    def add_documents(self, texts, batch_size=32):
        """Add documents to the store: chunk, embed, and index.

        Args:
            texts: list of document strings
            batch_size: encoding batch size
        """
        doc_start_idx = len(self.documents)
        new_chunks = []

        for i, text in enumerate(texts):
            doc_idx = doc_start_idx + i
            doc_chunks = self._chunk_text(text)
            for chunk_pos, chunk in enumerate(doc_chunks):
                new_chunks.append(chunk)
                self.chunk_to_doc.append(doc_idx)
                self.chunk_metadata.append({
                    "doc_idx": doc_idx,
                    "chunk_pos": chunk_pos,
                    "total_chunks": len(doc_chunks),
                })
            self.documents.append(text)

        new_embeddings = self.encoder.encode(new_chunks, batch_size=batch_size)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

        self.chunks.extend(new_chunks)

    def add_single(self, text):
        """Add a single document."""
        self.add_documents([text])

    def search(self, query, k=5):
        """Search for the k most similar chunks to a query.

        Args:
            query: query string
            k: number of results

        Returns:
            list of (chunk_text, similarity_score, metadata) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_emb = self.encoder.encode([query])
        similarities = (query_emb @ self.embeddings.T).squeeze(0)
        k = min(k, len(self.chunks))
        top_k = torch.topk(similarities, k=k)

        results = []
        for score, idx in zip(top_k.values, top_k.indices):
            idx = idx.item()
            results.append((
                self.chunks[idx],
                score.item(),
                self.chunk_metadata[idx],
            ))
        return results

    def search_with_doc_dedup(self, query, k=5):
        """Search with document-level deduplication.

        Returns at most one chunk per source document, preferring the
        highest-scoring chunk from each document.
        """
        if self.embeddings is None:
            return []

        query_emb = self.encoder.encode([query])
        similarities = (query_emb @ self.embeddings.T).squeeze(0)
        sorted_indices = similarities.argsort(descending=True)

        results = []
        seen_docs = set()
        for idx in sorted_indices:
            idx = idx.item()
            doc_idx = self.chunk_to_doc[idx]
            if doc_idx in seen_docs:
                continue
            seen_docs.add(doc_idx)
            results.append((
                self.chunks[idx],
                similarities[idx].item(),
                self.chunk_metadata[idx],
            ))
            if len(results) >= k:
                break
        return results

    def save(self, path):
        """Save vector store to disk."""
        torch.save({
            "documents": self.documents,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "chunk_to_doc": self.chunk_to_doc,
            "chunk_metadata": self.chunk_metadata,
            "config": {
                "chunk_strategy": self.chunk_strategy,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        }, path)
        print(f"Saved vector store to {path} ({len(self.chunks)} chunks)")

    def load(self, path):
        """Load vector store from disk."""
        data = torch.load(path, weights_only=False)
        self.documents = data["documents"]
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        self.chunk_to_doc = data["chunk_to_doc"]
        self.chunk_metadata = data["chunk_metadata"]
        config = data.get("config", {})
        self.chunk_strategy = config.get("chunk_strategy", self.chunk_strategy)
        self.chunk_size = config.get("chunk_size", self.chunk_size)
        self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        print(f"Loaded vector store from {path} ({len(self.chunks)} chunks)")

    @property
    def num_documents(self):
        return len(self.documents)

    @property
    def num_chunks(self):
        return len(self.chunks)

    def stats(self):
        """Return store statistics."""
        chunk_lengths = [len(c.split()) for c in self.chunks]
        return {
            "num_documents": self.num_documents,
            "num_chunks": self.num_chunks,
            "avg_chunk_words": sum(chunk_lengths) / max(len(chunk_lengths), 1),
            "min_chunk_words": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_words": max(chunk_lengths) if chunk_lengths else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "strategy": self.chunk_strategy,
        }


# ============================================================================
# Demo
# ============================================================================


def main():
    print("=" * 70)
    print("Module 42: Vector Store for RAG")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SimpleTokenizer(vocab_size=256, max_length=64)
    model = EmbeddingModel(
        vocab_size=256, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, max_seq_len=64,
    )
    encoder = TextEncoder(model, tokenizer, device=device)

    # --- Chunking strategies comparison ---
    print("\n" + "-" * 70)
    print("1. Chunking Strategies Comparison")
    print("-" * 70)

    sample_doc = (
        "PyTorch autograd automatically computes gradients for tensor operations. "
        "It builds a dynamic computation graph during the forward pass. "
        "Each tensor operation is recorded as a node in the graph. "
        "When you call backward, gradients flow through the graph in reverse. "
        "This enables automatic differentiation for any computation. "
        "The gradient tape is rebuilt on every forward pass. "
        "This makes PyTorch flexible for dynamic architectures like RNNs. "
        "You can use torch.no_grad to disable gradient tracking for inference."
    )

    for strategy in ["fixed", "sentence", "overlapping"]:
        fn = CHUNKING_STRATEGIES[strategy]
        if strategy == "sentence":
            chunks = fn(sample_doc, max_words=20)
        elif strategy == "overlapping":
            chunks = fn(sample_doc, chunk_size=20, stride=10)
        else:
            chunks = fn(sample_doc, chunk_size=20, overlap=5)
        print(f"\n{strategy.upper()} strategy ({len(chunks)} chunks):")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: [{len(chunk.split())} words] {chunk[:80]}...")

    # --- Build vector store ---
    print("\n" + "-" * 70)
    print("2. Building Vector Store")
    print("-" * 70)

    documents = [
        "PyTorch tensors are multi-dimensional arrays similar to NumPy arrays but with GPU support. "
        "They support automatic differentiation through the autograd system. "
        "Tensors can be created from Python lists, NumPy arrays, or other tensors.",

        "The autograd engine in PyTorch tracks operations on tensors to build a computation graph. "
        "When backward is called, it computes gradients using the chain rule. "
        "This enables training neural networks with gradient descent.",

        "torch.compile is a function that optimizes PyTorch models by capturing the computation graph. "
        "It uses Dynamo for graph capture and Inductor for code generation. "
        "Compiled models run faster by fusing operations and reducing overhead.",

        "Data loading in PyTorch uses Dataset and DataLoader classes. "
        "Dataset defines how to access individual samples. "
        "DataLoader handles batching, shuffling, and multi-process loading.",

        "Distributed training in PyTorch supports DDP and FSDP2. "
        "DDP replicates the model across GPUs and synchronizes gradients. "
        "FSDP2 shards model parameters across devices for memory efficiency.",

        "Attention mechanisms compute weighted sums of value vectors. "
        "The weights are determined by the compatibility of query and key vectors. "
        "Multi-head attention runs several attention functions in parallel.",
    ]

    store = VectorStore(encoder, chunk_strategy="fixed", chunk_size=25, chunk_overlap=5)

    start = time.perf_counter()
    store.add_documents(documents)
    elapsed = time.perf_counter() - start

    print(f"Indexed {store.num_documents} documents -> {store.num_chunks} chunks in {elapsed:.3f}s")
    stats = store.stats()
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.1f}")
        else:
            print(f"  {key}: {val}")

    # --- Search ---
    print("\n" + "-" * 70)
    print("3. Search Queries")
    print("-" * 70)

    queries = [
        "How does gradient computation work?",
        "How to load data in batches?",
        "What is torch.compile?",
        "How to train on multiple GPUs?",
    ]

    for query in queries:
        results = store.search(query, k=3)
        print(f"\nQuery: '{query}'")
        for rank, (chunk, score, meta) in enumerate(results):
            print(f"  #{rank + 1} (sim={score:.4f}, doc={meta['doc_idx']}): {chunk[:70]}...")

    # --- Deduplicated search ---
    print("\n" + "-" * 70)
    print("4. Deduplicated Search (one chunk per document)")
    print("-" * 70)

    query = "How does autograd work?"
    results_dedup = store.search_with_doc_dedup(query, k=3)
    print(f"Query: '{query}'")
    for rank, (chunk, score, meta) in enumerate(results_dedup):
        print(f"  #{rank + 1} (sim={score:.4f}, doc={meta['doc_idx']}): {chunk[:70]}...")

    # --- Persistence ---
    print("\n" + "-" * 70)
    print("5. Save and Load")
    print("-" * 70)

    save_path = "/tmp/test_vector_store.pt"
    store.save(save_path)

    store2 = VectorStore(encoder)
    store2.load(save_path)
    print(f"Loaded store: {store2.num_documents} docs, {store2.num_chunks} chunks")

    results_loaded = store2.search("gradient computation", k=2)
    print("Search on loaded store:")
    for rank, (chunk, score, meta) in enumerate(results_loaded):
        print(f"  #{rank + 1} (sim={score:.4f}): {chunk[:70]}...")

    os.remove(save_path)
    print(f"Cleaned up {save_path}")

    # --- Batch add ---
    print("\n" + "-" * 70)
    print("6. Incremental Add")
    print("-" * 70)

    new_doc = (
        "Mixed precision training uses lower-precision datatypes like float16. "
        "This reduces memory usage and speeds up computation on GPUs. "
        "PyTorch AMP provides autocast and GradScaler for safe mixed precision."
    )
    store.add_single(new_doc)
    print(f"After adding 1 doc: {store.num_documents} docs, {store.num_chunks} chunks")

    results_new = store.search("mixed precision float16", k=2)
    print("Search for 'mixed precision float16':")
    for rank, (chunk, score, meta) in enumerate(results_new):
        print(f"  #{rank + 1} (sim={score:.4f}, doc={meta['doc_idx']}): {chunk[:70]}...")

    print("\n" + "=" * 70)
    print("Vector store demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
