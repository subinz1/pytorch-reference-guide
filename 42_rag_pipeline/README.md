# Module 42: Building a RAG Pipeline — Retrieval-Augmented Generation

Build a complete Retrieval-Augmented Generation (RAG) pipeline from scratch: embedding models, vector stores, document chunking, similarity search, prompt construction, and generation — all in pure PyTorch with no external dependencies.

**No external APIs required** — we build a self-contained RAG system using a simple transformer encoder for embeddings and a mini-LLM for generation, demonstrating every concept on synthetic PyTorch documentation.

| Input | Output |
|-------|--------|
| `"How does autograd work?"` | `Top-3 relevant documents + generated answer` |
| `30+ PyTorch knowledge base docs` | `Chunked, embedded, indexed vector store` |
| `Query + Retrieved Context` | `Grounded answer with source attribution` |

---

## Table of Contents

1. [Overview](#overview)
2. [What Is RAG?](#what-is-rag)
3. [Why RAG Beats Pure LLMs](#why-rag-beats-pure-llms)
4. [Embedding Model](#embedding-model)
5. [Mean Pooling Over Transformer Outputs](#mean-pooling-over-transformer-outputs)
6. [Vector Similarity Search](#vector-similarity-search)
7. [Document Chunking Strategies](#document-chunking-strategies)
8. [Building a Vector Store](#building-a-vector-store)
9. [Retrieval: Query to Documents](#retrieval-query-to-documents)
10. [Prompt Construction](#prompt-construction)
11. [Generation with Retrieved Context](#generation-with-retrieved-context)
12. [Complete End-to-End Pipeline](#complete-end-to-end-pipeline)
13. [Evaluation](#evaluation)
14. [Practical Tips](#practical-tips)
15. [Key Takeaways](#key-takeaways)

---

## Overview

Large Language Models (LLMs) are powerful but suffer from hallucination, knowledge cutoff, and lack of domain-specific knowledge. Retrieval-Augmented Generation (RAG) addresses these limitations by **retrieving relevant documents** from a knowledge base before generating an answer.

```
Traditional LLM:
  Query ──────────────────────────────────────────> LLM ──> Answer (may hallucinate)

RAG Pipeline:
  Query ──> Embed ──> Search Vector Store ──> Top-K Documents
                                                    │
                                                    ▼
  Query + Retrieved Context ──────────────────────> LLM ──> Grounded Answer
```

This module builds every component from scratch using PyTorch.

### Prerequisites

| Module | Topic | Why You Need It |
|--------|-------|-----------------|
| [04 — Neural Networks](../04_neural_networks/) | `nn.Module`, layers, embeddings | Building the embedding model |
| [07 — Training Pipelines](../07_training/) | Training loops, loss functions | Training embedding and generation models |
| [09 — Attention Mechanisms](../09_attention/) | Multi-head attention, transformers | Transformer encoder for embeddings |
| [22 — LLM Recipes](../22_llm_recipes/) | RoPE, KV Cache, generation | Mini-LLM for text generation |

### Files in This Module

| File | Lines | Description |
|------|-------|-------------|
| `embedding_model.py` | 200+ | Transformer encoder for text embeddings, mean pooling, similarity |
| `vector_store.py` | 200+ | In-memory vector store with chunking, indexing, search, persistence |
| `rag_pipeline.py` | 300+ | Complete RAG: knowledge base, retrieval, prompt construction, generation |

### Time Estimate

~3 hours to work through all material and experiments.

---

## What Is RAG?

**Retrieval-Augmented Generation (RAG)** is a two-stage approach to question answering:

1. **Retrieve**: Given a query, find the most relevant documents from a knowledge base
2. **Generate**: Feed the query along with retrieved documents to an LLM to produce a grounded answer

```
                    ┌──────────────────────────────────────────┐
                    │           RAG Pipeline                    │
                    │                                          │
  User Query ──────>│  1. Encode query with embedding model    │
                    │  2. Search vector store for similar docs  │
                    │  3. Retrieve top-K document chunks        │
                    │  4. Construct prompt: context + query     │
                    │  5. Generate answer with LLM              │
                    │                                          │
                    └──────────────────────────────────────────┘
```

The key insight is that instead of relying solely on what the LLM memorized during pre-training, we **dynamically inject relevant knowledge** at inference time.

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| Knowledge updates | Swap documents instantly | Retrain required |
| Hallucination | Reduced (grounded in sources) | Still possible |
| Cost | Lower (no training) | Higher (GPU hours) |
| Domain adaptation | Add domain docs | Need domain data + training |
| Interpretability | Can cite sources | Opaque |
| Latency | Retrieval adds ~50ms | No retrieval overhead |

---

## Why RAG Beats Pure LLMs

Pure LLMs have fundamental limitations that RAG addresses:

### 1. Reduces Hallucination

LLMs generate plausible-sounding text even when they don't "know" the answer. RAG grounds the generation in actual documents.

```
Without RAG:
  Q: "What is the learning rate for FSDP2?"
  A: "The default learning rate is 0.001" (hallucinated — FSDP2 doesn't set LR)

With RAG:
  Retrieved: "FSDP2 shards parameters across devices using fully_shard()..."
  A: "FSDP2 is a distributed training strategy that shards parameters.
      It doesn't set a learning rate — that's controlled by your optimizer."
```

### 2. Uses Up-to-Date Information

LLMs have a knowledge cutoff. RAG can access the latest documentation.

```
Knowledge cutoff problem:
  LLM trained on data up to Jan 2025
  Can't answer: "What's new in PyTorch 2.14?"

RAG solution:
  Vector store contains PyTorch 2.14 release notes
  Query retrieves current documentation
```

### 3. Domain-Specific Knowledge

RAG lets you add specialized knowledge without retraining.

```
Add your company's internal docs, API references, or research papers
to the vector store — the LLM can now answer domain-specific questions.
```

### 4. Source Attribution

RAG can cite which documents were used, enabling verification.

---

## Embedding Model

The embedding model converts text into dense vectors that capture semantic meaning. Similar texts should have similar embeddings.

### Architecture

We use a simple transformer encoder with mean pooling:

```
Input text tokens:  ["How", "does", "autograd", "work", "?"]
                         │
                    ┌────▼────┐
                    │ Token   │
                    │ Embed   │  vocab_size -> d_model
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Pos     │
                    │ Embed   │  max_seq_len -> d_model
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Encoder │
                    │ Layer 1 │  Self-attention + FFN
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Encoder │
                    │ Layer N │
                    └────┬────┘
                         │
                    ┌────▼────────┐
                    │ Mean Pool   │  Average all token outputs
                    └────┬────────┘
                         │
                    ┌────▼────────┐
                    │ L2 Normalize│  Unit vector
                    └────┬────────┘
                         │
                    embedding (d_model,)
```

### Why Mean Pooling?

Given a transformer that outputs hidden states `H` of shape `(seq_len, d_model)`, we need to produce a single vector. Common strategies:

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| CLS token | `h[0]` | Simple | Relies on single token |
| Mean pooling | `mean(H, dim=0)` | Uses all tokens | Padding can distort |
| Max pooling | `max(H, dim=0)` | Captures strongest features | Loses nuance |
| Weighted mean | `sum(H * w) / sum(w)` | Customizable | Needs weight design |

Mean pooling (with proper masking for padding tokens) provides the best balance of simplicity and quality. Production systems like Sentence-BERT use masked mean pooling:

```python
def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (hidden_states * mask).sum(dim=1)   # (B, d_model)
    counts = mask.sum(dim=1).clamp(min=1e-9)     # (B, 1)
    return summed / counts                        # (B, d_model)
```

### Normalization

After mean pooling, we L2-normalize the embedding to a unit vector. This makes cosine similarity equivalent to a simple dot product:

```
cos(a, b) = (a . b) / (||a|| * ||b||)

If ||a|| = ||b|| = 1:
    cos(a, b) = a . b   (just a dot product!)
```

See `embedding_model.py` for the complete implementation.

---

## Mean Pooling Over Transformer Outputs

Let's walk through mean pooling step by step with concrete shapes:

```python
# Transformer encoder output
# hidden_states shape: (batch_size, seq_len, d_model)
# Example: (2, 5, 64) — 2 sentences, max 5 tokens, 64-dim embeddings

hidden_states = encoder(token_ids)  # (2, 5, 64)

# Attention mask: 1 for real tokens, 0 for padding
# Sentence 1: "How does autograd" (3 tokens + 2 padding)
# Sentence 2: "What is a tensor in PyTorch" (5 tokens)
attention_mask = torch.tensor([
    [1, 1, 1, 0, 0],  # 3 real tokens
    [1, 1, 1, 1, 1],  # 5 real tokens
])

# Expand mask for broadcasting: (2, 5) -> (2, 5, 1)
mask_expanded = attention_mask.unsqueeze(-1).float()  # (2, 5, 1)

# Zero out padding tokens
masked_hidden = hidden_states * mask_expanded  # (2, 5, 64)

# Sum over sequence length
summed = masked_hidden.sum(dim=1)  # (2, 64)

# Count real tokens (avoid division by zero)
counts = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (2, 1)

# Mean pool
embeddings = summed / counts  # (2, 64)

# L2 normalize
embeddings = F.normalize(embeddings, p=2, dim=-1)  # (2, 64), unit vectors
```

This gives us one embedding per sentence, regardless of sequence length.

---

## Vector Similarity Search

Given a query embedding and a collection of document embeddings, we need to find the most similar documents.

### Cosine Similarity

The standard similarity metric for text embeddings:

```
cos(q, d) = (q . d) / (||q|| * ||d||)

Range: [-1, 1]
  1.0 = identical direction
  0.0 = orthogonal (unrelated)
 -1.0 = opposite direction
```

With L2-normalized embeddings, cosine similarity reduces to a dot product:

```python
# query: (d_model,) — single query embedding
# docs:  (N, d_model) — N document embeddings

similarities = docs @ query  # (N,) — dot product with each doc
top_k_values, top_k_indices = similarities.topk(k=5)
```

### Brute Force Search

For small collections (< 100K documents), brute force search is fast enough:

```python
def brute_force_search(query, doc_embeddings, k=5):
    """Search all documents. O(N * d) where N=num_docs, d=embedding_dim."""
    similarities = torch.mv(doc_embeddings, query)  # matrix-vector multiply
    top_k = torch.topk(similarities, k=k)
    return top_k.indices, top_k.values
```

**Performance**: On CPU, brute force handles ~100K 768-dim vectors in <10ms. With GPU, it scales to millions.

### Approximate Nearest Neighbors (ANN)

For larger collections, exact search becomes slow. ANN algorithms trade a small amount of accuracy for dramatic speed improvements:

| Algorithm | Idea | Library |
|-----------|------|---------|
| IVF (Inverted File) | Partition space into clusters, search nearby clusters | FAISS |
| HNSW (Hierarchical NSW) | Build a multi-layer graph of neighbors | FAISS, Annoy |
| LSH (Locality-Sensitive Hashing) | Hash similar vectors to same buckets | Custom |
| PQ (Product Quantization) | Compress vectors into compact codes | FAISS |

```
Brute Force:  O(N * d)       Exact       For N < 100K
IVF:          O(sqrt(N) * d) ~99% recall  For N < 10M
HNSW:         O(log(N) * d)  ~99% recall  For N < 100M
```

In this module, we use brute force (sufficient for our scale). Production systems typically use FAISS or similar libraries.

---

## Document Chunking Strategies

Documents are often too long to embed as a single vector. We split them into overlapping chunks that fit within the embedding model's context window.

### Why Chunk?

1. **Context window limit**: Transformer models have a maximum input length
2. **Semantic focus**: Smaller chunks have more focused meaning
3. **Retrieval precision**: The right chunk is more relevant than an entire document

### Strategy 1: Fixed-Size Chunking

Split text into chunks of a fixed number of words/tokens:

```python
def chunk_fixed_size(text, chunk_size=50, overlap=10):
    """Split text into fixed-size word chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide by (chunk_size - overlap)
    return chunks
```

```
Document: "word1 word2 word3 ... word100"

chunk_size=50, overlap=10:
  Chunk 1: word1  ... word50
  Chunk 2: word41 ... word90   (10 words overlap with chunk 1)
  Chunk 3: word81 ... word100
```

### Strategy 2: Sentence-Based Chunking

Split at sentence boundaries, then group sentences until reaching the size limit:

```python
def chunk_by_sentences(text, max_words=50):
    """Split at sentence boundaries, group up to max_words."""
    sentences = text.replace("! ", ". ").replace("? ", ". ").split(". ")
    chunks, current = [], []
    current_len = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_len + sent_words > max_words and current:
            chunks.append(". ".join(current) + ".")
            current, current_len = [], 0
        current.append(sent.strip())
        current_len += sent_words

    if current:
        chunks.append(". ".join(current) + ".")
    return chunks
```

### Strategy 3: Overlapping Windows

A variant of fixed-size that ensures context isn't lost at boundaries:

```
Without overlap:
  [chunk 1: "The autograd engine tracks"] [chunk 2: "operations on tensors"]
  ^-- This split loses the connection between "tracks" and "operations"

With overlap (50% stride):
  [chunk 1: "The autograd engine tracks operations"]
  [chunk 2: "engine tracks operations on tensors"]
  ^-- The overlap preserves context across boundaries
```

### Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Fixed-size | Simple, predictable | May split mid-sentence | Uniform documents |
| Sentence-based | Respects boundaries | Uneven chunk sizes | Prose, documentation |
| Overlapping | No lost context | More chunks, redundancy | Technical content |

See `vector_store.py` for all three chunking implementations.

---

## Building a Vector Store

A vector store holds document embeddings and supports efficient similarity search.

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Vector Store               │
                    │                                     │
                    │  documents: List[str]    (raw text)  │
                    │  chunks:    List[str]    (chunked)   │
                    │  embeddings: Tensor      (N, d_model)│
                    │  chunk_to_doc: List[int] (mapping)   │
                    │                                     │
                    │  Methods:                           │
                    │    add_documents(texts)              │
                    │    search(query, k=5) -> results     │
                    │    save(path) / load(path)           │
                    └─────────────────────────────────────┘
```

### Implementation Overview

```python
class VectorStore:
    def __init__(self, embedding_model, chunk_size=50, chunk_overlap=10):
        self.model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.chunk_to_doc = []

    def add_documents(self, texts):
        for doc_idx, text in enumerate(texts):
            doc_chunks = self.chunk_text(text)
            for chunk in doc_chunks:
                self.chunks.append(chunk)
                self.chunk_to_doc.append(doc_idx)
            self.documents.append(text)

        all_embeddings = self.model.encode(self.chunks)
        self.embeddings = all_embeddings

    def search(self, query, k=5):
        query_emb = self.model.encode([query])          # (1, d)
        sims = query_emb @ self.embeddings.T             # (1, N)
        top_k = sims[0].topk(k)
        return [(self.chunks[i], top_k.values[j].item())
                for j, i in enumerate(top_k.indices)]
```

### Persistence

Save the vector store to disk for reuse:

```python
def save(self, path):
    torch.save({
        "documents": self.documents,
        "chunks": self.chunks,
        "embeddings": self.embeddings,
        "chunk_to_doc": self.chunk_to_doc,
    }, path)

def load(self, path):
    data = torch.load(path, weights_only=False)
    self.documents = data["documents"]
    self.chunks = data["chunks"]
    self.embeddings = data["embeddings"]
    self.chunk_to_doc = data["chunk_to_doc"]
```

See `vector_store.py` for the complete implementation with all chunking strategies and batch operations.

---

## Retrieval: Query to Documents

The retrieval stage converts a user query into the most relevant document chunks.

### Step 1: Query Embedding

```python
query = "How does PyTorch autograd work?"
query_embedding = embedding_model.encode([query])  # (1, d_model)
```

### Step 2: Top-K Search

```python
similarities = query_embedding @ store.embeddings.T  # (1, num_chunks)
top_k_scores, top_k_indices = similarities[0].topk(k=5)

retrieved_chunks = [store.chunks[i] for i in top_k_indices]
```

### Step 3: Re-Ranking (Optional)

Initial retrieval uses fast embedding similarity. Re-ranking applies a more expensive but accurate cross-encoder to the top candidates:

```
Stage 1 (fast):  Query embedding vs 10,000 doc embeddings → top 20 candidates
Stage 2 (slow):  Cross-encoder scores (query, candidate) → top 5 re-ranked

Cross-encoder architecture:
  Input:  [CLS] query [SEP] candidate [SEP]
  Output: relevance score (0 to 1)
```

The two-stage approach combines the speed of embedding search with the accuracy of cross-attention:

```python
def rerank(query, candidates, cross_encoder, k=5):
    """Re-rank candidates using a cross-encoder for better precision."""
    scores = []
    for candidate in candidates:
        score = cross_encoder(query, candidate)
        scores.append(score)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return ranked[:k]
```

### Retrieval Quality Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Recall@K | `relevant_in_top_k / total_relevant` | Did we find the right docs? |
| Precision@K | `relevant_in_top_k / k` | Are retrieved docs relevant? |
| MRR | `1 / rank_of_first_relevant` | How high is the best result? |
| NDCG@K | Normalized Discounted Cumulative Gain | Weighted ranking quality |

---

## Prompt Construction

After retrieval, we construct a prompt that combines the retrieved context with the user's query.

### Basic Template

```python
def build_prompt(query, retrieved_chunks, max_context_tokens=500):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
    return prompt
```

### Prompt Design Principles

1. **Instruction**: Tell the model to use the context
2. **Context first**: Place retrieved documents before the question
3. **Grounding instruction**: Ask the model to say "I don't know" when context is insufficient
4. **Clear separation**: Use markers to separate context from query

### Advanced Template

```python
def build_prompt_advanced(query, chunks_with_scores):
    """Include source attribution and confidence-based ordering."""
    context_parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores):
        context_parts.append(f"[Source {i+1}] (relevance: {score:.3f})\n{chunk}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY
the provided sources. Cite sources as [Source N]. If the sources don't
contain the answer, say "I cannot find this information in the provided sources."

Sources:
{context}

Question: {query}

Answer:"""
    return prompt
```

---

## Generation with Retrieved Context

The final stage feeds the constructed prompt to a language model.

### Mini-LLM for Generation

We use a small transformer decoder (similar to Module 22) for generation. In production, this would be a full-scale LLM like GPT-4 or Llama.

```
Prompt tokens: "Answer... Context: [retrieved docs]... Question: [query]... Answer:"
                                            │
                                    ┌───────▼───────┐
                                    │  Mini-LLM     │
                                    │  (Decoder)    │
                                    │               │
                                    │  - Token emb  │
                                    │  - Pos emb    │
                                    │  - N layers   │
                                    │  - Causal attn│
                                    │  - FFN        │
                                    │  - LM head    │
                                    └───────┬───────┘
                                            │
                                    Generated tokens
```

### Generation Strategy

```python
@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.7):
    """Autoregressive generation with temperature sampling."""
    tokens = prompt_tokens.clone()

    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :]         # last position
        logits = logits / temperature             # temperature scaling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)  # sample
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    return tokens
```

---

## Complete End-to-End Pipeline

Here is the full RAG pipeline assembled from all components:

```python
class RAGPipeline:
    def __init__(self, embedding_model, vector_store, generator):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.generator = generator

    def add_knowledge(self, documents):
        """Index documents into the vector store."""
        self.vector_store.add_documents(documents)

    def query(self, question, top_k=3):
        """Full RAG: retrieve, construct prompt, generate."""
        # 1. Retrieve relevant chunks
        results = self.vector_store.search(question, k=top_k)
        chunks = [chunk for chunk, score in results]

        # 2. Build prompt with context
        prompt = self.build_prompt(question, chunks)

        # 3. Generate answer
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "retrieved_chunks": results,
            "prompt": prompt,
            "answer": answer,
        }
```

### Pipeline Flow Diagram

```
┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│  Knowledge   │     │   User Query   │     │             │
│  Base (docs) │     │                │     │             │
└──────┬───────┘     └───────┬────────┘     │             │
       │                     │              │             │
       ▼                     ▼              │             │
┌──────────────┐     ┌────────────────┐     │             │
│  Chunk docs  │     │  Embed query   │     │             │
└──────┬───────┘     └───────┬────────┘     │             │
       │                     │              │             │
       ▼                     ▼              │             │
┌──────────────┐     ┌────────────────┐     │   Vector    │
│  Embed       │     │  Cosine        │     │   Store     │
│  chunks      │────>│  similarity    │<────│             │
└──────────────┘     └───────┬────────┘     │             │
                             │              │             │
                             ▼              │             │
                     ┌────────────────┐     │             │
                     │  Top-K chunks  │     │             │
                     └───────┬────────┘     └─────────────┘
                             │
                             ▼
                     ┌────────────────┐
                     │  Build prompt  │
                     │  context+query │
                     └───────┬────────┘
                             │
                             ▼
                     ┌────────────────┐
                     │  Generate      │
                     │  answer (LLM)  │
                     └───────┬────────┘
                             │
                             ▼
                     ┌────────────────┐
                     │  Answer +      │
                     │  sources       │
                     └────────────────┘
```

See `rag_pipeline.py` for the complete implementation with 30+ knowledge base documents, evaluation, and comparison with/without retrieval.

---

## Evaluation

### Retrieval Evaluation

Measure how well the retriever finds relevant documents:

```python
def evaluate_retrieval(store, queries, ground_truth, k=5):
    """
    queries: list of query strings
    ground_truth: list of sets of relevant doc indices
    """
    recalls, precisions, mrrs = [], [], []

    for query, relevant in zip(queries, ground_truth):
        results = store.search(query, k=k)
        retrieved_indices = {store.chunk_to_doc[i] for _, i in results}

        recall = len(retrieved_indices & relevant) / len(relevant)
        precision = len(retrieved_indices & relevant) / k
        rank = next((i+1 for i, (_, idx) in enumerate(results)
                     if store.chunk_to_doc[idx] in relevant), 0)
        mrr = 1.0 / rank if rank > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

    return {
        "recall@k": sum(recalls) / len(recalls),
        "precision@k": sum(precisions) / len(precisions),
        "mrr": sum(mrrs) / len(mrrs),
    }
```

### Answer Evaluation

Measure the quality of generated answers:

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Faithfulness** | Is the answer supported by context? | Check claims against retrieved docs |
| **Relevance** | Does the answer address the question? | Semantic similarity to expected answer |
| **Completeness** | Does it cover all aspects? | Compare against reference answer |
| **Hallucination rate** | Does it add unsupported claims? | Claims not in retrieved context |

```python
def evaluate_answer_relevance(answer, reference, embedding_model):
    """Measure semantic similarity between generated and reference answers."""
    answer_emb = embedding_model.encode([answer])
    ref_emb = embedding_model.encode([reference])
    similarity = (answer_emb @ ref_emb.T).item()
    return similarity
```

### With vs Without Retrieval

A key evaluation is comparing RAG against a pure LLM:

```
Query: "What optimizer should I use for fine-tuning?"

Without RAG (pure LLM):
  "Use Adam optimizer with lr=0.001"
  (Generic, possibly outdated)

With RAG:
  Retrieved: "AdamW with weight decay 0.01 is recommended for
              fine-tuning. Use a cosine LR schedule..."
  "Based on the documentation, use AdamW with weight decay 0.01
   and a cosine learning rate schedule for fine-tuning. [Source 1]"
  (Grounded, specific, citable)
```

---

## Practical Tips

### 1. Chunk Size Matters

```
Too small (10 words):  Loses context, fragments meaning
Too large (500 words): Dilutes relevance, wastes context window
Sweet spot (50-200):   Focused but complete thoughts
```

Experiment with your data. Technical docs often need larger chunks than conversational text.

### 2. Overlap Prevents Information Loss

Always use some overlap (10-20% of chunk size) to avoid splitting critical information across chunk boundaries.

### 3. Embedding Quality Is Critical

The embedding model is the bottleneck. A bad embedding model means bad retrieval, regardless of how good the LLM is.

```
Garbage embeddings → Irrelevant retrieval → Wrong context → Bad answer
Good embeddings   → Relevant retrieval   → Right context → Good answer
```

### 4. More Context Is Not Always Better

Retrieving too many chunks can confuse the LLM. Start with top-3, experiment up to top-5.

### 5. Metadata Filtering

Add metadata (source, date, category) to chunks and filter before similarity search:

```python
results = store.search(query, k=10, filter={"source": "pytorch_docs", "version": "2.14"})
```

### 6. Hybrid Search

Combine semantic search (embeddings) with keyword search (BM25) for better recall:

```
Semantic: "How to speed up training" → finds "optimization techniques"
Keyword:  "torch.compile" → finds exact mentions of torch.compile
Hybrid:   Combines both → best of both worlds
```

### 7. Query Expansion

Rewrite the user query to improve retrieval:

```python
def expand_query(query):
    """Generate multiple search queries for better coverage."""
    return [
        query,
        f"What is {query}?",
        f"How to {query}",
        f"Example of {query}",
    ]
```

### 8. Handle No-Context Gracefully

When retrieved context isn't relevant (low similarity scores), tell the user:

```python
if max_similarity < 0.3:
    return "I don't have enough information to answer this question."
```

---

## Key Takeaways

1. **RAG = Retrieve + Generate**: Fetch relevant documents first, then generate grounded answers
2. **Embedding model is king**: The quality of your embeddings determines retrieval quality
3. **Mean pooling + L2 normalize**: Standard recipe for sentence embeddings
4. **Cosine similarity**: The default metric for comparing text embeddings
5. **Chunking matters**: Balance chunk size, overlap, and boundary awareness
6. **Brute force is fine**: For < 100K documents, exact search is fast enough
7. **Prompt engineering**: How you format context + query affects generation quality
8. **Evaluate both stages**: Measure retrieval quality (Recall@K) and answer quality separately
9. **Start simple, iterate**: Begin with basic RAG, then add re-ranking, hybrid search, etc.

---

## References

- [Module 04 — Neural Networks](../04_neural_networks/) — `nn.Module`, embeddings, layers
- [Module 07 — Training Pipelines](../07_training/) — training loops, loss functions
- [Module 09 — Attention Mechanisms](../09_attention/) — multi-head attention, transformers
- [Module 22 — LLM Recipes](../22_llm_recipes/) — RoPE, KV Cache, mini-LLM generation
- [Module 41 — Diffusion Model](../41_diffusion_model/) — another end-to-end project
- [RAG Paper — Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT — Reimers & Gurevych 2019](https://arxiv.org/abs/1908.10084)
- [DPR — Karpukhin et al. 2020](https://arxiv.org/abs/2004.04906)
- [FAISS — Johnson et al. 2019](https://arxiv.org/abs/1702.08734)

---

### Upstream Updates (PyTorch 2.14+)

| Feature | Impact on RAG Pipelines |
|---------|------------------------|
| `torch.compile` | Compiles the embedding model for faster batch encoding |
| FlexAttention | Custom attention patterns for long-context embedding models |
| `torch.float8` | FP8 inference for embedding models at scale |
| FSDP2 | Distributed embedding model training on large corpora |
| `torch.export` | Export embedding models for production deployment |

---

<div align="center">

[← Previous Module (Diffusion Model)](../41_diffusion_model/) | [🏠 Home](../README.md) | Next Module → (coming soon)

**Notebook**: [`42_rag_pipeline.ipynb`](../notebooks/42_rag_pipeline.ipynb)

</div>
