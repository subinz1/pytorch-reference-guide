# Module 42: Building a RAG Pipeline with PyTorch

## Overview

Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative language model.
Instead of relying solely on parametric knowledge, the model retrieves relevant context from an
external knowledge base before generating a response. This module builds a complete RAG pipeline
using only PyTorch and standard libraries — no LangChain, no vector database services.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                           │
│                                                             │
│  Query ──► Encoder ──► Similarity Search ──► Top-K Docs    │
│                                                  │          │
│                                                  ▼          │
│            Context + Query ──► Generator ──► Response       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Document Encoder

Encodes documents and queries into dense vector representations using a pre-trained
transformer. We use mean-pooling over token embeddings to produce fixed-size vectors.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class DocumentEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        output = self.model(**encoded)
        embeddings = mean_pooling(output, encoded["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)
```

### 2. Vector Index (FAISS-free, Pure PyTorch)

A simple but effective in-memory vector store using cosine similarity:

```python
class VectorIndex:
    def __init__(self):
        self.embeddings: torch.Tensor | None = None
        self.documents: list[str] = []

    def add(self, documents: list[str], embeddings: torch.Tensor):
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.documents.extend(documents)

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        similarities = torch.mm(query_embedding, self.embeddings.T).squeeze(0)
        scores, indices = torch.topk(similarities, min(top_k, len(self.documents)))
        return [(self.documents[idx], scores[i].item()) for i, idx in enumerate(indices)]
```

### 3. Context Assembly

Assembles retrieved documents into a prompt for the generator:

```python
def build_prompt(query: str, retrieved_docs: list[tuple[str, float]], max_context_tokens: int = 1024) -> str:
    context_parts = []
    for doc, score in retrieved_docs:
        context_parts.append(f"[Relevance: {score:.3f}] {doc}")

    context = "\n\n".join(context_parts)

    return f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
```

### 4. Generator

Uses a causal language model to generate answers conditioned on the retrieved context:

```python
class RAGGenerator:
    def __init__(self, model_name="gpt2", device="cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
            self.device
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

## Complete Pipeline

```python
class RAGPipeline:
    def __init__(self, encoder: DocumentEncoder, index: VectorIndex, generator: RAGGenerator):
        self.encoder = encoder
        self.index = index
        self.generator = generator

    def ingest(self, documents: list[str], batch_size: int = 32):
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self.encoder.encode(batch)
            self.index.add(batch, embeddings)

    def query(self, question: str, top_k: int = 5) -> str:
        query_emb = self.encoder.encode([question])
        retrieved = self.index.search(query_emb, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        return self.generator.generate(prompt)
```

## Key Concepts

### Why RAG?

| Approach | Pros | Cons |
|----------|------|------|
| Fine-tuning | Fast inference, no retrieval latency | Expensive, stale knowledge, hallucinations |
| RAG | Up-to-date knowledge, verifiable sources | Retrieval latency, context window limits |
| RAG + Fine-tuning | Best of both worlds | Most complex to build and maintain |

### Embedding Quality Matters

The quality of your retrieval depends entirely on the embedding model. Key considerations:

- **Domain adaptation**: General-purpose encoders may not capture domain-specific semantics
- **Chunking strategy**: Document splitting affects what gets retrieved
- **Normalization**: Always L2-normalize embeddings for cosine similarity

### Chunking Strategies

```python
def chunk_by_sentences(text: str, chunk_size: int = 3) -> list[str]:
    sentences = text.split(". ")
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i : i + chunk_size])
        if not chunk.endswith("."):
            chunk += "."
        chunks.append(chunk)
    return chunks


def chunk_by_tokens(text: str, tokenizer, max_tokens: int = 256, overlap: int = 50) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks
```

## Performance Optimization

### Batch Encoding with torch.compile

```python
@torch.compile
def batch_encode_optimized(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    return mean_pooling(output, attention_mask)
```

### GPU-Accelerated Similarity Search

For large indices (>100K documents), move embeddings to GPU:

```python
class GPUVectorIndex(VectorIndex):
    def __init__(self, device="cuda"):
        super().__init__()
        self.search_device = device

    def search(self, query_embedding: torch.Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        query_gpu = query_embedding.to(self.search_device)
        embeddings_gpu = self.embeddings.to(self.search_device)
        similarities = torch.mm(query_gpu, embeddings_gpu.T).squeeze(0)
        scores, indices = torch.topk(similarities, min(top_k, len(self.documents)))
        return [(self.documents[idx.item()], scores[i].item()) for i, idx in enumerate(indices)]
```

## Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `rag_pipeline.py` | Complete RAG pipeline implementation |
| `chunking_strategies.py` | Document chunking utilities |
| `evaluation.py` | RAG evaluation metrics (retrieval recall, answer quality) |

## References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)

---

← [Module 41: Diffusion Model](../41_diffusion_model/) | [Module 43: Production Serving Patterns](../43_production_serving/) →
