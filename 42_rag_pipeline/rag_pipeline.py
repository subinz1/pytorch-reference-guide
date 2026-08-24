"""
RAG Pipeline — Complete Implementation

A minimal but functional Retrieval-Augmented Generation pipeline using PyTorch.
No external vector databases or orchestration frameworks required.

Usage:
    python rag_pipeline.py

Requirements:
    pip install torch transformers sentence-transformers
"""

import torch
import torch.nn.functional as F


# ============================================================
# Document Encoder
# ============================================================


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class DocumentEncoder:
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        output = self.model(**encoded)
        embeddings = mean_pooling(output, encoded["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)


# ============================================================
# Vector Index (Pure PyTorch, no FAISS)
# ============================================================


class VectorIndex:
    def __init__(self):
        self.embeddings: torch.Tensor | None = None
        self.documents: list[str] = []
        self.metadata: list[dict] = []

    def add(
        self,
        documents: list[str],
        embeddings: torch.Tensor,
        metadata: list[dict] | None = None,
    ):
        if self.embeddings is None:
            self.embeddings = embeddings.cpu()
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings.cpu()], dim=0)
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(
        self, query_embedding: torch.Tensor, top_k: int = 5
    ) -> list[tuple[str, float, dict]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
        similarities = torch.mm(query_embedding.cpu(), self.embeddings.T).squeeze(0)
        k = min(top_k, len(self.documents))
        scores, indices = torch.topk(similarities, k)
        return [
            (self.documents[idx], scores[i].item(), self.metadata[idx])
            for i, idx in enumerate(indices)
        ]

    def __len__(self) -> int:
        return len(self.documents)

    def save(self, path: str):
        torch.save(
            {
                "embeddings": self.embeddings,
                "documents": self.documents,
                "metadata": self.metadata,
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.embeddings = data["embeddings"]
        self.documents = data["documents"]
        self.metadata = data["metadata"]


# ============================================================
# Context Assembly
# ============================================================


def build_prompt(
    query: str, retrieved_docs: list[tuple[str, float, dict]]
) -> str:
    context_parts = []
    for doc, score, meta in retrieved_docs:
        source = meta.get("source", "unknown")
        context_parts.append(f"[Source: {source}, Relevance: {score:.3f}]\n{doc}")

    context = "\n\n".join(context_parts)

    return f"""Answer the question based only on the provided context. If the context
doesn't contain enough information, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""


# ============================================================
# Generator
# ============================================================


class RAGGenerator:
    def __init__(self, model_name="gpt2", device="cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================
# RAG Pipeline
# ============================================================


class RAGPipeline:
    def __init__(
        self,
        encoder: DocumentEncoder,
        index: VectorIndex,
        generator: RAGGenerator,
    ):
        self.encoder = encoder
        self.index = index
        self.generator = generator

    def ingest(
        self,
        documents: list[str],
        metadata: list[dict] | None = None,
        batch_size: int = 32,
    ):
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadata[i : i + batch_size] if metadata else None
            embeddings = self.encoder.encode(batch_docs)
            self.index.add(batch_docs, embeddings, batch_meta)
        print(f"Ingested {len(documents)} documents. Index size: {len(self.index)}")

    def query(self, question: str, top_k: int = 5, verbose: bool = False) -> str:
        query_emb = self.encoder.encode([question])
        retrieved = self.index.search(query_emb, top_k=top_k)

        if verbose:
            print(f"\nQuery: {question}")
            print(f"Retrieved {len(retrieved)} documents:")
            for doc, score, meta in retrieved:
                print(f"  [{score:.3f}] {doc[:80]}...")

        prompt = build_prompt(question, retrieved)
        return self.generator.generate(prompt)


# ============================================================
# Demo
# ============================================================


def main():
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)

    # Sample knowledge base
    documents = [
        "PyTorch is an open-source machine learning framework developed by Meta AI. "
        "It provides tensor computation with GPU acceleration and automatic differentiation.",
        "torch.compile is PyTorch's compiler that uses TorchDynamo to capture Python bytecode "
        "and TorchInductor to generate optimized kernels for CPU and GPU.",
        "FSDP2 (Fully Sharded Data Parallel) distributes model parameters, gradients, and "
        "optimizer states across data-parallel workers to train models larger than single-GPU memory.",
        "FlexAttention allows defining custom attention patterns via a score_mod function "
        "that modifies attention scores before softmax, enabling causal, sliding window, and more.",
        "torch.export produces an ExportedProgram with a graph representation suitable for "
        "deployment. It captures the entire model as a single graph without Python overhead.",
        "The PyTorch dispatcher routes operator calls through a priority chain of dispatch keys, "
        "enabling features like autograd, autocast, and custom backend dispatch.",
        "Activation checkpointing trades compute for memory by recomputing intermediate activations "
        "during the backward pass instead of storing them, reducing peak memory usage.",
        "Mixed precision training uses FP16 or BF16 for forward/backward passes while keeping "
        "a master copy of weights in FP32, achieving 2-3x speedup on modern GPUs.",
    ]

    metadata = [{"source": f"pytorch_docs_{i}"} for i in range(len(documents))]

    print("\nInitializing encoder...")
    encoder = DocumentEncoder(device="cpu")

    print("Building index...")
    index = VectorIndex()

    print("Initializing generator...")
    generator = RAGGenerator(device="cpu")

    pipeline = RAGPipeline(encoder, index, generator)
    pipeline.ingest(documents, metadata)

    questions = [
        "What is torch.compile and how does it work?",
        "How does FSDP2 help with large model training?",
        "What is FlexAttention used for?",
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        answer = pipeline.query(q, top_k=3, verbose=True)
        print(f"\nAnswer: {answer}")

    print(f"\n{'=' * 60}")
    print("Demo complete.")


if __name__ == "__main__":
    main()
