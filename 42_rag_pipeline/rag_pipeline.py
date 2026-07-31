"""
Module 42: Complete RAG Pipeline

Implements:
- Synthetic PyTorch knowledge base (30+ documents)
- Document chunking and vector store indexing
- Query pipeline: embed query -> retrieve top-K -> construct prompt -> generate
- Mini-LLM for text generation (small transformer decoder)
- Evaluation: retrieval accuracy, answer relevance, with/without comparison
- End-to-end demo

Usage:
    python rag_pipeline.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_model import EmbeddingModel, SimpleTokenizer, TextEncoder
from vector_store import VectorStore


# ============================================================================
# Knowledge Base: Synthetic PyTorch Documentation (30+ documents)
# ============================================================================

PYTORCH_KNOWLEDGE_BASE = [
    # Tensors & Autograd
    "PyTorch tensors are the fundamental data structure. They are multi-dimensional arrays "
    "that support automatic differentiation. Tensors can live on CPU or GPU. Use torch.tensor() "
    "to create tensors from Python lists. Use torch.zeros(), torch.ones(), torch.randn() for "
    "common initialization patterns. Tensors support broadcasting and are similar to NumPy arrays.",

    "Autograd is PyTorch's automatic differentiation engine. It records operations on tensors "
    "to build a dynamic computation graph. When you call .backward() on a scalar loss, autograd "
    "computes gradients for all tensors with requires_grad=True. The gradient is accumulated in "
    "the .grad attribute. Use torch.no_grad() to disable gradient tracking during inference.",

    "The computation graph in PyTorch is dynamic, rebuilt on every forward pass. This allows "
    "different control flow per sample, making it ideal for variable-length sequences and "
    "recursive structures. Each operation creates a Function node with a backward method. "
    "Calling .backward() traverses the graph in reverse topological order.",

    # Neural Networks
    "nn.Module is the base class for all neural network models. Subclass it and implement "
    "forward() to define your model. Modules automatically track parameters. Use model.parameters() "
    "to get all learnable weights. Key layers include nn.Linear, nn.Conv2d, nn.LSTM, nn.Embedding, "
    "nn.LayerNorm, nn.BatchNorm2d, nn.Dropout, and nn.MultiheadAttention.",

    "Loss functions in PyTorch are in the torch.nn module. Common losses include "
    "nn.CrossEntropyLoss for classification, nn.MSELoss for regression, nn.BCEWithLogitsLoss "
    "for binary classification, nn.L1Loss, nn.SmoothL1Loss, and nn.KLDivLoss. "
    "CrossEntropyLoss combines LogSoftmax and NLLLoss in one class.",

    "Hooks in PyTorch allow inspecting or modifying module behavior. Forward hooks are called "
    "after forward(), backward hooks during backpropagation. Register with "
    "module.register_forward_hook() or module.register_full_backward_hook(). "
    "Hooks are useful for feature extraction, gradient modification, and debugging.",

    # Optimizers & Training
    "PyTorch optimizers update model parameters based on computed gradients. Common optimizers "
    "include SGD, Adam, AdamW, and Adafactor. AdamW decouples weight decay from the gradient "
    "update. Always call optimizer.zero_grad() before loss.backward() and optimizer.step() after. "
    "Use torch.optim.lr_scheduler for learning rate schedules.",

    "Training loops in PyTorch follow the pattern: forward pass, compute loss, backward pass, "
    "optimizer step. Use model.train() and model.eval() to switch between training and evaluation "
    "modes. Gradient accumulation allows effective larger batch sizes. Mixed precision training "
    "with torch.amp reduces memory and speeds up training.",

    "Transfer learning uses a pretrained model as a starting point. Freeze the backbone layers "
    "and only train the classifier head, or fine-tune all layers with a smaller learning rate. "
    "Common in computer vision with ImageNet-pretrained models and in NLP with pretrained "
    "language models. Use model.requires_grad_(False) to freeze layers.",

    # Data Loading
    "torch.utils.data.Dataset defines how to access individual samples. Implement __len__() "
    "and __getitem__(). Map-style datasets support random access. IterableDataset is for "
    "streaming data. DataLoader handles batching, shuffling, and parallel loading with "
    "num_workers. Use collate_fn for custom batching logic.",

    "Data augmentation transforms training data to improve generalization. In torchvision, "
    "use transforms.Compose to chain operations: RandomCrop, RandomHorizontalFlip, "
    "ColorJitter, RandomRotation, Normalize. MixUp and CutMix blend training samples. "
    "Augmentations should only be applied during training, not evaluation.",

    # torch.compile
    "torch.compile optimizes PyTorch models by capturing the computation graph using TorchDynamo "
    "and generating optimized code with TorchInductor. Use model = torch.compile(model) to "
    "compile. Modes include 'default' for balanced optimization, 'reduce-overhead' for small "
    "models, and 'max-autotune' for maximum performance. Graph breaks occur when Dynamo cannot "
    "trace through Python code.",

    "Debugging torch.compile issues: use TORCH_LOGS=dynamo to see graph captures. Check for "
    "graph breaks with torch._dynamo.explain(model, input). Common causes of graph breaks: "
    "data-dependent control flow, unsupported Python features, and dynamic shapes. Use "
    "torch.compiler.disable() to skip problematic regions.",

    # Attention & Transformers
    "Scaled dot-product attention computes attention weights from queries, keys, and values. "
    "Use torch.nn.functional.scaled_dot_product_attention() which automatically selects the "
    "fastest backend: FlashAttention-2 or memory-efficient attention. "
    "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.",

    "Multi-head attention splits the embedding into multiple heads, applies attention "
    "independently, then concatenates. Use nn.MultiheadAttention(d_model, num_heads). "
    "This allows the model to attend to different aspects of the input simultaneously. "
    "Each head learns different attention patterns.",

    "FlexAttention in PyTorch allows custom attention patterns using score_mod functions. "
    "It supports causal masking, sliding window attention, prefix LM patterns, and more. "
    "FlexAttention is compiled and fused for performance. Use torch.nn.attention.flex_attention "
    "to apply custom attention logic without materializing full attention matrices.",

    # Distributed
    "Distributed Data Parallel (DDP) replicates the model on each GPU and synchronizes "
    "gradients after each backward pass using all-reduce. Initialize with "
    "torch.distributed.init_process_group(). Wrap your model with "
    "DistributedDataParallel(model). Each process handles a subset of the data.",

    "FSDP2 (Fully Sharded Data Parallel) shards model parameters, gradients, and optimizer "
    "states across GPUs using fully_shard(). It reduces per-GPU memory compared to DDP. "
    "Parameters are gathered during forward and backward, then resharded. FSDP2 uses "
    "DTensor for sharding and DeviceMesh for device topology.",

    "Pipeline Parallelism splits the model into stages across GPUs, with micro-batches flowing "
    "through the pipeline. Reduces memory per GPU but introduces pipeline bubbles. "
    "Strategies include GPipe, 1F1B, ZeroBubble, and DualPipeV. Combine with FSDP and TP "
    "for 3D parallelism in large-scale training.",

    # Export & Deploy
    "torch.export captures a PyTorch model into an ExportedProgram for deployment. It uses "
    "symbolic tracing to create a fully functional graph. Supports dynamic shapes via Dim. "
    "Export preserves Python semantics. Use torch.export.export(model, args) to export, then "
    "serialize or compile with AOTInductor.",

    "AOTInductor compiles exported models into standalone C++ shared libraries for deployment. "
    "Use torch._export.aot_compile() or aoti_compile_and_package(). The output runs without "
    "Python, making it suitable for production C++ inference. Supports CUDA and CPU targets.",

    # Advanced Features
    "Functorch provides functional transforms for PyTorch: vmap (vectorized map), grad "
    "(functional gradients), jacrev/jacfwd (Jacobians), and hessian. These enable "
    "per-sample gradients, meta-learning, and efficient Jacobian computation. "
    "Use torch.func.vmap to vectorize over a batch dimension.",

    "Custom operators in PyTorch are registered with torch.library. Define the schema, "
    "implement for each backend (CPU, CUDA), and add FakeTensor support for torch.compile. "
    "Use @torch.library.custom_op for simple cases or the full Library API for complex ops. "
    "Custom ops work with autograd, torch.compile, and torch.export.",

    # Memory & Performance
    "Mixed precision training uses lower-precision types (float16, bfloat16) to reduce memory "
    "and speed up computation. Use torch.amp.autocast('cuda') for automatic precision selection. "
    "GradScaler prevents underflow with float16. bfloat16 has the same range as float32 "
    "so it does not need GradScaler. FP8 training is emerging for even more efficiency.",

    "Memory optimization techniques include gradient checkpointing (recompute instead of store), "
    "mixed precision (reduce memory per element), gradient accumulation (reduce peak memory), "
    "model sharding (FSDP), and CPU offloading. Use torch.cuda.memory_stats() and the memory "
    "profiler to identify bottlenecks.",

    "CUDA Graphs capture a sequence of GPU operations and replay them with minimal CPU overhead. "
    "Use torch.cuda.CUDAGraph with capture and replay. Static inputs must be reused between "
    "captures. torch.compile with mode='reduce-overhead' automatically uses CUDA Graphs. "
    "Best for small models where CPU launch overhead dominates.",

    # LLM-related
    "Rotary Position Embedding (RoPE) encodes position by rotating query and key vectors. "
    "It preserves relative position information and supports extrapolation to longer sequences. "
    "RoPE applies a rotation matrix based on position and dimension index. Used in LLaMA, "
    "Mistral, and most modern LLMs. Implements complex multiplication in the frequency domain.",

    "KV Cache stores previously computed key and value tensors during autoregressive generation. "
    "This avoids recomputing attention for all previous tokens at each step. The cache grows "
    "linearly with sequence length. Grouped-Query Attention (GQA) reduces KV cache size by "
    "sharing key-value heads across multiple query heads.",

    "LoRA (Low-Rank Adaptation) fine-tunes LLMs by adding small trainable rank-decomposition "
    "matrices to frozen pretrained weights. Instead of updating a full d x d weight matrix, "
    "LoRA learns two small matrices A (d x r) and B (r x d) where r << d. This reduces "
    "trainable parameters by 100-1000x while maintaining performance.",

    # Testing & Debugging
    "PyTorch testing uses TestCase from torch.testing._internal.common_utils. Use assertEqual "
    "for tensor comparison with tolerances. Reproduce tests with manual seeds. The @parametrize "
    "decorator generates tests over multiple inputs. Use instantiate_device_type_tests for "
    "device-generic tests.",

    "Debugging PyTorch models: use torch.autograd.set_detect_anomaly(True) to catch NaN "
    "gradients. Register hooks to inspect intermediate values. For torch.compile issues, "
    "use TORCH_LOGS=dynamo,graph_code. Check tensor shapes with assertions. Use torch.isnan() "
    "and torch.isinf() to detect numerical issues.",

    # Quantization
    "torchao provides post-training quantization and sparsity tools. Use quantize_() to apply "
    "INT8, INT4, or FP8 quantization to model weights. Quantization reduces model size and "
    "speeds up inference on supported hardware. It integrates with torch.compile for optimized "
    "quantized kernels. 2:4 sparsity achieves roughly 2x speedup on supported GPUs.",
]


# ============================================================================
# Mini-LLM (Simple Transformer Decoder for Generation)
# ============================================================================


class MiniLM(nn.Module):
    """Small transformer decoder for demonstration generation.

    In production, this would be replaced by a full-scale LLM.
    Here we use it to demonstrate the RAG pipeline flow.
    """

    def __init__(self, vocab_size=256, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.decoder(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        """Autoregressive generation with temperature and top-k sampling."""
        self.eval()
        tokens = prompt_ids.clone()
        for _ in range(max_new_tokens):
            if tokens.shape[1] >= self.max_seq_len:
                break
            logits = self.forward(tokens)[:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k)
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


# ============================================================================
# Simple Tokenizer for Generator (reuse char-level approach)
# ============================================================================


class GeneratorTokenizer:
    """Character-level tokenizer for the mini-LLM generator."""

    def __init__(self, vocab_size=256, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def encode(self, text):
        ids = [min(ord(c), self.vocab_size - 1) for c in text]
        return torch.tensor([ids[:self.max_length]], dtype=torch.long)

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.squeeze(0).tolist()
        return "".join(chr(min(t, 127)) for t in token_ids if 32 <= t <= 126)


# ============================================================================
# RAG Pipeline
# ============================================================================


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline.

    Components:
        1. Embedding model: encodes text into dense vectors
        2. Vector store: indexes and searches document embeddings
        3. Generator: produces text given a prompt with context
    """

    def __init__(self, encoder, vector_store, generator, gen_tokenizer):
        self.encoder = encoder
        self.store = vector_store
        self.generator = generator
        self.gen_tokenizer = gen_tokenizer

    def add_knowledge(self, documents, batch_size=32):
        """Index documents into the vector store."""
        self.store.add_documents(documents, batch_size=batch_size)

    def build_prompt(self, query, retrieved_chunks):
        """Construct a prompt combining retrieved context with the query."""
        context_parts = []
        for i, (chunk, score, _meta) in enumerate(retrieved_chunks):
            context_parts.append(f"[Source {i + 1}] {chunk}")
        context = "\n".join(context_parts)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return prompt

    def retrieve(self, query, top_k=3):
        """Retrieve the most relevant chunks for a query."""
        return self.store.search(query, k=top_k)

    def generate_answer(self, prompt, max_tokens=80):
        """Generate an answer using the mini-LLM."""
        prompt_ids = self.gen_tokenizer.encode(prompt)
        device = next(self.generator.parameters()).device
        prompt_ids = prompt_ids.to(device)
        output_ids = self.generator.generate(prompt_ids, max_new_tokens=max_tokens, temperature=0.7)
        new_tokens = output_ids[:, prompt_ids.shape[1]:]
        return self.gen_tokenizer.decode(new_tokens)

    def query(self, question, top_k=3, max_tokens=80):
        """Full RAG pipeline: retrieve, construct prompt, generate."""
        retrieved = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, retrieved)
        answer = self.generate_answer(prompt, max_tokens=max_tokens)
        return {
            "question": question,
            "retrieved": [(chunk, score) for chunk, score, _ in retrieved],
            "prompt_length": len(prompt),
            "answer": answer,
        }

    def query_without_retrieval(self, question, max_tokens=80):
        """Generate without retrieval (baseline for comparison)."""
        prompt = f"Question: {question}\n\nAnswer:"
        answer = self.generate_answer(prompt, max_tokens=max_tokens)
        return {
            "question": question,
            "retrieved": [],
            "prompt_length": len(prompt),
            "answer": answer,
        }


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_retrieval(store, encoder, queries_and_keywords):
    """Evaluate retrieval quality using keyword matching as ground truth.

    For each query, we check if retrieved chunks contain expected keywords.
    """
    results = {"recall": [], "precision": [], "mrr": []}

    for query, expected_keywords in queries_and_keywords:
        retrieved = store.search(query, k=5)

        relevant_found = 0
        first_relevant_rank = 0
        for rank, (chunk, score, meta) in enumerate(retrieved):
            chunk_lower = chunk.lower()
            if any(kw.lower() in chunk_lower for kw in expected_keywords):
                relevant_found += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = rank + 1

        recall = min(relevant_found / max(len(expected_keywords), 1), 1.0)
        precision = relevant_found / len(retrieved) if retrieved else 0.0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0

        results["recall"].append(recall)
        results["precision"].append(precision)
        results["mrr"].append(mrr)

    return {
        "avg_recall@5": sum(results["recall"]) / len(results["recall"]),
        "avg_precision@5": sum(results["precision"]) / len(results["precision"]),
        "avg_mrr": sum(results["mrr"]) / len(results["mrr"]),
    }


def evaluate_answer_relevance(encoder, questions, answers, references):
    """Measure semantic similarity between generated and reference answers."""
    answer_embs = encoder.encode(answers)
    ref_embs = encoder.encode(references)
    similarities = (answer_embs * ref_embs).sum(dim=1)
    return {
        "avg_similarity": similarities.mean().item(),
        "per_question": [
            {"question": q, "similarity": s.item()}
            for q, s in zip(questions, similarities)
        ],
    }


# ============================================================================
# End-to-End Demo
# ============================================================================


def main():
    print("=" * 70)
    print("Module 42: Complete RAG Pipeline")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Initialize components ---
    print("\n" + "-" * 70)
    print("1. Initializing Components")
    print("-" * 70)

    tokenizer = SimpleTokenizer(vocab_size=256, max_length=64)
    emb_model = EmbeddingModel(
        vocab_size=256, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, max_seq_len=64,
    )
    encoder = TextEncoder(emb_model, tokenizer, device=device)
    print(f"Embedding model: {sum(p.numel() for p in emb_model.parameters()):,} params")

    gen_model = MiniLM(
        vocab_size=256, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, max_seq_len=512,
    ).to(device)
    gen_tokenizer = GeneratorTokenizer(vocab_size=256, max_length=512)
    print(f"Generator model: {sum(p.numel() for p in gen_model.parameters()):,} params")

    store = VectorStore(encoder, chunk_strategy="fixed", chunk_size=30, chunk_overlap=5)

    rag = RAGPipeline(encoder, store, gen_model, gen_tokenizer)

    # --- 2. Index knowledge base ---
    print("\n" + "-" * 70)
    print("2. Indexing Knowledge Base")
    print("-" * 70)

    start = time.perf_counter()
    rag.add_knowledge(PYTORCH_KNOWLEDGE_BASE)
    elapsed = time.perf_counter() - start

    stats = store.stats()
    print(f"Indexed {stats['num_documents']} documents -> {stats['num_chunks']} chunks in {elapsed:.3f}s")
    print(f"Avg chunk size: {stats['avg_chunk_words']:.1f} words")
    print(f"Embedding dim: {stats['embedding_dim']}")

    # --- 3. Retrieval demo ---
    print("\n" + "-" * 70)
    print("3. Retrieval Demo")
    print("-" * 70)

    demo_queries = [
        "How does autograd compute gradients?",
        "What is torch.compile and how do I use it?",
        "How do I train on multiple GPUs?",
        "What is LoRA fine-tuning?",
        "How to debug NaN gradients?",
        "What is KV cache in transformers?",
    ]

    for query in demo_queries:
        results = store.search(query, k=3)
        print(f"\nQuery: '{query}'")
        for rank, (chunk, score, meta) in enumerate(results):
            preview = chunk[:80].replace("\n", " ")
            print(f"  #{rank + 1} (sim={score:.4f}, doc={meta['doc_idx']:2d}): {preview}...")

    # --- 4. Full RAG queries ---
    print("\n" + "-" * 70)
    print("4. Full RAG Pipeline (retrieve + generate)")
    print("-" * 70)

    rag_queries = [
        "How does PyTorch autograd work?",
        "What optimizer should I use for training?",
        "How to speed up model inference?",
    ]

    for query in rag_queries:
        result = rag.query(query, top_k=3, max_tokens=60)
        print(f"\nQuery: '{result['question']}'")
        print(f"  Retrieved {len(result['retrieved'])} chunks (prompt: {result['prompt_length']} chars)")
        for i, (chunk, score) in enumerate(result["retrieved"]):
            print(f"    Source {i + 1} (sim={score:.4f}): {chunk[:60]}...")
        print(f"  Generated answer: '{result['answer'][:120]}'")

    # --- 5. With vs Without Retrieval ---
    print("\n" + "-" * 70)
    print("5. Comparison: With vs Without Retrieval")
    print("-" * 70)

    comparison_queries = [
        "What is FSDP2?",
        "How does RoPE encoding work?",
        "What is mixed precision training?",
    ]

    for query in comparison_queries:
        with_rag = rag.query(query, top_k=3, max_tokens=50)
        without_rag = rag.query_without_retrieval(query, max_tokens=50)
        print(f"\nQuery: '{query}'")
        print(f"  With RAG    (prompt={with_rag['prompt_length']:4d} chars): {with_rag['answer'][:80]}")
        print(f"  Without RAG (prompt={without_rag['prompt_length']:4d} chars): {without_rag['answer'][:80]}")

    # --- 6. Retrieval evaluation ---
    print("\n" + "-" * 70)
    print("6. Retrieval Evaluation")
    print("-" * 70)

    eval_queries = [
        ("How to compute gradients?", ["autograd", "backward", "gradient"]),
        ("What is torch.compile?", ["compile", "dynamo", "inductor"]),
        ("How to train on multiple GPUs?", ["distributed", "ddp", "fsdp"]),
        ("What is LoRA?", ["lora", "low-rank", "fine-tun"]),
        ("What is attention?", ["attention", "query", "key", "value"]),
        ("How to debug models?", ["debug", "nan", "anomaly"]),
        ("What is CUDA Graphs?", ["cuda graph", "capture", "replay"]),
        ("How does KV cache work?", ["kv cache", "key", "value", "autoregressive"]),
        ("What is quantization?", ["quantiz", "int8", "int4"]),
        ("How to export models?", ["export", "aotinductor", "deploy"]),
    ]

    eval_results = evaluate_retrieval(store, encoder, eval_queries)
    print(f"Retrieval metrics over {len(eval_queries)} queries:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")

    # --- 7. Pipeline statistics ---
    print("\n" + "-" * 70)
    print("7. Pipeline Statistics")
    print("-" * 70)

    total_emb_params = sum(p.numel() for p in emb_model.parameters())
    total_gen_params = sum(p.numel() for p in gen_model.parameters())
    print(f"Embedding model parameters: {total_emb_params:,}")
    print(f"Generator model parameters: {total_gen_params:,}")
    print(f"Total parameters: {total_emb_params + total_gen_params:,}")
    print(f"Knowledge base: {store.num_documents} documents, {store.num_chunks} chunks")
    print(f"Embedding dimension: {emb_model.d_model}")
    print(f"Chunk strategy: {store.chunk_strategy} (size={store.chunk_size}, overlap={store.chunk_overlap})")

    # --- 8. Timing breakdown ---
    print("\n" + "-" * 70)
    print("8. Latency Breakdown")
    print("-" * 70)

    query = "How does autograd compute gradients?"

    t0 = time.perf_counter()
    query_emb = encoder.encode([query])
    t1 = time.perf_counter()
    sims = query_emb @ store.embeddings.T
    top_k = sims[0].topk(3)
    t2 = time.perf_counter()
    chunks = [store.chunks[i] for i in top_k.indices]
    prompt = rag.build_prompt(query, [(c, s.item(), {}) for c, s in zip(chunks, top_k.values)])
    t3 = time.perf_counter()
    prompt_ids = gen_tokenizer.encode(prompt).to(device)
    output_ids = gen_model.generate(prompt_ids, max_new_tokens=50, temperature=0.7)
    t4 = time.perf_counter()

    print(f"  Query encoding:   {(t1 - t0) * 1000:7.2f} ms")
    print(f"  Similarity search: {(t2 - t1) * 1000:7.2f} ms")
    print(f"  Prompt construction: {(t3 - t2) * 1000:7.2f} ms")
    print(f"  Generation:       {(t4 - t3) * 1000:7.2f} ms")
    print(f"  Total:            {(t4 - t0) * 1000:7.2f} ms")

    print("\n" + "=" * 70)
    print("RAG pipeline demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
