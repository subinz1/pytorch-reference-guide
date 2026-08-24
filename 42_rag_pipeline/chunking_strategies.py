"""
Document Chunking Strategies for RAG

Different approaches to splitting documents into retrieval-friendly chunks.
The choice of chunking strategy significantly impacts retrieval quality.

Usage:
    python chunking_strategies.py
"""


# ============================================================
# Fixed-Size Chunking
# ============================================================


def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ============================================================
# Sentence-Based Chunking
# ============================================================


def chunk_by_sentences(text: str, sentences_per_chunk: int = 3, overlap: int = 1) -> list[str]:
    """Split text into chunks of N sentences with sentence-level overlap."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    start = 0
    while start < len(sentences):
        end = start + sentences_per_chunk
        chunk = " ".join(sentences[start:end])
        if chunk:
            chunks.append(chunk)
        start += sentences_per_chunk - overlap
    return chunks


# ============================================================
# Token-Based Chunking
# ============================================================


def chunk_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 256,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into chunks based on token count (tokenizer-aware)."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        start += max_tokens - overlap_tokens
    return chunks


# ============================================================
# Semantic Chunking (paragraph-aware)
# ============================================================


def chunk_by_paragraphs(
    text: str, max_chunk_size: int = 1000, merge_short: bool = True
) -> list[str]:
    """Split by paragraph boundaries, merging short paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not merge_short:
        return paragraphs

    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# ============================================================
# Recursive Chunking (hierarchical splitting)
# ============================================================


def chunk_recursive(
    text: str,
    max_chunk_size: int = 500,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text using hierarchical separators."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= max_chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= max_chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(part) > max_chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1 :]
                        chunks.extend(
                            chunk_recursive(part, max_chunk_size, remaining_seps)
                        )
                    else:
                        current = part
            if current:
                chunks.append(current.strip())
            return [c for c in chunks if c]

    # Last resort: hard split
    return chunk_by_characters(text, max_chunk_size, overlap=0)


# ============================================================
# Markdown-Aware Chunking
# ============================================================


def chunk_markdown(text: str, max_chunk_size: int = 800) -> list[str]:
    """Split markdown by headers, keeping section context."""
    import re

    sections = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks = []
    current_header = ""
    current_content = ""

    for section in sections:
        if re.match(r"^#{1,3}\s+", section):
            if current_content.strip():
                chunk = f"{current_header}\n{current_content}".strip()
                if len(chunk) > max_chunk_size:
                    sub_chunks = chunk_by_paragraphs(chunk, max_chunk_size)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk)
            current_header = section.strip()
            current_content = ""
        else:
            current_content += section

    if current_content.strip():
        chunk = f"{current_header}\n{current_content}".strip()
        chunks.append(chunk)

    return [c for c in chunks if c]


# ============================================================
# Demo
# ============================================================


def main():
    sample_text = """PyTorch is an open-source machine learning framework. It was developed by Meta AI
and released in 2016. PyTorch provides two high-level features: tensor computation with GPU
acceleration, and deep neural networks built on a tape-based autograd system.

The framework has become the most popular choice for research. It offers dynamic computation
graphs, which allow flexible model architectures. Researchers can use standard Python control
flow in their models.

torch.compile was introduced in PyTorch 2.0. It uses TorchDynamo to capture Python bytecode
into an FX graph. The graph is then optimized by TorchInductor, which generates fast kernels
for both CPU and GPU. This can provide 1.5-2x speedups on many models.

FSDP2 (Fully Sharded Data Parallel) is the next generation of distributed training in PyTorch.
It shards model parameters, gradients, and optimizer states across data-parallel workers.
This enables training models that are larger than single-GPU memory."""

    print("=" * 60)
    print("Document Chunking Strategies Demo")
    print("=" * 60)

    # Character-based
    print("\n1. Character-based chunking (size=200, overlap=50):")
    chunks = chunk_by_characters(sample_text, chunk_size=200, overlap=50)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")

    # Sentence-based
    print("\n2. Sentence-based chunking (3 sentences per chunk):")
    chunks = chunk_by_sentences(sample_text, sentences_per_chunk=3, overlap=1)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")

    # Paragraph-based
    print("\n3. Paragraph-based chunking (max 400 chars):")
    chunks = chunk_by_paragraphs(sample_text, max_chunk_size=400)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")

    # Recursive
    print("\n4. Recursive chunking (max 300 chars):")
    chunks = chunk_recursive(sample_text, max_chunk_size=300)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")

    # Summary
    print(f"\n{'=' * 60}")
    print("Strategy Selection Guide:")
    print("  - Fixed-size: Simple, predictable, but may split mid-sentence")
    print("  - Sentence-based: Preserves meaning, good default choice")
    print("  - Paragraph-based: Best for well-structured documents")
    print("  - Recursive: Adaptive, handles heterogeneous content")
    print("  - Markdown-aware: Best for documentation and technical content")


if __name__ == "__main__":
    main()
