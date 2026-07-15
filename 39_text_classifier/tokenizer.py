"""
Module 39: Tokenizer — Character-Level and Word-Level Tokenization from Scratch

Runnable on CPU:
    python tokenizer.py
"""

import json
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]


# ---------------------------------------------------------------------------
# Character-Level Tokenizer
# ---------------------------------------------------------------------------
class CharacterTokenizer:
    """Tokenizes text at the character level.

    Every unique character gets its own token ID.  Special tokens are reserved
    at the start of the vocabulary.
    """

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    # -- build vocabulary from data ------------------------------------------

    def build_vocab(self, texts: list[str]) -> None:
        """Scan *texts* and assign an ID to every unique character."""
        chars: set[str] = set()
        for text in texts:
            chars.update(text.lower())
        for ch in sorted(chars):
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

    # -- encode / decode -----------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        return list(text.lower())

    def encode(self, text: str, max_length: int = 128) -> list[int]:
        """text -> [CLS, char_ids..., PAD...] of exactly *max_length*."""
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        ids = [self.cls_id] + ids
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self.id_to_token.get(i, UNK_TOKEN)
            for i in ids
            if self.id_to_token.get(i, UNK_TOKEN) not in SPECIAL_TOKENS
        ]
        return "".join(tokens)

    def encode_batch(self, texts: list[str], max_length: int = 128) -> list[list[int]]:
        return [self.encode(t, max_length) for t in texts]

    # -- save / load ----------------------------------------------------------

    def save(self, path: str) -> None:
        data = {"type": "character", "token_to_id": self.token_to_id}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls.__new__(cls)
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        return tok


# ---------------------------------------------------------------------------
# Word-Level Tokenizer
# ---------------------------------------------------------------------------
class WordTokenizer:
    """Tokenizes text at the word level with frequency-based vocabulary."""

    def __init__(self, max_vocab_size: int = 10_000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    _WORD_RE = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str) -> list[str]:
        return self._WORD_RE.findall(text.lower())

    def build_vocab(self, texts: list[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        budget = self.max_vocab_size - len(SPECIAL_TOKENS)
        for word, _ in counter.most_common(budget):
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word

    def encode(self, text: str, max_length: int = 128) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        ids = [self.cls_id] + ids
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self.id_to_token.get(i, UNK_TOKEN)
            for i in ids
            if self.id_to_token.get(i, UNK_TOKEN) not in SPECIAL_TOKENS
        ]
        return " ".join(tokens)

    def encode_batch(self, texts: list[str], max_length: int = 128) -> list[list[int]]:
        return [self.encode(t, max_length) for t in texts]

    def save(self, path: str) -> None:
        data = {
            "type": "word",
            "max_vocab_size": self.max_vocab_size,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls.__new__(cls)
        tok.max_vocab_size = data["max_vocab_size"]
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        return tok


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    sample_texts = [
        "This movie was absolutely fantastic!",
        "Terrible acting and awful script.",
        "The film was okay, nothing special.",
        "I loved every moment of this masterpiece.",
        "Waste of time. Horrible experience.",
        "A decent movie with some good moments.",
    ]

    # --- Character tokenizer -------------------------------------------------
    print("=" * 70)
    print("CHARACTER-LEVEL TOKENIZER")
    print("=" * 70)

    char_tok = CharacterTokenizer()
    char_tok.build_vocab(sample_texts)
    print(f"Vocabulary size: {char_tok.vocab_size}")
    print(f"Sample vocab: {dict(list(char_tok.token_to_id.items())[:15])}...")
    print()

    for text in sample_texts[:3]:
        ids = char_tok.encode(text, max_length=50)
        decoded = char_tok.decode(ids)
        non_pad = [i for i in ids if i != char_tok.pad_id]
        print(f"Text:    {text}")
        print(f"Tokens:  {len(non_pad)} (excl. padding)")
        print(f"IDs:     {ids[:20]}...")
        print(f"Decoded: {decoded}")
        print()

    # --- Word tokenizer ------------------------------------------------------
    print("=" * 70)
    print("WORD-LEVEL TOKENIZER")
    print("=" * 70)

    word_tok = WordTokenizer(max_vocab_size=200)
    word_tok.build_vocab(sample_texts)
    print(f"Vocabulary size: {word_tok.vocab_size}")
    print(f"Full vocab: {dict(list(word_tok.token_to_id.items()))}")
    print()

    for text in sample_texts[:3]:
        tokens = word_tok.tokenize(text)
        ids = word_tok.encode(text, max_length=20)
        decoded = word_tok.decode(ids)
        non_pad = [i for i in ids if i != word_tok.pad_id]
        print(f"Text:    {text}")
        print(f"Tokens:  {tokens}")
        print(f"IDs:     {ids}")
        print(f"Decoded: {decoded}")
        print()

    # --- Batch encoding -------------------------------------------------------
    print("=" * 70)
    print("BATCH ENCODING")
    print("=" * 70)

    batch_ids = word_tok.encode_batch(sample_texts[:3], max_length=15)
    for text, ids in zip(sample_texts[:3], batch_ids):
        print(f"{text:50s} -> {ids}")
    print()

    # --- Save / load round-trip -----------------------------------------------
    print("=" * 70)
    print("SAVE / LOAD ROUND-TRIP")
    print("=" * 70)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        word_tok.save(f.name)
        loaded = WordTokenizer.load(f.name)
        print(f"Original vocab size: {word_tok.vocab_size}")
        print(f"Loaded vocab size:   {loaded.vocab_size}")
        test = "This movie was fantastic!"
        assert word_tok.encode(test) == loaded.encode(test)
        print(f"Encode match: True")

    print("\nDone!")


if __name__ == "__main__":
    main()
