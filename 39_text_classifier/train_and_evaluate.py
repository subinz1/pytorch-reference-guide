"""
Module 39: Train & Evaluate — End-to-End Text Classification Pipeline

Generates a synthetic sentiment dataset, trains a TransformerTextClassifier,
evaluates with per-class metrics and confusion matrix, runs inference, and
benchmarks torch.compile.

Runnable on CPU:
    python train_and_evaluate.py
"""

import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from text_classifier import TransformerTextClassifier
from tokenizer import WordTokenizer


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------
_POSITIVE_TEMPLATES = [
    "This was absolutely {adj}",
    "I loved {adj} every moment",
    "A {adj} experience overall",
    "Really {adj} and well done",
    "{adj} movie I highly recommend",
    "What a {adj} film this was",
    "The acting was {adj} and the story was {adj}",
    "One of the most {adj} films I have seen",
    "Truly {adj} from start to finish",
    "A {adj} masterpiece of cinema",
]
_POSITIVE_ADJS = [
    "great", "fantastic", "wonderful", "amazing", "excellent", "brilliant",
    "outstanding", "superb", "incredible", "marvelous", "delightful",
    "perfect", "magnificent", "spectacular", "phenomenal",
]

_NEGATIVE_TEMPLATES = [
    "This was absolutely {adj}",
    "I found it {adj} and boring",
    "A {adj} waste of time",
    "Really {adj} do not recommend",
    "{adj} movie with {adj} acting",
    "What a {adj} film this was",
    "The acting was {adj} and the plot was {adj}",
    "One of the most {adj} films ever made",
    "Truly {adj} from start to finish",
    "A {adj} disaster of a movie",
]
_NEGATIVE_ADJS = [
    "terrible", "horrible", "awful", "dreadful", "atrocious", "pathetic",
    "abysmal", "lousy", "appalling", "miserable", "dismal",
    "horrendous", "ghastly", "wretched", "deplorable",
]

_NEUTRAL_TEMPLATES = [
    "The movie was {adj}",
    "It was {adj} nothing more",
    "A {adj} film overall",
    "The film was {adj} I suppose",
    "Not bad not great just {adj}",
    "I thought it was {adj} at best",
    "An {adj} experience all around",
    "{adj} movie with {adj} moments",
    "Pretty {adj} if you ask me",
    "The movie felt {adj} throughout",
]
_NEUTRAL_ADJS = [
    "okay", "average", "mediocre", "decent", "passable", "ordinary",
    "unremarkable", "fair", "moderate", "standard", "typical",
    "alright", "so-so", "tolerable", "bland",
]

LABEL_NAMES = ["positive", "negative", "neutral"]


def generate_synthetic_data(
    n_per_class: int = 200,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []

    for _ in range(n_per_class):
        tmpl = rng.choice(_POSITIVE_TEMPLATES)
        text = tmpl.replace("{adj}", rng.choice(_POSITIVE_ADJS), 1)
        text = text.replace("{adj}", rng.choice(_POSITIVE_ADJS))
        texts.append(text)
        labels.append(0)

    for _ in range(n_per_class):
        tmpl = rng.choice(_NEGATIVE_TEMPLATES)
        text = tmpl.replace("{adj}", rng.choice(_NEGATIVE_ADJS), 1)
        text = text.replace("{adj}", rng.choice(_NEGATIVE_ADJS))
        texts.append(text)
        labels.append(1)

    for _ in range(n_per_class):
        tmpl = rng.choice(_NEUTRAL_TEMPLATES)
        text = tmpl.replace("{adj}", rng.choice(_NEUTRAL_ADJS), 1)
        text = text.replace("{adj}", rng.choice(_NEUTRAL_ADJS))
        texts.append(text)
        labels.append(2)

    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: WordTokenizer,
        max_length: int = 64,
    ):
        self.encodings = [tokenizer.encode(t, max_length) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.encodings[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.stack([item["label"] for item in batch])
    attention_mask = (input_ids != 0).float()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    max_grad_norm: float = 1.0,
    device: str = "cpu",
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cpu")):
            logits = model(ids, mask)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    preds: list[int],
    labels: list[int],
    num_classes: int,
    label_names: list[str],
) -> dict:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, l in zip(preds, labels):
        confusion[l][p] += 1

    per_class = {}
    f1_scores = []
    for c in range(num_classes):
        tp = confusion[c][c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label_names[c]] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": confusion[c].sum().item(),
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return {"confusion": confusion, "per_class": per_class, "macro_f1": macro_f1}


def print_confusion_matrix(confusion: torch.Tensor, label_names: list[str]) -> None:
    n = len(label_names)
    header = "          " + "".join(f"{name:>12s}" for name in label_names)
    print(header)
    print("          " + "-" * (12 * n))
    for i in range(n):
        row = f"{label_names[i]:>9s} |"
        for j in range(n):
            row += f"{confusion[i][j].item():>12d}"
        print(row)


def print_classification_report(metrics: dict) -> None:
    print(f"\n{'':>12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 55)
    for name, m in metrics["per_class"].items():
        print(f"{name:>12s} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")
    print("-" * 55)
    total_support = sum(m["support"] for m in metrics["per_class"].values())
    print(f"{'macro avg':>12s} {'':>10s} {'':>10s} {metrics['macro_f1']:>10.4f} {total_support:>10d}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(
    text: str,
    model: nn.Module,
    tokenizer: WordTokenizer,
    label_names: list[str],
    max_length: int = 64,
) -> dict:
    model.eval()
    ids = torch.tensor([tokenizer.encode(text, max_length)])
    mask = (ids != 0).float()
    with torch.no_grad():
        logits = model(ids, mask)
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    return {
        "text": text,
        "label": label_names[pred.item()],
        "confidence": conf.item(),
        "probs": {n: probs[0][i].item() for i, n in enumerate(label_names)},
    }


def predict_batch(
    texts: list[str],
    model: nn.Module,
    tokenizer: WordTokenizer,
    label_names: list[str],
    max_length: int = 64,
) -> list[dict]:
    model.eval()
    encoded = [tokenizer.encode(t, max_length) for t in texts]
    ids = nn.utils.rnn.pad_sequence(
        [torch.tensor(e) for e in encoded], batch_first=True, padding_value=0,
    )
    mask = (ids != 0).float()
    with torch.no_grad():
        logits = model(ids, mask)
    probs = torch.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    return [
        {"text": t, "label": label_names[p.item()], "confidence": c.item()}
        for t, p, c in zip(texts, preds, confs)
    ]


# ---------------------------------------------------------------------------
# Benchmark torch.compile
# ---------------------------------------------------------------------------
def benchmark_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids, mask)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(input_ids, mask)
        elapsed = time.perf_counter() - start
    return elapsed / n_runs * 1000  # ms


# ---------------------------------------------------------------------------
# Main — end-to-end pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = "cpu"

    # ---- 1. Generate data ----------------------------------------------------
    print("=" * 70)
    print("1. GENERATING SYNTHETIC DATA")
    print("=" * 70)
    texts, labels = generate_synthetic_data(n_per_class=200, seed=42)
    print(f"Total samples: {len(texts)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {labels.count(i)}")
    print(f"\nSample: '{texts[0]}' -> {LABEL_NAMES[labels[0]]}")
    print(f"Sample: '{texts[1]}' -> {LABEL_NAMES[labels[1]]}")
    print(f"Sample: '{texts[2]}' -> {LABEL_NAMES[labels[2]]}")

    # ---- 2. Tokenizer --------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. BUILDING TOKENIZER")
    print("=" * 70)
    tokenizer = WordTokenizer(max_vocab_size=500)
    tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # ---- 3. Dataset & DataLoader ---------------------------------------------
    print("\n" + "=" * 70)
    print("3. CREATING DATASETS")
    print("=" * 70)
    max_length = 32
    dataset = TextDataset(texts, labels, tokenizer, max_length)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=64, collate_fn=collate_fn)

    # ---- 4. Model ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. BUILDING MODEL")
    print("=" * 70)
    model = TransformerTextClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=len(LABEL_NAMES),
        max_len=max_length,
        dropout=0.1,
        padding_idx=tokenizer.pad_id,
        pool="cls",
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # ---- 5. Training ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. TRAINING")
    print("=" * 70)
    num_epochs = 15
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_loss = float("inf")
    patience = 4
    patience_counter = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device=device,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device=device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val loss: {best_val_loss:.4f})")

    # ---- 6. Evaluation -------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. EVALUATION ON TEST SET")
    print("=" * 70)
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device=device,
    )
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

    metrics = compute_metrics(test_preds, test_labels, len(LABEL_NAMES), LABEL_NAMES)

    print("\nConfusion Matrix:")
    print_confusion_matrix(metrics["confusion"], LABEL_NAMES)

    print("\nClassification Report:")
    print_classification_report(metrics)

    # ---- 7. Inference --------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. INFERENCE DEMO")
    print("=" * 70)

    test_texts = [
        "This was absolutely wonderful and amazing",
        "Terrible movie with horrible acting",
        "The film was okay nothing special",
        "I loved every moment of this masterpiece",
        "Dreadful experience would not recommend",
        "A decent movie with some good parts",
    ]

    print("Single predictions:")
    for text in test_texts:
        result = predict(text, model, tokenizer, LABEL_NAMES, max_length)
        probs_str = ", ".join(f"{k}: {v:.3f}" for k, v in result["probs"].items())
        print(f"  '{text}'")
        print(f"    -> {result['label']} (conf: {result['confidence']:.3f})  [{probs_str}]")

    print("\nBatch predictions:")
    batch_results = predict_batch(test_texts, model, tokenizer, LABEL_NAMES, max_length)
    for r in batch_results:
        print(f"  '{r['text']}' -> {r['label']} ({r['confidence']:.3f})")

    # ---- 8. torch.compile benchmark ------------------------------------------
    print("\n" + "=" * 70)
    print("8. torch.compile BENCHMARK")
    print("=" * 70)

    dummy_ids = torch.randint(1, tokenizer.vocab_size, (1, max_length))
    dummy_ids[:, 0] = tokenizer.cls_id
    dummy_mask = torch.ones(1, max_length)

    eager_ms = benchmark_model(model, dummy_ids, dummy_mask, n_warmup=5, n_runs=50)
    print(f"Eager inference:    {eager_ms:.2f} ms")

    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        compiled_ms = benchmark_model(compiled_model, dummy_ids, dummy_mask, n_warmup=10, n_runs=50)
        print(f"Compiled inference: {compiled_ms:.2f} ms")
        print(f"Speedup:            {eager_ms / compiled_ms:.2f}x")
    except Exception as e:
        print(f"torch.compile benchmark skipped: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
