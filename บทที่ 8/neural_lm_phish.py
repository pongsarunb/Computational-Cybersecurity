"""
Neural Language Model (Neural LM) for Cybersecurity:
Use-case: Phishing / anomaly detection by "language surprise" (perplexity)

Concept:
- Train a neural LM (LSTM) on LEGIT messages only (normal language baseline).
- Compute perplexity of each message under the legit-LM.
- If perplexity is high -> text is linguistically "unusual" vs legit -> flag as suspicious.
- Evaluate Precision/Recall/F1 on labeled CSV.

Input CSV (matches what you created):
- text,label   where label in {legit, phish} (or 0/1)

Install:
  pip install torch pandas scikit-learn

Run:
  python neural_lm_phish.py --csv emails.csv --text_col text --label_col label
"""

from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report


# ----------------------------
# Tokenization + vocab
# ----------------------------

TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?|[^\s]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())

def map_label(x) -> int:
    s = str(x).strip().lower()
    if s in {"phish", "phishing", "spam", "malicious", "1", "true", "yes"}:
        return 1
    if s in {"legit", "ham", "benign", "0", "false", "no"}:
        return 0
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unrecognized label: {x!r}")


SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int

def build_vocab(token_lists: List[List[str]], min_count: int = 2, max_vocab: int = 30000) -> Vocab:
    from collections import Counter
    c = Counter()
    for toks in token_lists:
        c.update(toks)

    words = [w for w, f in c.items() if f >= min_count]
    words.sort(key=lambda w: c[w], reverse=True)
    words = words[: max(0, max_vocab - len(SPECIALS))]

    itos = SPECIALS + words
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        unk_id=stoi["<unk>"],
        bos_id=stoi["<bos>"],
        eos_id=stoi["<eos>"],
    )

def encode(tokens: List[str], vocab: Vocab) -> List[int]:
    return [vocab.stoi.get(t, vocab.unk_id) for t in tokens]

def add_markers(ids: List[int], vocab: Vocab) -> List[int]:
    return [vocab.bos_id] + ids + [vocab.eos_id]


# ----------------------------
# Dataset for LM
# ----------------------------

class LMDataset(Dataset):
    """
    Each item returns (x, y) where:
    x = token ids [bos, w1, w2, ..., wn]
    y = token ids [w1, w2, ..., wn, eos]
    """
    def __init__(self, sequences: List[List[int]]):
        self.seqs = sequences

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        s = self.seqs[idx]
        x = torch.tensor(s[:-1], dtype=torch.long)
        y = torch.tensor(s[1:], dtype=torch.long)
        return x, y

def collate_pad(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    X = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    Y = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, : x.size(0)] = x
        Y[i, : y.size(0)] = y
    return X, Y


# ----------------------------
# Neural LM: LSTM
# ----------------------------

class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hid_dim: int = 256, num_layers: int = 2, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hid_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        e = self.emb(x)               # (B, T, E)
        h, _ = self.lstm(e)           # (B, T, H)
        logits = self.proj(h)         # (B, T, V)
        return logits


# ----------------------------
# Training + scoring
# ----------------------------

def train_lm(model, loader, optimizer, pad_id: int, device: str, epochs: int = 5):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            logits = model(X)  # (B,T,V)

            loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                mask = (Y != pad_id)
                total_tokens += int(mask.sum().item())
                total_loss += float(loss.item()) * int(mask.sum().item())

        ppl = math.exp(total_loss / max(1, total_tokens))
        print(f"Epoch {ep}/{epochs} | token_loss={total_loss/max(1,total_tokens):.4f} | ppl={ppl:.2f}")

@torch.no_grad()
def perplexity_of_text(model, text: str, vocab: Vocab, device: str) -> float:
    model.eval()
    toks = tokenize(text)
    ids = add_markers(encode(toks, vocab), vocab)
    if len(ids) < 2:
        return float("inf")

    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)   # (1,T)

    logits = model(x)  # (1,T,V)
    log_probs = torch.log_softmax(logits, dim=-1)
    # gather log P(y_t | x_<=t)
    lp = log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # (1,T)
    nll = -lp.mean().item()
    return float(math.exp(nll))

def choose_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    # predict phish if score >= threshold (higher perplexity = more suspicious)
    uniq = np.unique(scores)
    best_thr, best_f1 = float(uniq[0]), -1.0
    for thr in uniq:
        pred = (scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="emails.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=20000)

    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hid_dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df["y"] = df[args.label_col].apply(map_label)
    df["tokens"] = df[args.text_col].apply(tokenize)

    # Split train_val/test (stratified)
    train_val, test = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["y"])
    # Split train/val
    train, val = train_test_split(train_val, test_size=args.val_size, random_state=args.seed, stratify=train_val["y"])

    # Train LM on LEGIT only
    legit_train = train[train["y"] == 0].copy()
    if len(legit_train) < 20:
        raise ValueError(f"Need more legit samples to train LM. Found legit_train={len(legit_train)}")

    vocab = build_vocab(legit_train["tokens"].tolist(), min_count=args.min_count, max_vocab=args.max_vocab)

    train_seqs = [add_markers(encode(toks, vocab), vocab) for toks in legit_train["tokens"].tolist()]
    lm_ds = LMDataset(train_seqs)
    lm_dl = DataLoader(
        lm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, vocab.pad_id),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMLM(
        vocab_size=len(vocab.itos),
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device={device} | Vocab={len(vocab.itos)} | Legit train sequences={len(train_seqs)}")
    train_lm(model, lm_dl, optimizer, vocab.pad_id, device, epochs=args.epochs)

    # Score validation set -> pick threshold by max F1
    val_scores = np.array([perplexity_of_text(model, t, vocab, device) for t in val[args.text_col].tolist()], dtype=float)
    val_y = val["y"].to_numpy(dtype=int)

    thr = choose_threshold_max_f1(val_y, val_scores)
    print(f"\nChosen threshold (max F1 on val) = {thr:.3f}  (predict phish if ppl >= thr)")

    # Evaluate on test
    test_scores = np.array([perplexity_of_text(model, t, vocab, device) for t in test[args.text_col].tolist()], dtype=float)
    test_y = test["y"].to_numpy(dtype=int)
    test_pred = (test_scores >= thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(test_y, test_pred, average="binary", zero_division=0)
    cm = confusion_matrix(test_y, test_pred)

    print("\n=== Neural LM (LSTM) Perplexity Detector: Test Metrics (phish=positive) ===")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print("\nConfusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(test_y, test_pred, target_names=["legit", "phish"], zero_division=0))

    # Show most suspicious examples
    out = test.copy()
    out["ppl"] = test_scores
    out["pred"] = test_pred
    out = out.sort_values("ppl", ascending=False).head(10)

    print("\n=== Top 10 highest-perplexity test messages (most suspicious by LM) ===")
    for _, row in out.iterrows():
        true_lbl = "phish" if row["y"] == 1 else "legit"
        pred_lbl = "phish" if row["pred"] == 1 else "legit"
        preview = str(row[args.text_col]).replace("\n", " ")[:160]
        print(f"ppl={row['ppl']:.2f}  true={true_lbl:5s}  pred={pred_lbl:5s}  | {preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
