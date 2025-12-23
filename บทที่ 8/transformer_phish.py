"""
Transformer + Self-Attention for Cybersecurity (Phishing Classification)
- Train a small Transformer Encoder on emails.csv (text,label)
- Evaluate Precision/Recall/F1
- Extract and visualize self-attention weights for a sample message (text explanation + matrix)

Input CSV:
  - text,label   label in {legit, phish} (or 0/1)

Install:
  pip install torch pandas scikit-learn numpy

Run:
  python transformer_phish.py --csv emails.csv --text_col text --label_col label --epochs 6
"""

from __future__ import annotations
import argparse
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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


SPECIALS = ["<pad>", "<unk>", "<cls>", "<sep>"]

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int
    cls_id: int
    sep_id: int

def build_vocab(token_lists: List[List[str]], min_count: int = 2, max_vocab: int = 20000) -> Vocab:
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
        cls_id=stoi["<cls>"],
        sep_id=stoi["<sep>"],
    )

def encode(tokens: List[str], vocab: Vocab) -> List[int]:
    return [vocab.stoi.get(t, vocab.unk_id) for t in tokens]

def add_cls_sep(ids: List[int], vocab: Vocab, max_len: int) -> List[int]:
    # [CLS] ... [SEP]
    ids = [vocab.cls_id] + ids[: max_len - 2] + [vocab.sep_id]
    return ids


# ----------------------------
# Dataset
# ----------------------------

class TxtDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocab, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        toks = tokenize(self.texts[idx])
        ids = add_cls_sep(encode(toks, self.vocab), self.vocab, self.max_len)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, toks  # toks for optional inspection

def collate_pad(batch, pad_id: int):
    xs, ys, toks = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    X = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(xs), max_len), dtype=torch.bool)  # True for real tokens
    for i, x in enumerate(xs):
        X[i, : x.size(0)] = x
        attn_mask[i, : x.size(0)] = True
    Y = torch.stack(list(ys))
    return X, Y, attn_mask, toks


# ----------------------------
# Positional Encoding
# ----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ----------------------------
# Transformer Encoder with attention capture
# ----------------------------

class EncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.last_attn: Optional[torch.Tensor] = None  # (B, heads, T, T)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: True for PAD positions (PyTorch convention)
        attn_out, attn_w = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn = attn_w  # save
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayerWithAttn(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x_ids: torch.Tensor, attn_mask_real: torch.Tensor) -> torch.Tensor:
        """
        x_ids: (B,T)
        attn_mask_real: (B,T) True for real tokens
        """
        x = self.emb(x_ids)                  # (B,T,D)
        x = self.pos(x)
        x = self.dropout(x)

        # PyTorch MultiheadAttention expects key_padding_mask: True for pads
        key_padding_mask = ~attn_mask_real   # (B,T)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        # Use [CLS] representation at position 0
        cls_vec = x[:, 0, :]                 # (B,D)
        logits = self.cls(cls_vec)           # (B,C)
        return logits

    def get_last_attention(self) -> List[Optional[torch.Tensor]]:
        return [layer.last_attn for layer in self.layers]


# ----------------------------
# Train/Eval
# ----------------------------

def train_epoch(model, loader, opt, device: str):
    model.train()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for X, Y, attn_mask, _ in loader:
        X, Y, attn_mask = X.to(device), Y.to(device), attn_mask.to(device)
        opt.zero_grad()
        logits = model(X, attn_mask)
        loss = crit(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item()) * X.size(0)
        n += X.size(0)
    return total_loss / max(1, n)

@torch.no_grad()
def eval_model(model, loader, device: str):
    model.eval()
    ys, ps = [], []
    for X, Y, attn_mask, _ in loader:
        X, attn_mask = X.to(device), attn_mask.to(device)
        logits = model(X, attn_mask)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(Y.numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p

@torch.no_grad()
def predict_with_attention(model, text: str, vocab: Vocab, max_len: int, device: str):
    toks = tokenize(text)
    ids = add_cls_sep(encode(toks, vocab), vocab, max_len)
    X = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    attn_mask = torch.ones((1, X.size(1)), dtype=torch.bool).to(device)
    logits = model(X, attn_mask)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(prob.argmax())

    # Collect attention matrices (per layer)
    attns = model.get_last_attention()  # list of (B, heads, T, T)
    # Map tokens including [CLS]/[SEP]
    shown_tokens = ["<cls>"] + toks[: max_len - 2] + ["<sep>"]
    return pred, prob, shown_tokens, attns


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="emails.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")

    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=20000)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--show_attention_text", default="", help="If provided, print attention matrix stats for this text")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df["y"] = df[args.label_col].apply(map_label)
    df["tokens"] = df[args.text_col].apply(tokenize)

    # Split
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["y"])

    # Build vocab on train
    vocab = build_vocab(train_df["tokens"].tolist(), min_count=args.min_count, max_vocab=args.max_vocab)

    # Datasets / loaders
    train_ds = TxtDataset(train_df[args.text_col].tolist(), train_df["y"].tolist(), vocab, args.max_len)
    test_ds  = TxtDataset(test_df[args.text_col].tolist(),  test_df["y"].tolist(),  vocab, args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_pad(b, vocab.pad_id))
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_pad(b, vocab.pad_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformerClassifier(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_classes=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Device={device} | Train={len(train_ds)} Test={len(test_ds)} | Vocab={len(vocab.itos)}")
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_dl, opt, device)
        y_true, y_pred = eval_model(model, test_dl, device)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        print(f"Epoch {ep}/{args.epochs} | loss={loss:.4f} | Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")

    # Final report
    y_true, y_pred = eval_model(model, test_dl, device)
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Final Test Report ===")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["legit", "phish"], zero_division=0))

    # Optional attention inspection for a provided text
    if args.show_attention_text.strip():
        text = args.show_attention_text.strip()
        pred, prob, shown_tokens, attns = predict_with_attention(model, text, vocab, args.max_len, device)
        print("\n=== Attention Inspection ===")
        print("Text:", text)
        print("Pred:", "phish" if pred == 1 else "legit", "| probs=", prob)

        for li, A in enumerate(attns):
            if A is None:
                continue
            # A shape: (B=1, heads, T, T)
            A0 = A[0].detach().cpu().numpy()          # (heads, T, T)
            head_mean = A0.mean(axis=0)               # (T, T)
            T = head_mean.shape[0]

            # Print a compact attention summary: top attended tokens from CLS row
            cls_row = head_mean[0]                    # attention from <cls> to others
            top_idx = np.argsort(-cls_row)[: min(10, T)]
            print(f"\nLayer {li} (mean over heads):")
            print("Top tokens attended by <cls>:")
            for j in top_idx:
                print(f"  {shown_tokens[j]:>12s}  attn={cls_row[j]:.3f}")

            # If small, print the matrix
            if T <= 20:
                print("\nAttention matrix (mean over heads):")
                np.set_printoptions(precision=2, suppress=True)
                print(head_mean)

    print("\nDone.")


if __name__ == "__main__":
    main()
