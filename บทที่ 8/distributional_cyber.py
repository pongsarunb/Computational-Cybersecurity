"""
Distributional Hypothesis (Cybersecurity Example) - Python Demo

Goal:
Show how "words that occur in similar contexts have similar meanings"
by building:
1) Co-occurrence matrix (window-based)
2) PPMI (Positive PMI) vectors
3) Similarity search (cosine) for cybersecurity terms
4) Optional: simple phishing-vs-legit scoring using similarity to "phish cue" seed words

Input options:
A) Use a CSV like emails.csv with column "text" (label optional)
B) Or provide a plain text file (one document per line)

Install:
  pip install pandas numpy scikit-learn

Run:
  python distributional_cyber.py --csv emails.csv --text_col text
"""

from __future__ import annotations
import argparse
import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())

def build_vocab(docs_tokens: List[List[str]], min_count: int = 2, max_vocab: int = 2000) -> List[str]:
    c = Counter()
    for toks in docs_tokens:
        c.update(toks)
    items = [(w, f) for w, f in c.items() if f >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    vocab = [w for w, _ in items[:max_vocab]]
    return vocab

def build_cooc_matrix(
    docs_tokens: List[List[str]],
    vocab: List[str],
    window: int = 2,
) -> np.ndarray:
    """
    Simple symmetric co-occurrence: count context words within +/- window.
    """
    v_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    cooc = np.zeros((V, V), dtype=np.float64)

    for toks in docs_tokens:
        ids = [v_index.get(w, -1) for w in toks]
        for i, wi in enumerate(ids):
            if wi < 0:
                continue
            left = max(0, i - window)
            right = min(len(ids), i + window + 1)
            for j in range(left, right):
                if j == i:
                    continue
                wj = ids[j]
                if wj < 0:
                    continue
                cooc[wi, wj] += 1.0

    return cooc

def ppmi_matrix(cooc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute PPMI from co-occurrence matrix.
    PMI(i,j) = log( P(i,j) / (P(i)P(j)) )
    PPMI = max(PMI, 0)
    """
    total = cooc.sum()
    if total <= 0:
        raise ValueError("Co-occurrence matrix is empty.")

    p_ij = cooc / total
    p_i = p_ij.sum(axis=1, keepdims=True)
    p_j = p_ij.sum(axis=0, keepdims=True)

    # Avoid divide-by-zero
    denom = (p_i @ p_j) + eps
    pmi = np.log((p_ij + eps) / denom)
    ppmi = np.maximum(pmi, 0.0)
    return ppmi

def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms

def nearest_neighbors(word: str, vocab: List[str], Xn: np.ndarray, topn: int = 10) -> List[Tuple[str, float]]:
    v_index = {w: i for i, w in enumerate(vocab)}
    if word not in v_index:
        return []
    i = v_index[word]
    sims = (Xn @ Xn[i].reshape(-1, 1)).ravel()  # cosine since normalized
    order = np.argsort(-sims)
    out = []
    for j in order[: topn + 1]:
        if j == i:
            continue
        out.append((vocab[j], float(sims[j])))
        if len(out) >= topn:
            break
    return out

def seed_similarity_score(
    doc_tokens: List[str],
    seed_words: List[str],
    vocab: List[str],
    Xn: np.ndarray,
) -> float:
    """
    Score a document by how semantically close its words are to a set of seed words.
    This is a simple demonstration (NOT a production detector).
    """
    v_index = {w: i for i, w in enumerate(vocab)}
    seed_ids = [v_index[w] for w in seed_words if w in v_index]
    if not seed_ids:
        return 0.0

    # Compute max similarity of each token to any seed, then average
    token_ids = [v_index[w] for w in doc_tokens if w in v_index]
    if not token_ids:
        return 0.0

    seed_vecs = Xn[seed_ids]         # (S, D)
    tok_vecs = Xn[token_ids]         # (T, D)
    sims = tok_vecs @ seed_vecs.T    # (T, S)
    best = sims.max(axis=1)          # (T,)
    return float(best.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="CSV path (uses --text_col)")
    ap.add_argument("--text_col", default="text", help="Text column name in CSV")
    ap.add_argument("--txt", default="", help="Plain text file (one doc per line)")
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=2000)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    if not args.csv and not args.txt:
        raise SystemExit("Provide --csv emails.csv OR --txt corpus.txt")

    # Load documents
    docs: List[str] = []
    if args.csv:
        df = pd.read_csv(args.csv).dropna(subset=[args.text_col])
        docs = df[args.text_col].astype(str).tolist()
    else:
        with open(args.txt, "r", encoding="utf-8") as f:
            docs = [line.strip() for line in f if line.strip()]

    docs_tokens = [tokenize(d) for d in docs]

    # 1) Build vocab
    vocab = build_vocab(docs_tokens, min_count=args.min_count, max_vocab=args.max_vocab)
    print(f"Vocab size={len(vocab)} (min_count={args.min_count})")

    # 2) Co-occurrence
    cooc = build_cooc_matrix(docs_tokens, vocab, window=args.window)
    print(f"Co-occurrence matrix shape={cooc.shape}, total_counts={cooc.sum():.0f}")

    # 3) PPMI vectors
    X = ppmi_matrix(cooc)
    Xn = normalize_rows(X)

    # 4) Nearest neighbors: cybersecurity probes
    probes = ["verify", "confirm", "account", "password", "urgent", "click", "login", "invoice", "payroll"]
    print("\n=== Distributional Similarity (Nearest Neighbors from PPMI space) ===")
    for w in probes:
        nn = nearest_neighbors(w, vocab, Xn, topn=args.topn)
        if not nn:
            print(f"\n'{w}' not in vocab.")
            continue
        print(f"\nNeighbors for '{w}':")
        for ww, s in nn:
            print(f"  {ww:20s}  cos={s:.3f}")

    # 5) Optional mini-demo: semantic scoring vs phishing cue seeds
    seed_phish = ["verify", "urgent", "password", "login", "click", "account"]
    if any(w in vocab for w in seed_phish):
        print("\n=== Simple semantic scoring demo (Distributional Hypothesis in action) ===")
        for i in range(min(8, len(docs_tokens))):
            score = seed_similarity_score(docs_tokens[i], seed_phish, vocab, Xn)
            preview = " ".join(docs_tokens[i][:20])
            print(f"doc#{i:02d} score={score:.3f} | {preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
