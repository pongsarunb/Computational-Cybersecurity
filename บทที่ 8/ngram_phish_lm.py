"""
python ngram_phish_lm.py --csv emails.csv --text_col text --label_col label --n 3 --alpha 1.0 --threshold_mode max_f1
หรือเลือก threshold ให้ คุม False Positive Rate ของ “legit” (เช่น 5%):
python ngram_phish_lm.py --csv emails.csv --text_col text --label_col label --n 3 --threshold_mode target_fpr --target_fpr 0.05
"""
from __future__ import annotations
import re
import math
import argparse
from dataclasses import dataclass
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# ----------------------------
# Tokenization / preprocessing
# ----------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    # Basic English-ish tokenization. Replace for Thai if needed.
    return TOKEN_RE.findall(str(text).lower())

def add_sentence_markers(tokens: List[str], n: int) -> List[str]:
    if n <= 1:
        return tokens + ["</s>"]
    return ["<s>"] * (n - 1) + tokens + ["</s>"]

def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

# ----------------------------
# n-gram LM (Laplace smoothing)
# ----------------------------

@dataclass
class NGramLM:
    n: int
    alpha: float = 1.0
    vocab: Counter = None
    ngram_counts: Counter = None
    context_counts: Counter = None

    def __post_init__(self):
        self.vocab = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    @property
    def V(self) -> int:
        return max(1, len(self.vocab))

    def fit(self, texts: Iterable[str]) -> "NGramLM":
        for text in texts:
            toks = add_sentence_markers(tokenize(text), self.n)
            self.vocab.update(toks)
            for ng in ngrams(toks, self.n):
                self.ngram_counts[ng] += 1
                self.context_counts[ng[:-1]] += 1
        return self

    def prob(self, ng: Tuple[str, ...]) -> float:
        ctx = ng[:-1]
        c_ng = self.ngram_counts.get(ng, 0)
        c_ctx = self.context_counts.get(ctx, 0)
        return (c_ng + self.alpha) / (c_ctx + self.alpha * self.V)

    def perplexity(self, text: str) -> float:
        toks = add_sentence_markers(tokenize(text), self.n)
        ngs = list(ngrams(toks, self.n))
        if not ngs:
            return float("inf")
        lp = 0.0
        for ng in ngs:
            lp += math.log(self.prob(ng))
        return math.exp(-lp / len(ngs))

# ----------------------------
# Detector + threshold selection
# ----------------------------

def choose_threshold_max_f1(y_true: List[int], scores: List[float]) -> float:
    """
    Choose threshold that maximizes F1 on validation set.
    Convention: predict phish if score >= threshold.
    """
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    unique_scores = sorted(set(scores))
    if not unique_scores:
        return float("inf")

    best_thr = unique_scores[0]
    best_f1 = -1.0

    # Evaluate thresholds at each unique score
    for thr in unique_scores:
        y_pred = [1 if s >= thr else 0 for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr

def choose_threshold_target_fpr(y_true: List[int], scores: List[float], target_fpr: float) -> float:
    """
    Choose threshold such that false positive rate on legit samples (y=0) is ~target_fpr.
    """
    legit_scores = sorted([s for s, y in zip(scores, y_true) if y == 0])
    if not legit_scores:
        return float("inf")
    idx = int((1.0 - target_fpr) * (len(legit_scores) - 1))
    return legit_scores[max(0, min(idx, len(legit_scores) - 1))]

# ----------------------------
# Data loading / label mapping
# ----------------------------

def map_label(x) -> int:
    """
    Returns 0=legit, 1=phish
    Supports: 'legit'/'phish', 'ham'/'spam', 0/1, true/false, etc.
    Customize as needed.
    """
    if pd.isna(x):
        raise ValueError("Label contains NaN.")
    s = str(x).strip().lower()
    if s in {"phish", "phishing", "spam", "malicious", "1", "true", "yes"}:
        return 1
    if s in {"legit", "ham", "benign", "0", "false", "no"}:
        return 0
    # Fallback: try numeric
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unrecognized label value: {x!r}")

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--text_col", default="text", help="Name of text column")
    ap.add_argument("--label_col", default="label", help="Name of label column")
    ap.add_argument("--n", type=int, default=3, help="n for n-gram (e.g., 2 or 3)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    ap.add_argument("--val_size", type=float, default=0.2, help="Validation split ratio (from train)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--threshold_mode", choices=["max_f1", "target_fpr"], default="max_f1")
    ap.add_argument("--target_fpr", type=float, default=0.05, help="Used when threshold_mode=target_fpr")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns: {args.text_col!r} and {args.label_col!r}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[args.text_col, args.label_col]].dropna()
    df["y"] = df[args.label_col].apply(map_label)
    df["text"] = df[args.text_col].astype(str)

    # Split: train_val / test (stratified)
    train_val, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["y"]
    )

    # Split train_val into train / val (stratified)
    val_ratio_of_trainval = args.val_size
    train, val = train_test_split(
        train_val, test_size=val_ratio_of_trainval, random_state=args.seed, stratify=train_val["y"]
    )

    # Train LM ONLY on legit training samples
    legit_train_texts = train.loc[train["y"] == 0, "text"].tolist()
    if len(legit_train_texts) < 5:
        raise ValueError(
            f"Too few legit training samples ({len(legit_train_texts)}). "
            "Need more legit examples to train LM."
        )

    lm = NGramLM(n=args.n, alpha=args.alpha).fit(legit_train_texts)

    # Score validation and choose threshold
    val_scores = [lm.perplexity(t) for t in val["text"].tolist()]
    val_y = val["y"].tolist()

    if args.threshold_mode == "max_f1":
        thr = choose_threshold_max_f1(val_y, val_scores)
    else:
        thr = choose_threshold_target_fpr(val_y, val_scores, target_fpr=args.target_fpr)

    # Evaluate on test
    test_scores = [lm.perplexity(t) for t in test["text"].tolist()]
    test_y = test["y"].tolist()
    test_pred = [1 if s >= thr else 0 for s in test_scores]

    p, r, f1, _ = precision_recall_fscore_support(test_y, test_pred, average="binary", zero_division=0)
    cm = confusion_matrix(test_y, test_pred)  # [[TN, FP],[FN, TP]]

    print("\n=== Settings ===")
    print(f"CSV: {args.csv}")
    print(f"text_col={args.text_col} label_col={args.label_col}")
    print(f"n={args.n} alpha={args.alpha}")
    print(f"threshold_mode={args.threshold_mode} threshold={thr:.4f}")
    if args.threshold_mode == "target_fpr":
        print(f"target_fpr={args.target_fpr}")

    print("\n=== Split sizes ===")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"legit_train={len(legit_train_texts)}")

    print("\n=== Test metrics (phish=positive) ===")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print("\nConfusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)

    print("\n=== Classification report ===")
    print(classification_report(test_y, test_pred, target_names=["legit", "phish"], zero_division=0))

    # Show top suspicious examples (highest perplexity)
    test_out = test.copy()
    test_out["perplexity"] = test_scores
    test_out["pred"] = test_pred
    top = test_out.sort_values("perplexity", ascending=False).head(10)
    print("\n=== Top 10 highest-perplexity test samples (most suspicious by LM) ===")
    for _, row in top.iterrows():
        label = "phish" if row["y"] == 1 else "legit"
        pred = "phish" if row["pred"] == 1 else "legit"
        text_preview = row["text"][:160].replace("\n", " ")
        print(f"ppl={row['perplexity']:.2f}  true={label:5s}  pred={pred:5s}  | {text_preview}")

if __name__ == "__main__":
    main()
