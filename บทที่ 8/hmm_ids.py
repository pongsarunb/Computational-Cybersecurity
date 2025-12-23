"""
pip install hmmlearn scikit-learn pandas numpy
python hmm_ids.py --csv systemcalls.csv --seq_col seq --label_col label --n_states 6 --n_iter 100
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# ---- Optional dependency: hmmlearn (recommended) ----
# pip install hmmlearn
try:
    from hmmlearn.hmm import MultinomialHMM
except ImportError as e:
    raise SystemExit(
        "This script requires 'hmmlearn'. Install it with:\n"
        "  pip install hmmlearn\n"
    ) from e


def parse_sequence(seq_str: str) -> list[int]:
    # Expect: "1 2 3 4" or "1,2,3,4" (we support both)
    s = str(seq_str).strip().replace(",", " ")
    toks = [t for t in s.split() if t]
    return [int(t) for t in toks]


def build_X_lengths(seqs: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    """
    hmmlearn expects:
    - X: shape (N, 1) of integer observations
    - lengths: list of sequence lengths
    """
    lengths = [len(s) for s in seqs]
    X = np.array([o for s in seqs for o in s], dtype=int).reshape(-1, 1)
    return X, lengths


def fit_hmm_normal_only(
    normal_seqs: list[list[int]],
    n_states: int = 6,
    n_iter: int = 100,
    random_state: int = 42,
) -> MultinomialHMM:
    """
    Train a Multinomial HMM on normal sequences only.
    Observations are discrete event IDs (integers).
    """
    X_train, lengths = build_X_lengths(normal_seqs)
    n_symbols = int(X_train.max()) + 1  # assumes IDs start at 0; if start at 1, still OK

    model = MultinomialHMM(
        n_components=n_states,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    # hmmlearn needs n_features to be set for MultinomialHMM in some versions
    model.n_features = n_symbols

    model.fit(X_train, lengths)
    return model


def score_sequences(model: MultinomialHMM, seqs: list[list[int]]) -> np.ndarray:
    """
    Return per-sequence log-likelihood scores (higher is more normal).
    """
    scores = []
    for s in seqs:
        X, lengths = build_X_lengths([s])
        # If the test sequence contains a symbol unseen in training, scoring may fail.
        # We'll map unseen symbols to a known "UNK" symbol if needed. Here we clamp.
        max_symbol = model.n_features - 1
        X = np.clip(X, 0, max_symbol)

        ll = model.score(X, lengths)  # log P(X | model)
        scores.append(ll)
    return np.array(scores, dtype=float)


def choose_threshold_max_f1(y_true: np.ndarray, normality_scores: np.ndarray) -> float:
    """
    We flag as suspicious when score <= threshold (low likelihood).
    Choose threshold that maximizes F1 on validation set.
    """
    unique_scores = np.unique(normality_scores)
    best_thr = unique_scores[0]
    best_f1 = -1.0

    for thr in unique_scores:
        y_pred = (normality_scores <= thr).astype(int)  # 1=suspicious
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return float(best_thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with columns: seq,label")
    ap.add_argument("--seq_col", default="seq", help="Sequence column name")
    ap.add_argument("--label_col", default="label", help="Label column name (0=normal,1=attack)")
    ap.add_argument("--n_states", type=int, default=6, help="Number of hidden states")
    ap.add_argument("--n_iter", type=int, default=100, help="EM iterations")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    ap.add_argument("--val_size", type=float, default=0.2, help="Validation split ratio (from train)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.seq_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns {args.seq_col!r} and {args.label_col!r}. "
            f"Found: {list(df.columns)}"
        )

    df = df[[args.seq_col, args.label_col]].dropna()
    df["y"] = df[args.label_col].astype(int)
    df["seq_list"] = df[args.seq_col].apply(parse_sequence)

    # Split train_val/test (stratified)
    train_val, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["y"]
    )
    # Split train/val (stratified)
    train, val = train_test_split(
        train_val, test_size=args.val_size, random_state=args.seed, stratify=train_val["y"]
    )

    # Train only on normal sequences
    normal_train = train.loc[train["y"] == 0, "seq_list"].tolist()
    if len(normal_train) < 10:
        raise ValueError(
            f"Too few normal sequences for training: {len(normal_train)}. "
            "Provide more normal data."
        )

    model = fit_hmm_normal_only(
        normal_train,
        n_states=args.n_states,
        n_iter=args.n_iter,
        random_state=args.seed,
    )

    # Validation: choose threshold
    val_seqs = val["seq_list"].tolist()
    val_y = val["y"].to_numpy()
    val_scores = score_sequences(model, val_seqs)

    thr = choose_threshold_max_f1(val_y, val_scores)

    # Test evaluation
    test_seqs = test["seq_list"].tolist()
    test_y = test["y"].to_numpy()
    test_scores = score_sequences(model, test_seqs)

    test_pred = (test_scores <= thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(test_y, test_pred, average="binary", zero_division=0)
    cm = confusion_matrix(test_y, test_pred)

    print("\n=== HMM Sequence Anomaly Detection ===")
    print(f"csv={args.csv}")
    print(f"n_states={args.n_states} n_iter={args.n_iter}")
    print(f"threshold (log-likelihood) = {thr:.4f}  (flag if score <= threshold)")
    print("\n=== Split sizes ===")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(f"normal_train={len(normal_train)}")

    print("\n=== Test metrics (attack=positive) ===")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print("\nConfusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)

    print("\n=== Classification report ===")
    print(classification_report(test_y, test_pred, target_names=["normal", "attack"], zero_division=0))

    # Show most suspicious sequences (lowest scores)
    out = test.copy()
    out["score"] = test_scores
    out["pred"] = test_pred
    out = out.sort_values("score", ascending=True).head(10)

    print("\n=== Top 10 most suspicious test sequences (lowest likelihood) ===")
    for _, row in out.iterrows():
        true_lbl = "attack" if row["y"] == 1 else "normal"
        pred_lbl = "attack" if row["pred"] == 1 else "normal"
        seq_preview = " ".join(map(str, row["seq_list"][:40]))
        if len(row["seq_list"]) > 40:
            seq_preview += " ..."
        print(f"score={row['score']:.2f}  true={true_lbl:6s}  pred={pred_lbl:6s}  | {seq_preview}")


if __name__ == "__main__":
    main()
