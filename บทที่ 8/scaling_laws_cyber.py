"""
Scaling Laws (Cybersecurity Example) — Python Experiment Script
Goal:
- Empirically demonstrate a "scaling-law-like" relationship between
  (1) data size (N), (2) model capacity, and (3) performance (loss/F1)
  on a cybersecurity NLP task (phishing detection).

We do:
1) Load emails.csv (text,label: legit/phish)
2) Build TF-IDF features
3) Train models across multiple training set sizes N and capacities
   - Logistic Regression with different C (capacity proxy)
   - (Optional) MLPClassifier with different hidden sizes (capacity proxy)
4) Measure:
   - Log loss (cross-entropy)  [good for scaling laws]
   - Precision/Recall/F1
5) Fit a simple power-law-like curve for loss vs N:
   L(N) ≈ c + a * N^(-b)
   We estimate (a,b,c) by scanning c and doing linear regression on log(L-c) vs log(N).

Install:
  pip install pandas numpy scikit-learn matplotlib

Run:
  python scaling_laws_cyber.py --csv emails.csv --text_col text --label_col label --use_mlp 0

Outputs:
- results.csv
- scaling_plot.png (optional)
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

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

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))

def fit_power_law_loss(N: np.ndarray, L: np.ndarray, c_grid: np.ndarray) -> Dict[str, float]:
    """
    Fit L(N) ≈ c + a*N^(-b) by scanning c:
      y = log(L - c) = log(a) - b*log(N)
    Choose c giving best R^2 on transformed space.

    Returns dict with {a,b,c,r2}.
    """
    N = N.astype(float)
    L = L.astype(float)
    x = safe_log(N)

    best = {"a": float("nan"), "b": float("nan"), "c": float("nan"), "r2": -1.0}

    for c in c_grid:
        y_raw = L - c
        if np.any(y_raw <= 0):
            continue
        y = safe_log(y_raw)

        # Linear regression closed-form: y = alpha + beta*x
        # beta = cov(x,y)/var(x), alpha = mean(y) - beta*mean(x)
        vx = np.var(x)
        if vx <= 0:
            continue
        beta = np.cov(x, y, bias=True)[0, 1] / vx
        alpha = float(np.mean(y) - beta * np.mean(x))

        # Predict and compute R^2
        y_hat = alpha + beta * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        # Convert: y = log(a) - b log(N) => alpha=log(a), beta=-b
        a = float(math.exp(alpha))
        b = float(-beta)

        if r2 > best["r2"]:
            best = {"a": a, "b": b, "c": float(c), "r2": float(r2)}

    return best

def make_train_subset(X, y, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


# ----------------------------
# Main experiment
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to emails.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--seed", type=int, default=42)

    # Scaling grid
    ap.add_argument("--train_sizes", default="50,100,200,400,800",
                    help="Comma-separated training sizes N (must be <= train pool)")
    ap.add_argument("--repeats", type=int, default=3, help="Runs per (N,capacity) with different subsampling seeds")

    # Vectorizer
    ap.add_argument("--max_features", type=int, default=20000)
    ap.add_argument("--ngram_max", type=int, default=2)

    # Logistic capacities (C)
    ap.add_argument("--Cs", default="0.1,0.3,1,3,10", help="Comma-separated C values for LogisticRegression")

    # Optional MLP
    ap.add_argument("--use_mlp", type=int, default=0, help="1 to run MLP scaling too, else 0")
    ap.add_argument("--mlp_sizes", default="32,64,128,256", help="Comma-separated hidden sizes for MLP")
    ap.add_argument("--mlp_max_iter", type=int, default=30)

    # Output
    ap.add_argument("--out_csv", default="scaling_results.csv")
    ap.add_argument("--plot", type=int, default=1, help="1 to save plot, else 0")
    ap.add_argument("--plot_file", default="scaling_plot.png")

    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df["y"] = df[args.label_col].apply(map_label)
    texts = df[args.text_col].astype(str).tolist()
    y = df["y"].to_numpy(dtype=int)

    # Train/test split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # TF-IDF (fit on train only)
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, int(args.ngram_max)),
        max_features=int(args.max_features),
        token_pattern=r"(?u)\b\w+\b",  # simple word tokens
    )
    Xtr = vec.fit_transform(X_train_txt)
    Xte = vec.transform(X_test_txt)

    train_sizes = [int(s) for s in args.train_sizes.split(",") if s.strip()]
    Cs = [float(c) for c in args.Cs.split(",") if c.strip()]

    # Validate size feasibility
    maxN = max(train_sizes)
    if maxN > Xtr.shape[0]:
        raise ValueError(f"Max train_size={maxN} exceeds available train pool={Xtr.shape[0]}")

    rows: List[Dict[str, object]] = []

    # ---- Logistic Regression scaling ----
    for C in Cs:
        for N in train_sizes:
            for r in range(args.repeats):
                seed = args.seed + 1000 * r + int(N * 7) + int(C * 13)

                X_sub, y_sub = make_train_subset(Xtr, y_train, N, seed=seed)

                clf = LogisticRegression(
                    C=C,
                    max_iter=2000,
                    solver="liblinear",  # stable for sparse + small data
                )
                clf.fit(X_sub, y_sub)

                proba = clf.predict_proba(Xte)[:, 1]
                pred = (proba >= 0.5).astype(int)

                ll = log_loss(y_test, np.vstack([1 - proba, proba]).T, labels=[0, 1])
                p, rcl, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

                rows.append({
                    "model": "logreg",
                    "capacity": f"C={C}",
                    "C": C,
                    "N": N,
                    "repeat": r,
                    "logloss": float(ll),
                    "precision": float(p),
                    "recall": float(rcl),
                    "f1": float(f1),
                })

    # ---- Optional: MLP scaling (capacity via hidden size) ----
    if int(args.use_mlp) == 1:
        mlp_sizes = [int(s) for s in args.mlp_sizes.split(",") if s.strip()]
        for H in mlp_sizes:
            for N in train_sizes:
                for r in range(args.repeats):
                    seed = args.seed + 5000 + 1000 * r + int(N * 11) + int(H * 3)

                    X_sub, y_sub = make_train_subset(Xtr, y_train, N, seed=seed)

                    mlp = MLPClassifier(
                        hidden_layer_sizes=(H,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=int(args.mlp_max_iter),
                        random_state=seed,
                        early_stopping=True,
                        n_iter_no_change=5,
                        verbose=False,
                    )
                    mlp.fit(X_sub, y_sub)

                    proba = mlp.predict_proba(Xte)[:, 1]
                    pred = (proba >= 0.5).astype(int)

                    ll = log_loss(y_test, np.vstack([1 - proba, proba]).T, labels=[0, 1])
                    p, rcl, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)

                    rows.append({
                        "model": "mlp",
                        "capacity": f"H={H}",
                        "hidden": H,
                        "N": N,
                        "repeat": r,
                        "logloss": float(ll),
                        "precision": float(p),
                        "recall": float(rcl),
                        "f1": float(f1),
                    })

    # Save results
    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}  rows={len(res)}")

    # Aggregate mean over repeats
    agg = (
        res.groupby(["model", "capacity", "N"], as_index=False)
           .agg(logloss=("logloss", "mean"), f1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"))
    )

    # Fit a scaling curve for "best capacity per N" (lowest loss) for each model family
    fits = []
    for model_name in agg["model"].unique():
        sub = agg[agg["model"] == model_name].copy()

        # Choose the best capacity per N (min logloss)
        best_per_N = sub.loc[sub.groupby("N")["logloss"].idxmin()].sort_values("N")
        Nvals = best_per_N["N"].to_numpy(dtype=float)
        Lvals = best_per_N["logloss"].to_numpy(dtype=float)

        # c grid: from 0 to slightly below min(L)
        c_grid = np.linspace(0.0, max(1e-6, float(np.min(Lvals) * 0.95)), 60)
        fit = fit_power_law_loss(Nvals, Lvals, c_grid=c_grid)
        fit["model"] = model_name
        fits.append(fit)

        print(f"\n=== Scaling fit (best capacity) for model={model_name} ===")
        print(f"Loss(N) ≈ c + a*N^(-b)")
        print(f"a={fit['a']:.6f}  b={fit['b']:.6f}  c={fit['c']:.6f}  R2={fit['r2']:.4f}")

    fits_df = pd.DataFrame(fits)
    fits_df.to_csv("scaling_fits.csv", index=False)
    print("Saved: scaling_fits.csv")

    # Plot (loss vs N for each capacity + best-per-N + fitted curve)
    if int(args.plot) == 1:
        plt.figure()

        for model_name in agg["model"].unique():
            sub = agg[agg["model"] == model_name].copy()

            # Plot each capacity line
            for cap in sub["capacity"].unique():
                s2 = sub[sub["capacity"] == cap].sort_values("N")
                plt.plot(s2["N"], s2["logloss"], marker="o", linewidth=1, label=f"{model_name}:{cap}")

            # Best-per-N
            best = sub.loc[sub.groupby("N")["logloss"].idxmin()].sort_values("N")
            plt.plot(best["N"], best["logloss"], marker="o", linewidth=3, label=f"{model_name}:BEST")

            # Fit curve
            fit = fits_df[fits_df["model"] == model_name].iloc[0].to_dict()
            a, b, c = fit["a"], fit["b"], fit["c"]
            Ngrid = np.linspace(best["N"].min(), best["N"].max(), 200)
            Lgrid = c + a * (Ngrid ** (-b))
            plt.plot(Ngrid, Lgrid, linewidth=2, label=f"{model_name}:fit")

        plt.xscale("log")
        plt.yscale("linear")
        plt.xlabel("Training size N (log scale)")
        plt.ylabel("Test Log Loss (cross-entropy)")
        plt.title("Scaling Laws (Cybersecurity) — Loss vs Data Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_file, dpi=200)
        print(f"Saved: {args.plot_file}")

    # Also print a quick "best capacity per N" table
    print("\n=== Best capacity per N (by lowest logloss) ===")
    for model_name in agg["model"].unique():
        sub = agg[agg["model"] == model_name]
        best = sub.loc[sub.groupby("N")["logloss"].idxmin()].sort_values("N")
        print(f"\nModel={model_name}")
        print(best[["N", "capacity", "logloss", "f1", "precision", "recall"]].to_string(index=False))

if __name__ == "__main__":
    main()
