"""
Input:
- emails.csv with columns:
    text,label   (label: legit/phish)  <-- matches the dataset you created earlier

Install:
  pip install pandas numpy scikit-learn gensim

Run:
  python embeddings_cyber.py --csv emails.csv --text_col text --label_col label

ถ้าต้องการเดโมแบบ SOC clustering เพิ่ม (จัดกลุ่ม alert/ข้อความคล้ายกัน):
  python embeddings_cyber.py --csv emails.csv --do_cluster --k 6  
"""

from __future__ import annotations
import argparse
import re
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?", re.UNICODE)

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())

def map_label(x) -> int:
    s = str(x).strip().lower()
    if s in {"phish", "phishing", "spam", "malicious", "1", "true", "yes"}:
        return 1
    if s in {"legit", "ham", "benign", "0", "false", "no"}:
        return 0
    # try numeric
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unrecognized label: {x!r}")

def build_message_vector(tokens: list[str], w2v: Word2Vec, dim: int) -> np.ndarray:
    # Average of word vectors (simple + interpretable baseline)
    vecs = []
    for t in tokens:
        if t in w2v.wv:
            vecs.append(w2v.wv[t])
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to emails.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--vector_size", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--do_cluster", action="store_true", help="Also run KMeans clustering demo")
    ap.add_argument("--k", type=int, default=6, help="Number of clusters for KMeans (if enabled)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df["y"] = df[args.label_col].apply(map_label)
    df["tokens"] = df[args.text_col].apply(tokenize)

    # ---------
    # 1) Train Word2Vec
    # ---------
    corpus = df["tokens"].tolist()
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=1,                # 1=skip-gram (often better for small data)
        negative=10,
        workers=4,
        seed=args.seed,
    )
    w2v.train(corpus, total_examples=len(corpus), epochs=args.epochs)

    dim = args.vector_size

    # ---------
    # 2) Vector Semantics demo (nearest neighbors)
    # ---------
    print("\n=== Vector Semantics Demo (Nearest Neighbors) ===")
    probe_words = ["verify", "account", "password", "urgent", "click", "invoice", "payroll"]
    for w in probe_words:
        if w in w2v.wv:
            nn = w2v.wv.most_similar(w, topn=5)
            print(f"\nTop neighbors for '{w}':")
            for ww, sim in nn:
                print(f"  {ww:20s}  sim={sim:.3f}")
        else:
            print(f"\nWord '{w}' not in vocab (min_count too high or absent).")

    # Example cosine similarity between two words (if present)
    if "verify" in w2v.wv and "confirm" in w2v.wv:
        sim_vc = cosine(w2v.wv["verify"], w2v.wv["confirm"])
        print(f"\ncosine(verify, confirm) = {sim_vc:.3f}")

    # ---------
    # 3) Build message embeddings
    # ---------
    X = np.vstack([build_message_vector(toks, w2v, dim) for toks in df["tokens"]])
    y = df["y"].to_numpy()

    # ---------
    # 4) Train/Test classification
    # ---------
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    print("\n=== Phishing Classification (Embeddings → Logistic Regression) ===")
    print(f"Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    print("\nConfusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_te, y_pred, target_names=["legit", "phish"], zero_division=0))

    # Show a few high-confidence predictions
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_te)[:, 1]
        df_te = df_te.copy()
        df_te["p_phish"] = proba
        top_phish = df_te.sort_values("p_phish", ascending=False).head(5)
        top_legit = df_te.sort_values("p_phish", ascending=True).head(5)

        print("\n=== Top 5 most-phish-like (by probability) ===")
        for _, row in top_phish.iterrows():
            print(f"p={row['p_phish']:.3f} | {row[args.text_col][:160]}")

        print("\n=== Top 5 most-legit-like (by probability) ===")
        for _, row in top_legit.iterrows():
            print(f"p={row['p_phish']:.3f} | {row[args.text_col][:160]}")

    # ---------
    # 5) (Optional) Clustering for SOC triage
    # ---------
    if args.do_cluster:
        print("\n=== SOC-style Clustering Demo (KMeans on Message Embeddings) ===")
        km = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
        clusters = km.fit_predict(X)
        df["cluster"] = clusters

        # Print cluster summaries
        for c in range(args.k):
            sub = df[df["cluster"] == c]
            if len(sub) == 0:
                continue
            phish_rate = sub["y"].mean()
            print(f"\nCluster {c}: n={len(sub)}, phish_rate={phish_rate:.2f}")
            # show a few representative samples (closest to centroid)
            centroid = km.cluster_centers_[c]
            sub = sub.copy()
            sub["dist"] = np.linalg.norm(X[sub.index] - centroid, axis=1)
            reps = sub.sort_values("dist").head(3)
            for _, row in reps.iterrows():
                label = "phish" if row["y"] == 1 else "legit"
                print(f"  [{label}] {row[args.text_col][:160]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
