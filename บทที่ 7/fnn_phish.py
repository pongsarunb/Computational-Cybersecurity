"""
Feedforward Neural Network (FNN/MLP) for Cybersecurity: Phishing Detection
- Read emails_50000_hardcore.csv (text,label)
- Build features: TF-IDF + (optional) obfuscation indicators (zero-width, bidi, combining, punycode, homoglyph)
- Train an FNN (MLPClassifier) and evaluate Precision/Recall/F1

Install:
  pip install pandas numpy scikit-learn scipy

Run:
  python fnn_phish.py --csv emails_50000_hardcore.csv
"""

from __future__ import annotations
import argparse
import re
import unicodedata
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix


# -----------------------------
# Label normalization
# -----------------------------
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


# -----------------------------
# Cyber-linguistic obfuscation features (numeric)
# -----------------------------
ZW = ["\u200b", "\u200c", "\u200d", "\ufeff"]  # ZWSP, ZWNJ, ZWJ, BOM

BIDI = [
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # LRE,RLE,PDF,LRO,RLO
    "\u2066", "\u2067", "\u2068", "\u2069",            # LRI,RLI,FSI,PDI
]

# common Cyrillic homoglyphs used to spoof Latin (subset)
CYR_HOMO = set(["\u0430", "\u0441", "\u0435", "\u0456", "\u043e", "\u0440", "\u0445", "\u0443", "\u04bb", "\u043c", "\u043f", "\u0442", "\u0455"])

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def is_combining(ch: str) -> bool:
    # combining diacritics block + combining class
    return ("\u0300" <= ch <= "\u036f") or unicodedata.combining(ch) != 0

def extract_obfuscation_features(texts: np.ndarray) -> csr_matrix:
    """
    Returns sparse matrix (n_samples, k_features)
    Features are scaled-ish counts (not standardized here).
    """
    feats = []
    for s in texts:
        s = str(s)

        n_chars = max(1, len(s))
        n_spaces = s.count(" ")

        # zero-width / bidi / combining
        zw_count = sum(s.count(z) for z in ZW)
        bidi_count = sum(s.count(b) for b in BIDI)
        comb_count = sum(1 for ch in s if is_combining(ch))

        # punycode / url count
        puny = s.count("xn--")
        url_count = len(URL_RE.findall(s))

        # case/punct/entropy-ish proxies
        upper = sum(1 for ch in s if ch.isupper())
        digit = sum(1 for ch in s if ch.isdigit())
        punct = sum(1 for ch in s if (not ch.isalnum()) and (not ch.isspace()))

        # "spaced words" / "dotted words" pattern indicators
        spaced_letters = len(re.findall(r"\b(?:[a-zA-Z]\s){3,}[a-zA-Z]\b", s))
        dotted_letters = len(re.findall(r"\b(?:[a-zA-Z]\.){3,}[a-zA-Z]\b", s))

        # Cyrillic homoglyph presence (quick signal)
        cyr_homo = sum(1 for ch in s if ch in CYR_HOMO)

        # ratio features
        feats.append([
            zw_count,
            bidi_count,
            comb_count,
            puny,
            url_count,
            upper / n_chars,
            digit / n_chars,
            punct / n_chars,
            (n_spaces / n_chars),
            spaced_letters,
            dotted_letters,
            cyr_homo,
        ])

    X = np.array(feats, dtype=np.float32)
    return csr_matrix(X)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="emails_50000_hardcore.csv", help="Path to CSV")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--use_obf_features", type=int, default=1, help="1=TFIDF + obfuscation features, 0=TFIDF only")

    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--ngram_max", type=int, default=2)

    ap.add_argument("--hidden", default="256,128", help="Hidden layer sizes, e.g. 256,128")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--max_iter", type=int, default=20)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df["y"] = df[args.label_col].apply(map_label)
    texts = df[args.text_col].astype(str).to_numpy()
    y = df["y"].to_numpy(dtype=int)

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, int(args.ngram_max)),
        max_features=int(args.max_features),
        token_pattern=r"(?u)\b\w+\b",
    )
    Xtr_tfidf = vec.fit_transform(X_train_txt)
    Xte_tfidf = vec.transform(X_test_txt)

    if int(args.use_obf_features) == 1:
        Xtr_obf = extract_obfuscation_features(X_train_txt)
        Xte_obf = extract_obfuscation_features(X_test_txt)
        Xtr = hstack([Xtr_tfidf, Xtr_obf]).tocsr()
        Xte = hstack([Xte_tfidf, Xte_obf]).tocsr()
    else:
        Xtr, Xte = Xtr_tfidf, Xte_tfidf

    hidden = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())

    # Feedforward Neural Network (MLP)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=float(args.alpha),
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=int(args.max_iter),
        random_state=args.seed,
        early_stopping=True,
        n_iter_no_change=3,
        verbose=True,
    )
    clf.fit(Xtr, y_train)

    proba = clf.predict_proba(Xte)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== FNN/MLP Test Metrics ===")
    print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phish"], zero_division=0))

    # Quick sanity: show top suspicious samples (highest phish probability) from test
    topk = 5
    top_idx = np.argsort(-proba)[:topk]
    print(f"\n=== Top {topk} test samples predicted as PHISH (highest probability) ===")
    for i in top_idx:
        print(f"\nProb(phish)={proba[i]:.4f} | true={y_test[i]}")
        print(str(X_test_txt[i])[:500])

if __name__ == "__main__":
    main()
