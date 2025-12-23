"""
Dependencies:
  pip install pandas scikit-learn sklearn-crfsuite

Run:
  python crf_phish_label.py --csv phishing_tokens.csv --text_col text --labels_col labels
"""
from __future__ import annotations
import argparse
import re
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn_crfsuite import metrics


# ----------------------------
# Tokenization
# ----------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?|[^\s]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    # Keeps URLs as single tokens + splits punctuation.
    return TOKEN_RE.findall(str(text))


# ----------------------------
# Feature extraction
# ----------------------------

def token2features(tokens: List[str], i: int) -> Dict[str, object]:
    tok = tokens[i]
    feats: Dict[str, object] = {}

    feats["bias"] = 1.0
    feats["tok"] = tok.lower()
    feats["is_upper"] = tok.isupper()
    feats["is_title"] = tok.istitle()
    feats["is_digit"] = tok.isdigit()
    feats["has_digit"] = any(ch.isdigit() for ch in tok)
    feats["has_at"] = "@" in tok
    feats["has_dash"] = "-" in tok
    feats["has_dot"] = "." in tok
    feats["len"] = len(tok)
    feats["prefix1"] = tok[:1].lower()
    feats["prefix2"] = tok[:2].lower()
    feats["prefix3"] = tok[:3].lower()
    feats["suffix1"] = tok[-1:].lower()
    feats["suffix2"] = tok[-2:].lower()
    feats["suffix3"] = tok[-3:].lower()
    feats["is_url"] = bool(URL_RE.fullmatch(tok)) or tok.lower().startswith(("http://", "https://", "www."))

    # Common phishing cue lexicon (very small baseline; expand as needed)
    cue_words = {
        "urgent", "immediately", "important", "action", "required", "verify", "confirm",
        "password", "credentials", "login", "sign", "suspend", "locked", "failure",
        "click", "update", "bank", "payroll", "invoice"
    }
    feats["is_cue_word"] = tok.lower() in cue_words

    # Context window
    if i > 0:
        prev = tokens[i - 1]
        feats["-1:tok"] = prev.lower()
        feats["-1:is_url"] = prev.lower().startswith(("http://", "https://", "www."))
    else:
        feats["BOS"] = True

    if i < len(tokens) - 1:
        nxt = tokens[i + 1]
        feats["+1:tok"] = nxt.lower()
        feats["+1:is_url"] = nxt.lower().startswith(("http://", "https://", "www."))
    else:
        feats["EOS"] = True

    return feats


def sent2features(tokens: List[str]) -> List[Dict[str, object]]:
    return [token2features(tokens, i) for i in range(len(tokens))]


# ----------------------------
# Data loading + alignment checks
# ----------------------------

def load_sequences(df: pd.DataFrame, text_col: str, labels_col: str) -> Tuple[List[List[str]], List[List[str]]]:
    X_tokens: List[List[str]] = []
    Y_tags: List[List[str]] = []

    for idx, row in df.iterrows():
        tokens = tokenize(row[text_col])
        labels = str(row[labels_col]).split()

        if len(tokens) != len(labels):
            raise ValueError(
                f"Row {idx}: token/label length mismatch.\n"
                f"Text: {row[text_col]!r}\n"
                f"Tokens({len(tokens)}): {tokens}\n"
                f"Labels({len(labels)}): {labels}\n"
                "Fix by ensuring labels are aligned to this tokenizer."
            )

        X_tokens.append(tokens)
        Y_tags.append(labels)

    return X_tokens, Y_tags


# ----------------------------
# Train + Evaluate
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--text_col", default="text", help="Text column name")
    ap.add_argument("--labels_col", default="labels", help="Space-separated BIO tags column")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iterations", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.labels_col])

    # Build sequences
    X_tokens, Y_tags = load_sequences(df, args.text_col, args.labels_col)

    # Split into train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tokens, Y_tags, test_size=args.test_size, random_state=args.seed
    )

    X_tr_feats = [sent2features(s) for s in X_tr]
    X_te_feats = [sent2features(s) for s in X_te]

    # CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,     # L1
        c2=0.1,     # L2
        max_iterations=args.max_iterations,
        all_possible_transitions=True
    )

    crf.fit(X_tr_feats, y_tr)
    y_pred = crf.predict(X_te_feats)

    # Metrics
    labels = list(crf.classes_)
    # Remove 'O' from per-class summary if you want
    labels_no_o = [l for l in labels if l != "O"]

    print("\n=== CRF Token Labeling Evaluation ===")
    print(f"csv={args.csv} test_size={args.test_size}")
    print(f"labels={labels}")

    print("\n--- Per-class report (excluding 'O') ---")
    print(
        metrics.flat_classification_report(
            y_te, y_pred, labels=labels_no_o, digits=4, zero_division=0
        )
    )

    # Overall token-level F1 micro (excluding 'O' is common for NER-style tasks)
    f1_micro = metrics.flat_f1_score(y_te, y_pred, average="micro", labels=labels_no_o)
    f1_macro = metrics.flat_f1_score(y_te, y_pred, average="macro", labels=labels_no_o)
    print(f"\nToken-level F1 (micro, excl 'O') = {f1_micro:.4f}")
    print(f"Token-level F1 (macro, excl 'O') = {f1_macro:.4f}")

    # Show a few predictions
    print("\n=== Sample predictions ===")
    for i in range(min(5, len(X_te))):
        toks = X_te[i]
        true = y_te[i]
        pred = y_pred[i]
        print("\nText tokens:")
        print(toks)
        print("True tags:")
        print(true)
        print("Pred tags:")
        print(pred)


if __name__ == "__main__":
    main()
