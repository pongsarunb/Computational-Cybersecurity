"""
RNN (LSTM/GRU) for Cybersecurity: System Call Sequence Attack Detection
- Reads systemcalls.csv (sequence,label: normal/attack)
- Tokenize system calls -> pad sequences -> Embedding -> BiLSTM/GRU -> Dense -> Sigmoid
- Evaluates Precision/Recall/F1

Assumes CSV has:
  - a column with system-call sequence text (e.g., "sequence" or "calls" or "text")
    example: "open,read,write,execve,connect,..."
  - a label column (e.g., "label") with values like normal/attack (or 0/1)

Install:
  pip install pandas numpy scikit-learn tensorflow

Run:
  python rnn_syscalls.py --csv /mnt/data/systemcalls.csv --seq_col sequence --label_col label
"""

from __future__ import annotations
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, Model


def map_label(x) -> int:
    s = str(x).strip().lower()
    if s in {"attack", "malicious", "intrusion", "1", "true", "yes"}:
        return 1
    if s in {"normal", "benign", "0", "false", "no"}:
        return 0
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unrecognized label: {x!r}")


def infer_columns(df: pd.DataFrame, seq_col: str | None, label_col: str | None) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}

    # label
    if label_col is None:
        for k in ["label", "y", "target", "class"]:
            if k in cols:
                label_col = cols[k]
                break

    # sequence
    if seq_col is None:
        for k in ["sequence", "seq", "calls", "syscalls", "text", "events"]:
            if k in cols:
                seq_col = cols[k]
                break

    if seq_col is None or label_col is None:
        raise ValueError(f"Cannot infer columns. Found columns: {df.columns.tolist()}")

    return seq_col, label_col


def split_to_tokens(seq: str) -> list[str]:
    """
    Accepts sequences like:
      "open,read,write,execve"
      "open read write execve"
      "open|read|write|execve"
    """
    s = str(seq).strip()
    # normalize separators to spaces
    s = re.sub(r"[,\|;/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")


def build_rnn_model(
    vocab_size: int,
    max_len: int,
    emb_dim: int = 64,
    rnn_type: str = "lstm",   # "lstm" or "gru"
    rnn_units: int = 128,
    bidir: int = 1,
    dropout: float = 0.3,
) -> Model:
    inp = layers.Input(shape=(max_len,), dtype="int32")

    x = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len)(inp)
    x = layers.SpatialDropout1D(dropout)(x)

    if rnn_type.lower() == "gru":
        core = layers.GRU(rnn_units, return_sequences=False, dropout=dropout, recurrent_dropout=0.0)
    else:
        core = layers.LSTM(rnn_units, return_sequences=False, dropout=dropout, recurrent_dropout=0.0)

    if bidir == 1:
        x = layers.Bidirectional(core)(x)
    else:
        x = core(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/data/systemcalls.csv")
    ap.add_argument("--seq_col", default=None, help="Column name for syscall sequence (optional)")
    ap.add_argument("--label_col", default=None, help="Column name for label (optional)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--vocab_size", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--emb_dim", type=int, default=64)

    ap.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--rnn_units", type=int, default=128)
    ap.add_argument("--bidir", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    df = pd.read_csv(args.csv)
    seq_col, label_col = infer_columns(df, args.seq_col, args.label_col)

    df = df.dropna(subset=[seq_col, label_col]).copy()
    df["y"] = df[label_col].apply(map_label).astype(np.int32)

    sequences = df[seq_col].astype(str).tolist()
    y = df["y"].to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Tokenize system calls as "words"
    # IMPORTANT: keep filters empty; syscalls are already token-like
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=args.vocab_size,
        oov_token="<OOV>",
        lower=True,
        filters="",  # keep punctuation if any
        split=" "
    )

    # Convert sequences to normalized space-separated syscall tokens
    X_train_norm = [" ".join(split_to_tokens(s)) for s in X_train]
    X_test_norm  = [" ".join(split_to_tokens(s)) for s in X_test]

    tokenizer.fit_on_texts(X_train_norm)

    Xtr_seq = tokenizer.texts_to_sequences(X_train_norm)
    Xte_seq = tokenizer.texts_to_sequences(X_test_norm)

    Xtr = tf.keras.preprocessing.sequence.pad_sequences(
        Xtr_seq, maxlen=args.max_len, padding="post", truncating="post"
    )
    Xte = tf.keras.preprocessing.sequence.pad_sequences(
        Xte_seq, maxlen=args.max_len, padding="post", truncating="post"
    )

    model = build_rnn_model(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        emb_dim=args.emb_dim,
        rnn_type=args.rnn_type,
        rnn_units=args.rnn_units,
        bidir=args.bidir,
        dropout=args.dropout,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ]

    model.fit(
        Xtr, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    proba = model.predict(Xte, batch_size=args.batch_size).reshape(-1)
    y_pred = (proba >= 0.5).astype(np.int32)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== RNN Test Metrics (System Calls) ===")
    print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"], zero_division=0))

    # Show some sequences with highest attack probability (often multi-stage patterns)
    topk = 5
    top_idx = np.argsort(-proba)[:topk]
    print(f"\n=== Top {topk} sequences predicted as ATTACK (highest probability) ===")
    for i in top_idx:
        print(f"\nProb(attack)={proba[i]:.4f} | true={y_test[i]}")
        print(X_test[i][:500])

if __name__ == "__main__":
    main()
