"""
CNN (TextCNN / 1D CNN) for Cybersecurity: Phishing Detection
- Uses Keras (TensorFlow) to build a CNN over token sequences
- Reads emails_50000_hardcore.csv (text,label)
- Tokenize -> pad sequences -> Embedding -> Conv1D (multiple kernel sizes) -> GlobalMaxPool -> Dense -> Sigmoid
- Evaluates Precision/Recall/F1

Install:
  pip install pandas numpy scikit-learn tensorflow

Run:
  python cnn_text_phish.py --csv emails_50000_hardcore.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, Model


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


def build_textcnn(vocab_size: int, max_len: int, emb_dim: int, filters: int, kernel_sizes: list[int], dropout: float):
    inp = layers.Input(shape=(max_len,), dtype="int32")

    x = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len)(inp)

    conv_pools = []
    for k in kernel_sizes:
        c = layers.Conv1D(filters=filters, kernel_size=k, activation="relu", padding="valid")(x)
        p = layers.GlobalMaxPooling1D()(c)
        conv_pools.append(p)

    x = layers.Concatenate()(conv_pools) if len(conv_pools) > 1 else conv_pools[0]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="emails_50000_hardcore.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--vocab_size", type=int, default=60000)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--emb_dim", type=int, default=128)

    ap.add_argument("--filters", type=int, default=128)
    ap.add_argument("--kernel_sizes", default="3,4,5")
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)

    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.label_col]).copy()
    df["y"] = df[args.label_col].apply(map_label)
    texts = df[args.text_col].astype(str).to_numpy()
    y = df["y"].to_numpy(dtype=np.int32)

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=args.vocab_size,
        oov_token="<OOV>",
        lower=True,
        filters=""  # keep punctuation because obfuscation patterns matter
    )
    tokenizer.fit_on_texts(X_train_txt)

    Xtr_seq = tokenizer.texts_to_sequences(X_train_txt)
    Xte_seq = tokenizer.texts_to_sequences(X_test_txt)

    Xtr = tf.keras.preprocessing.sequence.pad_sequences(Xtr_seq, maxlen=args.max_len, padding="post", truncating="post")
    Xte = tf.keras.preprocessing.sequence.pad_sequences(Xte_seq, maxlen=args.max_len, padding="post", truncating="post")

    kernel_sizes = [int(k.strip()) for k in args.kernel_sizes.split(",") if k.strip()]

    model = build_textcnn(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        emb_dim=args.emb_dim,
        filters=args.filters,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ]

    model.fit(
        Xtr, y_train,
        validation_split=0.1,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    proba = model.predict(Xte, batch_size=args.batch_size).reshape(-1)
    y_pred = (proba >= 0.5).astype(np.int32)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== TextCNN Test Metrics ===")
    print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phish"], zero_division=0))

    # Show a few high-risk examples
    topk = 5
    top_idx = np.argsort(-proba)[:topk]
    print(f"\n=== Top {topk} test samples predicted as PHISH (highest probability) ===")
    for i in top_idx:
        print(f"\nProb(phish)={proba[i]:.4f} | true={y_test[i]}")
        print(str(X_test_txt[i])[:500])

if __name__ == "__main__":
    main()
