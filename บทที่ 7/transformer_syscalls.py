import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report

from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# -----------------------------
# Label mapping
# -----------------------------
def map_label(x):
    x = str(x).lower().strip()
    if x in ["attack", "malicious", "1"]:
        return 1
    return 0


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="systemcalls_50000.csv")
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out_dir", default="./transformer_syscall_out")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    df["label_id"] = df[args.label_col].apply(map_label)

    X_train, X_test, y_train, y_test = train_test_split(
        df[args.seq_col].astype(str).tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )

    train_ds = Dataset.from_dict({"text": X_train, "labels": y_train})
    test_ds = Dataset.from_dict({"text": X_test, "labels": y_test})

    # Tokenizer (treat syscalls as words)
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )

    # Training arguments (OLD VERSION SAFE)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=200,
        save_steps=1000,
        no_cuda=not torch.cuda.is_available(),
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Evaluation
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=1)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print("\n=== Transformer (System Calls) Results ===")
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))


if __name__ == "__main__":
    main()
