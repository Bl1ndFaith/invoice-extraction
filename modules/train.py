"""Fine-tuning loop for LayoutLMv3 on the aligned batch1 dataset.

Hardware target: GTX 1660 Ti 6GB -> fp16 + small per-device batch +
gradient accumulation. ~450 train documents, single template, so the
model converges in tens of minutes per epoch budget.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from transformers import (
    Trainer,
    TrainingArguments,
)

from .align import ID2LABEL, LABELS
from .dataset import InvoiceTokenDataset, load_dataset_records, split_records
from .model import BASE_MODEL, load_model, load_processor


def compute_metrics(eval_prediction):
    """Token-level accuracy + entity-level seqeval F1 (BIO scheme)."""

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    true_predictions, true_labels = [], []
    for prediction_row, label_row in zip(predictions, labels):
        filtered_pred, filtered_label = [], []
        for prediction, label in zip(prediction_row, label_row):
            if label != -100:
                filtered_pred.append(ID2LABEL[prediction])
                filtered_label.append(ID2LABEL[label])
        true_predictions.append(filtered_pred)
        true_labels.append(filtered_label)

    accuracy = float(
        np.mean(
            [
                prediction == label
                for prediction_row, label_row in zip(true_predictions, true_labels)
                for prediction, label in zip(prediction_row, label_row)
            ]
        )
    )

    f1 = _seqeval_f1(true_predictions, true_labels)

    return {"token_accuracy": accuracy, "f1": f1}


def _seqeval_f1(predictions: list[list[str]], references: list[list[str]]) -> float:
    from seqeval.metrics import f1_score

    return float(f1_score(references, predictions))


def train(
    cache_dir: str = "cache/dataset",
    output_dir: str = "runs/layoutlmv3-invoice",
    base_model: str = BASE_MODEL,
    epochs: int = 30,
    learning_rate: float = 2e-5,
    max_docs: int = 0,
) -> None:
    records = load_dataset_records(cache_dir)
    if not records:
        raise SystemExit(
            f"No cached documents in {cache_dir}. Run `python main.py prepare` first."
        )

    if max_docs and len(records) > max_docs:
        import random

        random.Random(42).shuffle(records)
        records = records[:max_docs]
        print(f"Subsampled to {len(records)} documents (max_docs={max_docs})")

    train_records, val_records = split_records(records)
    print(f"Dataset: {len(train_records)} train / {len(val_records)} val")

    # Persist the split so evaluate can score documents the model has
    # never seen (test = everything outside train+val).
    import json

    split_path = Path(output_dir) / "splits.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": [r["image"] for r in train_records],
                "val": [r["image"] for r in val_records],
            },
            handle,
            indent=1,
        )

    processor = load_processor(base_model)
    train_dataset = InvoiceTokenDataset(train_records, processor)
    val_dataset = InvoiceTokenDataset(val_records, processor)

    model = load_model(len(LABELS), base_model)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_dir = f"{output_dir}/best"
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"Best model saved to {best_dir}")


if __name__ == "__main__":
    train()
