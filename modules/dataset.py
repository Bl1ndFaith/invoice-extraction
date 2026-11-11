"""PyTorch Dataset for LayoutLMv3 token classification.

Each cached document is {image: path, words: [{text, box, label}]}.
The image is re-preprocessed with the v1 chain at load time so that
pixel content and OCR boxes always share the same coordinate space
(upscale + deskew change the canvas), then normalized boxes 0..1000
are produced for the processor.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from .align import LABEL2ID
from .image_processor import preprocess_image

MAX_LENGTH = 512


class InvoiceTokenDataset(Dataset):
    def __init__(self, records: list[dict], processor: AutoProcessor):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        image = cv2.imread(record["image"])
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {record['image']}")
        processed = preprocess_image(image)
        pil_image = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))

        words = [entry["text"] for entry in record["words"]]
        boxes = [_normalize_box(entry["box"], processed.shape) for entry in record["words"]]
        labels = [LABEL2ID.get(entry["label"], 0) for entry in record["words"]]

        encoding = self.processor(
            pil_image,
            words,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Word labels -> subtoken labels: first subtoken carries the label,
        # the rest (and special/padding tokens) are masked with -100.
        word_ids = encoding.word_ids(batch_index=0)
        token_labels = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)
            elif word_id != previous_word:
                token_labels.append(labels[word_id])
            else:
                token_labels.append(-100)
            previous_word = word_id

        encoding["labels"] = torch.tensor(token_labels, dtype=torch.long)

        return {key: value.squeeze(0) for key, value in encoding.items()}


def _normalize_box(box: list[int], shape: tuple[int, ...]) -> list[int]:
    """Pixel box (x0, y0, x1, y1) -> LayoutLMv3's 0..1000 coordinate space."""

    height, width = shape[:2]
    x0, y0, x1, y1 = box

    return [
        max(0, min(1000, round(1000 * x0 / width))),
        max(0, min(1000, round(1000 * y0 / height))),
        max(0, min(1000, round(1000 * x1 / width))),
        max(0, min(1000, round(1000 * y1 / height))),
    ]


def load_dataset_records(cache_dir: str | Path) -> list[dict]:
    """Load all cached aligned documents, sorted by file name."""

    records = []
    for path in sorted(Path(cache_dir).glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def split_records(records: list[dict], val_fraction: float = 0.1, seed: int = 42):
    """Deterministic document-level train/val split."""

    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))

    return shuffled[val_count:], shuffled[:val_count]
