"""Inference: image -> OCR -> LayoutLMv3 -> BIO -> JSON (v1 schema)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .align import ID2LABEL, LABEL2ID
from .image_processor import preprocess_image
from .ocr_engine import run_ocr
from .model import load_processor

# Entity groups concatenated into scalar fields (order = document order).
SCALAR_ENTITIES = [
    ("SELLER_NAME", "seller_name"),
    ("SELLER_ADDRESS", "seller_address"),
    ("CLIENT_NAME", "client_name"),
    ("CLIENT_ADDRESS", "client_address"),
    ("INVOICE_NUMBER", "invoice_number"),
    ("INVOICE_DATE", "invoice_date"),
    ("DUE_DATE", "due_date"),
    ("TAX", "tax"),
    ("DISCOUNT", "discount"),
    ("TOTAL", "total"),
]


class InvoiceModel:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        from transformers import AutoModelForTokenClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = load_processor_from_dir(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict_words(self, image_path: str | Path) -> list[dict]:
        """OCR the page and attach a predicted label to every word.

        Returns [{text, box, label}] in reading order (same shape as the
        aligned dataset cache entries).
        """

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        processed = preprocess_image(image)
        words = run_ocr(processed)

        seq = sorted(words, key=lambda w: (w.par_num, w.line_num, w.left))
        texts = [word.text for word in seq]
        boxes = [_normalize_box(word, processed.shape) for word in seq]

        encoding = self.processor(
            Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)),
            texts,
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits

        predictions = logits.argmax(-1).squeeze(0).cpu().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        labels = ["O"] * len(seq)
        previous_word = None
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word:
                continue
            labels[word_id] = ID2LABEL[predictions[position]]
            previous_word = word_id

        return [
            {
                "text": word.text,
                "box": [word.left, word.top, word.left + word.width,
                        word.top + word.height],
                "label": label,
            }
            for word, label in zip(seq, labels)
        ]

    def extract_json(self, image_path: str | Path) -> dict:
        """Full v1-schema JSON for one invoice image."""

        entries = self.predict_words(image_path)
        return build_json(entries)


def load_processor_from_dir(model_dir: str | Path):
    """Processor saved alongside the fine-tuned checkpoint (fallback: base)."""

    try:
        return load_processor(str(model_dir))
    except Exception:
        from .model import BASE_MODEL, load_processor

        return load_processor(BASE_MODEL)


def build_json(entries: list[dict]) -> dict:
    """Group BIO-labeled words into the v1 JSON schema.

    Items are reconstructed geometrically (like v1's table engine), not
    by reading order: wrapped description lines of neighboring items
    interleave in reading order, but each item owns the horizontal band
    between its own row and the next one.
    """

    scalars = _extract_scalars(entries)
    items = _group_items(entries)

    # A price token clearly below the last item row is the summary
    # 'Total' row mislabeled as B-ITEM_TOTAL_PRICE; rescue it when the
    # model produced no B-TOTAL of its own.
    if not scalars.get("total"):
        price_tokens = [e for e in entries if e["label"] == "B-ITEM_TOTAL_PRICE"]
        qty_tokens = [e for e in entries if e["label"] == "B-ITEM_QUANTITY"]
        if price_tokens and qty_tokens:
            heights = sorted(e["box"][3] - e["box"][1] for e in entries)
            median_height = heights[len(heights) // 2]
            last_row_y = max(_y_center(q) for q in qty_tokens)
            lowest = max(price_tokens, key=_y_center)
            if _y_center(lowest) > last_row_y + 2.0 * median_height:
                scalars["total"] = lowest["text"]

    due_date = scalars.get("due_date", "")

    return {
        "invoice": {
            "client_name": scalars.get("client_name", ""),
            "client_address": scalars.get("client_address", ""),
            "seller_name": scalars.get("seller_name", ""),
            "seller_address": scalars.get("seller_address", ""),
            "invoice_number": scalars.get("invoice_number", ""),
            "invoice_date": scalars.get("invoice_date", ""),
            "due_date": due_date,
        },
        "items": items,
        "subtotal": {
            "tax": scalars.get("tax", ""),
            "discount": scalars.get("discount", ""),
            "total": scalars.get("total", ""),
        },
        "payment_instructions": {
            "due_date": due_date,
            "bank_name": "",
            "account_number": "",
            "payment_method": "",
        },
    }


# Multi-occurrence handling per entity: addresses/names can be split into
# several groups (two-column layout interleaves seller/client lines in
# reading order) and must be joined; money values repeat down the page,
# where the GT meaning lives in the last group; identifiers in the first.
_SCALAR_JOIN_ALL = {"SELLER_NAME", "SELLER_ADDRESS", "CLIENT_NAME", "CLIENT_ADDRESS"}
_SCALAR_LAST = {"TAX", "TOTAL"}


def _extract_scalars(entries: list[dict]) -> dict[str, str]:
    """Collapse labeled tokens into scalar field values.

    Consecutive tokens of one entity form a group; a gap (other tokens)
    starts a new group. Group policy depends on the entity (see above).
    """

    mapping = {entity: key for entity, key in SCALAR_ENTITIES}
    groups: dict[str, list[list[str]]] = {}
    current: str | None = None

    for entry in entries:
        label = entry["label"]
        base = label[2:] if label != "O" else None

        if base not in mapping:
            current = None
            continue

        groups.setdefault(base, [])
        if base == current and groups[base]:
            groups[base][-1].append(entry["text"])
        else:
            groups[base].append([entry["text"]])
        current = base

    scalars: dict[str, str] = {}
    for base, group_list in groups.items():
        if base in _SCALAR_JOIN_ALL:
            scalars[mapping[base]] = " ".join(
                word for group in group_list for word in group
            )
        elif base in _SCALAR_LAST:
            scalars[mapping[base]] = " ".join(group_list[-1])
        else:
            scalars[mapping[base]] = " ".join(group_list[0])

    return scalars


def _y_center(entry: dict) -> float:
    top, bottom = entry["box"][1], entry["box"][3]
    return (top + bottom) / 2.0


def _sorted_by_lines(entries: list[dict]) -> list[dict]:
    """Reading order for words of one item: cluster words into visual
    lines by y (tolerance ~60% of word height — exact y_center sorting
    scrambles words whose baselines differ slightly), then left-to-right
    inside each line."""

    if not entries:
        return []

    ordered = sorted(entries, key=_y_center)
    lines: list[list[dict]] = [[ordered[0]]]

    for entry in ordered[1:]:
        height = entry["box"][3] - entry["box"][1]
        current_line = lines[-1]
        line_y = sum(_y_center(e) for e in current_line) / len(current_line)
        if abs(_y_center(entry) - line_y) <= 0.6 * height:
            current_line.append(entry)
        else:
            lines.append([entry])

    result: list[dict] = []
    for line in lines:
        result.extend(sorted(line, key=lambda e: e["box"][0]))
    return result


def _group_items(entries: list[dict]) -> list[dict]:
    """Geometry-based item reconstruction.

    Each B-ITEM_QUANTITY token anchors one item row (qty sits on the
    row's first line, like the description's first line). The item owns:
      - description: DESC tokens from its row down to the next item's row
        (wrapped description lines fall between the two anchors);
      - total_price: the PRICE token on the item's own line (net/other
        money columns are not PRICE-labeled).
    """

    qty_tokens = [e for e in entries if e["label"] == "B-ITEM_QUANTITY"]
    if not qty_tokens:
        return _group_items_reading_order(entries)

    desc_tokens = [e for e in entries if e["label"].endswith("ITEM_DESC")]
    price_tokens = [e for e in entries if e["label"] == "B-ITEM_TOTAL_PRICE"]

    heights = sorted(e["box"][3] - e["box"][1] for e in entries)
    median_height = heights[len(heights) // 2] if heights else 20.0
    pad = median_height / 2.0

    qty_tokens = sorted(qty_tokens, key=_y_center)
    items: list[dict] = []

    for index, qty in enumerate(qty_tokens):
        row_y = _y_center(qty)
        next_y = _y_center(qty_tokens[index + 1]) if index + 1 < len(qty_tokens) else None

        band_top = row_y - pad
        band_bottom = (next_y - pad) if next_y is not None else float("inf")

        desc_words = _sorted_by_lines(
            [e for e in desc_tokens if band_top <= _y_center(e) < band_bottom]
        )

        own_price = min(
            (e for e in price_tokens if abs(_y_center(e) - row_y) <= 2.0 * median_height),
            key=lambda e: abs(_y_center(e) - row_y),
            default=None,
        )

        items.append({
            "description": " ".join(e["text"] for e in desc_words),
            "quantity": qty["text"],
            "total_price": own_price["text"] if own_price else "",
        })

    # A price token far below the last item row is the summary 'Total'
    # row mislabeled as an item price; build_json rescues it for
    # subtotal.total, so nothing to do here beyond the row bands.
    return items


def _group_items_reading_order(entries: list[dict]) -> list[dict]:
    """Fallback: reading-order grouping (used when no quantity was found)."""

    items: list[dict] = []
    current: dict | None = None

    for entry in entries:
        label = entry["label"]
        if label == "O":
            continue

        base = label[2:]
        starts_new = label.startswith("B-")

        if base == "ITEM_DESC":
            if starts_new or current is None:
                current = {"description": entry["text"], "quantity": "",
                           "total_price": ""}
                items.append(current)
            else:
                current["description"] += " " + entry["text"]
            continue

        if base in ("ITEM_QUANTITY", "ITEM_TOTAL_PRICE"):
            key = "quantity" if base == "ITEM_QUANTITY" else "total_price"
            if current is None:
                current = {"description": "", "quantity": "", "total_price": ""}
                items.append(current)
            if starts_new or not current[key]:
                current[key] = entry["text"]
            else:
                current[key] += " " + entry["text"]

    return items


def _normalize_box(word, shape: tuple[int, ...]) -> list[int]:
    height, width = shape[:2]
    return [
        max(0, min(1000, round(1000 * word.left / width))),
        max(0, min(1000, round(1000 * word.top / height))),
        max(0, min(1000, round(1000 * (word.left + word.width) / width))),
        max(0, min(1000, round(1000 * (word.top + word.height) / height))),
    ]


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/layoutlmv3-invoice/best"
    image_path = sys.argv[2]
    document = InvoiceModel(model_dir).extract_json(image_path)
    print(json.dumps(document, ensure_ascii=False, indent=2))
