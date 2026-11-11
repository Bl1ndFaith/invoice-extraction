"""Field-level evaluation of the ML extractor, compatible with v1's
evaluate.py metrics: per-field exact match + style-insensitive
value-level match, plus per-item metrics for the items array.

Usage:
    python main.py evaluate --limit 20
    python main.py evaluate --split val
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from .align import normalize_text
from .dataset import load_dataset_records, split_records
from .infer import InvoiceModel

FIELD_LABELS = {
    "client_name": "invoice.client_name",
    "client_address": "invoice.client_address",
    "seller_name": "invoice.seller_name",
    "seller_address": "invoice.seller_address",
    "invoice_number": "invoice.invoice_number",
    "invoice_date": "invoice.invoice_date",
    "due_date": "invoice.due_date",
    "tax": "subtotal.tax",
    "discount": "subtotal.discount",
    "total": "subtotal.total",
}


def normalize_for_value_check(value: str) -> str:
    """Same normalization as v1 evaluate.py."""

    cleaned = "".join(value.split()).lower().strip()
    return cleaned.replace(",", ".")


def _load_gt_rows(data_dir: Path) -> dict[str, dict]:
    """File Name -> raw GT JSON, from every CSV in data/."""

    ground_truth: dict[str, dict] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        with open(csv_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ground_truth[row["File Name"].strip()] = json.loads(row["Json Data"])
    return ground_truth


def evaluate(
    model_dir: str = "runs/layoutlmv3-invoice/best",
    data_dir: str = "data",
    cache_dir: str = "cache/dataset",
    split: str = "val",
    limit: int = 0,
) -> None:
    records = load_dataset_records(cache_dir)
    if not records:
        raise SystemExit("Empty cache — run `python main.py prepare` first.")

    by_image = {record["image"]: record for record in records}

    splits_path = Path(model_dir).parent / "splits.json"
    if splits_path.exists():
        with open(splits_path, encoding="utf-8") as handle:
            splits = json.load(handle)
        if split == "val":
            records = [by_image[image] for image in splits["val"] if image in by_image]
        elif split == "train":
            records = [by_image[image] for image in splits["train"] if image in by_image]
        elif split == "test":
            seen = set(splits["train"]) | set(splits["val"])
            records = [r for image, r in by_image.items() if image not in seen]
    elif split == "val":
        _, records = split_records(records)
    elif split == "train":
        records, _ = split_records(records)

    ground_truth = _load_gt_rows(Path(data_dir))
    model = InvoiceModel(model_dir)

    total_fields: Counter = Counter()
    matched_fields: Counter = Counter()
    value_fields: Counter = Counter()
    value_matched: Counter = Counter()

    item_fields: Counter = Counter()
    item_matched: Counter = Counter()
    documents = records if limit == 0 else records[:limit]

    for number, record in enumerate(documents, 1):
        image_name = Path(record["image"]).name
        gt = ground_truth.get(image_name)
        if gt is None:
            print(f"[MISS] {image_name}: no GT row")
            continue

        predicted = model.extract_json(record["image"])

        # -- scalar fields ---------------------------------------------------
        for key, field in FIELD_LABELS.items():
            predicted_value = str(
                (predicted.get("invoice", {}) if key in predicted.get("invoice", {})
                 else predicted.get("subtotal", {})).get(key, "")
            )
            gt_invoice = gt.get("invoice", {})
            gt_subtotal = gt.get("subtotal", {})
            reference_value = str(
                (gt_invoice if key in gt_invoice else gt_subtotal).get(key, "") or ""
            )

            total_fields[field] += 1
            value_fields[field] += 1
            if predicted_value == reference_value:
                matched_fields[field] += 1
                value_matched[field] += 1
            elif (
                normalize_for_value_check(predicted_value)
                == normalize_for_value_check(reference_value)
            ):
                value_matched[field] += 1

        # -- items (positional comparison) -------------------------------------
        gt_items = gt.get("items", []) or []
        pred_items = predicted.get("items", []) or []

        item_count = max(len(gt_items), len(pred_items))
        item_fields["items_count"] += item_count
        item_fields["items_count_diff"] += abs(len(gt_items) - len(pred_items))

        for gt_item, pred_item in zip(gt_items, pred_items):
            for key in ("description", "quantity", "total_price"):
                item_fields[key] += 1
                gt_value = str(gt_item.get(key, "") or "")
                pred_value = str(pred_item.get(key, "") or "")

                if key == "description":
                    ratio = SequenceMatcher(
                        None, normalize_text(gt_value), normalize_text(pred_value)
                    ).ratio()
                    ok = ratio >= 0.95
                else:
                    ok = (
                        normalize_for_value_check(pred_value)
                        == normalize_for_value_check(gt_value)
                    )
                if ok:
                    item_matched[key] += 1

        if number % 10 == 0:
            print(f"  ... {number}/{len(documents)}")

    print("\n" + "=" * 64)
    print(f"FIELD-LEVEL ACCURACY ({split}, {len(documents)} documents)")
    print("=" * 64)

    overall_total = overall_matched = 0
    for key, field in FIELD_LABELS.items():
        count, match = total_fields[field], matched_fields[field]
        vcount, vmatch = value_fields[field], value_matched[field]
        overall_total += count
        overall_matched += match
        exact = 100.0 * match / count if count else 0.0
        value = 100.0 * vmatch / vcount if vcount else 0.0
        print(f"{field:<28} {match:>4}/{count:<4} {exact:6.1f}%   (value-level {value:5.1f}%)")

    if overall_total:
        print(
            f"\nOVERALL {overall_matched}/{overall_total} = "
            f"{100.0 * overall_matched / overall_total:.1f}%"
        )

    value_total = sum(value_fields[field] for field in FIELD_LABELS.values())
    value_ok = sum(value_matched[field] for field in FIELD_LABELS.values())
    if value_total:
        print(
            f"VALUE-LEVEL (style-insensitive) {value_ok}/{value_total} = "
            f"{100.0 * value_ok / value_total:.1f}%"
        )

    print("\nITEMS")
    for key in ("items_count", "items_count_diff", "description", "quantity", "total_price"):
        count, match = item_fields[key], item_matched[key]
        if count:
            print(f"{key:<24} {match:>5}/{count:<5} {100.0 * match / count:6.1f}%")
