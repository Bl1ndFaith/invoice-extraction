"""CLI entry point: invoice image(s) -> JSON documents.

Usage:
    python main.py data/batch1_1/batch1-0494.jpg
    python main.py data/batch1_1 --out output/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modules.image_processor import load_image, preprocess_image
from modules.invoice_math import reconcile
from modules.json_builder import build_invoice_document, validate_document
from modules.kv_extractor import extract_kv_fields
from modules.ocr_engine import REOCR_CONFIG, group_words_into_lines, run_ocr
from modules.table_extractor import extract_items
from modules.zone_splitter import Zones, find_zones


# ---------------------------------------------------------------------------
# Lazy re-OCR fallback for the items zone
# ---------------------------------------------------------------------------

REOCR_CROP_PAD = 10          # pixels around the items band (descenders)
REOCR_MIN_CROP_HEIGHT = 20   # a narrower crop cannot hold a table
REOCR_FULL_MIN_ITEMS = 3     # fewer items on an unanchored page = fragment


def _items_need_fallback(items: list) -> bool:
    """The primary pass found no items, or most rows carry no price."""

    if not items:
        return True
    missing_totals = sum(1 for item in items if not item.total_price)
    return missing_totals > len(items) / 2


def _reocr_items(processed, zones: Zones) -> list:
    """Re-OCR the items-zone crop with single-block segmentation (psm 6).

    Returns RawItem list for the crop; never touches the primary result.
    """

    height, _ = processed.shape[:2]
    top = max(0, int(zones.items_top_y) - REOCR_CROP_PAD)
    bottom = min(height, int(zones.totals_top_y) + REOCR_CROP_PAD)
    if bottom - top < REOCR_MIN_CROP_HEIGHT:
        return []

    crop = processed[top:bottom, :]
    words = run_ocr(crop, config=REOCR_CONFIG)
    lines = group_words_into_lines(words)

    crop_height = float(bottom - top)
    fallback_zones = Zones(
        page_height=crop_height,
        items_top_y=0.0,
        totals_top_y=crop_height,
        items_bottom_y=crop_height,
    )
    return extract_items(lines, fallback_zones)


def _remove_table_borders(processed):
    """Paint detected table grid lines white (bordered-table recovery).

    Tesseract skips numeric cells inside ruled tables in every psm mode:
    the vertical borders break its text model. Removing the grid lines
    turns the table into plain text that both psm modes read completely.
    """

    import cv2

    height, width = processed.shape[:2]
    binary = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, -2,
    )

    # Long straight structures only: text strokes are short, grid lines
    # span whole rows/columns of the table.
    length = max(25, min(height, width) // 20)
    horizontal = cv2.erode(
        binary, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    )
    vertical = cv2.erode(
        binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    )
    horizontal = cv2.dilate(
        horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    )
    vertical = cv2.dilate(
        vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    )

    grid = cv2.bitwise_or(horizontal, vertical)
    cleaned = processed.copy()
    cleaned[grid > 0] = 255
    return cleaned


def _reocr_full_page(processed) -> tuple[list, Zones]:
    """Full-page re-OCR (psm 6) with re-zoning from the new lines.

    For FRAGMENT pages (no table header, few money-aligned rows) psm 3
    skips most cells of bordered/shaded tables, so even the zone finder
    sees too little. A single-block pass reads such tables completely;
    zones are then recomputed from the psm-6 lines. If the page is RULED
    (borders eat the cells in every mode), the grid lines are painted
    out first and the pass is retried on the cleaned image.
    """

    words = run_ocr(processed, config=REOCR_CONFIG)
    lines = group_words_into_lines(words)
    fallback_zones = find_zones(lines, float(processed.shape[0]))
    best_items = extract_items(lines, fallback_zones)

    if len(best_items) < REOCR_FULL_MIN_ITEMS:
        try:
            cleaned = _remove_table_borders(processed)
        except Exception as error:  # cv2 issues must not break the run
            print(f"[WARN] border removal failed: {error}")
            cleaned = None
        if cleaned is not None:
            words = run_ocr(cleaned, config=REOCR_CONFIG)
            lines = group_words_into_lines(words)
            cleaned_zones = find_zones(lines, float(cleaned.shape[0]))
            cleaned_items = extract_items(lines, cleaned_zones)
            if len(cleaned_items) > len(best_items):
                best_items = cleaned_items
                fallback_zones = cleaned_zones

    return best_items, fallback_zones


def run_pipeline(image_path: Path) -> tuple[dict, list[str], object]:
    """Full zone-based pipeline for one image.

    OCR -> zones -> KV extraction -> items table -> arithmetic report.
    Returns (document, validation_errors, math_report).
    """

    image = load_image(image_path)
    processed = preprocess_image(image)

    words = run_ocr(processed)
    lines = group_words_into_lines(words)
    zones = find_zones(lines, float(processed.shape[0]))

    if "ratio" in (zones.items_top_source, zones.totals_top_source):
        print(
            f"[WARN] partial document? zone anchors missing "
            f"(items_top: {zones.items_top_source}, "
            f"totals_top: {zones.totals_top_source}) — extraction may be "
            f"incomplete"
        )

    kv = extract_kv_fields(lines, zones)
    items = extract_items(lines, zones)

    # Lazy re-OCR fallbacks: the primary full-page pass (psm 3) is never
    # changed; a fallback result is adopted strictly when it finds MORE
    # items. Full page first (fragments where psm 3 skipped table cells
    # and zoning itself is unreliable), then the zone crop (Крок E).
    if zones.items_top_source != "header" and len(items) < REOCR_FULL_MIN_ITEMS:
        try:
            full_page_items, _ = _reocr_full_page(processed)
        except Exception as error:  # OCR failures must not break the run
            print(f"[WARN] full-page re-OCR failed: {error}")
            full_page_items = []
        if len(full_page_items) > len(items):
            print(
                f"[MATH] items re-OCR adopted (full page, psm 6): "
                f"{len(items)} -> {len(full_page_items)} rows"
            )
            items = full_page_items

    if _items_need_fallback(items):
        try:
            fallback_items = _reocr_items(processed, zones)
        except Exception as error:  # OCR failures must not break the run
            print(f"[WARN] items re-OCR failed: {error}")
            fallback_items = []
        if len(fallback_items) > len(items):
            print(
                f"[MATH] items re-OCR adopted: {len(items)} -> "
                f"{len(fallback_items)} rows (psm 6 crop)"
            )
            items = fallback_items

    raw_invoice = {
        "client_name": kv["client_name"],
        "client_address": kv["client_address"],
        "seller_name": kv["seller_name"],
        "seller_address": kv["seller_address"],
        "invoice_number": kv["invoice_number"],
        "invoice_date": kv["invoice_date"],
        "due_date": kv["due_date"],
    }
    raw_items = [
        {
            "description": item.description,
            "quantity": item.quantity,
            "total_price": item.total_price,
        }
        for item in items
    ]
    # discount: no label dictionary source yet; internal shipping/discount
    # amounts stay out of the schema on purpose.
    raw_subtotal = {"tax": kv["tax"], "discount": "", "total": kv["total"]}

    document = build_invoice_document(raw_invoice, raw_items, raw_subtotal)
    errors = validate_document(document)

    math_report = reconcile(
        [item["total_price"] for item in raw_items],
        kv["subtotal"], kv["tax"], kv["total"],
        shipping=kv.get("shipping", ""),
        discount=kv.get("discount", ""),
    )

    return document, errors, math_report


def process_image(image_path: Path) -> tuple[dict, list[str]]:
    """Full pipeline for one image: returns (document, validation_errors)."""

    document, errors, _ = run_pipeline(image_path)
    return document, errors


def write_document(document: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Invoice image to JSON extractor")
    parser.add_argument("target", help="invoice .jpg file or a folder with invoices")
    parser.add_argument(
        "--out", default="output", help="output folder for generated JSON files"
    )
    parser.add_argument(
        "--print", action="store_true", help="print JSON to stdout"
    )
    args = parser.parse_args()

    target = Path(args.target)
    out_dir = Path(args.out)

    if target.is_file():
        image_paths = [target]
    elif target.is_dir():
        image_paths = sorted(target.glob("*.jpg")) + sorted(target.glob("*.png"))
    else:
        print(f"Target not found: {target}")
        return 2

    failures = 0

    for image_path in image_paths:
        try:
            document, errors, math_report = run_pipeline(image_path)
        except Exception as error:  # keep batch running on failures
            print(f"[FAIL] {image_path.name}: {error}")
            failures += 1
            continue

        if errors:
            print(f"[WARN] {image_path.name}: {len(errors)} schema issue(s)")
            for message in errors:
                print(f"       - {message}")

        for name, status in math_report.checks.items():
            if not status.startswith("skip"):
                print(f"[MATH] {image_path.name}: {name}: {status}")

        destination = out_dir / (image_path.stem + ".json")
        write_document(document, destination)

        if args.print:
            print(json.dumps(document, ensure_ascii=False, indent=2))

        print(f"[OK] {image_path.name} -> {destination}")

    print(f"\nDone. {len(image_paths) - failures}/{len(image_paths)} files processed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
