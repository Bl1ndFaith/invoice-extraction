"""Zone images -> invoice.json."""
import json

from modules.classify import classify_zones
from modules.ocr import ocr_words, words_to_text
from modules.parse_header import parse_header
from modules.parse_items import parse_items
from modules.parse_parties import parse_parties
from modules.parse_summary import parse_summary
from modules.table_cells import remove_grid


# (upscale, tesseract page-segmentation mode); tried in this order until the totals agree
OCR_ATTEMPTS = [(2, 6), (1, 6), (3, 6), (2, 4), (3, 4)]


def _sum_diff(items, targets):
    """Arithmetic check: smallest |sum of item totals - target|, None if there is nothing to compare.
    Targets are the invoice total and the sub total (the total may include shipping)."""
    targets = [t for t in targets if t is not None]
    if not targets:
        return None
    s = sum(i["total_price"] or 0 for i in items)
    return min(abs(s - t) for t in targets)


def _public(items):
    return [{k: v for k, v in i.items() if not k.startswith("_")} for i in items]


def parse_table_validated(table_img, targets):
    """Parse the items table; if the item totals do not add up to the invoice total, re-run
    OCR with other settings and keep the attempt that agrees best.

    OCR runs often lose different rows. Rows are cut between the table's rules, so row k of one
    run is row k of another: if no single run adds up, the rows are combined from all runs.

    Returns (items, check) where check is 'ok', 'mismatch (diff x)' or 'n/a' (no total found).
    """
    clean, rules = remove_grid(table_img)   # grid borders are erased, rule positions kept for row cutting
    best, attempts = None, []
    for scale, psm in OCR_ATTEMPTS:
        items = parse_items(ocr_words(clean, scale=scale, psm=psm), rules, clean.shape[0])
        diff = _sum_diff(items, targets)
        if diff is None:
            return _public(items), "n/a"
        attempts.append(items)
        if best is None or diff < best[0]:
            best = (diff, items)
        if items and diff <= 0.05 + 0.01 * len(items):
            return _public(items), "ok"

    merged = {}
    for items in attempts:
        for it in items:
            if it["_band"] is not None:
                merged.setdefault(it["_band"], it)
    if merged:
        items = [merged[k] for k in sorted(merged)]
        diff = _sum_diff(items, targets)
        if diff < best[0]:
            best = (diff, items)
        if diff <= 0.05 + 0.01 * len(items):
            return _public(items), "ok (rows combined from several OCR runs)"
    return _public(best[1]), f"mismatch (diff {best[0]:.2f})"


def extract_invoice(table_img, other_imgs):
    """table_img: the items table zone. other_imgs: {zone_name: image} of the three other zones.

    The other zones are classified (header / parties / summary) only by the words in them.
    Returns (data, labels, check): labels maps zone names to categories, check is the result of
    the arithmetic validation (items add up to the total).
    """
    words = {name: ocr_words(img) for name, img in other_imgs.items()}
    texts = {name: words_to_text(w) for name, w in words.items()}
    cat_to_zone = classify_zones(texts)

    header = parse_header(texts[cat_to_zone["header"]]) if "header" in cat_to_zone else {}
    parties = parse_parties(words[cat_to_zone["parties"]]) if "parties" in cat_to_zone else {}
    summary = parse_summary(texts[cat_to_zone["summary"]]) if "summary" in cat_to_zone else {}
    items, check = parse_table_validated(table_img, [summary.get("total"), summary.get("sub")])

    data = {
        "invoice": {
            "client_name": parties.get("client_name"),
            "client_address": parties.get("client_address"),
            "seller_name": parties.get("seller_name"),
            "seller_address": parties.get("seller_address"),
            "invoice_number": header.get("invoice_number"),
            "invoice_date": header.get("invoice_date"),
        },
        "items": items,
        "subtotal": {"tax": summary.get("tax"), "total": summary.get("total")},
    }
    labels = {zone: cat for cat, zone in cat_to_zone.items()}
    return data, labels, check


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
