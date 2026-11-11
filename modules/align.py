"""Alignment of CSV ground-truth JSON to OCR words -> BIO labels.

Turns the `Json Data` column of a GT CSV row into a per-word label
sequence over the OCR words of the same document. Alignment is
geometry-aware: multi-line values (addresses, item descriptions) are
first searched as contiguous runs *inside a single OCR line* — invoice
pages have two-column address blocks whose words interleave in full-page
reading order — and only fall back to a whole-page window search when no
single-line match clears the similarity threshold.

Ambiguity policy: a match is accepted only if its word span does not
overlap an already-assigned span; otherwise the tokens stay O and the
miss is counted in the coverage stats. Phase 1 favors label precision
over coverage — a wrongly labeled span teaches the model worse than a
missing one.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .ocr_engine import Word, group_words_into_lines

# --- label set (single source of truth for model + align + infer) -----------

LABELS: list[str] = [
    "O",
    "B-SELLER_NAME", "I-SELLER_NAME",
    "B-SELLER_ADDRESS", "I-SELLER_ADDRESS",
    "B-CLIENT_NAME", "I-CLIENT_NAME",
    "B-CLIENT_ADDRESS", "I-CLIENT_ADDRESS",
    "B-INVOICE_NUMBER",
    "B-INVOICE_DATE",
    "B-DUE_DATE",
    "B-TAX",
    "B-DISCOUNT",
    "B-TOTAL",
    "B-ITEM_QUANTITY",
    "B-ITEM_TOTAL_PRICE",
    "B-ITEM_DESC", "I-ITEM_DESC",
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for index, label in enumerate(LABELS)}

# GT key -> (entity base name, supports I- continuation, prefer last match)
# Money fields appear both in item rows and in the summary/Total rows;
# the GT value refers to the summary, i.e. the LAST occurrence on the page.
SCALAR_FIELDS: dict[str, tuple[str, bool, bool]] = {
    "seller_name": ("SELLER_NAME", True, False),
    "seller_address": ("SELLER_ADDRESS", True, False),
    "client_name": ("CLIENT_NAME", True, False),
    "client_address": ("CLIENT_ADDRESS", True, False),
    "invoice_number": ("INVOICE_NUMBER", False, False),
    "invoice_date": ("INVOICE_DATE", False, False),
    "due_date": ("DUE_DATE", False, False),
    "tax": ("TAX", False, True),
    "discount": ("DISCOUNT", False, True),
    "total": ("TOTAL", False, True),
}

MATCH_THRESHOLD = 0.80   # SequenceMatcher ratio for text windows
MIN_CHUNK_WORDS = 2      # recursion floor for description-line splitting


# --- normalization ----------------------------------------------------------

def normalize_text(value: str) -> str:
    """Collapse whitespace, lowercase (style-insensitive comparisons)."""
    return " ".join(str(value).split()).lower()


def normalize_number(value: str) -> str:
    """Normalize a numeric literal: no spaces, European comma -> dot."""
    return re.sub(r"\s+", "", str(value)).replace(",", ".")


def flatten_gt(gt: dict) -> tuple[dict[str, str], list[dict]]:
    """Split a GT JSON document into scalar fields + the items array."""

    invoice = gt.get("invoice", {}) or {}
    subtotal = gt.get("subtotal", {}) or {}

    scalars = {
        key: str((invoice if key in invoice else subtotal).get(key, "") or "")
        for key in SCALAR_FIELDS
    }
    return scalars, gt.get("items", []) or []


# --- matching primitives ----------------------------------------------------

def _window_scores(
    texts: list[str], target: str, sizes: range, prefer_last: bool = False
) -> tuple[int, int, float] | None:
    """Best (start, end, ratio) over windows of the given sizes.

    prefer_last: on ties in ratio, prefer the window that occurs later in
    the text (money values repeat down the page; the GT meaning lives in
    the last one — the summary 'Total' row).
    """

    best: tuple[int, int, float] | None = None

    def _better(new: tuple[int, int, float], old: tuple[int, int, float]) -> bool:
        if prefer_last:
            return new[2] > old[2] or (new[2] == old[2] and new[0] > old[0])
        return new[2] > old[2]

    for size in sizes:
        for start in range(0, len(texts) - size + 1):
            candidate = " ".join(texts[start : start + size])
            ratio = SequenceMatcher(None, target, candidate).ratio()
            if ratio == 1.0 and not prefer_last:
                return start, start + size, 1.0
            if best is None or _better((start, start + size, ratio), best):
                best = (start, start + size, ratio)

    return best


def _match_in_lines(
    line_norms: list[list[str]], target: str, prefer_last: bool = False
) -> tuple[int, int, float] | None:
    """Search `target` as a contiguous run inside one OCR line.

    Returns global (start, end, ratio) using running offsets, or None.
    """

    best: tuple[int, int, float] | None = None
    offset = 0

    for line in line_norms:
        sizes = range(max(1, len(target.split()) - 2), len(target.split()) + 3)
        found = _window_scores(line, target, sizes, prefer_last=prefer_last)
        if found and found[2] >= MATCH_THRESHOLD:
            start, end, ratio = found
            candidate = (offset + start, offset + end, ratio)
            if best is None or _later_or_better(candidate, best, prefer_last):
                best = candidate
        offset += len(line)

    return best


def _later_or_better(
    new: tuple[int, int, float], old: tuple[int, int, float], prefer_last: bool
) -> bool:
    if prefer_last:
        return new[2] > old[2] or (new[2] == old[2] and new[0] > old[0])
    return new[2] > old[2]


def _match_page(
    seq_norms: list[str], target: str, prefer_last: bool = False
) -> tuple[int, int, float] | None:
    """Whole-page contiguous-window fallback."""

    sizes = range(max(1, len(target.split()) - 2), len(target.split()) + 3)
    found = _window_scores(seq_norms, target, sizes, prefer_last=prefer_last)
    if found and found[2] >= MATCH_THRESHOLD:
        return found
    return None


# --- alignment ---------------------------------------------------------------

def align_document(words: list[Word], gt: dict) -> tuple[list[str], dict]:
    """Compute one BIO label per OCR word for a document.

    Returns (labels, coverage_stats) with labels parallel to the reading
    order produced by `_reading_order`.
    """

    seq, lines = _reading_order(words)
    labels = ["O"] * len(seq)
    occupied: set[int] = set()
    stats: dict[str, int] = {}

    # Lowercase once: targets are normalized, so candidates must be too
    # (SequenceMatcher is case-sensitive; raw text would drop the ratio
    # below threshold for differently-cased matches like 'Becker Ltd').
    line_norms = [[word.text.lower() for word in line.words] for line in lines]
    page_norms = [word.text.lower() for word in seq]

    scalars, items = flatten_gt(gt)

    # -- scalar header / subtotal fields ------------------------------------
    for key, (entity, has_i, prefer_last) in SCALAR_FIELDS.items():
        value = scalars.get(key, "")
        gt_lines = [line for line in str(value).split("\n") if line.strip()]
        if not gt_lines:
            continue

        stats[f"{key}_total"] = len(gt_lines)
        matched_any = 0

        for gt_line in gt_lines:
            target = normalize_text(gt_line)
            found = _match_in_lines(line_norms, target, prefer_last=prefer_last)
            if found is None:
                found = _match_page(page_norms, target, prefer_last=prefer_last)

            if found is None:
                continue

            start, end, _ratio = found
            span = range(start, end)
            if any(index in occupied for index in span):
                continue

            for index in span:
                occupied.add(index)
                labels[index] = f"B-{entity}" if index == start else (
                    f"I-{entity}" if has_i else f"B-{entity}"
                )
            matched_any += 1

        stats[f"{key}_matched"] = matched_any

    # -- items ---------------------------------------------------------------
    # 1) description lines: collect every candidate match, then assign the
    #    best-scoring non-overlapping ones (item descriptions share lines
    #    across items, so pure first-fit would mislabel).
    #
    #    GT descriptions come in two flavors: with explicit \n wraps (each
    #    line matches a visual OCR line) and as ONE long logical line that
    #    OCR wraps with money tokens interleaved. The latter has no
    #    contiguous window, so unmatched lines are recursively split in
    #    half until fragments find contiguous runs on the page.
    candidates: list[tuple[float, int, int, int]] = []  # score, start, end, item_idx

    def _find_desc_chunks(target: str) -> list[tuple[int, int, float]]:
        found = _match_page(page_norms, target)
        if found:
            return [found]
        words = target.split()
        if len(words) <= MIN_CHUNK_WORDS:
            return []
        mid = len(words) // 2
        left = " ".join(words[:mid])
        right = " ".join(words[mid:])
        return _find_desc_chunks(left) + _find_desc_chunks(right)

    for item_idx, item in enumerate(items):
        description = str(item.get("description", "") or "")
        for gt_line in description.split("\n"):
            target = normalize_text(gt_line)
            if not target:
                continue
            full = _match_in_lines(line_norms, target) or _match_page(page_norms, target)
            found_list = [full] if full else _find_desc_chunks(target)
            for found in found_list:
                score, start, end = found[2], found[0], found[1]
                candidates.append((score, start, end, item_idx))

    item_desc_words: dict[int, list[int]] = {}
    stats["item_desc_lines_total"] = len(candidates)

    for score, start, end, item_idx in sorted(candidates, reverse=True):
        span = range(start, end)
        if any(index in occupied for index in span):
            continue
        for index in span:
            occupied.add(index)
            labels[index] = "B-ITEM_DESC" if index == start else "I-ITEM_DESC"
            item_desc_words.setdefault(item_idx, []).append(index)
        stats.setdefault("item_desc_lines_matched", 0)
        stats["item_desc_lines_matched"] += 1

    # 2) quantity / total_price: exact normalized-number tokens inside the
    #    vertical band of the item's description; qty -> leftmost, gross
    #    price -> rightmost (they are the outer money columns).
    if seq:
        band_pad = max(word.height for word in seq) / 2.0
    else:
        band_pad = 0.0

    def _band_label(item: dict, key: str, label: str, pick) -> bool:
        """Label qty/price within the item's description band."""

        target = normalize_number(item.get(key, ""))
        if not target:
            return True  # nothing to label, not a miss

        in_band = [
            index
            for index, word in enumerate(seq)
            if index not in occupied
            and normalize_number(word.text) == target
            and band_top <= word.top + word.height / 2 <= band_bottom
        ]
        if not in_band:
            return False

        chosen = pick(in_band, key=lambda i: seq[i].x_center)
        occupied.add(chosen)
        labels[chosen] = label
        return True

    def _column_fallback(item: dict, key: str, label: str) -> bool:
        """X-column fallback for items without a description band.

        Item values share a vertical column (qty left of money columns,
        gross worth rightmost); anchor on the x-center of already-labeled
        tokens of the same kind and take the unused match nearest to it.
        """

        target = normalize_number(item.get(key, ""))
        if not target:
            return False

        anchor_xs = [
            seq[i].x_center for i, existing in enumerate(labels) if existing == label
        ]
        if not anchor_xs:
            return False

        anchor = sorted(anchor_xs)[len(anchor_xs) // 2]
        tolerance = 2.5 * (sorted(w.width for w in seq)[len(seq) // 2])

        candidates = [
            index
            for index, word in enumerate(seq)
            if index not in occupied
            and normalize_number(word.text) == target
            and abs(word.x_center - anchor) <= tolerance
        ]
        if not candidates:
            return False

        chosen = min(candidates, key=lambda i: seq[i].x_center)
        occupied.add(chosen)
        labels[chosen] = label
        return True

    for item_idx, item in enumerate(items):
        desc_indices = item_desc_words.get(item_idx, [])

        if desc_indices:
            band_top = min(seq[i].top for i in desc_indices) - band_pad
            band_bottom = max(seq[i].top + seq[i].height for i in desc_indices) + band_pad

        for key, label, pick in (
            ("quantity", "B-ITEM_QUANTITY", min),
            ("total_price", "B-ITEM_TOTAL_PRICE", max),
        ):
            if not normalize_number(item.get(key, "")):
                continue
            stats[f"item_{key}_total"] = stats.get(f"item_{key}_total", 0) + 1

            matched = (
                _band_label(item, key, label, pick)
                if desc_indices
                else False
            )
            if matched:
                stats[f"item_{key}_matched"] = stats.get(f"item_{key}_matched", 0) + 1
            elif _column_fallback(item, key, label):
                # no description band or the value is outside it: anchor on
                # the shared x-column of the already-labeled tokens instead
                stats[f"item_{key}_matched"] = stats.get(f"item_{key}_matched", 0) + 1

    return labels, stats


def _reading_order(words: list[Word]) -> tuple[list[Word], list]:
    """Flatten OCR words into reading order (line by line, left to right)."""

    lines = group_words_into_lines(words)
    seq: list[Word] = []
    ordered_lines = []

    for line in lines:
        row = sorted(line.words, key=lambda word: word.left)
        ordered_lines.append(type(line)(words=row, block_num=line.block_num,
                                        par_num=line.par_num, line_num=line.line_num))
        seq.extend(row)

    return seq, ordered_lines


def words_to_records(seq: list[Word], labels: list[str]) -> list[dict]:
    """Serialize the aligned sequence for the dataset cache."""

    return [
        {
            "text": word.text,
            "box": [word.left, word.top, word.left + word.width, word.top + word.height],
            "label": label,
        }
        for word, label in zip(seq, labels)
    ]
