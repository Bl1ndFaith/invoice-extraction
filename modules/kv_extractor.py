"""Generic KV extractor: find invoice key fields by label + geometry.

Pipeline position: runs on ZONED pages (see zone_splitter). The rule:

    label (dictionary synonym) -> value = nearest phrase RIGHT of the label
    on the same visual row, or BELOW it (a few rows down, inside the
    label's x-band) — accepted only when it satisfies the field's
    expected value type (date / money / code).

Falls back for unlabeled layouts live inside this module (unlabeled
seller block), NOT as separate profiles/templates.

All functions are pure: data in, data out. No printing, no file IO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

from modules.labels import (
    FIELD_LABELS,
    STRONG_TABLE_HEADER_WORDS,
    TOTALS_FUZZY_WORDS,
    fuzzy_match,
    looks_like_code,
    looks_like_date,
    looks_like_email,
    looks_like_money,
    looks_like_phone,
    looks_like_postal,
)
from modules.ocr_engine import Line, Word
from modules.zone_splitter import Zones

# ---------------------------------------------------------------------------
# Field configuration
# ---------------------------------------------------------------------------

# Each KV field is searched in one zone with one expected-value validator.
FIELD_ZONES: dict[str, str] = {
    "invoice_number": "header",
    "invoice_date": "header",
    "due_date": "header",
}

FIELD_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "invoice_number": looks_like_code,
    "invoice_date": looks_like_date,
    "due_date": looks_like_date,
}

# Order fixes label ambiguity: two-word 'date' labels ('due date',
# 'date of issue') must claim their rows before the bare 'date' synonym runs.
FIELD_ORDER: tuple[str, ...] = ("due_date", "invoice_date", "invoice_number")

PARTY_LABELS: dict[str, tuple[str, ...]] = {
    "seller": FIELD_LABELS["seller"],
    "client": FIELD_LABELS["client"],
}

MAX_LINES_BELOW = 3           # value search depth under a label
MAX_PARTY_LINES = 4           # name + up to 3 address lines
MEDIAN_ROW_MULTIPLIER = 0.9   # same-row y-tolerance vs median word height

# Unlabeled-seller fallback: a row counts as part of the seller block when
# its leftmost word starts in the left part of the page. The Flora template's
# large 'INVOICE' title sits around x~0.59, so 0.5 keeps it out.
UNLABELED_SELLER_X = 0.5
UNLABELED_SELLER_X_RELAXED = 0.65  # only when NO strict candidate exists
BLOCK_GAP_HEIGHTS = 5.0            # vertical gap (in word heights) splitting runs


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def labelize(raw: str) -> str:
    """Normalize text for label matching ('Invoice number:' -> 'invoice number',
    'due_date:' -> 'due date')."""

    cleaned = raw.strip().rstrip(":!.,;")
    cleaned = cleaned.replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).lower()


def normalize_date(raw: str) -> str:
    """'Oct. 11, 2023' / 'Sep 17, 2017' -> '09/17/2017' style; else raw."""

    text = raw.strip().rstrip(",.:;")
    for fmt in (
        "%m/%d/%Y", "%m/%d/%y",
        "%b. %d, %Y", "%b %d, %Y", "%B %d, %Y",
        "%b. %d %Y", "%b %d %Y", "%B %d %Y",
        "%d %b %Y", "%d %B %Y",
    ):
        try:
            return datetime.strptime(text, fmt).strftime("%m/%d/%Y")
        except ValueError:
            continue
    return raw


def normalize_money(raw: str) -> str:
    """'$4573' -> '4573'; single values switch to dot decimals ('232,95'
    -> '232.95'); values with thousand groups keep OCR rendering
    ('15 339,50' stays)."""

    cleaned = raw.replace("$", "").strip()
    if " " in cleaned:
        return cleaned
    return cleaned.replace(",", ".")


def page_width(lines: Sequence[Line]) -> float:
    rights = [
        word.left + word.width for line in lines for word in line.words
    ]
    return float(max(rights)) if rights else 0.0


def zone_words(lines: Sequence[Line], zones: Zones, zone_name: str) -> list[Word]:
    top_bottom_pairs = zones.band(zone_name)
    return [
        word for line in lines for word in line.words
        if any(top <= word.y_center < bottom for top, bottom in top_bottom_pairs)
    ]


def visual_rows(words: Sequence[Word]) -> list[list[Word]]:
    """Group zone words into visual rows by vertical center proximity.

    Tesseract splits a printed row into several OCR Lines (one per column),
    so rows are rebuilt here with y-center clustering instead of trusting
    the block/line hierarchy.
    """

    if not words:
        return []

    heights = sorted(word.height for word in words)
    median_height = heights[len(heights) // 2]
    tolerance = max(median_height * MEDIAN_ROW_MULTIPLIER, 6.0)

    ordered = sorted(words, key=lambda w: (w.y_center, w.left))

    rows: list[list[Word]] = []
    anchor_y: float | None = None

    for word in ordered:
        if anchor_y is None or abs(word.y_center - anchor_y) > tolerance:
            rows.append([word])
            anchor_y = word.y_center
        else:
            rows[-1].append(word)

    for row in rows:
        row.sort(key=lambda w: w.left)

    return rows


def rows_in_zone(lines: Sequence[Line], zones: Zones, zone_name: str) -> list[list[Word]]:
    return visual_rows(zone_words(lines, zones, zone_name))


def _row_has_claim(row: Sequence[Word], claims: set[int]) -> bool:
    return any(id(word) in claims for word in row)


def _row_is_stop(row: Sequence[Word]) -> bool:
    """Rows that end a party block: 'Tax Id:' or a table-header word."""

    for word in row:
        text = labelize(word.text)
        if text.startswith("tax") or text == "items":
            return True
        if text in STRONG_TABLE_HEADER_WORDS or (
            fuzzy_match(text, STRONG_TABLE_HEADER_WORDS, min_candidate_len=6)
            is not None
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Label matching
# ---------------------------------------------------------------------------


@dataclass
class LabelHit:
    field: str
    synonym: str
    words: list[Word]    # matched label words, in reading order
    row_index: int


def _match_status(word: Word, token: str) -> str:
    """Match one label token to a word. '' = no match.

    'prefix-rest' covers glued tokens such as 'Invoice:102' against the
    single-token label 'invoice': the remainder becomes a value candidate.
    'fuzzy' covers OCR-mangled tokens ('invoce nomber') within the edit
    budget — the value side is still gated by the type validator.
    """

    text = labelize(word.text)
    if text == token:
        return "yes"
    if " " not in token and text.startswith(token + ":"):
        return "prefix-rest"
    if fuzzy_match(text, {token}) is not None:
        return "fuzzy"
    return ""


def find_label_hits(
    rows: list[list[Word]], synonyms: Sequence[str]
) -> list[LabelHit]:
    """Find label occurrences, longest synonyms first (avoids 'date' traps).

    A hit is a run of consecutive row words matching the synonym's tokens;
    a glued last token ('Invoice:102') is allowed on the final position.
    """

    hits: list[LabelHit] = []

    for synonym in sorted(synonyms, key=lambda s: -len(s)):
        tokens = labelize(synonym).split()
        if not tokens:
            continue

        for row_index, row in enumerate(rows):
            for start in range(len(row) - len(tokens) + 1):
                window = row[start : start + len(tokens)]
                statuses = [
                    _match_status(word, token)
                    for word, token in zip(window, tokens)
                ]

                if any(status == "" for status in statuses):
                    continue
                # Glued remainder only makes sense on the LAST label token.
                if any(status == "prefix-rest" for status in statuses[:-1]):
                    continue

                hits.append(
                    LabelHit(
                        field="",
                        synonym=synonym,
                        words=list(window),
                        row_index=row_index,
                    )
                )

    return hits


# ---------------------------------------------------------------------------
# Value search
# ---------------------------------------------------------------------------


def _iter_value_phrases(words: Sequence[Word]):
    """Candidate phrases: the whole joined text first, then every slice.

    Joined-first repairs values the OCR split into pieces ('15' + '339,50');
    slices rescue values polluted by neighbour columns in the same row.
    """

    texts = [word.text for word in words]

    yield " ".join(texts).strip()
    for i in range(len(texts)):
        for j in range(i + 1, len(texts) + 1):
            yield " ".join(texts[i:j]).strip()


def _pick_phrase(words: Sequence[Word], validator: Callable[[str], bool]) -> str | None:
    for candidate in _iter_value_phrases(words):
        if candidate and validator(candidate):
            return candidate
    return None


def _value_right_of_row(
    row: Sequence[Word], label_words: Sequence[Word], claims: set[int],
    validator: Callable[[str], bool],
) -> str | None:
    cutoff = max(word.left + word.width for word in label_words)
    pool = [
        word for word in row
        if word.left >= cutoff and id(word) not in claims
    ]
    return _pick_phrase(pool, validator)


def _value_below_label(
    rows: list[list[Word]], hit: LabelHit, claims: set[int],
    validator: Callable[[str], bool],
) -> str | None:
    """Value under the label (Umbra: 'INVOICE NUMBER' -> '564234' one row down).

    Overlap window: the label's x-band widened by the label height on the
    left and 4x height on the right, so column neighbours stay out while
    values aligned with the label stay in.
    """

    label_left = min(word.left for word in hit.words)
    label_right = max(word.left + word.width for word in hit.words)
    label_height = max(word.height for word in hit.words)

    for row in rows[hit.row_index + 1 : hit.row_index + 1 + MAX_LINES_BELOW]:
        pool = [
            word for word in row
            if word.left + word.width >= label_left - label_height
            and word.left <= label_right + label_height * 4
            and id(word) not in claims
        ]
        candidate = _pick_phrase(pool, validator)
        if candidate is not None:
            return candidate
    return None


def extract_label_value(
    rows: list[list[Word]], hit: LabelHit, claims: set[int],
    validator: Callable[[str], bool],
) -> str | None:
    """Value right of the label, else below it (few rows, x-overlap)."""

    value = _value_right_of_row(rows[hit.row_index], hit.words, claims, validator)
    if value is not None:
        return value

    # Glued label ('Invoice:102'): the remainder after ':' is the value.
    last_word = hit.words[-1]
    last_token = labelize(hit.synonym.split()[-1])
    if ":" in last_word.text and ":" not in last_token:
        tail = last_word.text.split(":")[1].strip().rstrip(",.:;")
        if tail and validator(tail):
            return tail

    return _value_below_label(rows, hit, claims, validator)


# ---------------------------------------------------------------------------
# Totals zone
# ---------------------------------------------------------------------------


def _row_totals_label(row: Sequence[Word]) -> str | None:
    texts = [labelize(word.text) for word in row]

    # Fuzzy pass for OCR-mangled totals labels ('subtoal', 'tota1',
    # 'hendling'); short ambiguous words ('tax', 'due') stay exact-only.
    fuzzy_hits = set()
    for text in texts:
        matched = fuzzy_match(text, TOTALS_FUZZY_WORDS)
        if matched is not None:
            fuzzy_hits.add(matched)
    texts = texts + sorted(fuzzy_hits)

    # Shipping & Handling rows carry money but are not part of the schema;
    # they are still captured (internal key) for arithmetic reconciliation.
    if any(text.startswith("hand") for text in texts):
        return "shipping"

    if "subtotal" in texts or ("sub" in texts and "total" in texts):
        return "subtotal"
    if "discount" in texts:
        return "discount"
    if "tax" in texts or "vat" in texts:
        return "tax"
    # 'total' as an exact word; 'due alone' covers seriously mangled OCR
    # rows such as '| torat DUE ove' (Umbra 'TOTAL DUE').
    if "total" in texts or "due" in texts:
        return "total"
    return None


def _row_amounts(row: Sequence[Word]) -> list[str]:
    """Monetary values on a row, re-joining thousand groups split by OCR.

    '15' + '339,50' printed as '15 339,50' -> one amount, when the group is
    a <=3-digit integer followed by an adjacent decimal group.
    """

    amounts: list[str] = []
    skip_next = False

    for index, word in enumerate(row):
        text = word.text.strip()
        if skip_next:
            skip_next = False
            continue
        if "%" in text or text in {"$", "€", "£"}:
            continue

        cleaned = text.replace("$", "").strip().lstrip(":,;.")
        if not cleaned:
            continue

        next_word = row[index + 1] if index + 1 < len(row) else None
        gap = (
            next_word.left - (word.left + word.width)
            if next_word is not None
            else 10**9
        )

        if (
            len(cleaned) <= 3
            and cleaned.isdigit()
            and next_word is not None
            and looks_like_money(next_word.text)
            and gap <= word.height * 2
        ):
            amounts.append(f"{cleaned} {next_word.text.replace('$', '').strip()}")
            skip_next = True
            continue

        if looks_like_money(cleaned):
            amounts.append(cleaned)

    return amounts


def extract_totals(lines: Sequence[Line], zones: Zones) -> dict[str, str]:
    """subtotal / tax / total from the totals zone.

    Priority: subtotal rows first, then tax rows, then total rows. When a
    total row carries two amounts and no tax row exists, the second-to-last
    amount is the tax (batch1 layout: '... Tax ... Total' merged rows).

    shipping/discount are internal keys (NOT in the JSON schema) used only
    by the arithmetic reconciliation.
    """

    result = {"subtotal": "", "tax": "", "total": "", "shipping": "", "discount": ""}
    total_row_amounts: list[str] | None = None

    for row in rows_in_zone(lines, zones, "totals"):
        label = _row_totals_label(row)
        if label is None:
            continue

        amounts = _row_amounts(row)
        if not amounts:
            continue

        if label == "total":
            total_row_amounts = amounts

        if not result[label]:
            result[label] = normalize_money(amounts[-1])

    # batch1 style: no explicit tax row -> second-to-last amount of the
    # exact-'total' row is the tax amount.
    if not result["tax"] and total_row_amounts and len(total_row_amounts) >= 2:
        result["tax"] = normalize_money(total_row_amounts[-2])

    return result


# ---------------------------------------------------------------------------
# Parties (seller / client blocks)
# ---------------------------------------------------------------------------


def _row_text(row: Sequence[Word]) -> str:
    return " ".join(word.text for word in row)


def _row_leftmost_x(row: Sequence[Word]) -> float:
    first = min(row, key=lambda w: w.left)
    return first.x_center


def _row_is_title(row: Sequence[Word]) -> bool:
    """Standalone 'invoice' title rows must not enter party blocks."""

    return any(labelize(word.text) == "invoice" for word in row)


def _collect_party_lines(
    rows: list[list[Word]],
    start_index: int,
    claims: set[int],
    boundary: float | None,
    side: str,
    claim_exempt_row: int | None = None,
) -> list[str]:
    """Party column text lines: name line first, then address lines.

    boundary/side split a two-column layout; boundary=None keeps whole rows
    (single-column blocks). Label words and claimed KV words never appear,
    and a row carrying any claimed label word ends the block. The exempted
    row (its own label row) may hold claimed words without ending anything.
    """

    def member(word: Word) -> bool:
        if id(word) in claims:
            return False
        if boundary is None:
            return True
        return word.x_center < boundary if side == "left" else word.x_center >= boundary

    lines_out: list[str] = []

    for index in range(start_index, min(start_index + MAX_PARTY_LINES * 2, len(rows))):
        row = rows[index]
        if _row_is_stop(row):
            break
        # A claimed label row is a block boundary (KV label or another
        # party label just above); only the exempted row continues.
        if index != (start_index if claim_exempt_row is None else claim_exempt_row):
            if _row_has_claim(row, claims):
                break
        if _row_is_title(row):
            continue

        members = [word for word in row if member(word)]
        if members:
            lines_out.append(_row_text(members))
        if len(lines_out) >= MAX_PARTY_LINES:
            break

    return lines_out


def _parties_from_lines(lines_out: list[str]) -> dict[str, str]:
    return {
        "name": lines_out[0] if lines_out else "",
        "address": "\n".join(lines_out[1:]),
    }


def _parties_two_columns(
    rows: list[list[Word]], seller_hit: LabelHit, client_hit: LabelHit,
    claims: set[int],
) -> dict[str, str]:
    """Both parties labeled: batch1 / Umbra two-column layout.

    Sides follow the label positions (most templates put the seller left,
    but 'Vendor' right / 'Bill To' left layouts exist just as well).
    """

    seller_right = max(word.left + word.width for word in seller_hit.words)
    client_left = min(word.left for word in client_hit.words)

    if client_left > seller_right:
        boundary = (seller_right + client_left) / 2
    else:
        # Stacked labels: fall back to the page midpoint.
        page_mid = max(
            word.left + word.width for row in rows for word in row
        ) / 2
        boundary = page_mid

    seller_side = (
        "left"
        if seller_hit.words[0].x_center <= client_hit.words[0].x_center
        else "right"
    )
    client_side = "right" if seller_side == "left" else "left"

    start = min(seller_hit.row_index, client_hit.row_index)
    seller = _parties_from_lines(
        _collect_party_lines(rows, start, claims, boundary, seller_side)
    )
    client = _parties_from_lines(
        _collect_party_lines(rows, start, claims, boundary, client_side)
    )
    return {"seller": seller, "client": client}


def _column_gap_boundary(row: Sequence[Word], after: int, page_mid: float) -> float:
    """Left bound of the right KV column relative to a party label row.

    The first word cluster right of the label (small gap) is the inline
    value/name; the NEXT big horizontal gap marks the right column. The
    gap threshold scales with the PAGE WIDTH (15%), because wide-spaced
    fonts on big scans have inter-word gaps that dwarf word heights.
    A label standing alone (huge first gap) has no inline value.
    """

    width = page_mid * 2

    right_words = sorted((w for w in row if w.left > after), key=lambda w: w.left)
    if not right_words:
        return page_mid

    gap_limit = 0.15 * width
    first = right_words[0]

    if first.left - after > gap_limit:
        return float(first.left) - gap_limit / 2

    edge = after
    for word in right_words[1:]:
        if word.left - edge > gap_limit:
            return float(word.left) - gap_limit / 2
        edge = max(edge, word.left + word.width)
    return page_mid


def _party_block_below(
    rows: list[list[Word]], client_hit: LabelHit, claims: set[int],
    page_mid: float,
) -> dict[str, str]:
    """Client-only layout (Flora): name inline right of the label, address
    block below — everything left of the right KV column."""

    label_row = rows[client_hit.row_index]
    label_end = client_hit.words[-1]

    boundary = _column_gap_boundary(
        label_row, label_end.left + label_end.width, page_mid
    )

    # Inline name: words right of the label, before the column gap.
    inline = [
        word for word in label_row
        if id(word) not in claims
        and word.left >= label_end.left + label_end.width
        and word.x_center < boundary
    ]
    name = _row_text(inline) if inline else ""

    lines_out = _collect_party_lines(
        rows, client_hit.row_index + 1, claims, boundary, "left",
        claim_exempt_row=client_hit.row_index,
    )
    if name:
        lines_out = [name] + lines_out
    return _parties_from_lines(lines_out)


def _row_top_y(row: Sequence[Word]) -> float:
    return float(min(word.top for word in row))


def _row_is_address_marked(row: Sequence[Word]) -> bool:
    """Address evidence without labels: postal code / phone / e-mail."""

    return any(
        looks_like_postal(word.text)
        or looks_like_phone(word.text)
        or looks_like_email(word.text)
        for word in row
    )


def _label_free_runs(
    rows: list[list[Word]],
    cut_index: int,
    claims: set[int],
    side: str,
    page_width: float,
    boundary: float | None = None,
    relaxed: bool = False,
) -> list[tuple[list[str], bool]]:
    """Runs of consecutive label-free rows on one page side.

    side='left': rows whose leftmost word starts in the left part of the
    page (seller blocks). side='right': rows whose text reaches into the
    right part (client blocks). Runs break on stop rows, claimed rows,
    title rows and vertical gaps. Each run carries an 'address-marked'
    flag — without a label that is the only party evidence.
    """

    threshold = page_width * (
        UNLABELED_SELLER_X_RELAXED if relaxed else UNLABELED_SELLER_X
    )

    def qualifying(row: Sequence[Word]) -> list[Word] | None:
        if _row_is_stop(row) or _row_has_claim(row, claims):
            return None
        if _row_is_title(row):
            return None
        if side == "left":
            if _row_leftmost_x(row) > threshold:
                return None
        else:
            right_edge = max(word.left + word.width for word in row)
            if right_edge < page_width - threshold:
                return None
        pool = [
            word for word in row
            if id(word) not in claims
            and (
                boundary is None
                or (
                    word.x_center < boundary
                    if side == "left"
                    else word.x_center >= boundary
                )
            )
        ]
        return pool or None

    runs: list[tuple[list[str], bool]] = []
    current: list[str] = []
    current_marked = False

    heights = [
        word.height for row in rows[:cut_index] for word in row
    ]
    median_height = sorted(heights)[len(heights) // 2] if heights else 0
    gap_limit = BLOCK_GAP_HEIGHTS * median_height

    prev_top: float | None = None

    for row in rows[:cut_index]:
        pool = qualifying(row)
        top = _row_top_y(row)

        if pool is None:
            if current:
                runs.append((current, current_marked))
                current, current_marked = [], False
            prev_top = None
            continue

        if prev_top is not None and top - prev_top > gap_limit:
            runs.append((current, current_marked))
            current, current_marked = [], False

        current.append(_row_text(pool))
        current_marked = current_marked or _row_is_address_marked(row)
        prev_top = top
    if current:
        runs.append((current, current_marked))

    return runs


def _unlabeled_seller(
    rows: list[list[Word]], cut_index: int, claims: set[int],
    page_width: float, boundary: float | None, relaxed: bool,
) -> dict[str, str]:
    """Seller block without any label (Flora, Geonics).

    The LAST left-side run is the seller block — the text nearest the
    client label row (Flora) or the only label-free block at the top
    (Geonics). Name = first run row, address = the following run rows.
    """

    runs = _label_free_runs(
        rows, cut_index, claims, "left", page_width, boundary, relaxed
    )
    if not runs:
        return {"name": "", "address": ""}

    last = runs[-1][0]
    return {"name": last[0], "address": "\n".join(last[1:])}


def _unlabeled_client(
    rows: list[list[Word]], cut_index: int, claims: set[int],
    page_width: float, boundary: float | None = None, relaxed: bool = False,
) -> dict[str, str]:
    """Client block without any label.

    Mirror of the seller fallback on the right page half, but GATED by
    address evidence: at least one run must contain a postal code, phone
    or e-mail, otherwise any right-side text could be claimed as a party.
    """

    runs = _label_free_runs(
        rows, cut_index, claims, "right", page_width, boundary, relaxed
    )
    marked = [lines for lines, is_marked in runs if is_marked]
    if not marked:
        return {"name": "", "address": ""}

    last = marked[-1]
    return {"name": last[0], "address": "\n".join(last[1:])}


def extract_parties(
    lines: Sequence[Line], zones: Zones, claims: set[int]
) -> dict[str, str]:
    """seller/client name+address from the header KV region.

    The region is order-agnostic: party labels are searched across BOTH
    header intervals (above the table and, for table-first layouts,
    between the table body and totals).

    Case 1: both labeled      -> two-column split (batch1, Umbra).
    Case 2: only client       -> client block below label; seller falls back
                                 to the unlabeled top-left block (Flora).
    Case 3: only seller       -> seller block below label; client falls
                                 back to an address-marked right-half block.
    Case 4: neither labeled   -> the top text block is the seller; a
                                 right-half block with address evidence
                                 becomes the client (Geonics-style).
    """

    rows = rows_in_zone(lines, zones, "header")
    pw = page_width(lines)
    mid = pw / 2

    # Region split for the unlabeled fallbacks: prefer rows above the
    # table; when the table starts at the very top, use the post-table
    # band instead (table-first layouts put parties below the table).
    rows_above = [row for row in rows if _row_top_y(row) < zones.items_top_y]
    rows_post = [
        row for row in rows if _row_top_y(row) >= zones.items_bottom_y
    ]

    claims = set(claims)

    seller_hits = [
        hit for hit in find_label_hits(rows, PARTY_LABELS["seller"])
        if not any(id(word) in claims for word in hit.words)
    ]
    client_hits = [
        hit for hit in find_label_hits(rows, PARTY_LABELS["client"])
        if not any(id(word) in claims for word in hit.words)
    ]

    result = {
        "seller_name": "",
        "seller_address": "",
        "client_name": "",
        "client_address": "",
    }

    if seller_hits and client_hits:
        for hit in (*seller_hits, *client_hits):
            claims.update(id(word) for word in hit.words)

        parties = _parties_two_columns(rows, seller_hits[0], client_hits[0], claims)
        result["seller_name"] = parties["seller"]["name"]
        result["seller_address"] = parties["seller"]["address"]
        result["client_name"] = parties["client"]["name"]
        result["client_address"] = parties["client"]["address"]
        return result

    if client_hits:
        hit = client_hits[0]
        claims.update(id(word) for word in hit.words)

        client = _party_block_below(rows, hit, claims, mid)
        result["client_name"] = client["name"]
        result["client_address"] = client["address"]

        seller = _unlabeled_seller(
            rows, hit.row_index, claims, pw, None, relaxed=False
        )
        result["seller_name"] = seller["name"]
        result["seller_address"] = seller["address"]
        return result

    if seller_hits:
        hit = seller_hits[0]
        claims.update(id(word) for word in hit.words)

        seller = _collect_party_lines(
            rows, hit.row_index, claims, None, "left"
        )
        parties = _parties_from_lines(seller)
        result["seller_name"] = parties["name"]
        result["seller_address"] = parties["address"]

        client = _unlabeled_client(rows, len(rows), claims, pw)
        result["client_name"] = client["name"]
        result["client_address"] = client["address"]
        return result

    # Case 4: neither labeled. Prefer the region above the table; only a
    # fully table-first page falls back to the post-table band.
    for region in (rows_above, rows_post):
        if not region:
            continue

        seller = _unlabeled_seller(
            region, len(region), claims, pw, None, relaxed=True
        )
        if not seller["name"]:
            continue

        result["seller_name"] = seller["name"]
        result["seller_address"] = seller["address"]

        client = _unlabeled_client(region, len(region), claims, pw)
        result["client_name"] = client["name"]
        result["client_address"] = client["address"]
        break

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_kv_fields(lines: Sequence[Line], zones: Zones) -> dict[str, str]:
    """All non-table fields: header labels + totals zone + party blocks.

    Returns: invoice_number, invoice_date, due_date, subtotal, tax, total,
    seller_name, seller_address, client_name, client_address (all strings).
    """

    claims: set[int] = set()

    result: dict[str, str] = {
        "invoice_number": "",
        "invoice_date": "",
        "due_date": "",
        "subtotal": "",
        "tax": "",
        "total": "",
        "seller_name": "",
        "seller_address": "",
        "client_name": "",
        "client_address": "",
    }

    header_rows = rows_in_zone(lines, zones, "header")

    for field in FIELD_ORDER:
        validator = FIELD_VALIDATORS[field]
        hits = [
            hit for hit in find_label_hits(header_rows, FIELD_LABELS[field])
            if not any(id(word) in claims for word in hit.words)
        ]

        chosen = None
        value = None
        for hit in hits:
            candidate = extract_label_value(header_rows, hit, claims, validator)
            if candidate is not None:
                chosen, value = hit, candidate
                break

        if chosen is not None and value is not None:
            claims.update(id(word) for word in chosen.words)
            if field.endswith("date"):
                result[field] = normalize_date(value)
            else:
                result[field] = value

    result.update(extract_totals(lines, zones))
    result.update(extract_parties(lines, zones, claims))

    return result
