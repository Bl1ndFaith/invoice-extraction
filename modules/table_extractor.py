"""Table engine: line items from the items zone.

Zone-driven replacement for the legacy anchor extraction. The idea is the
same as for KV fields — labels from the shared dictionary, geometry and
type validators — but the object is positional columns:

    header words (Description | Qty | ...) -> column anchors
    data-driven money columns -> right-aligned token clusters spanning
                 >=2 rows (the modern, header-independent way)
    item rows -> anchored by the Qty column ('qty'), else by the
                 rightmost money column ('column', the total), else by
                 the first price token per row ('fallback')
    values    -> quantity = the Qty token (integer quantities only ride
                 along when rows are money-column anchored)
              gross/total = the rightmost money value, with OCR-split
              thousand groups re-joined ('85' + '828,73')
    description -> non-numeric words on the row / between anchors

Ground truth joins wrapped description lines with spaces, single money
values switch to dot decimals, values with thousand groups keep rendering.

All functions are pure: data in, data out. No printing, no file IO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from modules.kv_extractor import labelize, rows_in_zone
from modules.labels import fuzzy_match, looks_like_money
from modules.ocr_engine import Line, Word
from modules.zone_splitter import Zones

DECIMAL_RE = re.compile(r"^-?\d+[.,]\d+$")
INT_RE = re.compile(r"^\d{1,3}$")
ROW_NUMBER_RE = re.compile(r"^\d{1,3}[.)]$")

QUANTITY_COLUMN_WORDS = frozenset({"qty", "quantity"})
MONEY_COLUMN_WORDS = frozenset(
    {"total", "amount", "gross", "worth", "net", "price", "sum", "cost", "rate"}
)

# Tolerances are derived from OCR text itself (median word height, column
# gaps), so the engine survives any preprocessing upscale.
TOL_GAP_SHARE = 0.45      # share of the smallest inter-column gap
TOL_MIN_HEIGHTS = 1.5     # never below this many median word heights


@dataclass
class RawItem:
    description: str = ""
    quantity: str = ""
    total_price: str = ""


def _normalize_decimal(raw: str) -> str:
    """'1,00' -> '1.00', '$232,95' -> '232.95'."""

    cleaned = raw.replace("$", "").replace(" ", "").strip()
    return cleaned.replace(",", ".")


def _format_decimal(raw: str) -> str:
    """Ground-truth split by shape: single values -> dot decimals,
    values with thousand groups keep rendering ('15 339,50')."""

    cleaned = raw.replace("$", "").strip()
    if " " in cleaned:
        return cleaned
    return cleaned.replace(",", ".")


def _median_height(rows: Sequence[Sequence[Word]]) -> float:
    heights = sorted(word.height for row in rows for word in row)
    return float(heights[len(heights) // 2]) if heights else 1.0


def _row_top(row: Sequence[Word]) -> float:
    return float(min(word.top for word in row))


def _nearest_neighbour_rows(
    rows: list[list[Word]], row_index: int, anchor: Word
) -> list[list[Word]]:
    """Immediate visual rows above/below the anchor row, ordered by how
    close their first word's y-center is to the anchor."""

    neighbours: list[tuple[float, list[Word]]] = []
    for delta in (-1, 1):
        index = row_index + delta
        if 0 <= index < len(rows) and rows[index]:
            distance = abs(rows[index][0].y_center - anchor.y_center)
            neighbours.append((distance, rows[index]))

    return [row for _, row in sorted(neighbours, key=lambda pair: pair[0])]


# ---------------------------------------------------------------------------
# Column anchors from the table header
# ---------------------------------------------------------------------------


@dataclass
class TableColumn:
    description_left: float | None = None
    description_word: Word | None = field(default=None, repr=False)
    quantity_x: float | None = None
    money_x: float | None = None
    header_row: int = -1


def _find_columns(rows: list[list[Word]]) -> TableColumn:
    """First header occurrence of each dictionary column, over ALL rows.

    Tesseract may split the header into several OCR lines, so anchors are
    gathered from every row, not only the first one after the zone top.
    Fuzzy variants of long column words are accepted ('quanity' -> qty);
    short ones stay exact to avoid body-text misfires.
    """

    column = TableColumn()

    for row_index, row in enumerate(rows):
        for word in row:
            text = labelize(word.text)

            if (
                text.startswith("description")
                and column.description_word is None
            ):
                column.description_left = float(word.left)
                column.description_word = word

            elif column.quantity_x is None and (
                text in QUANTITY_COLUMN_WORDS
                or fuzzy_match(text, QUANTITY_COLUMN_WORDS, min_candidate_len=6)
                is not None
            ):
                column.quantity_x = word.x_center

            elif text in MONEY_COLUMN_WORDS or (
                fuzzy_match(text, MONEY_COLUMN_WORDS, min_candidate_len=6)
                is not None
            ):
                if column.money_x is None or word.x_center > column.money_x:
                    column.money_x = word.x_center
                    column.header_row = row_index

        if column.header_row == -1 and (
            column.description_word is not None or column.quantity_x is not None
        ):
            column.header_row = row_index

    return column


def _column_tolerance(rows: list[list[Word]], column: TableColumn) -> float:
    """Tolerance for 'word belongs to this column' tests.

    Share of the smallest gap between known column anchors, floored by
    1.5 median word heights (tiny gaps between neighboring labels exist).
    """

    anchors = [
        x for x in (column.description_word.x_center if column.description_word else None,
                    column.quantity_x, column.money_x)
        if x is not None
    ]
    median_h = _median_height(rows)

    if len(anchors) >= 2:
        gaps = [b - a for a, b in zip(sorted(anchors), sorted(anchors)[1:]) if b > a]
        base = min(gaps) * TOL_GAP_SHARE if gaps else 2.0 * median_h
    else:
        base = 2.0 * median_h

    return max(base, TOL_MIN_HEIGHTS * median_h)


# ---------------------------------------------------------------------------
# Numeric tokens on a row
# ---------------------------------------------------------------------------


def _price_word(text: str) -> bool:
    """A price-like token: '$4573'/'$0.00' or a strict decimal '154.00'.

    Bare integers ('01480' part codes, '24 oz.') are NOT prices — they stay
    in descriptions and don't anchor rows.
    """

    if "$" in text:
        return looks_like_money(text)
    return bool(DECIMAL_RE.match(text))


def _money_tokens(words: Sequence[Word], limit_left: float) -> list[str]:
    """Price values on the row right of `limit_left`, thousand groups
    re-joined ('15' + '339,50' adjacent -> '15 339,50'), no percents."""

    numeric = sorted(
        (word for word in words if word.x_center >= limit_left and _price_word(word.text)),
        key=lambda w: w.left,
    )

    values: list[str] = []
    skip_next = False

    for index, word in enumerate(numeric):
        text = word.text.replace("$", "").strip()

        if text in {"", "$"} or re.match(r"^\d+[%]", text):
            continue
        if skip_next:
            skip_next = False
            continue

        next_text = (
            numeric[index + 1].text.replace("$", "").strip()
            if index + 1 < len(numeric)
            else None
        )
        adjacent_gap = (
            numeric[index + 1].left - (word.left + word.width)
            if next_text is not None
            else 10**9
        )

        if (
            INT_RE.match(text)
            and next_text is not None
            and DECIMAL_RE.match(next_text)
            and adjacent_gap <= word.height
        ):
            values.append(f"{text} {next_text}")
            skip_next = True
            continue

        if DECIMAL_RE.match(text) or "$" in word.text:
            values.append(text)

    return values


# ---------------------------------------------------------------------------
# Money columns from the data (header-independent)
# ---------------------------------------------------------------------------


@dataclass
class MoneyColumn:
    """A printed money column: prices are right-aligned, so the shared
    RIGHT edge of a token cluster is the stable column signature."""

    right_edge: float
    x_center: float
    row_indices: set[int]
    tokens: list[tuple[int, Word]]


def _find_money_columns(
    rows: list[list[Word]], median_height: float
) -> list[MoneyColumn]:
    """Cluster price tokens by right edge; clusters spanning >=2 distinct
    rows are printed money columns, sorted left -> right.

    Header words are not needed: a real column shows itself by alignment
    across rows, while money-shaped one-off values (part codes) stay out.
    """

    tokens = [
        (row_index, word)
        for row_index, row in enumerate(rows)
        for word in row
        if _price_word(word.text)
    ]
    if not tokens:
        return []

    tolerance = 1.5 * median_height
    ordered = sorted(tokens, key=lambda pair: pair[1].left + pair[1].width)

    clusters: list[list[tuple[int, Word]]] = [[ordered[0]]]
    for row_index, word in ordered[1:]:
        edge = word.left + word.width
        if edge - (clusters[-1][-1][1].left + clusters[-1][-1][1].width) <= tolerance:
            clusters[-1].append((row_index, word))
        else:
            clusters.append([(row_index, word)])

    columns: list[MoneyColumn] = []
    for cluster in clusters:
        row_indices = {row_index for row_index, _ in cluster}
        if len(row_indices) < 2:
            continue
        right_edge = sum(w.left + w.width for _, w in cluster) / len(cluster)
        x_center = sum(w.x_center for _, w in cluster) / len(cluster)
        columns.append(
            MoneyColumn(
                right_edge=right_edge,
                x_center=x_center,
                row_indices=row_indices,
                tokens=cluster,
            )
        )
    return columns


# ---------------------------------------------------------------------------
# Item row anchors
# ---------------------------------------------------------------------------


def _row_anchors(
    rows: list[list[Word]], column: TableColumn, tolerance: float,
    median_height: float,
) -> tuple[list[tuple[int, Word]], str]:
    """(anchor pairs, mode) — three anchor strategies in priority order:

    'qty'      decimals in the Qty column (classic batch1 tables);
    'column'   the RIGHTMOST data-driven money column (total column):
               works with missing/mangled headers and captures integer
               quantities as a side value, never as row anchors;
    'fallback' first price token per row (degenerate layouts).
    """

    start = column.header_row + 1 if column.header_row >= 0 else 0

    if column.quantity_x is not None:
        found = [
            (row_index, word)
            for row_index, row in enumerate(rows)
            if row_index >= start
            for word in row
            if DECIMAL_RE.match(word.text.replace("$", "").strip())
            and abs(word.x_center - column.quantity_x) <= tolerance
        ]
        if found:
            return found, "qty"

    money_columns = _find_money_columns(rows, median_height)
    if money_columns:
        # Anchor rows by ANY money column; the anchor is the row's
        # rightmost column token. Rows whose TOTAL was mangled by OCR but
        # whose price column token survives still make it (Geonics rows
        # with only a price), while one-off money-shaped values do not
        # create anchors (a real column spans >=2 rows).
        anchor_map: dict[int, Word] = {}
        for money_column in money_columns:
            for row_index, word in money_column.tokens:
                current = anchor_map.get(row_index)
                if current is None or (
                    word.left + word.width > current.left + current.width
                ):
                    anchor_map[row_index] = word
        if anchor_map:
            return sorted(anchor_map.items()), "column"

    # Fallback: anchor every price-bearing row by its FIRST price token.
    anchors: list[tuple[int, Word]] = []
    for row_index, row in enumerate(rows):
        if row_index < start:
            continue
        prices = sorted(
            (word for word in row if _price_word(word.text)),
            key=lambda w: w.left,
        )
        if prices:
            anchors.append((row_index, prices[0]))
    return anchors, "fallback"


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------


def _description_lines(
    rows: list[list[Word]], left: float, right: float
) -> list[tuple[float, float, str]]:
    """Text lines inside the description band (x between the description
    column start and the quantity column), row-number tokens excluded.

    Returns (line_top, band_y_center, text): the y-center allows matching
    a text cluster that Tesseract printed as a separate visual row of the
    SAME printed item row (text line sits slightly above the price line).
    """

    lines_out: list[tuple[float, float, str]] = []

    for row in rows:
        band = [
            word for word in row
            if left < word.x_center < right
            and not ROW_NUMBER_RE.match(word.text)
        ]
        if band:
            text = " ".join(word.text for word in sorted(band, key=lambda w: w.left))
            y_center = sum(w.y_center for w in band) / len(band)
            lines_out.append((_row_top(row), y_center, text))

    return lines_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_items(lines: Sequence[Line], zones: Zones) -> list[RawItem]:
    """Build item rows from the items zone."""

    rows = rows_in_zone(lines, zones, "items")
    column = _find_columns(rows)

    if not rows:
        return []

    tolerance = _column_tolerance(rows, column)
    median_height = _median_height(rows)
    money_columns = _find_money_columns(rows, median_height)
    anchors, mode = _row_anchors(rows, column, tolerance, median_height)

    if not anchors:
        return []

    items = [RawItem() for _ in anchors]

    # --- descriptions: band text assigned between anchor rows -----------
    quantities_sorted = sorted(anchors, key=lambda pair: pair[1].top)

    if column.description_word is not None:
        # Band right bound: the Qty column, else the leftmost money column.
        if column.quantity_x is not None:
            desc_right = column.quantity_x - tolerance * 0.5
        elif money_columns:
            desc_right = money_columns[0].x_center - tolerance * 0.5
        else:
            desc_right = None

        if desc_right is not None:
            desc_lines = _description_lines(
                rows, column.description_left, desc_right
            )
            have_positions = True
        else:
            have_positions = False
    else:
        have_positions = False

    for index, (row_index, anchor) in enumerate(quantities_sorted):
        row_top = anchor.top - 0.6 * anchor.height
        row_bottom = (
            quantities_sorted[index + 1][1].top
            if index + 1 < len(quantities_sorted)
            else float("inf")
        )
        if have_positions:
            same_row_tolerance = 0.75 * anchor.height
            texts = [
                text
                for position, y_center, text in desc_lines
                if (row_top <= position < row_bottom)
                or abs(y_center - anchor.y_center) <= same_row_tolerance
            ]
            items[index].description = " ".join(texts)

    # --- numeric values per anchor row ----------------------------------
    for index, (row_index, anchor) in enumerate(quantities_sorted):
        if mode == "qty":
            items[index].quantity = _normalize_decimal(anchor.text)
        elif mode == "column" and column.quantity_x is not None:
            # Quantities ride along: integer or decimal in the Qty band.
            # Row EXISTENCE is decided by the money column, so integer
            # quantities can never create phantom rows here. The Qty
            # token may sit in a neighbour cluster of the same printed
            # row (text split from prices), hence the neighbour pass.
            qty_tolerance = tolerance * 0.5

            def _qty_in(row: Sequence[Word]) -> Word | None:
                candidates = [
                    word for word in row
                    if abs(word.x_center - column.quantity_x) <= qty_tolerance
                    and not ROW_NUMBER_RE.match(word.text)
                    and "%" not in word.text
                    and (
                        DECIMAL_RE.match(word.text.replace("$", "").strip())
                        or INT_RE.match(word.text.strip())
                    )
                ]
                if not candidates:
                    return None
                return min(
                    candidates,
                    key=lambda w: abs(w.x_center - column.quantity_x),
                )

            qty_word = _qty_in(rows[row_index])
            if qty_word is None:
                for neighbour in _nearest_neighbour_rows(rows, row_index, anchor):
                    qty_word = _qty_in(neighbour)
                    if qty_word is not None:
                        break
            if qty_word is not None:
                items[index].quantity = _normalize_decimal(qty_word.text)

        if mode == "column":
            # The anchor IS the rightmost money column value (the total).
            items[index].total_price = _format_decimal(anchor.text)
        else:
            limit_left = anchor.x_center - tolerance * 0.3
            values = _money_tokens(rows[row_index], limit_left)

            # Tesseract can split one printed row into two visual clusters
            # (text line + price line): if the anchor row carries no prices,
            # check the immediate neighbours before giving up.
            if not values:
                for neighbour in _nearest_neighbour_rows(rows, row_index, anchor):
                    values = _money_tokens(neighbour, limit_left)
                    if values:
                        break

            items[index].total_price = _format_decimal(values[-1]) if values else ""

    # --- fallback templates without column anchors: textual build -------
    if not have_positions:
        for index, (row_index, anchor) in enumerate(quantities_sorted):
            members = [
                word for word in rows[row_index]
                if not _price_word(word.text)
                and not re.match(r"^\d+[%]", word.text.strip())
                and word.text.strip() not in {"$", "%"}
                and not ROW_NUMBER_RE.match(word.text)
            ]
            items[index].description = " ".join(
                word.text for word in sorted(members, key=lambda w: w.left)
            )

    return items
