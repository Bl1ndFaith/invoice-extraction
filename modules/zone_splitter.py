"""Page zoning: split an invoice page into horizontal regions.

Zones are order-agnostic: the header KV region is whatever is NOT the
items table and NOT the totals block. Usually that is one band above the
table, but table-first layouts (items at the top, invoice fields below
the table) get a SECOND interval between the table body and the totals.

    header KV band(s) | items table | totals

Boundaries are found by anchor labels and money rows (dictionary-based),
not by fixed pixel positions, so they adapt to any template. Proportional
fallbacks apply when a boundary cannot be found on the page.

All functions are pure: data in, data out. No printing, no file IO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from modules.labels import STRONG_TABLE_HEADER_WORDS
from modules.labels import TABLE_HEADER_MIN_WORDS, TABLE_HEADER_WORDS
from modules.labels import TOTALS_LABEL_WORDS, fuzzy_match, looks_like_money
from modules.labels import looks_like_price
from modules.ocr_engine import Line

# Fallback boundary ratios (page height) when anchors are not found.
HEADER_FALLBACK_RATIO = 0.35   # items start: below the top third
TOTALS_FALLBACK_RATIO = 0.70   # totals start: near the bottom quarter

# Safety margin (page-height fraction) so a table header word is never
# counted as the first totals label (the 'TOTAL' column header trap).
ZONE_MARGIN_RATIO = 0.01

# Post-table KV interval starts this many median word heights below the
# last money-bearing table row: trailing description lines belong to the
# table, everything further down (before totals) is header-searchable.
KV_BAND_MARGIN_HEIGHTS = 2.0


def _normalize_word(text: str) -> str:
    return text.strip().rstrip(":!%.,;").lower()


def _vocab_hit(word: str, vocabulary: frozenset[str]) -> bool:
    """Exact dictionary membership, else fuzzy (OCR-mangled words like
    'descrlptlon' -> 'description').

    Fuzzy floor is 6 chars: zone anchors must not misfire on running text
    ('North' -> 'worth'), so only specific long entries fuzzy-match.
    """

    if word in vocabulary:
        return True
    return fuzzy_match(word, vocabulary, min_candidate_len=6) is not None


@dataclass(frozen=True)
class Zones:
    """Horizontal split of the page by y coordinates (top of each region).

    items_bottom_y marks where the items TABLE body ends (last money row
    + margin); the header KV region re-opens between it and totals_top_y.

    *_source records how each boundary was anchored: 'header'/'label'
    (dictionary), 'money-column' (data-driven alignment), 'ratio'
    (proportional fallback — a hint that the page may be a fragment).
    """

    page_height: float
    items_top_y: float      # header zone ends here (table header)
    totals_top_y: float     # items zone ends here (first totals row)
    items_bottom_y: float   # table body ends here (post-table KV may start)
    items_top_source: str = "header"    # header | money-column | ratio
    totals_top_source: str = "label"    # label | money-column | ratio

    @property
    def header_zone(self) -> list[tuple[float, float]]:
        bands = [(0.0, self.items_top_y)]
        if self.items_bottom_y < self.totals_top_y:
            bands.append((self.items_bottom_y, self.totals_top_y))
        return bands

    @property
    def items_zone(self) -> tuple[float, float]:
        return self.items_top_y, self.totals_top_y

    @property
    def totals_zone(self) -> tuple[float, float]:
        return self.totals_top_y, self.page_height

    def band(self, name: str) -> list[tuple[float, float]]:
        return {
            "header": self.header_zone,
            "items": [self.items_zone],
            "totals": [self.totals_zone],
        }[name]


def _find_table_header_y(lines: Sequence[Line]) -> float | None:
    """y of the first line that reads like a table header row.

    A line qualifies when it holds a strong header word (or several weak
    ones) — single strong words survive Tesseract's column splitting.
    """

    for line in lines:
        words = [_normalize_word(word.text) for word in line.words]
        if any(_vocab_hit(word, STRONG_TABLE_HEADER_WORDS) for word in words):
            return float(line.top)
        # Distinct words only: a line holding 'Tax Id: ... Tax Id: ...'
        # repeats one weak word and must not look like a header.
        weak_hits = len(
            {word for word in words if _vocab_hit(word, TABLE_HEADER_WORDS)}
        )
        if weak_hits >= TABLE_HEADER_MIN_WORDS:
            return float(line.top)
    return None


# 'Total:$5698' — OCR merges the label and the value into one word when
# no space follows the colon; a plain vocabulary check sees 'total:$5698',
# not 'total'. The prefix before the first non-letter must be a label.
TOTALS_MERGED_PREFIX_RE = re.compile(r"^([a-z]+)(?=[:\s$€£(\d])")


def _totals_word_hit(text: str) -> bool:
    """Totals label as a whole word or as a merged label:value token."""

    if _vocab_hit(_normalize_word(text), TOTALS_LABEL_WORDS):
        return True
    match = TOTALS_MERGED_PREFIX_RE.match(text.lower())
    return bool(match and match.group(1) in TOTALS_LABEL_WORDS)


def _find_totals_top_y(
    lines: Sequence[Line], after_y: float, page_height: float
) -> float | None:
    """y of the first totals label strictly below the table header."""

    margin = page_height * ZONE_MARGIN_RATIO
    candidates = [
        float(line.top)
        for line in lines
        for word in line.words
        if _totals_word_hit(word.text)
        and float(line.top) > after_y + margin
    ]
    return min(candidates) if candidates else None


def _find_money_column_bounds(
    lines: Sequence[Line],
) -> tuple[float, float, float] | None:
    """(first_top, last_bottom, median_height) of page-wide money columns.

    Price-shaped tokens ('looks_like_price': currency symbol or decimals;
    bare integers like postal codes and row numbers stay out) are
    clustered by right edge across the WHOLE page. A cluster spanning
    >=2 distinct lines is a printed money column — the table shows itself
    by alignment even when its header words are missing (table fragments,
    borderless tables, unrecognized header words).

    Returns None when no money column exists on the page.
    """

    money = [
        (line, word)
        for line in lines
        for word in line.words
        if looks_like_price(word.text)
    ]
    if not money:
        return None

    heights = sorted(word.height for line in lines for word in line.words)
    median_height = heights[len(heights) // 2] if heights else 10.0
    tolerance = 1.5 * median_height

    edges = sorted(
        (word.left + word.width, id(line)) for line, word in money
    )

    clusters: list[list[tuple[float, int]]] = [[edges[0]]]
    for edge, line_id in edges[1:]:
        if edge - clusters[-1][-1][0] <= tolerance:
            clusters[-1].append((edge, line_id))
        else:
            clusters.append([(edge, line_id)])

    column_line_ids = {
        line_id
        for cluster in clusters
        if len({line_id for _, line_id in cluster}) >= 2
        for _, line_id in cluster
    }
    if not column_line_ids:
        return None

    column_lines = [
        line for line, _ in money if id(line) in column_line_ids
    ]
    first_top = float(min(line.top for line in column_lines))
    last_bottom = float(max(line.bottom for line in column_lines))
    return first_top, last_bottom, median_height


def _find_items_bottom_y(
    lines: Sequence[Line], items_top: float, totals_top: float
) -> float:
    """Where the items TABLE BODY ends: bottom of the last money row that
    belongs to a real money COLUMN (its right edge shared with another
    row's), plus a small margin.

    Requiring a column keeps money-shaped values OUTSIDE the table
    (invoice numbers '84652373', phone-like ids) from stretching the
    table body over the post-table KV block. Returns totals_top when no
    money column exists — the page keeps the classic single-band split.
    """

    table_money = [
        (line, word)
        for line in lines
        if items_top < line.top < totals_top
        for word in line.words
        if looks_like_money(word.text)
    ]
    if not table_money:
        return totals_top

    heights = sorted(word.height for line in lines for word in line.words)
    median_height = heights[len(heights) // 2] if heights else 10.0
    tolerance = 1.5 * median_height

    edges = sorted(
        (word.left + word.width, id(line)) for line, word in table_money
    )

    # Greedy clustering of right edges; a cluster holding >=2 distinct
    # lines is a printed money column (prices are right-aligned).
    clusters: list[list[tuple[float, int]]] = [ [edges[0]] ]
    for edge, line_id in edges[1:]:
        if edge - clusters[-1][-1][0] <= tolerance:
            clusters[-1].append((edge, line_id))
        else:
            clusters.append([(edge, line_id)])

    column_line_ids = {
        line_id
        for cluster in clusters
        if len({line_id for _, line_id in cluster}) >= 2
        for _, line_id in cluster
    }
    if not column_line_ids:
        return totals_top

    last = max(
        (line for line, _ in table_money if id(line) in column_line_ids),
        key=lambda line: line.top,
    )

    bottom = last.bottom + KV_BAND_MARGIN_HEIGHTS * median_height
    return min(bottom, totals_top)


def find_zones(lines: Sequence[Line], page_height: float) -> Zones:
    """Compute zone boundaries; fallbacks keep the pipeline safe.

    Boundary priority for each edge — anchor first, data second, ratio
    last (full invoices hit the anchors; fragments fall through):

        items_top:   table header words -> top of the page's money
                     columns -> 35% of page height
        totals_top:  totals labels -> bottom of the money columns ->
                     70% of page height
    """

    money_bounds = _find_money_column_bounds(lines)

    header_y = _find_table_header_y(lines)
    if header_y is not None:
        items_top, items_source = header_y, "header"
    elif money_bounds is not None:
        first_top, _, median_height = money_bounds
        items_top = max(0.0, first_top - 1.5 * median_height)
        items_source = "money-column"
    else:
        items_top, items_source = (
            page_height * HEADER_FALLBACK_RATIO,
            "ratio",
        )

    totals_top = _find_totals_top_y(lines, items_top, page_height)
    if totals_top is not None:
        totals_source = "label"
    elif money_bounds is not None:
        _, last_bottom, median_height = money_bounds
        totals_top = min(
            last_bottom + KV_BAND_MARGIN_HEIGHTS * median_height,
            page_height,
        )
        totals_source = "money-column"
    else:
        totals_top, totals_source = (
            page_height * TOTALS_FALLBACK_RATIO,
            "ratio",
        )

    # Never allow an inverted split.
    totals_top = max(totals_top, items_top + page_height * ZONE_MARGIN_RATIO)

    items_bottom = _find_items_bottom_y(lines, items_top, totals_top)

    return Zones(
        page_height=page_height,
        items_top_y=items_top,
        totals_top_y=totals_top,
        items_bottom_y=items_bottom,
        items_top_source=items_source,
        totals_top_source=totals_source,
    )
