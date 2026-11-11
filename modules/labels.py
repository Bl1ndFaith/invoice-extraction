"""Label dictionaries for generic invoice extraction.

One shared place where all known invoice labels live. Zone splitting,
KV extraction and the table engine all read from here, so a new invoice
family is usually supported by adding synonyms to this file — no new code.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Table header synonyms.
# A line matches the table header when enough of its words are in here:
# values are normalized (lowercase, trailing ':!%' stripped) whole words.
# ---------------------------------------------------------------------------

TABLE_HEADER_WORDS = frozenset(
    {
        "id",
        "no",
        "description",
        "qty",
        "um",
        "unit",
        "price",
        "net",
        "worth",
        "total",
        "amount",
        "vat",
        "gross",
        "item",
        "product",
        "rate",
        "sum",
        "uom",
        "each",
    }
)

# A line containing at least this many table header words is the table header.
TABLE_HEADER_MIN_WORDS = 2

# Strong table header words: a single occurrence reliably marks the table
# header row, because Tesseract often splits the header row into separate
# OCR lines ('Description' | 'Amount') that would fail a 2-word threshold.
# 'total'/'amount'/'price' are deliberately NOT strong: they occur in body
# text and totals labels too ('TOTAL DUE', 'Amount Due').
STRONG_TABLE_HEADER_WORDS = frozenset(
    {"description", "qty", "um", "net", "worth", "gross", "vat"}
)

# ---------------------------------------------------------------------------
# Totals-zone labels (start of the totals region on the page).
# Matched per word, because Tesseract splits compound labels
# ('TOTAL DUE', 'Sub Total :', 'Sales Tax 8%') into separate words.
# ---------------------------------------------------------------------------

TOTALS_LABEL_WORDS = frozenset(
    {
        "subtotal",
        "sub",
        "total",
        "tax",
        "sales",
        "vat",
        "balance",
        "summary",
        "paid",
        "networth",
        "payable",
        "freight",
        "handling",
        "grand",
    }
)

# Fuzzy-enabled subset for the totals row classifier: only words long
# enough for reliable matching ('tax'/'vat'/'sub'/'due' stay exact-only).
TOTALS_FUZZY_WORDS = frozenset(
    {"subtotal", "total", "discount", "handling", "freight", "payable"}
)

# ---------------------------------------------------------------------------
# KV field labels: field name -> synonyms (normalized whole-word/phrase form).
# The KV extractor searches the label phrases inside allowed zones only.
# Values are compared as joined line text, so multi-word phrases work too.
# Longest phrases must win over shorter generic ones ('due date' before
# 'date') — the extractor sorts synonyms by length, so order here is free.
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "invoice_number": (
        "invoice no",
        "invoice number",
        "invoice:",
        "invoice id",
        "issue number",
        "inv no",
        "inv number",
        "invoice num",
        "document no",
        "doc no",
        "ref no",
        "reference no",
        "bill no",
    ),
    "invoice_date": (
        "date of issue",
        "invoice date",
        "issue date",
        "date issued",
        "invoiced on",
        "invoiced date",
        "bill date",
        "document date",
        "date",
    ),
    "due_date": (
        "due date",
        "due_date",
        "payment due",
        "payment due date",
        "due on",
    ),
    "seller": (
        "bill from",
        "seller",
        "from",
        "shipper",
        "vendor",
        "supplier",
        "sold by",
        "ship from",
        "sender",
    ),
    "client": (
        "bill to",
        "client",
        "buyer",
        "sold to",
        "ship to",
        "billed to",
        "invoice to",
        "customer",
        "deliver to",
        "consignee",
        "recipient",
    ),
}

# ---------------------------------------------------------------------------
# Value validators by expected type (label -> value shape).
# Established-regex approach: strict patterns first, then real-value
# confirmation (dates go through datetime, money through grouping rules)
# so OCR garbage like '45/45/2021' never validates.
# ---------------------------------------------------------------------------

# -- dates -------------------------------------------------------------------

# Numeric with a consistent separator: 09/17/2017, 3.11.2020, 17-09-2020.
DATE_NUMERIC_RE = re.compile(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})$")
# ISO first: 2020-03-03.
DATE_ISO_RE = re.compile(r"^(\d{4})([-/.])(\d{1,2})\2(\d{1,2})$")

# Textual month spellings confirmed via strptime ('Oct. 11, 2023').
TEXTUAL_DATE_FORMATS = (
    "%b %d, %Y", "%B %d, %Y", "%b. %d, %Y",
    "%b %d %Y", "%B %d %Y", "%b. %d %Y",
    "%d %b %Y", "%d %B %Y",
)


def _calendar_ok(year: int, month: int, day: int) -> bool:
    """Real-calendar check: rejects 45/45/2021, 31/02/2020, 13/2021."""

    from datetime import date

    try:
        date(year, month, day)
        return True
    except ValueError:
        return False


def looks_like_date(text: str) -> bool:
    """True when the token/phrase is a real date in any common layout."""

    from datetime import datetime

    text = text.strip().rstrip(",.:;")

    # Textual months: strptime is the validator (and range-checks the day).
    for fmt in TEXTUAL_DATE_FORMATS:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue

    match = DATE_ISO_RE.match(text)
    if match:
        year, _, month, day = match.groups()
        return _calendar_ok(int(year), int(month), int(day))

    match = DATE_NUMERIC_RE.match(text)
    if match:
        first, _, second, raw_year = match.groups()
        year = int(raw_year)
        year = year + 2000 if year < 100 else year
        # US mm/dd preferred; dd/mm accepted when mm/dd is impossible.
        return _calendar_ok(year, int(first), int(second)) or _calendar_ok(
            year, int(second), int(first)
        )

    return False


# -- money -------------------------------------------------------------------

# Canonical money: optional currency symbol, optional parenthesised or
# signed negative, thousand groups in either convention (1,234.56 /
# 1 234,56 / 1.234,56), 1-2 decimal digits.
MONEY_RE = re.compile(
    r"^[-(]?\s*[$€£]?\s*"
    r"(?:\d{1,3}(?:[ .,]\d{3})+|\d+)"
    r"(?:[.,]\d{1,2})?"
    r"\s*\)?$"
)


def looks_like_money(text: str) -> bool:
    """'$4573', '$3679.20', '2281', '3400.00', '15 339,50' — all money."""

    text = text.strip().rstrip(",:;.")
    return bool(MONEY_RE.match(text))


PRICE_TOKEN_RE = re.compile(
    r"^[$€£]\s*(?:\d{1,3}(?:[ .,]\d{3})+|\d+)(?:[.,]\d{1,2})?$"
    r"|^(?:\d{1,3}(?:[ .,]\d{3})+|\d+)[.,]\d{1,2}$"
)


def looks_like_price(text: str) -> bool:
    """A price-SHAPED token: '$15', '$0.00', '154.00', '1 234,56'.

    Unlike looks_like_money, bare integers ('84178' postal codes, '6379'
    row numbers, quantities) are NOT prices. Page-wide money-column
    clustering relies on this to ignore non-table numeric junk.
    """

    return bool(PRICE_TOKEN_RE.match(text.strip()))


# -- invoice number ----------------------------------------------------------

# Alphanumeric id with -_/ separators; at least one digit required; not a
# single word without digits. Money-shaped ids ('102') stay allowed.
INVOICE_NO_RE = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9\-_/]{1,19}$")


def looks_like_code(text: str) -> bool:
    """Invoice ids: '564234', '102', 'PO-675353' — but NOT a date or a word.

    Note: money-shaped tokens ('102') are valid ids, so they are allowed.
    Zone/type rules elsewhere keep values from dates by the date check.
    """

    words = text.strip().split()
    if len(words) != 1:
        return False
    token = words[0].rstrip(",:;.")
    if looks_like_date(token):
        return False
    return bool(INVOICE_NO_RE.match(token))


# -- contact / address helpers ------------------------------------------------
# For the upcoming anchor-less party fallbacks (Крок C): recognizing address
# blocks by content instead of 'Bill To' labels.


EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

PHONE_RE = re.compile(r"^[(+\d][\d\s\-().]{5,}$")

# US-style ZIP ('78346', '69747-1234'); covers the batch corpora.
POSTAL_RE = re.compile(r"^\d{5}(?:-\d{4})?$")

PERCENT_RE = re.compile(r"^\d{1,3}(?:\.\d+)?\s?%$")


def looks_like_email(text: str) -> bool:
    return bool(EMAIL_RE.match(text.strip().rstrip(",.:;")))


def looks_like_phone(text: str) -> bool:
    """Phone: regex shape + enough structure that '84652373' (an invoice
    number) does not pass — needs separators/+ AND >=7 digits, or >=10."""

    text = text.strip().rstrip(",.:;")
    if not PHONE_RE.match(text):
        return False
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 10:
        return True
    structured = bool(re.search(r"[+\-()]", text))
    return structured and len(digits) >= 7


def looks_like_postal(text: str) -> bool:
    return bool(POSTAL_RE.match(text.strip().rstrip(",.:;")))


def looks_like_percent(text: str) -> bool:
    return bool(PERCENT_RE.match(text.strip()))


# ---------------------------------------------------------------------------
# Fuzzy label matching (OCR noise): 'descrlptlon' -> 'description',
# 'hendling' -> 'handling'. Pure stdlib, no new dependencies.
#
# Budgets by word length keep false positives out:
#   - words shorter than 4 chars are NEVER fuzzy-matched ('no', 'id', 'um');
#   - 4-5 char words allow 1 edit, 6+ char words allow 2 edits;
#   - a similarity-ratio cutoff rejects same-length lookalikes of unrelated
#     vocabulary ('get' never matches 'net').
# Fuzzy label hits are always gated downstream by value-type validators,
# so a misfired label cannot put garbage into a field.
# ---------------------------------------------------------------------------

FUZZY_MIN_LENGTH = 4
FUZZY_SHORT_EDITS = 1   # word length 4..5
FUZZY_LONG_EDITS = 2    # word length 6+
FUZZY_MIN_RATIO = 0.8   # rejects 'date'->'rate' / 'get'->'net' lookalikes


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute)."""

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,          # delete
                    current[j - 1] + 1,       # insert
                    previous[j - 1] + (char_a != char_b),  # substitute
                )
            )
        previous = current
    return previous[-1]


def fuzzy_match(
    text: str, vocabulary: Iterable[str], min_candidate_len: int = FUZZY_MIN_LENGTH
) -> str | None:
    """Best vocabulary entry within the length-dependent edit budget.

    Returns the matched entry, or None when nothing is close enough
    (including when the word is too short to fuzzy-match at all).
    Exact equality is the caller's job — this helper is for OCR noise.

    min_candidate_len raises the floor for *vocabulary entries* (not the
    scanned word): short candidates like 'net'/'worth' absorb unrelated
    lookalikes ('North' -> 'worth'), so call sites that must not misfire
    on running text pass 6+ and keep only specific long entries fuzzy.
    """

    text = text.strip().rstrip(":!.,;")
    if len(text) < min_candidate_len:
        return None

    budget = FUZZY_SHORT_EDITS if len(text) <= 5 else FUZZY_LONG_EDITS
    best: str | None = None
    best_distance = budget + 1

    for candidate in vocabulary:
        candidate = candidate.strip()
        if len(candidate) < min_candidate_len:
            continue
        if abs(len(candidate) - len(text)) > budget:
            continue
        distance = levenshtein(text, candidate)
        if distance > budget or distance >= best_distance:
            continue
        ratio = difflib.SequenceMatcher(None, text, candidate).ratio()
        if ratio < FUZZY_MIN_RATIO:
            continue
        best, best_distance = candidate, distance

    return best
