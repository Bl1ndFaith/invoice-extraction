"""Items table zone: description, quantity and total price of every row.

Columns are found from the table's header row (Description / Qty / Price / Total ...), so the
parser does not depend on one fixed column order. Rows are cut between the table's horizontal
rules when there are any (numbers may be vertically centred in a tall row); otherwise every text
line that carries a quantity and a total starts a row, and the following lines extend its
description. If no header row is found, a regex written for the
"No. Description Qty UM Net price Net worth VAT Gross" layout is used as a fallback.
"""
import re
from decimal import ROUND_HALF_UP, Decimal
from statistics import median

from modules.ocr import group_lines
from modules.text_utils import NUM, parse_number

# A price token as OCR may print it: the decimal separator is sometimes lost ("15557" for "15,57").
TOK = r"\d+(?: \d{3})*[,.]?\d{2}"
NUMERIC_RE = re.compile(r"^[\d\s,.;:%$£€S-]*\d[\d\s,.;:%$£€S-]*$")
AMT_RE = re.compile(r"\d+(?: \d{3})*(?:[,.]\d+)?")

DESC_RE = re.compile(r"^(description|desc\.?|item|items|product|products|service|services|details?|particulars)\b", re.I)
ROLE_RES = {
    "qty": re.compile(r"\b(qty|quantity|qnty|units?|pcs|hours?|hrs)\b", re.I),
    "price": re.compile(r"\b(price|rate)\b", re.I),
    "vat": re.compile(r"\b(vat|tax|gst)\b", re.I),
    "net": re.compile(r"(net\s*(worth|amount|total)|sub\s*total)", re.I),
    "total": re.compile(r"(total|amount|gross|cost|worth|sum)", re.I),
}


# ---------------------------------------------------------------- helpers

def _clean(desc):
    desc = re.sub(r"\s+\S?[\]\|\)]+$", "", desc)  # OCR debris such as "J]" or "|" at the end
    return re.sub(r"\s+", " ", desc).strip(" |[]_")


def _label(text):
    """Header title without grid / OCR debris around it ('_Description |' -> 'Description')."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9%\])]+$", "", text)


def _normalize(text):
    """Common OCR slips in numbers: "10;25" / "5; 75" for "10,25"; a currency-like glyph before a digit."""
    text = re.sub(r"(?<=\d)\s?[;:]\s?(?=\d{2}(?:\s|$))", ",", text)
    return re.sub(r"[£€$S](?=\d)", "", text)


def _amount(text, restore):
    """Last number of a cell as (value, reliable), or None.

    Without a decimal separator the number is either a whole amount ("$462") or a price whose
    separator OCR lost ("15557" for "15,57"). `restore` says which one applies to this column.
    """
    m = AMT_RE.findall(_normalize(text))
    if not m:
        return None
    tok = m[-1]
    if re.search(r"[,.]", tok):
        return parse_number(tok), True
    digits = int(tok.replace(" ", ""))
    if restore and len(tok.replace(" ", "")) >= 3:
        return float(Decimal(digits) / 100), False
    return digits, True


def _has_separator(text):
    m = AMT_RE.findall(_normalize(text))
    return bool(m) and bool(re.search(r"[,.]", m[-1]))


def _quantity(text):
    m = AMT_RE.search(_normalize(text))
    return parse_number(m.group(0)) if m else None


def _expect(base, vat):
    return float((Decimal(str(base)) * (1 + Decimal(str(vat)) / 100)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def _check_total(g, qty, price, net, vat, gross_like):
    """Arithmetic validation of one row. The printed total is trusted if it agrees with
    qty * price or the printed net worth (with or without VAT). Otherwise OCR misread a digit,
    and the total is rebuilt from the most reliable of those values."""
    p, p_ok = price if price else (None, False)
    n = net[0] if net else None
    qp = round(qty * p, 2) if (qty is not None and p is not None) else None

    cands = []
    for base in (qp, n):
        if base is not None:
            cands.append(base)
            if vat:
                cands.append(_expect(base, vat))
    if g is not None and any(abs(g - c) <= 0.02 for c in cands):
        return g
    if not cands:
        return g
    # With only qty * price as evidence (no net worth / VAT column) a disagreement is as likely
    # to come from the document itself or from the price as from the total: keep what is printed.
    if g is not None and net is None and not vat:
        return g
    if n is not None and qp is not None and abs(n - qp) <= 0.02:
        base = n
    elif qp is not None and p_ok:
        base = qp
    else:
        base = n if n is not None else qp
    return _expect(base, vat) if (vat and gross_like) else round(base, 2)


# ---------------------------------------------------------------- header-driven parser

def _chunks(line):
    """Words that are close together form one cell (a header title, a number like '1 394,67',
    a piece of a description). Larger gaps separate columns."""
    h = median(w["height"] for w in line)
    groups, cur = [], [line[0]]
    for w in line[1:]:
        if w["left"] - (cur[-1]["left"] + cur[-1]["width"]) > 1.0 * h:
            groups.append(cur)
            cur = []
        cur.append(w)
    groups.append(cur)
    return [{"text": " ".join(w["text"] for w in g), "left": g[0]["left"],
             "right": g[-1]["left"] + g[-1]["width"], "h": h} for g in groups]


def _find_header(lines):
    for i, line in enumerate(lines):
        cells = _chunks(line)
        d = [j for j, c in enumerate(cells) if DESC_RE.match(_label(c["text"]))]
        if d and any(ROLE_RES[r].search(_label(c["text"])) for r in ("qty", "price", "total") for c in cells):
            return i, cells, d[0]
    return None


def _roles(cells):
    roles = {}
    for r in ("qty", "price", "vat", "net"):
        for j, c in enumerate(cells):
            lab = _label(c["text"])
            if ROLE_RES[r].search(lab) and not (r == "price" and ROLE_RES["net"].search(lab)):
                roles[r] = j
                break
    tot = [j for j, c in enumerate(cells) if ROLE_RES["total"].search(_label(c["text"]))]
    roles["total"] = tot[-1] if tot else len(cells) - 1
    return roles


def _nearest(chunk, cells):
    def dist(c):
        return min(abs(chunk["right"] - c["right"]), abs(chunk["left"] - c["left"]),
                   abs((chunk["left"] + chunk["right"]) / 2 - (c["left"] + c["right"]) / 2))
    return min(range(len(cells)), key=lambda j: dist(cells[j]))


def _cells_of(lines, desc, data):
    """Split the lines of one row (or one text line) into description text and column cells."""
    margin = 0.8 * desc["h"]
    boundary = data[0]["left"] - margin
    desc_parts, cols = [], {}
    for line in lines:
        for ch in _chunks(line):
            if ch["right"] < desc["left"] - 0.3 * desc["h"]:
                continue                       # row-number / ID column (ends before the description starts)
            # description: starts at the description column's left edge, or sits left of the first
            # data column and is not a bare number (a centred quantity like "2" may start left of
            # its header)
            if ch["left"] <= desc["left"] + 2.5 * desc["h"] or (
                    ch["left"] < boundary and not NUMERIC_RE.match(ch["text"])):
                desc_parts.append(ch["text"])
            else:
                j = _nearest(ch, data)
                cols[j] = (cols.get(j, "") + " " + ch["text"]).strip()
    return " ".join(desc_parts), cols


def _finalize(raw_rows, roles, gross_like):
    """Turn raw cell texts into items; a column without decimal separators holds whole amounts."""
    restore = {}
    for r in ("price", "net", "total"):
        if r in roles:
            cells = [rw[1][roles[r]] for rw in raw_rows if roles[r] in rw[1]]
            seps = [_has_separator(c) for c in cells]
            restore[r] = bool(seps) and sum(seps) >= len(seps) / 2

    def num(row_cols, r):
        if r in roles and roles[r] in row_cols:
            return _amount(row_cols[roles[r]], restore.get(r, False))
        return None

    items = []
    for text, cols, band in raw_rows:
        total, qty_txt = num(cols, "total"), cols.get(roles.get("qty", -1))
        qty = _quantity(qty_txt) if qty_txt else None
        if total is None or ("qty" in roles and qty is None):
            if items and text and not cols:    # wrapped description of the previous row
                items[-1]["description"] = _clean(items[-1]["description"] + " " + text)
            continue
        vat_txt = cols.get(roles.get("vat", -1))
        vat = _quantity(vat_txt) if vat_txt else None
        q = qty if qty is not None else 1
        items.append({"description": _clean(text), "quantity": q,
                      "total_price": _check_total(total[0], q, num(cols, "price"), num(cols, "net"),
                                                  vat, gross_like),
                      "_band": band})   # row index between rules (None without rules); see extract_json
    return items


def _parse_by_header(lines, header, rules, height):
    hi, cells, di = header
    desc, data = cells[di], cells[di + 1:]
    if not data:
        return None
    roles = _roles(data)
    gross_like = bool(re.search(r"gross|incl", data[roles["total"]]["text"], re.I))

    raw = []
    header_bottom = max(w["top"] + w["height"] for w in lines[hi])
    below = [r for r in (rules or []) if r >= header_bottom - 5]
    if len(below) >= 2:                        # rows are the bands between horizontal rules
        edges = below + ([height] if height and height - below[-1] > 30 else [])
        for top, bottom in zip(edges, edges[1:]):
            band = [[w for w in ln if top <= w["top"] + w["height"] / 2 < bottom] for ln in lines[hi + 1:]]
            band = [b for b in band if b]
            if band:
                raw.append(_cells_of(band, desc, data) + (len(raw),))
    else:                                      # rows are text lines, wrapped lines extend the last row
        for line in lines[hi + 1:]:
            raw.append(_cells_of([line], desc, data) + (None,))
    return _finalize(raw, roles, gross_like)


# ---------------------------------------------------------------- regex fallback (known layout)

ROW_RE = re.compile(
    rf"^(?P<desc>.*?)\s*(?P<qty>{NUM})\s+(?P<um>[A-Za-z.]+)\s+(?P<price>{TOK})\s+(?P<net>{TOK})"
    rf"\s+(?P<vat>\d+)\s?%\s+(?P<gross>{TOK})\s*$")


def _parse_by_regex(lines):
    cut_x = None
    for line in lines:
        for w in line:
            if w["text"].lower().startswith("descr"):
                cut_x = w["left"] - 5
                break
        if cut_x is not None:
            break

    items = []
    for line in lines:
        ws = [w for w in line if cut_x is None or w["left"] >= cut_x]
        text = _normalize(" ".join(w["text"] for w in ws))
        m = ROW_RE.match(text)
        if m:
            qty = parse_number(m.group("qty"))
            total = _check_total(_amount(m.group("gross"), True)[0], qty, _amount(m.group("price"), True),
                                 _amount(m.group("net"), True), float(m.group("vat")), True)
            items.append({"description": _clean(m.group("desc")), "quantity": qty, "total_price": total,
                          "_band": None})
        elif items and text:
            items[-1]["description"] = _clean(items[-1]["description"] + " " + text)
    return items


def parse_items(words, rules=None, height=None):
    """words: OCR words of the table zone. rules: y of its horizontal rules (see table_cells)."""
    lines = group_lines(words)
    header = _find_header(lines)
    items = _parse_by_header(lines, header, rules, height) if header else None
    if not items:
        items = _parse_by_regex(lines)
    return items
