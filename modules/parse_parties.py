"""Parties zone: seller and client (name + address), two side-by-side columns."""
import re

from modules.ocr import group_lines

ROLE_LABELS = [
    ("seller", re.compile(r"^(seller|vendor|supplier|from)\b", re.I)),
    ("client", re.compile(r"^(client|buyer|customer|bill(ed)?|sold)\b", re.I)),
]
STOP_RE = re.compile(r"^(tax|iban|vat|swift|bic|phone|tel|email|e-mail)\b", re.I)


def _label_words(words):
    found = {}
    for w in words:
        for role, rx in ROLE_LABELS:
            if role not in found and rx.match(w["text"].strip(":")):
                found[role] = w
    return found


def _column_text(words, label, x0, x1):
    """Lines under the label inside [x0, x1): first one is the name, the rest the address."""
    col = [w for w in words if x0 <= w["left"] < x1 and w["top"] > label["top"] + label["height"] * 0.5]
    lines = []
    for line in group_lines(col):
        txt = " ".join(w["text"] for w in line).strip()
        if STOP_RE.match(txt):
            break
        if not txt or txt.lower() == "items":
            continue
        lines.append(txt)
    if not lines:
        return None, None
    return lines[0], (", ".join(lines[1:]) or None)


def parse_parties(words):
    out = {"seller_name": None, "seller_address": None,
           "client_name": None, "client_address": None}
    labels = _label_words(words)
    if len(labels) < 2:
        return out
    xs = sorted(labels.items(), key=lambda kv: kv[1]["left"])
    tol = 15
    for i, (role, lab) in enumerate(xs):
        x0 = lab["left"] - tol
        x1 = xs[i + 1][1]["left"] - tol if i + 1 < len(xs) else float("inf")
        name, addr = _column_text(words, lab, x0, x1)
        out[f"{role}_name"], out[f"{role}_address"] = name, addr
    return out
