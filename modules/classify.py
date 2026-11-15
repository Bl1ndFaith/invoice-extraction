"""Decide which zone is header / parties / summary from the words inside it (no geometry)."""
import itertools
import re

CATEGORIES = ["header", "parties", "summary"]

# (regex, weight). Words that may occur in several categories get a low weight; the
# one-to-one assignment below resolves the rest.
KEYWORDS = {
    "header": [
        (r"invoice\s*(no|number|#|date|:)", 3), (r"date of issue", 3), (r"\bdate\b", 1),
        (r"invoice", 1), (r"\bdue\b", 1),
    ],
    "parties": [
        (r"\bseller\b", 3), (r"\bclient\b", 3), (r"tax\s*id", 3), (r"\biban\b", 3),
        (r"bill(ed)?\s*to", 3), (r"\bvendor\b", 2), (r"\bbuyer\b", 2), (r"\bcustomer\b", 2),
    ],
    "summary": [
        (r"\bsummary\b", 3), (r"\btotal\b", 2), (r"\bsub\s*total\b", 2), (r"\bvat\b", 1),
        (r"net worth", 1), (r"gross worth", 1), (r"\btax\b", 1), (r"amount due", 2), (r"\$", 1),
    ],
}


def score(text, category):
    t = text.lower()
    return sum(w * len(re.findall(rx, t)) for rx, w in KEYWORDS[category])


def classify_zones(zone_texts):
    """zone_texts: {zone_name: text}. Return {category: zone_name} with the best total score.

    Every zone gets a different category, so a zone with a few ambiguous words
    ('Total', 'VAT') is not stolen from the zone that really matches better.
    """
    names = list(zone_texts)
    cats = CATEGORIES[:len(names)]
    table = {n: {c: score(zone_texts[n], c) for c in CATEGORIES} for n in names}
    best, best_score = None, -1
    for perm in itertools.permutations(names, len(cats)):
        s = sum(table[n][c] for n, c in zip(perm, cats))
        if s > best_score:
            best, best_score = perm, s
    return dict(zip(cats, best))
