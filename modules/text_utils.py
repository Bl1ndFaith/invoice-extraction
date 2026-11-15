"""Shared number helpers for the parsers."""
import re

# 1 234,56 | 1234.56 | 12,50 ; thousands separated by a space
NUM = r"\d{1,3}(?: \d{3})+[,.]\d{2}|\d+[,.]\d{2}"
NUM_RE = re.compile(NUM)


def parse_number(s):
    """'1 394,67' -> 1394.67, '$ 6,204.19' -> 6204.19; None if it is not a number."""
    s = re.sub(r"[^\d,.]", "", s or "")
    if not s:
        return None
    if "," in s and "." in s:  # the last separator is the decimal one
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v.is_integer() else v


def find_numbers(text):
    """All amounts found in a text line, as numbers."""
    return [parse_number(m) for m in NUM_RE.findall(text)]
