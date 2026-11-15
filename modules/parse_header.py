"""Header zone: invoice number and date."""
import re

NUMBER_RE = re.compile(
    r"invoice\s*(?:no\.?|number|nr\.?|#)?\s*[:#]?\s*#?\s*([A-Z0-9-]*\d[A-Z0-9-]*)", re.I)
DATE_RES = [
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
    re.compile(r"\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b"),
    re.compile(r"\b[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}\b"),
    re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}\b"),
]


def parse_header(text):
    number = None
    m = NUMBER_RE.search(text)
    if m:
        number = m.group(1)

    date = None
    for line in text.splitlines():
        if "date" in line.lower():  # prefer the line that is labelled as a date
            for rx in DATE_RES:
                m = rx.search(line)
                if m:
                    date = m.group(0)
                    break
        if date:
            break
    if date is None:
        for rx in DATE_RES:
            m = rx.search(text)
            if m:
                date = m.group(0)
                break
    return {"invoice_number": number, "invoice_date": date}
