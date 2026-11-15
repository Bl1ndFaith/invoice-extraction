"""Pytesseract OCR helpers: words with boxes, grouped into text lines."""
import os

import cv2
import numpy as np
import pytesseract

_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_WIN_ALT = r"E:\Program Files\Tesseract-OCR\tesseract.exe"
for _p in (os.environ.get("TESSERACT_CMD"), _WIN_DEFAULT, _WIN_ALT):
    if _p and os.path.exists(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        break


def ocr_words(img, scale=2, psm=6):
    """Return OCR words as dicts {text,left,top,width,height,conf} in the original pixel space."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if scale != 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # white border helps tesseract with text touching the crop edge
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    data = pytesseract.image_to_data(gray, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)

    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or float(data["conf"][i]) < 0:
            continue
        words.append({
            "text": text,
            "left": (data["left"][i] - 10) / scale,
            "top": (data["top"][i] - 10) / scale,
            "width": data["width"][i] / scale,
            "height": data["height"][i] / scale,
            "conf": float(data["conf"][i]),
        })
    return words


def group_lines(words):
    """Group words into text lines.

    Tesseract returns separate words. Meaning lives in the line ("Invoice no:" + its value,
    the cells of one table row, consecutive address lines), so words whose vertical centres
    are close are merged into one line, each line sorted left to right.
    """
    if not words:
        return []
    med_h = float(np.median([w["height"] for w in words]))
    ordered = sorted(words, key=lambda w: w["top"] + w["height"] / 2)
    lines, cur, cur_c = [], [], None
    for w in ordered:
        c = w["top"] + w["height"] / 2
        if cur and abs(c - cur_c) > 0.6 * med_h:
            lines.append(cur)
            cur = []
        cur.append(w)
        cur_c = float(np.mean([x["top"] + x["height"] / 2 for x in cur]))
    if cur:
        lines.append(cur)
    return [sorted(l, key=lambda w: w["left"]) for l in lines]


def line_text(line):
    return " ".join(w["text"] for w in line)


def words_to_text(words):
    return "\n".join(line_text(l) for l in group_lines(words))
