"""Table grid handling: erase dark table borders before OCR and report where the row rules are."""
import cv2
import numpy as np


def remove_grid(img):
    """Return (clean_img, rule_ys).

    Dark, long horizontal / vertical lines (the borders of a grid table) disturb Tesseract, which
    drops or garbles the text next to them. They are found with a long morphological opening and
    painted white. The y of every horizontal rule is returned: rows can then be cut between rules,
    even if the numbers of a row are not on the same text line as its description.
    Light-gray borders (shaded tables) are not touched.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape
    dark = (gray < 140).astype(np.uint8) * 255

    hor = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.3)), 1)))
    ver = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.25)))))
    mask = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((3, 3), np.uint8))

    clean = gray.copy()
    clean[mask > 0] = 255

    prof = (hor > 0).sum(axis=1)
    rows = np.where(prof > w * 0.3)[0]
    rules, start, prev = [], None, None
    for r in rows:                      # merge consecutive pixel rows of one thick line
        if start is None:
            start = prev = r
        elif r - prev <= 2:
            prev = r
        else:
            rules.append((start + prev) / 2)
            start = prev = r
    if start is not None:
        rules.append((start + prev) / 2)
    return clean, rules
