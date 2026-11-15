"""Find the items table and return the y coordinates of the lines above and under it."""
import cv2
import numpy as np


def detect_table(img):
    """Return (y_top, y_bottom) of the largest table (the items table).

    Ink profile: table borders are long light-gray lines. Pixels darker than the
    page background are kept, a long horizontal / vertical opening keeps only
    line-like ink (this drops text, short heading bars and the page-edge strip),
    and the largest connected frame is the items table.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    ink = (gray < 215).astype(np.uint8) * 255

    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.3), 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.03))))
    # drop the full-height page-edge strip (very thin in a row-profile sense: it lies at the border)
    ver[:, int(w * 0.9):] = 0
    lines = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((5, 5), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(lines)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_HEIGHT] * stats[i, cv2.CC_STAT_WIDTH])
    y = stats[best, cv2.CC_STAT_TOP]
    hh = stats[best, cv2.CC_STAT_HEIGHT]

    # refine to the exact line rows using the row ink profile of the horizontal mask
    prof = (hor > 0).sum(axis=1)
    rows = np.where(prof > w * 0.3)[0]
    rows = rows[(rows >= y - 5) & (rows <= y + hh + 5)]
    return int(rows.min()), int(rows.max())
