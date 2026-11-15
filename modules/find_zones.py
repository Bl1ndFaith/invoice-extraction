"""Find the line separating the header (invoice no./date) from the parties block."""
import cv2
import numpy as np


def find_split_line(img, y_top, y_bottom=None):
    """Return y of the line between header and seller/client info.

    Ink profile of the area above the table (dark text only, so gray heading bars
    are ignored): the widest blank gap between text blocks is the separator and the
    line goes through its middle. If nothing is found above the table, the area
    below it is searched instead.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    ink = (gray < 140)
    ink[:, int(w * 0.9):] = False
    prof = ink.sum(axis=1)

    def widest_gap(a, b):
        text = np.where(prof[a:b] > 0)[0]
        if len(text) < 2:
            return None
        text += a
        gaps = np.diff(text)
        i = int(gaps.argmax())
        if gaps[i] < 3:
            return None
        return int((text[i] + text[i + 1]) // 2)

    y = widest_gap(0, y_top)
    if y is None and y_bottom is not None:
        y = widest_gap(y_bottom, h)
    return y
