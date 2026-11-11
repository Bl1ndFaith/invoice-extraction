"""Image loading and preprocessing with OpenCV.

Single responsibility: turn an image file path into a numpy array
optimized for OCR. No OCR, no parsing here.

Pipeline (order matters):

    1. grayscale      color carries no information for OCR;
    2. upscale        Tesseract works best around 300 DPI (existing
                      behaviour, kept byte-compatible);
    3. deskew         scans are often 1-3 degrees off; a Hough-based
                      angle estimate + rotation levels the page. Only
                      fires for |angle| >= DESKEW_MIN_ANGLE so digital
                      PDFs (perfectly axis-aligned) skip the
                      interpolation loss entirely;
    4. thresholding   global Otsu: hardens noisy scans into black-on-white
                      without fragmenting text strokes (adaptive/local
                      binarization was tested and rejected — it collapses
                      Tesseract's line detection on clean pages);
    5. denoising      light 3x3 Gaussian on the binarized page smooths
                      jagged stroke edges ('HI' -> 'Hl' misreads) and
                      leftover speckle. Pre-threshold blur was tested
                      and rejected — it erodes thin strokes here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Tesseract works best around 300 DPI. Scans below this height are
# upscaled before OCR to improve recognition quality.
MIN_HEIGHT_FOR_OCR = 1700

# -- binarization smoothing ---------------------------------------------------
# Post-threshold Gaussian smoothing: hard binarization leaves jagged
# stroke edges that LSTM sometimes reads as different glyphs ('HI' ->
# 'Hl'). A light 3x3 / sigma 0.6 pass smooths those edges. Pre-threshold
# blur was tested and REJECTED — it erodes thin strokes ('11/15' ->
# '01/15', lost digits) on the batch1 suite.
BINARIZE_SMOOTH_KERNEL = (3, 3)
BINARIZE_SMOOTH_SIGMA = 0.6

# -- deskew -------------------------------------------------------------------
DESKEW_MIN_ANGLE = 0.3        # degrees; smaller deviations are left alone
DESKEW_MAX_ANGLE = 10.0       # anything larger is not a scan skew
HOUGH_THRESHOLD = 200         # votes for a line segment
HOUGH_MIN_LINE_FRACTION = 3   # segment length = page width / this
HOUGH_MAX_LINE_GAP = 20


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk, raising FileNotFoundError for unreadable files."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    return image


def _denoise(binary: np.ndarray) -> np.ndarray:
    """Light smoothing of the binarized page (stroke-edge jaggies and
    leftover speckle). Deliberately AFTER binarization: pre-threshold
    blur erodes thin strokes on this dataset."""

    return cv2.GaussianBlur(
        binary, BINARIZE_SMOOTH_KERNEL, BINARIZE_SMOOTH_SIGMA
    )


def _estimate_skew(gray: np.ndarray) -> float:
    """Page skew in degrees, from the median angle of long horizontal
    segments (table rules, underlines, text baselines).

    Returns 0.0 when no reliable segment exists — pages without rules
    simply skip the deskew rather than guess.
    """

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    min_line_length = gray.shape[1] // HOUGH_MIN_LINE_FRACTION
    segments = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=math.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if segments is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= DESKEW_MAX_ANGLE:
            angles.append(angle)

    if len(angles) < 5:
        return 0.0

    return float(np.median(angles))


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by `angle` to level the page (image-coords convention:
    positive baseline angle needs a positive cv2 rotation)."""

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1.0
    )
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _threshold(gray: np.ndarray) -> np.ndarray:
    """Global Otsu binarization.

    Local adaptive binarization was tried and REJECTED: on clean pages it
    fragments text strokes (Tesseract line detection collapses: 66 -> 30
    OCR lines, 5 -> 0 items). Otsu keeps the global bimodal separation —
    identical behaviour to what Tesseract would do internally — while
    still hardening noisy scans, and it even recovers shaded bands on
    fragment pages (0 -> 2 items on batch3-1918).
    """

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return binary


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Grayscale -> upscale -> deskew -> Otsu threshold -> smoothing."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.shape[0] < MIN_HEIGHT_FOR_OCR:
        scale = 2.0
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    angle = _estimate_skew(gray)
    if abs(angle) >= DESKEW_MIN_ANGLE:
        gray = _rotate(gray, angle)

    return _denoise(_threshold(gray))
