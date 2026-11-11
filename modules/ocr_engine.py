"""OCR layer on top of pytesseract.

Converts a preprocessed image into structured word records and groups
them into lines. All geometry lives here and below: extraction logic
never talks to Tesseract directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytesseract
from pytesseract import Output

# Set explicitly if Tesseract is not on PATH. Checked in order.
_TESSERACT_CANDIDATES = (
    r"E:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

OCR_CONFIG = "--oem 1 --psm 3"
# Fallback re-OCR of a cropped zone: single uniform block of text. Only
# used when the primary full-page pass yields no usable item rows.
REOCR_CONFIG = "--oem 1 --psm 6"


@dataclass
class Word:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block_num: int
    par_num: int
    line_num: int

    @property
    def x_center(self) -> float:
        return self.left + self.width / 2

    @property
    def y_center(self) -> float:
        return self.top + self.height / 2


@dataclass
class Line:
    words: list[Word]
    block_num: int
    par_num: int
    line_num: int

    @property
    def top(self) -> int:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> int:
        return max(word.top + word.height for word in self.words)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in sorted(self.words, key=lambda w: w.left))


def _ensure_tesseract() -> None:
    """Point pytesseract at the installed binary if it is not on PATH."""

    try:
        pytesseract.get_tesseract_version()
        return
    except Exception:
        pass

    for candidate in _TESSERACT_CANDIDATES:
        if Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return

    raise RuntimeError(
        "Tesseract binary not found. Install it or check _TESSERACT_CANDIDATES."
    )


def run_ocr(image, config: str = OCR_CONFIG) -> list[Word]:
    """Run Tesseract and return non-empty word records with coordinates."""

    _ensure_tesseract()

    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)

    words: list[Word] = []

    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except ValueError:
            continue

        words.append(
            Word(
                text=text,
                confidence=confidence,
                left=data["left"][index],
                top=data["top"][index],
                width=data["width"][index],
                height=data["height"][index],
                block_num=data["block_num"][index],
                par_num=data["par_num"][index],
                line_num=data["line_num"][index],
            )
        )

    return words


def group_words_into_lines(words: list[Word]) -> list[Line]:
    """Group words by Tesseract's block/paragraph/line hierarchy."""

    grouped: dict[tuple[int, int, int], list[Word]] = {}

    for word in words:
        key = (word.block_num, word.par_num, word.line_num)
        grouped.setdefault(key, []).append(word)

    lines = [
        Line(words=grouped_words, block_num=key[0], par_num=key[1], line_num=key[2])
        for key, grouped_words in grouped.items()
    ]

    lines.sort(key=lambda line: line.top)

    return lines
