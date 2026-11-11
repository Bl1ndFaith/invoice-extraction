# invoice_to_json

Extracts key information from scanned invoice images (`.jpg`) and writes it as JSON,
using OpenCV for image preprocessing and Tesseract (via `pytesseract`) for OCR.

The target output schema corresponds to the ground truth in `data/*.csv`
(see `prompt.md` for the full task description).

## Usage

```sh
# single image (also prints JSON)
python main.py data/batch1_1/batch1-0494.jpg --out output --print

# whole folder -> one JSON per image in output/
python main.py data/batch1_1 --out output

# evaluate against the ground-truth CSVs (10 sample files per CSV)
python evaluate.py --limit 10
```

Both scripts pick their Python configuration from the project `.venv`
(`venv\Scripts\python.exe` on Windows) and expect `tesseract.exe` either on
`PATH` or in `E:\Program Files\Tesseract-OCR` (see `modules/ocr_engine.py`).

## Pipeline (current — zone-based)

```
Image Preprocessing -> OCR -> Semantic Analysis -> Data Validation -> Output Formatting
```

1. **`modules/image_processor.py`** — load image, grayscale, upscale small
   scans (<1700 px), deskew (Hough lines), threshold (global Otsu), denoise
   (Gaussian blur 3x3, sigma=0.6 on the binary image).
2. **`modules/ocr_engine.py`** — Tesseract `image_to_data` -> word records with
   coordinates/confidences -> group into lines (block/par/line keys).
3. **`modules/zone_splitter.py`** — horizontal zoning:
   - items_top: table-header words -> money-column top -> 35% ratio;
   - totals_top: totals labels -> money-column bottom -> 70% ratio;
   - items_bottom: last money row of a real column + margin.
   - `Zones` may hold two header bands (classic / table-first).
4. **`modules/kv_extractor.py`** — label dictionary + synonyms + fuzzy
   matching, zone-aware parties, address-marked fallbacks.
5. **`modules/table_extractor.py`** — data-driven money columns (right-edge
   clustering), row anchors (qty / column / fallback), quantity ride-along.
6. **`modules/json_builder.py`** — schema assembly (fixed key order, empty
   strings for missing fields) and validation.
7. **`main.py`** — orchestration, lazy re-OCR fallbacks, math reconcile,
   CLI (single/batch), errors never stop a batch.

### Lazy re-OCR fallbacks (primary `--oem 1 --psm 3` never changes)

- If zones are unanchored (source `ratio`) or items < 3 -> full-page re-OCR
  with `--oem 1 --psm 6`, then border-removal (OpenCV) -> retry if better.
- If items empty / >50% rows without total -> zone crop re-OCR with
  `--oem 1 --psm 6` -> adopt only if strictly more items.

## Evaluation notes

From analysis of the ground truth (see `prompt.md` section Notes):

- **Description line breaks:** the ground truth keeps the `\n` of the source
  product listing for some items and drops it for others; the image renders
  both the same way (soft wrap). This is not recoverable from pixels, so
  descriptions are joined with spaces, matching the majority style.
- **Money formats:** ground truth uses several formatting styles
  (dot `232.95`, comma `360,58`, space-groups `3 838.87` / `15 339,50`)
  inconsistently across invoices. The extractor joins split thousands tokens
  and applies the majority style per number shape; values are therefore always
  correct even when the exact ground-truth string differs.
- **Greek-Alpha defect:** GT of `batch1-0505` contains Greek small-cap `Α`
  instead of ASCII `A` in some ids/names; mismatch is reported but not
  treated as a pipeline defect.

Consequences visible in `evaluate.py`:

- exact field match: high on header fields, ~75% on money fields because of
  the style ambiguity above, ~17% on whole `items` arrays because one wrong
  whitespace in any description fails the whole array;
- value-level (style-insensitive) match on structured fields: ~100%.

## Current regression baseline (batch1, 25 samples)

- Exact accuracy: **86.8%** (716/825), Value-level: **98.0%** (735/750).

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- Tesseract OCR binary installed separately (5.x recommended)
- OpenCV (`pip install opencv-python`), pytesseract (`pip install pytesseract`)
