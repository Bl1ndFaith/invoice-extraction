# Invoice extraction (OpenCV + Tesseract)

Splits an invoice image into zones with OpenCV, reads the text with Tesseract OCR, and writes the main fields to JSON. No ML and no external API.

## Requirements

- Python 3.10+
- `opencv-python`, `numpy`, `pytesseract`
- Tesseract OCR installed locally. The code looks for it in `TESSERACT_CMD`, `C:\Program Files\Tesseract-OCR\` and `E:\Program Files\Tesseract-OCR\`.

## Usage

```
python main.py
```

Reads every `.jpg` / `.png` in `data/` and `data_test/`, and writes one folder per image to `outputs/<image name>/`:

- `zone_1.png`, `zone_2.png`, `zone_3.png`: the three non-table zones, numbered top to bottom
- `zone_table.png`: the items table
- `lines_labeled.png`: the full invoice with the three dividing lines and each zone's category (`header`, `parties`, `items`, `summary`)
- `invoice.json`: the extracted data

`data_example/` contains a few sample images to try the pipeline on.

## Pipeline

1. **Zone split (OpenCV)**
   - `table_detect.py`: finds the items table as the largest block of long horizontal and vertical lines. Returns its top and bottom.
   - `find_zones.py`: finds the header / parties separator as the widest blank gap in the text above the table.
   - `main.py` crops the four zones and saves them.
2. **OCR (Tesseract)**
   - `ocr.py`: runs Tesseract on the upscaled image, returns words with positions, and groups them into lines.
3. **Classification (words only)**
   - `classify.py`: the three non-table zones are assigned to header, parties and summary by keyword score, using the best one-to-one assignment. The table zone is always the items zone.
4. **Parsing**
   - `parse_header.py`: invoice number and date.
   - `parse_parties.py`: seller and client names and addresses.
   - `parse_items.py`: table rows (description, quantity, total). Columns come from the header row. Rows are cut between the table's rules.
   - `parse_summary.py`: tax, total and sub total.
   - `table_cells.py`: erases the table grid lines before OCR, so the text next to them is not garbled.
5. **Validation**
   - `extract_json.py`: the item totals are summed and compared with the summary total. If they do not agree, OCR is re-run with other settings, and rows from different runs can be combined. The result is printed as `check=ok` / `check=mismatch`.

## JSON output

```json
{
  "invoice": {
    "client_name": null, "client_address": null,
    "seller_name": null, "seller_address": null,
    "invoice_number": null, "invoice_date": null
  },
  "items": [
    {"description": null, "quantity": null, "total_price": null}
  ],
  "subtotal": {"tax": null, "total": null}
}
```

Fields that cannot be found are `null`.

## Known problems

- **Zone split fails** on some images: no header separator, or a crash in `detect_table`. These produce no JSON.
- **No bordered table**: on the images the table zone could be wrong, so no items are found.
- **Template-specific rules**: the parsers were tuned on the sample templates. New layouts may need new keywords, regexes or column names.
- **OCR errors**: small or light text, lost decimal separators and misread digits. Some are repaired, but not all. Check the `check=` result before trusting a file.
- **Speed**: about 40 seconds for 25 images, mostly OCR re-runs.