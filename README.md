# invoice-layoutlm

Invoice information extraction from scanned documents (image → JSON) by fine-tuning **Microsoft LayoutLMv3** for token classification with BIO tagging.

## Pipeline

```
prepare   : OCR (image preprocessing + Tesseract) + ground-truth alignment → BIO labels → cache/
train     : fine-tune microsoft/layoutlmv3-base (fp16, GPU)
infer     : model → BIO tags → fields → JSON
evaluate  : field-level accuracy on the held-out validation set
```

## Data (not included)

The repository contains **code only**. Training/evaluation uses a private set of 499 invoice scans (single template, English) with full ground-truth JSON annotations stored in CSV (`File Name`, `Json Data`, `OCRed Text` columns). Place the images and ground-truth CSV under `data/` before running `prepare`.

## Trained model (not included)

The fine-tuned checkpoint is **not uploaded** (weights exceed GitHub size limits; `runs/` is git-ignored). Reproduce it locally:

```bash
python main.py prepare          # OCR + alignment → cache/
python main.py train            # ~25 min on a GTX 1660 Ti 6 GB (fp16)
```

## Accuracy

Evaluated on 50 held-out documents. *Value-level* = normalized string match (case/whitespace/punctuation-insensitive); F1 = seqeval entity-level score on BIO tags.

| Field | Value-level accuracy |
|---|---|
| invoice_number / invoice_date / due_date / discount | 100% |
| subtotal.tax | 46% |
| subtotal.total | 28% |
| client_name | 38% |
| client_address | 70% |
| seller_name | 40% |
| seller_address | 16% |
| **items.description** | 93.7% |
| **items.quantity** | 100% |
| **items.total_price** | 29.3% |

**Overall value-level: 63.8%** · token-level F1: **0.813** · token accuracy: 93.6%

## Usage

```bash
python main.py prepare --limit 50     # OCR + alignment → cache/ (small run first)
python main.py train                  # fine-tune LayoutLMv3
python main.py evaluate               # field-level metrics on the validation split
python main.py infer data/invoices/sample.jpg --print
```

## Modules

- `modules/ocr_engine.py`, `image_processor.py`, `labels.py` — OCR, image preprocessing, value validators
- `modules/align.py` — ground-truth ↔ OCR word alignment (fuzzy matching + coordinates) → BIO labels
- `modules/dataset.py` — PyTorch dataset for LayoutLMv3
- `modules/model.py` — LayoutLMv3 token-classification wrapper
- `modules/train.py` — training loop (HF Trainer, fp16)
- `modules/infer.py` — inference → JSON
- `modules/evaluate_ml.py` — field-level metrics
