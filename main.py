"""CLI entry point for the LayoutLMv3 invoice extractor.

Commands:
    prepare                  OCR + GT alignment -> cache/dataset/*.json
    train                    fine-tune microsoft/layoutlmv3-base
    infer <image>            extract JSON from one image
    evaluate                 field-level metrics on the val split

Examples:
    python main.py prepare --limit 50
    python main.py prepare --workers 6
    python main.py train
    python main.py infer data\\batch1_1\\batch1-0494.jpg --print
    python main.py evaluate --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Tesseract LSTM shows OpenMP-related nondeterminism with multiple
# threads; pin to one thread for reproducible OCR across runs.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

sys.path.insert(0, str(Path(__file__).parent))


def _prepare_one(args: tuple[str, str]) -> dict | None:
    """Worker: OCR one image + align GT -> a cached dataset record.

    Receives (image_path, gt_json) explicitly: on Windows, child
    processes re-import the module, so module-level state is empty.
    """

    image_path, gt_json = args
    try:
        from modules.align import _reading_order, align_document, words_to_records
        from modules.image_processor import load_image, preprocess_image
        from modules.ocr_engine import run_ocr

        image = load_image(image_path)
        processed = preprocess_image(image)
        words = run_ocr(processed)

        gt_row = json.loads(gt_json)
        labels, stats = align_document(words, gt_row)
        seq, _lines = _reading_order(words)

        return {
            "image": image_path,
            "words": words_to_records(seq, labels),
            "stats": stats,
        }
    except Exception as error:  # noqa: BLE001 - report and continue
        return {"image": image_path, "error": str(error)}


_GT_CACHE: dict[str, dict] = {}


def _load_gt_cache(data_dir: Path) -> dict[str, dict]:
    import csv

    cache: dict[str, dict] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        with open(csv_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_name = row["File Name"].strip()
                image_path = data_dir / csv_path.stem / image_name
                if image_path.exists():
                    cache[str(image_path)] = json.loads(row["Json Data"])
    return cache


def command_prepare(args) -> int:
    data_dir = Path(args.data_dir)
    gt_cache = _load_gt_cache(data_dir)
    if not gt_cache:
        print(f"No CSV + image pairs found in {data_dir}")
        return 2

    items = sorted(gt_cache.items())
    if args.limit:
        items = items[: args.limit]

    out_dir = Path("cache/dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = failed = 0
    stats_total: dict[str, int] = {}

    tasks = [(path, json.dumps(gt)) for path, gt in items]

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_prepare_one, tasks, chunksize=4))
    else:
        results = [_prepare_one(task) for task in tasks]

    from tqdm import tqdm  # local import: keeps CLI startup light

    for record in tqdm(results, desc="saving"):
        if record is None or "error" in record:
            failed += 1
            name = Path(record["image"]).name if record else "?"
            print(f"[FAIL] {name}: {record.get('error') if record else 'no result'}")
            continue

        stem = Path(record["image"]).stem
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as handle:
            json.dump({"image": record["image"], "words": record["words"]},
                      handle, ensure_ascii=False)
        saved += 1

        for key, value in record["stats"].items():
            stats_total[key] = stats_total.get(key, 0) + value

    print(f"\nSaved {saved} documents to {out_dir}, failed: {failed}")

    print("\nALIGNMENT COVERAGE (matched/total):")
    for key in sorted({k.replace("_matched", "") for k in stats_total}):
        matched = stats_total.get(f"{key}_matched", 0)
        total = stats_total.get(f"{key}_total", 0)
        if total:
            percent = 100.0 * matched / total
            print(f"  {key:<20} {matched:>5}/{total:<5} {percent:6.1f}%")

    return 0 if saved else 1


def command_train(args) -> int:
    from modules.train import train

    train(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_docs=args.max_docs,
    )
    return 0


def command_infer(args) -> int:
    from modules.infer import InvoiceModel

    model = InvoiceModel(args.model_dir)
    document = model.extract_json(args.image)

    if args.print:
        print(json.dumps(document, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        print(f"Saved to {out_path}")
    if not args.print and not args.out:
        print(json.dumps(document, ensure_ascii=False, indent=2))

    return 0


def command_evaluate(args) -> int:
    from modules.evaluate_ml import evaluate

    evaluate(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        limit=args.limit,
    )
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="LayoutLMv3 invoice extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="OCR + GT alignment -> cache")
    prepare.add_argument("--data-dir", default="data")
    prepare.add_argument("--limit", type=int, default=0, help="0 = all 499")
    prepare.add_argument("--workers", type=int, default=1, help="parallel OCR workers")
    prepare.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    prepare.set_defaults(func=command_prepare)

    train_parser = subparsers.add_parser("train", help="fine-tune LayoutLMv3")
    train_parser.add_argument("--cache-dir", default="cache/dataset")
    train_parser.add_argument("--output-dir", default="runs/layoutlmv3-invoice")
    train_parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--max-docs", type=int, default=0,
                              help="subsample dataset for quick runs (0 = all)")
    train_parser.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer", help="extract JSON from one image")
    infer.add_argument("image")
    infer.add_argument("--model-dir", default="runs/layoutlmv3-invoice/best")
    infer.add_argument("--print", action="store_true")
    infer.add_argument("--out", default=None, help="save JSON to file")
    infer.set_defaults(func=command_infer)

    evaluate = subparsers.add_parser("evaluate", help="field-level metrics")
    evaluate.add_argument("--model-dir", default="runs/layoutlmv3-invoice/best")
    evaluate.add_argument("--data-dir", default="data")
    evaluate.add_argument("--cache-dir", default="cache/dataset")
    evaluate.add_argument("--split", choices=["val", "train", "test"], default="val")
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.set_defaults(func=command_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
