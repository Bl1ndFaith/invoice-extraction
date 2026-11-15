import glob
import os
import sys

import cv2

from modules.extract_json import extract_invoice, save_json
from modules.find_zones import find_split_line
from modules.table_detect import detect_table
from modules.visual_lines import draw_labeled_lines, draw_lines

INPUT_DIRS = ["data", "data_test"]
OUTPUT_ROOT = "outputs"


def process(path, out_dir):
    img = cv2.imread(path)
    if img is None:
        return f"cannot read image"

    y_top, y_bottom = detect_table(img)
    y_split = find_split_line(img, y_top, y_bottom)
    if y_split is None:
        return "no header separator found"
    ys = sorted([y_split, y_top, y_bottom])

    os.makedirs(out_dir, exist_ok=True)

    bounds = [0, *ys, img.shape[0]]
    n = 0
    zone_names, zones = [], {}
    for i in range(4):
        zone = img[bounds[i]:bounds[i + 1]]
        if bounds[i] == y_top and bounds[i + 1] == y_bottom:
            name = "zone_table"
        else:
            n += 1
            name = f"zone_{n}"
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), zone)
        zone_names.append(name)
        zones[name] = zone

    # OCR: the three non-table zones are classified by their words, the table zone is the items
    table = zones.pop("zone_table")
    data, labels, check = extract_invoice(table, zones)
    save_json(data, os.path.join(out_dir, "invoice.json"))

    categories = [labels.get(n, "items") for n in zone_names]
    cv2.imwrite(os.path.join(out_dir, "lines_labeled.png"), draw_labeled_lines(img, ys, categories))
    return (f"split={ys[0]} table_top={ys[1]} table_bottom={ys[2]} "
            f"items={len(data['items'])} check={check} zones={categories}")


def main():
    files = []
    for d in INPUT_DIRS:
        files += sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))

    failed = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            status = process(path, os.path.join(OUTPUT_ROOT, name))
        except Exception as e:  # keep going on bad files
            status = f"error: {e}"
        if status.startswith(("cannot", "no ", "error")):
            failed.append(path)
        print(f"{path}: {status}")

    print(f"\nprocessed {len(files)} files, failed {len(failed)}")
    for p in failed:
        print("  FAILED:", p)


if __name__ == "__main__":
    main()
