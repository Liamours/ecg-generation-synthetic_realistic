"""Draws each page's panel bboxes (blue) and staple bboxes (orange) directly
on top of the page image, from manifest_panels.csv's own page-space
bbox_x0..y1 and staple_bboxes columns -- no transform, one overlay PNG per
page into pages_overlayed/.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/overlay_pages_bboxes.py -> ecg-generation-synthetic_realistic -> repo -> project root
DATASET_ROOT = PROJECT_ROOT / "datasets" / "ecg-synthetic_realistic"
PANEL_COLOR = (30, 90, 220)
STAPLE_COLOR = (230, 120, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-manifest", type=Path, default=DATASET_ROOT / "labels" / "manifest_pages.csv")
    parser.add_argument("--panels-manifest", type=Path, default=DATASET_ROOT / "labels" / "manifest_panels.csv")
    parser.add_argument("--out-dir", type=Path, default=DATASET_ROOT / "pages_overlayed")
    parser.add_argument("--box-width", type=int, default=4)
    args = parser.parse_args()

    with args.pages_manifest.open(encoding="utf-8") as f:
        pages = list(csv.DictReader(f))
    with args.panels_manifest.open(encoding="utf-8") as f:
        panels_by_page: dict[str, list[dict]] = defaultdict(list)
        for row in csv.DictReader(f):
            panels_by_page[row["page_id"]].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = errors = 0
    for page in tqdm(pages, desc="overlay pages"):
        out_path = args.out_dir / f"{page['page_id']}.png"
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(PROJECT_ROOT / page["full_path"]).convert("RGB")
            draw = ImageDraw.Draw(img)
            for panel in panels_by_page.get(page["page_id"], []):
                box = (float(panel["bbox_x0"]), float(panel["bbox_y0"]), float(panel["bbox_x1"]), float(panel["bbox_y1"]))
                draw.rectangle(box, outline=PANEL_COLOR, width=args.box_width)
                for staple_box in json.loads(panel["staple_bboxes"] or "[]"):
                    draw.rectangle(tuple(staple_box), outline=STAPLE_COLOR, width=args.box_width)
            img.save(out_path)
            written += 1
        except Exception as exc:
            print(f"error: {page['page_id']}: {exc}")
            errors += 1

    print(f"written {written}, skipped {skipped} (already existed), errors {errors}")
    print(f"out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
