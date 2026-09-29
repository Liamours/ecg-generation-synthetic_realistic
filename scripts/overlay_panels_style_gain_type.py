"""Draws each panel's style (header + footer text fields), gain
(calibration step), and type (lead label) bboxes on top of its
noised+rectified crop, straight from manifest_panels.csv's own panel-local
text_fields / calibration_step_bbox columns -- these already share the
rectified crop's frame (see _rectify_panel_crop), so no transform is
needed. One overlay PNG per panel into panels_noised_rectified_overlayed/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/overlay_panels_style_gain_type.py -> ecg-generation-synthetic_realistic -> repo -> project root
DATASET_ROOT = PROJECT_ROOT / "datasets" / "ecg-synthetic_realistic"
STYLE_COLOR = (30, 90, 220)
GAIN_COLOR = (230, 120, 20)
TYPE_COLOR = (0, 160, 60)
BOX_WIDTH = 3


def _role_color(role: str) -> tuple[int, int, int] | None:
    if role == "header" or role.startswith("footer_top_") or role.startswith("footer_bottom_"):
        return STYLE_COLOR
    if role.startswith("lead_label_"):
        return TYPE_COLOR
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATASET_ROOT / "labels" / "manifest_panels.csv")
    parser.add_argument("--out-dir", type=Path, default=DATASET_ROOT / "panels_noised_rectified_overlayed")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = missing = errors = 0
    for row in tqdm(rows, desc="overlay panels"):
        rectified_rel = row["panel_image_noised_rectified_path"]
        if not rectified_rel:
            missing += 1
            continue

        out_path = args.out_dir / f"{row['panel_id']}.png"
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = Image.open(PROJECT_ROOT / rectified_rel).convert("RGB")
            draw = ImageDraw.Draw(img)

            for tf in json.loads(row["text_fields"] or "[]"):
                color = _role_color(tf["role"])
                if color is not None:
                    draw.rectangle(tuple(tf["bbox"]), outline=color, width=BOX_WIDTH)

            if row["calibration_step_bbox_x0"]:
                box = (float(row["calibration_step_bbox_x0"]), float(row["calibration_step_bbox_y0"]),
                       float(row["calibration_step_bbox_x1"]), float(row["calibration_step_bbox_y1"]))
                draw.rectangle(box, outline=GAIN_COLOR, width=BOX_WIDTH)

            img.save(out_path)
            written += 1
        except Exception as exc:
            print(f"error: {row['panel_id']}: {exc}")
            errors += 1

    print(f"written {written}, skipped {skipped} (already existed), missing {missing} (no rectified crop), errors {errors}")
    print(f"out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
