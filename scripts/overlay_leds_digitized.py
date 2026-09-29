"""Draws each lead's digitized signal directly onto its own led crop, using
the crop's own recorded geometry (x_lead_out, px_per_sample, px_per_mv,
baseline_y, y_top) with no additional registration or warp -- writes one
overlay PNG per manifest_leds.csv row. --source-field picks which crop
column to overlay (led_image_path or led_image_noised_rectified_path); both
share the same panel-local frame as y_top/baseline_y.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/overlay_leds_digitized.py -> ecg-generation-synthetic_realistic -> repo -> project root
DATASET_ROOT = PROJECT_ROOT / "datasets" / "ecg-synthetic_realistic"
OVERLAY_COLOR = (220, 20, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATASET_ROOT / "labels" / "manifest_leds.csv")
    parser.add_argument("--out-dir", type=Path, default=DATASET_ROOT / "leds_noised_rectified_overlayed")
    parser.add_argument("--source-field", default="led_image_noised_rectified_path")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows (for a quick sample check)")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = missing = errors = 0
    for row in tqdm(rows, desc="overlay"):
        rectified_rel = row[args.source_field]
        if not rectified_rel:
            missing += 1
            continue

        out_path = args.out_dir / f"{row['panel_id']}_{row['lead_name']}.png"
        if out_path.exists():
            skipped += 1
            continue

        try:
            led_img = Image.open(PROJECT_ROOT / rectified_rel).convert("RGB")
            signal = np.load(PROJECT_ROOT / row["digitized_path"])

            x_lead_out = float(row["x_lead_out"])
            px_per_sample = float(row["px_per_sample"])
            px_per_mv = float(row["px_per_mv"])
            baseline_y = float(row["baseline_y"]) - float(row["y_top"])
            line_width = int(row["trace_line_width_px"]) + 2  # wider than the original stroke so it fully occludes it; two separate rasterizations of the same polyline don't land on identical pixels at every joint

            xs = x_lead_out + np.arange(len(signal)) * px_per_sample
            ys = baseline_y - signal * px_per_mv
            points = list(zip(xs.tolist(), ys.tolist()))

            draw = ImageDraw.Draw(led_img)
            if len(points) >= 2:
                draw.line(points, fill=OVERLAY_COLOR, width=line_width, joint="curve")
            led_img.save(out_path)
            written += 1
        except Exception as exc:
            print(f"error: {row['panel_id']}_{row['lead_name']}: {exc}")
            errors += 1

    print(f"written {written}, skipped {skipped} (already existed), missing {missing} (no {args.source_field}), errors {errors}")
    print(f"out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
