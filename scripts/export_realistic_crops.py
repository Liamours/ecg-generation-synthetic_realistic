"""Exports two QA/training artifacts derived from an already-generated
synthetic dataset, using the recorded ground-truth labels directly -- no
detector run needed, the exact bbox is already known:

  panels_detected/<panel_id>.png   -- crops pages/<page_id>.png at each
                                       panel's page-space bbox (from
                                       manifest_panels.csv), carrying every
                                       page-level augmentation (table
                                       background, 3D warp, tilt, shadow,
                                       staples, degradation) the pristine
                                       panels/ folder deliberately excludes
                                       -- the "as-detected, before
                                       correction" stage a real detector
                                       would produce, matching this
                                       project's own panels_paper/
                                       panels_detected convention (see
                                       wiki/overview/directory.md).

  digitized_overlay/<panel_id>_<lead>.png -- the pristine leds/ crop with
                                       its own digitized/*.npy ground truth
                                       redrawn on top in red, so a mismatch
                                       between the recorded signal and the
                                       actual printed ink is visible
                                       directly. Drawn on the PRISTINE crop,
                                       not the fully-warped page: only each
                                       column's own axis-aligned bbox is
                                       tracked through the row-tilt + 3D
                                       page warp, not a per-sample-point
                                       forward transform, so an accurate
                                       post-warp overlay isn't possible
                                       without building that tracking first.

Usage:
    uv run python scripts/export_realistic_crops.py --dataset-dir ../../datasets/ecg-synthetic_realistic
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_synthetic_realistic.config import GRID_BOX_SECONDS
from ecg_synthetic_realistic.template import signal_start_x

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/export_realistic_crops.py -> repo -> project root
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def export_panel_crops(dataset_dir: Path, logger: logging.Logger) -> None:
    panels_csv = dataset_dir / "labels" / "manifest_panels.csv"
    out_dir = dataset_dir / "panels_detected"
    out_dir.mkdir(exist_ok=True)
    with panels_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = {p.stem for p in out_dir.glob("*.png")}
    pending = [r for r in rows if r["panel_id"] not in done]

    current_page_id, current_page_img = None, None
    for row in tqdm(pending, desc="panels_detected"):
        page_id = row["page_id"]
        if page_id != current_page_id:
            current_page_img = Image.open(dataset_dir / "pages" / f"{page_id}.png").convert("RGB")
            current_page_id = page_id
        x0, y0, x1, y1 = (float(row[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
        crop = current_page_img.crop((round(x0), round(y0), round(x1), round(y1)))
        crop.save(out_dir / f"{row['panel_id']}.png")

    msg = f"panels_detected: {len(pending)} cropped, {len(done)} already done -> {out_dir}"
    print(msg)
    logger.info(msg)


def export_digitized_overlay(dataset_dir: Path, logger: logging.Logger) -> None:
    leds_csv = dataset_dir / "labels" / "manifest_leds.csv"
    out_dir = dataset_dir / "digitized_overlay"
    out_dir.mkdir(exist_ok=True)
    with leds_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = {p.stem for p in out_dir.glob("*.png")}
    pending = [r for r in rows if f"{r['panel_id']}_{r['lead_name']}" not in done]

    for row in tqdm(pending, desc="digitized_overlay"):
        led_img = Image.open(PROJECT_ROOT / row["led_image_path"]).convert("RGB")
        signal = np.load(PROJECT_ROOT / row["digitized_path"])

        px_per_mv = float(row["px_per_mv"])
        px_per_second = float(row["px_per_second"])
        signal_fs = float(row["signal_fs"])
        px_per_sample = px_per_second / signal_fs
        y_top, y_bottom = float(row["y_top"]), float(row["y_bottom"])
        baseline_y_local = (y_bottom - y_top) / 2  # led crop's own local frame starts at y_top, matching template.py's build_panel_template
        grid_px = px_per_second * GRID_BOX_SECONDS  # inverts px_per_second's own derivation (grid_px / GRID_BOX_SECONDS) -- no separate stored constant needed
        x_lead_out = signal_start_x(grid_px)

        points = [(x_lead_out + i * px_per_sample, baseline_y_local - sample * px_per_mv) for i, sample in enumerate(signal)]
        if len(points) >= 2:
            ImageDraw.Draw(led_img).line(points, fill=(220, 30, 30), width=1, joint="curve")
        led_img.save(out_dir / f"{row['panel_id']}_{row['lead_name']}.png")

    msg = f"digitized_overlay: {len(pending)} drawn, {len(done)} already done -> {out_dir}"
    print(msg)
    logger.info(msg)


def main() -> None:
    args = parse_args()
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DEFAULT_LOG_DIR / f"export_realistic_crops_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    logger = logging.getLogger("export_realistic_crops")
    print(f"log -> {log_path}")

    export_panel_crops(args.dataset_dir, logger)
    export_digitized_overlay(args.dataset_dir, logger)


if __name__ == "__main__":
    main()
