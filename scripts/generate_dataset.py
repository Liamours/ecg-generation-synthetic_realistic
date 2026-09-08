"""Exports the full synthetic panel-page dataset: composes pages via
compose_page(), then decomposes each page down to its panels and leads,
mirroring this project's real page -> panel -> lead hierarchy (see
wiki/plans/plan.md) so a training pipeline can target each pipeline stage
directly instead of only the final composed page.

Layout, one row per unit at each level:
  pages/page_NNNN.png              -- one full composed page (photo-realistic: tilted, warped, stapled, handwritten-on)
  panels/<panel_id>.png            -- one CANONICAL, pre-transform 3-lead panel render, at its own native resolution -- the rectification target. The as-detected (still tilted/warped) version needing rectification is not stored separately: crop pages/<page_id>.png at manifest_panels.csv's bbox_* to get it, since that's a deterministic function of two already-saved things.
  leds/<panel_id>_<lead>.png        -- one lead's row, cropped from the canonical panel image above
  digitized/<panel_id>_<lead>.npy   -- the exact real PTB-XL sample array drawn as that lead's trace (float32, physical mV)
  labels/manifest_pages.csv        -- one row per page
  labels/manifest_panels.csv       -- one row per panel: page-space bbox + orientation (detection target), tilt_deg/perspective_side/perspective_shift_frac/page_tilt_deg (rectification target), and the calibration step's own bbox (panel-local frame, None when has_calibration_step is False) -- its height reads the panel's own printed gain, already in this row
  labels/manifest_leds.csv         -- one row per lead: crop path, digitized-signal path, the scale (px_per_mv, px_per_second) needed to relate them, and label_bbox_* -- the exact box of that row's own printed lead-name text (I, aVR, V4, ...), panel-local frame

Resumable: a page already present in manifest_pages.csv is skipped, so a
re-run after an interruption continues from where it stopped. One page's
image, panels, leds, digitized arrays, and manifest rows are all written
together before moving to the next page, so a crash mid-run never leaves
a manifest row without its files.

Pages are independent (own rng seed each), so generation is parallelized
across worker processes -- one page per worker task, results collected
and written to the manifests as they complete.
"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_synthetic_realistic.config import GRID_BOX_SECONDS
from ecg_synthetic_realistic.page import PAGE_STYLES, compose_page

GRID_PX_AT_RESOLUTION_200 = 39.37
BASE_SEED = 42
DEFAULT_OUT_DIR = Path(r"C:\research\research-ecg-digitization\datasets\synthetic-ptbxl-panels")
DEFAULT_LOG_DIR = Path(r"C:\research\research-ecg-digitization\logs")
DEFAULT_WORKERS = min(os.cpu_count() or 4, 8)

# 2026-09-08: half the pages standard, the rest split between the two new
# augmentation directions (see page.py's PAGE_STYLES docstring).
PAGE_STYLE_WEIGHTS = {"standard": 0.5, "ideal": 0.25, "torn": 0.25}
STYLE_SEED_OFFSET = 999983  # arbitrary, just far from any real page index so it never collides with a per-page content seed


def _page_styles(count: int, seed: int) -> list[str]:
    """One style per page index, 0..count-1, as a pure function of the
    index and base seed -- independent of a page's own content rng and
    of how many pages are already done, so a resumed run assigns page i
    the exact same style an unbroken run would have."""
    style_rng = random.Random(seed + STYLE_SEED_OFFSET)
    styles = list(PAGE_STYLE_WEIGHTS)
    weights = list(PAGE_STYLE_WEIGHTS.values())
    return [style_rng.choices(styles, weights=weights, k=1)[0] for _ in range(count)]

PAGE_FIELDS = ["page_id", "full_path", "width", "height", "num_pieces", "num_columns"]
PANEL_FIELDS = [
    "panel_id", "page_id", "piece_index", "column_index", "panel_image_path",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "orientation_deg", "column_type", "row_labels",
    "tilt_deg", "perspective_side", "perspective_shift_frac", "page_scale_factor", "page_tilt_deg", "page_tilt_applied",
    "style", "gain", "calibration_step_mode", "has_calibration_step",
    "calibration_step_bbox_x0", "calibration_step_bbox_y0", "calibration_step_bbox_x1", "calibration_step_bbox_y1",
    "bottom_row_color", "record_ref", "ptbxl_record_id",
    "header_text", "footer_top", "footer_bottom",
]
LED_FIELDS = [
    "panel_id", "lead_name", "led_image_path", "digitized_path",
    "label_bbox_x0", "label_bbox_y0", "label_bbox_x1", "label_bbox_y1",
    "y_top", "y_bottom", "num_samples", "signal_fs", "gain", "px_per_mv", "px_per_second",
]


def _page_row(page_id: str, full_path: Path, page) -> dict:
    return {
        "page_id": page_id,
        "full_path": full_path.as_posix(),
        "width": page.image.width,
        "height": page.image.height,
        "num_pieces": len(page.pieces),
        "num_columns": sum(len(piece.columns) for piece in page.pieces),
    }


def _panel_row(panel_id: str, page_id: str, piece_idx: int, col_idx: int, panel_image_path: Path, col, project_root: Path) -> dict:
    meta = col.meta
    # calibration_step_bbox is in panel_image's own pristine local frame
    # (same frame as manifest_leds.csv's y_top/y_bottom), not page space --
    # None when has_calibration_step is False.
    step_bbox = meta["calibration_step_bbox"] or (None, None, None, None)
    return {
        "panel_id": panel_id,
        "page_id": page_id,
        "piece_index": piece_idx,
        "column_index": col_idx,
        "panel_image_path": panel_image_path.relative_to(project_root).as_posix(),
        "bbox_x0": round(col.bbox[0], 2),
        "bbox_y0": round(col.bbox[1], 2),
        "bbox_x1": round(col.bbox[2], 2),
        "bbox_y1": round(col.bbox[3], 2),
        "orientation_deg": col.orientation_deg,
        "column_type": col.column_type,
        "row_labels": ";".join(meta["row_labels"]),
        "tilt_deg": round(meta["tilt_deg"], 3),
        "perspective_side": meta["perspective_side"],
        "perspective_shift_frac": meta["perspective_shift_frac"],
        "page_scale_factor": round(meta["page_scale_factor"], 4),
        "page_tilt_deg": round(meta["page_tilt_deg"], 3),
        "page_tilt_applied": meta["page_tilt_applied"],
        "style": meta["style"],
        "gain": meta["gain"],
        "calibration_step_mode": meta["calibration_step_mode"],
        "has_calibration_step": meta["has_calibration_step"],
        "calibration_step_bbox_x0": step_bbox[0], "calibration_step_bbox_y0": step_bbox[1],
        "calibration_step_bbox_x1": step_bbox[2], "calibration_step_bbox_y1": step_bbox[3],
        "bottom_row_color": "|".join(str(c) for c in meta["bottom_row_color"]),
        "record_ref": meta["record_ref"],
        "ptbxl_record_id": meta["ptbxl_record_id"],
        "header_text": meta["header_text"],
        "footer_top": ";".join(meta["footer_top"]),
        "footer_bottom": ";".join(meta["footer_bottom"]),
    }


def _led_rows(panel_id: str, col, leds_dir: Path, digitized_dir: Path, grid_px: float, project_root: Path) -> list[dict]:
    meta = col.meta
    gain_mm_per_mv = float(meta["gain"].split("mm/mV")[0])
    px_per_mv = (grid_px / 5.0) * gain_mm_per_mv
    px_per_second = grid_px / GRID_BOX_SECONDS

    rows = []
    panel_image = meta["panel_image"]
    for lead_name, (y_top, y_bottom), label_bbox in zip(meta["row_labels"], meta["row_y_ranges"], meta["label_bboxes"]):
        led_image_path = leds_dir / f"{panel_id}_{lead_name}.png"
        panel_image.crop((0, y_top, panel_image.width, y_bottom)).save(led_image_path)

        signal = meta["lead_signals"][lead_name]
        digitized_path = digitized_dir / f"{panel_id}_{lead_name}.npy"
        np.save(digitized_path, signal)

        rows.append({
            "panel_id": panel_id,
            "lead_name": lead_name,
            "led_image_path": led_image_path.relative_to(project_root).as_posix(),
            "digitized_path": digitized_path.relative_to(project_root).as_posix(),
            "label_bbox_x0": label_bbox[0], "label_bbox_y0": label_bbox[1],
            "label_bbox_x1": label_bbox[2], "label_bbox_y1": label_bbox[3],
            "y_top": y_top,
            "y_bottom": y_bottom,
            "num_samples": len(signal),
            "signal_fs": meta["signal_fs"],
            "gain": meta["gain"],
            "px_per_mv": round(px_per_mv, 4),
            "px_per_second": round(px_per_second, 4),
        })
    return rows


def _load_done_page_ids(pages_csv: Path) -> set[str]:
    if not pages_csv.exists():
        return set()
    with pages_csv.open(encoding="utf-8") as f:
        return {row["page_id"] for row in csv.DictReader(f)}


def _generate_one_page(task: tuple[int, int, float, Path, Path, Path, Path, Path]) -> tuple[dict, list[dict], list[dict]]:
    """Runs in a worker process: composes one page, saves the page image,
    every panel's canonical crop, every lead's crop and digitized signal
    array, and returns the three manifests' rows for the main process to
    write (keeps the CSVs single-writer, so resumability stays a simple
    "page_id already present" check with no cross-process file locking)."""
    i, seed, grid_px, pages_dir, panels_dir, leds_dir, digitized_dir, project_root = task
    page_id = f"page_{i:04d}"
    rng = random.Random(seed)
    page = compose_page(rng, grid_px=grid_px)

    page_image_path = pages_dir / f"{page_id}.png"
    page.image.save(page_image_path)
    page_row = _page_row(page_id, page_image_path.relative_to(project_root), page)

    panel_rows, led_rows = [], []
    for piece_idx, piece in enumerate(page.pieces):
        for col_idx, col in enumerate(piece.columns):
            panel_id = f"{page_id}_p{piece_idx}_c{col_idx}"
            panel_image_path = panels_dir / f"{panel_id}.png"
            col.meta["panel_image"].save(panel_image_path)

            panel_rows.append(_panel_row(panel_id, page_id, piece_idx, col_idx, panel_image_path, col, project_root))
            led_rows.extend(_led_rows(panel_id, col, leds_dir, digitized_dir, grid_px, project_root))

    return page_row, panel_rows, led_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--grid-px", type=float, default=GRID_PX_AT_RESOLUTION_200)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    pages_dir = args.out_dir / "pages"
    panels_dir = args.out_dir / "panels"
    leds_dir = args.out_dir / "leds"
    digitized_dir = args.out_dir / "digitized"
    labels_dir = args.out_dir / "labels"
    for d in (pages_dir, panels_dir, leds_dir, digitized_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = DEFAULT_LOG_DIR / f"synthetic-ptbxl-panels-generate-{datetime.now():%Y%m%d-%H%M}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")

    pages_csv = labels_dir / "manifest_pages.csv"
    panels_csv = labels_dir / "manifest_panels.csv"
    leds_csv = labels_dir / "manifest_leds.csv"
    done_page_ids = _load_done_page_ids(pages_csv)
    project_root = args.out_dir.parents[1]

    pending = [i for i in range(args.count) if f"page_{i:04d}" not in done_page_ids]
    logging.info("starting run: %d total, %d already done, %d pending, %d workers", args.count, len(done_page_ids), len(pending), args.workers)
    print(f"{len(done_page_ids)} pages already done, {len(pending)} pending, {args.workers} workers")

    if not pending:
        print("nothing to do")
        return

    tasks = [(i, args.seed + i, args.grid_px, pages_dir, panels_dir, leds_dir, digitized_dir, project_root) for i in pending]
    start_time = time.time()

    with pages_csv.open("a", newline="", encoding="utf-8") as pf, panels_csv.open("a", newline="", encoding="utf-8") as nf, leds_csv.open("a", newline="", encoding="utf-8") as lf:
        page_writer = csv.DictWriter(pf, fieldnames=PAGE_FIELDS)
        panel_writer = csv.DictWriter(nf, fieldnames=PANEL_FIELDS)
        led_writer = csv.DictWriter(lf, fieldnames=LED_FIELDS)
        if pf.tell() == 0:
            page_writer.writeheader()
        if nf.tell() == 0:
            panel_writer.writeheader()
        if lf.tell() == 0:
            led_writer.writeheader()

        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_generate_one_page, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="pages"):
                page_row, panel_rows, led_rows = future.result()
                page_writer.writerow(page_row)
                for row in panel_rows:
                    panel_writer.writerow(row)
                for row in led_rows:
                    led_writer.writerow(row)
                pf.flush()
                nf.flush()
                lf.flush()

                completed += 1
                elapsed = time.time() - start_time
                remaining = len(pending) - completed
                eta = timedelta(seconds=round(elapsed / completed * remaining)) if completed else None
                logging.info("wrote %s (%d/%d, elapsed %s, ETA %s)", page_row["page_id"], completed, len(pending), timedelta(seconds=round(elapsed)), eta)

    print(f"pages written to: {pages_dir}")
    print(f"panels written to: {panels_dir}")
    print(f"leds + digitized written to: {leds_dir}, {digitized_dir}")
    print(f"manifests: {pages_csv}, {panels_csv}, {leds_csv}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
