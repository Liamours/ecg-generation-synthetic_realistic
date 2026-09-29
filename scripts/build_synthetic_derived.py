"""Builds datasets/ecg-synthetic_realistic-derived/{panels,leds}/ from the raw,
fully-augmented pages in datasets/ecg-synthetic_realistic/pages/, using the
recorded labels to reconstruct each panel's true geometry -- not by copying
the generator's pristine pre-composition render, and not by re-detecting
anything from pixels.

Per panel: crop the page at its recorded bbox, undo the exact forward
transform chain (orientation, tilt, warp, page tilt) via the four recorded
corner points (a single homography, panel_pt0..3), then snap the residual
resampling error to the pristine truth with ECC. The delivered pixels still
come entirely from the noisy, augmented page crop -- ECC only decides how to
resample it, never what to draw. This is the same reconstruction
build_digitization_training_set.py already validated (mean ground-truth MAE
43.60/255, exact homography + ECC beating the old label-only chain on every
`ideal`-style panel) -- reused here via import, not reimplemented.

Per lead: crops the reconstructed canonical (709x591) panel at its recorded
y_top/y_bottom row bounds, already given in that same canonical frame -- no
further correction needed, and no approximation, since the row bounds are the
generator's own recorded values.

Usage:
    uv run python scripts/build_synthetic_derived.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_digitization_training_set import (  # noqa: E402
    CANONICAL_HEIGHT_PX,
    CANONICAL_WIDTH_PX,
    PROJECT_ROOT,
    ecc_refine,
    make_realistic,
    make_realistic_exact,
    panel_suffix,
)

DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "ecg-synthetic_realistic"
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "ecg-synthetic_realistic-derived"

PANEL_MANIFEST_COLUMNS = ["panel_id", "page_id", "panel_path", "reconstruction", "ecc_correlation"]
LED_MANIFEST_COLUMNS = ["panel_id", "lead_name", "led_path", "panel_path", "y_top", "y_bottom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--page-ids", nargs="+", default=None, help="omit to process every page_id in the manifest")
    return parser.parse_args()


def reconstruct_panel(page_bgr: np.ndarray, page_img: Image.Image, row: dict) -> tuple[np.ndarray, str, float | None]:
    exact = make_realistic_exact(page_bgr, row)
    if exact is not None:
        reconstructed, reconstruction = exact, "exact_homography"
    else:
        x0, y0, x1, y1 = (round(float(row[f"bbox_{k}"])) for k in ("x0", "y0", "x1", "y1"))
        distorted_bgr = cv2.cvtColor(np.array(page_img.crop((x0, y0, x1, y1))), cv2.COLOR_RGB2BGR)
        reconstructed, _residual_source = make_realistic(distorted_bgr, row)
        reconstruction = "label_reconstruction"

    truth_bgr = cv2.imread(str(PROJECT_ROOT / row["panel_image_path"]))
    reconstructed, ecc_correlation = ecc_refine(reconstructed, truth_bgr)
    return reconstructed, reconstruction, ecc_correlation


def main() -> None:
    args = parse_args()
    labels_dir = args.dataset_dir / "labels"
    with (labels_dir / "manifest_panels.csv").open(encoding="utf-8") as f:
        panel_rows = list(csv.DictReader(f))
    with (labels_dir / "manifest_leds.csv").open(encoding="utf-8") as f:
        led_rows = list(csv.DictReader(f))

    if args.page_ids is not None:
        wanted = set(args.page_ids)
        panel_rows = [r for r in panel_rows if r["page_id"] in wanted]

    leds_by_panel: dict[str, list[dict]] = defaultdict(list)
    for row in led_rows:
        leds_by_panel[row["panel_id"]].append(row)

    pages: dict[str, list[dict]] = defaultdict(list)
    for row in panel_rows:
        pages[row["page_id"]].append(row)

    panels_dir = args.out_dir / "panels"
    leds_dir = args.out_dir / "leds"
    out_labels_dir = args.out_dir / "labels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    leds_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    panel_manifest_path = out_labels_dir / "manifest_panels.csv"
    led_manifest_path = out_labels_dir / "manifest_leds.csv"
    done_page_ids: set[str] = set()
    if panel_manifest_path.exists():
        with panel_manifest_path.open(encoding="utf-8") as f:
            done_page_ids = {r["page_id"] for r in csv.DictReader(f)}
    pending_pages = {pid: rows for pid, rows in pages.items() if pid not in done_page_ids}

    panel_write_header = not panel_manifest_path.exists() or panel_manifest_path.stat().st_size == 0
    led_write_header = not led_manifest_path.exists() or led_manifest_path.stat().st_size == 0
    correlations, n_panels, n_leds = [], 0, 0

    with panel_manifest_path.open("a", newline="", encoding="utf-8") as pf, \
         led_manifest_path.open("a", newline="", encoding="utf-8") as lf:
        panel_writer = csv.DictWriter(pf, fieldnames=PANEL_MANIFEST_COLUMNS)
        led_writer = csv.DictWriter(lf, fieldnames=LED_MANIFEST_COLUMNS)
        if panel_write_header:
            panel_writer.writeheader()
        if led_write_header:
            led_writer.writeheader()

        for page_id, rows in tqdm(pending_pages.items(), desc="pages", initial=len(done_page_ids), total=len(pages)):
            page_img = Image.open(args.dataset_dir / "pages" / f"{page_id}.png").convert("RGB")
            page_bgr = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)

            for row in rows:
                panel_id = row["panel_id"]
                reconstructed, reconstruction, ecc_correlation = reconstruct_panel(page_bgr, page_img, row)

                panel_out_dir = panels_dir / page_id
                panel_out_dir.mkdir(exist_ok=True)
                panel_path = panel_out_dir / f"{panel_suffix(panel_id, page_id)}.png"
                cv2.imwrite(str(panel_path), reconstructed)
                if ecc_correlation is not None:
                    correlations.append(ecc_correlation)
                panel_writer.writerow({
                    "panel_id": panel_id, "page_id": page_id, "panel_path": str(panel_path),
                    "reconstruction": reconstruction, "ecc_correlation": ecc_correlation,
                })
                n_panels += 1

                for led_row in leds_by_panel.get(panel_id, []):
                    y_top, y_bottom = round(float(led_row["y_top"])), round(float(led_row["y_bottom"]))
                    led_crop = reconstructed[y_top:y_bottom, 0:CANONICAL_WIDTH_PX]
                    led_path = leds_dir / f"{panel_id}_{led_row['lead_name']}.png"
                    cv2.imwrite(str(led_path), led_crop)
                    led_writer.writerow({
                        "panel_id": panel_id, "lead_name": led_row["lead_name"], "led_path": str(led_path),
                        "panel_path": str(panel_path), "y_top": y_top, "y_bottom": y_bottom,
                    })
                    n_leds += 1
            pf.flush()
            lf.flush()

    print(f"{n_panels} panels, {n_leds} leds this run ({len(done_page_ids)} pages already done) -> {args.out_dir}")
    if correlations:
        print(f"ECC correlation: mean {np.mean(correlations):.4f}, min {np.min(correlations):.4f}, {n_panels - len(correlations)} panels fell back to unrefined (ECC did not converge)")


if __name__ == "__main__":
    main()
