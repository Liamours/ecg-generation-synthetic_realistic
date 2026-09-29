"""Adds trace-only ground truth masks to the existing 1000-page corpus
without re-running generation (2026-09-15, see TODO.md's digitizer
architecture-comparison item). A full regeneration would re-render every
page/panel/led from scratch at the original run's full CPU/time cost just
to add one new file per lead; everything the mask needs is already sitting
in manifest_leds.csv plus the saved digitized/*.npy signal, so this
reconstructs it directly instead.

Exact, not approximate: template.py draws each trace at x_lead_out + i *
px_per_sample, baseline_y - sample * px_per_mv, where x_lead_out depends
only on the fixed grid_px this whole corpus used (a single constant, not
per-panel), and baseline_y is simply the row's own vertical center once
working in the led crop's own coordinate frame (y_top subtracted out) --
both reconstructable from manifest_leds.csv's existing columns with no
information the original render had that this script lacks.

Usage:
    uv run python scripts/backfill_trace_masks.py \
        --manifest-dir ../../datasets/ecg-synthetic_realistic/labels \
        --project-root ../.. \
        --grid-px 39.37
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecg_synthetic_realistic.template import signal_start_x  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--grid-px", type=float, required=True, help="the fixed grid_px this corpus was generated with (GRID_PX_AT_RESOLUTION_200 in generate_dataset.py)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N leds -- for a quick timing/correctness check before a full run")
    args = parser.parse_args()

    x_lead_out = signal_start_x(args.grid_px)
    leds_csv = args.manifest_dir / "manifest_leds.csv"
    full_leds = pd.read_csv(leds_csv)
    leds = full_leds.head(args.limit) if args.limit else full_leds
    leds_dir = args.project_root / Path(leds.iloc[0]["led_image_path"]).parent
    masks_dir = leds_dir.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = []
    for row in tqdm(leds.itertuples(), total=len(leds), desc="backfilling masks"):
        signal = np.load(args.project_root / row.digitized_path).astype(np.float64)
        height = int(row.y_bottom - row.y_top)
        led_image = Image.open(args.project_root / row.led_image_path)
        width = led_image.width
        px_per_sample = row.px_per_second / row.signal_fs
        baseline_y = height / 2.0

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        points = [(x_lead_out + i * px_per_sample, baseline_y - sample * row.px_per_mv) for i, sample in enumerate(signal)]
        if len(points) >= 2:
            draw.line(points, fill=255, width=2, joint="curve")

        mask_path = masks_dir / f"{row.panel_id}_{row.lead_name}.png"
        mask.save(mask_path)
        mask_paths.append(mask_path.relative_to(args.project_root).as_posix())

    print(f"{len(leds)} masks written to {masks_dir}")
    if args.limit:
        print(f"--limit set: manifest NOT updated, rerun without --limit for the full {len(full_leds)}-row backfill")
    else:
        full_leds["mask_image_path"] = mask_paths
        full_leds.to_csv(leds_csv, index=False)
        print(f"manifest updated: {leds_csv}")


if __name__ == "__main__":
    main()
