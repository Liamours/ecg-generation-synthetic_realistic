"""Verification figure: overlays each sampled lead's digitized signal
(cyan) on top of its own led crop image, in the same pixel coordinates
template.py drew it in originally (x_lead_out, px_per_sample, px_per_mv,
baseline at the crop's own vertical center) -- confirms the two files
actually agree with each other, not just that both exist.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecg_synthetic_realistic.config import GRID_BOX_SECONDS  # noqa: E402
from ecg_synthetic_realistic.template import signal_start_x  # noqa: E402

DATASET_ROOT = Path(r"C:\research\research-ecg-digitization\datasets\synthetic-ptbxl-panels")
PROJECT_ROOT = Path(r"C:\research\research-ecg-digitization")
GRID_PX_AT_RESOLUTION_200 = 39.37
OUT_DIR = Path(r"C:\research\research-ecg-digitization\analyses\synthetic-digitzation-leds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with (DATASET_ROOT / "labels" / "manifest_leds.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.count)

    x_lead_out = signal_start_x(GRID_PX_AT_RESOLUTION_200)

    ncols = 2
    nrows = -(-len(sample) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = axes.flatten()

    for ax, row in zip(axes, sample):
        led_img = Image.open(PROJECT_ROOT / row["led_image_path"]).convert("RGB")
        signal = np.load(PROJECT_ROOT / row["digitized_path"])

        px_per_second = float(row["px_per_second"])
        signal_fs = float(row["signal_fs"])
        px_per_sample = px_per_second / signal_fs
        px_per_mv = float(row["px_per_mv"])
        baseline_y = led_img.height / 2.0

        xs = x_lead_out + np.arange(len(signal)) * px_per_sample
        ys = baseline_y - signal * px_per_mv

        ax.imshow(led_img)
        ax.plot(xs, ys, color="cyan", linewidth=1.2)
        ax.set_title(f"{row['panel_id']} / {row['lead_name']}  (gain={row['gain']})", fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle("Led crop vs. digitized signal overlay (cyan = digitized .npy, drawn at its own recorded scale)", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "led_digitized_overlay.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
