"""CLI: build a connected odd+even panel pair (same leads, continued
across two joined panels -- see template.build_connected_panel_template).

Usage:
    uv run python scripts/generate_connected_template.py --out connected.png --row-labels I II III
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_synthetic_realistic.template import build_connected_panel_template

GRID_PX_AT_RESOLUTION_200 = 39.37


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("connected.png"))
    parser.add_argument("--row-labels", nargs=3, default=["I", "II", "III"])
    parser.add_argument("--even-calibration-step", choices=["correlated", "free", "none"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image, left, right = build_connected_panel_template(
        row_labels=args.row_labels,
        grid_px=GRID_PX_AT_RESOLUTION_200,
        rng=random.Random(args.seed),
        even_calibration_step_mode=args.even_calibration_step,
    )
    image.save(args.out)
    print(f"wrote {args.out} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
