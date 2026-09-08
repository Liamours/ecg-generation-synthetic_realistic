"""CLI: build a standalone panel template (no lead signal yet).

Usage:
    uv run python scripts/generate_template.py --out template.png --row-labels I II III
    uv run python scripts/generate_template.py --out template.png --style even --calibration-step free
    uv run python scripts/generate_template.py --out-dir batch/ --count 20 --random --seed 0
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_synthetic_realistic.template import build_panel_template
from ecg_synthetic_realistic.variants import sample_panel_variant

GRID_PX_AT_RESOLUTION_200 = 39.37


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="single output file")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory for --count > 1")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--row-labels", nargs=3, default=None)
    parser.add_argument("--style", choices=["odd", "even"], default="odd",
                         help="odd = MAC 400/V1.02 header, Man/speed/gain/code footer; "
                              "even = GE date/time header, notch/bandpass/BPM footer -- same sizes, mutually exclusive")
    parser.add_argument("--calibration-step", choices=["correlated", "free", "none"], default=None)
    parser.add_argument("--random", action="store_true", help="sample style + row-labels randomly per image (see variants.py)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.count > 1 and args.out_dir is None:
        parser.error("--count > 1 requires --out-dir")
    if args.count == 1 and args.out is None and args.out_dir is None:
        args.out = Path("template.png")

    rng = random.Random(args.seed)

    for i in range(args.count):
        if args.random:
            kwargs = sample_panel_variant(rng)
        else:
            kwargs = {
                "row_labels": args.row_labels or ["I", "II", "III"],
                "style": args.style,
            }
            if args.calibration_step is not None:
                kwargs["calibration_step_mode"] = args.calibration_step

        result = build_panel_template(grid_px=GRID_PX_AT_RESOLUTION_200, record_ref=f"{i:07d}", rng=rng, **kwargs)

        out_path = args.out if args.count == 1 and args.out else (args.out_dir / f"panel-{i:03d}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(out_path)
        print(f"wrote {out_path} ({result.image.width}x{result.image.height}) row_labels={kwargs['row_labels']}")


if __name__ == "__main__":
    main()
