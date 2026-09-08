"""CLI for step 1 (wiki/plans/plan.md): render PTB-XL records and crop them
into per-lead ground-truth images + signal arrays.

Usage:
    uv run python scripts/generate_leads.py --count 20 --output-root ../../datasets/synthetic-ecg-realistic
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_synthetic_realistic import config
from ecg_synthetic_realistic.step1_leads import generate_leads_for_records


def pick_record_ids(count: int, seed: int) -> list[str]:
    hea_files = sorted(config.PTBXL_DATA_DIR.glob("*_hr.hea"))
    if not hea_files:
        raise FileNotFoundError(f"no *_hr.hea files under {config.PTBXL_DATA_DIR}")
    record_ids = [f.stem for f in hea_files]
    random.Random(seed).shuffle(record_ids)
    return record_ids[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20, help="number of PTB-XL records to render")
    parser.add_argument("--output-root", type=Path, default=config.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    record_ids = pick_record_ids(args.count, args.seed)
    print(f"rendering {len(record_ids)} records -> {args.output_root}")

    results = generate_leads_for_records(
        record_ids=record_ids,
        output_root=args.output_root,
        resolution=args.resolution,
        seed=args.seed,
    )

    manifest_path = args.output_root / "leads" / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id", "record_id", "column_index", "lead_name",
                "lead_png_path", "signal_npy_path",
                "x0", "y0", "x1", "y1",
                "start_sample", "end_sample", "sampling_frequency",
                "x_grid", "y_grid",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    f"{r.record_id}-c{r.column_index}-{r.lead_name}",
                    r.record_id, r.column_index, r.lead_name,
                    r.lead_png_path.relative_to(args.output_root),
                    r.signal_npy_path.relative_to(args.output_root),
                    *r.bbox_xyxy,
                    r.start_sample, r.end_sample, r.sampling_frequency,
                    r.x_grid, r.y_grid,
                ]
            )

    print(f"wrote {len(results)} lead crops, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
