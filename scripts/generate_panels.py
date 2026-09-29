"""CLI for step 2 (wiki/overview/synthetic_data_flow.md): assemble step 1's per-lead crops into
full panel images with header/footer/label text.

Usage:
    uv run python scripts/generate_panels.py --leads-root ../../datasets/ecg-synthetic_realistic
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from ecg_synthetic_realistic import config
from ecg_synthetic_realistic.step2_panels import assemble_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads-root", type=Path, default=config.DEFAULT_OUTPUT_ROOT,
                         help="output-root used by generate_leads.py (must contain leads/manifest.csv)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    leads_manifest = args.leads_root / "leads" / "manifest.csv"
    with open(leads_manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["record_id"], int(row["column_index"]))
        groups[key][row["lead_name"]] = row

    panels_dir = args.leads_root / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    manifest_rows = []
    for (record_id, column_index), by_lead in groups.items():
        lead_crops = {
            name: Image.open(args.leads_root / row["lead_png_path"])
            for name, row in by_lead.items()
        }
        lead_bboxes = {
            name: (int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"]))
            for name, row in by_lead.items()
        }
        grid_px = float(next(iter(by_lead.values()))["x_grid"])

        result = assemble_panel(lead_crops, lead_bboxes, column_index, record_id, grid_px=grid_px, rng=rng)

        panel_path = panels_dir / f"{record_id}-p{column_index}.png"
        result.panel_image.save(panel_path)

        manifest_rows.append({
            "id": f"{record_id}-p{column_index}",
            "record_id": record_id,
            "column_index": column_index,
            "panel_path": panel_path.relative_to(args.leads_root),
            "lead_names": "/".join(result.lead_names),
            "lead_y_ranges": ";".join(f"{y0}-{y1}" for y0, y1 in result.lead_y_ranges),
            "header_icon_xyxy": ",".join(map(str, result.header_icon_xyxy)),
            "header_text": result.header_text,
            "footer_fields": ";".join(f"{f.top}/{f.bottom}" for f in result.footer_fields),
            "width": result.panel_image.width,
            "height": result.panel_image.height,
        })

    manifest_path = panels_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"wrote {len(manifest_rows)} panels, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
