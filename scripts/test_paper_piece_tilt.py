"""Manual verification: build a paper piece, tilt it, draw the computed
per-column bboxes on top -- if the rotation math is right, each drawn box
should tightly and correctly bound its column's visible content.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import ImageDraw

from ecg_synthetic_realistic.paper_piece import apply_perspective_warp, apply_tilt, build_paper_piece

GRID_PX_AT_RESOLUTION_200 = 39.37

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
num_columns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
angle = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(".")

piece = build_paper_piece(rng, grid_px=GRID_PX_AT_RESOLUTION_200, num_columns=num_columns)

flat_preview = piece.image.copy()
draw = ImageDraw.Draw(flat_preview)
for col in piece.columns:
    draw.rectangle(col.bbox, outline=(0, 150, 255), width=4)
flat_preview.save(out_dir / "piece_flat.png")

tilted = apply_tilt(piece, angle_deg=angle)
tilted_preview = tilted.image.copy()
draw = ImageDraw.Draw(tilted_preview)
for col in tilted.columns:
    draw.rectangle(col.bbox, outline=(0, 150, 255), width=4)
tilted_preview.save(out_dir / "piece_tilted.png")

warped = apply_perspective_warp(piece, rng)
warped_preview = warped.image.copy()
draw = ImageDraw.Draw(warped_preview)
for col in warped.columns:
    draw.rectangle(col.bbox, outline=(255, 0, 150), width=4)
warped_preview.save(out_dir / "piece_warped.png")

both = apply_perspective_warp(tilted, rng)
both_preview = both.image.copy()
draw = ImageDraw.Draw(both_preview)
for col in both.columns:
    draw.rectangle(col.bbox, outline=(0, 200, 0), width=4)
both_preview.save(out_dir / "piece_tilted_warped.png")

print(f"flat: {piece.image.size}, columns: {[c.bbox for c in piece.columns]}")
print(f"tilted {angle} deg: {tilted.image.size}, columns: {[c.bbox for c in tilted.columns]}")
print(f"warped: {warped.image.size}, columns: {[c.bbox for c in warped.columns]}")
print(f"tilted+warped: {both.image.size}, columns: {[c.bbox for c in both.columns]}")
