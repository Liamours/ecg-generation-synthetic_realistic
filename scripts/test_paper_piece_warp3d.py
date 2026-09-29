"""Manual verification for apply_perspective_warp_3d: build a paper piece,
warp it at a few tilt-angle combinations, draw the computed per-column
bboxes on top -- if the projection math is right, each drawn box should
tightly and correctly bound its column's visible content, the same check
test_paper_piece_tilt.py already does for the 2D warp.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import ImageDraw

from ecg_synthetic_realistic.paper_piece import apply_perspective_warp_3d, build_paper_piece

GRID_PX_AT_RESOLUTION_200 = 39.37

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
num_columns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

piece = build_paper_piece(rng, grid_px=GRID_PX_AT_RESOLUTION_200, num_columns=num_columns)

flat_preview = piece.image.copy()
draw = ImageDraw.Draw(flat_preview)
for col in piece.columns:
    draw.rectangle(col.bbox, outline=(0, 150, 255), width=4)
flat_preview.save(out_dir / "warp3d_flat.png")
print(f"flat: {piece.image.size}, columns: {[c.bbox for c in piece.columns]}")

trials = [
    ("tilt_x", 22.0, 0.0),
    ("tilt_y", 0.0, 22.0),
    ("both", 16.0, 16.0),
    ("extreme", 30.0, 30.0),
]

for name, tx, ty in trials:
    warped = apply_perspective_warp_3d(piece, tilt_x_deg=tx, tilt_y_deg=ty)
    preview = warped.image.copy()
    draw = ImageDraw.Draw(preview)
    for col in warped.columns:
        draw.rectangle(col.bbox, outline=(255, 0, 150), width=4)
    preview.save(out_dir / f"warp3d_{name}.png")
    print(f"{name} (x={tx}, y={ty}): {warped.image.size}, columns: {[c.bbox for c in warped.columns]}")
