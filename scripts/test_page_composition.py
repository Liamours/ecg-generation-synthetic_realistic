"""Manual verification: compose a page and draw the resulting per-column
bboxes + orientation labels on top."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import ImageDraw, ImageFont

from ecg_synthetic_realistic.page import compose_page

GRID_PX_AT_RESOLUTION_200 = 39.37

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

rng = random.Random(seed)
page = compose_page(rng, grid_px=GRID_PX_AT_RESOLUTION_200)

preview = page.image.copy()
draw = ImageDraw.Draw(preview)
try:
    font = ImageFont.truetype("arialbd.ttf", 28)
except OSError:
    font = ImageFont.load_default()

all_column_types = []
for piece_idx, piece in enumerate(page.pieces):
    for col in piece.columns:
        draw.rectangle(col.bbox, outline=(0, 150, 255), width=5)
        draw.text((col.bbox[0] + 4, col.bbox[1] + 4), f"{col.orientation_deg}deg {col.column_type}", fill=(255, 0, 0), font=font)
        all_column_types.append(col.column_type)

preview.save(out_dir / "page_preview.png")
print(f"page size: {page.image.size}")
print(f"pieces: {len(page.pieces)}, total columns: {sum(len(p.columns) for p in page.pieces)}")
print(f"column types present: {sorted(set(all_column_types))} (need all 4: romanic, aV, first_v, second_v)")
for i, piece in enumerate(page.pieces):
    print(f"  piece {i}: pos={piece.position}, size={piece.image.size}, columns={[(c.bbox, c.orientation_deg, c.column_type) for c in piece.columns]}")
