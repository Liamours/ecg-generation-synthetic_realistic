"""Step 3 of the synthetic plan (wiki/overview/synthetic_data_flow.md): panel-level realism the
generator itself doesn't do -- paper color, tilt/rectification-style warp,
and aging. Split out from step 2 deliberately: step 2 owns layout/text,
step 3 owns "does this look like a real photographed piece of paper".

Only the paper-tint piece is implemented so far (2026-09-05, in response to
a direct reference-photo comparison). Tilt/perspective warp and tear/aging
are still open per wiki/TODO.md and not built here yet.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Sampled by eye against the 2026-09-05 reference photo (warm off-white,
# not pure white) -- a multiplicative tint, not a hard-coded fill, so ink
# (black/red) stays dark and only the paper itself shifts color.
DEFAULT_PAPER_TINT = (0.94, 0.90, 0.80)


def apply_paper_tint(image: Image.Image, tint: tuple[float, float, float] = DEFAULT_PAPER_TINT) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    tint_arr = np.array(tint, dtype=np.float32).reshape(1, 1, 3)
    tinted = arr * tint_arr
    return Image.fromarray(np.clip(tinted, 0, 255).astype(np.uint8))
