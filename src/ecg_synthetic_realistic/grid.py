"""The panel grid, as two dot tiers (decided 2026-09-05, from a zoomed
reference photo of 4 real major boxes): the major box boundaries are dots
packed so tight they read as a line, while each box's interior carries a
sparse, clearly-separated 4x4 dot array (the four 1mm marks between
0mm/5mm on each axis -- the boundary itself is the 0mm/5mm mark, not a
fifth interior dot). Not drawn as actual lines: still dots throughout,
just two different densities.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

# Standardized 2026-09-25 from 18 to 20, and fixed rather than the prior
# probabilistic 18/19/20 crop-margin-variance model (PANEL_WIDTH_VARIANTS,
# retired the same day, see template.py) -- measured directly against two
# AI-regenerated reference crops of a real GE MAC 400 panel (one MAC-400/
# "odd" style, one GE/"even" style, same patient/device, see
# datasets/panel_templates/ and inferences/grid_warp_check/register.md
# entries): "odd" style's own printed footer content genuinely spans all
# 20 boxes (its trailing code sits past column 17), not an 18-box core
# with incidental crop margin. "even" style's real content stays within
# the first 18 boxes; the rightmost 2 are always blank, not randomly
# present -- this is unaffected by the width change itself, see
# build_panel_template's style handling.
PANEL_WIDTH_BOXES = 20
PANEL_HEIGHT_BOXES = 15
INTERIOR_DOTS_PER_BOX = 4  # marks at 1/5, 2/5, 3/5, 4/5 of the box, excluding the boundary itself

BOUNDARY_DOT_COLOR = (120, 120, 120)
BOUNDARY_DOT_SPACING_PX = 4.5
BOUNDARY_DOT_RADIUS_PX = 0.65  # shrunk from 0.9 2026-09-21 (user call: dots read too big against a real reference)

INTERIOR_DOT_COLOR = (150, 150, 150)
INTERIOR_DOT_RADIUS_PX = 0.75  # shrunk from 1.0 2026-09-21, same call, same ratio to the boundary dot kept

# Grid color variants, added 2026-09-21 (wiki/TODO.md item 4): real photos
# show both a grey/black grid and a red/pink one (panel_geometry.py's
# ink_mask_robust comments -- the real-photo pipeline already separates ink
# from a red grid by color channel, so this closes a synthetic/real gap
# that already has a real-side workaround). "grey" keeps the original
# BOUNDARY_DOT_COLOR/INTERIOR_DOT_COLOR pair as the default; each variant
# keeps the same ~30-lightness-step relationship between boundary and
# interior the grey pair already had.
GRID_COLOR_VARIANTS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "grey": (BOUNDARY_DOT_COLOR, INTERIOR_DOT_COLOR),
    "black": ((40, 40, 40), (70, 70, 70)),
    "red": ((170, 60, 60), (200, 95, 95)),
}
GRID_COLOR_WEIGHTS = {"grey": 0.5, "black": 0.2, "red": 0.3}  # grey stays the common case; red/black are real but minority


def _dotted_line(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
    horizontal = y0 == y1
    length = (x1 - x0) if horizontal else (y1 - y0)
    n = max(1, int(length / BOUNDARY_DOT_SPACING_PX))
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        r = BOUNDARY_DOT_RADIUS_PX
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def draw_dot_grid(draw: ImageDraw.ImageDraw, xyxy: tuple[int, int, int, int], box_px: float,
                   boundary_color: tuple[int, int, int] = BOUNDARY_DOT_COLOR,
                   interior_color: tuple[int, int, int] = INTERIOR_DOT_COLOR) -> None:
    x0, y0, x1, y1 = xyxy

    # Major boundaries: dense dot-lines at every box_px step.
    x = x0
    while x <= x1 + 0.5:
        _dotted_line(draw, x, y0, x, y1, boundary_color)
        x += box_px
    y = y0
    while y <= y1 + 0.5:
        _dotted_line(draw, x0, y, x1, y, boundary_color)
        y += box_px

    # Interior: sparse 4x4 dots strictly inside each box.
    r = INTERIOR_DOT_RADIUS_PX
    bx = x0
    while bx < x1 - 0.5:
        by = y0
        while by < y1 - 0.5:
            for i in range(1, INTERIOR_DOTS_PER_BOX + 1):
                for j in range(1, INTERIOR_DOTS_PER_BOX + 1):
                    px = bx + box_px * i / (INTERIOR_DOTS_PER_BOX + 1)
                    py = by + box_px * j / (INTERIOR_DOTS_PER_BOX + 1)
                    if px > x1 or py > y1:
                        continue
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=interior_color)
            by += box_px
        bx += box_px


# Faint outer page-edge line, added 2026-09-25 -- a real, separate feature
# from the dot grid itself: two AI-regenerated reference crops both show a
# thin solid line just inside the panel's own top/bottom edge, distinct
# from the grid's dense boundary dots. Runs the panel's FULL width
# including any blank margin (confirmed with the user). Drawn a few px
# inside the existing canvas edge, NOT by adding new canvas padding --
# every y-coordinate in template.py (header/footer/row/calibration-step
# placement, and every bbox recorded off them) assumes canvas y=0 is the
# grid's own top edge, and padding the canvas would silently offset all of
# that without a compensating change everywhere it's used.
OUTER_BORDER_COLOR = (170, 170, 170)
OUTER_BORDER_WIDTH_PX = 1
OUTER_BORDER_INSET_PX = 3  # distance from the canvas edge, inside the existing bounds


def draw_outer_border(draw: ImageDraw.ImageDraw, xyxy: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = xyxy
    draw.line([(x0, y0 + OUTER_BORDER_INSET_PX), (x1, y0 + OUTER_BORDER_INSET_PX)], fill=OUTER_BORDER_COLOR, width=OUTER_BORDER_WIDTH_PX)
    draw.line([(x0, y1 - OUTER_BORDER_INSET_PX), (x1, y1 - OUTER_BORDER_INSET_PX)], fill=OUTER_BORDER_COLOR, width=OUTER_BORDER_WIDTH_PX)


def blank_panel_canvas(box_px: float, bg: tuple[int, int, int], grid_color: str = "grey") -> Image.Image:
    size = (round(PANEL_WIDTH_BOXES * box_px), round(PANEL_HEIGHT_BOXES * box_px))
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    boundary_color, interior_color = GRID_COLOR_VARIANTS[grid_color]
    draw_dot_grid(draw, (0, 0, size[0], size[1]), box_px, boundary_color, interior_color)
    draw_outer_border(draw, (0, 0, size[0], size[1]))
    return img
