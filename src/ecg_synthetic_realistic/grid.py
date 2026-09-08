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

PANEL_WIDTH_BOXES = 18
PANEL_HEIGHT_BOXES = 15
INTERIOR_DOTS_PER_BOX = 4  # marks at 1/5, 2/5, 3/5, 4/5 of the box, excluding the boundary itself

BOUNDARY_DOT_COLOR = (120, 120, 120)
BOUNDARY_DOT_SPACING_PX = 4.5
BOUNDARY_DOT_RADIUS_PX = 0.9

INTERIOR_DOT_COLOR = (150, 150, 150)
INTERIOR_DOT_RADIUS_PX = 1.0


def _dotted_line(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float) -> None:
    horizontal = y0 == y1
    length = (x1 - x0) if horizontal else (y1 - y0)
    n = max(1, int(length / BOUNDARY_DOT_SPACING_PX))
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        r = BOUNDARY_DOT_RADIUS_PX
        draw.ellipse([x - r, y - r, x + r, y + r], fill=BOUNDARY_DOT_COLOR)


def draw_dot_grid(draw: ImageDraw.ImageDraw, xyxy: tuple[int, int, int, int], box_px: float) -> None:
    x0, y0, x1, y1 = xyxy

    # Major boundaries: dense dot-lines at every box_px step.
    x = x0
    while x <= x1 + 0.5:
        _dotted_line(draw, x, y0, x, y1)
        x += box_px
    y = y0
    while y <= y1 + 0.5:
        _dotted_line(draw, x0, y, x1, y)
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
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=INTERIOR_DOT_COLOR)
            by += box_px
        bx += box_px


def blank_panel_canvas(box_px: float, bg: tuple[int, int, int]) -> Image.Image:
    size = (round(PANEL_WIDTH_BOXES * box_px), round(PANEL_HEIGHT_BOXES * box_px))
    img = Image.new("RGB", size, bg)
    draw_dot_grid(ImageDraw.Draw(img), (0, 0, size[0], size[1]), box_px)
    return img
