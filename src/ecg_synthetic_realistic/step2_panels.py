"""Step 2 of the synthetic plan (wiki/overview/synthetic_data_flow.md): combine step 1's three
independently-cropped lead strips for one column into a single panel image,
blending the seams so there's no hard boundary between leads, and draw the
panel-level elements real panels carry: a header (device icon + "MAC 400"
+ "U1.02"), a footer (gain/speed/serial fields, each a bold top field over
a smaller bottom field), and each lead's name label -- all on one grid
matching a real panel photo (2026-09-05 reference), phase-locked to each
crop's own baked-in grid so hand-drawn margins tile with it instead of
visibly starting their own pattern at every seam.

Per-lead calibration pulses come baked into each lead crop from step 1 (the
generator's own to-scale dc pulse), but the reference photo only shows one
clearly, on the bottom-left lead of the panel -- the other rows' pulses
(and every row's right-edge column-separator tick, which the reference
doesn't show at all) are patched out here as grid, not blanked to white.

Ground truth saved here: each lead's (y0, y1) placement inside the
assembled panel (the training target for a future panel -> lead cropper).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import text_vocab
from .config import PANEL_ROW_ORDER
from .grid import draw_grid

PAPER_BG = (255, 255, 255)

SEAM_OVERLAP_FRAC = 0.12  # fraction of a lead crop's height that blends into its neighbor
SIDE_MARGIN_PX = 20
HEADER_HEIGHT_PX = 55
FOOTER_HEIGHT_PX = 55

# One device typeface for every printed field, in both weights it actually
# needs -- a real panel's printer uses one font throughout, not a UI mix.
FONT_CANDIDATES_REGULAR = ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf", "DejaVuSans.ttf"]
FONT_CANDIDATES_BOLD = ["consolab.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSans-Bold.ttf"]

LABEL_FONT_SIZE = 22
HEADER_FONT_SIZE = 16
FOOTER_TOP_FONT_SIZE = 14
FOOTER_BOTTOM_FONT_SIZE = 11

HEADER_ICON_GRID_OFFSET = 3.0
HEADER_ICON_GRID_SIZE = 1.0
LABEL_GRID_OFFSET = 4.0

FOOTER_COLUMN_FRACS = [0.03, 0.28, 0.55, 0.80]

# The calibration pulse a real lead crop carries occupies roughly its
# first 2.5 grid-boxes of width, empirically (2026-09-05, against this
# project's own rendered crops) -- patched out for every row except the
# panel's last, matching the reference photo.
CALIBRATION_PULSE_GRID_WIDTH = 2.5


@dataclass
class PanelGroundTruth:
    panel_image: Image.Image
    lead_names: list[str]
    lead_y_ranges: list[tuple[int, int]]  # matches lead_names order
    header_icon_xyxy: tuple[int, int, int, int]
    header_text: str
    footer_fields: list[text_vocab.FooterField]


_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _load_font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    for name in (FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR):
        key = (name, size)
        if key in _font_cache:
            return _font_cache[key]
        try:
            font = ImageFont.truetype(name, size)
        except OSError:
            continue
        _font_cache[key] = font
        return font
    return ImageFont.load_default()


def _feather_paste(canvas: Image.Image, crop: Image.Image, x_left: int, y_top: int, overlap_px: int) -> None:
    """Pastes `crop` at (x_left, y_top), alpha-feathering its top
    `overlap_px` rows into whatever is already on the canvas there, so
    consecutive leads don't show a hard seam."""
    if overlap_px <= 0 or y_top <= 0:
        canvas.paste(crop, (x_left, max(0, y_top)))
        return

    crop_arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    canvas_arr = np.asarray(canvas.convert("RGB"), dtype=np.float32)

    h, w, _c = crop_arr.shape
    alpha = np.ones((h, 1, 1), dtype=np.float32)
    alpha[:overlap_px, 0, 0] = np.linspace(0.0, 1.0, overlap_px)

    y0 = y_top
    y1 = y_top + h
    dest_y0 = max(0, y0)
    dest_y1 = min(canvas_arr.shape[0], y1)
    src_y0 = dest_y0 - y0
    src_y1 = src_y0 + (dest_y1 - dest_y0)

    x0 = x_left
    x1 = x_left + w

    region = canvas_arr[dest_y0:dest_y1, x0:x1]
    src = crop_arr[src_y0:src_y1]
    a = alpha[src_y0:src_y1]
    blended = a * src + (1 - a) * region
    canvas_arr[dest_y0:dest_y1, x0:x1] = blended

    canvas.paste(Image.fromarray(canvas_arr.astype(np.uint8)), (0, 0))


def assemble_panel(
    lead_crops: dict[str, Image.Image],
    lead_bboxes: dict[str, tuple[int, int, int, int]],
    column_index: int,
    record_id: str,
    grid_px: float,
    rng: random.Random | None = None,
) -> PanelGroundTruth:
    rng = rng or random.Random()
    row_order = PANEL_ROW_ORDER[column_index]
    missing = [name for name in row_order if name not in lead_crops]
    if missing:
        raise ValueError(f"assemble_panel: missing lead crops for {missing}")

    crops = [lead_crops[name].convert("RGB") for name in row_order]
    width = max(c.width for c in crops) + 2 * SIDE_MARGIN_PX

    y_ranges: list[tuple[int, int]] = []
    y_cursor = HEADER_HEIGHT_PX
    for crop in crops:
        overlap = int(crop.height * SEAM_OVERLAP_FRAC) if y_cursor > HEADER_HEIGHT_PX else 0
        y_top = y_cursor - overlap
        y_ranges.append((y_top, y_top + crop.height))
        y_cursor = y_top + crop.height
    total_height = y_cursor + FOOTER_HEIGHT_PX

    # Every row shares the same x0 in the original sheet (same printed
    # column) -- confirmed empirically 2026-09-05 -- so one x-phase covers
    # the whole panel; each row keeps its own y-phase for its own crop.
    shared_x0 = lead_bboxes[row_order[0]][0]
    x_phase = SIDE_MARGIN_PX - shared_x0
    y_phases = [y_top - lead_bboxes[name][1] for name, (y_top, _y_bottom) in zip(row_order, y_ranges)]

    # Plain background first, crops pasted on top (they already carry their
    # own aligned grid) -- hand-drawn grid below fills only the areas crops
    # don't cover, phase-locked so it tiles with each crop's own grid.
    canvas = Image.new("RGB", (width, total_height), PAPER_BG)

    for crop, (y_top, _y_bottom) in zip(crops, y_ranges):
        overlap_px = int(crop.height * SEAM_OVERLAP_FRAC)
        _feather_paste(canvas, crop, SIDE_MARGIN_PX, y_top, overlap_px if y_top > HEADER_HEIGHT_PX else 0)

    draw = ImageDraw.Draw(canvas)
    draw_grid(draw, (0, 0, width, HEADER_HEIGHT_PX), grid_px, phase=(x_phase, y_phases[0]))
    draw_grid(draw, (0, total_height - FOOTER_HEIGHT_PX, width, total_height), grid_px, phase=(x_phase, y_phases[-1]))
    for (y_top, y_bottom), y_phase in zip(y_ranges, y_phases):
        draw_grid(draw, (0, y_top, SIDE_MARGIN_PX, y_bottom), grid_px, phase=(x_phase, y_phase))
        draw_grid(draw, (width - SIDE_MARGIN_PX, y_top, width, y_bottom), grid_px, phase=(x_phase, y_phase))

    # Only the panel's last row keeps its baked-in calibration pulse
    # (matches the reference photo); erase the others back to blank paper,
    # then redraw grid over the erased rectangle, so no ink ghosts through.
    # Every row's right edge also carries the generator's own column-
    # separator tick (a second step-like mark) -- not in the reference
    # photo at all, so it's erased on every row, last one included.
    patch_width = int(grid_px * CALIBRATION_PULSE_GRID_WIDTH)
    for (y_top, y_bottom), y_phase in zip(y_ranges[:-1], y_phases[:-1]):
        patch_box = (SIDE_MARGIN_PX, y_top, SIDE_MARGIN_PX + patch_width, y_bottom)
        draw.rectangle(patch_box, fill=PAPER_BG)
        draw_grid(draw, patch_box, grid_px, phase=(x_phase, y_phase))
    for (y_top, y_bottom), y_phase in zip(y_ranges, y_phases):
        patch_box = (width - SIDE_MARGIN_PX - patch_width, y_top, width - SIDE_MARGIN_PX, y_bottom)
        draw.rectangle(patch_box, fill=PAPER_BG)
        draw_grid(draw, patch_box, grid_px, phase=(x_phase, y_phase))

    icon_size = max(10, round(grid_px * HEADER_ICON_GRID_SIZE))
    icon_x0 = SIDE_MARGIN_PX + round(grid_px * HEADER_ICON_GRID_OFFSET)
    icon_y0 = (HEADER_HEIGHT_PX - icon_size) // 2
    icon_xyxy = (icon_x0, icon_y0, icon_x0 + icon_size, icon_y0 + icon_size)
    draw.rectangle(icon_xyxy, fill=(0, 0, 0))

    header_font = _load_font(HEADER_FONT_SIZE, bold=True)
    draw.text((icon_x0 + icon_size + 8, 12), text_vocab.HEADER_TEXT, fill=(0, 0, 0), font=header_font)

    patient_ref = text_vocab.synthetic_patient_ref(record_id)
    fields = text_vocab.footer_fields(row_order, patient_ref, rng)
    top_font = _load_font(FOOTER_TOP_FONT_SIZE, bold=True)
    bottom_font_plain = _load_font(FOOTER_BOTTOM_FONT_SIZE, bold=False)
    bottom_font_bold = _load_font(FOOTER_BOTTOM_FONT_SIZE, bold=True)
    usable_width = width - 2 * SIDE_MARGIN_PX
    for field, frac in zip(fields, FOOTER_COLUMN_FRACS):
        x = SIDE_MARGIN_PX + round(frac * usable_width)
        draw.text((x, total_height - FOOTER_HEIGHT_PX + 6), field.top, fill=(0, 0, 0), font=top_font)
        if field.bottom:
            bottom_font = bottom_font_bold if field.bottom_bold else bottom_font_plain
            draw.text((x, total_height - FOOTER_HEIGHT_PX + 28), field.bottom, fill=(180, 0, 0), font=bottom_font)

    label_font = _load_font(LABEL_FONT_SIZE, bold=True)
    label_x = SIDE_MARGIN_PX + round(grid_px * LABEL_GRID_OFFSET)
    for name, (y_top, y_bottom) in zip(row_order, y_ranges):
        draw.text((label_x, (y_top + y_bottom) // 2 - 26), name, fill=(0, 0, 0), font=label_font)

    return PanelGroundTruth(
        panel_image=canvas,
        lead_names=row_order,
        lead_y_ranges=y_ranges,
        header_icon_xyxy=icon_xyxy,
        header_text=text_vocab.HEADER_TEXT,
        footer_fields=fields,
    )
