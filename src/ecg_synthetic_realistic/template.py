"""Panel template, decoupled from real lead signal (decided 2026-09-05):
build the static paper look -- dot grid, paper color, header, footer,
row labels, one calibration step -- with the 3 row interiors left empty.
Real per-lead trace rendering plugs into those empty rows later, once the
template itself is right; nothing here depends on ecg-image-kit.

Two mutually exclusive header/footer styles exist on real panels, "odd"
and "even" (2026-09-05, re-examined against a reference photo that turned
out to show two separate panels side by side, not one panel combining
both). They are NOT additive: "even" REPLACES the header text and the
footer's top row with different content, in the exact same header/footer
height as "odd" -- it does not add a third line or grow the panel.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import text_vocab
from .config import GRID_BOX_SECONDS
from .grid import PANEL_HEIGHT_BOXES, PANEL_WIDTH_BOXES, blank_panel_canvas, draw_dot_grid

PAPER_TINT = (240, 234, 216)  # warm off-white, sampled by eye against a reference photo

FONT_CANDIDATES_REGULAR = ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf", "DejaVuSans.ttf"]
FONT_CANDIDATES_BOLD = ["consolab.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSans-Bold.ttf"]

HEADER_HEIGHT_BOXES = 1.5
FOOTER_HEIGHT_BOXES = 1.5
# Fixed for BOTH styles -- "even" must not grow the panel, see module docstring.
ROW_HEIGHT_BOXES = (PANEL_HEIGHT_BOXES - HEADER_HEIGHT_BOXES - FOOTER_HEIGHT_BOXES) / 3

LABEL_FONT_SIZE = 22
HEADER_FONT_SIZE = 24
FOOTER_TOP_FONT_SIZE = 21
FOOTER_BOTTOM_FONT_SIZE = 11

HEADER_ICON_GRID_OFFSET = 3.0
HEADER_ICON_GRID_SIZE = 1.0
LABEL_GRID_OFFSET = 4.0

# Footer column center positions, in grid boxes from the left edge -- not
# spread across the full width. "odd" starts at box 5 with tight 2.5-box
# spacing between its 4 columns (2026-09-05: previously spread out at
# ~1/6/11/15 boxes, corrected to start further right and sit closer
# together). "even" keeps the original wider spacing (not requested to
# change) for its 3 columns.
FOOTER_COLUMN_GRID_BOXES_ODD = [5.0, 8.0, 11.0, 14.0]
FOOTER_COLUMN_GRID_BOXES_EVEN = [6.0, 9.5, 13.0]

# The footer's bottom row (patient ref / serial / INNOQ) is red in most
# real photos, black in at least one confirmed real example -- both real,
# randomized per panel (2026-09-05).
BOTTOM_ROW_COLORS = [(180, 0, 0), (0, 0, 0)]

# The one calibration step this project's reference photo actually shows
# clearly (bottom-left lead only, see step2_panels.py's prior notes) --
# a 1mV pulse, not real signal, at the panel's own printed gain. Shape is
# a plateau with flat lead-in/lead-out segments (_|-|_), not a bare open
# "|-|" -- confirmed against a zoomed reference photo 2026-09-05. Height
# is tuned empirically at 10mm/mV (limb leads); a 5mm/mV panel
# (precordial leads) scales it proportionally, not to a separately-tuned
# number -- 1mV is 1mV, the gain is what changes how tall that reads.
CALIBRATION_STEP_FLAT_BOXES = 0.5
CALIBRATION_STEP_PLATEAU_BOXES = 1.0
CALIBRATION_STEP_HEIGHT_BOXES_AT_10MM = 3.25
CALIBRATION_STEP_LINE_WIDTH_PX = 3
# "free" mode (uncorrelated with gain): height sampled from roughly the
# same range the correlated 5mm/10mm cases already produce (10mm/mV ->
# 3.25 boxes, 5mm/mV -> 1.625 boxes), not an arbitrary range. Relevant
# only for "even" style, which prints no gain field to correlate against.
CALIBRATION_STEP_FREE_HEIGHT_RANGE_BOXES = (1.5, 3.5)

PanelStyle = str  # "odd" | "even"
CalibrationStepMode = str  # "correlated" | "free" | "none"


@dataclass
class PanelTemplate:
    image: Image.Image
    row_y_ranges: list[tuple[int, int]]  # matches row_labels order; where real lead traces plug in later
    header_icon_xyxy: tuple[int, int, int, int]
    header_text: str
    footer_fields: list[text_vocab.FooterField]
    row_labels: list[str]
    style: PanelStyle
    gain: str
    calibration_step_mode: CalibrationStepMode
    has_calibration_step: bool
    bottom_row_color: tuple[int, int, int]
    record_ref: str
    lead_signals: dict[str, np.ndarray] | None  # exact real samples drawn per row_labels entry, see signal_source.py -- None when the panel was built without real signal
    signal_fs: float | None
    label_bboxes: list[tuple[int, int, int, int]]  # matches row_labels order; exact rendered box of each row's own printed lead-name text
    calibration_step_bbox: tuple[int, int, int, int] | None  # None when has_calibration_step is False


def signal_start_x(grid_px: float) -> float:
    """Where a row's real trace should start -- after the reserved
    calibration-zone width, so all 3 rows align regardless of which one
    (if any) actually draws the step there. Exposed so callers (e.g.
    signal_source-driven code) can compute how many samples actually fit
    in the remaining panel width, instead of assuming the full
    config.PANEL_WIDTH_GRID_BOXES worth."""
    flat_w = CALIBRATION_STEP_FLAT_BOXES * grid_px
    plateau_w = CALIBRATION_STEP_PLATEAU_BOXES * grid_px
    total_w = 2 * flat_w + plateau_w
    x_lead_in = round(grid_px * 1.0 - total_w / 2)
    return round(x_lead_in + total_w)


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


def build_panel_template(
    row_labels: list[str],
    grid_px: float,
    record_ref: str = "0000000",
    rng: random.Random | None = None,
    style: PanelStyle = "odd",
    calibration_step_mode: CalibrationStepMode | None = None,
    lead_signals: dict[str, np.ndarray] | None = None,
    signal_fs: float | None = None,
) -> PanelTemplate:
    """`style`: "odd" is the base template (MAC 400 / V1.02 header, 4-field
    footer top row: Man / speed / gain / code). "even" REPLACES the header
    text with "GE {date} {time}" and the footer's top row with a 3-field
    notch-filter / bandpass / heart-rate row -- same header height, same
    2-line footer height, same bottom-row fields (patient ref / serial /
    INNOQ) as "odd". Not additive.

    `calibration_step_mode`: "correlated" ties the step height to the
    panel's own printed gain (only meaningful for "odd", which is the
    only style that prints a gain field) -- default None resolves to
    "correlated" for "odd" and "free" for "even", since "even" has no
    displayed gain to correlate against. "none" omits the step entirely.

    `lead_signals`/`signal_fs`: real per-lead sample arrays (see
    signal_source.py), one per row_labels entry, drawn as the actual
    trace. Omitted (default) leaves the row interiors empty, as they were
    before 2026-09-05 -- template-only testing still works unchanged.
    """
    if style not in ("odd", "even"):
        raise ValueError(f"style must be 'odd' or 'even', got {style!r}")
    if calibration_step_mode is None:
        calibration_step_mode = "correlated" if style == "odd" else "free"
    if calibration_step_mode not in ("correlated", "free", "none"):
        raise ValueError(f"calibration_step_mode must be 'correlated', 'free', or 'none', got {calibration_step_mode!r}")
    if len(row_labels) != 3:
        raise ValueError(f"a panel has exactly 3 lead rows, got {row_labels}")
    rng = rng or random.Random()

    canvas = blank_panel_canvas(grid_px, PAPER_TINT)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)

    header_px = HEADER_HEIGHT_BOXES * grid_px
    footer_px = FOOTER_HEIGHT_BOXES * grid_px
    row_px = ROW_HEIGHT_BOXES * grid_px

    row_y_ranges = []
    y = header_px
    for _ in row_labels:
        row_y_ranges.append((round(y), round(y + row_px)))
        y += row_px

    icon_size = max(10, round(grid_px * HEADER_ICON_GRID_SIZE))
    icon_x0 = round(grid_px * HEADER_ICON_GRID_OFFSET)
    icon_y0 = round((header_px - icon_size) / 2)
    icon_xyxy = (icon_x0, icon_y0, icon_x0 + icon_size, icon_y0 + icon_size)
    draw.rectangle(icon_xyxy, fill=(0, 0, 0))

    header_text = text_vocab.HEADER_TEXT if style == "odd" else text_vocab.ge_datetime_header(rng)
    header_font = _load_font(HEADER_FONT_SIZE, bold=True)
    draw.text((icon_x0 + icon_size - 2, round(header_px * 0.45)), header_text, fill=(0, 0, 0), font=header_font)

    label_font = _load_font(LABEL_FONT_SIZE, bold=True)
    label_x = round(grid_px * LABEL_GRID_OFFSET)
    label_bboxes: list[tuple[int, int, int, int]] = []
    for name, (y_top, y_bottom) in zip(row_labels, row_y_ranges):
        label_pos = (label_x, (y_top + y_bottom) // 2 - 26)
        draw.text(label_pos, name, fill=(0, 0, 0), font=label_font)
        # Exact rendered-glyph box (textbbox accounts for real font metrics,
        # not just the string's advance width) -- one box per row's own
        # printed lead-name text (I, aVR, V4, ...), not one box spanning the
        # whole 3-row group.
        label_bboxes.append(draw.textbbox(label_pos, name, font=label_font))

    # One calibration step, bottom-left lead only (matches the reference
    # photo) -- a fixed rectangular pulse shape, not derived from any
    # signal, since real lead data isn't wired in at this stage. Decided
    # once here (not inside footer_fields) so that, in "correlated" mode,
    # the step height and the printed gain field always agree.
    gain = text_vocab.gain_for_leads(row_labels, rng)

    if calibration_step_mode == "correlated":
        gain_mm_per_mv = float(gain.split("mm/mV")[0])
        step_height_boxes = CALIBRATION_STEP_HEIGHT_BOXES_AT_10MM * (gain_mm_per_mv / 10.0)
    elif calibration_step_mode == "free":
        step_height_boxes = rng.uniform(*CALIBRATION_STEP_FREE_HEIGHT_RANGE_BOXES)
    else:
        step_height_boxes = None

    # Geometry computed unconditionally (not just when a step is actually
    # drawn) -- every row's real trace starts at the same x_lead_out, so
    # all 3 rows stay aligned regardless of which row (if any) shows the
    # calibration step in its reserved zone.
    flat_w = CALIBRATION_STEP_FLAT_BOXES * grid_px
    plateau_w = CALIBRATION_STEP_PLATEAU_BOXES * grid_px
    total_w = 2 * flat_w + plateau_w
    # Centered on the boundary between the 1st and 2nd major box, i.e.
    # the middle of their combined width.
    x_lead_in = round(grid_px * 1.0 - total_w / 2)
    x_rise = round(x_lead_in + flat_w)
    x_fall = round(x_rise + plateau_w)
    x_lead_out = round(x_fall + flat_w)

    calibration_step_bbox: tuple[int, int, int, int] | None = None
    if step_height_boxes is not None:
        last_y_top, last_y_bottom = row_y_ranges[-1]
        step_h = round(step_height_boxes * grid_px)
        step_mid_y = (last_y_top + last_y_bottom) // 2
        step_y0 = step_mid_y - step_h // 2
        draw.line(
            [
                (x_lead_in, step_mid_y),
                (x_rise, step_mid_y),
                (x_rise, step_y0),
                (x_fall, step_y0),
                (x_fall, step_mid_y),
                (x_lead_out, step_mid_y),
            ],
            fill=(0, 0, 0),
            width=CALIBRATION_STEP_LINE_WIDTH_PX,
            joint="curve",
        )
        # Exact extent of the drawn polyline (x_lead_in..x_lead_out, the
        # baseline..plateau y-range) padded by half the stroke width on
        # every side, since the line itself extends that far past its own
        # path. Ties to the panel's own printed gain (5mm/mV vs 10mm/mV,
        # see `gain` below) since "correlated" mode's step height is
        # derived directly from it -- the scale a reader would infer from
        # this box's height is always the one this panel actually prints.
        half_lw = CALIBRATION_STEP_LINE_WIDTH_PX / 2
        calibration_step_bbox = (
            round(x_lead_in - half_lw), round(min(step_y0, step_mid_y) - half_lw),
            round(x_lead_out + half_lw), round(max(step_y0, step_mid_y) + half_lw),
        )

    if lead_signals is not None:
        if signal_fs is None:
            raise ValueError("signal_fs is required when lead_signals is given")
        gain_mm_per_mv = float(gain.split("mm/mV")[0])
        px_per_second = grid_px / GRID_BOX_SECONDS
        px_per_sample = px_per_second / signal_fs
        px_per_mv = (grid_px / 5.0) * gain_mm_per_mv  # grid_px is px per 5mm major box

        for name, (y_top, y_bottom) in zip(row_labels, row_y_ranges):
            signal = lead_signals[name]
            baseline_y = (y_top + y_bottom) / 2
            points = [
                (x_lead_out + i * px_per_sample, baseline_y - sample * px_per_mv)
                for i, sample in enumerate(signal)
            ]
            if len(points) >= 2:
                draw.line(points, fill=(0, 0, 0), width=2, joint="curve")

    patient_ref = text_vocab.synthetic_patient_ref(record_ref)
    if style == "odd":
        fields = text_vocab.footer_fields(row_labels, patient_ref, rng, gain=gain)
    else:
        fields = text_vocab.even_footer_fields(patient_ref, rng)

    top_font = _load_font(FOOTER_TOP_FONT_SIZE, bold=True)
    bottom_font_plain = _load_font(FOOTER_BOTTOM_FONT_SIZE, bold=False)
    bottom_font_bold = _load_font(FOOTER_BOTTOM_FONT_SIZE, bold=True)
    footer_y0 = height - footer_px
    column_boxes = FOOTER_COLUMN_GRID_BOXES_ODD if style == "odd" else FOOTER_COLUMN_GRID_BOXES_EVEN
    # Real photos show this row in red ink most of the time, but at least
    # one confirmed example prints it in black -- one color per panel, not
    # per field, since a real printer doesn't switch ink mid-line.
    bottom_row_color = rng.choice(BOTTOM_ROW_COLORS)
    for field, box_offset in zip(fields, column_boxes):
        center_x = box_offset * grid_px
        top_w = draw.textlength(field.top, font=top_font)
        draw.text((round(center_x - top_w / 2), round(footer_y0 + footer_px * 0.1)), field.top, fill=(0, 0, 0), font=top_font)
        if field.bottom:
            bottom_font = bottom_font_bold if field.bottom_bold else bottom_font_plain
            bottom_w = draw.textlength(field.bottom, font=bottom_font)
            draw.text((round(center_x - bottom_w / 2), round(footer_y0 + footer_px * 0.55)), field.bottom, fill=bottom_row_color, font=bottom_font)

    return PanelTemplate(
        image=canvas,
        row_y_ranges=row_y_ranges,
        header_icon_xyxy=icon_xyxy,
        header_text=header_text,
        footer_fields=fields,
        row_labels=row_labels,
        style=style,
        gain=gain,
        calibration_step_mode=calibration_step_mode,
        has_calibration_step=step_height_boxes is not None,
        bottom_row_color=bottom_row_color,
        record_ref=record_ref,
        lead_signals=lead_signals,
        signal_fs=signal_fs,
        label_bboxes=label_bboxes,
        calibration_step_bbox=calibration_step_bbox,
    )


def build_connected_panel_template(
    row_labels: list[str],
    grid_px: float,
    record_ref: str = "0000000",
    rng: random.Random | None = None,
    even_calibration_step_mode: CalibrationStepMode | None = None,
) -> tuple[Image.Image, PanelTemplate, PanelTemplate]:
    """An "odd" panel directly followed by an "even" panel, same row
    labels on both -- the same continued reading shown twice, once per
    style (matches the 2026-09-05 reference photo: I/II/III duplicated
    across two joined panels, one MAC-400-style, one GE-style). Each half
    keeps its own icon, header, footer, and row labels; they're placed
    edge to edge with no gap, not merged into one shared header/footer."""
    left = build_panel_template(row_labels, grid_px, record_ref=record_ref, rng=rng, style="odd")
    right = build_panel_template(
        row_labels, grid_px, record_ref=record_ref, rng=rng, style="even",
        calibration_step_mode=even_calibration_step_mode,
    )
    width = left.image.width + right.image.width
    height = max(left.image.height, right.image.height)
    canvas = Image.new("RGB", (width, height), PAPER_TINT)
    canvas.paste(left.image, (0, 0))
    canvas.paste(right.image, (left.image.width, 0))
    return canvas, left, right
