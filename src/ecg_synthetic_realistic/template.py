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

import math
import random
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.interpolate import RBFInterpolator

from . import text_vocab
from .config import GRID_BOX_SECONDS
from .grid import (GRID_COLOR_VARIANTS, GRID_COLOR_WEIGHTS, OUTER_BORDER_INSET_PX, PANEL_HEIGHT_BOXES, PANEL_MARGIN_TOP_BOXES, PANEL_WIDTH_BOXES,
                   blank_panel_canvas, draw_dot_grid)

PAPER_TINT = (240, 234, 216)  # warm off-white, sampled by eye against a reference photo

FONT_CANDIDATES_REGULAR = ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf", "DejaVuSans.ttf"]
FONT_CANDIDATES_BOLD = ["consolab.ttf", "courbd.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSans-Bold.ttf"]

HEADER_HEIGHT_BOXES = 1.5
FOOTER_HEIGHT_BOXES = 1.5
# Fixed for BOTH styles -- "even" must not grow the panel, see module docstring.
ROW_HEIGHT_BOXES = (PANEL_HEIGHT_BOXES - HEADER_HEIGHT_BOXES - FOOTER_HEIGHT_BOXES) / 3

# Printed text on the lead panels and the report panel's header, rebuilt 2026-09-29 from measurements on the
# two reference crops (datasets/panel_templates/, wiki/TODO.md "Ideal panel
# template prototype: text rendering"). The old fixed pixel sizes (22/24/21/11
# px) rendered text about half the real height at this project's grid_px
# (39.37 px per 5 mm). Text is grouped by measured glyph cap height, one font
# and one size per group, natural spacing, no stretching. Body font is Arial
# Bold as a placeholder: no installed font fit the body glyphs well, Arial
# Bold lands within about 0.5 mm of the real widths.
FONT_CANDIDATES_SANS = ["arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"]
FONT_CANDIDATES_SANS_BOLD = ["arialbd.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
TEXT_CAP_MM = {"body": 2.45, "footer_serial": 1.52, "footer_device": 1.85, "footer_org": 1.85}
# Left ink edge of each printed field in grid boxes from the panel's left edge,
# and its cap-height center in boxes from the grid's top line (measured on the
# references); add PANEL_MARGIN_TOP_BOXES for canvas coordinates. The footer's
# second row prints below the grid's bottom line, in the bottom margin.
HEADER_LEFT_BOXES = {"odd": [3.076, 9.848], "even": [3.446, 9.297, 13.676]}
HEADER_CAP_CENTER_BOXES = 0.73
LABEL_LEFT_BOXES = 4.19
LABEL_CAP_CENTER_BOXES = [2.52, 6.02, 9.99]
FOOTER_TOP_LEFT_BOXES = {"odd": [4.636, 9.015, 13.0, 17.485], "even": [4.865, 9.446, 14.757]}
FOOTER_BOTTOM_LEFT_BOXES = {"odd": [2.606, 6.788, 12.894], "even": [2.257, 6.041, 11.797]}
FOOTER_TOP_CAP_CENTER_BOXES = 14.52
FOOTER_BOTTOM_CAP_CENTER_BOXES = 15.2
FOOTER_BOTTOM_GROUPS = ["footer_serial", "footer_device", "footer_org"]
HEADER_ICON_GRID_HEIGHT = 1.16  # taller than wide, top edge on the outer border line (prototype, 2026-09-26)

# All four constants below corrected 2026-09-25 to EXACT measured values
# (not rounded/compromise numbers) from the two AI-regenerated reference
# crops -- pixel-thresholded (icon) and OCR bbox center (labels/footer),
# origin/pitch established directly per template from the same reference
# (see inferences/grid_warp_check/text_positions.json for the full
# measurement, both templates cross-checked against each other). The
# prior values (icon offset 3.0, label offset 4.0, footer columns
# [5,8,11,14]/[6.0,9.5,13.0]) were 2026-09-05-era estimates never
# re-verified against precise measurement -- icon offset in particular was
# off by a full box (measured 2.07 even / 2.03 odd, independently
# consistent, not 3.0).
HEADER_ICON_GRID_OFFSET = 2.05  # mean of even (2.07) and odd (2.03), independently measured
HEADER_ICON_GRID_SIZE = 1.0
LABEL_GRID_OFFSET = 4.09  # mean of odd's I/II/III label column positions (3.94, 4.08, 4.26)

# "even"'s real content still occupies only the first 18 of the panel's 20
# boxes, unaffected by the 2026-09-25 width change -- see grid.py's
# PANEL_WIDTH_BOXES note. "odd"'s own printed footer genuinely spans out
# toward the panel's new right edge (trailing code sits past box 17), not an
# 18-box core with incidental margin like "even". Footer field positions:
# FOOTER_TOP_LEFT_BOXES / FOOTER_BOTTOM_LEFT_BOXES above.

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
TRACE_LINE_WIDTH_PX = 2  # stroke width of every real lead trace; named 2026-09-18 so it can be recorded per lead rather than re-guessed by a consumer

# Calibration-step SHAPE variants, added 2026-09-21 (wiki/TODO.md item 3):
# real photos show this pulse drawn thin-and-narrow-legged to
# thick-and-wide-legged depending on the device/printer. Each entry is
# (line_width_px, flat_frac): flat_frac is the share of the step's FIXED
# total reserved width (CALIBRATION_STEP_FLAT_BOXES*2 + PLATEAU_BOXES,
# unchanged -- this is what signal_start_x() also assumes, so every row's
# trace still starts at the same x regardless of which variant this panel
# draws) given to the two flat lead-in/lead-out legs combined, rest to the
# plateau. Height is NOT part of this variation -- it stays derived from
# the panel's own gain exactly as before, only the pulse's proportions and
# stroke thickness change. "medium" reproduces the original single fixed
# shape exactly (line width 3, the original 0.5 flat/plateau split).
CALIBRATION_STEP_SHAPE_VARIANTS: dict[str, tuple[int, float]] = {
    "thin_no_leg": (2, 0.10),
    "thin_short_leg": (2, 0.30),
    "medium": (3, 0.50),
    "thick_long_leg": (4, 0.65),
    "thick_expanded_leg": (5, 0.80),
}

# Panel width variants (probabilistic 18/19/20 crop-margin simulation, and
# the _apply_width_margin paste function that implemented it) -- RETIRED
# 2026-09-25, user call ("we always do the same fixed size perfect grid"),
# once the true per-style design width was established directly (see
# grid.py's PANEL_WIDTH_BOXES note) rather than modeled as random crop
# variance.
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
    trace_mask: Image.Image | None  # "L" mode, same size as image, 255 where a real lead trace was drawn, 0 elsewhere -- excludes grid/header/labels/calibration step. None when lead_signals is None (nothing was drawn to mask)
    # Exact geometry of what was drawn, added 2026-09-18 so a training target
    # spanning the WHOLE row can be reconstructed, not just the trace segment.
    # The real trace starts at trace_x0, and the zone before it holds the
    # calibration step on one row and bare paper on the others. The per-lead
    # .npy holds only the trace's own samples, which is why every training
    # example carried an unlabelled dead zone at the row start (TODO gap #3).
    trace_x0: int | None            # panel-local x where the real trace begins; None when no signal was drawn
    px_per_sample: float | None
    px_per_mv: float | None
    baseline_ys: list[float]        # per row_labels entry, panel-local y of each row's resting baseline
    step_row_index: int | None      # which row carries the calibration step; None when there is no step
    step_geometry: tuple[int, int, int, int, int, int] | None  # (x_lead_in, x_rise, x_fall, x_lead_out, plateau_y, baseline_y)
    # Every printed text field's exact rendered box, added 2026-09-18:
    # (role, text, (x0, y0, x1, y1)) in panel-local coordinates, where role is
    # "header", "lead_label_<i>", or "footer_top_<i>" / "footer_bottom_<i>".
    # Only the lead labels had boxes recorded before, so a field DETECTOR could
    # not be trained at all -- the strings existed but not where they sit.
    # Carried as one JSON column rather than a column per field, since a panel
    # has a variable number of them.
    text_fields: list[tuple[str, str, tuple[int, int, int, int]]]
    grid_color: str  # "grey" | "black" | "red", see grid.py's GRID_COLOR_VARIANTS -- added 2026-09-21
    calibration_step_shape: str | None  # key into CALIBRATION_STEP_SHAPE_VARIANTS; None when has_calibration_step is False -- added 2026-09-21
    panel_width_boxes: int  # always PANEL_WIDTH_BOXES (20) since 2026-09-25 -- field kept for schema stability, the random 18/19/20 variant this recorded is retired, see grid.py's PANEL_WIDTH_BOXES note
    panel_noise_applied: bool  # see add_panel_noise, added 2026-09-21 (wiki/TODO.md)
    panel_wrinkle_applied: bool  # see add_panel_wrinkle, added 2026-09-21 (wiki/TODO.md)
    paper_bend_applied: bool  # see add_panel_bend, added 2026-09-22 (wiki/TODO.md)
    paper_bend_control_points: list[list[float]] | None  # [[ideal_x, ideal_y, bent_x, bent_y], ...]; None when paper_bend_applied is False


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


_cap_font_cache: dict[tuple[bool, int], tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]] = {}


def _font_for_cap(cap_mm: float, grid_px: float, bold: bool) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    """Sans font whose capital-letter ink height is closest to `cap_mm`
    (grid_px is px per 5 mm), with that font's bbox of "H" so callers can
    center the cap block vertically."""
    cap_px = cap_mm / 5.0 * grid_px
    key = (bold, round(cap_px * 4))
    if key in _cap_font_cache:
        return _cap_font_cache[key]
    for name in (FONT_CANDIDATES_SANS_BOLD if bold else FONT_CANDIDATES_SANS):
        try:
            best = None
            for size in range(6, 200):
                font = ImageFont.truetype(name, size)
                hb = font.getbbox("H")
                gap = abs((hb[3] - hb[1]) - cap_px)
                if best is None or gap < best[0]:
                    best = (gap, font, hb)
                if hb[3] - hb[1] > cap_px:
                    break
            _cap_font_cache[key] = (best[1], best[2])
            return _cap_font_cache[key]
        except OSError:
            continue
    fallback = ImageFont.load_default()
    return fallback, fallback.getbbox("H")


def _draw_cap_text(draw: ImageDraw.ImageDraw, left_x: float, cap_center_y: float, text: str, font: ImageFont.FreeTypeFont,
                   hb: tuple[int, int, int, int], fill: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Draw `text` with its first glyph's ink starting at `left_x` and its
    capital-letter block centered on `cap_center_y`; returns the ink bbox."""
    x = left_x - font.getbbox(text[0])[0]
    y = cap_center_y - (hb[3] - hb[1]) / 2 - hb[1]
    draw.text((x, y), text, fill=fill, font=font)
    return tuple(draw.textbbox((x, y), text, font=font))


def _draw_header(draw: ImageDraw.ImageDraw, header_text: str, style: PanelStyle, grid_px: float) -> tuple[int, int, int, int]:
    """Header text as its separate printed fields (`MAC 400` / `V1.02`, or
    `GE` / date / time) at their measured columns; returns the union bbox,
    which is what the single "header" text field records."""
    pieces = re.split(r"(?<=^GE) |\s{2,}", header_text)
    font, hb = _font_for_cap(TEXT_CAP_MM["body"], grid_px, bold=True)
    boxes = [
        _draw_cap_text(draw, left * grid_px, (PANEL_MARGIN_TOP_BOXES + HEADER_CAP_CENTER_BOXES) * grid_px, piece, font, hb, (0, 0, 0))
        for piece, left in zip(pieces, HEADER_LEFT_BOXES[style])
    ]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


# _shift_bbox and _apply_width_margin (random 18/19/20 crop-margin paste)
# deleted 2026-09-25, retired the same day PANEL_WIDTH_VARIANTS was -- see
# grid.py's PANEL_WIDTH_BOXES note. Not kept dormant: plain utility code,
# not a trained model or a documented negative research result.


# Per-panel noise and wrinkle texture, added 2026-09-21 (wiki/TODO.md, user
# request with 6 reference photos). Panel-level, not page-level: every
# existing noise source (brush mark, sticker, scribble, shadows,
# page_augment.py's whole set) operates on the composed PAGE after panels
# are placed, so two panels on one page currently always show identical
# "paper condition" apart from those page-wide effects. These two instead
# vary independently per panel, as if each physical strip has its own
# handling history -- applied here, at the very end of
# build_panel_template, to the finished (and now possibly width-margined)
# canvas, so any added margin gets the same paper surface treatment as the
# original content rather than looking like a pasted-on patch.
PANEL_NOISE_PROBABILITY = 0.6  # most panels show at least faint grain on a real photo; a minority (fresh print, good lighting) don't
PANEL_NOISE_STD_RANGE = (2.0, 7.0)  # per-channel Gaussian std, 0-255 scale -- faint grain, not visible speckle; kept subtle since this stacks with JPEG's own compression noise on a real photo
PANEL_WRINKLE_PROBABILITY = 0.25  # a real crease is a real physical event, not something every strip has -- rarer than noise
PANEL_WRINKLE_DARKEN_RANGE = (0.12, 0.28)  # multiplicative darkening on the crease's shadow side
PANEL_WRINKLE_HIGHLIGHT_RANGE = (0.05, 0.15)  # multiplicative brightening on its highlight side -- a real fold catches light on one edge, not just shadows the other
PANEL_WRINKLE_WIDTH_FRAC = 0.03  # of the panel's own diagonal, how wide the shadow/highlight band is before blurring

# Added 2026-09-22 (wiki/TODO.md, rectification-model entry): every other
# transform this generator applies -- rotation, tilt, the 2D/3D perspective
# warps, CamScanner -- composes into a single homography, fully recoverable
# from 4 corner points. Real paper also bends/curls, which a homography
# can't represent. This is the only source of that kind of distortion in
# the corpus, so a dense-field rectification model has real non-planar
# ground truth to learn from, and so grid_rectify.refine_canonical_panel's
# classical TPS correction -- which never fires on synthetic panels today,
# since the grid is always a perfect lattice -- can finally be scored
# against known truth instead of only ever running on real photos.
PAPER_BEND_PROBABILITY = 0.4  # a real photographed sheet is rarely perfectly flat; not applied to every panel since a fresh page pressed flat under glass or scanned is still common
PAPER_BEND_GRID_SHAPE = (3, 3)  # interior control points (rows, cols); the grid's own corner nodes are included too (rows+2 x cols+2 total) but are NOT zero-displacement in general -- see add_panel_bend's own note
# Fixed 2026-09-22, user call ("very flawed result... make it more
# straight"): v1 drew each interior control point's offset fully
# independently, which TPS still interpolates smoothly between but reads
# as several small conflicting wiggles, not one coherent bend -- a real
# curled page bends as one surface, not a grid of independent bumps. Now
# every control point's offset comes from ONE shared smooth profile (a
# single random bend axis, parabolic falloff -- zero at the two edges
# along that axis, peak at the center, like a page curling along one
# fold), so all points move together. Amplitude also lowered -- v1's
# range read as a visibly broken grid, not a gentle real-world bend.
PAPER_BEND_AMPLITUDE_FRAC_RANGE = (0.05, 0.18)  # of grid_px (one box pitch), peak displacement at the bend's center
PAPER_BEND_SMOOTHING = 1.0  # thin-plate-spline smoothing -- matches grid_rectify.tps_warp's own default so this distortion sits in the same family that corrector assumes


def _tps_warp(image: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Mirrors grid_rectify.py's own tps_warp exactly (duplicated, not
    imported -- this repo and the digitization repo are separate installed
    packages with their own venvs). Fits the inverse mapping (dst -> src)
    and backward-samples every output pixel, so src=ideal/dst=bent produces
    an image where content that was at an ideal position now appears at the
    corresponding bent position -- the exact reverse of what the classical
    corrector does when it removes this kind of distortion, which is the
    point: the two are meant to be inverses of each other."""
    h, w = image.shape[:2]
    inverse = RBFInterpolator(dst, src, kernel="thin_plate_spline", smoothing=PAPER_BEND_SMOOTHING)
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    sampled = inverse(query)
    map_x = sampled[:, 0].reshape(h, w).astype(np.float32)
    map_y = sampled[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_panel_bend(image: Image.Image, trace_mask: Image.Image | None, grid_px: float, rng: random.Random) -> tuple[Image.Image, Image.Image | None, list[list[float]]]:
    """Genuine non-homography geometric distortion, not per-pixel noise: a
    coarse control grid warped via thin-plate-spline (the same family
    grid_rectify.py's own TPS fit assumes). Every control point's offset
    comes from ONE shared smooth profile -- a random bend axis and a
    parabolic falloff along it, zero at the two edges of the fold axis,
    peak at the centre -- so the whole panel bends as one coherent surface
    (like a page curling along a single fold), not several independently-
    wiggling points. The falloff is zero along the fold axis itself, but
    the 4 image corners are NOT generally on that axis (only when the
    panel is square or the fold happens to land axis-aligned), so a
    corner's own `t` can land inside (-1, 1) and pick up real displacement
    -- confirmed 2026-09-24, up to ~15px on this project's wide panels at
    fold angles near 45 degrees. Callers that need the panel's true final
    corners (e.g. paper_piece.py's panel_points, feeding the rectification
    homography) must read them off `control_points` below, not assume the
    nominal (0,0)-(W,H) rectangle. Returns the bent image,
    the identically-bent trace mask (so mask-to-image correspondence
    survives -- warping only the visible image and leaving the mask
    untouched would silently break the IoU=1.0 reconstruction the
    manifest's geometry columns otherwise guarantee), and the exact
    (ideal, bent) control point correspondences as the ground truth a
    training pipeline reads back."""
    w, h = image.size
    rows, cols = PAPER_BEND_GRID_SHAPE
    amp = grid_px * rng.uniform(*PAPER_BEND_AMPLITUDE_FRAC_RANGE)
    theta = rng.uniform(0, math.pi)  # the bend's fold axis, in-plane direction
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    ideal_pts, bent_pts = [], []
    for gy in np.linspace(0, h, rows + 2):
        for gx in np.linspace(0, w, cols + 2):
            ideal_pts.append([gx, gy])
            # t in [-1, 1] along the fold axis, centred on the panel;
            # (1 - t^2) is exactly 0 at the two edges (t=+-1) and 1 at the
            # centre (t=0) -- one smooth dome, not per-point noise.
            t = ((gx - w / 2) * cos_t + (gy - h / 2) * sin_t) / (max(w, h) / 2)
            disp = amp * max(0.0, 1 - t**2)
            # displaced perpendicular to the fold axis, matching how a
            # page curling along a fold actually moves
            bent_pts.append([gx - sin_t * disp, gy + cos_t * disp])

    ideal_arr = np.array(ideal_pts, dtype=np.float64)
    bent_arr = np.array(bent_pts, dtype=np.float64)

    arr = np.asarray(image.convert("RGB"))
    bent_image = Image.fromarray(_tps_warp(arr, src=ideal_arr, dst=bent_arr))

    bent_mask = None
    if trace_mask is not None:
        mask_arr = np.asarray(trace_mask.convert("L"))
        bent_mask = Image.fromarray(_tps_warp(mask_arr, src=ideal_arr, dst=bent_arr))

    control_points = [[float(i[0]), float(i[1]), float(b[0]), float(b[1])] for i, b in zip(ideal_pts, bent_pts)]
    return bent_image, bent_mask, control_points


def add_panel_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    """Faint per-panel photographic grain -- ECG-Image-Kit and this
    project's own reference photos both show this at the panel/strip
    level, not uniformly across a whole multi-strip page (see
    wiki/research/ecg_datasets_frameworks.md, wiki/research/xai_methods.md).
    Independent per panel (own rng draw), so two panels on the same page
    read as two separately-handled pieces of paper, not one shared texture."""
    std = rng.uniform(*PANEL_NOISE_STD_RANGE)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, std, size=arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def add_panel_wrinkle(image: Image.Image, rng: random.Random) -> Image.Image:
    """One soft crease line across the panel: a shadow band on one side, a
    thin highlight on the other, blurred so it reads as a fold in the
    paper rather than a drawn line -- the "blurred-line paper-fold
    approximation" ECG-Image-Kit itself uses (wiki/research/xai_methods.md
    flagged this project's own generator had no crease modeling at all).
    Distance-from-line shading, not a drawn stroke, so it survives the
    panel's own tilt/warp/scale transforms exactly like the printed
    content does -- it's baked into the pixels before any of those run."""
    w, h = image.size
    diag = (w**2 + h**2) ** 0.5
    band_px = max(3.0, diag * PANEL_WRINKLE_WIDTH_FRAC)

    # A random line across the panel: two points on opposite edges, biased
    # toward roughly-diagonal (a crease from folding/handling rarely runs
    # perfectly axis-aligned with the printed grid).
    if rng.random() < 0.5:
        p0, p1 = (rng.uniform(0, w), 0), (rng.uniform(0, w), h)
    else:
        p0, p1 = (0, rng.uniform(0, h)), (w, rng.uniform(0, h))
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    line_len = max(1e-6, (dx**2 + dy**2) ** 0.5)

    yy, xx = np.mgrid[0:h, 0:w]
    # signed perpendicular distance from every pixel to the crease line
    signed_dist = ((xx - x0) * dy - (yy - y0) * dx) / line_len

    darken = rng.uniform(*PANEL_WRINKLE_DARKEN_RANGE)
    highlight = rng.uniform(*PANEL_WRINKLE_HIGHLIGHT_RANGE)
    # shadow on the negative side, highlight on the positive side, both
    # falling off to 1.0 (no change) within one band width
    factor = np.ones_like(signed_dist)
    factor = np.where(signed_dist < 0, 1.0 - darken * np.clip(1 - (-signed_dist) / band_px, 0, 1), factor)
    factor = np.where(signed_dist >= 0, 1.0 + highlight * np.clip(1 - signed_dist / band_px, 0, 1), factor)
    factor_img = Image.fromarray((np.clip(factor, 0, 2) * 127.5).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=band_px * 0.8))
    factor = np.asarray(factor_img, dtype=np.float32) / 127.5

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) * factor[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build_panel_template(
    row_labels: list[str],
    grid_px: float,
    record_ref: str = "0000000",
    rng: random.Random | None = None,
    style: PanelStyle = "odd",
    calibration_step_mode: CalibrationStepMode | None = None,
    lead_signals: dict[str, np.ndarray] | None = None,
    signal_fs: float | None = None,
    gain: str | None = None,
    grid_color: str | None = None,
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

    `gain`: overrides the internally-sampled value (default None samples
    it from `text_vocab.gain_for_leads`). Added 2026-09-21 (wiki/TODO.md
    item 2) so a caller pairing this panel with an odd-style neighbor
    covering the same leads can force both to report the same gain --
    the even panel prints no gain field of its own, but its true value
    should still match the odd panel showing the same continued reading."""
    if style not in ("odd", "even"):
        raise ValueError(f"style must be 'odd' or 'even', got {style!r}")
    if calibration_step_mode is None:
        calibration_step_mode = "correlated" if style == "odd" else "free"
    if calibration_step_mode not in ("correlated", "free", "none"):
        raise ValueError(f"calibration_step_mode must be 'correlated', 'free', or 'none', got {calibration_step_mode!r}")
    if len(row_labels) != 3:
        raise ValueError(f"a panel has exactly 3 lead rows, got {row_labels}")
    rng = rng or random.Random()

    # Grid color variant, added 2026-09-21 (wiki/TODO.md item 4): drawn per
    # panel, not per page -- real photos show both grid colors even within
    # one casebook/session (different clinic visits, different printer
    # stock), and this project's own casebook-clipping notes already record
    # panels as independently sourced, not one uniform printout. Override
    # available (2026-09-23) for the connected-panel case, where several
    # panels ARE one single physically continuous printout and must share
    # one grid style, not sample independently.
    grid_color = grid_color or rng.choices(list(GRID_COLOR_WEIGHTS), weights=list(GRID_COLOR_WEIGHTS.values()))[0]
    # Fixed canvas width now (PANEL_WIDTH_BOXES=20 for both styles, see
    # grid.py) -- the random per-panel width variant this rng draw used to
    # feed was retired 2026-09-25.
    canvas = blank_panel_canvas(grid_px, PAPER_TINT, grid_color=grid_color)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)

    header_px = (PANEL_MARGIN_TOP_BOXES + HEADER_HEIGHT_BOXES) * grid_px
    row_px = ROW_HEIGHT_BOXES * grid_px

    row_y_ranges = []
    y = header_px
    for _ in row_labels:
        row_y_ranges.append((round(y), round(y + row_px)))
        y += row_px

    icon_size = max(10, round(grid_px * HEADER_ICON_GRID_SIZE))
    icon_x0 = round(grid_px * HEADER_ICON_GRID_OFFSET)
    icon_xyxy = (icon_x0, OUTER_BORDER_INSET_PX, icon_x0 + icon_size, OUTER_BORDER_INSET_PX + round(grid_px * HEADER_ICON_GRID_HEIGHT))
    draw.rectangle(icon_xyxy, fill=(0, 0, 0))

    header_text = text_vocab.HEADER_TEXT if style == "odd" else text_vocab.ge_datetime_header(rng)
    text_fields: list[tuple[str, str, tuple[int, int, int, int]]] = [
        ("header", header_text, _draw_header(draw, header_text, style, grid_px)),
    ]

    label_font, label_hb = _font_for_cap(TEXT_CAP_MM["body"], grid_px, bold=True)
    label_bboxes: list[tuple[int, int, int, int]] = []
    for name, cap_center_boxes in zip(row_labels, LABEL_CAP_CENTER_BOXES):
        # Exact rendered-glyph box (textbbox accounts for real font metrics,
        # not just the string's advance width) -- one box per row's own
        # printed lead-name text (I, aVR, V4, ...), not one box spanning the
        # whole 3-row group.
        label_bboxes.append(_draw_cap_text(draw, LABEL_LEFT_BOXES * grid_px, (PANEL_MARGIN_TOP_BOXES + cap_center_boxes) * grid_px, name, label_font, label_hb, (0, 0, 0)))
        text_fields.append((f"lead_label_{len(label_bboxes) - 1}", name, tuple(label_bboxes[-1])))

    # One calibration step, bottom-left lead only (matches the reference
    # photo) -- a fixed rectangular pulse shape, not derived from any
    # signal, since real lead data isn't wired in at this stage. Decided
    # once here (not inside footer_fields) so that, in "correlated" mode,
    # the step height and the printed gain field always agree.
    gain = gain if gain is not None else text_vocab.gain_for_leads(row_labels, rng)

    step_geometry: tuple[int, int, int, int, int, int] | None = None
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
    calibration_step_shape: str | None = None
    if step_height_boxes is not None:
        # Shape variant chosen per panel (2026-09-21, wiki/TODO.md item 3):
        # only the internal rise/fall split and stroke width change --
        # x_lead_in/x_lead_out stay the ones computed above from the fixed
        # constants, so every row's trace start (signal_start_x, which
        # assumes those same fixed constants) is unaffected by which shape
        # this particular panel draws.
        calibration_step_shape = rng.choice(list(CALIBRATION_STEP_SHAPE_VARIANTS))
        step_line_width, flat_frac = CALIBRATION_STEP_SHAPE_VARIANTS[calibration_step_shape]
        variant_flat_w = total_w * flat_frac / 2
        x_rise = round(x_lead_in + variant_flat_w)
        x_fall = round(x_lead_out - variant_flat_w)

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
            width=step_line_width,
            joint="curve",
        )
        # Exact extent of the drawn polyline (x_lead_in..x_lead_out, the
        # baseline..plateau y-range) padded by half the stroke width on
        # every side, since the line itself extends that far past its own
        # path. Ties to the panel's own printed gain (5mm/mV vs 10mm/mV,
        # see `gain` below) since "correlated" mode's step height is
        # derived directly from it -- the scale a reader would infer from
        # this box's height is always the one this panel actually prints.
        half_lw = step_line_width / 2
        calibration_step_bbox = (
            round(x_lead_in - half_lw), round(min(step_y0, step_mid_y) - half_lw),
            round(x_lead_out + half_lw), round(max(step_y0, step_mid_y) + half_lw),
        )
        # The actual drawn plateau and baseline y, not the pre-halving box count:
        # step_y0 sits step_h // 2 above step_mid_y, so recording step_h would
        # overstate the drawn rise by 2x. These two values plus the x edges are
        # exactly what a consumer needs to rebuild this step's waveform.
        step_geometry = (x_lead_in, x_rise, x_fall, x_lead_out, step_y0, step_mid_y)

    trace_mask: Image.Image | None = None
    px_per_sample = px_per_mv = None
    if lead_signals is not None:
        if signal_fs is None:
            raise ValueError("signal_fs is required when lead_signals is given")
        gain_mm_per_mv = float(gain.split("mm/mV")[0])
        px_per_second = grid_px / GRID_BOX_SECONDS
        px_per_sample = px_per_second / signal_fs
        px_per_mv = (grid_px / 5.0) * gain_mm_per_mv  # grid_px is px per 5mm major box

        # Trace-only ground truth mask, same pixels the visible line below
        # draws, on its own blank canvas -- excludes grid/header/labels/
        # calibration step, unlike the visible canvas they all share. Real
        # photos have no equivalent (this is a training target only a
        # synthetic generator can give for free), see TODO.md's digitizer
        # architecture-comparison item (2026-09-15).
        trace_mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(trace_mask)

        for name, (y_top, y_bottom) in zip(row_labels, row_y_ranges):
            signal = lead_signals[name]
            baseline_y = (y_top + y_bottom) / 2
            points = [
                (x_lead_out + i * px_per_sample, baseline_y - sample * px_per_mv)
                for i, sample in enumerate(signal)
            ]
            if len(points) >= 2:
                draw.line(points, fill=(0, 0, 0), width=TRACE_LINE_WIDTH_PX, joint="curve")
                mask_draw.line(points, fill=255, width=TRACE_LINE_WIDTH_PX, joint="curve")

    patient_ref = text_vocab.synthetic_patient_ref(record_ref)
    if style == "odd":
        fields = text_vocab.footer_fields(row_labels, patient_ref, rng, gain=gain)
    else:
        fields = text_vocab.even_footer_fields(patient_ref, rng)

    top_font, top_hb = _font_for_cap(TEXT_CAP_MM["body"], grid_px, bold=True)
    # Real photos show this row in red ink most of the time, but at least
    # one confirmed example prints it in black -- one color per panel, not
    # per field, since a real printer doesn't switch ink mid-line.
    bottom_row_color = rng.choice(BOTTOM_ROW_COLORS)
    for field_idx, (field, top_left, bottom_left) in enumerate(zip(fields, FOOTER_TOP_LEFT_BOXES[style], FOOTER_BOTTOM_LEFT_BOXES[style] + [None])):
        top_box = _draw_cap_text(draw, top_left * grid_px, (PANEL_MARGIN_TOP_BOXES + FOOTER_TOP_CAP_CENTER_BOXES) * grid_px, field.top, top_font, top_hb, (0, 0, 0))
        text_fields.append((f"footer_top_{field_idx}", field.top, top_box))
        if field.bottom:
            bottom_font, bottom_hb = _font_for_cap(TEXT_CAP_MM[FOOTER_BOTTOM_GROUPS[field_idx]], grid_px, bold=field.bottom_bold)
            bottom_box = _draw_cap_text(draw, bottom_left * grid_px, (PANEL_MARGIN_TOP_BOXES + FOOTER_BOTTOM_CAP_CENTER_BOXES) * grid_px, field.bottom, bottom_font, bottom_hb, bottom_row_color)
            text_fields.append((f"footer_bottom_{field_idx}", field.bottom, bottom_box))

    panel = PanelTemplate(
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
        trace_mask=trace_mask,
        trace_x0=x_lead_out if lead_signals is not None else None,
        px_per_sample=px_per_sample,
        px_per_mv=px_per_mv,
        baseline_ys=[(t + b) / 2 for t, b in row_y_ranges],
        step_row_index=(len(row_labels) - 1) if step_height_boxes is not None else None,
        step_geometry=step_geometry,
        text_fields=text_fields,
        grid_color=grid_color,
        calibration_step_shape=calibration_step_shape,
        panel_width_boxes=PANEL_WIDTH_BOXES,
        panel_noise_applied=False,
        panel_wrinkle_applied=False,
        paper_bend_applied=False,
        paper_bend_control_points=None,
    )

    # Bend applies before noise/wrinkle: it's a genuine geometric event (the
    # physical paper moving), while noise/wrinkle are pixel-level effects
    # from the camera/lighting at capture time, which happens after the
    # paper is already however-bent it is. Warps trace_mask identically so
    # mask-to-image correspondence survives.
    from dataclasses import replace
    if rng.random() < PAPER_BEND_PROBABILITY:
        bent_image, bent_mask, control_points = add_panel_bend(panel.image, panel.trace_mask, grid_px, rng)
        panel = replace(panel, image=bent_image, trace_mask=bent_mask, paper_bend_applied=True, paper_bend_control_points=control_points)

    # Noise and wrinkle apply last, to the final (possibly width-margined
    # and bent) canvas, so any added margin gets the same paper-surface
    # treatment as the original content instead of looking like a
    # pasted-on patch.
    if rng.random() < PANEL_NOISE_PROBABILITY:
        panel = replace(panel, image=add_panel_noise(panel.image, rng), panel_noise_applied=True)
    if rng.random() < PANEL_WRINKLE_PROBABILITY:
        panel = replace(panel, image=add_panel_wrinkle(panel.image, rng), panel_wrinkle_applied=True)
    return panel


REPORT_FONT_SIZE = 20
REPORT_LINE_HEIGHT_BOXES = 0.62  # fits ~19 lines in the body height a normal 3-row panel reserves for its rows


def build_report_panel_template(
    grid_px: float,
    record_ref: str = "0000000",
    rng: random.Random | None = None,
    grid_color: str | None = None,
) -> PanelTemplate:
    """The GE 12SL measurement/interpretation report block (2026-09-23,
    reference photo scan_0004.jpg): a text-only page element a real GE
    printout prints alongside its lead panels -- Serial Number, a handful
    of blank patient-identity label rows, a measurement table (QRS/QT/QTC/
    PR/P/RR-PP/axis), then free-text interpretation lines. No ECG trace at
    all, so this has none of build_panel_template's lead/calibration
    machinery -- returned as a PanelTemplate for structural compatibility
    with the rest of the pipeline (same canvas size, same header format),
    but every lead-specific field (lead_signals, trace_mask, calibration_step_*,
    label_bboxes, step_geometry, baseline_ys) is empty/None by construction.
    Callers must treat a panel with row_labels == [] as non-digitizable and
    skip it when writing panel-detection or digitization training labels --
    it carries no bbox for any of those, by design (there is nothing to box)."""
    rng = rng or random.Random()
    grid_color = grid_color or rng.choices(list(GRID_COLOR_WEIGHTS), weights=list(GRID_COLOR_WEIGHTS.values()))[0]
    canvas = blank_panel_canvas(grid_px, PAPER_TINT, grid_color=grid_color)
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)

    header_px = (PANEL_MARGIN_TOP_BOXES + HEADER_HEIGHT_BOXES) * grid_px
    icon_size = max(10, round(grid_px * HEADER_ICON_GRID_SIZE))
    icon_x0 = round(grid_px * HEADER_ICON_GRID_OFFSET)
    icon_xyxy = (icon_x0, OUTER_BORDER_INSET_PX, icon_x0 + icon_size, OUTER_BORDER_INSET_PX + round(grid_px * HEADER_ICON_GRID_HEIGHT))
    draw.rectangle(icon_xyxy, fill=(0, 0, 0))

    header_text = text_vocab.ge_datetime_header(rng)
    text_fields: list[tuple[str, str, tuple[int, int, int, int]]] = [
        ("header", header_text, _draw_header(draw, header_text, "even", grid_px)),
    ]

    body_font = _load_font(REPORT_FONT_SIZE, bold=False)
    label_x = round(grid_px * LABEL_GRID_OFFSET)
    value_x = round(grid_px * (LABEL_GRID_OFFSET + 5.0))
    line_h = round(grid_px * REPORT_LINE_HEIGHT_BOXES)
    y = round(header_px + line_h * 0.5)

    def draw_line(label: str, value: str = "") -> None:
        nonlocal y
        draw.text((label_x, y), label, fill=(0, 0, 0), font=body_font)
        text_fields.append((f"report_field_{len(text_fields)}", label, tuple(draw.textbbox((label_x, y), label, font=body_font))))
        if value:
            draw.text((value_x, y), value, fill=(0, 0, 0), font=body_font)
            text_fields.append((f"report_field_{len(text_fields)}", value, tuple(draw.textbbox((value_x, y), value, font=body_font))))
        y += line_h

    draw_line("Serial Number:", text_vocab.report_serial_number(rng))
    for label in ("Last Name", "First Name", "Date of Birth", "Sex"):
        draw_line(label)
    draw_line("Measurement Results:")
    measurements = text_vocab.report_measurements(rng)
    draw_line("QRS", f"{measurements['QRS']} ms")
    draw_line("QT/QTC", f"{measurements['QT']} / {measurements['QTC']} ms")
    draw_line("PR", f"{measurements['PR']} ms")
    draw_line("P", f"{measurements['P']} ms")
    draw_line("RR/PP", f"{measurements['RR']} / {measurements['PP']} ms")
    draw_line("P/QRS/T", f"{measurements['axis']} Deg")
    draw_line("Interpretation:")
    for line in text_vocab.report_interpretation(rng):
        draw_line(line)

    return PanelTemplate(
        image=canvas, row_y_ranges=[], header_icon_xyxy=icon_xyxy, header_text=header_text,
        footer_fields=[], row_labels=[], style="even", gain="", calibration_step_mode="none",
        has_calibration_step=False, bottom_row_color=(0, 0, 0), record_ref=record_ref,
        lead_signals=None, signal_fs=None, label_bboxes=[], calibration_step_bbox=None,
        trace_mask=None, trace_x0=None, px_per_sample=None, px_per_mv=None, baseline_ys=[],
        step_row_index=None, step_geometry=None, text_fields=text_fields, grid_color=grid_color,
        calibration_step_shape=None, panel_width_boxes=PANEL_WIDTH_BOXES,
        panel_noise_applied=False, panel_wrinkle_applied=False, paper_bend_applied=False,
        paper_bend_control_points=None,
    )


def build_blank_panel_template(grid_px: float, rng: random.Random | None = None, grid_color: str | None = None) -> PanelTemplate:
    """An unused page slot -- bare paper, dot grid, no printed content at
    all (2026-09-23, reference photo scan_0004.jpg: the bottom-right of its
    3x2 layout carries only a handwritten patient note, nothing this
    generator synthesizes). Same non-digitizable contract as
    build_report_panel_template: row_labels == [], no bbox of any kind."""
    rng = rng or random.Random()
    grid_color = grid_color or rng.choices(list(GRID_COLOR_WEIGHTS), weights=list(GRID_COLOR_WEIGHTS.values()))[0]
    canvas = blank_panel_canvas(grid_px, PAPER_TINT, grid_color=grid_color)
    return PanelTemplate(
        image=canvas, row_y_ranges=[], header_icon_xyxy=(0, 0, 0, 0), header_text="",
        footer_fields=[], row_labels=[], style="even", gain="", calibration_step_mode="none",
        has_calibration_step=False, bottom_row_color=(0, 0, 0), record_ref="0000000",
        lead_signals=None, signal_fs=None, label_bboxes=[], calibration_step_bbox=None,
        trace_mask=None, trace_x0=None, px_per_sample=None, px_per_mv=None, baseline_ys=[],
        step_row_index=None, step_geometry=None, text_fields=[], grid_color=grid_color,
        calibration_step_shape=None, panel_width_boxes=PANEL_WIDTH_BOXES,
        panel_noise_applied=False, panel_wrinkle_applied=False, paper_bend_applied=False,
        paper_bend_control_points=None,
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
