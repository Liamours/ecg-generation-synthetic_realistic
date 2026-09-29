"""Global page-level augmentation (2026-09-05): handwriting on the page
background (drawn before pieces are pasted, so pieces can partially cover
it -- matches real pages where taped strips sit over handwritten notes),
a faint hand-shaped shadow (drawn after everything, since a real
photographer's hand shadow falls across the whole photographed scene),
and an optional whole-page tilt.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# Real handwriting-style fonts (Windows), varied per mark since real
# handwriting differs person to person -- not one fixed "handwriting font".
HANDWRITING_FONTS = ["SEGOESC.TTF", "MISTRAL.TTF", "FREESCPT.TTF", "BRADHITC.TTF", "segoepr.ttf"]
HANDWRITING_FONT_SIZE_RANGE = (24, 44)
HANDWRITING_COLOR_CHOICES = [(30, 30, 30), (20, 20, 60), (25, 25, 90)]  # black or blue-black ink
HANDWRITING_ROTATION_RANGE_DEG = (-10.0, 10.0)
HANDWRITING_MARKS_RANGE = (8, 14)  # bumped again 2026-09-05: "much more handwriting"
HANDWRITING_MARGIN_BAND_FRAC = 0.20
HANDWRITING_MULTILINE_PROBABILITY = 0.45  # fraction of marks that are a multi-line block instead of one short line
HANDWRITING_MULTILINE_ROWS_RANGE = (3, 4)
HANDWRITING_LINE_SPACING_FRAC = 1.25  # of font size

# Short clinical annotations plausible on a real page (matches this
# project's Indonesian pediatric-cardiology context, and the word list
# already used for this same purpose in the deleted pre-reset plan).
HANDWRITING_WORDS = [
    "hasil", "normal", "ulang", "cek", "revisi", "baik", "tgl", "anak",
    "ASD", "VSD", "PDA", "OK", "an.", "rawat",
]

# Longer sentence/list-line fragments, for multi-line notes -- matches
# the real handwritten diagnosis line found on an actual page
# (observed-page-samples.md: "Dx: Moderate VSD PMO... Moderate ARD...")
# in tone, not copied verbatim.
HANDWRITING_SENTENCE_LINES = [
    "Dx: VSD PDA ASD",
    "Hasil EKG normal",
    "Anak rawat ulang",
    "Cek ulang tgl 12",
    "Kontrol bulan depan",
    "Rujuk sp. jantung anak",
    "Riwayat sesak nafas",
    "Tidak ada keluhan",
    "BB naik, ASI lancar",
    "Jadwal kontrol berikut",
    "1. cek lab ulang",
    "2. rontgen thorax",
    "3. echo jantung",
    "tgl kontrol: minggu depan",
]


def _load_handwriting_font(rng: random.Random, size: int) -> ImageFont.FreeTypeFont:
    name = rng.choice(HANDWRITING_FONTS)
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _render_mark(rng: random.Random, w: int, h: int) -> Image.Image | None:
    """Renders one handwriting mark (RGBA, cropped to content) -- either a
    short 1-3 word line, or (per HANDWRITING_MULTILINE_PROBABILITY) a
    3-4 line note/list block, one consistent font/color/rotation per
    block since it's meant to read as one sitting of handwriting."""
    font_size = rng.randint(*HANDWRITING_FONT_SIZE_RANGE)
    font = _load_handwriting_font(rng, font_size)
    color = rng.choice(HANDWRITING_COLOR_CHOICES)

    if rng.random() < HANDWRITING_MULTILINE_PROBABILITY:
        num_lines = rng.randint(*HANDWRITING_MULTILINE_ROWS_RANGE)
        lines = [rng.choice(HANDWRITING_SENTENCE_LINES) for _ in range(num_lines)]
    else:
        num_words = rng.randint(1, 3)
        lines = [" ".join(rng.choice(HANDWRITING_WORDS) for _ in range(num_words))]

    line_height = round(font_size * HANDWRITING_LINE_SPACING_FRAC)
    tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    for i, line in enumerate(lines):
        jitter_x = rng.randint(-6, 6)  # each line starts slightly differently, like real handwriting
        draw.text((jitter_x, i * line_height), line, font=font, fill=(*color, 255))

    bbox = tmp.getbbox()
    if bbox is None:
        return None
    return tmp.crop(bbox)


def add_handwriting(image: Image.Image, rng: random.Random) -> None:
    """Draws several handwritten-looking marks (short lines and/or multi-
    line notes) at random positions and rotations directly onto `image`,
    in place. Called on the blank page background before any piece is
    pasted.

    Placement is biased into the 4 edge margins, not uniform across the
    page -- both more realistic (notes get written near a pasted strip's
    edge, not in the middle of empty paper) and less likely to land fully
    under a piece, which mostly fill the page's interior (found
    2026-09-05: uniform placement was invisible in every full-page test
    since pieces covered it every time)."""
    w, h = image.size
    num_marks = rng.randint(*HANDWRITING_MARKS_RANGE)
    margin_x = max(1, round(w * HANDWRITING_MARGIN_BAND_FRAC))
    margin_y = max(1, round(h * HANDWRITING_MARGIN_BAND_FRAC))

    for _ in range(num_marks):
        text_crop = _render_mark(rng, w, h)
        if text_crop is None:
            continue

        angle = rng.uniform(*HANDWRITING_ROTATION_RANGE_DEG)
        rotated = text_crop.rotate(angle, expand=True, resample=Image.BICUBIC)

        max_x = max(1, w - rotated.width)
        max_y = max(1, h - rotated.height)
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            x, y = rng.randint(0, max_x), rng.randint(0, min(margin_y, max_y))
        elif edge == "bottom":
            x, y = rng.randint(0, max_x), rng.randint(max(0, h - margin_y - rotated.height), max_y)
        elif edge == "left":
            x, y = rng.randint(0, min(margin_x, max_x)), rng.randint(0, max_y)
        else:
            x, y = rng.randint(max(0, w - margin_x - rotated.width), max_x), rng.randint(0, max_y)
        image.paste(rotated, (x, y), rotated)


HAND_SHADOW_MAX_DARKEN = 0.50  # bumped from 0.38 (2026-09-21, user call: shadow should appear more overwhelming) -- was bumped from 0.22 (2026-09-05): "bigger shadow" -- still blurred/soft, just more visible
HAND_SHADOW_BLUR_RADIUS_FRAC = 0.03  # of the page's shorter side
HAND_SHADOW_COLOR = (15, 12, 10)
HAND_SHADOW_PALM_RADIUS_FRAC_RANGE = (0.16, 0.24)  # bumped from (0.10, 0.16)
HAND_SHADOW_FINGER_LENGTH_FRAC_RANGE = (0.22, 0.34)  # bumped from (0.14, 0.24)


def add_hand_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    """A very faint, soft-edged hand/finger-shaped darkening intruding
    from one edge or corner, as if a photographer's hand were partially
    in frame between the light source and the page. Not a sharp
    silhouette -- heavily blurred and capped at a low peak opacity."""
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    edge = rng.choice(["left", "right", "top", "bottom"])
    if edge in ("left", "right"):
        base_x = -w * 0.08 if edge == "left" else w * 1.08
        base_y = rng.uniform(h * 0.2, h * 0.8)
    else:
        base_x = rng.uniform(w * 0.2, w * 0.8)
        base_y = -h * 0.08 if edge == "top" else h * 1.08

    # A "palm" blob plus 2-3 "finger" ellipses extending inward.
    palm_r = min(w, h) * rng.uniform(*HAND_SHADOW_PALM_RADIUS_FRAC_RANGE)
    draw.ellipse([base_x - palm_r, base_y - palm_r, base_x + palm_r, base_y + palm_r], fill=255)

    num_fingers = rng.randint(2, 3)
    inward = {"left": (1, 0), "right": (-1, 0), "top": (0, 1), "bottom": (0, -1)}[edge]
    for i in range(num_fingers):
        length = min(w, h) * rng.uniform(*HAND_SHADOW_FINGER_LENGTH_FRAC_RANGE)
        width = palm_r * rng.uniform(0.35, 0.55)
        spread = rng.uniform(-1, 1) * palm_r * 0.6
        perp = (-inward[1], inward[0])
        fx = base_x + inward[0] * length * 0.6 + perp[0] * spread
        fy = base_y + inward[1] * length * 0.6 + perp[1] * spread
        draw.ellipse([fx - length / 2, fy - width / 2, fx + length / 2, fy + width / 2], fill=255)

    blur_radius = min(w, h) * HAND_SHADOW_BLUR_RADIUS_FRAC
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0 * HAND_SHADOW_MAX_DARKEN
    base_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    shadow_color = np.array(HAND_SHADOW_COLOR, dtype=np.float32)
    blended = base_arr * (1 - mask_arr[..., None]) + shadow_color * mask_arr[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


BIG_SHADOW_COVERAGE_FRAC_RANGE = (0.35, 0.65)  # fraction of the page's area the shadow covers -- widened from (0.25, 0.50) 2026-09-21, user call: shadow should be more overwhelming
BIG_SHADOW_MAX_DARKEN = 0.70  # much stronger than the hand shadow -- a real cast shadow, not a dim patch. Bumped from 0.55 2026-09-21, same call
BIG_SHADOW_BLUR_RADIUS_FRAC = 0.05  # of the page's shorter side, soft edge
BIG_SHADOW_COLOR = HAND_SHADOW_COLOR


def add_big_cast_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    """A large, soft-edged shadow covering a big fraction of the page, as
    if a bystander or the photographer's own body blocked the light
    across part of the shot (reference: a real ekg-757 photo where roughly
    a third of the frame sits in shade). Unlike the localized hand shadow
    above, this one is a straight-edged band at a random angle, sized by
    area fraction directly: a linear field over the page is thresholded at
    the percentile matching the target coverage, so the shadow always
    covers exactly that fraction regardless of angle, then heavily blurred
    so the cut reads as a soft shadow edge, not a hard line."""
    w, h = image.size
    yy, xx = np.mgrid[0:h, 0:w]
    angle = rng.uniform(0, 2 * math.pi)
    field = xx * math.cos(angle) + yy * math.sin(angle)

    coverage = rng.uniform(*BIG_SHADOW_COVERAGE_FRAC_RANGE)
    cutoff = np.percentile(field, coverage * 100)
    mask = (field <= cutoff).astype(np.float32) * 255

    blur_radius = min(w, h) * BIG_SHADOW_BLUR_RADIUS_FRAC
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)

    mask_arr = mask / 255.0 * BIG_SHADOW_MAX_DARKEN
    base_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    shadow_color = np.array(BIG_SHADOW_COLOR, dtype=np.float32)
    blended = base_arr * (1 - mask_arr[..., None]) + shadow_color * mask_arr[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# "torn/bad data" direction (2026-09-08): a hand-torn paper edge, cut into
# the composed page rather than a clean rectangle. TORN_EDGE_BACKGROUND is
# deliberately not PAPER_TINT -- it stands for paper being physically
# missing (whatever surface is behind it), not a discolored patch of paper
# still present.
TORN_EDGE_BACKGROUND = (74, 71, 66)
TORN_EDGE_DEPTH_FRAC_RANGE = (0.03, 0.12)  # of the image's shorter side
TORN_EDGE_SEGMENTS = 14  # points along the tear line -- more = raggeder edge


def torn_edge_max_depth_px(image: Image.Image, rng: random.Random) -> float:
    """The depth apply_torn_edge is about to use, exposed so a caller can
    clip ground-truth bboxes to the guaranteed-safe remaining area before
    the tear's exact (random) jagged path is drawn."""
    return min(image.size) * rng.uniform(*TORN_EDGE_DEPTH_FRAC_RANGE)


def apply_torn_edge(image: Image.Image, rng: random.Random, side: str, max_depth: float) -> Image.Image:
    """Cuts an irregular, hand-torn-looking edge into one side of the
    page: a random-walk depth profile (not a straight diagonal cut) so the
    boundary reads as ragged, tapering back to 0 at both corners so the
    tear doesn't also eat into the adjacent edges. The torn-away area is
    replaced with TORN_EDGE_BACKGROUND, not blurred or discolored --
    content there is genuinely gone, the same as a real torn photo."""
    w, h = image.size
    along = w if side in ("top", "bottom") else h
    n = TORN_EDGE_SEGMENTS

    depths = [0.0]
    for _ in range(n):
        depths.append(max(0.0, min(max_depth, depths[-1] + rng.uniform(-max_depth * 0.35, max_depth * 0.35))))
    depths[0] = depths[-1] = 0.0
    positions = [round(i * along / n) for i in range(n + 1)]

    if side == "top":
        points = [(positions[i], depths[i]) for i in range(n + 1)]
        polygon = [(0, 0), *points, (w, 0)]
    elif side == "bottom":
        points = [(positions[i], h - depths[i]) for i in range(n + 1)]
        polygon = [(0, h), *points, (w, h)]
    elif side == "left":
        points = [(depths[i], positions[i]) for i in range(n + 1)]
        polygon = [(0, 0), *points, (0, h)]
    else:
        points = [(w - depths[i], positions[i]) for i in range(n + 1)]
        polygon = [(w, 0), *points, (w, h)]

    mask = Image.new("L", (w, h), 255)
    ImageDraw.Draw(mask).polygon(polygon, fill=0)
    background = Image.new("RGB", (w, h), TORN_EDGE_BACKGROUND)
    return Image.composite(image.convert("RGB"), background, mask)


# Real degradations documented in wiki/research/internal_dataset_characteristics.md
# and still missing after the 2026-09-08 augmentation pass (2026-09-08,
# this session): pen scribbles crossing the trace ink, black sticker/blot
# occlusions, broad uneven page lighting, and CamScanner-style re-scan
# processing. Signal-level ones (dead leads, motion-artifact bursts) live
# in paper_piece.py where the signal arrays exist. All four here are
# photo/paper-level: they touch no ground-truth geometry, so they need no
# bbox bookkeeping.
PEN_SCRIBBLE_COLOR_CHOICES = [(28, 28, 28), (25, 25, 70)]
PEN_SCRIBBLE_STROKE_WIDTH_RANGE = (2, 4)
PEN_SCRIBBLE_SEGMENT_COUNT_RANGE = (14, 24)  # wobble points per stroke -- more = cursive-er
PEN_SCRIBBLE_STROKE_COUNT_RANGE = (1, 3)
STICKER_BLOT_RADIUS_PX_RANGE = (35, 110)
STICKER_BLOT_COLOR = (12, 12, 12)
STICKER_BLOT_EDGE_COLOR = (45, 42, 40)
LIGHTING_GRADIENT_DARKEN_RANGE = (0.10, 0.30)  # peak multiplicative darkening at the darkest corner
CAMSCANNER_MIN_QUAD_AREA_FRAC = 0.15  # a detected paper quad smaller than this fraction of the page is treated as a bad detection, not a tiny real paper
CAMSCANNER_THRESHOLD_BLOCK_SIZE = 21  # andrewdcampbell/OpenCV-Document-Scanner's own adaptiveThreshold convention
CAMSCANNER_THRESHOLD_C = 15


def add_pen_scribble(image: Image.Image, rng: random.Random, region: tuple[float, float, float, float]) -> None:
    """Wavy ballpoint-pen strokes drawn straight over the panel area --
    unlike the margin-biased handwriting above, this one deliberately
    crosses the trace ink (the real-page failure mode the handwriting
    augmentation never covered). region is the placed cluster's bbox; a
    stroke starts on one side and random-walks across part of it, so it
    reliably lands on printed content."""
    x0, y0, x1, y1 = region
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(*PEN_SCRIBBLE_STROKE_COUNT_RANGE)):
        color = rng.choice(PEN_SCRIBBLE_COLOR_CHOICES)
        width = rng.randint(*PEN_SCRIBBLE_STROKE_WIDTH_RANGE)
        seg = rng.randint(*PEN_SCRIBBLE_SEGMENT_COUNT_RANGE)
        span_w = (x1 - x0) * rng.uniform(0.35, 0.8)
        span_h = (y1 - y0) * rng.uniform(0.2, 0.7)
        sx = rng.uniform(x0, max(x0, x1 - span_w))
        sy = rng.uniform(y0, max(y0, y1 - span_h))
        points = []
        for i in range(seg + 1):
            t = i / seg
            points.append((sx + t * span_w + rng.uniform(-6, 6), sy + t * span_h + rng.uniform(-18, 18)))
        draw.line(points, fill=color, width=width, joint="curve")


def add_sticker_blot(image: Image.Image, rng: random.Random, region: tuple[float, float, float, float]) -> None:
    """One near-black circular sticker/blot pasted over the panel area --
    full occlusion of whatever is under it (the real black-circle marks
    seen on ekg-757 pages). Opaque, with a slightly lighter rim so it
    reads as a physical sticker rather than a shadow."""
    x0, y0, x1, y1 = region
    r = rng.randint(*STICKER_BLOT_RADIUS_PX_RANGE)
    cx = rng.uniform(x0 + r, max(x0 + r, x1 - r))
    cy = rng.uniform(y0 + r, max(y0 + r, y1 - r))
    draw = ImageDraw.Draw(image)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=STICKER_BLOT_COLOR, outline=STICKER_BLOT_EDGE_COLOR, width=3)


BRUSH_MARK_COLOR = (24, 20, 16)  # dark dirt/ink smudge, not pure black -- distinct from the sticker blot's near-black opaque fill
BRUSH_MARK_OPACITY_RANGE = (0.12, 0.30)  # low opacity: content under it stays legible, unlike the sticker blot's full occlusion
BRUSH_MARK_WIDTH_RANGE = (8, 22)  # brush-stroke width, wider than a pen scribble's line width
BRUSH_MARK_LENGTH_FRAC_RANGE = (0.25, 0.55)  # of the placed cluster's own span
BRUSH_MARK_DAB_COUNT_RANGE = (18, 32)  # ellipses laid along the path
BRUSH_MARK_DAB_SKIP_PROB = 0.35  # each dab has this chance of being skipped -- the broken, dotted tire-track edge
BRUSH_MARK_JITTER_PX = 5.0  # perpendicular-to-path wobble per dab, for a messy (not a clean stripe) look


def add_brush_mark(image: Image.Image, rng: random.Random, region: tuple[float, float, float, float]) -> None:
    """A low-opacity, irregular brush-stroke/tire-track mark: a wavy path
    of overlapping dabs, randomly sized and randomly skipped, so the edge
    reads as broken and dotted rather than one clean stroke -- distinct
    from `add_pen_scribble` (opaque, thin, follows a pen's line) and
    `add_sticker_blot` (opaque, one clean circle). Added 2026-09-21
    (wiki/TODO.md item 10) for the real "other marks not part of the ECG
    itself" category plan.md's own caveats already name but nothing drew.
    Alpha-blended onto its own layer first (same technique as the hand/
    cast shadows), since a low, uneven opacity can't be expressed as a
    single flat PIL fill color."""
    x0, y0, x1, y1 = region
    width = image.width
    length = (x1 - x0) * rng.uniform(*BRUSH_MARK_LENGTH_FRAC_RANGE)
    angle = rng.uniform(0, 2 * math.pi)
    sx = rng.uniform(x0, max(x0, x1 - length * abs(math.cos(angle))))
    sy = rng.uniform(y0, max(y0, y1 - length * abs(math.sin(angle))))
    dab_count = rng.randint(*BRUSH_MARK_DAB_COUNT_RANGE)
    base_opacity = rng.uniform(*BRUSH_MARK_OPACITY_RANGE)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    perp = angle + math.pi / 2
    for i in range(dab_count):
        if rng.random() < BRUSH_MARK_DAB_SKIP_PROB:
            continue
        t = i / max(1, dab_count - 1)
        wobble = rng.uniform(-BRUSH_MARK_JITTER_PX, BRUSH_MARK_JITTER_PX)
        cx = sx + t * length * math.cos(angle) + wobble * math.cos(perp)
        cy = sy + t * length * math.sin(angle) + wobble * math.sin(perp)
        r = rng.uniform(*BRUSH_MARK_WIDTH_RANGE) / 2
        alpha = round(255 * base_opacity * rng.uniform(0.6, 1.0))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*BRUSH_MARK_COLOR, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=2.0))  # softens each dab's edge into a smudge, not a hard-edged dot
    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"), (0, 0))


# Whole-page color temperature, added 2026-09-21 (wiki/TODO.md, user
# request with 6 reference ekg-757 photos). Measured against those
# references (bright-paper pixels only, gray > 180, same technique the
# grid-color check used) rather than guessed: warm R-B +28.6/+28.7,
# G-B +16.7/+16.4; cold R-B -33.4/-31.7, G-B -39.7/-28.7; "neutral" itself
# reads slightly cool, R-B -5.3/-12.9, not exactly zero. A real, large,
# consistent swing (~62 R-B units warm-to-cold), not a subtle effect.
# Distinct from apply_lighting_gradient above: that's the light source's
# spatial falloff (a gradient), this is the light source's COLOR (a flat
# per-page cast) -- two different real effects, kept as separate,
# composable augmentations rather than merged into one.
PAGE_TEMPERATURE_R_B_RANGE = (-33.0, 29.0)  # matches the measured cold-to-warm range directly, not a rounder invented number
PAGE_TEMPERATURE_NEUTRAL_R_B = -9.0  # the measured "neutral" midpoint (mean of -5.3/-12.9), not an assumed 0
PAGE_TEMPERATURE_G_B_FRAC = 0.55  # G-B moves about half as far as R-B in every reference sample above -- kept proportional, not independently random


def apply_page_temperature(image: Image.Image, rng: random.Random, r_b_shift: float | None = None) -> Image.Image:
    """Shifts the whole page's color cast warm or cold by adjusting R and G
    relative to B, additively (not multiplicative like apply_lighting_gradient,
    since a color cast is a constant offset in the source light's spectrum,
    not something that scales with local brightness). r_b_shift lets a
    caller pick a specific point on the measured range directly (e.g. for
    a demo or a paired A/B check) instead of a random draw."""
    if r_b_shift is None:
        r_b_shift = rng.uniform(*PAGE_TEMPERATURE_R_B_RANGE)
    delta = r_b_shift - PAGE_TEMPERATURE_NEUTRAL_R_B
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    arr[..., 0] += delta          # R
    arr[..., 1] += delta * PAGE_TEMPERATURE_G_B_FRAC  # G
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_lighting_gradient(image: Image.Image, rng: random.Random) -> Image.Image:
    """Broad, smooth exposure unevenness across the whole page -- the
    diffuse light-falloff real photos show, distinct from the localized
    hand shadow above. A low-resolution random field is upscaled and
    blurred, so the gradient has no visible cell structure; multiplicative
    so it darkens ink and paper together like real uneven lighting."""
    w, h = image.size
    lo = 1.0 - rng.uniform(*LIGHTING_GRADIENT_DARKEN_RANGE)
    field_rng = np.random.default_rng(rng.randrange(2**32))
    field = np.asarray(
        Image.fromarray((field_rng.random((6, 6)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(radius=min(w, h) * 0.08)),
        dtype=np.float32,
    )
    field = (field / 255.0) * (1.0 - lo) + lo
    field /= field.max()  # keep the brightest point at full exposure
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) * field[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """top-left/top-right/bottom-right/bottom-left via sum/diff of
    coordinates -- the standard document-scanner ordering trick both
    reference repos use (pyimagesearch's four_point_transform)."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _detect_document_quad(bgr: np.ndarray) -> np.ndarray | None:
    """Finds the paper's own quad against its table background --
    vipul-sharma20/document-scanner's Canny -> findContours -> largest-
    contour approach, simplified since this project's own synthetic
    paper-on-table contrast is far cleaner than an arbitrary real photo
    (no LSD needed, unlike andrewdcampbell/OpenCV-Document-Scanner's more
    robust but heavier version). Returns None (caller falls back to the
    full image, matching that second repo's own fallback) when no
    big-enough quad is found, rather than the first repo's undefined
    crash on zero contours."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < CAMSCANNER_MIN_QUAD_AREA_FRAC * w * h:
        return None
    approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
    quad = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(largest))
    return _order_quad_corners(np.asarray(quad, dtype=np.float32))


def apply_camscanner_look(image: Image.Image, rng: random.Random) -> tuple[Image.Image, np.ndarray]:
    """Real CamScanner-style re-scan: detects the paper's own quad, warps
    it flat, then adaptive-thresholds to true black-and-white -- the two
    things the old autocontrast-only version never did (stayed RGB, no
    perspective correction; see wiki/TODO.md). Falls back to the full
    image, uncropped, when no quad is found.

    Returns the new image plus the 3x3 perspective matrix mapping OLD
    pixel coordinates to NEW ones (identity when no quad was found) --
    this is the one page-level augmentation that changes the canvas's own
    geometry rather than only its pixels, so callers must run every
    already-placed column bbox through it (see page.py's call site)."""
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    quad = _detect_document_quad(bgr)

    if quad is None:
        rectified_bgr, matrix = bgr, np.eye(3, dtype=np.float32)
    else:
        tl, tr, br, bl = quad
        out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad, dst)
        rectified_bgr = cv2.warpPerspective(bgr, matrix, (out_w, out_h), borderValue=(255, 255, 255))

    gray = cv2.GaussianBlur(cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                CAMSCANNER_THRESHOLD_BLOCK_SIZE, CAMSCANNER_THRESHOLD_C)
    out = Image.fromarray(bw).convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=rng.randint(40, 90), threshold=3))
    return out, matrix
