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

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


HAND_SHADOW_MAX_DARKEN = 0.38  # bumped from 0.22 (2026-09-05): "bigger shadow" -- still blurred/soft, just more visible
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
