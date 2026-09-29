"""Page composition (2026-09-05): place 2-3 paper pieces (see
paper_piece.py) onto one background page, sized from `ekg-757`'s own real
image dimensions (not an invented size), each piece given a discrete
facing (0/90/180/270) and its own tilt. The real 3D perspective warp
(`apply_page_perspective_warp_3d`, +-13 degrees per axis) is applied once,
last, to the whole composed page -- paper, table background, and every
other page-level augmentation together -- not per piece; see that
function's own docstring for why (2026-09-21, wiki/TODO.md item 7,
reduced from an earlier +-26deg per-piece-or-per-paper version). Ground
truth: every column's page-space bbox and orientation.

Total columns per page: 5-6, split across 2-3 pieces (per spec) --
matches this project's earlier-established "4-6 panels per page"
observation, not a new number invented here.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from importlib import resources

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import variants
from .config import TABLE_TEXTURE_DIR
from .page_augment import (
    add_big_cast_shadow,
    add_brush_mark,
    add_hand_shadow,
    add_handwriting,
    add_pen_scribble,
    add_sticker_blot,
    apply_camscanner_look,
    apply_lighting_gradient,
    apply_page_temperature,
    apply_torn_edge,
    torn_edge_max_depth_px,
    PAGE_TEMPERATURE_R_B_RANGE,
)
from .paper_piece import BLANK_SENTINEL, REPORT_SENTINEL, Column, PaperPiece, _bbox_from_meta, _project_plane_corners, apply_perspective_warp, apply_tilt, build_paper_piece, map_panel_points, pil_rotate_transform, scale_piece
from .template import PAPER_TINT

with resources.files("ecg_synthetic_realistic.data").joinpath("page_sizes_ekg757.json").open() as f:
    _REAL_PAGE_SIZES: list[tuple[int, int]] = [tuple(wh) for wh in json.load(f)]

DISCRETE_ORIENTATIONS = [0, 90, 180, 270]
TILT_RANGE_DEG = (-6.0, 6.0)  # reduced from +-15 (2026-09-05): looked too tilted once combined with discrete orientation on a full page

# Curated real-observed row layouts (2026-09-16, user-supplied), each a
# ragged list of per-row panel counts, e.g. [2, 2, 1] for "12/34/5" --
# replaces the old random 5-6-flat-columns-split-into-2-3-pieces default.
# Chosen to keep the resulting paper's own aspect ratio close to a real
# single photographed page (see compose_page's row_layout docstring).
PAGE_ROW_TEMPLATES: list[list[int]] = [
    [2, 2, 1], [2, 2, 2], [3, 3], [3, 2], [2, 2], [2, 2, 1, 2], [4, 4],
    [2, 2, 2, 2], [1, 2, 2], [2, 1, 2], [2, 1, 2, 1], [2, 3], [3, 3, 2],
]

# Pieces render at full quality (grid_px tuned for legible text/grid, see
# template.py), then get scaled for placement -- rescaling grid_px itself
# would shrink the fixed-pixel font sizes too and break the already-tuned
# legibility. Calibrated 2026-09-05 against a real single-column
# reference photo (903x776px) against our native render (709x591) --
# an UPSCALE (real single panels are bigger than our native render), not
# the downscale an earlier, wrong worst-case guess assumed.
PIECE_SCALE_TO_PAGE = 903 / 709
NEAR_BEST_AREA_FRAC = 0.15  # candidates within 15% of the smallest area are all "good enough" -- pick among them at random for variety instead of always the single tightest one

PAGE_TILT_PROBABILITY = 0.6
PAGE_TILT_RANGE_DEG = (-4.0, 4.0)  # smaller than a single piece's tilt -- this is the whole photographed page, not one strip

# Three page styles (2026-09-08), sampled per page by the caller (see
# generate_dataset.py's PAGE_STYLE_WEIGHTS) and passed into compose_page:
#   "standard" -- existing behavior, unchanged.
#   "ideal"    -- "ideal good looking data that is already tidy but hard
#                 to separate into their columns": no handwriting/shadow/
#                 page-tilt degradation, but every piece shares one
#                 orientation and near-zero tilt/warp so adjacent pieces'
#                 edges line up smoothly, and no staple marks -- nothing
#                 left to visually tell two touching columns apart except
#                 the printed content itself.
#   "torn"     -- "bad data with awful croppings, tear paper, and hard to
#                 crop sections": standard degradation plus a hand-torn
#                 edge cut into the page and a harsher, sometimes-
#                 negative placement margin that can crop straight into a
#                 piece's own printed content.
PAGE_STYLES = ("standard", "ideal", "torn")

# 2026-09-08, real-degradation coverage (see page_augment.py's own note):
# per-page probabilities for the four page-level ones, applied only on
# non-"ideal" pages (ideal is deliberately the clean style). Scribble and
# sticker go on the paper before the page tilt so they rotate with it;
# lighting and the CamScanner look are camera-side, so they apply after
# the tilt, matching how a real photo is produced.
PEN_SCRIBBLE_PROBABILITY = 0.25
STICKER_BLOT_PROBABILITY = 0.10
BRUSH_MARK_PROBABILITY = 0.15  # added 2026-09-21, wiki/TODO.md item 10 -- a distraction mark distinct from the scribble/sticker pair above
LIGHTING_GRADIENT_PROBABILITY = 0.35
CAMSCANNER_LOOK_PROBABILITY = 0.12
BIG_SHADOW_PROBABILITY = 0.45  # raised from 0.25 2026-09-21, user call: shadow should appear more often. Was 0.25 2026-09-15: reference ekg-757 photo shows a cast shadow over roughly a third of the frame

# Reduced from (-26.0, 26.0) 2026-09-21 (wiki/TODO.md item 7, user call:
# "a bit less violent"). Originally validated 2026-09-15 against
# apply_perspective_warp_3d's own test trials
# (analyses/warp3d-experiment/warp3d_{tilt_x,tilt_y,both,extreme}.png) at
# the wider range; halved here rather than re-deriving from new trials,
# since the ask was specifically to soften the existing effect, not
# re-pick it from scratch. Also moved from a per-paper draw (inside
# _build_paper) to compose_page's own last step -- see PAGE_WARP3D note there.
PAGE_WARP3D_TILT_RANGE_DEG = (-13.0, 13.0)

# 2026-09-16: a real photo always shows some of the surface the paper sits
# on around/behind it -- page_img used to be a flat PAPER_TINT fill for the
# whole canvas, which has no such surface at all (see wiki/TODO.md's
# synthetic-page-generator gaps). User-supplied reference photos live in
# datasets/table_textures/ (excludes the *_square previews built for quick
# viewing, not for compositing -- those are pre-stretched and would double-
# distort here).
_TABLE_TEXTURE_PATHS = sorted(
    p for p in TABLE_TEXTURE_DIR.glob("*.*")
    if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "_square" not in p.stem
)

# 2026-09-16, user call: most real ekg-757 photos don't show table surface
# at all (paper fills the frame), so a visible table background should be
# the minority case, not the default -- confirmed after the compositing
# order itself was fixed and approved.
TABLE_BACKGROUND_PROBABILITY = 0.15

# 2026-09-25: build_report_panel_template/build_blank_panel_template already
# existed (2026-09-23, reference photo scan_0004.jpg) but were only reachable
# via compose_page's forced_row_labels override, never the default random
# path -- so the corpus had zero non-digitizable panels in it, and
# detection_panels_leds (trained only on real-lead rows) had no way to learn
# what "not a lead row" looks like. Confirmed on a real photo: it drew a
# confident led_row box over a report panel's measurement-text block. Applied
# per filler slot (see _sample_page_row_labels), not per page -- a filler
# slot is already a random duplicate of a standard group, so this just
# changes what a fraction of those duplicates become. 0.15 is a starting
# value, not measured against a specific real-photo report/blank frequency.
REPORT_OR_BLANK_FILLER_PROBABILITY = 0.15


def _add_paper_shadow(page_img: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Soft drop shadow offset from the paper's own placement rect, blurred,
    drawn before the paper itself is pasted -- so the paper reads as a
    physical sheet resting on the table, not a flat texture painted onto
    it (2026-09-16, user call: "flapped shadow ... looks like an actual
    paper on top")."""
    offset = round(min(w, h) * 0.015) + 6
    blur_radius = max(4, offset)
    shadow_layer = Image.new("RGBA", page_img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow_layer).rectangle([x + offset, y + offset, x + w + offset, y + h + offset], fill=(0, 0, 0, 110))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.alpha_composite(page_img.convert("RGBA"), shadow_layer).convert("RGB")


def _make_table_background(rng: random.Random, page_w: int, page_h: int) -> tuple[Image.Image, str | None]:
    """15% of pages get a random real table photo, squeezed (stretched,
    not cover-cropped) directly to page_w x page_h; the other 85% keep the
    old flat PAPER_TINT fill (no visible surface around the paper at all).
    page_w/page_h themselves already come from _choose_page_size's real
    ekg-757 page-size dataset either way, so this only changes what fills
    the canvas, never the page's own dimensions.

    Returns the image plus the texture path actually used (None for the flat
    fill), added 2026-09-18: which surface a page was composited onto was
    previously unrecoverable once the pixels existed."""
    if not _TABLE_TEXTURE_PATHS or rng.random() >= TABLE_BACKGROUND_PROBABILITY:
        return Image.new("RGB", (page_w, page_h), PAPER_TINT), None
    path = rng.choice(_TABLE_TEXTURE_PATHS)
    texture = Image.open(path).convert("RGB")
    # Fixed 2026-09-22 (user call, "wrong orientation on the background"):
    # squeeze-stretching straight to page_w x page_h ignored the texture
    # photo's own natural orientation -- a portrait tray/table photo
    # squeezed into a landscape page canvas came out visibly wrong (the
    # object's real proportions, rounded corners etc. land on the wrong
    # edge). Rotate first when the two orientations disagree so the
    # texture's own "up" still roughly matches the page's, then the
    # existing stretch only has to correct the aspect ratio, not the
    # orientation too. "square" classifications skip this (no rotation
    # is more correct than another there).
    if classify_orientation(*texture.size) != classify_orientation(page_w, page_h) and classify_orientation(page_w, page_h) != "square":
        texture = texture.transpose(Image.ROTATE_90)
    return texture.resize((page_w, page_h), Image.LANCZOS), str(path)


def _cluster_bbox(page: Page) -> tuple[float, float, float, float]:
    """The union of every surviving column's bbox -- the placed paper
    area. Scribbles and stickers land inside this, not on empty page.
    Falls back to the full page when a harsh crop has (in principle,
    never observed in practice at today's HARSH_CROP_OVERFLOW_FRAC)
    clipped away every column -- min()/max() over an empty sequence
    would otherwise raise."""
    boxes = [col.bbox for piece in page.pieces for col in piece.columns]
    if not boxes:
        return (0.0, 0.0, float(page.image.width), float(page.image.height))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

IDEAL_TILT_RANGE_DEG = (-0.5, 0.5)
IDEAL_PERSPECTIVE_MAX_SHIFT_FRAC = 0.0
HARSH_CROP_OVERFLOW_FRAC = 0.08  # of the placed cluster's own size, how far it's allowed to hang off the page edge
MIN_VISIBLE_AREA_FRAC = 0.5  # a column bbox surviving a crop/tear at less than this fraction of its original area is dropped, not just shrunk


@dataclass
class PlacedPiece:
    image: Image.Image
    position: tuple[int, int]  # top-left, in page coordinates
    columns: list[Column]  # bboxes already in page coordinates


@dataclass
class Page:
    image: Image.Image
    pieces: list[PlacedPiece]
    # Which optional augmentations actually fired for this page, and the few
    # parameters worth keeping (2026-09-18). None of this was recorded before,
    # so a page could not be described, only looked at: "is this panel's trace
    # obscured by a scribble" was unanswerable from the manifest, and so was
    # "did this page go through the scanner filter". Populated by compose_page;
    # every entry is absent rather than False when the step never ran.
    augmentations: dict = field(default_factory=dict)


def _sample_page_row_labels(rng: random.Random, total: int) -> list[list[str]]:
    """All 4 standard groups (romanic/aV/first_v/second_v) at least once
    -- a real requirement, not just variety for its own sake -- then
    fills any remaining slots (page has 5-6 columns, only 4 mandatory
    groups) with random duplicates, matching the duplicate-panel pattern
    already observed on real pages, or (2026-09-25) a non-digitizable
    report/blank slot (REPORT_OR_BLANK_FILLER_PROBABILITY) -- real pages
    sometimes carry one (scan_0004.jpg), and detection_panels_leds needs
    real negative examples in its training data, not just real-lead rows,
    to learn what "not a lead row" looks like (see the probability
    constant's own comment). Shuffled so group order isn't fixed.

    total < 4 (2026-09-16, only reachable via compose_page's explicit
    column_counts override -- the default random total_columns is always
    5-6): can't fit all 4 groups on paper this small, so this picks
    `total` distinct groups instead of guaranteeing full coverage --
    compose_page skips its usual all-4-groups assertion in that case. No
    report/blank filler here either, same reasoning -- too few columns to
    spare one on a non-digitizable slot."""
    if total < len(variants.STANDARD_ROW_LABEL_SETS):
        return rng.sample(variants.STANDARD_ROW_LABEL_SETS, total)
    row_labels = list(variants.STANDARD_ROW_LABEL_SETS)
    for _ in range(total - len(row_labels)):
        if rng.random() < REPORT_OR_BLANK_FILLER_PROBABILITY:
            row_labels.append(rng.choice([REPORT_SENTINEL, BLANK_SENTINEL]))
        else:
            row_labels.append(rng.choice(variants.STANDARD_ROW_LABEL_SETS))
    rng.shuffle(row_labels)
    return row_labels


def _split_row_labels_into_pieces(row_labels: list[list[str]], column_counts: list[int]) -> list[list[list[str]]]:
    pieces = []
    idx = 0
    for count in column_counts:
        pieces.append(row_labels[idx:idx + count])
        idx += count
    return pieces


def _merge_pieces_into_paper(rng: random.Random, pieces: list[PaperPiece]) -> PaperPiece:
    """Fuses 1-3 already-built, not-yet-tilted/warped pieces into one
    single paper object (2026-09-16), reusing _choose_layout's tight
    zero-gap packing -- multiple physical strips taped into one sheet
    still pack edge to edge, but as one shared canvas this time, before
    any rotation or warp is applied to it (see _build_paper)."""
    if len(pieces) == 1:
        return pieces[0]
    layout_w, layout_h, offsets = _choose_layout(rng, pieces)
    canvas = Image.new("RGB", (round(layout_w), round(layout_h)), PAPER_TINT)
    columns: list[Column] = []
    for piece, (x, y) in zip(pieces, offsets):
        canvas.paste(piece.image, (round(x), round(y)))
        for col in piece.columns:
            bx0, by0, bx1, by1 = col.bbox
            columns.append(Column(bbox=(bx0 + x, by0 + y, bx1 + x, by1 + y), orientation_deg=col.orientation_deg,
                                  column_type=col.column_type,
                                  meta=map_panel_points(col.meta, lambda px, py: (px + x, py + y))))
    return PaperPiece(image=canvas, columns=columns)


def _build_paper(rng: random.Random, grid_px: float, piece_row_labels: list[list[list[str]]], page_style: str = "standard", row_layout: list[int] | None = None,
                  uniform_style: str | None = None, uniform_grid_color: bool = False, no_row_tilt: bool = False) -> list[PaperPiece]:
    """Builds every physical piece (fused columns, no per-piece transform
    yet), merges them into one paper object (a no-op when there's only
    one piece), then applies exactly one discrete orientation + continuous
    tilt + perspective warp + page-scale to that single merged paper.
    Returns a length-1 list so the caller (_place_pieces, which already
    supports a single piece) needs no further change.

    2026-09-16, corrects a real bug flagged directly by the user: each
    piece used to get its OWN independent rotation and 3D warp before
    being placed, which made a multi-piece page look like a collage of
    separately-photographed clippings tumbled at different angles, not
    one camera shot of one physical sheet. A real photo warps as a single
    rigid plane -- multiple strips taped together move as one object
    when the camera tilts, they don't each pick their own random angle.

    "ideal" style still gets near-zero tilt/perspective and skips
    staples (see build_paper_piece's own docstring) -- with only one
    merged paper now, "hard to separate into their columns" is
    automatic (no per-piece boundary rotation exists to give it away),
    not something that needs a shared-orientation special case anymore."""
    ideal = page_style == "ideal"

    pieces = []
    for row_labels_list in piece_row_labels:
        # row_layout (2026-09-16) only applies when this is the page's one
        # and only piece -- a multi-piece page's per-piece internal grid
        # isn't a case this has been asked for yet.
        piece_row_layout = row_layout if (row_layout is not None and len(piece_row_labels) == 1) else None
        pieces.append(build_paper_piece(rng, grid_px=grid_px, row_labels_list=row_labels_list, draw_staples=not ideal, row_layout=piece_row_layout,
                                         uniform_style=uniform_style, uniform_grid_color=uniform_grid_color, no_row_tilt=no_row_tilt))

    paper = _merge_pieces_into_paper(rng, pieces)

    orientation = rng.choice(DISCRETE_ORIENTATIONS)
    tilt_angle = rng.uniform(*(IDEAL_TILT_RANGE_DEG if ideal else TILT_RANGE_DEG))
    paper = apply_tilt(paper, angle_deg=orientation + tilt_angle, orientation_delta=orientation, tilt_deg=tilt_angle)
    if ideal:
        paper = apply_perspective_warp(paper, rng, max_shift_frac=IDEAL_PERSPECTIVE_MAX_SHIFT_FRAC)
    # 2026-09-21 (wiki/TODO.md item 7): the 3D perspective warp used to be
    # applied here, per merged paper, before the paper was even placed onto
    # the page/table background -- moved to compose_page's own last step
    # instead, so it warps the whole photographed scene (paper + table
    # background + every other page-level augmentation already applied),
    # matching how a real camera angle affects everything in frame at once,
    # not just the paper. See apply_page_perspective_warp_3d below.
    paper = scale_piece(paper, PIECE_SCALE_TO_PAGE)
    return [paper]


Size = tuple[int, int]
Offset = tuple[float, float]


def _pack_two_pieces(sizes: list[Size]) -> list[tuple[float, float, list[Offset]]]:
    """Side by side only (2026-09-16, user call): physical pieces only
    ever join horizontally, never stacked top-down -- two separate pieces
    of paper aren't taped into a single vertical sheet."""
    (w0, h0), (w1, h1) = sizes
    return [(w0 + w1, max(h0, h1), [(0, 0), (w0, 0)])]


def _pack_three_pieces(sizes: list[Size]) -> list[tuple[float, float, list[Offset]]]:
    """One horizontal row of all three (2026-09-16, user call: pieces only
    join horizontally, never stacked) -- replaces the old pair+lone/
    stacked candidate search now that vertical joins are off the table."""
    (w0, h0), (w1, h1), (w2, h2) = sizes
    return [(w0 + w1 + w2, max(h0, h1, h2), [(0.0, 0.0), (float(w0), 0.0), (float(w0 + w1), 0.0)])]


def _choose_layout(rng: random.Random, pieces: list[PaperPiece]) -> tuple[float, float, list[Offset]]:
    """Deterministically packs every piece with no overlap (never drops
    one, unlike the old random-rejection placement it replaced -- see
    2026-09-06 perf notes below), picking among the near-tightest
    packings at random for shape variety."""
    sizes = [p.image.size for p in pieces]
    if len(sizes) == 1:
        # A single piece has nothing to pack against -- its own size is
        # the layout (2026-09-16, for the single-paper "background -> page
        # (paper + panels)" case with no multi-strip composition at all).
        (w0, h0) = sizes[0]
        candidates = [(float(w0), float(h0), [(0.0, 0.0)])]
    elif len(sizes) == 2:
        candidates = _pack_two_pieces(sizes)
    elif len(sizes) == 3:
        candidates = _pack_three_pieces(sizes)
    else:
        raise ValueError(f"expected 1-3 pieces, got {len(sizes)}")

    candidates.sort(key=lambda c: c[0] * c[1])
    best_area = candidates[0][0] * candidates[0][1]
    near_best = [c for c in candidates if c[0] * c[1] <= best_area * (1 + NEAR_BEST_AREA_FRAC)]
    return rng.choice(near_best)


ORIENTATION_ASPECT_TOLERANCE = 1.05  # matches the ratio the real page-size dataset itself splits on (4 portrait / 6 landscape / 0 square at this threshold)


def classify_orientation(w: float, h: float) -> str:
    """"portrait" / "landscape" / "square", the same threshold used to
    split _REAL_PAGE_SIZES. A single shared function so the paper's own
    shape and a candidate frame size are always judged the same way --
    see _choose_page_size and _make_table_background, both of which
    need the two orientations to agree, not just eyeball "wide vs tall"
    separately with their own tolerance."""
    if h > w * ORIENTATION_ASPECT_TOLERANCE:
        return "portrait"
    if w > h * ORIENTATION_ASPECT_TOLERANCE:
        return "landscape"
    return "square"


def _choose_page_size(rng: random.Random, layout_w: float, layout_h: float) -> tuple[int, int]:
    """A real ekg-757 page size, constrained 2026-09-22 (user call, "we
    need consistency") to match the paper cluster's own orientation --
    before this, the page/photo-frame size was drawn fully independently
    of the paper (2026-09-16 decision), which could hand a portrait sheet
    of paper a landscape camera frame or vice versa. `layout_w`/`layout_h`
    are the cluster's size AFTER _build_paper's own discrete 0/90/180/270
    rotation has already been applied, so this reads the paper's actual
    photographed shape, not its pre-rotation content orientation -- a
    paper rotated 90deg for the shot legitimately wants a frame to match
    that rotation, not its upright reading orientation.

    Falls back to the full, unfiltered list when no real page size shares
    the paper's orientation (e.g. a near-square cluster: the real dataset
    has zero square-classified entries), so this never raises on an edge
    case, it just can't guarantee the match there."""
    orientation = classify_orientation(layout_w, layout_h)
    matching = [wh for wh in _REAL_PAGE_SIZES if classify_orientation(*wh) == orientation]
    return rng.choice(matching or _REAL_PAGE_SIZES)


def _place_pieces(rng: random.Random, pieces: list[PaperPiece], harsh: bool = False) -> tuple[int, int, list[PaperPiece], list[tuple[int, int]]]:
    """Packs `pieces` into one cluster (see _choose_layout), picks a real
    page size independent of that cluster's own size (see
    _choose_page_size), then randomizes where the cluster sits relative
    to the page -- slack can come out negative when the page is smaller
    than the cluster, which is fine: `rng.uniform(0, slack)` handles a
    negative bound the same way, distributing the crop naturally instead
    of needing a special case.

    harsh=True (the "torn" page style's "awful croppings, hard to crop
    sections" direction) adds its own extra overflow allowance on top,
    up to HARSH_CROP_OVERFLOW_FRAC of the cluster's own size -- a
    deliberately more dramatic version of the same off-frame idea.
    compose_page clips every column bbox to the final page bounds
    afterward, so a piece cut this way never claims content that isn't
    actually in the image."""
    layout_w, layout_h, offsets = _choose_layout(rng, pieces)
    page_w, page_h = _choose_page_size(rng, layout_w, layout_h)

    for piece in pieces:
        for col in piece.columns:
            col.meta["page_scale_factor"] = 1.0  # page and paper sizes are independent now -- no shrink-to-fit ever fires

    slack_w, slack_h = page_w - layout_w, page_h - layout_h
    if harsh:
        overflow_w, overflow_h = layout_w * HARSH_CROP_OVERFLOW_FRAC, layout_h * HARSH_CROP_OVERFLOW_FRAC
        pad_left = rng.uniform(-overflow_w, max(-overflow_w, slack_w + overflow_w))
        pad_top = rng.uniform(-overflow_h, max(-overflow_h, slack_h + overflow_h))
    else:
        pad_left = rng.uniform(0, slack_w)
        pad_top = rng.uniform(0, slack_h)
    positions = [(round(pad_left + x), round(pad_top + y)) for x, y in offsets]
    return page_w, page_h, pieces, positions


REQUIRED_COLUMN_TYPES = {"romanic", "aV", "first_v", "second_v"}
MAX_COVERAGE_ATTEMPTS = 6


def _surviving_coverage(pieces: list[PaperPiece], positions: list[tuple[int, int]], page_w: int, page_h: int) -> set[str]:
    """Which column_types would still be >=MIN_VISIBLE_AREA_FRAC visible
    after clipping to [0,page_w]x[0,page_h] -- same rule
    _clip_columns_to_bounds applies, computed early so a bad placement
    draw can be retried before committing to it (see
    _place_pieces_ensuring_coverage)."""
    types = set()
    for piece, (x, y) in zip(pieces, positions):
        for col in piece.columns:
            bx0, by0, bx1, by1 = col.bbox
            bx0, by0, bx1, by1 = bx0 + x, by0 + y, bx1 + x, by1 + y
            original_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
            cx0, cy0 = max(bx0, 0), max(by0, 0)
            cx1, cy1 = min(bx1, page_w), min(by1, page_h)
            clipped_area = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
            if original_area > 0 and clipped_area / original_area >= MIN_VISIBLE_AREA_FRAC:
                types.add(col.column_type)
    return types


def _place_pieces_ensuring_coverage(rng: random.Random, pieces: list[PaperPiece], harsh: bool) -> tuple[int, int, list[PaperPiece], list[tuple[int, int]]]:
    """_place_pieces, retried up to MAX_COVERAGE_ATTEMPTS times, so the
    page/paper size decoupling (2026-09-16) can still crop realistically
    without regularly producing a page too small to show all 4 lead
    groups (12 leads) -- 3 panels is 9 leads, not a real "12-lead ECG"
    anymore, flagged directly by the user. Falls back to whichever real
    page size actually contains the cluster (smallest such page, still
    independent of any margin/fit preference -- just the deterministic
    best option) if every random attempt fails, rather than looping
    forever or silently shipping an under-covered page. Only the rare
    cluster bigger than every real page size can still lose a group."""
    best = None
    for _ in range(MAX_COVERAGE_ATTEMPTS):
        page_w, page_h, placed, positions = _place_pieces(rng, pieces, harsh=harsh)
        if REQUIRED_COLUMN_TYPES.issubset(_surviving_coverage(placed, positions, page_w, page_h)):
            return page_w, page_h, placed, positions
        best = (page_w, page_h, placed, positions)

    layout_w, layout_h, offsets = _choose_layout(rng, pieces)
    containing = [(w, h) for (w, h) in _REAL_PAGE_SIZES if w >= layout_w and h >= layout_h]
    fallback_sizes = sorted(containing, key=lambda wh: wh[0] * wh[1]) or [max(_REAL_PAGE_SIZES, key=lambda wh: wh[0] * wh[1])]
    for page_w, page_h in fallback_sizes:
        slack_w, slack_h = page_w - layout_w, page_h - layout_h
        pad_left, pad_top = rng.uniform(0, max(0.0, slack_w)), rng.uniform(0, max(0.0, slack_h))
        positions = [(round(pad_left + x), round(pad_top + y)) for x, y in offsets]
        if REQUIRED_COLUMN_TYPES.issubset(_surviving_coverage(pieces, positions, page_w, page_h)):
            return page_w, page_h, pieces, positions
    return best  # even the largest real page can't fit this cluster with full coverage -- best effort


def _clip_columns_to_bounds(page: Page, x0: float, y0: float, x1: float, y1: float) -> Page:
    """Intersects every column's bbox with [x0,y0,x1,y1] -- used after a
    harsh/negative piece placement or a torn edge, both of which can put
    real page content outside the frame. A column left with under
    MIN_VISIBLE_AREA_FRAC of its original (unclipped) area is dropped
    entirely rather than kept as a barely-visible sliver: ground truth
    should never claim a column is "there" when a crop/tear removed most
    of what would identify it."""
    new_pieces = []
    for piece in page.pieces:
        kept = []
        for col in piece.columns:
            bx0, by0, bx1, by1 = col.bbox
            original_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
            cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
            clipped_area = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
            if original_area <= 0 or clipped_area / original_area < MIN_VISIBLE_AREA_FRAC:
                continue
            kept.append(Column(bbox=(cx0, cy0, cx1, cy1), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=col.meta))
        if kept:
            new_pieces.append(PlacedPiece(image=piece.image, position=piece.position, columns=kept))
    return Page(image=page.image, pieces=new_pieces)


def apply_page_tilt(page: Page, angle_deg: float) -> Page:
    """Rotates the whole composed page (photo-level tilt, smaller than a
    single piece's own tilt) and maps every already-placed column's bbox
    through the same transform -- reuses the exact rotation math verified
    for paper_piece.apply_tilt, just applied to the full page instead of
    one piece."""
    old_w, old_h = page.image.size
    rotated = page.image.rotate(angle_deg, expand=True, fillcolor=PAPER_TINT, resample=Image.BICUBIC)
    new_w, new_h = rotated.size

    transform_point, (pred_w, pred_h) = pil_rotate_transform(old_w, old_h, angle_deg)
    if (pred_w, pred_h) != (new_w, new_h):
        raise AssertionError(f"pil_rotate_transform disagrees with PIL: predicted {(pred_w, pred_h)}, PIL produced {(new_w, new_h)}")

    new_pieces = []
    for piece in page.pieces:
        new_columns = []
        for col in piece.columns:
            meta = map_panel_points({**col.meta, "page_tilt_deg": angle_deg, "page_tilt_applied": True}, transform_point)
            new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, transform_point), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))
        new_pieces.append(PlacedPiece(image=piece.image, position=piece.position, columns=new_columns))

    return Page(image=rotated, pieces=new_pieces)


def apply_page_perspective_warp_3d(page: Page, tilt_x_deg: float, tilt_y_deg: float, focal_frac: float = 2.2, fill=PAPER_TINT) -> Page:
    """Page-level twin of paper_piece.apply_perspective_warp_3d (2026-09-21,
    wiki/TODO.md item 7): rotates the WHOLE composed page -- paper, table
    background, staples, shadows, every already-applied augmentation -- as
    one flat plane in 3D and projects it back through a pinhole camera,
    instead of warping the paper alone before it's even placed on the page.
    A real camera's perspective distortion comes from its angle to the
    whole photographed scene, not from the paper flexing independently of
    the table it sits on -- see _project_plane_corners for the actual
    camera math, reused unchanged from the piece-level version.

    Called last in compose_page, after every other page-level step, so the
    warp is the final thing applied, matching how a real photo's
    perspective is a property of the whole shot, not an earlier layer
    other effects get pasted onto."""
    w, h = page.image.size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = _project_plane_corners(w, h, tilt_x_deg, tilt_y_deg, focal_frac)

    min_x, min_y = dst[:, 0].min(), dst[:, 1].min()
    max_x, max_y = dst[:, 0].max(), dst[:, 1].max()
    new_w, new_h = int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))
    dst_shifted = dst - [min_x, min_y]

    matrix = cv2.getPerspectiveTransform(src, dst_shifted)

    arr = np.array(page.image)
    warped_arr = cv2.warpPerspective(arr, matrix, (new_w, new_h), borderValue=fill)
    warped = Image.fromarray(warped_arr)

    def transform_point(x: float, y: float) -> tuple[float, float]:
        vec = matrix @ np.array([x, y, 1.0])
        return (vec[0] / vec[2], vec[1] / vec[2])

    new_pieces = []
    for piece in page.pieces:
        new_columns = []
        for col in piece.columns:
            # Reuses the SAME manifest columns the old per-paper warp wrote
            # (warp3d_tilt_x_deg/_tilt_y_deg/_focal_frac) -- same meaning
            # ("this panel's own 3D warp parameters"), just set at a
            # different stage now, so no new panel-level columns needed.
            meta = map_panel_points({**col.meta, "warp3d_tilt_x_deg": tilt_x_deg, "warp3d_tilt_y_deg": tilt_y_deg, "warp3d_focal_frac": focal_frac}, transform_point)
            new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, transform_point), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))
        new_pieces.append(PlacedPiece(image=piece.image, position=piece.position, columns=new_columns))

    return Page(image=warped, pieces=new_pieces)


def _apply_document_transform_to_page(page: Page, new_image: Image.Image, matrix: np.ndarray) -> Page:
    """Runs every already-placed column bbox through
    apply_camscanner_look's own perspective matrix -- reuses the
    corner-then-axis-align pattern apply_page_tilt already uses for its
    rotation, generalized to a full homography since CamScanner's crop
    isn't just a rotation."""
    def transform_points(pts: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 2)

    def transform_point(x: float, y: float) -> tuple[float, float]:
        return tuple(transform_points(np.array([(x, y)]))[0])

    new_pieces = []
    for piece in page.pieces:
        new_columns = []
        for col in piece.columns:
            meta = map_panel_points({**col.meta, "camscanner_applied": True}, transform_point)
            new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, transform_point),
                                       orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))
        new_pieces.append(PlacedPiece(image=piece.image, position=piece.position, columns=new_columns))
    return Page(image=new_image, pieces=new_pieces)


def compose_page(rng: random.Random, grid_px: float, page_style: str = "standard", column_counts: list[int] | None = None, row_layout: list[int] | None = None,
                  forced_row_labels: list[list[str]] | None = None, uniform_style: str | None = None, uniform_grid_color: bool = False, no_row_tilt: bool = False) -> Page:
    """Composes one page in a single deterministic pass -- no retries.

    Before 2026-09-06 this placed pieces by repeated random rejection
    sampling (pick a random spot, accept if it doesn't overlap anything
    already placed), which failed often enough to need a whole-page retry
    loop: pieces scaled to page size commonly cover ~70-80% of a real
    page's area, and 2-3 rectangles that large almost never land
    non-overlapping by pure chance. Measured before this fix: an average
    of ~7.7 full-page rebuilds per page actually kept (worst seen: 18),
    since every retry re-ran signal loading, grid/signal drawing, and
    both rotations for every piece from scratch. `_place_pieces` now
    packs pieces deterministically (see _choose_layout) so every piece
    always fits by construction -- the row_labels group-coverage
    guarantee from `_sample_page_row_labels` can no longer be silently
    broken by a dropped piece, so this needs one pass, not a search.

    page_style controls which of the two 2026-09-08 augmentation
    directions apply (see PAGE_STYLES's own comment): "ideal" trims
    degradation and piece-to-piece visual variety down to make columns
    hard to tell apart on an otherwise clean page; "torn" adds a torn
    edge and a harsher, sometimes off-frame placement on top of the
    standard degradation. Both route through the same bbox-clipping path
    (_clip_columns_to_bounds) as "standard" (a no-op there, since nothing
    pushes content out of frame in that style), so there's exactly one
    code path answering "is this column still really visible", not one
    per style.

    row_layout (2026-09-16): drives the default random path now -- each
    page is one single piece of paper, its panels arranged in a ragged
    row layout drawn from PAGE_ROW_TEMPLATES (this project's curated list
    of real observed layouts, e.g. [2, 2, 1] for "12/34/5"), not split
    across 2-3 independently-tilted/warped pieces. Pass an explicit list
    to force one specific template instead of a random draw. Templates
    were also chosen to keep the resulting paper's own aspect ratio close
    to a real single photographed page, so _place_pieces_ensuring_coverage
    below can actually find a real page size that contains it -- an
    arbitrary flat 6-in-a-row shape often couldn't.

    column_counts (legacy, rarely needed): forces 2-3 SEPARATE pieces
    (each its own flat row, no row_layout) merged together instead of
    the single-piece template default -- e.g. [3, 3] for 2 pieces of 3
    columns each. Mutually exclusive with row_layout. Kept for a caller
    that specifically wants the old multi-piece composition."""
    if page_style not in PAGE_STYLES:
        raise ValueError(f"page_style must be one of {PAGE_STYLES}, got {page_style!r}")
    ideal = page_style == "ideal"
    torn = page_style == "torn"

    if row_layout is not None and column_counts is not None:
        raise ValueError("pass either column_counts or row_layout, not both")
    if row_layout is not None:
        column_counts = [sum(row_layout)]
    elif column_counts is not None:
        if not (1 <= len(column_counts) <= 3) or not all(1 <= c <= 3 for c in column_counts):
            raise ValueError(f"column_counts must have 1-3 entries, each 1-3, got {column_counts!r}")
    else:
        row_layout = rng.choice(PAGE_ROW_TEMPLATES)
        column_counts = [sum(row_layout)]

    # forced_row_labels (2026-09-23, reference photo scan_0004.jpg, connected
    # non-digitizable report/blank slots): bypasses the random group-coverage
    # sampling entirely -- a caller building a specific fixed archetype
    # supplies its own exact row_labels, including paper_piece.REPORT_SENTINEL/
    # BLANK_SENTINEL for the non-led slots.
    row_labels = forced_row_labels if forced_row_labels is not None else _sample_page_row_labels(rng, sum(column_counts))
    piece_row_labels = _split_row_labels_into_pieces(row_labels, column_counts)
    pieces = _build_paper(rng, grid_px, piece_row_labels, page_style=page_style, row_layout=row_layout,
                           uniform_style=uniform_style, uniform_grid_color=uniform_grid_color, no_row_tilt=no_row_tilt)

    page_w, page_h, pieces, positions = _place_pieces_ensuring_coverage(rng, pieces, harsh=torn)

    page_img, table_texture_path = _make_table_background(rng, page_w, page_h)
    aug: dict = {"table_texture_path": table_texture_path} if table_texture_path else {}
    if not ideal:
        add_handwriting(page_img, rng)  # background only -- pieces paste on top and may partially cover it, matching real taped-over-notes pages
        aug["handwriting"] = True

    placed_pieces: list[PlacedPiece] = []
    for piece, (x, y) in zip(pieces, positions):
        page_img = _add_paper_shadow(page_img, x, y, piece.image.width, piece.image.height)
        page_img.paste(piece.image, (x, y))
        page_columns = [
            Column(
                bbox=(bx0 + x, by0 + y, bx1 + x, by1 + y), orientation_deg=col.orientation_deg, column_type=col.column_type,
                # map_panel_points shifts panel_points AND staple_points by
                # the same paste offset -- this site used to hand-roll just
                # panel_points, silently leaving staple_points at their
                # pre-paste piece-local coordinates (found 2026-09-22, same
                # bug class as the col.bbox/panel_points drift fix: two
                # fields that must agree, only one kept in step).
                meta={**map_panel_points(col.meta, lambda px, py, x=x, y=y: (px + x, py + y)),
                      "page_style": page_style, "page_tilt_deg": 0.0, "page_tilt_applied": False,
                      # where this panel's own paper piece was pasted, which was
                      # stored on PlacedPiece.position and never read by the
                      # writer, so a multi-piece page's layout was unrecoverable
                      "piece_x": x, "piece_y": y},  # tilt fields overwritten below if page-level tilt actually applies
            )
            for col in piece.columns
            for bx0, by0, bx1, by1 in [col.bbox]
        ]
        placed_pieces.append(PlacedPiece(image=piece.image, position=(x, y), columns=page_columns))

    page = Page(image=page_img, pieces=placed_pieces)
    # a harsh "torn" crop, or (2026-09-16) any style's independently-sized
    # page frame simply being smaller than the paper, can hang real content
    # off the fixed page_w x page_h canvas -- clipping it here is correct,
    # not a bug: page and paper are separate concerns now (user call).
    # _place_pieces_ensuring_coverage already retried the placement draw
    # to keep all 4 lead groups (12 leads) visible when it can, so this
    # clip is expected to only ever trim margin/duplicate columns, not a
    # required group -- except the rare cluster too big for even the
    # largest real page, where losing a group is the accepted last resort.
    page = _clip_columns_to_bounds(page, 0, 0, page_w, page_h)

    # Moved here 2026-09-16 (was after page-level tilt, alongside the
    # lighting gradient): a real CamScanner-style app scans the physical
    # paper itself, so its crop+B&W effect belongs before the later
    # steps that simulate photographing the (now scanned-flat) result --
    # pen marks/stickers/hand shadow/page tilt/lighting all still make
    # sense on top of a camscanned page, not the reverse.
    if not ideal and rng.random() < CAMSCANNER_LOOK_PROBABILITY:
        scanned, cam_matrix = apply_camscanner_look(page.image, rng)
        page = _apply_document_transform_to_page(page, scanned, cam_matrix)
        page = _clip_columns_to_bounds(page, 0, 0, page.image.width, page.image.height)
        aug["camscanner"] = True

    if not ideal:
        cluster = _cluster_bbox(page)
        if rng.random() < PEN_SCRIBBLE_PROBABILITY:
            add_pen_scribble(page.image, rng, cluster)
            aug["pen_scribble"] = True
        if rng.random() < STICKER_BLOT_PROBABILITY:
            add_sticker_blot(page.image, rng, cluster)
            aug["sticker_blot"] = True
        if rng.random() < BRUSH_MARK_PROBABILITY:
            add_brush_mark(page.image, rng, cluster)
            aug["brush_mark"] = True
        page_img = add_hand_shadow(page.image, rng)
        aug["hand_shadow"] = True
        if rng.random() < BIG_SHADOW_PROBABILITY:
            page_img = add_big_cast_shadow(page_img, rng)
            aug["big_cast_shadow"] = True
        page = Page(image=page_img, pieces=page.pieces)
    if not ideal and rng.random() < PAGE_TILT_PROBABILITY:
        page = apply_page_tilt(page, rng.uniform(*PAGE_TILT_RANGE_DEG))
        aug["page_tilt"] = True
    if not ideal:
        if rng.random() < LIGHTING_GRADIENT_PROBABILITY:
            lit = apply_lighting_gradient(page.image, rng)
            page = Page(image=lit, pieces=page.pieces)
            aug["lighting_gradient"] = True
        # Applied on every non-ideal page, no probability gate (2026-09-21,
        # wiki/TODO.md): every one of the 6 reference photos measured showed
        # SOME color cast, none neutral=0 -- "neutral" is just a point near
        # the cool end of the same continuum (R-B -9), not a separate
        # off-state, so drawing a random point on the whole range every time
        # matches the evidence better than a sparse on/off augmentation.
        r_b_shift = rng.uniform(*PAGE_TEMPERATURE_R_B_RANGE)
        page = Page(image=apply_page_temperature(page.image, rng, r_b_shift=r_b_shift), pieces=page.pieces)
        aug["page_temperature_r_b"] = round(r_b_shift, 2)

    if torn:
        side = rng.choice(["left", "right", "top", "bottom"])
        depth = torn_edge_max_depth_px(page.image, rng)
        torn_img = apply_torn_edge(page.image, rng, side=side, max_depth=depth)
        w, h = torn_img.size
        safe_bounds = {
            "left": (depth, 0, w, h), "right": (0, 0, w - depth, h),
            "top": (0, depth, w, h), "bottom": (0, 0, w, h - depth),
        }[side]
        page = _clip_columns_to_bounds(Page(image=torn_img, pieces=page.pieces), *safe_bounds)
        aug["torn_edge"] = True
        aug["torn_edge_side"] = side
        aug["torn_edge_depth_px"] = round(float(depth), 2)

    # Last step, deliberately (2026-09-21, wiki/TODO.md item 7): the whole
    # composed page -- paper, table background, staples, shadows, every
    # augmentation above -- gets one 3D perspective warp, matching how a
    # real camera's own angle to the whole scene is the final thing
    # between "flat composed content" and "what the sensor actually
    # captured". Not applied to "ideal" pages, matching that style's
    # existing near-zero-perspective design.
    if not ideal:
        # triangular, not uniform (kept from the pre-2026-09-21 per-paper
        # version, same reasoning: biases toward 0 so both axes landing
        # near the extreme at once, the harshest combination, gets rarer).
        tilt_x = rng.triangular(*PAGE_WARP3D_TILT_RANGE_DEG, 0.0)
        tilt_y = rng.triangular(*PAGE_WARP3D_TILT_RANGE_DEG, 0.0)
        page = apply_page_perspective_warp_3d(page, tilt_x_deg=tilt_x, tilt_y_deg=tilt_y)
        aug["page_warp3d"] = True

    # Read off the page's own final size, not the pre-crop page_w/page_h
    # chosen above: apply_camscanner_look, apply_torn_edge and the 3D warp
    # just above can all resize the canvas after that point (rectify to
    # the paper's quad, cut a torn side down, expand for the warped
    # corners), so an early label can silently go stale. Caught by a
    # direct check after building this feature: 5 of 40 smoke-test pages
    # had a "portrait" label sitting on a canvas that measured wider than
    # tall by the time the page was actually finished.
    aug["page_orientation"] = classify_orientation(page.image.width, page.image.height)
    page.augmentations = aug
    return page
