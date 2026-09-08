"""Page composition (2026-09-05): place 2-3 paper pieces (see
paper_piece.py) onto one background page, sized from `ekg-757`'s own real
image dimensions (not an invented size), each piece independently given a
discrete facing (0/90/180/270) plus its own small tilt and perspective
warp. Ground truth: every column's page-space bbox and orientation.

Total columns per page: 5-6, split across 2-3 pieces (per spec) --
matches this project's earlier-established "4-6 panels per page"
observation, not a new number invented here.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from importlib import resources

from PIL import Image

from . import variants
from .page_augment import add_hand_shadow, add_handwriting, apply_torn_edge, torn_edge_max_depth_px
from .paper_piece import Column, PaperPiece, apply_perspective_warp, apply_tilt, build_paper_piece, scale_piece
from .template import PAPER_TINT

with resources.files("ecg_synthetic_realistic.data").joinpath("page_sizes_ekg757.json").open() as f:
    _REAL_PAGE_SIZES: list[tuple[int, int]] = [tuple(wh) for wh in json.load(f)]

MIN_TOTAL_COLUMNS = 5
MAX_TOTAL_COLUMNS = 6
DISCRETE_ORIENTATIONS = [0, 90, 180, 270]
TILT_RANGE_DEG = (-6.0, 6.0)  # reduced from +-15 (2026-09-05): looked too tilted once combined with discrete orientation on a full page
EDGE_MARGIN_FRAC = 0.03  # keeps pieces off the literal page edge

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


def _sample_column_counts(rng: random.Random, total: int) -> list[int]:
    """Splits `total` columns across 2-3 pieces, each 1-3 columns."""
    num_pieces = rng.choice([2, 3])
    counts = [1] * num_pieces
    remaining = total - num_pieces
    while remaining > 0:
        candidates = [i for i, c in enumerate(counts) if c < 3]
        if not candidates:
            # Can't fit the remainder within the 1-3-per-piece cap; drop
            # to whatever total is already reached rather than violate it.
            break
        i = rng.choice(candidates)
        counts[i] += 1
        remaining -= 1
    return counts


def _sample_page_row_labels(rng: random.Random, total: int) -> list[list[str]]:
    """All 4 standard groups (romanic/aV/first_v/second_v) at least once
    -- a real requirement, not just variety for its own sake -- then
    fills any remaining slots (page has 5-6 columns, only 4 mandatory
    groups) with random duplicates, matching the duplicate-panel pattern
    already observed on real pages. Shuffled so group order isn't fixed."""
    row_labels = list(variants.STANDARD_ROW_LABEL_SETS)
    for _ in range(total - len(row_labels)):
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


def _build_pieces(rng: random.Random, grid_px: float, piece_row_labels: list[list[list[str]]], page_style: str = "standard") -> list[PaperPiece]:
    """Builds every piece: fused columns, then one combined rotation (the
    discrete 0/90/180/270 facing plus the continuous tilt noise, applied
    in a single rotate call -- see apply_tilt's docstring for why summing
    the angle is exactly equivalent to two sequential rotations, just
    faster and with one resample pass instead of two), perspective warp,
    and page-scale.

    "ideal" style shares one discrete orientation across every piece on
    the page (sampled once, not per piece) and shrinks tilt/perspective
    to near zero: pieces placed edge to edge at the same angle blend into
    what reads as one continuous sheet, the "hard to separate into their
    columns" property, and skips staples (see build_paper_piece's own
    docstring for why that's the last remaining boundary cue)."""
    ideal = page_style == "ideal"
    shared_orientation = rng.choice(DISCRETE_ORIENTATIONS) if ideal else None
    tilt_range = IDEAL_TILT_RANGE_DEG if ideal else TILT_RANGE_DEG

    pieces = []
    for row_labels_list in piece_row_labels:
        piece = build_paper_piece(rng, grid_px=grid_px, row_labels_list=row_labels_list, draw_staples=not ideal)

        orientation = shared_orientation if ideal else rng.choice(DISCRETE_ORIENTATIONS)
        tilt_angle = rng.uniform(*tilt_range)
        piece = apply_tilt(piece, angle_deg=orientation + tilt_angle, orientation_delta=orientation, tilt_deg=tilt_angle)
        if ideal:
            piece = apply_perspective_warp(piece, rng, max_shift_frac=IDEAL_PERSPECTIVE_MAX_SHIFT_FRAC)
        else:
            piece = apply_perspective_warp(piece, rng)
        piece = scale_piece(piece, PIECE_SCALE_TO_PAGE)
        pieces.append(piece)
    return pieces


Size = tuple[int, int]
Offset = tuple[float, float]


def _pack_two_pieces(sizes: list[Size]) -> list[tuple[float, float, list[Offset]]]:
    (w0, h0), (w1, h1) = sizes
    return [
        (w0 + w1, max(h0, h1), [(0, 0), (w0, 0)]),  # side by side
        (max(w0, w1), h0 + h1, [(0, 0), (0, h0)]),  # stacked
    ]


def _pack_three_pieces(sizes: list[Size]) -> list[tuple[float, float, list[Offset]]]:
    """Every 2-of-3 "pair" combined with the remaining "lone" piece, pair
    arranged side by side or stacked, lone piece placed on either side of
    that pair -- plus the plain single row and single column of all
    three. Small enough a search space (3 lone choices x 2 pair
    arrangements x up to 4 lone placements, plus 2) to enumerate outright
    rather than run a general bin-packer for just 3 rectangles."""
    candidates: list[tuple[float, float, dict[int, Offset]]] = []
    for lone_idx in range(3):
        a_idx, b_idx = [i for i in range(3) if i != lone_idx]
        (wa, ha), (wb, hb) = sizes[a_idx], sizes[b_idx]
        wl, hl = sizes[lone_idx]
        pair_layouts = [
            (wa + wb, max(ha, hb), {a_idx: (0.0, 0.0), b_idx: (float(wa), 0.0)}),
            (max(wa, wb), ha + hb, {a_idx: (0.0, 0.0), b_idx: (0.0, float(ha))}),
        ]
        for pair_w, pair_h, pair_offsets in pair_layouts:
            candidates.append((max(pair_w, wl), pair_h + hl, {**pair_offsets, lone_idx: (0.0, pair_h)}))
            candidates.append((max(pair_w, wl), pair_h + hl, {**{k: (x, y + hl) for k, (x, y) in pair_offsets.items()}, lone_idx: (0.0, 0.0)}))
            candidates.append((pair_w + wl, max(pair_h, hl), {**pair_offsets, lone_idx: (pair_w, 0.0)}))
            candidates.append((pair_w + wl, max(pair_h, hl), {**{k: (x + wl, y) for k, (x, y) in pair_offsets.items()}, lone_idx: (0.0, 0.0)}))

    (w0, h0), (w1, h1), (w2, h2) = sizes
    candidates.append((w0 + w1 + w2, max(h0, h1, h2), {0: (0.0, 0.0), 1: (float(w0), 0.0), 2: (float(w0 + w1), 0.0)}))
    candidates.append((max(w0, w1, w2), h0 + h1 + h2, {0: (0.0, 0.0), 1: (0.0, float(h0)), 2: (0.0, float(h0 + h1))}))

    return [(w, h, [offsets[i] for i in range(3)]) for w, h, offsets in candidates]


def _choose_layout(rng: random.Random, pieces: list[PaperPiece]) -> tuple[float, float, list[Offset]]:
    """Deterministically packs every piece with no overlap (never drops
    one, unlike the old random-rejection placement it replaced -- see
    2026-09-06 perf notes below), picking among the near-tightest
    packings at random for shape variety."""
    sizes = [p.image.size for p in pieces]
    if len(sizes) == 2:
        candidates = _pack_two_pieces(sizes)
    elif len(sizes) == 3:
        candidates = _pack_three_pieces(sizes)
    else:
        raise ValueError(f"expected 2 or 3 pieces, got {len(sizes)}")

    candidates.sort(key=lambda c: c[0] * c[1])
    best_area = candidates[0][0] * candidates[0][1]
    near_best = [c for c in candidates if c[0] * c[1] <= best_area * (1 + NEAR_BEST_AREA_FRAC)]
    return rng.choice(near_best)


def _pick_near_smallest(rng: random.Random, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    candidates = sorted(candidates, key=lambda wh: wh[0] * wh[1])
    smallest_area = candidates[0][0] * candidates[0][1]
    near_smallest = [c for c in candidates if c[0] * c[1] <= smallest_area * (1 + NEAR_BEST_AREA_FRAC)]
    return rng.choice(near_smallest)


def _choose_page_size(rng: random.Random, layout_w: float, layout_h: float) -> tuple[int, int]:
    """Prefers the tightest-fitting real page for this layout instead of
    just any big-enough one -- a close fit keeps the composed page
    looking like paper pieces filling the sheet (matching real reference
    photos), not a small cluster adrift on a needlessly large blank page.
    Tries a margin-satisfying fit first, then plain containment, and only
    when no real page is big enough at all falls back to whichever page
    needs the least shrinkage (minimizes wasted blank space in that
    unavoidable case, rather than the biggest-area page regardless of
    aspect ratio, which used to leave large stretches of empty page next
    to a scaled-down piece cluster)."""
    margin_scale = 1 - 2 * EDGE_MARGIN_FRAC
    with_margin = [(w, h) for (w, h) in _REAL_PAGE_SIZES if w * margin_scale >= layout_w and h * margin_scale >= layout_h]
    if with_margin:
        return _pick_near_smallest(rng, with_margin)

    contains = [(w, h) for (w, h) in _REAL_PAGE_SIZES if w >= layout_w and h >= layout_h]
    if contains:
        return _pick_near_smallest(rng, contains)

    return max(_REAL_PAGE_SIZES, key=lambda wh: min(wh[0] / layout_w, wh[1] / layout_h))


def _place_pieces(rng: random.Random, pieces: list[PaperPiece], harsh: bool = False) -> tuple[int, int, list[PaperPiece], list[tuple[int, int]]]:
    """Packs `pieces` into one cluster (see _choose_layout), picks a real
    page size big enough to hold it with margin, then randomizes where
    that cluster sits in the remaining slack -- the placement variety the
    old random-rejection sampler gave, without its failure mode of
    sometimes not finding a fit and dropping a piece (which used to cost
    a full page-worth of rebuilt pieces per retry, see 2026-09-06 perf
    notes at the top of compose_page).

    harsh=True (the "torn" page style's "awful croppings, hard to crop
    sections" direction) lets the cluster's slack range go slightly
    negative, so the placed cluster can hang off the page edge by up to
    HARSH_CROP_OVERFLOW_FRAC of its own size -- a real bad photo sometimes
    has the paper's own edge run off-frame, not just centered with room
    to spare. compose_page clips every column bbox to the final page
    bounds afterward, so a piece cut this way never claims content that
    isn't actually in the image."""
    layout_w, layout_h, offsets = _choose_layout(rng, pieces)
    page_w, page_h = _choose_page_size(rng, layout_w, layout_h)

    for piece in pieces:
        for col in piece.columns:
            col.meta["page_scale_factor"] = 1.0  # overwritten below only in the rare shrink-to-fit fallback

    if layout_w > page_w or layout_h > page_h:
        # Only the largest real page was available and it's still too
        # small for this layout -- shrink every piece just enough to
        # fit, rather than drop one and lose the group-coverage guarantee
        # already baked into piece_row_labels. Unlike PIECE_SCALE_TO_PAGE
        # (a fixed constant applied to every piece, already implied by
        # panel_image's own native resolution vs. the page-space bbox
        # size), this extra factor varies per page and was previously
        # applied then discarded with no record -- recorded here so the
        # page-space bbox size can still be related back to panel_image's
        # native resolution when it fires.
        scale = min(page_w / layout_w, page_h / layout_h)
        pieces = [scale_piece(p, scale) for p in pieces]
        offsets = [(x * scale, y * scale) for x, y in offsets]
        layout_w, layout_h = layout_w * scale, layout_h * scale
        for piece in pieces:
            for col in piece.columns:
                col.meta["page_scale_factor"] = scale

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


def _page_column_types(page: Page) -> set[str]:
    return {col.column_type for piece in page.pieces for col in piece.columns}


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

    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    old_cx, old_cy = old_w / 2, old_h / 2
    new_cx, new_cy = new_w / 2, new_h / 2

    def transform_point(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - old_cx, y - old_cy
        rx = dx * cos_t + dy * sin_t
        ry = -dx * sin_t + dy * cos_t
        return (rx + new_cx, ry + new_cy)

    new_pieces = []
    for piece in page.pieces:
        new_columns = []
        for col in piece.columns:
            x0, y0, x1, y1 = col.bbox
            corners = [transform_point(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            meta = {**col.meta, "page_tilt_deg": angle_deg, "page_tilt_applied": True}
            new_columns.append(Column(bbox=(min(xs), min(ys), max(xs), max(ys)), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))
        new_pieces.append(PlacedPiece(image=piece.image, position=piece.position, columns=new_columns))

    return Page(image=rotated, pieces=new_pieces)


def compose_page(rng: random.Random, grid_px: float, page_style: str = "standard") -> Page:
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
    per style."""
    if page_style not in PAGE_STYLES:
        raise ValueError(f"page_style must be one of {PAGE_STYLES}, got {page_style!r}")
    ideal = page_style == "ideal"
    torn = page_style == "torn"

    total_columns = rng.choice([MIN_TOTAL_COLUMNS, MAX_TOTAL_COLUMNS])
    column_counts = _sample_column_counts(rng, total_columns)
    row_labels = _sample_page_row_labels(rng, sum(column_counts))
    piece_row_labels = _split_row_labels_into_pieces(row_labels, column_counts)
    pieces = _build_pieces(rng, grid_px, piece_row_labels, page_style=page_style)

    page_w, page_h, pieces, positions = _place_pieces(rng, pieces, harsh=torn)

    page_img = Image.new("RGB", (page_w, page_h), PAPER_TINT)
    if not ideal:
        add_handwriting(page_img, rng)  # background only -- pieces paste on top and may partially cover it, matching real taped-over-notes pages

    placed_pieces: list[PlacedPiece] = []
    for piece, (x, y) in zip(pieces, positions):
        page_img.paste(piece.image, (x, y))
        page_columns = [
            Column(
                bbox=(bx0 + x, by0 + y, bx1 + x, by1 + y), orientation_deg=col.orientation_deg, column_type=col.column_type,
                meta={**col.meta, "page_style": page_style, "page_tilt_deg": 0.0, "page_tilt_applied": False},  # tilt fields overwritten below if page-level tilt actually applies
            )
            for col in piece.columns
            for bx0, by0, bx1, by1 in [col.bbox]
        ]
        placed_pieces.append(PlacedPiece(image=piece.image, position=(x, y), columns=page_columns))

    page = Page(image=page_img, pieces=placed_pieces)
    page = _clip_columns_to_bounds(page, 0, 0, page_w, page_h)  # a harsh placement can hang pieces off the fixed page_w x page_h canvas

    coverage = _page_column_types(page)
    if not REQUIRED_COLUMN_TYPES.issubset(coverage):
        # Only "torn" can legitimately lose a group (a harsh crop or the
        # tear itself can remove enough of a column that _clip_columns_to_bounds
        # correctly drops it) -- "standard"/"ideal" never push content out
        # of frame, so losing coverage there means the deterministic
        # packing broke, which is the real bug this assert exists to catch.
        if not torn:
            raise AssertionError(f"lost group coverage, missing {REQUIRED_COLUMN_TYPES - coverage} -- deterministic packing should make this impossible outside 'torn'")

    if not ideal:
        page_img = add_hand_shadow(page.image, rng)
        page = Page(image=page_img, pieces=page.pieces)
    if not ideal and rng.random() < PAGE_TILT_PROBABILITY:
        page = apply_page_tilt(page, rng.uniform(*PAGE_TILT_RANGE_DEG))

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

    return page
