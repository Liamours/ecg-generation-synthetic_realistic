"""A "paper piece" (2026-09-05): 1-3 panel templates fused edge to edge on
one continuous physical strip (no visible seam other than each column's
own bounding box), then given a small whole-piece tilt (the piece rotates
as one rigid object, not per-column) and a subtle perspective warp. This
is the unit later placed onto a page (see plan.md's step 4).

Ground truth tracked: each column's own axis-aligned bounding box, kept
correct through every transform (never diagonal), plus its orientation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import config, signal_source, variants
from .template import PAPER_TINT, build_panel_template, signal_start_x

ASSUMED_SIGNAL_FS = 500.0  # PTB-XL's _hr variant is always this; verified against what load_panel_signals actually returns, not blindly trusted


def _compute_duration_samples(grid_px: float) -> int:
    panel_width_px = config.PANEL_WIDTH_GRID_BOXES * grid_px
    available_px = panel_width_px - signal_start_x(grid_px)
    px_per_second = grid_px / config.GRID_BOX_SECONDS
    return max(1, round(available_px / px_per_second * ASSUMED_SIGNAL_FS))

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1, axis-aligned

STAPLE_LENGTH_FRAC = 0.12  # of the piece's shorter side
STAPLE_WIDTH_PX = 5
STAPLE_COLOR = (20, 20, 20)
STAPLE_MARGIN_PX = 6


@dataclass
class Column:
    bbox: BBox
    orientation_deg: int  # 0/90/180/270, upright-reading orientation before any tilt
    column_type: str  # "romanic" | "aV" | "first_v" | "second_v"
    meta: dict  # everything decided at panel-build time (style, gain, lead names, ...) -- see build_paper_piece; carried through every transform unchanged


@dataclass
class PaperPiece:
    image: Image.Image
    columns: list[Column]


def _corners(bbox: BBox) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _bbox_of_points(points: list[tuple[float, float]]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def build_paper_piece(
    rng: random.Random,
    grid_px: float,
    num_columns: int | None = None,
    row_labels_list: list[list[str]] | None = None,
    draw_staples: bool = True,
) -> PaperPiece:
    """Fuses 1-3 panel templates side by side with no gap, then draws one
    staple mark on a random side (unless draw_staples=False -- the "ideal"
    page style's own hard-to-separate-into-columns direction: a staple is
    currently the only visual cue marking where one column's paper joins
    the next, since columns already share a border with no whitespace
    gap; without it there is nothing but the printed content itself to
    tell them apart). `row_labels_list` fixes which group each column
    shows, in order (e.g. so a caller can guarantee page-level group
    coverage) -- when omitted, each column's group is sampled
    independently at random, same as before."""
    if row_labels_list is not None:
        num_columns = len(row_labels_list)
    elif num_columns is None:
        num_columns = rng.randint(1, 3)
    if not 1 <= num_columns <= 3:
        raise ValueError(f"num_columns must be 1-3, got {num_columns}")

    duration_samples = _compute_duration_samples(grid_px)

    panels = []
    column_types = []
    ptbxl_record_ids = []
    for i in range(num_columns):
        style_kwargs = variants.sample_style_variant(rng)
        row_labels = row_labels_list[i] if row_labels_list is not None else rng.choice(variants.STANDARD_ROW_LABEL_SETS)
        record_id, lead_signals, fs = signal_source.load_panel_signals(rng, row_labels, duration_samples)
        if fs != ASSUMED_SIGNAL_FS:
            raise RuntimeError(f"{record_id}: expected {ASSUMED_SIGNAL_FS}Hz, got {fs}Hz")
        panel = build_panel_template(
            row_labels=row_labels, grid_px=grid_px, record_ref=f"{rng.randint(0, 9999999):07d}", rng=rng,
            lead_signals=lead_signals, signal_fs=fs, **style_kwargs,
        )
        panels.append(panel)
        column_types.append(variants.column_type_for_labels(row_labels))
        ptbxl_record_ids.append(record_id)

    width = sum(p.image.width for p in panels)
    height = max(p.image.height for p in panels)
    canvas = Image.new("RGB", (width, height), PAPER_TINT)

    columns: list[Column] = []
    x_cursor = 0
    for panel, column_type, ptbxl_record_id in zip(panels, column_types, ptbxl_record_ids):
        canvas.paste(panel.image, (x_cursor, 0))
        meta = {
            "row_labels": panel.row_labels,
            "style": panel.style,
            "gain": panel.gain,
            "calibration_step_mode": panel.calibration_step_mode,
            "has_calibration_step": panel.has_calibration_step,
            "bottom_row_color": panel.bottom_row_color,
            "record_ref": panel.record_ref,
            "ptbxl_record_id": ptbxl_record_id,
            "header_text": panel.header_text,
            "footer_top": [f.top for f in panel.footer_fields],
            "footer_bottom": [f.bottom for f in panel.footer_fields],
            # Canonical, pre-transform ground truth for the rectification/
            # lead-cropping/digitization training targets (2026-09-06) --
            # `panel_image` is the pristine render at its own native
            # resolution, untouched by any later tilt/warp/scale applied
            # to the piece or page it ends up on, so it stays the correct
            # "what rectification should produce" target regardless of
            # where this column lands. `row_y_ranges` locates each lead
            # inside it (same order as row_labels); `lead_signals` is the
            # exact real sample array drawn into each lead's trace.
            "panel_image": panel.image,
            "row_y_ranges": panel.row_y_ranges,
            "lead_signals": panel.lead_signals,
            "signal_fs": panel.signal_fs,
            # Also in panel_image's own pristine local frame, same as
            # row_y_ranges above (2026-09-06). `label_bboxes` matches
            # row_labels order, one box per row's own printed lead-name
            # text. `calibration_step_bbox` is None when
            # has_calibration_step is False.
            "label_bboxes": panel.label_bboxes,
            "calibration_step_bbox": panel.calibration_step_bbox,
        }
        columns.append(Column(
            bbox=(x_cursor, 0, x_cursor + panel.image.width, panel.image.height),
            orientation_deg=0,
            column_type=column_type,
            meta=meta,
        ))
        x_cursor += panel.image.width

    if draw_staples:
        _draw_staples(canvas, rng, columns)

    return PaperPiece(image=canvas, columns=columns)


def _draw_one_staple(draw: ImageDraw.ImageDraw, w: int, h: int, side: str, length: int, center_frac: float | None = None) -> None:
    if side in ("left", "right"):
        x = STAPLE_MARGIN_PX if side == "left" else w - STAPLE_MARGIN_PX
        center_y = center_frac * h if center_frac is not None else None
        y0 = round(center_y - length / 2) if center_y is not None else 0
        y0 = max(0, min(y0, h - length))
        draw.line([(x, y0), (x, y0 + length)], fill=STAPLE_COLOR, width=STAPLE_WIDTH_PX)
    else:
        y = STAPLE_MARGIN_PX if side == "top" else h - STAPLE_MARGIN_PX
        center_x = center_frac * w if center_frac is not None else None
        x0 = round(center_x - length / 2) if center_x is not None else 0
        x0 = max(0, min(x0, w - length))
        draw.line([(x0, y), (x0 + length, y)], fill=STAPLE_COLOR, width=STAPLE_WIDTH_PX)


def _draw_staples(image: Image.Image, rng: random.Random, columns: list[Column]) -> None:
    """One staple per column, not one per whole piece -- a real multi-
    column strip is a physically longer document, plausibly stapled at
    several points along it, not just once (2026-09-05). For a
    multi-column piece, staples go on the top or bottom long edge
    (spread along the growing width); a single-column piece keeps the
    original any-side behavior."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    length = round(min(w, h) * STAPLE_LENGTH_FRAC)

    if len(columns) == 1:
        side = rng.choice(["left", "right", "top", "bottom"])
        _draw_one_staple(draw, w, h, side, length)
        return

    side = rng.choice(["top", "bottom"])
    for col in columns:
        x0, _y0, x1, _y1 = col.bbox
        center_frac = ((x0 + x1) / 2) / w
        center_frac += rng.uniform(-0.15, 0.15) / len(columns)
        _draw_one_staple(draw, w, h, side, length, center_frac=center_frac)


def apply_tilt(piece: PaperPiece, angle_deg: float, fill=PAPER_TINT, orientation_delta: int = 0, tilt_deg: float = 0.0) -> PaperPiece:
    """Rotates the whole piece as one rigid object (PIL, expand=True so
    nothing is cropped), then maps every column's bbox through the same
    transform and re-derives its axis-aligned bounding box from the
    rotated corners (never diagonal, per spec).

    `angle_deg` is the physical rotation applied to the image (can be a
    discrete 90/180/270 facing plus continuous tilt noise combined into
    one call -- two sequential rotate(expand=True) calls about the same
    center compose exactly into one rotation by the angle sum, so doing
    it in one call is both faster, one resample pass instead of two, and
    sharper, per 2026-09-06 perf pass). `orientation_delta` is the
    separate, always-90-multiple amount to add to each column's
    `orientation_deg` LABEL -- kept independent of `angle_deg` so tilt
    noise combined into the same call never leaks into the discrete
    facing label. `tilt_deg` is the continuous part of `angle_deg` alone
    (excluding `orientation_delta`), recorded into each column's meta as
    an explicit rectification regression target -- otherwise this value
    is used once to compute the final bbox and then discarded, leaving no
    record of how a piece got tilted (2026-09-06, closing a gap flagged
    by the experiments session)."""
    old_w, old_h = piece.image.size
    rotated = piece.image.rotate(angle_deg, expand=True, fillcolor=fill, resample=Image.BICUBIC)
    new_w, new_h = rotated.size

    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    old_cx, old_cy = old_w / 2, old_h / 2
    new_cx, new_cy = new_w / 2, new_h / 2

    def transform_point(x: float, y: float) -> tuple[float, float]:
        # PIL's Image.rotate(angle) rotates counterclockwise on-screen for
        # positive angle; verified empirically 2026-09-05 (see
        # tests/test_paper_piece.py) rather than assumed from docs.
        dx, dy = x - old_cx, y - old_cy
        rx = dx * cos_t + dy * sin_t
        ry = -dx * sin_t + dy * cos_t
        return (rx + new_cx, ry + new_cy)

    new_columns = []
    for col in piece.columns:
        corners = [transform_point(x, y) for x, y in _corners(col.bbox)]
        new_orientation = (col.orientation_deg + orientation_delta) % 360
        meta = {**col.meta, "tilt_deg": tilt_deg}
        new_columns.append(Column(bbox=_bbox_of_points(corners), orientation_deg=new_orientation, column_type=col.column_type, meta=meta))

    return PaperPiece(image=rotated, columns=new_columns)


def scale_piece(piece: PaperPiece, factor: float) -> PaperPiece:
    """Resizes the whole piece (e.g. for placing a full-quality-rendered
    piece onto a smaller page) -- scaling down at this stage instead of
    rendering at a smaller grid_px, since font sizes are fixed pixel
    constants and would stop looking right if grid_px itself shrank."""
    w, h = piece.image.size
    resized = piece.image.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.LANCZOS)
    new_columns = [
        Column(bbox=tuple(v * factor for v in col.bbox), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=col.meta)
        for col in piece.columns
    ]
    return PaperPiece(image=resized, columns=new_columns)


def apply_perspective_warp(piece: PaperPiece, rng: random.Random, max_shift_frac: float = 0.025, fill=PAPER_TINT) -> PaperPiece:
    """A very subtle depth cue -- one edge (randomly chosen) pulled
    slightly inward on both its corners, as if that side of the paper
    were tilted a little further from the camera. `max_shift_frac` is a
    fraction of the piece's own width/height, kept small since this
    should read as "not perfectly flat", not an obvious skew.

    Column bboxes are mapped through the same forward homography (not
    approximated), then re-flattened to axis-aligned -- required to stay
    a real bounding box, not a diagonal quad, per spec."""
    w, h = piece.image.size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    side = rng.choice(["left", "right", "top", "bottom"])
    shift_x = w * max_shift_frac
    shift_y = h * max_shift_frac
    dst = src.copy()
    if side == "left":
        dst[0][0] += shift_x
        dst[3][0] += shift_x
    elif side == "right":
        dst[1][0] -= shift_x
        dst[2][0] -= shift_x
    elif side == "top":
        dst[0][1] += shift_y
        dst[1][1] += shift_y
    else:
        dst[2][1] -= shift_y
        dst[3][1] -= shift_y

    matrix = cv2.getPerspectiveTransform(src, dst)  # forward: original coords -> warped coords

    # array is RGB (straight from PIL, not cv2.imread), so `fill` (an RGB
    # tuple) is passed to borderValue as-is -- no BGR conversion needed.
    arr = np.array(piece.image)
    warped_arr = cv2.warpPerspective(arr, matrix, (w, h), borderValue=fill)
    warped = Image.fromarray(warped_arr)

    def transform_point(x: float, y: float) -> tuple[float, float]:
        vec = matrix @ np.array([x, y, 1.0])
        return (vec[0] / vec[2], vec[1] / vec[2])

    new_columns = []
    for col in piece.columns:
        corners = [transform_point(x, y) for x, y in _corners(col.bbox)]
        meta = {**col.meta, "perspective_side": side, "perspective_shift_frac": max_shift_frac}
        new_columns.append(Column(bbox=_bbox_of_points(corners), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))

    return PaperPiece(image=warped, columns=new_columns)
