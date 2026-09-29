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
from PIL import Image, ImageDraw, ImageFilter

from . import config, signal_source, variants
from .grid import GRID_COLOR_WEIGHTS
from .template import PAPER_BEND_GRID_SHAPE, PAPER_TINT, build_panel_template, build_report_panel_template, build_blank_panel_template, signal_start_x

# Sentinels for build_paper_piece's row_labels_list: a real lead group is
# always a 3-name list, so a single-name sentinel is unambiguous. Added
# 2026-09-23 (reference photo scan_0004.jpg) for the non-digitizable page
# elements a real GE printout carries alongside its lead panels -- see
# template.build_report_panel_template's own docstring for the contract.
REPORT_SENTINEL = ["REPORT"]
BLANK_SENTINEL = ["BLANK"]

ASSUMED_SIGNAL_FS = 500.0  # PTB-XL's _hr variant is always this; verified against what load_panel_signals actually returns, not blindly trusted


def _compute_duration_samples(grid_px: float) -> int:
    panel_width_px = config.PANEL_WIDTH_GRID_BOXES * grid_px
    available_px = panel_width_px - signal_start_x(grid_px)
    px_per_second = grid_px / config.GRID_BOX_SECONDS
    return max(1, round(available_px / px_per_second * ASSUMED_SIGNAL_FS))

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1, axis-aligned

ROW_PANEL_TILT_RANGE_DEG = (0.4, 2.2)  # subtle per-panel micro-tilt magnitude within a row -- real paper isn't perfectly flat even when taped together at once

# Odd-even pairing, added 2026-09-21 (wiki/TODO.md item 2). Only ever
# considered for a column that's already a duplicate of its row-adjacent
# neighbor's group (row_labels_list assigns each of the 4 standard groups
# once, then fills any remaining slots with random duplicates -- pairing
# only touches those extra duplicate slots, never one of the 4 mandatory
# ones, so the page's group-coverage guarantee is untouched). When it
# fires, the second column becomes "even" and inherits the first's own
# gain (an "even" panel prints no gain field of its own, but the real
# pipeline's neighbor-gain fill already assumes an adjacent same-page
# panel is a legitimate scale source -- this makes that assumption true
# by construction for a deliberately paired column, not just a coincidence).
PAIRED_ODD_EVEN_PROBABILITY = 0.35
STAPLE_LENGTH_FRAC = 0.14  # of the row's own height
STAPLE_ANGLE_RANGE_DEG = (25.0, 65.0)  # diagonal, not axis-aligned -- a hand-held stapler doesn't land perfectly straight
STAPLE_MARGIN_FRAC = 0.2  # keep the staple's center off the row's own top/bottom edge

# Signal-level real degradations (2026-09-08, documented in
# wiki/research/internal_dataset_characteristics.md): a dead lead prints a
# flat line (simulated lead dropout -- before this, every lead always
# carried real signal), and a motion-artifact burst is the "patient moved
# during screening" corruption the real data shows. Applied to the signal
# dict BEFORE rendering and BEFORE it becomes panel.lead_signals ground
# truth, so image and digitized .npy stay consistent by construction.
DEAD_LEAD_PROBABILITY = 0.05  # per panel: one lead of three prints flat
MOTION_BURST_PROBABILITY = 0.16  # per panel: one lead carries a movement burst. Raised from 0.08 2026-09-21, user call: motion burst should be more common
MOTION_BURST_SPAN_FRAC = (0.10, 0.30)  # fraction of the recording the burst lasts
MOTION_BURST_AMPLITUDE_STD_MULT = (2.0, 5.0)  # burst wander amplitude, in multiples of the lead's own std
MOTION_BURST_NOISE_STD_MULT = (0.5, 1.5)
# New 2026-09-21, user call: a second motion variant where the patient's
# movement disturbs every lead at once (all 3 rows share the same burst
# window), not just one row over a partial span -- a real recording shows
# this when the patient moves during the whole panel's acquisition, not
# mid-lead. Same span/amplitude ranges as the single-lead case, but the
# start/span/wander frequency are shared across leads (one physical event),
# while each lead keeps its own independent noise draw (real per-electrode
# contact noise isn't identical across leads even during one shared event).
MOTION_BURST_WHOLE_PANEL_PROBABILITY = 0.06


def _burst_waveform(rng: random.Random, span: int, wander_freq: float, phase: float, amplitude: float, noise_std: float) -> np.ndarray:
    """One ramped-sine-plus-noise motion burst, shared by the single-lead
    and whole-panel variants below so the ramp/ noise shape stays identical
    between them -- only how many leads it's applied to, and whether the
    timing is drawn once or per lead, differs."""
    t = np.arange(span, dtype=np.float64) / ASSUMED_SIGNAL_FS
    burst = amplitude * np.sin(2 * np.pi * wander_freq * t + phase)
    ramp = np.minimum(np.arange(span) / (span * 0.2), (span - np.arange(span)) / (span * 0.2))
    ramp = np.clip(ramp, 0.0, 1.0)  # cosine-free linear ramp: 0 at both burst edges, 1 in the middle
    burst *= ramp
    burst += np.random.default_rng(rng.randrange(2**32)).normal(0.0, noise_std, size=span)
    return burst


def _apply_signal_degradation(rng: random.Random, lead_signals: dict[str, np.ndarray]) -> tuple[str, dict]:
    """Flattens one lead (dead lead), injects a movement burst into one
    lead, or injects a shared movement burst into every lead at once, in
    place. Returns (manifest note, parameters), the note being '' when none
    fired -- at most one degradation per panel, mirroring how a real page
    shows one thing wrong at a time, not a zoo.

    The numeric parameters were previously discarded, leaving only the note
    string, so "this lead was flattened to its own median" and "this lead got
    a 2.4Hz burst from sample 180 at 0.3mV" were indistinguishable in the
    manifest (2026-09-18). This is the one augmentation that changes the
    training TARGET rather than only the input, so the numbers matter."""
    if rng.random() < DEAD_LEAD_PROBABILITY:
        name = rng.choice(sorted(lead_signals))
        signal = lead_signals[name]
        flat = float(np.median(signal))
        lead_signals[name] = np.full_like(signal, flat)
        return f"dead:{name}", {"dead_lead_name": name, "dead_lead_flat_mv": round(flat, 6)}
    if rng.random() < MOTION_BURST_PROBABILITY:
        name = rng.choice(sorted(lead_signals))
        signal = lead_signals[name]
        n = len(signal)
        span = int(n * rng.uniform(*MOTION_BURST_SPAN_FRAC))
        start = rng.randint(0, n - span)
        wander_freq = rng.uniform(1.0, 3.0)
        amplitude = float(np.std(signal)) * rng.uniform(*MOTION_BURST_AMPLITUDE_STD_MULT)
        noise_std = max(1e-6, float(np.std(signal)) * rng.uniform(*MOTION_BURST_NOISE_STD_MULT))
        burst = _burst_waveform(rng, span, wander_freq, rng.uniform(0, 2 * np.pi), amplitude, noise_std)
        degraded = signal.copy()
        degraded[start:start + span] += burst.astype(signal.dtype)
        lead_signals[name] = degraded
        return f"motion:{name}", {
            "motion_lead_name": name, "motion_start_sample": int(start), "motion_span_samples": int(span),
            "motion_wander_freq_hz": round(float(wander_freq), 4),
            "motion_amplitude_mv": round(float(amplitude), 6),
            "motion_noise_std_mv": round(float(noise_std), 6),
        }
    if rng.random() < MOTION_BURST_WHOLE_PANEL_PROBABILITY:
        names = sorted(lead_signals)
        n = min(len(sig) for sig in lead_signals.values())
        span = int(n * rng.uniform(*MOTION_BURST_SPAN_FRAC))
        start = rng.randint(0, n - span)
        wander_freq = rng.uniform(1.0, 3.0)
        phase = rng.uniform(0, 2 * np.pi)  # one shared timing/frequency/phase -- one physical movement event
        per_lead: dict[str, dict] = {}
        for name in names:
            signal = lead_signals[name]
            amplitude = float(np.std(signal)) * rng.uniform(*MOTION_BURST_AMPLITUDE_STD_MULT)
            noise_std = max(1e-6, float(np.std(signal)) * rng.uniform(*MOTION_BURST_NOISE_STD_MULT))  # own noise draw per lead -- real per-electrode contact noise isn't shared
            burst = _burst_waveform(rng, span, wander_freq, phase, amplitude, noise_std)
            degraded = signal.copy()
            degraded[start:start + span] += burst.astype(signal.dtype)
            lead_signals[name] = degraded
            per_lead[name] = {"amplitude_mv": round(float(amplitude), 6), "noise_std_mv": round(float(noise_std), 6)}
        return "motion_panel:" + ",".join(names), {
            "motion_panel_lead_names": names, "motion_panel_start_sample": int(start), "motion_panel_span_samples": int(span),
            "motion_panel_wander_freq_hz": round(float(wander_freq), 4), "motion_panel_per_lead": per_lead,
        }
    return "", {}


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


def _bent_panel_corners(panel) -> list[tuple[float, float]] | None:
    """The panel's own 4 corners after add_panel_bend's TPS displacement, in
    the same (TL, TR, BR, BL) order as _corners(), read straight off
    paper_bend_control_points -- add_panel_bend's and PAPER_BEND_GRID_SHAPE's
    own comments claim the corners are anchored at zero displacement, but
    that only holds when the panel is square or the random fold axis is
    exactly axis-aligned; for the actual wide ECG panels at other fold
    angles the corners move by real pixels (confirmed 2026-09-24, up to
    ~15px). panel_points must start from where the bent canvas's edges
    actually are, not the nominal rectangle, or every downstream homography
    (rectification, page-space placement) is silently off. None when no
    bend fired, so the caller falls back to the nominal rectangle."""
    if not panel.paper_bend_applied or not panel.paper_bend_control_points:
        return None
    rows, cols = PAPER_BEND_GRID_SHAPE
    row_len = cols + 2
    cp = panel.paper_bend_control_points
    tl, tr, bl, br = cp[0], cp[row_len - 1], cp[-row_len], cp[-1]
    return [(tl[2], tl[3]), (tr[2], tr[3]), (br[2], br[3]), (bl[2], bl[3])]


def pil_rotate_transform(old_w: float, old_h: float, angle_deg: float):
    """Returns (point_fn, (new_w, new_h)) reproducing PIL's
    Image.rotate(angle, expand=True) exactly: rotate about (w/2, h/2), then
    translate so the rotated content's own bounding box starts at 0 using
    floor() of its minimum corner.

    The centre-only form this module used before 2026-09-18 is off by up to a
    pixel from what PIL actually does, because PIL's final translation is
    floor-based rather than centre-based. That error was invisible inside an
    axis-aligned bbox (it just shifted the AABB slightly) but is not invisible
    in a recorded corner: inverting a generated page with the old form showed a
    consistent ~+0.8px x offset. Both consumers now use this one function, so
    the bbox and the recorded corners agree by construction."""
    theta = math.radians(angle_deg % 360.0)
    cos_t, sin_t = round(math.cos(theta), 15), round(math.sin(theta), 15)
    cx, cy = old_w / 2, old_h / 2

    def about_centre(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cos_t * dx + sin_t * dy + cx, -sin_t * dx + cos_t * dy + cy)

    corners = [about_centre(x, y) for x, y in ((0, 0), (old_w, 0), (old_w, old_h), (0, old_h))]
    shift_x = math.floor(min(c[0] for c in corners))
    shift_y = math.floor(min(c[1] for c in corners))

    def point_fn(x: float, y: float) -> tuple[float, float]:
        rx, ry = about_centre(x, y)
        return (rx - shift_x, ry - shift_y)

    return point_fn, (math.ceil(max(c[0] for c in corners)) - shift_x,
                      math.ceil(max(c[1] for c in corners)) - shift_y)


def map_panel_points(meta: dict, transform) -> dict:
    """Carries the panel's own four pristine-local corner points, and any
    staple marks touching this column, through the same transform the
    column's bbox just went through. bbox is re-flattened to an
    axis-aligned box at every step, which discards rotation and shear; the
    four points do not, so four correspondences recover the exact homography
    from the panel's pristine local frame to wherever it ended up. Added
    2026-09-18 (wiki/TODO.md): without it, reconstructing a panel's geometry
    after generation is an approximation, which is what forced
    build_digitization_training_set.py into ECC alignment to paper over the
    error instead of inverting exactly.

    staple_points (added 2026-09-21, wiki/TODO.md item 1) is a list of
    4-point corner lists, one per staple mark touching this column (0-2: a
    column touches one staple per seam it sits next to) -- transformed the
    same way and for the same reason, so a staple's bbox can be recovered
    in whatever frame the panel ends up in, not just the piece-local frame
    it was drawn in."""
    out = meta
    points = meta.get("panel_points")
    if points:
        out = {**out, "panel_points": [transform(px, py) for px, py in points]}
    staples = meta.get("staple_points")
    if staples:
        out = {**out, "staple_points": [[transform(px, py) for px, py in staple] for staple in staples]}
    return out


def _bbox_from_meta(meta: dict, old_bbox: BBox, transform) -> BBox:
    """The new axis-aligned bbox for a column that just went through
    `transform`, taken from the panel's own just-updated `panel_points`
    (call `map_panel_points` on `meta` first) -- NOT by independently
    re-transforming `old_bbox`'s own corners, which was a real bug found
    2026-09-22: `old_bbox` is itself only an axis-aligned re-flattening
    from the PREVIOUS step, so once any non-90-degree rotation has
    happened, its 4 corners are different points than the true panel
    corners. Warping those independently let `bbox` and `panel_points`
    drift apart at every subsequent transform -- 20-130px apart by the
    final page in the case that exposed it (a staple's tracked position
    landing over 100px from where it's actually drawn). Falls back to the
    old (buggy but non-crashing) method only if `panel_points` is somehow
    missing, which shouldn't happen given `_column_meta` always sets it."""
    points = meta.get("panel_points")
    if points:
        return _bbox_of_points(points)
    return _bbox_of_points([transform(x, y) for x, y in _corners(old_bbox)])


def _add_panel_shadow(canvas: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Soft drop shadow for one panel resting on the paper -- same
    offset+blur technique as page.py's paper-on-table shadow, one level
    down in the panel -> paper -> table hierarchy (2026-09-16), so a
    panel reads as a distinct object laid onto the paper rather than a
    flat printed region with a mechanically uniform gap band."""
    offset = max(3, round(min(w, h) * 0.012))
    blur_radius = max(3, offset)
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow_layer).rectangle([x + offset, y + offset, x + w + offset, y + h + offset], fill=(0, 0, 0, 80))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.alpha_composite(canvas.convert("RGBA"), shadow_layer).convert("RGB")


def build_paper_piece(
    rng: random.Random,
    grid_px: float,
    num_columns: int | None = None,
    row_labels_list: list[list[str]] | None = None,
    draw_staples: bool = True,
    row_layout: list[int] | None = None,
    uniform_style: str | None = None,
    uniform_grid_color: bool = False,
    no_row_tilt: bool = False,
) -> PaperPiece:
    """Fuses 1-3 panel templates side by side with no gap (or a
    `row_layout` -- a ragged list of per-row panel counts, 2026-09-16,
    e.g. [2, 2, 1] lays 5 panels out as "12/34/5", [1, 2, 2] as "1/23/45"
    -- matching this project's curated real-template list instead of one
    flat row), then draws one staple mark on a random side (unless
    draw_staples=False -- the "ideal" page style's own
    hard-to-separate-into-columns direction: a staple is currently the
    only visual cue marking where one column's paper joins the next,
    since columns already share a border with no whitespace gap; without
    it there is nothing but the printed content itself to tell them
    apart). `row_labels_list` fixes which group each column shows, in
    order (e.g. so a caller can guarantee page-level group coverage) --
    when omitted, each column's group is sampled independently at
    random, same as before.

    uniform_style/uniform_grid_color/no_row_tilt (2026-09-23, reference
    photo scan_0004.jpg, user call): the "connected panels" case, where
    several panels are printed as ONE single continuous physical sheet
    (a real GE 12SL report page), not independently-sourced strips glued
    together -- every panel on it shares one printed style, one grid
    color/look, and there is no independent per-panel micro-tilt, because
    it is not independently placed paper, it is one piece. uniform_style
    forces every real panel to that style instead of sampling per panel;
    uniform_grid_color samples ONE grid color for the whole piece (leads,
    report, blank alike) instead of per panel; no_row_tilt zeroes the
    per-row micro-tilt this function otherwise always applies."""
    if row_labels_list is not None:
        num_columns = len(row_labels_list)
    elif num_columns is None:
        num_columns = rng.randint(1, 3)
    if row_layout is not None:
        if sum(row_layout) != num_columns:
            raise ValueError(f"row_layout {row_layout} doesn't sum to num_columns {num_columns}")
    elif not 1 <= num_columns <= 3:
        raise ValueError(f"num_columns must be 1-3, got {num_columns}")

    shared_grid_color = (rng.choices(list(GRID_COLOR_WEIGHTS), weights=list(GRID_COLOR_WEIGHTS.values()))[0]
                          if uniform_grid_color else None)
    duration_samples = _compute_duration_samples(grid_px)
    # Which indices start a new row -- pairing may never cross a row
    # boundary (physical panels only ever join horizontally within one
    # row, same rule _draw_row_staples already follows). No row_layout
    # means the whole piece is one row, so only index 0 starts one.
    row_starts = {0}
    if row_layout is not None:
        idx = 0
        for count in row_layout:
            row_starts.add(idx)
            idx += count

    panels = []
    column_types = []
    ptbxl_record_ids = []
    signal_degradations = []
    pair_partner_index: list[int | None] = []
    for i in range(num_columns):
        row_labels = row_labels_list[i] if row_labels_list is not None else rng.choice(variants.STANDARD_ROW_LABEL_SETS)
        pair_partner_index.append(None)
        # Non-digitizable slots (2026-09-23): resolved before any of the
        # real-panel machinery below (pairing, signal loading, column_type)
        # runs, since none of it applies -- these carry no leads at all.
        if row_labels in (REPORT_SENTINEL, BLANK_SENTINEL):
            panel = build_report_panel_template(grid_px=grid_px, rng=rng, grid_color=shared_grid_color) if row_labels == REPORT_SENTINEL \
                else build_blank_panel_template(grid_px=grid_px, rng=rng, grid_color=shared_grid_color)
            panels.append(panel)
            column_types.append("report" if row_labels == REPORT_SENTINEL else "blank")
            ptbxl_record_ids.append(None)
            signal_degradations.append((None, {}))
            continue
        paired_with: int | None = None
        gain_override: str | None = None
        if uniform_style is not None:
            style_kwargs = ({"style": "odd", "calibration_step_mode": "correlated"} if uniform_style == "odd"
                             else {"style": "even", "calibration_step_mode": rng.choice(variants.CALIBRATION_STEP_MODES_FOR_EVEN)})
        elif (i not in row_starts and row_labels_list is not None and row_labels_list[i] == row_labels_list[i - 1]
                and panels[i - 1].style == "odd" and rng.random() < PAIRED_ODD_EVEN_PROBABILITY):
            style_kwargs = {"style": "even", "calibration_step_mode": rng.choice(variants.CALIBRATION_STEP_MODES_FOR_EVEN)}
            gain_override = panels[i - 1].gain
            paired_with = i - 1
            pair_partner_index[i - 1] = i
        else:
            style_kwargs = variants.sample_style_variant(rng)
        record_id, lead_signals, fs = signal_source.load_panel_signals(rng, row_labels, duration_samples)
        if fs != ASSUMED_SIGNAL_FS:
            raise RuntimeError(f"{record_id}: expected {ASSUMED_SIGNAL_FS}Hz, got {fs}Hz")
        degradation, degradation_params = _apply_signal_degradation(rng, lead_signals)
        signal_degradations.append((degradation, degradation_params))
        panel = build_panel_template(
            row_labels=row_labels, grid_px=grid_px, record_ref=f"{rng.randint(0, 9999999):07d}", rng=rng,
            lead_signals=lead_signals, signal_fs=fs, gain=gain_override, grid_color=shared_grid_color, **style_kwargs,
        )
        panels.append(panel)
        column_types.append(variants.column_type_for_labels(row_labels))
        ptbxl_record_ids.append(record_id)
        pair_partner_index[i] = paired_with
        pair_partner_index.append(paired_with)

    def _column_meta(panel, ptbxl_record_id, degradation, paired_column_index=None) -> dict:
        degradation_note, degradation_kv = degradation
        return {
            "row_labels": panel.row_labels,
            "style": panel.style,
            "gain": panel.gain,
            # Odd-even pairing, added 2026-09-21 (wiki/TODO.md item 2): the
            # column_index (within this piece) of this panel's paired
            # partner, in either direction (the odd panel points at its
            # even duplicate and vice versa), or None when this column
            # isn't part of a pair. Resolved to a page-space panel_id at
            # the manifest-writing stage in generate_dataset.py, since
            # column indices alone aren't unique across a whole page.
            "odd_even_pair_index": paired_column_index,
            "calibration_step_mode": panel.calibration_step_mode,
            "has_calibration_step": panel.has_calibration_step,
            "bottom_row_color": panel.bottom_row_color,
            "grid_color": panel.grid_color,
            "calibration_step_shape": panel.calibration_step_shape,
            "panel_width_boxes": panel.panel_width_boxes,
            "panel_noise_applied": panel.panel_noise_applied,
            "panel_wrinkle_applied": panel.panel_wrinkle_applied,
            # Panel-LOCAL non-homography bend (2026-09-22, wiki/TODO.md
            # rectification-model entry), recorded before this panel is
            # placed on the page -- every subsequent transform (rotation,
            # tilt, the page-level warps, CamScanner) is already fully
            # recoverable from panel_pt0..pt3 below, so a consumer wanting
            # the full dense ground truth composes this panel-local TPS
            # with that homography rather than needing it re-threaded
            # through every later stage.
            "paper_bend_applied": panel.paper_bend_applied,
            "paper_bend_control_points": panel.paper_bend_control_points,
            "record_ref": panel.record_ref,
            "ptbxl_record_id": ptbxl_record_id,
            "header_text": panel.header_text,
            "footer_top": [f.top for f in panel.footer_fields],
            "footer_bottom": [f.bottom for f in panel.footer_fields],
            "signal_degradation": degradation_note, **degradation_kv,
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
            "panel_trace_mask": panel.trace_mask,
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
            # computed by template.py since the beginning and never written
            # anywhere until 2026-09-18; the header device block's own box
            "header_icon_xyxy": panel.header_icon_xyxy,
            "text_fields": panel.text_fields,
            # Exact drawn geometry (2026-09-18), so a consumer can build a target
            # covering the whole row rather than only the trace: the trace starts
            # at trace_x0, and everything before it is either the calibration step
            # (step_row_index's row) or bare paper. See template.PanelTemplate's
            # own note on why this closes TODO gap #3.
            "trace_x0": panel.trace_x0,
            "px_per_sample": panel.px_per_sample,
            "px_per_mv": panel.px_per_mv,
            "baseline_ys": panel.baseline_ys,
            "step_row_index": panel.step_row_index,
            "step_geometry": panel.step_geometry,
        }

    columns: list[Column] = []
    if row_layout is not None:
        rows: list[list] = []
        idx = 0
        for count in row_layout:
            rows.append(panels[idx:idx + count])
            idx += count

        # Per-panel micro-tilt (2026-09-16, user call): panels in the same
        # row share a tilt DIRECTION (as if handled/taped down together
        # at once) but each gets its own MAGNITUDE -- real paper isn't
        # perfectly flat even within one row. Rows are separate physical
        # objects (never joined to the row above/below), so each row
        # draws its own independent direction. `panel.image` in meta
        # stays the untilted pristine render (unchanged, same as every
        # other transform in this module) -- only the canvas paste uses
        # the rotated copy.
        rotated_rows: list[list] = []
        row_panel_tilts: list[list[float]] = []
        for row in rows:
            if no_row_tilt:
                angles = [0.0 for _ in row]
            else:
                sign = rng.choice([-1, 1])
                angles = [sign * rng.uniform(*ROW_PANEL_TILT_RANGE_DEG) for _ in row]
            rotated_rows.append([p.image.rotate(a, expand=True, fillcolor=PAPER_TINT, resample=Image.BICUBIC) for p, a in zip(row, angles)])
            row_panel_tilts.append(angles)

        row_heights = [max(img.height for img in row) for row in rotated_rows]
        row_widths = [sum(img.width for img in row) for row in rotated_rows]
        # Row gap (2026-09-16, user call): rows get real paper-colored
        # separation, columns within a row stay touching (no gap) -- one
        # grid box worth, so it reads as paper margin, not an invented unit.
        row_gap_px = round(grid_px)
        canvas_w = max(row_widths)
        canvas_h = sum(row_heights) + row_gap_px * (len(rows) - 1)
        canvas = Image.new("RGB", (canvas_w, canvas_h), PAPER_TINT)

        idx = 0
        y_cursor = 0
        for r, (row, rotated_row, tilts) in enumerate(zip(rows, rotated_rows, row_panel_tilts)):
            x_cursor = 0
            for panel, rimg, tilt in zip(row, rotated_row, tilts):
                column_type, ptbxl_record_id, degradation = column_types[idx], ptbxl_record_ids[idx], signal_degradations[idx]
                # Per-panel drop shadow (2026-09-16, user call: simulate
                # each panel as its own object laid onto the paper, the
                # paper laid onto the table -- same offset+blur technique
                # as page.py's paper-on-table shadow, one level down, so
                # it fades naturally instead of reading as a flat printed
                # band. Only visible where paper actually shows through
                # (the row gap above/below, or a shorter row's own bare
                # margin); touching columns hide it along their shared
                # edge, same as a real glued strip would.
                canvas = _add_panel_shadow(canvas, x_cursor, y_cursor, rimg.width, rimg.height)
                canvas.paste(rimg, (x_cursor, y_cursor))
                meta = _column_meta(panel, ptbxl_record_id, degradation, pair_partner_index[idx])
                # Found 2026-09-17: this per-panel micro-tilt was applied to
                # rimg above but never recorded anywhere, making every
                # downstream bbox/transform-label-based reconstruction of
                # this panel (e.g. reverse-labeling it back to a rectified
                # image) silently off by this amount -- recorded now the
                # same way apply_tilt already records its own tilt_deg.
                meta["row_panel_tilt_deg"] = tilt
                _rot_fn, _ = pil_rotate_transform(panel.image.width, panel.image.height, tilt)
                base_corners = _bent_panel_corners(panel) or _corners((0, 0, panel.image.width, panel.image.height))
                meta["panel_points"] = [
                    (rx + x_cursor, ry + y_cursor)
                    for rx, ry in (_rot_fn(px, py) for px, py in base_corners)
                ]
                columns.append(Column(
                    bbox=(x_cursor, y_cursor, x_cursor + rimg.width, y_cursor + rimg.height),
                    orientation_deg=0, column_type=column_type, meta=meta,
                ))
                x_cursor += rimg.width
                idx += 1
            y_cursor += row_heights[r] + row_gap_px
    else:
        width = sum(p.image.width for p in panels)
        height = max(p.image.height for p in panels)
        canvas = Image.new("RGB", (width, height), PAPER_TINT)

        x_cursor = 0
        for panel, column_type, ptbxl_record_id, degradation, paired_with in zip(panels, column_types, ptbxl_record_ids, signal_degradations, pair_partner_index):
            canvas.paste(panel.image, (x_cursor, 0))
            meta = _column_meta(panel, ptbxl_record_id, degradation, paired_with)
            meta["row_panel_tilt_deg"] = 0.0  # this branch pastes panel.image directly, no per-panel micro-tilt applied
            base_corners = _bent_panel_corners(panel) or _corners((0, 0, panel.image.width, panel.image.height))
            meta["panel_points"] = [(px + x_cursor, py) for px, py in base_corners]
            columns.append(Column(
                bbox=(x_cursor, 0, x_cursor + panel.image.width, panel.image.height),
                orientation_deg=0, column_type=column_type, meta=meta,
            ))
            x_cursor += panel.image.width

    if draw_staples:
        if row_layout is not None:
            idx = 0
            for count in row_layout:
                _draw_row_staples(canvas, rng, columns[idx:idx + count])
                idx += count
        else:
            _draw_row_staples(canvas, rng, columns)

    return PaperPiece(image=canvas, columns=columns)


STAPLE_BBOX_PAD_PX = 4.0  # half the widest drawn stroke (7px) rounded up -- keeps the recorded box tight to what's actually visible


def _draw_metallic_staple(image: Image.Image, cx: float, cy: float, length: float, angle_deg: float) -> None:
    """One diagonal staple pin with a simple 3-line metallic gradient
    (dark outline, mid-grey body, light highlight core) instead of a
    flat single-color bar -- matches a real reference photo's staple
    (2026-09-16, user-supplied), which reads as shiny metal, not ink."""
    draw = ImageDraw.Draw(image)
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta) * length / 2, math.sin(theta) * length / 2
    p0, p1 = (cx - dx, cy - dy), (cx + dx, cy + dy)
    draw.line([p0, p1], fill=(35, 35, 38), width=7)
    draw.line([p0, p1], fill=(150, 150, 155), width=4)
    draw.line([p0, p1], fill=(225, 225, 225), width=1)


def _staple_corners(cx: float, cy: float, length: float, angle_deg: float, pad: float = STAPLE_BBOX_PAD_PX) -> list[tuple[float, float]]:
    """The 4 corners of a rectangle hugging the drawn staple mark (its
    line segment plus half the stroke width as padding), in the same
    piece-canvas frame `_draw_metallic_staple` draws in -- the rotated
    equivalent of `_corners()`, needed because a staple is diagonal, not
    axis-aligned, so an axis-aligned box would either clip it or waste a
    lot of margin. Carried through every later transform via
    `map_panel_points`, same as `panel_points`."""
    theta = math.radians(angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)  # unit vector along the staple
    vx, vy = -uy, ux  # unit vector across it
    half_len, half_w = length / 2 + pad, pad
    return [
        (cx + s * half_len * ux + t * half_w * vx, cy + s * half_len * uy + t * half_w * vy)
        for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def _draw_row_staples(image: Image.Image, rng: random.Random, row_columns: list[Column]) -> None:
    """One diagonal metallic staple per internal seam within a single row
    of horizontally side-by-side panels -- physical panels only ever join
    horizontally (2026-09-16, user call, matching a real reference photo):
    never one spanning two vertically-stacked rows, since a row isn't the
    same physical piece of paper as the row above or below it.

    Each staple's corner points are recorded on both columns it sits
    between (`meta["staple_points"]`, a list since a middle column in a
    row of 3+ touches two seams, added 2026-09-21, wiki/TODO.md item 1) --
    staples always appear on real photos and were drawn but never
    labeled before this."""
    if len(row_columns) < 2:
        return
    y0 = min(col.bbox[1] for col in row_columns)
    y1 = max(col.bbox[3] for col in row_columns)
    row_h = y1 - y0
    length = row_h * STAPLE_LENGTH_FRAC
    for i in range(len(row_columns) - 1):
        seam_x = row_columns[i].bbox[2]  # shared edge between column i and i+1
        cy = rng.uniform(y0 + row_h * STAPLE_MARGIN_FRAC, y1 - row_h * STAPLE_MARGIN_FRAC)
        angle = rng.uniform(*STAPLE_ANGLE_RANGE_DEG) * rng.choice([1, -1])
        _draw_metallic_staple(image, seam_x, cy, length, angle)
        corners = _staple_corners(seam_x, cy, length, angle)
        row_columns[i].meta.setdefault("staple_points", []).append(corners)
        row_columns[i + 1].meta.setdefault("staple_points", []).append(corners)


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

    transform_point, (pred_w, pred_h) = pil_rotate_transform(old_w, old_h, angle_deg)
    if (pred_w, pred_h) != (new_w, new_h):
        raise AssertionError(f"pil_rotate_transform disagrees with PIL: predicted {(pred_w, pred_h)}, PIL produced {(new_w, new_h)}")

    new_columns = []
    for col in piece.columns:
        new_orientation = (col.orientation_deg + orientation_delta) % 360
        meta = map_panel_points({**col.meta, "tilt_deg": tilt_deg}, transform_point)
        new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, transform_point), orientation_deg=new_orientation, column_type=col.column_type, meta=meta))

    return PaperPiece(image=rotated, columns=new_columns)


def scale_piece(piece: PaperPiece, factor: float) -> PaperPiece:
    """Resizes the whole piece (e.g. for placing a full-quality-rendered
    piece onto a smaller page) -- scaling down at this stage instead of
    rendering at a smaller grid_px, since font sizes are fixed pixel
    constants and would stop looking right if grid_px itself shrank."""
    w, h = piece.image.size
    resized = piece.image.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.LANCZOS)
    scale = lambda x, y: (x * factor, y * factor)
    new_columns = []
    for col in piece.columns:
        meta = map_panel_points(col.meta, scale)
        new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, scale), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))
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
        meta = map_panel_points({**col.meta, "perspective_side": side, "perspective_shift_frac": max_shift_frac}, transform_point)
        new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, transform_point), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))

    return PaperPiece(image=warped, columns=new_columns)


def _project_plane_corners(w: float, h: float, tilt_x_deg: float, tilt_y_deg: float, focal_frac: float) -> np.ndarray:
    """Where the piece's 4 corners land after treating the piece as a flat
    plane in 3D, rotating it, and projecting it back through a pinhole
    camera -- a real camera model, not the edge-shift heuristic
    `apply_perspective_warp` above uses. tilt_x rotates around the
    horizontal axis (top edge moves toward/away from the camera);
    tilt_y around the vertical axis (left/right edge does). focal_frac
    sets the virtual camera's focal length as a multiple of the piece's
    own width -- how far back the camera sits; smaller exaggerates the
    perspective. The camera distance is set equal to the focal length,
    which is what makes an untilted plane project back to exactly its
    own original corners (verified: substituting tilt=0 collapses the
    projection to the identity)."""
    cx, cy = w / 2, h / 2
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts3d = np.array([[x - cx, y - cy, 0.0] for x, y in corners])

    tx, ty = math.radians(tilt_x_deg), math.radians(tilt_y_deg)
    rot_x = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    rot_y = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    rotated = pts3d @ (rot_y @ rot_x).T

    focal = focal_frac * w
    rotated[:, 2] += focal  # camera distance == focal length, see docstring
    return np.float32([[focal * x / z + cx, focal * y / z + cy] for x, y, z in rotated])


def apply_perspective_warp_3d(
    piece: PaperPiece, tilt_x_deg: float, tilt_y_deg: float, focal_frac: float = 2.2, fill=PAPER_TINT
) -> PaperPiece:
    """Experimental alternative to `apply_perspective_warp` above: instead
    of nudging one edge inward by a fixed fraction, actually rotates the
    piece as a flat plane in 3D (see `_project_plane_corners`) and
    projects it back through a pinhole camera. Physically meaningful
    parameters (two tilt angles, a focal length) in place of the 2D
    heuristic's single "which edge, how far" choice, so both axes can
    tilt at once and the amount is degrees, not a pixel fraction.

    Wired into page.py's _build_pieces (2026-09-15) for standard/torn page
    styles, replacing apply_perspective_warp above -- this is the
    2026-09-10 experiment from the backlog's "page-level 3D perspective
    warp" item (see scripts/test_paper_piece_warp3d.py), validated
    standalone before wiring in. apply_perspective_warp above still runs
    for the "ideal" style at max_shift_frac=0.0, i.e. no warp at all.

    Expands the output canvas to fit the warped corners (2026-09-16,
    fixes a real bug flagged directly by the user): a tilted plane's
    projected corners generally land outside the original [0,w]x[0,h]
    box on at least one side, and cv2.warpPerspective's output size was
    fixed at the ORIGINAL (w, h) -- silently cropping real paper content
    that projects past that box, unlike apply_tilt's plain rotation,
    which already uses PIL's expand=True for exactly this reason."""
    w, h = piece.image.size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = _project_plane_corners(w, h, tilt_x_deg, tilt_y_deg, focal_frac)

    min_x, min_y = dst[:, 0].min(), dst[:, 1].min()
    max_x, max_y = dst[:, 0].max(), dst[:, 1].max()
    new_w, new_h = int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))
    dst_shifted = dst - [min_x, min_y]

    matrix = cv2.getPerspectiveTransform(src, dst_shifted)

    arr = np.array(piece.image)
    warped_arr = cv2.warpPerspective(arr, matrix, (new_w, new_h), borderValue=fill)
    warped = Image.fromarray(warped_arr)

    def transform_point(x: float, y: float) -> tuple[float, float]:
        vec = matrix @ np.array([x, y, 1.0])
        return (vec[0] / vec[2], vec[1] / vec[2])

    new_columns = []
    for col in piece.columns:
        meta = map_panel_points({**col.meta, "warp3d_tilt_x_deg": tilt_x_deg, "warp3d_tilt_y_deg": tilt_y_deg, "warp3d_focal_frac": focal_frac}, transform_point)
        new_columns.append(Column(bbox=_bbox_from_meta(meta, col.bbox, transform_point), orientation_deg=col.orientation_deg, column_type=col.column_type, meta=meta))

    return PaperPiece(image=warped, columns=new_columns)
