"""Exports the full synthetic panel-page dataset: composes pages via
compose_page(), then decomposes each page down to its panels and leads,
mirroring this project's real page -> panel -> lead hierarchy (see
wiki/overview/synthetic_data_flow.md) so a training pipeline can target each pipeline stage
directly instead of only the final composed page.

Layout (2026-09-24 redesign), one row per unit at each level. Every folder
below is kept permanently (2026-09-24: panels-noised was briefly deleted as
an "in-between" output, then found to be orientation_classifier's own
training input -- the not-yet-rectified crop it needs specifically, since
rectification undoes the very rotation it's classifying -- so nothing here
is deleted after a run completes):
  pages/page_NNNN.png                                         -- one full composed page (photo-realistic: tilted, warped, stapled, handwritten-on)
  pages-overlayed_bbox/page_NNNN.png                          -- pages/ with every panel+staple box drawn on top, straight from page_boxes.csv -- a "check labels on actual pixels" verification image, not a training input
  panels/<panel_id>.png                                       -- one CANONICAL, pre-transform 3-lead panel render, at its own native resolution
  panels-noised/<panel_id>.png                                -- the same panel's bbox region cropped straight out of the final composited page, still tilted/warped -- orientation_classifier's own training input
  panels-noised-rectified/<panel_id>.png                      -- panels-noised/ warped back to the panel's own pristine (W,H) frame via its recorded corners -- the rectification target
  panels-noised-rectified-overlayed_bbox/<panel_id>.png       -- panels-noised-rectified/ with its style/gain/type sub-boxes drawn on top, straight from panel_field_boxes.csv
  leds_paper/<panel_id>_<lead>.png                            -- one lead's row, cropped from the canonical panel image
  leds_paper-noised/<panel_id>_<lead>.png                     -- the same row cropped from panels-noised
  leds_paper-noised-rectified/<panel_id>_<lead>.png           -- the same row cropped from panels-noised-rectified
  leds_paper-noised-rectified-overlayed_digitized/<panel_id>_<lead>.png -- leds_paper-noised-rectified/ with the digitized .npy signal drawn on top in red
  leds_paper-noised-rectified-overlayed_mask/<panel_id>_<lead>.png     -- leds_paper-noised-rectified/ with the trace mask alpha-blended on top in white
  leds_paper-masks/<panel_id>_<lead>.png                      -- trace-only ground truth mask, pixel-aligned with leds_paper/leds_paper-noised-rectified
  leds_digitized/<panel_id>_<lead>.npy                        -- the exact real PTB-XL sample array drawn as that lead's trace (float32, physical mV)
  labels/manifest_pages.csv        -- one row per page (provenance: seed, style, augmentation flags)
  labels/manifest_panels.csv       -- one row per panel (provenance: geometry, style/gain metadata, signal-degradation params) -- see page_boxes.csv/panel_field_boxes.csv below for the actual detection-training boxes
  labels/manifest_leds.csv         -- one row per lead: crop paths, digitized-signal path, the scale (px_per_mv, px_per_second) needed to relate them
  labels/page_boxes.csv            -- one row per page-level detection box: class "staple" or "panel" (with its full pristine-local -> page-space matrix, cw_rotation_to_upright_deg, pitch/yaw/roll, paper_bend_control_points)
  labels/panel_field_boxes.csv     -- one row per panel-level detection box, class-per-box (top_template, bottom_template-1/2/3, gain_symbol-<value>, type_notation-<lead>), panel-local frame
  labels/_backups/<timestamp>/     -- a full copy of every label CSV, written once a run completes ("stored more securely" -- see _backup_labels)

Resumable: a page already present in manifest_pages.csv is skipped, so a
re-run after an interruption continues from where it stopped. One page's
image, panels, leds, digitized arrays, and manifest rows are all written
together before moving to the next page, so a crash mid-run never leaves
a manifest row without its files.

Pages are independent (own rng seed each), so generation is parallelized
across worker processes -- one page per worker task, results collected
and written to the manifests as they complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_synthetic_realistic.config import GRID_BOX_SECONDS
from ecg_synthetic_realistic.page import PAGE_STYLES, compose_page
from ecg_synthetic_realistic.paper_piece import REPORT_SENTINEL, BLANK_SENTINEL
from ecg_synthetic_realistic.template import TRACE_LINE_WIDTH_PX

GRID_PX_AT_RESOLUTION_200 = 39.37
BASE_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/generate_dataset.py -> ecg-generation-synthetic_realistic -> repo -> project root
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "ecg-synthetic_realistic"  # renamed from datasets/training/synthetic-ptbxl-panels 2026-09-16, moved up to datasets/ root as a first-class dataset. Naming: "-" separates categories, "_" joins words within one category
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_WORKERS = min(os.cpu_count() or 4, 8)

# Below-normal OS scheduling priority, on by default (2026-09-24, user
# call: "i still want to use my laptop" while a long generation run is
# going in the background). This is pure CPU/PIL/cv2 drawing work with no
# GPU codepath to move it to -- there is nothing here a GPU accelerates,
# so the actual lever is letting the OS scheduler yield to whatever the
# user is doing in the foreground instead of competing with it, which a
# fixed worker-count cap can't do (it either wastes idle capacity or still
# competes at the moments that matter). --full-priority opts back into the
# old always-max-speed behavior. ctypes/stdlib only, no new dependency --
# ignored on non-Windows.
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _lower_process_priority() -> None:
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wintypes
        kernel32 = ctypes.windll.kernel32
        # Explicit types matter here: GetCurrentProcess's pseudo-handle
        # (-1) is 64-bit, and ctypes' default c_int return/arg type
        # truncates it, which made SetPriorityClass silently no-op when
        # first tried (checked directly: GetPriorityClass read back 0x0,
        # not the class just set, until these were added).
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)

# 2026-09-08: half the pages standard, the rest split between the two new
# augmentation directions (see page.py's PAGE_STYLES docstring).
# "ideal" lowered 0.25 -> 0.10 (2026-09-16, user call): it's the style that
# renders as flat, disconnected, unwarped single strips on a bare page --
# a real category worth keeping for variety, but it shouldn't be as common
# as the two realistic/warped styles. Redistributed evenly to the others.
PAGE_STYLE_WEIGHTS = {"standard": 0.55, "ideal": 0.10, "torn": 0.35}
assert set(PAGE_STYLE_WEIGHTS) == set(PAGE_STYLES), "PAGE_STYLE_WEIGHTS must cover exactly page.py's PAGE_STYLES"
STYLE_SEED_OFFSET = 999983  # arbitrary, just far from any real page index so it never collides with a per-page content seed


GE_REPORT_SEED_OFFSET = 999979  # arbitrary, distinct from STYLE_SEED_OFFSET, same resume-safe reasoning


def _ge_report_flags(count: int, seed: int, fraction: float) -> list[bool]:
    """One flag per page index, same resume-safe independence as
    _page_styles: a pure function of index+seed, not of what's already
    generated. Page style is ignored for a flagged page -- see
    _generate_one_page, which builds it via compose_page's
    forced_row_labels path instead of the normal random page_style."""
    if fraction <= 0:
        return [False] * count
    flag_rng = random.Random(seed + GE_REPORT_SEED_OFFSET)
    return [flag_rng.random() < fraction for _ in range(count)]


def _page_styles(count: int, seed: int) -> list[str]:
    """One style per page index, 0..count-1, as a pure function of the
    index and base seed -- independent of a page's own content rng and
    of how many pages are already done, so a resumed run assigns page i
    the exact same style an unbroken run would have."""
    style_rng = random.Random(seed + STYLE_SEED_OFFSET)
    styles = list(PAGE_STYLE_WEIGHTS)
    weights = list(PAGE_STYLE_WEIGHTS.values())
    return [style_rng.choices(styles, weights=weights, k=1)[0] for _ in range(count)]

PAGE_FIELDS = ["page_id", "full_path", "width", "height", "num_pieces", "num_columns", "page_style",
               "seed", "grid_px", "camscanner_applied",
               # Which optional augmentations fired. Recorded 2026-09-18 because
               # none of it was, so a page could be looked at but not described:
               # "is this panel's trace partly hidden by a scribble" had no
               # answer in the manifest, which matters when curating a training
               # set. Each is blank rather than False when the step never ran.
               "aug_handwriting", "aug_pen_scribble", "aug_sticker_blot", "aug_brush_mark", "aug_hand_shadow",
               "aug_big_cast_shadow", "aug_lighting_gradient", "aug_page_tilt",
               "aug_torn_edge", "torn_edge_side", "torn_edge_depth_px", "table_texture",
               # Added 2026-09-21 (wiki/TODO.md item 7): whether the whole-page
               # 3D perspective warp fired (compose_page's last step, replacing
               # the old per-paper warp -- see page.py's apply_page_perspective_warp_3d).
               "aug_page_warp3d",
               # Added 2026-09-21 (wiki/TODO.md, color-temperature request):
               # the R-B shift actually applied, not just a fire flag -- fires
               # on every non-ideal page, so the number itself is the useful
               # record, not a boolean.
               "page_temperature_r_b",
               # Added 2026-09-22 (wiki/TODO.md, user call: "we need
               # consistency"): "portrait" / "landscape" / "square", from the
               # final page_w/page_h -- derivable from width/height above, but
               # kept as its own column since it's what a downstream
               # full-image orientation check actually wants to read, not a
               # ratio to recompute every time.
               "page_orientation"]
PANEL_FIELDS = [
    "panel_id", "page_id", "piece_index", "column_index", "odd_even_pair_panel_id", "panel_image_path",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    # The panel's own four pristine-local corners mapped through every transform
    # to page space, added 2026-09-18. bbox above is an axis-aligned box and
    # discards rotation and shear; four point correspondences recover the exact
    # homography from the panel's pristine frame to the page, which is what makes
    # inverting the geometry exact instead of the ECC-approximated reconstruction
    # build_digitization_training_set.py had to settle for. Order matches
    # paper_piece._corners: (0,0), (W,0), (W,H), (0,H) in the pristine frame.
    "panel_pt0_x", "panel_pt0_y", "panel_pt1_x", "panel_pt1_y",
    "panel_pt2_x", "panel_pt2_y", "panel_pt3_x", "panel_pt3_y",
    "camscanner_applied",
    "orientation_deg", "column_type", "row_labels", "page_style",
    "tilt_deg", "row_panel_tilt_deg", "perspective_side", "perspective_shift_frac",
    "warp3d_tilt_x_deg", "warp3d_tilt_y_deg", "warp3d_focal_frac",
    "page_scale_factor", "page_tilt_deg", "page_tilt_applied",
    "style", "gain", "grid_color", "calibration_step_mode", "calibration_step_shape", "has_calibration_step", "panel_width_boxes",
    "panel_noise_applied", "panel_wrinkle_applied",
    # Genuine non-homography paper bend, added 2026-09-22 (wiki/TODO.md
    # rectification-model entry) -- ground truth for a dense-field
    # rectification model and for scoring the classical TPS corrector
    # (grid_rectify.refine_canonical_panel) on synthetic data for the first
    # time. panel-local (pre-placement) coordinates; JSON list of
    # [ideal_x, ideal_y, bent_x, bent_y] control points.
    "paper_bend_applied", "paper_bend_control_points",
    "calibration_step_bbox_x0", "calibration_step_bbox_y0", "calibration_step_bbox_x1", "calibration_step_bbox_y1",
    "header_icon_x0", "header_icon_y0", "header_icon_x1", "header_icon_y1",
    "piece_x", "piece_y",
    # Staple marks touching this panel, added 2026-09-21 (wiki/TODO.md item
    # 1) -- a staple always sits at a seam so it's never owned by one panel
    # alone; a JSON list since a middle column in a row of 3+ touches two
    # (one per seam). Each entry is the axis-aligned bbox in THIS panel's
    # final page-space frame (same frame as bbox_x0 above), flattened from
    # the exact rotated corners map_panel_points carried through every
    # transform, same pattern as panel_pt0_x etc.
    "staple_bboxes",
    # Every printed text field's exact box and string as a JSON list of
    # {role, text, bbox}, added 2026-09-18. One JSON column rather than a
    # column per field because a panel carries a variable number of them
    # (header, 3 lead labels, and up to 8 footer strings). Panel-local frame,
    # same as label_bbox_* and calibration_step_bbox_*.
    "text_fields",
    # Full-row target geometry, added 2026-09-18. The real trace begins at
    # trace_x0, so the region before it is unlabelled by the per-lead .npy;
    # step_* describe the calibration ramp drawn in the zone before it on one
    # row, which is what that region actually contains. Together these let a
    # training target span the whole row instead of leaving the row start
    # unlabelled (TODO gap #3's 11.1% dead zone).
    "trace_x0", "step_row_index",
    "step_x_lead_in", "step_x_rise", "step_x_fall", "step_x_lead_out",
    "step_plateau_y", "step_baseline_y",
    "bottom_row_color", "record_ref", "ptbxl_record_id",
    "header_text", "footer_top", "footer_bottom",
    "signal_degradation",
    # Numeric parameters behind signal_degradation, added 2026-09-18. This is
    # the one augmentation that alters the training TARGET rather than only the
    # input, and previously only the note string survived, so the actual burst
    # geometry and amplitude were unrecoverable from the manifest.
    "dead_lead_name", "dead_lead_flat_mv",
    "motion_lead_name", "motion_start_sample", "motion_span_samples",
    "motion_wander_freq_hz", "motion_amplitude_mv", "motion_noise_std_mv",
    # Whole-panel motion burst variant, added 2026-09-21 (user call): the
    # same burst but shared across all 3 leads at once (one physical
    # movement event), not one lead over a partial span. per_lead is a JSON
    # column (each lead's own amplitude/noise draw), same pattern as
    # text_fields, since the column count per panel is fixed at 3 but the
    # per-lead numbers still need their own place.
    "motion_panel_lead_names", "motion_panel_start_sample", "motion_panel_span_samples",
    "motion_panel_wander_freq_hz", "motion_panel_per_lead",
    # Added 2026-09-22, user call ("generate the panels from the noised
    # pages"): the panel's own bbox region cropped straight out of the
    # final fully-composited page (table background, shadows, tilt,
    # warp3d, CamScanner, paper bend -- everything), not dewarped. Pairs
    # with panel_image_path (the pristine render) as a distorted/straight
    # training pair for a rectification model. Blank on the rare panel
    # whose bbox falls entirely outside the final page canvas.
    "panel_image_noised_path",
    # Reinstated 2026-09-22 (originally added, then cut as unused the same
    # day, then found to actually be needed): panel_image_noised warped
    # back to the pristine panel's own (W,H) frame via the exact per-panel
    # homography -- undoes rotation/tilt/perspective/CamScanner but
    # deliberately leaves the panel-local paper-bend residual and page
    # noise/lighting in. The real consumer is pairing this against
    # digitized/masks (both drawn in the pristine panel-local frame,
    # unusable directly against the raw noised crop) for a realistic
    # digitization/segmentation retrain -- verified by overlay-checking 5
    # samples' digitized signal against this exact crop before wiring it
    # back in. Blank when panel_image_noised_path itself is blank.
    "panel_image_noised_rectified_path",
]
LED_FIELDS = [
    "panel_id", "lead_name", "led_image_path", "digitized_path", "mask_image_path",
    "label_bbox_x0", "label_bbox_y0", "label_bbox_x1", "label_bbox_y1",
    "y_top", "y_bottom", "num_samples", "signal_fs", "gain", "px_per_mv", "px_per_second",
    # Exact trace geometry, added 2026-09-18: x_lead_out is where this row's
    # trace actually begins in the row crop, px_per_sample is its sample
    # spacing, baseline_y is the resting line it was drawn about, and
    # trace_line_width_px is the stroke width. Without these a consumer can
    # only place the trace by re-deriving them and hoping the derivation
    # matches; x_lead_out in particular is not exactly recoverable from the
    # other columns, since grid_px was never recorded before this change.
    "x_lead_out", "px_per_sample", "baseline_y", "trace_line_width_px",
    # Added 2026-09-22, user call ("generate the panels from the noised
    # pages and the leds from the noised panels"): the realistic,
    # still-distorted crop, deliberately NOT dewarped -- see
    # _panel_local_to_page_homography's own note. Blank when the mapped
    # region fell entirely outside the final page canvas.
    "led_image_noised_path",
    # Reinstated 2026-09-22, same reasoning as panel_image_noised_rectified_path
    # above: this row's band sliced out of the rectified panel, so it lines
    # up with digitized_path/mask_image_path's own pixel coordinates while
    # still carrying realistic page texture/noise. Blank when the parent
    # panel has no rectified crop.
    "led_image_noised_rectified_path",
]
# Page-level detection boxes (2026-09-24 redesign): one row per box, class
# "staple" or "panel" -- replaces reconstructing these from manifest_panels'
# staple_bboxes/panel_pt* JSON+columns with a flat, directly-trainable
# table. matrix_a..i is the flattened 3x3 pristine-panel-local -> page-space
# homography (_panel_local_to_page_homography) -- strictly more information
# than the 4 corner points alone, and the corners remain trivially
# recoverable from it. cw_rotation_to_upright_deg is orientation_deg as-is
# (already the exact clockwise undo-rotation, PIL rotate is CCW-positive).
# pitch/yaw come from the real pinhole 3D warp (warp3d_tilt_x/y_deg, null on
# "ideal"-style pages, which use an unrelated 2D heuristic instead). roll is
# computed fresh from the matrix's own top-edge angle, not stored
# redundantly from tilt_deg+page_tilt_deg, so it can never drift from the
# matrix. paper_bend_control_points is the one genuinely non-matrix
# distortion (a panel-local TPS bend) and has to stay its own field.
PAGE_BOX_FIELDS = [
    "page_id", "class", "panel_id", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "matrix_a", "matrix_b", "matrix_c", "matrix_d", "matrix_e", "matrix_f", "matrix_g", "matrix_h", "matrix_i",
    "cw_rotation_to_upright_deg", "pitch_deg", "yaw_deg", "roll_deg", "paper_bend_control_points",
]
# Panel-level detection boxes (2026-09-24 redesign): one row per box, class-
# per-box (no grouped/class-less boxes), panel-local frame -- matches
# panels-noised-rectified/leds_paper-noised-rectified, since the
# rectification-corner fix (see _bent_panel_corners in paper_piece.py) now
# makes that frame line up with the pristine panel-local coordinates these
# boxes were measured in. gain_text intentionally excluded: not drawn on
# this GE-style template (2026-09-24 user call) -- add it only once/if that
# text actually gets rendered.
PANEL_FIELD_BOX_FIELDS = ["panel_id", "class", "text", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]


def _page_row(page_id: str, full_path: Path, page, page_style: str, seed: int, grid_px: float) -> dict:
    aug = getattr(page, "augmentations", None) or {}
    return {
        "page_id": page_id,
        "full_path": full_path.as_posix(),
        "width": page.image.width,
        "height": page.image.height,
        "num_pieces": len(page.pieces),
        "num_columns": sum(len(piece.columns) for piece in page.pieces),
        "page_style": page_style,
        # the per-page RNG seed was previously unrecoverable, so nothing about a
        # generated page could be reproduced from its manifest row alone
        "seed": seed,
        "grid_px": grid_px,
        "camscanner_applied": any(c.meta.get("camscanner_applied") for piece in page.pieces for c in piece.columns),
        **{f"aug_{k}": bool(aug.get(k)) for k in ("handwriting", "pen_scribble", "sticker_blot", "brush_mark", "hand_shadow",
                                                  "big_cast_shadow", "lighting_gradient", "page_tilt", "torn_edge", "page_warp3d")},
        "torn_edge_side": aug.get("torn_edge_side"),
        "torn_edge_depth_px": aug.get("torn_edge_depth_px"),
        "table_texture": (Path(aug["table_texture_path"]).name if aug.get("table_texture_path") else None),
        "page_temperature_r_b": aug.get("page_temperature_r_b"),
        "page_orientation": aug.get("page_orientation"),
    }


def _panel_local_to_page_homography(panel_size: tuple[int, int], panel_points: list) -> np.ndarray | None:
    """The exact homography from the pristine panel's own local frame (its
    own (0,0)-(W,H) rectangle) to final page-space, built from the same 4
    corner correspondences panel_pt0..pt3 already record. Every page-level
    transform this generator applies (rotation, tilt, both perspective
    warps, CamScanner) composes into one homography, so this single matrix
    is exact, not an approximation -- the panel-local paper bend is
    already baked into the pristine render's own pixels by this point, not
    a separate transform to compose."""
    if not panel_points or len(panel_points) != 4:
        return None
    w, h = panel_size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(panel_points)
    return cv2.getPerspectiveTransform(src, dst)


def _crop_mapped_region(page_image: Image.Image, local_rect: tuple[float, float, float, float], homography: np.ndarray) -> Image.Image | None:
    """Maps a pristine-panel-local rectangle through the homography into
    page space and crops that axis-aligned bounding box straight out of
    the final noised page -- rotation/tilt/perspective/paper-bend all stay
    exactly as they appear in the photographed page, deliberately NOT
    dewarped (2026-09-22, user call): the realistic, still-distorted half
    of a rectification-model training pair, matched against the
    already-existing pristine crop as the straight ground truth. None
    when the mapped region falls entirely outside the page canvas."""
    x0, y0, x1, y1 = local_rect
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    page_w, page_h = page_image.size
    cx0, cy0 = max(0, int(round(mapped[:, 0].min()))), max(0, int(round(mapped[:, 1].min())))
    cx1, cy1 = min(page_w, int(round(mapped[:, 0].max()))), min(page_h, int(round(mapped[:, 1].max())))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return page_image.crop((cx0, cy0, cx1, cy1))


def _rectify_panel_crop(panel_noised_image: Image.Image, panel_size: tuple[int, int], panel_points: list, crop_origin: tuple[int, int]) -> Image.Image | None:
    """Warps the already-cropped noised panel image back to the pristine
    panel's own (W,H) frame, via the exact per-panel homography shifted by
    the crop's own top-left origin -- panel_points are in PAGE space, but
    panel_noised_image's pixel (0,0) is the crop's top-left, not the
    page's, so the homography must be built against crop-local
    coordinates, not page coordinates directly. Undoes
    rotation/tilt/perspective/CamScanner but deliberately leaves the
    panel-local paper-bend residual and page noise/lighting in -- removing
    those isn't this step's job. Reinstated 2026-09-22: cut earlier the
    same day as unused, then found to be exactly what's needed to pair a
    realistic crop against digitized/masks ground truth (both drawn in
    this same pristine panel-local frame, unusable against the raw noised
    crop directly). None when panel_points is missing."""
    if not panel_points or len(panel_points) != 4:
        return None
    w, h = panel_size
    cx0, cy0 = crop_origin
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[px - cx0, py - cy0] for px, py in panel_points])
    homography = cv2.getPerspectiveTransform(src, dst)
    arr = np.array(panel_noised_image)
    rectified = cv2.warpPerspective(arr, homography, (w, h), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rectified)


def _panel_row(panel_id: str, page_id: str, piece_idx: int, col_idx: int, panel_image_path: Path, panel_image_noised_path: Path | None, panel_image_noised_rectified_path: Path | None, col, project_root: Path) -> dict:
    meta = col.meta
    points = list(meta.get("panel_points") or [])
    # calibration_step_bbox is in panel_image's own pristine local frame
    # (same frame as manifest_leds.csv's y_top/y_bottom), not page space --
    # None when has_calibration_step is False.
    step_bbox = meta["calibration_step_bbox"] or (None, None, None, None)
    pair_col_idx = meta.get("odd_even_pair_index")
    return {
        "panel_id": panel_id,
        "page_id": page_id,
        "piece_index": piece_idx,
        "column_index": col_idx,
        # Odd-even pairing, added 2026-09-21 (wiki/TODO.md item 2): the
        # paired panel's own panel_id, resolved from the within-piece
        # column index paper_piece.py recorded (pairing never crosses a
        # piece or row boundary, so page_id/piece_idx stay the same as
        # this panel's own). None when this column isn't part of a pair.
        "odd_even_pair_panel_id": (f"{page_id}_p{piece_idx}_c{pair_col_idx}" if pair_col_idx is not None else None),
        "panel_image_path": panel_image_path.relative_to(project_root).as_posix(),
        "panel_image_noised_path": (panel_image_noised_path.relative_to(project_root).as_posix() if panel_image_noised_path else None),
        "panel_image_noised_rectified_path": (panel_image_noised_rectified_path.relative_to(project_root).as_posix() if panel_image_noised_rectified_path else None),
        "bbox_x0": round(col.bbox[0], 2),
        "bbox_y0": round(col.bbox[1], 2),
        "bbox_x1": round(col.bbox[2], 2),
        "bbox_y1": round(col.bbox[3], 2),
        **{f"panel_pt{k}_{ax}": round(points[k][0 if ax == 'x' else 1], 3) for k in range(len(points)) for ax in ("x", "y")},
        "camscanner_applied": bool(meta.get("camscanner_applied")),
        "orientation_deg": col.orientation_deg,
        "column_type": col.column_type,
        "row_labels": ";".join(meta["row_labels"]),
        "page_style": meta["page_style"],
        "tilt_deg": round(meta["tilt_deg"], 3),
        "row_panel_tilt_deg": round(meta["row_panel_tilt_deg"], 3),
        # "ideal" pages still warp with apply_perspective_warp (perspective_side/
        # _shift_frac); standard/torn pages warp with apply_perspective_warp_3d
        # (warp3d_tilt_x/y_deg, warp3d_focal_frac) instead, 2026-09-15 -- a panel
        # carries exactly one of these two field groups, never both, so both
        # are read with .get() and blank when the other warp path produced this panel.
        "perspective_side": meta.get("perspective_side"),
        "perspective_shift_frac": meta.get("perspective_shift_frac"),
        "warp3d_tilt_x_deg": meta.get("warp3d_tilt_x_deg"),
        "warp3d_tilt_y_deg": meta.get("warp3d_tilt_y_deg"),
        "warp3d_focal_frac": meta.get("warp3d_focal_frac"),
        "page_scale_factor": round(meta["page_scale_factor"], 4),
        "page_tilt_deg": round(meta["page_tilt_deg"], 3),
        "page_tilt_applied": meta["page_tilt_applied"],
        "style": meta["style"],
        "gain": meta["gain"],
        "grid_color": meta["grid_color"],
        "calibration_step_mode": meta["calibration_step_mode"],
        "calibration_step_shape": meta.get("calibration_step_shape"),
        "has_calibration_step": meta["has_calibration_step"],
        "panel_width_boxes": meta["panel_width_boxes"],
        "panel_noise_applied": meta["panel_noise_applied"],
        "panel_wrinkle_applied": meta["panel_wrinkle_applied"],
        "paper_bend_applied": meta["paper_bend_applied"],
        "paper_bend_control_points": json.dumps(
            [[round(v, 3) for v in pt] for pt in meta["paper_bend_control_points"]],
            separators=(",", ":"),
        ) if meta.get("paper_bend_control_points") else None,
        "calibration_step_bbox_x0": step_bbox[0], "calibration_step_bbox_y0": step_bbox[1],
        "calibration_step_bbox_x1": step_bbox[2], "calibration_step_bbox_y1": step_bbox[3],
        "header_icon_x0": (meta["header_icon_xyxy"][0] if meta.get("header_icon_xyxy") else None),
        "header_icon_y0": (meta["header_icon_xyxy"][1] if meta.get("header_icon_xyxy") else None),
        "header_icon_x1": (meta["header_icon_xyxy"][2] if meta.get("header_icon_xyxy") else None),
        "header_icon_y1": (meta["header_icon_xyxy"][3] if meta.get("header_icon_xyxy") else None),
        "piece_x": meta.get("piece_x"), "piece_y": meta.get("piece_y"),
        "staple_bboxes": json.dumps([
            [round(min(p[0] for p in staple), 2), round(min(p[1] for p in staple), 2),
             round(max(p[0] for p in staple), 2), round(max(p[1] for p in staple), 2)]
            for staple in (meta.get("staple_points") or [])
        ]) if meta.get("staple_points") else None,
        "text_fields": json.dumps(
            [{"role": r, "text": t, "bbox": list(b)} for r, t, b in (meta.get("text_fields") or [])],
            ensure_ascii=False, separators=(",", ":"),
        ) if meta.get("text_fields") else None,
        "trace_x0": meta.get("trace_x0"),
        "step_row_index": meta.get("step_row_index"),
        "step_x_lead_in": (meta["step_geometry"][0] if meta.get("step_geometry") else None),
        "step_x_rise": (meta["step_geometry"][1] if meta.get("step_geometry") else None),
        "step_x_fall": (meta["step_geometry"][2] if meta.get("step_geometry") else None),
        "step_x_lead_out": (meta["step_geometry"][3] if meta.get("step_geometry") else None),
        "step_plateau_y": (meta["step_geometry"][4] if meta.get("step_geometry") else None),
        "step_baseline_y": (meta["step_geometry"][5] if meta.get("step_geometry") else None),
        "bottom_row_color": "|".join(str(c) for c in meta["bottom_row_color"]),
        "record_ref": meta["record_ref"],
        "ptbxl_record_id": meta["ptbxl_record_id"],
        "header_text": meta["header_text"],
        "footer_top": ";".join(meta["footer_top"]),
        "footer_bottom": ";".join(meta["footer_bottom"]),
        "signal_degradation": meta.get("signal_degradation", ""),
        **{k: meta.get(k) for k in ("dead_lead_name", "dead_lead_flat_mv",
                                    "motion_lead_name", "motion_start_sample", "motion_span_samples",
                                    "motion_wander_freq_hz", "motion_amplitude_mv", "motion_noise_std_mv",
                                    "motion_panel_start_sample", "motion_panel_span_samples", "motion_panel_wander_freq_hz")},
        "motion_panel_lead_names": json.dumps(meta["motion_panel_lead_names"]) if meta.get("motion_panel_lead_names") else None,
        "motion_panel_per_lead": json.dumps(meta["motion_panel_per_lead"]) if meta.get("motion_panel_per_lead") else None,
    }


def _led_rows(panel_id: str, col, leds_dir: Path, leds_noised_dir: Path, leds_noised_rectified_dir: Path, masks_dir: Path, digitized_dir: Path,
              leds_digitized_overlay_dir: Path, leds_mask_overlay_dir: Path, grid_px: float, page_image: Image.Image, homography: np.ndarray | None,
              rectified_panel_image: Image.Image | None, project_root: Path) -> list[dict]:
    meta = col.meta
    gain_mm_per_mv = float(meta["gain"].split("mm/mV")[0])
    px_per_mv = (grid_px / 5.0) * gain_mm_per_mv
    px_per_second = grid_px / GRID_BOX_SECONDS

    rows = []
    panel_image = meta["panel_image"]
    trace_mask = meta.get("panel_trace_mask")
    baselines = meta.get("baseline_ys") or [None] * len(meta["row_labels"])
    # homography is computed once by the caller (_generate_one_page already
    # needs it to build the panel-level noised crop) and passed in here so
    # a per-panel matrix isn't rebuilt per row.
    for row_idx, (lead_name, (y_top, y_bottom), label_bbox) in enumerate(zip(meta["row_labels"], meta["row_y_ranges"], meta["label_bboxes"])):
        led_image_path = leds_dir / f"{panel_id}_{lead_name}.png"
        panel_image.crop((0, y_top, panel_image.width, y_bottom)).save(led_image_path)

        led_image_noised_path = None
        if homography is not None:
            noised_crop = _crop_mapped_region(page_image, (0, y_top, panel_image.width, y_bottom), homography)
            if noised_crop is not None:
                led_image_noised_path = leds_noised_dir / f"{panel_id}_{lead_name}.png"
                noised_crop.save(led_image_noised_path)

        led_image_noised_rectified_path = None
        if rectified_panel_image is not None:
            led_image_noised_rectified_path = leds_noised_rectified_dir / f"{panel_id}_{lead_name}.png"
            rectified_panel_image.crop((0, y_top, rectified_panel_image.width, y_bottom)).save(led_image_noised_rectified_path)

        # Trace-only ground truth mask, cropped with the exact same box as
        # the led image above so the two stay pixel-aligned -- see TODO.md's
        # digitizer architecture-comparison item (2026-09-15).
        mask_image_path = masks_dir / f"{panel_id}_{lead_name}.png"
        mask_crop = trace_mask.crop((0, y_top, trace_mask.width, y_bottom))
        mask_crop.save(mask_image_path)

        signal = meta["lead_signals"][lead_name]
        digitized_path = digitized_dir / f"{panel_id}_{lead_name}.npy"
        np.save(digitized_path, signal)

        # Verification overlays, drawn straight on the rectified crop while
        # it's already in memory (2026-09-24 redesign) rather than a
        # separate full-corpus pass afterward -- red digitized-signal trace
        # and white/black mask, same "check labels on actual pixels"
        # purpose the earlier standalone overlay_leds_digitized.py served.
        if rectified_panel_image is not None:
            led_crop = rectified_panel_image.crop((0, y_top, rectified_panel_image.width, y_bottom))
            baseline_y_local = (baselines[row_idx] - y_top) if baselines[row_idx] is not None else None
            digitized_overlay = _draw_led_digitized_overlay(
                led_crop, signal, meta.get("trace_x0"), meta.get("px_per_sample"), px_per_mv, baseline_y_local, TRACE_LINE_WIDTH_PX)
            digitized_overlay.save(leds_digitized_overlay_dir / f"{panel_id}_{lead_name}.png")
            mask_overlay = _draw_led_mask_overlay(led_crop, mask_crop)
            mask_overlay.save(leds_mask_overlay_dir / f"{panel_id}_{lead_name}.png")

        rows.append({
            "panel_id": panel_id,
            "lead_name": lead_name,
            "led_image_path": led_image_path.relative_to(project_root).as_posix(),
            "led_image_noised_path": (led_image_noised_path.relative_to(project_root).as_posix() if led_image_noised_path else None),
            "led_image_noised_rectified_path": (led_image_noised_rectified_path.relative_to(project_root).as_posix() if led_image_noised_rectified_path else None),
            "digitized_path": digitized_path.relative_to(project_root).as_posix(),
            "mask_image_path": mask_image_path.relative_to(project_root).as_posix(),
            "label_bbox_x0": label_bbox[0], "label_bbox_y0": label_bbox[1],
            "label_bbox_x1": label_bbox[2], "label_bbox_y1": label_bbox[3],
            "y_top": y_top,
            "y_bottom": y_bottom,
            "num_samples": len(signal),
            "signal_fs": meta["signal_fs"],
            "gain": meta["gain"],
            "px_per_mv": round(px_per_mv, 4),
            "px_per_second": round(px_per_second, 4),
            "x_lead_out": meta.get("trace_x0"),
            "px_per_sample": (round(meta["px_per_sample"], 6) if meta.get("px_per_sample") else None),
            "baseline_y": (round(baselines[row_idx], 3) if baselines[row_idx] is not None else None),
            "trace_line_width_px": TRACE_LINE_WIDTH_PX,
        })
    return rows


def _roll_deg_from_points(panel_points: list) -> float | None:
    """Net in-plane rotation, read straight off the matrix's own top edge
    (panel_pt0 -> panel_pt1) rather than stored separately from
    tilt_deg+page_tilt_deg -- always consistent with matrix_a..i by
    construction, never a second number that can drift from the first."""
    if not panel_points or len(panel_points) != 4:
        return None
    (x0, y0), (x1, y1) = panel_points[0], panel_points[1]
    return round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 3)


def _flatten_matrix(matrix: np.ndarray | None) -> dict:
    if matrix is None:
        return {f"matrix_{c}": None for c in "abcdefghi"}
    flat = matrix.flatten().tolist()
    return {f"matrix_{c}": round(v, 8) for c, v in zip("abcdefghi", flat)}


def _staple_box_rows(page_id: str, page) -> list[dict]:
    """One row per physical staple, deduped across the 1-2 panels that
    share it at a seam (staple_points is recorded on every touching panel's
    own meta, see paper_piece.py) -- keyed on the rounded bbox, since the
    same physical staple carries the exact same points on each panel."""
    seen: dict[tuple, dict] = {}
    for piece in page.pieces:
        for col in piece.columns:
            for staple in (col.meta.get("staple_points") or []):
                bbox = (round(min(p[0] for p in staple), 2), round(min(p[1] for p in staple), 2),
                        round(max(p[0] for p in staple), 2), round(max(p[1] for p in staple), 2))
                seen[bbox] = {"page_id": page_id, "class": "staple", "panel_id": None,
                              "bbox_x0": bbox[0], "bbox_y0": bbox[1], "bbox_x1": bbox[2], "bbox_y1": bbox[3],
                              **_flatten_matrix(None), "cw_rotation_to_upright_deg": None,
                              "pitch_deg": None, "yaw_deg": None, "roll_deg": None, "paper_bend_control_points": None}
    return list(seen.values())


def _panel_box_row(page_id: str, panel_id: str, col, matrix: np.ndarray | None) -> dict:
    meta = col.meta
    return {
        "page_id": page_id, "class": "panel", "panel_id": panel_id,
        "bbox_x0": round(col.bbox[0], 2), "bbox_y0": round(col.bbox[1], 2),
        "bbox_x1": round(col.bbox[2], 2), "bbox_y1": round(col.bbox[3], 2),
        **_flatten_matrix(matrix),
        "cw_rotation_to_upright_deg": col.orientation_deg,
        "pitch_deg": meta.get("warp3d_tilt_x_deg"),
        "yaw_deg": meta.get("warp3d_tilt_y_deg"),
        "roll_deg": _roll_deg_from_points(meta.get("panel_points")),
        "paper_bend_control_points": json.dumps(
            [[round(v, 3) for v in pt] for pt in meta["paper_bend_control_points"]], separators=(",", ":"),
        ) if meta.get("paper_bend_control_points") else None,
    }


def _panel_field_boxes(panel_id: str, meta: dict) -> list[dict]:
    """Flattens a panel's text_fields/calibration_step_bbox into individual
    class-per-box rows in the panel-local frame (panels-noised-rectified,
    now that the bend-corner fix makes that frame line up with these
    pristine-measured coordinates) -- top_template (header), one
    bottom_template-N per footer column (top+bottom line unioned, not
    split), gain_symbol-<value> (calibration step, labeled by value only,
    clamped to the canvas since the recorded box pads 1-2px past the edge),
    type_notation-<lead> (one box per lead label, ungrouped). gain_text is
    intentionally excluded -- not drawn on this GE-style template."""
    boxes: list[tuple[str, str | None, tuple]] = []
    by_role: dict[str, list[tuple[str, tuple]]] = {}
    for role, text, bbox in (meta.get("text_fields") or []):
        by_role.setdefault(role, []).append((text, bbox))

    if "header" in by_role:
        text, bbox = by_role["header"][0]
        boxes.append(("top_template", text, bbox))

    # Footer column count varies by template/style (some panels carry a 4th
    # top-only column, e.g. "AC") -- read the actual indices present rather
    # than assuming a fixed 3, or a column silently drops with no box at all.
    footer_indices = sorted({int(k.rsplit("_", 1)[1]) for k in by_role if k.startswith("footer_top_") or k.startswith("footer_bottom_")})
    for n, i in enumerate(footer_indices):
        parts = [by_role[k][0] for k in (f"footer_top_{i}", f"footer_bottom_{i}") if k in by_role]
        if not parts:
            continue
        text = " / ".join(t for t, _ in parts)
        xs = [c for _, b in parts for c in (b[0], b[2])]
        ys = [c for _, b in parts for c in (b[1], b[3])]
        boxes.append((f"bottom_template-{n + 1}", text, (min(xs), min(ys), max(xs), max(ys))))

    for i, lead_name in enumerate(meta.get("row_labels") or []):
        field = by_role.get(f"lead_label_{i}")
        if field:
            text, bbox = field[0]
            boxes.append((f"type_notation-{lead_name}", text, bbox))

    # led_row-<lead>: the FULL row span (0, y_top, panel_width, y_bottom),
    # not just the label text position type_notation-<lead> above marks --
    # this is the same row_y_ranges _led_rows() already slices leds_paper/
    # crops with, exposed as a proper detection target (2026-09-25) so a
    # trained cropper can replace deriving row bounds from sparse
    # type_notation detections at inference time, the bug that caused
    # leds_paper/ crops to bundle multiple leads and header/footer text
    # together on real photos.
    panel_width = meta["panel_image"].width
    for lead_name, (y_top, y_bottom) in zip(meta.get("row_labels") or [], meta.get("row_y_ranges") or []):
        boxes.append((f"led_row-{lead_name}", None, (0, y_top, panel_width, y_bottom)))

    step_bbox = meta.get("calibration_step_bbox")
    if step_bbox and meta.get("has_calibration_step") and meta.get("gain"):
        gain_value = meta["gain"].split("mm/mV")[0]
        w, h = meta["panel_image"].size
        clamped = (max(0.0, step_bbox[0]), max(0.0, step_bbox[1]), min(float(w), step_bbox[2]), min(float(h), step_bbox[3]))
        boxes.append((f"gain_symbol-{gain_value}", None, clamped))

    return [{"panel_id": panel_id, "class": cls, "text": text,
             "bbox_x0": round(b[0], 2), "bbox_y0": round(b[1], 2), "bbox_x1": round(b[2], 2), "bbox_y1": round(b[3], 2)}
            for cls, text, b in boxes]


PANEL_BOX_COLOR = (30, 90, 220)
STAPLE_BOX_COLOR = (230, 120, 20)
GAIN_BOX_COLOR = (230, 120, 20)
TYPE_BOX_COLOR = (0, 160, 60)
LED_ROW_BOX_COLOR = (160, 0, 200)
DIGITIZED_OVERLAY_COLOR = (220, 20, 20)
OVERLAY_BOX_WIDTH = 3


def _panel_field_box_color(cls: str) -> tuple[int, int, int]:
    if cls.startswith("gain_symbol"):
        return GAIN_BOX_COLOR
    if cls.startswith("type_notation"):
        return TYPE_BOX_COLOR
    if cls.startswith("led_row"):
        return LED_ROW_BOX_COLOR
    return PANEL_BOX_COLOR  # top_template / bottom_template-N


def _draw_page_overlay(page_image: Image.Image, page_box_rows: list[dict]) -> Image.Image:
    img = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for row in page_box_rows:
        color = STAPLE_BOX_COLOR if row["class"] == "staple" else PANEL_BOX_COLOR
        draw.rectangle((row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]), outline=color, width=OVERLAY_BOX_WIDTH)
    return img


def _draw_panel_field_overlay(rectified_panel_image: Image.Image, field_box_rows: list[dict]) -> Image.Image:
    img = rectified_panel_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for row in field_box_rows:
        draw.rectangle((row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]),
                        outline=_panel_field_box_color(row["class"]), width=OVERLAY_BOX_WIDTH)
    return img


def _draw_led_digitized_overlay(led_crop: Image.Image, signal: np.ndarray, x_lead_out: float | None,
                                 px_per_sample: float | None, px_per_mv: float, baseline_y_local: float | None, line_width: int) -> Image.Image:
    """Draws the digitized signal directly on the rectified led crop, same
    frame/formula template.py used to draw the printed trace in the first
    place -- widened by 2px so it fully occludes the original stroke rather
    than leaving a rasterization sliver at sharp direction changes (see
    overlay_leds_digitized.py's own note)."""
    img = led_crop.convert("RGB").copy()
    if len(signal) >= 2 and x_lead_out is not None and px_per_sample is not None and baseline_y_local is not None:
        xs = x_lead_out + np.arange(len(signal)) * px_per_sample
        ys = baseline_y_local - signal * px_per_mv
        ImageDraw.Draw(img).line(list(zip(xs.tolist(), ys.tolist())), fill=DIGITIZED_OVERLAY_COLOR, width=line_width + 2, joint="curve")
    return img


def _draw_led_mask_overlay(led_crop: Image.Image, mask_crop: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Blends the raw black/white trace mask straight into the crop (not
    just brightening the trace pixels) so the background visibly dims where
    the mask is black and the trace visibly brightens where it's white --
    a highlight-only blend was nearly invisible, since the trace itself is
    already a thin 2px dark line indistinguishable from paper grain once
    whitened at this display scale."""
    base = led_crop.convert("RGB")
    mask_rgb = mask_crop.convert("L").convert("RGB")
    return Image.blend(base, mask_rgb, alpha)


def _load_done_page_ids(pages_csv: Path) -> set[str]:
    if not pages_csv.exists():
        return set()
    with pages_csv.open(encoding="utf-8") as f:
        return {row["page_id"] for row in csv.DictReader(f)}


def _backup_labels(labels_dir: Path) -> None:
    """Timestamped copy of every label CSV, written once a full run
    completes (2026-09-24, "stored more securely" -- manifest_pages.csv was
    accidentally truncated by an unrelated script this session, which had
    no recovery path since the live file was the only copy). Cheap
    (labels are small CSVs, not images) and doesn't touch the live-write
    path at all, so it can't itself introduce a resumability bug."""
    import shutil
    backup_dir = labels_dir / "_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in labels_dir.glob("*.csv"):
        shutil.copy2(csv_path, backup_dir / csv_path.name)
    print(f"labels backed up to: {backup_dir}")


GE_REPORT_ROW_LABELS = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"],
                        ["V4", "V5", "V6"], REPORT_SENTINEL, BLANK_SENTINEL]


def _generate_one_page(task: tuple) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    """Runs in a worker process: composes one page, saves the page image,
    every panel's canonical crop (plus its noised counterpart cropped
    straight out of the final composited page, added 2026-09-22, and that
    noised counterpart's own rectified version, reinstated 2026-09-22),
    every lead's crop (plus its noised and noised-rectified counterparts)
    and digitized signal array, plus the 2026-09-24 detection-box tables
    (page_boxes: staple/panel geometry+transform; panel_field_boxes:
    style/gain/type sub-regions) and their verification overlays, and
    returns all five manifests' rows for the main process to write (keeps
    the CSVs single-writer, so resumability stays a simple "page_id already
    present" check with no cross-process file locking)."""
    (i, seed, page_style, grid_px, column_counts, pages_dir, pages_overlayed_dir, panels_dir, panels_noised_dir, panels_noised_rectified_dir,
     panels_overlayed_dir, leds_dir, leds_noised_dir, leds_noised_rectified_dir, leds_digitized_overlay_dir, leds_mask_overlay_dir,
     masks_dir, digitized_dir, project_root, is_ge_report) = task
    page_id = f"page_{i:04d}"
    rng = random.Random(seed)
    if is_ge_report:
        # Connected GE 12SL report archetype (2026-09-23, reference photo
        # scan_0004.jpg): page_style forced to "ideal" here regardless of
        # the randomly-drawn one passed in -- this page IS the clean/
        # minimal-shadow/no-tilt case by construction, not a random draw
        # that happened to land there.
        page_style = "ideal"
        page = compose_page(rng, grid_px=grid_px, page_style=page_style, row_layout=[3, 3],
                             forced_row_labels=GE_REPORT_ROW_LABELS, uniform_style="even",
                             uniform_grid_color=True, no_row_tilt=True)
    else:
        page = compose_page(rng, grid_px=grid_px, page_style=page_style, column_counts=column_counts)

    page_image_path = pages_dir / f"{page_id}.png"
    page.image.save(page_image_path)
    page_row = _page_row(page_id, page_image_path.relative_to(project_root), page, page_style, seed, grid_px)

    panel_rows, led_rows, panel_field_box_rows = [], [], []
    page_box_rows = _staple_box_rows(page_id, page)
    for piece_idx, piece in enumerate(page.pieces):
        for col_idx, col in enumerate(piece.columns):
            panel_id = f"{page_id}_p{piece_idx}_c{col_idx}"
            homography = _panel_local_to_page_homography(col.meta["panel_image"].size, col.meta.get("panel_points"))
            page_box_rows.append(_panel_box_row(page_id, panel_id, col, homography))
            # Non-digitizable slots (2026-09-23, reference photo scan_0004.jpg,
            # user call: "won't have bounding box label since it is not a led
            # page"): report/blank columns carry no leads (row_labels == [])
            # and no printed gain to divide by, so _led_rows (which would
            # raise on meta["gain"] == "") is skipped for them below -- that
            # original "no lead bbox" intent is unchanged. But (2026-09-25)
            # they DO now get a normal panel_rows entry and noised/rectified
            # crop like any other panel: detection_panels_leds needs real
            # negative examples (a real image, zero led_row boxes) to learn
            # what "not a lead row" looks like, and excluding these panels
            # from manifest_panels.csv entirely meant prepare_panel_field_
            # boxes_dataset.py could never pick them up as training images at
            # all -- see wiki/TODO.md's 2026-09-25 entry for the real-photo
            # failure this addresses (a report panel's text confidently
            # mistaken for a led_row).
            is_non_digitizable = col.column_type in ("report", "blank")
            panel_image_path = panels_dir / f"{panel_id}.png"
            col.meta["panel_image"].save(panel_image_path)

            # Noised panel: the panel's own already-tracked bbox, cropped
            # straight out of the final composited page -- see
            # _crop_mapped_region's own note on why this stays distorted
            # rather than being dewarped.
            bx0, by0, bx1, by1 = col.bbox
            cx0, cy0 = max(0, int(round(bx0))), max(0, int(round(by0)))
            cx1, cy1 = min(page.image.width, int(round(bx1))), min(page.image.height, int(round(by1)))
            panel_image_noised_path = None
            panel_image_noised_rectified_path = None
            rectified_panel_image = None
            if cx1 > cx0 and cy1 > cy0:
                panel_image_noised = page.image.crop((cx0, cy0, cx1, cy1))
                panel_image_noised_path = panels_noised_dir / f"{panel_id}.png"
                panel_image_noised.save(panel_image_noised_path)

                rectified_panel_image = _rectify_panel_crop(panel_image_noised, col.meta["panel_image"].size, col.meta.get("panel_points"), (cx0, cy0))
                if rectified_panel_image is not None:
                    panel_image_noised_rectified_path = panels_noised_rectified_dir / f"{panel_id}.png"
                    rectified_panel_image.save(panel_image_noised_rectified_path)

            panel_rows.append(_panel_row(panel_id, page_id, piece_idx, col_idx, panel_image_path, panel_image_noised_path, panel_image_noised_rectified_path, col, project_root))
            if not is_non_digitizable:
                led_rows.extend(_led_rows(panel_id, col, leds_dir, leds_noised_dir, leds_noised_rectified_dir, masks_dir, digitized_dir,
                                           leds_digitized_overlay_dir, leds_mask_overlay_dir, grid_px, page.image, homography, rectified_panel_image, project_root))

            field_boxes = _panel_field_boxes(panel_id, col.meta)
            panel_field_box_rows.extend(field_boxes)
            if rectified_panel_image is not None and field_boxes:
                _draw_panel_field_overlay(rectified_panel_image, field_boxes).save(panels_overlayed_dir / f"{panel_id}.png")

    _draw_page_overlay(page.image, page_box_rows).save(pages_overlayed_dir / f"{page_id}.png")

    return page_row, panel_rows, led_rows, page_box_rows, panel_field_box_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--grid-px", type=float, default=GRID_PX_AT_RESOLUTION_200)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--column-counts", type=str, default=None,
                         help="force every page in this run to this exact per-piece column layout instead of drawing it randomly, e.g. '3,3' for 2 pieces of 3 columns each (6 total). Omit for the default fully-random layout")
    parser.add_argument("--ge-report-fraction", type=float, default=0.0,
                         help="2026-09-23 (reference photo scan_0004.jpg): fraction of pages built as the connected GE 12SL report archetype instead of the normal random composition -- 3x2 layout, leds/leds/leds on top, leds/report/blank on bottom, one shared style/grid, no per-panel tilt. 0 (default) never builds one")
    parser.add_argument("--full-priority", action="store_true",
                         help="2026-09-24: run at normal OS scheduling priority instead of the default below-normal. Default is considerate of whatever else the machine is doing at the time (this is pure CPU drawing work, nothing here uses a GPU); pass this for a dedicated/unattended machine where raw speed matters more")
    args = parser.parse_args()
    if not args.full_priority:
        _lower_process_priority()
    column_counts = [int(c) for c in args.column_counts.split(",")] if args.column_counts else None
    # Resolve before deriving page paths: every manifest row calls
    # Path.relative_to(PROJECT_ROOT), which is a purely lexical operation, so a
    # relative --out-dir containing ".." fails that check even when it resolves
    # inside the project. Found 2026-09-18 running the schema validation into
    # inferences/, which produced "'..\\..\\inferences\\pages\\page_0000.png' is
    # not in the subpath of '<project root>'".
    args.out_dir = args.out_dir.resolve()

    pages_dir = args.out_dir / "pages"
    pages_overlayed_dir = args.out_dir / "pages-overlayed_bbox"
    panels_dir = args.out_dir / "panels"
    panels_noised_dir = args.out_dir / "panels-noised"  # kept permanently: orientation_classifier's own training input, not just an inspection copy
    panels_noised_rectified_dir = args.out_dir / "panels-noised-rectified"
    panels_overlayed_dir = args.out_dir / "panels-noised-rectified-overlayed_bbox"
    leds_dir = args.out_dir / "leds_paper"
    leds_noised_dir = args.out_dir / "leds_paper-noised"
    leds_noised_rectified_dir = args.out_dir / "leds_paper-noised-rectified"
    leds_digitized_overlay_dir = args.out_dir / "leds_paper-noised-rectified-overlayed_digitized"
    leds_mask_overlay_dir = args.out_dir / "leds_paper-noised-rectified-overlayed_mask"
    masks_dir = args.out_dir / "leds_paper-masks"
    digitized_dir = args.out_dir / "leds_digitized"
    labels_dir = args.out_dir / "labels"
    for d in (pages_dir, pages_overlayed_dir, panels_dir, panels_noised_dir, panels_noised_rectified_dir, panels_overlayed_dir,
              leds_dir, leds_noised_dir, leds_noised_rectified_dir, leds_digitized_overlay_dir, leds_mask_overlay_dir,
              masks_dir, digitized_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = DEFAULT_LOG_DIR / f"synthetic-ptbxl-panels-generate-{datetime.now():%Y%m%d-%H%M}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")

    pages_csv = labels_dir / "manifest_pages.csv"
    panels_csv = labels_dir / "manifest_panels.csv"
    leds_csv = labels_dir / "manifest_leds.csv"
    page_boxes_csv = labels_dir / "page_boxes.csv"
    panel_field_boxes_csv = labels_dir / "panel_field_boxes.csv"
    done_page_ids = _load_done_page_ids(pages_csv)
    # PROJECT_ROOT (from __file__), not a fixed number of args.out_dir.parents[]
    # steps up -- that broke the moment --out-dir's own folder depth changed
    # (the D:\...\datasets\synthetic-ptbxl-panels -> datasets\training\synthetic-ptbxl-panels
    # fix above added a level), and would break again for any --out-dir a caller passes.
    project_root = PROJECT_ROOT

    pending = [i for i in range(args.count) if f"page_{i:04d}" not in done_page_ids]
    logging.info("starting run: %d total, %d already done, %d pending, %d workers", args.count, len(done_page_ids), len(pending), args.workers)
    print(f"{len(done_page_ids)} pages already done, {len(pending)} pending, {args.workers} workers")

    if not pending:
        print("nothing to do")
        return

    page_styles = _page_styles(args.count, args.seed)
    ge_report_flags = _ge_report_flags(args.count, args.seed, args.ge_report_fraction)
    tasks = [(i, args.seed + i, page_styles[i], args.grid_px, column_counts, pages_dir, pages_overlayed_dir, panels_dir, panels_noised_dir,
              panels_noised_rectified_dir, panels_overlayed_dir, leds_dir, leds_noised_dir, leds_noised_rectified_dir,
              leds_digitized_overlay_dir, leds_mask_overlay_dir, masks_dir, digitized_dir, project_root, ge_report_flags[i]) for i in pending]
    start_time = time.time()

    with pages_csv.open("a", newline="", encoding="utf-8") as pf, panels_csv.open("a", newline="", encoding="utf-8") as nf, \
         leds_csv.open("a", newline="", encoding="utf-8") as lf, page_boxes_csv.open("a", newline="", encoding="utf-8") as bf, \
         panel_field_boxes_csv.open("a", newline="", encoding="utf-8") as ff:
        page_writer = csv.DictWriter(pf, fieldnames=PAGE_FIELDS)
        panel_writer = csv.DictWriter(nf, fieldnames=PANEL_FIELDS)
        led_writer = csv.DictWriter(lf, fieldnames=LED_FIELDS)
        page_box_writer = csv.DictWriter(bf, fieldnames=PAGE_BOX_FIELDS)
        panel_field_box_writer = csv.DictWriter(ff, fieldnames=PANEL_FIELD_BOX_FIELDS)
        if pf.tell() == 0:
            page_writer.writeheader()
        if nf.tell() == 0:
            panel_writer.writeheader()
        if lf.tell() == 0:
            led_writer.writeheader()
        if bf.tell() == 0:
            page_box_writer.writeheader()
        if ff.tell() == 0:
            panel_field_box_writer.writeheader()

        completed = 0
        # Worker processes are spawned fresh on Windows (a re-exec of
        # python.exe per worker, not a fork that would inherit the parent's
        # already-lowered priority class), so each one needs to set its own
        # -- an initializer runs once per worker at startup, not once total.
        pool_kwargs = {} if args.full_priority else {"initializer": _lower_process_priority}
        with ProcessPoolExecutor(max_workers=args.workers, **pool_kwargs) as pool:
            futures = [pool.submit(_generate_one_page, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="pages"):
                page_row, panel_rows, led_rows, page_box_rows, panel_field_box_rows = future.result()
                page_writer.writerow(page_row)
                for row in panel_rows:
                    panel_writer.writerow(row)
                for row in led_rows:
                    led_writer.writerow(row)
                for row in page_box_rows:
                    page_box_writer.writerow(row)
                for row in panel_field_box_rows:
                    panel_field_box_writer.writerow(row)
                pf.flush()
                nf.flush()
                lf.flush()
                bf.flush()
                ff.flush()

                completed += 1
                elapsed = time.time() - start_time
                remaining = len(pending) - completed
                eta = timedelta(seconds=round(elapsed / completed * remaining)) if completed else None
                logging.info("wrote %s (%d/%d, elapsed %s, ETA %s)", page_row["page_id"], completed, len(pending), timedelta(seconds=round(elapsed)), eta)

    print(f"pages written to: {pages_dir} (overlayed: {pages_overlayed_dir})")
    print(f"panels: {panels_dir} (noised: {panels_noised_dir}, noised_rectified: {panels_noised_rectified_dir}, overlayed: {panels_overlayed_dir})")
    print(f"leds: {leds_dir} (noised: {leds_noised_dir}, noised_rectified: {leds_noised_rectified_dir}), digitized: {digitized_dir}, masks: {masks_dir}")
    print(f"led overlays: digitized={leds_digitized_overlay_dir}, mask={leds_mask_overlay_dir}")
    print(f"manifests: {pages_csv}, {panels_csv}, {leds_csv}, {page_boxes_csv}, {panel_field_boxes_csv}")
    print(f"log: {log_path}")

    run_complete = len(done_page_ids) + completed >= args.count
    if run_complete:
        _backup_labels(labels_dir)
    else:
        print(f"run incomplete ({len(done_page_ids) + completed}/{args.count} pages done) -- backup kept for the next resume")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
