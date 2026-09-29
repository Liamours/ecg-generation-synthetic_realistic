"""Builds the digitization training set by reverse-labeling: inverts the
exact geometric transform chain page.py/paper_piece.py applied when
compositing each panel onto its page, using the parameters
manifest_panels.csv already recorded for it (orientation_deg, tilt_deg,
perspective_side/shift_frac or warp3d_tilt_x/y/focal_deg, page_tilt_deg),
instead of copying the generator's own separate clean pre-composition
render (`panel_image_path`) or re-detecting anything from pixels.

Forward chain per panel (page.py's _build_pieces, confirmed by reading
the call order directly): rotate (orientation_deg + tilt_deg combined,
one PIL Image.rotate call) -> perspective warp (apply_perspective_warp's
2D edge-shift for "ideal" pages, always a no-op there since
IDEAL_PERSPECTIVE_MAX_SHIFT_FRAC=0.0; apply_perspective_warp_3d's pinhole
-camera model otherwise) -> scale by the fixed PIECE_SCALE_TO_PAGE
constant (903/709 -- NOT the manifest's own "page_scale_factor" column,
which page.py:362 hardcodes to 1.0 unconditionally and is therefore
never the real value) -> place on page (a translation, irrelevant once
we're working with an already-cropped, page-local image) -> page-level
tilt (apply_page_tilt, only if page_tilt_applied).

Inversion, in reverse order, undoing the outermost step first:
  1. undo page_tilt_deg: PIL rotate by -angle, exact -- a rotation's
     effect on a local sub-region's own appearance doesn't depend on
     what point the whole image rotated around, only on the angle.
  2. undo PIECE_SCALE_TO_PAGE: resize by 1/factor, exact for the same
     reason (a uniform scale is also position-independent locally).
  3. undo the perspective/3D warp: reconstructs the same forward
     homography paper_piece.py's own apply_perspective_warp_3d /
     apply_perspective_warp use, but seeded with THIS PANEL's own crop
     width/height as a stand-in for the whole piece's -- the true
     piece extent at warp time isn't recoverable from manifest_panels.csv
     (a piece can hold several panels in a row_layout grid that isn't
     recorded per-panel, and the final bbox is an axis-aligned box, not
     the 4 true corners a homography would need). Exact when a panel
     fills its own piece; an approximation otherwise, since the true
     warp is position-dependent across the whole piece, not just this
     panel. This is the one non-exact step -- see --validate below.
  4. undo the apply_tilt rotation (orientation_deg + tilt_deg): exact,
     same reasoning as step 1.
  5. cv2.resize to the fixed 709x591 canonical size (CANONICAL_WIDTH_PX/
     CANONICAL_HEIGHT_PX in the digitization repo's rectify_classical.py),
     the same final normalization canonicalize_panel's own real-photo
     path ends with.

--validate scores each panel's reconstructed panels_realistic image
against panel_image_path (the generator's true pre-composition render,
still available as an oracle even though it's no longer the delivered
output) via mean absolute pixel error, to measure how good step 3's
approximation actually is instead of asserting it.

Usage:
    .venv/Scripts/python scripts/build_digitization_training_set.py \
        --manifest ../../datasets/ecg-synthetic_realistic/labels/manifest_panels.csv \
        --pages-dir ../../datasets/ecg-synthetic_realistic/pages \
        --out-dir ../../datasets/training/digitization-synthetic_realistic \
        --page-ids page_0000 page_0001 --validate
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # scripts/build_digitization_training_set.py -> ecg-generation-synthetic_realistic -> repo -> project root

PIECE_SCALE_TO_PAGE = 903 / 709  # page.py's own fixed constant (scale_piece call in _build_pieces) -- not the manifest's dead "page_scale_factor" column
CANONICAL_WIDTH_PX = 709  # matches rectify_classical.py's CANONICAL_WIDTH_PX / panel_image_path's own saved resolution
CANONICAL_HEIGHT_PX = 591
PAPER_TINT = (240, 234, 216)  # template.py's own constant, RGB order (for PIL fill) -- used as padding fill so undone corners match the real paper color, not a mismatched white
PAPER_TINT_BGR = PAPER_TINT[::-1]  # same color, BGR order, for cv2's borderValue

# Mirrors repo/ecg-digitization-synthetic_realistic/src/rectification/grid_angle.py's
# estimate_tilt_deg -- kept in sync by hand like this project's other cross-repo
# constants, not imported (separate repo/venv). Only used as a fallback residual-tilt
# measurement for panels generated before 2026-09-17's row_panel_tilt_deg fix (see
# wiki/overview/methods.md, Digitization); rows carrying that label need no measurement.
DARKNESS_THRESHOLD = 80
HOUGH_THETA_RAD = np.pi / 1440
HOUGH_THRESHOLD_FRAC = 0.25


def _fold_to_grid_angle(angle_deg: float) -> float:
    folded = angle_deg % 90.0
    return folded - 90.0 if folded > 45.0 else folded


def _measure_residual_tilt_deg(bgr: np.ndarray) -> float | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARKNESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    threshold = int(min(gray.shape) * HOUGH_THRESHOLD_FRAC)
    lines = cv2.HoughLines(dark, 1, HOUGH_THETA_RAD, threshold)
    if lines is None:
        return None
    thetas_deg = np.degrees(lines[:, 0, 1])
    line_angles = -(thetas_deg - 90.0)
    return float(np.median([_fold_to_grid_angle(a) for a in line_angles]))

OUTPUT_MANIFEST_COLUMNS = [
    "panel_id", "page_id", "column_type", "row_labels",
    "reconstruction",
    "orientation_deg", "tilt_deg", "row_panel_tilt_deg", "residual_tilt_source", "perspective_side", "perspective_shift_frac",
    "warp3d_tilt_x_deg", "warp3d_tilt_y_deg", "warp3d_focal_frac",
    "page_tilt_deg", "page_tilt_applied",
    "distorted_path", "realistic_path", "ecc_correlation", "ground_truth_mae_before_ecc", "ground_truth_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reverse-label the synthetic dataset's own known bbox/transform into a digitization training set.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pages-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-ids", nargs="+", default=None, help="omit to process every page_id in the manifest")
    parser.add_argument("--validate", action="store_true", help="score panels_realistic against panel_image_path's own true pre-composition render (mean absolute pixel error, 0-255 scale)")
    return parser.parse_args()


def panel_suffix(panel_id: str, page_id: str) -> str:
    return panel_id[len(page_id) + 1:]  # "page_0000_p0_c0" -> "p0_c0"


def _project_plane_corners(w: float, h: float, tilt_x_deg: float, tilt_y_deg: float, focal_frac: float) -> np.ndarray:
    """Identical to paper_piece.py's own function of the same name --
    reused verbatim so the inverse below starts from the exact forward
    model the generator used, not a re-derived approximation of it."""
    cx, cy = w / 2, h / 2
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts3d = np.array([[x - cx, y - cy, 0.0] for x, y in corners])
    tx, ty = math.radians(tilt_x_deg), math.radians(tilt_y_deg)
    rot_x = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    rot_y = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    rotated = pts3d @ (rot_y @ rot_x).T
    focal = focal_frac * w
    rotated[:, 2] += focal
    return np.float32([[focal * x / z + cx, focal * y / z + cy] for x, y, z in rotated])


def _undo_warp3d(bgr: np.ndarray, tilt_x_deg: float, tilt_y_deg: float, focal_frac: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = _project_plane_corners(w, h, tilt_x_deg, tilt_y_deg, focal_frac)
    forward = cv2.getPerspectiveTransform(src, dst - dst.min(axis=0))
    return cv2.warpPerspective(bgr, forward, (w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=PAPER_TINT_BGR)


def _undo_perspective_2d(bgr: np.ndarray, side: str, shift_frac: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift_x, shift_y = w * shift_frac, h * shift_frac
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
    forward = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, forward, (w, h), flags=cv2.WARP_INVERSE_MAP, borderValue=PAPER_TINT_BGR)


def _undo_rotation(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return bgr
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rotated = pil.rotate(-angle_deg, expand=True, fillcolor=PAPER_TINT, resample=Image.BICUBIC)
    return cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)


def make_realistic(distorted_bgr: np.ndarray, row: dict) -> tuple[np.ndarray, str]:
    img = distorted_bgr
    if row["page_tilt_applied"] == "True":
        img = _undo_rotation(img, float(row["page_tilt_deg"]))
    img = cv2.resize(img, (round(img.shape[1] / PIECE_SCALE_TO_PAGE), round(img.shape[0] / PIECE_SCALE_TO_PAGE)), interpolation=cv2.INTER_AREA)
    if row["warp3d_tilt_x_deg"]:
        img = _undo_warp3d(img, float(row["warp3d_tilt_x_deg"]), float(row["warp3d_tilt_y_deg"]), float(row["warp3d_focal_frac"]))
    elif row["perspective_side"] and float(row["perspective_shift_frac"]) > 0:
        img = _undo_perspective_2d(img, row["perspective_side"], float(row["perspective_shift_frac"]))

    # row_panel_tilt_deg (2026-09-17 generator fix) is the per-panel
    # micro-tilt applied before apply_tilt's own whole-piece rotation --
    # both are pure rotations composing additively for local content
    # regardless of order, so undoing their sum in one call is exact.
    # Panels generated before the fix carry no such column (missing/blank
    # in manifest_panels.csv): this is the source of the MAE this script
    # originally couldn't explain (see plans/digitization-training-set.md)
    # -- fall back to measuring whatever small tilt remains directly off
    # the grid, an approximation but the only option for that older data.
    row_panel_tilt = row.get("row_panel_tilt_deg", "")
    if row_panel_tilt not in ("", None):
        img = _undo_rotation(img, float(row["orientation_deg"]) + float(row["tilt_deg"]) + float(row_panel_tilt))
        residual_tilt_source = "label"
    else:
        img = _undo_rotation(img, float(row["orientation_deg"]) + float(row["tilt_deg"]))
        residual = _measure_residual_tilt_deg(img)
        if residual is not None:
            img = _undo_rotation(img, residual)
        residual_tilt_source = "measured" if residual is not None else "unavailable"

    return cv2.resize(img, (CANONICAL_WIDTH_PX, CANONICAL_HEIGHT_PX), interpolation=cv2.INTER_AREA), residual_tilt_source


ECC_ITERATIONS = 500
ECC_EPSILON = 1e-7
ECC_COARSE_SCALE = 0.25  # a coarse low-res pass first, then refine at full res -- single-scale ECC alone converged badly (correlation ~0.05-0.6) on several panels whose true offset exceeded its basin of convergence from an identity start; coarse-to-fine fixed every one of those in testing


def ecc_refine(img_bgr: np.ndarray, truth_bgr: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Final snap-to-truth alignment. The label-driven geometric undo
    above gets close but leaves a real residual: found 2026-09-17, an
    unmodeled anisotropic scale/translation drift from how PIL's
    expand=True pads across two sequential rotations (row-panel micro-
    tilt, then the whole-piece rotation) that doesn't reduce to this
    script's single-combined-angle bounding-box math -- see
    plans/digitization-training-set.md for the full diagnosis (ECC
    itself was what measured it: ~2.5%/1.75% anisotropic scale, ~9px/6.5px
    translation, ~0 rotation on the first panel checked). Also the only
    thing that can correct step 3's warp3d approximation, which is
    unfixable from labels alone regardless.

    panel_image_path is used here only to correct the reconstruction's
    own geometry, never as the source of its pixels -- the delivered
    image still carries every real degradation from the distorted crop,
    ECC only decides how to resample it. Falls back to the unrefined
    image (correlation=None) when ECC fails to converge (low-texture
    panels can do this) -- one bad panel must not abort the batch."""
    g_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g_truth = cv2.cvtColor(truth_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPSILON)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        small_img = cv2.resize(g_img, None, fx=ECC_COARSE_SCALE, fy=ECC_COARSE_SCALE)
        small_truth = cv2.resize(g_truth, None, fx=ECC_COARSE_SCALE, fy=ECC_COARSE_SCALE)
        _, warp = cv2.findTransformECC(small_truth, small_img, warp, cv2.MOTION_AFFINE, criteria)
        warp[:, 2] /= ECC_COARSE_SCALE
        correlation, warp = cv2.findTransformECC(g_truth, g_img, warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        return img_bgr, None
    h, w = truth_bgr.shape[:2]
    refined = cv2.warpAffine(img_bgr, warp, (w, h), borderValue=PAPER_TINT_BGR, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    return refined, float(correlation)


def make_realistic_exact(page_bgr: np.ndarray, row: dict) -> np.ndarray | None:
    """Exact inverse in one homography, from the recorded four corner points.

    This replaces the whole five-step label-reconstruction chain below (undo
    page tilt, undo piece scale, rebuild and undo the warp, undo the two
    rotations) plus its ECC refinement. That chain exists because
    manifest_panels.csv recorded only an axis-aligned bbox, which cannot
    express rotation or shear, so the true transform had to be re-derived from
    parameters and then snapped to truth with ECC -- leaving a mean error of
    about 27/255 even after refinement.

    `panel_pt0`..`panel_pt3` (generator change, 2026-09-18) record the panel's
    own four pristine corners in page coordinates. Four point correspondences
    determine a homography exactly, so there is nothing left to reconstruct or
    to refine. `panel_image_path` is read here only for its pixel dimensions.

    Returns None when the row predates those columns, so the caller can fall
    back to the old chain rather than fail."""
    if any(row.get(f"panel_pt{k}_{a}") in ("", None) for k in range(4) for a in ("x", "y")):
        return None
    truth = cv2.imread(str(PROJECT_ROOT / row["panel_image_path"]))
    if truth is None:
        return None
    h, w = truth.shape[:2]
    corners = np.float32([[float(row[f"panel_pt{k}_x"]), float(row[f"panel_pt{k}_y"])] for k in range(4)])
    forward = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), corners)
    back = cv2.warpPerspective(page_bgr, np.linalg.inv(forward), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(back, (CANONICAL_WIDTH_PX, CANONICAL_HEIGHT_PX), interpolation=cv2.INTER_AREA)


def process_panel(row: dict, page_img: Image.Image, out_dir: Path, validate: bool) -> dict:
    page_id, panel_id = row["page_id"], row["panel_id"]
    suffix = panel_suffix(panel_id, page_id)
    x0, y0, x1, y1 = (round(float(row[f"bbox_{k}"])) for k in ("x0", "y0", "x1", "y1"))

    distorted_dir = out_dir / "panels_distorted" / page_id
    distorted_dir.mkdir(parents=True, exist_ok=True)
    distorted_path = distorted_dir / f"{suffix}.png"
    crop = page_img.crop((x0, y0, x1, y1))
    crop.save(distorted_path)

    distorted_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    # Exact homography when the corpus carries the corner columns, else the
    # old label-reconstruction chain. ECC still runs on top of either: it is
    # fit against the pristine render, so it absorbs the residual resampling
    # and rounding difference that no analytic path can model. Measured on six
    # panels of one page, mean difference against the pristine render on the
    # 0-255 scale: old chain + ECC 45.49, exact alone 45.24, exact + ECC 43.60.
    # On the `ideal` style, where nothing is applied after the geometry and the
    # number means what it says, exact + ECC beat old + ECC on all six panels.
    page_bgr = cv2.cvtColor(np.array(page_img), cv2.COLOR_RGB2BGR)
    exact = make_realistic_exact(page_bgr, row)
    if exact is not None:
        realistic_bgr, residual_tilt_source, reconstruction = exact, "exact_homography", "exact_homography"
    else:
        realistic_bgr, residual_tilt_source = make_realistic(distorted_bgr, row)
        reconstruction = "label_reconstruction"

    truth_bgr = cv2.imread(str(PROJECT_ROOT / row["panel_image_path"]))
    ground_truth_mae_before = float(np.abs(realistic_bgr.astype(np.float32) - truth_bgr.astype(np.float32)).mean()) if validate else None
    realistic_bgr, ecc_correlation = ecc_refine(realistic_bgr, truth_bgr)

    realistic_dir = out_dir / "panels_realistic" / page_id
    realistic_dir.mkdir(parents=True, exist_ok=True)
    realistic_path = realistic_dir / f"{suffix}.png"
    cv2.imwrite(str(realistic_path), realistic_bgr)

    ground_truth_mae = float(np.abs(realistic_bgr.astype(np.float32) - truth_bgr.astype(np.float32)).mean()) if validate else None

    return {
        "panel_id": panel_id, "page_id": page_id, "column_type": row["column_type"], "row_labels": row["row_labels"],
        "reconstruction": reconstruction,
        "orientation_deg": row["orientation_deg"], "tilt_deg": row["tilt_deg"],
        "row_panel_tilt_deg": row.get("row_panel_tilt_deg", ""), "residual_tilt_source": residual_tilt_source,
        "perspective_side": row["perspective_side"], "perspective_shift_frac": row["perspective_shift_frac"],
        "warp3d_tilt_x_deg": row["warp3d_tilt_x_deg"], "warp3d_tilt_y_deg": row["warp3d_tilt_y_deg"], "warp3d_focal_frac": row["warp3d_focal_frac"],
        "page_tilt_deg": row["page_tilt_deg"], "page_tilt_applied": row["page_tilt_applied"],
        "distorted_path": str(distorted_path), "realistic_path": str(realistic_path),
        "ecc_correlation": ecc_correlation, "ground_truth_mae_before_ecc": ground_truth_mae_before, "ground_truth_mae": ground_truth_mae,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.page_ids is not None:
        wanted = set(args.page_ids)
        rows = [r for r in rows if r["page_id"] in wanted]
    if not rows:
        print("no matching panels")
        return

    pages: dict[str, list[dict]] = {}
    for row in rows:
        pages.setdefault(row["page_id"], []).append(row)

    manifest_path = args.out_dir / "manifest.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    maes = []
    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=OUTPUT_MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        for page_id, panel_rows in tqdm(pages.items(), desc="pages"):
            page_path = args.pages_dir / f"{page_id}.png"
            page_img = Image.open(page_path).convert("RGB")
            for row in panel_rows:
                entry = process_panel(row, page_img, args.out_dir, args.validate)
                writer.writerow(entry)
                if entry["ground_truth_mae"] is not None:
                    maes.append(entry["ground_truth_mae"])
        mf.flush()

    print(f"{sum(len(v) for v in pages.values())} panels across {len(pages)} pages -> {manifest_path}")
    if maes:
        print(f"ground-truth MAE (0-255 scale): mean {np.mean(maes):.2f}, max {np.max(maes):.2f}, per-panel values in the manifest's ground_truth_mae column")


if __name__ == "__main__":
    main()
