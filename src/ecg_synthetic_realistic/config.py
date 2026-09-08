"""Filesystem layout shared by every step. This repo holds code only -- all
generated data (panels, pages, manifests) writes to ``datasets/`` at the
project root, never inside this repo, per the 2026-09-05 decision to keep
this repo small.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[1]

ECG_IMAGE_KIT_GENERATOR_DIR = PROJECT_ROOT / "external" / "ecg-image-kit" / "codes" / "ecg-image-generator"
ECG_IMAGE_KIT_BATCH_SCRIPT = ECG_IMAGE_KIT_GENERATOR_DIR / "gen_ecg_images_from_data_batch.py"

PTBXL_DATA_DIR = PROJECT_ROOT / "datasets" / "ptb-xl" / "data"

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "synthetic-ecg-realistic"

LEADS_PER_PANEL = 3
PANELS_PER_RECORD = 4

PANEL_ROW_ORDER = {
    0: ["I", "II", "III"],
    1: ["aVR", "aVL", "aVF"],
    2: ["V1", "V2", "V3"],
    3: ["V4", "V5", "V6"],
}

# A real panel is ~18 grid boxes wide (user measurement, confirmed against
# a reference photo 2026-09-05 -- an earlier attempt using ecg-image-kit's
# own --num_columns 4 split, which only gives ~12.5 boxes per column, was
# visibly too zoomed in). Rendered with --num_columns 1 instead (each lead
# gets its own full-duration row at the same fixed px/box scale), then
# cropped to this many boxes ourselves -- see step1_leads.py.
PANEL_WIDTH_GRID_BOXES = 18
GRID_BOX_SECONDS = 0.2  # ecg-image-kit's own standard_values['x_grid_size']
