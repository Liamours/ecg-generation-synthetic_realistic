"""Panel variant sampling (2026-09-05): combines the independent variation
axes in template.py -- the "odd"/"even" style (mutually exclusive, see
template.py's module docstring) and lead grouping -- for generating a
varied batch. Each axis stays independently callable on
`build_panel_template` directly; this module is just a convenience.
"""
from __future__ import annotations

import random

STANDARD_ROW_LABEL_SETS = [
    ["I", "II", "III"],
    ["aVR", "aVL", "aVF"],
    ["V1", "V2", "V3"],
    ["V4", "V5", "V6"],
]

# The column-type classifier's 4 classes (2026-09-05) -- one per standard
# group, keyed by its first lead so any of the 4 STANDARD_ROW_LABEL_SETS
# entries maps to exactly one type.
COLUMN_TYPE_BY_FIRST_LEAD = {
    "I": "romanic",
    "aVR": "aV",
    "V1": "first_v",
    "V4": "second_v",
}

# "even" prints no gain field, so "correlated" isn't meaningful for it by
# default -- still offered here at low weight since nothing rules it out
# as a real possibility, just not the expected case.
CALIBRATION_STEP_MODES_FOR_EVEN = ["free", "free", "none", "correlated"]


def column_type_for_labels(row_labels: list[str]) -> str:
    return COLUMN_TYPE_BY_FIRST_LEAD[row_labels[0]]


def sample_style_variant(rng: random.Random) -> dict:
    """Style + calibration-step-mode only, no row_labels -- for callers
    (page.py) that decide row_labels themselves, e.g. to guarantee all 4
    groups appear on a page."""
    style = rng.choice(["odd", "even"])
    if style == "odd":
        return {"style": "odd", "calibration_step_mode": "correlated"}
    return {"style": "even", "calibration_step_mode": rng.choice(CALIBRATION_STEP_MODES_FOR_EVEN)}


def sample_panel_variant(rng: random.Random) -> dict:
    """Returns kwargs ready to pass to `template.build_panel_template`
    (plus `rng`, `grid_px`, `record_ref`, which the caller still supplies)."""
    row_labels = rng.choice(STANDARD_ROW_LABEL_SETS)
    return {"row_labels": row_labels, **sample_style_variant(rng)}
