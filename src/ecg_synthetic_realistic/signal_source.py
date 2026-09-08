"""Real PTB-XL signal sourcing for a panel's 3 rows (2026-09-05): one
record per panel, so all 3 leads shown together are physiologically
consistent (the same patient, same recording), not independently random
traces stitched together.
"""
from __future__ import annotations

import random

import numpy as np
import wfdb

from . import config

_RECORD_IDS_CACHE: list[str] | None = None


def _record_ids() -> list[str]:
    """Record ids as PTBXL_DATA_DIR-relative POSIX paths without extension
    (e.g. "records500/00000/00001_hr"), not bare stems: the official
    PhysioNet distribution nests records under records500/<folder-per-
    1000>/, not flat in PTBXL_DATA_DIR itself (confirmed against the
    2026-09-08 re-download -- a bare-stem glob found 0 files there)."""
    global _RECORD_IDS_CACHE
    if _RECORD_IDS_CACHE is None:
        _RECORD_IDS_CACHE = [
            f.relative_to(config.PTBXL_DATA_DIR).with_suffix("").as_posix()
            for f in config.PTBXL_DATA_DIR.rglob("*_hr.hea")
        ]
        if not _RECORD_IDS_CACHE:
            raise FileNotFoundError(f"no *_hr.hea files under {config.PTBXL_DATA_DIR}")
    return _RECORD_IDS_CACHE


def load_panel_signals(rng: random.Random, lead_names: list[str], duration_samples: int) -> tuple[str, dict[str, np.ndarray], float]:
    """Picks one random PTB-XL record and a random `duration_samples`-long
    window from it, returning that window for each of `lead_names`.
    PTB-XL's .hea names augmented limb leads in all caps (AVR/AVL/AVF);
    the clinical mixed-case form (aVR/aVL/aVF) is matched case-
    insensitively -- same gotcha found and fixed in step1_leads.py."""
    record_id = rng.choice(_record_ids())
    record = wfdb.rdrecord(str(config.PTBXL_DATA_DIR / record_id))
    sig_names_upper = [n.upper() for n in record.sig_name]

    total_samples = record.p_signal.shape[0]
    if duration_samples > total_samples:
        raise ValueError(f"duration_samples ({duration_samples}) exceeds record length ({total_samples})")
    start = rng.randint(0, total_samples - duration_samples)

    signals = {}
    for name in lead_names:
        if name.upper() not in sig_names_upper:
            raise RuntimeError(f"{record_id}: lead {name!r} not in wfdb sig_name {record.sig_name}")
        col = sig_names_upper.index(name.upper())
        signals[name] = np.asarray(record.p_signal[start:start + duration_samples, col], dtype=np.float32)

    return record_id, signals, float(record.fs)
