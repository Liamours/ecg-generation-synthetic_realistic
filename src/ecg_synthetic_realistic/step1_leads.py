"""Step 1 of the synthetic plan (wiki/plans/plan.md): render PTB-XL records to
paper via ecg-image-kit, then crop out individual, per-lead paper-strip
images with their exact digital signal as ground truth.

Rendered with `--num_columns 1`: each of the 12 leads gets its own row
spanning its full recorded duration, at a fixed, resolution-determined
pixel-per-grid-box scale (`x_grid`/`y_grid` in the sidecar JSON -- ~39.37px
per 5mm box at resolution=200, confirmed empirically 2026-09-05 and
independent of --num_columns). A real panel is `config.PANEL_WIDTH_GRID_BOXES`
(18) boxes wide -- an earlier attempt using ecg-image-kit's own
`--num_columns 4` column split only gave ~12.5 boxes per lead (a 2.5s
segment), visibly too zoomed in against a reference photo. Panel width is
cropped ourselves instead of relying on the generator's own column split,
so it's independent of however many seconds that happens to correspond to.

Real interface confirmed directly 2026-09-05 (not assumed from
ecg-image-kit's docs): `--lead_bbox --store_config 2` writes a
`<record>-<frame>.json` next to the render, with one entry per lead under
`"leads"`: `lead_name`, `lead_bounding_box` (4 corners, **[row, col] i.e.
[y, x]**, not [x, y]), `start_sample`, `end_sample`. With `--num_columns 1`
every lead's (start_sample, end_sample) is (0, full_length) -- there's no
column grouping to infer from the JSON any more, so leads are grouped into
panels by name instead, via `config.PANEL_ROW_ORDER`.

Rendered with `--calibration_pulse 1` (the generator's real to-scale
per-lead calibration step pulse, drawn once at the start of every lead's
own trace -- confirmed against a real panel photo 2026-09-05, not a
decorative box: it is a genuine 1mV rectangular step at the printed gain)
and `--random_bw 1` (the vendored ecg_plot.py already has a project-specific
patch drawing a dashed-major/dotted-minor grid "matching this project's
real MAC 400 paper look" -- bw mode gives it a gray/black tone instead of
the default red palette, which is what real MAC 400 photos actually show).
The separate solid black header icon seen before "MAC 400" on real panels
is not a calibration mark and is not rendered here -- it is panel-level,
drawn in step 2. `--remove_lead_names` disables the generator's own
lead-name text (it draws one by default) since step 2 draws its own,
correctly positioned label -- without this flag the two overlap and
render as garbled/doubled text (found 2026-09-05, comparing output
against a reference photo).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from PIL import Image

from . import config

CROP_MARGIN_FRAC = 0.35  # vertical padding around a lead's tight trace bbox, as a fraction of its own height

LEAD_TO_PANEL = {
    name: column_index
    for column_index, names in config.PANEL_ROW_ORDER.items()
    for name in names
}


@dataclass
class LeadCrop:
    record_id: str
    column_index: int
    lead_name: str
    lead_png_path: Path
    signal_npy_path: Path
    bbox_xyxy: tuple[int, int, int, int]  # in the original full-sheet render, for provenance
    start_sample: int
    end_sample: int
    sampling_frequency: float
    x_grid: float
    y_grid: float


def _run_generator(input_dir: Path, output_dir: Path, resolution: int, seed: int) -> None:
    python = sys.executable
    cmd = [
        python,
        str(config.ECG_IMAGE_KIT_BATCH_SCRIPT),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "--num_columns", "1",
        "--lead_bbox",
        "--store_config", "2",
        "--calibration_pulse", "1",
        "--random_bw", "1",
        "--remove_lead_names",
        "-r", str(resolution),
        "--seed", str(seed),
    ]
    result = subprocess.run(cmd, cwd=config.ECG_IMAGE_KIT_GENERATOR_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ecg-image-kit generator failed:\n{result.stdout}\n{result.stderr}")


def _corners_to_xyxy(corners: dict) -> tuple[int, int, int, int]:
    ys = [pt[0] for pt in corners.values()]
    xs = [pt[1] for pt in corners.values()]
    return min(xs), min(ys), max(xs), max(ys)


def generate_leads_for_records(
    record_ids: list[str],
    output_root: Path = config.DEFAULT_OUTPUT_ROOT,
    resolution: int = 200,
    seed: int = 0,
    tmp_dir: Path | None = None,
) -> list[LeadCrop]:
    leads_dir = output_root / "leads"
    leads_dir.mkdir(parents=True, exist_ok=True)

    own_tmp = tmp_dir is None
    tmp_dir = tmp_dir or (output_root / "_tmp_render")
    render_in = tmp_dir / "in"
    render_out = tmp_dir / "out"
    render_in.mkdir(parents=True, exist_ok=True)
    render_out.mkdir(parents=True, exist_ok=True)

    for record_id in record_ids:
        for ext in (".hea", ".dat"):
            src = config.PTBXL_DATA_DIR / f"{record_id}{ext}"
            if not src.exists():
                raise FileNotFoundError(f"missing PTB-XL source file: {src}")
            shutil.copy2(src, render_in / f"{record_id}{ext}")

    _run_generator(render_in, render_out, resolution=resolution, seed=seed)

    results: list[LeadCrop] = []
    for record_id in record_ids:
        json_path = render_out / f"{record_id}-0.json"
        png_path = render_out / f"{record_id}-0.png"
        if not json_path.exists() or not png_path.exists():
            raise RuntimeError(f"generator did not produce output for {record_id} (expected {json_path})")

        meta = json.loads(json_path.read_text())
        grid_px = meta["x_grid"]
        panel_width_px = round(config.PANEL_WIDTH_GRID_BOXES * grid_px)
        panel_duration_samples = round(config.PANEL_WIDTH_GRID_BOXES * config.GRID_BOX_SECONDS * meta["sampling_frequency"])

        sheet = Image.open(png_path)
        wfdb_record = wfdb.rdrecord(str(config.PTBXL_DATA_DIR / record_id))
        sig_names = list(wfdb_record.sig_name)
        # PTB-XL's .hea names augmented limb leads in all caps (AVR/AVL/AVF);
        # ecg-image-kit prints the clinical mixed-case form (aVR/aVL/aVF).
        sig_names_upper = [n.upper() for n in sig_names]

        seen_panels: dict[int, list[str]] = {}
        for entry in meta["leads"]:
            # A stray extra entry with no "lead_name" key shows up alongside
            # the 12 real ones even at --num_columns 1 (found 2026-09-05,
            # cause not tracked down) -- skip it like any non-standard lead.
            lead_name = entry.get("lead_name")
            if lead_name not in LEAD_TO_PANEL:
                continue
            column_index = LEAD_TO_PANEL[lead_name]
            seen_panels.setdefault(column_index, []).append(lead_name)

            x0, y0, x1_full, y1 = _corners_to_xyxy(entry["lead_bounding_box"])
            x1 = min(x1_full, x0 + panel_width_px)
            pad = int((y1 - y0) * CROP_MARGIN_FRAC)
            crop_box = (x0, max(0, y0 - pad), x1, min(sheet.height, y1 + pad))
            crop = sheet.crop(crop_box)

            if lead_name.upper() not in sig_names_upper:
                raise RuntimeError(f"{record_id}: lead {lead_name!r} not in wfdb sig_name {sig_names}")
            lead_col = sig_names_upper.index(lead_name.upper())
            start, end = entry["start_sample"], min(entry["start_sample"] + panel_duration_samples, entry["end_sample"])
            signal_segment = np.asarray(wfdb_record.p_signal[start:end, lead_col], dtype=np.float32)

            stem = f"{record_id}-c{column_index}-{lead_name}"
            lead_png_path = leads_dir / f"{stem}.png"
            signal_npy_path = leads_dir / f"{stem}.npy"
            crop.save(lead_png_path)
            np.save(signal_npy_path, signal_segment)

            results.append(
                LeadCrop(
                    record_id=record_id,
                    column_index=column_index,
                    lead_name=lead_name,
                    lead_png_path=lead_png_path,
                    signal_npy_path=signal_npy_path,
                    bbox_xyxy=crop_box,
                    start_sample=start,
                    end_sample=end,
                    sampling_frequency=meta["sampling_frequency"],
                    x_grid=meta["x_grid"],
                    y_grid=meta["y_grid"],
                )
            )

        for column_index, names in seen_panels.items():
            if len(names) != config.LEADS_PER_PANEL:
                raise RuntimeError(f"{record_id} panel {column_index}: expected {config.LEADS_PER_PANEL} leads, got {names}")

    if own_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results
