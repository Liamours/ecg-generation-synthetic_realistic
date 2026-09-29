"""Builds the synthetic-pretraining dataset for the image-direct classifier:
joins the synthetic corpus's manifests' ptbxl_record_id to PTB-XL's
ptbxl_database.csv and emits a normal-vs-abnormal YOLO classification
dataset over the CANONICAL panel images.

Label definition: a panel is "normal" when its record's scp_codes carries
NORM at likelihood >= NORM_LIKELIHOOD_THRESHOLD (80, PTB-XL's own
"confident single-label" convention), regardless of what else is in the
dict; anything else is "abnormal". Not the same as requiring scp_codes to
be exactly {"NORM"}: real PTB-XL records almost always carry an
auxiliary rhythm code alongside a diagnostic one (e.g. "SR", sinus
rhythm) even when genuinely normal -- checked directly against the
2026-09-08 PTB-XL download, requiring exact-set-equality yields only 190
"normal" records out of 21799 (0.9%), an unusable class size for
pretraining; NORM >= 80 yields 8933 (41%), a normal-looking split. PTB-XL
has no junctional-beat class, so this supervises generic normal-vs-
abnormal morphology only -- the PJB task is learned from the real
ekg-757 labels at fine-tune time (see wiki/overview/synthetic_data_flow.md's synthetic
pretraining plan).

Split is by page (all panels of one page land in one split, seed 42,
80/10/10), mirroring the real-classifier's page-level split so a page's
panels can never straddle splits.

Usage:
    .venv/Scripts/python.exe scripts/prepare_pretrain_dataset.py \
        --synthetic-dir ../../datasets/ecg-synthetic_realistic \
        --ptbxl-csv ../../datasets/ptb-xl/ptbxl_database.csv \
        --out-dir ../../datasets/training/pretrain_scp_yolo
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CLASSES = ["normal", "abnormal"]
SPLIT_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
NORM_LIKELIHOOD_THRESHOLD = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--ptbxl-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_scp_codes(raw: str) -> dict[str, float]:
    """scp_codes is stored as a Python dict literal ("{'NORM': 100.0,
    'SR': 0.0}"), not a delimited string -- ast.literal_eval is the
    correct, safe parser for that; an earlier version here split on ","
    then ":" by hand, which left stray "{" / "'" characters on every key
    and meant no code ever matched a clean lookup."""
    return ast.literal_eval(raw)


def label_for_record(scp_codes_raw: str) -> str:
    codes = parse_scp_codes(scp_codes_raw)
    return "normal" if codes.get("NORM", 0.0) >= NORM_LIKELIHOOD_THRESHOLD else "abnormal"


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def main() -> None:
    args = parse_args()
    panels = pd.read_csv(args.synthetic_dir / "labels" / "manifest_panels.csv")
    database = pd.read_csv(args.ptbxl_csv, usecols=["filename_hr", "scp_codes"])
    # the manifest's ptbxl_record_id is the WFDB path ("records500/03000/03085_hr"),
    # which is exactly ptbxl_database.csv's filename_hr column; ecg_id is a plain
    # row number and does not match it
    scp_by_record = dict(zip(database["filename_hr"], database["scp_codes"]))

    unknown = panels[~panels["ptbxl_record_id"].isin(scp_by_record)]
    if len(unknown):
        raise SystemExit(f"{len(unknown)} panels reference records absent from ptbxl_database.csv, e.g. {unknown['ptbxl_record_id'].iloc[0]}")

    panels = panels.copy()
    panels["label"] = panels["ptbxl_record_id"].map(lambda r: label_for_record(scp_by_record[r]))

    rng = random.Random(SEED)
    page_ids = sorted(panels["page_id"].unique().tolist())
    rng.shuffle(page_ids)
    n = len(page_ids)
    n_train, n_val = round(n * SPLIT_FRACS["train"]), round(n * SPLIT_FRACS["val"])
    split_of = {pid: "train" for pid in page_ids[:n_train]}
    split_of.update({pid: "val" for pid in page_ids[n_train:n_train + n_val]})
    split_of.update({pid: "test" for pid in page_ids[n_train + n_val:]})

    panels["split"] = panels["page_id"].map(split_of)
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (args.out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    rows = []
    project_root = args.synthetic_dir.parents[1]  # manifest paths are relative to the research project root ("datasets/...")
    for row in tqdm(panels.itertuples(), total=len(panels), desc="link panels"):
        src = project_root / row.panel_image_path
        label = label_for_record(scp_by_record[row.ptbxl_record_id])
        dst = args.out_dir / row.split / label / f"{row.panel_id}.png"
        if not dst.exists():
            link(src, dst)
        rows.append({"panel_id": row.panel_id, "page_id": row.page_id, "ptbxl_record_id": row.ptbxl_record_id,
                     "label": label, "split": row.split})

    out_csv = args.synthetic_dir / "labels" / "labels_scp.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for split in ("train", "val", "test"):
        counts = panels[panels["split"] == split]["label"].value_counts().to_dict()
        print(f"{split}: {counts}")


if __name__ == "__main__":
    main()
