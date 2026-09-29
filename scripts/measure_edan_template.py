"""Measure grid pitch, tilt, and text-field positions on a clean template reference image.

Usage:
    python scripts/measure_edan_template.py --image <png> --texts <ocr.json> --out-dir <dir> [--clean-window X0 X1 Y0 Y1]

Writes grid.json, text_positions.json, and overlay.png to out-dir. `--clean-window` is a
grid region with no text or labels, used to fit the grid lines.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

TEXT_THRESHOLD = 60
TEXT_PAD = 2
BAND = 150
MAX_SHIFT_PX = 3


def line_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    return 255.0 - np.median(gray, axis=axis)


def fit_lines(profile: np.ndarray, lo: int, hi: int) -> dict:
    peaks, _ = find_peaks(profile, distance=3, prominence=3)
    peaks = peaks[(peaks >= lo) & (peaks < hi)]
    pitch = float((peaks[-1] - peaks[0]) / (len(peaks) - 1))
    for _ in range(4):
        k = np.round((peaks - peaks[0]) / pitch).astype(int)
        keep = np.ones(len(peaks), bool)
        for kk in np.unique(k):
            same = np.where(k == kk)[0]
            keep[same] = False
            keep[same[np.argmax(profile[peaks[same]])]] = True
        b, a = np.polyfit(k[keep], peaks[keep], 1)
        resid = peaks - (a + b * k)
        keep &= np.abs(resid) < 1.5
        b, a = np.polyfit(k[keep], peaks[keep], 1)
        pitch = float(b)
    k, peaks = k[keep], peaks[keep]
    amps = profile[peaks]
    means = [amps[k % 5 == ph].mean() for ph in range(5)]
    phase = int(np.argmax(means))
    others = np.mean([m for i, m in enumerate(means) if i != phase])
    half = len(k) // 2
    left, right = np.polyfit(k[:half], peaks[:half], 1)[0], np.polyfit(k[half:], peaks[half:], 1)[0]
    resid = peaks - (a + pitch * k)
    return {
        "minor_pitch_px": round(pitch, 3),
        "major_pitch_px": round(5 * pitch, 3),
        "major_line_offset_px": round((a + pitch * ((phase - k[0]) % 5 + k[0])) % (5 * pitch), 2),
        "n_minor_lines": int(len(k)),
        "major_line_contrast": round(float(means[phase] / others), 2),
        "pitch_first_half_px": round(float(left), 3),
        "pitch_second_half_px": round(float(right), 3),
        "residual_max_px": round(float(np.abs(resid).max()), 2),
    }


def band_shift(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    cc = np.correlate(a, b, "full")
    mid = len(a) - 1
    i = mid - MAX_SHIFT_PX + int(np.argmax(cc[mid - MAX_SHIFT_PX:mid + MAX_SHIFT_PX + 1]))
    y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
    return i - (len(a) - 1) + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)


def tilt_deg(gray: np.ndarray, win: tuple) -> dict:
    x0, x1, y0, y1 = win
    top, bottom = gray[y0:y0 + BAND], gray[y1 - BAND:y1]
    left, right = gray[y0:y1, x0:x0 + BAND], gray[y0:y1, x1 - BAND:x1]
    dy = (y1 - BAND) - y0
    dx = (x1 - BAND) - x0
    vertical = math.degrees(math.atan2(band_shift(line_profile(top, 0), line_profile(bottom, 0)), dy))
    horizontal = math.degrees(math.atan2(band_shift(line_profile(left, 1), line_profile(right, 1)), dx))
    return {"vertical_lines_deg": round(vertical, 3), "horizontal_lines_deg": round(horizontal, 3)}


def tighten(gray: np.ndarray, box: list) -> list:
    x0, y0, x1, y1 = box
    crop = gray[max(y0 - TEXT_PAD, 0):y1 + TEXT_PAD, max(x0 - TEXT_PAD, 0):x1 + TEXT_PAD] < TEXT_THRESHOLD
    cols, rows = np.where(crop.sum(0) >= 2)[0], np.where(crop.sum(1) >= 2)[0]
    if len(cols) == 0 or len(rows) == 0:
        return box
    ox, oy = max(x0 - TEXT_PAD, 0), max(y0 - TEXT_PAD, 0)
    return [int(ox + cols.min()), int(oy + rows.min()), int(ox + cols.max()) + 1, int(oy + rows.max()) + 1]


def text_fields(gray: np.ndarray, x_line: dict, y_line: dict, texts: list) -> list[dict]:
    mx, my = x_line["minor_pitch_px"], y_line["minor_pitch_px"]
    fields = []
    for i, t in enumerate(sorted(texts, key=lambda t: (t["bbox"][1] // 20, t["bbox"][0]))):
        x0, y0, x1, y1 = box = tighten(gray, t["bbox"])
        fields.append({
            "id": i,
            "ocr_text": t["text"],
            "bbox_px": box,
            "size_mm": [round((x1 - x0) / mx, 2), round((y1 - y0) / my, 2)],
            "grid_box": [
                round(((x0 + x1) / 2 - x_line["major_line_offset_px"]) / x_line["major_pitch_px"], 2),
                round(((y0 + y1) / 2 - y_line["major_line_offset_px"]) / y_line["major_pitch_px"], 2),
            ],
        })
    return fields


def draw(gray: np.ndarray, fields: list[dict], out: Path) -> None:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for f in fields:
        x0, y0, x1, y1 = f["bbox_px"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(img, str(f["id"]), (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    cv2.imwrite(str(out), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True, help="JSON list of {text, bbox: [x0, y0, x1, y1]} from an OCR pass")
    ap.add_argument("--clean-window", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), help="grid region with no text or labels")
    args = ap.parse_args()
    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    texts = json.loads(args.texts.read_text(encoding="utf-8"))
    h, w = gray.shape
    x0, x1, y0, y1 = args.clean_window or (0, w, 0, h)
    x_line = fit_lines(line_profile(gray[y0:y1], 0), 0, w)
    y_line = fit_lines(line_profile(gray[:, x0:x1], 1), y0, y1)
    fields = text_fields(gray, x_line, y_line, texts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = {"image": args.image.name, "size_px": [gray.shape[1], gray.shape[0]],
            "vertical_lines_x": x_line, "horizontal_lines_y": y_line, "clean_window_xyxy": [x0, x1, y0, y1], "tilt": tilt_deg(gray, (x0, x1, y0, y1))}
    (args.out_dir / "grid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    (args.out_dir / "text_positions.json").write_text(json.dumps(fields, indent=1), encoding="utf-8")
    draw(gray, fields, args.out_dir / "overlay.png")
    print(json.dumps(grid, indent=1))
    print(f"{len(fields)} text fields")


if __name__ == "__main__":
    main()
