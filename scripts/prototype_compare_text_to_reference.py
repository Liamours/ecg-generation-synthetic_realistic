import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import prototype_ideal_panel_template as tp

PROJECT_ROOT = Path(__file__).parents[3]
REAL_PATHS = {
    "odd": PROJECT_ROOT / "datasets/panel_templates/odd_template_i_ii_iii_gain10.png",
    "even": PROJECT_ROOT / "datasets/panel_templates/even_template_avr_avl_avf_no_gain.png",
}
REAL_ORIGIN_PITCH = {
    "odd": (-1.0, 38.5, 66.0, 70.0),
    "even": (3.0, 43.0, 74.0, 71.0),
}
PAD_MEASURE = 3
PAD_VISUAL = 20
OUT = PROJECT_ROOT / "inferences" / "ideal_panel_template_prototype" / "compare"


def bbox_dark(gray_arr, thresh=200):
    ys, xs = np.where(gray_arr < thresh)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def measure(style, text, col0, row0, col1, row1):
    real_path = REAL_PATHS[style]
    ox, oy, px, py = REAL_ORIGIN_PITCH[style]
    real_img = Image.open(real_path).convert("L")
    rx0, ry0 = int(ox + col0 * px) - PAD_MEASURE, int(oy + row0 * py) - PAD_MEASURE
    rx1, ry1 = int(ox + col1 * px) + PAD_MEASURE, int(oy + row1 * py) + PAD_MEASURE
    real_crop_measure = real_img.crop((rx0, ry0, rx1, ry1))
    real_bbox = bbox_dark(np.asarray(real_crop_measure))

    box_px = 70.0
    gen_full = tp.build_panel(box_px=box_px, style=style, show_bbox=False).convert("L")
    gmargin = tp.EXTRA_MARGIN_BOXES * box_px
    gx0, gy0 = int(col0 * box_px) - PAD_MEASURE, int(row0 * box_px + gmargin) - PAD_MEASURE
    gx1, gy1 = int(col1 * box_px) + PAD_MEASURE, int(row1 * box_px + gmargin) + PAD_MEASURE
    gen_crop_measure = gen_full.crop((gx0, gy0, gx1, gy1))
    gen_bbox = bbox_dark(np.asarray(gen_crop_measure))

    real_crop_visual = real_img.crop((rx0 - PAD_VISUAL, ry0 - PAD_VISUAL, rx1 + PAD_VISUAL, ry1 + PAD_VISUAL))
    gen_crop_visual = gen_full.crop((gx0 - PAD_VISUAL, gy0 - PAD_VISUAL, gx1 + PAD_VISUAL, gy1 + PAD_VISUAL))

    return real_crop_visual, gen_crop_visual, real_bbox, gen_bbox, (px, py), box_px


def report(style, text, col0, row0, col1, row1):
    real_crop, gen_crop, real_bbox, gen_bbox, (px, py), box_px = measure(style, text, col0, row0, col1, row1)

    target_h = 200
    real_r = real_crop.resize((round(real_crop.width * target_h / real_crop.height), target_h), Image.LANCZOS)
    gen_r = gen_crop.resize((round(gen_crop.width * target_h / gen_crop.height), target_h), Image.LANCZOS)

    w = max(real_r.width, gen_r.width)
    side_by_side = Image.new("L", (w * 2 + 30, target_h + 30), 255)
    side_by_side.paste(real_r, (0, 30))
    side_by_side.paste(gen_r, (w + 30, 30))
    d = ImageDraw.Draw(side_by_side)
    d.text((5, 5), "REAL", fill=0)
    d.text((w + 35, 5), "GENERATED", fill=0)

    diff = ImageChops.difference(real_r.convert("L").resize((w, target_h)), gen_r.convert("L").resize((w, target_h)))
    real_rgb = real_r.convert("RGB").resize((w, target_h))
    gen_rgb = gen_r.convert("RGB").resize((w, target_h))
    overlay = Image.blend(real_rgb, gen_rgb, 0.5)

    name = text.replace("/", "_").replace(" ", "_").replace(".", "")
    side_by_side.save(OUT / f"{style}_{name}_sidebyside.png")
    diff.save(OUT / f"{style}_{name}_diff.png")
    overlay.save(OUT / f"{style}_{name}_overlay.png")

    def to_mm(bbox, pitch_x, pitch_y):
        if bbox is None:
            return None
        return (round((bbox[2] - bbox[0]) * 5 / pitch_x, 2), round((bbox[3] - bbox[1]) * 5 / pitch_y, 2))

    real_wh_mm = to_mm(real_bbox, px, py) if real_bbox else None
    gen_wh_mm = to_mm(gen_bbox, box_px, box_px) if gen_bbox else None
    real_w_px = real_bbox[2] - real_bbox[0] if real_bbox else None
    real_h_px = real_bbox[3] - real_bbox[1] if real_bbox else None
    gen_w_px = gen_bbox[2] - gen_bbox[0] if gen_bbox else None
    gen_h_px = gen_bbox[3] - gen_bbox[1] if gen_bbox else None

    return {
        "text": text,
        "style": style,
        "real_w_px": real_w_px, "real_h_px": real_h_px,
        "gen_w_px": gen_w_px, "gen_h_px": gen_h_px,
        "real_w_mm": real_wh_mm[0] if real_wh_mm else None,
        "real_h_mm": real_wh_mm[1] if real_wh_mm else None,
        "gen_w_mm": gen_wh_mm[0] if gen_wh_mm else None,
        "gen_h_mm": gen_wh_mm[1] if gen_wh_mm else None,
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for style in ("odd", "even"):
        for text, col0, row0, col1, row1 in tp.TEXT_FIELDS[style]:
            rows.append(report(style, text, col0, row0, col1, row1))

    print(f"{'style':6} {'text':20} {'real_h_mm':10} {'gen_h_mm':10} {'d_h_mm':8} {'real_w_mm':10} {'gen_w_mm':10} {'d_w_mm':8}")
    for r in rows:
        dh = round(r["gen_h_mm"] - r["real_h_mm"], 2) if r["real_h_mm"] and r["gen_h_mm"] else None
        dw = round(r["gen_w_mm"] - r["real_w_mm"], 2) if r["real_w_mm"] and r["gen_w_mm"] else None
        print(f"{r['style']:6} {r['text']:20} {str(r['real_h_mm']):10} {str(r['gen_h_mm']):10} {str(dh):8} {str(r['real_w_mm']):10} {str(r['gen_w_mm']):10} {str(dw):8}")
