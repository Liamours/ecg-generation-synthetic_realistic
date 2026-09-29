from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/consolab.ttf"

TEXT_FIELDS = {
    "odd": [
        ("MAC 400", 3.050, 0.150, 5.940, 0.760),
        ("V1.02", 9.590, 0.379, 12.804, 1.121),
        ("I", 4.090, 2.119, 4.733, 2.861),
        ("II", 4.090, 5.609, 5.376, 6.351),
        ("III", 4.090, 9.599, 6.019, 10.341),
        ("Auto", 4.730, 14.169, 7.301, 14.911),
        ("25mm/s", 9.120, 14.169, 12.980, 14.911),
        ("10mm/mV", 12.980, 14.169, 17.494, 14.911),
        ("ADS", 17.580, 14.169, 19.509, 14.911),
        ("For 2030887-001", 2.360, 14.924, 6.550, 15.496),
        ("MDC72942181716008", 6.550, 14.924, 12.640, 15.496),
        ("INNOQ", 12.640, 14.924, 15.083, 15.496),
    ],
    "even": [
        ("GE", 2.930, 0.139, 4.216, 0.881),
        ("10.Sep'.24", 9.080, 0.379, 13.450, 1.121),
        ("09:14", 13.450, 0.379, 16.664, 1.121),
        ("50Hz", 4.840, 14.159, 9.530, 14.901),
        ("0.08-150Hz", 9.530, 14.159, 12.744, 14.901),
        ("80BPM", 14.840, 14.159, 18.054, 14.901),
        ("For 2030887-001", 2.040, 14.844, 5.810, 15.366),
        ("MDC72942181716008", 5.810, 14.844, 11.580, 15.366),
        ("INNOQ", 11.580, 14.844, 14.023, 15.366),
    ],
}

WIDTH_BOXES = 20
HEIGHT_BOXES = 15
EXTRA_MARGIN_BOXES = 0.42

BOUNDARY_DOTS_PER_BOX_H = 9
BOUNDARY_DOTS_PER_BOX_V = 10
INTERIOR_DOTS_PER_BOX = 4

DOT_DIAMETER_FRAC_OF_BOX = 0.036
SS = 4

BOUNDARY_COLOR = (20, 20, 20)
INTERIOR_COLOR = (20, 20, 20)
BORDER_COLOR = (170, 170, 170)
PAPER = (252, 252, 252)

MISSING_LINE_COL_START = 2
MISSING_LINE_COL_FULL_END = 12
MISSING_LINE_COL_HALF = 13
INTERIOR_DOTS_PER_BOX_KEPT_ROWS = 2


def dotted_segment(draw, x0, y0, x1, y1, n_dots, color, radius, from_fraction=0.0, to_fraction=1.0):
    n = n_dots + 1
    for i in range(n + 1):
        t = i / n
        if t < from_fraction or t > to_fraction:
            continue
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        r = radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def draw_dot_grid(draw, x0, y0, box_px, width_boxes, height_boxes, radius):
    missing_line_row = height_boxes - 1
    for col in range(width_boxes + 1):
        x = x0 + col * box_px
        for row in range(height_boxes):
            y0_ = y0 + row * box_px
            y1_ = y0_ + box_px
            to_frac = 1.0
            if row == missing_line_row and MISSING_LINE_COL_START + 1 <= col <= MISSING_LINE_COL_HALF:
                to_frac = 6 / (BOUNDARY_DOTS_PER_BOX_V + 1)
            dotted_segment(draw, x, y0_, x, y1_, BOUNDARY_DOTS_PER_BOX_V, BOUNDARY_COLOR, radius, to_fraction=to_frac)
    for row in range(height_boxes + 1):
        y = y0 + row * box_px
        for col in range(width_boxes):
            x0_ = x0 + col * box_px
            x1_ = x0_ + box_px
            if row == missing_line_row + 1:
                if MISSING_LINE_COL_START <= col <= MISSING_LINE_COL_FULL_END:
                    continue
                if col == MISSING_LINE_COL_HALF:
                    dotted_segment(draw, x0_, y, x1_, y, BOUNDARY_DOTS_PER_BOX_H, BOUNDARY_COLOR, radius, from_fraction=0.5)
                    continue
            dotted_segment(draw, x0_, y, x1_, y, BOUNDARY_DOTS_PER_BOX_H, BOUNDARY_COLOR, radius)

    for col in range(width_boxes):
        for row in range(height_boxes):
            bx, by = x0 + col * box_px, y0 + row * box_px
            row_count = INTERIOR_DOTS_PER_BOX
            if row == missing_line_row and MISSING_LINE_COL_START <= col <= MISSING_LINE_COL_HALF:
                row_count = INTERIOR_DOTS_PER_BOX_KEPT_ROWS
            for i in range(1, INTERIOR_DOTS_PER_BOX + 1):
                for j in range(1, row_count + 1):
                    px = bx + box_px * i / (INTERIOR_DOTS_PER_BOX + 1)
                    py = by + box_px * j / (INTERIOR_DOTS_PER_BOX + 1)
                    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=INTERIOR_COLOR)


def font_matching_height(text, target_h_px):
    size = round(target_h_px)
    font = ImageFont.truetype(FONT_PATH, max(size, 1))
    bbox = font.getbbox(text)
    rendered_h = bbox[3] - bbox[1]
    while rendered_h < target_h_px:
        size += 1
        font = ImageFont.truetype(FONT_PATH, size)
        bbox = font.getbbox(text)
        rendered_h = bbox[3] - bbox[1]
    return font, bbox


BOX_FILL_FRACTION = 0.9


def draw_text_in_box(img, draw, text, x0_px, y0_px, x1_px, y1_px, color, show_bbox=False):
    target_h = y1_px - y0_px
    box_w = round((x1_px - x0_px) * BOX_FILL_FRACTION)
    font, bbox = font_matching_height(text, target_h)
    glyph_w, glyph_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    target_w = min(box_w, glyph_w)
    layer = Image.new("RGBA", (max(1, glyph_w), max(1, glyph_h)), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((-bbox[0], -bbox[1]), text, fill=(*color, 255), font=font)
    layer = layer.resize((max(1, target_w), max(1, glyph_h)), Image.LANCZOS)
    img.paste(layer, (round(x0_px), round(y0_px)), layer)
    if show_bbox:
        draw.rectangle([x0_px, y0_px, x0_px + target_w, y0_px + glyph_h], outline=(255, 0, 0), width=1)


def draw_text_fields(img, draw, x0, y0, box_px, style, show_bbox=False):
    for text, col0, row0, col1, row1 in TEXT_FIELDS[style]:
        draw_text_in_box(
            img, draw, text,
            x0 + col0 * box_px, y0 + row0 * box_px,
            x0 + col1 * box_px, y0 + row1 * box_px,
            BOUNDARY_COLOR, show_bbox=show_bbox,
        )


def build_panel(box_px: float, style: str = "odd", show_bbox: bool = False) -> Image.Image:
    ss_box_px = box_px * SS
    margin_px = EXTRA_MARGIN_BOXES * ss_box_px
    canvas_w = round(WIDTH_BOXES * ss_box_px)
    text_bottom_pad = 0.15 * ss_box_px
    canvas_h = round(HEIGHT_BOXES * ss_box_px + 2 * margin_px + text_bottom_pad)

    img = Image.new("RGB", (canvas_w, canvas_h), PAPER)
    draw = ImageDraw.Draw(img)

    grid_x0, grid_y0 = 0, margin_px
    radius = DOT_DIAMETER_FRAC_OF_BOX * ss_box_px / 2
    draw_dot_grid(draw, grid_x0, grid_y0, ss_box_px, WIDTH_BOXES, HEIGHT_BOXES, radius)

    border_w = max(1, round(SS * 0.4))
    bottom_border_y = grid_y0 + HEIGHT_BOXES * ss_box_px + margin_px - border_w - 1
    draw.line([(0, border_w), (canvas_w, border_w)], fill=BORDER_COLOR, width=border_w)
    draw.line([(0, bottom_border_y), (canvas_w, bottom_border_y)], fill=BORDER_COLOR, width=border_w)

    ICON_X_OFFSET_BOXES = 0.08
    ICON_HEIGHT_BOXES = 1.16
    icon_x0 = grid_x0 + (2 + ICON_X_OFFSET_BOXES) * ss_box_px
    icon_y0 = border_w
    draw.rectangle([icon_x0, icon_y0, icon_x0 + ss_box_px, icon_y0 + ICON_HEIGHT_BOXES * ss_box_px], fill=(0, 0, 0))

    draw_text_fields(img, draw, grid_x0, grid_y0, ss_box_px, style, show_bbox=show_bbox)

    target_size = (round(canvas_w / SS), round(canvas_h / SS))
    return img.resize(target_size, Image.LANCZOS)


if __name__ == "__main__":
    out = Path(__file__).parents[3] / "inferences" / "ideal_panel_template_prototype"
    out.mkdir(parents=True, exist_ok=True)
    for style in ("odd", "even"):
        panel = build_panel(box_px=70.0, style=style, show_bbox=False)
        panel.save(out / f"panel_prototype_{style}.png")
        print("saved", out / f"panel_prototype_{style}.png", panel.size)
