from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import OCRItem, FieldResult

# 中文字体路径（Windows 微软雅黑）
_FONT_CANDIDATES = [
    Path('C:/Windows/Fonts/msyh.ttc'),
    Path('C:/Windows/Fonts/simhei.ttf'),
    Path('C:/Windows/Fonts/simsun.ttc'),
    Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'),
    Path('/System/Library/Fonts/PingFang.ttc'),
]


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in _FONT_CANDIDATES:
        if fp.exists():
            try:
                return ImageFont.truetype(str(fp), size)
            except Exception:
                continue
    return ImageFont.load_default()


# 调色板：每个 OCR 框用不同颜色（接近官方多彩风格）
_COLORS = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (155, 89, 182),
    (241, 196, 15), (230, 126, 34), (26, 188, 156), (52, 73, 94),
    (236, 64, 120), (41, 128, 185), (39, 174, 96), (142, 68, 173),
    (243, 156, 18), (211, 84, 0), (22, 160, 133), (44, 62, 80),
]


def _color(idx: int) -> tuple[int, int, int]:
    return _COLORS[idx % len(_COLORS)]


def _fit_text(text: str, draw: ImageDraw.ImageDraw, font_path: Path | None, box_w: int, box_h: int, base_size: int) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
    """Fit a label inside its OCR box; never place it outside the polygon."""
    for size in range(max(8, base_size), 7, -1):
        font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default()
        candidate = str(text).replace('\n', ' ').strip()
        while candidate:
            bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=1)
            if bbox[2] - bbox[0] <= max(4, box_w - 4) and bbox[3] - bbox[1] <= max(4, box_h - 4):
                return candidate, font, size
            candidate = candidate[:-2] + '…' if len(candidate) > 2 else ''
    return '…' if text else '', _get_font(8), 8


def _font_path() -> Path | None:
    return next((path for path in _FONT_CANDIDATES if path.exists()), None)


def draw_ocr_boxes(image: np.ndarray, items: list[OCRItem]) -> np.ndarray:
    """在原图上画彩色检测框，并把识别文字放在对应框内部。"""
    if not items:
        return image.copy()

    h, w = image.shape[:2]
    # 根据图片大小选择字体大小
    font_size = max(12, int(min(h, w) / 80))
    font_path = _font_path()

    # OpenCV BGR -> PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for idx, item in enumerate(items):
        color = _color(idx)
        pts = np.asarray(item.box, dtype=np.int32).reshape((-1, 2))
        # 画多边形框
        pil_pts = [tuple(p) for p in pts]
        draw.polygon(pil_pts, outline=color, width=2)
        # 标注文本：严格限制在 OCR 框内部，不再绘制到框上方
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        box_w = max(8, int(pts[:, 0].max() - pts[:, 0].min()))
        box_h = max(8, int(pts[:, 1].max() - pts[:, 1].min()))
        text, font, _ = _fit_text(str(item.text), draw, font_path, box_w, box_h, max(8, min(font_size, box_h - 4)))
        if text:
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x1 + max(2, (box_w - tw) // 2)
            ty = y1 + max(1, (box_h - th) // 2)
            draw.text((tx, ty), text, fill=color, font=font, stroke_width=1, stroke_fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_roi_boxes(image: np.ndarray, fields: list[FieldResult]) -> np.ndarray:
    canvas = image.copy()
    for idx, field in enumerate(fields):
        if not field.roi:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in field.roi]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
        label = field.field_name
        # ROI 字段标签也限制在字段框内部
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_path = _font_path()
        label, font, _ = _fit_text(label, draw, font_path, max(8, x2 - x1), max(8, y2 - y1), max(8, int(min(canvas.shape[:2]) / 90)))
        if label:
            bbox = draw.textbbox((0, 0), label, font=font, stroke_width=1)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x1 + max(2, ((x2 - x1) - tw) // 2)
            ty = y1 + max(1, ((y2 - y1) - th) // 2)
            draw.text((tx, ty), label, fill=(0, 140, 255), font=font, stroke_width=1, stroke_fill=(255, 255, 255))
        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def draw_all(image: np.ndarray, items: list[OCRItem], fields: list[FieldResult]) -> np.ndarray:
    return draw_roi_boxes(draw_ocr_boxes(image, items), fields)


def make_side_by_side(image: np.ndarray, annotated: np.ndarray, panel_width: int = 900) -> np.ndarray:
    """Create the reference-style original/annotated comparison image."""
    def fit(panel: np.ndarray) -> np.ndarray:
        scale = panel_width / max(panel.shape[1], 1)
        return cv2.resize(panel, (panel_width, max(1, int(panel.shape[0] * scale))))

    left, right = fit(image), fit(annotated)
    height = max(left.shape[0], right.shape[0])
    canvas = np.full((height, panel_width * 2 + 24, 3), 255, dtype=np.uint8)
    canvas[:left.shape[0], :panel_width] = left
    canvas[:right.shape[0], panel_width + 24:] = right
    cv2.line(canvas, (panel_width + 12, 0), (panel_width + 12, height), (220, 220, 220), 2)
    return canvas


def make_original_white_result_pair(image: np.ndarray, items: list[OCRItem], panel_width: int = 900) -> np.ndarray:
    """Create the reference layout: original source on the left, white OCR on the right."""
    white_result = draw_white_ocr_canvas(image.shape[:2], items)
    return make_side_by_side(image, white_result, panel_width=panel_width)


def draw_white_ocr_canvas(image_shape: tuple[int, int], items: list[OCRItem]) -> np.ndarray:
    """Render OCR boxes and text on a clean white document canvas like the official demo."""
    height, width = int(image_shape[0]), int(image_shape[1])
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_size = max(13, int(min(height, width) / 90))
    font_path = _font_path()
    for idx, item in enumerate(items):
        color = _color(idx)
        pts = np.asarray(item.box, dtype=np.int32).reshape((-1, 2))
        draw.polygon([tuple(p) for p in pts], outline=color, width=max(1, font_size // 8))
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        box_w = max(8, int(pts[:, 0].max() - pts[:, 0].min()))
        box_h = max(8, int(pts[:, 1].max() - pts[:, 1].min()))
        text, font, _ = _fit_text(f'{item.text} [{item.confidence:.2f}]', draw, font_path, box_w, box_h, max(8, min(font_size, box_h - 4)))
        if text:
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x1 + max(2, (box_w - tw) // 2), y1 + max(1, (box_h - th) // 2)), text, fill=color, font=font, stroke_width=1, stroke_fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
