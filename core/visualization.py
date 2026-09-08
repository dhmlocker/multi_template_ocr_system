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


def draw_ocr_boxes(image: np.ndarray, items: list[OCRItem]) -> np.ndarray:
    """在原图上画彩色检测框并标注识别文本（支持中文，接近官方可视化效果）。"""
    if not items:
        return image.copy()

    h, w = image.shape[:2]
    # 根据图片大小选择字体大小
    font_size = max(12, int(min(h, w) / 80))
    font = _get_font(font_size)

    # OpenCV BGR -> PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for idx, item in enumerate(items):
        color = _color(idx)
        pts = np.asarray(item.box, dtype=np.int32).reshape((-1, 2))
        # 画多边形框
        pil_pts = [tuple(p) for p in pts]
        draw.polygon(pil_pts, outline=color, width=2)
        # 标注文本：在框上方画白底文字
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        text = str(item.text)
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * font_size, font_size
        ty = max(0, y1 - th - 3)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 1), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_roi_boxes(image: np.ndarray, fields: list[FieldResult]) -> np.ndarray:
    canvas = image.copy()
    for idx, field in enumerate(fields):
        if not field.roi:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in field.roi]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
        label = field.field_name
        # ROI 标签用中文，用 PIL 画
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = _get_font(max(12, int(min(canvas.shape[:2]) / 80)))
        try:
            bbox = draw.textbbox((x1, y1), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 12, 14
        ty = max(0, y1 - th - 3)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=(0, 140, 255))
        draw.text((x1 + 3, ty + 1), label, fill=(255, 255, 255), font=font)
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


def draw_white_ocr_canvas(image_shape: tuple[int, int], items: list[OCRItem]) -> np.ndarray:
    """Render OCR boxes and text on a clean white document canvas like the official demo."""
    height, width = int(image_shape[0]), int(image_shape[1])
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_size = max(13, int(min(height, width) / 90))
    font = _get_font(font_size)
    for idx, item in enumerate(items):
        color = _color(idx)
        pts = np.asarray(item.box, dtype=np.int32).reshape((-1, 2))
        draw.polygon([tuple(p) for p in pts], outline=color, width=max(1, font_size // 8))
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        text = f'{item.text}  [{item.confidence:.2f}]'
        draw.text((x1 + 3, max(0, y1 - font_size - 2)), text, fill=color, font=font)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
