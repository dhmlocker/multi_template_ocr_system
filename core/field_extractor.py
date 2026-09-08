from __future__ import annotations

import re
from typing import Any

from .types import FieldResult, OCRItem


def _rect_iou(a: tuple[float,float,float,float], b: tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FieldExtractor:
    @staticmethod
    def extract(template_config: dict[str, Any], ocr_items: list[OCRItem], image_size: tuple[int,int]) -> list[FieldResult]:
        width, height = image_size
        ref_w = float(template_config.get("reference_width") or width)
        ref_h = float(template_config.get("reference_height") or height)
        sx = width / ref_w if ref_w else 1.0
        sy = height / ref_h if ref_h else 1.0
        results: list[FieldResult] = []
        for field in template_config.get("fields", []):
            name = str(field.get("field_name", ""))
            bbox = field.get("bbox", [0,0,0,0])
            roi = [float(bbox[0])*sx, float(bbox[1])*sy, float(bbox[2])*sx, float(bbox[3])*sy]
            rx1, ry1, rx2, ry2 = roi
            mode = str(field.get("match_mode", "center"))
            threshold = float(field.get("iou_threshold", 0.1))
            matched: list[tuple[int, OCRItem]] = []
            for idx, item in enumerate(ocr_items):
                cx, cy = item.center
                hit = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                if not hit and mode in {"iou", "center_or_iou"}:
                    hit = _rect_iou((rx1,ry1,rx2,ry2), item.bounds) >= threshold
                if hit:
                    matched.append((idx, item))
            matched.sort(key=lambda pair: (pair[1].bounds[1], pair[1].bounds[0]))
            value = " ".join(item.text.strip() for _, item in matched if item.text.strip())
            confidence = sum(item.confidence for _, item in matched) / len(matched) if matched else 0.0
            validation = "未识别"
            if matched:
                validation = "已识别"
                if any(token in name for token in ("日期", "时间")):
                    validation = "格式正常" if re.search(r"\d{2,4}[-/.年]\d{1,2}", value) else "需复核"
                elif any(token in name for token in ("金额", "税额", "合计")):
                    validation = "格式正常" if re.search(r"\d+[.，,]?\d*", value) else "需复核"
                elif any(token in name for token in ("号码", "税号", "代码")):
                    validation = "格式正常" if len(re.sub(r"\s+", "", value)) >= 6 else "需复核"
            results.append(FieldResult(
                field_name=name,
                value=value,
                missing=not bool(matched),
                confidence=round(float(confidence), 4),
                validation=validation,
                source_indices=[idx for idx, _ in matched],
                roi=roi,
            ))
        return results
