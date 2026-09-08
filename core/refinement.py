from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import OCRItem


def refine_low_confidence_items(
    image: np.ndarray,
    items: list[OCRItem],
    engine: Any,
    threshold: float = 0.72,
    max_regions: int = 8,
    padding: int = 8,
) -> tuple[list[OCRItem], dict[str, Any]]:
    """Re-recognize difficult text crops and keep only stronger candidates."""
    height, width = image.shape[:2]
    candidates = sorted(
        ((idx, item) for idx, item in enumerate(items) if item.confidence < threshold),
        key=lambda pair: pair[1].confidence,
    )[:max_regions]
    refined = list(items)
    attempted = 0
    accepted = 0
    for parent_index, item in candidates:
        x1, y1, x2, y2 = item.bounds
        ix1 = max(0, int(x1) - padding)
        iy1 = max(0, int(y1) - padding)
        ix2 = min(width, int(x2) + padding)
        iy2 = min(height, int(y2) + padding)
        if ix2 - ix1 < 8 or iy2 - iy1 < 8:
            continue
        crop = image[iy1:iy2, ix1:ix2]
        attempted += 1
        try:
            local_items = engine.recognize(crop)
        except Exception:
            continue
        if not local_items:
            continue
        best = max(local_items, key=lambda value: value.confidence)
        if best.confidence <= item.confidence or not best.text.strip():
            continue
        global_box = [[float(point[0] + ix1), float(point[1] + iy1)] for point in best.box]
        refined[parent_index] = OCRItem(
            box=global_box,
            text=best.text,
            confidence=best.confidence,
            source="roi_refine",
            parent_index=parent_index,
        )
        accepted += 1
    return refined, {
        "attempted": attempted,
        "accepted": accepted,
        "threshold": threshold,
        "max_regions": max_regions,
    }
