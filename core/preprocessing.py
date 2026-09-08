from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class ImageQuality:
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    skew_degrees: float
    quality_score: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_float(value: float) -> float:
    return float(round(float(value), 4))


def estimate_quality(image: np.ndarray) -> ImageQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(30, min(gray.shape) // 5), minLineLength=max(40, gray.shape[1] // 8), maxLineGap=12)
    angles: list[float] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = [int(x) for x in line]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) <= 15:
                angles.append(float(angle))
    skew = float(np.median(angles)) if angles else 0.0
    warnings: list[str] = []
    if blur < 55:
        warnings.append('图像可能偏模糊，建议补光或重新拍摄。')
    if brightness < 65:
        warnings.append('图像整体偏暗，已建议使用亮度增强。')
    elif brightness > 210:
        warnings.append('图像可能过曝，部分文字细节会丢失。')
    if contrast < 28:
        warnings.append('前景与背景对比度较低。')
    if abs(skew) > 3:
        warnings.append(f'检测到约 {abs(skew):.1f}° 倾斜，建议进行校正。')
    blur_score = min(1.0, blur / 350.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - 145.0) / 145.0)
    contrast_score = min(1.0, contrast / 75.0)
    skew_score = max(0.0, 1.0 - abs(skew) / 15.0)
    score = 100.0 * (0.40 * blur_score + 0.25 * brightness_score + 0.25 * contrast_score + 0.10 * skew_score)
    return ImageQuality(
        width=int(image.shape[1]), height=int(image.shape[0]), blur_score=_safe_float(blur),
        brightness=_safe_float(brightness), contrast=_safe_float(contrast),
        skew_degrees=_safe_float(skew), quality_score=_safe_float(score), warnings=warnings,
    )


def enhance_for_ocr(image: np.ndarray, mode: str = 'balanced') -> np.ndarray:
    """Apply conservative enhancement without changing the document geometry."""
    if mode in {'off', '原图', 'none'}:
        return image.copy()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0 if mode == 'balanced' else 3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    if mode == 'strong':
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
    return enhanced


def preprocessing_summary(original: np.ndarray, processed: np.ndarray) -> dict:
    before = estimate_quality(original)
    after = estimate_quality(processed)
    return {'before': before.to_dict(), 'after': after.to_dict(), 'changed': not np.array_equal(original, processed)}
