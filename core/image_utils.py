from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def decode_image(source: Any) -> np.ndarray:
    """Decode supported image input to OpenCV BGR ndarray."""
    if isinstance(source, np.ndarray):
        arr = source.copy()
    elif isinstance(source, Image.Image):
        rgb = np.asarray(source.convert('RGB'))
        arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(source, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(source), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f'image file not found: {path}')
        # np.fromfile handles non-ASCII Windows paths better than cv2.imread.
        buf = np.fromfile(str(path), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if arr is None or arr.size == 0:
        raise ValueError('Invalid or unsupported image data')
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f'Unsupported image shape: {arr.shape}')
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
