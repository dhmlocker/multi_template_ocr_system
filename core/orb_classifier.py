from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .types import ClassificationResult


_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


class ORBClassifier:
    def __init__(self, reference_dir: str | Path, nfeatures: int = 1200):
        self.reference_dir = Path(reference_dir)
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._index: dict[str, list[np.ndarray]] = {}

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        """兼容中文路径读图。"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _descriptors(self, image: np.ndarray) -> np.ndarray | None:
        if image is None:
            return None
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, desc = self.orb.detectAndCompute(gray, None)
        return desc

    def build_index(self, reference_dir: str | Path | None = None) -> int:
        if reference_dir is not None:
            self.reference_dir = Path(reference_dir)
        if not self.reference_dir.exists():
            raise FileNotFoundError(f'ORB reference directory not found: {self.reference_dir}')
        index: dict[str, list[np.ndarray]] = {}
        for class_dir in sorted(p for p in self.reference_dir.iterdir() if p.is_dir()):
            descriptors: list[np.ndarray] = []
            for path in sorted(class_dir.rglob('*')):
                if path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                img = self._read_image(path)
                desc = self._descriptors(img)
                if desc is not None and len(desc):
                    descriptors.append(desc)
            if descriptors:
                index[class_dir.name] = descriptors
        self._index = index
        return sum(len(v) for v in index.values())

    @staticmethod
    def _good_match_score(matches: Iterable[list[cv2.DMatch]], ratio: float = 0.75) -> int:
        count = 0
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < ratio * n.distance:
                count += 1
        return count

    def predict(self, image: np.ndarray) -> ClassificationResult:
        start = time.perf_counter()
        if not self._index:
            self.build_index()
        query = self._descriptors(image)
        if query is None or not len(query):
            return ClassificationResult(None, 0.0, 'orb', (time.perf_counter()-start)*1000, 'No ORB features found in image')
        scores: dict[str, int] = {}
        for label, refs in self._index.items():
            best = 0
            for ref_desc in refs:
                matches = self.matcher.knnMatch(query, ref_desc, k=2)
                best = max(best, self._good_match_score(matches))
            scores[label] = best
        if not scores:
            return ClassificationResult(None, 0.0, 'orb', (time.perf_counter()-start)*1000, 'No valid ORB reference templates')
        label = max(scores, key=scores.get)
        best = scores[label]
        total = sum(scores.values())
        confidence = float(best / total) if total > 0 else 0.0
        # 归一化各类得分（匹配数占比），用于分类卡片展示
        norm_scores = {k: float(v / total) for k, v in scores.items()} if total > 0 else {}
        warning = None
        if best == 0:
            warning = 'No reliable ORB matches'
        elif len(scores) > 1:
            second = sorted(scores.values(), reverse=True)[1]
            if second > 0 and best / second < 1.5:
                warning = f'ORB 最佳与次选差距较小（{best} vs {second}），分类置信度有限'
        return ClassificationResult(label if best > 0 else None, confidence, 'orb', (time.perf_counter()-start)*1000,
                                    warning, scores=norm_scores)
