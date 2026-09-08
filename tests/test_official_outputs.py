from __future__ import annotations

import numpy as np

from core.refinement import refine_low_confidence_items
from core.types import OCRItem
from core.visualization import draw_white_ocr_canvas


class FakeEngine:
    def recognize(self, image):
        return [OCRItem([[0, 0], [20, 0], [20, 12], [0, 12]], '复识别结果', 0.96)]


def test_refine_low_confidence_items_replaces_with_stronger_candidate():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    items = [OCRItem([[10, 10], [30, 10], [30, 22], [10, 22]], '原始结果', 0.4)]
    refined, stats = refine_low_confidence_items(image, items, FakeEngine(), threshold=0.8)
    assert refined[0].text == '复识别结果'
    assert refined[0].source == 'roi_refine'
    assert refined[0].parent_index == 0
    assert stats['accepted'] == 1


def test_white_ocr_canvas_preserves_image_shape():
    item = OCRItem([[2, 2], [20, 2], [20, 12], [2, 12]], '测试', 0.9)
    canvas = draw_white_ocr_canvas((30, 50), [item])
    assert canvas.shape == (30, 50, 3)
    assert int(canvas.mean()) > 200
