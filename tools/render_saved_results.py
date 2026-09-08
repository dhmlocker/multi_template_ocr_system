from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.types import OCRItem
from core.visualization import draw_ocr_boxes, draw_white_ocr_canvas, make_original_white_result_pair

OUT = ROOT / 'outputs' / 'demo_samples'


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(path)
    return image


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def main() -> int:
    for sample_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and (p / 'official').is_dir()):
        source_candidates = sorted((ROOT / 'data' / 'samples' / sample_dir.name).glob('*'))
        source = next((path for path in source_candidates if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}), None)
        result_files = sorted((sample_dir / 'official').glob('*_res.json'))
        if source is None or not result_files:
            continue
        payload = json.loads(result_files[-1].read_text(encoding='utf-8'))
        image = read_image(source)
        texts = payload.get('rec_texts', [])
        scores = payload.get('rec_scores', [])
        polys = payload.get('rec_polys', payload.get('dt_polys', []))
        items = [OCRItem(box=poly, text=str(text), confidence=float(score)) for text, score, poly in zip(texts, scores, polys) if str(text).strip()]
        annotated = draw_ocr_boxes(image, items)
        save(sample_dir / '02_ocr_boxes_inside.jpg', annotated)
        save(sample_dir / '04_original_vs_white.jpg', make_original_white_result_pair(image, items))
        save(sample_dir / '05_text_recognition_white_inside.jpg', draw_white_ocr_canvas(image.shape[:2], items))
        print(f'{sample_dir.name}: rendered {len(items)} boxes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
