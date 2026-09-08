from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis import summarize_result, write_csv, write_jsonl  # noqa: E402
from core.config import AppConfig  # noqa: E402
from core.pipeline import RecognitionPipeline  # noqa: E402
from core.visualization import draw_all, draw_ocr_boxes, make_side_by_side  # noqa: E402


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot decode image: {path}')
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or '.jpg'
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f'cannot encode image: {path}')
    encoded.tofile(str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', default='', help='只运行一个模板目录名称')
    args = parser.parse_args()
    cfg = AppConfig.load(ROOT / 'config' / 'settings.yaml', project_root=ROOT)
    pipeline = RecognitionPipeline(cfg)
    sample_root = ROOT / 'data' / 'samples'
    output_root = ROOT / 'outputs' / 'demo_samples'
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in sample_root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'})
    if args.only:
        paths = [p for p in paths if p.parent.name == args.only]
    rows = []
    details = []
    for index, path in enumerate(paths, 1):
        image = read_image(path)
        started = time.perf_counter()
        result = pipeline.run(image, filename=path.name, classifier_method='cnn', save=False)
        elapsed = (time.perf_counter() - started) * 1000
        label = path.parent.name
        sample_dir = output_root / label
        save_image(sample_dir / '01_original.jpg', image)
        save_image(sample_dir / '02_ocr_boxes.jpg', draw_ocr_boxes(image, result.ocr_items))
        save_image(sample_dir / '03_ocr_roi.jpg', draw_all(image, result.ocr_items, result.fields))
        save_image(sample_dir / '04_original_vs_ocr.jpg', make_side_by_side(image, draw_ocr_boxes(image, result.ocr_items)))
        summary = summarize_result(result)
        summary.update({
            'sample_index': index,
            'ground_truth_template': label,
            'predicted_template': result.template_name,
            'classification_correct': bool(result.template_name and result.template_name == label),
            'wall_time_ms': round(elapsed, 2),
            'source_path': str(path.relative_to(ROOT)),
            'result_dir': str(sample_dir.relative_to(ROOT)),
        })
        rows.append(summary)
        details.append({'summary': summary, 'result': result.to_dict()})
        print(f'[{index}/{len(paths)}] {label}: predicted={result.template_name}, text={len(result.ocr_items)}, total={summary["total_ms"]:.0f}ms')
    write_csv(rows, output_root / 'demo_summary.csv')
    write_jsonl(rows, output_root / 'demo_summary.jsonl')
    (output_root / 'demo_details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_root / 'README.md').write_text(
        '# 三类样例实际识别结果\n\n'
        '本目录由 `tools/run_demo_samples.py` 使用当前配置和 PaddleOCRv6 运行生成。\n\n'
        '- `01_original.jpg`：原始样例图\n'
        '- `02_ocr_boxes.jpg`：OCR 文本框可视化\n'
        '- `03_ocr_roi.jpg`：OCR 框与模板字段 ROI 可视化\n'
        '- `demo_summary.csv`：可用于论文统计的汇总数据\n'
        '- `demo_details.json`：完整识别结果、字段与耗时\n', encoding='utf-8')
    print(f'Wrote demo artifacts to {output_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
