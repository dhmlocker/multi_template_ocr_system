from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis import write_csv, write_jsonl, summarize_result  # noqa: E402
from core.config import AppConfig  # noqa: E402
from core.pipeline import RecognitionPipeline  # noqa: E402


def iter_images(root: Path):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description='Run reproducible PaddleOCRv6 form recognition evaluation.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'test')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'experiments')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--classifier', default=None, help='cnn, orb or ocr_only')
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        print(f'data directory not found: {args.data_dir}')
        return 2
    cfg = AppConfig.load(ROOT / 'config' / 'settings.yaml', project_root=ROOT)
    pipeline = RecognitionPipeline(cfg)
    rows = []
    paths = list(iter_images(args.data_dir))
    if args.limit > 0:
        paths = paths[:args.limit]
    for index, path in enumerate(paths, 1):
        data = cv2.imdecode(__import__('numpy').fromfile(str(path), dtype=__import__('numpy').uint8), cv2.IMREAD_COLOR)
        if data is None:
            continue
        started = time.perf_counter()
        try:
            result = pipeline.run(data, filename=path.name, classifier_method=args.classifier, save=False)
            row = summarize_result(result)
            row['relative_path'] = str(path.relative_to(args.data_dir))
            row['ground_truth_template'] = path.parent.name
            row['elapsed_wall_ms'] = round((time.perf_counter() - started) * 1000, 2)
            rows.append(row)
            print(f'[{index}/{len(paths)}] {path.name}: {row["template_name"]}, {row["total_ms"]:.0f} ms')
        except Exception as exc:
            rows.append({'relative_path': str(path.relative_to(args.data_dir)), 'error': str(exc)})
            print(f'[{index}/{len(paths)}] FAILED {path.name}: {exc}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.output_dir / 'recognition_results.jsonl')
    write_csv(rows, args.output_dir / 'recognition_results.csv')
    (args.output_dir / 'run_config.json').write_text(json.dumps({
        'data_dir': str(args.data_dir), 'count': len(paths), 'completed': len(rows),
        'paddleocr_engine': 'PaddleOCRv6', 'classifier': args.classifier or cfg.classifier.method,
        'ocr': cfg.to_dict()['ocr'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote experiment artifacts to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
