from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('paddle', 'paddleocr', 'paddlex', 'streamlit', 'cv2', 'PIL', 'numpy')
MODELS = ('PP-OCRv6_medium_det', 'PP-OCRv6_medium_rec', 'PP-LCNet_x1_0_doc_ori', 'UVDoc', 'PP-LCNet_x1_0_textline_ori')


def main() -> int:
    failures: list[str] = []
    if sys.version_info < (3, 10):
        failures.append(f'Python 版本过低：{sys.version.split()[0]}，需要 Python 3.10+')
    print(f'Python: {sys.version.split()[0]}')
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f'[OK] {name}: {getattr(module, "__version__", "loaded")}')
        except Exception as exc:
            failures.append(f'{name} 导入失败：{exc}')
            print(f'[FAIL] {name}: {exc}')
    cache = Path.home() / '.paddlex' / 'official_models'
    print(f'模型缓存：{cache}')
    for model in MODELS:
        path = cache / model
        if path.is_dir():
            print(f'[OK] model: {model}')
        else:
            failures.append(f'缺少模型：{model}')
            print(f'[FAIL] model: {model}')
    if failures:
        print('\n需要处理的问题：')
        for failure in failures:
            print(f'- {failure}')
        print('\n可尝试：python tools/download_models.py --device cpu')
        return 1
    print('环境和官方 PaddleOCRv6 模型检查通过。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
