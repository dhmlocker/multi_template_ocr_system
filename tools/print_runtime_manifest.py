from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, '__version__', 'unknown'))
    except Exception as exc:
        return f'not-importable: {exc}'


def git_commit() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    model_cache = Path(os.environ.get('PADDLEX_HOME', Path.home() / '.paddlex')) / 'official_models'
    config = ROOT / 'config' / 'settings.yaml'
    models = {
        name: (model_cache / name).is_dir()
        for name in (
            'PP-OCRv6_medium_det', 'PP-OCRv6_medium_rec',
            'PP-LCNet_x1_0_doc_ori', 'UVDoc', 'PP-LCNet_x1_0_textline_ori',
        )
    }
    manifest = {
        'project_root': str(ROOT),
        'git_commit': git_commit(),
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {
            'paddle': version('paddle'),
            'paddleocr': version('paddleocr'),
            'paddlex': version('paddlex'),
            'streamlit': version('streamlit'),
            'cv2': version('cv2'),
            'PIL': version('PIL'),
            'numpy': version('numpy'),
            'pandas': version('pandas'),
            'torch': version('torch'),
            'torchvision': version('torchvision'),
        },
        'config_path': str(config),
        'config_sha256': sha256(config),
        'model_cache': str(model_cache),
        'models_present': models,
        'device': os.environ.get('PADDLE_DEVICE', 'cpu'),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
