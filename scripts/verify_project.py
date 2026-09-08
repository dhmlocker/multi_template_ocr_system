from __future__ import annotations

from pathlib import Path

ALLOWED_NAMES = {'.gitkeep', 'README.md', 'README.txt'}
MODEL_EXTS = {'.pth','.pt','.pdparams','.pdmodel','.onnx','.bin','.safetensors','.nb'}
DATA_EXTS = {'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff','.pdf','.csv','.xlsx','.xls','.jsonl','.txt'}


def find_forbidden_payloads(root: str | Path) -> list[Path]:
    root = Path(root)
    hits: list[Path] = []
    data_root = root / 'data'
    model_root = root / 'models'
    if data_root.exists():
        for p in data_root.rglob('*'):
            if not p.is_file() or p.name in ALLOWED_NAMES:
                continue
            if p.suffix.lower() in DATA_EXTS or p.stat().st_size > 0:
                hits.append(p)
    if model_root.exists():
        for p in model_root.rglob('*'):
            if not p.is_file() or p.name in ALLOWED_NAMES:
                continue
            if p.suffix.lower() in MODEL_EXTS or p.stat().st_size > 0:
                hits.append(p)
    return sorted(hits)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hits = find_forbidden_payloads(root)
    if hits:
        print('Verification failed: packaged user data/model payloads detected:')
        for p in hits:
            print(' -', p.relative_to(root))
        return 1
    required = [
        root/'app.py', root/'config/settings.yaml', root/'core/pipeline.py',
        root/'database/schema.sql', root/'README.md', root/'requirements.txt'
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print('Verification failed: required project files missing:')
        for p in missing:
            print(' -', p.relative_to(root))
        return 2
    print('Project verification passed: no business data or model weights are bundled.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
