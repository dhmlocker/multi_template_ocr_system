from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def discover_classes(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'dataset directory not found: {root}')
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))


def validate_imagefolder_layout(root: str | Path) -> list[str]:
    root = Path(root)
    classes = discover_classes(root)
    if not classes:
        raise ValueError(f'no class directories found under: {root}')
    for class_name in classes:
        class_dir = root / class_name
        count = sum(1 for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if count == 0:
            raise ValueError(f'class directory {class_dir} contains no supported images')
    return classes
