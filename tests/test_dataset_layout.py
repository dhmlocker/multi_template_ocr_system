from pathlib import Path
import pytest

from training.dataset import discover_classes, validate_imagefolder_layout


def test_discover_classes_from_user_folders_only(tmp_path: Path):
    for name in ['invoice', 'hospital', 'shipping']:
        (tmp_path / name).mkdir()
    assert discover_classes(tmp_path) == ['hospital', 'invoice', 'shipping']


def test_validate_layout_rejects_empty_class_directory(tmp_path: Path):
    (tmp_path / 'invoice').mkdir()
    with pytest.raises(ValueError, match='contains no supported images'):
        validate_imagefolder_layout(tmp_path)
