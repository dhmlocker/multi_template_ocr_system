from pathlib import Path
import pytest

from core.config import AppConfig, resolve_project_path


def test_default_model_paths_resolve_inside_project(tmp_path: Path):
    cfg_file = tmp_path / 'settings.yaml'
    cfg_file.write_text('''\nocr:\n  det_model_dir: models/PP-OCRv6_medium_det\n  rec_model_dir: models/PP-OCRv6_medium_rec\ndatabase:\n  path: outputs/recognition.db\ntemplates:\n  config_dir: config/templates\n''', encoding='utf-8')
    cfg = AppConfig.load(cfg_file, project_root=tmp_path)
    assert cfg.ocr.det_model_dir == tmp_path / 'models/PP-OCRv6_medium_det'
    assert cfg.ocr.rec_model_dir == tmp_path / 'models/PP-OCRv6_medium_rec'
    assert cfg.database.path == tmp_path / 'outputs/recognition.db'


def test_invalid_yaml_root_raises_clear_error(tmp_path: Path):
    cfg_file = tmp_path / 'settings.yaml'
    cfg_file.write_text('- not\n- a\n- mapping\n', encoding='utf-8')
    with pytest.raises(ValueError, match='settings YAML root must be a mapping'):
        AppConfig.load(cfg_file, project_root=tmp_path)


def test_resolve_project_path_preserves_absolute(tmp_path: Path):
    absolute = tmp_path / 'x'
    assert resolve_project_path(absolute, tmp_path) == absolute

def test_pretrained_backbone_path_resolves_from_classifier_config(tmp_path: Path):
    cfg_file = tmp_path / 'settings.yaml'
    cfg_file.write_text('''\nocr:\n  det_model_dir: models/det\n  rec_model_dir: models/rec\nclassifier:\n  pretrained_backbone_path: models/mobilenet_v2_imagenet.pth\n''', encoding='utf-8')
    cfg = AppConfig.load(cfg_file, project_root=tmp_path)
    assert cfg.classifier.pretrained_backbone_path == tmp_path / 'models/mobilenet_v2_imagenet.pth'
