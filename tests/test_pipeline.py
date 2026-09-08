from pathlib import Path
import numpy as np

from core.config import AppConfig
from core.pipeline import RecognitionPipeline
from core.types import ClassificationResult, OCRItem


class FakeOCR:
    def recognize(self, image):
        return [OCRItem([[1,1],[20,1],[20,10],[1,10]], 'abc', .9)]


class FakeClassifier:
    def predict(self, image):
        return ClassificationResult('known_template', .95, 'cnn', 1.0)


class FailingRepo:
    def save_result(self, result):
        raise RuntimeError('db down')


def config(tmp_path: Path):
    (tmp_path/'config/templates').mkdir(parents=True)
    (tmp_path/'models').mkdir()
    (tmp_path/'outputs').mkdir()
    settings = tmp_path/'config/settings.yaml'
    settings.write_text('''\nocr:\n  det_model_dir: models/det\n  rec_model_dir: models/rec\nclassifier:\n  method: cnn\n  cnn_model_path: models/cnn.pth\n  class_names_path: models/classes.json\n  orb_reference_dir: models/orb\ndatabase:\n  path: outputs/db.sqlite\ntemplates:\n  config_dir: config/templates\nruntime:\n  device: cpu\n''', encoding='utf-8')
    return AppConfig.load(settings, project_root=tmp_path)


def test_ocr_only_mode_does_not_require_classifier(tmp_path: Path):
    cfg = config(tmp_path)
    pipe = RecognitionPipeline(cfg, ocr_engine=FakeOCR())
    image = np.full((20,30,3), 255, np.uint8)
    result = pipe.run(image, 'x.png', classifier_method='ocr_only')
    assert result.classification is None
    assert result.ocr_items[0].text == 'abc'
    assert result.fields == []


def test_missing_roi_config_returns_ocr_and_warning(tmp_path: Path):
    cfg = config(tmp_path)
    pipe = RecognitionPipeline(cfg, ocr_engine=FakeOCR(), cnn_classifier=FakeClassifier())
    result = pipe.run(np.full((20,30,3),255,np.uint8), 'x.png', classifier_method='cnn')
    assert result.template_name == 'known_template'
    assert result.ocr_items[0].text == 'abc'
    assert any('ROI' in w for w in result.warnings)


def test_database_save_failure_does_not_lose_result(tmp_path: Path):
    cfg = config(tmp_path)
    pipe = RecognitionPipeline(cfg, ocr_engine=FakeOCR(), repository=FailingRepo())
    result = pipe.run(np.full((20,30,3),255,np.uint8), 'x.png', classifier_method='ocr_only', save=True)
    assert result.record_id is None
    assert any('保存失败' in w for w in result.warnings)
