from pathlib import Path
import pytest

from core.config import OCRConfig
from core.ocr_engine import PaddleOCREngine


def test_validate_models_fails_without_attempting_download(tmp_path: Path):
    cfg = OCRConfig(tmp_path/'det', tmp_path/'rec')
    engine = PaddleOCREngine(cfg)
    with pytest.raises(FileNotFoundError, match='Local PP-OCRv6 model directory missing'):
        engine.validate_models()


def test_normalize_legacy_paddleocr_result():
    raw = [[
        [[[1,2],[20,2],[20,10],[1,10]], ('hello', 0.93)],
        [[[3,20],[30,20],[30,28],[3,28]], ('world', 0.88)],
    ]]
    items = PaddleOCREngine.normalize_result(raw)
    assert [x.text for x in items] == ['hello', 'world']
    assert items[0].confidence == pytest.approx(0.93)


def test_normalize_paddleocr_v3_json_result():
    class Result:
        json = {
            'res': {
                'rec_texts': ['甲', '乙'],
                'rec_scores': [0.91, 0.87],
                'rec_polys': [
                    [[1,1],[10,1],[10,5],[1,5]],
                    [[2,8],[12,8],[12,13],[2,13]],
                ],
            }
        }
    items = PaddleOCREngine.normalize_result([Result()])
    assert [x.text for x in items] == ['甲', '乙']
    assert items[1].box[0] == [2.0, 8.0]

def test_build_engine_with_generic_kwargs_still_receives_local_model_paths(tmp_path: Path, monkeypatch):
    import sys, types
    (tmp_path/'det').mkdir(); (tmp_path/'rec').mkdir()
    captured = {}
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    fake = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, 'paddleocr', fake)
    engine = PaddleOCREngine(OCRConfig(tmp_path/'det', tmp_path/'rec'))
    engine._build_engine()
    values = set(captured.values())
    assert str(tmp_path/'det') in values
    assert str(tmp_path/'rec') in values


def test_normalize_v3_numpy_arrays():
    import numpy as np
    class Result:
        json = {'res': {
            'rec_texts': np.array(['A']),
            'rec_scores': np.array([0.7]),
            'rec_polys': np.array([[[1,1],[2,1],[2,2],[1,2]]], dtype=float),
        }}
    items = PaddleOCREngine.normalize_result([Result()])
    assert items[0].text == 'A'
