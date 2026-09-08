from pathlib import Path
from types import SimpleNamespace

from ui.settings_page import model_status_rows


def test_model_status_rows_includes_local_mobilenet_backbone(tmp_path: Path):
    cfg = SimpleNamespace(
        ocr=SimpleNamespace(det_model_dir=tmp_path/'det', rec_model_dir=tmp_path/'rec'),
        classifier=SimpleNamespace(
            cnn_model_path=tmp_path/'cnn.pth', class_names_path=tmp_path/'classes.json',
            pretrained_backbone_path=tmp_path/'mobilenet_v2_imagenet.pth', orb_reference_dir=tmp_path/'orb'
        )
    )
    rows = model_status_rows(cfg)
    assert any(r['项目'] == 'MobileNetV2预训练' for r in rows)
