from pathlib import Path
import pytest

from training.train_classifier import resolve_backbone_source


def test_local_backbone_is_preferred_and_missing_never_downloads_implicitly(tmp_path: Path):
    local = tmp_path / 'mobilenet_v2_imagenet.pth'
    local.write_bytes(b'x')
    assert resolve_backbone_source(local, no_pretrained=False, allow_download=False) == 'local'
    with pytest.raises(FileNotFoundError, match='local MobileNetV2 ImageNet weight'):
        resolve_backbone_source(tmp_path/'missing.pth', no_pretrained=False, allow_download=False)


def test_pretrained_can_be_disabled_or_download_explicitly_enabled(tmp_path: Path):
    missing = tmp_path/'missing.pth'
    assert resolve_backbone_source(missing, no_pretrained=True, allow_download=False) == 'none'
    assert resolve_backbone_source(missing, no_pretrained=False, allow_download=True) == 'torchvision'
