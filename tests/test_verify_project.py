from pathlib import Path

from scripts.verify_project import find_forbidden_payloads


def test_verify_detects_business_data_and_model_weights(tmp_path: Path):
    (tmp_path/'data/train/class_a').mkdir(parents=True)
    (tmp_path/'models').mkdir()
    (tmp_path/'data/train/class_a/sample.jpg').write_bytes(b'x')
    (tmp_path/'models/model.pth').write_bytes(b'x')
    hits = find_forbidden_payloads(tmp_path)
    rel = {str(p.relative_to(tmp_path)).replace('\\','/') for p in hits}
    assert 'data/train/class_a/sample.jpg' in rel
    assert 'models/model.pth' in rel


def test_verify_allows_readmes_gitkeep_and_template_schema(tmp_path: Path):
    (tmp_path/'data/train').mkdir(parents=True)
    (tmp_path/'models').mkdir()
    (tmp_path/'config/templates').mkdir(parents=True)
    (tmp_path/'data/train/.gitkeep').write_text('')
    (tmp_path/'models/README.md').write_text('x')
    (tmp_path/'config/templates/example.schema.json').write_text('{}')
    assert find_forbidden_payloads(tmp_path) == []
