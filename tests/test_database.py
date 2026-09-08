from pathlib import Path

from core.types import ClassificationResult, FieldResult, OCRItem, PipelineResult
from database.db import RecognitionRepository


def make_result():
    return PipelineResult(
        filename='x.png', template_name='invoice',
        classification=ClassificationResult('invoice', .91, 'cnn', 12.0),
        ocr_items=[OCRItem([[0,0],[1,0],[1,1],[0,1]], 'abc', .88)],
        fields=[FieldResult('name', 'abc', False, [0], [0,0,10,10])],
        timings_ms={'total': 100.0}
    )


def test_repository_crud_and_export(tmp_path: Path):
    repo = RecognitionRepository(tmp_path / 'records.db')
    repo.initialize()
    rid = repo.save_result(make_result())
    rows = repo.list_records()
    assert rows[0]['id'] == rid
    assert rows[0]['template_name'] == 'invoice'
    detail = repo.get_record(rid)
    assert detail['fields'][0]['field_name'] == 'name'
    assert detail['fields'][0]['field_value'] == 'abc'
    exported = repo.export_rows()
    assert exported[0]['filename'] == 'x.png'
    repo.delete_record(rid)
    assert repo.list_records() == []
