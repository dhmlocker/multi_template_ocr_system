from core.field_extractor import FieldExtractor
from core.types import OCRItem


def box(x1, y1, x2, y2):
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


def test_scales_roi_and_merges_in_reading_order():
    cfg = {
        'template_name': 't', 'reference_width': 1000, 'reference_height': 500,
        'fields': [{'field_name': 'name', 'bbox': [100,100,500,200]}]
    }
    items = [
        OCRItem(box(90, 45, 140, 70), 'B', .9),
        OCRItem(box(50, 45, 80, 70), 'A', .9),
    ]
    # image is half reference size, ROI becomes [50,50,250,100]
    result = FieldExtractor.extract(cfg, items, (500,250))
    assert result[0].value == 'A B'
    assert result[0].missing is False
    assert result[0].roi == [50.0,50.0,250.0,100.0]


def test_iou_fallback_can_match_when_center_is_outside():
    cfg = {
        'template_name': 't', 'reference_width': 100, 'reference_height': 100,
        'fields': [{'field_name':'f','bbox':[10,10,30,30],'match_mode':'center_or_iou','iou_threshold':0.1}]
    }
    item = OCRItem(box(15, 20, 55, 40), 'X', .8)
    result = FieldExtractor.extract(cfg, [item], (100,100))
    assert result[0].value == 'X'


def test_missing_field_returns_empty_value_and_flag():
    cfg = {
        'template_name':'t','reference_width':100,'reference_height':100,
        'fields':[{'field_name':'f','bbox':[0,0,10,10]}]
    }
    result = FieldExtractor.extract(cfg, [], (100,100))
    assert result[0].value == ''
    assert result[0].missing is True
    assert result[0].source_indices == []
