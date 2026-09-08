from tools.roi_labeler import build_template_payload


def test_build_template_payload_uses_actual_image_size_and_named_boxes():
    payload = build_template_payload('demo', 800, 600, [('field_a', (10,20,30,40))])
    assert payload['template_name'] == 'demo'
    assert payload['reference_width'] == 800
    assert payload['fields'][0]['bbox'] == [10,20,40,60]
