from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def build_template_payload(template_name: str, width: int, height: int, named_rois: Iterable[tuple[str, tuple[int,int,int,int]]]) -> dict:
    fields = []
    for field_name, (x, y, w, h) in named_rois:
        fields.append({
            'field_name': str(field_name),
            'bbox': [int(x), int(y), int(x+w), int(y+h)],
            'match_mode': 'center_or_iou',
            'iou_threshold': 0.1,
        })
    return {
        'template_name': str(template_name),
        'reference_width': int(width),
        'reference_height': int(height),
        'fields': fields,
    }


def load_image(path: Path) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'cannot decode image: {path}')
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive ROI labeler for a user-provided template image.')
    parser.add_argument('image', help='Reference image path')
    parser.add_argument('--template-name', required=True)
    parser.add_argument('--output', default=None, help='Output JSON path; defaults to config/templates/<template>.json')
    args = parser.parse_args()

    image_path = Path(args.image)
    image = load_image(image_path)
    print('在弹出的窗口中依次框选字段 ROI；按 ENTER/SPACE 确认一个框，ESC 结束。')
    rois = cv2.selectROIs('ROI Labeler', image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if rois is None or len(rois) == 0:
        print('没有选择 ROI，未写入配置。')
        return
    named = []
    for i, roi in enumerate(rois, 1):
        default = f'field_{i}'
        name = input(f'第 {i} 个 ROI 字段名 [{default}]: ').strip() or default
        named.append((name, tuple(int(v) for v in roi)))
    payload = build_template_payload(args.template_name, image.shape[1], image.shape[0], named)
    output = Path(args.output) if args.output else Path('config/templates') / f'{args.template_name}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'ROI 配置已保存：{output}')


if __name__ == '__main__':
    main()
