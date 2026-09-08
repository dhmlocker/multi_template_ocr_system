from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs' / 'demo_samples'
OUTPUT = INPUT / 'all_samples_contact_sheet.jpg'

items = []
for label in sorted(p for p in INPUT.iterdir() if p.is_dir()):
    for name in ('01_original.jpg', '02_ocr_boxes.jpg', '03_ocr_roi.jpg'):
        path = label / name
        if path.is_file():
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                items.append((label.name, name, image))

tile_w, tile_h = 520, 330
rows = []
for i in range(0, len(items), 3):
    row = []
    for label, name, image in items[i:i + 3]:
        scale = min((tile_w - 20) / image.shape[1], (tile_h - 45) / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        y = 35 + (tile_h - 35 - resized.shape[0]) // 2
        x = (tile_w - resized.shape[1]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(tile, f'{label} | {name}', (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 54, 93), 1, cv2.LINE_AA)
        row.append(tile)
    while len(row) < 3:
        row.append(np.full((tile_h, tile_w, 3), 255, dtype=np.uint8))
    rows.append(np.hstack(row))
if rows:
    sheet = np.vstack(rows)
    cv2.imencode('.jpg', sheet)[1].tofile(str(OUTPUT))
    print(OUTPUT)
