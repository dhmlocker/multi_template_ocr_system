from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from core.cnn_classifier import CNNClassifier
from training.dataset import validate_imagefolder_layout, IMAGE_EXTS


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate trained CNN template classifier.')
    parser.add_argument('--data-dir', default='data/test')
    parser.add_argument('--model', default='models/classifier.pth')
    parser.add_argument('--class-names', default='models/class_names.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    classes = validate_imagefolder_layout(args.data_dir)
    clf = CNNClassifier(args.model, args.class_names, args.device)
    total = correct = 0; elapsed = 0.0
    for label in classes:
        for p in Path(args.data_dir, label).rglob('*'):
            if p.suffix.lower() not in IMAGE_EXTS: continue
            img = cv2.imread(str(p)); t = time.perf_counter(); result = clf.predict(img); elapsed += time.perf_counter()-t
            total += 1; correct += int(result.label == label)
    print({'samples': total, 'accuracy': correct/max(total,1), 'avg_ms': elapsed*1000/max(total,1)})


if __name__ == '__main__':
    main()
