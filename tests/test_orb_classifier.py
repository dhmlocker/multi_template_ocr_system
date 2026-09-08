from pathlib import Path
import cv2
import numpy as np

from core.orb_classifier import ORBClassifier


def pattern(kind: int) -> np.ndarray:
    img = np.full((240, 320, 3), 255, np.uint8)
    if kind == 0:
        cv2.putText(img, 'FORM-A', (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
        cv2.rectangle(img, (25, 135), (285, 200), (0,0,0), 3)
        cv2.line(img, (40, 160), (270, 160), (0,0,0), 2)
    else:
        cv2.putText(img, 'FORM-B', (45, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
        cv2.circle(img, (160, 160), 55, (0,0,0), 3)
        cv2.line(img, (105, 160), (215, 160), (0,0,0), 2)
    return img


def test_orb_classifies_nearest_reference(tmp_path: Path):
    refs = tmp_path / 'refs'
    (refs / 'a').mkdir(parents=True)
    (refs / 'b').mkdir(parents=True)
    cv2.imwrite(str(refs/'a'/'ref.png'), pattern(0))
    cv2.imwrite(str(refs/'b'/'ref.png'), pattern(1))
    clf = ORBClassifier(refs)
    clf.build_index()
    result = clf.predict(pattern(0))
    assert result.label == 'a'
    assert result.method == 'orb'
    assert result.confidence > 0
