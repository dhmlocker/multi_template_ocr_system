from __future__ import annotations

import argparse
from pathlib import Path

from core.orb_classifier import ORBClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description='检查并建立 ORB 参考模板索引（不复制、不生成业务图片）')
    parser.add_argument('--reference-dir', default='models/orb_templates')
    args = parser.parse_args()
    clf = ORBClassifier(Path(args.reference_dir))
    count = clf.build_index()
    print(f'ORB reference images indexed: {count}; classes: {sorted(clf._index)}')


if __name__ == '__main__':
    main()
