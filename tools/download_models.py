from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description='Prepare official PaddleOCRv6 pipeline models')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    os.environ.setdefault('FLAGS_use_mkldnn', '0')
    os.environ.setdefault('PADDLE_PDX_DISABLE_DEV_MODEL_WL', '1')
    from paddleocr import PaddleOCR

    PaddleOCR(
        device=args.device,
        lang='ch',
        text_detection_model_name='PP-OCRv6_medium_det',
        text_recognition_model_name='PP-OCRv6_medium_rec',
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
    )
    print('已准备官方模型：PP-OCRv6_medium_det、PP-OCRv6_medium_rec、PP-LCNet_x1_0_doc_ori、UVDoc、PP-LCNet_x1_0_textline_ori')
    print('模型默认缓存目录：~/.paddlex/official_models')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
