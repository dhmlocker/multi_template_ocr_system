# 三类样例增强链路运行日志

本日志记录逐类运行官方 PaddleOCRv6 Pipeline、文档方向分类、UVDoc 去畸变、文本行方向分类和低置信度 ROI 复识别后的实际终端结果。每类当前只有一张样例，数据用于演示和调试，不代表统计学准确率。

| 模板 | 模板分类 | OCR 文本框 | 总耗时 |
|---|---|---:|---:|
| 医院收费票据 | 正确 | 59 | 117.2 s |
| 增值稅发票 | 正确 | 72 | 322.5 s |
| 快递寄件面单 | 正确 | 22 | 161.7 s |

输出文件包括每个模板目录下的 `01_original.jpg`、`02_ocr_boxes.jpg`、`03_ocr_roi.jpg`、`04_original_vs_ocr.jpg`、`05_text_recognition_white.jpg` 和 `official/` 目录。`official/` 目录由 PaddleOCR Result 的 `save_to_img()` 和 `save_to_json()` 生成。

增值稅发票的白底识别结果是本轮重点验证素材，展示了原始坐标下的彩色检测框、识别文本和置信度。当前 CPU 环境中去畸变与 ROI 复识别明显增加耗时；后续现场演示应提供“快速识别”模式，论文实验再使用高精度模式。
