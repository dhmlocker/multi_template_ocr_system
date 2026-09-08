# 三类样例实际识别结果

本目录由 `tools/run_demo_samples.py` 使用当前配置和 PaddleOCRv6 运行生成。

- `01_original.jpg`：原始样例图
- `02_ocr_boxes.jpg`：OCR 文本框可视化
- `03_ocr_roi.jpg`：OCR 框与模板字段 ROI 可视化
- `04_original_vs_ocr.jpg`：参考 PaddleOCR 官方示例的原图/彩色 OCR 框并排图
- `demo_summary.csv`：可用于论文统计的汇总数据
- `demo_details.json`：完整识别结果、字段与耗时

当前增强配置启用了 PP-LCNet_x1_0_doc_ori 文档方向分类、UVDoc 文档去畸变、PP-LCNet_x1_0_textline_ori 文本行方向分类，以及 PP-OCRv6_medium_det/rec 检测识别模型。
