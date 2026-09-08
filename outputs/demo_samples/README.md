# 三类样例实际识别结果

本目录由 `tools/run_demo_samples.py` 使用当前配置和 PaddleOCRv6 运行生成。

- `01_original.jpg`：原始样例图
- `02_ocr_boxes.jpg`：OCR 文本框可视化
- `03_ocr_roi.jpg`：OCR 框与模板字段 ROI 可视化
- `04_original_vs_ocr.jpg`：原图与彩色 OCR 框并排图
- `05_text_recognition_white.jpg`：白底文本识别结果图
- `official/`：PaddleOCR 官方 save_to_img/save_to_json 输出
- `demo_summary.csv`：可用于论文统计的汇总数据
- `demo_details.json`：完整识别结果、字段与耗时
