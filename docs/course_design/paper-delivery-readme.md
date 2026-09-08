# 论文交付包使用说明

本目录由三类仓库样例的实际运行生成，核心 OCR 引擎为 PaddleOCRv6，模板分类器为项目自带 CNN 模型。

## 实际实验结果

- 医院收费票据：分类正确，61 个 OCR 文本框，字段覆盖率 83.33%，端到端耗时 69.56 秒。
- 增值稅发票：分类正确，74 个 OCR 文本框，字段覆盖率 80.00%，端到端耗时 146.63 秒。
- 快递寄件面单：分类正确，24 个 OCR 文本框，字段覆盖率 33.33%，端到端耗时 46.70 秒。
- 三类样例案例级分类准确率：3/3 = 100%。每类仅一张图片，不能代表大样本泛化性能。

## 目录说明

- `actual-demo-analysis.md/pdf`：根据真实三样例运行结果编写的论文分析素材。
- `outputs/demo_samples/demo_summary.csv`：结构化汇总数据，可继续制作论文图表。
- `outputs/demo_samples/demo_details.json`：完整 OCR 文本框、字段、置信度和耗时结果。
- `outputs/demo_samples/<模板>/01_original.jpg`：原图。
- `outputs/demo_samples/<模板>/02_ocr_boxes.jpg`：PaddleOCRv6 检测框图。
- `outputs/demo_samples/<模板>/03_ocr_roi.jpg`：检测框与模板 ROI 叠加图。
- `outputs/demo_samples/charts/`：真实数据生成的论文图表。
- `tools/run_demo_samples.py`：重新运行三类样例。
- `tools/make_analysis_charts.py`：根据 CSV 重新生成图表。

## 重新运行

```bash
pip install -r requirements.txt
python3 tools/run_demo_samples.py
python3 tools/make_analysis_charts.py
```

如果使用完整数据集，应将实验样本按 `data/test/<模板名>/*` 放置，并运行 `tools/run_experiment.py`。论文中的准确率、召回率和混淆矩阵必须使用人工标注数据计算，不应把本次每类单张样例的 100% 分类结果表述为模型总体准确率。
