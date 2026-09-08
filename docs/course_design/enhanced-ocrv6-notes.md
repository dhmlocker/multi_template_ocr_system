# 增强版 PaddleOCRv6 识别链路说明

本版本保留 PP-OCRv6 作为核心文字检测与识别引擎，同时启用 PaddleX OCR 流水线提供的三个文档级配套模型：PP-LCNet_x1_0_doc_ori 用于文档方向分类，UVDoc 用于文档去畸变，PP-LCNet_x1_0_textline_ori 用于文本行方向分类。模型在首次运行时自动下载到 PaddleX 官方模型缓存，后续运行复用缓存；也可以在配置中指定本地模型目录。

当前默认配置位于 `config/settings.yaml`，其中 `use_doc_orientation_classify`、`use_doc_unwarping` 和 `use_textline_orientation` 均为 `true`。PaddleOCRv6 检测和识别模型使用 PP-OCRv6_medium_det 与 PP-OCRv6_medium_rec。程序将方向与预处理信息写入 `result.preprocessing.paddleocrv6`，并在 Streamlit 的“PaddleOCRv6 文档预处理状态”面板中展示。

结果可视化包括原图、彩色 OCR 多边形框、文本标签、模板 ROI 框、`04_original_vs_ocr.jpg` 原图/彩色结果并排图，以及 `05_text_recognition_white.jpg` 白底文本识别结果图。程序同时调用官方 Result 的 `save_to_img()` 和 `save_to_json()`，保存官方结果文件，项目自己的统一结果只保留轻量坐标、文本、置信度和配置元数据。白底结果图的坐标来自官方文本框，形式接近 PaddleOCR 和 AI Studio 的 Text recognition 视图。

整页 OCR 完成后，系统会对低置信度文本框进行有限数量的 ROI 二次识别，并以候选置信度和字段校验结果选择更可信文本。结果中的 `source` 字段区分 `full_page` 与 `roi_refine`，`result.preprocessing.roi_refine` 保存尝试和采纳数量。默认主链路使用官方原图输入，避免项目自定义 CLAHE 对发票小字造成额外损伤；需要时仍可通过配置启用其他预处理模式。

三类仓库样例已分别运行增强链路。最新运行中，医院收费票据识别出 59 个文本框，增值稅发票识别出 72 个文本框，快递寄件面单识别出 22 个文本框；三类样例的 CNN 模板分类均正确。由于每类只有一张样例，这些数字用于展示和调试，不代表大规模数据集指标。

重新运行命令：

```bash
python3 tools/run_demo_samples.py --only 医院收费票据
python3 tools/run_demo_samples.py --only 增值稅发票
python3 tools/run_demo_samples.py --only 快递寄件面单
```

官方流程还支持 PP-StructureV3 版面解析、表格识别和文档元素识别；当前版本先启用与三类表单直接相关的方向校正、去畸变和文本行方向模型，以避免在 CPU 环境下引入过重的版面模型。后续若需要表格结构输出，可在独立实验模式中增加 PP-StructureV3，并与当前 ROI 字段抽取结果进行对比。
