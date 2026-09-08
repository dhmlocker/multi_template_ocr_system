# 多模板表单自动识别和管理系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个不内置业务数据、使用本地 PP-OCRv6 模型、支持 CNN/ORB 模板分类、ROI 字段抽取、SQLite 管理和 Streamlit 可视化的课程设计项目，并输出可下载 ZIP。

**Architecture:** 采用模块化单体架构。`RecognitionPipeline` 串联分类、OCR、字段抽取和持久化；模型与 PaddleOCR 返回结构隔离在适配层；Streamlit 仅调用业务接口，不直接依赖模型实现。

**Tech Stack:** Python 3.10+、Streamlit、PaddleOCR/PaddlePaddle、PyTorch/torchvision、OpenCV、SQLite、PyYAML、Pandas、Pillow、Matplotlib、pytest。

**Spec:** `docs/superpowers/specs/2026-09-07-multi-template-ocr-system-design.md`

## Global Constraints

- 项目不生成、不内置任何训练集、验证集、测试集或伪造业务表单。
- 不下载或打包 PaddleOCR 模型；默认只引用 `models/PP-OCRv6_medium_det` 和 `models/PP-OCRv6_medium_rec`。
- CNN 模板分类是主方案，ORB 是传统基线；允许在缺少 CNN 权重时切换 ORB 或 OCR-only。
- 字段抽取优先中心点落入 ROI，可按字段启用 IoU 阈值回退；空字段返回空字符串并标记 missing。
- SQLite 仅保存用户真实执行后的识别记录。
- 所有默认路径使用项目相对路径；Windows/CPU 为主要运行环境。
- 最终必须通过 `pytest` 与 Python 语法编译检查后再打 ZIP。

---

## File Map

- `app.py`：Streamlit 入口、页面导航与全局样式。
- `core/config.py`：YAML 配置读取、路径解析与保存。
- `core/types.py`：OCR、分类、字段、流水线结果数据结构。
- `core/image_utils.py`：图片解码、尺寸和排序辅助。
- `core/ocr_engine.py`：PP-OCRv6 本地模型适配层。
- `core/cnn_classifier.py`：MobileNetV2 推理封装。
- `core/orb_classifier.py`：ORB 索引与分类。
- `core/field_extractor.py`：ROI 缩放、中心点/IoU 分配和文本合并。
- `core/pipeline.py`：端到端识别流程。
- `core/visualization.py`：OCR/ROI 框绘制。
- `database/db.py` / `database/schema.sql`：SQLite 建表和 CRUD。
- `ui/*.py`：识别、模板、历史、实验、设置页面。
- `training/*.py`：用户数据集读取、CNN 训练和评估。
- `tools/*.py`：ROI 标注、ORB 参考模板初始化。
- `tests/*.py`：配置、字段抽取、数据库、ORB、流水线降级路径单元测试。
- `config/settings.yaml`：默认本地模型/数据库/模板路径。
- `README.md` / `.gitignore` / `requirements.txt`：运行、数据放置、模型放置、依赖说明。

---

### Task 1: 配置、类型与项目骨架

**Files:**
- Create: `core/config.py`, `core/types.py`, `config/settings.yaml`, `tests/test_config.py`
- Create: `data/{train,val,test}/.gitkeep`, `models/.gitkeep`, `outputs/.gitkeep`, `config/templates/.gitkeep`

**Interfaces:**
- Produces: `AppConfig.load(path)`, `AppConfig.save(path)`, `resolve_project_path()`, `OCRItem`, `ClassificationResult`, `FieldResult`, `PipelineResult`。

- [ ] 写配置解析失败/默认路径解析测试。
- [ ] 运行 `pytest tests/test_config.py -v`，确认先失败。
- [ ] 实现配置和数据类型。
- [ ] 再运行测试，确认通过。

### Task 2: ROI 字段抽取引擎

**Files:**
- Create: `core/field_extractor.py`, `tests/test_field_extractor.py`

**Interfaces:**
- Consumes: `OCRItem`, 模板 JSON。
- Produces: `FieldExtractor.extract(template_config, ocr_items, image_size) -> list[FieldResult]`。

- [ ] 写 ROI 缩放、中心点命中、IoU 回退、阅读顺序合并、missing 测试。
- [ ] 运行字段测试并确认失败。
- [ ] 实现最小可用抽取器。
- [ ] 运行字段测试确认通过。

### Task 3: SQLite 持久化

**Files:**
- Create: `database/schema.sql`, `database/db.py`, `tests/test_database.py`

**Interfaces:**
- Produces: `RecognitionRepository.initialize()`, `save_result()`, `list_records()`, `get_record()`, `delete_record()`, `export_rows()`。

- [ ] 写临时数据库 CRUD 测试。
- [ ] 运行测试确认失败。
- [ ] 实现 schema 和 repository。
- [ ] 运行数据库测试确认通过。

### Task 4: ORB 传统基线

**Files:**
- Create: `core/orb_classifier.py`, `tools/init_orb_templates.py`, `tests/test_orb_classifier.py`

**Interfaces:**
- Produces: `ORBClassifier.build_index(reference_dir)`, `predict(image) -> ClassificationResult`。

- [ ] 用内存几何图写 ORB 接口测试，不写入业务数据目录。
- [ ] 运行测试确认失败。
- [ ] 实现 ORB 描述子索引与 BFMatcher 分类。
- [ ] 运行测试确认通过。

### Task 5: CNN 数据集、训练和推理

**Files:**
- Create: `training/dataset.py`, `training/train_classifier.py`, `training/evaluate_classifier.py`, `core/cnn_classifier.py`, `tests/test_dataset_layout.py`

**Interfaces:**
- Produces: `discover_classes(root)`, CLI 训练脚本输出 `classifier.pth` 与 `class_names.json`; `CNNClassifier.predict(image) -> ClassificationResult`。

- [ ] 写仅验证用户目录布局与类名发现的测试。
- [ ] 运行测试确认失败。
- [ ] 实现数据目录检查、训练/评估 CLI 与 MobileNetV2 推理封装。
- [ ] 运行相关测试确认通过。

### Task 6: PP-OCRv6 本地模型适配

**Files:**
- Create: `core/ocr_engine.py`, `tests/test_ocr_adapter.py`

**Interfaces:**
- Produces: `PaddleOCREngine.validate_models()`, `recognize(image) -> list[OCRItem]`。

- [ ] 写“模型目录缺失时不联网且明确报错”与返回结构归一化测试（mock PaddleOCR）。
- [ ] 运行测试确认失败。
- [ ] 实现兼容多版本 PaddleOCR 的初始化/结果适配。
- [ ] 运行 mock 测试确认通过。

### Task 7: 端到端流水线与可视化

**Files:**
- Create: `core/image_utils.py`, `core/pipeline.py`, `core/visualization.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `RecognitionPipeline.run(image, filename, classifier_method, save=False) -> PipelineResult`。

- [ ] 写 OCR-only 降级、无 ROI 配置、保存失败不阻断展示的测试。
- [ ] 运行测试确认失败。
- [ ] 实现流水线、模板配置加载、耗时统计和框绘制。
- [ ] 运行测试确认通过。

### Task 8: Streamlit 管理界面

**Files:**
- Create: `app.py`, `ui/recognition_page.py`, `ui/template_page.py`, `ui/history_page.py`, `ui/experiments_page.py`, `ui/settings_page.py`, `ui/common.py`

**Interfaces:**
- Consumes: `AppConfig`, `RecognitionPipeline`, `RecognitionRepository`。
- Produces: 可启动的 PaddleOCR Studio 风格 Streamlit 管理系统。

- [ ] 实现左侧导航、顶部状态条和双栏识别工作区。
- [ ] 实现结构化字段 / 文本识别 / JSON 标签页与保存按钮。
- [ ] 实现模板 ROI JSON 编辑、历史检索/导出/删除、实验页、设置页。
- [ ] 运行 `streamlit run app.py --server.headless true` 做启动烟测。

### Task 9: 文档、依赖与无数据约束

**Files:**
- Create: `README.md`, `requirements.txt`, `.gitignore`, `data/README.md`, `models/README.md`, `config/templates/example.schema.json`

**Interfaces:**
- Produces: 可复现安装/启动说明、用户数据目录格式、模型路径说明、ROI 配置示例 schema（不含业务值）。

- [ ] 写 README：Windows/CPU 安装、模型放置、数据目录、训练、ROI 配置、运行、实验与导出命令。
- [ ] 确认 `data/`、`models/` 不含任何业务文件或模型权重。
- [ ] 固化依赖并说明 PaddlePaddle 建议按官方环境安装。

### Task 10: 全量验证与 ZIP 打包

**Files:**
- Create: `scripts/verify_project.py`
- Create: `/mnt/data/多模板表单自动识别和管理系统_A方案.zip`

**Interfaces:**
- Produces: 完整可下载项目包。

- [ ] 运行 `pytest -q`。
- [ ] 运行 `python -m compileall -q .`。
- [ ] 执行自检脚本，确认业务数据/模型权重未被打包。
- [ ] 检查 ZIP 清单与 README。
- [ ] 创建 ZIP 并再次列出内容确认。
