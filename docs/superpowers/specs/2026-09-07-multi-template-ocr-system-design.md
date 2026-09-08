# 多模板表单自动识别和管理系统 — 设计规格

## 1. 项目定位

- 课程：深度学习与应用课程设计，方向一。
- 系统名称：多模板表单自动识别和管理系统。
- 选题名称：基于深度学习OCR的多模板表单自动识别和管理系统。
- 已有数据类型：增值税发票、医院收费票据、快递寄件面单。
- 数据约束：项目不生成、不内置任何训练集、验证集、测试集或伪造业务表单。数据目录仅提供空目录和说明文件，由使用者自行放入已有数据。
- 模型约束：不重复下载或打包用户已下载的 PaddleOCR 模型；通过配置文件引用本地模型目录。

## 2. 总体技术路线

输入表单图片 → 模板分类 → OCR检测识别 → ROI字段抽取 → 结构化结果 → SQLite存储 → Streamlit界面展示/检索/导出。

模板分类以 CNN 迁移学习为主方案，ORB 特征匹配作为传统计算机视觉对比基线。OCR 使用本地 PP-OCRv6 检测与识别模型。字段抽取采用固定模板 ROI 坐标映射，以文本框中心点落入 ROI 为主规则、IoU 阈值为备用规则。同字段多文本框按阅读顺序合并。

## 3. 架构

### 3.1 表现层

使用 Streamlit 构建 Web 界面，视觉参考 PaddleOCR Studio：

- 左侧导航：新建识别、模板管理、历史记录、实验评估、系统设置。
- 主识别页：左侧显示源文件与原始表单预览；右侧提供“结构化字段 / 文本识别 / JSON”标签页。
- 支持上传单张或多张图片；不自动写入任何样例数据。
- 支持 OCR 框与 ROI 框可视化。
- 支持识别结果人工复核和保存。

### 3.2 业务逻辑层

统一 `RecognitionPipeline` 负责：

1. 读取图像并校验格式。
2. 通过 CNN 进行模板分类；可切换 ORB 基线。
3. 调用 PP-OCRv6 本地模型进行文本检测和识别。
4. 加载当前模板 ROI 配置并执行字段抽取。
5. 组装模板类别、模板置信度、OCR明细、结构化字段、耗时信息。
6. 按需保存 SQLite。

### 3.3 模型层

#### CNN 模板分类

- 默认网络：MobileNetV2。
- 使用 ImageNet 预训练权重进行迁移学习。
- 分类头输出类别数由训练数据目录自动推断。
- 训练脚本只读取用户提供的数据，不创建业务样本。
- 输出最佳权重、类别映射和训练指标。

#### ORB 基线

- 为每类模板参考图建立 ORB 描述子索引。
- 使用 BFMatcher + Hamming 距离进行匹配。
- 输出匹配类别、匹配分数和耗时。
- 用于课程报告中的 CNN vs ORB 对比实验。

#### PP-OCRv6

默认模型目录：

- `models/PP-OCRv6_medium_det`
- `models/PP-OCRv6_medium_rec`

模型路径可在 `config/settings.yaml` 或系统设置页修改。项目不联网下载模型。

为了兼容 PaddleOCR 版本差异，OCR 适配层集中处理初始化参数和返回结构，业务层不直接依赖具体 PaddleOCR 返回对象。

### 3.4 字段抽取层

每个模板对应一个 JSON 配置文件，包含：

- `template_name`
- `reference_width`
- `reference_height`
- `fields[]`
  - `field_name`
  - `bbox: [x1, y1, x2, y2]`
  - 可选 `match_mode`
  - 可选 `iou_threshold`

不同分辨率输入按参考尺寸进行坐标比例换算。

匹配流程：

1. 计算 OCR 文本框中心点。
2. 优先判断中心点是否位于字段 ROI。
3. 若中心点未命中且字段启用 IoU 规则，则按阈值判断。
4. 多文本框按 y、x 排序并拼接。
5. 无结果字段标记为空字符串并记录 `missing=true`，不伪造内容。

### 3.5 数据层

SQLite 使用两张核心表：

- `recognition_records`：识别任务主表，记录文件名、模板、分类方式、置信度、完整 JSON、时间戳和总耗时。
- `recognition_fields`：字段明细表，记录主表ID、字段名、字段值、字段状态。

系统启动时自动建表。数据库仅保存用户真实执行后的识别记录。

## 4. 项目目录

```text
multi_template_ocr_system/
├─ app.py
├─ requirements.txt
├─ README.md
├─ config/
│  ├─ settings.yaml
│  └─ templates/
├─ core/
│  ├─ config.py
│  ├─ image_utils.py
│  ├─ ocr_engine.py
│  ├─ cnn_classifier.py
│  ├─ orb_classifier.py
│  ├─ field_extractor.py
│  ├─ pipeline.py
│  └─ visualization.py
├─ database/
│  ├─ db.py
│  └─ schema.sql
├─ ui/
│  ├─ recognition_page.py
│  ├─ template_page.py
│  ├─ history_page.py
│  ├─ experiments_page.py
│  └─ settings_page.py
├─ training/
│  ├─ dataset.py
│  ├─ train_classifier.py
│  └─ evaluate_classifier.py
├─ tools/
│  ├─ roi_labeler.py
│  └─ init_orb_templates.py
├─ tests/
├─ data/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ models/
├─ outputs/
└─ docs/
```

## 5. UI 设计

### 5.1 识别页

- 左侧窄栏：文件列表、上传入口、最近识别记录。
- 中间：源文件预览，可显示原图 / OCR框 / ROI框。
- 右侧：识别结果。
  - “结构化字段”：表格展示字段名、识别值、状态。
  - “文本识别”：按坐标顺序展示 OCR 文本、置信度。
  - “JSON”：完整结构化结果。
- 顶部：当前模板分类方式、OCR模型路径状态、执行识别按钮。

### 5.2 模板管理

- 查看模板列表。
- 新建/编辑 ROI 配置。
- 上传模板参考图用于 ORB。
- ROI 标注工具采用交互式图像坐标方式，保存到 JSON。

### 5.3 历史记录

- 按日期、模板类型、文件名搜索。
- 查看详情。
- 导出 CSV / JSON。
- 删除选中记录。

### 5.4 实验评估

不生成实验数据，仅对用户指定的数据目录运行：

- CNN vs ORB：准确率、平均推理耗时。
- OCR：平均文本置信度、处理耗时。
- 字段抽取：在用户提供 Ground Truth 时计算字段级准确率。
- 端到端：平均总耗时与可选准确率。
- 结果支持 CSV 导出和 matplotlib 图表生成。

## 6. 配置

`config/settings.yaml` 主要字段：

- `ocr.det_model_dir`
- `ocr.rec_model_dir`
- `ocr.use_angle_cls`
- `ocr.lang`
- `classifier.method`
- `classifier.cnn_model_path`
- `classifier.class_names_path`
- `classifier.orb_reference_dir`
- `database.path`
- `templates.config_dir`
- `runtime.device`

所有路径默认使用项目相对路径。

## 7. 错误处理

- 模型目录不存在：界面显示明确错误，不触发自动下载。
- CNN 权重不存在：允许切换 ORB 或仅 OCR 模式，但不伪装分类结果。
- 模板 ROI 配置不存在：返回 OCR 全文，同时提示“当前模板未配置字段ROI”。
- 图片损坏：拒绝执行并提示格式错误。
- OCR 模型初始化失败：显示依赖或模型路径错误。
- 数据库异常：识别结果仍可显示，但提示保存失败。

## 8. 测试策略

- `field_extractor`：使用程序构造的坐标框单元测试；这些是算法测试坐标，不是业务表单数据。
- SQLite：使用临时数据库测试增删查改。
- 配置加载：测试路径和默认值。
- ORB：使用简单几何测试图验证接口，不写入项目数据目录。
- OCR/CNN 集成测试默认跳过，需本地模型或用户数据时由使用者显式运行。
- 最终打包前执行 `pytest` 和 Python 语法编译检查。

## 9. 数据与隐私边界

- 仓库不包含任何用户真实表单。
- 不生成增值税发票、医院收费票据、快递面单等业务数据。
- README 清楚说明数据放置格式。
- `.gitignore` 默认忽略 `data/**`、`models/**`、`outputs/**` 和本地 SQLite 数据库，仅保留说明文件/空目录标记。

## 10. 验收标准

1. 在 Windows/CPU 环境安装依赖后可启动 Streamlit。
2. 本地 PP-OCRv6 检测/识别模型路径可配置且不联网下载。
3. 数据目录为空，用户可自行放入三类已有数据。
4. CNN 训练脚本能读取用户数据并产出分类模型。
5. ORB 基线可建立参考模板索引并执行分类。
6. 识别流水线可输出 OCR 文本和结构化 JSON。
7. 配置 ROI 后可抽取对应字段。
8. 识别记录可保存、查询、查看与导出。
9. 实验页可对用户数据运行 CNN/ORB 对比并输出结果。
10. `pytest` 与语法检查通过后生成 ZIP 代码包。
