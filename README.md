# 多模板表单自动识别和管理系统

课程设计 · 方向一：基于 PaddleOCRv6 + CNN/ORB 模板分类的多类表单自动识别与管理系统。

- **OCR 引擎**：PaddleOCRv6（中文，CPU/GPU）
- **模板分类**：MobileNetV2 CNN（主方案） + ORB 特征匹配（对比基线）
- **前端**：Streamlit（参考 PaddleOCR AI Studio 风格 UI）
- **三类表单**：医院收费票据 / 增值税发票 / 快递寄件面单
- **结果分析**：图像质量评估、文字/字段置信度、格式校验、识别 JSON 与实验 CSV

## 目录结构

```
multi_template_ocr_system/
├── app.py                     # Streamlit 入口
├── core/                      # 核心模块
│   ├── ocr_engine.py          # PaddleOCR 封装（空路径自动下载模型）
│   ├── cnn_classifier.py      # CNN 模板分类器
│   ├── orb_classifier.py      # ORB 模板分类器
│   ├── pipeline.py            # 识别流程编排
│   ├── types.py               # 数据结构
│   └── ...
├── ui/                        # 各页面组件
├── config/settings.yaml       # 运行配置
├── models/
│   ├── classifier.pth         # CNN 分类器权重（8.7 MB）
│   ├── class_names.json      # 类别列表
│   └── orb_templates/         # ORB 参考图
├── data/
│   ├── samples/               # ✅ 随仓库分发：每类 1 张示例图（开箱即用）
│   ├── train/.gitkeep         # ← 完整训练集需自行下载
│   ├── val/.gitkeep
│   └── test/.gitkeep
└── outputs/                   # 运行时产物（自动创建）
```

## 环境要求

- Python 3.10+
- Windows / Linux / macOS（已在 Windows 验证）
- CPU 即可（OCR 约 1.5 分钟/张，CNN 分类 ~150 ms/张）

## 安装

```bash
# 1. 克隆
git clone https://github.com/dhmlocker/multi_template_ocr_system.git
cd multi_template_ocr_system

# 2. 创建虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. 安装 PaddlePaddle（选其一）
#    CPU 版：
pip install paddlepaddle==3.3.1
#    GPU 版（CUDA 11.8）：
# pip install paddlepaddle-gpu==3.3.1.post118

# 4. 安装其余依赖
pip install paddleocr==3.7.0 streamlit torch torchvision opencv-python pillow pandas
```

## 数据集

项目使用三类真实表单图片，约 650 MB，500+ 张。

**下载方式**：数据集包含个人隐私信息，不直接放入仓库。请联系课程负责人获取网盘链接，下载后按以下结构放置：

```
data/
├── train/
│   ├── 医院收费票据/    # 约 350 张
│   ├── 增值稅发票/      # 约 150 张
│   └── 快递寄件面单/    # 约 80 张
├── val/                 # 同结构，验证集
└── test/                # 同结构，测试集
```

每张图文件名无特殊要求，支持 `.jpg/.jpeg/.png/.bmp/.webp`。

## 模型说明

| 模型 | 大小 | 来源 | 进仓库 |
|------|------|------|--------|
| PP-OCRv6 中文 det + rec | ~132 MB | PaddleOCR 首次运行自动下载 | ❌ |
| CNN classifier.pth | 8.7 MB | 训练产物 | ✅ |
| CNN class_names.json | < 1 KB | 训练产物 | ✅ |
| ORB orb_templates/ | ~4 MB | 从 train/val 挑选的参考图 | ✅ |

> 若需使用本地 PP-OCRv6 模型而非自动下载，将 `config/settings.yaml` 中 `ocr.det_model_dir` 和 `ocr.rec_model_dir` 改为实际路径即可。

## 运行

```bash
# 启动 Streamlit（默认端口 8501）
python -m streamlit run app.py

# 打开浏览器访问
# http://localhost:8501
```

首次启动时 PaddleOCR 会自动下载 PP-OCRv6 模型（约 132 MB），需联网。下载完成后会缓存到本地。

**开箱体验**：`data/samples/` 内置 3 张示例图（医院票据/发票/面单各一张），无需下载数据集即可在"新建识别"页的"示例图片"下拉中选择并运行识别。

### 功能页说明

| 页面 | 功能 |
|------|------|
| **新建识别** | 上传/选择图片 → 模板分类（秒级） → OCR + 字段抽取（CPU ~1.5 min） |
| **历史记录** | 查看已保存的识别结果，与识别时展示完全一致（含原图） |
| **模板管理** | 管理三类模板的 ROI 配置（字段坐标） |
| **实验评估** | CNN vs ORB 分类器对比（准确率/耗时/混淆矩阵） |
| **系统设置** | 运行参数 |

## 训练（可选）

若需重新训练 CNN 分类器：

```bash
# 1. 目录结构准备好 data/train/<类别>/*.jpg
# 2. 训练脚本会自动下载 torchvision MobileNetV2 预训练权重
python -m training.train_cnn

# 3. 产出：
#    models/classifier.pth   (新分类器权重)
#    models/class_names.json (更新类别)
```

## 识别质量分析

系统默认使用 `balanced` 图像增强，对低对比度、轻度模糊和手机拍摄表单进行保守处理。每次识别会记录图像质量分数、亮度、对比度、模糊度、倾斜角、文字平均置信度、低置信度文本数量和字段覆盖率。若图片质量或字段置信度偏低，界面会提示人工复核，而不是把规则校验结果当作绝对正确。

可在 `config/settings.yaml` 中调整 `ocr.preprocess_mode`：`off` 保持原图，`balanced` 为默认方案，`strong` 适用于更困难的图片。对论文实验应固定配置并保存实际输出，不使用模拟指标。

## 已知限制

- CPU 上 OCR 较慢（约 1~2 分钟/张），有 GPU 时切换 `runtime.device: gpu`
- ORB 参考图仅 8 张，CNN 分类器准确率明显优于 ORB（CNN ~100% vs ORB ~82%）
- 中文字体需系统有 msyh.ttc / simhei.ttf（Windows 默认有）

## License

课程设计作品，请勿用于商业用途。
