# 本地模型目录说明

代码包不附带任何模型权重，也不会自动下载 PP-OCRv6 模型。

默认期望你已经下载的 OCR 模型位于：

```text
models/
├─ PP-OCRv6_medium_det/
└─ PP-OCRv6_medium_rec/
```

如果你已经把模型放在其他目录，可直接修改 `config/settings.yaml`，也可在系统“系统设置”页填绝对路径。

你截图中已有的 MobileNetV2 ImageNet 预训练权重可放为：

```text
models/mobilenet_v2_imagenet.pth
```

训练脚本默认优先读取这个本地文件，不会隐式下载；若不存在会提示你使用 `--no-pretrained` 或显式 `--allow-torchvision-download`。

CNN 模板分类训练完成后会生成：

```text
models/classifier.pth
models/class_names.json
models/training_metrics.json
```

ORB 参考图由你自己上传或放置：

```text
models/orb_templates/<模板类别>/<参考表单图片>
```

这些文件和目录均被 `.gitignore` 忽略，避免把数据或权重误打包进课程设计代码附件。
