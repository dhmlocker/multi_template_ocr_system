from __future__ import annotations

from pathlib import Path


def model_status_rows(config) -> list[dict[str, str]]:
    status = {
        '检测模型': Path(config.ocr.det_model_dir).is_dir(),
        '识别模型': Path(config.ocr.rec_model_dir).is_dir(),
        'CNN 权重': Path(config.classifier.cnn_model_path).is_file(),
        '类别映射': Path(config.classifier.class_names_path).is_file(),
        'MobileNetV2预训练': Path(config.classifier.pretrained_backbone_path).is_file(),
        'ORB 目录': Path(config.classifier.orb_reference_dir).is_dir(),
    }
    return [{'项目': k, '状态': '就绪' if v else '缺失'} for k, v in status.items()]


def render(config, pipeline, repository) -> None:
    import streamlit as st

    st.markdown('<div class="ocr-title">系统设置</div>', unsafe_allow_html=True)
    st.caption('所有路径均可改为绝对路径；保存后刷新页面使新配置完全生效。')
    det = st.text_input('PP-OCRv6 检测模型目录', value=str(config.ocr.det_model_dir))
    rec = st.text_input('PP-OCRv6 识别模型目录', value=str(config.ocr.rec_model_dir))
    cnn = st.text_input('CNN 分类模型', value=str(config.classifier.cnn_model_path))
    names = st.text_input('CNN 类别映射 JSON', value=str(config.classifier.class_names_path))
    backbone = st.text_input('MobileNetV2 ImageNet 本地预训练权重', value=str(config.classifier.pretrained_backbone_path))
    orb = st.text_input('ORB 参考模板目录', value=str(config.classifier.orb_reference_dir))
    method = st.selectbox('默认模板分类方式', ['cnn','orb','ocr_only'], index=['cnn','orb','ocr_only'].index(config.classifier.method if config.classifier.method in {'cnn','orb','ocr_only'} else 'cnn'))
    device = st.selectbox('运行设备', ['cpu','cuda'], index=0 if config.runtime.device == 'cpu' else 1)

    # Build status rows from the currently edited values rather than only the loaded config.
    edited = type('Edited', (), {})()
    edited.ocr = type('OCR', (), {'det_model_dir': Path(det), 'rec_model_dir': Path(rec)})()
    edited.classifier = type('Classifier', (), {
        'cnn_model_path': Path(cnn),
        'class_names_path': Path(names),
        'pretrained_backbone_path': Path(backbone),
        'orb_reference_dir': Path(orb),
    })()
    st.dataframe(model_status_rows(edited), use_container_width=True, hide_index=True)

    if st.button('保存设置', type='primary'):
        config.ocr.det_model_dir = Path(det)
        config.ocr.rec_model_dir = Path(rec)
        config.classifier.cnn_model_path = Path(cnn)
        config.classifier.class_names_path = Path(names)
        config.classifier.pretrained_backbone_path = Path(backbone)
        config.classifier.orb_reference_dir = Path(orb)
        config.classifier.method = method
        config.runtime.device = device
        config.save(config.project_root/'config/settings.yaml')
        st.success('设置已保存到 config/settings.yaml。')
