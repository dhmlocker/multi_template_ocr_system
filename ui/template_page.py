from __future__ import annotations

import json
import re
from pathlib import Path


def _safe_name(name: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '_', name.strip())
    return cleaned.strip('_') or 'template'


def render(config, pipeline, repository) -> None:
    import streamlit as st

    st.markdown('<div class="ocr-title">模板管理</div>', unsafe_allow_html=True)
    st.caption('每个模板只保存 ROI 配置和你主动上传的 ORB 参考图；系统不会生成业务图片。')
    root = config.templates.config_dir
    root.mkdir(parents=True, exist_ok=True)
    configs = sorted(root.glob('*.json'))
    names = ['（新建模板）'] + [p.name for p in configs]
    selected = st.selectbox('模板配置', names)

    default_payload = {
        'template_name': '',
        'reference_width': 1000,
        'reference_height': 1000,
        'fields': []
    }
    if selected != '（新建模板）':
        try:
            default_text = (root/selected).read_text(encoding='utf-8')
        except Exception:
            default_text = json.dumps(default_payload, ensure_ascii=False, indent=2)
    else:
        default_text = json.dumps(default_payload, ensure_ascii=False, indent=2)
    text = st.text_area('ROI JSON', value=default_text, height=420, key=f'tpl_editor::{selected}')
    c1, c2 = st.columns(2)
    with c1:
        if st.button('校验 JSON', use_container_width=True):
            try:
                payload = json.loads(text)
                required = {'template_name','reference_width','reference_height','fields'}
                missing = required - set(payload)
                if missing:
                    raise ValueError(f'缺少字段：{sorted(missing)}')
                if not isinstance(payload['fields'], list):
                    raise ValueError('fields 必须是数组')
                st.success('JSON 结构校验通过。')
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button('保存模板配置', type='primary', use_container_width=True):
            try:
                payload = json.loads(text)
                name = str(payload.get('template_name','')).strip()
                if not name:
                    raise ValueError('template_name 不能为空')
                if not isinstance(payload.get('fields'), list):
                    raise ValueError('fields 必须是数组')
                target = root / f'{_safe_name(name)}.json'
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                st.success(f'已保存：{target.name}')
            except Exception as exc:
                st.error(f'保存失败：{exc}')

    st.divider()
    st.subheader('ORB 参考模板')
    st.caption('目录结构：models/orb_templates/<模板类别>/<你上传的参考图片>')
    template_name = st.text_input('模板类别名（需与 CNN 类别 / ROI template_name 一致）')
    ref_file = st.file_uploader('上传一张已有参考表单图片', type=['png','jpg','jpeg','bmp','webp'], key='orb_ref_upload')
    if st.button('保存 ORB 参考图', disabled=not(template_name and ref_file)):
        target_dir = config.classifier.orb_reference_dir / _safe_name(template_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(ref_file.name).name
        target.write_bytes(ref_file.getvalue())
        st.success(f'已保存用户提供的参考图：{target.relative_to(config.project_root)}')
