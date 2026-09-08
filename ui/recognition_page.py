from __future__ import annotations

from core.image_utils import bgr_to_rgb, decode_image
from core.analysis import summarize_result
from core.types import ClassificationResult
from core.visualization import draw_all, draw_ocr_boxes, draw_roi_boxes
from ui.common import (
    ICON_LAYOUT,
    ICON_REFRESH,
    ICON_TEXT,
    default_classifier_method,
    doc_layout_html,
    result_field_rows,
    result_ocr_rows,
    sample_images,
)

VIEW_OPTIONS = ['原图', 'OCR 框', 'OCR + ROI']
METHOD_LABELS = {'cnn': 'CNN（主方案）', 'orb': 'ORB（对比基线）', 'ocr_only': '仅 OCR，不分类'}


def _classification_banner(c: ClassificationResult | None, warnings: list[str]) -> None:
    import streamlit as st

    if c is None:
        return
    color = '#22a06b' if c.confidence >= 0.6 else ('#e8a13a' if c.confidence >= 0.4 else '#e8686a')
    scores = sorted(c.scores.items(), key=lambda kv: -kv[1]) if c.scores else []
    scores_html = ''
    if scores:
        rows = ''.join(
            f'<div class="cls-score-row" style="margin:.25rem 0;">'
            f'<span style="width:6rem;">{name}</span>'
            f'<span class="bar"><span style="width:{score*100:.1f}%"></span></span>'
            f'<span style="width:3rem;text-align:right;">{score*100:.1f}%</span></div>'
            for name, score in scores
        )
        scores_html = f'<div style="margin-top:.4rem;">{rows}</div>'
    st.markdown(
        f'<div class="cls-banner">'
        f'<span class="lbl">模板分类</span>'
        f'<span class="val">{c.label or "未识别"}</span>'
        f'<span style="color:{color};">●</span>'
        f'<span class="conf" style="color:{color};">{c.confidence*100:.1f}%</span>'
        f'<span class="lbl">{c.method.upper()} · {c.elapsed_ms:.0f} ms</span>'
        f'</div>{scores_html}',
        unsafe_allow_html=True,
    )
    for w in (warnings or []):
        st.warning(w)


def _render_result(result, image) -> None:
    import pandas as pd
    import streamlit as st

    tab_text, tab_fields, tab_json = st.tabs(['文本识别', '结构化字段', 'JSON'])
    with tab_text:
        if image is not None:
            h, w = image.shape[:2]
            st.markdown(doc_layout_html(result.ocr_items, w, h), unsafe_allow_html=True)
        else:
            from ui.common import reading_order_html
            st.markdown(reading_order_html(result.ocr_items), unsafe_allow_html=True)
        rows = result_ocr_rows(result)
        if rows:
            with st.expander('全部文本框明细'):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tab_fields:
        rows = result_field_rows(result)
        if rows:
            st.caption(f'已根据模板"{result.template_name}"的字段配置抽取结果；低置信度字段建议人工复核。')
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info('当前结果没有结构化字段：模板未分类或该模板尚未配置 ROI。')
    with tab_json:
        st.json(result.to_dict(), expanded=False)

    total = result.timings_ms.get('total', 0.0)
    summary = summarize_result(result)
    metric_cols = st.columns(4)
    metric_cols[0].metric('平均文字置信度', f'{summary["avg_ocr_confidence"] * 100:.1f}%')
    metric_cols[1].metric('字段覆盖率', f'{summary["field_coverage"] * 100:.1f}%')
    metric_cols[2].metric('图像质量', f'{summary["quality_score"]:.0f}/100')
    metric_cols[3].metric('待复核文本', str(summary['low_confidence_items']))
    if result.quality.get('warnings'):
        with st.expander('图像质量分析', expanded=False):
            for warning in result.quality['warnings']:
                st.warning(warning)
    st.download_button(
        '下载识别 JSON', data=__import__('json').dumps(result.to_dict(), ensure_ascii=False, indent=2),
        file_name=f'{result.filename}.json', mime='application/json', use_container_width=True,
    )
    st.caption(
        f'总耗时 {total/1000:.1f} s · 分类 {result.timings_ms.get("classification", 0)/1000:.2f} s · '
        f'OCR {result.timings_ms.get("ocr", 0)/1000:.1f} s · 字段抽取 {result.timings_ms.get("field_extraction", 0)/1000:.2f} s'
    )


def render(config, pipeline, repository) -> None:
    import streamlit as st

    samples = sample_images(config)
    sample_names = ['（使用示例图片）'] + [f'{p.parent.name}/{p.name}' for p in samples]

    up_col, sample_col, method_col = st.columns([2.2, 1.6, 1.4])
    with up_col:
        uploaded = st.file_uploader(
            '上传表单图片', type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'],
            label_visibility='collapsed', help='支持上传自己准备的真实图片；示例图片来自本地 data/test。',
        )
    with sample_col:
        picked_sample = st.selectbox('示例图片', sample_names, key='rec_sample', label_visibility='collapsed')
    with method_col:
        default_method = default_classifier_method(config)
        method = st.selectbox(
            '模板分类方式', ['cnn', 'orb', 'ocr_only'],
            index=['cnn', 'orb', 'ocr_only'].index(default_method),
            format_func=METHOD_LABELS.__getitem__, key='rec_method', label_visibility='collapsed',
        )

    if uploaded is not None:
        payload, display_name = uploaded.getvalue(), uploaded.name
    elif picked_sample != sample_names[0]:
        path = next(p for p in samples if f'{p.parent.name}/{p.name}' == picked_sample)
        payload, display_name = path.read_bytes(), path.name
    else:
        payload, display_name = None, None

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button('执行识别', type='primary', use_container_width=True, disabled=payload is None)

    cls_state_key = f'cls_result::{display_name}' if display_name else None
    ocr_state_key = f'ocr_result::{display_name}' if display_name else None
    result = None
    image = None

    if payload is not None:
        if run_clicked:
            with st.spinner('正在进行模板分类...'):
                bgr, cls_result, cls_warnings, cls_ms = pipeline.classify(payload, method)
            st.session_state[cls_state_key] = (cls_result, cls_warnings)
            try:
                with st.spinner('正在执行 PP-OCRv6 文字识别与字段抽取（CPU 下约 1~2 分钟）...'):
                    st.session_state[ocr_state_key] = pipeline.run(
                        bgr, display_name, classifier_method=method, save=False,
                        reuse_classification=(bgr, cls_result, cls_warnings, cls_ms),
                    )
            except Exception as exc:
                st.error(f'识别失败：{exc}')

        cached_cls = st.session_state.get(cls_state_key)
        if cached_cls is not None:
            _classification_banner(cached_cls[0], cached_cls[1])
        result = st.session_state.get(ocr_state_key)
        image = decode_image(payload)

    left, right = st.columns([1, 1], gap='large')

    with left:
        st.markdown('<div class="panel-title">Source File</div>', unsafe_allow_html=True)
        if image is None:
            st.markdown(
                '<div class="source-card" style="display:flex;align-items:center;justify-content:center;height:300px;color:#b8bcc6;">'
                '上传或选择示例图片后显示</div>', unsafe_allow_html=True,
            )
        else:
            view_state = st.session_state.get('rec_view', '原图')
            vcols = st.columns(len(VIEW_OPTIONS))
            for i, opt in enumerate(VIEW_OPTIONS):
                with vcols[i]:
                    if st.button(opt, key=f'view_{opt}', use_container_width=True,
                                 type='primary' if view_state == opt else 'secondary'):
                        st.session_state['rec_view'] = opt
                        st.rerun()
            shown = image
            if result is not None:
                vm = st.session_state.get('rec_view', '原图')
                if vm == 'OCR 框':
                    shown = draw_ocr_boxes(image, result.ocr_items)
                elif vm == 'OCR + ROI':
                    shown = draw_all(image, result.ocr_items, result.fields)
            st.markdown('<div class="source-card">', unsafe_allow_html=True)
            st.image(bgr_to_rgb(shown), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f'{display_name} · {image.shape[1]}×{image.shape[0]}')

    with right:
        st.markdown('<div class="result-toolbar">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="tb-left"><span style="font-size:.85rem;color:#8a909e;">Parsing model</span>'
            f'<span style="font-weight:600;font-size:.9rem;">PP-OCRv6</span></div>'
            f'<div class="tb-right">'
            f'<span class="icon-btn" title="文本视图">{ICON_TEXT}</span>'
            f'<span class="icon-btn" title="布局视图">{ICON_LAYOUT}</span>'
            f'<span class="icon-btn" title="刷新">{ICON_REFRESH}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if result is None:
            st.info('选择图片并点击"执行识别"后，这里显示文本识别、结构化字段与 JSON 结果。')
        else:
            if result.record_id is None:
                if st.button('保存到历史记录', type='primary', use_container_width=True):
                    try:
                        rid = repository.save_result(result, image_bgr=image)
                        result.record_id = rid
                        st.success(f'已保存，记录 ID：{rid}（含原图）。侧边栏 Recents 已更新。')
                        st.rerun()
                    except Exception as exc:
                        st.error(f'保存失败：{exc}')
            else:
                st.success(f'该结果已保存为记录 #{result.record_id}。')
            _render_result(result, image)
