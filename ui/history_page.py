from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.types import OCRItem
from core.visualization import draw_ocr_boxes
from ui.common import (
    ICON_LAYOUT,
    ICON_REFRESH,
    ICON_TEXT,
    doc_layout_html,
    reading_order_html,
)


def _dict_to_ocr_items(raw_items: list[dict]) -> list[OCRItem]:
    return [OCRItem(box=it['box'], text=it['text'], confidence=it.get('confidence', 0.0)) for it in raw_items]


def _load_image(image_path: str | None) -> np.ndarray | None:
    if not image_path:
        return None
    p = Path(image_path)
    if not p.exists():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _render_detail(config, repository, record: dict) -> None:
    import pandas as pd
    import streamlit as st

    result = record.get('result') or {}
    ocr_items = _dict_to_ocr_items(result.get('ocr_items') or [])
    fields = result.get('fields') or []
    classification = result.get('classification')
    img = _load_image(record.get('image_path'))

    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem;">'
        f'<div><span style="font-size:1.15rem;font-weight:700;">{record.get("filename", "-")}</span>'
        f' <span style="color:#8a909e;font-size:.85rem;">· {record.get("created_at", "-")} · '
        f'{record.get("total_elapsed_ms", 0)/1000:.1f}s</span></div></div>',
        unsafe_allow_html=True,
    )

    if classification:
        conf = float(classification.get('confidence', 0))
        color = '#22a06b' if conf >= 0.6 else ('#e8a13a' if conf >= 0.4 else '#e8686a')
        scores = sorted((classification.get('scores') or {}).items(), key=lambda kv: -kv[1])
        scores_html = ''
        if scores:
            rows = ''.join(
                f'<div class="cls-score-row" style="margin:.25rem 0;">'
                f'<span style="width:6rem;">{n}</span>'
                f'<span class="bar"><span style="width:{s*100:.1f}%"></span></span>'
                f'<span style="width:3rem;text-align:right;">{s*100:.1f}%</span></div>'
                for n, s in scores
            )
            scores_html = f'<div style="margin-top:.4rem;">{rows}</div>'
        st.markdown(
            f'<div class="cls-banner"><span class="lbl">模板分类</span>'
            f'<span class="val">{classification.get("label") or "未识别"}</span>'
            f'<span style="color:{color};">●</span>'
            f'<span class="conf" style="color:{color};">{conf*100:.1f}%</span>'
            f'<span class="lbl">{(classification.get("method") or "").upper()}</span></div>{scores_html}',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1, 1], gap='large')

    with left:
        st.markdown('<div class="panel-title">Source File</div>', unsafe_allow_html=True)
        if img is not None:
            st.markdown('<div class="source-card">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f'{img.shape[1]}×{img.shape[0]}')
        else:
            st.info('原图未保存（旧记录无图片）。')

    with right:
        st.markdown('<div class="result-toolbar">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="tb-left"><span style="font-size:.85rem;color:#8a909e;">Parsing model</span>'
            f'<span style="font-weight:600;font-size:.9rem;">PP-OCRv6</span></div>'
            f'<div class="tb-right">'
            f'<span class="icon-btn">{ICON_TEXT}</span>'
            f'<span class="icon-btn">{ICON_LAYOUT}</span>'
            f'<span class="icon-btn">{ICON_REFRESH}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        tab_text, tab_fields, tab_json = st.tabs(['文本识别', '结构化字段', 'JSON'])
        with tab_text:
            if img is not None:
                h, w = img.shape[:2]
                st.markdown(doc_layout_html(ocr_items, w, h), unsafe_allow_html=True)
            else:
                st.markdown(reading_order_html(ocr_items), unsafe_allow_html=True)
        with tab_fields:
            if fields:
                rows = [{'字段名': f['field_name'], '识别值': f['value'],
                         '状态': '未识别' if f.get('missing') else '已识别'} for f in fields]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info('该记录没有结构化字段。')
        with tab_json:
            st.json(result, expanded=False)

    back1, back2 = st.columns(2)
    with back1:
        if st.button('← 返回记录列表', use_container_width=True):
            st.session_state.pop('open_record', None)
            st.rerun()
    with back2:
        if st.button('删除该记录', use_container_width=True):
            repository.delete_record(int(record['id']))
            st.session_state.pop('open_record', None)
            st.success('已删除。')
            st.rerun()


def render(config, pipeline, repository) -> None:
    import pandas as pd
    import streamlit as st

    open_id = st.session_state.get('open_record')
    if open_id is not None:
        record = repository.get_record(int(open_id))
        if record is None:
            st.warning(f'记录 {open_id} 不存在，可能已被删除。')
            st.session_state.pop('open_record', None)
        else:
            _render_detail(config, repository, record)
            return

    st.markdown('<div class="ocr-title">历史记录</div>', unsafe_allow_html=True)
    st.caption('点击侧边栏 Recents 可直接回看对应记录；详情页展示与识别时完全一致的原图与结果。')

    rows = repository.list_records(limit=200)
    if not rows:
        st.info('数据库中暂无识别记录。到"新建识别"完成一次识别并保存即可。')
        return

    df = pd.DataFrame(rows)[['id', 'filename', 'template_name', 'classifier_method', 'classifier_confidence', 'total_elapsed_ms', 'created_at']]
    df.columns = ['ID', '文件名', '模板', '分类方式', '置信度', '耗时(ms)', '时间']
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        rid = st.selectbox('查看详情', [int(r['id']) for r in rows], format_func=lambda i: f'记录 {i}')
    with c2:
        st.download_button('导出 CSV', df.to_csv(index=False).encode('utf-8-sig'), 'recognition_history.csv', 'text/csv', use_container_width=True)
    with c3:
        st.download_button('导出 JSON', json.dumps(rows, ensure_ascii=False, indent=2).encode('utf-8'), 'recognition_history.json', 'application/json', use_container_width=True)

    if st.button('打开选中记录详情', type='primary'):
        st.session_state['open_record'] = int(rid)
        st.rerun()
