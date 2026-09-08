from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.config import AppConfig
from core.pipeline import RecognitionPipeline
from database.db import RecognitionRepository
from ui.common import APP_CSS, ICON_DOC
from ui import experiments_page, history_page, recognition_page, settings_page, template_page

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'config' / 'settings.yaml'

st.set_page_config(page_title='多模板表单自动识别和管理系统', page_icon='📄', layout='wide', initial_sidebar_state='expanded')
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_services(settings_mtime: float):
    cfg = AppConfig.load(SETTINGS, project_root=ROOT)
    repo = RecognitionRepository(cfg.database.path)
    repo.initialize()
    pipe = RecognitionPipeline(cfg, repository=repo)
    return cfg, pipe, repo


config, pipeline, repository = load_services(SETTINGS.stat().st_mtime)

NAV_OPTIONS = ['新建识别', '历史记录', '模板管理', '实验评估', '系统设置']
st.session_state.setdefault('nav', '新建识别')

with st.sidebar:
    st.markdown('## 多模板表单识别')
    st.caption('多模板表单自动识别和管理系统')

    def _go_new():
        st.session_state['nav'] = '新建识别'
        st.session_state.pop('open_record', None)

    st.button('新建识别', type='primary', use_container_width=True, on_click=_go_new, key='sidebar_new_btn')

    st.divider()

    st.markdown(f'<div style="display:flex;align-items:center;gap:.35rem;font-size:.85rem;font-weight:600;color:#1f2430;margin-bottom:.4rem;">{ICON_DOC} Recents</div>', unsafe_allow_html=True)
    try:
        recents = repository.list_records(limit=8)
    except Exception:
        recents = []
    if recents:
        for r in recents:
            def _open(rid=int(r['id'])):
                st.session_state['nav'] = '历史记录'
                st.session_state['open_record'] = rid
            label = str(r['filename'])[:26]
            st.button(label, key=f'recent_{r["id"]}', on_click=_open, help=f"{r['filename']} · {r['created_at']}")
    else:
        st.caption('暂无记录。执行识别并保存后会出现在这里。')

    st.divider()
    page = st.radio('导航', NAV_OPTIONS, key='nav', label_visibility='collapsed')

pages = {
    '新建识别': recognition_page.render,
    '历史记录': history_page.render,
    '模板管理': template_page.render,
    '实验评估': experiments_page.render,
    '系统设置': settings_page.render,
}
pages[page](config, pipeline, repository)
