from __future__ import annotations

from pathlib import Path
from typing import Any

from core.types import OCRItem

APP_CSS = r'''
<style>
:root { --panel-border: #eceef2; --muted: #8a909e; --brand: #ff4d4f; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 1700px; }
.stApp { background: #ffffff; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--panel-border); }
[data-testid="stSidebar"] .stButton > button { width: 100%; text-align: left; border: none; background: transparent; box-shadow: none; padding: .25rem .5rem; font-size: .88rem; border-radius: 8px; }
[data-testid="stSidebar"] .stButton > button:hover { background: #f2f3f7; }
[data-testid="stSidebar"] .stButton > button[kind="primary"],
.stButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); color: white; }
.stButton > button[kind="primary"]:hover { background: #e64340; border-color: #e64340; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #ffffff !important; border: 1px solid #d0d3da !important; color: #1f2430 !important; border-radius: 8px; font-weight: 500; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover { background: #f5f6f8 !important; border-color: #bcbfc7 !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]::before { content: "+ "; font-size: 1.05rem; font-weight: 400; }
.panel-title { font-size:.92rem; font-weight:600; margin-bottom:.4rem; color:#1f2430; }
.small-muted { color:var(--muted); font-size:.8rem; }
.ocr-title { font-size: 1.3rem; font-weight: 700; margin-bottom: .15rem; }
.status-row { display:flex; gap:.6rem; flex-wrap:wrap; margin:.25rem 0 .8rem 0; }
.status-pill { border:1px solid var(--panel-border); border-radius:999px; padding:.2rem .7rem; font-size:.8rem; background:white; color:#3b4256; }
/* 文档式文本识别结果：统一字号、行高，行内文字按原始横向间距排列 */
.doc-text { border:1px solid var(--panel-border); border-radius:10px; background:white; padding:1.2rem 1.4rem; font-size:13px; line-height:1.9; color:#1f2430; box-shadow: 0 1px 2px rgba(0,0,0,.02); overflow-x:auto; white-space:pre; font-family:'Courier New',Consolas,monospace; }
.doc-text > div { margin: .1rem 0; }
/* 图片卡 */
.source-card { border:1px solid var(--panel-border); border-radius:10px; background:#fafbfc; padding:.6rem; }
/* 结果区工具栏 */
.result-toolbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:.6rem; }
.result-toolbar .tb-left { display:flex; align-items:center; gap:.5rem; }
.result-toolbar .tb-right { display:flex; gap:.25rem; }
.icon-btn { display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:6px; background:transparent; border:1px solid transparent; cursor:pointer; color:#5b6272; }
.icon-btn:hover { background:#f2f3f7; border-color:#e4e6ec; }
/* 分类结果轻量条 */
.cls-banner { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; padding:.5rem .9rem; background:#f7f8fa; border:1px solid var(--panel-border); border-radius:8px; margin-bottom:.7rem; font-size:.88rem; }
.cls-banner .lbl { color:var(--muted); }
.cls-banner .val { font-weight:600; color:#1f2430; }
.cls-banner .conf { font-weight:600; }
.cls-score-row { display:flex; align-items:center; gap:.6rem; font-size:.8rem; color:#5b6272; }
.cls-score-row .bar { flex:1; background:#eef0f5; border-radius:4px; height:8px; overflow:hidden; }
.cls-score-row .bar > div { height:100%; background:#ff4d4f; }
/* 页签 */
[data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid var(--panel-border); padding-bottom:0 !important; }
[data-baseweb="tab"] { padding:.5rem 0 !important; font-size:.95rem; }
[data-baseweb="tab"][aria-selected="true"] { color:#1f2430 !important; }
[data-baseweb="tab-highlight"] { background:#ff4d4f !important; height:2px !important; }
</style>
'''

# 线性 SVG 图标
ICON_GEAR = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
ICON_TEXT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
ICON_LAYOUT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>'
ICON_REFRESH = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'
ICON_DOC = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'


def path_status(path: Path) -> tuple[str, bool]:
    ok = Path(path).exists()
    return ("就绪" if ok else "缺失", ok)


def sort_reading_order(items: list[OCRItem]) -> list[list[OCRItem]]:
    """按阅读顺序聚类成行。

    改进点：
    - 用框顶部 y1 分行（而非中心），避免大高度框干扰；
    - 过滤高度异常大的框（>3 倍中位数），通常是 OCR 检测到的区域块而非文字行；
    - 分行阈值 = 中位数框高 × 0.8。
    """
    if not items:
        return []
    heights = sorted(max(it.bounds[3] - it.bounds[1], 1.0) for it in items)
    median_h = heights[len(heights) // 2]
    max_h = median_h * 3.0
    tol = max(median_h * 0.8, 12.0)

    filtered = [it for it in items if (it.bounds[3] - it.bounds[1]) <= max_h]
    ordered = sorted(filtered, key=lambda it: (it.bounds[1], it.center[0]))

    lines: list[list[OCRItem]] = []
    current: list[OCRItem] = []
    cur_y: float | None = None
    for it in ordered:
        y1 = it.bounds[1]
        if cur_y is None or abs(y1 - cur_y) <= tol:
            current.append(it)
            cur_y = y1 if cur_y is None else (cur_y + y1) / 2.0
        else:
            lines.append(sorted(current, key=lambda i: i.center[0]))
            current, cur_y = [it], y1
    if current:
        lines.append(sorted(current, key=lambda i: i.center[0]))
    return lines


def reading_order_html(items: list[OCRItem]) -> str:
    lines = sort_reading_order(items)
    if not lines:
        return '<div class="doc-text">未识别到文字。</div>'
    parts = ['<div class="doc-text">']
    for line in lines:
        parts.append('<div>' + ' '.join(str(it.text) for it in line) + '</div>')
    parts.append('</div>')
    return ''.join(parts)


def _char_width(text: str) -> int:
    """估算等宽字体下的字符宽度：中文/全角占 2，ASCII/半角占 1。"""
    w = 0
    for ch in text:
        if ord(ch) > 0x2E80:  # CJK 及全角范围
            w += 2
        else:
            w += 1
    return w


def doc_layout_html(items: list[OCRItem], img_w: float, img_h: float) -> str:
    """按阅读顺序分行，用等宽字体 + 空格填充还原文档横向布局（AI Studio 风格）。

    实现：
    - 按 y 分行、行内按 x 排序；
    - 把图片宽度映射到固定字符宽度（默认 100 列）；
    - 每个文字根据 x1 计算起始列，用空格填充到该位置。
    """
    lines = sort_reading_order(items)
    if not lines:
        return '<div class="doc-text">未识别到文字。</div>'

    COLS = 100
    scale = COLS / max(img_w, 1.0)
    style = 'font-family:Consolas,"Courier New",monospace;font-size:13px;line-height:1.9;color:#1f2430;background:#fff;border:1px solid #eceef2;border-radius:10px;padding:1.2rem 1.4rem;overflow-x:auto;white-space:pre;'
    parts = [f'<pre style="{style}">']
    for line in lines:
        sorted_line = sorted(line, key=lambda it: it.bounds[0])
        cols: list[str] = [' '] * (COLS + 200)
        for it in sorted_line:
            x1 = it.bounds[0]
            col = int(x1 * scale)
            text = str(it.text)
            for i, ch in enumerate(text):
                pos = col + i
                if 0 <= pos < len(cols):
                    cols[pos] = ch
        line_str = ''.join(cols).rstrip()
        parts.append(line_str)
    parts.append('</pre>')
    return '\n'.join(parts)


def result_field_rows(result: Any) -> list[dict[str, Any]]:
    return [
        {'字段名': f.field_name, '识别值': f.value, '状态': '未识别' if f.missing else '已识别'}
        for f in result.fields
    ]


def result_ocr_rows(result: Any) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(result.ocr_items, 1):
        rows.append({'序号': i, '文本': item.text, '置信度': round(float(item.confidence), 4)})
    return rows


def sample_images(config, limit: int = 50) -> list[Path]:
    root: Path = config.project_root
    picks: list[Path] = []
    for base in (root / 'data' / 'test', root / 'data' / 'val'):
        if not base.is_dir():
            continue
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
                picks.append(p)
        if len(picks) >= limit:
            break
    return picks[:limit]


def default_classifier_method(config) -> str:
    if Path(config.classifier.cnn_model_path).is_file():
        return 'cnn'
    if Path(config.classifier.orb_reference_dir).is_dir() and any(Path(config.classifier.orb_reference_dir).iterdir()):
        return 'orb'
    return 'ocr_only'


def template_class_dirs(config) -> list[str]:
    names: list[str] = []
    train_dir = config.project_root / 'data' / 'train'
    if train_dir.is_dir():
        names.extend(p.name for p in sorted(train_dir.iterdir()) if p.is_dir())
    for p in sorted(config.templates.config_dir.glob('*.json')):
        if p.stem not in names:
            names.append(p.stem)
    return names
