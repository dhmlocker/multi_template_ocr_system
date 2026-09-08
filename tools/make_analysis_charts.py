from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs' / 'demo_samples' / 'demo_summary.csv'
OUT = ROOT / 'outputs' / 'demo_samples' / 'charts'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
with DATA.open(encoding='utf-8-sig', newline='') as handle:
    rows = list(csv.DictReader(handle))
labels = [r['ground_truth_template'] for r in rows]
coverage = [float(r['field_coverage']) * 100 for r in rows]
confidence = [float(r['avg_ocr_confidence']) * 100 for r in rows]
quality = [float(r['quality_score']) for r in rows]
latency = [float(r['total_ms']) / 1000 for r in rows]

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
x = np.arange(len(labels))
width = 0.36
ax.bar(x - width/2, coverage, width, label='字段覆盖率', color='#1A365D')
ax.bar(x + width/2, confidence, width, label='平均 OCR 置信度', color='#C92A2A')
ax.set_ylim(0, 110)
ax.set_ylabel('百分比 (%)')
ax.set_title('三类表单识别质量：字段覆盖率与 OCR 置信度')
ax.set_xticks(x, labels)
ax.legend()
ax.grid(axis='y', alpha=.25)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'quality_comparison.png', bbox_inches='tight')
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(10, 5.6), dpi=180)
bar = ax1.bar(labels, latency, color='#1A365D', label='端到端耗时')
ax1.set_ylabel('耗时 (秒)')
ax1.set_title('三类表单识别耗时与输入质量')
ax1.grid(axis='y', alpha=.25)
ax1.bar_label(bar, fmt='%.1f s', padding=2, fontsize=8)
ax2 = ax1.twinx()
line = ax2.plot(labels, quality, color='#C92A2A', marker='o', linewidth=2.5, label='图像质量分')
ax2.set_ylabel('图像质量分 (0–100)')
ax2.set_ylim(0, 100)
for xval, yval in zip(labels, quality):
    ax2.annotate(f'{yval:.1f}', (xval, yval), textcoords='offset points', xytext=(0, 8), ha='center', color='#C92A2A', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'latency_quality.png', bbox_inches='tight')
plt.close(fig)

(OUT / 'README.md').write_text(
    '# 三类样例真实数据图表\n\n'
    '图表由 `tools/make_analysis_charts.py` 根据 `demo_summary.csv` 生成。样本量为每类 1 张，仅用于课程设计案例展示，不代表大样本泛化性能。\n', encoding='utf-8')
print(f'Wrote charts to {OUT}')
