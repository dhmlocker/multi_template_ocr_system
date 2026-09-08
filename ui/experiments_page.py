from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from core.cnn_classifier import CNNClassifier
from core.orb_classifier import ORBClassifier
from training.dataset import IMAGE_EXTS, validate_imagefolder_layout


def _evaluate(data_dir: Path, classifier, class_names: list[str]) -> dict:
    classes = validate_imagefolder_layout(data_dir)
    all_images: list[tuple[str, Path]] = []
    for label in classes:
        for path in (data_dir / label).rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                all_images.append((label, path))

    total = correct = 0
    elapsed_ms = 0.0
    per_class: dict[str, dict] = {c: {'total': 0, 'correct': 0} for c in classes}
    confusion: dict[str, dict[str, int]] = {c: {p: 0 for p in classes} for c in classes}

    for label, path in all_images:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        t = time.perf_counter()
        result = classifier.predict(image)
        elapsed_ms += (time.perf_counter() - t) * 1000
        total += 1
        pred = result.label or '未识别'
        per_class[label]['total'] += 1
        if pred == label:
            correct += 1
            per_class[label]['correct'] += 1
        confusion[label][pred if pred in confusion[label] else '未识别'] = confusion[label].get(pred, 0) + 1

    per_class_acc = {c: (d['correct'] / d['total'] if d['total'] else 0.0) for c, d in per_class.items()}
    return {
        'samples': total,
        'accuracy': correct / max(total, 1),
        'avg_ms': elapsed_ms / max(total, 1),
        'per_class': per_class_acc,
        'confusion': confusion,
    }


def render(config, pipeline, repository) -> None:
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="ocr-title">实验评估</div>', unsafe_allow_html=True)
    st.caption('只评估你已经放入目录的数据，不生成任何实验样本。建议目录按 data/test/<类别名>/*.jpg 组织。')

    data_dir_text = st.text_input('测试集目录', value=str(config.project_root / 'data/test'))
    methods = st.multiselect('比较方法', ['CNN', 'ORB'], default=['CNN', 'ORB'])

    run = st.button('运行模板分类对比实验', type='primary')
    if run:
        data_dir = Path(data_dir_text)
        if not data_dir.is_dir():
            st.error(f'目录不存在：{data_dir}')
            return
        try:
            classes = validate_imagefolder_layout(data_dir)
        except Exception as exc:
            st.error(f'目录结构校验失败：{exc}')
            return

        results = []
        with st.status('正在运行实验...', expanded=True) as status:
            for method in methods:
                if method == 'CNN':
                    st.write('加载 CNN 分类器...')
                    clf = CNNClassifier(config.classifier.cnn_model_path, config.classifier.class_names_path, config.runtime.device)
                    st.write(f'CNN 评估中（{len(classes)} 类）...')
                    r = _evaluate(data_dir, clf, classes)
                    r['method'] = 'CNN'
                    results.append(r)
                    st.write(f'CNN 完成：{r["samples"]} 张，准确率 {r["accuracy"]*100:.1f}%')
                elif method == 'ORB':
                    st.write('加载 ORB 分类器并建索引...')
                    clf = ORBClassifier(config.classifier.orb_reference_dir)
                    n = clf.build_index()
                    st.write(f'ORB 索引完成：{n} 个描述子')
                    st.write(f'ORB 评估中（{len(classes)} 类）...')
                    r = _evaluate(data_dir, clf, classes)
                    r['method'] = 'ORB'
                    results.append(r)
                    st.write(f'ORB 完成：{r["samples"]} 张，准确率 {r["accuracy"]*100:.1f}%')
            status.update(label='实验完成', state='complete')
        st.session_state['experiment_results'] = results

    results = st.session_state.get('experiment_results', [])
    if not results:
        return

    # 汇总表
    df = pd.DataFrame([{'方法': r['method'], '样本数': r['samples'],
                        '准确率': f'{r["accuracy"]*100:.1f}%',
                        '平均耗时(ms)': round(r['avg_ms'], 1)} for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 每类准确率
    all_classes = sorted({c for r in results for c in r['per_class']})
    per_class_rows = []
    for c in all_classes:
        row = {'类别': c}
        for r in results:
            row[f'{r["method"]}准确率'] = f'{r["per_class"].get(c, 0)*100:.1f}%'
        per_class_rows.append(row)
    st.markdown('#### 每类准确率')
    st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

    # 柱状图
    chart_df = pd.DataFrame([{'方法': r['method'], '准确率': r['accuracy']*100} for r in results])
    st.bar_chart(chart_df.set_index('方法'))
    time_df = pd.DataFrame([{'方法': r['method'], '平均耗时(ms)': r['avg_ms']} for r in results])
    st.bar_chart(time_df.set_index('方法'))

    # 混淆矩阵
    with st.expander('混淆矩阵'):
        for r in results:
            st.markdown(f'**{r["method"]}**')
            cm = r['confusion']
            cm_classes = sorted(cm.keys())
            cm_rows = []
            for c in cm_classes:
                row = {'真实类别': c}
                for p in cm_classes:
                    row[p] = cm[c].get(p, 0)
                cm_rows.append(row)
            st.dataframe(pd.DataFrame(cm_rows), use_container_width=True, hide_index=True)

    st.download_button('导出实验结果 CSV', df.to_csv(index=False).encode('utf-8-sig'),
                       'classifier_experiment.csv', 'text/csv')
