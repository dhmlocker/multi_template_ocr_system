from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def summarize_result(result: Any) -> dict[str, Any]:
    items = list(getattr(result, 'ocr_items', []) or [])
    fields = list(getattr(result, 'fields', []) or [])
    confidence = [float(item.confidence) for item in items]
    recognized_fields = [f for f in fields if not f.missing and str(f.value).strip()]
    low_conf = [item for item in items if float(item.confidence) < 0.80]
    quality = getattr(result, 'quality', None) or {}
    return {
        'filename': result.filename,
        'template_name': result.template_name,
        'ocr_items': len(items),
        'avg_ocr_confidence': round(sum(confidence) / len(confidence), 4) if confidence else 0.0,
        'low_confidence_items': len(low_conf),
        'fields_total': len(fields),
        'fields_recognized': len(recognized_fields),
        'field_coverage': round(len(recognized_fields) / len(fields), 4) if fields else 0.0,
        'quality_score': round(float(quality.get('quality_score', 0.0)), 2) if isinstance(quality, dict) else 0.0,
        'total_ms': round(float(result.timings_ms.get('total', 0.0)), 2),
        'ocr_ms': round(float(result.timings_ms.get('ocr', 0.0)), 2),
        'warnings': len(result.warnings),
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return path
    keys = list(rows[0].keys())
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_result_bundle(result: Any, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'result_summary.json'
    detail_path = output_dir / 'result_detail.json'
    summary_path.write_text(json.dumps(summarize_result(result), ensure_ascii=False, indent=2), encoding='utf-8')
    detail_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    return {'summary': str(summary_path), 'detail': str(detail_path)}
