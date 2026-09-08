from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import OCRConfig
from .types import OCRItem


class PaddleOCREngine:
    """PaddleOCR adapter.

    当 config 中 det_model_dir / rec_model_dir 为空字符串时，PaddleOCR 会
    自动从 Paddle 云下载 PP-OCRv6 模型（首次运行）。路径非空时使用本地模型。

    始终禁用 oneDNN / mkldnn，规避 Paddle 3.x PIR 的 ConvertPirAttribute2RuntimeAttribute bug。
    """

    def __init__(self, config: OCRConfig):
        self.config = config
        self._engine: Any | None = None
        self.last_metadata: dict[str, Any] = {}
        self.last_raw: Any | None = None
        self.last_result: Any | None = None

    def _is_local_model_dir(self, path: str) -> bool:
        """路径非空且目录存在，视为使用本地模型。"""
        return path.strip() not in {'', '.', './'} and Path(path).is_dir()

    def validate_models(self) -> None:
        """Validate explicitly configured local PP-OCRv6 directories without downloading."""
        det_dir = str(self.config.det_model_dir).strip()
        rec_dir = str(self.config.rec_model_dir).strip()
        if not self._is_local_model_dir(det_dir) or not self._is_local_model_dir(rec_dir):
            raise FileNotFoundError('Local PP-OCRv6 model directory missing')

    def _build_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        # 始终禁用 oneDNN (Paddle 3.x PIR bug)
        os.environ['FLAGS_use_mkldnn'] = '0'
        os.environ['PADDLE_PDX_DISABLE_DEV_MODEL_WL'] = '1'

        from paddleocr import PaddleOCR

        det_dir = str(self.config.det_model_dir).strip()
        rec_dir = str(self.config.rec_model_dir).strip()
        use_local = self._is_local_model_dir(det_dir) and self._is_local_model_dir(rec_dir)

        # 本地模型目录都存在时才校验；否则让 PaddleOCR 自动下载
        if use_local:
            missing = [p for p in (det_dir, rec_dir) if not Path(p).is_dir()]
            if missing:
                raise FileNotFoundError(f'本地模型目录缺失: {missing}')

        # 构建参数：空路径时省略 *_model_dir 键，让 PaddleOCR 用默认值（自动下载）
        base = {
            'use_doc_orientation_classify': bool(self.config.use_doc_orientation_classify),
            'use_doc_unwarping': bool(self.config.use_doc_unwarping),
            'use_textline_orientation': bool(self.config.use_textline_orientation or self.config.use_angle_cls),
            'lang': self.config.lang,
            'enable_mkldnn': False,
            'det_limit_side_len': int(self.config.det_limit_side_len),
            'text_detection_model_name': self.config.det_model_name,
            'text_recognition_model_name': self.config.rec_model_name,
        }
        if use_local:
            new_kwargs = {**base, 'text_detection_model_dir': det_dir, 'text_recognition_model_dir': rec_dir}
            classic_kwargs = {**base, 'det_model_dir': det_dir, 'rec_model_dir': rec_dir,
                              'use_angle_cls': bool(self.config.use_angle_cls), 'show_log': False}
        else:
            # 自动下载：不传任何 model_dir，PaddleOCR 会下载默认的 PP-OCRv6
            new_kwargs = {**base}
            classic_kwargs = {**base, 'use_angle_cls': bool(self.config.use_angle_cls), 'show_log': False}

        try:
            parameters = inspect.signature(PaddleOCR.__init__).parameters
        except (TypeError, ValueError):
            parameters = {}
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())

        attempts: list[dict[str, Any]]
        if parameters and not has_var_kwargs:
            merged = {**new_kwargs, **classic_kwargs}
            filtered = {k: v for k, v in merged.items() if k in parameters}
            attempts = [filtered]
        else:
            attempts = [new_kwargs, classic_kwargs]

        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                self._engine = PaddleOCR(**kwargs)
                mode = '本地模型' if use_local else '自动下载模型'
                print(f'[OCR] PaddleOCR 初始化成功（{mode}）')
                return self._engine
            except TypeError as exc:
                last_exc = exc
                continue
        raise RuntimeError(f'PaddleOCR 初始化失败。最后错误: {last_exc}')

    def recognize(self, image: np.ndarray) -> list[OCRItem]:
        engine = self._build_engine()
        try:
            if hasattr(engine, 'predict'):
                raw = engine.predict(image)
            else:
                raw = engine.ocr(image, cls=bool(self.config.use_angle_cls))
        except Exception as exc:
            raise RuntimeError(f'PaddleOCR inference failed: {exc}') from exc
        self.last_raw = raw
        self.last_result = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        self.last_metadata = self.extract_metadata(raw)
        return self.normalize_result(raw)

    def save_official_outputs(self, output_dir: str | Path, stem: str = 'ocr_result') -> dict[str, str]:
        """Use PaddleOCR's own result exporters when available."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        result = self.last_result
        if result is not None and hasattr(result, 'save_to_img'):
            result.save_to_img(save_path=str(output))
            paths['official_img_dir'] = str(output)
        if result is not None and hasattr(result, 'save_to_json'):
            result.save_to_json(save_path=str(output))
            paths['official_json_dir'] = str(output)
        return paths

    def predict_official_source(self, source: str | Path) -> list[OCRItem]:
        """Run the official pipeline on a file path so exporters retain input geometry."""
        engine = self._build_engine()
        raw = engine.predict(str(source)) if hasattr(engine, 'predict') else engine.ocr(str(source), cls=True)
        if not isinstance(raw, (list, tuple)) and hasattr(raw, '__iter__'):
            raw = list(raw)
        self.last_raw = raw
        self.last_result = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        self.last_metadata = self.extract_metadata(raw)
        return self.normalize_result(raw)

    @staticmethod
    def extract_metadata(raw: Any) -> dict[str, Any]:
        """Extract orientation/unwarping and model settings for UI and paper logs."""
        def _jsonable(value: Any) -> Any:
            if hasattr(value, 'tolist'):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            return str(value)

        objects = raw if isinstance(raw, (list, tuple)) else [raw]
        metadata: dict[str, Any] = {}
        for obj in objects:
            payload = obj if isinstance(obj, dict) else getattr(obj, 'json', None)
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                try:
                    import json as _json
                    payload = _json.loads(payload)
                except Exception:
                    payload = None
            if not isinstance(payload, dict):
                continue
            res = payload.get('res', payload)
            if not isinstance(res, dict):
                continue
            for key in ('model_settings', 'textline_orientation_angles', 'input_path'):
                if key not in res:
                    continue
                value = res[key]
                if key == 'textline_orientation_angles':
                    values = _jsonable(value)
                    metadata[key] = {
                        'count': len(values) if isinstance(values, list) else 0,
                        'values': values[:200] if isinstance(values, list) else values,
                    }
                else:
                    metadata[key] = _jsonable(value)
            break
        return metadata

    @staticmethod
    def _to_box(poly: Any) -> list[list[float]]:
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f'invalid OCR polygon: {poly!r}')
        return [[float(x), float(y)] for x, y in arr[:, :2]]

    @classmethod
    def normalize_result(cls, raw: Any) -> list[OCRItem]:
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple, dict)) and hasattr(raw, '__iter__'):
            raw = list(raw)

        objects = raw if isinstance(raw, (list, tuple)) else [raw]
        new_items: list[OCRItem] = []
        found_new = False
        for obj in objects:
            payload = None
            if isinstance(obj, dict):
                payload = obj
            else:
                try:
                    payload = getattr(obj, 'json', None)
                    if callable(payload):
                        payload = payload()
                except Exception:
                    payload = None
            if isinstance(payload, str):
                try:
                    import json as _json
                    payload = _json.loads(payload)
                except Exception:
                    payload = None
            if not isinstance(payload, dict):
                continue
            res = payload.get('res', payload)
            if not isinstance(res, dict):
                continue
            def _first_value(*keys):
                for key in keys:
                    value = res.get(key)
                    if value is not None:
                        return value
                return None
            texts = _first_value('rec_texts', 'texts')
            scores = _first_value('rec_scores', 'scores')
            polys = _first_value('rec_polys', 'dt_polys', 'polys')
            if texts is None or polys is None:
                continue
            found_new = True
            scores = scores if scores is not None else [0.0] * len(texts)
            for text, score, poly in zip(texts, scores, polys):
                if str(text).strip():
                    new_items.append(OCRItem(cls._to_box(poly), str(text), float(score)))
        if found_new:
            return new_items

        legacy = raw
        if isinstance(legacy, tuple):
            legacy = list(legacy)
        if isinstance(legacy, list) and len(legacy) == 1 and isinstance(legacy[0], list):
            inner = legacy[0]
            if not inner or (isinstance(inner[0], (list, tuple)) and len(inner[0]) >= 2):
                legacy = inner
        items: list[OCRItem] = []
        if not isinstance(legacy, list):
            return items
        for line in legacy:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            box, rec = line[0], line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2 and isinstance(rec[0], str):
                text, score = rec[0], rec[1]
                if text.strip():
                    items.append(OCRItem(cls._to_box(box), text, float(score)))
        return items
