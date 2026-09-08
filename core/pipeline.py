from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .cnn_classifier import CNNClassifier
from .config import AppConfig
from .field_extractor import FieldExtractor
from .image_utils import decode_image
from .ocr_engine import PaddleOCREngine
from .orb_classifier import ORBClassifier
from .types import ClassificationResult, PipelineResult

try:
    from database.db import RecognitionRepository
except Exception:  # pragma: no cover - package import safety
    RecognitionRepository = None  # type: ignore


class RecognitionPipeline:
    def __init__(
        self,
        config: AppConfig,
        *,
        ocr_engine: Any | None = None,
        cnn_classifier: Any | None = None,
        orb_classifier: Any | None = None,
        repository: Any | None = None,
    ):
        self.config = config
        self.ocr_engine = ocr_engine or PaddleOCREngine(config.ocr)
        self.cnn_classifier = cnn_classifier
        self.orb_classifier = orb_classifier
        self.repository = repository

    def _cnn(self) -> Any:
        if self.cnn_classifier is None:
            self.cnn_classifier = CNNClassifier(
                self.config.classifier.cnn_model_path,
                self.config.classifier.class_names_path,
                self.config.runtime.device,
            )
        return self.cnn_classifier

    def _orb(self) -> Any:
        if self.orb_classifier is None:
            self.orb_classifier = ORBClassifier(self.config.classifier.orb_reference_dir)
        return self.orb_classifier

    def _repo(self) -> Any:
        if self.repository is None:
            if RecognitionRepository is None:
                raise RuntimeError('database repository is unavailable')
            repo = RecognitionRepository(self.config.database.path)
            repo.initialize()
            self.repository = repo
        return self.repository

    def _classify(self, image, method: str) -> tuple[ClassificationResult | None, list[str]]:
        warnings: list[str] = []
        method = method.lower()
        if method in {'ocr_only', 'none', '仅ocr', 'ocr'}:
            return None, warnings
        try:
            if method == 'cnn':
                result = self._cnn().predict(image)
            elif method == 'orb':
                result = self._orb().predict(image)
            else:
                raise ValueError(f'unsupported classifier method: {method}')
        except Exception as exc:
            warnings.append(f'模板分类不可用，已继续执行 OCR：{exc}')
            return None, warnings
        if result.warning:
            warnings.append(result.warning)
        if result.label is None:
            warnings.append('模板分类未得到有效类别，字段 ROI 抽取将跳过。')
        return result, warnings

    def _load_template_config(self, label: str) -> dict[str, Any] | None:
        root = self.config.templates.config_dir
        if not root.exists():
            return None
        direct = root / f'{label}.json'
        candidates = [direct] if direct.is_file() else []
        candidates.extend(p for p in sorted(root.glob('*.json')) if p not in candidates)
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if path == direct or str(payload.get('template_name', '')) == label:
                return payload
        return None

    def classify(self, image: Any, classifier_method: str | None = None) -> tuple[Any, ClassificationResult | None, list[str], float]:
        """仅执行模板分类（秒级），返回 (BGR图像, 分类结果, 警告, 耗时ms)。"""
        t = time.perf_counter()
        bgr = decode_image(image)
        method = classifier_method or self.config.classifier.method
        classification, warnings = self._classify(bgr, method)
        elapsed = (time.perf_counter() - t) * 1000
        return bgr, classification, warnings, elapsed

    def run(self, image: Any, filename: str = 'uploaded_image', classifier_method: str | None = None,
            save: bool = False, reuse_classification: tuple | None = None) -> PipelineResult:
        total_start = time.perf_counter()
        if reuse_classification is not None:
            bgr, classification, warnings, cls_ms = reuse_classification
            timings = {'classification': cls_ms}
        else:
            bgr = decode_image(image)
            method = classifier_method or self.config.classifier.method
            timings = {}
            t = time.perf_counter()
            classification, warnings = self._classify(bgr, method)
            timings['classification'] = (time.perf_counter() - t) * 1000
        template_name = classification.label if classification else None

        t = time.perf_counter()
        ocr_items = self.ocr_engine.recognize(bgr)
        timings['ocr'] = (time.perf_counter() - t) * 1000

        fields = []
        t = time.perf_counter()
        if template_name:
            template_cfg = self._load_template_config(template_name)
            if template_cfg is None:
                warnings.append(f'当前模板“{template_name}”未配置字段 ROI，已返回 OCR 全文。')
            else:
                fields = FieldExtractor.extract(template_cfg, ocr_items, (bgr.shape[1], bgr.shape[0]))
        timings['field_extraction'] = (time.perf_counter() - t) * 1000
        timings['total'] = (time.perf_counter() - total_start) * 1000

        result = PipelineResult(
            filename=filename,
            template_name=template_name,
            classification=classification,
            ocr_items=ocr_items,
            fields=fields,
            timings_ms=timings,
            warnings=warnings,
        )
        if save:
            try:
                result.record_id = int(self._repo().save_result(result))
            except Exception as exc:
                result.warnings.append(f'识别结果保存失败，但当前结果仍可查看：{exc}')
        return result
