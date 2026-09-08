from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OCRItem:
    box: list[list[float]]
    text: str
    confidence: float = 0.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bounds
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class ClassificationResult:
    label: str | None
    confidence: float
    method: str
    elapsed_ms: float = 0.0
    warning: str | None = None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class FieldResult:
    field_name: str
    value: str
    missing: bool
    confidence: float = 0.0
    validation: str = "未校验"
    source_indices: list[int] = field(default_factory=list)
    roi: list[float] | None = None


@dataclass
class PipelineResult:
    filename: str
    template_name: str | None
    classification: ClassificationResult | None
    ocr_items: list[OCRItem]
    fields: list[FieldResult]
    timings_ms: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    record_id: int | None = None
    image_path: str | None = None
    image_shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "template_name": self.template_name,
            "classification": asdict(self.classification) if self.classification else None,
            "ocr_items": [asdict(x) for x in self.ocr_items],
            "fields": [asdict(x) for x in self.fields],
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
            "quality": self.quality,
            "preprocessing": self.preprocessing,
            "record_id": self.record_id,
        }
