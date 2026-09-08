from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


@dataclass
class OCRConfig:
    det_model_dir: Path
    rec_model_dir: Path
    use_angle_cls: bool = False
    lang: str = "ch"
    det_limit_side_len: int = 1600


@dataclass
class ClassifierConfig:
    method: str = "cnn"
    cnn_model_path: Path = Path("models/classifier.pth")
    class_names_path: Path = Path("models/class_names.json")
    pretrained_backbone_path: Path = Path("models/mobilenet_v2_imagenet.pth")
    orb_reference_dir: Path = Path("models/orb_templates")
    min_confidence: float = 0.0


@dataclass
class DatabaseConfig:
    path: Path = Path("outputs/recognition.db")


@dataclass
class TemplatesConfig:
    config_dir: Path = Path("config/templates")


@dataclass
class RuntimeConfig:
    device: str = "cpu"


@dataclass
class AppConfig:
    project_root: Path
    ocr: OCRConfig
    classifier: ClassifierConfig
    database: DatabaseConfig
    templates: TemplatesConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path, project_root: str | Path | None = None) -> "AppConfig":
        path = Path(path)
        root = Path(project_root).resolve() if project_root else path.resolve().parent.parent
        if not path.exists():
            raise FileNotFoundError(f"settings file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("settings YAML root must be a mapping")

        ocr_raw = raw.get("ocr", {}) or {}
        cls_raw = raw.get("classifier", {}) or {}
        db_raw = raw.get("database", {}) or {}
        tpl_raw = raw.get("templates", {}) or {}
        run_raw = raw.get("runtime", {}) or {}

        ocr = OCRConfig(
            det_model_dir=resolve_project_path(ocr_raw.get("det_model_dir", "models/PP-OCRv6_medium_det"), root),
            rec_model_dir=resolve_project_path(ocr_raw.get("rec_model_dir", "models/PP-OCRv6_medium_rec"), root),
            use_angle_cls=bool(ocr_raw.get("use_angle_cls", False)),
            lang=str(ocr_raw.get("lang", "ch")),
            det_limit_side_len=int(ocr_raw.get("det_limit_side_len", 1600)),
        )
        classifier = ClassifierConfig(
            method=str(cls_raw.get("method", "cnn")),
            cnn_model_path=resolve_project_path(cls_raw.get("cnn_model_path", "models/classifier.pth"), root),
            class_names_path=resolve_project_path(cls_raw.get("class_names_path", "models/class_names.json"), root),
            pretrained_backbone_path=resolve_project_path(cls_raw.get("pretrained_backbone_path", "models/mobilenet_v2_imagenet.pth"), root),
            orb_reference_dir=resolve_project_path(cls_raw.get("orb_reference_dir", "models/orb_templates"), root),
            min_confidence=float(cls_raw.get("min_confidence", 0.0)),
        )
        database = DatabaseConfig(path=resolve_project_path(db_raw.get("path", "outputs/recognition.db"), root))
        templates = TemplatesConfig(config_dir=resolve_project_path(tpl_raw.get("config_dir", "config/templates"), root))
        runtime = RuntimeConfig(device=str(run_raw.get("device", "cpu")))
        return cls(root, ocr, classifier, database, templates, runtime)

    def to_dict(self) -> dict[str, Any]:
        def rel(path: Path) -> str:
            try:
                return str(path.resolve().relative_to(self.project_root.resolve())).replace("\\", "/")
            except ValueError:
                return str(path)

        return {
            "ocr": {
                "det_model_dir": rel(self.ocr.det_model_dir),
                "rec_model_dir": rel(self.ocr.rec_model_dir),
                "use_angle_cls": self.ocr.use_angle_cls,
                "lang": self.ocr.lang,
                "det_limit_side_len": self.ocr.det_limit_side_len,
            },
            "classifier": {
                "method": self.classifier.method,
                "cnn_model_path": rel(self.classifier.cnn_model_path),
                "class_names_path": rel(self.classifier.class_names_path),
                "pretrained_backbone_path": rel(self.classifier.pretrained_backbone_path),
                "orb_reference_dir": rel(self.classifier.orb_reference_dir),
                "min_confidence": self.classifier.min_confidence,
            },
            "database": {"path": rel(self.database.path)},
            "templates": {"config_dir": rel(self.templates.config_dir)},
            "runtime": {"device": self.runtime.device},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
