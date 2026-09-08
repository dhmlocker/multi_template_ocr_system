from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.types import PipelineResult


class RecognitionRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(recognition_records)").fetchall()]
            if "image_path" not in cols:
                conn.execute("ALTER TABLE recognition_records ADD COLUMN image_path TEXT")

    def save_result(self, result: PipelineResult, image_bgr: np.ndarray | None = None) -> int:
        classification = result.classification
        record_id_holder: list[int] = []
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO recognition_records
                (filename, template_name, classifier_method, classifier_confidence, result_json, total_elapsed_ms, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.filename,
                    result.template_name,
                    classification.method if classification else None,
                    classification.confidence if classification else None,
                    json.dumps(result.to_dict(), ensure_ascii=False),
                    float(result.timings_ms.get("total", 0.0)),
                    result.image_path,
                ),
            )
            record_id = int(cur.lastrowid)
            record_id_holder.append(record_id)
            conn.executemany(
                "INSERT INTO recognition_fields(record_id, field_name, field_value, missing) VALUES(?,?,?,?)",
                [(record_id, f.field_name, f.value, int(f.missing)) for f in result.fields],
            )

        if image_bgr is not None:
            rec_id = record_id_holder[0]
            img_dir = self.db_path.parent / "records"
            img_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(result.filename).stem
            img_path = img_dir / f"record_{rec_id}_{safe_name}.jpg"
            ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                img_path.write_bytes(buf.tobytes())
                with self._connect() as conn:
                    conn.execute("UPDATE recognition_records SET image_path=? WHERE id=?", (str(img_path), rec_id))
        return record_id_holder[0]

    def list_records(self, *, template: str | None = None, filename: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT id, filename, template_name, classifier_method, classifier_confidence, total_elapsed_ms, created_at FROM recognition_records WHERE 1=1"
        params: list[Any] = []
        if template:
            sql += " AND template_name = ?"
            params.append(template)
        if filename:
            sql += " AND filename LIKE ?"
            params.append(f"%{filename}%")
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def get_record(self, record_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM recognition_records WHERE id=?", (record_id,)).fetchone()
            if row is None:
                return None
            data = dict(row)
            data["fields"] = [dict(x) for x in conn.execute(
                "SELECT field_name, field_value, missing FROM recognition_fields WHERE record_id=? ORDER BY id",
                (record_id,),
            ).fetchall()]
            try:
                data["result"] = json.loads(data["result_json"])
            except Exception:
                data["result"] = None
            return data

    def delete_record(self, record_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM recognition_records WHERE id=?", (record_id,))

    def export_rows(self) -> list[dict[str, Any]]:
        return self.list_records(limit=100000)
