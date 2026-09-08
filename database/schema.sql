PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS recognition_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    template_name TEXT,
    classifier_method TEXT,
    classifier_confidence REAL,
    result_json TEXT NOT NULL,
    total_elapsed_ms REAL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS recognition_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    field_value TEXT NOT NULL DEFAULT '',
    missing INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(record_id) REFERENCES recognition_records(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_records_created_at ON recognition_records(created_at);
CREATE INDEX IF NOT EXISTS idx_records_template ON recognition_records(template_name);
CREATE INDEX IF NOT EXISTS idx_fields_record ON recognition_fields(record_id);
