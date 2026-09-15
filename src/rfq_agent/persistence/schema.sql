PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS legacy_checkpoints (
 id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, complete INTEGER NOT NULL,
 payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS answers (
 case_id TEXT REFERENCES cases(case_id), record_id TEXT, payload TEXT NOT NULL,
 PRIMARY KEY(case_id, record_id));
CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY, case_id TEXT REFERENCES cases(case_id), payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS drafts (
 case_id TEXT REFERENCES cases(case_id), version INTEGER, payload TEXT NOT NULL,
 PRIMARY KEY(case_id, version));
CREATE TABLE IF NOT EXISTS section_versions (
 case_id TEXT REFERENCES cases(case_id), section_id TEXT, version INTEGER, payload TEXT NOT NULL,
 PRIMARY KEY(case_id, section_id, version));
CREATE TABLE IF NOT EXISTS change_requests (
 case_id TEXT REFERENCES cases(case_id), record_id TEXT, payload TEXT NOT NULL,
 PRIMARY KEY(case_id, record_id));
CREATE TABLE IF NOT EXISTS review_issues (
 case_id TEXT REFERENCES cases(case_id), record_id TEXT, payload TEXT NOT NULL,
 PRIMARY KEY(case_id, record_id));
CREATE TABLE IF NOT EXISTS model_runs (
 case_id TEXT REFERENCES cases(case_id), record_id TEXT, payload TEXT NOT NULL,
 PRIMARY KEY(case_id, record_id));
CREATE TABLE IF NOT EXISTS documents (
 id INTEGER PRIMARY KEY, case_id TEXT REFERENCES cases(case_id), payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_cases (
 case_id TEXT PRIMARY KEY, initial_request TEXT NOT NULL, status TEXT NOT NULL,
 document_path TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS graph_messages (
 id INTEGER PRIMARY KEY, case_id TEXT NOT NULL REFERENCES graph_cases(case_id),
 role TEXT NOT NULL, content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_documents (
 id INTEGER PRIMARY KEY, case_id TEXT NOT NULL REFERENCES graph_cases(case_id),
 draft_version INTEGER NOT NULL, path TEXT NOT NULL,
 UNIQUE(case_id, draft_version, path));
CREATE TABLE IF NOT EXISTS graph_model_runs (
 run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES graph_cases(case_id),
 capability TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
 started_at TEXT NOT NULL, finished_at TEXT, error_type TEXT);
