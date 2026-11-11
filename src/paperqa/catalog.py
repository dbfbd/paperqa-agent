"""SQLite 登记资料、入库任务与已发布快照。构建期间无长事务。"""

import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from paperqa.utils import dump_json


class PublishConflict(RuntimeError):
    pass


class IdempotencyConflict(ValueError):
    pass


class Catalog:
    def __init__(self, settings):
        name = settings.index_name
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("index_name 必须是单个目录名")
        self.directory = Path(settings.index_directory).resolve() / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "catalog.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL").fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, manifest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active (
                    slot INTEGER PRIMARY KEY CHECK (slot = 1),
                    version TEXT NOT NULL REFERENCES versions(id)
                );
                CREATE TABLE IF NOT EXISTS sources (
                    doc_id TEXT PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingestions (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    version TEXT, error TEXT, error_code INTEGER
                );
                CREATE TABLE IF NOT EXISTS ingestion_requests (
                    request_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                    job TEXT NOT NULL REFERENCES ingestions(id)
                );
            """)

    @contextmanager
    def connect(self):
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            yield db

    def active(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT id, manifest FROM active JOIN versions ON version = id"
            ).fetchone()
            return (row["id"], json.loads(row["manifest"])) if row else (None, None)

    def publish(self, generation, manifest, previous, job=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT version FROM active").fetchone()
            if (active["version"] if active else None) != previous:
                raise PublishConflict("构建期间索引版本已更新，请重试入库")
            if generation != previous:
                db.execute("INSERT INTO versions VALUES (?, ?)", (generation, dump_json(manifest)))
                db.executemany(
                    "INSERT OR IGNORE INTO sources VALUES (?, ?, ?)",
                    ((x["doc_id"], x["original"], x["name"]) for x in manifest["files"].values()),
                )
                db.execute(
                    "INSERT INTO active VALUES (1, ?) "
                    "ON CONFLICT(slot) DO UPDATE SET version=excluded.version",
                    (generation,),
                )
            if job:
                db.execute(
                    "UPDATE ingestions SET status='ready', version=? WHERE id=?", (generation, job)
                )

    def source(self, doc_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT path, name FROM sources WHERE doc_id=?", (doc_id,)
            ).fetchone()
            return (self.directory / row["path"], row["name"]) if row else None

    def version_manifest(self, version):
        with self.connect() as db:
            row = db.execute("SELECT manifest FROM versions WHERE id=?", (version,)).fetchone()
            return json.loads(row["manifest"]) if row else None

    def new_job(self, request_key=None, fingerprint=None, *, accept_new=True):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if request_key is not None:
                previous = db.execute(
                    "SELECT fingerprint, job FROM ingestion_requests WHERE request_key=?",
                    (request_key,),
                ).fetchone()
                if previous:
                    if previous["fingerprint"] != fingerprint:
                        raise IdempotencyConflict("同一 Idempotency-Key 已用于不同的上传内容")
                    return previous["job"], False
            if not accept_new:
                return None, False
            job = uuid4().hex
            db.execute(
                "INSERT INTO ingestions (id, status) VALUES (?, 'queued')", (job,)
            )
            if request_key is not None:
                db.execute(
                    "INSERT INTO ingestion_requests VALUES (?, ?, ?)",
                    (request_key, fingerprint, job),
                )
        return job, True

    def update_job(self, job, status, *, error=None, code=None):
        with self.connect() as db:
            db.execute(
                "UPDATE ingestions SET status=?, error=?, error_code=? WHERE id=?",
                (status, error, code, job),
            )

    def job(self, job):
        with self.connect() as db:
            row = db.execute("SELECT * FROM ingestions WHERE id=?", (job,)).fetchone()
            return dict(row) if row else None

    def interrupt_unfinished(self):
        with self.connect() as db:
            db.execute(
                "UPDATE ingestions SET status='failed', error='服务重启，请重新上传', "
                "error_code=503 WHERE status IN ('queued', 'building')"
            )
