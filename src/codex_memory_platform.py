"""Codex-native Obsidian memory runtime, CLI, and MCP stdio server."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for Codex memory platform") from exc

VAULT = Path(os.environ.get("CODEX_MEMORY_VAULT", r"D:\hermes\hermes_obsidian_vault"))
STORE = Path(os.environ.get("CODEX_MEMORY_STORE", r"D:\hermes\codex_memory_store"))
DB = STORE / "codex_memory.db"
LOCK = STORE / ".codex_memory.lock"
SEMANTIC_LOCK_NAME = ".codex_semantic.lock"
BACKUPS = STORE / "backups"
ONNX_MODEL = Path(os.environ.get("CODEX_MEMORY_ONNX_MODEL", r"D:\hermes\models\bge_m3_onnx_export\model.onnx"))
TOKENIZER_ROOT = Path(os.environ.get("CODEX_MEMORY_TOKENIZER", r"D:\hermes\models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"))
EMBEDDING_MODEL = "BAAI/bge-m3-onnx-cpu"
SECRET_RE = re.compile(r"(?i)(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|api[_-]?key\s*[:=]\s*[^\s]+|secret[_-]?key\s*[:=]\s*[^\s]+|password\s*[:=]\s*[^\s]+|Bearer\s+[A-Za-z0-9._-]+)")
_SEMANTIC_PROVIDER = None
_SEMANTIC_GATE = threading.Lock()
_SEMANTIC_WORKER = None
_SEMANTIC_WORKER_RESTARTS = 0
SEMANTIC_WORKER_COMMAND = [sys.executable, str(Path(__file__).resolve()), "semantic-worker"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: str) -> str:
    cleaned = str(text or "").encode("utf-8", "replace").decode("utf-8")
    return SECRET_RE.sub("[REDACTED]", cleaned)


def bounded_semantic_text(text: str, max_chars: int = 12000) -> str:
    """Keep semantic input proportional to the model token budget and CPU profile."""
    safe = redact(text)
    if len(safe) <= max_chars:
        return safe
    half = max_chars // 2
    return safe[:half] + "\n[semantic middle truncated]\n" + safe[-half:]


def parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    try:
        value = yaml.safe_load(text[3:end]) or {}
        return value if isinstance(value, dict) else {}
    except yaml.YAMLError:
        return {}


def extract_links(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[\[([^\]|#]+)", text or "")))


def file_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SemanticProvider:
    """Local CPU-only BGE-M3 provider with bounded operation lifetime."""
    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._np = None

    @property
    def available(self) -> bool:
        return ONNX_MODEL.exists() and TOKENIZER_ROOT.exists()

    def load(self) -> None:
        if self._session is not None:
            return
        if not self.available:
            raise RuntimeError(f"semantic model files missing: {ONNX_MODEL}")
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("CODEX_MEMORY_CPU_THREADS", "4"))
        options.inter_op_num_threads = 1
        # The E3-1230v3 profile cannot afford ONNX's large arena and full graph
        # optimization peak during model initialization. The worker boundary
        # contains native failures; these settings reduce the allocation peak.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self._session = ort.InferenceSession(str(ONNX_MODEL), sess_options=options, providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT), local_files_only=True)
        self._np = np

    def encode(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        self.load()
        vectors = []
        input_names = {item.name for item in self._session.get_inputs()}
        max_length = int(os.environ.get("CODEX_MEMORY_MAX_LENGTH", "128"))
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self._tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=max_length)
            inputs = {key: value.astype(self._np.int64) for key, value in encoded.items() if key in input_names}
            outputs = self._session.run(None, inputs)
            hidden = self._np.asarray(outputs[0], dtype=self._np.float32)
            mask = encoded["attention_mask"].astype(self._np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1) / self._np.maximum(mask.sum(axis=1), 1e-6)
            pooled /= self._np.maximum(self._np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            vectors.extend(pooled.tolist())
            del encoded, inputs, outputs, hidden, mask, pooled
        return vectors

    def close(self) -> None:
        """Release native ONNX and tokenizer allocations after one operation."""
        self._session = None
        self._tokenizer = None
        self._np = None


class SemanticWorkerClient:
    """Keep native ONNX failures outside the MCP process."""
    def __init__(self, command: list[str] | None = None, timeout: float = 180.0) -> None:
        self.command = command or SEMANTIC_WORKER_COMMAND
        self.timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self._process.poll() if self._process is not None else None

    def _start(self) -> subprocess.Popen[str]:
        global _SEMANTIC_WORKER_RESTARTS
        if self.active:
            return self._process  # type: ignore[return-value]
        if self._process is not None:
            _SEMANTIC_WORKER_RESTARTS += 1
            self._terminate()
        env = os.environ.copy()
        env["CODEX_MEMORY_SEMANTIC_WORKER"] = "child"
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(self.command, **kwargs)
        return self._process

    def _readline(self, process: subprocess.Popen[str]) -> str:
        if process.stdout is None:
            raise RuntimeError("semantic worker stdout is unavailable")
        responses: queue.Queue[str] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                responses.put(process.stdout.readline())
            except Exception as exc:  # pragma: no cover - platform pipe failure
                responses.put(json.dumps({"ok": False, "error": str(exc)}))

        threading.Thread(target=read, daemon=True).start()
        try:
            line = responses.get(timeout=self.timeout)
        except queue.Empty as exc:
            self._terminate()
            raise TimeoutError(f"semantic worker timeout after {self.timeout:.0f}s") from exc
        if not line:
            code = process.poll()
            self._terminate()
            raise RuntimeError(f"semantic worker exited before response (code={code})")
        return line

    def encode(self, texts: list[str], batch_size: int = 1) -> list[list[float]]:
        with self._lock:
            process = self._start()
            try:
                if process.stdin is None:
                    raise RuntimeError("semantic worker stdin is unavailable")
                process.stdin.write(json.dumps({"texts": texts, "batch_size": batch_size}, ensure_ascii=False) + "\n")
                process.stdin.flush()
                response = json.loads(self._readline(process))
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error") or "semantic worker failed"))
                vectors = response.get("vectors")
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise RuntimeError("semantic worker returned an invalid vector count")
                return vectors
            except Exception:
                self._terminate()
                raise

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def close(self) -> None:
        with self._lock:
            self._terminate()


def semantic_worker() -> int:
    """Serve isolated ONNX requests over JSONL until the parent closes stdin."""
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    provider = SemanticProvider()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            vectors = provider.encode(request.get("texts", []), int(request.get("batch_size", 1)))
            result = {"ok": True, "vectors": vectors}
        except Exception as exc:
            result = {"ok": False, "error": redact(str(exc))[:500]}
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    provider.close()
    return 0


def semantic_provider() -> SemanticProvider:
    global _SEMANTIC_PROVIDER
    if _SEMANTIC_PROVIDER is None:
        _SEMANTIC_PROVIDER = SemanticProvider()
    return _SEMANTIC_PROVIDER


def semantic_worker_client() -> SemanticWorkerClient:
    global _SEMANTIC_WORKER
    if _SEMANTIC_WORKER is None:
        _SEMANTIC_WORKER = SemanticWorkerClient()
    return _SEMANTIC_WORKER


def semantic_encode(texts: list[str], batch_size: int = 1) -> list[list[float]]:
    """Encode through an isolated worker; the MCP process never loads ONNX."""
    if os.environ.get("CODEX_MEMORY_SEMANTIC_WORKER", "process") == "inproc":
        return semantic_provider().encode(texts, batch_size=batch_size)
    return semantic_worker_client().encode(texts, batch_size=batch_size)


@atexit.register
def close_semantic_worker() -> None:
    if _SEMANTIC_WORKER is not None:
        _SEMANTIC_WORKER.close()


def record_semantic_event(db: sqlite3.Connection, *, error: str = "") -> None:
    """Persist semantic health without hiding a degraded retrieval path."""
    if error:
        previous = db.execute("SELECT value FROM meta WHERE key='semantic_failures'").fetchone()
        failures = int(previous[0]) if previous and str(previous[0]).isdigit() else 0
        db.execute("INSERT INTO meta(key,value) VALUES('semantic_failures',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(failures + 1),))
        db.execute("INSERT INTO meta(key,value) VALUES('last_semantic_error',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (redact(error)[:500],))
    else:
        db.execute("INSERT INTO meta(key,value) VALUES('last_semantic_success',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now(),))
        db.execute("INSERT INTO meta(key,value) VALUES('last_semantic_error','') ON CONFLICT(key) DO UPDATE SET value=excluded.value")


@contextmanager
def file_lock(lock_path: Path, timeout: float = 30.0):
    STORE.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    deadline = time.monotonic() + timeout
    locked = False
    try:
        while not locked:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Codex memory write lock timeout")
                time.sleep(0.2)
        yield
    finally:
        if locked:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


@contextmanager
def write_lock(timeout: float = 30.0):
    with file_lock(LOCK, timeout):
        yield


@contextmanager
def semantic_operation(timeout: float = 30.0):
    """Serialize ONNX model work in-process and across MCP/CLI processes."""
    with _SEMANTIC_GATE:
        with file_lock(STORE / SEMANTIC_LOCK_NAME, timeout):
            try:
                yield
            finally:
                if os.environ.get("CODEX_MEMORY_SEMANTIC_WORKER", "process") == "inproc" and _SEMANTIC_PROVIDER is not None:
                    _SEMANTIC_PROVIDER.close()


def connect() -> sqlite3.Connection:
    STORE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS notes (
          path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, title TEXT, type TEXT,
          category TEXT, status TEXT, updated TEXT, content TEXT NOT NULL,
          created TEXT DEFAULT '', source TEXT DEFAULT '', scope TEXT DEFAULT 'global',
          confidence REAL DEFAULT 0.5, importance REAL DEFAULT 0.5,
          valid_from TEXT DEFAULT '', valid_until TEXT DEFAULT '', links_json TEXT DEFAULT '[]'
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
          path UNINDEXED, title, content, tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS note_embeddings (
          path TEXT PRIMARY KEY, model TEXT NOT NULL, sha256 TEXT NOT NULL,
          dimensions INTEGER NOT NULL, vector BLOB NOT NULL, updated TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS note_links (
          source_path TEXT NOT NULL, target TEXT NOT NULL,
          PRIMARY KEY(source_path, target)
        );
        CREATE TABLE IF NOT EXISTS observations (
          id TEXT PRIMARY KEY, created TEXT NOT NULL, kind TEXT NOT NULL,
          scope TEXT NOT NULL, confidence REAL NOT NULL, summary TEXT NOT NULL,
          tags_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'captured',
          promoted_path TEXT DEFAULT '', supersedes_id TEXT DEFAULT '',
          conflict_group TEXT DEFAULT '', valid_from TEXT DEFAULT '',
          valid_until TEXT DEFAULT '', importance REAL DEFAULT 0.5,
          last_confirmed TEXT DEFAULT '', source_event_id TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS conflicts (
          id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
          paths_json TEXT NOT NULL, detected TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
          kind TEXT NOT NULL DEFAULT 'duplicate-title', conflict_group TEXT DEFAULT '',
          winner_id TEXT DEFAULT '', resolved_at TEXT DEFAULT '', resolution_note TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    columns = {row[1] for row in db.execute("PRAGMA table_info(notes)")}
    for name, definition in {
        "created": "TEXT DEFAULT ''", "source": "TEXT DEFAULT ''", "scope": "TEXT DEFAULT 'global'",
        "confidence": "REAL DEFAULT 0.5", "importance": "REAL DEFAULT 0.5",
        "valid_from": "TEXT DEFAULT ''", "valid_until": "TEXT DEFAULT ''", "links_json": "TEXT DEFAULT '[]'",
    }.items():
        if name not in columns:
            db.execute(f"ALTER TABLE notes ADD COLUMN {name} {definition}")
    observation_columns = {row[1] for row in db.execute("PRAGMA table_info(observations)")}
    for name, definition in {
        "promoted_path": "TEXT DEFAULT ''", "supersedes_id": "TEXT DEFAULT ''",
        "conflict_group": "TEXT DEFAULT ''", "valid_from": "TEXT DEFAULT ''",
        "valid_until": "TEXT DEFAULT ''", "importance": "REAL DEFAULT 0.5",
        "last_confirmed": "TEXT DEFAULT ''", "source_event_id": "TEXT DEFAULT ''",
    }.items():
        if name not in observation_columns:
            db.execute(f"ALTER TABLE observations ADD COLUMN {name} {definition}")
    conflict_columns = {row[1] for row in db.execute("PRAGMA table_info(conflicts)")}
    for name, definition in {
        "kind": "TEXT NOT NULL DEFAULT 'duplicate-title'", "conflict_group": "TEXT DEFAULT ''",
        "winner_id": "TEXT DEFAULT ''", "resolved_at": "TEXT DEFAULT ''",
        "resolution_note": "TEXT DEFAULT ''",
    }.items():
        if name not in conflict_columns:
            db.execute(f"ALTER TABLE conflicts ADD COLUMN {name} {definition}")
    db.commit()
    return db


def vault_files() -> list[Path]:
    return [p for p in VAULT.rglob("*.md") if ".obsidian" not in p.parts and "Archive" not in p.parts]


def note_fields(path: Path, content: str) -> dict[str, Any]:
    fm = parse_frontmatter(content)
    def number(name: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(fm.get(name, default))))
        except (TypeError, ValueError):
            return default
    return {
        "title": str(fm.get("title") or path.stem),
        "type": str(fm.get("type") or "knowledge"),
        "category": str(fm.get("category") or "Knowledge"),
        "status": str(fm.get("status") or "active"),
        "updated": str(fm.get("updated") or ""),
        "created": str(fm.get("created") or ""),
        "source": str(fm.get("source") or "vault"),
        "scope": str(fm.get("project_scope") or fm.get("scope") or "global"),
        "confidence": number("confidence", 0.5),
        "importance": number("importance", 0.5),
        "valid_from": str(fm.get("valid_from") or ""),
        "valid_until": str(fm.get("valid_until") or ""),
        "links": extract_links(content),
    }


def sync() -> dict[str, Any]:
    with write_lock():
        db = connect()
        current = {str(p.relative_to(VAULT)): p for p in vault_files()}
        changed = removed = 0
        pending_embeddings: list[tuple[str, str, str]] = []
        with db:
            old = {row[0] for row in db.execute("SELECT path FROM notes")}
            for rel, path in current.items():
                content = path.read_text(encoding="utf-8-sig", errors="replace")
                digest = file_digest(content)
                existing = db.execute("SELECT sha256 FROM notes WHERE path=?", (rel,)).fetchone()
                if existing and existing[0] == digest:
                    fields = note_fields(path, content)
                    db.execute("UPDATE notes SET title=?,type=?,category=?,status=?,updated=?,created=?,source=?,scope=?,confidence=?,importance=?,valid_from=?,valid_until=?,links_json=? WHERE path=?",
                               (fields["title"], fields["type"], fields["category"], fields["status"], fields["updated"], fields["created"], fields["source"], fields["scope"], fields["confidence"], fields["importance"], fields["valid_from"], fields["valid_until"], json.dumps(fields["links"], ensure_ascii=False), rel))
                    db.execute("DELETE FROM note_links WHERE source_path=?", (rel,))
                    db.executemany("INSERT OR IGNORE INTO note_links(source_path,target) VALUES(?,?)", [(rel, x) for x in fields["links"]])
                    embedding = db.execute("SELECT sha256 FROM note_embeddings WHERE path=?", (rel,)).fetchone()
                    if not embedding or embedding[0] != digest:
                         pending_embeddings.append((rel, bounded_semantic_text(fields["title"] + "\n" + content), digest))
                    continue
                fields = note_fields(path, content)
                db.execute(
                    """INSERT INTO notes(path,sha256,title,type,category,status,updated,content,created,source,scope,confidence,importance,valid_from,valid_until,links_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256,title=excluded.title,type=excluded.type,
                    category=excluded.category,status=excluded.status,updated=excluded.updated,content=excluded.content,
                    created=excluded.created,source=excluded.source,scope=excluded.scope,confidence=excluded.confidence,
                    importance=excluded.importance,valid_from=excluded.valid_from,valid_until=excluded.valid_until,links_json=excluded.links_json""",
                    (rel, digest, fields["title"], fields["type"], fields["category"], fields["status"], fields["updated"],
                     redact(content), fields["created"], fields["source"], fields["scope"], fields["confidence"],
                     fields["importance"], fields["valid_from"], fields["valid_until"], json.dumps(fields["links"], ensure_ascii=False)),
                )
                db.execute("DELETE FROM notes_fts WHERE path=?", (rel,))
                db.execute("INSERT INTO notes_fts(path,title,content) VALUES(?,?,?)", (rel, fields["title"], redact(content)))
                db.execute("DELETE FROM note_links WHERE source_path=?", (rel,))
                db.executemany("INSERT OR IGNORE INTO note_links(source_path,target) VALUES(?,?)", [(rel, x) for x in fields["links"]])
                pending_embeddings.append((rel, bounded_semantic_text(fields["title"] + "\n" + content), digest))
                changed += 1
            for rel in old - set(current):
                db.execute("DELETE FROM notes WHERE path=?", (rel,))
                db.execute("DELETE FROM notes_fts WHERE path=?", (rel,))
                db.execute("DELETE FROM note_links WHERE source_path=?", (rel,))
                db.execute("DELETE FROM note_embeddings WHERE path=?", (rel,))
                removed += 1
            db.execute("DELETE FROM conflicts WHERE kind='duplicate-title'")
            for title, paths, _count in db.execute("SELECT title, GROUP_CONCAT(path), COUNT(*) FROM notes GROUP BY title HAVING COUNT(*) > 1"):
                db.execute("INSERT INTO conflicts(title,paths_json,detected,kind) VALUES(?,?,?,?)", (title, json.dumps(paths.split(",")), now(), "duplicate-title"))
            db.execute("INSERT INTO meta(key,value) VALUES('last_sync',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (now(),))
        db.close()
        if pending_embeddings and os.environ.get("CODEX_MEMORY_SEMANTIC", "required") != "disabled":
            provider = semantic_provider()
            db = connect()
            batch_size = int(os.environ.get("CODEX_MEMORY_EMBED_BATCH", "1"))
            semantic_error = ""
            try:
                with semantic_operation():
                    for start in range(0, len(pending_embeddings), batch_size):
                        batch = pending_embeddings[start:start + batch_size]
                        vectors = semantic_encode([item[1] for item in batch], batch_size=batch_size)
                        with db:
                            for (path, _text, digest), vector in zip(batch, vectors):
                                db.execute("INSERT INTO note_embeddings(path,model,sha256,dimensions,vector,updated) VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET model=excluded.model,sha256=excluded.sha256,dimensions=excluded.dimensions,vector=excluded.vector,updated=excluded.updated",
                                           (path, EMBEDDING_MODEL, digest, len(vector), sqlite3.Binary(__import__('array').array('f', vector).tobytes()), now()))
                with db:
                    record_semantic_event(db)
            except Exception as exc:
                semantic_error = redact(str(exc))[:500]
                with db:
                    record_semantic_event(db, error=semantic_error)
            db.close()
        else:
            semantic_error = ""
            if os.environ.get("CODEX_MEMORY_SEMANTIC", "required") != "disabled":
                db = connect()
                with db:
                    record_semantic_event(db)
                db.close()
        db = connect()
        count = db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        conflicts = db.execute("SELECT COUNT(*) FROM conflicts WHERE status='open'").fetchone()[0]
        db.close()
        embeddings = sqlite3.connect(DB).execute("SELECT COUNT(*) FROM note_embeddings").fetchone()[0]
        return {"notes": count, "changed": changed, "removed": removed, "conflicts": conflicts, "embeddings": embeddings, "semantic_error": semantic_error or None, "db": str(DB)}


def status() -> dict[str, Any]:
    db = connect()
    worker_exit_code = _SEMANTIC_WORKER.exit_code if _SEMANTIC_WORKER and _SEMANTIC_WORKER.active else None
    result = {
        "db": str(DB),
        "notes": db.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
        "fts_docs": db.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0],
        "links": db.execute("SELECT COUNT(*) FROM note_links").fetchone()[0],
        "observations": db.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
        "open_conflicts": db.execute("SELECT COUNT(*) FROM conflicts WHERE status='open'").fetchone()[0],
        "effective_observations": db.execute("SELECT COUNT(*) FROM observations WHERE status IN ('active','promoted') AND (valid_until='' OR valid_until>?)", (now(),)).fetchone()[0],
        "expired_observations": db.execute("SELECT COUNT(*) FROM observations WHERE status='expired'").fetchone()[0],
        "superseded_observations": db.execute("SELECT COUNT(*) FROM observations WHERE status='superseded'").fetchone()[0],
        "conflicted_observations": db.execute("SELECT COUNT(*) FROM observations WHERE status='conflicted'").fetchone()[0],
        "semantic_index_docs": db.execute("SELECT COUNT(*) FROM note_embeddings").fetchone()[0],
        "retrieval_mode": "fts5+onnx-bge-m3+confidence+importance+recency" if db.execute("SELECT COUNT(*) FROM note_embeddings").fetchone()[0] else "pending-semantic-index",
        "semantic_provider": "onnx-bge-m3-cpu" if semantic_provider().available else "unavailable",
        "semantic_failures": int((db.execute("SELECT value FROM meta WHERE key='semantic_failures'").fetchone() or ["0"])[0]),
        "last_semantic_error": (db.execute("SELECT value FROM meta WHERE key='last_semantic_error'").fetchone() or [""])[0],
        "semantic_health": "degraded" if (db.execute("SELECT value FROM meta WHERE key='last_semantic_error'").fetchone() or [""])[0] else ("ready" if db.execute("SELECT COUNT(*) FROM note_embeddings").fetchone()[0] else "pending"),
        "semantic_resource_policy": "isolated-worker-serialized-release",
        "semantic_session_loaded": bool(_SEMANTIC_PROVIDER and _SEMANTIC_PROVIDER._session is not None),
        "semantic_worker_active": bool(_SEMANTIC_WORKER and _SEMANTIC_WORKER.active),
        "semantic_worker_restarts": _SEMANTIC_WORKER_RESTARTS,
        "semantic_worker_exit_code": worker_exit_code,
        "last_sync": (db.execute("SELECT value FROM meta WHERE key='last_sync'").fetchone() or [""])[0],
        "integrity_check": db.execute("PRAGMA integrity_check").fetchone()[0],
    }
    db.close()
    return result


def recency_score(updated: str) -> float:
    if not updated:
        return 0.25
    try:
        value = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - value.astimezone(timezone.utc)).total_seconds() / 86400)
        return max(0.0, 1.0 - min(age_days, 365.0) / 365.0)
    except ValueError:
        return 0.25


def _fts_fallback_query(query: str) -> str:
    """Build a tolerant OR query after the exact FTS query misses."""
    tokens = re.findall(r"[\w]+", query or "", flags=re.UNICODE)
    unique = list(dict.fromkeys(token for token in tokens if token))
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in unique)


def search(query: str, limit: int = 10, scope: str = "", kind: str = "") -> list[dict[str, Any]]:
    db = connect()
    params: list[Any] = [query, max(limit * 4, limit)]
    try:
        rows = db.execute(
            "SELECT n.path,n.title,n.type,n.scope,n.confidence,n.importance,n.updated,n.content,bm25(notes_fts) "
            "FROM notes_fts JOIN notes n ON n.path=notes_fts.path WHERE notes_fts MATCH ? AND n.status NOT IN ('draft','deprecated') LIMIT ?", params
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        tolerant_query = _fts_fallback_query(query)
        if tolerant_query and tolerant_query != query:
            try:
                rows = db.execute(
                    "SELECT n.path,n.title,n.type,n.scope,n.confidence,n.importance,n.updated,n.content,bm25(notes_fts) "
                    "FROM notes_fts JOIN notes n ON n.path=notes_fts.path WHERE notes_fts MATCH ? AND n.status NOT IN ('draft','deprecated') LIMIT ?",
                    [tolerant_query, max(limit * 4, limit)],
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
    if not rows:
        tokens = re.findall(r"[\w]+", query or "", flags=re.UNICODE) or [query]
        clauses = ["status NOT IN ('draft','deprecated')", "(" + " OR ".join("content LIKE ?" for _ in tokens) + ")"]
        args: list[Any] = [f"%{token}%" for token in tokens]
        if scope:
            clauses.append("scope IN ('global', ?)")
            args.append(scope)
        if kind:
            clauses.append("type=?")
            args.append(kind)
        rows = [(*row, 0.0) for row in db.execute(
            f"SELECT path,title,type,scope,confidence,importance,updated,content FROM notes WHERE {' AND '.join(clauses)} LIMIT ?",
            (*args, limit * 4),
        ).fetchall()]
    semantic_by_path: dict[str, tuple[float, tuple[Any, ...]]] = {}
    if db.execute("SELECT COUNT(*) FROM note_embeddings").fetchone()[0] and os.environ.get("CODEX_MEMORY_SEMANTIC", "required") != "disabled":
        import numpy as np
        try:
            with semantic_operation():
                query_vector = np.asarray(semantic_encode([query], batch_size=1)[0], dtype=np.float32)
                semantic_rows = db.execute("SELECT n.path,n.title,n.type,n.scope,n.confidence,n.importance,n.updated,n.content,e.vector FROM note_embeddings e JOIN notes n ON n.path=e.path WHERE n.status NOT IN ('draft','deprecated')").fetchall()
            with db:
                record_semantic_event(db)
        except Exception as exc:
            with db:
                record_semantic_event(db, error=str(exc))
            semantic_rows = []
        for item in semantic_rows:
            path = item[0]
            if scope and item[3] not in ("global", scope):
                continue
            if kind and item[2] != kind:
                continue
            vector = np.frombuffer(item[8], dtype=np.float32)
            semantic_by_path[path] = (float(np.dot(query_vector, vector)), item[:8] + (0.0,))
    result = []
    seen: set[str] = set()
    for path, title, typ, row_scope, confidence, importance, updated, content, rank in rows:
        if scope and row_scope not in ("global", scope):
            continue
        if kind and typ != kind:
            continue
        lexical = 1.0 / (1.0 + max(float(rank), 0.0))
        semantic = semantic_by_path.get(path, (0.0, ()))[0]
        score = round(0.35 * semantic + 0.35 * lexical + 0.15 * float(confidence) + 0.10 * float(importance) + 0.05 * recency_score(updated), 6)
        result.append({"path": path, "title": title, "type": typ, "scope": row_scope, "score": score,
                       "snippet": redact(content).replace("\n", " ")[:500]})
        seen.add(path)
    for path, (semantic, item) in sorted(semantic_by_path.items(), key=lambda x: x[1][0], reverse=True):
        if path in seen:
            continue
        _path, title, typ, row_scope, confidence, importance, updated, content, _rank = item
        result.append({"path": path, "title": title, "type": typ, "scope": row_scope,
                       "score": round(0.35 * semantic + 0.15 * float(confidence) + 0.10 * float(importance) + 0.05 * recency_score(updated), 6),
                       "snippet": redact(content).replace("\n", " ")[:500]})
    db.close()
    return sorted(result, key=lambda x: x["score"], reverse=True)[:limit]


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _is_effective(status: str, valid_from: str, valid_until: str, at: datetime | None = None) -> tuple[bool, str]:
    at = at or datetime.now(timezone.utc)
    if status not in ("active", "promoted"):
        return False, status
    start = _parse_timestamp(valid_from)
    if start and start > at:
        return False, "not-yet-valid"
    end = _parse_timestamp(valid_until)
    if end and end <= at:
        return False, "expired"
    return True, "effective"


def _update_promoted_note(path_value: str, status: str, valid_until: str = "") -> None:
    if not path_value:
        return
    path = Path(path_value)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.startswith("---"):
        return
    end = text.find("\n---", 3)
    if end < 0:
        return
    try:
        frontmatter = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return
    if not isinstance(frontmatter, dict):
        return
    frontmatter["status"] = status
    frontmatter["updated"] = now()
    if valid_until:
        frontmatter["valid_until"] = valid_until
    rewritten = "---\n" + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False) + "---\n" + text[end + 4:]
    path.write_text(redact(rewritten), encoding="utf-8")


def _observation_row(db: sqlite3.Connection, observation_id: str) -> sqlite3.Row | tuple[Any, ...] | None:
    return db.execute(
        """SELECT id,created,kind,scope,confidence,summary,tags_json,status,promoted_path,
                  supersedes_id,conflict_group,valid_from,valid_until,importance,last_confirmed,source_event_id
           FROM observations WHERE id=?""", (observation_id,)
    ).fetchone()


def _observation_dict(row: tuple[Any, ...], at: datetime | None = None) -> dict[str, Any]:
    effective, reason = _is_effective(str(row[7]), str(row[11] or ""), str(row[12] or ""), at)
    return {
        "id": row[0], "created": row[1], "kind": row[2], "scope": row[3],
        "confidence": row[4], "summary": redact(row[5]), "tags": json.loads(row[6] or "[]"),
        "status": row[7], "promoted_path": row[8], "supersedes_id": row[9] or "",
        "conflict_group": row[10] or "", "valid_from": row[11] or "", "valid_until": row[12] or "",
        "importance": row[13], "last_confirmed": row[14] or "", "source_event_id": row[15] or "",
        "effective": effective, "reason": reason,
    }


def capture(summary: str, kind: str = "observation", scope: str = "global", confidence: float = 0.5,
            tags: list[str] | None = None, event_id: str | None = None, importance: float = 0.5,
            valid_from: str = "", valid_until: str = "", supersedes_id: str = "",
            conflict_group: str = "", source_event_id: str = "") -> dict[str, Any]:
    summary = redact(summary).strip()
    if not summary:
        raise ValueError("summary is required")
    scope = str(scope or "global").strip() or "global"
    kind = str(kind or "observation").strip() or "observation"
    ident = event_id or str(uuid.uuid4())
    with write_lock():
        db = connect()
        existing = db.execute("SELECT status, scope FROM observations WHERE id=?", (ident,)).fetchone()
        if existing:
            db.close()
            return {"id": ident, "status": "already_captured", "scope": existing[1]}
        with db:
            db.execute(
                """INSERT INTO observations(id,created,kind,scope,confidence,summary,tags_json,status,
                   supersedes_id,conflict_group,valid_from,valid_until,importance,last_confirmed,source_event_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ident, now(), kind, scope, max(0.0, min(1.0, confidence)), summary,
                 json.dumps(sorted(set(tags or [])), ensure_ascii=False), "captured", supersedes_id,
                 conflict_group, valid_from, valid_until, max(0.0, min(1.0, importance)), "", source_event_id or ident),
            )
        db.close()
    return {"id": ident, "status": "captured", "scope": scope}


def promote(observation_id: str) -> dict[str, Any]:
    db = connect()
    row = _observation_row(db, observation_id)
    if not row:
        db.close()
        raise ValueError("observation not found")
    if row[7] in ("superseded", "conflicted", "expired"):
        db.close()
        raise ValueError(f"observation is not promotable: {row[7]}")
    date = datetime.now().strftime("%Y-%m-%d")
    folder = VAULT / "Hermes" / "Memory" / "Codex" / "Promoted"
    folder.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "-", observation_id)[:96]
    path = folder / f"{date}-{safe_id}.md"
    if row[7] in ("promoted", "active") and row[8] and Path(row[8]) == path and path.exists():
        db.close()
        return {"id": observation_id, "status": "already_active"}
    tags = json.loads(row[6])
    title = f"{row[2].title()} {safe_id}"
    content = "---\n" + yaml.safe_dump({"id": f"codex-observation-{observation_id}", "title": title, "type": "lesson", "created": row[1], "updated": now(), "source": "codex", "tags": ["codex", "promoted", *tags], "status": "active", "category": "Workflow", "project_scope": row[3], "confidence": row[4], "importance": row[13], "valid_from": row[11], "valid_until": row[12]}, allow_unicode=True, sort_keys=False) + "---\n\n# " + title + "\n\n" + row[5] + "\n"
    path.write_text(redact(content), encoding="utf-8")
    with db:
        db.execute("UPDATE observations SET status='active', promoted_path=?, last_confirmed=? WHERE id=?", (str(path), now(), observation_id))
    db.close()
    return {"id": observation_id, "status": "active", "path": str(path)}


def auto_curate(min_confidence: float = 0.8) -> dict[str, Any]:
    db = connect()
    ids = [row[0] for row in db.execute("SELECT id FROM observations WHERE status='captured' AND confidence >= ? ORDER BY created", (min_confidence,))]
    db.close()
    promoted = [promote(ident) for ident in ids]
    return {"eligible": len(ids), "promoted": len(promoted), "items": promoted}


def memory_supersede(new_id: str, old_id: str) -> dict[str, Any]:
    if not new_id or not old_id or new_id == old_id:
        raise ValueError("new_id and old_id must be distinct")
    with write_lock():
        db = connect()
        new_row, old_row = _observation_row(db, new_id), _observation_row(db, old_id)
        if not new_row or not old_row:
            db.close()
            raise ValueError("both observations must exist")
        timestamp = now()
        with db:
            db.execute("UPDATE observations SET status='superseded', valid_until=?, last_confirmed=? WHERE id=?", (timestamp, timestamp, old_id))
            db.execute("UPDATE observations SET status='active', supersedes_id=?, last_confirmed=? WHERE id=?", (old_id, timestamp, new_id))
        db.close()
    _update_promoted_note(str(old_row[8] or ""), "deprecated", timestamp)
    return {"status": "superseded", "active_id": new_id, "superseded_id": old_id}


def memory_conflict(observation_ids: list[str], conflict_group: str = "", title: str = "", reason: str = "") -> dict[str, Any]:
    ids = list(dict.fromkeys(str(item) for item in observation_ids if str(item)))
    if len(ids) < 2:
        raise ValueError("at least two observation ids are required")
    group = conflict_group or "conflict-" + hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]
    with write_lock():
        db = connect()
        rows = [_observation_row(db, ident) for ident in ids]
        if any(row is None for row in rows):
            db.close()
            raise ValueError("all conflict observations must exist")
        detected = now()
        with db:
            db.executemany("UPDATE observations SET status='conflicted', conflict_group=?, last_confirmed=? WHERE id=?", [(group, detected, ident) for ident in ids])
            cursor = db.execute("INSERT INTO conflicts(title,paths_json,detected,status,kind,conflict_group,resolution_note) VALUES(?,?,?,?,?,?,?)",
                                (title or f"Observation conflict {group}", json.dumps(ids), detected, "open", "observation", group, reason))
            conflict_id = int(cursor.lastrowid)
        db.close()
    for row in rows:
        _update_promoted_note(str(row[8] or ""), "deprecated", detected)
    return {"conflict_id": conflict_id, "status": "open", "conflict_group": group, "observation_ids": ids}


def memory_resolve(conflict_id: int, winner_id: str, resolution_note: str = "") -> dict[str, Any]:
    with write_lock():
        db = connect()
        conflict = db.execute("SELECT paths_json,status FROM conflicts WHERE id=? AND kind='observation'", (int(conflict_id),)).fetchone()
        if not conflict:
            db.close()
            raise ValueError("open observation conflict not found")
        ids = json.loads(conflict[0])
        if winner_id not in ids:
            db.close()
            raise ValueError("winner_id must belong to the conflict")
        timestamp = now()
        with db:
            db.execute("UPDATE observations SET status='active', conflict_group='', last_confirmed=? WHERE id=?", (timestamp, winner_id))
            losers = [ident for ident in ids if ident != winner_id]
            db.executemany("UPDATE observations SET status='superseded', valid_until=?, conflict_group='' WHERE id=?", [(timestamp, ident) for ident in losers])
            db.execute("UPDATE conflicts SET status='resolved',winner_id=?,resolved_at=?,resolution_note=? WHERE id=?", (winner_id, timestamp, redact(resolution_note), int(conflict_id)))
            winner = _observation_row(db, winner_id)
            loser_rows = [_observation_row(db, ident) for ident in losers]
        db.close()
    _update_promoted_note(str(winner[8] or ""), "active")
    for row in loser_rows:
        if row:
            _update_promoted_note(str(row[8] or ""), "deprecated", timestamp)
    return {"conflict_id": int(conflict_id), "status": "resolved", "winner_id": winner_id, "superseded_ids": losers}


def memory_expire(observation_ids: list[str] | None = None, before: str = "") -> dict[str, Any]:
    with write_lock():
        db = connect()
        at = _parse_timestamp(before) or datetime.now(timezone.utc)
        if observation_ids:
            ids = list(dict.fromkeys(str(item) for item in observation_ids if str(item)))
            placeholders = ",".join("?" for _ in ids)
            rows = db.execute(f"SELECT id FROM observations WHERE id IN ({placeholders}) AND status IN ('active','promoted')", ids).fetchall()
        else:
            rows = db.execute("SELECT id FROM observations WHERE status IN ('active','promoted') AND valid_until<>'' AND valid_until<=?", (at.isoformat(),)).fetchall()
        ids = [row[0] for row in rows]
        timestamp = now()
        with db:
            db.executemany("UPDATE observations SET status='expired', valid_until=?, last_confirmed=? WHERE id=?", [(timestamp, timestamp, ident) for ident in ids])
            note_rows = [_observation_row(db, ident) for ident in ids]
        db.close()
    for row in note_rows:
        if row:
            _update_promoted_note(str(row[8] or ""), "deprecated", timestamp)
    return {"status": "expired", "count": len(ids), "observation_ids": ids}


def memory_merge(target_id: str, source_ids: list[str]) -> dict[str, Any]:
    sources = [ident for ident in dict.fromkeys(source_ids) if ident and ident != target_id]
    if not sources:
        raise ValueError("at least one source observation is required")
    with write_lock():
        db = connect()
        target = _observation_row(db, target_id)
        source_rows = [_observation_row(db, ident) for ident in sources]
        if not target or any(row is None for row in source_rows):
            db.close()
            raise ValueError("target and all source observations must exist")
        summaries = [str(row[5]) for row in source_rows if row and str(row[5]) not in str(target[5])]
        tags = sorted(set(json.loads(target[6] or "[]")).union(*(set(json.loads(row[6] or "[]")) for row in source_rows if row)))
        merged_summary = str(target[5]) + ("\n\nMerged evidence:\n" + "\n".join(f"- {item}" for item in summaries) if summaries else "")
        timestamp = now()
        with db:
            db.execute("UPDATE observations SET summary=?,tags_json=?,importance=?,status='active',last_confirmed=? WHERE id=?", (redact(merged_summary), json.dumps(tags, ensure_ascii=False), max([float(target[13]), *[float(row[13]) for row in source_rows if row]]), timestamp, target_id))
            db.executemany("UPDATE observations SET status='superseded',supersedes_id=?,valid_until=?,last_confirmed=? WHERE id=?", [(target_id, timestamp, timestamp, ident) for ident in sources])
        db.close()
    for row in source_rows:
        if row:
            _update_promoted_note(str(row[8] or ""), "deprecated", timestamp)
    return {"status": "merged", "target_id": target_id, "source_ids": sources}


def memory_lifecycle(scope: str = "", include_expired: bool = False) -> dict[str, Any]:
    db = connect()
    rows = db.execute("""SELECT id,created,kind,scope,confidence,summary,tags_json,status,promoted_path,
                              supersedes_id,conflict_group,valid_from,valid_until,importance,last_confirmed,source_event_id
                       FROM observations ORDER BY created DESC""").fetchall()
    conflicts = db.execute("SELECT id,title,paths_json,detected,status,kind,conflict_group,winner_id,resolved_at,resolution_note FROM conflicts WHERE status='open' ORDER BY detected DESC").fetchall()
    db.close()
    items = [_observation_dict(row) for row in rows if not scope or row[3] in ("global", scope)]
    if not include_expired:
        items = [item for item in items if item["effective"] or item["status"] in ("captured", "conflicted")]
    effective = [item for item in items if item["effective"]]
    effective.sort(key=lambda item: (0.45 * float(item["importance"]) + 0.35 * float(item["confidence"]) + 0.20 * recency_score(item["last_confirmed"] or item["created"])), reverse=True)
    return {"scope": scope or "all", "effective_count": len(effective), "effective": effective, "items": items, "open_conflicts": [{"id": row[0], "title": row[1], "observation_ids": json.loads(row[2]), "detected": row[3], "status": row[4], "kind": row[5], "conflict_group": row[6], "winner_id": row[7], "resolved_at": row[8], "resolution_note": redact(row[9])} for row in conflicts]}


def memory_audit() -> dict[str, Any]:
    db = connect()
    violations: list[str] = []
    ids = {row[0] for row in db.execute("SELECT id FROM observations")}
    for ident, supersedes in db.execute("SELECT id,supersedes_id FROM observations WHERE supersedes_id<>''"):
        if supersedes not in ids:
            violations.append(f"dangling_supersedes:{ident}->{supersedes}")
    at = datetime.now(timezone.utc)
    for row in db.execute("SELECT id,status,valid_from,valid_until FROM observations"):
        effective, reason = _is_effective(row[1], row[2] or "", row[3] or "", at)
        if row[1] in ("active", "promoted") and not effective and reason == "expired":
            violations.append(f"unexpired_status:{row[0]}")
    for ident_json, status, kind in db.execute("SELECT paths_json,status,kind FROM conflicts WHERE status='open'"):
        members = json.loads(ident_json)
        if kind == "observation" and any(ident not in ids for ident in members):
            violations.append("conflict_member_missing")
        if kind == "observation" and any(db.execute("SELECT status FROM observations WHERE id=?", (ident,)).fetchone()[0] != "conflicted" for ident in members if ident in ids):
            violations.append("open_conflict_member_not_conflicted")
    counts = {row[0]: row[1] for row in db.execute("SELECT status,COUNT(*) FROM observations GROUP BY status")}
    db.close()
    return {"ok": not violations, "violations": violations, "status_counts": counts, "checked_at": now()}


def backup() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    target = BACKUPS / datetime.now().strftime("%Y%m%d_%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    source = connect()
    dest = sqlite3.connect(target / "codex_memory.db")
    source.backup(dest)
    dest.close(); source.close()
    db_path = target / "codex_memory.db"
    check = sqlite3.connect(db_path)
    manifest = {
        "schema": 1,
        "created": now(),
        "db_sha256": binary_digest(db_path),
        "notes": check.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
        "fts_docs": check.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0],
        "integrity_check": check.execute("PRAGMA integrity_check").fetchone()[0],
    }
    check.close()
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return {"path": str(db_path), "manifest": str(target / "manifest.json"), "sha256": manifest["db_sha256"], "bytes": db_path.stat().st_size}


def verify_backup(path: str) -> dict[str, Any]:
    backup_path = Path(path)
    if not backup_path.exists():
        raise ValueError(f"backup not found: {backup_path}")
    db = sqlite3.connect(backup_path)
    result = {
        "path": str(backup_path),
        "notes": db.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
        "fts_docs": db.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0],
        "integrity_check": db.execute("PRAGMA integrity_check").fetchone()[0],
    }
    db.close()
    manifest_path = backup_path.with_name("manifest.json")
    manifest_verified = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_verified = (
                manifest.get("db_sha256") == binary_digest(backup_path)
                and manifest.get("notes") == result["notes"]
                and manifest.get("fts_docs") == result["fts_docs"]
                and manifest.get("integrity_check") == result["integrity_check"]
            )
        except (OSError, json.JSONDecodeError):
            manifest_verified = False
    result["manifest"] = str(manifest_path)
    result["manifest_verified"] = manifest_verified
    result["verified"] = result["integrity_check"] == "ok" and result["notes"] == result["fts_docs"] and manifest_verified
    return result


def evaluate() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "eval" / "golden.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    passed = 0
    results = []
    for case in cases:
        found = search(case["query"], limit=10, scope=case.get("scope", ""))
        titles = " ".join(x["title"] for x in found)
        ok = all(term.lower() in titles.lower() for term in case["expected_titles"])
        passed += int(ok)
        results.append({"query": case["query"], "passed": ok, "results": len(found), "titles": [x["title"] for x in found[:5]]})
    return {"cases": len(cases), "passed": passed, "failed": len(cases) - passed, "results": results}


TOOLS = [
    {"name": "memory_status", "description": "Return Codex native memory health and counts.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_sync", "description": "Synchronize the Obsidian vault into the Codex-owned index.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_search", "description": "Search durable Codex memory with scope and confidence-aware ranking.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "scope": {"type": "string"}, "type": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["query"]}},
    {"name": "memory_capture", "description": "Capture a concise redacted observation for later promotion.", "inputSchema": {"type": "object", "properties": {"summary": {"type": "string"}, "kind": {"type": "string"}, "scope": {"type": "string"}, "confidence": {"type": "number"}, "importance": {"type": "number"}, "valid_from": {"type": "string"}, "valid_until": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["summary"]}},
    {"name": "memory_promote", "description": "Promote one captured observation into an Obsidian lesson note.", "inputSchema": {"type": "object", "properties": {"observation_id": {"type": "string"}}, "required": ["observation_id"]}},
    {"name": "memory_curate", "description": "Promote only high-confidence captured observations.", "inputSchema": {"type": "object", "properties": {"min_confidence": {"type": "number", "minimum": 0, "maximum": 1}}}},
    {"name": "memory_supersede", "description": "Make a newer observation active and supersede an older one.", "inputSchema": {"type": "object", "properties": {"new_id": {"type": "string"}, "old_id": {"type": "string"}}, "required": ["new_id", "old_id"]}},
    {"name": "memory_conflict", "description": "Mark mutually incompatible observations as an open conflict.", "inputSchema": {"type": "object", "properties": {"observation_ids": {"type": "array", "items": {"type": "string"}}, "conflict_group": {"type": "string"}, "title": {"type": "string"}, "reason": {"type": "string"}}, "required": ["observation_ids"]}},
    {"name": "memory_resolve", "description": "Resolve an observation conflict with an explicit winning observation.", "inputSchema": {"type": "object", "properties": {"conflict_id": {"type": "integer"}, "winner_id": {"type": "string"}, "resolution_note": {"type": "string"}}, "required": ["conflict_id", "winner_id"]}},
    {"name": "memory_expire", "description": "Expire selected or due active observations and deprecate their notes.", "inputSchema": {"type": "object", "properties": {"observation_ids": {"type": "array", "items": {"type": "string"}}, "before": {"type": "string"}}}},
    {"name": "memory_merge", "description": "Merge source observations into one active target and supersede the sources.", "inputSchema": {"type": "object", "properties": {"target_id": {"type": "string"}, "source_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["target_id", "source_ids"]}},
    {"name": "memory_lifecycle", "description": "Return effective memories, explainable status reasons, and open conflicts.", "inputSchema": {"type": "object", "properties": {"scope": {"type": "string"}, "include_expired": {"type": "boolean"}}}},
    {"name": "memory_audit", "description": "Audit lifecycle invariants, dangling supersedes links, and conflict membership.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "memory_evaluate", "description": "Run Codex-native golden retrieval evaluation.", "inputSchema": {"type": "object", "properties": {}}},
]


def call_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "memory_status": return status()
    if name == "memory_sync": return sync()
    if name == "memory_search": return search(args.get("query", ""), int(args.get("limit", 10)), args.get("scope", ""), args.get("type", ""))
    if name == "memory_capture": return capture(args.get("summary", ""), args.get("kind", "observation"), args.get("scope", "global"), float(args.get("confidence", 0.5)), args.get("tags"), importance=float(args.get("importance", 0.5)), valid_from=args.get("valid_from", ""), valid_until=args.get("valid_until", ""))
    if name == "memory_promote": return promote(args.get("observation_id", ""))
    if name == "memory_curate": return auto_curate(float(args.get("min_confidence", 0.8)))
    if name == "memory_supersede": return memory_supersede(args.get("new_id", ""), args.get("old_id", ""))
    if name == "memory_conflict": return memory_conflict(args.get("observation_ids", []), args.get("conflict_group", ""), args.get("title", ""), args.get("reason", ""))
    if name == "memory_resolve": return memory_resolve(int(args.get("conflict_id", 0)), args.get("winner_id", ""), args.get("resolution_note", ""))
    if name == "memory_expire": return memory_expire(args.get("observation_ids") or [], args.get("before", ""))
    if name == "memory_merge": return memory_merge(args.get("target_id", ""), args.get("source_ids", []))
    if name == "memory_lifecycle": return memory_lifecycle(args.get("scope", ""), bool(args.get("include_expired", False)))
    if name == "memory_audit": return memory_audit()
    if name == "memory_evaluate": return evaluate()
    raise ValueError(f"unknown tool: {name}")


def serve() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        method = request.get("method")
        ident = request.get("id")
        if method == "notifications/initialized":
            continue
        if method == "initialize":
            result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "codex-native-memory", "version": "1.1.0"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                params = request.get("params", {})
                output = call_tool(params.get("name", ""), params.get("arguments", {}))
                result = {"content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}], "structuredContent": output}
            except Exception as exc:
                result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        else:
            if ident is None:
                continue
            result = {"error": {"code": -32601, "message": f"Method not found: {method}"}}
        if ident is not None:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sync", "status", "search", "capture", "promote", "curate", "supersede", "conflict", "resolve", "expire", "merge", "lifecycle", "audit", "backup", "verify-backup", "evaluate", "serve", "semantic-worker"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--scope", default="")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--kind", default="observation")
    parser.add_argument("--event-id", default="")
    parser.add_argument("--new-id", default="")
    parser.add_argument("--old-id", default="")
    parser.add_argument("--target-id", default="")
    parser.add_argument("--winner-id", default="")
    parser.add_argument("--conflict-id", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--conflict-group", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--resolution-note", default="")
    parser.add_argument("--before", default="")
    parser.add_argument("--include-expired", action="store_true")
    parser.add_argument("--importance", type=float, default=0.5)
    parser.add_argument("--valid-from", default="")
    parser.add_argument("--valid-until", default="")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()
    if args.command == "serve": return serve()
    elif args.command == "semantic-worker": return semantic_worker()
    elif args.command == "sync": print(json.dumps(sync(), ensure_ascii=False))
    elif args.command == "status": print(json.dumps(status(), ensure_ascii=False))
    elif args.command == "search": print(json.dumps(search(args.query or "", args.limit, args.scope), ensure_ascii=False, indent=2))
    elif args.command == "capture": print(json.dumps(capture(args.query or "", kind=args.kind, scope=args.scope or "global", confidence=args.confidence, tags=args.tag, event_id=args.event_id or None, importance=args.importance, valid_from=args.valid_from, valid_until=args.valid_until), ensure_ascii=False))
    elif args.command == "promote": print(json.dumps(promote(args.query or ""), ensure_ascii=False))
    elif args.command == "curate": print(json.dumps(auto_curate(args.min_confidence), ensure_ascii=False))
    elif args.command == "supersede": print(json.dumps(memory_supersede(args.new_id, args.old_id), ensure_ascii=False))
    elif args.command == "conflict": print(json.dumps(memory_conflict(args.ids, args.conflict_group, args.title, args.reason), ensure_ascii=False))
    elif args.command == "resolve": print(json.dumps(memory_resolve(args.conflict_id, args.winner_id, args.resolution_note), ensure_ascii=False))
    elif args.command == "expire": print(json.dumps(memory_expire(args.ids, args.before), ensure_ascii=False))
    elif args.command == "merge": print(json.dumps(memory_merge(args.target_id, args.ids), ensure_ascii=False))
    elif args.command == "lifecycle": print(json.dumps(memory_lifecycle(args.scope, args.include_expired), ensure_ascii=False, indent=2))
    elif args.command == "audit": print(json.dumps(memory_audit(), ensure_ascii=False, indent=2))
    elif args.command == "backup": print(json.dumps(backup(), ensure_ascii=False))
    elif args.command == "verify-backup": print(json.dumps(verify_backup(args.query or ""), ensure_ascii=False))
    elif args.command == "evaluate": print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
