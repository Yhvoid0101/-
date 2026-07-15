import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import textwrap

os.environ["CODEX_MEMORY_SEMANTIC"] = "disabled"
from pathlib import Path


MODULE = Path(__file__).parents[1] / "src" / "codex_memory_platform.py"
spec = importlib.util.spec_from_file_location("codex_memory_platform", MODULE)
platform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(platform)


def test_schema_and_status(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "Hermes" / "Decisions" / "Codex").mkdir(parents=True)
    note = vault / "Hermes" / "Decisions" / "Codex" / "one.md"
    note.write_text("---\nid: one\ntitle: One\ntype: decision\nstatus: active\ncategory: Decisions\nupdated: 2026-07-14T00:00:00+00:00\n---\n\n# One\n", encoding="utf-8")
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    result = platform.sync()
    assert result["notes"] == 1
    health = platform.status()
    assert health["notes"] == health["fts_docs"] == 1
    assert health["integrity_check"] == "ok"
    assert health["semantic_resource_policy"] == "isolated-worker-serialized-release"
    assert health["semantic_session_loaded"] is False


def test_capture_and_promote(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    captured = platform.capture("A reusable redacted lesson", kind="lesson", scope="demo", confidence=0.9)
    promoted = platform.promote(captured["id"])
    assert promoted["status"] == "active"
    assert Path(promoted["path"]).exists()


def test_promoted_observations_have_unique_paths(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    first = platform.capture("first task end", kind="task-end", scope="demo", confidence=0.95)
    second = platform.capture("second task end", kind="task-end", scope="demo", confidence=0.95)
    first_path = Path(platform.promote(first["id"])["path"])
    second_path = Path(platform.promote(second["id"])["path"])
    assert first_path != second_path
    assert first_path.exists() and second_path.exists()
    first_text = first_path.read_text(encoding="utf-8")
    second_text = second_path.read_text(encoding="utf-8")
    first_title = first_text.split("title: ", 1)[1].splitlines()[0]
    second_title = second_text.split("title: ", 1)[1].splitlines()[0]
    assert first_title.startswith("Task-End ")
    assert second_title.startswith("Task-End ")
    assert first_title != second_title


def test_backup_manifest_is_verified(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    platform.capture("backup manifest observation", scope="demo")
    platform.sync()
    backup = platform.backup()
    verified = platform.verify_backup(backup["path"])
    assert Path(backup["manifest"]).exists()
    assert verified["manifest_verified"] is True
    assert verified["verified"] is True


def test_capture_event_id_is_idempotent(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    first = platform.capture("first summary", scope="demo", event_id="event-123")
    replay = platform.capture("replayed summary", scope="demo", event_id="event-123")
    assert first["status"] == "captured"
    assert replay["status"] == "already_captured"


def test_lifecycle_supersede_conflict_resolve_expire_and_audit(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")

    old = platform.capture("old workflow", scope="demo", confidence=0.8, importance=0.4)
    new = platform.capture("new workflow", scope="demo", confidence=0.95, importance=0.9)
    platform.promote(new["id"])
    superseded = platform.memory_supersede(new["id"], old["id"])
    assert superseded["active_id"] == new["id"]

    left = platform.capture("left decision", scope="demo")
    right = platform.capture("right decision", scope="demo")
    conflict = platform.memory_conflict([left["id"], right["id"]], title="decision conflict")
    assert platform.memory_lifecycle("demo")["open_conflicts"]
    resolved = platform.memory_resolve(conflict["conflict_id"], left["id"], "evidence favored left")
    assert resolved["winner_id"] == left["id"]

    expiring = platform.capture("temporary rule", scope="demo")
    platform.promote(expiring["id"])
    expired = platform.memory_expire([expiring["id"]])
    assert expired["count"] == 1
    lifecycle = platform.memory_lifecycle("demo")
    assert any(item["id"] == new["id"] and item["effective"] for item in lifecycle["effective"])
    assert platform.memory_audit()["ok"] is True


def test_lifecycle_merge_is_lossless_and_auditable(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    target = platform.capture("canonical lesson", tags=["one"], importance=0.5)
    source = platform.capture("supporting evidence", tags=["two"], importance=0.9)
    result = platform.memory_merge(target["id"], [source["id"]])
    assert result["status"] == "merged"
    row = platform._observation_row(platform.connect(), target["id"])
    assert "supporting evidence" in row[5]
    assert platform.memory_audit()["ok"] is True


def test_mcp_handshake(tmp_path):
    env = os.environ.copy()
    env["CODEX_MEMORY_STORE"] = str(tmp_path / "store")
    env["CODEX_MEMORY_VAULT"] = str(tmp_path / "vault")
    (tmp_path / "vault").mkdir()
    process = subprocess.Popen(
        [sys.executable, str(MODULE), "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
    )
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_status", "arguments": {}}},
    ]
    output, _ = process.communicate("".join(json.dumps(x) + "\n" for x in requests), timeout=10)
    responses = [json.loads(line) for line in output.splitlines() if line.strip()]
    assert responses[0]["result"]["serverInfo"]["name"] == "codex-native-memory"
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert "memory_search" in tool_names
    assert {"memory_supersede", "memory_conflict", "memory_resolve", "memory_expire", "memory_merge", "memory_lifecycle", "memory_audit"}.issubset(tool_names)
    assert responses[2]["result"]["structuredContent"]["integrity_check"] == "ok"


def test_search_falls_back_for_fts_syntax(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    note.write_text("---\ntitle: Lifecycle\ntype: lesson\nstatus: active\ncategory: Workflow\n---\n\nhigh-confidence lifecycle observation", encoding="utf-8")
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    platform.sync()
    assert platform.search("high-confidence lifecycle", limit=3)


def test_search_tolerates_partial_multiword_queries(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "pagination.md"
    note.write_text(
        "---\ntitle: Pagination Contract\ntype: knowledge\nstatus: active\ncategory: Knowledge\n---\n\nREST pagination error envelopes",
        encoding="utf-8",
    )
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    platform.sync()
    results = platform.search("pagination error envelope", limit=3)
    assert results and results[0]["title"] == "Pagination Contract"


def test_capture_sanitizes_invalid_surrogates(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "VAULT", tmp_path / "vault")
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    result = platform.capture("bad surrogate \udcac but durable", confidence=0.9)
    assert result["status"] == "captured"


def test_semantic_input_is_bounded_without_losing_edges():
    text = "begin " + ("x" * 20000) + " end"
    bounded = platform.bounded_semantic_text(text, max_chars=100)
    assert len(bounded) <= 100 + len("\n[semantic middle truncated]\n")
    assert bounded.startswith("begin ")
    assert bounded.endswith(" end")


def test_semantic_operations_are_serialized(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    active = 0
    maximum = 0
    guard = threading.Lock()

    def worker():
        nonlocal active, maximum
        with platform.semantic_operation(timeout=2):
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.05)
            with guard:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_semantic_operations_are_serialized_across_processes(tmp_path):
    store = tmp_path / "store"
    script = textwrap.dedent(f"""
        import importlib.util
        import time
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("platform", r"{MODULE}")
        platform = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(platform)
        with platform.semantic_operation(timeout=5):
            print("entered", flush=True)
            time.sleep(0.25)
    """)
    env = os.environ.copy()
    env["CODEX_MEMORY_STORE"] = str(store)
    env["CODEX_MEMORY_SEMANTIC"] = "disabled"
    first = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env)
    second = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env)
    first_output, _ = first.communicate(timeout=10)
    second_output, _ = second.communicate(timeout=10)
    assert first.returncode == second.returncode == 0
    assert first_output.strip() == second_output.strip() == "entered"


def test_search_preserves_fts_when_semantic_provider_fails(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "reliability.md"
    note.write_text("---\ntitle: Reliability\ntype: lesson\nstatus: active\ncategory: Workflow\n---\n\nserial resource gate", encoding="utf-8")
    monkeypatch.setattr(platform, "VAULT", vault)
    monkeypatch.setattr(platform, "STORE", tmp_path / "store")
    monkeypatch.setattr(platform, "DB", tmp_path / "store" / "codex_memory.db")
    monkeypatch.setattr(platform, "LOCK", tmp_path / "store" / ".lock")
    monkeypatch.setenv("CODEX_MEMORY_SEMANTIC", "disabled")
    platform.sync()
    db = platform.connect()
    db.execute("INSERT INTO note_embeddings(path,model,sha256,dimensions,vector,updated) VALUES(?,?,?,?,?,?)", ("reliability.md", "test", "x", 1, sqlite3.Binary(b"\x00\x00\x80?"), platform.now()))
    db.commit()
    db.close()
    monkeypatch.setenv("CODEX_MEMORY_SEMANTIC", "required")
    def failing_encode(texts, batch_size=1):
        raise RuntimeError("simulated semantic allocation failure")
    monkeypatch.setattr(platform, "semantic_encode", failing_encode)
    results = platform.search("serial resource gate", limit=3)
    assert results and results[0]["title"] == "Reliability"
    assert platform.status()["semantic_health"] == "degraded"


def test_semantic_worker_jsonl_protocol(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(textwrap.dedent("""
        import json
        import sys
        for line in sys.stdin:
            request = json.loads(line)
            vectors = [[float(len(text))] for text in request["texts"]]
            print(json.dumps({"ok": True, "vectors": vectors}), flush=True)
    """), encoding="utf-8")
    client = platform.SemanticWorkerClient([sys.executable, str(worker)], timeout=5)
    try:
        assert client.encode(["a", "abcd"], batch_size=1) == [[1.0], [4.0]]
        assert client.active is True
    finally:
        client.close()
    assert client.active is False
