import importlib.util
import json
import sqlite3
from pathlib import Path


MODULE = Path(__file__).parents[1] / "tools" / "codex_auto_memory_capture.py"
spec = importlib.util.spec_from_file_location("codex_auto_memory_capture", MODULE)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def _make_db(path: Path, rollout: Path, cwd: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE threads (
            id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, updated_at_ms INTEGER,
            title TEXT NOT NULL, cwd TEXT NOT NULL, first_user_message TEXT NOT NULL,
            preview TEXT NOT NULL, recency_at_ms INTEGER NOT NULL
        )"""
    )
    connection.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("thread-1", str(rollout), 1000, "Test task", str(cwd), "Build the memory loop", "Build the memory loop", 1000),
    )
    connection.commit()
    connection.close()


def _write_rollout(path: Path, terminal: str = "task_complete") -> None:
    records = [
        {"timestamp": "2026-07-15T00:00:00Z", "type": "event_msg", "payload": {"type": "task_started"}},
        {"timestamp": "2026-07-15T00:00:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "Build a memory loop token=secret-value"}},
        {"timestamp": "2026-07-15T00:00:02Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "pytest passed; verification gate PASS"}]}},
        {"timestamp": "2026-07-15T00:00:03Z", "type": "event_msg", "payload": {"type": terminal}},
    ]
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


def test_extracts_latest_completed_task_before_new_active_task(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(rollout)
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"payload": {"type": "task_started"}}) + "\n")
    segment = collector.extract_latest_terminal(rollout)
    assert segment is not None
    assert segment["terminal"] == "task_complete"


def test_capture_is_redacted_and_idempotent(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(rollout)
    db = tmp_path / "state.sqlite"
    _make_db(db, rollout, tmp_path)
    inbox = tmp_path / "inbox"
    state = tmp_path / "state.json"
    first = collector.run_capture(db, inbox, state, min_age_seconds=0, now_ms=2000)
    second = collector.run_capture(db, inbox, state, min_age_seconds=0, now_ms=2000)
    assert first["queued"] == 1
    assert second["queued"] == 0
    packet = json.loads(next(inbox.glob("*.json")).read_text(encoding="utf-8"))
    assert "secret-value" not in json.dumps(packet)
    assert packet["metadataSource"] == "codex-state-sqlite-rollout-jsonl"


def test_active_rollout_is_not_queued(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(json.dumps({"timestamp": "2026-07-15T00:00:04Z", "payload": {"type": "task_started"}}) + "\n", encoding="utf-8")
    db = tmp_path / "state.sqlite"
    _make_db(db, rollout, tmp_path)
    result = collector.run_capture(db, tmp_path / "inbox", tmp_path / "state.json", min_age_seconds=0, now_ms=2000)
    assert result["queued"] == 0


def test_redaction_produces_json_safe_utf8(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    _write_rollout(rollout)
    db = tmp_path / "state.sqlite"
    _make_db(db, rollout, tmp_path)
    packet = collector.build_event(
        {"id": "thread", "rollout_path": str(rollout), "cwd": str(tmp_path), "first_user_message": "fallback", "updated_at_ms": 1},
        {"terminal": "task_complete", "status": "PASS", "timestamp": "now", "eventTexts": ["bad \udcac text"], "fallbackTexts": [], "assistant": []},
        now_ms=2,
    )
    encoded = json.dumps(packet, ensure_ascii=False).encode("utf-8")
    assert json.loads(encoded.decode("utf-8"))["objective"] == "bad � text"
