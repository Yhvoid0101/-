"""Distill completed Codex rollouts into the durable memory inbox.

The collector reads only local Codex metadata and JSONL rollout evidence. It
never stores a transcript; it writes bounded, redacted task summaries with
content-derived event ids so repeated scans are idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


TERMINAL_TYPES = {"task_complete": "PASS", "turn_aborted": "BLOCKED"}
SECRET_PATTERNS = (
    re.compile(r"(?i)ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret(?:key)?)\s*[:=]\s*[^\s,;]+"),
)
TEST_SIGNAL = re.compile(
    r"(?i)(pytest|unittest|test\b|verification|golden|coverage|gate\b|passed|pass\b|audit)"
)
FAILURE_SIGNAL = re.compile(
    r"(?i)(error|failed|failure|blocked|exception|traceback|residual risk|incomplete|not verified|could not)"
)
DECISION_SIGNAL = re.compile(r"(?i)(decision|decided|chosen|architecture|tradeoff)")


def redact(value: Any, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = re.sub(r"[\ud800-\udfff]", "�", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "\n[truncated]"
    return text


def normalize_path(value: str) -> Path:
    text = str(value or "").strip()
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text)


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:24]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or "")
    return ""


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _append_unique(items: list[str], value: str, limit: int = 240) -> None:
    cleaned = redact(value, limit).strip()
    if cleaned and cleaned not in items and len(items) < 64:
        items.append(cleaned)


def iter_rollout(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def extract_latest_terminal(path: Path) -> dict[str, Any] | None:
    """Return the latest completed task segment, ignoring a newer active task."""

    current: dict[str, Any] | None = None
    latest: dict[str, Any] | None = None
    for event in iter_rollout(path):
        payload = _event_payload(event)
        event_type = payload.get("type")
        if event_type == "task_started":
            current = {"eventTexts": [], "fallbackTexts": [], "assistant": [], "start": event.get("timestamp", "")}
            continue
        if current is None:
            continue
        if event_type == "user_message":
            _append_unique(current["eventTexts"], payload.get("message", ""), 1200)
        elif event.get("type") == "response_item" and payload.get("type") == "message":
            text = _content_text(payload.get("content"))
            if payload.get("role") == "user":
                _append_unique(current["fallbackTexts"], text, 1200)
            elif payload.get("role") == "assistant":
                _append_unique(current["assistant"], text, 1000)
        if event_type in TERMINAL_TYPES:
            current["terminal"] = event_type
            current["status"] = TERMINAL_TYPES[event_type]
            current["timestamp"] = event.get("timestamp") or payload.get("completed_at") or ""
            latest = current
            current = None
    return latest


def _signal_lines(text: str, pattern: re.Pattern[str], limit: int = 6) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not pattern.search(line):
            continue
        _append_unique(lines, line, 260)
        if len(lines) >= limit:
            break
    return lines


def _project_scope(cwd: Path) -> str:
    lower = str(cwd).lower().replace("/", "\\")
    marker = "\\codex_projects\\"
    if marker in lower:
        index = lower.index(marker) + len(marker)
        tail = str(cwd).replace("/", "\\")[index:].strip("\\")
        return f"codex-projects/{tail.split('\\')[0]}"
    return f"workspace/{cwd.name or 'unknown'}"


def _git_root(cwd: Path) -> Path | None:
    candidate = cwd if cwd.is_dir() else cwd.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def changed_files(cwd: Path) -> list[dict[str, str]]:
    root = _git_root(cwd)
    if root is None:
        return []
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    items: list[dict[str, str]] = []
    for raw in result.stdout.splitlines():
        if len(raw) < 4:
            continue
        relative = raw[3:].strip()
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        target = root / relative
        digest = "missing"
        if target.is_file():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        items.append({"path": relative, "sha256": digest})
        if len(items) >= 100:
            break
    return items


def load_threads(db_path: Path, limit: int) -> list[dict[str, Any]]:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT id, rollout_path, updated_at_ms, title, cwd, first_user_message,
                      preview, recency_at_ms
                 FROM threads
                ORDER BY recency_at_ms DESC
                LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("events", {}), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"schema": 1, "events": {}, "lastScanAt": None}


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    # ASCII-only JSON keeps Windows PowerShell 5.1 ConvertFrom-Json stable for legacy rollout text.
    temp.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def build_event(thread: dict[str, Any], segment: dict[str, Any], now_ms: int | None = None) -> dict[str, Any]:
    rollout = normalize_path(thread["rollout_path"])
    cwd = normalize_path(thread["cwd"])
    texts = segment.get("eventTexts") or segment.get("fallbackTexts") or []
    objective = texts[0] if texts else thread.get("first_user_message") or thread.get("title")
    evidence = "\n".join(segment.get("assistant", []))
    artifacts = changed_files(cwd)
    terminal = segment["terminal"]
    status = segment["status"]
    identity = {
        "threadId": thread["id"],
        "terminal": terminal,
        "terminalTimestamp": segment.get("timestamp", ""),
        "rollout": str(rollout),
        "projectScope": _project_scope(cwd),
    }
    event_id = "auto-task-end-" + short_hash(json.dumps(identity, sort_keys=True))
    tests = _signal_lines(evidence, TEST_SIGNAL)
    failures = _signal_lines(evidence, FAILURE_SIGNAL)
    decisions = _signal_lines(evidence, DECISION_SIGNAL)
    risks = _signal_lines(evidence, re.compile(r"(?i)(residual risk|blocked|pending|conditional|not verified|incomplete)"))
    captured_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if now_ms is None else str(now_ms)
    summary = (
        f"Automatically distilled Codex task terminal={terminal}; "
        f"objective={redact(objective, 600)}; project={_project_scope(cwd)}; "
        f"changed_files={len(artifacts)}; tests={len(tests)}; failures={len(failures)}; "
        f"residual_risks={len(risks)}."
    )
    event = {
        "schema": 3,
        "eventId": event_id,
        "observedAt": captured_at,
        "kind": "task-end",
        "taskId": f"{thread['id']}:{short_hash(segment.get('timestamp', ''))}",
        "event": terminal,
        "projectScope": _project_scope(cwd),
        "scope": _project_scope(cwd),
        "objective": redact(objective, 1200),
        "status": status,
        "confidence": 0.95 if status == "PASS" else 0.8,
        "summary": redact(summary, 2400),
        "changedFiles": [item["path"] for item in artifacts],
        "changedFileHashes": artifacts,
        "tests": tests,
        "decisions": decisions,
        "failures": failures,
        "residualRisks": risks,
        "metadataSource": "codex-state-sqlite-rollout-jsonl",
        "provenance": {
            "rolloutSha256": hashlib.sha256(rollout.read_bytes()).hexdigest() if rollout.is_file() else "missing",
            "rolloutPath": str(rollout),
            "threadUpdatedAtMs": thread.get("updated_at_ms"),
            "capturedAt": captured_at,
        },
    }
    event["provenance"]["payloadSha256"] = short_hash(json.dumps(event, ensure_ascii=True, sort_keys=True))
    return event


def run_capture(
    db_path: Path,
    inbox: Path,
    state_path: Path,
    min_age_seconds: int = 90,
    max_threads: int = 100,
    now_ms: int | None = None,
) -> dict[str, Any]:
    now = int(time.time() * 1000) if now_ms is None else now_ms
    state = _load_state(state_path)
    events = state.setdefault("events", {})
    scanned = queued = skipped = 0
    failures: list[str] = []
    candidates: list[str] = []
    for thread in load_threads(db_path, max_threads):
        scanned += 1
        updated = int(thread.get("updated_at_ms") or thread.get("recency_at_ms") or 0)
        if updated and now - updated < min_age_seconds * 1000:
            skipped += 1
            continue
        rollout = normalize_path(thread["rollout_path"])
        if not rollout.is_file():
            continue
        try:
            segment = extract_latest_terminal(rollout)
            if not segment:
                continue
            event = build_event(thread, segment, now_ms=now)
            event_id = event["eventId"]
            candidates.append(event_id)
            if event_id in events:
                skipped += 1
                continue
            target = inbox / f"{event_id}.json"
            if not target.exists():
                _write_json_atomic(target, event)
            events[event_id] = {
                "threadId": thread["id"],
                "status": "queued",
                "queuedAt": now,
            }
            queued += 1
            _write_json_atomic(state_path, {"schema": 1, "events": events, "lastScanAt": now})
        except Exception as exc:  # preserve other threads for the next scan
            failures.append(f"{thread.get('id')}: {redact(exc, 500)}")
    state["lastScanAt"] = now
    _write_json_atomic(state_path, {"schema": 1, "events": events, "lastScanAt": now})
    result = {
        "status": "PASS" if not failures else "PARTIAL",
        "scanned": scanned,
        "queued": queued,
        "skipped": skipped,
        "candidates": candidates,
        "failures": failures,
        "inbox": str(inbox),
        "state": str(state_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--inbox", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--min-age-seconds", type=int, default=90)
    parser.add_argument("--max-threads", type=int, default=100)
    args = parser.parse_args()
    result = run_capture(args.db, args.inbox, args.state, args.min_age_seconds, args.max_threads)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] != "PARTIAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
