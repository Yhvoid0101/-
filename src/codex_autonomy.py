"""Codex-owned durable autonomy and evidence-gated evolution primitives.

This is a bounded local control plane inspired by OpenClaw's durable gateway
state and Hermes' learning loop.  It deliberately does not execute arbitrary
code, modify repositories, commit, or deploy.  Callers provide a small,
already-authorized executor and an independent verifier for evolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _command_ok(command: list[str], *, timeout: int = 10) -> bool:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _run_checked(command: list[str], *, timeout: int = 30) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()
        raise RuntimeError(detail[-2000:])
    return completed.stdout


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','dead')),
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  next_run_at REAL NOT NULL,
  claimed_at REAL,
  finished_at REAL,
  last_error TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_due_idx ON jobs(status, next_run_at);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  state_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('active','paused','completed','lost')),
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evolution_candidates (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK(kind IN ('memory','skill','workflow')),
  title TEXT NOT NULL,
  proposal_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  risk TEXT NOT NULL CHECK(risk IN ('low','medium','high')),
  confidence REAL NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('proposed','verified','promoted','rejected')),
  created_at REAL NOT NULL,
  verified_at REAL,
  promoted_at REAL,
  rejection_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS control_events (
  id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evolution_policies (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  job_kind TEXT NOT NULL,
  failure_signature TEXT NOT NULL,
  version INTEGER NOT NULL,
  parent_id TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK(status IN ('active','retired','rolled_back')),
  proposal_json TEXT NOT NULL,
  evaluation_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  activated_at REAL,
  retired_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS evolution_policy_version_idx
  ON evolution_policies(job_kind, failure_signature, version);
CREATE TABLE IF NOT EXISTS evolution_checkpoints (
  cycle_key TEXT PRIMARY KEY,
  phase TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
  state_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS policy_outcomes (
  id TEXT PRIMARY KEY,
  policy_id TEXT NOT NULL,
  passed INTEGER NOT NULL CHECK(passed IN (0,1)),
  metrics_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS policy_outcomes_idx ON policy_outcomes(policy_id, created_at);
"""


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


class AutonomyStore:
    """SQLite-backed queue/session/evolution store with crash recovery."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript(SCHEMA)

    def record_control_event(self, event_type: str, payload: dict[str, Any]) -> str:
        """Persist a bounded, redacted learning signal for later evolution analysis."""
        safe = json.dumps(payload, ensure_ascii=False)
        safe = re.sub(r"(?i)(token|password|secret|api[_-]?key)\s*[:=]\s*[^,}\s]+", r"\1=[REDACTED]", safe)
        event_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO control_events(id,event_type,payload_json,created_at) VALUES (?,?,?,?)",
                (event_id, event_type, safe[:4000], time.time()),
            )
        return event_id

    def enqueue(self, kind: str, payload: dict[str, Any], *, idempotency_key: str,
                max_attempts: int = 3, delay_seconds: float = 0) -> dict[str, Any]:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        now = time.time()
        job_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO jobs
                (id,idempotency_key,kind,payload_json,status,max_attempts,next_run_at,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (job_id, idempotency_key, kind, json.dumps(payload, ensure_ascii=False),
                 "pending", max(1, int(max_attempts)), now + max(0, delay_seconds), now, now),
            )
            row = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        return dict(row)

    def recover_orphans(self, *, stale_after: float = 300) -> dict[str, int]:
        cutoff = time.time() - stale_after
        now = time.time()
        with self.connect() as db:
            cur = db.execute(
                """UPDATE jobs SET status='pending', next_run_at=?, claimed_at=NULL, updated_at=?
                   WHERE status='running' AND claimed_at IS NOT NULL AND claimed_at < ?
                   AND attempts < max_attempts""", (now, now, cutoff))
            recovered = cur.rowcount
            cur = db.execute(
                """UPDATE jobs SET status='dead', finished_at=?, updated_at=?,
                   last_error=CASE WHEN last_error='' THEN 'orphaned after retry budget' ELSE last_error END
                   WHERE status='running' AND claimed_at IS NOT NULL AND claimed_at < ?
                   AND attempts >= max_attempts""", (now, now, cutoff))
            dead = cur.rowcount
        return {"recovered": recovered, "dead": dead}

    def claim(self, *, now: float | None = None) -> Job | None:
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='pending' AND next_run_at<=? ORDER BY created_at LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                db.execute("COMMIT")
                return None
            db.execute("UPDATE jobs SET status='running', attempts=attempts+1, claimed_at=?, updated_at=? WHERE id=?",
                       (now, now, row["id"]))
            db.execute("COMMIT")
            return Job(row["id"], row["kind"], json.loads(row["payload_json"]),
                       row["attempts"] + 1, row["max_attempts"])

    def succeed(self, job_id: str) -> None:
        now = time.time()
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='succeeded', finished_at=?, updated_at=? WHERE id=?", (now, now, job_id))

    def _active_policy(self, kind: str, error: str) -> dict[str, Any] | None:
        signature = re.sub(r"[0-9a-f]{8,}", "<id>", error, flags=re.IGNORECASE)
        with self.connect() as db:
            row = db.execute("""SELECT * FROM evolution_policies
                               WHERE job_kind=? AND failure_signature=? AND status='active'
                               ORDER BY version DESC LIMIT 1""", (kind, signature)).fetchone()
        return dict(row) if row else None

    def _adaptive_delay_multiplier(self, kind: str, error: str) -> float:
        policy = self._active_policy(kind, error)
        if policy:
            try:
                proposal = json.loads(policy["proposal_json"])
                return min(4.0, max(1.0, float(proposal.get("retry_delay_multiplier", 1.0))))
            except (ValueError, TypeError, json.JSONDecodeError):
                return 1.0
        return 1.0

    def record_policy_outcome(self, policy_id: str, passed: bool, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = metrics or {}
        with self.connect() as db:
            policy = db.execute("SELECT status FROM evolution_policies WHERE id=?", (policy_id,)).fetchone()
            if policy is None:
                raise KeyError(policy_id)
            db.execute("INSERT INTO policy_outcomes(id,policy_id,passed,metrics_json,created_at) VALUES (?,?,?,?,?)",
                       (str(uuid.uuid4()), policy_id, int(bool(passed)), json.dumps(metrics, ensure_ascii=False)[:2000], time.time()))
            recent = db.execute("SELECT passed FROM policy_outcomes WHERE policy_id=? ORDER BY created_at DESC LIMIT 5",
                                (policy_id,)).fetchall()
        failures = sum(1 for row in recent if not row[0])
        regression = len(recent) >= 3 and failures / len(recent) > 0.5
        rollback = self.rollback_policy(policy_id, f"automatic regression: {failures}/{len(recent)} failures") if regression and policy["status"] == "active" else None
        return {"policy_id": policy_id, "samples": len(recent), "failures": failures,
                "regression": regression, "rollback": rollback}
        for row in rows:
            try:
                proposal = json.loads(row[0])
            except json.JSONDecodeError:
                continue
            if proposal.get("job_kind") == kind and proposal.get("failure_signature") == signature:
                return min(4.0, max(1.0, float(proposal.get("retry_delay_multiplier", 1.0))))
        return 1.0

    def _checkpoint(self, cycle_key: str, phase: str, status: str, state: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True)[:4000]
        with self.connect() as db:
            db.execute(
                """INSERT INTO evolution_checkpoints(cycle_key,phase,status,state_json,updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(cycle_key) DO UPDATE SET phase=excluded.phase,status=excluded.status,
                   state_json=excluded.state_json,updated_at=excluded.updated_at""",
                (cycle_key, phase, status, payload, time.time()),
            )
            row = db.execute("SELECT * FROM evolution_checkpoints WHERE cycle_key=?", (cycle_key,)).fetchone()
        return dict(row)

    def _get_checkpoint(self, cycle_key: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM evolution_checkpoints WHERE cycle_key=?", (cycle_key,)).fetchone()
        return dict(row) if row else None

    def fail(self, job_id: str, error: str, *, base_delay: float = 5) -> dict[str, Any]:
        now = time.time()
        with self.connect() as db:
            job = db.execute("SELECT kind FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise KeyError(job_id)
        policy = self._active_policy(job["kind"], error)
        with self.connect() as db:
            row = db.execute("SELECT attempts,max_attempts,kind FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            terminal = row["attempts"] >= row["max_attempts"]
            status = "dead" if terminal else "pending"
            delay = base_delay * self._adaptive_delay_multiplier(row["kind"], error) * (2 ** max(0, row["attempts"] - 1))
            db.execute("""UPDATE jobs SET status=?, last_error=?, next_run_at=?, finished_at=?, updated_at=?
                         WHERE id=?""", (status, str(error)[:2000], now + delay, now if terminal else None, now, job_id))
        if policy:
            self.record_policy_outcome(policy["id"], False, {"job_id": job_id, "error": error[:500]})
        return {"status": status, "retry_in": 0 if terminal else delay}

    def run_once(self, executor: Callable[[Job], None]) -> dict[str, Any]:
        self.recover_orphans()
        job = self.claim()
        if job is None:
            return {"status": "idle"}
        try:
            executor(job)
        except Exception as exc:  # persist exact bounded failure before returning
            result = self.fail(job.id, f"{type(exc).__name__}: {exc}")
            self.record_control_event("job_failed", {
                "job_id": job.id, "kind": job.kind, "attempt": job.attempts,
                "error": f"{type(exc).__name__}: {exc}"[:1500], "status": result["status"],
            })
            return {"status": result["status"], "job_id": job.id, "error": str(exc), "retry_in": result["retry_in"]}
        self.succeed(job.id)
        self.record_control_event("job_succeeded", {"job_id": job.id, "kind": job.kind, "attempt": job.attempts})
        return {"status": "succeeded", "job_id": job.id}

    def status(self) -> dict[str, Any]:
        with self.connect() as db:
            rows = db.execute("SELECT status,COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
            candidates = db.execute("SELECT status,COUNT(*) AS n FROM evolution_candidates GROUP BY status").fetchall()
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        with self.connect() as db:
            event_count = db.execute("SELECT COUNT(*) FROM control_events").fetchone()[0]
        return {"db": str(self.db_path), "integrity_check": integrity,
                "jobs": {r["status"]: r["n"] for r in rows},
                "evolution": {r["status"]: r["n"] for r in candidates},
                "control_events": event_count}

    def capability_report(self) -> dict[str, Any]:
        """Report real local capability prerequisites without exposing values."""
        env_requirements = {
            "github": ["GH_TOKEN", "GITHUB_TOKEN"],
            "telegram": ["TELEGRAM_BOT_TOKEN"],
            "discord": ["DISCORD_BOT_TOKEN"],
            "slack": ["SLACK_BOT_TOKEN"],
            "whatsapp": ["WHATSAPP_TOKEN"],
        }
        external = {}
        for name, keys in env_requirements.items():
            if name == "github":
                ready = _command_ok(["gh", "auth", "status"]) or _command_ok(
                    ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", "gh auth status"]
                )
            else:
                ready = any(bool(os.environ.get(key)) for key in keys)
            external[name] = {"ready": ready, "required_env": keys}
        local = {
            "python": True,
            "native_memory_store": Path(r"D:\hermes\codex_memory_store").exists(),
            "obsidian_vault": Path(r"D:\hermes\hermes_obsidian_vault").exists(),
            "codegraph_mcp_transport": bool(shutil.which("wsl")) or bool(shutil.which("codegraph")),
            "windows_scheduler": bool(shutil.which("schtasks")),
            "node": bool(shutil.which("node")),
        }
        optional = {
            "docker": bool(shutil.which("docker")) or _command_ok(
                ["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", "docker info >/dev/null 2>&1"]
            )
        }
        return {"status": "PASS" if all(local.values()) else "BLOCKED", "local": local,
                "optional_backends": optional,
                "external": external, "policy": "missing credentials are blocked, never simulated"}

    def github_snapshot(self) -> dict[str, Any]:
        """Capture a read-only GitHub repository snapshot through authenticated gh."""
        owner = os.environ.get("CODEX_GITHUB_OWNER", "Yhvoid0101").strip()
        repo = os.environ.get("CODEX_GITHUB_REPO", "-").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
            raise ValueError("invalid GitHub owner/repository name")
        repository = json.loads(_run_checked(["gh", "api", f"repos/{owner}/{repo}"], timeout=30))
        issues = json.loads(_run_checked(["gh", "issue", "list", "--repo", f"{owner}/{repo}",
                                          "--state", "open", "--limit", "50", "--json",
                                          "number,title,url,updatedAt"], timeout=30) or "[]")
        pull_requests = json.loads(_run_checked(["gh", "pr", "list", "--repo", f"{owner}/{repo}",
                                                 "--state", "open", "--limit", "50", "--json",
                                                 "number,title,url,updatedAt"], timeout=30) or "[]")
        report = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "repository": {"full_name": repository.get("full_name"), "private": repository.get("private"),
                            "default_branch": (repository.get("default_branch") or "")},
            "issues": issues,
            "pull_requests": pull_requests,
        }
        target = self.db_path.parent / "github" / "last_snapshot.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "PASS", "path": str(target), "issues": len(issues), "pull_requests": len(pull_requests)}

    def builtin_worker_once(self) -> dict[str, Any]:
        """Run one fixed, low-risk maintenance cycle.

        The payload is intentionally ignored for command selection.  This
        prevents a durable queue entry from becoming an arbitrary shell.
        """
        bucket = int(time.time() // 600)
        self.enqueue("memory-check", {"bucket": bucket}, idempotency_key=f"memory-check:{bucket}", max_attempts=2)
        self.enqueue("github-check", {"bucket": bucket}, idempotency_key=f"github-check:{bucket}", max_attempts=2)
        self.enqueue("evolution-check", {"bucket": bucket}, idempotency_key=f"evolution-check:{bucket}", max_attempts=2)

        def execute(job: Job) -> None:
            if job.kind == "memory-check":
                script = Path(os.environ.get("CODEX_MEMORY_AUTOMATION", r"D:\hermes\codex_memory_automation.ps1"))
                if not script.exists():
                    raise FileNotFoundError(script)
                completed = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Mode", "Check"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError((completed.stderr or completed.stdout or "memory check failed")[-2000:])
            elif job.kind == "github-check":
                self.github_snapshot()
            elif job.kind == "evolution-check":
                self.evolution_cycle()
            else:
                raise RuntimeError(f"unsupported builtin job kind: {job.kind}")

        return self.run_once(execute)

    def evolution_cycle(self, *, min_occurrences: int = 2) -> dict[str, Any]:
        """Turn repeated failures into evaluated, versioned, rollback-capable policies."""
        cycle_key = f"evolution:{int(time.time() // 600)}"
        previous = self._get_checkpoint(cycle_key)
        resumed_from = previous["phase"] if previous and previous["status"] != "completed" else ""
        self._checkpoint(cycle_key, "scan", "running", {"min_occurrences": min_occurrences})
        try:
            with self.connect() as db:
                events = db.execute(
                    "SELECT id,payload_json FROM control_events WHERE event_type='job_failed' ORDER BY created_at"
                ).fetchall()
            groups: dict[tuple[str, str], list[str]] = {}
            for event in events:
                try:
                    payload = json.loads(event["payload_json"])
                except json.JSONDecodeError:
                    continue
                kind = str(payload.get("kind", "unknown"))
                error = str(payload.get("error", "unknown"))
                signature = re.sub(r"[0-9a-f]{8,}", "<id>", error, flags=re.IGNORECASE)
                groups.setdefault((kind, signature), []).append(event["id"])
            candidates = []
            self._checkpoint(cycle_key, "evaluate", "running", {"patterns": len(groups)})
            for (kind, signature), event_ids in groups.items():
                if len(event_ids) < min_occurrences:
                    continue
                candidate = self.propose(
                    "workflow",
                    f"repeated failure: {kind}",
                    {"job_kind": kind, "failure_signature": signature,
                     "recommended_action": "increase bounded retry delay for this repeated failure",
                     "retry_delay_multiplier": 2.0},
                    event_ids[-20:], risk="low", confidence=min(0.99, 0.85 + 0.03 * (len(event_ids) - 2)),
                )
                candidate_id = candidate["id"]
                verification = self.verify_candidate(
                    candidate_id,
                    lambda proposal, ids=event_ids: proposal.get("job_kind") == kind
                    and proposal.get("failure_signature") == signature
                    and len(ids) >= min_occurrences,
                )
                evaluation = self.evaluate_candidate(candidate_id)
                if evaluation["status"] != "passed":
                    self._checkpoint(cycle_key, "evaluate", "failed", {"candidate_id": candidate_id,
                                                                          "evaluation": evaluation})
                    continue
                promoted = self.promote_candidate(candidate_id)
                policy = self.activate_policy(candidate_id, evaluation)
                candidates.append({"id": candidate_id, "status": promoted["status"], "verified": verification["status"],
                                   "job_kind": kind, "occurrences": len(event_ids), "automatic": True,
                                   "policy_id": policy["id"], "policy_version": policy["version"]})
            result = {"status": "PASS", "patterns": len(candidates), "candidates": candidates,
                      "resumed_from": resumed_from,
                      "checkpoint": self._checkpoint(cycle_key, "complete", "completed", {"patterns": len(candidates)})}
            return result
        except Exception as exc:
            self._checkpoint(cycle_key, "failed", "failed", {"error": str(exc)[:1500]})
            raise

    def evaluate_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT proposal_json,status FROM evolution_candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        proposal = json.loads(row["proposal_json"])
        multiplier = proposal.get("retry_delay_multiplier")
        passed = (row["status"] == "verified" and proposal.get("job_kind") and proposal.get("failure_signature")
                  and isinstance(multiplier, (int, float)) and 1.0 <= float(multiplier) <= 4.0)
        result = {"status": "passed" if passed else "failed", "baseline_delay": 5.0,
                  "candidate_delay": 5.0 * float(multiplier or 0), "regression": not passed}
        with self.connect() as db:
            db.execute("INSERT INTO control_events(id,event_type,payload_json,created_at) VALUES (?,?,?,?)",
                       (str(uuid.uuid4()), "policy_evaluated", json.dumps({"candidate_id": candidate_id, **result}), time.time()))
        return result

    def activate_policy(self, candidate_id: str, evaluation: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as db:
            candidate = db.execute("SELECT proposal_json FROM evolution_candidates WHERE id=?", (candidate_id,)).fetchone()
            if candidate is None:
                raise KeyError(candidate_id)
            proposal = json.loads(candidate["proposal_json"])
            job_kind, signature = proposal["job_kind"], proposal["failure_signature"]
            parent = db.execute("""SELECT * FROM evolution_policies WHERE job_kind=? AND failure_signature=?
                                  AND status='active' ORDER BY version DESC LIMIT 1""", (job_kind, signature)).fetchone()
            version = int(db.execute("SELECT COALESCE(MAX(version),0)+1 FROM evolution_policies WHERE job_kind=? AND failure_signature=?",
                                     (job_kind, signature)).fetchone()[0])
            policy_id = str(uuid.uuid4())
            now = time.time()
            if parent:
                db.execute("UPDATE evolution_policies SET status='retired',retired_at=? WHERE id=?", (now, parent["id"]))
            db.execute("""INSERT INTO evolution_policies
                (id,candidate_id,job_kind,failure_signature,version,parent_id,status,proposal_json,evaluation_json,created_at,activated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                       (policy_id, candidate_id, job_kind, signature, version, parent["id"] if parent else "", "active",
                        json.dumps(proposal, ensure_ascii=False), json.dumps(evaluation, ensure_ascii=False), now, now))
        return {"id": policy_id, "version": version, "parent_id": parent["id"] if parent else "", "status": "active"}

    def rollback_policy(self, policy_id: str, reason: str) -> dict[str, Any]:
        with self.connect() as db:
            policy = db.execute("SELECT parent_id,status FROM evolution_policies WHERE id=?", (policy_id,)).fetchone()
            if policy is None:
                raise KeyError(policy_id)
            if policy["status"] != "active":
                return {"id": policy_id, "status": policy["status"]}
            db.execute("UPDATE evolution_policies SET status='rolled_back',retired_at=? WHERE id=?", (time.time(), policy_id))
            if policy["parent_id"]:
                db.execute("UPDATE evolution_policies SET status='active',activated_at=? WHERE id=?", (time.time(), policy["parent_id"]))
            db.execute("INSERT INTO control_events(id,event_type,payload_json,created_at) VALUES (?,?,?,?)",
                       (str(uuid.uuid4()), "policy_rollback", json.dumps({"policy_id": policy_id, "reason": reason[:500]}), time.time()))
        return {"id": policy_id, "status": "rolled_back", "restored_parent": policy["parent_id"]}

    def propose(self, kind: str, title: str, proposal: dict[str, Any], evidence: list[str],
                *, risk: str = "medium", confidence: float = 0.8) -> dict[str, Any]:
        if kind not in {"memory", "skill", "workflow"} or risk not in {"low", "medium", "high"}:
            raise ValueError("invalid candidate kind or risk")
        candidate_id = hashlib.sha256((kind + "\0" + title + "\0" + json.dumps(proposal, sort_keys=True)).encode()).hexdigest()[:24]
        now = time.time()
        with self.connect() as db:
            db.execute("""INSERT OR IGNORE INTO evolution_candidates
              (id,kind,title,proposal_json,evidence_json,risk,confidence,status,created_at)
              VALUES (?,?,?,?,?,?,?,'proposed',?)""",
              (candidate_id, kind, title, json.dumps(proposal, ensure_ascii=False),
               json.dumps(evidence, ensure_ascii=False), risk, float(confidence), now))
            row = db.execute("SELECT * FROM evolution_candidates WHERE id=?", (candidate_id,)).fetchone()
        return dict(row)

    def verify_candidate(self, candidate_id: str, verifier: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM evolution_candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        proposal = json.loads(row["proposal_json"])
        passed = bool(verifier(proposal))
        now = time.time()
        with self.connect() as db:
            db.execute("UPDATE evolution_candidates SET status=?, verified_at=? WHERE id=?",
                       ("verified" if passed else "rejected", now if passed else None, candidate_id))
        return {"id": candidate_id, "status": "verified" if passed else "rejected"}

    def promote_candidate(self, candidate_id: str) -> dict[str, Any]:
        now = time.time()
        with self.connect() as db:
            row = db.execute("SELECT status,risk,confidence FROM evolution_candidates WHERE id=?", (candidate_id,)).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            if row["status"] != "verified":
                raise RuntimeError("candidate must pass verification before promotion")
            if row["risk"] == "high" or row["confidence"] < 0.85:
                raise RuntimeError("high-risk or low-confidence candidate requires explicit review")
            db.execute("UPDATE evolution_candidates SET status='promoted', promoted_at=? WHERE id=?", (now, candidate_id))
        return {"id": candidate_id, "status": "promoted"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Codex durable autonomy control plane")
    parser.add_argument("command", choices=["status", "recover", "worker", "github-check", "evolution-check", "policy-rollback", "policy-outcome", "capabilities"])
    parser.add_argument("--db", type=Path, default=Path(r"D:\hermes\codex_memory_store\codex_autonomy.db"))
    parser.add_argument("--policy-id", default="")
    parser.add_argument("--reason", default="operator-requested rollback")
    parser.add_argument("--passed", action="store_true")
    args = parser.parse_args()
    store = AutonomyStore(args.db)
    if args.command == "status":
        result = store.status()
    elif args.command == "recover":
        result = store.recover_orphans()
    elif args.command == "worker":
        result = store.builtin_worker_once()
    elif args.command == "github-check":
        result = store.github_snapshot()
    elif args.command == "evolution-check":
        result = store.evolution_cycle()
    elif args.command == "policy-rollback":
        if not args.policy_id:
            parser.error("--policy-id is required for policy-rollback")
        result = store.rollback_policy(args.policy_id, args.reason)
    elif args.command == "policy-outcome":
        if not args.policy_id:
            parser.error("--policy-id is required for policy-outcome")
        result = store.record_policy_outcome(args.policy_id, args.passed, {"source": "cli"})
    else:
        result = store.capability_report()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
