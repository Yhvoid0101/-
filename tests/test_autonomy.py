import importlib.util
import sys
import sqlite3
import time
from pathlib import Path


MODULE = Path(__file__).parents[1] / "src" / "codex_autonomy.py"
spec = importlib.util.spec_from_file_location("codex_autonomy", MODULE)
autonomy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = autonomy
spec.loader.exec_module(autonomy)


def test_idempotent_queue_and_success(tmp_path):
    store = autonomy.AutonomyStore(tmp_path / "state.db")
    first = store.enqueue("sync", {"scope": "demo"}, idempotency_key="same")
    replay = store.enqueue("sync", {"scope": "changed"}, idempotency_key="same")
    assert first["id"] == replay["id"]
    assert store.run_once(lambda job: None)["status"] == "succeeded"
    assert store.status()["jobs"] == {"succeeded": 1}


def test_failure_backoff_and_dead_letter(tmp_path):
    store = autonomy.AutonomyStore(tmp_path / "state.db")
    store.enqueue("sync", {}, idempotency_key="dead", max_attempts=2)
    assert store.run_once(lambda job: (_ for _ in ()).throw(RuntimeError("first")))["status"] == "pending"
    with store.connect() as db:
        db.execute("UPDATE jobs SET next_run_at=0")
    result = store.run_once(lambda job: (_ for _ in ()).throw(RuntimeError("second")))
    assert result["status"] == "dead"
    assert store.status()["jobs"] == {"dead": 1}


def test_restart_recovery_requeues_stale_running(tmp_path):
    store = autonomy.AutonomyStore(tmp_path / "state.db")
    row = store.enqueue("sync", {}, idempotency_key="recover")
    claimed = store.claim()
    assert claimed and claimed.id == row["id"]
    with store.connect() as db:
        db.execute("UPDATE jobs SET claimed_at=?", (time.time() - 9999,))
    assert store.recover_orphans(stale_after=300)["recovered"] == 1
    assert store.run_once(lambda job: None)["status"] == "succeeded"


def test_evolution_requires_evidence_and_verification(tmp_path):
    store = autonomy.AutonomyStore(tmp_path / "state.db")
    candidate = store.propose("skill", "bounded retry", {"rule": "retry twice"}, ["test passed"], confidence=0.9)
    try:
        store.promote_candidate(candidate["id"])
        assert False, "unverified candidate was promoted"
    except RuntimeError as exc:
        assert "verification" in str(exc)
    assert store.verify_candidate(candidate["id"], lambda proposal: proposal["rule"] == "retry twice")["status"] == "verified"
    assert store.promote_candidate(candidate["id"])["status"] == "promoted"


def test_high_risk_candidate_stays_blocked(tmp_path):
    store = autonomy.AutonomyStore(tmp_path / "state.db")
    candidate = store.propose("workflow", "deploy", {"action": "deploy"}, ["manual review"], risk="high", confidence=0.99)
    store.verify_candidate(candidate["id"], lambda _: True)
    try:
        store.promote_candidate(candidate["id"])
        assert False, "high-risk candidate was promoted"
    except RuntimeError as exc:
        assert "high-risk" in str(exc)
