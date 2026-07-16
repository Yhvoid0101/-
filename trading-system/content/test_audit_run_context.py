import json

from sandbox_trading.production_hardening import AuditLogger


def test_audit_logger_merges_run_context_into_persisted_event(tmp_path):
    log_file = tmp_path / "audit.log"
    logger = AuditLogger(log_file=str(log_file), context={"run_id": "run-123"})

    logger.log_trade("agent-1", "open", {"symbol": "BTCUSDT"})

    event = json.loads(log_file.read_text(encoding="utf-8"))
    assert event["details"]["run_id"] == "run-123"