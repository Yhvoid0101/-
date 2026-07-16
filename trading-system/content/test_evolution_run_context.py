from pathlib import Path


def test_evolution_loop_assigns_a_run_id_to_the_audit_logger():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")

    assert "import uuid" in source
    assert "self.run_id" in source
    assert "context={\"run_id\": self.run_id}" in source