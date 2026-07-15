import importlib.util
import sys
from pathlib import Path


MODULE = Path(__file__).parents[1] / "src" / "codex_autonomy.py"
spec = importlib.util.spec_from_file_location("codex_autonomy_capabilities", MODULE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_capability_report_redacts_and_blocks_missing_external_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(module, "_command_ok", lambda command, timeout=10: False)
    report = module.AutonomyStore(tmp_path / "state.db").capability_report()
    assert report["external"]["github"]["ready"] is False
    assert report["policy"] == "missing credentials are blocked, never simulated"
    assert "value" not in report["external"]["github"]


def test_wsl_docker_is_detected(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "_command_ok", lambda command, timeout=10: "docker info" in " ".join(command))
    report = module.AutonomyStore(tmp_path / "state.db").capability_report()
    assert report["optional_backends"]["docker"] is True
