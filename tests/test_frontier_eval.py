import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "tools" / "memory_frontier_eval.py"
spec = importlib.util.spec_from_file_location("memory_frontier_eval", MODULE)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)


def test_frontier_eval_passes_in_isolated_fixture(tmp_path):
    result = eval_module.run_evaluation(tmp_path)
    assert result["status"] == "PASS"
    assert result["retrieval"]["recall_at_5"] >= 0.9
    assert result["retrieval"]["mrr"] >= 0.75
    assert result["retrieval"]["scope_isolation"] is True
    assert result["pass_at_3"] is True
    assert result["lifecycle"]["audit_ok"] is True
    assert result["lifecycle"]["expiration"] is True
    assert result["backup"]["verified"] is True
    assert result["backup"]["restore_drill"] is True
