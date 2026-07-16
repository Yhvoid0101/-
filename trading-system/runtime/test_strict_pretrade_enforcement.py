from pathlib import Path


def test_evolution_loop_stops_after_every_pre_trade_rejection():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")
    start = source.index("if not can_trade:")
    end = source.index("# v3.38: symbol-level circuit breaker", start)
    rejection_block = source[start:end]

    assert "if reason in _GLOBAL_RISK_REASONS" not in rejection_block
    assert "continue" in rejection_block

def test_agent_coordinator_uses_package_relative_imports_in_runtime_loop():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")
    start = source.index("# v598 Phase B: AgentCoordinator 主循环真实调度")
    end = source.index("# v598 Phase 1: 复盘-经验库联动闭环", start)
    runtime_block = source[start:end]

    assert "from multi_agent_framework import" not in runtime_block
    assert runtime_block.count("from .multi_agent_framework import") == 3


def test_hard_risk_rejections_become_generation_lessons_in_both_evolution_paths():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")

    assert source.count("risk_hard_rejection") == 2
    assert source.count("risk_rejections_by_agent.items()") == 2
    assert source.count("hard_risk_rejection_count") == 2
