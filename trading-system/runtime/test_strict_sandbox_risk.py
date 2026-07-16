from sandbox_trading.production_hardening import KillSwitch, KillSwitchState
from sandbox_trading.risk_control import CircuitState, RiskControlManager


def test_sandbox_kill_switch_flattens_and_locks_after_daily_loss():
    flattened = []
    kill_switch = KillSwitch(sandbox_mode=True)
    kill_switch.register_flatten_callback(lambda reason: flattened.append(reason) or 2)

    assert kill_switch.check_daily_loss(kill_switch.DAILY_LOSS_HARD_STOP)
    assert flattened == ["DAILY_LOSS_HARD_STOP"]
    assert kill_switch.state == KillSwitchState.FLATTENED


def test_sandbox_pre_trade_rejects_agent_drawdown_limit():
    risk = RiskControlManager(initial_capital=1000.0, sandbox_mode=True)
    profile = risk.get_agent_profile("agent-1")
    profile.current_drawdown = risk._config["l1_max_drawdown_pct"] + 0.01

    allowed, reason = risk.pre_trade_check("agent-1", "BTCUSDT", 1.0, 100.0)

    assert not allowed
    assert reason == "AGENT_DRAWDOWN_LIMIT"


def test_sandbox_pre_trade_rejects_global_daily_loss_limit():
    risk = RiskControlManager(initial_capital=1000.0, sandbox_mode=True)
    risk._global.daily_pnl = -(risk._config["l3_max_daily_loss"] + 1.0)

    allowed, reason = risk.pre_trade_check("agent-1", "BTCUSDT", 1.0, 100.0)

    assert not allowed
    assert reason == "DAILY_LOSS_LIMIT"
    assert risk._global.global_circuit == CircuitState.OPEN


def test_sandbox_pre_trade_rejects_consecutive_losses_with_production_cooldown():
    risk = RiskControlManager(initial_capital=1000.0, sandbox_mode=True)
    profile = risk.get_agent_profile("agent-1")
    profile.consecutive_losses = risk._config["l1_max_consecutive_losses"]

    allowed, reason = risk.pre_trade_check("agent-1", "BTCUSDT", 1.0, 100.0)

    assert not allowed
    assert reason == "AGENT_CONSECUTIVE_LOSSES"
    assert profile.circuit_cooldown_seconds >= 3600.0