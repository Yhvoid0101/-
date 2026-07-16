from sandbox_trading.production_hardening import KillSwitch


def test_kill_switch_uses_production_drawdown_and_daily_loss_limits():
    assert KillSwitch.GLOBAL_DD_HARD_STOP == 0.12
    assert KillSwitch.AGENT_DD_HARD_STOP == 0.25
    assert KillSwitch.DAILY_LOSS_HARD_STOP == 0.05