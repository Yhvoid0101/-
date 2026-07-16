"""RiskBudget — 第6层风险控制层输出契约。

铁律：SL/TP、仓位、相关性上限、事件缩仓、KillSwitch均输出可解释RiskBudget。
     动态参数只能在不可变安全包络内优化。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import ContractBase, make_blocked


@dataclass
class SafetyEnvelope:
    """不可变安全包络 — 不可被学习器放宽。"""
    max_single_loss_pct: float = 0.02      # 单笔最大损失（% 仓位）
    max_daily_loss_pct: float = 0.06       # 日最大损失（% 仓位）
    max_portfolio_exposure: float = 1.0    # 组合最大敞口
    kill_switch_threshold: float = 0.10    # KillSwitch 阈值（% 总资金）


@dataclass
class RiskBudget(ContractBase):
    """第6层输出：风险预算。"""
    allowed_position_size: float = 0.0     # 允许仓位大小
    stop_loss: float = 0.0                 # 止损价
    take_profit: float = 0.0               # 止盈价
    max_loss_amount: float = 0.0           # 最大损失金额
    portfolio_correlation: float = 0.0     # 组合相关性
    event_constraint: str = ""             # 事件约束（如 "高影响事件缩仓"）
    bear_protection_active: bool = False   # 熊市保护激活
    kill_switch_active: bool = False       # KillSwitch 激活

    # 安全包络（不可变）
    safety: SafetyEnvelope = field(default_factory=SafetyEnvelope)

    # 可解释性
    reasoning: str = ""                    # 风控决策理由
    regime_adjustment: str = ""            # regime 调整说明

    def __post_init__(self):
        """KillSwitch 激活时阻断。"""
        if self.kill_switch_active:
            self.status = self.status  # KillSwitch 不改变 status，由执行层拒绝

    def is_within_envelope(self) -> bool:
        """检查是否在安全包络内。"""
        return (
            self.max_loss_amount <= self.safety.max_single_loss_pct
            and not self.kill_switch_active
        )


def blocked_kill_switch(correlation_id: str = "") -> RiskBudget:
    """KillSwitch 激活，阻断。"""
    rb = RiskBudget(kill_switch_active=True)
    return make_blocked(rb, "KILL_SWITCH",
                        "KillSwitch 激活，所有交易阻断", correlation_id)
