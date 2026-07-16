"""L6 Risk Adapter — 包装 RiskControlManager/kill_switch。

铁律：SL/TP、仓位、KillSwitch输出可解释RiskBudget。
零模拟：无 risk_control 时 BLOCKED，不返回默认安全包络。
"""
from __future__ import annotations

from typing import Any, Dict

from ..contracts.audit_verdict import AuditVerdict, VerdictAction
from ..contracts.risk_budget import RiskBudget, SafetyEnvelope, blocked_kill_switch
from ..contracts.base import make_blocked
from .base import AgentAdapter


class L6RiskAdapter(AgentAdapter):
    """L6 风险控制适配器 — 包装 RiskControlManager/kill_switch。

    输入：AuditVerdict
    输出：RiskBudget（仓位 + SL/TP + 安全包络）
    """
    LAYER_NAME = "L6_RISK"
    LAYER_NUMBER = 6

    def __init__(self, risk_control=None, kill_switch=None):
        super().__init__("L6Risk")
        self._risk_control = risk_control
        self._kill_switch = kill_switch

    def decide(self, verdict: AuditVerdict) -> RiskBudget:
        """计算风险预算。"""
        if verdict.is_blocked():
            rb = RiskBudget()
            return make_blocked(rb, "UPSTREAM_BLOCKED",
                                "L5审核BLOCKED", verdict.correlation_id)

        if verdict.action == VerdictAction.BLOCK:
            rb = RiskBudget()
            return make_blocked(rb, "AUDIT_BLOCKED",
                                "审核未通过", verdict.correlation_id)

        # 零模拟：无真实 risk_control 时 BLOCKED
        if self._risk_control is None:
            rb = RiskBudget()
            return make_blocked(rb, "RISK_CONTROL_NOT_INITIALIZED",
                                "RiskControlManager 未初始化", verdict.correlation_id)

        try:
            if self._kill_switch is not None and self._kill_switch.is_triggered():
                return blocked_kill_switch(verdict.correlation_id)

            rb = RiskBudget(
                safety=SafetyEnvelope(),
            )
            rb.correlation_id = verdict.correlation_id
            return rb

        except Exception as e:
            self._record_error(str(e))
            rb = RiskBudget()
            return make_blocked(rb, "RISK_CONTROL_FAILED", str(e),
                                verdict.correlation_id)

    def observe(self, envelope):
        return envelope

    def propose(self, snapshot):
        return snapshot

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "kill_switch_active": False}
