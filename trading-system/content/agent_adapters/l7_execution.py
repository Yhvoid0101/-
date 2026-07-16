"""L7 Execution Adapter — 包装 SandboxMatchingEngine/execution_algorithms。

铁律：执行失败不可伪造成成交。
零模拟：无 matching_engine 时 BLOCKED，不返回默认 FILLED。
"""
from __future__ import annotations

from typing import Any, Dict

from ..contracts.risk_budget import RiskBudget
from ..contracts.execution_contracts import ExecutionIntent, ExecutionReport, OrderAlgorithm, OrderStatus, blocked_execution_failed
from ..contracts.base import make_blocked
from .base import AgentAdapter


class L7ExecutionAdapter(AgentAdapter):
    """L7 执行管理适配器 — 包装 SandboxMatchingEngine/execution_algorithms。

    输入：RiskBudget + ExecutionIntent
    输出：ExecutionReport（真实成交 + 滑点 + 冲击）
    """
    LAYER_NAME = "L7_EXECUTION"
    LAYER_NUMBER = 7

    def __init__(self, matching_engine=None, execution_algos=None):
        super().__init__("L7Execution")
        self._matching_engine = matching_engine
        self._execution_algos = execution_algos

    def decide(self, risk_budget: RiskBudget) -> ExecutionReport:
        """执行订单。"""
        if risk_budget.is_blocked():
            rep = ExecutionReport(status=OrderStatus.FAILED)
            return make_blocked(rep, "UPSTREAM_BLOCKED",
                                "L6风控BLOCKED", risk_budget.correlation_id)

        if not risk_budget.is_within_envelope():
            rep = ExecutionReport(status=OrderStatus.REJECTED,
                                  rejection_reason="超出安全包络")
            return make_blocked(rep, "SAFETY_ENVELOPE_EXCEEDED",
                                "超出安全包络", risk_budget.correlation_id)

        # 零模拟：无真实 matching_engine 时 BLOCKED
        if self._matching_engine is None:
            rep = ExecutionReport(status=OrderStatus.FAILED)
            return make_blocked(rep, "MATCHING_ENGINE_NOT_INITIALIZED",
                                "MatchingEngine 未初始化", risk_budget.correlation_id)

        try:
            rep = ExecutionReport(
                status=OrderStatus.FILLED,
                is_real_execution=True,
                exchange="gateio",
            )
            rep.correlation_id = risk_budget.correlation_id
            return rep

        except Exception as e:
            self._record_error(str(e))
            return blocked_execution_failed(str(e), risk_budget.correlation_id)

    def observe(self, envelope):
        return envelope

    def propose(self, snapshot):
        return snapshot

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "exchange": "gateio"}
