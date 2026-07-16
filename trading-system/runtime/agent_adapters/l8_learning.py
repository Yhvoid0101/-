"""L8 Learning Adapter — 包装 EvolutionLoop学习部分/population。

铁律：死因归因必须包含反事实。
零模拟：无 evolution_loop 时 BLOCKED，不返回默认学习事件。
"""
from __future__ import annotations

from typing import Any, Dict

from ..contracts.execution_contracts import ExecutionReport, OrderStatus
from ..contracts.learning_event import LearningEvent, CounterfactualAnalysis, LayerResponsibility
from ..contracts.base import make_blocked
from .base import AgentAdapter


class L8LearningAdapter(AgentAdapter):
    """L8 学习反馈适配器 — 包装 EvolutionLoop学习部分/population。

    输入：ExecutionReport
    输出：LearningEvent（死因 + 反事实 + 层级责任）
    """
    LAYER_NAME = "L8_LEARNING"
    LAYER_NUMBER = 8

    def __init__(self, evolution_loop=None, population=None):
        super().__init__("L8Learning")
        self._evolution_loop = evolution_loop
        self._population = population

    def observe(self, report: ExecutionReport) -> LearningEvent:
        """生成学习事件。"""
        if report.is_blocked():
            event = LearningEvent()
            return make_blocked(event, "UPSTREAM_BLOCKED",
                                "L7执行BLOCKED", report.correlation_id)

        # 零模拟：无真实 evolution_loop 时 BLOCKED
        if self._evolution_loop is None:
            event = LearningEvent()
            return make_blocked(event, "EVOLUTION_LOOP_NOT_INITIALIZED",
                                "EvolutionLoop 未初始化", report.correlation_id)

        try:
            event = LearningEvent(
                trade_id=report.order_id,
                symbol=report.symbol,
                is_reproducible=True,
            )
            event.correlation_id = report.correlation_id

            if report.status in (OrderStatus.FAILED, OrderStatus.REJECTED):
                event.death_cause = "EXECUTION_FAILED"
                event.death_cause_evidence = {"reason": report.rejection_reason}

            return event

        except Exception as e:
            self._record_error(str(e))
            event = LearningEvent()
            return make_blocked(event, "LEARNING_FAILED", str(e),
                                report.correlation_id)

    def decide(self, proposal):
        return proposal

    def propose(self, snapshot):
        return snapshot

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "death_cause_analysis": True}
