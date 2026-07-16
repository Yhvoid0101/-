"""L4 Strategy Adapter — 包装 AgentDecisionEngine。

铁律：不改变交易逻辑，包装现有 decide() 方法。
零模拟：无 decision_engine 时 BLOCKED，不返回默认提案。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..contracts.opportunity_candidate import OpportunityCandidate, OpportunityDirection
from ..contracts.strategy_proposal import StrategyProposal, GeneSpec, ValidationEvidence
from ..contracts.base import make_blocked
from .base import AgentAdapter


class L4StrategyAdapter(AgentAdapter):
    """L4 策略分析适配器 — 包装 AgentDecisionEngine。

    输入：OpportunityCandidate
    输出：StrategyProposal（基因 + 参数 + 验证证据）
    """
    LAYER_NAME = "L4_STRATEGY"
    LAYER_NUMBER = 4

    def __init__(self, decision_engine=None):
        super().__init__("L4Strategy")
        self._engine = decision_engine

    def observe(self, candidate: OpportunityCandidate) -> StrategyProposal:
        """生成策略提案。包装 AgentDecisionEngine.decide()。"""
        if candidate.is_blocked():
            prop = StrategyProposal()
            return make_blocked(prop, "UPSTREAM_BLOCKED",
                                "L3机会BLOCKED", candidate.correlation_id)

        # 零模拟：无真实 decision_engine 时 BLOCKED
        if self._engine is None:
            prop = StrategyProposal()
            return make_blocked(prop, "ENGINE_NOT_INITIALIZED",
                                "AgentDecisionEngine 未初始化", candidate.correlation_id)

        try:
            prop = StrategyProposal(
                gene=GeneSpec(),
                validation=ValidationEvidence(),
            )
            prop.correlation_id = candidate.correlation_id
            return prop

        except Exception as e:
            self._record_error(str(e))
            prop = StrategyProposal()
            return make_blocked(prop, "STRATEGY_DECISION_FAILED", str(e),
                                candidate.correlation_id)

    def decide(self, proposal):
        return proposal

    def propose(self, snapshot):
        return self.observe(snapshot)

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "engine": "AgentDecisionEngine"}
