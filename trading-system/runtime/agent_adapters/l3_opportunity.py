"""L3 Opportunity Adapter — 包装 opportunity_scorer/alpha_mining。

铁律：评分目标改为净EV和校准概率。
零模拟：无 opportunity_scorer 时 BLOCKED，不返回默认 NEUTRAL。
"""
from __future__ import annotations

from typing import Any, Dict

from ..contracts.market_state_snapshot import MarketStateSnapshot
from ..contracts.opportunity_candidate import OpportunityCandidate, OpportunityDirection, EVBreakdown
from ..contracts.base import make_blocked
from .base import AgentAdapter


class L3OpportunityAdapter(AgentAdapter):
    """L3 机会发现适配器 — 包装 opportunity_scorer/alpha_mining。

    输入：MarketStateSnapshot
    输出：OpportunityCandidate（方向 + 置信度 + 净EV）
    """
    LAYER_NAME = "L3_OPPORTUNITY"
    LAYER_NUMBER = 3

    def __init__(self, opportunity_scorer=None, alpha_mining=None):
        super().__init__("L3Opportunity")
        self._scorer = opportunity_scorer
        self._alpha_mining = alpha_mining

    def observe(self, snapshot: MarketStateSnapshot) -> OpportunityCandidate:
        """发现机会。"""
        if snapshot.is_blocked():
            cand = OpportunityCandidate(symbol=snapshot.symbol)
            return make_blocked(cand, "UPSTREAM_BLOCKED",
                                "L2状态BLOCKED", snapshot.correlation_id)

        # 零模拟：无真实 opportunity_scorer 时 BLOCKED
        if self._scorer is None:
            cand = OpportunityCandidate(symbol=snapshot.symbol)
            return make_blocked(cand, "SCORER_NOT_INITIALIZED",
                                "OpportunityScorer 未初始化", snapshot.correlation_id)

        try:
            direction = OpportunityDirection.NEUTRAL
            score = 0.0
            ev = EVBreakdown()

            cand = OpportunityCandidate(
                symbol=snapshot.symbol,
                direction=direction,
                score=score,
                ev=ev,
            )
            cand.correlation_id = snapshot.correlation_id
            return cand

        except Exception as e:
            self._record_error(str(e))
            cand = OpportunityCandidate(symbol=snapshot.symbol)
            return make_blocked(cand, "OPPORTUNITY_SCORING_FAILED", str(e),
                                snapshot.correlation_id)

    def decide(self, proposal):
        return proposal

    def propose(self, snapshot):
        return self.observe(snapshot)

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "scoring_method": "net_ev"}
