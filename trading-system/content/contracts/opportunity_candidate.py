"""OpportunityCandidate — 第3层机会发现层输出契约。

铁律：评分目标改为净EV和校准概率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .base import ContractBase, make_blocked


class OpportunityDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class EVBreakdown:
    """预期EV分解。"""
    gross_ev: float = 0.0
    fee_cost: float = 0.0
    slippage_cost: float = 0.0
    impact_cost: float = 0.0
    funding_cost: float = 0.0
    net_ev: float = 0.0

    def compute_net(self):
        self.net_ev = self.gross_ev - self.fee_cost - self.slippage_cost - self.impact_cost - self.funding_cost
        return self.net_ev


@dataclass
class OpportunityCandidate(ContractBase):
    """第3层输出：机会候选。"""
    symbol: str = ""
    direction: OpportunityDirection = OpportunityDirection.NEUTRAL
    confidence: float = 0.0
    calibrated_probability: float = 0.0
    ev: EVBreakdown = field(default_factory=EVBreakdown)
    assumptions: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    threshold: float = 70.0
    is_passed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.ev.compute_net()
        self.is_passed = self.score >= self.threshold and self.ev.net_ev > 0

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["direction"] = self.direction.value
        return d


def blocked_no_opportunity(symbol: str, reason: str, correlation_id: str = "") -> OpportunityCandidate:
    cand = OpportunityCandidate(symbol=symbol, rejection_reasons=[reason])
    return make_blocked(cand, "NO_OPPORTUNITY",
                        f"无有效机会: {symbol} - {reason}", correlation_id)
