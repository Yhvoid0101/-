"""MarketStateSnapshot — 第2层市场理解层输出契约。

铁律：未知状态进入第5层审核，而非强行归类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .base import ContractBase, ContractStatus


class MarketRegime(str, Enum):
    """市场状态分类。"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeEvidence:
    """Regime 判定证据。"""
    label: str = ""
    confidence: float = 0.0
    ood_distance: float = 0.0
    change_points: List[float] = field(default_factory=list)
    supporting_features: Dict[str, float] = field(default_factory=dict)
    duration_bars: int = 0
    cross_symbol_consistency: float = 1.0


@dataclass
class MarketStateSnapshot(ContractBase):
    """第2层输出：市场状态快照。"""
    symbol: str = ""
    timeframe: str = ""
    regime: MarketRegime = MarketRegime.UNKNOWN
    evidence: RegimeEvidence = field(default_factory=RegimeEvidence)
    is_ood: bool = False
    uncertainty: float = 0.0
    is_stable: bool = False

    def __post_init__(self):
        if self.is_ood:
            self.regime = MarketRegime.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["regime"] = self.regime.value
        return d
