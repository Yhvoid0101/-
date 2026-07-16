"""StrategyProposal — 第4层策略分析层输出契约。

铁律：基因、参数、适用regime、训练窗口、验证证据、版本。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .base import ContractBase, make_blocked
from .market_state_snapshot import MarketRegime


@dataclass
class GeneSpec:
    """基因规格。"""
    worldview_primary: str = ""
    worldview_secondary: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    risk_params: Dict[str, Any] = field(default_factory=dict)
    gene_hash: str = ""  # 基因哈希，用于版本追踪


@dataclass
class ValidationEvidence:
    """验证证据。"""
    train_window: str = ""            # 训练窗口
    validation_window: str = ""       # 验证窗口
    walk_forward_pass: bool = False   # walk-forward 通过
    purged_cv_pass: bool = False      # purged CV 通过
    pbo_score: float = 0.0           # PBO（过拟合概率）
    mc_pass: bool = False             # Monte Carlo 通过
    min_trades_met: bool = False      # 最小交易数满足
    cost_shock_pass: bool = False     # 成本冲击通过


@dataclass
class StrategyProposal(ContractBase):
    """第4层输出：策略提案。"""
    gene: GeneSpec = field(default_factory=GeneSpec)
    applicable_regimes: List[MarketRegime] = field(default_factory=list)
    validation: ValidationEvidence = field(default_factory=ValidationEvidence)
    expected_fitness: float = 0.0      # 预期适应度
    min_trades_required: int = 30      # 最小交易数要求
    is_eligible: bool = False          # 是否符合精英条件

    def __post_init__(self):
        self.is_eligible = (
            self.validation.min_trades_met
            and self.validation.walk_forward_pass
            and self.validation.pbo_score < 0.5
        )


def blocked_invalid_strategy(reason: str, correlation_id: str = "") -> StrategyProposal:
    prop = StrategyProposal()
    return make_blocked(prop, "INVALID_STRATEGY", reason, correlation_id)
