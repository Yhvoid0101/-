"""LearningEvent — 第8层学习反馈层输出契约。

铁律：死因归因必须包含反事实；结构建议只能提交 ChangeProposal。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ContractBase


@dataclass
class CounterfactualAnalysis:
    """反事实分析 — 若不翻转方向/不同RR/不同执行算法，结果是否改善。"""
    original_action: str = ""
    original_result: float = 0.0
    counterfactual_action: str = ""
    counterfactual_result: float = 0.0
    improvement: float = 0.0
    is_counterfactual_better: bool = False


@dataclass
class LayerResponsibility:
    """层级责任图 — 每笔交易的贡献分解。"""
    l1_data_quality: float = 0.0        # 数据层贡献
    l2_regime_accuracy: float = 0.0     # 理解层贡献
    l3_opportunity_score: float = 0.0   # 机会层贡献
    l4_strategy_fitness: float = 0.0    # 策略层贡献
    l5_audit_decision: float = 0.0      # 审核层贡献
    l6_risk_control: float = 0.0        # 风控层贡献
    l7_execution_quality: float = 0.0   # 执行层贡献
    total_attribution: float = 0.0      # 总归因（应=1.0）


@dataclass
class LearningEvent(ContractBase):
    """第8层输出：学习事件。"""
    trade_id: str = ""
    symbol: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_bars: int = 0

    # 死因归因（10种）
    death_cause: str = ""               # 死因（如 "方向翻转"、"RR不足"等）
    death_cause_evidence: Dict[str, Any] = field(default_factory=dict)

    # 反事实分析
    counterfactual: Optional[CounterfactualAnalysis] = None

    # 层级责任
    responsibility: LayerResponsibility = field(default_factory=LayerResponsibility)

    # 可复现证据
    replay_seed: str = ""               # 重放种子
    replay_data_hash: str = ""          # 重放数据哈希
    is_reproducible: bool = False       # 是否可复现

    def __post_init__(self):
        """验证归因完整性。"""
        total = (self.responsibility.l1_data_quality
                 + self.responsibility.l2_regime_accuracy
                 + self.responsibility.l3_opportunity_score
                 + self.responsibility.l4_strategy_fitness
                 + self.responsibility.l5_audit_decision
                 + self.responsibility.l6_risk_control
                 + self.responsibility.l7_execution_quality)
        self.responsibility.total_attribution = total
