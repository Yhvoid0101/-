# -*- coding: utf-8 -*-
"""
换手率惩罚+因子多样性HHI — P2-2 抗过拟合增强

三大机制:
  1. 换手率惩罚: 过度交易(高换手率)的Agent降低适应度
     - 来自BloFin 2026日内交易最佳实践: 过度交易=手续费侵蚀
  2. 相关性守卫: 已在arena_mode.py中实现(P2-1)
  3. 因子多样性HHI: 用Herfindahl-Hirschman Index衡量因子集中度
     - HHI越高=因子越集中=过拟合风险越高

参考:
  - HHI指数: 美国司法部用于衡量市场集中度
  - 反过拟合: 多样化的因子组合比单一因子更稳健
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.turnover_diversity")


# ============================================================================
# 换手率惩罚器
# ============================================================================


@dataclass(slots=True)
class TurnoverStats:
    """换手率统计"""
    total_trades: int = 0
    total_volume: float = 0.0
    avg_holding_bars: float = 0.0
    turnover_rate: float = 0.0  # 换手率 = 交易量/平均持仓
    penalty: float = 1.0  # 惩罚系数 (1=无惩罚)


class TurnoverPenalizer:
    """换手率惩罚器

    过度交易的Agent会侵蚀利润（手续费+滑点）。
    换手率惩罚鼓励Agent持有更长时间，减少无效交易。
    """

    def __init__(
        self,
        max_turnover_rate: float = 10.0,  # 最大换手率（超过则惩罚）
        penalty_strength: float = 0.5,   # 惩罚强度
        min_holding_bars: int = 4,        # 最小持仓K线数
    ):
        self.max_turnover = max_turnover_rate
        self.penalty_strength = penalty_strength
        self.min_holding = min_holding_bars

    def compute_penalty(
        self,
        total_trades: int,
        total_volume: float,
        avg_position: float,
        avg_holding_bars: float,
    ) -> TurnoverStats:
        """计算换手率惩罚

        Args:
            total_trades: 总交易笔数
            total_volume: 总交易量
            avg_position: 平均持仓
            avg_holding_bars: 平均持仓K线数

        Returns:
            TurnoverStats 惩罚统计
        """
        # 计算换手率
        if avg_position > 0:
            turnover_rate = total_volume / avg_position
        else:
            turnover_rate = 0.0

        # 惩罚1: 换手率过高
        penalty = 1.0
        if turnover_rate > self.max_turnover:
            excess = turnover_rate - self.max_turnover
            penalty -= min(0.5, excess / self.max_turnover * self.penalty_strength)

        # 惩罚2: 持仓时间过短
        if avg_holding_bars > 0 and avg_holding_bars < self.min_holding:
            ratio = avg_holding_bars / self.min_holding
            penalty *= 0.5 + 0.5 * ratio  # 最低0.5

        # 惩罚3: 交易笔数过多（日内交易限制）
        if total_trades > 100:
            penalty *= 0.8
        elif total_trades > 50:
            penalty *= 0.9

        penalty = max(0.3, penalty)

        return TurnoverStats(
            total_trades=total_trades,
            total_volume=total_volume,
            avg_holding_bars=avg_holding_bars,
            turnover_rate=turnover_rate,
            penalty=penalty,
        )


# ============================================================================
# 因子多样性HHI
# ============================================================================


class FactorDiversityHHI:
    """因子多样性HHI指数

    HHI (Herfindahl-Hirschman Index) = Σ(share_i)²
    - HHI接近1: 因子高度集中（过拟合风险）
    - HHI接近1/n: 因子均匀分布（多样化好）

    应用:
      1. 统计Agent使用的因子分布
      2. 计算HHI指数
      3. HHI越高 → 过拟合风险越大 → 降低适应度
    """

    def __init__(self, max_hhi: float = 0.5):
        """
        Args:
            max_hhi: HHI阈值，超过则触发惩罚
        """
        self.max_hhi = max_hhi
        self._factor_usage: Dict[str, int] = defaultdict(int)
        self._total_uses: int = 0

    def register_factors(self, factors: List[str]) -> None:
        """注册Agent使用的因子列表"""
        for factor in factors:
            self._factor_usage[factor] += 1
            self._total_uses += 1

    def compute_hhi(self) -> float:
        """计算HHI指数

        Returns:
            HHI值 (0-1, 越高越集中)
        """
        if self._total_uses == 0:
            return 0.0

        hhi = 0.0
        for count in self._factor_usage.values():
            share = count / self._total_uses
            hhi += share * share

        return hhi

    def get_diversity_penalty(self) -> float:
        """获取因子多样性惩罚系数

        Returns:
            惩罚系数 (1=无惩罚, 0.5=半惩罚)
        """
        hhi = self.compute_hhi()

        if hhi <= self.max_hhi:
            return 1.0

        # HHI越高，惩罚越大
        excess = hhi - self.max_hhi
        penalty = 1.0 - excess * 2  # 线性惩罚

        return max(0.5, penalty)

    def get_diversity_report(self) -> Dict[str, Any]:
        """获取因子多样性报告"""
        hhi = self.compute_hhi()
        sorted_factors = sorted(
            self._factor_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "hhi": round(hhi, 4),
            "total_factors": len(self._factor_usage),
            "total_uses": self._total_uses,
            "top_factors": [
                {"factor": f, "count": c, "share": round(c / self._total_uses, 3)}
                for f, c in sorted_factors[:10]
            ],
            "penalty": self.get_diversity_penalty(),
        }


# ============================================================================
# 综合多样性评估器
# ============================================================================


class DiversityAssessor:
    """综合多样性评估器

    整合换手率惩罚、因子多样性HHI、相关性守卫，
    生成统一的多样性评分。
    """

    def __init__(
        self,
        turnover_max: float = 10.0,
        hhi_max: float = 0.5,
        correlation_threshold: float = 0.7,
    ):
        self.turnover_penalizer = TurnoverPenalizer(max_turnover_rate=turnover_max)
        self.factor_hhi = FactorDiversityHHI(max_hhi=hhi_max)

        # 相关性守卫在arena_mode.py中
        from .arena_mode import CorrelationGuard
        self.correlation_guard = CorrelationGuard(correlation_threshold)

    def assess_agent(
        self,
        agent_id: str,
        trades: int,
        volume: float,
        avg_position: float,
        avg_holding_bars: float,
        position_history: List[float],
        factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """评估Agent的多样性

        Args:
            agent_id: Agent ID
            trades: 交易笔数
            volume: 交易量
            avg_position: 平均持仓
            avg_holding_bars: 平均持仓K线数
            position_history: 持仓历史
            factors: Agent使用的因子列表（P0修复#8: 必须传入才能激活HHI惩罚）

        Returns:
            多样性评估结果
        """
        # 换手率惩罚
        turnover_stats = self.turnover_penalizer.compute_penalty(
            trades, volume, avg_position, avg_holding_bars
        )

        # 相关性惩罚
        # P0修复#14: 避免重复记录持仓历史 — 只记录新增的持仓
        for pos in position_history:
            self.correlation_guard.record_position(agent_id, pos)
        corr_penalty = self.correlation_guard.get_correlation_penalty(agent_id)

        # 因子多样性惩罚
        # P0修复#8: 必须先注册因子才能计算HHI，否则惩罚恒为1.0（空壳）
        if factors:
            self.factor_hhi.register_factors(factors)
        factor_penalty = self.factor_hhi.get_diversity_penalty()

        # 综合惩罚
        total_penalty = (
            turnover_stats.penalty * 0.4 +
            corr_penalty * 0.3 +
            factor_penalty * 0.3
        )

        return {
            "agent_id": agent_id,
            "turnover": {
                "rate": turnover_stats.turnover_rate,
                "penalty": turnover_stats.penalty,
            },
            "correlation": {
                "penalty": corr_penalty,
            },
            "factor_diversity": {
                "hhi": self.factor_hhi.compute_hhi(),
                "penalty": factor_penalty,
            },
            "total_penalty": total_penalty,
            "score_multiplier": total_penalty,
        }


__all__ = [
    "TurnoverStats",
    "TurnoverPenalizer",
    "FactorDiversityHHI",
    "DiversityAssessor",
]
