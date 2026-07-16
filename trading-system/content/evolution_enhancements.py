# -*- coding: utf-8 -*-
"""
进化增强集成模块 — P1-2优化

将3个独立设计的增强模块集成到进化流程:
  1. Arena Mode (arena_mode.py) — 拥挤信号惩罚+相关性守卫
  2. 换手率惩罚 (turnover_diversity.py) — 过度交易Agent降低适应度
  3. 因子多样性HHI (turnover_diversity.py) — 因子集中度过拟合惩罚

设计原则: 不破坏现有进化流程,作为可选增强层叠加
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.evolution_enhancements")

# 尝试导入增强模块
try:
    from .arena_mode import ArenaMode
    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False

try:
    from .turnover_diversity import (
        FactorDiversityHHI,
        TurnoverPenalizer,
    )
    _TURNOVER_AVAILABLE = True
except ImportError:
    _TURNOVER_AVAILABLE = False


class EvolutionEnhancements:
    """进化增强集成器

    在进化评估阶段叠加3个增强:
      1. Arena Mode拥挤惩罚: 信号拥挤的Agent降低适应度
      2. 换手率惩罚: 过度交易的Agent降低适应度
      3. 因子多样性HHI: 因子集中度高的Agent降低适应度

    使用方式:
        enhancer = EvolutionEnhancements(enabled=True)
        # 在评估阶段调用
        penalized_scores = enhancer.apply_penalties(scores, agent_trades, agent_signals)
    """

    def __init__(
        self,
        enabled: bool = True,
        arena_enabled: bool = True,
        turnover_enabled: bool = True,
        hhi_enabled: bool = True,
        turnover_max: float = 10.0,
        hhi_max: float = 0.5,
    ):
        """
        Args:
            enabled: 总开关
            arena_enabled: Arena Mode拥挤惩罚开关
            turnover_enabled: 换手率惩罚开关
            hhi_enabled: 因子多样性HHI开关
            turnover_max: 最大换手率阈值
            hhi_max: HHI阈值
        """
        self.enabled = enabled
        self.arena_enabled = arena_enabled and _ARENA_AVAILABLE
        self.turnover_enabled = turnover_enabled and _TURNOVER_AVAILABLE
        self.hhi_enabled = hhi_enabled and _TURNOVER_AVAILABLE

        # 初始化各增强模块
        if self.arena_enabled:
            self.arena = ArenaMode()
            self.crowding_tracker = self.arena  # ArenaMode内部包含crowding_tracker
        else:
            self.arena = None
            self.crowding_tracker = None

        if self.turnover_enabled:
            self.turnover_penalizer = TurnoverPenalizer(max_turnover_rate=turnover_max)
        else:
            self.turnover_penalizer = None

        if self.hhi_enabled:
            self.factor_hhi = FactorDiversityHHI(max_hhi=hhi_max)
        else:
            self.factor_hhi = None

        # 统计
        self._stats = {
            "arena_penalties_applied": 0,
            "turnover_penalties_applied": 0,
            "hhi_penalties_applied": 0,
            "total_evaluations": 0,
        }

    def set_total_agents(self, total: int) -> None:
        """设置Agent总数(用于Arena Mode拥挤度计算)"""
        if self.arena:
            self.arena.update_agent_count(total)

    def register_agent_signal(
        self,
        agent_id: str,
        signals: set = None,
        direction: int = 0,
    ) -> None:
        """注册Agent信号(用于Arena Mode拥挤度跟踪)"""
        if self.arena_enabled and self.arena:
            sig_set = signals if signals else set()
            self.arena.register_agent(agent_id, sig_set)

    def register_factors(self, factors: List[str]) -> None:
        """注册Agent使用的因子列表(用于HHI计算)"""
        if self.factor_hhi:
            self.factor_hhi.register_factors(factors)

    def apply_penalties(
        self,
        scores: Dict[str, Any],
        agent_trades: Dict[str, List[Dict]],
        agent_factors: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """应用所有惩罚

        Args:
            scores: agent_id -> FitnessScore 的字典
            agent_trades: agent_id -> [trade_dict] 的字典
            agent_factors: agent_id -> [factor_name] 的字典(可选)

        Returns:
            (penalized_scores, penalty_report)
        """
        if not self.enabled:
            return scores, {"status": "disabled"}

        self._stats["total_evaluations"] += 1
        report = {
            "arena_penalties": {},
            "turnover_penalties": {},
            "hhi_penalty": 1.0,
            "total_penalized": 0,
        }

        penalized_scores = dict(scores)  # 浅拷贝
        total_penalized = 0

        # 1. Arena Mode拥挤惩罚
        if self.arena_enabled:
            arena_penalty_count = 0
            for agent_id, score in penalized_scores.items():
                # 获取该agent的Arena惩罚
                if hasattr(score, "gt_score"):
                    original_gt = score.gt_score
                    if self.arena:
                        # 使用compute_arena_penalty获取综合惩罚
                        arena_result = self.arena.compute_arena_penalty(agent_id)
                        arena_penalty = arena_result.get("total_penalty", 1.0)
                        if arena_penalty < 1.0:
                            # 应用惩罚但不低于原值50%
                            penalized_gt = max(original_gt * 0.5, original_gt * arena_penalty)
                            try:
                                score.gt_score = penalized_gt
                                arena_penalty_count += 1
                            except AttributeError:
                                pass
            report["arena_penalties"] = {
                "count": arena_penalty_count,
                "status": "applied" if arena_penalty_count > 0 else "no_penalty",
            }
            self._stats["arena_penalties_applied"] += arena_penalty_count

        # 2. 换手率惩罚
        if self.turnover_enabled:
            turnover_penalty_count = 0
            for agent_id, trades in agent_trades.items():
                if not trades or agent_id not in penalized_scores:
                    continue

                # 计算换手率统计
                total_trades = len(trades)
                total_volume = sum(abs(t.get("quantity", 0)) for t in trades)
                avg_position = total_volume / max(total_trades, 1)
                # 估算平均持仓K线数
                holding_bars = []
                for t in trades:
                    hb = t.get("holding_bars", 0)
                    if hb > 0:
                        holding_bars.append(hb)
                avg_holding_bars = sum(holding_bars) / len(holding_bars) if holding_bars else 0

                turnover_stats = self.turnover_penalizer.compute_penalty(
                    total_trades=total_trades,
                    total_volume=total_volume,
                    avg_position=avg_position,
                    avg_holding_bars=avg_holding_bars,
                )

                if turnover_stats.penalty < 1.0:
                    score = penalized_scores[agent_id]
                    if hasattr(score, "gt_score"):
                        original_gt = score.gt_score
                        penalized_gt = max(original_gt * 0.3, original_gt * turnover_stats.penalty)
                        try:
                            score.gt_score = penalized_gt
                            turnover_penalty_count += 1
                        except AttributeError:
                            pass

            report["turnover_penalties"] = {
                "count": turnover_penalty_count,
                "status": "applied" if turnover_penalty_count > 0 else "no_penalty",
            }
            self._stats["turnover_penalties_applied"] += turnover_penalty_count

        # 3. 因子多样性HHI惩罚
        if self.hhi_enabled and agent_factors:
            # 注册所有agent的因子
            for agent_id, factors in agent_factors.items():
                self.factor_hhi.register_factors(factors)

            hhi_penalty = self.factor_hhi.get_diversity_penalty()
            report["hhi_penalty"] = hhi_penalty

            if hhi_penalty < 1.0:
                # 应用全局HHI惩罚到所有agent
                hhi_count = 0
                for agent_id, score in penalized_scores.items():
                    if hasattr(score, "gt_score"):
                        original_gt = score.gt_score
                        penalized_gt = max(original_gt * 0.5, original_gt * hhi_penalty)
                        try:
                            score.gt_score = penalized_gt
                            hhi_count += 1
                        except AttributeError:
                            pass
                report["hhi_penalties"] = {
                    "count": hhi_count,
                    "hhi_value": self.factor_hhi.compute_hhi(),
                    "penalty_factor": hhi_penalty,
                }
                self._stats["hhi_penalties_applied"] += hhi_count

        report["total_penalized"] = (
            report.get("arena_penalties", {}).get("count", 0)
            + report.get("turnover_penalties", {}).get("count", 0)
            + report.get("hhi_penalties", {}).get("count", 0)
        )

        return penalized_scores, report

    def get_stats(self) -> Dict[str, Any]:
        """获取增强统计"""
        return dict(self._stats)

    def reset(self) -> None:
        """重置增强状态(用于新一代)"""
        if self.crowding_tracker and self.arena_enabled:
            # Phase 14.2: 修复no-op — 重新初始化ArenaMode清除跨代拥挤度状态
            # 原bug: reset()是pass空操作, 拥挤度状态跨代累积, 导致惩罚失真
            self.arena = ArenaMode(
                crowding_penalty_weight=getattr(self.arena, "crowding_weight", 0.3),
                correlation_penalty_weight=getattr(self.arena, "correlation_weight", 0.2),
                correlation_threshold=0.7,
            )
            self.crowding_tracker = self.arena
        # HHI和换手率是每次评估时重新计算的,无需重置
__all__ = ["EvolutionEnhancements"]
