# -*- coding: utf-8 -*-
"""
Arena Mode 策略对抗 — P2-1 拥挤信号惩罚+相关性守卫

借鉴 QuantEvolve 和 Arena-based Evolution 的设计:
  - 多个Agent在同一个市场中竞争
  - 拥挤信号惩罚: 当多个Agent使用相同信号时，降低该信号的收益预期
  - 相关性守卫: 防止多个Agent持仓过度相关

核心机制:
  1. 信号拥挤度计算: 统计每个信号被多少Agent使用
  2. 拥挤惩罚: 高拥挤度信号 → 降低适应度
  3. 持仓相关性: 计算Agent间持仓的相关系数
  4. 相关性守卫: 高相关Agent → 降低适应度（鼓励多样化）

参考:
  - QuantEvolve: 行为特征地图维护多样性
  - Crowding Distance in NSGA-II: 拥挤距离
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger("hermes.arena")


# ============================================================================
# 信号拥挤度
# ============================================================================


@dataclass(slots=True)
class SignalUsage:
    """信号使用统计"""
    signal_name: str
    agent_ids: Set[str] = field(default_factory=set)
    total_uses: int = 0
    total_agents: int = 0  # P0修复#15: 由tracker设置，用于计算crowding_score

    @property
    def crowding_score(self) -> float:
        """拥挤度评分 (0=无人用, 1=所有人用)

        P0修复#15: 实现真实计算，不再返回空壳 0.0
        """
        if self.total_agents <= 0:
            return 0.0
        return min(1.0, len(self.agent_ids) / self.total_agents)


class SignalCrowdingTracker:
    """信号拥挤度追踪器

    统计每个信号被多少Agent使用，计算拥挤度。
    拥挤度高的信号 → 预期收益降低（市场已price-in）。
    """

    def __init__(self):
        self._signal_usage: Dict[str, SignalUsage] = defaultdict(lambda: SignalUsage(""))
        self._agent_signals: Dict[str, Set[str]] = defaultdict(set)
        self._total_agents: int = 0

    def register_agent(self, agent_id: str, signals: Set[str]) -> None:
        """注册Agent使用的信号集"""
        # 移除旧信号
        old_signals = self._agent_signals.get(agent_id, set())
        for sig in old_signals:
            if agent_id in self._signal_usage[sig].agent_ids:
                self._signal_usage[sig].agent_ids.discard(agent_id)
                self._signal_usage[sig].total_uses -= 1

        # 注册新信号
        self._agent_signals[agent_id] = signals.copy()
        for sig in signals:
            usage = self._signal_usage[sig]
            usage.signal_name = sig
            usage.agent_ids.add(agent_id)
            usage.total_uses += 1

    def set_total_agents(self, n: int) -> None:
        """设置总Agent数（用于拥挤度归一化）"""
        self._total_agents = max(1, n)
        # P0修复#15: 同步更新所有SignalUsage的total_agents，使crowding_score可用
        for usage in self._signal_usage.values():
            usage.total_agents = self._total_agents

    def get_crowding_penalty(self, agent_id: str) -> float:
        """获取Agent的拥挤惩罚系数

        Returns:
            惩罚系数 0-1 (1=无惩罚, 0=完全惩罚)
        """
        signals = self._agent_signals.get(agent_id, set())
        if not signals or self._total_agents == 0:
            return 1.0

        # 计算平均拥挤度
        total_crowding = 0.0
        for sig in signals:
            usage = self._signal_usage.get(sig)
            if usage:
                crowding = len(usage.agent_ids) / self._total_agents
                total_crowding += crowding

        avg_crowding = total_crowding / len(signals) if signals else 0

        # 拥挤惩罚: 拥挤度越高，惩罚越大
        # crowding=0 → penalty=1.0 (无惩罚)
        # crowding=0.5 → penalty=0.7
        # crowding=1.0 → penalty=0.3
        penalty = 1.0 - 0.7 * avg_crowding

        return max(0.3, penalty)

    def get_signal_crowding(self, signal_name: str) -> float:
        """获取单个信号的拥挤度"""
        usage = self._signal_usage.get(signal_name)
        if not usage or self._total_agents == 0:
            return 0.0
        return len(usage.agent_ids) / self._total_agents

    def get_crowding_report(self) -> Dict[str, Any]:
        """获取拥挤度报告"""
        return {
            "total_agents": self._total_agents,
            "signals_tracked": len(self._signal_usage),
            "top_crowded": sorted(
                [(sig, len(u.agent_ids)) for sig, u in self._signal_usage.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }


# ============================================================================
# 持仓相关性守卫
# ============================================================================


class CorrelationGuard:
    """持仓相关性守卫

    计算Agent间持仓的相关系数，防止过度集中。
    高相关Agent → 降低适应度（鼓励多样化策略）。
    """

    def __init__(self, correlation_threshold: float = 0.7):
        """
        Args:
            correlation_threshold: 相关性阈值，超过则触发惩罚
        """
        self.threshold = correlation_threshold
        self._position_history: Dict[str, List[float]] = defaultdict(list)

    def record_position(self, agent_id: str, position_value: float) -> None:
        """记录Agent的持仓价值"""
        self._position_history[agent_id].append(position_value)
        # 只保留最近100个数据点
        if len(self._position_history[agent_id]) > 100:
            self._position_history[agent_id] = self._position_history[agent_id][-100:]

    def compute_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """计算Agent间持仓相关系数矩阵"""
        agent_ids = list(self._position_history.keys())
        n = len(agent_ids)
        if n < 2:
            return {}

        # 构建持仓矩阵
        min_len = min(len(self._position_history[aid]) for aid in agent_ids)
        if min_len < 5:
            return {}

        matrix = np.array([
            self._position_history[aid][-min_len:]
            for aid in agent_ids
        ])

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(matrix)

        result: Dict[str, Dict[str, float]] = {}
        for i, aid_i in enumerate(agent_ids):
            result[aid_i] = {}
            for j, aid_j in enumerate(agent_ids):
                if i != j:
                    result[aid_i][aid_j] = float(corr_matrix[i, j])

        return result

    def get_correlation_penalty(self, agent_id: str) -> float:
        """获取Agent的相关性惩罚系数

        Returns:
            惩罚系数 0-1 (1=无惩罚, 0=完全惩罚)
        """
        corr_matrix = self.compute_correlation_matrix()
        if agent_id not in corr_matrix:
            return 1.0

        correlations = corr_matrix[agent_id]
        if not correlations:
            return 1.0

        # 找出高相关的Agent数
        high_corr_count = sum(
            1 for corr in correlations.values()
            if abs(corr) > self.threshold
        )

        # 惩罚: 高相关Agent越多，惩罚越大
        total_peers = len(correlations)
        if total_peers == 0:
            return 1.0

        penalty_ratio = high_corr_count / total_peers
        penalty = 1.0 - 0.5 * penalty_ratio

        return max(0.5, penalty)


# ============================================================================
# Arena Mode 协调器
# ============================================================================


class ArenaMode:
    """Arena Mode 策略对抗系统

    在同一个市场中运行多个Agent，通过拥挤惩罚和相关性守卫
    鼓励策略多样化，避免同质化竞争。
    """

    def __init__(
        self,
        crowding_penalty_weight: float = 0.3,
        correlation_penalty_weight: float = 0.2,
        correlation_threshold: float = 0.7,
    ):
        """
        Args:
            crowding_penalty_weight: 拥挤惩罚权重 (0-1)
            correlation_penalty_weight: 相关性惩罚权重 (0-1)
            correlation_threshold: 相关性阈值
        """
        self.crowding_tracker = SignalCrowdingTracker()
        self.correlation_guard = CorrelationGuard(correlation_threshold)
        self.crowding_weight = crowding_penalty_weight
        self.correlation_weight = correlation_penalty_weight

    def register_agent(self, agent_id: str, signals: Set[str]) -> None:
        """注册Agent"""
        self.crowding_tracker.register_agent(agent_id, signals)

    def update_agent_count(self, n: int) -> None:
        """更新Agent总数"""
        self.crowding_tracker.set_total_agents(n)

    def record_position(self, agent_id: str, position_value: float) -> None:
        """记录持仓"""
        self.correlation_guard.record_position(agent_id, position_value)

    def compute_arena_penalty(self, agent_id: str) -> Dict[str, float]:
        """计算Arena惩罚

        Returns:
            {
                "crowding_penalty": float,  # 拥挤惩罚系数
                "correlation_penalty": float,  # 相关性惩罚系数
                "total_penalty": float,  # 总惩罚系数
                "adjusted_score": float,  # 调整后的分数倍率
            }
        """
        crowding = self.crowding_tracker.get_crowding_penalty(agent_id)
        correlation = self.correlation_guard.get_correlation_penalty(agent_id)

        # P0修复#7: 删除死代码，修复公式 — 使用归一化加权平均
        # 原代码有死代码(total变量计算后丢弃)和数学错误(权重不归一化导致惩罚互相抵消)
        # 修复: total_penalty = Σ(penalty_i * weight_i) / Σ(weight_i)
        # 当 crowding=1.0 且 correlation=1.0 (无惩罚) → total=1.0 (无惩罚)
        # 当 crowding=0.3 (高拥挤) 且 correlation=1.0 → total=(0.3*0.3+1.0*0.2)/0.5=0.58 (合理惩罚)
        weight_sum = self.crowding_weight + self.correlation_weight
        if weight_sum > 0:
            total_penalty = (crowding * self.crowding_weight + correlation * self.correlation_weight) / weight_sum
        else:
            total_penalty = 1.0
        total_penalty = max(0.3, min(1.0, total_penalty))

        return {
            "crowding_penalty": crowding,
            "correlation_penalty": correlation,
            "total_penalty": total_penalty,
            "score_multiplier": total_penalty,  # 适应度乘以此系数
        }

    def adjust_fitness(
        self, agent_id: str, original_score: float,
    ) -> Tuple[float, Dict[str, float]]:
        """调整Agent适应度

        Args:
            agent_id: Agent ID
            original_score: 原始适应度分数

        Returns:
            (调整后分数, 惩罚详情)
        """
        penalty = self.compute_arena_penalty(agent_id)
        adjusted = original_score * penalty["score_multiplier"]

        return adjusted, penalty

    def get_arena_report(self) -> Dict[str, Any]:
        """获取Arena报告"""
        return {
            "crowding": self.crowding_tracker.get_crowding_report(),
            "correlation_matrix": self.correlation_guard.compute_correlation_matrix(),
            "weights": {
                "crowding": self.crowding_weight,
                "correlation": self.correlation_weight,
            },
        }


__all__ = [
    "SignalUsage",
    "SignalCrowdingTracker",
    "CorrelationGuard",
    "ArenaMode",
]
