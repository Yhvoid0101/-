# -*- coding: utf-8 -*-
"""
NSGA-III 多目标帕累托选择器 — P1-1 替换GT-Score标量化

借鉴 NSGA-III (Deb & Jain, 2014) 的参考点机制：
  - 在目标空间中生成均匀分布的参考点
  - 将个体关联到最近的参考点
  - 在niching过程中保持多目标多样性

相比NSGA-II的拥挤距离，NSGA-III的优势：
  1. 在3个以上目标上保持更好的多样性分布
  2. 避免NSGA-II在多目标时偏向边界解的问题
  3. 参考点机制确保帕累托前沿被均匀覆盖

4个优化目标（与FitnessScore对齐）：
  - 最大化夏普比率
  - 最大化胜率
  - 最大化盈利因子
  - 最小化最大回撤

参考:
  - Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
    Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I.
  - NSGA-III Python实现: github.com/anyoptimization/pymoo
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .population import (
    AgentGene,
    FitnessScore,
    fast_non_dominated_sort,
)

logger = logging.getLogger("hermes.nsga3")


# ============================================================================
# 参考点生成 (Das-Dennis方法)
# ============================================================================


def generate_reference_points(
    n_objectives: int,
    n_partitions: int,
) -> np.ndarray:
    """生成Das-Dennis参考点

    在(n_objectives-1)维单纯形上均匀分布参考点。
    参考点数量 = C(n_objectives + n_partitions - 1, n_partitions)

    Args:
        n_objectives: 目标数量
        n_partitions: 每个维度的分割数

    Returns:
        (n_points, n_objectives) 参考点矩阵，每行和为1
    """
    if n_objectives == 1:
        return np.array([[1.0]])

    # 生成所有分割组合
    def _generate_recursive(n_obj: int, n_part: int) -> List[List[float]]:
        if n_obj == 1:
            return [[float(n_part)]]
        results = []
        for i in range(n_part + 1):
            for rest in _generate_recursive(n_obj - 1, n_part - i):
                results.append([float(i)] + rest)
        return results

    points = _generate_recursive(n_objectives, n_partitions)

    # 归一化：每个参考点的和为1
    ref_points = np.array(points, dtype=np.float64)
    ref_points = ref_points / n_partitions

    return ref_points


# ============================================================================
# 归一化与关联
# ============================================================================


def normalize_objectives(
    scores: Dict[str, FitnessScore],
    agent_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """归一化目标值到[0,1]范围

    NSGA-III使用理想点和天底点进行归一化。

    Args:
        scores: 适应度字典
        agent_ids: 要归一化的Agent ID列表

    Returns:
        normalized: (n, 4) 归一化后的目标矩阵
        ideal: (4,) 理想点（每列最小值）
        nadir: (4,) 天底点（每列最大值）
    """
    n = len(agent_ids)
    if n == 0:
        return np.zeros((0, 6)), np.zeros(6), np.zeros(6)  # Phase 14.12: 4→6目标

    # 提取目标值 (n, 4)
    # 前3列最大化，第4列最小化
    objectives = np.array([
        [scores[aid].sharpe_ratio,
         scores[aid].win_rate,
         scores[aid].profit_factor,
         scores[aid].max_drawdown,
         scores[aid].calmar_ratio,      # Phase 14.12: 新增Calmar
         scores[aid].sortino_ratio]     # Phase 14.12: 新增Sortino
        for aid in agent_ids
    ], dtype=np.float64)

    # 统一为最大化问题：第4列取负
    objectives_max = objectives.copy()
    objectives_max[:, 3] = -objectives_max[:, 3]

    # 理想点：每列最小值（归一化后为0）
    ideal = objectives_max.min(axis=0)

    # 平移使理想点为原点
    translated = objectives_max - ideal

    # 天底点：每列最大值
    nadir = translated.max(axis=0)

    # 防止除零
    nadir[nadir < 1e-10] = 1.0

    # 归一化到[0,1]
    normalized = translated / nadir

    return normalized, ideal, nadir


def associate_to_reference_points(
    normalized: np.ndarray,
    ref_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将每个个体关联到最近的参考点

    Args:
        normalized: (n, n_obj) 归一化后的目标值
        ref_points: (n_ref, n_obj) 参考点

    Returns:
        associations: (n,) 每个个体关联的参考点索引
        distances: (n,) 每个个体到参考点的垂直距离
    """
    n = normalized.shape[0]
    n_ref = ref_points.shape[0]

    if n == 0 or n_ref == 0:
        return np.zeros(0, dtype=int), np.zeros(0)

    # 计算每个个体到每个参考点的距离
    # 使用垂直距离（点到射线的距离）
    associations = np.zeros(n, dtype=int)
    distances = np.zeros(n)

    for i in range(n):
        point = normalized[i]
        min_dist = float('inf')
        best_ref = 0

        for j in range(n_ref):
            ref = ref_points[j]
            # 垂直距离: ||p - (p·w / ||w||²) * w||
            # 参考点已归一化（和为1），||w||² = sum(w²)
            w_norm_sq = np.dot(ref, ref)
            if w_norm_sq < 1e-10:
                continue

            projection = np.dot(point, ref) / w_norm_sq
            if projection < 1e-10:
                projection = 1e-10

            projected = projection * ref
            perp_dist = np.linalg.norm(point - projected)

            if perp_dist < min_dist:
                min_dist = perp_dist
                best_ref = j

        associations[i] = best_ref
        distances[i] = min_dist

    return associations, distances


# ============================================================================
# Niching选择
# ============================================================================


def nsga3_niching(
    population: List[AgentGene],
    scores: Dict[str, FitnessScore],
    n_select: int,
    n_partitions: int = 12,
) -> List[AgentGene]:
    """NSGA-III选择：参考点导向的多目标选择

    Args:
        population: 当前种群
        scores: 每个Agent的适应度
        n_select: 需要选择的数量
        n_partitions: 参考点分割数（影响参考点数量）

    Returns:
        被选中的Agent列表
    """
    n_objectives = 6  # Phase 14.12: 4→6 (夏普、胜率、盈利因子、回撤、Calmar、Sortino)
    ref_points = generate_reference_points(n_objectives, n_partitions)

    # 第1步：非支配排序
    fronts = fast_non_dominated_sort(scores)
    aid_to_gene = {a.agent_id: a for a in population}

    selected: List[AgentGene] = []
    last_front: List[str] = []

    # 第2步：逐层填充，直到超过n_select
    for front in fronts:
        if len(selected) + len(front) <= n_select:
            selected.extend(aid_to_gene[aid] for aid in front if aid in aid_to_gene)
        else:
            last_front = [aid for aid in front if aid in aid_to_gene]
            break

    if not last_front:
        return selected[:n_select]

    # 第3步：NSGA-III niching选择最后一层
    remaining = n_select - len(selected)
    if remaining <= 0:
        return selected[:n_select]

    # 合并已选和待选的Agent ID
    all_aids = [a.agent_id for a in selected] + last_front

    # 归一化
    normalized, ideal, nadir = normalize_objectives(scores, all_aids)

    if normalized.shape[0] == 0:
        # 降级到NSGA-II拥挤距离
        from .population import crowding_distance
        distances = crowding_distance(last_front, scores)
        sorted_front = sorted(last_front, key=lambda aid: distances.get(aid, 0), reverse=True)
        selected.extend(aid_to_gene[aid] for aid in sorted_front[:remaining])
        return selected

    # 关联到参考点
    associations, distances = associate_to_reference_points(normalized, ref_points)

    # 已选个体在all_aids中的索引
    n_already = len(selected)
    last_front_indices = list(range(n_already, len(all_aids)))

    # 统计每个参考点已有的个体数
    ref_counts: Dict[int, int] = {}
    for i in range(n_already):
        ref_idx = associations[i]
        ref_counts[ref_idx] = ref_counts.get(ref_idx, 0) + 1

    # 第4步：Niching选择
    # 对每个参考点，选择距离最近的个体
    niching_selected: List[str] = []
    available = set(last_front)

    while len(niching_selected) < remaining and available:
        # 找到参考点中个体数最少的
        min_count = min(ref_counts.get(j, 0) for j in range(len(ref_points)))

        # 在最小计数的参考点中选择距离最近的个体
        best_aid: Optional[str] = None
        best_dist = float('inf')
        best_ref = -1

        for aid_idx in last_front_indices:
            aid = all_aids[aid_idx]
            if aid not in available:
                continue

            ref_idx = associations[aid_idx]
            if ref_counts.get(ref_idx, 0) > min_count:
                continue

            dist = distances[aid_idx]
            if dist < best_dist:
                best_dist = dist
                best_aid = aid
                best_ref = ref_idx

        if best_aid is None:
            # 所有参考点都超过最小计数，选距离最近的
            for aid_idx in last_front_indices:
                aid = all_aids[aid_idx]
                if aid not in available:
                    continue
                dist = distances[aid_idx]
                if dist < best_dist:
                    best_dist = dist
                    best_aid = aid
                    best_ref = associations[aid_idx]

        if best_aid is None:
            break

        niching_selected.append(best_aid)
        available.discard(best_aid)
        ref_counts[best_ref] = ref_counts.get(best_ref, 0) + 1

    selected.extend(aid_to_gene[aid] for aid in niching_selected)

    return selected[:n_select]


# ============================================================================
# 统一选择接口
# ============================================================================


def multi_objective_select(
    population: List[AgentGene],
    scores: Dict[str, FitnessScore],
    n_select: int,
    method: str = "nsga3",
) -> List[AgentGene]:
    """多目标选择统一接口

    Args:
        population: 当前种群
        scores: 适应度字典
        n_select: 选择数量
        method: "nsga2" 或 "nsga3"

    Returns:
        被选中的Agent列表
    """
    # Phase 14.9.4: 入口按agent_id去重 (金融知识: 精英池必须唯一个体)
    # 原bug: population中存在重复agent_id, 导致NSGA3 niching选择失效
    _seen_ids = set()
    _unique_pop = []
    for a in population:
        aid = a.agent_id
        if aid not in _seen_ids:
            _seen_ids.add(aid)
            _unique_pop.append(a)

    if method == "nsga3":
        return nsga3_niching(_unique_pop, scores, n_select)
    elif method == "nsga2":
        from .population import nsga2_select
        return nsga2_select(_unique_pop, scores, n_select)
    else:
        raise ValueError(f"不支持的选择方法: {method}")


class NSGA3Selector:
    """NSGA-III多目标选择器(Phase 8类封装)

    封装5个函数式实现,提供面向对象接口 + 自适应权重能力

    4个优化目标(与FitnessScore对齐):
      - 最大化夏普比率
      - 最大化胜率
      - 最大化盈利因子
      - 最小化最大回撤

    Phase 8新增: 自适应权重
      根据市场regime动态调整目标权重:
      - 趋势市场: 夏普权重↑, 回撤权重↓
      - 震荡市场: 胜率权重↑, 夏普权重↓
      - 危机市场: 回撤权重↑↑, 夏普权重↓
    """

    def __init__(self, n_objectives: int = 6, n_partitions: int = 4,  # Phase 14.12: 4→6目标, 12→4分区
                 regime_weights: dict = None):
        """初始化NSGA-III选择器

        Args:
            n_objectives: 优化目标数(默认4)
            n_partitions: 参考点分区数(默认12)
            regime_weights: 按regime的目标权重映射
                {"trend": {"sharpe": 0.4, "win_rate": 0.2, "profit_factor": 0.2, "drawdown": 0.2},
                 "range": {"sharpe": 0.2, "win_rate": 0.4, "profit_factor": 0.2, "drawdown": 0.2},
                 "crisis": {"sharpe": 0.1, "win_rate": 0.2, "profit_factor": 0.2, "drawdown": 0.5}}
        """
        self.n_objectives = n_objectives
        self.n_partitions = n_partitions
        self._reference_points = None
        self._current_regime = "neutral"
        self._regime_weights = regime_weights or {
            # Phase 14.12: 6目标regime权重 — 添加Calmar(收益/回撤)和Sortino(下行风险)
            "trend":   {"sharpe": 0.25, "win_rate": 0.15, "profit_factor": 0.15,
                        "drawdown": 0.15, "calmar": 0.15, "sortino": 0.15},
            "range":   {"sharpe": 0.15, "win_rate": 0.25, "profit_factor": 0.15,
                        "drawdown": 0.15, "calmar": 0.15, "sortino": 0.15},
            "crisis":  {"sharpe": 0.08, "win_rate": 0.12, "profit_factor": 0.12,
                        "drawdown": 0.35, "calmar": 0.20, "sortino": 0.13},
            "neutral": {"sharpe": 0.17, "win_rate": 0.17, "profit_factor": 0.16,
                        "drawdown": 0.17, "calmar": 0.16, "sortino": 0.17},
        }

    @property
    def reference_points(self):
        """惰性生成参考点"""
        if self._reference_points is None:
            self._reference_points = generate_reference_points(
                self.n_objectives, self.n_partitions
            )
        return self._reference_points

    def set_regime(self, regime: str) -> None:
        """设置当前市场regime,调整目标权重

        Args:
            regime: "trend" | "range" | "crisis" | "neutral"
        """
        if regime in self._regime_weights:
            self._current_regime = regime
        else:
            self._current_regime = "neutral"

    def get_current_weights(self) -> dict:
        """获取当前regime的目标权重"""
        return self._regime_weights.get(self._current_regime, self._regime_weights["neutral"])

    def select(self, population, scores, n_select: int, method: str = "nsga3"):
        """多目标选择

        Args:
            population: Agent列表
            scores: {agent_id: FitnessScore} 映射
            n_select: 选择数量
            method: "nsga3" | "nsga2"

        Returns:
            被选中的Agent列表
        """
        return multi_objective_select(population, scores, n_select, method=method)

    def select_with_regime(self, population, scores, n_select: int,
                           regime: str = None) -> list:
        """带regime感知的多目标选择

        Phase 8新增: 根据regime调整选择偏好
        - crisis regime: 优先选择低回撤agent
        - trend regime: 优先选择高夏普agent
        - range regime: 优先选择高胜率agent

        Args:
            population: Agent列表
            scores: {agent_id: FitnessScore} 映射
            n_select: 选择数量
            regime: 市场regime(如不指定用当前设置)

        Returns:
            被选中的Agent列表
        """
        if regime:
            self.set_regime(regime)

        # Phase 14.9.3: 候选池过滤 — 只保留已运行个体 (金融知识: 精英必须基于已验证个体)
        # 原bug: 不过滤注入个体(survival_rounds=0, gt=0.0), 归一化后gt=0.0 > 亏损个体(负数)
        #        导致注入个体被错误选为精英, 进化无积累
        # 来源: LO-CCAC 2026 Portfolio Evolutionary Strategies — 精英池必须基于已验证个体
        _eligible = [a for a in population if getattr(a, 'survival_rounds', 0) > 0]
        if not _eligible:
            _eligible = [a for a in population if getattr(a, 'total_trades', 0) > 0]
        if not _eligible:
            _eligible = list(population)  # 极端降级: 全部是注入个体时避免空集

        # Phase 14.9.4: 按agent_id去重 (避免重复个体导致niching失效)
        _seen_ids = set()
        _eligible_unique = []
        for a in _eligible:
            if a.agent_id not in _seen_ids:
                _seen_ids.add(a.agent_id)
                _eligible_unique.append(a)

        # 先用NSGA-III选择候选池(2倍) — 从已过滤去重的_eligible_unique中选择
        candidates = self.select(_eligible_unique, scores, n_select * 2, method="nsga3")

        # 按regime权重排序候选池
        weights = self.get_current_weights()

        def regime_score(agent):
            fs = scores.get(agent.agent_id)
            if fs is None:
                return 0.0
            # 标准化各目标到[0,1]
            sharpe_n = min(max(fs.sharpe_ratio / 3.0, 0), 1)
            win_n = min(max(fs.win_rate, 0), 1)
            pf_n = min(max(fs.profit_factor / 3.0, 0), 1)
            dd_n = 1.0 - min(max(fs.max_drawdown, 0), 1)  # 回撤越低越好
            calmar_n = min(max(fs.calmar_ratio / 5.0, 0), 1)    # Phase 14.12: Calmar=5→满分
            sortino_n = min(max(fs.sortino_ratio / 5.0, 0), 1)  # Phase 14.12: Sortino=5→满分

            return (sharpe_n * weights["sharpe"] +
                    win_n * weights["win_rate"] +
                    pf_n * weights["profit_factor"] +
                    dd_n * weights["drawdown"] +
                    calmar_n * weights.get("calmar", 0) +
                    sortino_n * weights.get("sortino", 0))

        # 按regime加权得分排序
        sorted_candidates = sorted(candidates, key=regime_score, reverse=True)
        return sorted_candidates[:n_select]


__all__ = [
    "generate_reference_points",
    "normalize_objectives",
    "associate_to_reference_points",
    "nsga3_niching",
    "multi_objective_select",
    "NSGA3Selector",
]
