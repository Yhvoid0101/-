"""Population NSGA-II module (Phase 7.12 extracted from population.py)

Contains:
  - _dominates: pareto dominance check
  - fast_non_dominated_sort: NSGA-II fast non-dominated sort
  - crowding_distance: crowding distance calculation
  - nsga2_select: NSGA-II selection
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .population_fitness import FitnessScore


def _dominates(a: FitnessScore, b: FitnessScore) -> bool:
    """判断个体a是否支配个体b（多目标：最大化夏普、最大化胜率、最大化盈利因子、最小化回撤）

    a支配b当且仅当：a在所有目标上不劣于b，且至少一个目标严格优于b。
    """
    # 最大化目标：夏普、胜率、盈利因子
    # 最小化目标：最大回撤
    # Phase 14.12: 添加Calmar和Sortino目标
    better_or_equal = (
        a.sharpe_ratio >= b.sharpe_ratio
        and a.win_rate >= b.win_rate
        and a.profit_factor >= b.profit_factor
        and a.max_drawdown <= b.max_drawdown
        and a.calmar_ratio >= b.calmar_ratio
        and a.sortino_ratio >= b.sortino_ratio
    )
    strictly_better = (
        a.sharpe_ratio > b.sharpe_ratio
        or a.win_rate > b.win_rate
        or a.profit_factor > b.profit_factor
        or a.max_drawdown < b.max_drawdown
        or a.calmar_ratio > b.calmar_ratio
        or a.sortino_ratio > b.sortino_ratio
    )
    return better_or_equal and strictly_better


def fast_non_dominated_sort(
    scores: Dict[str, FitnessScore],
) -> List[List[str]]:
    """NSGA-II快速非支配排序

    将种群分为多个帕累托前沿层：
      Front 0 = 不被任何个体支配的精英（帕累托最优集）
      Front 1 = 只被Front 0支配
      Front 2 = 只被Front 0和1支配
      ...

    P4#12: 向量化支配关系计算 — 用 numpy 广播一次性计算所有对的支配矩阵，
    替代 O(n²) 的 _dominates 双重循环调用。

    Returns:
        [[agent_id, ...], ...] 按前沿层级排序
    """
    agent_ids = list(scores.keys())
    n = len(agent_ids)

    # P4#12: 空字典处理（与原始实现行为一致，返回 [[]]）
    if n == 0:
        return [[]]

    # P4#12: 向量化支配关系计算
    # 提取目标值矩阵 (n × 4)：sharpe, win_rate, profit_factor, max_drawdown
    # 前3列最大化，第4列最小化
    objectives = np.array([
        [scores[aid].sharpe_ratio, scores[aid].win_rate,
         scores[aid].profit_factor, scores[aid].max_drawdown,
         scores[aid].calmar_ratio, scores[aid].sortino_ratio]
        for aid in agent_ids
    ], dtype=np.float64)

    # P4#12: 用 2D 矩阵直接计算，避免 3D 中间数组开销
    # 提取各列：前3列最大化，第4列最小化
    col_sharpe = objectives[:, 0]       # (n,) 夏普
    col_win = objectives[:, 1]          # (n,) 胜率
    col_pf = objectives[:, 2]           # (n,) 盈利因子
    col_dd = objectives[:, 3]           # (n,) 最大回撤
    col_calmar = objectives[:, 4]       # (n,) Calmar (Phase 14.12)
    col_sortino = objectives[:, 5]      # (n,) Sortino (Phase 14.12)

    # better_or_equal[i,j] = i 在所有目标上不劣于 j
    # 广播：(n,1) vs (1,n) → (n,n)
    ge0 = col_sharpe[:, np.newaxis] >= col_sharpe[np.newaxis, :]
    ge1 = col_win[:, np.newaxis] >= col_win[np.newaxis, :]
    ge2 = col_pf[:, np.newaxis] >= col_pf[np.newaxis, :]
    le3 = col_dd[:, np.newaxis] <= col_dd[np.newaxis, :]
    ge4 = col_calmar[:, np.newaxis] >= col_calmar[np.newaxis, :]
    ge5 = col_sortino[:, np.newaxis] >= col_sortino[np.newaxis, :]
    better_or_equal = ge0 & ge1 & ge2 & le3 & ge4 & ge5

    # strictly_better[i,j] = i 至少一个目标严格优于 j
    gt0 = col_sharpe[:, np.newaxis] > col_sharpe[np.newaxis, :]
    gt1 = col_win[:, np.newaxis] > col_win[np.newaxis, :]
    gt2 = col_pf[:, np.newaxis] > col_pf[np.newaxis, :]
    lt3 = col_dd[:, np.newaxis] < col_dd[np.newaxis, :]
    gt4 = col_calmar[:, np.newaxis] > col_calmar[np.newaxis, :]
    gt5 = col_sortino[:, np.newaxis] > col_sortino[np.newaxis, :]
    strictly_better = gt0 | gt1 | gt2 | lt3 | gt4 | gt5

    # 支配矩阵：dominates_matrix[i,j] = True 表示 i 支配 j
    dominates_matrix = better_or_equal & strictly_better
    np.fill_diagonal(dominates_matrix, False)  # 排除自身

    # 构建 domination_count 和 dominated_set
    domination_count_arr = np.sum(dominates_matrix, axis=0)  # 每列和 = 被支配次数
    domination_count: Dict[str, int] = {
        aid: int(domination_count_arr[idx]) for idx, aid in enumerate(agent_ids)
    }
    dominated_set: Dict[str, List[str]] = {aid: [] for aid in agent_ids}
    # P4#12: 用 argwhere 一次性提取所有支配对，避免 n² 双重循环
    dominating_pairs = np.argwhere(dominates_matrix)
    for i, j in dominating_pairs:
        dominated_set[agent_ids[int(i)]].append(agent_ids[int(j)])

    # 分层
    fronts: List[List[str]] = []
    current_front = [aid for aid in agent_ids if domination_count[aid] == 0]
    fronts.append(current_front)

    while current_front:
        next_front = []
        for aid in current_front:
            for dominated_aid in dominated_set[aid]:
                domination_count[dominated_aid] -= 1
                if domination_count[dominated_aid] == 0:
                    next_front.append(dominated_aid)
        if next_front:
            fronts.append(next_front)
        current_front = next_front

    return fronts


def crowding_distance(
    front: List[str],
    scores: Dict[str, FitnessScore],
) -> Dict[str, float]:
    """计算同一前沿层中每个个体的拥挤距离

    拥挤距离大的个体位于稀疏区域，应优先保留以维持多样性。
    """
    if len(front) <= 2:
        return {aid: float("inf") for aid in front}

    distances = {aid: 0.0 for aid in front}

    # 4个目标维度
    objectives = [
        ("sharpe_ratio", False),   # 最大化
        ("win_rate", False),       # 最大化
        ("profit_factor", False),  # 最大化
        ("max_drawdown", True),    # 最小化（取反后最大化）
    ]

    for attr, reverse in objectives:
        # 按该目标排序
        sorted_front = sorted(
            front,
            key=lambda aid: getattr(scores[aid], attr),
            reverse=reverse,
        )

        # 边界个体距离无穷大
        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")

        # 计算中间个体的距离
        values = [getattr(scores[aid], attr) for aid in sorted_front]
        val_range = max(values) - min(values)
        if val_range > 1e-10:
            for i in range(1, len(sorted_front) - 1):
                if distances[sorted_front[i]] != float("inf"):
                    distances[sorted_front[i]] += (
                        values[i + 1] - values[i - 1]
                    ) / val_range

    return distances


def nsga2_select(
    population: List[AgentGene],
    scores: Dict[str, FitnessScore],
    n_select: int,
) -> List[AgentGene]:
    """NSGA-II选择：先按前沿层级，同层按拥挤距离

    Args:
        population: 当前种群
        scores: 每个Agent的适应度
        n_select: 需要选择的数量

    Returns:
        被选中的Agent列表
    """
    fronts = fast_non_dominated_sort(scores)
    selected: List[AgentGene] = []
    aid_to_gene = {a.agent_id: a for a in population}

    for front in fronts:
        if len(selected) + len(front) <= n_select:
            # 整层保留
            selected.extend(aid_to_gene[aid] for aid in front)
        else:
            # 部分保留：按拥挤距离选
            remaining = n_select - len(selected)
            distances = crowding_distance(front, scores)
            sorted_front = sorted(front, key=lambda aid: distances[aid], reverse=True)
            selected.extend(aid_to_gene[aid] for aid in sorted_front[:remaining])
            break

    return selected
