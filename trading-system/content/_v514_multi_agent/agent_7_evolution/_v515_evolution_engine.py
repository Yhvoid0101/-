#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v515 NSGA-II 多目标进化引擎 — Agent-7 核心交付
================================================================
用户铁律:
  "构建一代比一代强的进化体系, 实现策略持续优化, 逐步提升
   盈利能力、盈亏比和胜率, 增强对各种行情、状况和波动的适应能力"

相比 v515 单目标 GA 的关键进化:
  1. NSGA-II 核心算法:
     - 快速非支配排序 (Fast Non-dominated Sort, O(MN^2))
     - 拥挤距离 (Crowding Distance) 保持多样性
     - 精英保留 + 锦标赛选择 + 均匀交叉
  2. 真帕累托前沿 (非加权和):
     - 5 个目标同时优化, 保留非支配解集合
     - 不再"压扁"成单一 fitness, 而是给决策者完整权衡空间
  3. 自适应变异率:
     - mut_rate = 0.5 * (1 - gen/max_gen) + 0.1
     - 前期高变异 (探索), 后期低变异 (开发)
  4. 进化参数升级:
     - 种群: 50 (v515 是 20)
     - 代数: 20 (v515 是 8)
  5. 与知识库联动:
     - 从 KnowledgeBase 读取策略模板初始化种群
     - 历史最优基因注入 (避免重复探索)
     - 进化结果写回记忆

5 个优化目标 (max/min 明确):
  - max sharpe          (盈利能力)
  - max win_rate        (胜率)
  - max n_trades        (统计显著性, 软约束)
  - min monthly_loss    (月亏概率)
  - max profit_loss_ratio  (盈亏比)

依赖:
  - _v515_knowledge_base.py (本目录)
  - _v515_ensemble_strategy.py (上级目录, 提供 backtest/evaluate 基础)
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np

# ============================================================
# 路径设置: 让本模块能 import v515 策略与知识库
# ============================================================
HERE = Path(__file__).parent.resolve()
SANDBOX_DIR = HERE.parent.parent  # sandbox_trading 根目录
sys.path.insert(0, str(HERE))                       # 知识库在本目录
sys.path.insert(0, str(HERE.parent.parent))         # v515 策略在 sandbox 根目录

# 导入 v515 策略 (复用 backtest / 指标 / 数据加载)
from _v515_ensemble_strategy import (  # type: ignore
    GENE_RANGES,
    TARGETS,
    WF_WINDOWS,
    IS_RATIO,
    PURGE_RATIO,
    MIN_TRADES_FOR_SIGNIFICANCE,
    Individual as V515Individual,
    random_genes,
    load_klines,
    calc_indicators,
    backtest_window,
)
# 导入知识库
from _v515_knowledge_base import KnowledgeBase, GeneMemoryEntry  # type: ignore


# ============================================================
# 1. NSGA-II 数据结构
# ============================================================
@dataclass
class NSGAIndividual:
    """NSGA-II 个体: 基因 + 5目标值 + 帕累托层级 + 拥挤距离"""
    genes: Dict[str, float]
    # 5 个目标 (原始值, 未经归一化)
    sharpe: float = 0.0           # maximize
    win_rate: float = 0.0         # maximize
    n_trades: float = 0.0         # maximize (统计显著性)
    monthly_loss: float = 1.0     # minimize
    profit_loss_ratio: float = 0.0  # maximize

    # 辅助指标 (不参与帕累托, 用于报告/KPI)
    max_dd: float = 0.0
    annual_return: float = 0.0
    consec_loss: float = 0.0
    win_rate_ci_lo: float = 0.0
    is_sharpe: float = 0.0
    df: float = 0.0
    kpi_pass: int = 0
    fitness_scalar: float = 0.0   # 仅用于报告对比

    # NSGA-II 元数据
    rank: int = -1                # 帕累托层级 (1=最优前沿)
    crowding_distance: float = 0.0
    evaluated: bool = False


# ============================================================
# 2. 评估函数: 把 v515 backtest 结果映射为 5 目标
# ============================================================
def evaluate_individual(
    ind: NSGAIndividual,
    all_data: Dict[str, Tuple],
    seed: int = 0,
) -> NSGAIndividual:
    """
    在 Walk-Forward 4 窗口 × 5 标的上评估个体, 计算 5 个目标值

    设计:
      - 复用 v515 的 backtest_window (含成本/滑点/IS-OOS)
      - 跨窗口取均值, 提高稳定性
      - DF (退化因子) 用于检测过拟合 (辅助, 不参与帕累托)
    """
    is_sharpes: List[float] = []
    oos_sharpes: List[float] = []
    oos_winrates: List[float] = []
    oos_pfs: List[float] = []
    oos_trades: List[float] = []
    oos_monthly_losses: List[float] = []
    oos_dds: List[float] = []
    oos_annuals: List[float] = []
    oos_consec_losses: List[float] = []
    oos_winrate_cis: List[float] = []

    for symbol, (bars, indicators) in all_data.items():
        n = len(bars)
        window_size = max(50, n // WF_WINDOWS)
        for w in range(WF_WINDOWS):
            ws = w * window_size
            we = (w + 1) * window_size
            if we - ws < 100:
                continue
            is_end = int(ws + (we - ws) * IS_RATIO)
            purge_end = int(is_end + (we - ws) * PURGE_RATIO)
            oos_start = purge_end

            is_metrics = backtest_window(bars, indicators, ws, is_end, ind.genes)
            oos_metrics = backtest_window(bars, indicators, oos_start, we, ind.genes)

            is_sharpes.append(is_metrics["sharpe"])
            oos_sharpes.append(oos_metrics["sharpe"])
            oos_winrates.append(oos_metrics["win_rate"])
            oos_pfs.append(oos_metrics["profit_loss_ratio"])
            oos_trades.append(oos_metrics["n_trades"])
            oos_monthly_losses.append(oos_metrics["monthly_loss_prob"])
            oos_dds.append(oos_metrics["max_dd"])
            oos_annuals.append(oos_metrics["annual_return"])
            oos_consec_losses.append(oos_metrics["max_consecutive_losses"])
            oos_winrate_cis.append(oos_metrics.get("win_rate_ci_lo", 0.0))

    ind.sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    ind.win_rate = float(np.mean(oos_winrates)) if oos_winrates else 0.0
    ind.n_trades = float(np.mean(oos_trades)) if oos_trades else 0.0
    ind.monthly_loss = float(np.mean(oos_monthly_losses)) if oos_monthly_losses else 1.0
    ind.profit_loss_ratio = float(np.mean(oos_pfs)) if oos_pfs else 0.0

    ind.max_dd = float(np.mean(oos_dds)) if oos_dds else 0.0
    ind.annual_return = float(np.mean(oos_annuals)) if oos_annuals else 0.0
    ind.consec_loss = float(np.mean(oos_consec_losses)) if oos_consec_losses else 0.0
    ind.win_rate_ci_lo = float(np.mean(oos_winrate_cis)) if oos_winrate_cis else 0.0
    ind.is_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0

    # DF (退化因子): OOS/IS, 检测过拟合
    if ind.is_sharpe > 0:
        ind.df = max(0.0, min(1.0, ind.sharpe / ind.is_sharpe))
    elif ind.sharpe > 0:
        ind.df = 0.5
    else:
        ind.df = 0.0

    # KPI 通过数 (用户7关卡)
    kpi = 0
    if ind.sharpe >= TARGETS["sharpe"]: kpi += 1
    if ind.max_dd <= TARGETS["max_drawdown"]: kpi += 1
    if ind.win_rate >= TARGETS["win_rate"]: kpi += 1
    if ind.n_trades >= TARGETS["trades_min"]: kpi += 1
    if ind.monthly_loss <= TARGETS["monthly_loss_prob"]: kpi += 1
    if ind.consec_loss <= TARGETS["consecutive_losses"]: kpi += 1
    if ind.annual_return >= TARGETS["annual_return"]: kpi += 1
    ind.kpi_pass = kpi

    # 标量 fitness (仅用于报告对比, NSGA-II 不用它做选择)
    fit = ind.sharpe
    if ind.df >= 0.4: fit += 3.0
    fit += kpi * 0.5
    if ind.n_trades < MIN_TRADES_FOR_SIGNIFICANCE:
        fit -= 10.0
        if ind.n_trades < 15:
            fit -= 10.0
    elif 30 <= ind.n_trades <= 150:
        fit += 2.0
    elif ind.n_trades > 200:
        fit -= 1.0
    if ind.win_rate_ci_lo >= 0.55: fit += 3.0
    elif ind.win_rate >= 0.55: fit += 1.0
    if ind.monthly_loss <= 0.05: fit += 3.0
    elif ind.monthly_loss <= 0.10: fit += 1.0
    # 过拟合惩罚
    if ind.is_sharpe > 3.0 and ind.sharpe < ind.is_sharpe / 3.0:
        fit -= 5.0
    # 小样本偏差惩罚
    if ind.sharpe > 5.0 and ind.n_trades < 50:
        fit -= 5.0
    ind.fitness_scalar = fit

    ind.evaluated = True
    return ind


# ============================================================
# 3. NSGA-II 核心: 快速非支配排序
# ============================================================
# 目标方向: True=maximize, False=minimize
OBJECTIVE_DIRECTIONS = {
    "sharpe": True,
    "win_rate": True,
    "n_trades": True,
    "monthly_loss": False,
    "profit_loss_ratio": True,
}


def _dominates(a: NSGAIndividual, b: NSGAIndividual) -> bool:
    """
    判断 a 是否支配 b (a ≺ b)
    a 在所有目标上 ≥ b, 且至少一个目标严格 > b
    (考虑 max/min 方向)
    """
    better_or_equal = True
    strictly_better = False
    for obj, is_max in OBJECTIVE_DIRECTIONS.items():
        av = getattr(a, obj)
        bv = getattr(b, obj)
        if is_max:
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
        else:  # minimize
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
    return better_or_equal and strictly_better


def fast_non_dominated_sort(population: List[NSGAIndividual]) -> List[List[int]]:
    """
    NSGA-II 快速非支配排序
    返回: [[层1索引], [层2索引], ...]  层1 = 帕累托前沿
    """
    n = len(population)
    domination_count = [0] * n          # 被 a 支配的个体数
    dominated_set: List[List[int]] = [[] for _ in range(n)]  # a 支配的个体索引
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(population[p], population[q]):
                dominated_set[p].append(q)
            elif _dominates(population[q], population[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            population[p].rank = 1
            fronts[0].append(p)

    # 逐层扩展
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    # 移除空层
    return [f for f in fronts if f]


# ============================================================
# 4. NSGA-II 核心: 拥挤距离
# ============================================================
def crowding_distance_assignment(
    population: List[NSGAIndividual],
    front_indices: List[int],
) -> None:
    """
    计算指定前沿层内每个个体的拥挤距离
    拥挤距离大 = 该个体周围稀疏 = 应保留 (维持多样性)
    """
    n = len(front_indices)
    if n == 0:
        return
    for idx in front_indices:
        population[idx].crowding_distance = 0.0

    for obj, is_max in OBJECTIVE_DIRECTIONS.items():
        # 按目标值排序
        sorted_idx = sorted(
            front_indices,
            key=lambda i: getattr(population[i], obj),
            reverse=not is_max,  # max: 降序; min: 升序
        )
        # 边界个体设为无穷大 (必保留)
        population[sorted_idx[0]].crowding_distance = float("inf")
        population[sorted_idx[-1]].crowding_distance = float("inf")

        # 取值范围
        vals = [getattr(population[i], obj) for i in sorted_idx]
        v_min = vals[-1]
        v_max = vals[0]
        denom = v_max - v_min
        if abs(denom) < 1e-12:
            continue

        # 中间个体: 拥挤距离 = |邻居差| / 范围 (NSGA-II 原始论文使用绝对值)
        for k in range(1, n - 1):
            i_prev = sorted_idx[k - 1]
            i_next = sorted_idx[k + 1]
            v_prev = getattr(population[i_prev], obj)
            v_next = getattr(population[i_next], obj)
            population[sorted_idx[k]].crowding_distance += abs(v_next - v_prev) / denom


# ============================================================
# 5. NSGA-II 选择: 锦标赛 (基于 rank + crowding)
# ============================================================
def tournament_select(
    population: List[NSGAIndividual],
    k: int = 2,
) -> NSGAIndividual:
    """
    锦标赛选择: 随机选 k 个, 取 rank 最小 (帕累托层级最优) 的;
    若 rank 相同, 取 crowding_distance 大的
    """
    candidates = random.sample(population, min(k, len(population)))
    best = candidates[0]
    for c in candidates[1:]:
        # (rank 升序, crowding 降序) 取最优
        if c.rank < best.rank or (
            c.rank == best.rank and c.crowding_distance > best.crowding_distance
        ):
            best = c
    return best


# ============================================================
# 6. NSGA-II 交叉与变异
# ============================================================
def uniform_crossover(g1: Dict[str, float], g2: Dict[str, float]) -> Dict[str, float]:
    """均匀交叉: 每个基因独立 50% 概率从父代1或父代2继承"""
    return {k: (g1[k] if random.random() < 0.5 else g2[k]) for k in GENE_RANGES}


def mutate(genes: Dict[str, float], rate: float) -> Dict[str, float]:
    """
    自适应变异: 每个基因以 rate 概率重置为范围内随机值
    rate = 0.5 * (1 - gen/max_gen) + 0.1  (前期探索, 后期开发)
    """
    new = dict(genes)
    for k, (lo, hi) in GENE_RANGES.items():
        if random.random() < rate:
            # 50% 概率小幅扰动 (开发), 50% 概率大幅跳变 (探索)
            if random.random() < 0.5:
                # 小幅扰动: ±10% 范围内
                cur = new[k]
                delta = (hi - lo) * 0.1
                new[k] = max(lo, min(hi, cur + random.uniform(-delta, delta)))
            else:
                # 大幅跳变: 完全随机
                new[k] = random.uniform(lo, hi)
    return new


def adaptive_mutation_rate(gen: int, max_gen: int) -> float:
    """
    自适应变异率: 前期高 (探索), 后期低 (开发)
    公式: 0.5 * (1 - gen/max_gen) + 0.1
    gen=0: 0.6, gen=max_gen/2: 0.35, gen=max_gen: 0.1
    """
    if max_gen <= 0:
        return 0.3
    return 0.5 * (1.0 - gen / max_gen) + 0.1


# ============================================================
# 7. NSGA-II 主循环
# ============================================================
class NSGA2Engine:
    """
    NSGA-II 多目标进化引擎

    参数:
      pop_size: 种群大小 (默认 50, 比 v515 的 20 大)
      generations: 进化代数 (默认 20, 比 v515 的 8 多)
      elite_ratio: 精英保留比例
      crossover_rate: 交叉概率
      inject_memory: 是否注入历史最优基因
    """

    def __init__(
        self,
        all_data: Dict[str, Tuple],
        pop_size: int = 50,
        generations: int = 20,
        elite_ratio: float = 0.1,
        crossover_rate: float = 0.9,
        inject_memory: bool = True,
        knowledge_base: Optional[KnowledgeBase] = None,
        seed: int = 5152,
    ):
        self.all_data = all_data
        self.pop_size = pop_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.inject_memory = inject_memory
        self.kb = knowledge_base or KnowledgeBase()
        self.kb.load_memory()
        self.history: List[Dict[str, Any]] = []  # 每代进展
        self.pareto_front: List[NSGAIndividual] = []
        random.seed(seed)
        np.random.seed(seed)

    def _init_population(self) -> List[NSGAIndividual]:
        """初始化种群: 部分随机 + 部分注入历史最优"""
        pop: List[NSGAIndividual] = []
        n_inject = 0

        if self.inject_memory and self.kb.memory:
            # 注入历史最优基因 (最多注入 30%)
            top = self.kb.top_genes(n=max(1, self.pop_size // 3))
            for entry in top:
                # 把历史基因映射回当前 GENE_RANGES (兼容性)
                g = {}
                for k, (lo, hi) in GENE_RANGES.items():
                    if k in entry.genes:
                        g[k] = max(lo, min(hi, entry.genes[k]))
                    else:
                        g[k] = random.uniform(lo, hi)
                pop.append(NSGAIndividual(genes=g))
                n_inject += 1
            print(f"  [NSGA-II] 注入历史最优基因: {n_inject} 条")

        # 用知识库策略模板初始化一部分 (覆盖各 regime)
        # 从 mean_reversion_relaxed 模板取值 (v515 的核心策略)
        mr_template = self.kb.gene_template_for("mean_reversion_relaxed")
        n_template = 0
        if mr_template:
            for _ in range(max(2, self.pop_size // 5)):
                g = random_genes()
                # 用模板范围覆盖部分基因 (偏向已知有效区域)
                for k, (lo, hi) in mr_template.items():
                    if k in GENE_RANGES:
                        g[k] = random.uniform(lo, hi)
                pop.append(NSGAIndividual(genes=g))
                n_template += 1
            print(f"  [NSGA-II] 策略模板初始化: {n_template} 条")

        # 剩余随机
        while len(pop) < self.pop_size:
            pop.append(NSGAIndividual(genes=random_genes()))

        return pop[:self.pop_size]

    def _make_new_pop(
        self,
        parents: List[NSGAIndividual],
        mut_rate: float,
    ) -> List[NSGAIndividual]:
        """通过选择+交叉+变异产生子代"""
        children: List[NSGAIndividual] = []
        n_elite = max(2, int(self.pop_size * self.elite_ratio))

        # 精英保留: 直接复制最优前沿前 n_elite 个 (按 crowding 排序)
        elite_sorted = sorted(
            parents,
            key=lambda x: (x.rank, -x.crowding_distance),
        )[:n_elite]
        for e in elite_sorted:
            children.append(NSGAIndividual(genes=dict(e.genes)))

        # 交叉+变异
        while len(children) < self.pop_size:
            p1 = tournament_select(parents, k=2)
            p2 = tournament_select(parents, k=2)
            if random.random() < self.crossover_rate:
                child_genes = uniform_crossover(p1.genes, p2.genes)
            else:
                child_genes = dict(p1.genes)
            child_genes = mutate(child_genes, rate=mut_rate)
            children.append(NSGAIndividual(genes=child_genes))

        return children

    def run(self, verbose: bool = True) -> List[NSGAIndividual]:
        """
        运行 NSGA-II 进化
        返回: 最终帕累托前沿 (rank=1 的个体集合)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("v515 NSGA-II 多目标进化引擎启动")
            print("=" * 70)
            print(f"种群: {self.pop_size}, 代数: {self.generations}")
            print(f"目标: {len(OBJECTIVE_DIRECTIONS)} 个 (夏普/胜率/交易数/月亏/盈亏比)")
            print(f"自适应变异率: 0.5*(1-gen/max)+0.1 (前期探索, 后期开发)")
            print(f"符号: {list(self.all_data.keys())}")
            print(f"知识库: {len(self.kb.memory)} 条历史记忆")
            print()

        # 1. 初始化 + 评估
        population = self._init_population()
        if verbose:
            print(f"  [初始化] 评估 {len(population)} 个个体...")
        for i, ind in enumerate(population):
            if not ind.evaluated:
                population[i] = evaluate_individual(ind, self.all_data)
            if verbose and (i + 1) % 10 == 0:
                print(f"    评估 {i+1}/{len(population)}")

        # 2. 主循环
        for gen in range(self.generations):
            t0 = time.time()
            mut_rate = adaptive_mutation_rate(gen, self.generations)

            # 非支配排序
            fronts = fast_non_dominated_sort(population)
            # 拥挤距离 (每层独立计算)
            for front in fronts:
                crowding_distance_assignment(population, front)

            # 记录前沿
            front1 = [population[i] for i in fronts[0]]
            self.pareto_front = front1

            # 进展统计
            front_sharpe = np.mean([x.sharpe for x in front1]) if front1 else 0
            front_winrate = np.mean([x.win_rate for x in front1]) if front1 else 0
            front_trades = np.mean([x.n_trades for x in front1]) if front1 else 0
            front_monthly_loss = np.mean([x.monthly_loss for x in front1]) if front1 else 1
            front_pf = np.mean([x.profit_loss_ratio for x in front1]) if front1 else 0
            front_kpi = np.mean([x.kpi_pass for x in front1]) if front1 else 0
            front_size = len(front1)

            self.history.append({
                "gen": gen + 1,
                "mut_rate": round(mut_rate, 4),
                "front_size": front_size,
                "front_sharpe": round(float(front_sharpe), 4),
                "front_winrate": round(float(front_winrate), 4),
                "front_trades": round(float(front_trades), 2),
                "front_monthly_loss": round(float(front_monthly_loss), 4),
                "front_pf": round(float(front_pf), 4),
                "front_kpi_avg": round(float(front_kpi), 2),
                "elapsed_sec": round(time.time() - t0, 2),
            })

            if verbose:
                print(f"\n--- 代 {gen+1}/{self.generations} (mut_rate={mut_rate:.3f}) ---")
                print(f"  帕累托前沿大小: {front_size}")
                print(f"  前沿均值: 夏普={front_sharpe:.3f}, 胜率={front_winrate*100:.1f}%, "
                      f"交易数={front_trades:.1f}, 月亏={front_monthly_loss*100:.1f}%, "
                      f"盈亏比={front_pf:.2f}, KPI={front_kpi:.1f}/7")

            # 3. 产生子代 (最后一代不再产生, 直接结束)
            if gen == self.generations - 1:
                break
            children = self._make_new_pop(population, mut_rate)

            # 4. 评估子代
            if verbose:
                print(f"  评估 {len(children)} 个子代...")
            for i, ind in enumerate(children):
                if not ind.evaluated:
                    children[i] = evaluate_individual(ind, self.all_data)
                if verbose and (i + 1) % 10 == 0:
                    print(f"    评估 {i+1}/{len(children)}")

            # 5. 合并父子 + 选择下一代 (μ+λ)
            combined = population + children
            combined_fronts = fast_non_dominated_sort(combined)
            for front in combined_fronts:
                crowding_distance_assignment(combined, front)

            # 按 (rank, -crowding) 选前 pop_size 个
            combined.sort(key=lambda x: (x.rank, -x.crowding_distance))
            population = combined[:self.pop_size]

        # 最终前沿
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance_assignment(population, front)
        self.pareto_front = [population[i] for i in fronts[0]]

        # 把前沿写入知识库记忆 (避免下次重复探索)
        for ind in self.pareto_front:
            self.kb.save_best_gene(
                genes=ind.genes,
                fitness=ind.fitness_scalar,
                objectives={
                    "sharpe": ind.sharpe,
                    "win_rate": ind.win_rate,
                    "n_trades": ind.n_trades,
                    "monthly_loss": ind.monthly_loss,
                    "profit_loss_ratio": ind.profit_loss_ratio,
                },
                kpi_pass=ind.kpi_pass,
                generation=self.generations,
                notes=f"NSGA-II v515 front1, KPI={ind.kpi_pass}/7",
            )
        self.kb.save_memory()
        if verbose:
            print(f"\n  [记忆] 帕累托前沿 {len(self.pareto_front)} 个解已写入知识库")

        return self.pareto_front


# ============================================================
# 8. 主入口
# ============================================================
def main():
    """主入口: 加载数据 + 运行 NSGA-II + 输出结果"""
    import argparse
    parser = argparse.ArgumentParser(description="v515 NSGA-II 多目标进化引擎")
    parser.add_argument("--pop", type=int, default=50, help="种群大小")
    parser.add_argument("--gen", type=int, default=20, help="进化代数")
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
                        help="标的列表")
    args = parser.parse_args()

    print("=" * 70)
    print("v515 NSGA-II 多目标进化引擎 — Agent-7")
    print("=" * 70)
    print(f"策略: 动量 + 均值回归(严格/放宽) + 突破(压缩释放) [复用 v515]")
    print(f"算法: NSGA-II (快速非支配排序 + 拥挤距离)")
    print(f"目标: 5 个 (夏普↑, 胜率↑, 交易数↑, 月亏↓, 盈亏比↑)")
    print(f"种群: {args.pop}, 代数: {args.gen}")
    print(f"标的: {args.symbols}")
    print()

    # 加载数据
    all_data: Dict[str, Tuple] = {}
    for sym in args.symbols:
        print(f"加载 {sym}...", end=" ")
        bars = load_klines(sym)
        if not bars:
            print("FAIL, 跳过")
            continue
        indicators = calc_indicators(bars)
        all_data[sym] = (bars, indicators)
        print(f"OK ({len(bars)} bars)")

    if not all_data:
        print("[ERROR] 无可用数据, 退出")
        return

    # 运行 NSGA-II
    engine = NSGA2Engine(
        all_data=all_data,
        pop_size=args.pop,
        generations=args.gen,
    )
    pareto_front = engine.run(verbose=True)

    # 输出结果
    print("\n" + "=" * 70)
    print("v515 NSGA-II 最终结果 — 帕累托前沿")
    print("=" * 70)
    print(f"前沿大小: {len(pareto_front)}")
    print()
    print(f"{'#':>3} {'夏普':>7} {'胜率':>7} {'交易数':>7} {'月亏':>7} {'盈亏比':>7} {'KPI':>5} {'年化':>8} {'回撤':>7}")
    for i, ind in enumerate(pareto_front):
        print(f"{i+1:>3} {ind.sharpe:>7.3f} {ind.win_rate*100:>6.1f}% "
              f"{ind.n_trades:>7.1f} {ind.monthly_loss*100:>6.1f}% "
              f"{ind.profit_loss_ratio:>7.2f} {ind.kpi_pass:>3}/7 "
              f"{ind.annual_return*100:>7.1f}% {ind.max_dd*100:>6.1f}%")

    # 保存帕累托前沿
    result = {
        "version": "v515-nsga2",
        "timestamp": time.time(),
        "pop_size": args.pop,
        "generations": args.gen,
        "objectives": list(OBJECTIVE_DIRECTIONS.keys()),
        "objective_directions": OBJECTIVE_DIRECTIONS,
        "pareto_front_size": len(pareto_front),
        "pareto_front": [
            {
                "genes": ind.genes,
                "objectives": {
                    "sharpe": ind.sharpe,
                    "win_rate": ind.win_rate,
                    "n_trades": ind.n_trades,
                    "monthly_loss": ind.monthly_loss,
                    "profit_loss_ratio": ind.profit_loss_ratio,
                },
                "kpi": {
                    "kpi_pass": ind.kpi_pass,
                    "max_dd": ind.max_dd,
                    "annual_return": ind.annual_return,
                    "consec_loss": ind.consec_loss,
                    "win_rate_ci_lo": ind.win_rate_ci_lo,
                    "is_sharpe": ind.is_sharpe,
                    "df": ind.df,
                },
                "fitness_scalar": ind.fitness_scalar,
            }
            for ind in pareto_front
        ],
        "history": engine.history,
        "targets": TARGETS,
    }
    out_path = HERE / "_v515_pareto_front.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n帕累托前沿已保存: {out_path}")


if __name__ == "__main__":
    main()
