# -*- coding: utf-8 -*-
"""
行为多样性增强模块 (Behavior Diversity Enhancement) — Novelty Search + MAP-Elites

v2.7新增（来源：Lehman & Stanley 2011 Novelty Search + Mouret & Clune 2015 MAP-Elites）：
  1. Novelty Search: 奖励行为新颖的Agent，防止种群收敛到单一策略
  2. MAP-Elites: 4维行为特征网格，每个niche保留Top-1，实现Quality-Diversity

核心思想：
  - 传统Fitness只奖励"表现好"的Agent → 种群收敛到单一最优策略 → 实盘崩溃
  - Novelty Search奖励"行为新颖"的Agent → 维持行为多样性 → 实盘适应性强
  - MAP-Elites在行为空间网格中保留每个niche的最优 → 策略图鉴 → regime-aware选策略

行为特征向量（5维）：
  1. 策略类型索引（worldview_primary的哈希映射到[0,1]）
  2. 持仓时长（exit_risk_max_hold_hours归一化）
  3. 交易频率（1/(1+max_trades_per_day)归一化）
  4. 风险敞口（position_sizing_max_position_pct）
  5. 信号敏感度（macd_sensitivity归一化）

引用来源：
  - Lehman & Stanley, "Abandoning Objectives", Evolutionary Computation 2011
  - Mouret & Clune, "Illuminating Behavior Spaces", MAP-Elites arXiv:1504.04909
  - Kistemaker & Whiteson, GECCO 2011 — Novelty Search关键因素
  - de Witt & Pakkanen, MAP-Elites for Optimal Execution, arXiv:2601.22113 (2026)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 行为特征向量计算
# ============================================================================


# 策略类型到索引的映射（用于行为特征第1维）
WORLDVIEW_INDEX: Dict[str, float] = {
    "trend_following": 0.0,
    "mean_reversion": 0.1,
    "breakout": 0.2,
    "momentum": 0.3,
    "value_investing": 0.4,
    "safe_margin": 0.5,
    "stat_arbitrage": 0.6,
    "risk_parity": 0.7,
    "multi_strategy": 0.8,
    "market_making": 0.9,
}


def compute_behavior_vector(agent: Any) -> np.ndarray:
    """计算Agent的5维行为特征向量

    行为特征（Behavior Characterization, BC）选择依据：
    Kistemaker & Whiteson GECCO 2011 — 策略类型/持仓时长/交易频率是关键维度
    Jackson & Daley 2019 — 动作序列特征有效但计算成本高，这里用代理特征

    Args:
        agent: AgentGene实例

    Returns:
        5维归一化行为特征向量，每维在[0, 1]
    """
    # 1. 策略类型索引
    wv_idx = WORLDVIEW_INDEX.get(agent.worldview_primary, 0.5)

    # 2. 持仓时长归一化（exit_risk_max_hold_hours: 1-720 → 0-1）
    holding = min(1.0, max(0.0, getattr(agent, "exit_risk_max_hold_hours", 48) / 720.0))

    # 3. 交易频率代理（aggressiveness越高=越频繁）
    aggressiveness = getattr(agent, "entry_logic_entry_aggressiveness", 0.5)

    # 4. 风险敞口（position_sizing_max_position_pct: 0.05-0.5 → 0-1）
    risk = min(1.0, max(0.0, (getattr(agent, "position_sizing_max_position_pct", 0.25) - 0.05) / 0.45))

    # 5. 信号敏感度（macd_sensitivity: 0.001-0.01 → 0-1，反向：越敏感=值越小=越接近1）
    macd_sens = getattr(agent, "entry_logic_macd_sensitivity", 0.005)
    sensitivity = min(1.0, max(0.0, (0.01 - macd_sens) / 0.009))

    return np.array([wv_idx, holding, aggressiveness, risk, sensitivity], dtype=np.float64)


# ============================================================================
# Novelty Search Archive
# ============================================================================


class NoveltyArchive:
    """Novelty Search档案 — 维护历史行为特征点

    来源：Lehman & Stanley 2011 — 维护一个行为特征档案，
    计算当前Agent到档案中k个最近邻的平均距离作为新颖性度量。

    参数选择依据：
    - k=15: 经典值（Lehman & Stanley 2011推荐）
    - archive_max=200: 防止档案爆炸，保留真正新颖的个体
    - add_probability=0.05: 5%概率添加新个体到档案
    """

    def __init__(self, k: int = 15, archive_max: int = 200, add_probability: float = 0.05):
        self.k = k
        self.archive_max = archive_max
        self.add_probability = add_probability
        self._archive: List[np.ndarray] = []
        self._rng = np.random.default_rng(42)  # 确定性RNG

    def compute_novelty(self, behavior_vec: np.ndarray) -> float:
        """计算行为特征向量的新颖性分数

        Novelty(b) = (1/k) × Σᵢ₌₁ᵏ || b - bᵢⁿⁿ ||₂

        如果档案为空或少于k个点，返回最大新颖性1.0。

        Args:
            behavior_vec: 5维行为特征向量

        Returns:
            新颖性分数 [0, +∞)，归一化到[0, 1]
        """
        if len(self._archive) == 0:
            return 1.0

        # 计算到档案中所有点的欧氏距离
        archive_array = np.array(self._archive)
        distances = np.linalg.norm(archive_array - behavior_vec, axis=1)

        # 取k个最近邻的平均距离
        k_actual = min(self.k, len(distances))
        if k_actual == 0:
            return 1.0

        # 取最近的k个距离
        k_nearest = np.sort(distances)[:k_actual]
        novelty = np.mean(k_nearest)

        # 归一化到[0, 1]（行为特征每维在[0,1]，最大距离=√5≈2.236）
        max_possible = math.sqrt(5)
        return min(1.0, novelty / max_possible)

    def maybe_add(self, behavior_vec: np.ndarray, novelty: float) -> None:
        """以一定概率将行为特征添加到档案

        只有新颖性足够高的个体才被添加，防止档案被普通个体淹没。
        来源：Lehman & Stanley 2011 — 动态新颖性阈值
        """
        # 档案未满时直接添加
        if len(self._archive) < self.archive_max:
            self._archive.append(behavior_vec.copy())
            return

        # 档案已满时，只有新颖性高于平均的个体才替换最旧的
        if novelty > 0.3 and self._rng.random() < self.add_probability:
            # 替换档案中最旧的个体（FIFO）
            self._archive.pop(0)
            self._archive.append(behavior_vec.copy())

    def clear(self) -> None:
        """清空档案（每轮进化后重置）"""
        self._archive.clear()

    def __len__(self) -> int:
        return len(self._archive)


# ============================================================================
# MAP-Elites 网格
# ============================================================================


class MAPElitesGrid:
    """MAP-Elites网格 — 4维行为特征网格，每个niche保留Top-1

    来源：Mouret & Clune 2015 — 在行为空间网格中保留每个niche的最优个体，
    形成策略图鉴（repertoire），实现Quality-Diversity。

    4维行为特征（从5维中选取最关键的4维）：
    1. 策略类型（10格）
    2. 持仓时长（5格）
    3. 风险敞口（5格）
    4. 信号敏感度（5格）

    总格子数：10×5×5×5 = 1250个niche
    """

    def __init__(
        self,
        bins_strategy: int = 10,
        bins_holding: int = 5,
        bins_risk: int = 5,
        bins_sensitivity: int = 5,
    ):
        self.bins_strategy = bins_strategy
        self.bins_holding = bins_holding
        self.bins_risk = bins_risk
        self.bins_sensitivity = bins_sensitivity

        # 网格：{cell_key: (agent_id, gt_score)}
        self._grid: Dict[Tuple[int, int, int, int], Tuple[str, float]] = {}

    def _get_cell_key(self, behavior_vec: np.ndarray) -> Tuple[int, int, int, int]:
        """将行为特征向量映射到网格坐标"""
        # 1. 策略类型（第0维）
        s_idx = min(self.bins_strategy - 1, int(behavior_vec[0] * self.bins_strategy))

        # 2. 持仓时长（第1维）
        h_idx = min(self.bins_holding - 1, int(behavior_vec[1] * self.bins_holding))

        # 3. 风险敞口（第3维）
        r_idx = min(self.bins_risk - 1, int(behavior_vec[3] * self.bins_risk))

        # 4. 信号敏感度（第4维）
        sens_idx = min(self.bins_sensitivity - 1, int(behavior_vec[4] * self.bins_sensitivity))

        return (s_idx, h_idx, r_idx, sens_idx)

    def update(self, agent_id: str, behavior_vec: np.ndarray, gt_score: float) -> bool:
        """更新网格中的个体

        如果该格子为空或新个体的GT-Score更高，则替换。

        Returns:
            True如果被接受（新格子或替换了旧个体）
        """
        cell_key = self._get_cell_key(behavior_vec)

        if cell_key not in self._grid:
            # 新格子，直接添加
            self._grid[cell_key] = (agent_id, gt_score)
            return True

        # 已有个体，比较GT-Score
        _, existing_score = self._grid[cell_key]
        if gt_score > existing_score:
            self._grid[cell_key] = (agent_id, gt_score)
            return True

        return False

    def get_diverse_elite_ids(self, top_n: int) -> List[str]:
        """获取行为多样化的精英ID列表

        从不同niche中选取GT-Score最高的N个个体。
        来源：Mouret & Clune 2015 — MAP-Elites输出策略图鉴
        """
        if not self._grid:
            return []

        # 按GT-Score降序排序所有格子中的个体
        sorted_cells = sorted(self._grid.values(), key=lambda x: -x[1])

        # 返回前N个不同niche的agent_id
        return [agent_id for agent_id, _ in sorted_cells[:top_n]]

    def get_cell_count(self) -> int:
        """获取已占据的格子数"""
        return len(self._grid)

    def clear(self) -> None:
        """清空网格"""
        self._grid.clear()


# ============================================================================
# 行为多样性管理器
# ============================================================================


class BehaviorDiversityManager:
    """行为多样性管理器 — 集成Novelty Search + MAP-Elites

    使用方式：
      1. 每轮进化开始前调用 update_archive(population) 更新档案
      2. 对每个Agent调用 compute_novelty_score(agent) 获取新颖性分数
      3. 将新颖性分数加权到GT-Score: final = 0.7*gt + 0.3*novelty
      4. 每轮进化结束后调用 update_elite_grid(population, scores) 更新MAP-Elites网格
      5. get_elite时优先从MAP-Elites网格获取多样化精英
    """

    def __init__(
        self,
        novelty_weight: float = 0.3,  # v2.17.2: 从0.2提升到0.3，增加diversity和探索
        k_neighbors: int = 15,
        archive_max: int = 200,
    ):
        self.novelty_weight = novelty_weight
        self.gt_weight = 1.0 - novelty_weight

        self.archive = NoveltyArchive(k=k_neighbors, archive_max=archive_max)
        self.elite_grid = MAPElitesGrid()

        # 缓存行为特征向量（避免重复计算）
        self._behavior_cache: Dict[str, np.ndarray] = {}

    def compute_behavior_vector(self, agent: Any) -> np.ndarray:
        """计算Agent的行为特征向量（带缓存）"""
        if agent.agent_id not in self._behavior_cache:
            self._behavior_cache[agent.agent_id] = compute_behavior_vector(agent)
        return self._behavior_cache[agent.agent_id]

    def update_archive(self, population: List[Any]) -> None:
        """每轮进化开始前更新Novelty Archive

        从当前种群中采样个体添加到档案。
        """
        for agent in population:
            bv = self.compute_behavior_vector(agent)
            novelty = self.archive.compute_novelty(bv)
            self.archive.maybe_add(bv, novelty)

    def compute_novelty_score(self, agent: Any) -> float:
        """计算Agent的Novelty Score"""
        bv = self.compute_behavior_vector(agent)
        return self.archive.compute_novelty(bv)

    def apply_novelty_to_scores(self, population: List[Any], scores: Dict[str, Any],
                                overfitted_ids: Optional[set] = None) -> None:
        """将Novelty Score加权到GT-Score (v4.0 Phase 1: 乘法门控重构)

        v4.0 公式：final_gt = original_gt × diversity_gate
          diversity_gate = 1.0 + 0.2 × norm_novelty ∈ [1.0, 1.2]
        （乘法门控：novelty只boost已有性能，不替代性能）

        来源：MENS (Mixed Evolutionary Novelty Search) + 乘法适应度门控
              (Lehman & Stanley 2011 + Multiplicative Gating)

        v4.0 Phase 1 修复: 旧加法混合 (0.7*gt + 0.3*novelty*100) 导致:
          - 高novelty/低performance的agent得分虚高
          - 与GT-Score乘法门控重构不一致 (additive bonus已全部改为multiplicative)
          - 零性能agent可通过novelty获得正分 (绕过base_score=0铁律)
        修复: 改为乘法门控，novelty仅boost已有性能，base_score=0 → boost=0

        v2.20 CRITICAL修复：添加overfitted_ids参数，过拟合Agent不得获得novelty bonus
        原问题：novelty bonus绕过PSR/DSR门控，过拟合策略可通过新颖性获得分数
        修复：过拟合Agent跳过novelty bonus，关闭过拟合逃逸通道
        """
        # 先归一化Novelty Score到群体范围
        novelties = {}
        for agent in population:
            novelties[agent.agent_id] = self.compute_novelty_score(agent)

        if not novelties:
            return

        max_novelty = max(novelties.values()) if novelties else 1.0
        if max_novelty < 1e-10:
            return  # 所有新颖性为0，跳过

        for agent in population:
            s = scores.get(agent.agent_id)
            if s is None:
                continue

            # v2.20 CRITICAL: 过拟合Agent跳过novelty bonus
            # 关闭过拟合逃逸通道——过拟合策略不得通过新颖性获得分数
            if overfitted_ids and agent.agent_id in overfitted_ids:
                continue

            # 归一化novelty到[0, 1]
            norm_novelty = novelties[agent.agent_id] / max_novelty

            # v4.0 Phase 1: 乘法 diversity_gate (替代旧加法混合)
            # 铁律: gt_score=0 → diversity_gate×0=0 (novelty不创造分数，只boost)
            # 范围: diversity_gate ∈ [1.0, 1.2] (最多+20% boost)
            if s.gt_score > 0:
                # 交易盈利Agent：乘法boost (最多+20%)
                diversity_gate = 1.0 + 0.2 * norm_novelty
                s.gt_score = s.gt_score * diversity_gate
                agent.fitness_score = s.gt_score
            elif s.total_trades == 0:
                # v2.41 CRITICAL 修复 (用户反馈: "沙盘的优势应该是agent一代只会比一代强,
                #   不知道哪里出了问题" — 进化机制失效根因)
                #
                # 根因 (v2.17.2 引入的回归):
                #   原代码 s.gt_score = max(0.0, exploration_bonus - 2.0)
                #   把零交易 agent 的 gt_score 从 compute_fitness() 返回的 -5.0
                #   覆写成 0~13 的正数, 导致:
                #     1. 零交易 agent 进入精英层 (正分 > 真正交易但亏损的 agent)
                #     2. 交叉变异产生下一代零交易 agent
                #     3. 形成"不交易 > 亏损交易"的反向选择压力
                #     4. v316/v323 数据显示 87.2%~99.1% 零交易率
                #     5. fitness-PnL 相关系数 = 0.020 (几乎无关系)
                #
                # 修复原则 (与用户 anti-overfitting 警告一致):
                #   "不要因有的真实数据去委曲求全然后贴合数据做调整,
                #    那样还是过拟合, 还是会模拟牛逼, 实盘亏钱!"
                #   v2.17.2 的"探索奖励"就是为了让更多 agent 有正分而妥协 → 过拟合
                #
                # 修复方案:
                #   1. 零交易 agent 的 gt_score 必须保持负值 (淘汰压力)
                #   2. 保留 novelty 的区分度 (零交易 agent 之间仍有差异)
                #      但区分度不允许跨越 0 阈值 (不允许变成正数)
                #   3. 范围: -5.0 (最不新颖) ~ -4.5 (最新颖)
                #      -5.0 来自 compute_fitness() 的零交易惩罚 (population.py:265-267)
                #      +0.5 是 novelty 提供的轻微区分度 (不足以跨越 0)
                #   4. 真正交易但亏损的 agent (gt_score ≤ 0) 仍然优先于零交易 agent
                #      因为 -5.0 < 任何交易 agent 的 gt_score (即使亏损也被 min(.,0) 限制)
                #
                # 验证: 见 _verify_v241_zero_trade_penalty.py
                exploration_bonus = norm_novelty * 0.5  # 0~0.5 的轻微区分度
                s.gt_score = -5.0 + exploration_bonus   # 范围 -5.0 ~ -4.5 (始终为负)
                agent.fitness_score = s.gt_score

    def update_elite_grid(
        self,
        population: List[Any],
        scores: Dict[str, Any],
    ) -> None:
        """每轮进化结束后更新MAP-Elites网格"""
        self.elite_grid.clear()

        for agent in population:
            s = scores.get(agent.agent_id)
            if s is None:
                continue
            bv = self.compute_behavior_vector(agent)
            self.elite_grid.update(agent.agent_id, bv, s.gt_score)

    def get_diverse_elite(
        self,
        population: List[Any],
        top_n: int,
    ) -> List[Any]:
        """从MAP-Elites网格获取行为多样化的精英

        如果网格中个体不足，回退到GT-Score排名。
        """
        diverse_ids = set(self.elite_grid.get_diverse_elite_ids(top_n))

        result: List[Any] = []
        result_ids: set = set()

        # 从种群中收集网格中的精英
        if diverse_ids:
            for agent in population:
                if agent.agent_id in diverse_ids:
                    result.append(agent)
                    result_ids.add(agent.agent_id)

        # 如果不足top_n，按GT-Score排名补充
        if len(result) < top_n:
            sorted_pop = sorted(
                population,
                key=lambda a: a.fitness_score,
                reverse=True,
            )
            for agent in sorted_pop:
                if len(result) >= top_n:
                    break
                if agent.agent_id not in result_ids:
                    result.append(agent)
                    result_ids.add(agent.agent_id)

        return result[:top_n]

    def clear_cache(self) -> None:
        """清空行为特征缓存（每轮进化后调用）"""
        self._behavior_cache.clear()
