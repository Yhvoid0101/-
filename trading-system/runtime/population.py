# -*- coding: utf-8 -*-
"""
种群管理器 (Population Manager) — 子Agent种群的生成、进化、淘汰 v2.25

管理50-200个子Agent的完整生命周期：
  - 初始种群生成（随机基因 + 种子库注入）
  - 适应度排名（GT-Score：夏普+回撤+胜率+盈亏比）
  - 精英保留 + 锦标赛选择
  - 交叉变异产生下一代
  - 多样性维护（避免近亲繁殖）
  - 灭绝与重生（淘汰连续亏损的个体）
"""

from __future__ import annotations

import copy
import random
import uuid
import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .gene_codec import AgentGene, WORLDVIEW_SEEDS
from .gene_generation import generate_random_gene, generate_seed_based_gene
from .gene_evolution import crossover, gene_similarity_from_vectors
from .gene_io import genes_from_jsonl, genes_to_jsonl
# P1-3优化: 集成NSGA-III多目标选择器(4目标帕累托前沿覆盖更均匀)
# 注意: 延迟导入避免循环依赖(nsga3_selector内部导入population)
# R12修复: 真实try-import替代硬编码True,避免导入失败时仍声称可用
try:
    # 仅做语法检查式导入,实际调用时再延迟导入
    import importlib.util as _importlib_util
    _nsga3_spec = _importlib_util.find_spec(".nsga3_selector", package=__package__)
    _NSGA3_AVAILABLE = _nsga3_spec is not None
except (ImportError, ModuleNotFoundError, ValueError):
    _NSGA3_AVAILABLE = False
# P0-1确定性事件驱动重构：使用确定性RNG替代模块级random
# 保证同一代数+同一种群→同一进化结果→可复现的进化对比
from .deterministic_rng import global_rng_manager
# v2.7 行为多样性增强（Novelty Search + MAP-Elites）
from .behavior_diversity import BehaviorDiversityManager
# v499 P0-3修复: 集成反过拟合验证器 (970行完整实现但从未被调用 → 现在打通)
# 来源: anti_overfitting.py 5大验证器 (Walk-Forward/蒙特卡洛/真实滑点/PBO/参数敏感性)
# 这是"沙盘赚钱实盘亏钱"的根源修复 — Bailey 2014 "Probability of Backtest Overfitting"
from .anti_overfitting import AntiOverfittingValidator

logger = logging.getLogger(__name__)


def _anti_overfit_metrics(pnls: List[float]) -> Dict[str, float]:
    values = np.asarray(pnls, dtype=float)
    if values.size == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "return": 0.0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sharpe = mean / std * math.sqrt(values.size) if std > 1e-10 else 0.0
    wins = values[values > 0]
    losses = values[values < 0]
    loss_total = abs(float(losses.sum()))
    profit_factor = float(wins.sum()) / loss_total if loss_total > 1e-10 else (float("inf") if wins.size else 0.0)
    equity = 10000.0 + np.cumsum(values)
    peaks = np.maximum.accumulate(np.insert(equity, 0, 10000.0))[1:]
    drawdown = float(np.max((peaks - equity) / np.maximum(peaks, 1e-10))) if equity.size else 0.0
    return {
        "sharpe": float(sharpe),
        "max_drawdown": drawdown,
        "win_rate": float(np.mean(values > 0)),
        "profit_factor": profit_factor,
        "return": float(values.sum()),
    }


# ============================================================================
# GT-Score 计算
# ============================================================================






# ============================================================================
# NSGA-II 多目标帕累托排序 (基于 Deb et al, 2002)
# ============================================================================










# ============================================================================
# Feature Map 行为多样性维护 (基于 QuantEvolve, 2025)
# ============================================================================


# ============================================================================
# 种群管理器
# ============================================================================


@dataclass(slots=True)
class PopulationStats:
    """种群统计信息"""

    generation: int = 0
    population_size: int = 0
    mean_gt_score: float = 0.0
    median_gt_score: float = 0.0
    max_gt_score: float = 0.0
    max_gt_score_raw: float = 0.0  # Phase 14.16p: 惩罚前原始max_gt_score
    min_gt_score: float = 0.0
    mean_sharpe: float = 0.0
    mean_win_rate: float = 0.0
    mean_drawdown: float = 0.0
    diversity: float = 0.0  # 平均基因相似度（越低越多样）
    elite_count: int = 0
    eliminated_count: int = 0
    new_born_count: int = 0
    worldview_distribution: Dict[str, int] = field(default_factory=dict)
    # R8进化升级：NSGA-II + Feature Map统计
    pareto_front_size: int = 0  # 帕累托前沿大小（Front 0的个体数）
    feature_map_cells: int = 0  # Feature Map占据的格子数
    # R9进化升级：Walk-forward抗过拟合统计
    overfit_detected_count: int = 0  # 检测到过拟合的Agent数
    mean_wf_ratio: float = 0.0  # v2.23: 默认0.0（无数据=异常），<0.5=过拟合，>0.7=健康
    wf_valid_count: int = 0
    wf_negative_count: int = 0
    wf_status: str = "insufficient_trades"
    adaptive_mutation_rate: float = 0.0  # 本轮自适应变异率
    random_injection_count: int = 0  # 本轮随机注入的个体数
    timestamp: float = field(default_factory=time.time)


class PopulationManager:
    """种群管理器 — 管理子Agent种群的完整生命周期

    使用方式:
      pm = PopulationManager(population_size=50)
      pm.initialize_population()
      pm.run_evolution_round(trades_by_agent)  # 每轮沙盘结束后调用
      elite = pm.get_elite(5)  # 获取精英前5
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_ratio: float = 0.50,  # Phase 14.10: 0.20→0.50 (elite_size 1→4, 精英积累硬约束)
        mutation_boost_threshold: float = 0.3,
        min_diversity: float = 0.3,
        data_dir: str = "",
        n_islands: int = 4,
        migration_interval: int = 5,
        migration_rate: float = 0.10,
        use_nsga2: bool = True,
        use_nsga3: bool = True,  # R12修复: NSGA-III默认启用(4目标帕累托前沿覆盖更均匀)
        use_feature_map: bool = True,
        max_generations: int = 0,
        live_mode: bool = False,  # v2.19 P0: 实盘硬门控开关
    ):
        """
        Args:
            population_size: 种群规模
            elite_ratio: 精英保留比例
            mutation_boost_threshold: 多样性低于此阈值时提高变异率
            min_diversity: 最小多样性阈值
            data_dir: 数据存储目录
            n_islands: 岛屿数量（岛屿模型，来自QuantEvolve）
            migration_interval: 每N代迁移一次
            migration_rate: 迁移比例（每岛迁移出去的比例）
            use_nsga2: 是否使用NSGA-II多目标帕累托排序
            use_feature_map: 是否使用Feature Map行为多样性维护
            max_generations: 最大进化代数（v2.6新增，用于分阶段变异率）
            live_mode: v2.19 P0 实盘模式硬门控
                False(默认)=沙盘模式：过拟合Agent GT×0.5软惩罚，仍参与进化
                True=实盘模式：过拟合Agent GT=-999硬淘汰，禁止进入精英
                核心铁律："ai量化技能包不要模拟厉害，实盘亏钱！"
                来源：Bailey & López de Prado 2014 DSR + 1500错误知识库
        """
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_boost_threshold = mutation_boost_threshold
        self.min_diversity = min_diversity
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "sandbox_trading"
        )
        # v2.19 P0: 实盘硬门控开关
        self._live_mode = live_mode
        # v2.35 日内交易模式开关 — 控制 crossover 是否应用日内硬约束 clamp
        # 来源: 500轮进化诊断 — 变异破坏初始日内约束,74% 后代违反 max_hold<=24
        self._intraday_mode: bool = False

        self.population: List[AgentGene] = []
        self.generation: int = 0
        self.history: List[PopulationStats] = []
        self._elite_size = max(1, int(population_size * elite_ratio))

        # v4.0 Phase 6 Task #29.2: ERR教训注入映射 (闭环验证用)
        # 用户铁律: "迭代经验和教训库使用起来，而不是和复盘一样，在那儿摆看！"
        # _err_injection_count: ERR教训注入crossover的次数 (L7元验证用)
        # _gene_err_map: {child_key: [err_id, ...]} 记录每个子代基因携带的ERR教训
        #   child_key 格式: "gen{N}_idx{M}" (因为 AgentGene 是 slots=True, 不能动态加字段)
        self._err_injection_count: int = 0
        self._gene_err_map: Dict[str, List[str]] = {}

        # NSGA-II + Feature Map (R8进化升级)
        self.use_nsga2 = use_nsga2
        # P1-3优化: NSGA-III多目标选择(4目标帕累托前沿覆盖更均匀)
        self.use_nsga3 = use_nsga3 and _NSGA3_AVAILABLE  # 仅在可用时启用
        # Phase 8.1: NSGA3Selector instance for regime weight adaptation
        self._nsga3_selector = None
        self._current_regime = "neutral"  # default regime
        if self.use_nsga3:
            try:
                from .nsga3_selector import NSGA3Selector
                self._nsga3_selector = NSGA3Selector()
                logger.info("Phase 8.1: NSGA3Selector initialized with regime weight adaptation")
            except Exception as e:
                logger.warning("Phase 8.1: NSGA3Selector init failed, using function-style: %s", e)
                self._nsga3_selector = None
        if use_nsga3 and not _NSGA3_AVAILABLE:
            logger.warning("NSGA-III不可用(nsga3_selector导入失败),回退到NSGA-II")
        self.use_feature_map = use_feature_map
        self.feature_map = FeatureMap() if use_feature_map else None

        # P1-2优化: 进化增强集成器(Arena Mode+换手率+HHI)
        # R12修复: 默认启用(消除空壳交付),惩罚拥挤/过度交易/因子集中
        try:
            from .evolution_enhancements import EvolutionEnhancements
            self.enhancements = EvolutionEnhancements(enabled=True)
        except ImportError:
            self.enhancements = None
            logger.warning("EvolutionEnhancements不可用,Arena/换手率/HHI惩罚未启用")

        # 岛屿模型 (来自QuantEvolve)
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self._islands: List[List[AgentGene]] = []

        # Walk-forward验证 (来自Stratevo / Robert Pardo WFO)
        # 核心思想：训练集上优化的参数必须在未见过的测试集上验证
        # 过拟合惩罚 = test_score / train_score，<0.5视为严重过拟合
        self._wf_train_scores: Dict[str, float] = {}
        self._wf_test_scores: Dict[str, float] = {}
        # execution_realism_penalty: 存储最近一轮的scores和trades供evolution_loop重评估
        self._last_scores: Dict[str, FitnessScore] = {}
        self._last_trades_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        self._overfit_penalty: Dict[str, float] = {}
        # Phase 14.16f: 滑动窗口PnL历史 (每个agent最近N代的total_pnl)
        # 用于硬门控判断: rolling_avg_pnl<0 才衰减(不归零)
        self._agent_pnl_history: Dict[str, list] = {}
        # Phase 14.17: 滑动窗口GT-Score历史 (每个agent最近N代的gt_score)
        # 用于gt_score平滑: 防止数据段红利误判导致下一代崩溃
        self._agent_gt_history: Dict[str, list] = {}
        self._rolling_window_size: int = 5  # Phase 14.17: 3→5 增大窗口更稳健
        # Phase 14.16o: HIGH GT基因存档 — 防止HIGH GT基因跨代丢失
        # 根因: gen 27,44,71的HIGH GT个体到gen 90-99完全消失
        # 修复: gt_score>10的top-1个体deepcopy存档, 每10代回注
        self._high_gt_archive: list = []  # [(gt_score, agent, gen), ...]
        self._high_gt_threshold: float = 10.0  # HIGH GT阈值
        self._high_gt_inject_interval: int = 3  # 每3代检查一次精英回注
        self._high_gt_inject_k: int = 1  # Phase N: 单代最多回注一个,保留探索空间
        self._high_gt_inject_cooldown: int = 9  # 同一基因指纹9代内不重复回注
        self._high_gt_last_injected: Dict[str, int] = {}  # gene_hash -> generation
        # Phase 14.16p: 惩罚前原始max_gt_score (每代重置)
        self._current_max_gt_raw: float = 0.0
        self.use_walk_forward: bool = True
        self.wf_train_ratio: float = 0.7  # 训练集比例

        # v499 P0-3修复: 反过拟合验证器实例 (970行完整实现终于被打通)
        # 来源: anti_overfitting.py AntiOverfittingValidator (5大验证器)
        # 职责: 在run_evolution_round中对每个agent的trades执行综合反过拟合验证
        #       过拟合策略 → gt_score × 0.3 严厉惩罚 (Bailey 2014 PBO理论)
        #       这是"沙盘赚钱实盘亏钱"的根源修复
        # Phase R: 沙盘进化使用有界Monte Carlo预算, 保留OOS/PBO/滑点验证但避免单代无限计算
        # 生产环境仍由独立部署验证器使用完整默认预算
        _sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
        if _sandbox_mode:
            from .anti_overfitting import MonteCarloPermutationTest
            self.anti_overfitting_validator = AntiOverfittingValidator(
                monte_carlo=MonteCarloPermutationTest(n_iterations=200)
            )
        else:
            self.anti_overfitting_validator = AntiOverfittingValidator()
        self.wf_overfit_threshold: float = 0.7  # v3.12: 0.5→0.7 收紧过拟合检测阈值 (DF退化64.6%根因)

        # v2.43 实盘预测 PnL 适应度 (用户核心需求: 杜绝"模拟牛逼实盘亏钱")
        # 来源: _v243_live_pnl_fitness.py (LivePnLFitnessEvaluator)
        # 原理: 在模拟PnL上应用6大真实成本(spread/slippage/liquidity/latency/fee/failure),
        #       得到 live-predicted PnL, 然后基于 live PnL 计算 GT-Score
        # 用户警告: "不要因有的真实数据去委曲求全然后贴合数据做调整, 那样还是过拟合"
        # 默认 False (向后兼容), 通过配置启用
        self.use_live_pnl_fitness: bool = False
        self._live_pnl_evaluator = None  # 延迟初始化

        # v2.6 分阶段变异率（来源：CSDN 2026-06 GA最佳实践）
        self.max_generations: int = max_generations

        # v2.7 进化停滞检测与突破（来源：jestersimmps Evolving Trader 2026 + Rajabi & Witt GECCO 2020）
        # 核心思想：检测连续N代无改进，触发分级突破策略
        #   N=8: 激进变异模式（变异率×5）
        #   N=16: 横向移动接受（GT-Score≥0.7×best即保留）
        #   N=25: IPOP式重启（保留Top-5精英，其余重新随机+种子库注入）
        self.best_gt_score_history: List[float] = []  # 每代最佳GT-Score历史
        self.generations_since_improvement: int = 0  # 自上次显著改进以来的代数
        self.stagnation_improvement_threshold: float = 0.01  # 1%改进才算显著改进
        self.stagnation_level1: int = 8   # 触发激进变异
        self.stagnation_level2: int = 16  # 触发横向移动接受
        self.stagnation_level3: int = 25  # 触发IPOP式重启

        # v2.7 行为多样性增强（Novelty Search + MAP-Elites）
        # 来源：Lehman & Stanley 2011 + Mouret & Clune 2015
        # 核心思想：奖励行为新颖的Agent，在行为空间网格中保留每个niche的最优
        # v2.17.2调优：novelty_weight=0.3，平衡GT-Score权重和diversity
        # 理由：v2.17.1的0.2权重导致diversity=0.156偏低，提升到0.3增加探索
        #       不交易Agent exploration权重0.15，让更多Agent有正GT-Score
        self.behavior_diversity = BehaviorDiversityManager(
            novelty_weight=0.3,  # v2.17.2: 0.2→0.3，增加diversity
            k_neighbors=15,
            archive_max=200,
        )

        # v2.4 高级反过拟合门控注入的过拟合Agent集合
        # 来源：Bailey & López de Prado DSR/PSR/MinBTL
        # 被标记的Agent在进化时强制淘汰，不参与交叉变异，防止过拟合基因传播
        self._overfitted_agents: set = set()
        # v2.22: IPOP restart触发标志，evolution_loop检测后重置DSR状态
        self._restart_triggered: bool = False

        # R13优化: Live Reality Check WARN Agent集合(轻度惩罚)
        # 来源: Quantopian cost/latency + hidden-regime + Gandalf 2026
        # BLOCK的Agent已加入_overfitted_agents(GT×0.5或-999)
        # WARN的Agent在此集合中,GT×0.8轻度惩罚(仍参与进化,但优先级降低)
        # 防止"模拟牛逼实盘亏钱": 成本/延迟敏感但未到BLOCK的Agent也需惩罚
        self._live_reality_warned_agents: Dict[str, List[str]] = {}

        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 初始种群
    # ------------------------------------------------------------------

    def initialize_population(self, seed_count: int = 0) -> List[AgentGene]:
        """生成初始种群

        Args:
            seed_count: 从种子库注入的种子策略数量（0=全部随机）

        Returns:
            初始种群列表
        """
        self.population = []
        self.generation = 0

        # 种子策略注入：确保每种世界观至少有一个代表
        if seed_count > 0:
            # Phase E: 沙盘模式下过滤非现货/永续策略
            _sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
            if _sandbox_mode:
                from .strategy_registry import StrategyRegistry
                _unsupported = StrategyRegistry.SANDBOX_UNSUPPORTED_SEEDS
                seeds = [s for s in WORLDVIEW_SEEDS.keys() if s not in _unsupported]
            else:
                seeds = list(WORLDVIEW_SEEDS.keys())
            # P0-1: 使用全局确定性RNG替代random.shuffle
            global_rng_manager.global_rng.shuffle(seeds)
            for i, seed_name in enumerate(seeds[:seed_count]):
                gene = generate_random_gene(generation=0)
                gene.worldview_primary = seed_name
                gene.agent_id = f"seed-{seed_name}-{i:03d}"
                self.population.append(gene)

        # 填充剩余到population_size
        remaining = self.population_size - len(self.population)
        for i in range(remaining):
            gene = generate_random_gene(generation=0)
            gene.agent_id = f"agent-gen0-{i:03d}"
            self.population.append(gene)

        self._save_population()
        logger.info(
            "Initialized population: %d agents (generation %d)",
            len(self.population), self.generation,
        )
        return self.population

    # ------------------------------------------------------------------
    # v2.35: 日内交易场景初始化
    # ------------------------------------------------------------------

    def initialize_intraday_population(self) -> List[AgentGene]:
        """生成日内交易场景的初始种群 (v2.35)

        与 initialize_population 的区别:
          1. 全部使用 generate_intraday_gene (应用日内硬约束)
          2. 按 INTRADAY_WORLDVIEW_WEIGHTS 加权分布高胜率世界观
          3. 不再注入全部 WORLDVIEW_SEEDS(避免低胜率世界观浪费进化槽位)

        来源: 500轮进化诊断 — 16个亏钱参数 + 高胜率世界观排名

        Returns:
            日内交易场景的初始种群
        """
        # 延迟导入避免循环依赖
        from .gene_codec import generate_intraday_gene

        self.population = []
        self.generation = 0
        # v2.35: 启用日内模式,后续 crossover 将应用日内硬约束 clamp
        self._intraday_mode = True

        for i in range(self.population_size):
            gene = generate_intraday_gene(generation=0)
            gene.agent_id = f"intraday-gen0-{i:03d}"
            self.population.append(gene)

        self._save_population()
        logger.info(
            "Initialized INTRADAY population: %d agents (generation %d) "
            "[event_driven=40%%/liquidity=20%%/funding=15%%/stat_arb=15%%/transformer=10%%]",
            len(self.population), self.generation,
        )
        return self.population

    # ------------------------------------------------------------------
    # 进化轮次
    # ------------------------------------------------------------------

    def run_evolution_round(
        self,
        trades_by_agent: Dict[str, List[Dict[str, Any]]],
        initial_capital: float = 10000.0,
        err_lessons: Optional[List[Dict]] = None,
        risk_rejections_by_agent: Optional[Dict[str, int]] = None,
    ) -> PopulationStats:
        """执行一轮进化

        R8进化升级：集成NSGA-II多目标帕累托排序 + Feature Map行为多样性

        1. 计算每个Agent的适应度（多目标：夏普/回撤/胜率/盈利因子）
        2. NSGA-II非支配排序 + 拥挤距离选择精英
        3. Feature Map更新（行为多样性维护）
        4. 交叉变异产生后代（从帕累托前沿选父代）
        5. 岛屿迁移（每N代）

        Args:
            trades_by_agent: {agent_id: [trade_records]}
            initial_capital: 初始资金
            err_lessons: v4.0 Phase 6 ERR教训列表 (闭环核心)
                None=不注入(向后兼容); List[Dict]=传递给 crossover 应用
                用户铁律: "迭代经验和教训库使用起来"

        Returns:
            本轮种群统计
        """
        # Step 1: 计算适应度（含Walk-forward抗过拟合验证）
        scores: Dict[str, FitnessScore] = {}
        self._wf_train_scores.clear()
        self._wf_test_scores.clear()
        self._overfit_penalty.clear()
        risk_rejections_by_agent = risk_rejections_by_agent or {}

        # v2.43: 实盘预测 PnL 评估器 (延迟初始化)
        # 用户核心需求: 杜绝"模拟牛逼实盘亏钱"
        live_evaluator = None
        if self.use_live_pnl_fitness:
            try:
                from . _v243_live_pnl_fitness import LivePnLFitnessEvaluator
                live_evaluator = LivePnLFitnessEvaluator()
                logger.info("v2.43 LivePnLFitnessEvaluator 已启用 (实盘预测PnL适应度)")
            except ImportError as e:
                logger.warning("v2.43 LivePnLFitnessEvaluator 导入失败, 回退到 compute_fitness: %s", e)
                self.use_live_pnl_fitness = False

        def _compute_fitness_v243(trades_list):
            """v2.43 适应度计算辅助 — 根据 use_live_pnl_fitness 选择评估器"""
            if live_evaluator is not None:
                score_obj, meta = live_evaluator.evaluate(trades_list, initial_capital)
                # 兼容性: 确保 score_obj 是 FitnessScore 或类似对象
                return score_obj
            else:
                return compute_fitness(trades_list, initial_capital)

        for agent in self.population:
            trades = trades_by_agent.get(agent.agent_id, [])
            risk_rejection_count = risk_rejections_by_agent.get(agent.agent_id, 0)
            
            # Walk-forward验证：将交易分为训练集和测试集
            # v3.12.1 修复 (独立审查P0): 早检测保持>=5, 但硬门控需>=10避免2笔统计不稳定
            if self.use_walk_forward and len(trades) >= 5:  # v3.12: 10→5 早检测过拟合
                # 按50%分窗：前70%训练，后30%测试
                split_idx = max(1, int(len(trades) * self.wf_train_ratio))
                train_trades = trades[:split_idx]
                test_trades = trades[split_idx:]

                train_fitness = _compute_fitness_v243(train_trades)
                test_fitness = _compute_fitness_v243(test_trades) if test_trades else FitnessScore()

                # 记录Walk-forward分数
                self._wf_train_scores[agent.agent_id] = train_fitness.gt_score
                self._wf_test_scores[agent.agent_id] = test_fitness.gt_score

                # 计算过拟合惩罚因子
                # v2.36 修复 (协调 LOSS-HUNTER 建议): 阈值 1.0 → 0.3
                # 根因: 原阈值 train_gt > 1.0 太高, 大部分 agent gt_score < 1.0, 进不了检测
                #       导致过拟合策略仍可参与进化繁殖, 可能主导种群
                # 修复: 降低阈值到 0.3, 让更多 agent 进入过拟合检测
                if train_fitness.gt_score > 0.3:
                    wf_ratio = test_fitness.gt_score / train_fitness.gt_score
                    # wf_ratio < 1.0 表示测试集表现不如训练集（可能过拟合）
                    # v3.12: 沙盘模式硬门控 - wf_ratio<0.5直接淘汰(原软门控×0.5仍参与进化)
                    # v3.12: wf_ratio<0.7标记DEGRADED (原0.5阈值漏检)
                    # v3.12.1: 硬门控需 len(trades)>=10 避免测试集仅2笔统计不稳定
                    #   (独立审查P0: 5笔时split=3/2, 2笔全亏触发硬淘汰是随机波动非过拟合)
                    # wf_ratio > 1.0 表示泛化良好，无惩罚
                    # Phase 12: Walk-forward硬门控增加盈利豁免
                    # 根因: gen8盈利策略在gen10+可能因测试集方差触发wf_ratio<0.5被-999误杀
                    # 修复: 全量交易仍盈利(total_pnl>0)时降级为软门控, 不硬淘汰
                    _agent_total_pnl = getattr(train_fitness, 'total_pnl', 0) + getattr(test_fitness, 'total_pnl', 0)
                    if wf_ratio < 0.5 and len(trades) >= 10 and _agent_total_pnl <= 0:
                        # v3.12 硬门控: 严重过拟合直接淘汰 (替代原软门控)
                        # v3.12.1: 仅在样本充足(>=10笔)时启用, 避免误杀
                        # Phase 12: 仅在全量交易亏损时硬淘汰, 盈利策略降级为软门控
                        self._overfit_penalty[agent.agent_id] = -999.0
                        logger.warning(
                            "Agent %s HARD-OVERFIT (v3.12): train=%.2f test=%.2f ratio=%.2f n=%d pnl=%.2f → eliminated",
                            agent.agent_id[:8], train_fitness.gt_score, test_fitness.gt_score,
                            wf_ratio, len(trades), _agent_total_pnl,
                        )
                    elif wf_ratio < 0.5 and len(trades) >= 10 and _agent_total_pnl > 0:
                        # Phase 12: 盈利策略豁免硬淘汰, 降级为软门控 (防止误杀真盈利)
                        self._overfit_penalty[agent.agent_id] = wf_ratio * 0.5
                        logger.info(
                            "Agent %s PROFIT-EXEMPT (Phase 12): train=%.2f test=%.2f ratio=%.2f n=%d pnl=%.2f → soft penalty=%.2f",
                            agent.agent_id[:8], train_fitness.gt_score, test_fitness.gt_score,
                            wf_ratio, len(trades), _agent_total_pnl, wf_ratio * 0.5,
                        )
                    elif wf_ratio < 0.5 and len(trades) < 10:
                        # v3.12.1: 样本不足时降级为软门控, 避免统计不稳定误杀
                        self._overfit_penalty[agent.agent_id] = wf_ratio * 0.5
                        logger.warning(
                            "Agent %s SOFT-OVERFIT (v3.12.1, n=%d<10): train=%.2f test=%.2f ratio=%.2f penalty=%.2f",
                            agent.agent_id[:8], len(trades), train_fitness.gt_score, test_fitness.gt_score,
                            wf_ratio, wf_ratio * 0.5,
                        )
                    elif wf_ratio < self.wf_overfit_threshold:
                        penalty = wf_ratio  # 直接用ratio作为惩罚
                        self._overfit_penalty[agent.agent_id] = penalty
                        logger.warning(
                            "Agent %s overfitting detected: train=%.1f test=%.1f ratio=%.2f penalty=%.2f",
                            agent.agent_id[:8], train_fitness.gt_score, test_fitness.gt_score,
                            wf_ratio, penalty,
                        )

                # 使用全量交易计算最终适应度
                fitness = _compute_fitness_v243(trades)

                fitness.risk_rejection_count = risk_rejection_count

                # Phase 14.17: 设置滑动窗口平均值 (主分支也必须设置, 原BUG: 只在else分支设置)
                _pnl_hist = self._agent_pnl_history.get(agent.agent_id, [])
                fitness.rolling_avg_pnl = sum(_pnl_hist) / len(_pnl_hist) if _pnl_hist else 0.0
                _gt_hist = self._agent_gt_history.get(agent.agent_id, [])
                fitness.rolling_avg_gt = sum(_gt_hist) / len(_gt_hist) if _gt_hist else 0.0
                # Phase 14.18: 调试日志 — 确认rolling_avg_gt是否正确读取
                if _gt_hist:
                    logger.warning(
                        "Phase 14.18 DEBUG main: agent=%s gt_hist=%s rolling_avg_gt=%.2f "
                        "(hist_len=%d, pnl_hist_len=%d)",
                        agent.agent_id[:16], [round(v, 2) for v in _gt_hist[-3:]],
                        fitness.rolling_avg_gt, len(_gt_hist), len(_pnl_hist),
                    )
                # 用历史平均值重新计算gt_score (应用rolling_avg_pnl门控 + rolling_avg_gt平滑)
                fitness.compute_gt_score()
                # Phase 14.18: 调试日志 — 确认平滑后的gt_score
                if fitness.rolling_avg_gt > 0:
                    logger.warning(
                        "Phase 14.18 DEBUG smooth: agent=%s raw_gt→smoothed_gt=%.2f "
                        "(rolling_avg_gt=%.2f, final=%.2f)",
                        agent.agent_id[:16], 0.0, fitness.rolling_avg_gt, fitness.gt_score,
                    )

                # 应用过拟合惩罚
                if agent.agent_id in self._overfit_penalty:
                    penalty_val = self._overfit_penalty[agent.agent_id]
                    if penalty_val < 0:
                        # v3.12 硬门控: 直接淘汰
                        fitness.gt_score = -999.0
                    else:
                        fitness.gt_score *= penalty_val
            else:
                # 交易数太少，无法Walk-forward，直接计算
                fitness = _compute_fitness_v243(trades)

                fitness.risk_rejection_count = risk_rejection_count

                # Phase 14.16f: 设置滑动窗口平均PnL (即使交易少也要设置)
                pnl_history = self._agent_pnl_history.get(agent.agent_id, [])
                if pnl_history:
                    fitness.rolling_avg_pnl = sum(pnl_history) / len(pnl_history)
                else:
                    fitness.rolling_avg_pnl = 0.0

                # Phase 14.17: 设置滑动窗口平均GT-Score
                gt_history = self._agent_gt_history.get(agent.agent_id, [])
                if gt_history:
                    fitness.rolling_avg_gt = sum(gt_history) / len(gt_history)
                else:
                    fitness.rolling_avg_gt = 0.0
                # Phase 14.18: 调试日志
                if gt_history:
                    logger.warning(
                        "Phase 14.18 DEBUG else: agent=%s gt_hist=%s rolling_avg_gt=%.2f",
                        agent.agent_id[:16], [round(v, 2) for v in gt_history[-3:]],
                        fitness.rolling_avg_gt,
                    )

                # 重新计算gt_score (使用rolling_avg_pnl门控 + rolling_avg_gt平滑)
                fitness.compute_gt_score()
            
            scores[agent.agent_id] = fitness

            # 更新基因中的运行时状态

            # Phase 14.16f: 更新滑动窗口PnL历史 (用于下一代硬门控判断)
            if agent.agent_id not in self._agent_pnl_history:
                self._agent_pnl_history[agent.agent_id] = []
            self._agent_pnl_history[agent.agent_id].append(fitness.total_pnl)
            # 只保留最近N代 (滑动窗口)
            if len(self._agent_pnl_history[agent.agent_id]) > self._rolling_window_size:
                self._agent_pnl_history[agent.agent_id] = self._agent_pnl_history[agent.agent_id][-self._rolling_window_size:]
            # Phase 14.17: 更新滑动窗口GT-Score历史 (用于下一代gt_score平滑)
            # 排除硬淘汰值(-999)避免污染历史平均
            if fitness.gt_score > -100.0:
                if agent.agent_id not in self._agent_gt_history:
                    self._agent_gt_history[agent.agent_id] = []
                self._agent_gt_history[agent.agent_id].append(fitness.gt_score)
                # 只保留最近N代 (滑动窗口)
                if len(self._agent_gt_history[agent.agent_id]) > self._rolling_window_size:
                    self._agent_gt_history[agent.agent_id] = self._agent_gt_history[agent.agent_id][-self._rolling_window_size:]
            agent.fitness_score = fitness.gt_score
            agent.sharpe_ratio = fitness.sharpe_ratio
            agent.win_rate = fitness.win_rate
            agent.total_trades = fitness.total_trades
            agent.total_pnl = fitness.total_pnl
            agent.max_drawdown_realized = fitness.max_drawdown
            agent.survival_rounds += 1

        # Step 1.5: 共享函数适应度惩罚（R9.2调优版，来自前沿GA小生境技术）
        # 核心思想：对周围有相似个体的Agent降低适应度，防止优势策略过度繁殖
        # 公式: f'_i = f_i / (1 + β * sum_j(sh(d_ij)))
        # sh(d) = 1 - (d/σ)^α  if d < σ_share, else 0
        #
        # R9.2修复（解决R9.1过度惩罚问题）:
        #   1. share_radius: 0.15 → 0.25 (只惩罚真正相似的个体)
        #   2. 惩罚强度系数 β: 1.0 → 0.3 (降低每个邻居的惩罚贡献)
        #   3. 精英保护: Top 20% 不应用共享函数惩罚
        #   4. 惩罚下限: GT-Score 不低于原值的 50%
        if len(self.population) > 2:
            share_radius = 0.25  # 共享半径（R9.2: 0.15→0.25，只惩罚真正相似）
            alpha = 2.0  # 共享函数形状参数
            beta = 0.3  # 惩罚强度系数（R9.2新增：1.0→0.3，降低单邻居惩罚）
            min_score_ratio = 0.5  # 惩罚下限（R9.2新增：保留至少50%原值得分）

            # R9.2: 精英保护 — Top 20% 不惩罚
            sorted_agents = sorted(
                self.population,
                key=lambda a: scores[a.agent_id].gt_score,
                reverse=True,
            )
            elite_count = max(1, len(sorted_agents) // 5)  # Top 20%
            elite_ids = {a.agent_id for a in sorted_agents[:elite_count]}

            # Phase 14.16o: HIGH GT基因存档 — 保存gt_score>10的top-1个体
            # 根因: HIGH GT个体(gen 27=134.9, gen 44=113.7, gen 71=116.6)到后期完全丢失
            # 修复: 存档HIGH GT基因, 每10代回注确保优质基因不永久消失
            try:
                _top_agent = sorted_agents[0] if sorted_agents else None
                _top_score = scores.get(_top_agent.agent_id) if _top_agent is not None else None
                _max_gt_this_gen = _top_score.gt_score if _top_score is not None else -999.0
                _top_trades = trades_by_agent.get(_top_agent.agent_id, []) if _top_agent is not None else []
                _top_pnls = [
                    float(t.get("pnl", 0.0)) if isinstance(t, dict) else float(getattr(t, "pnl", 0.0))
                    for t in _top_trades
                ]
                _top_split = max(1, int(len(_top_pnls) * 0.6))
                _top_total_pnl = sum(_top_pnls)
                _top_oos_pnl = sum(_top_pnls[_top_split:]) if len(_top_pnls) - _top_split >= 2 else -1.0
                _archive_eligible = (
                    _max_gt_this_gen > self._high_gt_threshold
                    and _top_total_pnl > 0.0
                    and _top_oos_pnl > 0.0
                )
                if _archive_eligible:
                    _archive_agent = copy.deepcopy(_top_agent)
                    self._high_gt_archive.append((_max_gt_this_gen, _archive_agent, self.generation))
                    self._high_gt_archive.sort(key=lambda x: x[0], reverse=True)
                    if len(self._high_gt_archive) > 10:
                        self._high_gt_archive = self._high_gt_archive[:10]
                    logger.info(
                        "Phase Q HIGH GT ARCHIVE: gen=%d gt=%.4f agent=%s pnl=%.4f oos_pnl=%.4f archive_size=%d",
                        self.generation, _max_gt_this_gen, _top_agent.agent_id[:12],
                        _top_total_pnl, _top_oos_pnl, len(self._high_gt_archive),
                    )
                elif _max_gt_this_gen > self._high_gt_threshold:
                    logger.info(
                        "Phase Q HIGH GT REJECTED: gen=%d gt=%.4f agent=%s pnl=%.4f oos_pnl=%.4f",
                        self.generation, _max_gt_this_gen, _top_agent.agent_id[:12],
                        _top_total_pnl, _top_oos_pnl,
                    )
            except Exception as _e:
                logger.warning("Phase 14.16o archive failed (non-fatal): %s", _e)

            # Phase 14.16p: 记录惩罚前原始max_gt_score (共享函数惩罚前的真实表现)
            # 根因: ARCHIVE记录原始gt, stats记录惩罚后gt, 两者不一致导致验证2误判
            # 修复: 在共享函数惩罚前记录max_gt_score_raw, 用于验证2(>100)
            try:
                _raw_gt_scores = [scores[a.agent_id].gt_score for a in self.population]
                if _raw_gt_scores:
                    self._current_max_gt_raw = max(_raw_gt_scores)
                else:
                    self._current_max_gt_raw = 0.0
            except Exception:
                self._current_max_gt_raw = 0.0

            # P4#2: 预计算所有 style_vector，避免 O(n²) 循环中重复调用 compute_style_vector
            _style_vectors = {a.agent_id: a.compute_style_vector() for a in self.population}

            for agent_i in self.population:
                # 精英跳过惩罚
                if agent_i.agent_id in elite_ids:
                    continue

                niche_count = 0.0
                va = _style_vectors[agent_i.agent_id]
                for agent_j in self.population:
                    if agent_i.agent_id == agent_j.agent_id:
                        continue
                    sim = gene_similarity_from_vectors(va, _style_vectors[agent_j.agent_id])
                    # sim越接近1表示越相似，d = 1 - sim 表示距离
                    d = 1.0 - sim
                    if d < share_radius:
                        niche_count += 1.0 - (d / share_radius) ** alpha

                # 应用共享函数惩罚（R9.2: 带强度系数β和下限保护）
                if niche_count > 0 and scores[agent_i.agent_id].gt_score > 0:
                    original_score = scores[agent_i.agent_id].gt_score
                    penalty = 1.0 / (1.0 + beta * niche_count)
                    new_score = original_score * penalty
                    # 惩罚下限：不低于原值的 min_score_ratio
                    min_allowed = original_score * min_score_ratio
                    scores[agent_i.agent_id].gt_score = max(new_score, min_allowed)
                    agent_i.fitness_score = scores[agent_i.agent_id].gt_score

        # Step 1.6: v2.6 GT-Score乘法结构升级 — 显著性门控(ln(z))
        # 来源：arXiv:2602.00080 GT-Score论文 (Sheppert 2026-01)
        # 核心思想：用乘法结构替代加法，任一维度低分整体低分
        # 公式：gt_score *= significance_gate(z)
        #   z = (agent_pnl - benchmark_pnl) / (std_pnl / sqrt(N))
        #   z <= 0: 跑输基准，惩罚（gate < 1.0）
        #   z > 0: 跑赢基准，奖励（gate > 1.0，但ln压缩防主导）
        # 优势：
        #   1. 不交易或亏损的Agent被显著惩罚（z<0 → gate<1）
        #   2. 仅在群体中相对优秀才得高分（基准=群体均值）
        #   3. ln(z)压缩大值，防止单一维度主导
        if len(self.population) > 1:
            # 计算群体基准（所有Agent的平均PnL）
            agent_pnls = []
            agent_pnl_stds = []
            for agent in self.population:
                s = scores[agent.agent_id]
                if s.total_trades > 0:
                    agent_pnls.append(s.total_pnl)
                    # 估计PnL标准差（用max_drawdown近似）
                    agent_pnl_stds.append(abs(s.total_pnl * s.max_drawdown) + 1.0)

            if agent_pnls:
                benchmark_pnl = sum(agent_pnls) / len(agent_pnls)
                mean_std = sum(agent_pnl_stds) / len(agent_pnl_stds) if agent_pnl_stds else 1.0

                for agent in self.population:
                    s = scores[agent.agent_id]
                    if s.total_trades == 0:
                        # v2.41 修复: 零交易 agent 已在 compute_fitness() 中得 -5.0 (非 0 分)
                        # 原注释错误: "已在compute_fitness中得0分" — 实际是 -5.0 (BUG-E 修复)
                        # 显著性门控基于 PnL z-score, 零交易 agent 无 PnL, 无法计算 z
                        # 保持 -5.0 不变 (淘汰压力), 由后续 Step 1.7 novelty 提供轻微区分度
                        # v2.41 已修复 Step 1.7: novelty 不再把 -5.0 覆写为正数
                        continue

                    # Z-score: 相对于群体基准的统计显著性
                    N = max(s.total_trades, 1)
                    if mean_std > 1e-10:
                        z = (s.total_pnl - benchmark_pnl) / (mean_std / math.sqrt(N))
                    else:
                        z = 0.0

                    # 显著性门控（ln压缩）
                    # z <= 0: gate = max(0.1, 1 + z*0.1)  # 跑输基准，惩罚但不归零
                    # z > 0: gate = 1 + ln(1 + z) * 0.3  # 跑赢基准，奖励但ln压缩
                    if z <= 0:
                        significance_gate = max(0.1, 1.0 + z * 0.1)
                    else:
                        significance_gate = 1.0 + math.log1p(z) * 0.3  # log1p(z) = ln(1+z)

                    # 应用门控（乘法结构）
                    original_gt = s.gt_score
                    s.gt_score = original_gt * significance_gate
                    agent.fitness_score = s.gt_score  # 更新运行时状态


        # Step 1.7: v2.7 Novelty Search + MAP-Elites 行为多样性增强
        # v2.20 CRITICAL修复：novelty bonus移到过拟合门控之后（原Step 1.7在Step 2之前）
        # 原问题：novelty bonus先应用→过拟合门控5折→novelty bonus部分被保留
        #         一个novel但overfit的Agent可获15-30分人造GT-Score，绕过PSR/DSR门控
        # 修复：先执行过拟合门控（Step 2），再应用novelty bonus，且对过拟合Agent不应用
        # 来源：AlgoXpert 2026 IS-WFA-OOS协议 — 过拟合策略不得通过其他通道获得分数
        # (novelty bonus代码移到Step 2之后)

        # Step 1.8: v2.13 回退 Fitness Sharing
        # v2.12实验结论：Fitness Sharing在当前种群下效果反向
        #   - σ_share=0.75: 39/50 Agent被惩罚, GT max 98.3→16.6
        #   - σ_share=0.85: 32 Agent被惩罚, GT max→11.2
        #   - σ_share=0.85+cap=2.0: GT max→46.2 (仍远低于v2.11的98.3)
        #   - diversity 没有显著提升 (0.142 vs v2.11的0.152)
        # 根因：平均基因相似度=0.86, σ_share阈值难以选择
        #       Fitness Sharing只惩罚GT-Score,不改变基因,无法直接提升diversity
        # 决策：回退到v2.11状态,改用Crowding(在替换时考虑相似度)更有效
        # 详见 v2.13 Crowding Step 4.5

        # ===== v499 P0-3修复: Step 1.75 反过拟合强制验证 (WF1-T5) =====
        # 来源: anti_overfitting.py 5大验证器 (Walk-Forward/蒙特卡洛/真实滑点/PBO/参数敏感性)
        # 核心使命: 消除"沙盘赚钱实盘亏钱"的根源 — 970行完整实现终于被调用
        # 任何策略过拟合 → gt_score × 0.3 严厉惩罚 (Bailey 2014 PBO理论)
        # 注意: 此验证独立于内置的Walk-forward(_wf_*机制),作为第二道防线
        #       _wf_*机制只做IS/OOS分数比对,本验证器还包含蒙特卡洛/PBO/参数敏感性
        # 插入位置: 所有GT-Score计算(Step 1.5/1.6/1.7)之后、精英选择(Step 2)之前
        for agent in self.population:
            agent_id = agent.agent_id
            fitness = scores.get(agent_id)
            if fitness is None:
                continue

            # 取该agent的交易记录 (trades_by_agent是传入参数)
            trades = trades_by_agent.get(agent_id, [])
            if len(trades) < 10:
                continue  # 交易不足10笔,样本不足,跳过验证(避免统计不稳定)

            # 提取trade_pnls (兼容dict和object两种交易记录格式)
            trade_pnls = []
            for t in trades:
                if isinstance(t, dict):
                    trade_pnls.append(float(t.get('pnl', 0.0)))
                else:
                    trade_pnls.append(float(getattr(t, 'pnl', 0.0)))

            split = max(1, int(len(trade_pnls) * 0.6))
            if len(trade_pnls) - split < 2:
                continue
            is_metrics = _anti_overfit_metrics(trade_pnls[:split])
            oos_metrics = _anti_overfit_metrics(trade_pnls[split:])

            try:
                result = self.anti_overfitting_validator.validate(
                    is_metrics=is_metrics,
                    oos_metrics=oos_metrics,
                    trade_pnls=trade_pnls,
                    apply_slippage=True,
                )
                _is_return = float(is_metrics.get("return", 0.0))
                _oos_return = float(oos_metrics.get("return", 0.0))
                _has_positive_alpha = _is_return > 0.0 or _oos_return > 0.0
                _return_reversal = _is_return > 0.0 and _oos_return <= 0.0
                _explicit_overfit = bool(result.pbo >= 0.5 or _return_reversal)
                if result.is_overfitted and _has_positive_alpha and _explicit_overfit:
                    original_gt = fitness.gt_score
                    fitness.gt_score = original_gt * 0.6
                    fitness.overfitting_flag = True
                    fitness.pbo_value = float(result.pbo)
                    self._overfit_penalty[agent_id] = float(result.pbo)
                    _wf_ratio = float(result.oos_is_ratio)
                    _mc_p = float(result.monte_carlo_pvalue)
                    _reasons = "; ".join(result.recommendations) or "综合验证门控"
                    logger.warning(
                        "v499 P0-3 反过拟合: agent=%s gt_score %.2f→%.2f "
                        "(×0.6惩罚), pbo=%.3f, mc_p=%.4f, wf_ratio=%.3f, reasons=%s",
                        agent_id[:12], original_gt, fitness.gt_score,
                        result.pbo, _mc_p, _wf_ratio, _reasons,
                    )
                elif result.is_overfitted and not _has_positive_alpha:
                    logger.debug(
                        "v499 P0-3: agent=%s 无正向alpha, 保留原始GT用于探索淘汰 mc_p=%.4f",
                        agent_id[:12], float(result.monte_carlo_pvalue),
                    )
            except Exception as e:
                # 验证失败不阻塞进化,但记录警告
                self._overfit_penalty[agent_id] = -1.0
                logger.warning(
                    "v499 P0-3 反过拟合验证异常: agent=%s, error=%s",
                    agent_id[:12], e,
                )

        # Step 2: 精英选择（NSGA-II 或 GT-Score排名）
        # v2.19 P0: 真正实现实盘硬门控（之前只有注释，无代码分支）
        # 来源：Bailey & López de Prado "Deflated Sharpe Ratio" 2014 + 核心铁律
        # 核心铁律："ai量化技能包不要模拟厉害，实盘亏钱！"
        # 沙盘模式(_live_mode=False)：过拟合Agent的GT-Score降低50%（软门控），仍然参与进化
        # 实盘模式(_live_mode=True)：过拟合Agent的GT-Score设为-999（硬门控），直接淘汰
        # 理由：沙盘模式下DSR/PSR不稳定（交易次数少），硬门控会过度淘汰
        #       但实盘部署时必须严格——过拟合策略进入实盘=必然亏钱
        if self._overfitted_agents:
            n_marked = 0
            for agent in self.population:
                if agent.agent_id in self._overfitted_agents:
                    original_gt = scores[agent.agent_id].gt_score
                    if self._live_mode:
                        # v2.19 P0: 实盘硬门控 — GT=-999，直接淘汰
                        # 真正实现"实盘不进化，进化不实盘"的铁律
                        penalized_gt = -999.0
                        scores[agent.agent_id].gt_score = penalized_gt
                        agent.fitness_score = penalized_gt
                        logger.warning(
                            "v2.19 HARD GATE: 实盘硬淘汰 %s (原GT=%.2f, DSR/PSR未通过)",
                            agent.agent_id[:12], original_gt,
                        )
                    else:
                        # v2.8: 沙盘软门控 — GT-Score降低20% (Phase 14.16k.2: 50%→20%)
                        # 理由: 沙盘进化模式需要保留更多gt_score信号, 原50%惩罚过于严厉
                        #       实盘部署时过拟合门控通过production_oms/risk_control严格生效
                        penalized_gt = original_gt * 0.8
                        scores[agent.agent_id].gt_score = penalized_gt
                        agent.fitness_score = penalized_gt
                    n_marked += 1
            if n_marked > 0:
                gate_type = "HARD(实盘)" if self._live_mode else "SOFT(沙盘)"
                logger.info(
                    "v2.19 %s Gate: 处理 %d 个过拟合Agent",
                    gate_type, n_marked,
                )

        # R13优化: Live Reality Check WARN 轻度惩罚
        # 来源: Quantopian cost/latency + hidden-regime + Gandalf 2026
        # BLOCK的Agent已被上面的过拟合门控处理(GT×0.5或-999)
        # WARN的Agent在此处理: 沙盘GT×0.8, 实盘GT×0.5
        # 理由: 成本/延迟敏感但未到BLOCK的Agent仍需惩罚,降低其进化优先级
        #       防止"模拟牛逼实盘亏钱": WARN是早期预警,不惩罚会进化出更敏感的策略
        if self._live_reality_warned_agents:
            n_warned = 0
            warn_penalty_factor = 0.5 if self._live_mode else 0.8  # 实盘更严格
            for agent in self.population:
                if agent.agent_id in self._live_reality_warned_agents:
                    # 跳过已被过拟合门控处理的Agent(避免双重惩罚)
                    if agent.agent_id in self._overfitted_agents:
                        continue
                    original_gt = scores[agent.agent_id].gt_score
                    penalized_gt = original_gt * warn_penalty_factor
                    scores[agent.agent_id].gt_score = penalized_gt
                    agent.fitness_score = penalized_gt
                    n_warned += 1
                    logger.info(
                        "R13 LiveReality WARN Gate: %s GT %.2f→%.2f (×%.1f) reasons=%s",
                        agent.agent_id[:12], original_gt, penalized_gt,
                        warn_penalty_factor,
                        self._live_reality_warned_agents[agent.agent_id][:2],  # 只显示前2个原因
                    )
            if n_warned > 0:
                logger.info("R13 LiveReality WARN Gate: 处理 %d 个WARN Agent", n_warned)
            # 清空WARN集合(每轮重新评估)
            self._live_reality_warned_agents.clear()

        # Step 2.1: v2.20 CRITICAL修复 — Novelty bonus移到过拟合门控之后
        # 来源：Lehman & Stanley 2011 Novelty Search + Mouret & Clune 2015 MAP-Elites
        # 核心思想：奖励行为新颖的Agent，防止种群收敛到单一策略
        # 公式：final_gt = 0.7 * original_gt + 0.3 * novelty_score * 100
        # v2.20修复：过拟合Agent不得获得novelty bonus（关闭过拟合逃逸通道）
        try:
            self.behavior_diversity.update_archive(self.population)
            # v2.20: 临时清空_overfitted_agents，让apply_novelty_to_scores跳过这些Agent
            # 通过传入overfitted_ids参数让novelty bonus跳过过拟合Agent
            self.behavior_diversity.apply_novelty_to_scores(
                self.population, scores,
                overfitted_ids=self._overfitted_agents if self._overfitted_agents else None,
            )
        except Exception as e:
            logger.warning("Novelty Search application failed: %s", e)

        # P1-2优化: 应用进化增强惩罚(Arena Mode+换手率+HHI)
        if self.enhancements and self.enhancements.enabled:
            scores, enh_report = self.enhancements.apply_penalties(
                scores, trades_by_agent
            )
            if enh_report.get("total_penalized", 0) > 0:
                logger.info(
                    "P1-2增强: 应用惩罚到 %d 个Agent (Arena=%d, 换手率=%d, HHI=%d)",
                    enh_report["total_penalized"],
                    enh_report.get("arena_penalties", {}).get("count", 0),
                    enh_report.get("turnover_penalties", {}).get("count", 0),
                    enh_report.get("hhi_penalties", {}).get("count", 0),
                )

        # Phase 14.16m: bear regime精英保护 (移到Step 2.1后, 避免被novelty bonus覆盖)
        # 根因: bear regime下所有agent亏损导致-4.5回退, 好基因丢失
        # 修复: bear保护从Step 1.75后移到此处(Step 2.1后), 确保是最后执行的score修改
        # 之前的问题: novelty bonus对零交易agent强制设gt_score=-5.0+bonus, 覆盖了bear保护的-0.5
        _bear_active = getattr(self, '_bear_regime_active', False)
        if _bear_active:
            _bear_sorted = sorted(self.population, key=lambda a: scores[a.agent_id].gt_score, reverse=True)
            _bear_elite_n = max(1, len(_bear_sorted) // 5)  # Top 20%
            _bear_protected = 0
            for _be_agent in _bear_sorted[:_bear_elite_n]:
                _be_aid = _be_agent.agent_id
                if scores[_be_aid].gt_score < -0.5:
                    _be_orig = scores[_be_aid].gt_score
                    scores[_be_aid].gt_score = -0.5  # 保护精英不低于-0.5
                    _be_agent.fitness_score = -0.5
                    _bear_protected += 1
                    logger.info(
                        "Phase 14.16m BEAR PROTECTION: agent=%s gt_score %.2f→-0.5 (精英保护, post-novelty)",
                        _be_aid[:12], _be_orig,
                    )
            if _bear_protected > 0:
                logger.info(
                    "Phase 14.16m BEAR PROTECTION: gen=%d, protected %d/%d elite agents",
                    self.generation, _bear_protected, _bear_elite_n,
                )

        # Phase 14.16p/q: 零交易精英保护 (所有regime)
        # 根因: gen 74 sideways regime下bear保护不触发, 但零交易导致novelty bonus设-4.5
        # Phase 14.16p: 在所有regime下保护top 20%精英的零交易agent到-0.5
        # Phase 14.16q-1修复: 原用gt_score<-4.0判断, 但scores已被共享函数惩罚(L644)修改
        #                   从-5.0→-0.178, 导致从未触发. 改用total_trades==0直接判断
        # 与bear保护互补: bear保护bear regime所有精英, 零交易保护所有regime零交易精英
        try:
            _zt_sorted = sorted([a for a in self.population if a.agent_id in scores], key=lambda a: scores[a.agent_id].gt_score, reverse=True)  # Phase 14.16q-2: 过滤不在scores中的agent, 避免KeyError
            _zt_elite_n = max(1, len(_zt_sorted) // 5)  # Top 20%
            _zt_protected = 0
            # Phase 14.16q-3: 调试日志 - 确认Top 20%精英的total_trades值
            _zt_debug_tts = [(a.agent_id[:12], scores[a.agent_id].total_trades, round(scores[a.agent_id].gt_score, 3)) for a in _zt_sorted[:_zt_elite_n]]
            logger.info("Phase 14.16q-3 DEBUG: gen=%d elite_n=%d elite_tt_gt=%s", self.generation, _zt_elite_n, _zt_debug_tts[:5])
            for _zt_agent in _zt_sorted[:_zt_elite_n]:
                _zt_aid = _zt_agent.agent_id
                if (
                    scores[_zt_aid].total_trades == 0
                    and scores[_zt_aid].risk_rejection_count == 0
                ):
                    _zt_orig = scores[_zt_aid].gt_score
                    scores[_zt_aid].gt_score = -0.5  # 保护精英不低于-0.5
                    _zt_agent.fitness_score = -0.5
                    _zt_protected += 1
                    logger.info(
                        "Phase 14.16q ZERO-TRADE PROTECTION: agent=%s total_trades=0, gt_score %.2f→-0.5 (零交易精英保护, all-regime)",
                        _zt_aid[:12], _zt_orig,
                    )
            if _zt_protected > 0:
                logger.info(
                    "Phase 14.16q ZERO-TRADE PROTECTION: gen=%d, protected %d/%d elite agents (all-regime, total_trades==0)",
                    self.generation, _zt_protected, _zt_elite_n,
                )
        except Exception as _e:
            logger.warning("Phase 14.16q zero-trade protection failed (non-fatal): %s", _e)  # Phase 14.16q-2: DEBUG→WARNING, 确认修复效果

        # Phase 14.9.3b: 统一过滤 — 所有精英选择路径只考虑已运行个体
        # 金融知识: 精英池必须基于已验证个体, 注入个体(gt=0.0)不应参与精英选择
        # 来源: LO-CCAC 2026 Portfolio Evolutionary Strategies + 用户铁律"杜绝模拟牛逼实盘亏"
        _evolve_eligible = [a for a in self.population if getattr(a, 'survival_rounds', 0) > 0]
        if not _evolve_eligible:
            _evolve_eligible = [a for a in self.population if getattr(a, 'total_trades', 0) > 0]
        if not _evolve_eligible:
            _evolve_eligible = list(self.population)  # 极端降级
            # Phase 14.10: 退化代警告 — 全种群无运行经验, 精英选择无意义
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Phase 14.10: 退化代 gen=%d — 全种群 survival=0 trades=0, "
                "精英将从注入个体中选择(无运行经验), 考虑提高注入质量",
                self.generation
            )

        # P1-3优化: 优先NSGA-III(4目标帕累托前沿更均匀),其次NSGA-II,最后GT-Score排名
        # Phase 8.1: 使用NSGA3Selector.select_with_regime()实现regime权重自适应
        # Phase 14.9.3b: 使用_evolve_eligible替代self.population
        if self.use_nsga3 and len(_evolve_eligible) > 1:
            if self._nsga3_selector is not None:
                # Phase 8.1: regime-aware selection (NSGA-III + regime weights)
                elite = self._nsga3_selector.select_with_regime(
                    _evolve_eligible, scores, self._elite_size,
                )
            else:
                # 回退: 函数式NSGA-III (无regime权重)
                from .nsga3_selector import multi_objective_select as _nsga3_select
                elite = _nsga3_select(_evolve_eligible, scores, self._elite_size, method="nsga3")
        elif self.use_nsga2 and len(_evolve_eligible) > 1:
            elite = nsga2_select(_evolve_eligible, scores, self._elite_size)
        else:
            ranked = sorted(
                _evolve_eligible,
                key=lambda a: scores[a.agent_id].gt_score,
                reverse=True,
            )
            elite = ranked[:self._elite_size]

        # Phase 14.16c: 统一agent_id去重 (应用到NSGA-III/NSGA-II/GT-Score所有路径)
        # 根因: v2.22去重只在GT-Score回退路径, NSGA-III(默认)路径无去重
        # 岛屿迁移deepcopy保留agent_id -> 重复副本进入精英 -> 跨代累积
        # 修复: 所有选择路径后统一去重
        seen_ids: set = set()
        deduped_elite = []
        for agent in elite:
            if agent.agent_id not in seen_ids:
                deduped_elite.append(agent)
                seen_ids.add(agent.agent_id)
                if len(deduped_elite) >= self._elite_size:
                    break
        _orig_elite_count = len(elite)
        elite = deduped_elite
        # Phase 14.19: 去重后补充到 elite_size
        # 原因: NSGA-III 返回重复 agent_id, 去重后数量不足 elite_size
        #       导致下一代基因多样性下降
        # 修复: 用 _evolve_eligible 中其他 agent 补足
        if len(elite) < self._elite_size and len(_evolve_eligible) > len(elite):
            _elite_id_set = {a.agent_id for a in elite}
            _filled = 0
            # 按 gt_score 排序, 补充最高分的未入选个体
            _fill_candidates = sorted(
                [a for a in _evolve_eligible if a.agent_id not in _elite_id_set],
                key=lambda a: scores[a.agent_id].gt_score,
                reverse=True,
            )
            for _a in _fill_candidates:
                if len(elite) >= self._elite_size:
                    break
                elite.append(_a)
                _elite_id_set.add(_a.agent_id)
                _filled += 1
            if _filled > 0:
                logger.warning(
                    "Phase 14.19 精英补足: 去重后%d→%d个 (补充%d个, 原NSGA-III返回%d个含重复)",
                    _orig_elite_count, len(elite), _filled, _orig_elite_count,
                )

        # Phase 14.16c: 顶级gt_score精英保护 (防止好基因被NSGA-III多目标选择漏掉)
        # 根因: NSGA-III是多目标帕累托前沿, 不保证gt_score最高的个体被选中
        # gen18 max_gt=17.54 -> gen19 max_gt=-4.5, 好基因丢失
        # 修复: 确保top-3 gt_score个体始终在精英中
        try:
            all_ranked = sorted(
                _evolve_eligible,
                key=lambda a: scores[a.agent_id].gt_score,
                reverse=True,
            )
            elite_ids = {a.agent_id for a in elite}
            top_preserve_count = min(3, len(all_ranked))
            for top_agent in all_ranked[:top_preserve_count]:
                if top_agent.agent_id not in elite_ids:
                    # 顶级个体不在精英中, 替换精英中最低分的个体
                    if len(elite) > 0:
                        min_idx = min(
                            range(len(elite)),
                            key=lambda i: scores[elite[i].agent_id].gt_score,
                        )
                        replaced = elite[min_idx]
                        elite[min_idx] = top_agent
                        elite_ids.discard(replaced.agent_id)
                        elite_ids.add(top_agent.agent_id)
                        logger.info(
                            "Phase 14.16c 精英保护: 保留顶级gt=%.4f个体 %s (替换gt=%.4f %s)",
                            scores[top_agent.agent_id].gt_score,
                            top_agent.agent_id[:12],
                            scores[replaced.agent_id].gt_score,
                            replaced.agent_id[:12],
                        )
        except Exception:
            pass  # 保护逻辑不应阻断主流程

        # v2.44 RC-004 根治: 世界观多样性硬约束 (每个世界观至少占精英的10%)
        # 根因: 180天回测Top5全部trend_following, 单一策略集中度过高
        # 修复: 精英选择后, 检查世界观分布, 补充占比<10%的世界观最优个体
        # 原则: 多市场状态适应, 不是单一策略集中
        # 来源: 用户核心诉求 + v2.44架构根因分析
        try:
            if len(elite) > 0 and len(self.population) > 0:
                # 统计精英群体世界观分布
                elite_wv_count: Dict[str, int] = {}
                for a in elite:
                    wv = getattr(a, 'worldview_primary', 'unknown')
                    elite_wv_count[wv] = elite_wv_count.get(wv, 0) + 1

                # 统计全种群世界观分布
                pop_wv_agents: Dict[str, List[Any]] = {}
                for a in self.population:
                    wv = getattr(a, 'worldview_primary', 'unknown')
                    if wv not in pop_wv_agents:
                        pop_wv_agents[wv] = []
                    pop_wv_agents[wv].append(a)

                # Phase 13.3: 提高worldview配额 10%→25% (diversity=0.0439需更强约束)
                min_quota = max(1, len(elite) // 4)  # Phase 13.3: //10→//4
                # 如果世界观数量 > 10, 调整为 max(1, elite_size // 世界观数量)
                n_worldviews = len(pop_wv_agents)
                if n_worldviews > 0:
                    min_quota = max(1, min(min_quota, len(elite) // n_worldviews))

                # 检查并补充不足配额的世界观
                elite_ids = {a.agent_id for a in elite}
                for wv, agents_in_wv in pop_wv_agents.items():
                    current_count = elite_wv_count.get(wv, 0)
                    if current_count < min_quota:
                        # 该世界观在精英中不足, 从全种群补充最优个体
                        # 按gt_score降序排列该世界观的所有个体
                        wv_ranked = sorted(
                            agents_in_wv,
                            key=lambda a: scores[a.agent_id].gt_score,
                            reverse=True,
                        )
                        # 找到该世界观中不在精英里的最优个体
                        for candidate in wv_ranked:
                            if candidate.agent_id not in elite_ids:
                                # 替换精英中最低分的个体(保持精英总数)
                                # 找到精英中最低分的(且不是当前世界观的)
                                min_idx = -1
                                min_score = float('inf')
                                for idx, e in enumerate(elite):
                                    e_wv = getattr(e, 'worldview_primary', 'unknown')
                                    if e_wv != wv and scores[e.agent_id].gt_score < min_score:
                                        min_score = scores[e.agent_id].gt_score
                                        min_idx = idx
                                if min_idx >= 0:
                                    replaced = elite[min_idx]
                                    elite[min_idx] = candidate
                                    elite_ids.discard(replaced.agent_id)
                                    elite_ids.add(candidate.agent_id)
                                    logger.info(
                                        "v2.44 世界观多样性: 替换 %s→%s (wv=%s, gt=%.2f→%.2f)",
                                        replaced.agent_id[:8], candidate.agent_id[:8],
                                        wv, min_score, scores[candidate.agent_id].gt_score,
                                    )
                                    current_count += 1
                                    if current_count >= min_quota:
                                        break
        except Exception as e:
            logger.warning("v2.44 世界观多样性约束失败(不阻塞): %s", e)

        # Step 3: Feature Map更新
        if self.feature_map:
            self.feature_map._cells.clear()
            for agent in self.population:
                self.feature_map.update(agent, scores[agent.agent_id], self.generation)

        # Step 3.5: v2.7 MAP-Elites网格更新（行为多样性）
        # 来源：Mouret & Clune 2015 — 在行为空间网格中保留每个niche的最优
        # 每轮进化后更新网格，用于get_elite时获取行为多样化的精英
        try:
            self.behavior_diversity.update_elite_grid(self.population, scores)
        except Exception as e:
            logger.warning("MAP-Elites grid update failed: %s", e)

        # Step 4: 统计收集
        ranked = sorted(
            self.population,
            key=lambda a: scores[a.agent_id].gt_score,
            reverse=True,
        )
        eliminated = ranked[self._elite_size:]

        stats = PopulationStats(
            generation=self.generation,
            population_size=len(self.population),
            elite_count=len(elite),
            eliminated_count=len(eliminated),
        )

        gt_scores = [scores[a.agent_id].gt_score for a in ranked]
        if gt_scores:
            stats.mean_gt_score = sum(gt_scores) / len(gt_scores)
            stats.max_gt_score = max(gt_scores)
            # Phase 14.16p: 记录惩罚前原始max_gt_score (用于验证2, 与ARCHIVE一致)
            stats.max_gt_score_raw = getattr(self, '_current_max_gt_raw', max(gt_scores))
            stats.min_gt_score = min(gt_scores)
            sorted_scores = sorted(gt_scores)
            mid = len(sorted_scores) // 2
            stats.median_gt_score = sorted_scores[mid]

        sharpes = [scores[a.agent_id].sharpe_ratio for a in ranked if scores[a.agent_id].total_trades > 0]
        if sharpes:
            stats.mean_sharpe = sum(sharpes) / len(sharpes)

        win_rates = [scores[a.agent_id].win_rate for a in ranked if scores[a.agent_id].total_trades > 0]
        if win_rates:
            stats.mean_win_rate = sum(win_rates) / len(win_rates)

        drawdowns = [scores[a.agent_id].max_drawdown for a in ranked]
        if drawdowns:
            stats.mean_drawdown = sum(drawdowns) / len(drawdowns)

        for a in self.population:
            wv = a.worldview_primary
            stats.worldview_distribution[wv] = stats.worldview_distribution.get(wv, 0) + 1

        # R8: 帕累托前沿大小 (P1-3优化: NSGA-III也用同样的fast_non_dominated_sort)
        if (self.use_nsga2 or self.use_nsga3) and len(self.population) > 1:
            fronts = fast_non_dominated_sort(scores)
            stats.pareto_front_size = len(fronts[0]) if fronts else 0

        # R8: Feature Map格子数
        if self.feature_map:
            stats.feature_map_cells = self.feature_map.get_cell_count()

        # R9: Walk-forward抗过拟合统计
        stats.overfit_detected_count = len(self._overfit_penalty)
        ratios = []
        low_train_count = 0
        for aid, train_s in self._wf_train_scores.items():
            if aid not in self._wf_test_scores:
                continue
            if train_s > 1.0:
                ratios.append(self._wf_test_scores[aid] / train_s)
            else:
                low_train_count += 1
        stats.wf_valid_count = len(ratios)
        stats.wf_negative_count = sum(1 for ratio in ratios if ratio < 0)
        if ratios:
            stats.mean_wf_ratio = sum(ratios) / len(ratios)
            stats.wf_status = "direction_reversal" if stats.wf_negative_count else "valid"
        elif low_train_count:
            stats.mean_wf_ratio = 0.0
            stats.wf_status = "low_train_score"
        else:
            stats.mean_wf_ratio = 0.0
            stats.wf_status = "insufficient_trades"

        # Step 5: 计算基因多样性
        # P4#2: 预计算 style_vector，避免 O(n²) 循环中重复调用 compute_style_vector
        _vectors = [a.compute_style_vector() for a in self.population]
        total_sim = 0.0
        pair_count = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                total_sim += gene_similarity_from_vectors(_vectors[i], _vectors[j])
                pair_count += 1
        stats.diversity = 1.0 - (total_sim / pair_count if pair_count > 0 else 0)

        # Step 6: 交叉变异产生下一代（R9: 自适应变异率 + v2.6分阶段变异率 + v2.7停滞突破）
        # v2.6升级：分阶段变异率（来源：CSDN 2026-06 GA最佳实践）
        #   前期(0-30%): 高探索，变异率0.02-0.05
        #   中期(30-70%): 自适应，基于多样性调整
        #   后期(70%+): 强收敛，变异率0.005
        #
        # v2.7升级：进化停滞检测与突破（来源：jestersimmps Evolving Trader 2026 + SD-EA）
        #   连续8代无改进: 激进变异模式（变异率×5）
        #   连续16代无改进: 横向移动接受（GT-Score≥0.7×best即保留）
        #   连续25代无改进: IPOP式重启（保留Top-5精英，其余重新随机）
        #
        # 原始自适应变异率：多样性越低，变异率越高（来自前沿GA自适应算子调控）
        # diversity > 0.3: mutation_boost = 0.0（正常）
        # diversity 0.20-0.3: mutation_boost = 0.2（轻度提升）
        # diversity 0.12-0.20: mutation_boost = 0.4（中度提升）
        # diversity < 0.12: mutation_boost = 0.6 + 随机注入（重度提升）

        # v2.7: 进化停滞检测
        # 更新最佳GT-Score历史和停滞计数器
        current_best = stats.max_gt_score
        if self.best_gt_score_history:
            # 根治(2026-07-16): 使用滚动窗口最近100代最佳，而非全历史最佳
            # 原bug: historical_best=322.62 是早期过拟合异常值，导致后续25代都判定为"无改进"
            # 修复: 只比较最近100代的最佳，避免过拟合异常值永久污染停滞检测
            _recent_history = self.best_gt_score_history[-100:] if len(self.best_gt_score_history) > 100 else self.best_gt_score_history
            historical_best = max(_recent_history)
            # 显著改进阈值：相对历史最佳的1%改进
            improvement = (current_best - historical_best) / max(abs(historical_best), 1.0)
            if improvement > self.stagnation_improvement_threshold:
                self.generations_since_improvement = 0  # 重置停滞计数器
            else:
                self.generations_since_improvement += 1
        else:
            self.generations_since_improvement = 0
        self.best_gt_score_history.append(current_best)

        stagnation_level = 0
        if self.generations_since_improvement >= self.stagnation_level3:
            stagnation_level = 3
        elif self.generations_since_improvement >= self.stagnation_level2:
            stagnation_level = 2
        elif self.generations_since_improvement >= self.stagnation_level1:
            stagnation_level = 1

        if stagnation_level > 0:
            logger.warning(
                "Stagnation detected: level=%d, generations_since_improvement=%d, "
                "current_best=%.2f, historical_best=%.2f",
                stagnation_level, self.generations_since_improvement,
                current_best, max(self.best_gt_score_history),
            )

        # v2.6: 计算进化阶段（基于generation和max_generations）
        # 来源：CSDN 2026-01 分阶段变异率策略
        if self.max_generations > 0:
            progress = self.generation / self.max_generations  # 0.0-1.0
        else:
            progress = 0.0  # 未知最大代数时，默认前期

        # 分阶段基础变异率
        if progress < 0.3:
            # 前期：高探索
            phase_base_boost = 0.15  # 前期加成15%
        elif progress < 0.7:
            # 中期：自适应
            phase_base_boost = 0.05  # 中期加成5%
        else:
            # 后期：强收敛
            phase_base_boost = 0.0   # 后期无加成

        # 多样性驱动的变异率（保留R9逻辑）
        # v2.11优化：提升0.12-0.20区间的boost从0.4到0.6，增强多样性驱动
        diversity_boost = 0.0
        if stats.diversity < 0.12:
            diversity_boost = 1.0  # Phase 14.3: 0.6→1.0 — diversity=0.035时需要更强变异
        elif stats.diversity < 0.20:
            diversity_boost = 0.6  # v2.11: 0.4→0.6，增强中度多样性驱动
        elif stats.diversity < self.min_diversity:
            diversity_boost = 0.2

        # v2.7: 停滞突破变异率加成
        stagnation_boost = 0.0
        if stagnation_level == 1:
            # Level 1: 激进变异模式（变异率大幅提升）
            stagnation_boost = 0.5
        elif stagnation_level == 2:
            # Level 2: 超激进变异模式
            stagnation_boost = 1.0
        elif stagnation_level == 3:
            # Level 3: IPOP式重启 — 极端变异率
            stagnation_boost = 2.0

        # 综合变异率 = 阶段基础 + 多样性驱动 + 停滞突破
        mutation_boost = phase_base_boost + diversity_boost + stagnation_boost

        if mutation_boost > 0:
            logger.info(
                "Mutation boost=%.2f (phase=%.2f, diversity=%.3f→boost=%.2f, "
                "stagnation=%d→boost=%.2f, gen=%d/%d)",
                mutation_boost, progress, stats.diversity, diversity_boost,
                stagnation_level, stagnation_boost,
                self.generation, self.max_generations,
            )
        stats.adaptive_mutation_rate = mutation_boost

        # Phase 14.18: 精英强制去重 — 防止同一agent_id重复进入new_generation
        # 根因: gen10 Top3全是agent-gen9-7e9d10d5(同一ID重复3次)
        # 修复: deepcopy前按agent_id去重, 重复的用随机新个体补充
        _elite_seen = set()
        _deduped_elite = []
        for _a in elite:
            if _a.agent_id not in _elite_seen:
                _deduped_elite.append(_a)
                _elite_seen.add(_a.agent_id)
        if len(_deduped_elite) < len(elite):
            logger.warning(
                "Phase 14.18 精英去重: %d→%d (移除%d个重复)",
                len(elite), len(_deduped_elite), len(elite) - len(_deduped_elite),
            )
        elite = _deduped_elite
        new_generation = [copy.deepcopy(a) for a in elite]
        new_generation_count = self.population_size - len(new_generation)

        # v2.7: IPOP式重启 — 停滞Level 3时保留Top-10%精英，其余重新随机
        # 来源：Auger & Hansen 2005 IPOP-CMA-ES + jestersimmps Evolving Trader 2026
        # 核心思想：连续25代无改进说明种群陷入深度局部最优，需要大规模重启
        # v2.18: 保留比例从Top-5提升到Top-10%（至少5个）
        #   原因：v2.17.2沙盘显示GT max=125.5但mean=9.4，Top-5丢失了过多优质基因
        #   依据：CMA-ES社区实践推荐保留10-15%避免重启后优质基因流失
        if stagnation_level >= 3:
            # v2.18: 保留Top-10%精英（至少5个）
            restart_elite_count = max(5, len(elite) // 10)
            restart_elite_count = min(restart_elite_count, len(elite))
            restart_elite = [copy.deepcopy(a) for a in elite[:restart_elite_count]]
            # 其余全部重新随机生成
            new_generation = restart_elite
            new_generation_count = self.population_size - len(new_generation)
            # v2.18 P1c: 重启时50%使用种子库驱动基因，避免丢失策略多样性
            # 来源：v2.17.2沙盘显示40+策略类型分布，IPOP重启后丢失部分策略
            # 设计：让一半个体使用种子库（保证策略覆盖），另一半完全随机（保证探索）
            seed_injection_count = new_generation_count // 2
            for i in range(new_generation_count):
                if i < seed_injection_count:
                    # v2.35: 日内模式下 IPOP restart 也走日内约束
                    if self._intraday_mode:
                        from .gene_codec import generate_intraday_gene
                        random_gene = generate_intraday_gene(generation=self.generation + 1)
                    else:
                        random_gene = generate_seed_based_gene(
                            generation=self.generation + 1, err_lessons=err_lessons,
                        )
                    random_gene.agent_id = f"restart-seed-{self.generation + 1}-{i:03d}"
                else:
                    if self._intraday_mode:
                        from .gene_codec import generate_intraday_gene
                        random_gene = generate_intraday_gene(generation=self.generation + 1)
                    else:
                        random_gene = generate_random_gene(
                            generation=self.generation + 1, err_lessons=err_lessons,
                        )
                    random_gene.agent_id = f"restart-{self.generation + 1}-{i:03d}"
                new_generation.append(random_gene)
            # 重置停滞计数器（给重启后的种群新的机会）
            self.generations_since_improvement = 0
            # v2.22 CRITICAL修复：IPOP restart时设置重置标志
            # 原BUG: _all_trial_sharpes和_n_trials永不重置，IPOP后新种群被旧trials惩罚
            #         50轮×200 Agent=10000+试验，V[SR_k]极大，DSR→0（无论Sharpe多高）
            # 修复：设置restart_triggered标志，evolution_loop检测后重置DSR状态
            # 来源：RCMAES 2026 (CEC竞赛) — restart应重置搜索状态
            self._restart_triggered = True
            logger.warning(
                "IPOP restart: kept %d elites, regenerated %d random individuals "
                "(stagnation level 3, gen %d)",
                restart_elite_count, new_generation_count, self.generation,
            )
            stats.random_injection_count = new_generation_count
            stats.new_born_count = new_generation_count

            self._save_evaluated_population(ranked, scores)
            self.population = new_generation
            self.generation += 1

            # Step 7: 岛屿迁移（每N代）
            if self.n_islands > 1 and self.generation % self.migration_interval == 0:
                self._migrate()

            self.history.append(stats)
            self._save_population()

            # Feature Map统计
            fm_cells = self.feature_map.get_cell_count() if self.feature_map else 0
            logger.info(
                "Evolution round %d (IPOP restart): GT-Score mean=%.1f max=%.1f diversity=%.3f fm_cells=%d "
                "overfit=%d wf_ratio=%.2f wf_status=%s wf_valid=%d wf_negative=%d mut_boost=%.2f inject=%d",
                self.generation, stats.mean_gt_score, stats.max_gt_score, stats.diversity, fm_cells,
                stats.overfit_detected_count, stats.mean_wf_ratio, stats.wf_status,
                stats.wf_valid_count, stats.wf_negative_count,
                stats.adaptive_mutation_rate, stats.random_injection_count,
            )
            return stats

        # R9: 随机注入机制 — 多样性低时注入随机个体
        # 来自前沿GA的重启机制：保留精英+随机重置部分个体
        # R9.1优化：触发阈值从0.08提高到0.12，注入比例从20%提高到30%
        # v2.11优化：触发阈值从0.12提高到0.15，注入比例从30%提高到40%
        #   理由：v2.10沙盘显示diversity=0.13-0.14，刚好高于0.12阈值不触发注入
        #         提高阈值到0.15，让diversity=0.13-0.14时也能触发随机注入
        #         增加注入比例到40%，更有效地打破基因同质化
        random_injection_count = 0
        # Phase 11.9: diversity根治 — 修复Phase 11.5结构bug
        #   原bug: if diversity<0.10只设injection_count但没注入循环(循环在elif内)
        #   修复: if/elif只设injection_count, 注入循环统一在后面执行
        #   diversity<0.10: 重度注入60% (原40%不够, gen466-491连续-4.5证明)
        #   diversity<0.15: 轻度注入40% (原逻辑保留)
        # Phase 13.3: 提高注入阈值 (diversity=0.0439远低于0.10)
        if stats.diversity < 0.12:  # Phase 13.3: 0.10→0.12
            # 根治(2026-07-16): 注入量从15%→30%，最小3个个体
            # 原bug: injection_count=max(1,int(10*0.15))=1, 每次只注入1个, diversity永远卡在0.05
            # 修复: max(3, 30%)确保至少注入3个个体(seed+perturb+xover各1)
            injection_count = max(3, int(new_generation_count * 0.30))  # 根治: 15%→30%, min 3
        elif stats.diversity < 0.18:  # Phase 13.3: 0.15→0.18
            injection_count = max(2, int(new_generation_count * 0.20))  # 根治: min 2
        else:
            injection_count = 0

        if injection_count > 0:
            # Phase 14.16h: Multi-strategy diversity injection
            # 旧: 60% seed + 40% random (2 strategies, diversity stuck at 0.04)
            # Phase 14.16j: 20% seed + 20% perturbation + 20% crossover + 40% random (aggressive exploration)
            # 目标: diversity 0.057→0.06+ (aggressive random exploration)
            # Phase 14.16n: 注入前预计算现有种群style_vector, 注入时去重(避免agent[10]==agent[17]重复)
            _existing_vectors = [a.compute_style_vector() for a in self.population]
            _dedup_max = 5  # 每个注入个体最多重试5次去重
            seed_inject_count = max(1, int(injection_count * 0.20))  # Phase 14.16j: 30%→20%
            perturb_inject_count = max(1, int(injection_count * 0.30))  # Phase 14.16l.2: 20%→30%
            crossover_inject_count = max(1, int(injection_count * 0.20))  # Phase 14.16j: 20% (unchanged)
            for i in range(injection_count):
                if i < seed_inject_count:
                    # Strategy 1: Seed-based (worldview coverage)
                    if self._intraday_mode:
                        from .gene_codec import generate_intraday_gene
                        random_gene = generate_intraday_gene(generation=self.generation + 1)
                    else:
                        random_gene = generate_seed_based_gene(
                            generation=self.generation + 1, err_lessons=err_lessons,
                        )
                    random_gene.agent_id = f"injected-seed-{self.generation + 1}-{i:03d}-{uuid.uuid4().hex[:8]}"
                elif i < seed_inject_count + perturb_inject_count:
                    # Phase 14.16h Strategy 2: Gene perturbation (elite + significant mutation)
                    # Take a random elite, deepcopy, perturb key genes by ±50%
                    if elite:
                        _perturb_src = copy.deepcopy(random.choice(elite))
                        _perturb_src.generation = self.generation + 1
                        _perturb_src.survival_rounds = 0
                        # Perturb key genes for diversity
                        for _fn in ['sl_multiplier', 'tp_multiplier', 'position_size_multiplier',
                                     'signal_score_multiplier', 'contrarian_bias', 'trend_filter_strength',
                                     'max_hold_minutes', 'max_positions']:
                            if hasattr(_perturb_src, _fn):
                                _orig = getattr(_perturb_src, _fn)
                                if isinstance(_orig, (int, float)) and _orig != 0:
                                    _factor = random.uniform(0.2, 1.8)  # Phase 14.16j: ±50%→±80% aggressive perturbation
                                    setattr(_perturb_src, _fn, type(_orig)(_orig * _factor))
                        random_gene = _perturb_src
                        random_gene.agent_id = f"injected-perturb-{self.generation + 1}-{i:03d}-{uuid.uuid4().hex[:8]}"
                    else:
                        random_gene = generate_random_gene(generation=self.generation + 1, err_lessons=err_lessons)
                        random_gene.agent_id = f"injected-{self.generation + 1}-{i:03d}"
                elif i < seed_inject_count + perturb_inject_count + crossover_inject_count:
                    # Phase 14.16h Strategy 3: Crossover-based (novel combinations from elite)
                    if len(elite) >= 2:
                        _pa = random.choice(elite)
                        _pb = random.choice(elite)
                        _guard = 0
                        while _pb is _pa and len(elite) > 1 and _guard < 5:
                            _pb = random.choice(elite)
                            _guard += 1
                        try:
                            from .gene_evolution import crossover as gene_crossover
                            random_gene = gene_crossover(
                                _pa, _pb,
                                generation=self.generation + 1,
                                mutation_rate_boost=1.5,  # Phase 14.16j: 1.0→1.5 aggressive crossover mutation
                                intraday_mode=self._intraday_mode,
                                err_lessons=err_lessons,
                            )
                            random_gene.agent_id = f"injected-xover-{self.generation + 1}-{i:03d}-{uuid.uuid4().hex[:8]}"
                        except Exception:
                            random_gene = generate_random_gene(generation=self.generation + 1, err_lessons=err_lessons)
                            random_gene.agent_id = f"injected-{self.generation + 1}-{i:03d}"
                    else:
                        random_gene = generate_random_gene(generation=self.generation + 1, err_lessons=err_lessons)
                        random_gene.agent_id = f"injected-{self.generation + 1}-{i:03d}"
                else:
                    # Strategy 4: Random (exploration)
                    if self._intraday_mode:
                        from .gene_codec import generate_intraday_gene
                        random_gene = generate_intraday_gene(generation=self.generation + 1)
                    else:
                        random_gene = generate_random_gene(
                            generation=self.generation + 1, err_lessons=err_lessons,
                        )
                    random_gene.agent_id = f"injected-{self.generation + 1}-{i:03d}"
                # Phase 14.16n: 注入去重检查 - 如果新个体与现有种群余弦相似度>0.99则强制扰动
                _new_vec = random_gene.compute_style_vector()
                _dedup_attempts_local = 0
                while _dedup_attempts_local < _dedup_max:
                    _is_dup = False
                    for _ev in _existing_vectors:
                        # 简化余弦相似度检查
                        _dot = sum(_new_vec[k] * _ev[k] for k in _new_vec)
                        _na = sum(v * v for v in _new_vec.values()) ** 0.5
                        _nb = sum(v * v for v in _ev.values()) ** 0.5
                        if _na > 0 and _nb > 0:
                            _sim = _dot / (_na * _nb)
                            if _sim > 0.99:
                                _is_dup = True
                                break
                    if not _is_dup:
                        break
                    # 重复, 强制扰动关键维度
                    _dedup_attempts_local += 1
                    if hasattr(random_gene, 'entry_logic_entry_aggressiveness'):
                        _factor = 0.3 + 0.4 * _dedup_attempts_local  # 0.3-0.7
                        _orig = random_gene.entry_logic_entry_aggressiveness
                        random_gene.entry_logic_entry_aggressiveness = min(1.0, max(0.0, _orig * (1.0 + _factor)))
                    if hasattr(random_gene, 'exit_risk_max_hold_hours'):
                        random_gene.exit_risk_max_hold_hours = max(1, int(random_gene.exit_risk_max_hold_hours * (0.5 + 0.3 * _dedup_attempts_local)))
                    _new_vec = random_gene.compute_style_vector()
                if _dedup_attempts_local >= _dedup_max:
                    logger.warning("Phase 14.16n dedup: agent=%s still duplicate after %d attempts",
                                 random_gene.agent_id[:12], _dedup_max)
                _existing_vectors.append(_new_vec)  # 更新现有向量库
                new_generation.append(random_gene)
                random_injection_count += 1
            new_generation_count -= injection_count
            _rand_count = injection_count - seed_inject_count - perturb_inject_count - crossover_inject_count
            logger.warning(
                "Phase 14.16h diversity multi-inject: diversity=%.3f, total=%d (seed=%d, perturb=%d, xover=%d, random=%d)",
                stats.diversity, injection_count, seed_inject_count, perturb_inject_count,
                crossover_inject_count, max(0, _rand_count),
            )
        stats.random_injection_count = random_injection_count

        # Phase 11.6: 连续3代max_gt<=-4.5提前触发部分重启 (不等stagnation_level 3的25代)
        # 根因: gen466-491连续-4.5表明种群多样性崩溃, 现有stagnation检测需要25代太慢
        # 根治: 连续3代-4.5时, 额外注入20%纯随机个体 (不依赖stagnation计数器)
        _consecutive_neg45 = 0
        for _hist_stats in reversed(self.history[-5:]):  # 检查最近5代
            if hasattr(_hist_stats, 'max_gt_score') and _hist_stats.max_gt_score <= -4.0:
                _consecutive_neg45 += 1
            else:
                break  # 遇到正分代次, 中断计数
        # 当前代也是-4.5时, 计数+1
        if stats.max_gt_score <= -4.0:
            _consecutive_neg45 += 1
        if _consecutive_neg45 >= 3:
            _extra_inject = max(1, int(new_generation_count * 0.20))
            for _i in range(_extra_inject):
                if self._intraday_mode:
                    from .gene_codec import generate_intraday_gene
                    _extra_gene = generate_intraday_gene(generation=self.generation + 1)
                else:
                    _extra_gene = generate_random_gene(
                        generation=self.generation + 1, err_lessons=err_lessons,
                    )
                _extra_gene.agent_id = f"neg45-rescue-{self.generation + 1}-{_i:03d}"
                new_generation.append(_extra_gene)
                random_injection_count += 1
            new_generation_count -= _extra_inject
            stats.random_injection_count = random_injection_count
            logger.warning(
                "Phase 11.6: 连续%d代GT<=-4.5, 额外注入%d纯随机个体 (提前重启)",
                _consecutive_neg45, _extra_inject,
            )

        # Phase 14.19: HIGH GT 回注 — 从 HallOfFame 注入高分策略
        # 根因: SCORE_CRASHED 时只注入随机个体, 丢失历史好基因
        # 修复: 每代检查 HallOfFame, 如果有存档则注入 Top-2 高分策略
        # 来源: CMA-ES HallOfFame 重启机制 (Auger & Hansen 2005)
        try:
            _hof = getattr(self, '_hall_of_fame', [])
            if _hof and len(_hof) > 0:
                # 每10代注入2个 HallOfFame 策略 (与随机注入互补)
                if self.generation > 0 and self.generation % 10 == 0:
                    _hof_inject_count = min(2, len(_hof))
                    for _hi in range(_hof_inject_count):
                        _hof_entry = _hof[_hi]  # Top-N 已排序
                        # 从 HallOfFame 条目重建基因
                        if 'gene_dict' in _hof_entry:
                            from .gene_codec import AgentGene as Gene  # Phase 14.23b: Gene→AgentGene
                            try:
                                _hof_gene = Gene.from_dict(_hof_entry['gene_dict'])
                                _hof_gene.agent_id = f"hof-inject-{self.generation + 1}-{_hi:03d}"
                                new_generation.append(_hof_gene)
                                logger.warning(
                                    "Phase 14.19 HOF注入: gen=%d 注入HallOfFame策略#%d (sharpe=%.2f dsr=%.2f)",
                                    self.generation, _hi,
                                    _hof_entry.get('sharpe', 0), _hof_entry.get('dsr', 0),
                                )
                            except Exception as _he:
                                logger.warning("Phase 14.19 HOF注入失败(非致命): %s", _he)
        except Exception as _e:
            logger.warning("Phase 14.19 HOF注入异常(非致命): %s", _e)

        # Phase 14.19: SCORE_CRASHED 紧急回注
        # 检测: 本代 max_gt < 前代 max_gt * 0.3 (大幅崩溃)
        # 修复: 从最近一代有正分的 population 文件回注 top 2
        try:
            if len(self.history) >= 2:
                _prev_stats = self.history[-1]
                _prev_max_gt = getattr(_prev_stats, 'max_gt_score', 0.0)
                _cur_max_gt = getattr(stats, 'max_gt_score', 0.0)
                if _prev_max_gt > 10.0 and _cur_max_gt < _prev_max_gt * 0.3:
                    # SCORE_CRASHED 检测到
                    logger.warning(
                        "Phase 14.19 SCORE_CRASHED: prev_max_gt=%.2f → cur_max_gt=%.2f (跌幅%.0f%%)",
                        _prev_max_gt, _cur_max_gt,
                        (1 - _cur_max_gt / max(_prev_max_gt, 0.01)) * 100,
                    )
                    # 紧急回注: 从 HallOfFame 注入
                    _hof = getattr(self, '_hall_of_fame', [])
                    if _hof:
                        _crash_inject = min(3, len(_hof))
                        for _ci in range(_crash_inject):
                            _hof_entry = _hof[_ci]
                            if 'gene_dict' in _hof_entry:
                                from .gene_codec import AgentGene as Gene  # Phase 14.23b: Gene→AgentGene
                                try:
                                    _crash_gene = Gene.from_dict(_hof_entry['gene_dict'])
                                    _crash_gene.agent_id = f"crash-rescue-{self.generation + 1}-{_ci:03d}"
                                    new_generation.append(_crash_gene)
                                    logger.warning(
                                        "Phase 14.19 CRASH_RESCUE: 注入HallOfFame策略#%d (sharpe=%.2f)",
                                        _ci, _hof_entry.get('sharpe', 0),
                                    )
                                except Exception:
                                    pass
        except Exception as _e:
            logger.warning("Phase 14.19 SCORE_CRASHED检测异常(非致命): %s", _e)

        # Phase 14.22: CRASH_RESCUE从population文件回注 (HOF为空时的fallback)
        # 根因: HOF为空时CRASH_RESCUE无策略可注, SCORE_CRASHED后无法恢复
        # 修复: 从最近一代有正分的population文件回注top 3策略
        try:
            if len(self.history) >= 2:
                _prev_stats = self.history[-1]
                _prev_max_gt = getattr(_prev_stats, 'max_gt_score', 0.0)
                _cur_max_gt = getattr(stats, 'max_gt_score', 0.0)
                _hof = getattr(self, '_hall_of_fame', [])
                if _prev_max_gt > 1.0 and _cur_max_gt < _prev_max_gt * 0.3 and not _hof:
                    # HOF为空且SCORE_CRASHED → 从population文件回注
                    import json as _json
                    _data_dir = getattr(self, 'data_dir', '')
                    _crash_gen = self.generation  # 当前崩溃的代
                    # 向前搜索最近一代有正分的population文件
                    for _search_gen in range(_crash_gen, max(_crash_gen - 5, -1), -1):  # Phase 14.23: 包含当前崩溃世代(含回测结果)
                        _pop_fp = os.path.join(_data_dir, f'population_gen{_search_gen:04d}.jsonl')
                        if not os.path.exists(_pop_fp):
                            continue
                        _pop_agents = []
                        with open(_pop_fp) as _pf:
                            for _line in _pf:
                                try:
                                    _pop_agents.append(_json.loads(_line))
                                except Exception:
                                    pass
                        if not _pop_agents:
                            continue
                        # 按fitness_score或sharpe_ratio排序
                        def _get_score(a):
                            _s = a.get('fitness_score', a.get('sharpe_ratio', -999))
                            try:
                                return float(_s)
                            except:
                                return -999.0
                        _pop_agents.sort(key=_get_score, reverse=True)
                        _top_n = min(3, len(_pop_agents))
                        _injected = 0
                        for _ti in range(_top_n):
                            _pa = _pop_agents[_ti]
                            _score = _get_score(_pa)
                            if _score <= 0:
                                continue  # 跳过负分策略
                            try:
                                from .gene_codec import AgentGene
                                _rescue_gene = AgentGene.from_dict(_pa)
                                _rescue_gene.agent_id = f"crash-rescue-{self.generation + 1}-{_ti:03d}"
                                new_generation.append(_rescue_gene)
                                _injected += 1
                                logger.warning(
                                    "Phase 14.22 CRASH_RESCUE from FILE: gen=%d 注入gen%d策略#%d (score=%.2f, sharpe=%.2f, win=%.1f%%)",
                                    self.generation, _search_gen, _ti,
                                    _score, float(_pa.get('sharpe_ratio', 0)),
                                    float(_pa.get('win_rate', 0)) * 100,
                                )
                            except Exception as _re:
                                logger.warning("Phase 14.22 CRASH_RESCUE重建失败(非致命): %s", _re)
                        if _injected > 0:
                            logger.warning(
                                "Phase 14.22 CRASH_RESCUE: 从gen%d文件回注%d个策略到gen%d",
                                _search_gen, _injected, self.generation + 1,
                            )
                            break  # 成功回注,不再搜索更早的代
        except Exception as _e:
            logger.warning("Phase 14.22 CRASH_RESCUE文件回注异常(非致命): %s", _e)

        # v2.14.1性能优化：预计算所有ranked的style_vector，避免在_tournament_select中重复计算
        # 每次进化调用_tournament_select约190次，每次计算5个候选的style_vector
        # 预计算后只需1次遍历，减少约950次重复计算
        ranked_vectors = {a.agent_id: a.compute_style_vector() for a in ranked}

        for i in range(new_generation_count):
            # 从帕累托前沿选父代a，锦标赛选父代b
            # P0-1: 使用全局确定性RNG替代random.choice
            parent_a = global_rng_manager.global_rng.choice(elite)
            # v2.14: Restricted Mating — 传入parent_a，选择基因差异大的父代b
            # v2.14.1: 传入ranked_vectors，使用预计算的style_vector
            parent_b = self._tournament_select(ranked, scores, parent_a=parent_a, ranked_vectors=ranked_vectors)

            child = crossover(
                parent_a, parent_b,
                generation=self.generation + 1,
                mutation_rate_boost=mutation_boost,
                intraday_mode=self._intraday_mode,
                err_lessons=err_lessons,  # v4.0 Phase 6: ERR教训注入闭环
            )
            new_generation.append(child)

            # v4.0 Phase 6 Task #29.2: 记录 ERR 教训注入映射 (闭环验证用)
            # 用户铁律: "迭代经验和教训库使用起来"
            if err_lessons:
                self._err_injection_count += 1
                child_key = f"gen{self.generation + 1}_idx{i}"
                self._gene_err_map[child_key] = [l.get("err_id", "unknown") for l in err_lessons]

        # v2.16.2: 回退Crowding机制 — 实验证明代价过高
        # 来源：De Jong 1975, Mahfoud 1992 "Crowding and Niching"
        #
        # 实验数据（500轮沙盘）：
        #   v2.15.2（无Crowding）: GT max=117.7, mean=2.4, diversity=0.138
        #   v2.16（threshold=0.95）: GT max=107.0, mean=1.8, diversity=0.158
        #   v2.16.1（threshold=0.98, max_removal=20%）: GT max=42.1, diversity=0.169
        #
        # 结论：
        #   - Crowding确实提升diversity（+0.031），但GT-Score严重下降（-75.6）
        #   - 每轮移除30+个体并注入随机新个体，拉低整体种群质量
        #   - 随机新个体大多不交易，导致GT-Score mean下降
        #   - diversity提升的收益远不抵GT-Score下降的代价
        #
        # v2.16.2决策：回退Crowding，保留v2.15.2的Novelty bonus改革为最优方案
        # _crowding_dedup方法保留在下方以供未来研究参考，但不再调用
        # new_generation = self._crowding_dedup(new_generation, scores)

        # Phase 14.16o Fix 1c: HIGH GT基因回注 — 每10代注入top-2存档个体
        # 根因: 后10代(90-99)max_gt平均=0.068, HIGH GT基因完全丢失
        # 修复: 每10代从存档注入2个HIGH GT个体, 替换new_generation中fitness最低的
        try:
            if (len(self._high_gt_archive) > 0 and
                self.generation > 0 and
                self.generation % self._high_gt_inject_interval == 0):
                _inject_k = min(self._high_gt_inject_k, len(self._high_gt_archive))
                _injected = 0
                _seen_injection_hashes = set()
                for _idx in range(len(self._high_gt_archive)):
                    if _injected >= _inject_k:
                        break
                    _gt_val, _archive_agent, _archive_gen = self._high_gt_archive[_idx]
                    _gene_payload = _archive_agent.to_dict()
                    _gene_payload = {
                        k: v for k, v in _gene_payload.items()
                        if k not in ("agent_id", "generation", "survival_rounds", "fitness_score")
                    }
                    _gene_hash = str(hash(json.dumps(_gene_payload, sort_keys=True, default=str)))
                    _last_injected = self._high_gt_last_injected.get(_gene_hash, -10**9)
                    if _gene_hash in _seen_injection_hashes:
                        continue
                    if self.generation - _last_injected < self._high_gt_inject_cooldown:
                        continue
                    _seen_injection_hashes.add(_gene_hash)
                    _reborn = copy.deepcopy(_archive_agent)
                    _reborn.agent_id = f"reborn-hgt-{self.generation + 1}-{_idx:03d}"
                    # 清除fitness_score, 强制重新评估
                    _reborn.fitness_score = 0.0
                    # 替换new_generation中fitness_score最低的个体
                    if len(new_generation) > 0:
                        _min_idx = min(
                            range(len(new_generation)),
                            key=lambda i: getattr(new_generation[i], 'fitness_score', -999.0),
                        )
                        _replaced = new_generation[_min_idx]
                        new_generation[_min_idx] = _reborn
                        _injected += 1
                        self._high_gt_last_injected[_gene_hash] = self.generation
                        logger.info(
                            "Phase N HIGH GT REBORN: gen=%d reborn agent=%s (gt=%.4f from gen %d hash=%s) replaced %s",
                            self.generation + 1, _reborn.agent_id[:12], _gt_val, _archive_gen, _gene_hash, _replaced.agent_id[:12],
                        )
                if _injected > 0:
                    logger.info(
                        "Phase 14.16o: gen=%d injected %d HIGH GT agents from archive (archive_size=%d)",
                        self.generation + 1, _injected, len(self._high_gt_archive),
                    )
        except Exception as _e:
            logger.warning("Phase 14.16o HIGH GT inject failed (non-fatal): %s", _e)

        # Phase 14.16o Fix 2: 全种群去重 — 确保无重复agent_id
        # 根因: gen 45有5个相同agent_id(injected-perturb-44-002-6)的重复个体
        # 修复: new_generation最终组装后,按agent_id去重, 重复个体重新分配ID
        try:
            _seen_ids = set()
            _deduped_pop = []
            _dup_count = 0
            for _a in new_generation:
                if _a.agent_id not in _seen_ids:
                    _deduped_pop.append(_a)
                    _seen_ids.add(_a.agent_id)
                else:
                    # 重复个体, 生成新agent_id
                    _a.agent_id = f"dedup-fix-{self.generation + 1}-{len(_deduped_pop):03d}"
                    _deduped_pop.append(_a)
                    _dup_count += 1
            if _dup_count > 0:
                logger.warning(
                    "Phase 14.16o DEDUP: gen=%d fixed %d duplicate agent_ids in new_generation",
                    self.generation + 1, _dup_count,
                )
            new_generation = _deduped_pop
        except Exception as _e:
            logger.warning("Phase 14.16o dedup failed (non-fatal): %s", _e)

        self._save_evaluated_population(ranked, scores)
        self.population = new_generation
        self.generation += 1
        stats.new_born_count = new_generation_count

        # Step 7: 岛屿迁移（每N代）
        if self.n_islands > 1 and self.generation % self.migration_interval == 0:
            self._migrate()

        # execution_realism_penalty: 存储本轮scores和trades供evolution_loop重评估
        self._last_scores = scores
        self._last_trades_by_agent = trades_by_agent
        self.history.append(stats)
        self._save_population()

        # v2.7: 清空行为特征缓存（避免内存泄漏，新种群需要重新计算）
        self.behavior_diversity.clear_cache()

        # Feature Map统计
        fm_cells = self.feature_map.get_cell_count() if self.feature_map else 0
        map_elites_cells = self.behavior_diversity.elite_grid.get_cell_count()
        logger.info(
            "Evolution round %d: GT-Score mean=%.1f max=%.1f diversity=%.3f fm_cells=%d "
            "me_cells=%d overfit=%d wf_ratio=%.2f wf_status=%s wf_valid=%d wf_negative=%d mut_boost=%.2f inject=%d",
            self.generation, stats.mean_gt_score, stats.max_gt_score, stats.diversity, fm_cells,
            map_elites_cells,
            stats.overfit_detected_count, stats.mean_wf_ratio, stats.wf_status,
            stats.wf_valid_count, stats.wf_negative_count,
            stats.adaptive_mutation_rate, stats.random_injection_count,
        )

        # Phase 14.16e: Top-3精英基因追踪 (用户指令1)
        self._log_top3_elite_tracking(ranked, scores)

        return stats

    def _log_top3_elite_tracking(
        self,
        ranked: List["AgentGene"],
        scores: Dict[str, "FitnessScore"],
    ) -> None:
        """Phase 14.16e: Top-3精英基因哈希追踪 + 下一代存活追踪

        用户指令1: 输出Top-3精英的基因哈希, 追踪其是否在下一代中完整存活。
        区分两种情况:
        - 个体丢失: 下一代中无相同基因哈希 → 被淘汰
        - 分数崩溃: 个体存活但gt_score从高变0 → 硬门控触发(total_pnl<0)
        """
        import hashlib

        # 取Top-3精英
        top3 = ranked[:3] if len(ranked) >= 3 else ranked[:]
        if not top3:
            return

        # 计算当前代Top-3的基因哈希
        current_hashes: Dict[str, str] = {}  # agent_id -> gene_hash
        current_info: List[Dict[str, Any]] = []
        for agent in top3:
            try:
                gene_dict = agent.to_dict()
                # 排除agent_id和generation(这些会变), 只追踪基因参数
                gene_for_hash = {k: v for k, v in gene_dict.items()
                                 if k not in ("agent_id", "generation", "survival_rounds")}
                gene_str = json.dumps(gene_for_hash, sort_keys=True, default=str)
                gene_hash = hashlib.md5(gene_str.encode("utf-8")).hexdigest()[:12]
            except Exception:
                gene_hash = "ERROR"

            gt = scores[agent.agent_id].gt_score if agent.agent_id in scores else 0.0
            pnl = scores[agent.agent_id].total_pnl if agent.agent_id in scores else 0.0
            current_hashes[agent.agent_id] = gene_hash
            current_info.append({
                "agent_id": agent.agent_id,
                "gene_hash": gene_hash,
                "gt_score": round(gt, 4),
                "total_pnl": round(pnl, 4),
                "worldview": getattr(agent, "worldview_primary", "?"),
            })

        # 与上一代对比, 追踪存活
        prev_hashes = getattr(self, "_last_top3_hashes", {})
        prev_info = getattr(self, "_last_top3_info", [])

        if prev_info:
            # 输出上一代Top-3在当前代的存活情况
            # 当前代所有agent_id集合
            current_agent_ids = {a.agent_id for a in self.population}
            # 当前代所有基因哈希集合
            current_gene_hashes = set()
            for a in self.population:
                try:
                    gd = a.to_dict()
                    gf = {k: v for k, v in gd.items()
                          if k not in ("agent_id", "generation", "survival_rounds")}
                    gs = json.dumps(gf, sort_keys=True, default=str)
                    current_gene_hashes.add(hashlib.md5(gs.encode("utf-8")).hexdigest()[:12])
                except Exception:
                    pass

            logger.info(
                "=== Phase 14.16e Top-3 Elite Tracking (gen %d → gen %d) ===",
                self.generation - 1, self.generation,
            )
            for prev in prev_info:
                aid = prev["agent_id"]
                gh = prev["gene_hash"]
                # 检查是否存活: 1)相同agent_id 2)相同基因哈希
                survived_by_id = aid in current_agent_ids
                survived_by_gene = gh in current_gene_hashes
                # 如果存活, 查看当前分数
                if survived_by_id:
                    # 找到当前代中该agent的分数
                    curr_gt = None
                    curr_pnl = None
                    for a in self.population:
                        if a.agent_id == aid and aid in scores:
                            curr_gt = scores[aid].gt_score
                            curr_pnl = scores[aid].total_pnl
                            break
                    status = "ALIVE" if curr_gt and curr_gt > 0.01 else "SCORE_CRASHED"
                    logger.info(
                        "  [%s] %s hash=%s gen%d_gt=%.4f gen%d_gt=%.4f gen%d_pnl=%.4f → %s",
                        prev["worldview"], aid[:24], gh,
                        self.generation - 1, prev["gt_score"],
                        self.generation, curr_gt or 0.0,
                        self.generation, curr_pnl or 0.0,
                        status,
                    )
                elif survived_by_gene:
                    logger.info(
                        "  [GENE_COPIED] %s %s hash=%s gen%d_gt=%.4f → 基因在gen%d中存活(ID变更)",
                        prev["worldview"], aid[:24], gh,
                        self.generation - 1, prev["gt_score"],
                        self.generation,
                    )
                else:
                    logger.info(
                        "  [LOST] %s %s hash=%s gen%d_gt=%.4f → gen%d中未存活(被淘汰)",
                        prev["worldview"], aid[:24], gh,
                        self.generation - 1, prev["gt_score"],
                        self.generation,
                    )

        # 输出当前代Top-3
        logger.info(
            "=== Top-3 Elite gen %d ===", self.generation,
        )
        for i, info in enumerate(current_info):
            logger.info(
                "  #%d %s hash=%s gt=%.4f pnl=%.4f wv=%s",
                i + 1, info["agent_id"][:24], info["gene_hash"],
                info["gt_score"], info["total_pnl"], info["worldview"],
            )

        # 存储当前代哈希供下一代对比
        self._last_top3_hashes = current_hashes
        self._last_top3_info = current_info

    def _crowding_dedup(
        self,
        population: List[AgentGene],
        scores: Dict[str, FitnessScore],
        similarity_threshold: float = 0.98,
    ) -> List[AgentGene]:
        """v2.16.1: Crowding去重 — 移除基因过于相似的个体，提升diversity

        来源：De Jong 1975, Mahfoud 1992 "Crowding and Niching"
        核心思想：新种群中如果存在基因相似度>阈值的个体对，保留GT-Score更高的
        被移除的个体用随机新个体填充，保持种群规模

        v2.16.1调整：
          - 阈值从0.95提高到0.98（只去除几乎完全相同的个体）
          - 限制移除数量最多20%（防止过度移除导致种群不稳定）
          - v2.16问题：阈值0.95移除82%个体，导致GT-Score max/mean下降

        参数：
          similarity_threshold: 相似度阈值，默认0.98（几乎完全相同才去重）
          - 0.98: 只去除几乎完全相同的个体（保守，v2.16.1默认）
          - 0.95: 去除高度相似的个体（v2.16，过度激进）
          - 0.90: 去除中等相似的个体（可能过度去重）

        性能：O(n²)相似度计算，200个体约20ms
        """
        if len(population) <= 1:
            return population

        # 预计算所有个体的style_vector
        vectors = {a.agent_id: a.compute_style_vector() for a in population}

        # 标记需要移除的个体（被更优的相似个体击败）
        removed_ids: Set[str] = set()
        n = len(population)
        # v2.16.1: 限制移除数量最多20%，防止过度移除
        max_removal = max(1, n // 5)  # 200个体最多移除40个

        for i in range(n):
            if len(removed_ids) >= max_removal:
                break  # 达到移除上限，停止
            agent_i = population[i]
            if agent_i.agent_id in removed_ids:
                continue

            for j in range(i + 1, n):
                agent_j = population[j]
                if agent_j.agent_id in removed_ids:
                    continue

                sim = gene_similarity_from_vectors(
                    vectors[agent_i.agent_id],
                    vectors[agent_j.agent_id],
                )

                if sim > similarity_threshold:
                    # 相似度>阈值，保留GT-Score更高的
                    gt_i = scores.get(agent_i.agent_id, FitnessScore()).gt_score
                    gt_j = scores.get(agent_j.agent_id, FitnessScore()).gt_score

                    if gt_i >= gt_j:
                        removed_ids.add(agent_j.agent_id)
                    else:
                        removed_ids.add(agent_i.agent_id)
                        break  # agent_i被移除，跳出内循环

            if len(removed_ids) >= max_removal:
                break  # 达到移除上限，停止

        if not removed_ids:
            return population

        # 构建新种群：保留未被移除的个体 + 随机新个体填充
        result = [a for a in population if a.agent_id not in removed_ids]
        removed_count = len(removed_ids)

        # 用随机新个体填充
        for k in range(removed_count):
            # v2.35: 日内模式下 Crowding 注入也走日内约束
            if self._intraday_mode:
                from .gene_codec import generate_intraday_gene
                new_gene = generate_intraday_gene(generation=self.generation + 1)
            else:
                new_gene = generate_random_gene(generation=self.generation + 1)
            new_gene.agent_id = f"crowding-{self.generation + 1}-{k:03d}"
            result.append(new_gene)

        logger.info(
            "v2.16.1 Crowding: removed %d similar individuals (max=%d), injected %d random "
            "(threshold=%.2f, pop_size=%d)",
            removed_count, max_removal, removed_count, similarity_threshold, len(result),
        )

        return result

    def _tournament_select(
        self,
        ranked: List[AgentGene],
        scores: Dict[str, FitnessScore],
        tournament_size: int = 3,
        parent_a: Optional[AgentGene] = None,
        ranked_vectors: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> AgentGene:
        """锦标赛选择：从随机候选者中选适应度最高的

        v2.14 Restricted Mating（限制竞争选择）：
          来源：Deb & Goldberg 1989 "An Investigation of Niche and Species Formation"
          核心思想：仅允许基因差异大的个体交配，避免近亲繁殖
          实现方式：当提供parent_a时，选择候选中GT-Score最高且与parent_a基因差异最大的
          公式：selection_score = gt_score + diversity_bonus
                 diversity_bonus = (1 - similarity_to_parent_a) * gt_score * 0.3
          效果：保持GT-Score选择压力的同时，增加后代diversity

        v2.14.1性能优化：
          预计算ranked_vectors，避免在每次调用中重复计算候选的style_vector
          每次进化调用_tournament_select约190次，预计算后减少约950次重复计算
        """
        # P0-1: 使用全局确定性RNG替代random.sample
        # v2.14: 候选数从3提升到5，增加Restricted Mating的选择空间
        k = 5 if parent_a is not None else tournament_size
        candidates = global_rng_manager.global_rng.sample(
            ranked, min(k, len(ranked))
        )
        # v2.14: 标准锦标赛选择（无parent_a时）
        if parent_a is None:
            return max(candidates, key=lambda a: scores[a.agent_id].gt_score)
        # v2.14: Restricted Mating — 综合GT-Score和基因差异
        # v2.14.1: 使用预计算的style_vector
        try:
            parent_a_vector = (
                ranked_vectors[parent_a.agent_id]
                if ranked_vectors and parent_a.agent_id in ranked_vectors
                else parent_a.compute_style_vector()
            )
            best_candidate = None
            best_score = -1.0
            for c in candidates:
                gt = scores[c.agent_id].gt_score
                c_vector = (
                    ranked_vectors[c.agent_id]
                    if ranked_vectors and c.agent_id in ranked_vectors
                    else c.compute_style_vector()
                )
                sim = gene_similarity_from_vectors(parent_a_vector, c_vector)
                # diversity_bonus: 基因差异越大，奖励越高（最多+30% GT-Score）
                diversity_bonus = (1.0 - sim) * gt * 0.3
                selection_score = gt + diversity_bonus
                if selection_score > best_score:
                    best_score = selection_score
                    best_candidate = c
            return best_candidate if best_candidate is not None else candidates[0]
        except Exception:
            # 降级为标准锦标赛选择
            return max(candidates, key=lambda a: scores[a.agent_id].gt_score)

    def _migrate(self) -> None:
        """岛屿迁移 — 单向环拓扑，混合精英+多样性迁移

        v2.7升级（来源：Zychowski et al. PPRAI 2025 + OpenEvolve Island Model）：
          1. 修复migrant效应：保留原agent_id，避免下一轮找不到交易记录导致GT-Score=0
          2. 混合迁移策略：50%精英 + 50%多样性最大化，避免过度传播单一基因
          3. 单向环拓扑（理论证明优于全连接，ACM 2016动态优化）
        """
        if len(self.population) < self.n_islands * 2:
            return

        # 将种群分为岛屿
        island_size = len(self.population) // self.n_islands
        islands = []
        for i in range(self.n_islands):
            start = i * island_size
            end = start + island_size if i < self.n_islands - 1 else len(self.population)
            islands.append(self.population[start:end])

        # 迁移：每岛迁移前migrate_rate%到下一岛（单向环）
        n_migrate = max(1, int(island_size * self.migration_rate))
        # v2.7: 混合策略 — 一半精英，一半多样性最大化
        n_elite_migrate = max(1, n_migrate // 2)
        n_diverse_migrate = n_migrate - n_elite_migrate

        for i in range(self.n_islands):
            src_island = islands[i]
            dst_island = islands[(i + 1) % self.n_islands]

            # 50%精英迁移：选src岛中最优的
            elite_migrants = sorted(
                src_island, key=lambda a: a.fitness_score, reverse=True
            )[:n_elite_migrate]

            # 50%多样性迁移：选src岛中行为特征最独特的
            # 来源：Zychowski 2025实证 — 混合策略最优
            if n_diverse_migrate > 0 and len(src_island) > n_elite_migrate:
                # 计算src岛中每个Agent到岛内其他Agent的平均距离
                src_vectors = [a.compute_style_vector() for a in src_island]
                diverse_candidates = []
                for idx, agent in enumerate(src_island):
                    if agent in elite_migrants:
                        continue
                    va = src_vectors[idx]
                    # 到岛内其他Agent的平均距离（越大越独特）
                    dists = []
                    for jdx, other_vec in enumerate(src_vectors):
                        if idx == jdx:
                            continue
                        sim = gene_similarity_from_vectors(va, other_vec)
                        dists.append(1.0 - sim)
                    avg_dist = sum(dists) / len(dists) if dists else 0.0
                    diverse_candidates.append((avg_dist, agent))
                diverse_candidates.sort(key=lambda x: -x[0])  # 距离大的优先
                diverse_migrants = [c[1] for c in diverse_candidates[:n_diverse_migrate]]
            else:
                diverse_migrants = []

            all_migrants = elite_migrants + diverse_migrants

            # 替换dst岛中最差的n_migrate个
            dst_sorted = sorted(range(len(dst_island)), key=lambda idx: dst_island[idx].fitness_score)
            for j, idx in enumerate(dst_sorted[:n_migrate]):
                migrant = copy.deepcopy(all_migrants[j % len(all_migrants)])
                # v2.7修复：保留原agent_id，避免下一轮trades_by_agent找不到交易记录
                # 迁移历史记录在parent_ids中（不改变agent_id）
                # 来源：OpenEvolve Island Model — 迁移时携带完整metadata
                migrant.parent_ids = migrant.parent_ids + [f"migrated_from_island{i}"]
                dst_island[idx] = migrant

        self.population = []
        for island in islands:
            self.population.extend(island)

        logger.info(
            "Island migration: %d islands, %d migrants each (50%% elite + 50%% diverse, ring topology)",
            self.n_islands, n_migrate,
        )

    # ------------------------------------------------------------------
    # 精英查询
    # ------------------------------------------------------------------
    def set_regime(self, regime: str) -> None:
        """Phase 8.1: Set current market regime for NSGA-III weight adaptation.

        Called by EvolutionLoop when regime_detector detects a regime change.
        NSGA3Selector uses this to dynamically adjust sharpe/win_rate/
        profit_factor/drawdown weights.

        Regime name mapping (regime_detector -> NSGA3Selector):
            bull     -> trend   (bull market = trend, weight sharpe)
            sideways -> range   (sideways = range, weight win_rate)
            bear     -> crisis  (bear = crisis, weight low drawdown)
            neutral  -> neutral

        Args:
            regime: Market regime string from regime_detector
                    ("bull", "bear", "sideways", "neutral")
        """
        # Phase 8.1: Map regime_detector names to NSGA3Selector names
        _REGIME_MAP = {
            "bull": "trend",
            "sideways": "range",
            "bear": "crisis",
            "neutral": "neutral",
        }
        self._current_regime = regime
        if self._nsga3_selector is not None:
            mapped = _REGIME_MAP.get(regime, "neutral")
            self._nsga3_selector.set_regime(mapped)


    def get_elite(self, top_n: int = 5) -> List[AgentGene]:
        """获取当前种群中适应度最高的N个Agent

        v2.7升级：优先从MAP-Elites网格获取行为多样化的精英
        来源：Mouret & Clune 2015 — Quality-Diversity算法

        优先级：
          1. MAP-Elites网格（v2.7新增）— 行为空间每个niche的最优
          2. Feature Map（R8）— 4维行为特征分格
          3. GT-Score排名（回退）
        """
        # Phase 14.8.6: 统一过滤 — 所有路径只返回已运行个体
        # 原bug: MAP-Elites/Feature Map路径不过滤注入个体(survival_rounds=0)
        _eligible = [a for a in self.population if getattr(a, 'survival_rounds', 0) > 0]
        if not _eligible:
            _eligible = [a for a in self.population if getattr(a, 'total_trades', 0) > 0]
        if not _eligible:
            _eligible = list(self.population)

        # v2.7: 优先从MAP-Elites网格获取行为多样化的精英
        if self.behavior_diversity.elite_grid.get_cell_count() >= top_n:
            diverse_elite = self.behavior_diversity.get_diverse_elite(_eligible, top_n)
            if len(diverse_elite) >= top_n:
                return diverse_elite[:top_n]

        if self.feature_map and self.feature_map.get_cell_count() > 0:
            # 从Feature Map获取行为多样化的精英ID
            diverse_ids = set(self.feature_map.get_diverse_elite(top_n))
            if len(diverse_ids) >= top_n:
                # Feature Map有足够的格子，直接返回
                result = []
                for a in _eligible:
                    if a.agent_id in diverse_ids:
                        result.append(a)
                if len(result) >= top_n:
                    return result[:top_n]

        # 回退到GT-Score排名 (使用已过滤的_eligible)
        sorted_pop = sorted(
            _eligible,
            key=lambda a: (getattr(a, 'total_trades', 0) > 0, a.fitness_score),
            reverse=True,
        )
        return sorted_pop[:top_n]

    def get_elite_signals(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取精英Agent的交易信号（供实盘使用）

        返回格式:
          [{"agent_id": ..., "gt_score": ..., "worldview": ..., "signal": "long"/"short"/"neutral"}]
        """
        elite = self.get_elite(top_n)
        return [
            {
                "agent_id": a.agent_id,
                "gt_score": a.fitness_score,
                "sharpe": a.sharpe_ratio,
                "win_rate": a.win_rate,
                "worldview": a.worldview_primary,
                "style_label": a.style_label,
                "survival_rounds": a.survival_rounds,
                "trade_count": a.total_trades,
            }
            for a in elite
        ]

    # ------------------------------------------------------------------
    # 多样性维护
    # ------------------------------------------------------------------

    def compute_diversity(self) -> float:
        """计算当前种群多样性（1-平均相似度）"""
        if len(self.population) < 2:
            return 1.0
        # P4#2: 预计算 style_vector，避免 O(n²) 循环中重复调用 compute_style_vector
        _vectors = [a.compute_style_vector() for a in self.population]
        total_sim = 0.0
        pair_count = 0
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                total_sim += gene_similarity_from_vectors(_vectors[i], _vectors[j])
                pair_count += 1
        return 1.0 - (total_sim / pair_count)

    def inject_diversity(self, count: int = 10) -> None:  # Phase 14.3: 5→10
        """注入随机新个体以维持多样性（替换最差的）"""
        if len(self.population) < 2:
            return

        # 淘汰最差的
        ranked = sorted(self.population, key=lambda a: a.fitness_score)
        self.population = ranked[count:]

        # 注入随机新个体
        for i in range(count):
            # v2.35: 日内模式下多样性注入也走日内约束
            if self._intraday_mode:
                from .gene_codec import generate_intraday_gene
                gene = generate_intraday_gene(generation=self.generation)
            else:
                gene = generate_random_gene(generation=self.generation)
            gene.agent_id = f"diversity-gen{self.generation}-{i:03d}"
            self.population.append(gene)

        logger.info("Injected %d random agents for diversity", count)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_evaluated_population(
        self,
        evaluated_population: List[AgentGene],
        scores: Dict[str, FitnessScore],
    ) -> None:
        """保存与当前统计严格对应的已评估种群快照"""
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, f"evaluated_gen{self.generation:04d}.jsonl")
        fd, temp_path = tempfile.mkstemp(dir=self.data_dir, prefix=".evaluated_population_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for agent in evaluated_population:
                    score = scores[agent.agent_id]
                    record = agent.to_dict()
                    record.update({
                        "fitness_score": score.gt_score,
                        "sharpe_ratio": score.sharpe_ratio,
                        "total_trades": score.total_trades,
                        "win_rate": score.win_rate,
                        "total_pnl": score.total_pnl,
                        "max_drawdown_realized": score.max_drawdown,
                    })
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temp_path, filepath)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _save_population(self) -> None:
        """保存当前种群到文件"""
        # 确保数据目录存在（修复 E2E 测试发现的目录缺失 bug）
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, f"population_gen{self.generation:04d}.jsonl")
        fd, temp_path = tempfile.mkstemp(dir=self.data_dir, prefix=".population_", suffix=".tmp")
        os.close(fd)
        try:
            genes_to_jsonl(self.population, temp_path)
            os.replace(temp_path, filepath)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

        # 同时保存统计历史
        stats_path = os.path.join(self.data_dir, "evolution_stats.jsonl")
        with open(stats_path, "a", encoding="utf-8") as f:
            for stat in self.history[-1:]:
                f.write(json.dumps({
                    "generation": stat.generation,
                    "population_size": stat.population_size,
                    "mean_gt_score": stat.mean_gt_score,
                    "max_gt_score": stat.max_gt_score,
                    "max_gt_score_raw": getattr(stat, 'max_gt_score_raw', stat.max_gt_score),  # Phase 14.16p
                    "mean_sharpe": stat.mean_sharpe,
                    "diversity": stat.diversity,
                    "elite_count": stat.elite_count,
                    "pareto_front_size": stat.pareto_front_size,
                    "feature_map_cells": stat.feature_map_cells,
                    "worldview_distribution": stat.worldview_distribution,
                    "timestamp": stat.timestamp,
                }, ensure_ascii=False) + "\n")

    def load_population(self, generation: int = -1) -> List[AgentGene]:
        """加载指定代的种群（-1=最新）

        v2.9修复：取所有Agent中最大的generation，而非第一个的generation
        原bug：精英deepcopy保留旧generation，导致加载Gen 2而非Gen 10
        """
        if generation < 0:
            # 找最新的代
            candidates = []
            for filename in os.listdir(self.data_dir):
                if not (filename.startswith("population_gen") and filename.endswith(".jsonl")):
                    continue
                try:
                    candidate_generation = int(filename[len("population_gen"):-len(".jsonl")])
                except ValueError:
                    continue
                candidate_path = os.path.join(self.data_dir, filename)
                if os.path.isfile(candidate_path) and os.path.getsize(candidate_path) > 0:
                    candidates.append((candidate_generation, candidate_path))
            if not candidates:
                raise FileNotFoundError(f"No population files found in {self.data_dir}")
            filepath = max(candidates, key=lambda item: item[0])[1]
        else:
            filepath = os.path.join(self.data_dir, f"population_gen{generation:04d}.jsonl")

        self.population = genes_from_jsonl(filepath)
        if self.population:
            # v2.9修复：取所有Agent中最大的generation（精英可能保留旧代数）
            self.generation = max(a.generation for a in self.population)
        return self.population

    def get_stats_history(self) -> List[Dict[str, Any]]:
        """获取进化统计历史"""
        stats_path = os.path.join(self.data_dir, "evolution_stats.jsonl")
        if not os.path.exists(stats_path):
            return []
        history = []
        with open(stats_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
        return history

    # ------------------------------------------------------------------
    # 风格回溯标签
    # ------------------------------------------------------------------

    def label_styles(self) -> None:
        """为存活的Agent回溯贴风格标签

        基于存活下来的Agent的实际表现（夏普、胜率、交易频率等），
        而非预设，贴标签。标签用于人类理解，不影响进化。
        """
        for agent in self.population:
            if agent.total_trades < 10:
                agent.style_label = "rookie"
                continue

            if agent.sharpe_ratio > 1.5 and agent.max_drawdown_realized < 0.15:
                agent.style_label = "elite_hunter"
            elif agent.sharpe_ratio > 0.8 and agent.win_rate > 0.55:
                agent.style_label = "steady_earner"
            elif agent.win_rate > 0.70 and agent.total_trades > 50:
                agent.style_label = "scalper"
            elif agent.max_drawdown_realized > 0.30:
                agent.style_label = "high_roller"
            elif agent.sharpe_ratio < 0:
                agent.style_label = "struggler"
            elif agent.total_trades < 20:
                agent.style_label = "patient_hunter"
            else:
                agent.style_label = "generalist"

        # 分布统计
        labels = defaultdict(int)
        for a in self.population:
            labels[a.style_label] += 1
        logger.info("Style labels: %s", dict(labels))

# Phase 7.12: Extracted modules (backward-compatible re-export)
from .population_fitness import (
    FitnessScore,
    compute_fitness,
)
from .population_nsga import (
    fast_non_dominated_sort,
    nsga2_select,
)
from .population_feature_map import (
    FeatureMap,
)
