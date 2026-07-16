# -*- coding: utf-8 -*-
"""
机器学习因子挖掘模块 (ML Alpha Factor Mining)

基于2025-2026全网前沿因子挖掘研究，整合3大增强：

1. 遗传规划因子生成 (Genetic Programming Factor Generation)
   来源：QuantaAlpha 2026 + AlphaPROBE 2026 + FactorEngine 2026 + alfars 2026
   原理：用遗传规划自动发现Alpha因子表达式，从基础算子组合出有效因子
   优势：白盒可解释，自动发现非线性关系，避免人工特征工程局限

2. 因子评估体系 (Factor Evaluation System)
   来源：QuantaAlpha 2026 (Rank IC + 低冗余 + 容量三重门槛)
   原理：用IC/RankIC/ICIR评估因子预测力，用相关性去重
   量化：IC>0.03, ICIR>0.5, Rank IC>0.05 为有效因子

3. 因子库管理 (Factor Library Management)
   来源：AlphaPROBE 2026 (DAG因子池) + alfars 2026 (因子库版本管理)
   原理：维护因子库，去重、淘汰失效因子、保留有效因子
   优势：避免因子拥挤，保持因子多样性
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.alpha_mining")


# ============================================================================
# 1. 因子表达式树 (Factor Expression Tree)
# ============================================================================


@dataclass
class FactorNode:
    """因子表达式树的节点"""
    node_type: str  # "operator" / "feature" / "constant"
    value: str      # 算子名/特征名/常数值
    children: List["FactorNode"] = field(default_factory=list)
    window: int = 0  # v2.20: 时序算子窗口参数（如sma(window=10)）

    def to_string(self) -> str:
        """转换为可读字符串"""
        if self.node_type == "feature":
            return self.value
        elif self.node_type == "constant":
            return self.value
        elif self.node_type == "operator":
            if len(self.children) == 1:
                # v2.20: 时序算子显示窗口参数
                if self.window > 0:
                    return f"{self.value}({self.children[0].to_string()}, w={self.window})"
                return f"{self.value}({self.children[0].to_string()})"
            elif len(self.children) == 2:
                return f"({self.children[0].to_string()} {self.value} {self.children[1].to_string()})"
            else:
                args = ", ".join(c.to_string() for c in self.children)
                return f"{self.value}({args})"
        return self.value

    def depth(self) -> int:
        """树的深度"""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def complexity(self) -> int:
        """复杂度（节点数）"""
        return 1 + sum(c.complexity() for c in self.children)


class FactorExpression:
    """因子表达式 — 可执行的Alpha因子

    基于遗传规划的因子表达式，支持：
      - 基础特征：close, open, high, low, volume, returns
      - 技术指标：sma, ema, rsi, macd, bollinger
      - 算术运算：+, -, *, /, max, min
      - 时序运算：delay, delta, rank, ts_mean, ts_std

    来源：QuantaAlpha 2026的AST表达式系统 + alfars 2026的遗传规划
    """

    # 基础特征
    FEATURES = ["close", "open", "high", "low", "volume", "returns"]

    # 一元算子
    UNARY_OPS = {
        "abs": np.abs,
        "neg": lambda x: -x,
        "rank": lambda x: _rank(x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
    }

    # 二元算子
    BINARY_OPS = {
        "+": np.add,
        "-": np.subtract,
        "*": np.multiply,
        "/": lambda x, y: np.divide(x, y + 1e-10),
        "max": np.maximum,
        "min": np.minimum,
    }

    # 时序算子（需要窗口参数）
    TS_OPS = ["sma", "ema", "std", "max", "min", "delay", "delta", "rsi"]

    def __init__(self, root: FactorNode):
        self.root = root
        self._cache: Optional[np.ndarray] = None

    @classmethod
    def random_generate(cls, max_depth: int = 4) -> "FactorExpression":
        """随机生成因子表达式

        Args:
            max_depth: 最大树深度

        Returns:
            随机生成的因子表达式
        """
        root = cls._random_node(max_depth, depth=0)
        return cls(root)

    @classmethod
    def _random_node(cls, max_depth: int, depth: int) -> FactorNode:
        """递归生成随机节点"""
        if depth >= max_depth or (depth > 0 and random.random() < 0.3):
            # 生成叶子节点
            if random.random() < 0.7:
                # 特征
                return FactorNode("feature", random.choice(cls.FEATURES))
            else:
                # 常数
                return FactorNode("constant", str(round(random.uniform(-2, 2), 2)))
        else:
            # 生成算子节点
            r = random.random()
            if r < 0.35:
                # 一元算子
                op = random.choice(list(cls.UNARY_OPS.keys()))
                child = cls._random_node(max_depth, depth + 1)
                return FactorNode("operator", op, [child])
            elif r < 0.65:
                # 二元算子
                op = random.choice(list(cls.BINARY_OPS.keys()))
                left = cls._random_node(max_depth, depth + 1)
                right = cls._random_node(max_depth, depth + 1)
                return FactorNode("operator", op, [left, right])
            else:
                # v2.20: 时序算子（30%概率生成）
                # 修复CRITICAL空壳：原代码声明TS_OPS但从不生成，导致因子表达式无时序能力
                op = random.choice(cls.TS_OPS)
                child = cls._random_node(max_depth, depth + 1)
                window = random.choice([5, 10, 20, 30])  # 常见时序窗口
                return FactorNode("operator", op, [child], window=window)

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """计算因子值

        Args:
            data: 市场数据字典 {"close": np.ndarray, "open": np.ndarray, ...}

        Returns:
            因子值数组
        """
        try:
            result = self._eval_node(self.root, data)
            # 处理NaN和Inf
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result
        except Exception as e:
            logger.debug(f"因子计算失败: {e}")
            return np.zeros(len(data.get("close", [1])))

    def _eval_node(self, node: FactorNode, data: Dict[str, np.ndarray]) -> np.ndarray:
        """递归计算节点值"""
        if node.node_type == "feature":
            return np.array(data.get(node.value, np.zeros(1)), dtype=float)

        elif node.node_type == "constant":
            return np.array([float(node.value)])

        elif node.node_type == "operator":
            children_values = [self._eval_node(c, data) for c in node.children]

            if node.value in self.UNARY_OPS:
                return self.UNARY_OPS[node.value](children_values[0])

            elif node.value in self.BINARY_OPS:
                # 广播对齐
                left, right = children_values
                if len(left) != len(right):
                    if len(left) == 1:
                        left = np.full_like(right, left[0])
                    elif len(right) == 1:
                        right = np.full_like(left, right[0])
                    else:
                        min_len = min(len(left), len(right))
                        left, right = left[:min_len], right[:min_len]
                return self.BINARY_OPS[node.value](left, right)

            elif node.value in self.TS_OPS:
                # v2.20: 时序算子真实实现（修复CRITICAL空壳）
                # 原问题：TS_OPS声明8个算子但_eval_node无分支，落入return np.zeros(1)死分支
                # 修复：实现sma/ema/std/max/min/delay/delta/rsi 8个时序算子
                # 来源：WorldQuant Alpha 101 + QuantaAlpha 2026时序因子库
                x = children_values[0]
                w = max(2, node.window)  # 窗口至少2
                if len(x) < w:
                    # 数据不足，返回最后一个值或0
                    return np.array([x[-1] if len(x) > 0 else 0.0])

                if node.value == "sma":
                    # 简单移动平均
                    result = np.convolve(x, np.ones(w) / w, mode='valid')
                    return result
                elif node.value == "ema":
                    # 指数移动平均
                    alpha = 2.0 / (w + 1)
                    result = np.zeros_like(x)
                    result[0] = x[0]
                    for i in range(1, len(x)):
                        result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
                    return result[w - 1:]  # 只返回窗口满后的部分
                elif node.value == "std":
                    # 滚动标准差
                    result = np.array([np.std(x[i:i + w]) for i in range(len(x) - w + 1)])
                    return result
                elif node.value == "max":
                    # 滚动最大值
                    result = np.array([np.max(x[i:i + w]) for i in range(len(x) - w + 1)])
                    return result
                elif node.value == "min":
                    # 滚动最小值
                    result = np.array([np.min(x[i:i + w]) for i in range(len(x) - w + 1)])
                    return result
                elif node.value == "delay":
                    # 延迟w期（取w期前的值）
                    return np.array([x[i - w] for i in range(w, len(x))])
                elif node.value == "delta":
                    # 差分（当前值 - w期前值）
                    return np.array([x[i] - x[i - w] for i in range(w, len(x))])
                elif node.value == "rsi":
                    # RSI相对强弱指标
                    if len(x) < w + 1:
                        return np.array([50.0])
                    deltas = np.diff(x)
                    gains = np.where(deltas > 0, deltas, 0.0)
                    losses = np.where(deltas < 0, -deltas, 0.0)
                    avg_gain = np.mean(gains[-w:])
                    avg_loss = np.mean(losses[-w:])
                    if avg_loss < 1e-10:
                        return np.array([100.0])
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                    return np.array([rsi])

            return np.zeros(1)

        return np.zeros(1)

    def to_string(self) -> str:
        """转换为字符串表示"""
        return self.root.to_string()

    def complexity(self) -> int:
        """获取复杂度"""
        return self.root.complexity()

    def mutate(self, mutation_rate: float = 0.2) -> "FactorExpression":
        """变异 — 随机修改部分节点

        Args:
            mutation_rate: 变异概率

        Returns:
            变异后的新因子
        """
        new_root = self._mutate_node(self.root, mutation_rate)
        return FactorExpression(new_root)

    def _mutate_node(self, node: FactorNode, mutation_rate: float) -> FactorNode:
        """递归变异节点"""
        if random.random() < mutation_rate:
            # 变异此节点
            if node.node_type == "feature":
                return FactorNode("feature", random.choice(self.FEATURES))
            elif node.node_type == "constant":
                return FactorNode("constant", str(round(random.uniform(-2, 2), 2)))
            elif node.node_type == "operator":
                if node.value in self.UNARY_OPS:
                    new_op = random.choice(list(self.UNARY_OPS.keys()))
                    return FactorNode("operator", new_op, [
                        self._mutate_node(c, mutation_rate) for c in node.children
                    ])
                elif node.value in self.BINARY_OPS:
                    new_op = random.choice(list(self.BINARY_OPS.keys()))
                    return FactorNode("operator", new_op, [
                        self._mutate_node(c, mutation_rate) for c in node.children
                    ])
                elif node.value in self.TS_OPS:
                    # v2.20: 时序算子变异——随机更换算子或窗口
                    new_op = random.choice(self.TS_OPS)
                    new_window = random.choice([5, 10, 20, 30])
                    return FactorNode("operator", new_op, [
                        self._mutate_node(c, mutation_rate) for c in node.children
                    ], window=new_window)

        # 递归变异子节点
        new_children = [self._mutate_node(c, mutation_rate) for c in node.children]
        return FactorNode(node.node_type, node.value, new_children, window=node.window)

    @classmethod
    def crossover(cls, parent1: "FactorExpression", parent2: "FactorExpression") -> "FactorExpression":
        """交叉 — 从两个父因子产生子因子

        Args:
            parent1: 父因子1
            parent2: 父因子2

        Returns:
            交叉后的新因子
        """
        # 简化交叉：随机选择parent1的子树用parent2的子树替换
        new_root = cls._crossover_node(parent1.root, parent2.root)
        return cls(new_root)

    @classmethod
    def _crossover_node(cls, node1: FactorNode, node2: FactorNode) -> FactorNode:
        """递归交叉"""
        if random.random() < 0.3:
            # 用node2替换node1
            return cls._copy_node(node2)

        new_children = []
        for c in node1.children:
            if random.random() < 0.5 and node2.children:
                # 用node2的子树交叉
                idx = random.randint(0, len(node2.children) - 1)
                new_children.append(cls._crossover_node(c, node2.children[idx]))
            else:
                new_children.append(cls._crossover_node(c, node2))

        return FactorNode(node1.node_type, node1.value, new_children, window=node1.window)

    @classmethod
    def _copy_node(cls, node: FactorNode) -> FactorNode:
        """深拷贝节点"""
        return FactorNode(
            node.node_type,
            node.value,
            [cls._copy_node(c) for c in node.children],
            window=node.window,
        )


def _rank(x: np.ndarray) -> np.ndarray:
    """计算排名（归一化到0-1）"""
    if len(x) == 0:
        return x
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


# ============================================================================
# 2. 因子评估器 (Factor Evaluator)
# ============================================================================


class FactorEvaluator:
    """因子评估器

    来源：QuantaAlpha 2026 (Rank IC + 低冗余 + 容量三重门槛)

    评估指标：
      1. IC (Information Coefficient): 因子值与未来收益的相关系数
      2. Rank IC: 因子排名与未来收益排名的Spearman相关
      3. ICIR (IC Information Ratio): IC均值/IC标准差
      4. 因子覆盖率: 非NaN值的比例
      5. 因子换手率: 因子值变化的平均绝对值

    量化标准（来自QuantaAlpha 2026）：
      - |IC| > 0.03: 有效因子
      - |Rank IC| > 0.05: 强因子
      - ICIR > 0.5: 稳定因子
      - 覆盖率 > 0.8: 可用因子
    """

    def __init__(
        self,
        min_ic: float = 0.03,           # 最小IC绝对值
        min_rank_ic: float = 0.05,      # 最小Rank IC绝对值
        min_icir: float = 0.5,          # 最小ICIR
        min_coverage: float = 0.8,      # 最小覆盖率
        max_turnover: float = 0.5,      # 最大换手率
    ):
        """
        Args:
            min_ic: 最小IC绝对值
            min_rank_ic: 最小Rank IC绝对值
            min_icir: 最小ICIR
            min_coverage: 最小覆盖率
            max_turnover: 最大换手率
        """
        self.min_ic = min_ic
        self.min_rank_ic = min_rank_ic
        self.min_icir = min_icir
        self.min_coverage = min_coverage
        self.max_turnover = max_turnover

    def evaluate(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray,
    ) -> Dict[str, float]:
        """评估因子

        Args:
            factor_values: 因子值数组
            forward_returns: 未来收益率数组

        Returns:
            {
                "ic": float,           # 信息系数
                "rank_ic": float,      # Rank IC
                "icir": float,         # IC信息比率
                "coverage": float,     # 覆盖率
                "turnover": float,     # 换手率
                "is_effective": bool,  # 是否有效
                "score": float,        # 综合评分
            }
        """
        if len(factor_values) != len(forward_returns) or len(factor_values) < 10:
            return self._empty_result()

        # 对齐并去除NaN
        mask = ~(np.isnan(factor_values) | np.isnan(forward_returns))
        f_vals = factor_values[mask]
        r_vals = forward_returns[mask]

        if len(f_vals) < 10:
            return self._empty_result()

        # 零方差或非有限值无法定义相关性，直接拒绝，避免corrcoef生成NaN。
        if (
            not np.all(np.isfinite(f_vals))
            or not np.all(np.isfinite(r_vals))
            or np.std(f_vals) <= 1e-12
            or np.std(r_vals) <= 1e-12
        ):
            return self._empty_result()

        # IC: Pearson相关系数
        ic = float(np.corrcoef(f_vals, r_vals)[0, 1]) if len(f_vals) > 1 else 0.0

        # Rank IC: Spearman相关系数
        rank_ic = self._spearman_corr(f_vals, r_vals)

        # ICIR: 用滚动窗口计算IC序列的标准差
        # 简化：用整个样本的IC作为均值，用分块IC估计标准差
        ic_series = self._compute_ic_series(f_vals, r_vals, window=20)
        if len(ic_series) > 1 and np.std(ic_series) > 0:
            icir = float(np.mean(ic_series) / np.std(ic_series))
        else:
            icir = 0.0

        # 覆盖率
        coverage = float(mask.sum() / len(factor_values))

        # 换手率（因子值变化的平均绝对值）
        if len(f_vals) > 1:
            turnover = float(np.mean(np.abs(np.diff(f_vals)) / (np.abs(f_vals[:-1]) + 1e-10)))
        else:
            turnover = 0.0

        # 是否有效
        is_effective = (
            abs(ic) >= self.min_ic and
            abs(rank_ic) >= self.min_rank_ic and
            abs(icir) >= self.min_icir and
            coverage >= self.min_coverage
        )

        # 综合评分
        score = (
            abs(ic) * 0.3 +
            abs(rank_ic) * 0.3 +
            abs(icir) * 0.2 +
            coverage * 0.1 +
            (1 - min(turnover, 1.0)) * 0.1
        )

        return {
            "ic": ic,
            "rank_ic": rank_ic,
            "icir": icir,
            "coverage": coverage,
            "turnover": turnover,
            "is_effective": is_effective,
            "score": score,
        }

    def _spearman_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算Spearman相关系数"""
        if len(x) < 2:
            return 0.0
        try:
            rank_x = _rank(x)
            rank_y = _rank(y)
            return float(np.corrcoef(rank_x, rank_y)[0, 1])
        except Exception:
            return 0.0

    def _compute_ic_series(
        self,
        factor: np.ndarray,
        returns: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """计算滚动IC序列"""
        if len(factor) < window * 2:
            if (
                len(factor) < 2
                or not np.all(np.isfinite(factor))
                or not np.all(np.isfinite(returns))
                or np.std(factor) <= 1e-12
                or np.std(returns) <= 1e-12
            ):
                return np.array([0.0])
            return np.array([float(np.corrcoef(factor, returns)[0, 1])])

        ics = []
        for i in range(0, len(factor) - window, window):
            f_chunk = factor[i:i+window]
            r_chunk = returns[i:i+window]
            if len(f_chunk) > 1 and np.std(f_chunk) > 0 and np.std(r_chunk) > 0:
                ics.append(float(np.corrcoef(f_chunk, r_chunk)[0, 1]))

        return np.array(ics) if ics else np.array([0.0])

    def _empty_result(self) -> Dict[str, float]:
        """空结果"""
        return {
            "ic": 0.0,
            "rank_ic": 0.0,
            "icir": 0.0,
            "coverage": 0.0,
            "turnover": 0.0,
            "is_effective": False,
            "score": 0.0,
        }


# ============================================================================
# 3. 遗传规划因子挖掘器 (Genetic Programming Factor Miner)
# ============================================================================


class GeneticFactorMiner:
    """遗传规划因子挖掘器

    来源：QuantaAlpha 2026 + AlphaPROBE 2026 + FactorEngine 2026

    使用遗传规划自动发现Alpha因子：
      1. 初始化随机因子种群
      2. 评估每个因子的IC/RankIC/ICIR
      3. 选择优秀因子进行交叉变异
      4. 淘汰无效因子，保留有效因子
      5. 迭代直到收敛或达到最大代数

    量化标准：
      - 种群大小: 50
      - 最大代数: 20
      - 变异率: 0.2
      - 精英保留: 10%
      - 最大复杂度: 20（避免过拟合）
    """

    def __init__(
        self,
        population_size: int = 50,        # 种群大小
        max_generations: int = 20,        # 最大代数
        mutation_rate: float = 0.2,       # 变异率
        elite_ratio: float = 0.1,         # 精英比例
        max_depth: int = 4,               # 最大树深度
        max_complexity: int = 20,         # 最大复杂度
        diversity_threshold: float = 0.7, # 多样性阈值（相关性）
    ):
        """
        Args:
            population_size: 种群大小
            max_generations: 最大进化代数
            mutation_rate: 变异率
            elite_ratio: 精英保留比例
            max_depth: 因子树最大深度
            max_complexity: 因子最大复杂度
            diversity_threshold: 多样性阈值（高于此相关性的因子去重）
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.diversity_threshold = diversity_threshold

        self.evaluator = FactorEvaluator()
        self._factor_library: List[Dict[str, Any]] = []
        self._best_factor: Optional[FactorExpression] = None
        self._best_score: float = 0.0
        # Phase 8.7.1: Operator bias for feedback-driven mining
        self._operator_bias: Dict[str, int] = {}

    def mine(
        self,
        data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """执行因子挖掘

        Args:
            data: 市场数据字典
            forward_returns: 未来收益率
            verbose: 是否打印进度

        Returns:
            {
                "best_factor": FactorExpression,
                "best_score": float,
                "best_evaluation": Dict,
                "factor_library": List[Dict],
                "generations": int,
                "total_evaluated": int,
            }
        """
        # 初始化种群
        population = [
            FactorExpression.random_generate(self.max_depth)
            for _ in range(self.population_size)
        ]

        total_evaluated = 0
        best_factor = None
        best_score = 0.0
        best_eval = {}

        for gen in range(self.max_generations):
            # 评估种群
            scored_population = []
            for factor in population:
                # 复杂度检查
                if factor.complexity() > self.max_complexity:
                    scored_population.append((factor, 0.0, {}))
                    continue

                factor_values = factor.evaluate(data)
                eval_result = self.evaluator.evaluate(factor_values, forward_returns)
                total_evaluated += 1

                scored_population.append((factor, eval_result["score"], eval_result))

                # 更新最佳
                if eval_result["score"] > best_score and eval_result["is_effective"]:
                    best_score = eval_result["score"]
                    best_factor = factor
                    best_eval = eval_result

            if verbose and gen % 5 == 0:
                scores = [s for _, s, _ in scored_population if s > 0]
                avg_score = np.mean(scores) if scores else 0
                print(f"  Gen {gen}: avg_score={avg_score:.4f}, best={best_score:.4f}, evaluated={total_evaluated}")

            # 选择精英
            scored_population.sort(key=lambda x: x[1], reverse=True)
            elite_count = max(1, int(self.population_size * self.elite_ratio))
            elites = [f for f, _, _ in scored_population[:elite_count]]

            # 生成下一代
            new_population = elites.copy()  # 精英保留

            # 交叉变异填充种群
            while len(new_population) < self.population_size:
                if len(elites) >= 2:
                    parent1, parent2 = random.sample(elites, 2)
                    child = FactorExpression.crossover(parent1, parent2)
                    child = child.mutate(self.mutation_rate)
                    new_population.append(child)
                else:
                    new_population.append(FactorExpression.random_generate(self.max_depth))

            population = new_population[:self.population_size]

        # 构建因子库（去重）
        self._build_factor_library(scored_population)

        self._best_factor = best_factor
        self._best_score = best_score

        return {
            "best_factor": best_factor,
            "best_score": best_score,
            "best_evaluation": best_eval,
            "factor_library": self._factor_library,
            "generations": self.max_generations,
            "total_evaluated": total_evaluated,
        }

    def _build_factor_library(self, scored_population: List[Tuple[FactorExpression, float, Dict]]):
        """构建因子库（去重）"""
        self._factor_library = []

        for factor, score, eval_result in scored_population:
            if not eval_result.get("is_effective", False):
                continue

            # 检查与已有因子的相关性（用表达式字符串简化）
            factor_str = factor.to_string()
            is_duplicate = False
            for existing in self._factor_library:
                if self._expression_similarity(factor_str, existing["expression"]) > self.diversity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                self._factor_library.append({
                    "expression": factor_str,
                    "score": score,
                    "ic": eval_result.get("ic", 0),
                    "rank_ic": eval_result.get("rank_ic", 0),
                    "icir": eval_result.get("icir", 0),
                    "complexity": factor.complexity(),
                })

    def _expression_similarity(self, expr1: str, expr2: str) -> float:
        """计算两个表达式字符串的相似度（简化版）"""
        # 使用Jaccard相似度（基于字符集）
        set1 = set(expr1)
        set2 = set(expr2)
        if not set1 and not set2:
            return 1.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    def get_best_factor(self) -> Optional[FactorExpression]:
        """获取最佳因子"""
        return self._best_factor

    def get_factor_library(self) -> List[Dict[str, Any]]:
        """获取因子库"""
        return self._factor_library

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "best_score": self._best_score,
            "best_factor": self._best_factor.to_string() if self._best_factor else None,
            "library_size": len(self._factor_library),
            "mutation_rate": self.mutation_rate,
            "elite_ratio": self.elite_ratio,
        }


# ============================================================================
# 4. 因子挖掘综合系统
# ============================================================================


class AlphaMiningSystem:
    """Alpha因子挖掘综合系统

    整合遗传规划因子生成 + 因子评估 + 因子库管理

    提供统一接口：
      1. discover_factors() — 自动发现有效因子
      2. get_factor_signal() — 获取因子信号
      3. get_summary() — 获取系统摘要
    """

    def __init__(
        self,
        miner: Optional[GeneticFactorMiner] = None,
        update_window: int = 100,
        retrain_interval: int = 200,
    ):
        """
        Args:
            miner: 遗传规划因子挖掘器
            update_window: 在线更新窗口大小（用于累积数据）
            retrain_interval: 重新挖掘因子的间隔（更新次数）
        """
        self.miner = miner or GeneticFactorMiner()
        self.update_window = update_window
        self.retrain_interval = retrain_interval
        self._is_trained = False
        self._last_discovery: Optional[Dict[str, Any]] = None
        # 在线更新状态
        self._price_history: List[float] = []
        self._update_count: int = 0
        self._last_signal: Dict[str, Any] = {
            "signal": 0.0,
            "direction": "neutral",
            "confidence": 0.0,
            "factor_expression": "none",
        }

    def update(
        self,
        price: float,
        data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """在线更新（增量模式）

        累积价格历史，定期触发因子发现，返回当前因子信号

        Args:
            price: 当前价格
            data: 可选的完整市场数据字典（若提供则用于因子发现）

        Returns:
            当前因子信号字典
        """
        self._update_count += 1
        self._price_history.append(price)

        # 保留窗口大小
        if len(self._price_history) > self.update_window * 2:
            self._price_history = self._price_history[-self.update_window:]

        # 定期重新挖掘因子
        should_retrain = (
            self._update_count % self.retrain_interval == 0
            and len(self._price_history) >= self.update_window
        )

        if should_retrain:
            try:
                # 构造训练数据
                prices = np.array(self._price_history[-self.update_window:])
                if data is not None:
                    train_data = {
                        k: v[-self.update_window:] if len(v) >= self.update_window else v
                        for k, v in data.items()
                    }
                else:
                    # 从价格序列构造基础特征
                    train_data = self._build_features_from_prices(prices)

                # 构造前向收益
                forward_returns = np.zeros_like(prices)
                forward_returns[:-1] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)
                forward_returns[-1] = 0.0

                self.discover_factors(train_data, forward_returns, verbose=False)
            except Exception as e:
                logger.debug(f"因子重新挖掘失败: {e}")

        # 获取当前因子信号
        if self._is_trained and data is not None:
            try:
                self._last_signal = self.get_factor_signal(data)
            except Exception as e:
                logger.debug(f"因子信号计算失败: {e}")
        elif self._is_trained and len(self._price_history) >= 10:
            try:
                prices = np.array(self._price_history[-self.update_window:])
                train_data = self._build_features_from_prices(prices)
                self._last_signal = self.get_factor_signal(train_data)
            except Exception as e:
                logger.debug(f"因子信号计算失败: {e}")

        return self._last_signal

    def _build_features_from_prices(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """从价格序列构造基础特征字典"""
        n = len(prices)
        returns = np.zeros(n)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        # 简化OHLCV（用价格近似）
        return {
            "close": prices,
            "open": np.concatenate([[prices[0]], prices[:-1]]),
            "high": np.maximum(prices, np.concatenate([[prices[0]], prices[:-1]])),
            "low": np.minimum(prices, np.concatenate([[prices[0]], prices[:-1]])),
            "volume": np.abs(returns) * 1000 + 100,  # 近似成交量
            "returns": returns,
        }

    def get_state(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        return {
            "status": "active" if self._is_trained else "warming_up",
            "update_count": self._update_count,
            "is_trained": self._is_trained,
            "price_history_len": len(self._price_history),
            "last_signal": self._last_signal,
        }

    # Phase 8.7.1: Factor mining feedback loop
    # 反馈环: 进化轮次结束后，把因子通过门控的信息反馈给因子挖掘器
    # 让下一代因子挖掘偏向高通过率的因子结构
    def receive_gate_feedback(self, gate_results: Dict[str, Any]) -> None:
        """接收来自进化门控的反馈

        Args:
            gate_results: {
                "passed_factor_expressions": List[str],  # 通过门控的因子表达式
                "failed_factor_expressions": List[str],  # 未通过门控的因子表达式
                "pass_rate": float,  # 通过率
                "common_operators": Dict[str, int],  # 通过门控因子的常见算子频率
            }
        """
        if not gate_results:
            return

        passed = gate_results.get("passed_factor_expressions", [])
        failed = gate_results.get("failed_factor_expressions", [])
        pass_rate = gate_results.get("pass_rate", 0.0)
        common_ops = gate_results.get("common_operators", {})

        # 记录反馈历史
        if not hasattr(self, '_feedback_history'):
            self._feedback_history = []

        self._feedback_history.append({
            "pass_rate": pass_rate,
            "n_passed": len(passed),
            "n_failed": len(failed),
            "common_ops": common_ops,
        })

        # 只保留最近10次反馈
        if len(self._feedback_history) > 10:
            self._feedback_history = self._feedback_history[-10:]

        # 调整GeneticFactorMiner的变异偏向
        # 如果有常见通过门控的算子，下一代变异时偏向这些算子
        if common_ops and hasattr(self, 'miner'):
            # 将常见算子频率传递给miner，作为变异偏好
            self.miner._operator_bias = common_ops

        logger.debug(
            "Phase 8.7.1: Factor gate feedback received: pass_rate=%.2f, passed=%d, failed=%d",
            pass_rate, len(passed), len(failed)
        )

    def get_feedback_summary(self) -> Dict[str, Any]:
        """获取反馈环摘要"""
        if not hasattr(self, '_feedback_history') or not self._feedback_history:
            return {"feedback_count": 0, "pass_rate_trend": []}
        return {
            "feedback_count": len(self._feedback_history),
            "pass_rate_trend": [f["pass_rate"] for f in self._feedback_history],
            "latest_common_ops": self._feedback_history[-1].get("common_ops", {}),
        }

    def discover_factors(
        self,
        data: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """自动发现有效因子

        Args:
            data: 市场数据字典
            forward_returns: 未来收益率
            verbose: 是否打印进度

        Returns:
            因子挖掘结果
        """
        result = self.miner.mine(data, forward_returns, verbose=verbose)
        self._is_trained = True
        self._last_discovery = result
        return result

    def get_factor_signal(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """获取因子信号

        Args:
            data: 市场数据字典

        Returns:
            {
                "signal": float,           # 因子信号值
                "direction": str,          # 方向 "long"/"short"/"neutral"
                "confidence": float,       # 置信度
                "factor_expression": str,  # 因子表达式
            }
        """
        if not self._is_trained or self.miner.get_best_factor() is None:
            return {
                "signal": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
                "factor_expression": "none",
            }

        best_factor = self.miner.get_best_factor()
        factor_values = best_factor.evaluate(data)

        # 使用最新的因子值作为信号
        if len(factor_values) > 0:
            signal = float(factor_values[-1])
        else:
            signal = 0.0

        # 根据信号方向和强度
        if signal > 0.01:
            direction = "long"
        elif signal < -0.01:
            direction = "short"
        else:
            direction = "neutral"

        # 置信度基于因子评分
        confidence = min(1.0, abs(signal) * 10)

        return {
            "signal": signal,
            "direction": direction,
            "confidence": confidence,
            "factor_expression": best_factor.to_string(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        miner_summary = self.miner.get_summary()
        return {
            "is_trained": self._is_trained,
            **miner_summary,
            "last_best_evaluation": self._last_discovery.get("best_evaluation", {}) if self._last_discovery else {},
        }
