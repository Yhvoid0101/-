#!/usr/bin/env python3
"""特征自动生成模块 — Layer 8 学习反馈层

让系统在进化过程中自动尝试对原始数据做数学变换（移动平均、标准差、变化率等），
发现好的变换就保留，无用的就淘汰。系统自己"发明"适合当前市场的衍生特征。

核心理念:
  - 不手动写死RSI/MACD/KDJ等指标
  - 提供"数学运算工具箱"，让进化自己组合
  - 优胜劣汰：有用的变换保留，无用的淘汰

特征生成操作:
  - MA (移动平均): window=2~128
  - STD (标准差): window=2~128
  - ROC (变化率): window=1~32
  - RANK (分位数排名): window=10~100
  - DIFF (差分): window=1~10
  - ZSCORE (Z分数): window=10~100

用法:
    from sandbox_trading.feature_autogen import FeatureAutoGenerator
    gen = FeatureAutoGenerator(enabled=True)
    features = gen.generate(price=40000, volume=1000, oi=500000, funding_rate=0.0001)
    gen.feedback(score=10.5)  # 反馈特征效果, 用于优胜劣汰
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.feature_autogen")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass(slots=True)
class FeatureTransform:
    """单个特征变换定义"""
    name: str = ""
    input_field: str = ""      # 输入字段: price/volume/oi/funding_rate
    operation: str = ""        # 操作: ma/std/roc/rank/diff/zscore
    window: int = 10           # 窗口大小
    weight: float = 1.0        # 权重 (用于优胜劣汰)
    use_count: int = 0         # 使用次数
    feedback_count: int = 0    # 反馈次数
    score_sum: float = 0.0     # 累计评分
    avg_score: float = 0.0     # 平均评分 (基于反馈次数)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input_field": self.input_field,
            "operation": self.operation,
            "window": self.window,
            "weight": self.weight,
            "use_count": self.use_count,
            "feedback_count": self.feedback_count,
            "avg_score": self.avg_score,
        }


@dataclass(slots=True)
class GeneratedFeature:
    """生成的特征值"""
    name: str = ""
    value: float = 0.0
    transform: Optional[FeatureTransform] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
        }


# ============================================================================
# 特征自动生成器
# ============================================================================

class FeatureAutoGenerator:
    """特征自动生成器 — 数学工具箱 + 优胜劣汰

    工作流程:
      1. 初始化: 生成N个随机特征变换
      2. 生成: 对输入数据应用变换, 输出衍生特征
      3. 反馈: 接收策略评分, 更新特征权重
      4. 进化: 每M代淘汰低分特征, 生成新的
    """

    # 输入字段
    INPUT_FIELDS = ["price", "volume", "oi", "funding_rate"]

    # 数学操作
    OPERATIONS = ["ma", "std", "roc", "rank", "diff", "zscore"]

    # 窗口范围
    WINDOW_MIN = 2
    WINDOW_MAX = 64

    # 进化参数
    INITIAL_FEATURES = 10       # 初始特征数量
    MAX_FEATURES = 20           # 最大特征数量
    EVOLVE_EVERY = 10           # 每10代进化一次
    KILL_THRESHOLD = -1.0       # 平均评分低于此值淘汰
    MUTATION_RATE = 0.2         # 变异率

    def __init__(
        self,
        enabled: bool = True,
        initial_features: int = None,
        seed: int = 42,
    ):
        """初始化特征自动生成器

        Args:
            enabled: 是否启用
            initial_features: 初始特征数量 (默认INITIAL_FEATURES)
            seed: 随机种子
        """
        self.enabled = enabled
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        self._transforms: List[FeatureTransform] = []
        self._history: Dict[str, List[float]] = {}  # field -> history values
        self._generation = 0
        self._total_generated = 0
        self._total_feedback = 0

        # 统计
        self._stats = {
            "evolutions": 0,
            "killed": 0,
            "created": 0,
            "best_score": 0.0,
            "worst_score": 0.0,
        }

        if enabled:
            n_init = initial_features or self.INITIAL_FEATURES
            self._init_transforms(n_init)
            logger.info(
                "FeatureAutoGenerator initialized: transforms=%d fields=%s ops=%s",
                len(self._transforms), self.INPUT_FIELDS, self.OPERATIONS,
            )

    def _init_transforms(self, n: int):
        """初始化N个随机特征变换"""
        for i in range(n):
            t = self._random_transform()
            self._transforms.append(t)
        self._stats["created"] += n

    def _random_transform(self) -> FeatureTransform:
        """生成一个随机特征变换"""
        field = self._rng.choice(self.INPUT_FIELDS)
        op = self._rng.choice(self.OPERATIONS)
        window = self._rng.randint(self.WINDOW_MIN, self.WINDOW_MAX)
        name = f"{op}_{field}_{window}"
        return FeatureTransform(
            name=name,
            input_field=field,
            operation=op,
            window=window,
            weight=1.0,
        )

    def generate(
        self,
        price: float = 0.0,
        volume: float = 0.0,
        oi: float = 0.0,
        funding_rate: float = 0.0,
    ) -> Dict[str, float]:
        """生成衍生特征

        Args:
            price: 当前价格
            volume: 当前成交量
            oi: 当前持仓量
            funding_rate: 当前资金费率

        Returns:
            Dict[str, float]: 特征名 -> 特征值
        """
        if not self.enabled:
            return {}

        inputs = {
            "price": price,
            "volume": volume,
            "oi": oi,
            "funding_rate": funding_rate,
        }

        # 更新历史
        for field, val in inputs.items():
            if field not in self._history:
                self._history[field] = []
            self._history[field].append(val)
            if len(self._history[field]) > self.WINDOW_MAX * 2:
                self._history[field].pop(0)

        features: Dict[str, float] = {}

        for t in self._transforms:
            try:
                value = self._apply_transform(t, inputs)
                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    features[t.name] = value
                    t.use_count += 1
                    self._total_generated += 1
            except Exception as e:
                logger.debug("Feature transform %s failed: %s", t.name, e)

        return features

    def _apply_transform(
        self,
        t: FeatureTransform,
        inputs: Dict[str, float],
    ) -> Optional[float]:
        """应用单个变换"""
        history = self._history.get(t.input_field, [])
        current = inputs.get(t.input_field, 0.0)

        if current == 0:
            return None

        window = min(t.window, len(history)) if history else 0

        if t.operation == "ma":
            if window < 2:
                return None
            return float(np.mean(history[-window:]))
        elif t.operation == "std":
            if window < 2:
                return None
            return float(np.std(history[-window:]))
        elif t.operation == "roc":
            if window < 1 or len(history) < 2:
                return None
            prev = history[-min(window + 1, len(history))]
            if prev == 0:
                return None
            return float((current - prev) / prev)
        elif t.operation == "rank":
            if window < 2:
                return None
            arr = np.array(history[-window:])
            rank = np.searchsorted(np.sort(arr), current) / len(arr)
            return float(rank)
        elif t.operation == "diff":
            if len(history) < 2:
                return None
            prev = history[-min(t.window, len(history) - 1) - 1]
            return float(current - prev)
        elif t.operation == "zscore":
            if window < 2:
                return None
            arr = np.array(history[-window:])
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                return 0.0
            return float((current - mean) / std)
        else:
            return None

    def feedback(self, score: float):
        """反馈策略评分, 更新特征权重

        Args:
            score: 策略评分 (正=好, 负=差)
        """
        if not self.enabled:
            return

        self._total_feedback += 1

        for t in self._transforms:
            if t.use_count > 0:
                t.score_sum += score
                t.feedback_count += 1
                t.avg_score = t.score_sum / max(1, t.feedback_count)

        # 更新统计
        if score > self._stats["best_score"]:
            self._stats["best_score"] = score
        if score < self._stats["worst_score"]:
            self._stats["worst_score"] = score

    def evolve(self):
        """进化: 淘汰低分特征, 生成新的"""
        if not self.enabled:
            return

        self._generation += 1
        if self._generation % self.EVOLVE_EVERY != 0:
            return

        self._stats["evolutions"] += 1

        # 淘汰低分特征
        before = len(self._transforms)
        survivors = []
        killed = []

        for t in self._transforms:
            if t.feedback_count < 3:
                # 反馈次数太少, 保留观察
                survivors.append(t)
            elif t.avg_score < self.KILL_THRESHOLD:
                killed.append(t)
            else:
                survivors.append(t)

        self._transforms = survivors
        self._stats["killed"] += len(killed)

        # 生成新特征 (保持数量)
        n_new = min(
            self.MAX_FEATURES - len(self._transforms),
            len(killed) + 1,
        )
        for _ in range(max(0, n_new)):
            new_t = self._random_transform()
            # 变异: 偶尔改变窗口
            if self._rng.random() < self.MUTATION_RATE and killed:
                parent = self._rng.choice(killed)
                new_t.input_field = parent.input_field
                new_t.operation = parent.operation
                new_t.window = max(
                    self.WINDOW_MIN,
                    min(self.WINDOW_MAX, parent.window + self._rng.randint(-5, 5)),
                )
                new_t.name = f"{new_t.operation}_{new_t.input_field}_{new_t.window}"
            self._transforms.append(new_t)

        self._stats["created"] += n_new

        logger.info(
            "FeatureAutoGen evolved gen=%d: before=%d killed=%d new=%d after=%d best=%.3f worst=%.3f",
            self._generation, before, len(killed), n_new,
            len(self._transforms),
            self._stats["best_score"], self._stats["worst_score"],
        )

    def get_transforms(self) -> List[Dict[str, Any]]:
        """获取所有变换定义"""
        return [t.to_dict() for t in self._transforms]

    def get_best_transforms(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取评分最高的N个变换"""
        sorted_t = sorted(
            self._transforms,
            key=lambda t: t.avg_score,
            reverse=True,
        )
        return [t.to_dict() for t in sorted_t[:n]]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            **self._stats,
            "enabled": self.enabled,
            "generation": self._generation,
            "n_transforms": len(self._transforms),
            "total_generated": self._total_generated,
            "total_feedback": self._total_feedback,
            "history_fields": list(self._history.keys()),
            "history_lengths": {k: len(v) for k, v in self._history.items()},
        }

    def reset(self):
        """重置 (新进化周期)"""
        self._transforms.clear()
        self._history.clear()
        self._generation = 0
        self._total_generated = 0
        self._total_feedback = 0
        self._stats = {
            "evolutions": 0, "killed": 0, "created": 0,
            "best_score": 0.0, "worst_score": 0.0,
        }
        if self.enabled:
            self._init_transforms(self.INITIAL_FEATURES)
        logger.info("FeatureAutoGenerator reset")


# ============================================================================
# 便捷函数
# ============================================================================

def create_feature_generator(enabled: bool = True) -> FeatureAutoGenerator:
    """创建特征自动生成器"""
    return FeatureAutoGenerator(enabled=enabled)
