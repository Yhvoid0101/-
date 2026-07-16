# -*- coding: utf-8 -*-
# v447_optimize_candidate: 此模块已标记为优化候选
# 原因: 零影响+中等重叠 33.3% > 30% (可合并)
# 建议操作: merge_or_lazy_load
# 标记时间: 2026-06-28T20:18:41
# 详情: _archive_v447/MANIFEST_v447.json

"""
stat_arb_pairs.py — 协整配对交易引擎 v1.0

来源：
  - voiceofchain 2026 (Statistical Arbitrage Crypto Guide)
  - traderabyss 2026 (Crypto Statistical Arbitrage Guide 2026)
  - Pindza & Mba 2026 (Adaptive copula-based pairs trading, QFE期刊)
  - Tsoku & Makatjane 2026 (Deep learning-based pairs trading, Frontiers期刊)

核心算法：
  1. 协整检验（Engle-Granger 两步法）
     - OLS回归: log(A) = α + β × log(B) + ε
     - ADF检验残差: p < 0.05 → 协整
  2. Z-Score 计算
     - spread = log(A) - β × log(B)
     - z = (spread - mean) / std
  3. 半衰期估计
     - Ornstein-Uhlenbeck过程: ds = -κ(s-μ)dt + σdW
     - half_life = ln(2) / κ
  4. 交易信号
     - z < -2.0 → 做多价差（多A空B）
     - z > +2.0 → 做空价差（空A多B）
     - |z| < 0.5 → 平仓
     - |z| > 3.5 → 止损（结构断裂）

实测结果（来源：traderabyss 2026, ETH/BTC对）：
  - 协整p值: 0.003
  - 半衰期: 12天
  - 每笔预期收益: 1.5-3%
  - 持仓周期: 1-3周
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PairSignal(Enum):
    """配对交易信号"""
    LONG_SPREAD = "long_spread"      # 做多价差（多A空B）
    SHORT_SPREAD = "short_spread"    # 做空价差（空A多B）
    CLOSE_LONG = "close_long"        # 平多仓
    CLOSE_SHORT = "close_short"      # 平空仓
    STOP_LOSS = "stop_loss"          # 止损（结构断裂）
    NEUTRAL = "neutral"              # 中性


class PairStatus(Enum):
    """配对状态"""
    NOT_TESTED = "not_tested"        # 未检验
    NOT_COINTEGRATED = "not_coint"   # 不协整
    COINTEGRATED = "cointegrated"    # 协整
    BROKEN = "broken"                # 结构断裂


@dataclass
class PairAnalysis:
    """配对分析结果"""
    asset_a: str
    asset_b: str
    status: PairStatus = PairStatus.NOT_TESTED
    hedge_ratio: float = 1.0          # β 对冲比率
    intercept: float = 0.0            # α 截距
    z_score: float = 0.0               # 当前Z-Score
    spread: float = 0.0                # 当前价差
    spread_mean: float = 0.0           # 价差均值
    spread_std: float = 0.0            # 价差标准差
    half_life: float = 0.0             # 半衰期（天）
    coint_p_value: float = 1.0         # 协整p值
    correlation: float = 0.0            # 相关系数
    signal: PairSignal = PairSignal.NEUTRAL
    confidence: float = 0.0
    entry_z: float = 0.0               # 入场Z
    exit_z: float = 0.0                # 出场Z
    stop_z: float = 0.0                # 止损Z
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


class StatArbPairs:
    """
    统计套利配对交易引擎

    基于协整性的市场中性策略，做多低估资产、做空高估资产，
    从价差均值回归中获利。

    核心流程：
      1. 资产筛选（相关性 > 0.75）
      2. 协整检验（Engle-Granger ADF）
      3. 半衰期估计（OU过程）
      4. Z-Score 信号生成
      5. 动态对冲比率（可选卡尔曼滤波）
      6. 风险管理（止损/时间限制/仓位调整）
    """

    # 协整检验参数
    MIN_CORRELATION = 0.75           # 最小相关系数
    MAX_COINT_P_VALUE = 0.05         # 协整p值阈值
    MIN_HALF_LIFE = 1                 # 最小半衰期（tick数，原值3天针对日线，tick数据降至1）
    MAX_HALF_LIFE = 30                # 最大半衰期（天）

    # 交易信号阈值
    ENTRY_Z = 2.0                    # 入场Z-Score
    EXIT_Z = 0.5                     # 出场Z-Score
    STOP_Z = 3.5                     # 止损Z-Score

    # 回看窗口
    LOOKBACK_WINDOW = 90             # 回看窗口（天）
    Z_SCORE_WINDOW = 20              # Z-Score计算窗口

    def __init__(self, asset_a: str = "ETH", asset_b: str = "BTC"):
        self.asset_a = asset_a
        self.asset_b = asset_b

        # 价格历史
        self._prices_a: deque = deque(maxlen=200)
        self._prices_b: deque = deque(maxlen=200)
        self._timestamps: deque = deque(maxlen=200)

        # 分析结果缓存
        self._last_analysis: Optional[PairAnalysis] = None
        # 当前持仓
        self._current_position: Optional[PairSignal] = None
        # 入场时的Z-Score
        self._entry_z_score: float = 0.0

    def update_prices(self, price_a: float, price_b: float,
                      timestamp: Optional[float] = None):
        """更新价格数据

        Args:
            price_a: 资产A价格
            price_b: 资产B价格
            timestamp: 时间戳
        """
        ts = timestamp if timestamp is not None else time.time()
        self._prices_a.append(price_a)
        self._prices_b.append(price_b)
        self._timestamps.append(ts)

    def analyze(self) -> PairAnalysis:
        """分析配对，生成交易信号

        Returns:
            PairAnalysis: 分析结果
        """
        if len(self._prices_a) < 30:
            return PairAnalysis(asset_a=self.asset_a, asset_b=self.asset_b)

        prices_a = np.array(list(self._prices_a))
        prices_b = np.array(list(self._prices_b))

        # 1. 相关性检查
        correlation = self._calculate_correlation(prices_a, prices_b)
        if abs(correlation) < self.MIN_CORRELATION:
            result = PairAnalysis(
                asset_a=self.asset_a,
                asset_b=self.asset_b,
                status=PairStatus.NOT_COINTEGRATED,
                correlation=correlation,
                detail=f"相关性{correlation:.3f} < {self.MIN_CORRELATION}，不适合配对",
            )
            self._last_analysis = result
            return result

        # 2. 协整检验（Engle-Granger 两步法）
        hedge_ratio, intercept, residuals = self._engle_granger_test(
            prices_a, prices_b
        )

        # ADF检验（简化版：使用残差的自相关系数）
        coint_p_value = self._adf_test_simplified(residuals)

        if coint_p_value > self.MAX_COINT_P_VALUE:
            result = PairAnalysis(
                asset_a=self.asset_a,
                asset_b=self.asset_b,
                status=PairStatus.NOT_COINTEGRATED,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                intercept=intercept,
                coint_p_value=coint_p_value,
                detail=f"协整p值{coint_p_value:.4f} > 0.05，不协整",
            )
            self._last_analysis = result
            return result

        # 3. 半衰期估计
        half_life = self._estimate_half_life(residuals)

        if half_life < self.MIN_HALF_LIFE or half_life > self.MAX_HALF_LIFE:
            result = PairAnalysis(
                asset_a=self.asset_a,
                asset_b=self.asset_b,
                status=PairStatus.NOT_COINTEGRATED,
                correlation=correlation,
                hedge_ratio=hedge_ratio,
                intercept=intercept,
                coint_p_value=coint_p_value,
                half_life=half_life,
                detail=f"半衰期{half_life:.1f}天不在[{self.MIN_HALF_LIFE},{self.MAX_HALF_LIFE}]范围",
            )
            self._last_analysis = result
            return result

        # 4. Z-Score 计算
        spread_mean = float(np.mean(residuals[-self.Z_SCORE_WINDOW:]))
        spread_std = float(np.std(residuals[-self.Z_SCORE_WINDOW:]))
        current_spread = float(residuals[-1])

        if spread_std > 0:
            z_score = (current_spread - spread_mean) / spread_std
        else:
            z_score = 0.0

        # 5. 信号生成
        signal, confidence, detail = self._generate_signal(z_score)

        result = PairAnalysis(
            asset_a=self.asset_a,
            asset_b=self.asset_b,
            status=PairStatus.COINTEGRATED,
            hedge_ratio=hedge_ratio,
            intercept=intercept,
            z_score=z_score,
            spread=current_spread,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            coint_p_value=coint_p_value,
            correlation=correlation,
            signal=signal,
            confidence=confidence,
            entry_z=self.ENTRY_Z,
            exit_z=self.EXIT_Z,
            stop_z=self.STOP_Z,
            detail=detail,
        )
        self._last_analysis = result
        return result

    def _calculate_correlation(self, prices_a: np.ndarray,
                               prices_b: np.ndarray) -> float:
        """计算对数收益率相关系数"""
        if len(prices_a) < 2:
            return 0.0
        log_ret_a = np.diff(np.log(prices_a))
        log_ret_b = np.diff(np.log(prices_b))
        if len(log_ret_a) == 0:
            return 0.0
        corr_matrix = np.corrcoef(log_ret_a, log_ret_b)
        return float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    def _engle_granger_test(self, prices_a: np.ndarray,
                            prices_b: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Engle-Granger 两步法协整检验

        步骤1: OLS回归 log(A) = α + β × log(B) + ε
        步骤2: 对残差 ε 进行 ADF 检验

        Returns:
            (hedge_ratio β, intercept α, residuals)
        """
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)

        # OLS回归
        n = len(log_a)
        x = np.column_stack([np.ones(n), log_b])
        try:
            beta, _, _, _ = np.linalg.lstsq(x, log_a, rcond=None)
            intercept = float(beta[0])
            hedge_ratio = float(beta[1])
        except np.linalg.LinAlgError:
            return 1.0, 0.0, log_a - log_b

        residuals = log_a - (intercept + hedge_ratio * log_b)
        return hedge_ratio, intercept, residuals

    def _adf_test_simplified(self, residuals: np.ndarray) -> float:
        """简化版 ADF 检验

        完整ADF检验需要回归: Δε_t = α + ρ × ε_{t-1} + Σ γ_i × Δε_{t-i} + u_t
        检验 H0: ρ = 0（非平稳）vs H1: ρ < 0（平稳）

        简化版：使用残差的一阶自相关系数 ρ 来近似
        如果 ρ 接近 0 → 平稳（协整）
        如果 ρ 接近 1 → 非平稳

        Returns:
            p_value (0-1)，越小越显著
        """
        if len(residuals) < 10:
            return 1.0

        # 一阶差分
        delta_res = np.diff(residuals)
        lag_res = residuals[:-1]

        # OLS: Δε_t = ρ × ε_{t-1}
        n = len(delta_res)
        x = lag_res.reshape(-1, 1)
        x_with_const = np.column_stack([np.ones(n), x])

        try:
            beta, _, _, _ = np.linalg.lstsq(x_with_const, delta_res, rcond=None)
            rho = beta[1]  # 自回归系数
        except np.linalg.LinAlgError:
            return 1.0

        # ρ 越负 → 越平稳 → p值越小
        # 近似映射：ρ = -0.5 → p ≈ 0.01
        #          ρ = -0.3 → p ≈ 0.05
        #          ρ = -0.1 → p ≈ 0.3
        #          ρ = 0    → p = 1.0
        if rho >= 0:
            return 1.0

        # Phase 7B 修复: 公式应为 exp(rho*10) 以匹配上述映射（原代码 *5 是笔误）
        # 验证: exp(-0.3*10)=0.050 ✓, exp(-0.5*10)=0.0067 ✓, exp(-0.1*10)=0.368 ✓
        p_value = math.exp(rho * 10)  # 简化的指数映射
        return min(max(p_value, 0.0), 1.0)

    def _estimate_half_life(self, residuals: np.ndarray) -> float:
        """估计半衰期

        基于 Ornstein-Uhlenbeck 过程:
          ds = -κ(s-μ)dt + σdW
          half_life = ln(2) / κ

        通过 OLS 回归 Δs_t = -κ × s_{t-1} + const 估计 κ
        """
        if len(residuals) < 10:
            return 0.0

        delta_s = np.diff(residuals)
        lag_s = residuals[:-1]

        # OLS: Δs = -κ × s_{t-1}
        n = len(delta_s)
        x = lag_s.reshape(-1, 1)
        x_with_const = np.column_stack([np.ones(n), x])

        try:
            beta, _, _, _ = np.linalg.lstsq(x_with_const, delta_s, rcond=None)
            kappa = -beta[1]  # κ = -ρ
        except np.linalg.LinAlgError:
            return 0.0

        if kappa <= 0:
            return 999.0  # 不收敛

        half_life = math.log(2) / kappa
        return half_life

    def _generate_signal(self, z_score: float) -> Tuple[PairSignal, float, str]:
        """生成交易信号

        信号逻辑：
          z < -2.0 → LONG_SPREAD（多A空B，价差会扩大回归）
          z > +2.0 → SHORT_SPREAD（空A多B，价差会缩小回归）
          |z| < 0.5 → 平仓
          |z| > 3.5 → 止损（结构断裂）
        """
        # 止损检查
        if abs(z_score) > self.STOP_Z:
            return (
                PairSignal.STOP_LOSS,
                0.80,
                f"Z={z_score:.2f}超过止损线±{self.STOP_Z}，结构断裂，止损"
            )

        # 平仓检查
        if self._current_position is not None:
            if abs(z_score) < self.EXIT_Z:
                if self._current_position == PairSignal.LONG_SPREAD:
                    return (PairSignal.CLOSE_LONG, 0.70,
                            f"Z={z_score:.2f}回归到±{self.EXIT_Z}内，平多仓")
                elif self._current_position == PairSignal.SHORT_SPREAD:
                    return (PairSignal.CLOSE_SHORT, 0.70,
                            f"Z={z_score:.2f}回归到±{self.EXIT_Z}内，平空仓")

        # 入场信号
        if z_score < -self.ENTRY_Z:
            confidence = min(abs(z_score) / 4.0, 0.95)
            return (
                PairSignal.LONG_SPREAD,
                confidence,
                f"Z={z_score:.2f}<-{self.ENTRY_Z}，做多价差（多{self.asset_a}空{self.asset_b}）"
            )

        if z_score > self.ENTRY_Z:
            confidence = min(abs(z_score) / 4.0, 0.95)
            return (
                PairSignal.SHORT_SPREAD,
                confidence,
                f"Z={z_score:.2f}>+{self.ENTRY_Z}，做空价差（空{self.asset_a}多{self.asset_b}）"
            )

        return (PairSignal.NEUTRAL, 0.0, f"Z={z_score:.2f}在中性区间")

    def set_position(self, signal: PairSignal):
        """设置当前持仓状态"""
        if signal in (PairSignal.LONG_SPREAD, PairSignal.SHORT_SPREAD):
            self._current_position = signal
            self._entry_z_score = self._last_analysis.z_score if self._last_analysis else 0.0
        elif signal in (PairSignal.CLOSE_LONG, PairSignal.CLOSE_SHORT,
                        PairSignal.STOP_LOSS):
            self._current_position = None
            self._entry_z_score = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "data_points": len(self._prices_a),
            "has_position": self._current_position is not None,
            "current_position": self._current_position.value if self._current_position else None,
            "entry_z_score": self._entry_z_score,
            "last_analysis": self._last_analysis is not None,
        }


def scan_cointegrated_pairs(price_data: Dict[str, List[float]],
                            min_correlation: float = 0.75,
                            max_p_value: float = 0.05) -> List[PairAnalysis]:
    """扫描协整配对

    Args:
        price_data: {asset: [prices]} 字典
        min_correlation: 最小相关系数
        max_p_value: 最大协整p值

    Returns:
        List[PairAnalysis]: 通过筛选的协整配对列表
    """
    assets = list(price_data.keys())
    results = []

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            prices_a = np.array(price_data[a])
            prices_b = np.array(price_data[b])

            if len(prices_a) != len(prices_b) or len(prices_a) < 30:
                continue

            engine = StatArbPairs(asset_a=a, asset_b=b)
            for pa, pb in zip(prices_a, prices_b):
                engine.update_prices(pa, pb)

            analysis = engine.analyze()
            if (analysis.status == PairStatus.COINTEGRATED and
                analysis.coint_p_value <= max_p_value and
                analysis.correlation >= min_correlation):
                results.append(analysis)

    return results
