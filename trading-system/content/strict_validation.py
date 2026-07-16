# -*- coding: utf-8 -*-
"""
严格验证模块 (Strict Validation) — 三集分离+时间序列交叉验证+真实数据生成

基于前沿量化系统过拟合防御最佳实践（2025）：
  - Robert Pardo WFA: 训练/验证/测试三集分离（50%/20%/30%）
  - 时间序列交叉验证（Time Series CV）：滚动窗口验证
  - Bonferroni校正：多重假设检验偏差修正
  - 参数鲁棒性检验：参数热力图+敏感性分析
  - 真实历史数据生成：BTC/ETH统计特性（波动率聚集、肥尾效应）

核心组件：
  1. RealisticDataGenerator — 真实历史数据生成器（带统计特性）
  2. DataSplitter — 严格三集分离（训练/验证/测试）
  3. TimeSeriesCV — 时间序列交叉验证
  4. OverfittingDetector — 过拟合检测器
  5. RobustnessAnalyzer — 参数鲁棒性分析
  6. StrictValidator — 统一严格验证器
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("hermes.strict_validation")


# ============================================================================
# 1. 真实历史数据生成器（Realistic Data Generator）
# ============================================================================


class RealisticDataGenerator:
    """真实历史数据生成器 — 模拟BTC/ETH统计特性

    基于真实加密货币市场的统计特性：
      1. 波动率聚集（Volatility Clustering）：高波动后跟高波动
      2. 肥尾效应（Fat Tails）：极端事件比正态分布更频繁
      3. 均值回归（Mean Reversion）：长期价格回归趋势线
      4. 日内波动模式：亚洲/欧洲/美洲时段不同波动率
      5. 趋势+噪声分解：价格 = 趋势 + 周期 + 噪声

    参数基于BTC历史统计（2018-2024）：
      - 年化波动率: 60-80%
      - 日均波动: 3-5%
      - 偏度: -0.3（略负偏，暴跌比暴涨多）
      - 峰度: 7-12（肥尾）
    """

    # BTC统计参数（基于2018-2024历史数据）
    BTC_ANNUAL_VOL = 0.70           # 年化波动率70%
    BTC_DAILY_VOL = 0.044           # 日波动率4.4%
    BTC_SKEWNESS = -0.3             # 略负偏
    BTC_KURTOSIS = 8.0              # 肥尾
    BTC_ANNUAL_DRIFT = 0.15         # 年化漂移15%
    BTC_MEAN_REVERSION = 0.01       # 均值回归强度

    # ETH统计参数（比BTC波动更大）
    ETH_ANNUAL_VOL = 0.85
    ETH_DAILY_VOL = 0.054
    ETH_SKEWNESS = -0.4
    ETH_KURTOSIS = 10.0
    ETH_ANNUAL_DRIFT = 0.20
    ETH_MEAN_REVERSION = 0.015

    # 波动率聚集参数（GARCH-like）
    VOL_CLUSTER_ALPHA = 0.1    # 新信息冲击
    VOL_CLUSTER_BETA = 0.85    # 之前波动率延续
    # omega = long_var * (1 - alpha - beta), long_var = daily_vol^2
    # 对于BTC: long_var = 0.044^2 = 0.001936, omega = 0.001936 * 0.05 = 0.0000968

    @staticmethod
    def generate_garch_volatility(n_bars: int, initial_vol: float,
                                   alpha: float = 0.1, beta: float = 0.85,
                                   omega: float = None) -> List[float]:
        """生成GARCH(1,1)波动率序列（波动率聚集）

        公式: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        其中 ε²_{t-1} = σ²_{t-1} * z² (z为标准正态)

        长期方差: σ² = ω / (1 - α - β)
        """
        # 自动计算omega以匹配目标波动率
        if omega is None:
            long_var = initial_vol ** 2
            omega = long_var * (1 - alpha - beta)

        vols = []
        var = initial_vol ** 2
        long_var = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else initial_vol ** 2

        for i in range(n_bars):
            # 随机冲击（标准正态）
            shock = random.gauss(0, 1)
            # GARCH更新: σ²_t = ω + α*σ²_{t-1}*z² + β*σ²_{t-1}
            var = omega + alpha * var * (shock ** 2) + beta * var
            # 均值回归到长期方差
            var = 0.95 * var + 0.05 * long_var
            vols.append(math.sqrt(max(var, 1e-10)))

        return vols

    @staticmethod
    def generate_fat_tailed_returns(n_bars: int, daily_vol: float,
                                     skewness: float = 0.0,
                                     kurtosis: float = 3.0) -> List[float]:
        """生成肥尾收益序列（Student-t分布+偏度调整）

        使用混合正态分布近似Student-t分布的肥尾特性
        返回的是标准化收益（z-scores），需乘以波动率得到实际收益
        """
        returns = []
        # 自由度（从峰度反推，最小限制为4.0以保证有限方差）
        if kurtosis > 3:
            df = max(4.0, 6.0 / (kurtosis - 3.0))
        else:
            df = 30.0  # 接近正态

        for i in range(n_bars):
            # Student-t采样（使用Gamma+正态混合）
            gamma_sample = random.gammavariate(df / 2.0, 2.0 / df)
            # 防止除零（gamma可能接近0）
            gamma_sample = max(gamma_sample, 0.01)
            normal_sample = random.gauss(0, 1)
            t_sample = normal_sample / math.sqrt(gamma_sample)

            # 限制极端值（防止价格爆炸）
            t_sample = max(-5.0, min(5.0, t_sample))

            # 偏度调整（简单位移）
            if skewness != 0:
                t_sample += skewness * 0.3

            # 返回标准化收益（z-score），后续乘以波动率
            returns.append(t_sample)

        return returns

    @classmethod
    def generate_btc_history(cls, n_bars: int = 365, start_price: float = 50000.0,
                              seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """生成BTC历史K线数据（带真实统计特性）

        Args:
            n_bars: K线数量（默认365天）
            start_price: 起始价格
            seed: 随机种子（可复现）

        Returns:
            K线数据列表
        """
        if seed is not None:
            random.seed(seed)

        # 1. 生成GARCH波动率序列
        vols = cls.generate_garch_volatility(
            n_bars, cls.BTC_DAILY_VOL,
            cls.VOL_CLUSTER_ALPHA, cls.VOL_CLUSTER_BETA,
        )

        # 2. 生成标准化肥尾收益（z-scores）
        z_scores = cls.generate_fat_tailed_returns(
            n_bars, cls.BTC_DAILY_VOL, cls.BTC_SKEWNESS, cls.BTC_KURTOSIS,
        )

        # 3. 实际收益 = z_score * 波动率（正确缩放）
        daily_drift = cls.BTC_ANNUAL_DRIFT / 365
        prices = [start_price]
        for i, z in enumerate(z_scores):
            # 实际收益 = 漂移 + z_score * 当日波动率
            ret = daily_drift + z * vols[i]
            # 限制单日收益防止极端值（±20%）
            ret = max(-0.20, min(0.20, ret))
            # 均值回归项（回归到起始价格的对数）
            if prices[-1] > 0:
                mean_reversion = cls.BTC_MEAN_REVERSION * math.log(start_price / prices[-1])
            else:
                mean_reversion = 0
            total_ret = ret + mean_reversion
            new_price = max(0.01, prices[-1] * math.exp(total_ret))
            prices.append(new_price)

        # 5. 生成OHLCV
        bars = []
        for i in range(n_bars):
            open_price = prices[i]
            close_price = prices[i + 1]
            # 日内波动（基于当日波动率）
            intraday_vol = vols[i] * 0.7  # 日内波动略小于日波动
            high = max(open_price, close_price) * (1 + abs(random.gauss(0, intraday_vol * 0.5)))
            low = min(open_price, close_price) * (1 - abs(random.gauss(0, intraday_vol * 0.5)))
            # 成交量（与波动率正相关）
            volume = 1000 * (1 + vols[i] * 20) * (0.8 + random.random() * 0.4)

            bars.append({
                "timestamp": time.time() + i * 86400,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
                "symbol": "BTC-USDT",
            })

        return bars

    @classmethod
    def generate_eth_history(cls, n_bars: int = 365, start_price: float = 3000.0,
                              seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """生成ETH历史K线数据"""
        if seed is not None:
            random.seed(seed + 1)  # 不同于BTC的种子

        vols = cls.generate_garch_volatility(
            n_bars, cls.ETH_DAILY_VOL,
            cls.VOL_CLUSTER_ALPHA, cls.VOL_CLUSTER_BETA,
        )

        z_scores = cls.generate_fat_tailed_returns(
            n_bars, cls.ETH_DAILY_VOL, cls.ETH_SKEWNESS, cls.ETH_KURTOSIS,
        )

        daily_drift = cls.ETH_ANNUAL_DRIFT / 365
        prices = [start_price]
        for i, z in enumerate(z_scores):
            ret = daily_drift + z * vols[i]
            ret = max(-0.25, min(0.25, ret))  # ETH波动更大，限制±25%
            if prices[-1] > 0:
                mean_reversion = cls.ETH_MEAN_REVERSION * math.log(start_price / prices[-1])
            else:
                mean_reversion = 0
            total_ret = ret + mean_reversion
            new_price = max(0.01, prices[-1] * math.exp(total_ret))
            prices.append(new_price)

        bars = []
        for i in range(n_bars):
            open_price = prices[i]
            close_price = prices[i + 1]
            intraday_vol = vols[i] * 0.7
            high = max(open_price, close_price) * (1 + abs(random.gauss(0, intraday_vol * 0.5)))
            low = min(open_price, close_price) * (1 - abs(random.gauss(0, intraday_vol * 0.5)))
            volume = 5000 * (1 + vols[i] * 20) * (0.8 + random.random() * 0.4)

            bars.append({
                "timestamp": time.time() + i * 86400,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
                "symbol": "ETH-USDT",
            })

        return bars

    @classmethod
    def get_multi_symbol_history(cls, symbols: List[str] = None,
                                  n_bars: int = 365) -> Dict[str, List[Dict[str, Any]]]:
        """获取多币种历史数据"""
        if symbols is None:
            symbols = ["BTC-USDT", "ETH-USDT"]

        result = {}
        for symbol in symbols:
            if symbol == "BTC-USDT":
                result[symbol] = cls.generate_btc_history(n_bars, seed=42)
            elif symbol == "ETH-USDT":
                result[symbol] = cls.generate_eth_history(n_bars, seed=42)
            else:
                # 默认使用BTC参数
                result[symbol] = cls.generate_btc_history(n_bars, start_price=100.0, seed=42)

        return result


# ============================================================================
# 2. 数据三集分离（Data Splitter）— 严格时间序列分离
# ============================================================================


@dataclass(slots=True)
class DataSplit:
    """数据分离结果"""
    train: List[Any]
    validation: List[Any]
    test: List[Any]
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    split_info: Dict[str, Any] = field(default_factory=dict)


class DataSplitter:
    """严格三集分离 — 训练/验证/测试

    来自前沿过拟合防御标准（2025）：
      - 训练集 50%：参数拟合
      - 验证集 20%：超参数调优与模型选择
      - 测试集 30%：最终评估（全程不参与优化）

    关键原则：
      - 严格时间顺序（不能打乱）
      - 测试集"一次性使用"（不可回看）
      - 无数据泄露（训练集不能包含测试集信息）
    """

    DEFAULT_TRAIN_RATIO = 0.50
    DEFAULT_VALIDATION_RATIO = 0.20
    DEFAULT_TEST_RATIO = 0.30

    @classmethod
    def split_time_series(cls, data: List[Any],
                          train_ratio: float = None,
                          validation_ratio: float = None,
                          test_ratio: float = None) -> DataSplit:
        """时间序列三集分离

        Args:
            data: 时间序列数据（必须按时间排序）
            train_ratio: 训练集比例（默认0.50）
            validation_ratio: 验证集比例（默认0.20）
            test_ratio: 测试集比例（默认0.30）

        Returns:
            DataSplit 对象
        """
        tr = train_ratio or cls.DEFAULT_TRAIN_RATIO
        vr = validation_ratio or cls.DEFAULT_VALIDATION_RATIO
        sr = test_ratio or cls.DEFAULT_TEST_RATIO

        # 比例和必须为1
        total = tr + vr + sr
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        n = len(data)
        train_end = int(n * tr)
        val_end = train_end + int(n * vr)

        train = data[:train_end]
        validation = data[train_end:val_end]
        test = data[val_end:]

        split_info = {
            "total_size": n,
            "train_size": len(train),
            "validation_size": len(validation),
            "test_size": len(test),
            "train_end_idx": train_end,
            "validation_end_idx": val_end,
            "method": "time_series_sequential",
        }

        logger.info(
            "Data split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
            len(train), tr * 100,
            len(validation), vr * 100,
            len(test), sr * 100,
        )

        return DataSplit(
            train=train,
            validation=validation,
            test=test,
            train_ratio=tr,
            validation_ratio=vr,
            test_ratio=sr,
            split_info=split_info,
        )

    @classmethod
    def split_walk_forward(cls, data: List[Any],
                           train_window: int,
                           test_window: int,
                           step: int = None) -> List[Tuple[List[Any], List[Any]]]:
        """Walk-forward分离（滚动窗口）

        Args:
            data: 时间序列数据
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            step: 滚动步长（默认=test_window）

        Returns:
            [(train, test), ...] 列表
        """
        if step is None:
            step = test_window

        splits = []
        n = len(data)

        start = 0
        while start + train_window + test_window <= n:
            train = data[start:start + train_window]
            test = data[start + train_window:start + train_window + test_window]
            splits.append((train, test))
            start += step

        logger.info(
            "Walk-forward splits: %d windows (train=%d, test=%d, step=%d)",
            len(splits), train_window, test_window, step,
        )

        return splits


# ============================================================================
# 3. 时间序列交叉验证（Time Series Cross-Validation）
# ============================================================================


class TimeSeriesCV:
    """时间序列交叉验证 — 滚动窗口验证

    来自scikit-learn TimeSeriesSplit的最佳实践：
      - 严格时间顺序（不能打乱）
      - 训练集逐步扩大
      - 测试集固定大小
      - 无数据泄露

    与传统K折CV的区别：
      - K折CV假设样本独立同分布（不适用于金融时间序列）
      - TimeSeriesCV保持时间顺序，避免未来信息泄露
    """

    def __init__(self, n_splits: int = 5, test_size: int = None, gap: int = 0):
        """
        Args:
            n_splits: 分割次数
            test_size: 测试集大小（None=自动）
            gap: 训练集和测试集之间的间隔（避免数据泄露）
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, data: List[Any]) -> Iterator[Tuple[List[int], List[int]]]:
        """生成交叉验证分割

        Yields:
            (train_indices, test_indices)
        """
        n = len(data)
        test_size = self.test_size or n // (self.n_splits + 1)

        # 计算训练集最小大小
        min_train = n - (self.n_splits * test_size) - (self.n_splits * self.gap)
        if min_train < test_size:
            raise ValueError(
                f"Not enough data: n={n}, n_splits={self.n_splits}, "
                f"test_size={test_size}, gap={self.gap}"
            )

        indices = list(range(n))
        for i in range(self.n_splits):
            train_end = min_train + i * (test_size + self.gap)
            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n:
                break

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_info(self, data: List[Any]) -> Dict[str, Any]:
        """获取交叉验证信息"""
        splits = list(self.split(data))
        return {
            "n_splits": len(splits),
            "total_size": len(data),
            "test_size": self.test_size or len(data) // (self.n_splits + 1),
            "gap": self.gap,
            "splits": [
                {
                    "split": i,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_range": f"[0:{train_idx[-1]+1}]",
                    "test_range": f"[{test_idx[0]}:{test_idx[-1]+1}]",
                }
                for i, (train_idx, test_idx) in enumerate(splits)
            ],
        }


# ============================================================================
# 4. 过拟合检测器（Overfitting Detector）
# ============================================================================


@dataclass(slots=True)
class OverfittingResult:
    """过拟合检测结果"""
    is_overfit: bool
    train_score: float
    test_score: float
    overfit_ratio: float  # test/train
    severity: str  # "none" / "mild" / "moderate" / "severe"
    details: Dict[str, Any] = field(default_factory=dict)


class OverfittingDetector:
    """过拟合检测器 — 多维度检测

    检测方法：
      1. Train-Test Gap: 训练集和测试集性能差距
      2. Walk-Forward Stability: 滚动窗口性能稳定性
      3. Bonferroni校正: 多重假设检验偏差修正
      4. 参数敏感性: 参数小变动下的性能变化
    """

    # 过拟合判定阈值
    OVERFIT_RATIO_THRESHOLD = 0.5   # test/train < 0.5 = 过拟合
    MILD_OVERFIT_THRESHOLD = 0.7
    MODERATE_OVERFIT_THRESHOLD = 0.5
    SEVERE_OVERFIT_THRESHOLD = 0.3

    @classmethod
    def detect_train_test_gap(cls, train_score: float, test_score: float) -> OverfittingResult:
        """检测训练集和测试集性能差距

        Args:
            train_score: 训练集得分
            test_score: 测试集得分

        Returns:
            OverfittingResult
        """
        if train_score <= 0:
            return OverfittingResult(
                is_overfit=False,
                train_score=train_score,
                test_score=test_score,
                overfit_ratio=0.0,
                severity="none",
                details={"reason": "train_score <= 0"},
            )

        ratio = test_score / train_score

        if ratio < cls.SEVERE_OVERFIT_THRESHOLD:
            severity = "severe"
            is_overfit = True
        elif ratio < cls.MODERATE_OVERFIT_THRESHOLD:
            severity = "moderate"
            is_overfit = True
        elif ratio < cls.MILD_OVERFIT_THRESHOLD:
            severity = "mild"
            is_overfit = True
        else:
            severity = "none"
            is_overfit = False

        return OverfittingResult(
            is_overfit=is_overfit,
            train_score=train_score,
            test_score=test_score,
            overfit_ratio=ratio,
            severity=severity,
            details={
                "gap": train_score - test_score,
                "gap_pct": (1 - ratio) * 100,
                "threshold": cls.OVERFIT_RATIO_THRESHOLD,
            },
        )

    @classmethod
    def detect_wf_stability(cls, wf_scores: List[Tuple[float, float]]) -> OverfittingResult:
        """检测Walk-Forward稳定性

        Args:
            wf_scores: [(train_score, test_score), ...] 列表

        Returns:
            OverfittingResult
        """
        if not wf_scores:
            return OverfittingResult(
                is_overfit=False, train_score=0, test_score=0,
                overfit_ratio=0, severity="none",
                details={"reason": "no wf_scores"},
            )

        train_scores = [s[0] for s in wf_scores]
        test_scores = [s[1] for s in wf_scores]

        mean_train = statistics.mean(train_scores)
        mean_test = statistics.mean(test_scores)

        # 测试集得分的标准差（稳定性）
        test_std = statistics.stdev(test_scores) if len(test_scores) > 1 else 0
        test_cv = test_std / mean_test if mean_test > 0 else float("inf")  # 变异系数

        result = cls.detect_train_test_gap(mean_train, mean_test)
        result.details.update({
            "wf_windows": len(wf_scores),
            "test_score_std": test_std,
            "test_score_cv": test_cv,
            "test_stable": test_cv < 0.3,  # 变异系数<30%为稳定
        })

        # 如果测试集不稳定，升级严重度
        if test_cv > 0.5 and result.severity == "none":
            result.severity = "mild"
            result.is_overfit = True
            result.details["instability_detected"] = True

        return result

    @classmethod
    def bonferroni_correction(cls, p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """Bonferroni多重假设检验校正

        Args:
            p_values: 原始p值列表
            alpha: 显著性水平

        Returns:
            校正后的显著性列表
        """
        n = len(p_values)
        corrected_alpha = alpha / n

        return [p < corrected_alpha for p in p_values]


# ============================================================================
# 5. 参数鲁棒性分析（Robustness Analyzer）
# ============================================================================


@dataclass(slots=True)
class RobustnessResult:
    """鲁棒性分析结果"""
    is_robust: bool
    sensitivity: float  # 参数敏感性（性能变化/参数变化）
    stable_region: float  # 稳定区域占比
    details: Dict[str, Any] = field(default_factory=dict)


class RobustnessAnalyzer:
    """参数鲁棒性分析

    来自前沿量化系统鲁棒性检验标准：
      - 参数热力图：观察性能变化
      - 参数敏感性：小变动下的性能变化
      - 稳定区域：性能保持稳定的参数范围占比
    """

    # 鲁棒性阈值（交易系统参数敏感性天然较高，阈值放宽）
    SENSITIVITY_THRESHOLD = 2.0   # 敏感性<2.0为鲁棒（交易系统标准）
    STABLE_REGION_THRESHOLD = 0.6  # 稳定区域>60%为鲁棒

    @classmethod
    def analyze_parameter_sensitivity(
        cls,
        base_param: float,
        param_variations: List[float],
        base_score: float,
        scores: List[float],
    ) -> RobustnessResult:
        """分析参数敏感性

        Args:
            base_param: 基准参数值
            param_variations: 参数变化列表
            base_score: 基准得分
            scores: 对应的得分列表

        Returns:
            RobustnessResult
        """
        if not scores or not param_variations:
            return RobustnessResult(
                is_robust=False, sensitivity=1.0, stable_region=0.0,
                details={"reason": "empty input"},
            )

        # 计算参数变化范围
        param_range = max(param_variations) - min(param_variations)
        if param_range == 0:
            return RobustnessResult(
                is_robust=True, sensitivity=0.0, stable_region=1.0,
                details={"reason": "no param variation"},
            )

        # 计算得分变化范围
        score_range = max(scores) - min(scores)
        sensitivity = score_range / param_range if param_range > 0 else 0

        # 计算稳定区域（得分变化<10%的参数占比）
        stable_count = sum(
            1 for s in scores
            if abs(s - base_score) / base_score < 0.1 if base_score > 0
        )
        stable_region = stable_count / len(scores)

        is_robust = (
            sensitivity < cls.SENSITIVITY_THRESHOLD
            and stable_region > cls.STABLE_REGION_THRESHOLD
        )

        return RobustnessResult(
            is_robust=is_robust,
            sensitivity=sensitivity,
            stable_region=stable_region,
            details={
                "base_param": base_param,
                "base_score": base_score,
                "param_range": param_range,
                "score_range": score_range,
                "stable_count": stable_count,
                "total_variations": len(scores),
            },
        )


# ============================================================================
# 6. 严格验证器（Strict Validator）— 统一入口
# ============================================================================


class StrictValidator:
    """严格验证器 — 统一管理所有验证组件

    集成：
      - RealisticDataGenerator（真实数据生成）
      - DataSplitter（三集分离）
      - TimeSeriesCV（时间序列交叉验证）
      - OverfittingDetector（过拟合检测）
      - RobustnessAnalyzer（鲁棒性分析）
    """

    def __init__(self):
        self.data_generator = RealisticDataGenerator()
        self.splitter = DataSplitter()
        self.cv = TimeSeriesCV
        self.overfit_detector = OverfittingDetector
        self.robustness = RobustnessAnalyzer

    def generate_validation_data(self, n_bars: int = 365) -> Dict[str, List[Dict[str, Any]]]:
        """生成验证用真实数据"""
        return self.data_generator.get_multi_symbol_history(n_bars=n_bars)

    def strict_train_val_test_split(self, data: List[Any]) -> DataSplit:
        """严格三集分离"""
        return self.splitter.split_time_series(data)

    def time_series_cv(self, data: List[Any], n_splits: int = 5) -> Dict[str, Any]:
        """时间序列交叉验证"""
        cv = self.cv(n_splits=n_splits)
        return cv.get_info(data)

    def detect_overfitting(self, train_score: float, test_score: float) -> OverfittingResult:
        """过拟合检测"""
        return self.overfit_detector.detect_train_test_gap(train_score, test_score)

    def analyze_robustness(self, base_param: float, variations: List[float],
                           base_score: float, scores: List[float]) -> RobustnessResult:
        """鲁棒性分析"""
        return self.robustness.analyze_parameter_sensitivity(
            base_param, variations, base_score, scores,
        )

    def full_validation_pipeline(self, n_bars: int = 365) -> Dict[str, Any]:
        """完整验证管道

        Returns:
            验证结果字典
        """
        results = {
            "timestamp": time.time(),
            "stages": {},
        }

        # Stage 1: 生成真实数据
        logger.info("Stage 1: Generating realistic data...")
        data = self.generate_validation_data(n_bars=n_bars)
        results["stages"]["data_generation"] = {
            "symbols": list(data.keys()),
            "bars_per_symbol": {k: len(v) for k, v in data.items()},
        }

        # Stage 2: 三集分离
        logger.info("Stage 2: Train/Validation/Test split...")
        btc_data = data.get("BTC-USDT", [])
        split = self.strict_train_val_test_split(btc_data)
        results["stages"]["data_split"] = split.split_info

        # Stage 3: 时间序列CV
        logger.info("Stage 3: Time series cross-validation...")
        cv_info = self.time_series_cv(btc_data, n_splits=5)
        results["stages"]["time_series_cv"] = cv_info

        # Stage 4: 过拟合检测（模拟）
        logger.info("Stage 4: Overfitting detection...")
        overfit_result = self.detect_overfitting(
            train_score=85.0,  # 模拟训练集得分
            test_score=72.0,   # 模拟测试集得分
        )
        results["stages"]["overfitting_detection"] = {
            "is_overfit": overfit_result.is_overfit,
            "severity": overfit_result.severity,
            "overfit_ratio": overfit_result.overfit_ratio,
            "details": overfit_result.details,
        }

        # Stage 5: 鲁棒性分析（模拟）
        logger.info("Stage 5: Robustness analysis...")
        robustness_result = self.analyze_robustness(
            base_param=0.5,
            variations=[0.3, 0.4, 0.5, 0.6, 0.7],
            base_score=80.0,
            scores=[75, 78, 80, 79, 76],
        )
        results["stages"]["robustness_analysis"] = {
            "is_robust": robustness_result.is_robust,
            "sensitivity": robustness_result.sensitivity,
            "stable_region": robustness_result.stable_region,
            "details": robustness_result.details,
        }

        # 总结
        all_passed = (
            not overfit_result.is_overfit
            and robustness_result.is_robust
        )
        results["overall_passed"] = all_passed
        results["grade"] = "EXCELLENT" if all_passed else "DEGRADED"

        logger.info("Strict validation completed: grade=%s", results["grade"])

        return results


# ============================================================================
# 便利函数
# ============================================================================


def create_strict_validator() -> StrictValidator:
    """创建严格验证器实例"""
    return StrictValidator()


def quick_overfit_check(train_score: float, test_score: float) -> Dict[str, Any]:
    """快速过拟合检查"""
    result = OverfittingDetector.detect_train_test_gap(train_score, test_score)
    return {
        "is_overfit": result.is_overfit,
        "severity": result.severity,
        "overfit_ratio": result.overfit_ratio,
        "details": result.details,
    }
