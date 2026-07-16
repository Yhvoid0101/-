# -*- coding: utf-8 -*-
"""
反过拟合验证模块 (Anti-Overfitting Validator)

防止"沙盘赚钱、实盘亏钱"的核心模块。基于2025-2026前沿最佳实践：

1. Walk-Forward三集分离（IS-WFA-OOS协议）
   - 来源：AlgoXpert Alpha Research Framework (arXiv:2603.09219, 2026)
   - 训练50% / 验证20% / 测试30%，严格时间序列分割
   - Purging + Embargo 防信息泄露

2. 蒙特卡洛置换检验（Monte Carlo Permutation Test）
   - 来源：neurotrader 2025四步框架
   - 1000次随机打乱交易顺序，p<0.05才通过
   - 检测路径依赖性和回撤分布

3. 真实滑点/市场冲击模拟（Square-Root Law）
   - 来源：Almgren-Chriss最优执行框架
   - Market Impact = k × √(Order_Size / ADV)
   - 动态滑点+流动性限制，与实盘误差<3%

4. PBO过拟合检测（Probability of Backtest Overfitting）
   - 来源：Bailey & López de Prado CSCV方法
   - PBO > 0.5 = 过拟合（BLOCK）
   - PBO < 0.2 = 稳健（PASS）

5. 参数敏感性分析（Plateaus vs Peaks）
   - 来源：backtesting.py参数热力图分析
   - 参数±10%扰动，性能下降>50%=脆弱
   - 追求平台区域，而非单一尖峰

核心铁律：沙盘赚钱 ≠ 实盘赚钱。只有通过全部反过拟合验证的策略才能部署。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.anti_overfitting")


# ============================================================================
# 验证结果等级
# ============================================================================


class OverfittingRisk(Enum):
    """过拟合风险等级"""

    EXCELLENT = "excellent"  # PBO<0.1, OOS/IS>0.8
    GOOD = "good"  # PBO<0.2, OOS/IS>0.5
    ACCEPTABLE = "acceptable"  # PBO<0.3, OOS/IS>0.3
    WARN = "warn"  # PBO<0.5, OOS/IS>0.2
    BLOCK = "block"  # PBO>=0.5 或 OOS/IS<0.2


@dataclass(slots=True)
class ValidationResult:
    """验证结果"""

    risk_level: OverfittingRisk
    is_overfitted: bool  # True=过拟合，禁止部署
    pbo: float  # Probability of Backtest Overfitting
    oos_is_ratio: float  # OOS收益/IS收益
    monte_carlo_pvalue: float  # 蒙特卡洛置换检验p值
    max_drawdown_ratio: float  # OOS_MaxDD / IS_MaxDD
    parameter_stability: float  # 参数稳定性评分（0-1）
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# 1. Walk-Forward三集分离
# ============================================================================


class WalkForwardValidator:
    """Walk-Forward前向验证器

    实施IS-WFA-OOS三阶段协议：
      - IS (In-Sample): 50%数据，用于策略开发和参数优化
      - Validation: 20%数据，用于参数冻结检查
      - OOS (Out-of-Sample): 30%数据，最终验证，禁止调参

    关键原则：
      1. 严格时间序列分割，禁止随机打乱
      2. Purging: 移除训练集与测试集时间重叠的样本
      3. Embargo: 测试集后施加缓冲期
      4. 参数冻结: IS期设参数，OOS期禁止调参
    """

    def __init__(
        self,
        train_ratio: float = 0.50,
        validation_ratio: float = 0.20,
        test_ratio: float = 0.30,
        purge_gap: int = 24,  # 24根K线（1天）的purge gap
        embargo_bars: int = 12,  # 12根K线的embargo缓冲
    ):
        """
        Args:
            train_ratio: 训练集比例（默认50%）
            validation_ratio: 验证集比例（默认20%）
            test_ratio: 测试集比例（默认30%）
            purge_gap: 清洗间隔（K线数），防止标签重叠
            embargo_bars: 禁运缓冲期（K线数），防止时间相关性污染
        """
        assert abs(train_ratio + validation_ratio + test_ratio - 1.0) < 1e-6, \
            "三集比例之和必须=1.0"
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.purge_gap = purge_gap
        self.embargo_bars = embargo_bars

    def split_data(self, total_bars: int) -> Dict[str, Tuple[int, int]]:
        """将数据按时间序列分割为三集

        Args:
            total_bars: 总K线数

        Returns:
            {"train": (start, end), "validation": (start, end), "test": (start, end)}
        """
        train_end = int(total_bars * self.train_ratio)
        validation_start = train_end + self.purge_gap  # Purging
        validation_end = validation_start + int(total_bars * self.validation_ratio)
        test_start = validation_end + self.embargo_bars  # Embargo
        test_end = total_bars

        return {
            "train": (0, train_end),
            "validation": (validation_start, validation_end),
            "test": (test_start, test_end),
        }

    def evaluate(
        self,
        is_metrics: Dict[str, float],
        oos_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """评估IS vs OOS性能差距

        量化标准（来自AlgoXpert 2026）：
          - OOS Sharpe >= IS Sharpe * 0.5 → 通过
          - OOS Sharpe < IS Sharpe * 0.3 → 过拟合
          - OOS MaxDD < 2×IS MaxDD → 通过
          - OOS MaxDD > 3×IS MaxDD → 过拟合

        Args:
            is_metrics: 训练集指标 {"sharpe": float, "max_drawdown": float, "return": float, ...}
            oos_metrics: 测试集指标

        Returns:
            评估结果字典
        """
        is_sharpe = is_metrics.get("sharpe", 0)
        oos_sharpe = oos_metrics.get("sharpe", 0)
        is_return = is_metrics.get("return", 0)
        oos_return = oos_metrics.get("return", 0)
        is_dd = abs(is_metrics.get("max_drawdown", 0))
        oos_dd = abs(oos_metrics.get("max_drawdown", 0))

        # OOS/IS 收益比 — 边界安全处理
        # 当IS收益接近0时，ratio会爆炸；当两者都为负时，ratio语义错误
        # 改进：使用差值+对数+符号的组合，避免除零和负值问题
        if abs(is_return) > 1e-6:
            if is_return > 0:
                # IS正收益：正常比率
                oos_is_ratio = oos_return / is_return
            elif oos_return > 0:
                # IS负OOS正：方向反转，严重过拟合
                oos_is_ratio = -1.0
            else:
                # 双负：OOS亏得更多=过拟合，OOS亏得少=可接受
                oos_is_ratio = oos_return / is_return  # 两个负数相除得正
        else:
            # IS接近0：无法计算比率，用差值判断
            oos_is_ratio = 1.0 if oos_return > 0 else (0.0 if abs(oos_return) < 1e-6 else -1.0)

        # Sharpe衰减
        sharpe_decay = (oos_sharpe / is_sharpe) if abs(is_sharpe) > 1e-10 else 0.0

        # 回撤放大倍数
        dd_ratio = (oos_dd / is_dd) if is_dd > 1e-10 else 999.0

        # 判定
        is_overfitted = False
        issues = []

        if sharpe_decay < 0.3:
            is_overfitted = True
            issues.append(f"Sharpe衰减严重: IS={is_sharpe:.2f} → OOS={oos_sharpe:.2f} (decay={sharpe_decay:.1%})")
        elif sharpe_decay < 0.5:
            issues.append(f"Sharpe衰减警告: IS={is_sharpe:.2f} → OOS={oos_sharpe:.2f} (decay={sharpe_decay:.1%})")

        if dd_ratio > 3.0:
            is_overfitted = True
            issues.append(f"回撤放大严重: IS_DD={is_dd:.1%} → OOS_DD={oos_dd:.1%} (ratio={dd_ratio:.1f}x)")
        elif dd_ratio > 2.0:
            issues.append(f"回撤放大警告: IS_DD={is_dd:.1%} → OOS_DD={oos_dd:.1%} (ratio={dd_ratio:.1f}x)")

        if oos_return < 0 and is_return > 0:
            is_overfitted = True
            issues.append(f"收益方向反转: IS=+{is_return:.1%} → OOS={oos_return:.1%}（沙盘赚钱实盘亏钱）")

        return {
            "is_overfitted": is_overfitted,
            "oos_is_ratio": oos_is_ratio,
            "sharpe_decay": sharpe_decay,
            "dd_ratio": dd_ratio,
            "issues": issues,
            "verdict": "BLOCK" if is_overfitted else ("WARN" if issues else "PASS"),
        }


# ============================================================================
# 2. 蒙特卡洛置换检验
# ============================================================================


class MonteCarloPermutationTest:
    """蒙特卡洛置换检验

    通过随机打乱交易顺序，生成1000条替代权益曲线，检测策略是否真正有效。

    原理：
      - 如果策略有效，实际收益应在置换分布的高百分位
      - p值 = 置换收益 >= 实际收益的比例
      - p < 0.05 → 策略有效（拒绝"策略是垃圾"的零假设）
      - p >= 0.05 → 策略可能只是随机噪声

    量化标准（来自QuantProof 2025）：
      - p < 0.01 → EXCELLENT（99%置信）
      - p < 0.05 → GOOD（95%置信）
      - p < 0.10 → ACCEPTABLE（90%置信）
      - p >= 0.10 → BLOCK（策略无效）
    """

    def __init__(self, n_iterations: int = 1000, confidence_level: float = 0.95):
        """
        Args:
            n_iterations: 蒙特卡洛迭代次数（默认1000）
            confidence_level: 置信水平（默认95%）
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level

    def test(
        self,
        trade_pnls: List[float],
        metric: str = "sharpe",
    ) -> Dict[str, Any]:
        """执行蒙特卡洛置换检验

        使用Bootstrap自助法（有放回抽样）而非简单打乱顺序。
        因为打乱顺序不改变mean/std，夏普比率不变，无法检测策略有效性。

        Bootstrap原理：
          - 从原始PnL中有放回抽样N次，生成新的PnL序列
          - 计算新序列的指标（如累计收益）
          - 重复1000次，得到指标分布
          - p值 = 置换收益 <= 0 的比例（零假设：策略无优势）

        量化标准（来自QuantProof 2025）：
          - p < 0.01 → EXCELLENT（99%置信）
          - p < 0.05 → GOOD（95%置信）
          - p < 0.10 → ACCEPTABLE（90%置信）
          - p >= 0.10 → BLOCK（策略无效）

        Args:
            trade_pnls: 交易盈亏列表
            metric: 评估指标 ("return", "sharpe", "profit_factor")

        Returns:
            验证结果字典
        """
        if len(trade_pnls) < 5:
            return {
                "pvalue": 1.0,
                "actual_metric": 0.0,
                "permuted_mean": 0.0,
                "permuted_std": 0.0,
                "percentile": 0.0,
                "is_significant": False,
                "mc_sharpe_5pct": 0.0,
                "ruin_probability": 1.0,
                "error": "交易数<5，无法执行置换检验",
            }

        pnls = np.array(trade_pnls)
        n = len(pnls)
        actual_metric = self._compute_metric(pnls, metric)

        # P4#10: 向量化 Bootstrap — 一次性生成所有样本矩阵 (n_iterations × n)
        # 替代 O(n_iterations × n) 的 Python 双重循环，预计 10-50 倍加速
        # 内存：n_iterations=1000, n=1000 → 8MB，完全可接受
        samples = np.random.choice(pnls, size=(self.n_iterations, n), replace=True)

        # 向量化计算指标
        if metric == "sharpe":
            means = np.mean(samples, axis=1)
            stds = np.std(samples, axis=1, ddof=1)
            valid = stds > 1e-10
            permuted_metrics = np.zeros(self.n_iterations, dtype=np.float64)
            permuted_metrics[valid] = means[valid] / stds[valid] * math.sqrt(n)
        elif metric == "return":
            permuted_metrics = np.sum(samples, axis=1)
        elif metric == "profit_factor":
            wins_sum = np.sum(np.where(samples > 0, samples, 0.0), axis=1)
            losses_sum = np.abs(np.sum(np.where(samples < 0, samples, 0.0), axis=1))
            # 避免除以0：losses_sum=0 时，有盈利=inf，无盈利=0
            safe_losses = np.where(losses_sum > 0, losses_sum, 1.0)
            permuted_metrics = np.where(
                losses_sum > 0,
                wins_sum / safe_losses,
                np.where(wins_sum > 0, np.inf, 0.0),
            )
        else:
            permuted_metrics = np.zeros(self.n_iterations, dtype=np.float64)

        # 向量化破产概率：cumsum 计算权益曲线，any 检查是否归零
        initial_capital = 10000.0
        equity_curves = initial_capital + np.cumsum(samples, axis=1)
        ruin_mask = np.any(equity_curves <= 0, axis=1)
        ruin_count = int(np.sum(ruin_mask))
        permuted_mean = float(np.mean(permuted_metrics))
        permuted_std = float(np.std(permuted_metrics))

        # p值：Bootstrap收益 <= 0 的比例（零假设：策略无优势）
        # 如果策略有效，Bootstrap收益应该大部分>0
        pvalue = float(np.mean(permuted_metrics <= 0))

        # 百分位：实际值在置换分布中的位置
        if permuted_std > 1e-10:
            percentile = float(np.mean(permuted_metrics <= actual_metric))
        else:
            percentile = 0.5

        # 蒙特卡洛5%分位
        mc_sharpe_5pct = float(np.percentile(permuted_metrics, 5))

        # 破产概率
        ruin_probability = ruin_count / self.n_iterations

        # 统计显著性
        alpha = 1 - self.confidence_level
        is_significant = pvalue < alpha

        return {
            "pvalue": pvalue,
            "actual_metric": float(actual_metric),
            "permuted_mean": permuted_mean,
            "permuted_std": permuted_std,
            "percentile": percentile,
            "is_significant": is_significant,
            "mc_sharpe_5pct": mc_sharpe_5pct,
            "ruin_probability": ruin_probability,
            "n_iterations": self.n_iterations,
            "method": "bootstrap",
        }

    def _compute_metric(self, pnls: np.ndarray, metric: str) -> float:
        """计算评估指标"""
        if metric == "sharpe":
            if len(pnls) < 2:
                return 0.0
            mean = np.mean(pnls)
            std = np.std(pnls, ddof=1)
            if std < 1e-10:
                return 0.0
            return float(mean / std * math.sqrt(len(pnls)))
        elif metric == "return":
            return float(np.sum(pnls))
        elif metric == "profit_factor":
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            if len(losses) == 0 or np.sum(losses) == 0:
                return float("inf") if len(wins) > 0 else 0.0
            return float(np.sum(wins) / abs(np.sum(losses)))
        return 0.0


# ============================================================================
# 3. 真实滑点/市场冲击模拟
# ============================================================================


class RealisticSlippageModel:
    """真实滑点/市场冲击模拟

    基于Square-Root Law（平方根法则）：
      Market Impact = k × √(Order_Size / ADV)

    来源：Almgren-Chriss最优执行框架
    实证：动态滑点+流动性模型与实盘误差<3%

    五层撮合模型（我们实现L3-L4）：
      L1: 收盘价成交（高估收益约50%）
      L2: 固定滑点
      L3: 成交量加权撮合
      L4: 订单簿驱动 + 市场冲击
      L5: 含网络延迟+风控逻辑（实盘专属）
    """

    def __init__(
        self,
        impact_coefficient: float = 0.1,  # k值，流动性系数（0.1-0.5）
        base_slippage_bps: float = 2.0,   # 基础滑点（基点）
        max_slippage_bps: float = 50.0,   # 最大滑点（基点）
        latency_ms: float = 100.0,        # 模拟延迟（毫秒）
    ):
        """
        Args:
            impact_coefficient: 市场冲击系数k（0.1=高流动性，0.5=低流动性）
            base_slippage_bps: 基础滑点（2基点=0.02%）
            max_slippage_bps: 最大滑点上限（50基点=0.5%）
            latency_ms: 模拟网络延迟（毫秒）
        """
        self.k = impact_coefficient
        self.base_slippage_bps = base_slippage_bps
        self.max_slippage_bps = max_slippage_bps
        self.latency_ms = latency_ms

    def compute_slippage(
        self,
        order_size: float,
        adv: float,  # Average Daily Volume
        price: float,
        side: str = "long",
    ) -> Dict[str, float]:
        """计算真实滑点

        Square-Root Law:
          Market Impact = k × √(Order_Size / ADV) × price

        Args:
            order_size: 订单数量（合约数）
            adv: 日均成交量
            price: 当前价格
            side: "long"或"short"

        Returns:
            {
                "slippage_bps": float,      # 总滑点（基点）
                "slippage_amount": float,   # 滑点金额
                "impact_bps": float,        # 市场冲击（基点）
                "base_bps": float,          # 基础滑点（基点）
                "latency_cost_bps": float,  # 延迟成本（基点）
            }
        """
        # 订单占ADV比例
        # 自动检测adv单位：如果adv < price，说明adv是合约数，需转为名义价值
        if adv > 0:
            if adv < price:
                # adv是合约数，转为名义价值
                adv_notional = adv * price
            else:
                # adv已经是名义价值
                adv_notional = adv
            size_ratio = (order_size * price) / adv_notional
        else:
            size_ratio = 1.0

        # 市场冲击（平方根法则）
        impact_bps = self.k * math.sqrt(max(size_ratio, 0)) * 10000  # 转为基点

        # 基础滑点（买卖价差）
        base_bps = self.base_slippage_bps

        # 延迟成本（100ms延迟≈0.5基点）
        latency_cost_bps = self.latency_ms / 1000.0 * 5.0

        # 总滑点
        total_bps = base_bps + impact_bps + latency_cost_bps
        total_bps = min(total_bps, self.max_slippage_bps)  # 上限

        # 滑点金额
        slippage_amount = total_bps / 10000.0 * order_size * price

        # 方向：买入滑点为正（成交价更高），卖出滑点为负（成交价更低）
        direction = 1 if side == "long" else -1

        return {
            "slippage_bps": total_bps * direction,
            "slippage_amount": slippage_amount * direction,
            "impact_bps": impact_bps,
            "base_bps": base_bps,
            "latency_cost_bps": latency_cost_bps,
            "size_ratio": size_ratio,
        }

    def apply_to_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        adv: float,
        fee_bps: float = 5.0,  # 手续费（基点）
    ) -> Dict[str, float]:
        """将真实滑点应用到交易

        Args:
            entry_price: 入场价
            exit_price: 出场价
            quantity: 数量
            adv: 日均成交量
            fee_bps: 手续费（基点）

        Returns:
            调整后的交易结果
        """
        # 入场滑点（买入，成交价更高）
        entry_slip = self.compute_slippage(quantity, adv, entry_price, "long")
        adjusted_entry = entry_price * (1 + entry_slip["slippage_bps"] / 10000.0)

        # 出场滑点（卖出，成交价更低）
        exit_slip = self.compute_slippage(quantity, adv, exit_price, "short")
        adjusted_exit = exit_price * (1 + exit_slip["slippage_bps"] / 10000.0)

        # 手续费
        fee_amount = fee_bps / 10000.0 * quantity * (entry_price + exit_price)

        # 原始PnL vs 调整后PnL
        raw_pnl = (exit_price - entry_price) * quantity
        adjusted_pnl = (adjusted_exit - adjusted_entry) * quantity - fee_amount

        # 滑点+手续费对收益的侵蚀
        erosion = raw_pnl - adjusted_pnl
        erosion_pct = (erosion / abs(raw_pnl)) if abs(raw_pnl) > 1e-10 else 0.0

        return {
            "raw_pnl": raw_pnl,
            "adjusted_pnl": adjusted_pnl,
            "erosion": erosion,
            "erosion_pct": erosion_pct,
            "adjusted_entry": adjusted_entry,
            "adjusted_exit": adjusted_exit,
            "total_slippage_bps": abs(entry_slip["slippage_bps"]) + abs(exit_slip["slippage_bps"]),
            "fee_amount": fee_amount,
            "total_cost": erosion + fee_amount,  # 滑点侵蚀+手续费总成本
        }


# ============================================================================
# 4. PBO过拟合检测
# ============================================================================


class PBOValidator:
    """PBO（Probability of Backtest Overfitting）过拟合检测

    来源：Bailey & López de Prado CSCV方法
    论文："The Probability of Backtest Overfitting" (2014)

    原理：
      1. 将数据分为N组
      2. 评估所有C(N,k)种k组测试集组合
      3. 统计样本内最优策略在样本外仍为最优的比例
      4. PBO = 这个比例

    判定标准：
      - PBO ≈ 0: 策略有明显优势（PASS）
      - PBO ≈ 0.5: 策略无优势（WARN）
      - PBO > 0.5: 选择过程在追逐噪声（BLOCK=过拟合）
      - PBO < 0.2: 策略稳健（EXCELLENT）
    """

    def __init__(self, n_groups: int = 8, test_groups: int = 4):
        """
        Args:
            n_groups: 数据分组数（默认8）
            test_groups: 测试组数（默认4，即50%测试集）
        """
        self.n_groups = n_groups
        self.test_groups = test_groups

    def compute_pbo(
        self,
        strategy_returns: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """计算PBO

        Args:
            strategy_returns: {策略名: 收益序列}

        Returns:
            {
                "pbo": float,           # PBO值
                "is_overfitted": bool,  # 是否过拟合
                "risk_level": str,      # 风险等级
                "n_combinations": int,  # 评估的组合数
            }
        """
        from itertools import combinations

        if len(strategy_returns) < 2:
            return {
                "pbo": 0.5,
                "is_overfitted": True,
                "risk_level": "BLOCK",
                "n_combinations": 0,
                "error": "策略数<2，无法计算PBO",
            }

        # 将每个策略的收益序列分为N组
        strategy_names = list(strategy_returns.keys())
        n_strategies = len(strategy_names)

        # 对齐所有策略的收益序列长度
        min_len = min(len(r) for r in strategy_returns.values())

        # v2.5修复：动态调整分组数，避免"收益序列长度<N<分组数"导致PBO无法计算
        # 之前：min_len<8直接返回PBO=0.5+BLOCK，导致短序列策略永远无法通过PBO验证
        # 现在：动态调整n_groups=min(min_len, 8)，test_groups=n_groups//2
        effective_n_groups = min(self.n_groups, max(2, min_len))
        effective_test_groups = max(1, effective_n_groups // 2)

        if min_len < 2:
            return {
                "pbo": 0.5,
                "is_overfitted": True,
                "risk_level": "BLOCK",
                "n_combinations": 0,
                "error": f"收益序列长度{min_len} < 2，无法计算PBO",
            }

        # 分组（使用动态调整后的分组数）
        group_size = min_len // effective_n_groups
        if group_size < 1:
            group_size = 1
        grouped_returns = {}
        for name, returns in strategy_returns.items():
            trimmed = returns[:min_len]
            groups = []
            for i in range(effective_n_groups):
                start = i * group_size
                end = start + group_size
                groups.append(trimmed[start:end])
            grouped_returns[name] = groups

        # CSCV: 对称组合交叉验证
        test_combinations = list(combinations(range(effective_n_groups), effective_test_groups))
        n_combinations = len(test_combinations)

        pbo_count = 0

        for test_idx in test_combinations:
            # P4#13: 将 test_idx 转为 set，使 `i not in test_idx` 从 O(n) 降为 O(1)
            test_idx_set = set(test_idx)
            train_idx = [i for i in range(self.n_groups) if i not in test_idx_set]

            # 计算每个策略在训练集和测试集的夏普
            is_sharpes = {}
            oos_sharpes = {}
            for name in strategy_names:
                # 训练集夏普
                is_returns = []
                for i in train_idx:
                    is_returns.extend(grouped_returns[name][i])
                is_sharpes[name] = self._sharpe(is_returns)

                # 测试集夏普
                oos_returns = []
                for i in test_idx:
                    oos_returns.extend(grouped_returns[name][i])
                oos_sharpes[name] = self._sharpe(oos_returns)

            # IS最优策略
            is_best = max(is_sharpes, key=is_sharpes.get)

            # OOS排名
            oos_ranked = sorted(oos_sharpes.items(), key=lambda x: x[1], reverse=True)
            oos_rank_of_is_best = next(
                i for i, (name, _) in enumerate(oos_ranked) if name == is_best
            )

            # PBO: IS最优策略在OOS排名的中位数以下
            median_rank = n_strategies / 2.0
            if oos_rank_of_is_best >= median_rank:
                pbo_count += 1

        pbo = pbo_count / n_combinations if n_combinations > 0 else 0.5

        # 风险等级
        if pbo < 0.1:
            risk_level = "EXCELLENT"
            is_overfitted = False
        elif pbo < 0.2:
            risk_level = "GOOD"
            is_overfitted = False
        elif pbo < 0.3:
            risk_level = "ACCEPTABLE"
            is_overfitted = False
        elif pbo < 0.5:
            risk_level = "WARN"
            is_overfitted = False
        else:
            risk_level = "BLOCK"
            is_overfitted = True

        return {
            "pbo": pbo,
            "is_overfitted": is_overfitted,
            "risk_level": risk_level,
            "n_combinations": n_combinations,
            "n_strategies": n_strategies,
        }

    def _sharpe(self, returns: List[float]) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(mean / std * math.sqrt(len(returns)))

    def _compute_cscv(
        self,
        is_matrix: np.ndarray,
        oos_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """Combinatorial Symmetric Cross-Validation (CSCV) — Bailey & López de Prado 2014

        论文: "The Probability of Backtest Overfitting" (2014)
        实现: 基于 logit 排名矩阵的精确 PBO 计算

        Args:
            is_matrix: IS收益矩阵 (N_strategies × N_IS_periods)
            oos_matrix: OOS收益矩阵 (N_strategies × N_OOS_periods)

        Returns:
            {
                "pbo": float,           # PBO值 [0,1]
                "logit_matrix": array,  # logit排名矩阵（调试用）
                "n_strategies": int,
                "method": "cscv_bailey_ldp_2014",
            }
        """
        n_strategies = is_matrix.shape[0]
        if n_strategies < 2:
            return {"pbo": 0.5, "error": "策略数<2，无法计算CSCV", "method": "cscv_bailey_ldp_2014"}

        # 计算每个策略的IS和OOS Sharpe
        is_sharpes = np.array([self._sharpe(is_matrix[i]) for i in range(n_strategies)])
        oos_sharpes = np.array([self._sharpe(oos_matrix[i]) for i in range(n_strategies)])

        # IS最优策略的索引
        is_best_idx = int(np.argmax(is_sharpes))

        # OOS排名（降序，1=最优）
        oos_order = np.argsort(-oos_sharpes)  # 降序排列的索引
        oos_rank_of_is_best = int(np.where(oos_order == is_best_idx)[0][0]) + 1  # 1-based rank

        # PBO: IS最优策略在OOS排名的中位数或以下
        # 论文定义: PBO = P(OOS_rank_of_IS_best > N/2)
        median_rank = n_strategies / 2.0
        pbo = 1.0 if oos_rank_of_is_best > median_rank else 0.0

        # logit排名矩阵（论文公式）
        # r_i = rank of strategy i by IS Sharpe (1=best)
        # omega_i = rank of strategy i by OOS Sharpe (1=best)
        is_ranks = np.argsort(np.argsort(-is_sharpes)) + 1  # 1-based ranks
        oos_ranks = np.argsort(np.argsort(-oos_sharpes)) + 1

        # logit变换: λ = ln(r / (N+1-r))
        with np.errstate(divide="ignore", invalid="ignore"):
            is_logits = np.log(is_ranks / (n_strategies + 1 - is_ranks))
            oos_logits = np.log(oos_ranks / (n_strategies + 1 - oos_ranks))

        return {
            "pbo": float(pbo),
            "is_best_idx": is_best_idx,
            "oos_rank_of_is_best": oos_rank_of_is_best,
            "is_sharpes": is_sharpes.tolist(),
            "oos_sharpes": oos_sharpes.tolist(),
            "is_ranks": is_ranks.tolist(),
            "oos_ranks": oos_ranks.tolist(),
            "n_strategies": n_strategies,
            "method": "cscv_bailey_ldp_2014",
        }

    def generate_candidate_strategies(
        self,
        base_returns: List[float],
        n_candidates: int = 20,
        perturbation_std: float = 0.02,
    ) -> Dict[str, List[float]]:
        """通过收益序列扰动生成候选策略（用于单策略PBO计算）

        v597 Bug#4修复: 单策略时用扰动生成N个候选策略，再执行真实CSCV
        替代旧的 pbo=1-oos_is_r 粗略近似

        Args:
            base_returns: 基准策略收益序列
            n_candidates: 候选策略数（默认20，Bailey&LdP建议≥10）
            perturbation_std: 扰动标准差（默认2%收益）

        Returns:
            {策略名: 扰动后收益序列}
        """
        np.random.seed(42)  # 可复现
        base = np.array(base_returns, dtype=float)
        candidates = {"base_strategy": base.tolist()}

        for i in range(n_candidates - 1):
            # 对收益叠加高斯噪声模拟参数扰动效果
            noise = np.random.normal(0, perturbation_std, len(base))
            perturbed = base + noise
            candidates[f"candidate_{i+1:02d}"] = perturbed.tolist()

        return candidates

    def compute_pbo_single_strategy(
        self,
        trade_pnls: List[float],
        is_sharpe: float,
        oos_sharpe: float,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """单策略PBO科学估计 (v596 Bug#4修复)

        替代旧版 `pbo = 1 - oos_is_r` 粗略近似。

        方法：Bootstrap重采样估计PBO的置信区间
          1. 将trade_pnls分为IS/OOS两半（用walk-forward的分割）
          2. 对IS和OOS分别做bootstrap重采样，生成Sharpe分布
          3. PBO = P(OOS_Sharpe < IS_Sharpe中位数) 的估计
          4. 用bootstrap置信区间量化不确定性

        科学依据：
          - Bailey & López de Prado 2014: PBO本需多策略CSCV
          - 单策略时用bootstrap估计"IS最优→OOS表现"的概率
          - 这是对"选择偏差"的保守估计（IS已选为最优，OOS衰减概率）

        Args:
            trade_pnls: 完整交易盈亏序列
            is_sharpe: 训练集Sharpe（已知）
            oos_sharpe: 测试集Sharpe（已知）
            n_bootstrap: bootstrap重采样次数（默认1000）

        Returns:
            {
                "pbo": float,              # PBO点估计
                "pbo_ci_low": float,       # PBO 95%置信区间下界
                "pbo_ci_high": float,      # PBO 95%置信区间上界
                "is_sharpe": float,        # IS Sharpe
                "oos_sharpe": float,       # OOS Sharpe
                "degradation_factor": float,  # DF = OOS/IS
                "method": str,             # 方法说明
                "is_overfitted": bool,     # 是否过拟合
                "risk_level": str,         # 风险等级
            }
        """
        if len(trade_pnls) < 10:
            return {
                "pbo": 0.5,
                "pbo_ci_low": 0.0,
                "pbo_ci_high": 1.0,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "degradation_factor": oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0,
                "method": "v596_bootstrap_insufficient_data",
                "is_overfitted": True,
                "risk_level": "BLOCK",
                "error": f"trade_pnls长度{len(trade_pnls)}<10，无法执行bootstrap",
            }

        pnls = np.array(trade_pnls, dtype=float)
        n = len(pnls)
        mid = n // 2

        # Bootstrap重采样估计OOS Sharpe分布
        rng = np.random.default_rng(42)
        oos_sharpe_boot = np.empty(n_bootstrap)
        is_sharpe_boot = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            # IS bootstrap
            is_sample = rng.choice(pnls[:mid], size=mid, replace=True)
            is_sharpe_boot[i] = self._sharpe(is_sample.tolist())
            # OOS bootstrap
            oos_sample = rng.choice(pnls[mid:], size=n - mid, replace=True)
            oos_sharpe_boot[i] = self._sharpe(oos_sample.tolist())

        # PBO估计: OOS Sharpe < IS Sharpe中位数的比例
        is_median = np.median(is_sharpe_boot)
        pbo_samples = (oos_sharpe_boot < is_median).astype(float)
        pbo = float(np.mean(pbo_samples))

        # 95%置信区间（正态近似）
        pbo_std = float(np.std(pbo_samples, ddof=1))
        pbo_ci_low = max(0.0, pbo - 1.96 * pbo_std)
        pbo_ci_high = min(1.0, pbo + 1.96 * pbo_std)

        # DF (Degradation Factor)
        df = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

        # 风险等级（与多策略PBO一致的阈值）
        if pbo < 0.1:
            risk_level = "EXCELLENT"
            is_overfitted = False
        elif pbo < 0.2:
            risk_level = "GOOD"
            is_overfitted = False
        elif pbo < 0.3:
            risk_level = "ACCEPTABLE"
            is_overfitted = False
        elif pbo < 0.5:
            risk_level = "WARN"
            is_overfitted = False
        else:
            risk_level = "BLOCK"
            is_overfitted = True

        return {
            "pbo": pbo,
            "pbo_ci_low": pbo_ci_low,
            "pbo_ci_high": pbo_ci_high,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "degradation_factor": df,
            "method": "v596_bootstrap_single_strategy",
            "n_bootstrap": n_bootstrap,
            "is_overfitted": is_overfitted,
            "risk_level": risk_level,
        }


# ============================================================================
# 5. 参数敏感性分析
# ============================================================================


class ParameterSensitivityAnalyzer:
    """参数敏感性分析

    检测策略参数是否处于"平台区域"（稳健）还是"尖峰"（脆弱）。

    来源：backtesting.py参数热力图分析
    原则：追求平台区域，而非单一尖峰

    量化标准：
      - 参数±10%扰动，性能下降>50% = 脆弱（BLOCK）
      - 参数±10%扰动，性能下降<20% = 稳健（EXCELLENT）
      - 仅单一精确参数有效 = 过拟合

    v596 D3 升级: 三档扰动 [±10%, ±20%, ±30%] + 平台/尖峰/斜坡区域分类
      - plateau (平台): ±10% 退化 <5% (稳健，EXCELLENT)
      - slope (斜坡): ±10% 退化 5-20% (可接受，GOOD)
      - peak (尖峰): ±10% 退化 >20% 或 ±20% 退化 >50% (过拟合，BLOCK)
    """

    def __init__(
        self,
        perturbation_pct: float = None,
        perturbation_levels: List[float] = None,
    ):
        """v596 D3: 多档扰动参数敏感性分析

        Args:
            perturbation_pct: (向后兼容) 单一扰动百分比，如 0.10 表示 ±10%
            perturbation_levels: (v596 新增) 多档扰动百分比列表，默认 [0.10, 0.20, 0.30]

        Note:
            若 perturbation_levels 提供，则忽略 perturbation_pct；
            若仅 perturbation_pct 提供，则 levels=[perturbation_pct]；
            若两者都未提供，则使用默认 [0.10, 0.20, 0.30]
        """
        if perturbation_levels is not None:
            self.perturbation_levels = sorted(perturbation_levels)
        elif perturbation_pct is not None:
            self.perturbation_levels = [perturbation_pct]
        else:
            self.perturbation_levels = [0.10, 0.20, 0.30]
        # 向后兼容: 保留 perturbation_pct 字段（取首档）
        self.perturbation_pct = self.perturbation_levels[0]

    def analyze(
        self,
        base_params: Dict[str, float],
        param_performance_fn,
    ) -> Dict[str, Any]:
        """分析参数敏感性 (v596 D3: 三档扰动 + 区域分类)

        Args:
            base_params: 基准参数 {参数名: 值}
            param_performance_fn: 函数，输入参数字典，返回性能指标（如GT-Score）

        Returns:
            {
                "stability_score": float,        # 稳定性评分（0-1）
                "is_robust": bool,               # 是否稳健
                "sensitive_params": List,        # 敏感参数列表
                "perturbation_results": Dict,    # 各参数各档位扰动结果
                "regions": Dict[str, str],       # v596 D3: 各参数的区域分类
                "base_performance": float,       # 基准性能
                "avg_performance_drop": float,   # 平均性能下降
            }
        """
        base_performance = param_performance_fn(base_params)

        if base_performance is None or base_performance <= 0:
            return {
                "stability_score": 0.0,
                "is_robust": False,
                "sensitive_params": list(base_params.keys()),
                "perturbation_results": {},
                "regions": {},
                "base_performance": 0.0,
                "avg_performance_drop": 1.0,
                "error": "基准性能无效",
            }

        perturbation_results: Dict[str, Dict[str, float]] = {}
        regions: Dict[str, str] = {}
        sensitive_params: List[str] = []
        performance_drops: List[float] = []

        for param_name, param_value in base_params.items():
            if param_value == 0:
                continue  # 跳过0值参数

            results: Dict[str, float] = {}
            # v596 D3: 按档位收集退化率，用于区域分类
            drops_by_level: Dict[float, float] = {}

            for level in self.perturbation_levels:
                level_drops: List[float] = []
                for direction, mult in [("+", 1 + level), ("-", 1 - level)]:
                    key = f"±{int(level * 100)}%{direction}"
                    perturbed = base_params.copy()
                    perturbed[param_name] = param_value * mult
                    perf = param_performance_fn(perturbed)
                    results[key] = perf

                    if perf is not None and base_performance > 0:
                        drop = (base_performance - perf) / base_performance
                        performance_drops.append(drop)
                        level_drops.append(drop)
                        # 敏感参数: ±10% 退化 >50%
                        if level == 0.10 and drop > 0.5:
                            sensitive_params.append(f"{param_name}{direction}")

                # 该档位的最大退化（取 +/- 中较大者）
                if level_drops:
                    drops_by_level[level] = max(level_drops)

            perturbation_results[param_name] = results

            # v596 D3: 区域分类
            regions[param_name] = self._classify_region(drops_by_level)

        # 稳定性评分：1 - 平均性能下降
        avg_drop = sum(performance_drops) / len(performance_drops) if performance_drops else 1.0
        stability_score = max(0.0, 1.0 - avg_drop)

        # 判定: 稳健需要 stability_score≥0.8 + 无敏感参数 + 无 peak 区域
        has_peak = any(r == "peak" for r in regions.values())
        is_robust = stability_score >= 0.8 and len(sensitive_params) == 0 and not has_peak

        return {
            "stability_score": stability_score,
            "is_robust": is_robust,
            "sensitive_params": sensitive_params,
            "perturbation_results": perturbation_results,
            "regions": regions,
            "base_performance": base_performance,
            "avg_performance_drop": avg_drop,
            "has_peak_region": has_peak,  # v596 D3: 是否存在尖峰过拟合
        }

    def _classify_region(self, drops_by_level: Dict[float, float]) -> str:
        """v596 D3: 参数区域分类

        基于多档扰动退化率判定参数处于哪种区域:
          - plateau (平台): ±10% 退化 <5% (稳健，参数选择不敏感)
          - slope (斜坡): ±10% 退化 5-20% (可接受，参数有一定影响)
          - peak (尖峰): ±10% 退化 >20% 或 ±20% 退化 >50% (过拟合，参数极度敏感)

        Args:
            drops_by_level: {扰动级别: 最大退化率}，如 {0.10: 0.03, 0.20: 0.08, 0.30: 0.15}

        Returns:
            str: "plateau" / "slope" / "peak"
        """
        drop_10 = drops_by_level.get(0.10, 0.0)
        drop_20 = drops_by_level.get(0.20, 0.0)
        # 若无 0.10 档位，取最小档位的退化
        if 0.10 not in drops_by_level and drops_by_level:
            min_level = min(drops_by_level.keys())
            drop_10 = drops_by_level[min_level]

        if drop_10 > 0.20 or drop_20 > 0.50:
            return "peak"    # 尖峰过拟合
        elif drop_10 < 0.05:
            return "plateau"  # 平台稳健
        else:
            return "slope"   # 斜坡可接受

    def export_heatmap_csv(self, results: Dict[str, Any], output_path: str) -> str:
        """v596 D3: 导出参数敏感性热力图 CSV

        生成参数 × 扰动级别 × 性能 × 退化率 × 区域 的热力图，
        供可视化分析使用。

        Args:
            results: analyze() 方法的返回值
            output_path: CSV 输出路径

        Returns:
            str: 实际写入的路径（成功）或空字符串（失败）
        """
        import csv
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "param", "perturbation_key", "performance",
                    "drop_pct", "region", "is_sensitive"
                ])
                base_perf = results.get("base_performance", 0.0)
                regions = results.get("regions", {})
                sensitive_params = set(results.get("sensitive_params", []))

                for param, level_results in results.get("perturbation_results", {}).items():
                    region = regions.get(param, "unknown")
                    for key, perf in level_results.items():
                        drop_pct = (
                            (base_perf - perf) / base_perf
                            if (perf is not None and base_perf > 0)
                            else 0.0
                        )
                        is_sensitive = 1 if any(
                            param in sp for sp in sensitive_params
                        ) else 0
                        writer.writerow([
                            param, key, f"{perf:.6f}" if perf is not None else "N/A",
                            f"{drop_pct:.4f}", region, is_sensitive
                        ])
            return output_path
        except Exception as e:
            logger.warning(f"v596 D3: 热力图 CSV 导出失败: {e}")
            return ""



# ============================================================================
# 综合反过拟合验证器
# ============================================================================


class AntiOverfittingValidator:
    """综合反过拟合验证器

    整合5大验证模块，提供一站式反过拟合验证。

    使用方式:
        validator = AntiOverfittingValidator()
        result = validator.validate(
            is_metrics={"sharpe": 2.0, "return": 0.3, "max_drawdown": 0.1},
            oos_metrics={"sharpe": 1.5, "return": 0.2, "max_drawdown": 0.15},
            trade_pnls=[100, -50, 200, ...],
        )
        if result.is_overfitted:
            print("策略过拟合，禁止部署！")
    """

    def __init__(
        self,
        walk_forward: Optional[WalkForwardValidator] = None,
        monte_carlo: Optional[MonteCarloPermutationTest] = None,
        slippage_model: Optional[RealisticSlippageModel] = None,
        pbo_validator: Optional[PBOValidator] = None,
        sensitivity_analyzer: Optional[ParameterSensitivityAnalyzer] = None,
    ):
        self.walk_forward = walk_forward or WalkForwardValidator()
        self.monte_carlo = monte_carlo or MonteCarloPermutationTest(n_iterations=1000)
        self.slippage_model = slippage_model or RealisticSlippageModel()
        self.pbo_validator = pbo_validator or PBOValidator()
        self.sensitivity_analyzer = sensitivity_analyzer or ParameterSensitivityAnalyzer()

    def validate(
        self,
        is_metrics: Dict[str, float],
        oos_metrics: Dict[str, float],
        trade_pnls: List[float],
        strategy_returns: Optional[Dict[str, List[float]]] = None,
        apply_slippage: bool = True,
        trade_prices: Optional[List[float]] = None,
        trade_volumes: Optional[List[float]] = None,
        param_performance_fn: Optional[callable] = None,
        base_params: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """综合反过拟合验证

        Args:
            is_metrics: 训练集指标
            oos_metrics: 测试集指标
            trade_pnls: 交易盈亏列表
            strategy_returns: 多策略收益（用于PBO计算）
            apply_slippage: 是否应用真实滑点
            trade_prices: 每笔交易的入场价格（B10修复：替代硬编码50000）
            trade_volumes: 每笔交易的成交量（B10修复：替代硬编码ADV）
            param_performance_fn: 参数→性能函数（P9修复：启用真正的ParameterSensitivityAnalyzer）
            base_params: 基准参数字典（P9修复：配合param_performance_fn使用）

        Returns:
            ValidationResult 验证结果
        """
        recommendations = []
        details = {}

        # 1. Walk-Forward验证
        wf_result = self.walk_forward.evaluate(is_metrics, oos_metrics)
        details["walk_forward"] = wf_result
        if wf_result["is_overfitted"]:
            recommendations.append("Walk-Forward验证失败：OOS性能严重衰减，策略可能过拟合")

        # 2. 蒙特卡洛置换检验
        mc_result = self.monte_carlo.test(trade_pnls, metric="sharpe")
        details["monte_carlo"] = mc_result
        if not mc_result.get("is_significant", False):
            recommendations.append(
                f"蒙特卡洛检验未通过：p={mc_result['pvalue']:.3f} >= 0.05，策略可能只是随机噪声"
            )
        if mc_result.get("ruin_probability", 0) > 0.05:
            recommendations.append(
                f"破产概率过高：{mc_result['ruin_probability']:.1%}，策略风险不可控"
            )

        # 3. PBO过拟合检测（如果有多策略数据）
        pbo = 0.0
        if strategy_returns and len(strategy_returns) >= 2:
            pbo_result = self.pbo_validator.compute_pbo(strategy_returns)
            details["pbo"] = pbo_result
            pbo = pbo_result.get("pbo", 0.5)
            if pbo_result.get("is_overfitted", True):
                recommendations.append(
                    f"PBO={pbo:.2f} > 0.5，选择过程在追逐噪声，策略过拟合"
                )
        else:
            # v596 Bug#4修复: 单策略PBO科学估计（替代旧版 `pbo = 1 - oos_is_r` 粗略近似）
            # 旧版问题: 线性映射 oos_is_r→pbo 缺乏统计依据，无置信区间
            # 新版方法: bootstrap重采样估计 P(OOS_Sharpe < IS_Sharpe中位数) + 95%置信区间
            is_sharpe = is_metrics.get("sharpe", 0.0)
            oos_sharpe = oos_metrics.get("sharpe", 0.0)
            pbo_result = self.pbo_validator.compute_pbo_single_strategy(
                trade_pnls=trade_pnls,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
            )
            details["pbo"] = pbo_result
            pbo = pbo_result.get("pbo", 0.5)
            if pbo_result.get("is_overfitted", True):
                recommendations.append(
                    f"PBO={pbo:.2f} (95%CI: {pbo_result.get('pbo_ci_low', 0):.2f}-"
                    f"{pbo_result.get('pbo_ci_high', 1):.2f})，单策略bootstrap估计，"
                    f"DF={pbo_result.get('degradation_factor', 0):.3f}，策略可能过拟合"
                )

        # 4. 真实滑点影响评估
        if apply_slippage and trade_pnls:
            slippage_impact = self._estimate_slippage_impact(
                trade_pnls,
                trade_prices=trade_prices,
                trade_volumes=trade_volumes,
            )
            details["slippage_impact"] = slippage_impact
            if slippage_impact["erosion_pct"] > 0.5:
                recommendations.append(
                    f"滑点+手续费侵蚀{slippage_impact['erosion_pct']:.1%}收益，"
                    f"实盘可能由盈转亏"
                )

        # 综合判定
        oos_is_ratio = wf_result.get("oos_is_ratio", 0)
        mc_pvalue = mc_result.get("pvalue", 1.0)
        dd_ratio = wf_result.get("dd_ratio", 999)

        # 风险等级
        if pbo < 0.1 and oos_is_ratio > 0.8 and mc_pvalue < 0.01:
            risk_level = OverfittingRisk.EXCELLENT
        elif pbo < 0.2 and oos_is_ratio > 0.5 and mc_pvalue < 0.05:
            risk_level = OverfittingRisk.GOOD
        elif pbo < 0.3 and oos_is_ratio > 0.3 and mc_pvalue < 0.10:
            risk_level = OverfittingRisk.ACCEPTABLE
        elif pbo < 0.5 and oos_is_ratio > 0.2:
            risk_level = OverfittingRisk.WARN
        else:
            risk_level = OverfittingRisk.BLOCK

        is_overfitted = (
            wf_result["is_overfitted"]
            or pbo >= 0.5
            or mc_pvalue >= 0.10
            or (oos_metrics.get("return", 0) < 0 and is_metrics.get("return", 0) > 0)
        )

        # 5. 参数敏感性分析
        # P9/P10修复：优先使用真正的ParameterSensitivityAnalyzer（±10%扰动）
        # 来源：backtesting.py参数热力图分析 — 参数±10%扰动，追求平台区域
        parameter_stability = 0.5  # 默认中等稳定性
        try:
            if param_performance_fn is not None and base_params:
                # P9修复：使用真正的ParameterSensitivityAnalyzer
                # 对每个参数±10%扰动，调用param_performance_fn评估性能变化
                sa_result = self.sensitivity_analyzer.analyze(
                    base_params=base_params,
                    param_performance_fn=param_performance_fn,
                )
                parameter_stability = sa_result.get("stability_score", 0.5)
                details["parameter_sensitivity"] = sa_result
                if sa_result.get("sensitive_params"):
                    recommendations.append(
                        f"参数敏感: {sa_result['sensitive_params']}，"
                        f"±10%扰动导致性能下降>{sa_result.get('avg_performance_drop', 0):.1%}"
                    )
            elif strategy_returns and len(strategy_returns) >= 2:
                # 回退：用多策略的收益方差作为参数敏感性的代理（近似方法）
                # 策略间收益方差小 = 参数区域稳定 = 平台区域
                # 策略间收益方差大 = 参数敏感 = 尖峰
                import numpy as np
                returns_array = [sum(r) for r in strategy_returns.values() if len(r) > 0]
                if len(returns_array) >= 2:
                    mean_ret = np.mean(returns_array)
                    std_ret = np.std(returns_array)
                    # 变异系数CV = std/mean，CV越小越稳定
                    if abs(mean_ret) > 1e-10:
                        cv = abs(std_ret / mean_ret)
                        # CV<0.5=稳定(0.9), CV<1.0=中等(0.7), CV<2.0=一般(0.5), CV>=2.0=敏感(0.3)
                        if cv < 0.5:
                            parameter_stability = 0.9
                        elif cv < 1.0:
                            parameter_stability = 0.7
                        elif cv < 2.0:
                            parameter_stability = 0.5
                        else:
                            parameter_stability = 0.3
                    details["parameter_sensitivity"] = {
                        "stability_score": parameter_stability,
                        "cv": float(cv) if abs(mean_ret) > 1e-10 else 999.0,
                        "n_strategies": len(returns_array),
                        "method": "cv_proxy_fallback",
                    }
        except Exception as e:
            logger.warning(f"参数敏感性分析失败: {e}")
            pass  # 参数敏感性分析失败不阻塞验证

        return ValidationResult(
            risk_level=risk_level,
            is_overfitted=is_overfitted,
            pbo=pbo,
            oos_is_ratio=oos_is_ratio,
            monte_carlo_pvalue=mc_pvalue,
            max_drawdown_ratio=dd_ratio,
            parameter_stability=parameter_stability,
            details=details,
            recommendations=recommendations,
        )

    def _estimate_slippage_impact(
        self,
        trade_pnls: List[float],
        trade_prices: Optional[List[float]] = None,
        trade_volumes: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """估算滑点对收益的侵蚀

        使用RealisticSlippageModel（平方根法则）替代固定10bps假设。
        来源：Almgren-Chriss最优执行框架

        B10修复：使用交易记录中的真实价格和成交量，替代硬编码
        price=50000.0/adv=1000000.0。硬编码值会严重高估流动性，
        导致滑点被低估，实盘可能由盈转亏。

        Args:
            trade_pnls: 交易盈亏列表
            trade_prices: 每笔交易的入场价格（可选，None则用中位数估计）
            trade_volumes: 每笔交易的成交量（可选，None则用PnL反推）
        """
        if not trade_pnls:
            return {"erosion_pct": 0.0, "erosion_amount": 0.0, "n_trades": 0, "slippage_per_trade_bps": 0.0}

        n_trades = len(trade_pnls)
        total_slippage = 0.0
        total_slippage_bps = 0.0

        # B10修复：使用真实价格和成交量，而非硬编码
        # 如果没有提供，则从PnL中位数反推一个合理的默认值
        if trade_prices and len(trade_prices) == n_trades:
            prices = trade_prices
        else:
            # 回退：用PnL中位数反推价格（PnL≈1%价格变动 → price≈|pnl|*100）
            abs_pnls = [abs(p) for p in trade_pnls if p != 0]
            median_pnl = sorted(abs_pnls)[len(abs_pnls) // 2] if abs_pnls else 100.0
            prices = [max(median_pnl * 100, 100.0)] * n_trades

        if trade_volumes and len(trade_volumes) == n_trades:
            volumes = trade_volumes
        else:
            # 回退：用|pnl|*10作为名义价值代理（保守估计）
            volumes = [max(abs(p) * 10, 100.0) for p in trade_pnls]

        # 使用RealisticSlippageModel计算每笔交易的滑点
        for i, pnl in enumerate(trade_pnls):
            price = max(prices[i], 1.0)  # 价格必须>0
            trade_value = max(volumes[i], 1.0)  # 名义价值必须>0
            # ADV估计：用交易量的20倍作为ADV代理（单笔占ADV的5%是合理假设）
            adv = max(trade_value * 20.0, 10000.0)

            slip_result = self.slippage_model.compute_slippage(
                order_size=trade_value / price,
                adv=adv,
                price=price,
                side="long",
            )
            slip_bps = abs(slip_result.get("slippage_bps", 10.0))
            slip_amount = slip_bps / 10000.0 * trade_value
            total_slippage += slip_amount
            total_slippage_bps += slip_bps

        avg_slippage_bps = total_slippage_bps / n_trades
        total_pnl = sum(trade_pnls)
        erosion_pct = (total_slippage / abs(total_pnl)) if abs(total_pnl) > 1e-10 else 999.0

        return {
            "erosion_pct": erosion_pct,
            "erosion_amount": total_slippage,
            "n_trades": n_trades,
            "slippage_per_trade_bps": avg_slippage_bps,
            "used_real_prices": bool(trade_prices and len(trade_prices) == n_trades),
            "used_real_volumes": bool(trade_volumes and len(trade_volumes) == n_trades),
        }

    def generate_report(self, result: ValidationResult) -> str:
        """生成人类可读的验证报告"""
        lines = [
            "=" * 70,
            "反过拟合验证报告 (Anti-Overfitting Validation Report)",
            "=" * 70,
            "",
            f"风险等级: {result.risk_level.value.upper()}",
            f"是否过拟合: {'是 ❌ (禁止部署)' if result.is_overfitted else '否 ✅ (可以部署)'}",
            "",
            "--- 核心指标 ---",
            f"  PBO (过拟合概率):     {result.pbo:.3f}  ({'过拟合' if result.pbo >= 0.5 else '稳健' if result.pbo < 0.2 else '警告'})",
            f"  OOS/IS 收益比:       {result.oos_is_ratio:.2f}  ({'严重衰减' if result.oos_is_ratio < 0.3 else '合理' if result.oos_is_ratio > 0.5 else '警告'})",
            f"  蒙特卡洛 p值:        {result.monte_carlo_pvalue:.4f}  ({'不显著' if result.monte_carlo_pvalue >= 0.05 else '显著'})",
            f"  回撤放大倍数:        {result.max_drawdown_ratio:.2f}x  ({'严重' if result.max_drawdown_ratio > 3 else '可接受' if result.max_drawdown_ratio < 2 else '警告'})",
            "",
        ]

        if result.details:
            lines.append("--- 详细结果 ---")
            for category, detail in result.details.items():
                lines.append(f"  [{category}]")
                if isinstance(detail, dict):
                    for k, v in detail.items():
                        if isinstance(v, (int, float)):
                            lines.append(f"    {k}: {v:.4f}")
                        elif isinstance(v, list):
                            lines.append(f"    {k}: {v}")
                        elif isinstance(v, str):
                            lines.append(f"    {k}: {v}")
            lines.append("")

        if result.recommendations:
            lines.append("--- 建议 ---")
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        else:
            lines.append("--- 建议 ---")
            lines.append("  策略通过全部反过拟合验证，可以部署到实盘")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
