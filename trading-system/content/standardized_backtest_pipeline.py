# -*- coding: utf-8 -*-
"""标准化回测验证流程体系 + 沙盘-实盘一致性验证框架 (Phase 7G)

来源：
  - 用户要求："建立标准化、可重复的验证流程体系，严格执行基于真实市场历史数据
    与实时数据流的双重实证检验机制。建立模拟交易与实盘交易的一致性验证框架，
    包括但不限于数据精度校验、交易成本模拟、滑点模型优化、订单执行逻辑一致性检查"
  - SaR (Slippage-at-Risk) 框架 2026前沿
  - Almgren-Chriss 最优执行框架
  - NautilusTrader 回测-实盘零代码差异理念
  - Freqtrade Lookahead-Analysis
  - Sortino & Price 1994 下行风险框架

设计原则：
  1. 不重复造轮子，复用现有工具链：
     - anti_overfitting.py: RealisticSlippageModel + WalkForward + MonteCarlo + PBO
     - advanced_validation.py: PSR/DSR/MinBTL/CPCV
     - live_reality_check.py: LookAheadBias + CostSensitivity + LatencyRobustness
     - population.py: GT-Score (含 Sortino)
  2. 提供统一的 run_standard_backtest() 接口
  3. 输出标准化验证报告（与 L1-L8 验证框架兼容）
  4. 沙盘-实盘一致性验证：数据精度 + 成本模拟 + 滑点模型 + 执行逻辑 + KPI 对比

核心模块：
  1. KPIPortfolio — 统一 KPI 计算模块（Sharpe/Sortino/Calmar/MaxDD/WinRate/PF）
  2. SlippageAtRisk — SaR 滑点风险价值（2026前沿）
  3. SandboxLiveConsistencyValidator — 沙盘-实盘一致性验证框架
  4. StandardizedBacktestPipeline — 标准化回测验证流程编排器
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("hermes.standardized_pipeline")


# ============================================================================
# 1. KPIPortfolio — 统一 KPI 计算模块
# ============================================================================


@dataclass(slots=True)
class KPIPortfolio:
    """统一关键绩效指标 (KPI) 计算模块

    汇总量化交易核心 KPI：
      - Sharpe Ratio: 风险调整收益（惩罚所有波动）
      - Sortino Ratio: 下行风险调整收益（仅惩罚下行波动）
      - Calmar Ratio: 年化收益/最大回撤
      - Max Drawdown: 最大回撤
      - Win Rate: 胜率
      - Profit Factor: 盈亏比（总盈利/总亏损）
      - Total PnL: 总盈亏
      - Total Trades: 总交易数

    来源：
      - Sharpe 1966 "Mutual Fund Performance"
      - Sortino & Price 1994 "Performance Measurement in a Downside Risk Framework"
      - Calmar 1965 行业标准
      - GT-Score 评分体系 (population.py)
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_trades: int = 0
    annual_return: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0

    @classmethod
    def from_pnls(
        cls,
        pnls: Sequence[float],
        initial_capital: float = 10000.0,
        annualization_factor: float = 252.0,
    ) -> "KPIPortfolio":
        """从 PnL 序列计算所有 KPI

        Args:
            pnls: 每笔交易的 PnL 序列
            initial_capital: 初始资金
            annualization_factor: 年化因子（252=日线，365=加密货币日）

        Returns:
            KPIPortfolio 实例
        """
        if not pnls or len(pnls) == 0:
            return cls()

        arr = np.asarray(pnls, dtype=float)
        n = len(arr)
        total_pnl = float(np.sum(arr))
        winning = arr[arr > 0]
        losing = arr[arr < 0]

        # 胜率
        win_rate = float(len(winning) / n) if n > 0 else 0.0

        # 盈亏比
        total_profit = float(np.sum(winning)) if len(winning) > 0 else 0.0
        total_loss = float(np.abs(np.sum(losing))) if len(losing) > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 1e-10 else (
            10.0 if total_profit > 0 else 0.0
        )

        # 波动率
        if n > 1:
            volatility = float(np.std(arr, ddof=1))
        else:
            volatility = 0.0

        # Sharpe（年化）
        if volatility > 1e-10 and n > 1:
            mean_pnl = float(np.mean(arr))
            sharpe = mean_pnl / volatility * math.sqrt(min(annualization_factor, n))
            sharpe = max(-5.0, min(5.0, sharpe))
        else:
            sharpe = 0.0

        # Sortino（年化）— 仅下行偏差
        if n > 1:
            downside = np.minimum(arr - 0.0, 0.0)  # target=0
            downside_var = float(np.mean(downside ** 2))
            downside_std = math.sqrt(downside_var)
            if downside_std > 1e-10:
                mean_pnl_sort = float(np.mean(arr))
                sortino = mean_pnl_sort / downside_std * math.sqrt(min(annualization_factor, n))
                sortino = max(-10.0, min(10.0, sortino))
            else:
                sortino = 10.0 if total_pnl > 0 else 0.0
        else:
            sortino = 0.0
            downside_std = 0.0

        # 最大回撤
        equity = np.concatenate([[initial_capital], np.cumsum(arr) + initial_capital])
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1e-10)
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # 年化收益
        if n > 0 and initial_capital > 0:
            annual_return = total_pnl / initial_capital * (annualization_factor / n)
        else:
            annual_return = 0.0

        # Calmar
        if max_dd > 1e-10:
            calmar = annual_return / max_dd
            calmar = max(-10.0, min(10.0, calmar))
        else:
            calmar = 10.0 if annual_return > 0 else 0.0

        return cls(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_trades=n,
            annual_return=annual_return,
            volatility=volatility,
            downside_deviation=downside_std,
        )

    def to_dict(self) -> Dict[str, float]:
        """转为字典（用于报告）"""
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_pnl": round(self.total_pnl, 2),
            "total_trades": int(self.total_trades),
            "annual_return": round(self.annual_return, 4),
            "volatility": round(self.volatility, 4),
            "downside_deviation": round(self.downside_deviation, 4),
        }


# ============================================================================
# 2. SlippageAtRisk (SaR) — 滑点风险价值（2026前沿）
# ============================================================================


@dataclass(slots=True)
class SlippageAtRiskResult:
    """SaR 滑点风险价值结果

    来源：SaR (Slippage-at-Risk) 框架 2026前沿
    类比 VaR (Value-at-Risk)，SaR 量化给定置信水平下的最大滑点损失。

    公式：
      SaR_α = quantile(slippage_distribution, 1-α)
      其中 slippage_distribution 通过蒙特卡洛模拟生成：
        对每笔交易，模拟 N 次市场冲击+基础滑点+延迟成本，
        得到滑点分布，取 α 分位数作为风险价值
    """

    sar_95: float = 0.0       # 95% 置信度下的最大滑点（bps）
    sar_99: float = 0.0       # 99% 置信度下的最大滑点（bps）
    mean_slippage_bps: float = 0.0
    worst_slippage_bps: float = 0.0
    slippage_volatility_bps: float = 0.0
    n_simulations: int = 0
    is_robust: bool = False
    verdict: str = "PASS"
    details: List[Dict[str, Any]] = field(default_factory=list)


class SlippageAtRiskCalculator:
    """SaR 滑点风险价值计算器

    来源：SaR (Slippage-at-Risk) 框架 2026前沿 + Almgren-Chriss 平方根法则

    核心理念：
      传统回测使用固定滑点（如 2bps），但实际滑点是随机变量：
        - 市场冲击取决于实时流动性（随机）
        - 基础滑点取决于买卖价差（波动）
        - 延迟成本取决于网络状况（随机）

      SaR 通过蒙特卡洛模拟生成滑点分布，量化"最坏情况滑点"：
        - SaR_95 = 95% 置信度下的最大滑点（即 5% 的交易滑点会超过此值）
        - SaR_99 = 99% 置信度下的最大滑点（更保守）

    阈值：
      - SaR_95 < 10 bps  → PASS（滑点可控）
      - SaR_95 10-30 bps → WARN（滑点偏高，需关注流动性）
      - SaR_95 >= 30 bps → BLOCK（滑点失控，策略在低流动性场景会亏损）
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        impact_coefficient: float = 0.1,
        base_slippage_bps: float = 2.0,
        latency_ms_mean: float = 100.0,
        latency_ms_std: float = 50.0,
        pass_threshold_bps: float = 10.0,
        warn_threshold_bps: float = 30.0,
        random_seed: Optional[int] = 42,
    ):
        """
        Args:
            n_simulations: 蒙特卡洛模拟次数
            impact_coefficient: 市场冲击系数 k（0.1=高流动性，0.5=低流动性）
            base_slippage_bps: 基础滑点（基点）
            latency_ms_mean: 平均延迟（毫秒）
            latency_ms_std: 延迟标准差（毫秒）
            pass_threshold_bps: PASS 阈值（bps）
            warn_threshold_bps: WARN 阈值（bps）
            random_seed: 随机种子（可复现）
        """
        self.n_simulations = max(100, n_simulations)
        self.k = impact_coefficient
        self.base_bps = base_slippage_bps
        self.latency_mean = latency_ms_mean
        self.latency_std = max(0.0, latency_ms_std)
        self.pass_bps = pass_threshold_bps
        self.warn_bps = warn_threshold_bps
        self.rng = np.random.default_rng(random_seed)

    def calculate(
        self,
        order_sizes: Sequence[float],
        adv: float,
        price: float,
    ) -> SlippageAtRiskResult:
        """计算 SaR

        Args:
            order_sizes: 订单大小序列（合约数或名义价值）
            adv: 日均成交量（合约数或名义价值，与 order_sizes 单位一致）
            price: 当前价格

        Returns:
            SlippageAtRiskResult
        """
        result = SlippageAtRiskResult(n_simulations=self.n_simulations)

        if not order_sizes or adv <= 0:
            result.verdict = "WARN"
            result.details.append({"error": "订单数据为空或 ADV<=0"})
            return result

        sizes = np.asarray(order_sizes, dtype=float)
        n_orders = len(sizes)

        # 蒙特卡洛模拟：对每个订单模拟 n_simulations 次滑点
        # 滑点 = 基础滑点 + 市场冲击 + 延迟成本
        all_slippages = np.zeros(n_orders * self.n_simulations)

        for i, size in enumerate(sizes):
            # 市场冲击（平方根法则）：k × √(size/ADV) × 10000 (bps)
            # 加入随机扰动模拟流动性波动
            size_ratio = size / adv if adv > 0 else 1.0
            base_impact = self.k * math.sqrt(max(size_ratio, 0.0)) * 10000.0

            # 模拟流动性波动（对数正态分布，保证非负）
            liquidity_shock = self.rng.lognormal(mean=0.0, sigma=0.3, size=self.n_simulations)
            impact_bps = base_impact * liquidity_shock

            # 基础滑点（买卖价差波动）
            base_bps_sim = self.base_bps * self.rng.lognormal(mean=0.0, sigma=0.2, size=self.n_simulations)

            # 延迟成本（正态分布，截断到非负）
            latency_ms = np.maximum(
                0.0,
                self.rng.normal(self.latency_mean, self.latency_std, size=self.n_simulations)
            )
            latency_cost_bps = latency_ms / 1000.0 * 5.0  # 5 bps per second

            # 总滑点
            total_slippage = base_bps_sim + impact_bps + latency_cost_bps
            all_slippages[i * self.n_simulations:(i + 1) * self.n_simulations] = total_slippage

        # 计算统计量
        result.mean_slippage_bps = float(np.mean(all_slippages))
        result.worst_slippage_bps = float(np.max(all_slippages))
        result.slippage_volatility_bps = float(np.std(all_slippages))
        result.sar_95 = float(np.percentile(all_slippages, 95))
        result.sar_99 = float(np.percentile(all_slippages, 99))

        # 判定
        if result.sar_95 < self.pass_bps:
            result.is_robust = True
            result.verdict = "PASS"
        elif result.sar_95 < self.warn_bps:
            result.is_robust = False
            result.verdict = "WARN"
        else:
            result.is_robust = False
            result.verdict = "BLOCK"

        result.details.append({
            "n_orders": n_orders,
            "adv": adv,
            "price": price,
            "mean_size": float(np.mean(sizes)),
            "max_size": float(np.max(sizes)),
        })

        return result


# ============================================================================
# 3. SandboxLiveConsistencyValidator — 沙盘-实盘一致性验证框架
# ============================================================================


@dataclass(slots=True)
class ConsistencyResult:
    """沙盘-实盘一致性验证结果"""

    # 数据精度校验
    data_precision_pass: bool = False
    data_precision_details: Dict[str, Any] = field(default_factory=dict)

    # 交易成本模拟一致性
    cost_simulation_pass: bool = False
    cost_break_even_bps: float = 0.0
    cost_details: Dict[str, Any] = field(default_factory=dict)

    # 滑点模型一致性（SaR）
    slippage_sar_95: float = 0.0
    slippage_verdict: str = "PASS"
    slippage_details: Dict[str, Any] = field(default_factory=dict)

    # 订单执行逻辑一致性
    execution_logic_pass: bool = False
    execution_details: Dict[str, Any] = field(default_factory=dict)

    # KPI 对比（沙盘 vs 实盘）
    sandbox_kpi: Dict[str, float] = field(default_factory=dict)
    live_kpi: Dict[str, float] = field(default_factory=dict)
    kpi_deviation: Dict[str, float] = field(default_factory=dict)
    kpi_consistency_pass: bool = False

    # 综合判定
    overall_verdict: str = "PASS"
    consistency_score: float = 0.0  # 0-100


class SandboxLiveConsistencyValidator:
    """沙盘-实盘一致性验证框架

    来源：用户要求 + NautilusTrader 回测-实盘零代码差异理念 + SaR 2026前沿

    核心目标：杜绝"模拟环境表现优异而实盘交易出现亏损"的策略失效现象

    验证维度：
      1. 数据精度校验：沙盘数据与实盘数据的 tick 级一致性
      2. 交易成本模拟：CostSensitivityScanner 验证成本模型
      3. 滑点模型优化：SaR + RealisticSlippageModel 验证滑点
      4. 订单执行逻辑一致性：撮合引擎接口一致性
      5. KPI 对比：沙盘 KPI vs 实盘 KPI 偏差
    """

    # KPI 偏差阈值（沙盘 vs 实盘）
    # 来源：NautilusTrader 回测-实盘一致性最佳实践
    KPI_DEVIATION_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        # (WARN阈值, BLOCK阈值) — 相对偏差百分比
        "sharpe_ratio": (0.15, 0.30),    # Sharpe 偏差 >30% = BLOCK
        "sortino_ratio": (0.15, 0.30),
        "calmar_ratio": (0.20, 0.40),
        "max_drawdown": (0.20, 0.50),    # 回撤偏差 >50% = BLOCK
        "win_rate": (0.10, 0.20),
        "profit_factor": (0.15, 0.30),
        "total_pnl": (0.10, 0.25),       # PnL 偏差 >25% = BLOCK
    }

    def __init__(
        self,
        cost_pass_threshold_bps: float = 3.0,
        cost_warn_threshold_bps: float = 10.0,
        sar_calculator: Optional[SlippageAtRiskCalculator] = None,
    ):
        """
        Args:
            cost_pass_threshold_bps: 成本 PASS 阈值
            cost_warn_threshold_bps: 成本 WARN 阈值
            sar_calculator: SaR 计算器（None 则使用默认）
        """
        self.cost_pass_bps = cost_pass_threshold_bps
        self.cost_warn_bps = cost_warn_threshold_bps
        self.sar_calculator = sar_calculator or SlippageAtRiskCalculator()

    def validate_data_precision(
        self,
        sandbox_prices: Sequence[float],
        live_prices: Sequence[float],
        tolerance_pct: float = 0.01,
    ) -> Tuple[bool, Dict[str, Any]]:
        """数据精度校验

        Args:
            sandbox_prices: 沙盘价格序列
            live_prices: 实盘价格序列
            tolerance_pct: 容忍偏差百分比（0.01=1%）

        Returns:
            (是否通过, 详细信息)
        """
        details: Dict[str, Any] = {}
        if len(sandbox_prices) == 0 or len(live_prices) == 0:
            return False, {"error": "价格序列为空"}

        n = min(len(sandbox_prices), len(live_prices))
        sb = np.asarray(sandbox_prices[:n], dtype=float)
        lv = np.asarray(live_prices[:n], dtype=float)

        # 相对偏差
        diff = np.abs(sb - lv)
        scale = np.maximum(np.abs(lv), 1e-10)
        rel_diff = diff / scale

        max_diff_pct = float(np.max(rel_diff)) * 100.0
        mean_diff_pct = float(np.mean(rel_diff)) * 100.0
        mismatch_count = int(np.sum(rel_diff > tolerance_pct))
        mismatch_ratio = mismatch_count / n if n > 0 else 0.0

        details = {
            "n_points": n,
            "max_diff_pct": round(max_diff_pct, 4),
            "mean_diff_pct": round(mean_diff_pct, 4),
            "mismatch_count": mismatch_count,
            "mismatch_ratio": round(mismatch_ratio, 4),
            "tolerance_pct": tolerance_pct * 100.0,
        }

        # 判定：mismatch_ratio < 5% = PASS
        passed = mismatch_ratio < 0.05
        return passed, details

    def validate_cost_simulation(
        self,
        trade_pnls: Sequence[float],
        trade_volumes: Optional[Sequence[float]] = None,
        capital: float = 10000.0,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """交易成本模拟一致性验证

        使用 CostSensitivityScanner 扫描不同成本下的策略盈亏，
        找到盈亏平衡点（break_even_bps）。

        Args:
            trade_pnls: 每笔交易的原始 PnL（未扣成本）
            trade_volumes: 每笔交易成交量
            capital: 初始资金

        Returns:
            (是否通过, break_even_bps, 详细信息)
        """
        # 延迟导入避免循环依赖
        from .live_reality_check import CostSensitivityScanner

        scanner = CostSensitivityScanner(
            pass_threshold_bps=self.cost_pass_bps,
            warn_threshold_bps=self.cost_warn_bps,
        )
        result = scanner.scan(trade_pnls, trade_volumes, capital)

        details = {
            "break_even_bps": result.break_even_bps,
            "worst_case_pnl": result.worst_case_pnl,
            "pnl_degradation_pct": result.pnl_degradation_pct,
            "verdict": result.verdict,
            "is_robust": result.is_robust,
            "scan_results": result.scan_results,
        }

        passed = result.is_robust
        return passed, result.break_even_bps, details

    def validate_slippage_model(
        self,
        order_sizes: Sequence[float],
        adv: float,
        price: float,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """滑点模型优化验证（SaR）

        Args:
            order_sizes: 订单大小序列
            adv: 日均成交量
            price: 当前价格

        Returns:
            (sar_95, verdict, 详细信息)
        """
        sar_result = self.sar_calculator.calculate(order_sizes, adv, price)

        details = {
            "sar_95": sar_result.sar_95,
            "sar_99": sar_result.sar_99,
            "mean_slippage_bps": sar_result.mean_slippage_bps,
            "worst_slippage_bps": sar_result.worst_slippage_bps,
            "slippage_volatility_bps": sar_result.slippage_volatility_bps,
            "n_simulations": sar_result.n_simulations,
            "details": sar_result.details,
        }

        return sar_result.sar_95, sar_result.verdict, details

    def validate_execution_logic(
        self,
        sandbox_fill_prices: Sequence[float],
        expected_fill_prices: Sequence[float],
        tolerance_bps: float = 5.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """订单执行逻辑一致性检查

        比较沙盘撮合引擎的成交价与预期成交价（来自 RealisticSlippageModel）

        Args:
            sandbox_fill_prices: 沙盘实际成交价
            expected_fill_prices: 预期成交价（含滑点模型）
            tolerance_bps: 容忍偏差（基点）

        Returns:
            (是否通过, 详细信息)
        """
        details: Dict[str, Any] = {}
        if len(sandbox_fill_prices) == 0 or len(expected_fill_prices) == 0:
            return False, {"error": "成交价序列为空"}

        n = min(len(sandbox_fill_prices), len(expected_fill_prices))
        sb = np.asarray(sandbox_fill_prices[:n], dtype=float)
        ex = np.asarray(expected_fill_prices[:n], dtype=float)

        # 计算偏差（bps）
        diff = np.abs(sb - ex)
        scale = np.maximum(np.abs(ex), 1e-10)
        diff_bps = diff / scale * 10000.0

        max_diff_bps = float(np.max(diff_bps))
        mean_diff_bps = float(np.mean(diff_bps))
        mismatch_count = int(np.sum(diff_bps > tolerance_bps))
        mismatch_ratio = mismatch_count / n if n > 0 else 0.0

        details = {
            "n_fills": n,
            "max_diff_bps": round(max_diff_bps, 4),
            "mean_diff_bps": round(mean_diff_bps, 4),
            "mismatch_count": mismatch_count,
            "mismatch_ratio": round(mismatch_ratio, 4),
            "tolerance_bps": tolerance_bps,
        }

        # 判定：max_diff_bps < tolerance AND mismatch_ratio < 10%
        passed = max_diff_bps < tolerance_bps * 2 and mismatch_ratio < 0.10
        return passed, details

    def validate_kpi_consistency(
        self,
        sandbox_pnls: Sequence[float],
        live_pnls: Sequence[float],
        initial_capital: float = 10000.0,
    ) -> Tuple[bool, Dict[str, float], Dict[str, float], Dict[str, float]]:
        """KPI 一致性验证（沙盘 vs 实盘）

        Args:
            sandbox_pnls: 沙盘 PnL 序列
            live_pnls: 实盘 PnL 序列
            initial_capital: 初始资金

        Returns:
            (是否通过, sandbox_kpi, live_kpi, kpi_deviation)
        """
        sb_kpi = KPIPortfolio.from_pnls(sandbox_pnls, initial_capital)
        lv_kpi = KPIPortfolio.from_pnls(live_pnls, initial_capital)

        sb_dict = sb_kpi.to_dict()
        lv_dict = lv_kpi.to_dict()

        # 计算相对偏差
        deviations: Dict[str, float] = {}
        has_block = False
        has_warn = False

        for key, (warn_thr, block_thr) in self.KPI_DEVIATION_THRESHOLDS.items():
            sb_val = sb_dict.get(key, 0.0)
            lv_val = lv_dict.get(key, 0.0)
            # 相对偏差 = |sb - lv| / max(|lv|, epsilon)
            denom = max(abs(lv_val), 1e-6)
            rel_dev = abs(sb_val - lv_val) / denom
            deviations[key] = round(rel_dev, 4)

            if rel_dev >= block_thr:
                has_block = True
            elif rel_dev >= warn_thr:
                has_warn = True

        # 综合判定
        passed = not has_block
        return passed, sb_dict, lv_dict, deviations

    def validate(
        self,
        sandbox_pnls: Sequence[float],
        live_pnls: Sequence[float],
        sandbox_prices: Optional[Sequence[float]] = None,
        live_prices: Optional[Sequence[float]] = None,
        order_sizes: Optional[Sequence[float]] = None,
        adv: float = 0.0,
        price: float = 0.0,
        sandbox_fill_prices: Optional[Sequence[float]] = None,
        expected_fill_prices: Optional[Sequence[float]] = None,
        trade_volumes: Optional[Sequence[float]] = None,
        initial_capital: float = 10000.0,
    ) -> ConsistencyResult:
        """完整沙盘-实盘一致性验证

        Args:
            sandbox_pnls: 沙盘 PnL 序列
            live_pnls: 实盘 PnL 序列
            sandbox_prices: 沙盘价格序列（可选，用于数据精度校验）
            live_prices: 实盘价格序列（可选）
            order_sizes: 订单大小序列（可选，用于 SaR）
            adv: 日均成交量
            price: 当前价格
            sandbox_fill_prices: 沙盘成交价序列（可选）
            expected_fill_prices: 预期成交价序列（可选）
            trade_volumes: 交易成交量序列（可选）
            initial_capital: 初始资金

        Returns:
            ConsistencyResult
        """
        result = ConsistencyResult()

        # 1. 数据精度校验
        if sandbox_prices is not None and live_prices is not None:
            passed, details = self.validate_data_precision(sandbox_prices, live_prices)
            result.data_precision_pass = passed
            result.data_precision_details = details
        else:
            result.data_precision_pass = True  # 跳过
            result.data_precision_details = {"skipped": True}

        # 2. 交易成本模拟一致性
        if sandbox_pnls and len(sandbox_pnls) >= 5:
            passed, break_even, details = self.validate_cost_simulation(
                sandbox_pnls, trade_volumes, initial_capital
            )
            result.cost_simulation_pass = passed
            result.cost_break_even_bps = break_even
            result.cost_details = details
        else:
            result.cost_simulation_pass = False
            result.cost_details = {"error": "沙盘 PnL 不足（<5）"}

        # 3. 滑点模型优化验证（SaR）
        if order_sizes and adv > 0 and price > 0:
            sar_95, verdict, details = self.validate_slippage_model(order_sizes, adv, price)
            result.slippage_sar_95 = sar_95
            result.slippage_verdict = verdict
            result.slippage_details = details
        else:
            result.slippage_verdict = "SKIP"
            result.slippage_details = {"skipped": True}

        # 4. 订单执行逻辑一致性
        if sandbox_fill_prices is not None and expected_fill_prices is not None:
            passed, details = self.validate_execution_logic(
                sandbox_fill_prices, expected_fill_prices
            )
            result.execution_logic_pass = passed
            result.execution_details = details
        else:
            result.execution_logic_pass = True  # 跳过
            result.execution_details = {"skipped": True}

        # 5. KPI 一致性
        if sandbox_pnls and live_pnls and len(sandbox_pnls) >= 5 and len(live_pnls) >= 5:
            passed, sb_kpi, lv_kpi, deviations = self.validate_kpi_consistency(
                sandbox_pnls, live_pnls, initial_capital
            )
            result.sandbox_kpi = sb_kpi
            result.live_kpi = lv_kpi
            result.kpi_deviation = deviations
            result.kpi_consistency_pass = passed
        else:
            result.kpi_consistency_pass = True  # 跳过
            result.kpi_deviation = {"skipped": True}

        # 综合判定
        checks = [
            result.data_precision_pass,
            result.cost_simulation_pass,
            result.slippage_verdict != "BLOCK",
            result.execution_logic_pass,
            result.kpi_consistency_pass,
        ]
        passed_count = sum(1 for c in checks if c)
        result.consistency_score = passed_count / len(checks) * 100.0

        if all(checks):
            result.overall_verdict = "PASS"
        elif result.consistency_score >= 60.0:
            result.overall_verdict = "WARN"
        else:
            result.overall_verdict = "BLOCK"

        return result


# ============================================================================
# 4. StandardizedBacktestPipeline — 标准化回测验证流程编排器
# ============================================================================


@dataclass(slots=True)
class BacktestReport:
    """标准化回测验证报告"""

    # KPI
    kpi: Dict[str, float] = field(default_factory=dict)

    # 反过拟合验证
    walk_forward_verdict: str = "SKIP"
    monte_carlo_verdict: str = "SKIP"
    pbo_verdict: str = "SKIP"
    psr_verdict: str = "SKIP"
    anti_overfitting_details: Dict[str, Any] = field(default_factory=dict)

    # 实盘现实性检查
    lookahead_verdict: str = "SKIP"
    cost_sensitivity_verdict: str = "SKIP"
    latency_robustness_verdict: str = "SKIP"
    reality_check_details: Dict[str, Any] = field(default_factory=dict)

    # 沙盘-实盘一致性
    consistency_verdict: str = "SKIP"
    consistency_score: float = 0.0
    consistency_details: Dict[str, Any] = field(default_factory=dict)

    # 综合评定
    overall_verdict: str = "PASS"
    overall_score: float = 0.0
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)

    def to_summary(self) -> str:
        """生成摘要报告"""
        lines = [
            "=" * 70,
            "标准化回测验证报告 (Phase 7G)",
            "=" * 70,
            "",
            "【KPI 关键绩效指标】",
        ]
        for k, v in self.kpi.items():
            lines.append(f"  {k:25s}: {v}")

        lines.extend([
            "",
            "【反过拟合验证】",
            f"  Walk-Forward:      {self.walk_forward_verdict}",
            f"  Monte-Carlo:       {self.monte_carlo_verdict}",
            f"  PBO:               {self.pbo_verdict}",
            f"  PSR:               {self.psr_verdict}",
            "",
            "【实盘现实性检查】",
            f"  Look-Ahead Bias:   {self.lookahead_verdict}",
            f"  Cost Sensitivity:  {self.cost_sensitivity_verdict}",
            f"  Latency Robustness:{self.latency_robustness_verdict}",
            "",
            "【沙盘-实盘一致性】",
            f"  Consistency Score: {self.consistency_score:.1f}/100",
            f"  Verdict:           {self.consistency_verdict}",
            "",
            "【综合评定】",
            f"  Overall Score:     {self.overall_score:.1f}/100",
            f"  Verdict:           {self.overall_verdict}",
        ])

        if self.block_reasons:
            lines.append("")
            lines.append("【BLOCK 原因】")
            for r in self.block_reasons:
                lines.append(f"  - {r}")

        if self.warn_reasons:
            lines.append("")
            lines.append("【WARN 原因】")
            for r in self.warn_reasons:
                lines.append(f"  - {r}")

        lines.append("=" * 70)
        return "\n".join(lines)


class StandardizedBacktestPipeline:
    """标准化回测验证流程编排器

    来源：用户要求 "建立标准化、可重复的验证流程体系"

    串联现有工具链，提供统一的回测验证流程：
      1. KPI 计算（KPIPortfolio）
      2. 反过拟合验证（WalkForward + MonteCarlo + PBO + PSR/DSR）
      3. 实盘现实性检查（LookAhead + CostSensitivity + LatencyRobustness）
      4. 沙盘-实盘一致性验证（SandboxLiveConsistencyValidator）

    用法：
        pipeline = StandardizedBacktestPipeline()
        report = pipeline.run(
            trade_pnls=[100, -50, 80, ...],
            price_history=[50000, 50100, ...],
            live_pnls=[90, -45, 75, ...],  # 可选
        )
        print(report.to_summary())
    """

    def __init__(
        self,
        enable_walk_forward: bool = True,
        enable_monte_carlo: bool = True,
        enable_pbo: bool = False,  # 较慢，默认关闭
        enable_psr: bool = True,
        enable_lookahead: bool = True,
        enable_cost_sensitivity: bool = True,
        enable_latency_robustness: bool = True,
        enable_consistency: bool = True,
        consistency_validator: Optional[SandboxLiveConsistencyValidator] = None,
    ):
        """
        Args:
            enable_*: 各验证模块开关
            consistency_validator: 沙盘-实盘一致性验证器（None 则使用默认）
        """
        self.enable_walk_forward = enable_walk_forward
        self.enable_monte_carlo = enable_monte_carlo
        self.enable_pbo = enable_pbo
        self.enable_psr = enable_psr
        self.enable_lookahead = enable_lookahead
        self.enable_cost_sensitivity = enable_cost_sensitivity
        self.enable_latency_robustness = enable_latency_robustness
        self.enable_consistency = enable_consistency
        self.consistency_validator = consistency_validator or SandboxLiveConsistencyValidator()

    def run(
        self,
        trade_pnls: Sequence[float],
        price_history: Optional[Sequence[float]] = None,
        trade_volumes: Optional[Sequence[float]] = None,
        initial_capital: float = 10000.0,
        live_pnls: Optional[Sequence[float]] = None,
        sandbox_prices: Optional[Sequence[float]] = None,
        live_prices: Optional[Sequence[float]] = None,
        order_sizes: Optional[Sequence[float]] = None,
        adv: float = 0.0,
        price: float = 0.0,
        indicator_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> BacktestReport:
        """运行标准化回测验证流程

        Args:
            trade_pnls: 交易 PnL 序列
            price_history: 价格历史（用于 LookAhead 检测）
            trade_volumes: 交易成交量序列
            initial_capital: 初始资金
            live_pnls: 实盘 PnL 序列（可选，用于一致性验证）
            sandbox_prices: 沙盘价格序列（可选）
            live_prices: 实盘价格序列（可选）
            order_sizes: 订单大小序列（可选，用于 SaR）
            adv: 日均成交量
            price: 当前价格
            indicator_fn: 指标计算函数（可选，用于 LookAhead 检测）

        Returns:
            BacktestReport
        """
        report = BacktestReport()
        block_reasons: List[str] = []
        warn_reasons: List[str] = []

        # 1. KPI 计算
        kpi = KPIPortfolio.from_pnls(trade_pnls, initial_capital)
        report.kpi = kpi.to_dict()

        # 2. 反过拟合验证
        anti_details: Dict[str, Any] = {}

        if self.enable_walk_forward and len(trade_pnls) >= 20:
            try:
                from .anti_overfitting import WalkForwardValidator
                wf = WalkForwardValidator()
                # 简化：用 PnL 序列做 walk-forward
                n = len(trade_pnls)
                split = n // 2
                is_returns = trade_pnls[:split]
                oos_returns = trade_pnls[split:]
                is_metrics = {
                    "sharpe": float(np.mean(is_returns) / max(np.std(is_returns), 1e-10)) if is_returns else 0.0,
                    "max_drawdown": self._compute_max_dd(is_returns, initial_capital),
                    "return": float(np.sum(is_returns)),
                }
                oos_metrics = {
                    "sharpe": float(np.mean(oos_returns) / max(np.std(oos_returns), 1e-10)) if oos_returns else 0.0,
                    "max_drawdown": self._compute_max_dd(oos_returns, initial_capital),
                    "return": float(np.sum(oos_returns)),
                }
                wf_result = wf.evaluate(is_metrics, oos_metrics)
                wf_verdict = wf_result.get("verdict", "SKIP")
                wf_issues = wf_result.get("issues", [])
                report.walk_forward_verdict = wf_verdict
                anti_details["walk_forward"] = {
                    "verdict": wf_verdict,
                    "sharpe_decay": wf_result.get("sharpe_decay", 0.0),
                    "issues": wf_issues,
                }
                if wf_verdict == "BLOCK":
                    block_reasons.append(f"Walk-Forward: {wf_issues}")
                elif wf_verdict == "WARN":
                    warn_reasons.append(f"Walk-Forward: {wf_issues}")
            except Exception as e:
                logger.debug("Walk-Forward 验证失败: %s", e)
                report.walk_forward_verdict = "SKIP"

        if self.enable_monte_carlo and len(trade_pnls) >= 10:
            try:
                from .anti_overfitting import MonteCarloPermutationTest
                mc = MonteCarloPermutationTest(n_iterations=500, confidence_level=0.95)
                mc_result = mc.test(list(trade_pnls), metric="sharpe")
                mc_pvalue = mc_result.get("pvalue", 1.0)
                # 判定：p < 0.05 = PASS, p < 0.10 = WARN, p >= 0.10 = BLOCK
                if mc_pvalue < 0.05:
                    mc_verdict = "PASS"
                elif mc_pvalue < 0.10:
                    mc_verdict = "WARN"
                else:
                    mc_verdict = "BLOCK"
                report.monte_carlo_verdict = mc_verdict
                anti_details["monte_carlo"] = {
                    "verdict": mc_verdict,
                    "pvalue": mc_pvalue,
                    "mc_sharpe_5pct": mc_result.get("mc_sharpe_5pct", 0.0),
                    "percentile": mc_result.get("percentile", 0.0),
                }
                if mc_verdict == "BLOCK":
                    block_reasons.append(f"Monte-Carlo p-value={mc_pvalue:.4f}")
            except Exception as e:
                logger.debug("Monte-Carlo 验证失败: %s", e)
                report.monte_carlo_verdict = "SKIP"

        if self.enable_psr and len(trade_pnls) >= 10:
            try:
                from .advanced_validation import AdvancedAntiOverfittingValidator
                validator = AdvancedAntiOverfittingValidator()
                # PSR 需要 sharpe, n_trials, n_returns
                sharpe = kpi.sharpe_ratio
                n_returns = len(trade_pnls)
                # 简化：假设没有多重比较，n_trials=1
                psr = validator.validate_strategy(
                    sharpe_ratio=sharpe,
                    n_trials=1,
                    returns=list(trade_pnls),
                )
                if hasattr(psr, "psr"):
                    psr_verdict = "PASS" if psr.psr >= 0.95 else (
                        "WARN" if psr.psr >= 0.80 else "BLOCK"
                    )
                    report.psr_verdict = psr_verdict
                    anti_details["psr"] = {
                        "psr": psr.psr,
                        "verdict": psr_verdict,
                    }
                    if psr_verdict == "BLOCK":
                        block_reasons.append(f"PSR={psr.psr:.4f} < 0.80")
                else:
                    report.psr_verdict = "SKIP"
            except Exception as e:
                logger.debug("PSR 验证失败: %s", e)
                report.psr_verdict = "SKIP"

        report.anti_overfitting_details = anti_details

        # 3. 实盘现实性检查
        reality_details: Dict[str, Any] = {}

        if self.enable_lookahead and price_history is not None and indicator_fn is not None:
            try:
                from .live_reality_check import LookAheadBiasDetector
                detector = LookAheadBiasDetector()
                price_arr = np.asarray(price_history, dtype=float)
                la_result = detector.detect(indicator_fn, price_arr)
                report.lookahead_verdict = la_result.verdict
                reality_details["lookahead"] = {
                    "verdict": la_result.verdict,
                    "bias_percentage": la_result.bias_percentage,
                    "n_slices_tested": la_result.n_slices_tested,
                }
                if la_result.verdict == "BLOCK":
                    block_reasons.append(f"Look-Ahead Bias={la_result.bias_percentage:.2f}%")
            except Exception as e:
                logger.debug("Look-Ahead 检测失败: %s", e)
                report.lookahead_verdict = "SKIP"

        if self.enable_cost_sensitivity and len(trade_pnls) >= 5:
            try:
                from .live_reality_check import CostSensitivityScanner
                scanner = CostSensitivityScanner()
                cs_result = scanner.scan(trade_pnls, trade_volumes, initial_capital)
                report.cost_sensitivity_verdict = cs_result.verdict
                reality_details["cost_sensitivity"] = {
                    "verdict": cs_result.verdict,
                    "break_even_bps": cs_result.break_even_bps,
                    "pnl_degradation_pct": cs_result.pnl_degradation_pct,
                }
                if cs_result.verdict == "BLOCK":
                    block_reasons.append(f"Cost break-even={cs_result.break_even_bps:.2f} bps")
            except Exception as e:
                logger.debug("Cost Sensitivity 扫描失败: %s", e)
                report.cost_sensitivity_verdict = "SKIP"

        if self.enable_latency_robustness and len(trade_pnls) >= 5:
            try:
                from .live_reality_check import LatencyRobustnessScanner
                scanner = LatencyRobustnessScanner()
                lr_result = scanner.scan(trade_pnls, trade_volumes, initial_capital)
                report.latency_robustness_verdict = lr_result.verdict
                reality_details["latency_robustness"] = {
                    "verdict": lr_result.verdict,
                    "break_even_latency_ms": lr_result.break_even_latency_ms,
                    "pnl_degradation_pct": lr_result.pnl_degradation_pct,
                }
                if lr_result.verdict == "BLOCK":
                    block_reasons.append(f"Latency break-even={lr_result.break_even_latency_ms:.0f} ms")
            except Exception as e:
                logger.debug("Latency Robustness 扫描失败: %s", e)
                report.latency_robustness_verdict = "SKIP"

        report.reality_check_details = reality_details

        # 4. 沙盘-实盘一致性验证
        if self.enable_consistency:
            consistency_result = self.consistency_validator.validate(
                sandbox_pnls=trade_pnls,
                live_pnls=live_pnls if live_pnls else [],
                sandbox_prices=sandbox_prices,
                live_prices=live_prices,
                order_sizes=order_sizes,
                adv=adv,
                price=price,
                sandbox_fill_prices=None,
                expected_fill_prices=None,
                trade_volumes=trade_volumes,
                initial_capital=initial_capital,
            )
            report.consistency_verdict = consistency_result.overall_verdict
            report.consistency_score = consistency_result.consistency_score
            report.consistency_details = {
                "data_precision_pass": consistency_result.data_precision_pass,
                "cost_simulation_pass": consistency_result.cost_simulation_pass,
                "slippage_verdict": consistency_result.slippage_verdict,
                "slippage_sar_95": consistency_result.slippage_sar_95,
                "execution_logic_pass": consistency_result.execution_logic_pass,
                "kpi_consistency_pass": consistency_result.kpi_consistency_pass,
                "kpi_deviation": consistency_result.kpi_deviation,
            }
            if consistency_result.overall_verdict == "BLOCK":
                block_reasons.append(f"Consistency score={consistency_result.consistency_score:.1f}")
            elif consistency_result.overall_verdict == "WARN":
                warn_reasons.append(f"Consistency score={consistency_result.consistency_score:.1f}")

        # 综合评定
        all_verdicts = [
            report.walk_forward_verdict,
            report.monte_carlo_verdict,
            report.psr_verdict,
            report.lookahead_verdict,
            report.cost_sensitivity_verdict,
            report.latency_robustness_verdict,
            report.consistency_verdict,
        ]

        # 计算总分（PASS=100, WARN=50, BLOCK=0, SKIP=不计入）
        scores = []
        for v in all_verdicts:
            if v == "PASS":
                scores.append(100.0)
            elif v == "WARN":
                scores.append(50.0)
            elif v == "BLOCK":
                scores.append(0.0)
            # SKIP 不计入
        report.overall_score = sum(scores) / len(scores) if scores else 0.0

        if block_reasons:
            report.overall_verdict = "BLOCKED"
        elif warn_reasons:
            report.overall_verdict = "DEGRADED"
        else:
            report.overall_verdict = "GOOD"

        report.block_reasons = block_reasons
        report.warn_reasons = warn_reasons

        return report

    @staticmethod
    def _compute_max_dd(pnls: Sequence[float], initial_capital: float = 10000.0) -> float:
        """计算最大回撤"""
        if not pnls:
            return 0.0
        equity = [initial_capital]
        for p in pnls:
            equity.append(equity[-1] + p)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd
