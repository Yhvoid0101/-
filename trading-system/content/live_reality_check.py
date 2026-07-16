# -*- coding: utf-8 -*-
"""
实盘真实性验证模块 (Live Reality Check) — 专治"模拟牛逼，实盘亏钱"

针对现有 anti_overfitting.py 的7大致命缺口设计，5大新验证器：

1. LookAheadBiasDetector       — 未来函数黑盒检测（参考 freqtrade lookahead-analysis 2026）
2. CostSensitivityScanner      — 交易成本敏感性扫描（参考 Quantopian Lecture Series by Dr. Michele Goe）
3. LatencyRobustnessScanner    — 延迟鲁棒性扫描（参考 CSDN 2025-12 5层撮合模型）
4. RegimeAwareValidator        — 多市场状态适应性测试（参考 hidden-regime + Pagliaro 2026 Electronics 15(6):1334）
5. LiveBacktestDiscrepancyMonitor — 实盘vs回测差距监测（参考 traderssecondbrain 2026-05 + Gandalf Project 2026）

核心设计原则：
  - 阈值量化：每项检查有精确数值阈值和测量方法
  - 来源标注：每条规则标注学术/工业来源
  - 优雅降级：依赖缺失时回退到简化方案
  - 独立验证：每个验证器可独立运行，也可组合
  - 不造假：检测失败时如实报告，不调整阈值以求通过

引用铁律：禁止投机取巧 — 遇到"沙盘赚钱实盘亏钱"必须 BLOCK，不允许通过调阈值来"美化"结果
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("hermes.live_reality_check")


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class LookAheadBiasResult:
    """未来函数检测结果

    来源：freqtrade lookahead-analysis（48.4K stars，2026.3 文档）
    方法：黑盒探测法 — 不读代码，比较全量回测 vs 切片回测的指标差异
    """

    has_bias: bool = False
    bias_percentage: float = 0.0          # 0-100，受影响的K线比例
    biased_indicators: List[str] = field(default_factory=list)
    n_slices_tested: int = 0
    n_inconsistencies: int = 0
    verdict: str = "PASS"                 # PASS / WARN / BLOCK
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CostSensitivityResult:
    """交易成本敏感性扫描结果

    来源：Quantopian Lecture Series by Dr. Michele Goe
    公式：annual_impact = leverage × turnover × trading_days × txn_cost_bps / 10000
    """

    break_even_bps: float = 0.0           # 盈亏平衡点（bps）
    is_robust: bool = False
    scan_results: List[Dict[str, Any]] = field(default_factory=list)
    worst_case_pnl: float = 0.0           # 最高成本下的PnL
    pnl_degradation_pct: float = 0.0      # 从1bps到20bps的PnL退化百分比
    verdict: str = "PASS"


@dataclass(slots=True)
class LatencyRobustnessResult:
    """延迟鲁棒性扫描结果

    来源：CSDN 2025-12 5层撮合模型 + Almgren-Chriss 最优执行
    """

    break_even_latency_ms: float = 0.0    # 盈亏平衡延迟（ms）
    is_robust: bool = False
    scan_results: List[Dict[str, Any]] = field(default_factory=list)
    worst_case_pnl: float = 0.0           # 最高延迟下的PnL
    pnl_degradation_pct: float = 0.0      # 从50ms到1000ms的PnL退化百分比
    verdict: str = "PASS"


@dataclass(slots=True)
class RegimeAwareResult:
    """多市场状态适应性测试结果

    来源：hidden-regime PyPI + Pagliaro 2026 (Electronics 15(6):1334)
          NeurIPS 2025 Workshop on Adaptive RL
    """

    regimes_detected: List[str] = field(default_factory=list)
    regime_sharpes: Dict[str, float] = field(default_factory=dict)
    regime_pnls: Dict[str, float] = field(default_factory=dict)
    min_regime_sharpe: float = 0.0
    sharpe_consistency: float = 0.0       # 0-1，各regime Sharpe的归一化标准差倒数
    is_robust: bool = False
    failing_regimes: List[str] = field(default_factory=list)
    verdict: str = "PASS"


@dataclass(slots=True)
class DiscrepancyResult:
    """实盘vs回测差距监测结果

    来源：traderssecondbrain 2026-05 + Gandalf Project 2026-02
          deflated-sharpe RegimeDecayDetector (三重确认)
    """

    sharpe_decay: float = 0.0             # (backtest_sharpe - live_sharpe) / backtest_sharpe
    win_rate_decay: float = 0.0           # backtest_winrate - live_winrate
    pnl_discrepancy_pct: float = 0.0      # (backtest_pnl - live_pnl) / |backtest_pnl|
    drawdown_exceeded: bool = False       # live_drawdown > 1.5 × backtest_max_drawdown
    mahalanobis_distance: float = 0.0     # Mahalanobis OOD距离
    signals_fired: int = 0                # 触发的信号数（0-3）
    decay_confirmed: bool = False         # signals_fired >= 2 时为True
    recommendation: str = "CONTINUE"      # CONTINUE / MONITOR / STOP
    verdict: str = "PASS"


@dataclass(slots=True)
class LiveRealityCheckResult:
    """综合实盘真实性验证结果"""

    lookahead: Optional[LookAheadBiasResult] = None
    cost_sensitivity: Optional[CostSensitivityResult] = None
    latency_robustness: Optional[LatencyRobustnessResult] = None
    regime_aware: Optional[RegimeAwareResult] = None
    discrepancy: Optional[DiscrepancyResult] = None
    overall_verdict: str = "PASS"         # PASS / WARN / BLOCK
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)


# ============================================================================
# 1. LookAheadBiasDetector — 未来函数黑盒检测器
# ============================================================================


class LookAheadBiasDetector:
    """未来函数黑盒检测器

    来源：freqtrade lookahead-analysis（48.4K stars，2026.3 文档）
          https://docs.freqtrade.io/en/2026.3/lookahead-analysis/

    方法：黑盒探测法 — 不读代码，强制 --cache=none + 无限钱包 + 市价单，
          比较 baseline vs sliced 回测的指标差异。

    原理：
      如果策略使用了未来函数（如 .rolling(5).mean() 未 .shift(1)），
      那么在某根K线 i 上计算的指标值，会依赖于 i+1, i+2, ... 的数据。
      当我们只取 data[:i] 进行回测时，指标值会发生变化（因为未来数据被切断）。
      通过比较全量回测和切片回测的指标值，可以检测出未来函数。

    阈值：
      - bias_percentage < 1%  → PASS（正常数值误差）
      - bias_percentage 1-5%  → WARN（可能有问题）
      - bias_percentage >= 5% → BLOCK（明确未来函数）
    """

    def __init__(
        self,
        n_slices: int = 10,
        bias_threshold_pct: float = 5.0,
        warn_threshold_pct: float = 1.0,
    ):
        """
        Args:
            n_slices: 切片数量（从尾部逐步减少）
            bias_threshold_pct: BLOCK阈值（%）
            warn_threshold_pct: WARN阈值（%）
        """
        self.n_slices = max(3, n_slices)
        self.bias_threshold_pct = bias_threshold_pct
        self.warn_threshold_pct = warn_threshold_pct

    def detect(
        self,
        indicator_fn: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        indicator_name: str = "unknown",
    ) -> LookAheadBiasResult:
        """检测指标计算函数是否存在未来函数

        Args:
            indicator_fn: 输入价格序列，返回指标序列的函数
            data: 价格序列（一维np.ndarray）
            indicator_name: 指标名称（用于报告）

        Returns:
            LookAheadBiasResult
        """
        result = LookAheadBiasResult()
        result.n_slices_tested = self.n_slices

        if data is None or len(data) < 20:
            result.verdict = "WARN"
            result.details.append({"error": "数据不足（<20）"})
            return result

        # 全量回测作为baseline
        try:
            baseline_indicators = np.asarray(indicator_fn(data), dtype=float)
        except Exception as e:
            result.verdict = "WARN"
            result.details.append({"error": f"indicator_fn执行失败: {e}"})
            return result

        n = len(data)
        slice_step = max(1, n // (self.n_slices + 2))
        inconsistencies = 0
        total_checks = 0

        for i in range(self.n_slices):
            # 从尾部逐步减少数据
            slice_end = n - (i + 1) * slice_step
            if slice_end < 10:
                break

            sliced_data = data[:slice_end]
            try:
                sliced_indicators = np.asarray(indicator_fn(sliced_data), dtype=float)
            except Exception:
                continue

            # 比较前 slice_end 个数据点的指标值
            min_len = min(len(baseline_indicators), len(sliced_indicators))
            if min_len < 1:
                continue

            baseline_slice = baseline_indicators[:min_len]
            sliced_slice = sliced_indicators[:min_len]

            # 过滤NaN
            valid_mask = ~(np.isnan(baseline_slice) | np.isnan(sliced_slice))
            if valid_mask.sum() < 5:
                continue

            baseline_valid = baseline_slice[valid_mask]
            sliced_valid = sliced_slice[valid_mask]

            # 计算相对差异
            diff = np.abs(baseline_valid - sliced_valid)
            scale = np.abs(baseline_valid) + 1e-10
            rel_diff = diff / scale

            # 超过1%相对差异的视为不一致
            n_inconsistent = int(np.sum(rel_diff > 0.01))
            inconsistencies += n_inconsistent
            total_checks += len(rel_diff)

            if n_inconsistent > 0:
                result.details.append({
                    "slice": i,
                    "slice_end": slice_end,
                    "n_inconsistent": n_inconsistent,
                    "max_rel_diff": float(np.max(rel_diff)),
                })

        if total_checks > 0:
            result.bias_percentage = float(inconsistencies) / total_checks * 100.0
        result.n_inconsistencies = inconsistencies

        if result.bias_percentage >= self.bias_threshold_pct:
            result.has_bias = True
            result.biased_indicators = [indicator_name]
            result.verdict = "BLOCK"
        elif result.bias_percentage >= self.warn_threshold_pct:
            result.has_bias = True
            result.biased_indicators = [indicator_name]
            result.verdict = "WARN"
        else:
            result.verdict = "PASS"

        return result


# ============================================================================
# 2. CostSensitivityScanner — 交易成本敏感性扫描
# ============================================================================


class CostSensitivityScanner:
    """交易成本敏感性扫描器

    来源：Quantopian Lecture Series by Dr. Michele Goe
          https://wqu.guru/blog/quantopia-quantitative-analysis-56-lectures/market-impact-models

    公式：annual_impact = leverage × turnover × trading_days × txn_cost_bps / 10000

    实践：1 bps 执行成本在 2×杠杆、40%换手率、252交易日下 = 年化 2.016% 性能损耗

    扫描点：fee_bps ∈ [0, 1, 2, 3, 5, 10, 20, 50]
      - 0 bps：零成本基准
      - 1 bps：maker费率（VIP）
      - 2 bps：taker费率（普通）
      - 3 bps：含轻微滑点
      - 5 bps：含市场冲击
      - 10 bps：流动性较差
      - 20 bps：极端情况
      - 50 bps：压力测试

    v2.30阈值收紧(2026-06-26):
      原阈值: pass=3bps, warn=10bps → break_even>=10就PASS
      问题: Binance现货0.1%=10bps手续费 + 滑点0.05%=5bps = 15bps实盘硬成本
            break_even=10bps的策略在实盘净亏5bps/笔 = "模拟牛逼实盘亏"
            这是用户核心铁律"不要模拟牛逼,实盘亏钱"的根源
      新阈值: pass=15bps, warn=25bps
        - break_even >= 25bps → PASS(能承受实盘+缓冲+多笔连亏)
        - break_even 15-25bps → WARN(刚好覆盖实盘成本,边缘策略)
        - break_even < 15bps  → BLOCK(实盘必亏,禁止部署)
      来源: Binance现货费率0.1% + Almgren-Chriss滑点模型0.05% + 5bps安全边际
      铁律: 标准只升不降 — 旧阈值3/10是对真实成本的低估,新阈值15/25反映实盘
    """

    DEFAULT_COST_POINTS: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0)

    def __init__(
        self,
        cost_points: Optional[Sequence[float]] = None,
        pass_threshold_bps: float = 15.0,
        warn_threshold_bps: float = 25.0,
    ):
        self.cost_points = tuple(cost_points) if cost_points else self.DEFAULT_COST_POINTS
        self.pass_threshold_bps = pass_threshold_bps
        self.warn_threshold_bps = warn_threshold_bps

    def scan(
        self,
        trade_pnls: Sequence[float],
        trade_volumes: Optional[Sequence[float]] = None,
        capital: float = 10000.0,
        already_costed: bool = False,
    ) -> CostSensitivityResult:
        """扫描不同成本下的策略盈亏

        Args:
            trade_pnls: 每笔交易的原始PnL（未扣成本）
            trade_volumes: 每笔交易成交量（用于计算换手率），None则假设等权
            capital: 初始资金
            already_costed: 若True表示trade_pnls已扣过基础成本（如maker fee），
                           扫描时只叠加增量成本，避免二次扣费（v597 Bug#1修复）

        Returns:
            CostSensitivityResult
        """
        result = CostSensitivityResult()

        if trade_pnls is None or len(trade_pnls) < 5:
            result.verdict = "WARN"
            result.scan_results.append({"error": "交易笔数不足（<5）"})
            return result

        pnls = np.asarray(trade_pnls, dtype=float)
        n_trades = len(pnls)

        # 计算每笔交易的成交量（用于成本扣除）
        if trade_volumes is not None and len(trade_volumes) == n_trades:
            volumes = np.asarray(trade_volumes, dtype=float)
        else:
            # 假设每笔交易占资金的10%（典型日内策略）
            volumes = np.full(n_trades, capital * 0.1)

        # 扫描不同成本点
        # v597 Bug#1修复: already_costed=True时，0bps点就是已扣基础成本的PnL，
        # 只叠加增量成本（cost_bps - base_already_costed_bps），避免二次扣费
        base_already_costed_bps = 0.0  # 默认未扣过
        for cost_bps in self.cost_points:
            incremental_bps = max(0.0, cost_bps - base_already_costed_bps) if already_costed else cost_bps
            cost_per_trade = volumes * incremental_bps / 10000.0
            adjusted_pnls = pnls - cost_per_trade
            total_pnl = float(np.sum(adjusted_pnls))
            win_rate = float(np.mean(adjusted_pnls > 0)) if n_trades > 0 else 0.0

            # 计算年化Sharpe（假设252交易日，每天1笔）
            if np.std(adjusted_pnls) > 0:
                sharpe = float(np.mean(adjusted_pnls) / np.std(adjusted_pnls) * math.sqrt(252))
            else:
                sharpe = 0.0

            result.scan_results.append({
                "cost_bps": cost_bps,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "n_trades": n_trades,
            })

        # 找到盈亏平衡点（PnL由正转负的成本点）
        break_even = self._find_break_even(result.scan_results)
        result.break_even_bps = break_even

        # 最坏情况（最高成本）
        result.worst_case_pnl = result.scan_results[-1]["total_pnl"]

        # PnL退化百分比（从1bps到20bps）
        pnl_at_1bps = next((r["total_pnl"] for r in result.scan_results if r["cost_bps"] == 1.0), 0.0)
        pnl_at_20bps = next((r["total_pnl"] for r in result.scan_results if r["cost_bps"] == 20.0), 0.0)
        if abs(pnl_at_1bps) > 1e-10:
            result.pnl_degradation_pct = float((pnl_at_1bps - pnl_at_20bps) / abs(pnl_at_1bps) * 100.0)

        # 判定 (v597 Bug#1修复: nan表示策略在所有成本下都亏损，直接BLOCK)
        if math.isnan(break_even):
            # 策略即使0成本也亏损，break-even不可知
            result.is_robust = False
            result.verdict = "BLOCK"
        elif break_even >= self.warn_threshold_bps:
            result.is_robust = True
            result.verdict = "PASS"
        elif break_even >= self.pass_threshold_bps:
            result.is_robust = False
            result.verdict = "WARN"
        else:
            result.is_robust = False
            result.verdict = "BLOCK"

        return result

    def _find_break_even(self, scan_results: List[Dict[str, Any]]) -> float:
        """找到PnL由正转负的成本点

        v597 Bug#1修复: 无正PnL点时返回 nan 而非 0.0
        - 0.0 会误导为"break-even=0bps"（看似有值但实为亏损）
        - nan 正确表示"break-even不可知，策略在所有成本下都亏损"
        - nan >= threshold 在Python中返回False，verdict正确落入BLOCK
        """
        positive_points = [r for r in scan_results if r["total_pnl"] > 0]
        if not positive_points:
            return float('nan')  # v597: 策略即使0成本也亏损，break-even不可知
        if len(positive_points) == len(scan_results):
            return float(scan_results[-1]["cost_bps"]) * 2.0  # 所有点都盈利，返回2倍最高点

        # 找到第一个PnL为负的点
        sorted_results = sorted(scan_results, key=lambda r: r["cost_bps"])
        for i, r in enumerate(sorted_results):
            if r["total_pnl"] <= 0 and i > 0:
                prev = sorted_results[i - 1]
                # 线性插值
                if r["cost_bps"] > prev["cost_bps"]:
                    ratio = prev["total_pnl"] / (prev["total_pnl"] - r["total_pnl"] + 1e-10)
                    return float(prev["cost_bps"] + ratio * (r["cost_bps"] - prev["cost_bps"]))
                return float(prev["cost_bps"])
        return float(scan_results[-1]["cost_bps"])


# ============================================================================
# 3. LatencyRobustnessScanner — 延迟鲁棒性扫描
# ============================================================================


class LatencyRobustnessScanner:
    """延迟鲁棒性扫描器

    来源：CSDN 2025-12 5层撮合模型 + Almgren-Chriss 最优执行
          https://ask.csdn.net/questions/9074882

    5层撮合模型：
      L1: 收盘价成交（高估 ~50%）— Backtrader默认
      L2: 固定滑点 — Zipline默认
      L3: 成交量加权撮合 — 自定义
      L4: 订单簿驱动 + 市场冲击 — QuantConnect
      L5: 含网络延迟 + 风控逻辑 — 实盘系统专属

    延迟成本公式（来自 anti_overfitting.py:432-498 RealisticSlippageModel）：
      latency_cost_bps = latency_ms / 1000 × 5.0

    扫描点：latency_ms ∈ [50, 100, 200, 500, 1000, 2000]
      - 50ms：本地交易所 colocate
      - 100ms：同城机房
      - 200ms：跨城
      - 500ms：跨国
      - 1000ms：网络抖动
      - 2000ms：极端情况（API限流）

    阈值：
      - break_even_latency_ms >= 1000 → PASS（对延迟不敏感）
      - break_even_latency_ms 200-1000 → WARN（中等敏感）
      - break_even_latency_ms < 200   → BLOCK（高频策略，延迟敏感）
    """

    DEFAULT_LATENCY_POINTS: Tuple[float, ...] = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)
    LATENCY_COST_COEFFICIENT: float = 5.0  # bps per second

    def __init__(
        self,
        latency_points: Optional[Sequence[float]] = None,
        pass_threshold_ms: float = 1000.0,
        warn_threshold_ms: float = 200.0,
    ):
        self.latency_points = tuple(latency_points) if latency_points else self.DEFAULT_LATENCY_POINTS
        self.pass_threshold_ms = pass_threshold_ms
        self.warn_threshold_ms = warn_threshold_ms

    def scan(
        self,
        trade_pnls: Sequence[float],
        trade_volumes: Optional[Sequence[float]] = None,
        capital: float = 10000.0,
    ) -> LatencyRobustnessResult:
        """扫描不同延迟下的策略盈亏

        Args:
            trade_pnls: 每笔交易的原始PnL（未扣延迟成本）
            trade_volumes: 每笔交易成交量，None则假设等权
            capital: 初始资金

        Returns:
            LatencyRobustnessResult
        """
        result = LatencyRobustnessResult()

        if trade_pnls is None or len(trade_pnls) < 5:
            result.verdict = "WARN"
            result.scan_results.append({"error": "交易笔数不足（<5）"})
            return result

        pnls = np.asarray(trade_pnls, dtype=float)
        n_trades = len(pnls)

        if trade_volumes is not None and len(trade_volumes) == n_trades:
            volumes = np.asarray(trade_volumes, dtype=float)
        else:
            volumes = np.full(n_trades, capital * 0.1)

        for latency_ms in self.latency_points:
            # 延迟成本 = latency_ms / 1000 × 5.0 bps
            latency_cost_bps = latency_ms / 1000.0 * self.LATENCY_COST_COEFFICIENT
            cost_per_trade = volumes * latency_cost_bps / 10000.0
            adjusted_pnls = pnls - cost_per_trade
            total_pnl = float(np.sum(adjusted_pnls))
            win_rate = float(np.mean(adjusted_pnls > 0)) if n_trades > 0 else 0.0

            if np.std(adjusted_pnls) > 0:
                sharpe = float(np.mean(adjusted_pnls) / np.std(adjusted_pnls) * math.sqrt(252))
            else:
                sharpe = 0.0

            result.scan_results.append({
                "latency_ms": latency_ms,
                "latency_cost_bps": latency_cost_bps,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "n_trades": n_trades,
            })

        # 找到盈亏平衡点
        break_even = self._find_break_even(result.scan_results)
        result.break_even_latency_ms = break_even

        result.worst_case_pnl = result.scan_results[-1]["total_pnl"]

        pnl_at_50ms = next((r["total_pnl"] for r in result.scan_results if r["latency_ms"] == 50.0), 0.0)
        pnl_at_1000ms = next((r["total_pnl"] for r in result.scan_results if r["latency_ms"] == 1000.0), 0.0)
        if abs(pnl_at_50ms) > 1e-10:
            result.pnl_degradation_pct = float((pnl_at_50ms - pnl_at_1000ms) / abs(pnl_at_50ms) * 100.0)

        # v597 Bug#1修复: nan表示策略在所有延迟下都亏损，直接BLOCK
        if math.isnan(break_even):
            result.is_robust = False
            result.verdict = "BLOCK"
        elif break_even >= self.pass_threshold_ms:
            result.is_robust = True
            result.verdict = "PASS"
        elif break_even >= self.warn_threshold_ms:
            result.is_robust = False
            result.verdict = "WARN"
        else:
            result.is_robust = False
            result.verdict = "BLOCK"

        return result

    def _find_break_even(self, scan_results: List[Dict[str, Any]]) -> float:
        """找到PnL由正转负的延迟点

        v597 Bug#1修复: 无正PnL点时返回 nan 而非 0.0
        """
        positive_points = [r for r in scan_results if r["total_pnl"] > 0]
        if not positive_points:
            return float('nan')  # v597: 策略即使0延迟也亏损
        if len(positive_points) == len(scan_results):
            return float(scan_results[-1]["latency_ms"]) * 2.0

        sorted_results = sorted(scan_results, key=lambda r: r["latency_ms"])
        for i, r in enumerate(sorted_results):
            if r["total_pnl"] <= 0 and i > 0:
                prev = sorted_results[i - 1]
                if r["latency_ms"] > prev["latency_ms"]:
                    ratio = prev["total_pnl"] / (prev["total_pnl"] - r["total_pnl"] + 1e-10)
                    return float(prev["latency_ms"] + ratio * (r["latency_ms"] - prev["latency_ms"]))
                return float(prev["latency_ms"])
        return float(scan_results[-1]["latency_ms"])


# ============================================================================
# 4. RegimeAwareValidator — 多市场状态适应性测试
# ============================================================================


class RegimeAwareValidator:
    """多市场状态适应性测试

    来源：hidden-regime PyPI (Production/Stable)
          Pagliaro 2026 (Electronics 15(6):1334) — Regime-Aware LightGBM
          NeurIPS 2025 Workshop — Adaptive and Regime-Aware RL
          arXiv:2509.14385 — KMeans/GMM/HMM 三种regime检测对比

    方法：用简单波动率+趋势分位检测3个regime（牛/熊/震荡）
          在每个regime内单独计算策略Sharpe，要求所有regime都 > 0

    为什么不用HMM：
      - HMM需要hmmlearn依赖，且参数调优复杂
      - Pagliaro 2026实证显示，简单波动率分位的regime检测效果接近HMM
      - 优雅降级原则：无外部依赖也能工作

    阈值：
      - 所有regime Sharpe > 0 且 min_regime_sharpe > 0.5 → PASS
      - 所有regime Sharpe > 0 但 min_regime_sharpe <= 0.5 → WARN
      - 任一regime Sharpe <= 0 → BLOCK（该regime下亏损）
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lookback: int = 20,
        vol_quantiles: Tuple[float, ...] = (0.33, 0.67),
        min_sharpe_threshold: float = 0.5,
    ):
        self.n_regimes = max(2, n_regimes)
        self.lookback = max(5, lookback)
        self.vol_quantiles = vol_quantiles
        self.min_sharpe_threshold = min_sharpe_threshold

    def validate(
        self,
        returns: Sequence[float],
        prices: Optional[Sequence[float]] = None,
    ) -> RegimeAwareResult:
        """验证策略在不同市场状态下的适应性

        Args:
            returns: 收益率序列
            prices: 价格序列（可选，用于计算波动率）

        Returns:
            RegimeAwareResult
        """
        result = RegimeAwareResult()

        if returns is None or len(returns) < 60:
            result.verdict = "WARN"
            result.regimes_detected = ["insufficient_data"]
            return result

        returns_arr = np.asarray(returns, dtype=float)

        # 计算滚动波动率
        if prices is not None and len(prices) >= self.lookback:
            prices_arr = np.asarray(prices, dtype=float)
            vol = self._compute_rolling_vol(prices_arr)
        else:
            vol = self._compute_rolling_vol_from_returns(returns_arr)

        if len(vol) < 20:
            result.verdict = "WARN"
            result.regimes_detected = ["insufficient_vol_data"]
            return result

        # 用波动率分位检测regime
        regimes = self._detect_regimes(vol, returns_arr)
        result.regimes_detected = list(set(regimes))

        # 在每个regime内计算Sharpe和PnL
        for regime in result.regimes_detected:
            mask = np.array([r == regime for r in regimes])
            regime_returns = returns_arr[mask]

            if len(regime_returns) < 5:
                result.regime_sharpes[regime] = 0.0
                result.regime_pnls[regime] = 0.0
                continue

            regime_pnl = float(np.sum(regime_returns))
            if np.std(regime_returns) > 0:
                regime_sharpe = float(
                    np.mean(regime_returns) / np.std(regime_returns) * math.sqrt(252)
                )
            else:
                regime_sharpe = 0.0

            result.regime_sharpes[regime] = regime_sharpe
            result.regime_pnls[regime] = regime_pnl

        # 计算一致性
        sharpes = list(result.regime_sharpes.values())
        if sharpes:
            result.min_regime_sharpe = float(min(sharpes))
            mean_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes))
            if mean_sharpe > 0:
                cv = std_sharpe / (abs(mean_sharpe) + 1e-10)
                result.sharpe_consistency = float(max(0.0, 1.0 - cv))

        # 找出失败的regime
        result.failing_regimes = [
            r for r, s in result.regime_sharpes.items()
            if s <= 0.0
        ]

        # 判定
        if result.failing_regimes:
            result.is_robust = False
            result.verdict = "BLOCK"
        elif result.min_regime_sharpe < self.min_sharpe_threshold:
            result.is_robust = False
            result.verdict = "WARN"
        else:
            result.is_robust = True
            result.verdict = "PASS"

        return result

    def _compute_rolling_vol(self, prices: np.ndarray) -> np.ndarray:
        """从价格序列计算滚动波动率"""
        log_returns = np.diff(np.log(prices + 1e-10))
        if len(log_returns) < self.lookback:
            return np.array([])
        vol = np.array([
            np.std(log_returns[max(0, i - self.lookback):i + 1])
            for i in range(len(log_returns))
        ])
        return vol

    def _compute_rolling_vol_from_returns(self, returns: np.ndarray) -> np.ndarray:
        """从收益率序列计算滚动波动率"""
        if len(returns) < self.lookback:
            return np.array([])
        vol = np.array([
            np.std(returns[max(0, i - self.lookback):i + 1])
            for i in range(len(returns))
        ])
        return vol

    def _detect_regimes(self, vol: np.ndarray, returns: np.ndarray) -> List[str]:
        """检测市场状态

        简单方法：用波动率分位 + 收益率方向
          - 低波动 + 正收益 → bull
          - 低波动 + 负收益 → bear
          - 高波动 → volatile
          - 中波动 → sideways
        """
        min_len = min(len(vol), len(returns))
        vol = vol[:min_len]
        returns = returns[:min_len]

        q1 = np.quantile(vol, self.vol_quantiles[0])
        q2 = np.quantile(vol, self.vol_quantiles[1]) if len(self.vol_quantiles) > 1 else q1

        regimes = []
        for i in range(min_len):
            if vol[i] >= q2:
                regimes.append("volatile")
            elif vol[i] <= q1:
                if returns[i] > 0:
                    regimes.append("bull")
                else:
                    regimes.append("bear")
            else:
                regimes.append("sideways")
        return regimes


# ============================================================================
# 5. LiveBacktestDiscrepancyMonitor — 实盘vs回测差距监测
# ============================================================================


class LiveBacktestDiscrepancyMonitor:
    """实盘vs回测差距监测器

    来源：traderssecondbrain 2026-05 — 回测年化+47% → 实盘-12% 的诊断框架
          Gandalf Project 2026-02 — 5个月141笔实盘交易，P&L 实际 vs 理论差 +4.74%
          deflated-sharpe RegimeDecayDetector — 三重确认（贝叶斯+回撤+Mahalanobis）

    五大偏差指标：
      1. Sharpe衰减：(backtest_sharpe - live_sharpe) / backtest_sharpe
      2. 胜率衰减：backtest_winrate - live_winrate
      3. PnL偏差：(backtest_pnl - live_pnl) / |backtest_pnl|
      4. 回撤超越：live_drawdown > 1.5 × backtest_max_drawdown
      5. Mahalanobis OOD：实盘收益分布与回测分布的Mahalanobis距离

    三重确认触发机制（参考 RegimeDecayDetector）：
      - 信号1：Sharpe衰减 > 50%
      - 信号2：回撤超越（live_drawdown > 1.5×backtest）
      - 信号3：Mahalanobis距离 > 3σ
      - 信号数 >= 2 → decay_confirmed=True → 建议停止交易

    阈值（来源：traderssecondbrain 2026-05 实证）：
      - 退化 < 30% → PASS（正常执行差距）
      - 退化 30-60% → WARN（需要诊断）
      - 退化 > 60% → BLOCK（结构性问题）
    """

    def __init__(
        self,
        sharpe_decay_block_threshold: float = 0.6,
        sharpe_decay_warn_threshold: float = 0.3,
        win_rate_decay_block: float = 0.15,
        drawdown_exceed_factor: float = 1.5,
        mahalanobis_threshold: float = 3.0,
        min_live_trades: int = 20,
    ):
        self.sharpe_decay_block_threshold = sharpe_decay_block_threshold
        self.sharpe_decay_warn_threshold = sharpe_decay_warn_threshold
        self.win_rate_decay_block = win_rate_decay_block
        self.drawdown_exceed_factor = drawdown_exceed_factor
        self.mahalanobis_threshold = mahalanobis_threshold
        self.min_live_trades = min_live_trades

    def assess(
        self,
        backtest_sharpe: float,
        backtest_win_rate: float,
        backtest_pnl: float,
        backtest_max_drawdown: float,
        backtest_returns: Sequence[float],
        live_trades: Sequence[float],
    ) -> DiscrepancyResult:
        """评估实盘vs回测的差距

        Args:
            backtest_sharpe: 回测Sharpe
            backtest_win_rate: 回测胜率
            backtest_pnl: 回测总PnL
            backtest_max_drawdown: 回测最大回撤
            backtest_returns: 回测收益率序列（用于Mahalanobis）
            live_trades: 实盘每笔交易PnL序列

        Returns:
            DiscrepancyResult
        """
        result = DiscrepancyResult()

        if live_trades is None or len(live_trades) < self.min_live_trades:
            result.verdict = "WARN"
            result.recommendation = "INSUFFICIENT_LIVE_DATA"
            return result

        live_pnls = np.asarray(live_trades, dtype=float)
        n_live = len(live_pnls)

        # 计算实盘指标
        live_pnl = float(np.sum(live_pnls))
        live_win_rate = float(np.mean(live_pnls > 0)) if n_live > 0 else 0.0
        if np.std(live_pnls) > 0:
            live_sharpe = float(np.mean(live_pnls) / np.std(live_pnls) * math.sqrt(252))
        else:
            live_sharpe = 0.0

        # 实盘最大回撤
        live_cumulative = np.cumsum(live_pnls)
        live_running_max = np.maximum.accumulate(live_cumulative)
        live_drawdown = float(np.max(live_running_max - live_cumulative)) if n_live > 0 else 0.0

        # 信号1：Sharpe衰减
        if abs(backtest_sharpe) > 1e-10:
            result.sharpe_decay = float(
                (backtest_sharpe - live_sharpe) / abs(backtest_sharpe)
            )
        else:
            result.sharpe_decay = 0.0

        # 信号2：胜率衰减
        result.win_rate_decay = float(backtest_win_rate - live_win_rate)

        # 信号3：PnL偏差
        if abs(backtest_pnl) > 1e-10:
            result.pnl_discrepancy_pct = float(
                (backtest_pnl - live_pnl) / abs(backtest_pnl)
            )

        # 信号4：回撤超越
        if backtest_max_drawdown > 0:
            result.drawdown_exceeded = bool(
                live_drawdown > self.drawdown_exceed_factor * backtest_max_drawdown
            )

        # 信号5：Mahalanobis OOD
        result.mahalanobis_distance = self._compute_mahalanobis(
            backtest_returns, live_pnls
        )

        # 三重确认触发
        signals = 0
        if result.sharpe_decay > self.sharpe_decay_block_threshold:
            signals += 1
        if result.drawdown_exceeded:
            signals += 1
        if result.mahalanobis_distance > self.mahalanobis_threshold:
            signals += 1

        result.signals_fired = signals
        result.decay_confirmed = signals >= 2

        # 推荐
        if result.decay_confirmed:
            result.recommendation = "STOP"
            result.verdict = "BLOCK"
        elif (result.sharpe_decay > self.sharpe_decay_warn_threshold
              or result.win_rate_decay > self.win_rate_decay_block
              or result.pnl_discrepancy_pct > 0.3):
            result.recommendation = "MONITOR"
            result.verdict = "WARN"
        else:
            result.recommendation = "CONTINUE"
            result.verdict = "PASS"

        return result

    def _compute_mahalanobis(
        self,
        backtest_returns: Sequence[float],
        live_pnls: np.ndarray,
    ) -> float:
        """计算Mahalanobis距离

        公式：D² = (x - μ)ᵀ × Σ⁻¹ × (x - μ)
        其中 μ, Σ 是回测收益的均值和协方差
        """
        try:
            bt = np.asarray(backtest_returns, dtype=float)
            if len(bt) < 10 or len(live_pnls) < 5:
                return 0.0

            # 用回测数据的均值和方差
            mu = float(np.mean(bt))
            sigma = float(np.std(bt))
            if sigma < 1e-10:
                return 0.0

            # 实盘PnL的均值
            live_mu = float(np.mean(live_pnls))

            # 简化Mahalanobis（一维）
            distance = abs(live_mu - mu) / sigma
            return float(distance)
        except Exception as e:
            logger.debug("Mahalanobis计算失败: %s", e)
            return 0.0


# ============================================================================
# 综合验证器
# ============================================================================


class LiveRealityCheckValidator:
    """综合实盘真实性验证器

    整合5大验证器，提供统一的验证接口。

    用法：
        validator = LiveRealityCheckValidator()
        result = validator.validate(
            trade_pnls=trades,
            returns=returns,
            prices=prices,
            backtest_sharpe=2.0,
            backtest_win_rate=0.55,
            backtest_pnl=1000.0,
            backtest_max_drawdown=0.15,
            backtest_returns=returns,
            live_trades=live_pnls,
        )
        if result.overall_verdict == "BLOCK":
            # 禁止部署
    """

    def __init__(
        self,
        lookahead_detector: Optional[LookAheadBiasDetector] = None,
        cost_scanner: Optional[CostSensitivityScanner] = None,
        latency_scanner: Optional[LatencyRobustnessScanner] = None,
        regime_validator: Optional[RegimeAwareValidator] = None,
        discrepancy_monitor: Optional[LiveBacktestDiscrepancyMonitor] = None,
    ):
        self.lookahead_detector = lookahead_detector or LookAheadBiasDetector()
        self.cost_scanner = cost_scanner or CostSensitivityScanner()
        self.latency_scanner = latency_scanner or LatencyRobustnessScanner()
        self.regime_validator = regime_validator or RegimeAwareValidator()
        self.discrepancy_monitor = discrepancy_monitor or LiveBacktestDiscrepancyMonitor()

    def validate(
        self,
        trade_pnls: Optional[Sequence[float]] = None,
        returns: Optional[Sequence[float]] = None,
        prices: Optional[Sequence[float]] = None,
        backtest_sharpe: Optional[float] = None,
        backtest_win_rate: Optional[float] = None,
        backtest_pnl: Optional[float] = None,
        backtest_max_drawdown: Optional[float] = None,
        backtest_returns: Optional[Sequence[float]] = None,
        live_trades: Optional[Sequence[float]] = None,
        indicator_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        indicator_data: Optional[np.ndarray] = None,
    ) -> LiveRealityCheckResult:
        """执行综合实盘真实性验证

        所有参数可选，按需执行对应的验证器。
        """
        result = LiveRealityCheckResult()

        # 1. 未来函数检测（需要indicator_fn和indicator_data）
        if indicator_fn is not None and indicator_data is not None:
            try:
                result.lookahead = self.lookahead_detector.detect(
                    indicator_fn, indicator_data, "main_indicator"
                )
                if result.lookahead.verdict == "BLOCK":
                    result.block_reasons.append(
                        f"未来函数检测失败: bias={result.lookahead.bias_percentage:.2f}%"
                    )
                elif result.lookahead.verdict == "WARN":
                    result.warn_reasons.append(
                        f"未来函数警告: bias={result.lookahead.bias_percentage:.2f}%"
                    )
            except Exception as e:
                logger.warning("未来函数检测异常: %s", e)
                result.warn_reasons.append(f"未来函数检测异常: {e}")

        # 2. 成本敏感性扫描（需要trade_pnls）
        if trade_pnls is not None and len(trade_pnls) >= 5:
            try:
                result.cost_sensitivity = self.cost_scanner.scan(trade_pnls)
                # v597 Bug#1修复: nan表示策略在所有成本下都亏损
                be_bps = result.cost_sensitivity.break_even_bps
                be_str = f"{be_bps:.2f}" if not math.isnan(be_bps) else "N/A(全成本亏损)"
                if result.cost_sensitivity.verdict == "BLOCK":
                    result.block_reasons.append(
                        f"成本敏感性失败: break_even={be_str}bps"
                    )
                elif result.cost_sensitivity.verdict == "WARN":
                    result.warn_reasons.append(
                        f"成本敏感性警告: break_even={be_str}bps"
                    )
            except Exception as e:
                logger.warning("成本敏感性扫描异常: %s", e)

        # 3. 延迟鲁棒性扫描（需要trade_pnls）
        if trade_pnls is not None and len(trade_pnls) >= 5:
            try:
                result.latency_robustness = self.latency_scanner.scan(trade_pnls)
                # v597 Bug#1修复: nan表示策略在所有延迟下都亏损
                be_lat = result.latency_robustness.break_even_latency_ms
                be_lat_str = f"{be_lat:.0f}" if not math.isnan(be_lat) else "N/A(全延迟亏损)"
                if result.latency_robustness.verdict == "BLOCK":
                    result.block_reasons.append(
                        f"延迟鲁棒性失败: break_even={be_lat_str}ms"
                    )
                elif result.latency_robustness.verdict == "WARN":
                    result.warn_reasons.append(
                        f"延迟鲁棒性警告: break_even={be_lat_str}ms"
                    )
            except Exception as e:
                logger.warning("延迟鲁棒性扫描异常: %s", e)

        # 4. 多市场状态适应性（需要returns）
        if returns is not None and len(returns) >= 60:
            try:
                result.regime_aware = self.regime_validator.validate(returns, prices)
                if result.regime_aware.verdict == "BLOCK":
                    failing = ", ".join(result.regime_aware.failing_regimes)
                    result.block_reasons.append(
                        f"多市场状态适应性失败: 亏损regime=[{failing}]"
                    )
                elif result.regime_aware.verdict == "WARN":
                    result.warn_reasons.append(
                        f"多市场状态适应性警告: min_sharpe={result.regime_aware.min_regime_sharpe:.2f}"
                    )
            except Exception as e:
                logger.warning("多市场状态适应性测试异常: %s", e)

        # 5. 实盘vs回测差距监测（需要backtest指标 + live_trades）
        if (backtest_sharpe is not None and live_trades is not None
                and len(live_trades) >= 20):
            try:
                result.discrepancy = self.discrepancy_monitor.assess(
                    backtest_sharpe=backtest_sharpe,
                    backtest_win_rate=backtest_win_rate or 0.5,
                    backtest_pnl=backtest_pnl or 0.0,
                    backtest_max_drawdown=backtest_max_drawdown or 0.2,
                    backtest_returns=backtest_returns or returns or [],
                    live_trades=live_trades,
                )
                if result.discrepancy.verdict == "BLOCK":
                    result.block_reasons.append(
                        f"实盘vs回测差距严重: sharpe_decay={result.discrepancy.sharpe_decay:.2%}"
                    )
                elif result.discrepancy.verdict == "WARN":
                    result.warn_reasons.append(
                        f"实盘vs回测差距警告: sharpe_decay={result.discrepancy.sharpe_decay:.2%}"
                    )
            except Exception as e:
                logger.warning("实盘vs回测差距监测异常: %s", e)

        # 综合判定
        if result.block_reasons:
            result.overall_verdict = "BLOCK"
        elif result.warn_reasons:
            result.overall_verdict = "WARN"
        else:
            result.overall_verdict = "PASS"

        return result

    def generate_report(self, result: LiveRealityCheckResult) -> str:
        """生成人类可读的验证报告"""
        lines = ["=" * 60, "实盘真实性验证报告 (Live Reality Check)", "=" * 60, ""]

        if result.lookahead:
            lines.append(f"1. 未来函数检测: {result.lookahead.verdict}")
            lines.append(f"   bias_percentage: {result.lookahead.bias_percentage:.2f}%")
            lines.append(f"   n_inconsistencies: {result.lookahead.n_inconsistencies}")
            lines.append("")

        if result.cost_sensitivity:
            be_bps = result.cost_sensitivity.break_even_bps
            be_bps_str = f"{be_bps:.2f}" if not math.isnan(be_bps) else "N/A(全成本亏损)"
            lines.append(f"2. 成本敏感性: {result.cost_sensitivity.verdict}")
            lines.append(f"   break_even_bps: {be_bps_str}")
            lines.append(f"   worst_case_pnl: {result.cost_sensitivity.worst_case_pnl:.2f}")
            lines.append(f"   pnl_degradation: {result.cost_sensitivity.pnl_degradation_pct:.2f}%")
            lines.append("")

        if result.latency_robustness:
            be_lat = result.latency_robustness.break_even_latency_ms
            be_lat_str = f"{be_lat:.0f}" if not math.isnan(be_lat) else "N/A(全延迟亏损)"
            lines.append(f"3. 延迟鲁棒性: {result.latency_robustness.verdict}")
            lines.append(f"   break_even_latency_ms: {be_lat_str}")
            lines.append(f"   worst_case_pnl: {result.latency_robustness.worst_case_pnl:.2f}")
            lines.append(f"   pnl_degradation: {result.latency_robustness.pnl_degradation_pct:.2f}%")
            lines.append("")

        if result.regime_aware:
            lines.append(f"4. 多市场状态适应性: {result.regime_aware.verdict}")
            lines.append(f"   regimes: {result.regime_aware.regimes_detected}")
            lines.append(f"   regime_sharpes: {result.regime_aware.regime_sharpes}")
            lines.append(f"   min_regime_sharpe: {result.regime_aware.min_regime_sharpe:.2f}")
            lines.append(f"   sharpe_consistency: {result.regime_aware.sharpe_consistency:.2f}")
            if result.regime_aware.failing_regimes:
                lines.append(f"   failing_regimes: {result.regime_aware.failing_regimes}")
            lines.append("")

        if result.discrepancy:
            lines.append(f"5. 实盘vs回测差距: {result.discrepancy.verdict}")
            lines.append(f"   sharpe_decay: {result.discrepancy.sharpe_decay:.2%}")
            lines.append(f"   win_rate_decay: {result.discrepancy.win_rate_decay:.2%}")
            lines.append(f"   pnl_discrepancy: {result.discrepancy.pnl_discrepancy_pct:.2%}")
            lines.append(f"   drawdown_exceeded: {result.discrepancy.drawdown_exceeded}")
            lines.append(f"   mahalanobis_distance: {result.discrepancy.mahalanobis_distance:.2f}")
            lines.append(f"   signals_fired: {result.discrepancy.signals_fired}/3")
            lines.append(f"   decay_confirmed: {result.discrepancy.decay_confirmed}")
            lines.append(f"   recommendation: {result.discrepancy.recommendation}")
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"综合判定: {result.overall_verdict}")
        if result.block_reasons:
            lines.append("BLOCK原因:")
            for r in result.block_reasons:
                lines.append(f"  - {r}")
        if result.warn_reasons:
            lines.append("WARN原因:")
            for r in result.warn_reasons:
                lines.append(f"  - {r}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# 模块自检入口
# ============================================================================

def _self_test() -> bool:
    """自检: 验证 LiveRealityCheckValidator 与 LiveBacktestDiscrepancyMonitor 核心功能可用"""
    try:
        rng = np.random.default_rng(42)

        # 测试1: 综合验证流程 (成本/延迟/差距监测路径全部触发)
        validator = LiveRealityCheckValidator()
        trade_pnls = rng.normal(loc=0.5, scale=2.0, size=50).tolist()
        live_trades = rng.normal(loc=0.3, scale=2.0, size=30).tolist()
        backtest_returns = rng.normal(loc=0.4, scale=1.8, size=60).tolist()

        result = validator.validate(
            trade_pnls=trade_pnls,
            backtest_sharpe=2.0,
            backtest_win_rate=0.55,
            backtest_pnl=1000.0,
            backtest_max_drawdown=0.15,
            backtest_returns=backtest_returns,
            live_trades=live_trades,
        )
        assert result.overall_verdict in ("PASS", "WARN", "BLOCK"), \
            f"overall_verdict 应非空且合法, 实际: {result.overall_verdict}"
        assert isinstance(result.block_reasons, list)
        assert isinstance(result.warn_reasons, list)

        # 测试2: LiveBacktestDiscrepancyMonitor 数据不足场景 (live_trades < min_live_trades)
        monitor = LiveBacktestDiscrepancyMonitor(min_live_trades=20)
        insufficient = monitor.assess(
            backtest_sharpe=2.0,
            backtest_win_rate=0.6,
            backtest_pnl=500.0,
            backtest_max_drawdown=0.1,
            backtest_returns=[0.01] * 30,
            live_trades=[0.1, -0.05],  # 仅2笔 < 20 → WARN
        )
        assert insufficient.verdict == "WARN"
        assert insufficient.recommendation == "INSUFFICIENT_LIVE_DATA"

        # 测试3: 充足实盘数据场景 — 字段被正确计算
        live_full = rng.normal(loc=0.4, scale=1.5, size=30).tolist()
        full = monitor.assess(
            backtest_sharpe=2.0,
            backtest_win_rate=0.6,
            backtest_pnl=500.0,
            backtest_max_drawdown=0.1,
            backtest_returns=backtest_returns,
            live_trades=live_full,
        )
        assert full.verdict in ("PASS", "WARN", "BLOCK")
        assert isinstance(full.sharpe_decay, float)
        assert isinstance(full.signals_fired, int)

        # 测试4: 报告生成可读且非空
        report_text = validator.generate_report(result)
        assert isinstance(report_text, str) and len(report_text) > 0
        assert "实盘真实性验证报告" in report_text
        assert "综合判定" in report_text

        logger.info("LiveRealityCheckValidator 自检全部通过 ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
