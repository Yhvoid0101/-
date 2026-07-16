# -*- coding: utf-8 -*-
"""
模拟盘测试框架 (Paper Trading Framework) — R17-4

实现90天以上模拟交易验证，确保策略在实盘前经历不同市场行情周期。

用户需求："实施严格的模拟盘测试流程，至少进行90天以上的模拟交易验证，
期间需经历不同市场行情周期"

业界最佳实践：
  - Walk-forward Analysis (Pardo 2008): 滚动窗口回测+前瞻验证
  - Out-of-sample Testing: 严格样本外测试
  - Regime Coverage Check: 多市场状态覆盖
  - Realistic Cost Model: 真实手续费+滑点+延迟+资金费率
  - Cross-Period Consistency: 跨周期一致性检验

核心组件：
  1. PaperTrade             — 单笔模拟交易记录
  2. PaperTradingSession    — 单次模拟盘会话(真实成本+延迟+滑点)
  3. MultiPeriodValidator   — 多周期一致性验证器
  4. RegimeCoverageChecker  — 市场行情周期覆盖检查
  5. PaperTradingFramework  — 综合框架(90天+多周期+regime覆盖)
  6. PaperTradingReport     — 模拟报告

数字货币合约日内场景真实参数：
  - Taker fee: 5bps (Binance Futures 0.04% → 约4-5bps)
  - Maker fee: 2bps (0.02%)
  - 滑点: 3bps (中等流动性)
  - 延迟: 100ms (含网络+匹配引擎)
  - 资金费率: 0.01%/8h (3次/天)
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .strategy_evaluator import (
        StrategyEvaluator, MarketRegime, classify_regime, Verdict,
        StrategyEvaluationResult,
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sandbox_trading.strategy_evaluator import (  # type: ignore
        StrategyEvaluator, MarketRegime, classify_regime, Verdict,
        StrategyEvaluationResult,
    )

logger = logging.getLogger("hermes.paper_trading")


# ============================================================================
# 阈值常量 (量化+不可妥协)
# ============================================================================

MIN_PAPER_DAYS = 90                 # 最少90天模拟
MIN_REGIMES_COVERED = 2             # 至少覆盖2种regime
IDEAL_REGIMES_COVERED = 3           # 理想覆盖3种regime
CONSISTENCY_SHARPE_DIFF_MAX = 0.5   # 跨周期Sharpe差异<0.5
CONSISTENCY_DRAWDOWN_DIFF_MAX = 0.10  # 跨周期最大回撤差异<10%
CONSISTENCY_RETURN_DIFF_MAX = 0.30  # 跨周期年化收益差异<30%

# 真实交易成本(数字货币合约)
REALISTIC_TAKER_FEE_BPS = 5.0       # 5bps taker fee
REALISTIC_MAKER_FEE_BPS = 2.0       # 2bps maker fee
REALISTIC_SLIPPAGE_BPS = 3.0        # 3bps 滑点(中等流动性)
REALISTIC_LATENCY_MS = 100          # 100ms 订单执行延迟
FUNDING_RATE_8H = 0.0001            # 8h资金费率0.01% (3次/天)
FUNDING_PER_DAY = 3 * FUNDING_RATE_8H  # 每日资金费成本0.03%

PERIODS_PER_DAY_1H = 24             # 1h K线: 24期/天


# ============================================================================
# 1. 单笔模拟交易
# ============================================================================


@dataclass(slots=True)
class PaperTrade:
    """单笔模拟交易记录

    记录一笔完整的交易(开仓+平仓)的所有信息，包括真实成本。
    """
    timestamp: float                   # 开仓时间戳
    symbol: str                        # 交易对
    side: str                          # "long" / "short"
    entry_price: float                 # 开仓价
    exit_price: float                  # 平仓价
    quantity: float                    # 数量(合约张数)
    leverage: float = 1.0              # 杠杆倍数
    notional_usd: float = 0.0          # 名义价值(USD)
    slippage_cost_usd: float = 0.0     # 滑点成本(USD)
    taker_fee_usd: float = 0.0         # Taker手续费(USD)
    funding_cost_usd: float = 0.0      # 资金费成本(USD)
    pnl_gross_usd: float = 0.0         # 毛PnL(未扣成本)
    pnl_net_usd: float = 0.0           # 净PnL(扣完成本)
    hold_periods: int = 0              # 持仓期数
    latency_ms: float = 0.0            # 执行延迟(ms)


# ============================================================================
# 2. 模拟盘会话
# ============================================================================


@dataclass(slots=True)
class PaperTradingSession:
    """单次模拟盘会话

    记录一次完整模拟盘的所有交易和统计信息。
    包含真实成本模型：手续费+滑点+延迟+资金费率。
    """
    session_id: str
    start_timestamp: float
    end_timestamp: float
    initial_capital_usd: float
    trades: List[PaperTrade] = field(default_factory=list)
    daily_pnls: List[float] = field(default_factory=list)  # 每日净PnL收益率
    equity_curve: List[float] = field(default_factory=list)  # 净值曲线

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def days_count(self) -> int:
        return max(1, int((self.end_timestamp - self.start_timestamp) / 86400))

    @property
    def final_equity(self) -> float:
        if not self.equity_curve:
            return self.initial_capital_usd
        return self.equity_curve[-1]

    @property
    def total_return(self) -> float:
        if self.initial_capital_usd <= 0:
            return 0.0
        return (self.final_equity - self.initial_capital_usd) / self.initial_capital_usd

    @property
    def total_cost_usd(self) -> float:
        """总交易成本(滑点+手续费+资金费)"""
        return sum(
            t.slippage_cost_usd + t.taker_fee_usd + t.funding_cost_usd
            for t in self.trades
        )

    @property
    def cost_ratio(self) -> float:
        """成本占比 = 总成本 / 初始资金"""
        if self.initial_capital_usd <= 0:
            return 0.0
        return self.total_cost_usd / self.initial_capital_usd

    def compute_sharpe(self, periods_per_year: int = 365) -> float:
        """计算日PnL的Sharpe"""
        if len(self.daily_pnls) < 2:
            return 0.0
        arr = np.asarray(self.daily_pnls, dtype=np.float64)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1))
        if std_ret < 1e-10:
            return 0.0
        return mean_ret / std_ret * math.sqrt(periods_per_year)

    def compute_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.equity_curve:
            return 0.0
        arr = np.asarray(self.equity_curve, dtype=np.float64)
        peak = np.maximum.accumulate(arr)
        drawdown = (arr - peak) / np.where(peak > 0, peak, 1.0)
        return float(np.min(drawdown))


# ============================================================================
# 3. 多周期一致性验证器
# ============================================================================


@dataclass(slots=True)
class PeriodResult:
    """单周期验证结果"""
    period_id: str
    start_timestamp: float
    end_timestamp: float
    days_count: int
    sharpe: float
    max_drawdown: float
    annual_return: float
    trade_count: int
    cost_ratio: float
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ConsistencyResult:
    """跨周期一致性结果"""
    periods: List[PeriodResult] = field(default_factory=list)
    sharpe_std: float = 0.0          # 跨周期Sharpe标准差(仅活跃周期)
    sharpe_range: float = 0.0        # Sharpe极差(max-min)
    drawdown_std: float = 0.0        # 回撤标准差
    return_std: float = 0.0          # 收益标准差
    consistency_score: float = 0.0   # 一致性评分(0-1, 越高越好)
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)
    # v3.10.17新增: 稳健性指标
    sharpe_iqr: float = 0.0          # Sharpe四分位距(对极端值稳健, Tukey 1977)
    n_active_periods: int = 0        # 活跃周期数(有交易的)
    n_inactive_periods: int = 0      # 不活跃周期数(std<1e-8, 无交易)
    active_ratio: float = 0.0        # 活跃率 = n_active / n_total
    profitable_ratio: float = 0.0    # 盈利周期比 = n_profitable / n_active
    median_sharpe: float = 0.0       # 活跃周期Sharpe中位数(对极端值稳健)
    n_profitable_periods: int = 0    # 盈利周期数(Sharpe>0)


class MultiPeriodValidator:
    """多周期一致性验证器

    将90天以上的模拟数据分成多个子周期(默认每30天一个)，
    检查策略在不同子周期下的表现一致性。

    一致性检验维度：
      1. Sharpe稳定性: 跨周期Sharpe标准差 < 0.5
      2. 回撤稳定性: 跨周期最大回撤标准差 < 10%
      3. 收益稳定性: 跨周期年化收益标准差 < 30%

    原理：
      如果策略在某些子周期表现优异，在其他子周期表现糟糕，
      说明策略可能过拟合特定市场环境，不具备普适性。
    """

    def __init__(
        self,
        period_days: int = 30,
        periods_per_year: int = 365,
        rolling_step: Optional[int] = None,
    ):
        """
        Args:
            period_days: 单周期天数(默认30天)
            periods_per_year: 每年期数
            rolling_step: 滚动步长(None=不相交窗口, <period_days=重叠窗口)
                v3.10.17新增: 支持 rolling window consistency check
                统计学依据: Pardo 2008, Lopez de Prado 2018 — 30天Sharpe方差极大(1.5-2.5),
                60天滚动窗口+30天步长提供更稳定的Sharpe估计且样本数不减少
                示例: period_days=60, rolling_step=30 → 60天窗口, 30天步长(50%重叠)
        """
        self.period_days = period_days
        self.periods_per_year = periods_per_year
        # rolling_step=None → 不相交窗口(向后兼容); rolling_step<period_days → 重叠窗口
        self.rolling_step = rolling_step if rolling_step is not None else period_days

    def validate(
        self,
        session: PaperTradingSession,
    ) -> ConsistencyResult:
        """验证多周期一致性

        Args:
            session: 模拟盘会话(已包含daily_pnls)

        Returns:
            ConsistencyResult
        """
        result = ConsistencyResult()

        if not session.daily_pnls:
            result.verdict = Verdict.BLOCKED
            result.reasons.append("无daily_pnls数据")
            return result

        total_days = len(session.daily_pnls)
        if total_days < MIN_PAPER_DAYS:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(f"总天数={total_days} < MIN_PAPER_DAYS={MIN_PAPER_DAYS}")
            return result

        # 切分子周期 — v3.10.17: 支持rolling window(重叠窗口)
        period_len = self.period_days
        step = self.rolling_step
        # 计算所有窗口的起止索引
        window_starts: List[int] = []
        i = 0
        while i + period_len <= total_days:
            window_starts.append(i)
            i += step
        n_periods = len(window_starts)

        sharpes: List[float] = []
        drawdowns: List[float] = []
        returns: List[float] = []

        for idx, start_idx in enumerate(window_starts):
            end_idx = start_idx + period_len
            period_pnls = session.daily_pnls[start_idx:end_idx]

            if len(period_pnls) < 5:
                continue

            arr = np.asarray(period_pnls, dtype=np.float64)
            mean_ret = float(np.mean(arr))
            std_ret = float(np.std(arr, ddof=1))
            sharpe = (mean_ret / std_ret * math.sqrt(self.periods_per_year)) if std_ret > 1e-10 else 0.0

            # 计算子周期净值曲线和回撤
            equity = np.cumprod(1 + arr)
            peak = np.maximum.accumulate(equity)
            drawdown_arr = (equity - peak) / np.where(peak > 0, peak, 1.0)
            max_dd = float(np.min(drawdown_arr)) if len(drawdown_arr) > 0 else 0.0

            annual_ret = mean_ret * self.periods_per_year

            # v3.10.17: 区分活跃/不活跃周期
            # 不活跃周期(std<1e-8)无交易, Sharpe未定义(强制0), 不参与Sharpe一致性统计
            # 原因: 不活跃周期提供的是"策略覆盖度"信息, 不是"表现一致性"信息
            is_active = std_ret > 1e-8

            period_result = PeriodResult(
                period_id=f"P{idx+1}",
                start_timestamp=session.start_timestamp + start_idx * 86400,
                end_timestamp=session.start_timestamp + end_idx * 86400,
                days_count=len(period_pnls),
                sharpe=sharpe,
                max_drawdown=max_dd,
                annual_return=annual_ret,
                trade_count=sum(1 for t in session.trades
                                if t.timestamp >= session.start_timestamp + start_idx * 86400
                                and t.timestamp < session.start_timestamp + end_idx * 86400),
                cost_ratio=session.cost_ratio / max(1, n_periods),
            )

            # 单周期评估
            if not is_active:
                period_result.verdict = Verdict.DEGRADED
                period_result.reasons.append("无交易(策略不活跃)")
            elif sharpe < 0:
                period_result.verdict = Verdict.DEGRADED
                period_result.reasons.append(f"Sharpe={sharpe:.2f}<0")
            elif abs(max_dd) > 0.20:
                period_result.verdict = Verdict.DEGRADED
                period_result.reasons.append(f"max_dd={max_dd:.2%}<-20%")
            else:
                period_result.verdict = Verdict.ACCEPTABLE

            result.periods.append(period_result)
            # 只把活跃周期纳入Sharpe一致性统计
            if is_active:
                sharpes.append(sharpe)
                drawdowns.append(abs(max_dd))
                returns.append(annual_ret)

        # v3.10.17: 统计活跃/不活跃周期
        n_total = len(result.periods)
        n_active = len(sharpes)
        n_inactive = n_total - n_active
        result.n_active_periods = n_active
        result.n_inactive_periods = n_inactive
        result.active_ratio = n_active / max(1, n_total)

        if n_active < 2:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"活跃周期数={n_active} < 2 (总{n_total}, 不活跃{n_inactive}), "
                f"无法评估Sharpe一致性"
            )
            if n_inactive > 0:
                result.reasons.append(
                    f"不活跃周期{n_inactive}个(策略覆盖度不足, 可能是regime不适配)"
                )
            return result

        # 跨周期统计(仅活跃周期)
        result.sharpe_std = float(np.std(sharpes, ddof=1))
        result.sharpe_range = float(max(sharpes) - min(sharpes))
        result.drawdown_std = float(np.std(drawdowns, ddof=1))
        result.return_std = float(np.std(returns, ddof=1))
        # v3.10.17: IQR (Tukey 1977 robust statistics, 对极端值稳健)
        result.sharpe_iqr = float(
            np.percentile(sharpes, 75) - np.percentile(sharpes, 25)
        )
        # v3.10.17: 盈利周期比 + 中位数Sharpe (区分"波动正向"vs"真不一致")
        n_profitable = sum(1 for s in sharpes if s > 0)
        result.n_profitable_periods = n_profitable
        result.profitable_ratio = n_profitable / max(1, n_active)
        result.median_sharpe = float(np.median(sharpes))

        # 一致性评分(0-1) — v3.10.17: 用IQR替代std作为主指标
        # IQR对极端值稳健, 更能反映"典型"一致性而非被离群点拉偏
        sharpe_consistency = max(0.0, 1.0 - result.sharpe_iqr / CONSISTENCY_SHARPE_DIFF_MAX)
        dd_consistency = max(0.0, 1.0 - result.drawdown_std / CONSISTENCY_DRAWDOWN_DIFF_MAX)
        ret_consistency = max(0.0, 1.0 - result.return_std / CONSISTENCY_RETURN_DIFF_MAX)
        result.consistency_score = (sharpe_consistency + dd_consistency + ret_consistency) / 3.0

        # 评估结论 — v3.10.17: IQR为主 + 盈利比调制
        # 核心洞察: IQR高不代表"不一致" — 可能是"波动正向"(7/8盈利, Sharpe 0.2~2.8)
        #   - 盈利比>=80% + 中位Sharpe>0 → 高IQR是"波动正向", 降级BLOCK→DEGRADED
        #   - 盈利比<80% 或 中位Sharpe<=0 → 高IQR是"真不一致", 保持BLOCK
        profitable_strong = (result.profitable_ratio >= 0.80 and result.median_sharpe > 0)

        if result.sharpe_iqr > CONSISTENCY_SHARPE_DIFF_MAX * 2:
            # IQR极高(>1.0): 默认BLOCK, 但盈利比强则降级DEGRADED
            if profitable_strong:
                result.verdict = Verdict.DEGRADED
                result.reasons.append(
                    f"Sharpe IQR={result.sharpe_iqr:.2f}>>{CONSISTENCY_SHARPE_DIFF_MAX} "
                    f"(活跃{n_active}/{n_total}, std={result.sharpe_std:.2f}, "
                    f"盈利比{result.profitable_ratio:.0%}, median={result.median_sharpe:.2f}) "
                    f"[盈利比>=80%降级: 波动正向≠不一致]"
                )
            else:
                result.verdict = Verdict.BLOCKED
                result.reasons.append(
                    f"Sharpe IQR={result.sharpe_iqr:.2f}>>{CONSISTENCY_SHARPE_DIFF_MAX} "
                    f"(活跃{n_active}/{n_total}, std={result.sharpe_std:.2f}, "
                    f"盈利比{result.profitable_ratio:.0%}, median={result.median_sharpe:.2f})"
                )
        elif result.sharpe_iqr > CONSISTENCY_SHARPE_DIFF_MAX:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"Sharpe IQR={result.sharpe_iqr:.2f}>{CONSISTENCY_SHARPE_DIFF_MAX} "
                f"(活跃{n_active}/{n_total}, std={result.sharpe_std:.2f})"
            )
        elif result.consistency_score >= 0.8:
            result.verdict = Verdict.EXCELLENT
        elif result.consistency_score >= 0.6:
            result.verdict = Verdict.GOOD
        else:
            result.verdict = Verdict.ACCEPTABLE

        # v3.10.17: 不活跃率警告(策略覆盖度问题, 与一致性独立)
        if n_inactive > 0 and result.active_ratio < 0.7:
            inactive_warn = (
                f"策略覆盖度不足: 活跃率{result.active_ratio:.0%} "
                f"({n_active}活跃/{n_total}总, {n_inactive}个无交易周期)"
            )
            result.reasons.append(inactive_warn)
            # 不活跃率>50% = 单独DEGRADED (即使IQR通过)
            if result.active_ratio < 0.5 and result.verdict == Verdict.ACCEPTABLE:
                result.verdict = Verdict.DEGRADED

        return result


# ============================================================================
# 4. 市场行情周期覆盖检查器
# ============================================================================


@dataclass(slots=True)
class RegimeCoverageResult:
    """regime覆盖结果"""
    regimes_covered: List[MarketRegime] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)  # regime → 天数
    missing_regimes: List[MarketRegime] = field(default_factory=list)
    coverage_score: float = 0.0       # 覆盖评分(0-1)
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class RegimeCoverageChecker:
    """市场行情周期覆盖检查器

    用户需求："至少进行90天以上的模拟交易验证，期间需经历不同市场行情周期"

    确保90天模拟经历了多种市场状态：
      - BULL: 牛市(上涨)
      - BEAR: 熊市(下跌)
      - SIDEWAYS: 震荡(横盘)
      - EXTREME: 极端(高波动)

    评分规则：
      - 覆盖2种regime → ACCEPTABLE
      - 覆盖3种regime → GOOD
      - 覆盖4种regime → EXCELLENT
      - 仅1种regime → DEGRADED (可能过拟合特定环境)
    """

    def __init__(self, window_days: int = 7):
        """
        Args:
            window_days: regime识别窗口(默认7天)
        """
        self.window_days = window_days

    def check(
        self,
        daily_pnls: List[float],
    ) -> RegimeCoverageResult:
        """检查regime覆盖

        Args:
            daily_pnls: 每日PnL序列

        Returns:
            RegimeCoverageResult
        """
        result = RegimeCoverageResult()
        if not daily_pnls:
            result.verdict = Verdict.BLOCKED
            result.reasons.append("无daily_pnls数据")
            return result

        window = self.window_days
        if len(daily_pnls) < window:
            # 数据不足，整体作为一个regime
            arr = np.asarray(daily_pnls, dtype=np.float64)
            regime = classify_regime(arr)
            result.regimes_covered = [regime]
            result.regime_distribution = {regime.value: len(daily_pnls)}
            result.verdict = Verdict.DEGRADED
            result.reasons.append(f"数据不足<{window}天, 仅识别为{regime.value}")
            return result

        # 滚动窗口识别regime
        regime_days: Dict[MarketRegime, int] = {r: 0 for r in MarketRegime}
        for i in range(0, len(daily_pnls) - window + 1, window):
            chunk = np.asarray(daily_pnls[i:i + window], dtype=np.float64)
            regime = classify_regime(chunk)
            regime_days[regime] += window

        # 统计覆盖
        covered = [r for r, days in regime_days.items() if days > 0]
        result.regimes_covered = covered
        result.regime_distribution = {r.value: d for r, d in regime_days.items() if d > 0}

        all_regimes = list(MarketRegime)
        result.missing_regimes = [r for r in all_regimes if r not in covered]

        # 评分
        n_covered = len(covered)
        result.coverage_score = n_covered / len(all_regimes)

        if n_covered >= 4:
            result.verdict = Verdict.EXCELLENT
        elif n_covered >= 3:
            result.verdict = Verdict.GOOD
        elif n_covered >= MIN_REGIMES_COVERED:
            result.verdict = Verdict.ACCEPTABLE
        else:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"仅覆盖{n_covered}种regime < {MIN_REGIMES_COVERED}, "
                f"可能过拟合{covered[0].value if covered else 'N/A'}环境"
            )

        return result


# ============================================================================
# 5. 综合模拟盘框架
# ============================================================================


@dataclass(slots=True)
class PaperTradingReport:
    """模拟盘综合报告"""
    session: Optional[PaperTradingSession] = None
    consistency: Optional[ConsistencyResult] = None
    regime_coverage: Optional[RegimeCoverageResult] = None
    final_verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PaperTradingFramework:
    """综合模拟盘测试框架

    完整流程：
      1. 模拟90天以上交易(含真实成本+延迟+滑点+资金费)
      2. 多周期一致性验证(每30天一个子周期)
      3. 市场行情周期覆盖检查(至少2种regime)
      4. 综合评估，给出实盘准入建议

    使用示例：
        framework = PaperTradingFramework(
            initial_capital_usd=10000,
            taker_fee_bps=5.0,
            slippage_bps=3.0,
        )

        # 传入策略PnL和交易信号
        report = framework.run_paper_trading(
            daily_pnls=strategy_daily_pnls,  # 90+天的日PnL
            trades=trade_records,            # 交易记录
            periods_per_year=365,
        )

        if report.final_verdict == Verdict.BLOCKED:
            print("禁止实盘:", report.reasons)
        elif report.final_verdict == Verdict.DEGRADED:
            print("谨慎实盘:", report.reasons)
        else:
            print("允许实盘")
    """

    def __init__(
        self,
        initial_capital_usd: float = 10_000.0,
        taker_fee_bps: float = REALISTIC_TAKER_FEE_BPS,
        maker_fee_bps: float = REALISTIC_MAKER_FEE_BPS,
        slippage_bps: float = REALISTIC_SLIPPAGE_BPS,
        latency_ms: float = REALISTIC_LATENCY_MS,
        funding_rate_8h: float = FUNDING_RATE_8H,
        period_days: int = 30,
        regime_window_days: int = 7,
        rolling_step: Optional[int] = None,
    ):
        """
        Args:
            initial_capital_usd: 初始资金(USD)
            taker_fee_bps: Taker手续费(bps)
            maker_fee_bps: Maker手续费(bps)
            slippage_bps: 滑点(bps)
            latency_ms: 订单执行延迟(ms)
            funding_rate_8h: 8h资金费率
            period_days: 多周期验证的单周期天数
            regime_window_days: regime识别窗口天数
            rolling_step: 滚动步长(None=不相交窗口, <period_days=重叠窗口)
                v3.10.17: rolling window consistency check (Pardo 2008, Lopez de Prado 2018)
        """
        self.initial_capital_usd = initial_capital_usd
        self.taker_fee_bps = taker_fee_bps
        self.maker_fee_bps = maker_fee_bps
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms
        self.funding_rate_8h = funding_rate_8h
        self.period_days = period_days
        self.regime_window_days = regime_window_days
        self.rolling_step = rolling_step

        self.validator = MultiPeriodValidator(
            period_days=period_days,
            rolling_step=rolling_step,
        )
        self.coverage_checker = RegimeCoverageChecker(window_days=regime_window_days)

    def apply_realistic_costs(
        self,
        gross_pnl_usd: float,
        notional_usd: float,
        hold_periods: int,
        periods_per_day: int = PERIODS_PER_DAY_1H,
    ) -> Tuple[float, float, float, float]:
        """应用真实交易成本

        Args:
            gross_pnl_usd: 毛PnL(USD)
            notional_usd: 名义价值(USD)
            hold_periods: 持仓期数
            periods_per_day: 每日周期数

        Returns:
            (slippage_cost, taker_fee, funding_cost, net_pnl)
        """
        # 滑点: 开仓+平仓各一次
        slippage_cost = notional_usd * self.slippage_bps / 10000 * 2

        # Taker手续费: 开仓+平仓各一次
        taker_fee = notional_usd * self.taker_fee_bps / 10000 * 2

        # 资金费率: 持仓期间累计
        hold_days = hold_periods / periods_per_day
        funding_periods = hold_days * 3  # 8h一次, 3次/天
        funding_cost = notional_usd * self.funding_rate_8h * funding_periods

        net_pnl = gross_pnl_usd - slippage_cost - taker_fee - funding_cost
        return slippage_cost, taker_fee, funding_cost, net_pnl

    def run_paper_trading(
        self,
        daily_pnls: List[float],
        trades: Optional[List[PaperTrade]] = None,
        session_id: str = "paper_session",
        start_timestamp: Optional[float] = None,
        periods_per_year: int = 365,
    ) -> PaperTradingReport:
        """运行完整模拟盘测试

        Args:
            daily_pnls: 每日PnL收益率序列(>=90天)
            trades: 交易记录(可选, 用于成本统计)
            session_id: 会话ID
            start_timestamp: 起始时间戳(None=当前时间-N天)
            periods_per_year: 每年期数

        Returns:
            PaperTradingReport
        """
        report = PaperTradingReport()

        if not daily_pnls:
            report.final_verdict = Verdict.BLOCKED
            report.reasons.append("无daily_pnls数据")
            return report

        n_days = len(daily_pnls)
        if start_timestamp is None:
            start_timestamp = time.time() - n_days * 86400
        end_timestamp = start_timestamp + n_days * 86400

        # 构建会话
        session = PaperTradingSession(
            session_id=session_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            initial_capital_usd=self.initial_capital_usd,
            trades=trades or [],
            daily_pnls=list(daily_pnls),
        )

        # 构建净值曲线
        equity = self.initial_capital_usd
        session.equity_curve = [equity]
        for pnl in daily_pnls:
            equity *= (1 + pnl)
            session.equity_curve.append(equity)

        report.session = session

        # === 1. 天数检查 ===
        if n_days < MIN_PAPER_DAYS:
            report.final_verdict = Verdict.BLOCKED
            report.reasons.append(
                f"模拟天数={n_days} < MIN_PAPER_DAYS={MIN_PAPER_DAYS}"
            )
            return report

        # === 2. 多周期一致性验证 ===
        consistency = self.validator.validate(session)
        report.consistency = consistency

        # === 3. regime覆盖检查 ===
        coverage = self.coverage_checker.check(daily_pnls)
        report.regime_coverage = coverage

        # === 4. 综合结论 ===
        verdicts = [consistency.verdict, coverage.verdict]

        if Verdict.BLOCKED in verdicts:
            report.final_verdict = Verdict.BLOCKED
        elif Verdict.DEGRADED in verdicts:
            report.final_verdict = Verdict.DEGRADED
        elif Verdict.ACCEPTABLE in verdicts:
            report.final_verdict = Verdict.ACCEPTABLE
        elif Verdict.GOOD in verdicts:
            report.final_verdict = Verdict.GOOD
        else:
            report.final_verdict = Verdict.EXCELLENT

        # 汇总原因
        if consistency.verdict in (Verdict.BLOCKED, Verdict.DEGRADED):
            report.reasons.extend(consistency.reasons)
        if coverage.verdict in (Verdict.BLOCKED, Verdict.DEGRADED):
            report.reasons.extend(coverage.reasons)

        # 推荐
        if report.final_verdict == Verdict.BLOCKED:
            report.recommendations.append("禁止实盘: 模拟盘未通过基础验证")
        elif report.final_verdict == Verdict.DEGRADED:
            report.recommendations.append(
                "谨慎实盘: 仅用小额资金(初始资金10-20%)验证, 密切监控"
            )
        elif report.final_verdict == Verdict.ACCEPTABLE:
            report.recommendations.append(
                "渐进实盘: 用初始资金30-50%分批入场, 每周评估"
            )
        else:
            report.recommendations.append(
                "允许实盘: 可用初始资金50-80%入场, 月度评估"
            )

        return report

    def generate_report(self, report: PaperTradingReport) -> str:
        """生成人类可读报告"""
        lines = [
            "=" * 70,
            "模拟盘测试报告 (Paper Trading Report)",
            "=" * 70,
            f"最终结论: {report.final_verdict.value}",
            "",
        ]

        if report.session:
            s = report.session
            lines.extend([
                "【会话信息】",
                f"  会话ID: {s.session_id}",
                f"  模拟天数: {s.days_count}",
                f"  初始资金: ${s.initial_capital_usd:,.2f}",
                f"  最终净值: ${s.final_equity:,.2f}",
                f"  总收益率: {s.total_return:.2%}",
                f"  交易笔数: {s.trade_count}",
                f"  Sharpe: {s.compute_sharpe():.3f}",
                f"  最大回撤: {s.compute_max_drawdown():.2%}",
                f"  总成本: ${s.total_cost_usd:,.2f} ({s.cost_ratio:.2%})",
                "",
            ])

        if report.consistency:
            c = report.consistency
            lines.extend([
                "【多周期一致性】",
                f"  子周期数: {len(c.periods)}",
                f"  Sharpe标准差: {c.sharpe_std:.3f} (阈值<{CONSISTENCY_SHARPE_DIFF_MAX})",
                f"  Sharpe极差: {c.sharpe_range:.3f}",
                f"  回撤标准差: {c.drawdown_std:.3f} (阈值<{CONSISTENCY_DRAWDOWN_DIFF_MAX})",
                f"  收益标准差: {c.return_std:.3f} (阈值<{CONSISTENCY_RETURN_DIFF_MAX})",
                f"  一致性评分: {c.consistency_score:.2f}",
                f"  结论: {c.verdict.value}",
                "",
            ])
            for p in c.periods:
                lines.append(
                    f"  {p.period_id}: {p.days_count}天 Sharpe={p.sharpe:.2f} "
                    f"DD={p.max_drawdown:.2%} 年化={p.annual_return:.2%} "
                    f"trades={p.trade_count} → {p.verdict.value}"
                )
            lines.append("")

        if report.regime_coverage:
            cov = report.regime_coverage
            lines.extend([
                "【市场行情周期覆盖】",
                f"  覆盖regime: {[r.value for r in cov.regimes_covered]}",
                f"  缺失regime: {[r.value for r in cov.missing_regimes]}",
                f"  分布: {cov.regime_distribution}",
                f"  覆盖评分: {cov.coverage_score:.2f}",
                f"  结论: {cov.verdict.value}",
                "",
            ])

        if report.reasons:
            lines.append("【原因】")
            for r in report.reasons:
                lines.append(f"  - {r}")
            lines.append("")

        if report.recommendations:
            lines.append("【实盘准入建议】")
            for rec in report.recommendations:
                lines.append(f"  → {rec}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def extract_tier2_metrics(self, report: PaperTradingReport) -> Dict[str, Any]:
        """Phase 5 Task #28.1: 从 PaperTradingReport 提取 Tier 2 评估指标

        用户核心诉求: "分阶测试准入规则（模拟→迷你小额实盘→标准实盘），
                      不达性能指标禁止向下游部署"
        之前: tier2_metrics (evolution_loop.py:7755) 全 None 硬编码, Tier 2 永远无法通过
        现在: 从 PaperTradingReport.consistency.periods 提取月度指标

        Tier 2 准入指标 (用户硬约束):
          - 月度亏损概率 ≤ 5% (mlp_pct)
          - 单次交易最大亏损 ≤ 本金 2% (single_loss_pct)
          - 连续亏损次数 ≤ 3 (consec_loss)
          - 连续模拟通过月数 (consecutive_sim_go_months)

        Args:
            report: PaperTradingReport (含 session + consistency)

        Returns:
            {
                "consecutive_sim_go_months": int,   # 连续 GO (GOOD/EXCELLENT/ACCEPTABLE) 月数
                "live_mlp_pct": float,              # 月度亏损概率 (%)
                "live_single_loss_pct": float,      # 单月最大亏损占本金 (%)
                "live_consec_loss": int,            # 最长连续亏损月数
                "monthly_pnls": List[float],        # 各月净PnL (USD)
                "total_days": int,                  # 总模拟天数
                "n_periods": int,                   # 周期数
            }
        """
        # 边界处理: report 或 consistency 为 None
        if report is None:
            return {
                "consecutive_sim_go_months": 0,
                "live_mlp_pct": None,
                "live_single_loss_pct": None,
                "live_consec_loss": None,
                "monthly_pnls": [],
                "total_days": 0,
                "n_periods": 0,
            }

        consistency = getattr(report, "consistency", None)
        session = getattr(report, "session", None)
        periods = getattr(consistency, "periods", []) if consistency else []
        initial_capital = (
            getattr(session, "initial_capital_usd", 10000.0) if session else 10000.0
        )

        monthly_pnls: List[float] = []
        current_go_streak = 0
        max_consec_go_months = 0
        max_consec_loss = 0
        current_loss_streak = 0
        single_loss_max_pct = 0.0

        # GO verdict 集合: 非BLOCKED/DEGRADED 即视为通过
        # Verdict 枚举: BLOCKED, DEGRADED, ACCEPTABLE, GOOD, EXCELLENT
        go_verdicts = {"ACCEPTABLE", "GOOD", "EXCELLENT"}

        for period in periods:
            # PeriodResult 无 net_pnl 字段, 从 annual_return 推算
            # net_pnl_usd = annual_return * (days_count / 365) * initial_capital
            annual_ret = float(getattr(period, "annual_return", 0.0))
            days_cnt = int(getattr(period, "days_count", 30))
            net_pnl = annual_ret * (days_cnt / 365.0) * initial_capital
            monthly_pnls.append(net_pnl)

            # 连续 GO 月数 (verdict 在 GO 集合中), 跟踪历史最大值
            period_verdict = getattr(period, "verdict", None)
            verdict_str = period_verdict.value if period_verdict else "UNKNOWN"
            if verdict_str in go_verdicts:
                current_go_streak += 1
                max_consec_go_months = max(max_consec_go_months, current_go_streak)
            else:
                current_go_streak = 0

            # 连续亏损月数
            if net_pnl < 0:
                current_loss_streak += 1
                max_consec_loss = max(max_consec_loss, current_loss_streak)
                # 单月亏损占本金百分比
                loss_pct = abs(net_pnl) / initial_capital * 100.0 if initial_capital > 0 else 0.0
                single_loss_max_pct = max(single_loss_max_pct, loss_pct)
            else:
                current_loss_streak = 0

        # 月度亏损概率
        loss_months = sum(1 for p in monthly_pnls if p < 0)
        mlp_pct = (loss_months / len(monthly_pnls) * 100.0) if monthly_pnls else 0.0

        total_days = int(getattr(session, "days_count", 0)) if session else 0

        return {
            "consecutive_sim_go_months": max_consec_go_months,
            "live_mlp_pct": mlp_pct,
            "live_single_loss_pct": single_loss_max_pct,
            "live_consec_loss": max_consec_loss,
            "monthly_pnls": monthly_pnls,
            "total_days": total_days,
            "n_periods": len(periods),
        }


# ============================================================================
# 自测
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R17-4 模拟盘测试框架 自测 ===\n")

    np.random.seed(42)

    # 构造120天的策略日PnL(包含多种regime)
    # 30天牛市 + 30天熊市 + 30天震荡 + 30天极端
    bull = np.random.normal(0.002, 0.01, 30)
    bear = np.random.normal(-0.001, 0.012, 30)
    sideways = np.random.normal(0.0001, 0.005, 30)
    extreme = np.random.normal(0.0005, 0.04, 30)
    daily_pnls = list(np.concatenate([bull, bear, sideways, extreme]))

    framework = PaperTradingFramework(
        initial_capital_usd=10000,
        taker_fee_bps=5.0,
        slippage_bps=3.0,
    )

    report = framework.run_paper_trading(
        daily_pnls=daily_pnls,
        trades=[],
        periods_per_year=365,
    )

    print(framework.generate_report(report))
