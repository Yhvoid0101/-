# -*- coding: utf-8 -*-
"""
模拟vs实盘对比分析模块 (Live-Paper Comparator) — R17-5

实现"防模拟牛逼实盘亏钱"保障机制：深入分析模拟与实盘差异的根本原因，
建立模拟与实盘业绩对比分析框架，设置关键指标差异预警阈值。

用户需求：
  "深入分析模拟与实盘差异的根本原因，包括但不限于滑点、流动性、
   订单执行延迟、数据延迟等因素"
  "建立模拟与实盘业绩对比分析框架，设置关键指标差异预警阈值"

业界最佳实践：
  - Implementation Shortfall (Perold 1988): 决策价到实际成交价的总成本分解
  - Slippage Decomposition: 显性成本(手续费) + 隐性成本(滑点+市场冲击)
  - Live-Paper Spread Monitor: 实时监控模拟与实盘PnL差异
  - TCA (Transaction Cost Analysis): 交易成本分析标准框架

核心组件：
  1. TradePair               — 模拟交易+实盘交易配对
  2. DiscrepancyAnalyzer     — 差异分析器(价格/时间/数量/PnL)
  3. KPIDiffMonitor          — 关键指标差异监控(Sharpe/Return/DD/WinRate)
  4. RootCauseAnalyzer       — 根因分析(滑点/延迟/流动性归因)
  5. LivePaperComparator     — 综合对比器
  6. ComparisonReport        — 对比报告

差异预警阈值 (量化+不可妥协)：
  SHARPE_DIFF_WARN          = 0.3    # Sharpe差异>0.3 警告
  SHARPE_DIFF_BLOCK         = 0.8    # Sharpe差异>0.8 阻塞
  RETURN_DIFF_WARN          = 0.10   # 收益差异>10% 警告
  RETURN_DIFF_BLOCK         = 0.30   # 收益差异>30% 阻塞
  DRAWDOWN_DIFF_WARN        = 0.05   # 回撤差异>5% 警告
  SLIPPAGE_DIFF_WARN_BPS    = 2.0    # 滑点差异>2bps 警告
  LATENCY_DIFF_WARN_MS      = 50     # 延迟差异>50ms 警告
  WINRATE_DIFF_WARN         = 0.05   # 胜率差异>5% 警告
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from .strategy_evaluator import Verdict
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sandbox_trading.strategy_evaluator import Verdict  # type: ignore

logger = logging.getLogger("hermes.live_paper_comparator")


# ============================================================================
# 阈值常量 (量化+不可妥协)
# ============================================================================

# KPI差异预警阈值
SHARPE_DIFF_WARN = 0.3           # Sharpe差异>0.3 警告
SHARPE_DIFF_BLOCK = 0.8          # Sharpe差异>0.8 阻塞(严重背离)
RETURN_DIFF_WARN = 0.10          # 收益差异>10% 警告
RETURN_DIFF_BLOCK = 0.30         # 收益差异>30% 阻塞
DRAWDOWN_DIFF_WARN = 0.05        # 回撤差异>5% 警告
DRAWDOWN_DIFF_BLOCK = 0.15       # 回撤差异>15% 阻塞
WINRATE_DIFF_WARN = 0.05         # 胜率差异>5% 警告
WINRATE_DIFF_BLOCK = 0.15        # 胜率差异>15% 阻塞

# 交易层面差异阈值
SLIPPAGE_DIFF_WARN_BPS = 2.0     # 滑点差异>2bps 警告
SLIPPAGE_DIFF_BLOCK_BPS = 10.0   # 滑点差异>10bps 阻塞
LATENCY_DIFF_WARN_MS = 50        # 延迟差异>50ms 警告
LATENCY_DIFF_BLOCK_MS = 500      # 延迟差异>500ms 阻塞
QUANTITY_FILL_RATIO_WARN = 0.95  # 成交率<95% 警告
QUANTITY_FILL_RATIO_BLOCK = 0.80 # 成交率<80% 阻塞


# ============================================================================
# 1. 交易配对
# ============================================================================


@dataclass(slots=True)
class TradeRecord:
    """单笔交易记录(模拟或实盘通用)"""
    trade_id: str
    timestamp: float
    symbol: str
    side: str                          # "long"/"short"
    price: float                       # 成交价
    quantity: float                    # 成交数量
    fee_usd: float = 0.0               # 手续费(USD)
    slippage_bps: float = 0.0          # 滑点(bps)
    latency_ms: float = 0.0            # 执行延迟(ms)
    pnl_usd: float = 0.0               # 该笔交易PnL(USD)
    source: str = "paper"              # "paper"/"live"
    # R19-2 WARN-4修复: 扩展归因维度所需字段(可选,缺省时降级为估算)
    leverage: int = 1                       # 杠杆倍数(用于强平漂移归因)
    funding_rate_per_8h: float = 0.0        # 资金费率/8h(用于资金费率反转归因)
    holding_period_hours: float = 0.0       # 持仓时长(小时,用于资金费率累计)
    is_rate_limited: bool = False           # 是否触发429限流(用于API限流归因)


@dataclass(slots=True)
class TradePair:
    """模拟交易+实盘交易配对"""
    paper_trade: TradeRecord
    live_trade: Optional[TradeRecord] = None  # 实盘可能缺失(未成交)

    @property
    def is_matched(self) -> bool:
        return self.live_trade is not None

    @property
    def price_diff_bps(self) -> float:
        """价格差异(bps)"""
        if not self.live_trade:
            return 0.0
        if self.paper_trade.price <= 0:
            return 0.0
        return (self.live_trade.price - self.paper_trade.price) / self.paper_trade.price * 10000

    @property
    def quantity_fill_ratio(self) -> float:
        """成交率"""
        if not self.live_trade:
            return 0.0
        if self.paper_trade.quantity <= 0:
            return 1.0
        return self.live_trade.quantity / self.paper_trade.quantity

    @property
    def latency_diff_ms(self) -> float:
        """延迟差异(ms)"""
        if not self.live_trade:
            return 0.0
        return self.live_trade.latency_ms - self.paper_trade.latency_ms

    @property
    def slippage_diff_bps(self) -> float:
        """滑点差异(bps)"""
        if not self.live_trade:
            return 0.0
        return self.live_trade.slippage_bps - self.paper_trade.slippage_bps

    @property
    def pnl_diff_usd(self) -> float:
        """PnL差异(USD)"""
        if not self.live_trade:
            return -self.paper_trade.pnl_usd
        return self.live_trade.pnl_usd - self.paper_trade.pnl_usd


# ============================================================================
# 2. 差异分析器
# ============================================================================


@dataclass(slots=True)
class DiscrepancyResult:
    """差异分析结果"""
    total_pairs: int = 0
    matched_pairs: int = 0
    unmatched_paper_count: int = 0       # 模拟有但实盘没有
    avg_price_diff_bps: float = 0.0
    max_price_diff_bps: float = 0.0
    avg_slippage_diff_bps: float = 0.0
    avg_latency_diff_ms: float = 0.0
    avg_fill_ratio: float = 0.0
    total_pnl_diff_usd: float = 0.0
    avg_pnl_diff_usd: float = 0.0
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class DiscrepancyAnalyzer:
    """差异分析器

    分析模拟交易与实盘交易在以下维度的差异：
      1. 价格差异(滑点表现)
      2. 时间差异(执行延迟)
      3. 数量差异(部分成交)
      4. PnL差异(最终业绩差异)

    原理：
      模拟盘假设完美成交，实盘会有滑点、延迟、部分成交等问题。
      差异越大 → 模拟越失真 → 实盘风险越高。
    """

    def analyze(self, pairs: List[TradePair]) -> DiscrepancyResult:
        """分析交易配对差异

        Args:
            pairs: 交易配对列表

        Returns:
            DiscrepancyResult
        """
        result = DiscrepancyResult()
        result.total_pairs = len(pairs)
        if not pairs:
            result.verdict = Verdict.BLOCKED
            result.reasons.append("无交易配对")
            return result

        matched = [p for p in pairs if p.is_matched]
        result.matched_pairs = len(matched)
        result.unmatched_paper_count = result.total_pairs - len(matched)

        if not matched:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(f"全部{result.total_pairs}笔模拟交易无实盘对应")
            return result

        # 价格差异
        price_diffs = [abs(p.price_diff_bps) for p in matched]
        result.avg_price_diff_bps = float(np.mean(price_diffs))
        result.max_price_diff_bps = float(np.max(price_diffs))

        # 滑点差异
        slip_diffs = [p.slippage_diff_bps for p in matched]
        result.avg_slippage_diff_bps = float(np.mean(slip_diffs))

        # 延迟差异
        latency_diffs = [p.latency_diff_ms for p in matched]
        result.avg_latency_diff_ms = float(np.mean(latency_diffs))

        # 成交率
        fill_ratios = [p.quantity_fill_ratio for p in matched]
        result.avg_fill_ratio = float(np.mean(fill_ratios))

        # PnL差异
        pnl_diffs = [p.pnl_diff_usd for p in matched]
        result.total_pnl_diff_usd = float(np.sum(pnl_diffs))
        result.avg_pnl_diff_usd = float(np.mean(pnl_diffs))

        # 评估结论
        if result.max_price_diff_bps > SLIPPAGE_DIFF_BLOCK_BPS:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"最大价格差异={result.max_price_diff_bps:.1f}bps > {SLIPPAGE_DIFF_BLOCK_BPS}bps"
            )
        elif result.avg_latency_diff_ms > LATENCY_DIFF_BLOCK_MS:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"平均延迟差异={result.avg_latency_diff_ms:.0f}ms > {LATENCY_DIFF_BLOCK_MS}ms"
            )
        elif result.avg_fill_ratio < QUANTITY_FILL_RATIO_BLOCK:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"平均成交率={result.avg_fill_ratio:.1%} < {QUANTITY_FILL_RATIO_BLOCK:.0%}"
            )
        elif result.avg_slippage_diff_bps > SLIPPAGE_DIFF_WARN_BPS:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"平均滑点差异={result.avg_slippage_diff_bps:.1f}bps > {SLIPPAGE_DIFF_WARN_BPS}bps"
            )
        elif result.avg_latency_diff_ms > LATENCY_DIFF_WARN_MS:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"平均延迟差异={result.avg_latency_diff_ms:.0f}ms > {LATENCY_DIFF_WARN_MS}ms"
            )
        elif result.avg_fill_ratio < QUANTITY_FILL_RATIO_WARN:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"平均成交率={result.avg_fill_ratio:.1%} < {QUANTITY_FILL_RATIO_WARN:.0%}"
            )
        elif result.unmatched_paper_count > len(matched) * 0.5:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(
                f"未匹配率={result.unmatched_paper_count}/{result.total_pairs} > 50%"
            )
        else:
            result.verdict = Verdict.GOOD

        return result


# ============================================================================
# 3. KPI差异监控
# ============================================================================


@dataclass(slots=True)
class KPIDiffResult:
    """KPI差异结果"""
    paper_sharpe: float = 0.0
    live_sharpe: float = 0.0
    sharpe_diff: float = 0.0
    paper_return: float = 0.0
    live_return: float = 0.0
    return_diff: float = 0.0
    paper_max_drawdown: float = 0.0
    live_max_drawdown: float = 0.0
    drawdown_diff: float = 0.0
    paper_winrate: float = 0.0
    live_winrate: float = 0.0
    winrate_diff: float = 0.0
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class KPIDiffMonitor:
    """关键指标差异监控

    监控模拟盘与实盘在以下KPI上的差异：
      1. Sharpe ratio差异
      2. 收益率差异
      3. 最大回撤差异
      4. 胜率差异

    预警逻辑：
      |diff| > WARN阈值 → DEGRADED
      |diff| > BLOCK阈值 → BLOCKED
    """

    def compare(
        self,
        paper_pnls: List[float],
        live_pnls: List[float],
        periods_per_year: int = 365,
    ) -> KPIDiffResult:
        """对比模拟与实盘KPI

        Args:
            paper_pnls: 模拟盘每日PnL
            live_pnls: 实盘每日PnL
            periods_per_year: 每年期数

        Returns:
            KPIDiffResult
        """
        result = KPIDiffResult()

        if not paper_pnls or not live_pnls:
            result.verdict = Verdict.BLOCKED
            result.reasons.append("PnL数据为空")
            return result

        # 对齐长度
        min_len = min(len(paper_pnls), len(live_pnls))
        if min_len < 5:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(f"对齐后PnL长度={min_len}<5")
            return result

        paper_arr = np.asarray(paper_pnls[:min_len], dtype=np.float64)
        live_arr = np.asarray(live_pnls[:min_len], dtype=np.float64)

        # Sharpe
        def compute_sharpe(arr):
            if len(arr) < 2:
                return 0.0
            mean_r = float(np.mean(arr))
            std_r = float(np.std(arr, ddof=1))
            return (mean_r / std_r * math.sqrt(periods_per_year)) if std_r > 1e-10 else 0.0

        result.paper_sharpe = compute_sharpe(paper_arr)
        result.live_sharpe = compute_sharpe(live_arr)
        result.sharpe_diff = result.live_sharpe - result.paper_sharpe

        # 收益率
        result.paper_return = float(np.sum(paper_arr))
        result.live_return = float(np.sum(live_arr))
        result.return_diff = result.live_return - result.paper_return

        # 最大回撤
        def compute_max_dd(arr):
            equity = np.cumprod(1 + arr)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
            return float(np.min(dd)) if len(dd) > 0 else 0.0

        result.paper_max_drawdown = compute_max_dd(paper_arr)
        result.live_max_drawdown = compute_max_dd(live_arr)
        result.drawdown_diff = abs(result.live_max_drawdown - result.paper_max_drawdown)

        # 胜率
        result.paper_winrate = float(np.mean(paper_arr > 0))
        result.live_winrate = float(np.mean(live_arr > 0))
        result.winrate_diff = abs(result.live_winrate - result.paper_winrate)

        # 评估结论
        if abs(result.sharpe_diff) > SHARPE_DIFF_BLOCK:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"Sharpe差异={abs(result.sharpe_diff):.2f} > {SHARPE_DIFF_BLOCK}"
            )
        elif abs(result.return_diff) > RETURN_DIFF_BLOCK:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"收益差异={abs(result.return_diff):.1%} > {RETURN_DIFF_BLOCK:.0%}"
            )
        elif result.drawdown_diff > DRAWDOWN_DIFF_BLOCK:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"回撤差异={result.drawdown_diff:.1%} > {DRAWDOWN_DIFF_BLOCK:.0%}"
            )
        elif result.winrate_diff > WINRATE_DIFF_BLOCK:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"胜率差异={result.winrate_diff:.1%} > {WINRATE_DIFF_BLOCK:.0%}"
            )
        elif abs(result.sharpe_diff) > SHARPE_DIFF_WARN:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(f"Sharpe差异={abs(result.sharpe_diff):.2f} > {SHARPE_DIFF_WARN}")
        elif abs(result.return_diff) > RETURN_DIFF_WARN:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(f"收益差异={abs(result.return_diff):.1%} > {RETURN_DIFF_WARN:.0%}")
        elif result.drawdown_diff > DRAWDOWN_DIFF_WARN:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(f"回撤差异={result.drawdown_diff:.1%} > {DRAWDOWN_DIFF_WARN:.0%}")
        elif result.winrate_diff > WINRATE_DIFF_WARN:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(f"胜率差异={result.winrate_diff:.1%} > {WINRATE_DIFF_WARN:.0%}")
        else:
            result.verdict = Verdict.GOOD

        return result


# ============================================================================
# 4. 根因分析
# ============================================================================


@dataclass(slots=True)
class RootCauseResult:
    """根因分析结果"""
    slippage_attribution_pct: float = 0.0    # 滑点归因占比(%)
    latency_attribution_pct: float = 0.0     # 延迟归因占比(%)
    liquidity_attribution_pct: float = 0.0   # 流动性归因占比(%)
    fee_attribution_pct: float = 0.0         # 手续费归因占比(%)
    # R19-2 WARN-4修复: 扩展3类归因维度(针对数字货币合约场景)
    funding_rate_attribution_pct: float = 0.0     # 资金费率反转归因占比(%)
    liquidation_drift_attribution_pct: float = 0.0  # 强平漂移归因占比(%)
    api_rate_limit_attribution_pct: float = 0.0   # API限流(429)归因占比(%)
    other_attribution_pct: float = 0.0       # 其他归因占比(%)
    primary_cause: str = ""                  # 主要原因
    secondary_cause: str = ""                # 次要原因
    verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)


class RootCauseAnalyzer:
    """根因分析器

    分析模拟与实盘PnL差异的根本原因，归因到8个维度(R19-2扩展):
      1. 滑点 (Slippage): 实际成交价 vs 模拟价差异
      2. 延迟 (Latency): 信号生成到成交的时间差(排除429限流)
      3. 流动性 (Liquidity): 部分成交、未成交
      4. 手续费 (Fee): 实际费率 vs 模拟费率
      5. 资金费率反转 (Funding Rate Reversal): 持仓期间费率反向 [R19-2新增]
         - 来源: Binance永续合约资金费率每8h结算,极端行情可反转
         - 实盘后果: delta-neutral套利策略年化15-40%收益可被反转吞没
      6. 强平漂移 (Liquidation Drift): 高杠杆下强平价格漂移 [R19-2新增]
         - 来源: 杠杆越高强平越近,实盘剧烈波动可触发强制减仓
         - 实盘后果: 模拟未模拟强平→实盘强平→PnL差异巨大
      7. API限流 (API Rate Limit / 429): 请求超限触发延迟 [R19-2新增]
         - 来源: Binance 1200 weight/min,超限触发429需重试1-5s
         - 实盘后果: 信号过期成交或订单丢失
      8. 其他 (Other): 无法归因的残差

    归因方法：
      total_diff = Σ(live_pnl - paper_pnl)
      slippage_cost = Σ(price_diff × quantity)
      latency_cost = 估算(延迟期间价格变动 × 数量, 排除>1s的429事件)
      liquidity_cost = Σ(未成交部分 × 模拟PnL)
      fee_diff = Σ(live_fee - paper_fee)
      funding_rate_cost = Σ(funding_rate × notional × holding_periods)
      liquidation_drift_cost = Σ(强平价格漂移 × leverage × quantity)
      api_rate_limit_cost = Σ(429延迟期间价格变动 × quantity)
      other = total_diff - 上述7类
    """

    def analyze(
        self,
        pairs: List[TradePair],
        discrepancy: DiscrepancyResult,
    ) -> RootCauseResult:
        """分析差异根因

        Args:
            pairs: 交易配对列表
            discrepancy: 差异分析结果

        Returns:
            RootCauseResult
        """
        result = RootCauseResult()

        if not pairs:
            result.verdict = Verdict.BLOCKED
            result.reasons.append("无交易对, 无法归因")
            return result

        # 全部未匹配场景: 归因为流动性主导
        if discrepancy.matched_pairs == 0:
            total_cost = abs(discrepancy.total_pnl_diff_usd)
            liquidity_cost = sum(abs(p.paper_trade.pnl_usd) for p in pairs if not p.is_matched)
            if total_cost > 1e-6:
                result.liquidity_attribution_pct = min(100.0, liquidity_cost / total_cost * 100)
                result.other_attribution_pct = max(0, 100 - result.liquidity_attribution_pct)
            else:
                result.liquidity_attribution_pct = 100.0
            result.primary_cause = "流动性"
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"全部{len(pairs)}笔交易未匹配, 流动性主导(占比{result.liquidity_attribution_pct:.0f}%)"
            )
            return result

        # 计算各归因成本(USD)
        slippage_cost = 0.0
        latency_cost = 0.0
        liquidity_cost = 0.0
        fee_diff = 0.0
        # R19-2 WARN-4修复: 新增3类归因成本
        funding_rate_cost = 0.0       # 资金费率反转
        liquidation_drift_cost = 0.0  # 强平漂移
        api_rate_limit_cost = 0.0     # API限流(429)

        for pair in pairs:
            if pair.is_matched:
                # 滑点成本 = 价格差异(bps) × 名义价值 / 10000
                notional = pair.paper_trade.price * pair.paper_trade.quantity
                slippage_cost += abs(pair.price_diff_bps) * notional / 10000

                # 延迟成本(排除429限流事件, 避免双重计算)
                # R19-2: 429限流(latency>1000ms)单独归因, 不计入普通延迟
                latency_diff = abs(pair.latency_diff_ms)
                is_429_event = (
                    pair.live_trade.is_rate_limited
                    or pair.live_trade.latency_ms > 1000.0
                    or pair.paper_trade.is_rate_limited
                )
                if is_429_event:
                    # 429限流成本 = 延迟期间价格变动 × 数量
                    api_rate_limit_cost += latency_diff * notional * 0.0001 / 100
                else:
                    # 普通延迟成本 = 延迟差异 × 数量 × 价格变动率(估算0.01%/100ms)
                    latency_cost += latency_diff * notional * 0.0001 / 100

                # 手续费差异
                fee_diff += pair.live_trade.fee_usd - pair.paper_trade.fee_usd

                # R19-2: 资金费率反转成本
                # 来源: Binance永续每8h结算, 持仓期间累计资金费率
                # 当资金费率为正(多头付空头收)且模拟未计入时, 实盘成本 = rate × notional × periods
                funding_rate = pair.live_trade.funding_rate_per_8h or pair.paper_trade.funding_rate_per_8h
                holding_hours = pair.live_trade.holding_period_hours or pair.paper_trade.holding_period_hours
                if funding_rate != 0.0 and holding_hours > 0.0:
                    funding_periods = holding_hours / 8.0
                    # 多头付正费率, 空头收正费率; 模拟常忽略此项
                    side_sign = 1.0 if pair.paper_trade.side == "long" else -1.0
                    funding_rate_cost += abs(funding_rate * notional * funding_periods * side_sign)

                # R19-2: 强平漂移成本
                # 来源: 高杠杆下强平价格漂移, 模拟常未模拟强平事件
                # 估算: 当滑点 > 100/leverage bps时, 视为接近强平, 损失放大leverage倍
                leverage = max(1, pair.live_trade.leverage or pair.paper_trade.leverage or 1)
                if leverage > 1:
                    # 强平距离阈值 ≈ 100/leverage bps (例: 10x杠杆, 10bps即1%强平距离)
                    liq_threshold_bps = 100.0 / leverage
                    if abs(pair.price_diff_bps) > liq_threshold_bps:
                        # 强平漂移损失 = 超出阈值部分 × leverage × notional
                        excess_bps = abs(pair.price_diff_bps) - liq_threshold_bps
                        liquidation_drift_cost += excess_bps * leverage * notional / 10000
            else:
                # 流动性成本 = 未成交部分损失的PnL
                liquidity_cost += abs(pair.paper_trade.pnl_usd)

        total_cost = abs(discrepancy.total_pnl_diff_usd)
        if total_cost < 1e-6:
            # 差异为0, 无需归因
            result.slippage_attribution_pct = 0
            result.latency_attribution_pct = 0
            result.liquidity_attribution_pct = 0
            result.fee_attribution_pct = 0
            result.funding_rate_attribution_pct = 0
            result.liquidation_drift_attribution_pct = 0
            result.api_rate_limit_attribution_pct = 0
            result.other_attribution_pct = 0
            result.primary_cause = "无显著差异"
            result.verdict = Verdict.EXCELLENT
            return result

        # 归因占比
        result.slippage_attribution_pct = slippage_cost / total_cost * 100
        result.latency_attribution_pct = latency_cost / total_cost * 100
        result.liquidity_attribution_pct = liquidity_cost / total_cost * 100
        result.fee_attribution_pct = fee_diff / total_cost * 100
        result.funding_rate_attribution_pct = funding_rate_cost / total_cost * 100
        result.liquidation_drift_attribution_pct = liquidation_drift_cost / total_cost * 100
        result.api_rate_limit_attribution_pct = api_rate_limit_cost / total_cost * 100
        result.other_attribution_pct = max(0, 100 - (
            result.slippage_attribution_pct
            + result.latency_attribution_pct
            + result.liquidity_attribution_pct
            + result.fee_attribution_pct
            + result.funding_rate_attribution_pct
            + result.liquidation_drift_attribution_pct
            + result.api_rate_limit_attribution_pct
        ))

        # 主要原因(R19-2: 8类归因排序)
        attributions = [
            ("滑点", result.slippage_attribution_pct),
            ("延迟", result.latency_attribution_pct),
            ("流动性", result.liquidity_attribution_pct),
            ("手续费", result.fee_attribution_pct),
            ("资金费率反转", result.funding_rate_attribution_pct),
            ("强平漂移", result.liquidation_drift_attribution_pct),
            ("API限流", result.api_rate_limit_attribution_pct),
            ("其他", result.other_attribution_pct),
        ]
        attributions.sort(key=lambda x: x[1], reverse=True)
        result.primary_cause = attributions[0][0]
        result.secondary_cause = attributions[1][0] if len(attributions) > 1 else ""

        # 评估(R19-2: 资金费率/强平漂移主导也视为BLOCKED, 合约交易致命风险)
        if result.primary_cause == "流动性" and attributions[0][1] > 50:
            result.verdict = Verdict.BLOCKED
            result.reasons.append(f"流动性主导(占比{attributions[0][1]:.0f}%), 实盘成交不可靠")
        elif result.primary_cause == "强平漂移" and attributions[0][1] > 30:
            # R19-2: 强平漂移>30%即BLOCKED(合约交易致命风险)
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"强平漂移主导(占比{attributions[0][1]:.0f}%), 杠杆过高导致实盘强平, "
                f"模拟未模拟强平事件"
            )
        elif result.primary_cause == "资金费率反转" and attributions[0][1] > 30:
            # R19-2: 资金费率反转>30%即BLOCKED(套利策略失效)
            result.verdict = Verdict.BLOCKED
            result.reasons.append(
                f"资金费率反转主导(占比{attributions[0][1]:.0f}%), delta-neutral套利失效, "
                f"模拟未计入资金费率成本"
            )
        elif attributions[0][1] > 70:
            result.verdict = Verdict.DEGRADED
            result.reasons.append(f"{result.primary_cause}主导(占比{attributions[0][1]:.0f}%), 需优化")
        else:
            result.verdict = Verdict.ACCEPTABLE

        return result


# ============================================================================
# 5. 综合对比器
# ============================================================================


@dataclass(slots=True)
class ComparisonReport:
    """对比报告"""
    discrepancy: Optional[DiscrepancyResult] = None
    kpi_diff: Optional[KPIDiffResult] = None
    root_cause: Optional[RootCauseResult] = None
    final_verdict: Verdict = Verdict.ACCEPTABLE
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class LivePaperComparator:
    """模拟vs实盘综合对比器

    完整流程：
      1. 交易配对(模拟交易 ↔ 实盘交易)
      2. 差异分析(价格/时间/数量/PnL)
      3. KPI对比(Sharpe/Return/DD/WinRate)
      4. 根因分析(滑点/延迟/流动性/手续费归因)
      5. 综合评估，给出实盘准入建议

    使用示例：
        comparator = LivePaperComparator()

        pairs = comparator.match_trades(paper_trades, live_trades)
        report = comparator.compare(
            pairs=pairs,
            paper_pnls=paper_daily_pnls,
            live_pnls=live_daily_pnls,
        )

        if report.final_verdict == Verdict.BLOCKED:
            print("实盘严重背离模拟, 停止实盘:", report.reasons)
    """

    def __init__(self):
        self.discrepancy_analyzer = DiscrepancyAnalyzer()
        self.kpi_monitor = KPIDiffMonitor()
        self.root_cause_analyzer = RootCauseAnalyzer()

    def match_trades(
        self,
        paper_trades: List[TradeRecord],
        live_trades: List[TradeRecord],
        max_time_diff_s: float = 60.0,
    ) -> List[TradePair]:
        """配对模拟与实盘交易

        策略：按时间戳最近匹配(同一symbol+同方向+时间差<60s)

        Args:
            paper_trades: 模拟交易列表
            live_trades: 实盘交易列表
            max_time_diff_s: 最大时间差(秒)

        Returns:
            List[TradePair]
        """
        pairs: List[TradePair] = []
        used_live_indices: set = set()

        for p_trade in paper_trades:
            best_match = None
            best_diff = float('inf')
            for i, l_trade in enumerate(live_trades):
                if i in used_live_indices:
                    continue
                if l_trade.symbol != p_trade.symbol:
                    continue
                if l_trade.side != p_trade.side:
                    continue
                time_diff = abs(l_trade.timestamp - p_trade.timestamp)
                if time_diff > max_time_diff_s:
                    continue
                if time_diff < best_diff:
                    best_diff = time_diff
                    best_match = i

            if best_match is not None:
                pairs.append(TradePair(
                    paper_trade=p_trade,
                    live_trade=live_trades[best_match],
                ))
                used_live_indices.add(best_match)
            else:
                pairs.append(TradePair(paper_trade=p_trade, live_trade=None))

        return pairs

    def compare(
        self,
        pairs: List[TradePair],
        paper_pnls: Optional[List[float]] = None,
        live_pnls: Optional[List[float]] = None,
        periods_per_year: int = 365,
    ) -> ComparisonReport:
        """综合对比

        Args:
            pairs: 交易配对列表
            paper_pnls: 模拟盘每日PnL(可选)
            live_pnls: 实盘每日PnL(可选)
            periods_per_year: 每年期数

        Returns:
            ComparisonReport
        """
        report = ComparisonReport()

        # 1. 差异分析
        discrepancy = self.discrepancy_analyzer.analyze(pairs)
        report.discrepancy = discrepancy

        # 2. KPI对比(如有PnL数据)
        if paper_pnls and live_pnls:
            kpi_diff = self.kpi_monitor.compare(
                paper_pnls, live_pnls, periods_per_year=periods_per_year,
            )
            report.kpi_diff = kpi_diff

        # 3. 根因分析
        root_cause = self.root_cause_analyzer.analyze(pairs, discrepancy)
        report.root_cause = root_cause

        # 4. 综合结论
        verdicts = [discrepancy.verdict, root_cause.verdict]
        if report.kpi_diff:
            verdicts.append(report.kpi_diff.verdict)

        if Verdict.BLOCKED in verdicts:
            report.final_verdict = Verdict.BLOCKED
        elif Verdict.DEGRADED in verdicts:
            report.final_verdict = Verdict.DEGRADED
        elif Verdict.ACCEPTABLE in verdicts:
            report.final_verdict = Verdict.ACCEPTABLE
        else:
            report.final_verdict = Verdict.GOOD

        # 汇总原因
        for v in [discrepancy, root_cause, report.kpi_diff]:
            if v and v.verdict in (Verdict.BLOCKED, Verdict.DEGRADED):
                report.reasons.extend(v.reasons)

        # 推荐
        if report.final_verdict == Verdict.BLOCKED:
            report.recommendations.append("停止实盘: 模拟与实盘严重背离, 立即停止实盘交易")
        elif report.final_verdict == Verdict.DEGRADED:
            report.recommendations.append("降低仓位: 模拟与实盘存在显著差异, 降低实盘仓位50%")
        elif report.final_verdict == Verdict.ACCEPTABLE:
            report.recommendations.append("继续观察: 模拟与实盘基本一致, 维持当前仓位")
        else:
            report.recommendations.append("增加仓位: 模拟与实盘高度一致, 可适当增加仓位")

        return report

    def generate_report(self, report: ComparisonReport) -> str:
        """生成人类可读报告"""
        lines = [
            "=" * 70,
            "模拟vs实盘对比报告 (Live-Paper Comparison Report)",
            "=" * 70,
            f"最终结论: {report.final_verdict.value}",
            "",
        ]

        if report.discrepancy:
            d = report.discrepancy
            lines.extend([
                "【交易差异分析】",
                f"  总配对数: {d.total_pairs}",
                f"  匹配数: {d.matched_pairs}",
                f"  未匹配(仅模拟): {d.unmatched_paper_count}",
                f"  平均价格差异: {d.avg_price_diff_bps:.2f}bps",
                f"  最大价格差异: {d.max_price_diff_bps:.2f}bps",
                f"  平均滑点差异: {d.avg_slippage_diff_bps:.2f}bps",
                f"  平均延迟差异: {d.avg_latency_diff_ms:.0f}ms",
                f"  平均成交率: {d.avg_fill_ratio:.1%}",
                f"  总PnL差异: ${d.total_pnl_diff_usd:,.2f}",
                f"  平均PnL差异: ${d.avg_pnl_diff_usd:,.2f}",
                f"  结论: {d.verdict.value}",
                "",
            ])

        if report.kpi_diff:
            k = report.kpi_diff
            lines.extend([
                "【KPI差异】",
                f"  Sharpe:  模拟={k.paper_sharpe:.3f}  实盘={k.live_sharpe:.3f}  差异={k.sharpe_diff:+.3f}",
                f"  收益率:  模拟={k.paper_return:.2%}    实盘={k.live_return:.2%}    差异={k.return_diff:+.2%}",
                f"  最大回撤: 模拟={k.paper_max_drawdown:.2%}  实盘={k.live_max_drawdown:.2%}  差异={k.drawdown_diff:.2%}",
                f"  胜率:    模拟={k.paper_winrate:.1%}   实盘={k.live_winrate:.1%}   差异={k.winrate_diff:.1%}",
                f"  结论: {k.verdict.value}",
                "",
            ])

        if report.root_cause:
            r = report.root_cause
            lines.extend([
                "【根因分析】",
                f"  滑点归因:    {r.slippage_attribution_pct:.1f}%",
                f"  延迟归因:    {r.latency_attribution_pct:.1f}%",
                f"  流动性归因:  {r.liquidity_attribution_pct:.1f}%",
                f"  手续费归因:  {r.fee_attribution_pct:.1f}%",
                f"  其他归因:    {r.other_attribution_pct:.1f}%",
                f"  主要原因: {r.primary_cause}",
                f"  次要原因: {r.secondary_cause}",
                f"  结论: {r.verdict.value}",
                "",
            ])

        if report.reasons:
            lines.append("【原因】")
            for r in report.reasons:
                lines.append(f"  - {r}")
            lines.append("")

        if report.recommendations:
            lines.append("【实盘建议】")
            for rec in report.recommendations:
                lines.append(f"  → {rec}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 自测
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R17-5 模拟vs实盘对比 自测 ===\n")

    import time as _time

    np.random.seed(42)

    # 构造模拟交易+实盘交易
    base_time = _time.time()
    paper_trades = []
    live_trades = []

    for i in range(20):
        ts = base_time + i * 3600
        price = 50000 + i * 100
        # 模拟交易
        paper_trades.append(TradeRecord(
            trade_id=f"p{i}", timestamp=ts, symbol="BTCUSDT", side="long",
            price=price, quantity=1.0, fee_usd=2.5, slippage_bps=1.0,
            latency_ms=10, pnl_usd=100, source="paper",
        ))
        # 实盘交易(有滑点和延迟)
        live_trades.append(TradeRecord(
            trade_id=f"l{i}", timestamp=ts + 0.1, symbol="BTCUSDT", side="long",
            price=price + 5, quantity=0.98, fee_usd=3.0, slippage_bps=3.0,
            latency_ms=80, pnl_usd=85, source="live",
        ))

    comparator = LivePaperComparator()
    pairs = comparator.match_trades(paper_trades, live_trades)
    print(f"配对结果: {len(pairs)}对, 匹配{sum(1 for p in pairs if p.is_matched)}个")

    # 构造每日PnL(模拟vs实盘有差异)
    paper_daily = list(np.random.normal(0.001, 0.01, 30))
    live_daily = list(np.array(paper_daily) - 0.0002)  # 实盘略差

    report = comparator.compare(
        pairs=pairs,
        paper_pnls=paper_daily,
        live_pnls=live_daily,
    )
    print(comparator.generate_report(report))
