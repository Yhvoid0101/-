# -*- coding: utf-8 -*-
"""
R17-2: 策略评估框架 (Strategy Evaluator) — 裸K策略系统性评估+筛选标准

针对用户原话:
  "对裸K交易策略及其他潜在交易策略进行系统性评估,分析其在数字货币合约日内
   交易场景中的适用性、风险收益特征及市场适应性"
  "根据实际交易环境、资金规模、风险承受能力及历史业绩表现,制定策略筛选与融合标准"

5大评估维度(来源:量化交易策略评估前沿最佳实践 2025-2026):
  1. 收益性能 (Return Performance) — 年化收益/Sharpe/Sortino/Calmar
  2. 风险特征 (Risk Profile) — 最大回撤/VaR/CVaR/波动率/下行风险
  3. 市场适应性 (Market Adaptability) — 不同regime下表现(牛/熊/震荡/极端)
  4. 稳定性 (Stability) — 滚动Sharpe一致性/收益自相关/路径间方差
  5. 适用性 (Applicability) — 资金容量/换手率/滑点敏感度/合约适配性

评估对象:
  - 缠论背驰策略(ChanLunDivergence)
  - 裸K形态策略(PinBar/Engulfing)
  - 支撑阻力策略(SupportResistance)
  - 海龟策略(TurtleTrading)
  - 趋势跟踪(MA Cross)
  - 均值回归(Bollinger Bands)
  - 动量策略(CrossSectional Momentum)
  - 任何IStrategy实现

铁律(筛选标准):
  - Sharpe < 0.5 → BLOCK
  - MaxDrawdown > 30% → BLOCK
  - 胜率 < 30% (日内合约场景) → WARN
  - 资金容量 < 100K USD → WARN(不适合大资金)
  - 滑点敏感度 > 50% (Sharpe退化) → BLOCK
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.strategy_evaluator")


# ============================================================================
# 评估阈值(铁标准)
# ============================================================================

# 收益性能阈值(来源:量化对冲基金行业标准 + 加密合约实盘数据)
SHARPE_BLOCK = 0.5              # Sharpe<0.5 → BLOCK
SHARPE_WARN = 1.0               # Sharpe<1.0 → WARN
SHARPE_GOOD = 1.5               # Sharpe>=1.5 → GOOD
SHARPE_EXCELLENT = 2.0          # Sharpe>=2.0 → EXCELLENT

SORTINO_BLOCK = 0.7             # Sortino<0.7 → BLOCK (Sortino通常比Sharpe高40%)
SORTINO_GOOD = 2.0
CALMAR_BLOCK = 0.3              # Calmar<0.3 → BLOCK
CALMAR_GOOD = 1.0

ANNUAL_RETURN_WARN = 0.10       # 年化<10% → WARN(扣除成本后可能亏损)
ANNUAL_RETURN_GOOD = 0.25       # 年化>=25% → GOOD
ANNUAL_RETURN_EXCELLENT = 0.50  # 年化>=50% → EXCELLENT

# 风险特征阈值
MAX_DRAWDOWN_BLOCK = 0.30       # 最大回撤>30% → BLOCK
MAX_DRAWDOWN_WARN = 0.20        # 最大回撤>20% → WARN
MAX_DRAWDOWN_GOOD = 0.10        # 最大回撤<=10% → GOOD

VAR_95_BLOCK = 0.05             # 95% VaR > 5% (单日) → BLOCK
CVAR_BLOCK = 0.08               # CVaR > 8% → BLOCK

VOLATILITY_BLOCK = 0.50         # 年化波动率>50% → BLOCK
VOLATILITY_WARN = 0.30          # 年化波动率>30% → WARN

# 市场适应性阈值
REGIME_BETA_BLOCK = -0.2        # 在有利regime下Sharpe<-0.2 → BLOCK
REGIME_CONSISTENCY_GOOD = 0.7   # 70%以上regime表现正收益 → GOOD

# 稳定性阈值
ROLLING_SHARPE_STD_BLOCK = 1.5  # 滚动Sharpe标准差>1.5 → BLOCK(不稳定)
AUTOCORR_BLOCK = 0.3            # 收益自相关>0.3 → BLOCK(非独立)

# 适用性阈值(日内合约场景)
CAPACITY_WARN_USD = 100_000     # 资金容量<100K → WARN
TURNOVER_BLOCK = 50.0           # 日换手率>50倍 → BLOCK(成本过高)
SLIPPAGE_SENSITIVITY_BLOCK = 0.50  # 50bps滑点导致Sharpe退化>50% → BLOCK


# ============================================================================
# 评估等级
# ============================================================================


class Verdict(Enum):
    """评估结论"""
    BLOCKED = "BLOCKED"     # 禁止使用
    DEGRADED = "DEGRADED"   # 可用但需谨慎
    ACCEPTABLE = "ACCEPTABLE"  # 可用
    GOOD = "GOOD"           # 良好
    EXCELLENT = "EXCELLENT"  # 卓越


# ============================================================================
# 1. 收益性能评估 (Return Performance)
# ============================================================================


@dataclass(slots=True)
class ReturnPerformance:
    """收益性能评估结果"""
    annual_return: float = 0.0       # 年化收益率
    sharpe: float = 0.0              # 夏普比率
    sortino: float = 0.0             # Sortino比率(只考虑下行波动)
    calmar: float = 0.0              # Calmar比率(年化收益/最大回撤)
    omega: float = 0.0               # Omega比率
    verdict: Verdict = Verdict.BLOCKED
    reasons: List[str] = field(default_factory=list)


def evaluate_return_performance(
    pnls: List[float],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> ReturnPerformance:
    """评估收益性能

    Args:
        pnls: 收益率序列(每期收益率,非累计)
        periods_per_year: 每年期数(日线252/小时线8760)
        risk_free_rate: 无风险利率(年化)

    Returns:
        ReturnPerformance
    """
    result = ReturnPerformance()
    if not pnls or len(pnls) < 10:
        result.reasons.append("数据不足(<10期)")
        return result

    arr = np.asarray(pnls, dtype=np.float64)
    n = len(arr)

    # 年化收益
    mean_period = float(np.mean(arr))
    result.annual_return = mean_period * periods_per_year

    # Sharpe
    std_period = float(np.std(arr, ddof=1))
    if std_period > 1e-10:
        result.sharpe = (mean_period - risk_free_rate / periods_per_year) / std_period * math.sqrt(periods_per_year)
    else:
        result.sharpe = 0.0

    # Sortino (只考虑下行波动)
    downside = arr[arr < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 1e-10:
            result.sortino = (mean_period - risk_free_rate / periods_per_year) / downside_std * math.sqrt(periods_per_year)
        else:
            result.sortino = 0.0
    else:
        result.sortino = result.sharpe * 1.4  # 无下行时近似

    # Calmar (年化收益/最大回撤)
    cumsum = np.cumsum(arr)
    peak = np.maximum.accumulate(cumsum)
    dd = np.where(peak > 0, (peak - cumsum) / np.where(peak > 0, peak, 1), 0)
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    if max_dd > 1e-6:
        result.calmar = result.annual_return / max_dd
    else:
        result.calmar = 0.0

    # Omega比率(收益/损失比)
    gains = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    if losses > 1e-10:
        result.omega = gains / losses
    else:
        result.omega = 999.0

    # 评估结论
    if result.sharpe < SHARPE_BLOCK:
        result.verdict = Verdict.BLOCKED
        result.reasons.append(f"Sharpe={result.sharpe:.2f}<{SHARPE_BLOCK}")
    elif result.sharpe < SHARPE_WARN:
        result.verdict = Verdict.DEGRADED
        result.reasons.append(f"Sharpe={result.sharpe:.2f}<{SHARPE_WARN}")
    elif result.sharpe >= SHARPE_EXCELLENT:
        result.verdict = Verdict.EXCELLENT
    elif result.sharpe >= SHARPE_GOOD:
        result.verdict = Verdict.GOOD
    else:
        result.verdict = Verdict.ACCEPTABLE

    if result.annual_return < ANNUAL_RETURN_WARN and result.verdict != Verdict.BLOCKED:
        result.reasons.append(f"年化收益={result.annual_return:.1%}<{ANNUAL_RETURN_WARN:.0%}")

    return result


# ============================================================================
# 2. 风险特征评估 (Risk Profile)
# ============================================================================


@dataclass(slots=True)
class RiskProfile:
    """风险特征评估结果"""
    max_drawdown: float = 0.0       # 最大回撤
    var_95: float = 0.0             # 95% VaR(单日)
    cvar_95: float = 0.0            # 95% CVaR
    volatility_annual: float = 0.0  # 年化波动率
    downside_deviation: float = 0.0  # 下行偏差
    verdict: Verdict = Verdict.BLOCKED
    reasons: List[str] = field(default_factory=list)


def evaluate_risk_profile(
    pnls: List[float],
    periods_per_year: int = 252,
) -> RiskProfile:
    """评估风险特征"""
    result = RiskProfile()
    if not pnls or len(pnls) < 10:
        result.reasons.append("数据不足(<10期)")
        return result

    arr = np.asarray(pnls, dtype=np.float64)

    # 最大回撤
    cumsum = np.cumsum(arr)
    peak = np.maximum.accumulate(cumsum)
    # 使用对数权益避免初始值问题
    equity = 1.0 + cumsum
    equity = np.where(equity > 0.01, equity, 0.01)
    peak_eq = np.maximum.accumulate(equity)
    dd = (peak_eq - equity) / peak_eq
    result.max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0.0

    # VaR(95%) - 单期
    if len(arr) >= 20:
        result.var_95 = float(abs(np.percentile(arr, 5)))  # 5%分位数的绝对值
        # CVaR(95%) - 最差5%的平均
        var_threshold = np.percentile(arr, 5)
        tail = arr[arr <= var_threshold]
        result.cvar_95 = float(abs(np.mean(tail))) if len(tail) > 0 else result.var_95
    else:
        result.var_95 = 0.0
        result.cvar_95 = 0.0

    # 年化波动率
    std_period = float(np.std(arr, ddof=1))
    result.volatility_annual = std_period * math.sqrt(periods_per_year)

    # 下行偏差
    downside = arr[arr < 0]
    if len(downside) > 1:
        result.downside_deviation = float(np.std(downside, ddof=1)) * math.sqrt(periods_per_year)

    # 评估结论
    if result.max_drawdown > MAX_DRAWDOWN_BLOCK:
        result.verdict = Verdict.BLOCKED
        result.reasons.append(f"最大回撤={result.max_drawdown:.1%}>{MAX_DRAWDOWN_BLOCK:.0%}")
    elif result.max_drawdown > MAX_DRAWDOWN_WARN:
        result.verdict = Verdict.DEGRADED
        result.reasons.append(f"最大回撤={result.max_drawdown:.1%}>{MAX_DRAWDOWN_WARN:.0%}")
    elif result.max_drawdown <= MAX_DRAWDOWN_GOOD:
        result.verdict = Verdict.GOOD
    else:
        result.verdict = Verdict.ACCEPTABLE

    if result.var_95 > VAR_95_BLOCK:
        result.reasons.append(f"VaR95={result.var_95:.1%}>{VAR_95_BLOCK:.0%}")
        if result.verdict != Verdict.BLOCKED:
            result.verdict = Verdict.DEGRADED

    if result.volatility_annual > VOLATILITY_BLOCK:
        result.reasons.append(f"年化波动={result.volatility_annual:.1%}>{VOLATILITY_BLOCK:.0%}")
        if result.verdict != Verdict.BLOCKED:
            result.verdict = Verdict.DEGRADED

    return result


# ============================================================================
# 3. 市场适应性评估 (Market Adaptability)
# ============================================================================


class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"           # 牛市(上涨趋势)
    BEAR = "bear"           # 熊市(下跌趋势)
    SIDEWAYS = "sideways"   # 震荡(横盘)
    EXTREME = "extreme"     # 极端(高波动/黑天鹅)


@dataclass(slots=True)
class RegimePerformance:
    """单regime下表现"""
    regime: MarketRegime
    sharpe: float = 0.0
    return_pct: float = 0.0
    bars: int = 0


@dataclass(slots=True)
class MarketAdaptability:
    """市场适应性评估结果"""
    regime_performances: List[RegimePerformance] = field(default_factory=list)
    consistency_score: float = 0.0    # 一致性评分(0-1, 越高越好)
    best_regime: Optional[MarketRegime] = None
    worst_regime: Optional[MarketRegime] = None
    verdict: Verdict = Verdict.BLOCKED
    reasons: List[str] = field(default_factory=list)


def classify_regime(returns: np.ndarray) -> MarketRegime:
    """根据收益序列分类市场状态

    简化分类规则:
      - 平均收益>0.1% 且 波动率<3% → BULL
      - 平均收益<-0.1% 且 波动率<3% → BEAR
      - 波动率>=5% → EXTREME
      - 其他 → SIDEWAYS
    """
    if len(returns) < 5:
        return MarketRegime.SIDEWAYS

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))

    if std_ret >= 0.05:
        return MarketRegime.EXTREME
    if mean_ret >= 0.001:
        return MarketRegime.BULL
    if mean_ret <= -0.001:
        return MarketRegime.BEAR
    return MarketRegime.SIDEWAYS


def evaluate_market_adaptability(
    pnls: List[float],
    window_size: int = 60,
    periods_per_year: int = 252,
) -> MarketAdaptability:
    """评估市场适应性(滚动窗口分类regime+各regime下表现)"""
    result = MarketAdaptability()
    if len(pnls) < window_size * 2:
        result.reasons.append(f"数据不足(<{window_size*2}期)")
        return result

    arr = np.asarray(pnls, dtype=np.float64)

    # 滚动窗口分类regime
    regime_performances: Dict[MarketRegime, List[float]] = {
        r: [] for r in MarketRegime
    }
    for i in range(0, len(arr) - window_size, window_size // 2):
        window = arr[i:i + window_size]
        regime = classify_regime(window)
        regime_performances[regime].extend(window.tolist())

    # 计算各regime下表现
    for regime, regime_pnls in regime_performances.items():
        if not regime_pnls:
            continue
        rp_arr = np.asarray(regime_pnls, dtype=np.float64)
        mean_ret = float(np.mean(rp_arr))
        std_ret = float(np.std(rp_arr, ddof=1))
        sharpe = (mean_ret / std_ret * math.sqrt(periods_per_year)) if std_ret > 1e-10 else 0.0
        result.regime_performances.append(RegimePerformance(
            regime=regime,
            sharpe=sharpe,
            return_pct=mean_ret * len(regime_pnls),
            bars=len(regime_pnls),
        ))

    # 一致性评分: 正收益regime占比
    if result.regime_performances:
        positive_count = sum(1 for rp in result.regime_performances if rp.return_pct > 0)
        result.consistency_score = positive_count / len(result.regime_performances)
        # 最优/最差regime
        sorted_rps = sorted(result.regime_performances, key=lambda x: x.sharpe, reverse=True)
        result.best_regime = sorted_rps[0].regime
        result.worst_regime = sorted_rps[-1].regime

    # 评估结论
    if result.consistency_score >= REGIME_CONSISTENCY_GOOD:
        result.verdict = Verdict.GOOD
    elif result.consistency_score >= 0.5:
        result.verdict = Verdict.ACCEPTABLE
    elif result.consistency_score >= 0.25:
        result.verdict = Verdict.DEGRADED
        result.reasons.append(f"一致性评分={result.consistency_score:.2f}<0.5")
    else:
        result.verdict = Verdict.BLOCKED
        result.reasons.append(f"一致性评分={result.consistency_score:.2f}<0.25")

    # 极端regime下表现检查
    extreme_rp = next((rp for rp in result.regime_performances if rp.regime == MarketRegime.EXTREME), None)
    if extreme_rp and extreme_rp.sharpe < REGIME_BETA_BLOCK:
        result.reasons.append(f"极端regime下Sharpe={extreme_rp.sharpe:.2f}<{REGIME_BETA_BLOCK}")
        if result.verdict != Verdict.BLOCKED:
            result.verdict = Verdict.DEGRADED

    return result


# ============================================================================
# 4. 稳定性评估 (Stability)
# ============================================================================


@dataclass(slots=True)
class StabilityMetrics:
    """稳定性评估结果"""
    rolling_sharpe_mean: float = 0.0
    rolling_sharpe_std: float = 0.0
    autocorrelation_lag1: float = 0.0
    path_variance: float = 0.0
    verdict: Verdict = Verdict.BLOCKED
    reasons: List[str] = field(default_factory=list)


def evaluate_stability(
    pnls: List[float],
    rolling_window: int = 60,
    periods_per_year: int = 252,
) -> StabilityMetrics:
    """评估稳定性(滚动Sharpe一致性+收益自相关)"""
    result = StabilityMetrics()
    if len(pnls) < rolling_window * 2:
        result.reasons.append(f"数据不足(<{rolling_window*2}期)")
        return result

    arr = np.asarray(pnls, dtype=np.float64)

    # 滚动Sharpe
    rolling_sharpes: List[float] = []
    for i in range(0, len(arr) - rolling_window):
        window = arr[i:i + rolling_window]
        mean_w = float(np.mean(window))
        std_w = float(np.std(window, ddof=1))
        if std_w > 1e-10:
            sharpe_w = mean_w / std_w * math.sqrt(periods_per_year)
            rolling_sharpes.append(sharpe_w)

    if rolling_sharpes:
        result.rolling_sharpe_mean = float(np.mean(rolling_sharpes))
        result.rolling_sharpe_std = float(np.std(rolling_sharpes, ddof=1))

    # 自相关 lag=1
    if len(arr) > 2:
        mean_arr = float(np.mean(arr))
        numerator = sum((arr[i] - mean_arr) * (arr[i - 1] - mean_arr) for i in range(1, len(arr)))
        denominator = sum((x - mean_arr) ** 2 for x in arr)
        if denominator > 1e-10:
            result.autocorrelation_lag1 = float(numerator / denominator)

    # 评估结论
    if result.rolling_sharpe_std > ROLLING_SHARPE_STD_BLOCK:
        result.verdict = Verdict.BLOCKED
        result.reasons.append(f"滚动Sharpe标准差={result.rolling_sharpe_std:.2f}>{ROLLING_SHARPE_STD_BLOCK}")
    elif result.rolling_sharpe_std > 1.0:
        result.verdict = Verdict.DEGRADED
        result.reasons.append(f"滚动Sharpe标准差={result.rolling_sharpe_std:.2f}>1.0")
    elif result.rolling_sharpe_std <= 0.5:
        result.verdict = Verdict.GOOD
    else:
        result.verdict = Verdict.ACCEPTABLE

    if abs(result.autocorrelation_lag1) > AUTOCORR_BLOCK:
        result.reasons.append(f"自相关={result.autocorrelation_lag1:.2f}>{AUTOCORR_BLOCK}")
        if result.verdict != Verdict.BLOCKED:
            result.verdict = Verdict.DEGRADED

    return result


# ============================================================================
# 5. 适用性评估 (Applicability)
# ============================================================================


@dataclass(slots=True)
class ApplicabilityMetrics:
    """适用性评估结果"""
    capacity_usd: float = 0.0          # 资金容量(USD)
    daily_turnover: float = 0.0        # 日换手率
    slippage_sensitivity: float = 0.0  # 滑点敏感度(Sharpe退化%)
    contract_compatible: bool = False  # 合约兼容性
    verdict: Verdict = Verdict.BLOCKED
    reasons: List[str] = field(default_factory=list)


def evaluate_applicability(
    pnls: List[float],
    avg_position_usd: float = 10000.0,
    avg_trade_frequency_per_day: float = 5.0,
    avg_daily_volume_usd: float = 100_000_000.0,
    periods_per_day: int = 24,
    contract_mode: bool = True,
) -> ApplicabilityMetrics:
    """评估适用性(资金容量+换手率+滑点敏感度)

    Args:
        pnls: 收益率序列
        avg_position_usd: 平均持仓(USD)
        avg_trade_frequency_per_day: 日交易频率
        avg_daily_volume_usd: 标的日均成交量(USD)
        periods_per_day: 每日周期数(小时线=24)
        contract_mode: 是否合约模式
    """
    result = ApplicabilityMetrics()
    if not pnls:
        result.reasons.append("数据不足")
        return result

    arr = np.asarray(pnls, dtype=np.float64)

    # 1. 资金容量(基于ADV 10%限制)
    # 资金容量 = ADV × 10% / (日交易频率 × 单笔仓位占比)
    # 简化: 容量 = ADV × 0.1
    result.capacity_usd = avg_daily_volume_usd * 0.1

    # 2. 日换手率
    # 换手率 = 日交易频率 × 平均仓位 / 总资金
    # 假设总资金=avg_position_usd, 单笔仓位=avg_position_usd
    result.daily_turnover = avg_trade_frequency_per_day  # 简化:每日交易几次就换手几次

    # 3. 滑点敏感度
    # 模拟50bps滑点对Sharpe的影响
    slippage_cost_per_trade = 0.0050  # 50bps
    daily_slippage_cost = slippage_cost_per_trade * avg_trade_frequency_per_day
    # 转化为每期成本
    period_slippage_cost = daily_slippage_cost / periods_per_day

    original_mean = float(np.mean(arr))
    original_std = float(np.std(arr, ddof=1))
    if original_std > 1e-10:
        original_sharpe = original_mean / original_std
        adjusted_mean = original_mean - period_slippage_cost
        adjusted_sharpe = adjusted_mean / original_std
        if abs(original_sharpe) > 1e-6:
            result.slippage_sensitivity = 1.0 - (adjusted_sharpe / original_sharpe)
        else:
            result.slippage_sensitivity = 1.0
    else:
        result.slippage_sensitivity = 1.0

    # 4. 合约兼容性(简单判断:换手率<20且滑点敏感度<0.5)
    result.contract_compatible = (
        result.daily_turnover < 20.0
        and result.slippage_sensitivity < 0.5
        and contract_mode
    )

    # 评估结论
    if result.slippage_sensitivity > SLIPPAGE_SENSITIVITY_BLOCK:
        result.verdict = Verdict.BLOCKED
        result.reasons.append(f"滑点敏感度={result.slippage_sensitivity:.1%}>{SLIPPAGE_SENSITIVITY_BLOCK:.0%}")
    elif result.daily_turnover > TURNOVER_BLOCK:
        result.verdict = Verdict.BLOCKED
        result.reasons.append(f"日换手率={result.daily_turnover:.1f}>{TURNOVER_BLOCK}")
    elif result.capacity_usd < CAPACITY_WARN_USD:
        result.verdict = Verdict.DEGRADED
        result.reasons.append(f"资金容量={result.capacity_usd:.0f}<{CAPACITY_WARN_USD}")
    elif result.slippage_sensitivity < 0.2 and result.contract_compatible:
        result.verdict = Verdict.EXCELLENT
    elif result.slippage_sensitivity < 0.3:
        result.verdict = Verdict.GOOD
    else:
        result.verdict = Verdict.ACCEPTABLE

    return result


# ============================================================================
# 6. 综合策略评估器 (Strategy Evaluator)
# ============================================================================


@dataclass(slots=True)
class StrategyEvaluationResult:
    """策略综合评估结果"""
    strategy_name: str
    return_performance: ReturnPerformance = field(default_factory=ReturnPerformance)
    risk_profile: RiskProfile = field(default_factory=RiskProfile)
    market_adaptability: MarketAdaptability = field(default_factory=MarketAdaptability)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    applicability: ApplicabilityMetrics = field(default_factory=ApplicabilityMetrics)

    overall_verdict: Verdict = Verdict.BLOCKED
    overall_score: float = 0.0           # 0-100综合评分
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)  # 优势

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "overall_verdict": self.overall_verdict.value,
            "overall_score": round(self.overall_score, 2),
            "return_performance": {
                "annual_return": self.return_performance.annual_return,
                "sharpe": self.return_performance.sharpe,
                "sortino": self.return_performance.sortino,
                "calmar": self.return_performance.calmar,
                "omega": self.return_performance.omega,
                "verdict": self.return_performance.verdict.value,
            },
            "risk_profile": {
                "max_drawdown": self.risk_profile.max_drawdown,
                "var_95": self.risk_profile.var_95,
                "cvar_95": self.risk_profile.cvar_95,
                "volatility_annual": self.risk_profile.volatility_annual,
                "verdict": self.risk_profile.verdict.value,
            },
            "market_adaptability": {
                "consistency_score": self.market_adaptability.consistency_score,
                "best_regime": self.market_adaptability.best_regime.value if self.market_adaptability.best_regime else None,
                "worst_regime": self.market_adaptability.worst_regime.value if self.market_adaptability.worst_regime else None,
                "verdict": self.market_adaptability.verdict.value,
            },
            "stability": {
                "rolling_sharpe_mean": self.stability.rolling_sharpe_mean,
                "rolling_sharpe_std": self.stability.rolling_sharpe_std,
                "autocorrelation_lag1": self.stability.autocorrelation_lag1,
                "verdict": self.stability.verdict.value,
            },
            "applicability": {
                "capacity_usd": self.applicability.capacity_usd,
                "daily_turnover": self.applicability.daily_turnover,
                "slippage_sensitivity": self.applicability.slippage_sensitivity,
                "contract_compatible": self.applicability.contract_compatible,
                "verdict": self.applicability.verdict.value,
            },
            "block_reasons": self.block_reasons,
            "warn_reasons": self.warn_reasons,
            "strengths": self.strengths,
        }


class StrategyEvaluator:
    """策略综合评估器 — 5维评估+筛选标准

    使用方式:
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate(
            strategy_name="裸K-PinBar",
            pnls=strategy_pnls,
            periods_per_year=8760,  # 1h K线
            avg_position_usd=10000,
            avg_trade_frequency_per_day=5,
            contract_mode=True,
        )
        if result.overall_verdict == Verdict.BLOCKED:
            # 禁止使用
            print("策略被阻止:", result.block_reasons)
    """

    def evaluate(
        self,
        strategy_name: str,
        pnls: List[float],
        periods_per_year: int = 252,
        avg_position_usd: float = 10000.0,
        avg_trade_frequency_per_day: float = 5.0,
        avg_daily_volume_usd: float = 100_000_000.0,
        contract_mode: bool = True,
    ) -> StrategyEvaluationResult:
        """综合评估策略

        Args:
            strategy_name: 策略名称
            pnls: 每期收益率序列
            periods_per_year: 每年期数(日线252/小时线8760)
            avg_position_usd: 平均持仓(USD)
            avg_trade_frequency_per_day: 日交易频率
            avg_daily_volume_usd: 标的日均成交量
            contract_mode: 是否合约模式

        Returns:
            StrategyEvaluationResult
        """
        result = StrategyEvaluationResult(strategy_name=strategy_name)

        # 5维评估
        result.return_performance = evaluate_return_performance(
            pnls, periods_per_year=periods_per_year,
        )
        result.risk_profile = evaluate_risk_profile(
            pnls, periods_per_year=periods_per_year,
        )
        result.market_adaptability = evaluate_market_adaptability(
            pnls, periods_per_year=periods_per_year,
        )
        result.stability = evaluate_stability(
            pnls, periods_per_year=periods_per_year,
        )
        result.applicability = evaluate_applicability(
            pnls,
            avg_position_usd=avg_position_usd,
            avg_trade_frequency_per_day=avg_trade_frequency_per_day,
            avg_daily_volume_usd=avg_daily_volume_usd,
            periods_per_day=periods_per_year // 365 if periods_per_year >= 365 else 24,
            contract_mode=contract_mode,
        )

        # 综合评分(0-100)
        score = self._compute_overall_score(result)
        result.overall_score = score

        # 综合结论: 任一维度BLOCKED → BLOCKED
        verdicts = [
            result.return_performance.verdict,
            result.risk_profile.verdict,
            result.market_adaptability.verdict,
            result.stability.verdict,
            result.applicability.verdict,
        ]

        if Verdict.BLOCKED in verdicts:
            result.overall_verdict = Verdict.BLOCKED
        elif Verdict.DEGRADED in verdicts:
            result.overall_verdict = Verdict.DEGRADED
        elif Verdict.ACCEPTABLE in verdicts:
            result.overall_verdict = Verdict.ACCEPTABLE
        elif Verdict.GOOD in verdicts:
            result.overall_verdict = Verdict.GOOD
        else:
            result.overall_verdict = Verdict.EXCELLENT

        # 收集BLOCK/WARN/strengths
        for name, perf in [
            ("收益", result.return_performance),
            ("风险", result.risk_profile),
            ("适应性", result.market_adaptability),
            ("稳定性", result.stability),
            ("适用性", result.applicability),
        ]:
            for reason in perf.reasons:
                if perf.verdict == Verdict.BLOCKED:
                    result.block_reasons.append(f"[{name}] {reason}")
                else:
                    result.warn_reasons.append(f"[{name}] {reason}")
            if perf.verdict in (Verdict.GOOD, Verdict.EXCELLENT):
                result.strengths.append(f"[{name}] {perf.verdict.value}")

        return result

    def _compute_overall_score(self, result: StrategyEvaluationResult) -> float:
        """计算综合评分(0-100)"""
        # 各维度权重
        weights = {
            "return": 0.30,
            "risk": 0.25,
            "adaptability": 0.20,
            "stability": 0.15,
            "applicability": 0.10,
        }

        # 各等级对应分数
        grade_score = {
            Verdict.BLOCKED: 10.0,
            Verdict.DEGRADED: 40.0,
            Verdict.ACCEPTABLE: 60.0,
            Verdict.GOOD: 80.0,
            Verdict.EXCELLENT: 95.0,
        }

        score = (
            weights["return"] * grade_score[result.return_performance.verdict]
            + weights["risk"] * grade_score[result.risk_profile.verdict]
            + weights["adaptability"] * grade_score[result.market_adaptability.verdict]
            + weights["stability"] * grade_score[result.stability.verdict]
            + weights["applicability"] * grade_score[result.applicability.verdict]
        )

        return score

    def generate_report(self, result: StrategyEvaluationResult) -> str:
        """生成人类可读报告"""
        lines = [
            "=" * 70,
            f"策略评估报告: {result.strategy_name}",
            "=" * 70,
            f"综合评分: {result.overall_score:.1f}/100",
            f"综合结论: {result.overall_verdict.value}",
            "",
            "【收益性能】",
            f"  年化收益: {result.return_performance.annual_return:.2%}",
            f"  Sharpe:   {result.return_performance.sharpe:.3f}",
            f"  Sortino:  {result.return_performance.sortino:.3f}",
            f"  Calmar:   {result.return_performance.calmar:.3f}",
            f"  Omega:    {result.return_performance.omega:.3f}",
            f"  结论:     {result.return_performance.verdict.value}",
            "",
            "【风险特征】",
            f"  最大回撤: {result.risk_profile.max_drawdown:.2%}",
            f"  VaR95:    {result.risk_profile.var_95:.2%}",
            f"  CVaR95:   {result.risk_profile.cvar_95:.2%}",
            f"  年化波动: {result.risk_profile.volatility_annual:.2%}",
            f"  结论:     {result.risk_profile.verdict.value}",
            "",
            "【市场适应性】",
            f"  一致性:   {result.market_adaptability.consistency_score:.2f}",
            f"  最优regime: {result.market_adaptability.best_regime.value if result.market_adaptability.best_regime else 'N/A'}",
            f"  最差regime: {result.market_adaptability.worst_regime.value if result.market_adaptability.worst_regime else 'N/A'}",
            f"  结论:     {result.market_adaptability.verdict.value}",
            "",
            "【稳定性】",
            f"  滚动Sharpe均值: {result.stability.rolling_sharpe_mean:.3f}",
            f"  滚动Sharpe标准差: {result.stability.rolling_sharpe_std:.3f}",
            f"  自相关:   {result.stability.autocorrelation_lag1:.3f}",
            f"  结论:     {result.stability.verdict.value}",
            "",
            "【适用性】",
            f"  资金容量: ${result.applicability.capacity_usd:,.0f}",
            f"  日换手率: {result.applicability.daily_turnover:.1f}",
            f"  滑点敏感度: {result.applicability.slippage_sensitivity:.1%}",
            f"  合约兼容: {result.applicability.contract_compatible}",
            f"  结论:     {result.applicability.verdict.value}",
            "",
        ]

        if result.block_reasons:
            lines.append("【BLOCK原因】")
            for r in result.block_reasons:
                lines.append(f"  - {r}")
            lines.append("")

        if result.warn_reasons:
            lines.append("【WARN警告】")
            for r in result.warn_reasons:
                lines.append(f"  - {r}")
            lines.append("")

        if result.strengths:
            lines.append("【优势】")
            for s in result.strengths:
                lines.append(f"  + {s}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R17-2 策略评估框架 自测 ===\n")

    evaluator = StrategyEvaluator()

    # 模拟裸K-PinBar策略(高频日内)
    np.random.seed(42)
    pnls_good = list(np.random.normal(0.0002, 0.008, 2000))  # Sharpe ≈ 2.18
    pnls_bad = list(np.random.normal(-0.0001, 0.015, 2000))  # 负收益

    print("--- 评估策略1: 高Sharpe策略 ---")
    r1 = evaluator.evaluate(
        strategy_name="MockHighSharpe",
        pnls=pnls_good,
        periods_per_year=8760,
        avg_position_usd=10000,
        avg_trade_frequency_per_day=5,
        contract_mode=True,
    )
    print(evaluator.generate_report(r1))

    print("\n--- 评估策略2: 负收益策略 ---")
    r2 = evaluator.evaluate(
        strategy_name="MockNegativeReturn",
        pnls=pnls_bad,
        periods_per_year=8760,
    )
    print(evaluator.generate_report(r2))
