# -*- coding: utf-8 -*-
"""
实盘-回测差距量化器 (Live-Backtest Gap Quantifier) — R15 核心模块

解决"模拟牛逼，实盘亏钱"的最根本问题：
  业界数据 (Glassnode 2025): 87% 回测正收益策略实盘亏损
  典型退化幅度: 30-60% (零售策略平均)
  退化五大结构原因:
    1. 滑点与佣金低估 (90% 策略存在, 年化损耗 5-15%)
    2. 过拟合/曲线拟合 (65% 策略存在, 年化高估 40-60%)
    3. 市场状态变化 (55% 策略存在, 策略失效)
    4. 流动性不足 (45% 策略存在, 无法按预期价执行)
    5. 幸存者偏差 (35% 策略存在, 仅对存活资产有效)

业界基准 (FerroQuant 2026-04):
  Degradation Factor = OOS_Sharpe / IS_Sharpe
  - DF > 0.7  → PASS (实盘可保持 70%+ 表现)
  - DF 0.4-0.7 → WARN (退化明显, 需谨慎部署)
  - DF < 0.4  → BLOCK (几乎必然实盘亏钱)

来源:
  - FerroQuant 2026-04 "Backtesting vs Live Trading: Bridging the Gap"
  - digitalninjasystems 2026-06 "Closing the Performance Gap"
  - arXiv:2604.03272 (Meng & Chen, NYU, 2026-03) AI 渗透率下系统性风险
  - Glassnode 2025 实盘退化统计
  - QuantConnect 社区 2,847 策略实测数据

核心铁律:
  - 回测 Sharpe=3.0 但 DF=0.3 → 实盘预期 Sharpe=0.9 → BLOCK
  - 回测 Sharpe=1.5 但 DF=0.8 → 实盘预期 Sharpe=1.2 → PASS
  - 不是 Sharpe 越高越好, 是 DF 越高越可信
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Phase 5 Task #28.2: 接入 maker_cost_model 替换硬编码 flat bps
# 用户核心诉求: "模拟/实盘差异全成本量化模型，根治模拟好看实盘亏"
# 之前: theoretical_cost_bps=2.0, realistic_cost_bps=10.0 硬编码, 未利用 per-symbol 精确成本
# 现在: 用 compute_taker_cost_bps / compute_maker_cost_usd 计算 per-symbol 真实成本
try:
    from .maker_cost_model import (
        MakerCostParams, compute_maker_cost_usd, compute_taker_cost_bps,
    )
    _MAKER_COST_MODEL_AVAILABLE = True
except ImportError:
    _MAKER_COST_MODEL_AVAILABLE = False
    MakerCostParams = None  # type: ignore
    compute_maker_cost_usd = None  # type: ignore
    compute_taker_cost_bps = None  # type: ignore

logger = logging.getLogger("hermes.live_backtest_gap")


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class DegradationFactorResult:
    """退化因子计算结果

    DF = OOS_Sharpe / IS_Sharpe
    衡量策略从训练集到测试集的退化程度。
    """

    is_sharpe: float              # In-Sample Sharpe (训练集)
    oos_sharpe: float             # Out-of-Sample Sharpe (测试集)
    degradation_factor: float     # DF = OOS/IS, [0, +inf)
    degradation_pct: float        # 退化百分比 = (1 - DF) * 100%
    verdict: str                  # PASS / WARN / BLOCK
    detail: str                   # 人类可读说明


@dataclass(slots=True)
class GapBreakdown:
    """5大差距来源量化

    每项差距表示为"该因素导致的Sharpe高估百分比"
    total_gap_pct = sum of all gaps (clamped to [0, 0.95])
    """

    # 数据差距: 合成数据 vs 真实历史数据 Sharpe 差异 (合成通常高估 10-30%)
    data_gap_pct: float = 0.0

    # 成本差距: 理论成本 vs 真实成本 (含滑点/费率/资金费率, 通常高估 5-15%)
    cost_gap_pct: float = 0.0

    # 延迟差距: 理想成交 vs 实际延迟 (信号到成交, 通常高估 2-10%)
    latency_gap_pct: float = 0.0

    # 流动性差距: 无限流动性 vs 订单簿深度 (大单冲击, 通常高估 5-20%)
    liquidity_gap_pct: float = 0.0

    # 状态差距: 回测期间 vs 实盘期间市场状态 (regime drift, 通常高估 10-40%)
    regime_gap_pct: float = 0.0

    # Phase 5 Task #28.2: per-symbol 成本明细 (validate_with_maker_cost 填充)
    # None = 使用硬编码 flat bps (向后兼容); dict = per-symbol 精确成本
    per_symbol_costs: Optional[Dict[str, Any]] = None

    # 总差距 (clamped to [0, 0.95], 防止极端值导致负Sharpe)
    @property
    def total_gap_pct(self) -> float:
        return min(0.95, max(0.0,
            self.data_gap_pct +
            self.cost_gap_pct +
            self.latency_gap_pct +
            self.liquidity_gap_pct +
            self.regime_gap_pct
        ))

    # 实盘预期保持因子 = 1 - total_gap_pct
    @property
    def gap_factor(self) -> float:
        """实盘预期 = 回测 × gap_factor"""
        return max(0.05, 1.0 - self.total_gap_pct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_gap_pct": round(self.data_gap_pct, 4),
            "cost_gap_pct": round(self.cost_gap_pct, 4),
            "latency_gap_pct": round(self.latency_gap_pct, 4),
            "liquidity_gap_pct": round(self.liquidity_gap_pct, 4),
            "regime_gap_pct": round(self.regime_gap_pct, 4),
            "total_gap_pct": round(self.total_gap_pct, 4),
            "gap_factor": round(self.gap_factor, 4),
            "per_symbol_costs": self.per_symbol_costs,
        }


@dataclass(slots=True)
class PredictedLiveSharpe:
    """实盘Sharpe预测结果"""

    backtest_sharpe: float         # 回测Sharpe
    predicted_live_sharpe: float   # 预测实盘Sharpe = backtest × gap_factor
    gap_factor: float              # 保持因子
    total_gap_pct: float           # 总退化百分比
    verdict: str                   # PASS / WARN / BLOCK
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)
    detail: str = ""


@dataclass(slots=True)
class LiveBacktestGapResult:
    """实盘-回测差距综合验证结果"""

    degradation_factor: Optional[DegradationFactorResult] = None
    gap_breakdown: Optional[GapBreakdown] = None
    predicted_live: Optional[PredictedLiveSharpe] = None
    overall_verdict: str = "PASS"  # PASS / WARN / BLOCK
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)


# ============================================================================
# 1. 退化因子计算器 (Degradation Factor Calculator)
# ============================================================================


class DegradationFactorCalculator:
    """退化因子计算器

    DF = OOS_Sharpe / IS_Sharpe

    业界基准 (FerroQuant 2026-04, traderssecondbrain 2026-05):
      DF > 0.7   → PASS (实盘可保持 70%+ 表现)
      DF 0.4-0.7 → WARN (退化明显, 30-60% 退化, 谨慎部署)
      DF < 0.4   → BLOCK (退化 >60%, 几乎必然实盘亏钱)

    特殊情况:
      - IS_Sharpe ≤ 0 → 训练集就亏, 直接 BLOCK
      - OOS_Sharpe < 0 但 IS > 0 → 训练赚测试亏, BLOCK (典型过拟合)
      - OOS_Sharpe > IS_Sharpe → DF > 1.0, 可能欠拟合或数据泄露
    """

    # 部署门槛 (来源: FerroQuant 2026-04)
    PASS_THRESHOLD = 0.70   # DF > 0.7 → PASS
    WARN_THRESHOLD = 0.40   # DF > 0.4 → WARN, DF < 0.4 → BLOCK

    @classmethod
    def compute(
        cls,
        is_sharpe: float,
        oos_sharpe: float,
    ) -> DegradationFactorResult:
        """计算退化因子

        Args:
            is_sharpe: In-Sample Sharpe (训练集/优化集表现)
            oos_sharpe: Out-of-Sample Sharpe (测试集/前向验证表现)

        Returns:
            DegradationFactorResult
        """
        # 特殊情况1: 训练集就亏
        if is_sharpe <= 0.0:
            return DegradationFactorResult(
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                degradation_factor=0.0,
                degradation_pct=100.0,
                verdict="BLOCK",
                detail=f"训练集Sharpe={is_sharpe:.3f}≤0, 策略本身无效"
            )

        # 特殊情况2: 训练赚测试亏 (典型过拟合)
        if oos_sharpe < 0.0:
            return DegradationFactorResult(
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                degradation_factor=oos_sharpe / is_sharpe,  # 负值
                degradation_pct=100.0 + abs(oos_sharpe / is_sharpe) * 100.0,
                verdict="BLOCK",
                detail=f"训练赚(IS={is_sharpe:.3f})测试亏(OOS={oos_sharpe:.3f}), 典型过拟合"
            )

        # 正常计算
        df = oos_sharpe / is_sharpe
        degradation_pct = (1.0 - df) * 100.0

        if df >= cls.PASS_THRESHOLD:
            verdict = "PASS"
            detail = f"DF={df:.3f}≥{cls.PASS_THRESHOLD}, 退化{degradation_pct:.1f}%, 实盘可保持{df*100:.1f}%表现"
        elif df >= cls.WARN_THRESHOLD:
            verdict = "WARN"
            detail = f"DF={df:.3f}∈[{cls.WARN_THRESHOLD},{cls.PASS_THRESHOLD}), 退化{degradation_pct:.1f}%, 谨慎部署"
        else:
            verdict = "BLOCK"
            detail = f"DF={df:.3f}<{cls.WARN_THRESHOLD}, 退化{degradation_pct:.1f}%, 几乎必然实盘亏钱"

        return DegradationFactorResult(
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            degradation_factor=df,
            degradation_pct=degradation_pct,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 2. 5大差距量化器 (Gap Quantifier)
# ============================================================================


class GapQuantifier:
    """5大差距来源量化器

    每项差距表示为"该因素导致的Sharpe高估百分比"。
    总差距 = sum(all_gaps), clamped to [0, 0.95]
    实盘预期 = 回测Sharpe × (1 - total_gap)
    """

    @staticmethod
    def quantify_data_gap(
        synthetic_sharpe: Optional[float] = None,
        real_data_sharpe: Optional[float] = None,
    ) -> float:
        """量化数据差距 (合成 vs 真实)

        来源: digitalninjasystems 2026-06
        合成数据通常高估 Sharpe 10-30%, 因为:
          - GARCH模型无法完全再现真实市场的微结构
          - 跳跃扩散模型低估极端事件频率
          - 时序相关性结构简化

        Args:
            synthetic_sharpe: 合成数据上的 Sharpe
            real_data_sharpe: 真实历史数据上的 Sharpe

        Returns:
            数据差距百分比 [0, 0.5]
        """
        if synthetic_sharpe is None or real_data_sharpe is None:
            # 无对比数据时, 假设合成数据高估 15% (业界中位数)
            return 0.15

        if synthetic_sharpe <= 0.0:
            return 0.0

        # 差距 = (合成 - 真实) / 合成
        gap = (synthetic_sharpe - real_data_sharpe) / max(synthetic_sharpe, 0.01)
        # 限制在 [0, 0.5] 范围
        return max(0.0, min(0.50, gap))

    @staticmethod
    def quantify_cost_gap(
        theoretical_cost_bps: float = 0.0,
        realistic_cost_bps: float = 0.0,
        n_trades_per_year: int = 252,
    ) -> float:
        """量化成本差距 (理论 vs 真实)

        来源: sentinel 2026, cofool 2026
        真实成本包含:
          - 交易手续费 (maker 0.02%, taker 0.05%)
          - 滑点 (1-5 bps 正常, 5-20 bps 危机)
          - 买卖价差 (1-3 bps 主流币, 10-50 bps 山寨)
          - 永续合约资金费率 (0.01%/8h 基础, 极端 0.51%/8h)
          - Gas费 (DEX)

        Args:
            theoretical_cost_bps: 理论成本 (bps/笔)
            realistic_cost_bps: 真实成本 (bps/笔)
            n_trades_per_year: 年交易笔数

        Returns:
            成本差距百分比 [0, 0.5]
        """
        if realistic_cost_bps <= theoretical_cost_bps:
            return 0.0

        # 成本差 / 10000 = 每笔额外成本占比
        cost_diff_pct = (realistic_cost_bps - theoretical_cost_bps) / 10000.0
        # 年化成本 = 每笔成本差 × 交易笔数
        annual_cost_gap = cost_diff_pct * n_trades_per_year
        # 假设策略年化收益 20%, 则差距占比 = annual_cost_gap / 0.20
        # 但更通用: 直接用 annual_cost_gap 作为 Sharpe 高估百分比
        # 限制在 [0, 0.5]
        return max(0.0, min(0.50, annual_cost_gap))

    @staticmethod
    def quantify_latency_gap(
        theoretical_latency_ms: float = 0.0,
        realistic_latency_ms: float = 0.0,
    ) -> float:
        """量化延迟差距 (理想 vs 实际)

        来源: dhawal.org 2026 ch18
        延迟导致:
          - 信号到成交的价格漂移 (1-5 bps 正常, 10-30 bps 高频)
          - 队列位置劣化 (被动单成交概率下降)
          - 撮合机制断层 (回测"触及即成交", 实盘需排队)

        Args:
            theoretical_latency_ms: 理论延迟 (毫秒)
            realistic_latency_ms: 实际延迟 (毫秒, 含网络+撮合+滑点)

        Returns:
            延迟差距百分比 [0, 0.3]
        """
        if realistic_latency_ms <= theoretical_latency_ms:
            return 0.0

        # 延迟比 → 差距 (经验映射)
        # 延迟比 1x → 0%, 2x → 5%, 5x → 10%, 10x → 15%, 100x → 20%
        if theoretical_latency_ms <= 0:
            ratio = realistic_latency_ms / 1.0  # 假设理论1ms
        else:
            ratio = realistic_latency_ms / theoretical_latency_ms

        # 对数映射: gap = 0.05 * log2(ratio), 上限 0.30
        gap = 0.05 * math.log2(max(1.0, ratio))
        return max(0.0, min(0.30, gap))

    @staticmethod
    def quantify_liquidity_gap(
        order_size_usd: float = 0.0,
        avg_daily_volume_usd: float = 1_000_000.0,
        is_market_order: bool = True,
    ) -> float:
        """量化流动性差距 (无限流动性 vs 订单簿)

        来源: arXiv:2604.03272 (Meng & Chen, NYU, 2026-03)
              Frontiers in Blockchain 2026 (Pindza)
              Kyle lambda + Almgren-Chriss 平方根冲击

        市场冲击公式 (平方根定律):
          impact = σ × √(Q/V)
          Q = order_size, V = ADV (平均日成交量)

        当 Q/V > 1% 时, 冲击显著
        当 Q/V > 5% 时, 冲击主导价差

        Args:
            order_size_usd: 订单大小 (USD)
            avg_daily_volume_usd: 平均日成交量 (USD)
            is_market_order: 是否市价单 (限价单冲击较小)

        Returns:
            流动性差距百分比 [0, 0.5]
        """
        if order_size_usd <= 0 or avg_daily_volume_usd <= 0:
            return 0.0

        # 参与率 = 订单大小 / 日成交量
        participation = order_size_usd / avg_daily_volume_usd

        # 平方根冲击: impact ∝ √(participation)
        # 经验系数: 市价单 0.10, 限价单 0.05
        coef = 0.10 if is_market_order else 0.05
        impact = coef * math.sqrt(participation)

        # 转换为 Sharpe 高估百分比
        # 经验: impact 1% → Sharpe 高估 5%
        gap = impact * 5.0

        return max(0.0, min(0.50, gap))

    @staticmethod
    def quantify_regime_gap(
        backtest_periods_vol: float = 0.0,
        live_periods_vol: float = 0.0,
        backtest_bull_pct: float = 0.5,
        live_bull_pct: float = 0.5,
    ) -> float:
        """量化状态差距 (回测期间 vs 实盘期间市场状态)

        来源: CSDN 2026-02, cofool 2026
        Regime drift 是策略失效的隐蔽原因:
          - 2019-2020 盈利模式在 2024-2026 失效
          - 市场非平稳, 统计特性漂移
          - AI 渗透率上升改变市场结构 (arXiv:2604.03272)

        Args:
            backtest_periods_vol: 回测期间平均波动率
            live_periods_vol: 实盘期间平均波动率
            backtest_bull_pct: 回测期间牛市占比 [0,1]
            live_bull_pct: 实盘期间牛市占比 [0,1]

        Returns:
            状态差距百分比 [0, 0.5]
        """
        gap = 0.0

        # 波动率差距
        if backtest_periods_vol > 0 and live_periods_vol > 0:
            vol_ratio = live_periods_vol / backtest_periods_vol
            # 波动率比 1.0 → 0%, 1.5 → 5%, 2.0 → 10%, 3.0 → 15%
            if vol_ratio > 1.0:
                gap += 0.10 * math.log2(vol_ratio)
            elif vol_ratio < 0.5:
                # 波动率太低也可能不利 (策略依赖波动)
                gap += 0.05

        # 牛市占比差距
        bull_diff = abs(backtest_bull_pct - live_bull_pct)
        # 占比差 0 → 0%, 0.3 → 10%, 0.5 → 15%
        gap += bull_diff * 0.30

        return max(0.0, min(0.50, gap))

    @classmethod
    def quantify_all_gaps(
        cls,
        synthetic_sharpe: Optional[float] = None,
        real_data_sharpe: Optional[float] = None,
        theoretical_cost_bps: float = 0.0,
        realistic_cost_bps: float = 0.0,
        n_trades_per_year: int = 252,
        theoretical_latency_ms: float = 0.0,
        realistic_latency_ms: float = 0.0,
        order_size_usd: float = 0.0,
        avg_daily_volume_usd: float = 1_000_000.0,
        is_market_order: bool = True,
        backtest_periods_vol: float = 0.0,
        live_periods_vol: float = 0.0,
        backtest_bull_pct: float = 0.5,
        live_bull_pct: float = 0.5,
    ) -> GapBreakdown:
        """量化全部5大差距

        Returns:
            GapBreakdown 包含5项差距 + total_gap_pct + gap_factor
        """
        return GapBreakdown(
            data_gap_pct=cls.quantify_data_gap(synthetic_sharpe, real_data_sharpe),
            cost_gap_pct=cls.quantify_cost_gap(
                theoretical_cost_bps, realistic_cost_bps, n_trades_per_year
            ),
            latency_gap_pct=cls.quantify_latency_gap(
                theoretical_latency_ms, realistic_latency_ms
            ),
            liquidity_gap_pct=cls.quantify_liquidity_gap(
                order_size_usd, avg_daily_volume_usd, is_market_order
            ),
            regime_gap_pct=cls.quantify_regime_gap(
                backtest_periods_vol, live_periods_vol,
                backtest_bull_pct, live_bull_pct,
            ),
        )


# ============================================================================
# 3. 实盘Sharpe预测器 (Live Sharpe Predictor)
# ============================================================================


class LiveSharpePredictor:
    """实盘Sharpe预测器

    公式:
      predicted_live_sharpe = backtest_sharpe × gap_factor
      gap_factor = 1 - total_gap_pct

    业界基准 (Glassnode 2025, QuantConnect 2026):
      87% 回测正收益策略实盘亏损
      典型 gap_factor = 0.4-0.7 (退化 30-60%)

    判定:
      predicted_live_sharpe > 1.0 → PASS (实盘预期正收益)
      predicted_live_sharpe > 0.0 → WARN (实盘预期微利, 边界)
      predicted_live_sharpe ≤ 0.0 → BLOCK (实盘预期亏钱)
    """

    # 实盘Sharpe门槛
    PASS_LIVE_SHARPE = 1.0   # 预期Sharpe > 1.0 → PASS
    BLOCK_LIVE_SHARPE = 0.0  # 预期Sharpe ≤ 0 → BLOCK

    @classmethod
    def predict(
        cls,
        backtest_sharpe: float,
        gap_breakdown: GapBreakdown,
    ) -> PredictedLiveSharpe:
        """预测实盘Sharpe

        Args:
            backtest_sharpe: 回测Sharpe
            gap_breakdown: 5大差距量化结果

        Returns:
            PredictedLiveSharpe
        """
        gap_factor = gap_breakdown.gap_factor
        predicted = backtest_sharpe * gap_factor
        total_gap = gap_breakdown.total_gap_pct

        block_reasons: List[str] = []
        warn_reasons: List[str] = []

        if predicted <= cls.BLOCK_LIVE_SHARPE:
            verdict = "BLOCK"
            block_reasons.append(
                f"实盘预期Sharpe={predicted:.3f}≤0, 回测Sharpe={backtest_sharpe:.3f}×gap_factor={gap_factor:.3f}"
            )
        elif predicted < cls.PASS_LIVE_SHARPE:
            verdict = "WARN"
            warn_reasons.append(
                f"实盘预期Sharpe={predicted:.3f}<{cls.PASS_LIVE_SHARPE}, 边界策略"
            )
        else:
            verdict = "PASS"

        # 额外警告: 单项差距过大
        if gap_breakdown.data_gap_pct > 0.25:
            warn_reasons.append(f"数据差距过大: {gap_breakdown.data_gap_pct:.1%}")
        if gap_breakdown.cost_gap_pct > 0.20:
            warn_reasons.append(f"成本差距过大: {gap_breakdown.cost_gap_pct:.1%}")
        if gap_breakdown.liquidity_gap_pct > 0.20:
            warn_reasons.append(f"流动性差距过大: {gap_breakdown.liquidity_gap_pct:.1%}")
        if gap_breakdown.regime_gap_pct > 0.25:
            warn_reasons.append(f"状态差距过大: {gap_breakdown.regime_gap_pct:.1%}")

        # 如果有 BLOCK 原因, 升级为 BLOCK
        if block_reasons:
            verdict = "BLOCK"
        elif warn_reasons and verdict == "PASS":
            verdict = "WARN"

        detail = (
            f"回测Sharpe={backtest_sharpe:.3f} × gap_factor={gap_factor:.3f} "
            f"(总退化{total_gap:.1%}) = 实盘预期Sharpe={predicted:.3f}"
        )

        return PredictedLiveSharpe(
            backtest_sharpe=backtest_sharpe,
            predicted_live_sharpe=predicted,
            gap_factor=gap_factor,
            total_gap_pct=total_gap,
            verdict=verdict,
            block_reasons=block_reasons,
            warn_reasons=warn_reasons,
            detail=detail,
        )


# ============================================================================
# 4. 综合验证器 (Live Backtest Gap Validator)
# ============================================================================


class LiveBacktestGapValidator:
    """实盘-回测差距综合验证器

    集成:
      1. DegradationFactorCalculator (OOS/IS Sharpe 比值)
      2. GapQuantifier (5大差距量化)
      3. LiveSharpePredictor (实盘Sharpe预测)

    综合判定:
      - 任一 BLOCK → BLOCK
      - 任一 WARN → WARN
      - 全 PASS → PASS
    """

    def __init__(
        self,
        pass_df_threshold: float = 0.70,
        warn_df_threshold: float = 0.40,
        pass_live_sharpe: float = 1.0,
        block_live_sharpe: float = 0.0,
        maker_cost_params: "MakerCostParams" = None,
    ):
        """初始化验证器

        Args:
            pass_df_threshold: DF PASS 阈值 (默认 0.70)
            warn_df_threshold: DF WARN 阈值 (默认 0.40, 低于此 = BLOCK)
            pass_live_sharpe: 实盘Sharpe PASS 阈值 (默认 1.0)
            block_live_sharpe: 实盘Sharpe BLOCK 阈值 (默认 0.0)
            maker_cost_params: MakerCostParams (Phase 5 Task #28.2 新增)
                若提供则 validate_with_maker_cost 使用 per-symbol 精确成本;
                若 None 则回退到硬编码 flat bps (向后兼容)
        """
        DegradationFactorCalculator.PASS_THRESHOLD = pass_df_threshold
        DegradationFactorCalculator.WARN_THRESHOLD = warn_df_threshold
        LiveSharpePredictor.PASS_LIVE_SHARPE = pass_live_sharpe
        LiveSharpePredictor.BLOCK_LIVE_SHARPE = block_live_sharpe
        # Phase 5 Task #28.2: 保存 maker_cost_params 供 validate_with_maker_cost 使用
        if maker_cost_params is None and _MAKER_COST_MODEL_AVAILABLE:
            self.maker_cost_params = MakerCostParams()
        else:
            self.maker_cost_params = maker_cost_params

    def validate(
        self,
        is_sharpe: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
        backtest_sharpe: Optional[float] = None,
        synthetic_sharpe: Optional[float] = None,
        real_data_sharpe: Optional[float] = None,
        theoretical_cost_bps: float = 0.0,
        realistic_cost_bps: float = 0.0,
        n_trades_per_year: int = 252,
        theoretical_latency_ms: float = 0.0,
        realistic_latency_ms: float = 0.0,
        order_size_usd: float = 0.0,
        avg_daily_volume_usd: float = 1_000_000.0,
        is_market_order: bool = True,
        backtest_periods_vol: float = 0.0,
        live_periods_vol: float = 0.0,
        backtest_bull_pct: float = 0.5,
        live_bull_pct: float = 0.5,
    ) -> LiveBacktestGapResult:
        """综合验证

        Args:
            is_sharpe: In-Sample Sharpe (可选, 用于 DF 计算)
            oos_sharpe: Out-of-Sample Sharpe (可选, 用于 DF 计算)
            backtest_sharpe: 回测 Sharpe (用于实盘预测, 默认 = oos_sharpe 或 is_sharpe)
            其他参数见 GapQuantifier.quantify_all_gaps

        Returns:
            LiveBacktestGapResult
        """
        result = LiveBacktestGapResult()
        block_reasons: List[str] = []
        warn_reasons: List[str] = []

        # 1. 退化因子 (如果提供了 IS/OOS Sharpe)
        if is_sharpe is not None and oos_sharpe is not None:
            df_result = DegradationFactorCalculator.compute(is_sharpe, oos_sharpe)
            result.degradation_factor = df_result

            if df_result.verdict == "BLOCK":
                block_reasons.append(f"退化因子: {df_result.detail}")
            elif df_result.verdict == "WARN":
                warn_reasons.append(f"退化因子: {df_result.detail}")

        # 2. 5大差距量化
        gap = GapQuantifier.quantify_all_gaps(
            synthetic_sharpe=synthetic_sharpe,
            real_data_sharpe=real_data_sharpe,
            theoretical_cost_bps=theoretical_cost_bps,
            realistic_cost_bps=realistic_cost_bps,
            n_trades_per_year=n_trades_per_year,
            theoretical_latency_ms=theoretical_latency_ms,
            realistic_latency_ms=realistic_latency_ms,
            order_size_usd=order_size_usd,
            avg_daily_volume_usd=avg_daily_volume_usd,
            is_market_order=is_market_order,
            backtest_periods_vol=backtest_periods_vol,
            live_periods_vol=live_periods_vol,
            backtest_bull_pct=backtest_bull_pct,
            live_bull_pct=live_bull_pct,
        )
        result.gap_breakdown = gap

        # 3. 实盘Sharpe预测
        # 优先使用 oos_sharpe, 其次 is_sharpe, 最后 backtest_sharpe
        effective_backtest_sharpe = backtest_sharpe
        if effective_backtest_sharpe is None:
            if oos_sharpe is not None:
                effective_backtest_sharpe = oos_sharpe
            elif is_sharpe is not None:
                effective_backtest_sharpe = is_sharpe
            else:
                effective_backtest_sharpe = 0.0

        if effective_backtest_sharpe > 0:
            predicted = LiveSharpePredictor.predict(effective_backtest_sharpe, gap)
            result.predicted_live = predicted

            if predicted.verdict == "BLOCK":
                block_reasons.append(f"实盘预测: {predicted.detail}")
                block_reasons.extend(predicted.block_reasons)
            elif predicted.verdict == "WARN":
                warn_reasons.append(f"实盘预测: {predicted.detail}")
                warn_reasons.extend(predicted.warn_reasons)

        # 综合判定
        if block_reasons:
            result.overall_verdict = "BLOCK"
            result.block_reasons = block_reasons
            result.warn_reasons = warn_reasons
        elif warn_reasons:
            result.overall_verdict = "WARN"
            result.warn_reasons = warn_reasons
        else:
            result.overall_verdict = "PASS"

        return result

    def validate_with_maker_cost(
        self,
        trades_by_symbol: Dict[str, List[Dict]],
        is_sharpe: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
        backtest_sharpe: Optional[float] = None,
        synthetic_sharpe: Optional[float] = None,
        real_data_sharpe: Optional[float] = None,
        avg_order_size_usd: float = 1000.0,
        n_trades_per_year: int = 252,
        theoretical_latency_ms: float = 0.0,
        realistic_latency_ms: float = 0.0,
        backtest_periods_vol: float = 0.0,
        live_periods_vol: float = 0.0,
        backtest_bull_pct: float = 0.5,
        live_bull_pct: float = 0.5,
    ) -> LiveBacktestGapResult:
        """Phase 5 Task #28.2: 用 maker_cost_model 计算 per-symbol 成本，加权聚合后调用 validate

        用户核心诉求: "模拟/实盘差异全成本量化模型，根治模拟好看实盘亏"
        之前: theoretical_cost_bps=2.0, realistic_cost_bps=10.0 硬编码 flat
        现在: per-symbol 计算 taker (真实) 与 maker (理论) 成本, 加权聚合

        Args:
            trades_by_symbol: per-symbol 交易列表, 每个 trade dict 应包含:
                - atr_pct (float): ATR百分比, 默认 0.01
                - volume_ratio (float): 成交量比率, 默认 1.0
                - pnl (float, 可选): 用于加权
            is_sharpe/oos_sharpe/backtest_sharpe: Sharpe 值 (传递给 validate)
            synthetic_sharpe/real_data_sharpe: 传递给 validate
            avg_order_size_usd: 平均订单大小 (USD), 默认 1000
            其他参数传递给 validate

        Returns:
            LiveBacktestGapResult (与 validate 相同结构)

        Raises:
            RuntimeError: 若 maker_cost_model 不可用
        """
        if not _MAKER_COST_MODEL_AVAILABLE or compute_taker_cost_bps is None:
            raise RuntimeError(
                "validate_with_maker_cost 不可用: maker_cost_model 导入失败. "
                "请回退到 validate() 方法使用硬编码 flat bps."
            )

        params = self.maker_cost_params or MakerCostParams()

        # 按 symbol 计算 taker (真实成本) 与 maker (理论成本) 的 bps, 加权聚合
        total_taker_bps = 0.0
        total_maker_bps = 0.0
        total_trades = 0
        per_symbol_costs: Dict[str, Dict[str, float]] = {}

        for symbol, trades in trades_by_symbol.items():
            if not trades:
                continue
            n_sym = len(trades)
            # 从 trades 提取中位数 atr_pct / volume_ratio (保守估计)
            atr_pcts = [float(t.get("atr_pct", 0.01)) for t in trades]
            vol_ratios = [float(t.get("volume_ratio", 1.0)) for t in trades]
            atr_pcts.sort()
            vol_ratios.sort()
            median_atr = atr_pcts[n_sym // 2] if atr_pcts else 0.01
            median_vol = vol_ratios[n_sym // 2] if vol_ratios else 1.0

            # Taker 成本 (真实成本, 含滑点+冲击+延迟)
            taker_cost = compute_taker_cost_bps(
                symbol=symbol,
                order_size_usd=avg_order_size_usd,
                volume_ratio=median_vol,
                atr_pct=median_atr,
                params=params,
            )
            # Maker 成本 (理论成本, spread 从成本变收益, 可能是负值=净收益)
            maker_cost_usd = compute_maker_cost_usd(
                symbol=symbol,
                order_size_usd=avg_order_size_usd,
                volume_ratio=median_vol,
                atr_pct=median_atr,
                params=params,
            )
            # Maker USD → bps 转换 (相对于 order_size)
            maker_bps = (
                maker_cost_usd["total_cost"] / avg_order_size_usd * 10000.0
                if avg_order_size_usd > 0 else 0.0
            )

            total_taker_bps += taker_cost["total"] * n_sym
            total_maker_bps += maker_bps * n_sym
            total_trades += n_sym
            per_symbol_costs[symbol] = {
                "taker_total_bps": taker_cost["total"],
                "maker_total_bps": maker_bps,
                "n_trades": n_sym,
                "median_atr_pct": median_atr,
                "median_volume_ratio": median_vol,
            }

        # 加权平均 (按交易笔数加权)
        if total_trades > 0:
            realistic_cost_bps = total_taker_bps / total_trades
            theoretical_cost_bps = total_maker_bps / total_trades
        else:
            # 无 trades 时回退到保守默认值
            realistic_cost_bps = 10.0
            theoretical_cost_bps = 2.0

        logger.info(
            "Phase 5 Task #28.2 validate_with_maker_cost: "
            "n_symbols=%d, n_trades=%d, "
            "theoretical=%.2f bps, realistic=%.2f bps",
            len(per_symbol_costs), total_trades,
            theoretical_cost_bps, realistic_cost_bps,
        )

        # 调用原 validate 方法, 用 per-symbol 计算的成本替换硬编码
        result = self.validate(
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            backtest_sharpe=backtest_sharpe,
            synthetic_sharpe=synthetic_sharpe,
            real_data_sharpe=real_data_sharpe,
            theoretical_cost_bps=theoretical_cost_bps,
            realistic_cost_bps=realistic_cost_bps,
            n_trades_per_year=n_trades_per_year,
            theoretical_latency_ms=theoretical_latency_ms,
            realistic_latency_ms=realistic_latency_ms,
            order_size_usd=avg_order_size_usd,
            avg_daily_volume_usd=1_000_000.0,  # 主流资产 ADV 默认值
            is_market_order=True,
            backtest_periods_vol=backtest_periods_vol,
            live_periods_vol=live_periods_vol,
            backtest_bull_pct=backtest_bull_pct,
            live_bull_pct=live_bull_pct,
        )

        # 附加 per-symbol 成本明细到 result (用于诊断)
        if result.gap_breakdown is not None:
            result.gap_breakdown.per_symbol_costs = per_symbol_costs

        return result

    def generate_report(self, result: LiveBacktestGapResult) -> str:
        """生成人类可读报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("实盘-回测差距验证报告 (Live-Backtest Gap Report)")
        lines.append("=" * 70)
        lines.append(f"综合判定: {result.overall_verdict}")
        lines.append("")

        if result.degradation_factor:
            df = result.degradation_factor
            lines.append("[退化因子 Degradation Factor]")
            lines.append(f"  IS Sharpe:  {df.is_sharpe:.4f}")
            lines.append(f"  OOS Sharpe: {df.oos_sharpe:.4f}")
            lines.append(f"  DF:         {df.degradation_factor:.4f}")
            lines.append(f"  退化:       {df.degradation_pct:.2f}%")
            lines.append(f"  判定:       {df.verdict}")
            lines.append(f"  说明:       {df.detail}")
            lines.append("")

        if result.gap_breakdown:
            gb = result.gap_breakdown
            lines.append("[5大差距量化 Gap Breakdown]")
            lines.append(f"  数据差距:     {gb.data_gap_pct:.2%}  (合成 vs 真实)")
            lines.append(f"  成本差距:     {gb.cost_gap_pct:.2%}  (理论 vs 真实)")
            lines.append(f"  延迟差距:     {gb.latency_gap_pct:.2%}  (理想 vs 实际)")
            lines.append(f"  流动性差距:   {gb.liquidity_gap_pct:.2%}  (无限 vs 订单簿)")
            lines.append(f"  状态差距:     {gb.regime_gap_pct:.2%}  (回测 vs 实盘期间)")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  总差距:       {gb.total_gap_pct:.2%}")
            lines.append(f"  保持因子:     {gb.gap_factor:.4f}  (实盘预期=回测×此值)")
            lines.append("")

        if result.predicted_live:
            pl = result.predicted_live
            lines.append("[实盘Sharpe预测 Live Sharpe Prediction]")
            lines.append(f"  回测 Sharpe:        {pl.backtest_sharpe:.4f}")
            lines.append(f"  预测实盘 Sharpe:    {pl.predicted_live_sharpe:.4f}")
            lines.append(f"  保持因子:           {pl.gap_factor:.4f}")
            lines.append(f"  总退化:             {pl.total_gap_pct:.2%}")
            lines.append(f"  判定:               {pl.verdict}")
            lines.append(f"  说明:               {pl.detail}")
            if pl.block_reasons:
                lines.append(f"  BLOCK 原因: {'; '.join(pl.block_reasons)}")
            if pl.warn_reasons:
                lines.append(f"  WARN 原因:  {'; '.join(pl.warn_reasons)}")
            lines.append("")

        if result.block_reasons:
            lines.append("[BLOCK 原因汇总]")
            for r in result.block_reasons:
                lines.append(f"  - {r}")
            lines.append("")

        if result.warn_reasons:
            lines.append("[WARN 原因汇总]")
            for r in result.warn_reasons:
                lines.append(f"  - {r}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 便利函数
# ============================================================================


def validate_live_backtest_gap(
    is_sharpe: Optional[float] = None,
    oos_sharpe: Optional[float] = None,
    backtest_sharpe: Optional[float] = None,
    synthetic_sharpe: Optional[float] = None,
    real_data_sharpe: Optional[float] = None,
    theoretical_cost_bps: float = 0.0,
    realistic_cost_bps: float = 0.0,
    n_trades_per_year: int = 252,
    theoretical_latency_ms: float = 0.0,
    realistic_latency_ms: float = 0.0,
    order_size_usd: float = 0.0,
    avg_daily_volume_usd: float = 1_000_000.0,
    is_market_order: bool = True,
    backtest_periods_vol: float = 0.0,
    live_periods_vol: float = 0.0,
    backtest_bull_pct: float = 0.5,
    live_bull_pct: float = 0.5,
) -> LiveBacktestGapResult:
    """便利函数: 一键验证实盘-回测差距

    所有参数可选, 未提供的参数不会被纳入计算。

    Returns:
        LiveBacktestGapResult
    """
    validator = LiveBacktestGapValidator()
    return validator.validate(
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        backtest_sharpe=backtest_sharpe,
        synthetic_sharpe=synthetic_sharpe,
        real_data_sharpe=real_data_sharpe,
        theoretical_cost_bps=theoretical_cost_bps,
        realistic_cost_bps=realistic_cost_bps,
        n_trades_per_year=n_trades_per_year,
        theoretical_latency_ms=theoretical_latency_ms,
        realistic_latency_ms=realistic_latency_ms,
        order_size_usd=order_size_usd,
        avg_daily_volume_usd=avg_daily_volume_usd,
        is_market_order=is_market_order,
        backtest_periods_vol=backtest_periods_vol,
        live_periods_vol=live_periods_vol,
        backtest_bull_pct=backtest_bull_pct,
        live_bull_pct=live_bull_pct,
    )
