# -*- coding: utf-8 -*-
"""
市场冲击模型 (Market Impact Model) — R15 核心模块

解决"模拟牛逼实盘亏钱"的核心根因之一: 流动性幻觉
  业界数据 (dhawal.org 2026 ch18):
    - 45% 策略因流动性不足实盘亏损
    - 回测假设"无限流动性, 触及即成交"
    - 实盘: 大单推动价格, 订单簿深度有限, 队列位置决定成交

业界前沿 (2026):
  - 平方根冲击定律 (Almgren-Chriss 2000, 实证基准):
      impact = σ × √(Q/V)
      Q = order_size, V = ADV (平均日成交量)
  - Kyle λ 静态模型 (Kyle 1985):
      impact = λ × order_size
      λ = 信息不对称系数
  - 时变 Kyle λ (arXiv:2604.03272, Meng & Chen NYU 2026-03):
      λ'(ϕ) 随 AI 渗透率 ϕ 递减
      系统性风险 r(ϕ) = ϕρβ/λ'(ϕ) 对 ϕ 凸
      尾部损失放大 18-54%
  - 多智能体 Almgren-Chriss (arXiv:2605.20348, Koulouris & Campajola 2026-05):
      DRL 智能体的"超竞争结果"源于记忆与 intra-episode 反馈

核心公式:
  真实成交价格 = 期望价格 × (1 + market_impact_pct)
  market_impact_pct = sqrt_impact + kyle_impact + temporary_impact

来源:
  - Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
  - Kyle (1985) "Continuous Auctions and Insider Trading"
  - arXiv:2604.03272 (Meng & Chen, NYU, 2026-03)
  - arXiv:2605.20348 (Koulouris & Campajola, UCL, 2026-05)
  - arXiv:2410.04867v2 (Palmaria, Lillo, Eisler, 2026-03) 时变冲击参数
  - Frontiers in Blockchain 2026 (Pindza) 加密市场 Kyle λ 微结构 alpha
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional


logger = logging.getLogger("hermes.market_impact")


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class MarketImpactResult:
    """市场冲击计算结果"""

    order_size_usd: float           # 订单大小 (USD)
    avg_daily_volume_usd: float     # 平均日成交量 (USD)
    participation_rate: float       # 参与率 = order/ADV
    volatility: float               # 资产波动率 (日化)
    # 三大冲击分量
    sqrt_impact_pct: float = 0.0    # 平方根冲击 (Almgren-Chriss)
    kyle_impact_pct: float = 0.0    # Kyle λ 冲击 (信息不对称)
    temporary_impact_pct: float = 0.0  # 临时冲击 (买卖价差恢复)
    # 综合
    total_impact_pct: float = 0.0   # 总冲击百分比
    total_impact_bps: float = 0.0   # 总冲击 (bps)
    expected_fill_price: float = 0.0  # 预期成交价 (含冲击)
    original_price: float = 0.0     # 原始期望价格
    # 风险评估
    is_illiquid: bool = False       # 流动性不足标志
    verdict: str = "PASS"           # PASS / WARN / BLOCK
    detail: str = ""


@dataclass(slots=True)
class CostAdjustedSharpe:
    """成本调整后的Sharpe"""

    original_sharpe: float          # 原始Sharpe (无成本)
    impact_cost_bps: float          # 市场冲击成本 (bps/笔)
    fee_cost_bps: float             # 手续费 (bps/笔)
    slippage_cost_bps: float        # 滑点 (bps/笔)
    funding_cost_bps: float         # 资金费率成本 (bps/笔, 永续合约)
    total_cost_bps: float           # 总成本 (bps/笔)
    n_trades_per_year: int          # 年交易笔数
    annual_cost_pct: float          # 年化成本占比
    adjusted_sharpe: float          # 调整后Sharpe
    sharpe_degradation_pct: float   # Sharpe退化百分比
    verdict: str = "PASS"
    detail: str = ""


@dataclass(slots=True)
class MarketImpactValidationResult:
    """市场冲击综合验证结果"""

    impact_result: Optional[MarketImpactResult] = None
    cost_adjusted: Optional[CostAdjustedSharpe] = None
    overall_verdict: str = "PASS"
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)


# ============================================================================
# 1. 市场冲击计算器 (Market Impact Calculator)
# ============================================================================


class MarketImpactCalculator:
    """市场冲击计算器

    集成三大冲击模型:
      1. Almgren-Chriss 平方根冲击 (实证基准)
      2. Kyle λ 信息不对称冲击
      3. 临时冲击 (买卖价差恢复)

    业界基准 (Frontiers in Blockchain 2026):
      - 加密市场 Kyle λ 微结构信号含真实但弱信息
      - 标准零售费率下不可获利
      - >5% 参与率时冲击主导价差
    """

    # 平方根冲击系数 (Almgren-Chriss 经验值)
    # 来源: Almgren & Chriss 2000 + 实证校准
    SQRT_IMPACT_COEF = 0.10  # 主流资产
    SQRT_IMPACT_COEF_ILLIQUID = 0.20  # 流动性差的资产

    # Kyle λ 默认值 (来源: Frontiers in Blockchain 2026)
    # 加密市场: 1e-6 ~ 1e-4 (取决于资产流动性)
    KYLE_LAMBDA_DEFAULT = 1e-5
    KYLE_LAMBDA_HIGH_INFO = 1e-4  # 高信息不对称

    # 临时冲击恢复系数 (买卖价差的一半)
    TEMPORARY_IMPACT_FACTOR = 0.5

    # 流动性阈值
    LIQUIDITY_WARN_PARTICIPATION = 0.01  # 1% 参与率 → WARN
    LIQUIDITY_BLOCK_PARTICIPATION = 0.05  # 5% 参与率 → BLOCK
    LIQUIDITY_CRITICAL_PARTICIPATION = 0.20  # 20% 参与率 → 严重

    @classmethod
    def compute_impact(
        cls,
        order_size_usd: float,
        avg_daily_volume_usd: float,
        volatility: float = 0.02,
        current_price: float = 50000.0,
        is_buy: bool = True,
        kyle_lambda: Optional[float] = None,
        ai_penetration: float = 0.0,
        is_illiquid_asset: bool = False,
    ) -> MarketImpactResult:
        """计算市场冲击

        Args:
            order_size_usd: 订单大小 (USD)
            avg_daily_volume_usd: 平均日成交量 (USD)
            volatility: 资产日波动率 (如 0.02 = 2%)
            current_price: 当前价格
            is_buy: 是否买入 (买入冲击正向, 卖出冲击负向)
            kyle_lambda: Kyle λ 系数 (None=使用默认)
            ai_penetration: AI 渗透率 [0,1] (来源: arXiv:2604.03272)
                AI 渗透率高 → λ 下降但系统性风险上升
            is_illiquid_asset: 是否流动性差的资产

        Returns:
            MarketImpactResult
        """
        if order_size_usd <= 0 or avg_daily_volume_usd <= 0:
            return MarketImpactResult(
                order_size_usd=order_size_usd,
                avg_daily_volume_usd=avg_daily_volume_usd,
                participation_rate=0.0,
                volatility=volatility,
                expected_fill_price=current_price,
                original_price=current_price,
                verdict="PASS",
                detail="无订单或无成交量"
            )

        # 参与率
        participation = order_size_usd / avg_daily_volume_usd

        # 1. 平方根冲击 (Almgren-Chriss)
        # impact = σ × coef × √(Q/V)
        coef = cls.SQRT_IMPACT_COEF_ILLIQUID if is_illiquid_asset else cls.SQRT_IMPACT_COEF
        sqrt_impact = volatility * coef * math.sqrt(participation)

        # 2. Kyle λ 冲击 (信息不对称)
        # 时变 Kyle λ (arXiv:2604.03272): λ'(ϕ) = λ × (1 - 0.3×ϕ)
        # AI 渗透率上升 → λ 下降 (信息更对称)
        # 但系统性风险上升: 尾部损失放大 18-54%
        if kyle_lambda is None:
            kyle_lambda = cls.KYLE_LAMBDA_HIGH_INFO if is_illiquid_asset else cls.KYLE_LAMBDA_DEFAULT
        effective_lambda = kyle_lambda * (1.0 - 0.3 * ai_penetration)
        # Kyle 冲击 = λ × Q (Q 以 USD 计)
        kyle_impact = effective_lambda * order_size_usd
        # 转换为百分比 (除以订单大小得到单位成本)
        kyle_impact_pct = kyle_impact / max(order_size_usd, 1.0)
        # 限制在合理范围
        kyle_impact_pct = max(0.0, min(0.05, kyle_impact_pct))

        # 3. 临时冲击 (买卖价差恢复)
        # 假设买卖价差 = volatility × 0.1 (主流资产)
        # 临时冲击 = 价差 × 0.5 (一半恢复)
        spread = volatility * 0.1 if not is_illiquid_asset else volatility * 0.3
        temporary_impact = spread * cls.TEMPORARY_IMPACT_FACTOR

        # 总冲击
        total_impact_pct = sqrt_impact + kyle_impact_pct + temporary_impact
        total_impact_bps = total_impact_pct * 10000

        # 预期成交价 (买入正向, 卖出负向)
        direction = 1.0 if is_buy else -1.0
        expected_fill = current_price * (1.0 + direction * total_impact_pct)

        # 流动性评估
        is_illiquid = participation >= cls.LIQUIDITY_WARN_PARTICIPATION
        if participation >= cls.LIQUIDITY_CRITICAL_PARTICIPATION:
            verdict = "BLOCK"
            detail = f"参与率={participation:.2%}≥20%, 严重流动性不足, 冲击主导价差"
        elif participation >= cls.LIQUIDITY_BLOCK_PARTICIPATION:
            verdict = "BLOCK"
            detail = f"参与率={participation:.2%}≥5%, 流动性不足, 冲击={total_impact_bps:.1f}bps"
        elif participation >= cls.LIQUIDITY_WARN_PARTICIPATION:
            verdict = "WARN"
            detail = f"参与率={participation:.2%}≥1%, 流动性需关注, 冲击={total_impact_bps:.1f}bps"
        else:
            verdict = "PASS"
            detail = f"参与率={participation:.2%}<1%, 流动性充足, 冲击={total_impact_bps:.1f}bps"

        return MarketImpactResult(
            order_size_usd=order_size_usd,
            avg_daily_volume_usd=avg_daily_volume_usd,
            participation_rate=participation,
            volatility=volatility,
            sqrt_impact_pct=sqrt_impact,
            kyle_impact_pct=kyle_impact_pct,
            temporary_impact_pct=temporary_impact,
            total_impact_pct=total_impact_pct,
            total_impact_bps=total_impact_bps,
            expected_fill_price=expected_fill,
            original_price=current_price,
            is_illiquid=is_illiquid,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 2. 成本调整 Sharpe 计算器 (Cost Adjusted Sharpe)
# ============================================================================


class CostAdjustedSharpeCalculator:
    """成本调整 Sharpe 计算器

    在真实交易成本下重新计算 Sharpe Ratio。

    成本构成 (来源: sentinel 2026, coindaynow 2026):
      1. 市场冲击 (Market Impact): 大单推动价格
      2. 手续费 (Fee): maker 0.02%, taker 0.05%
      3. 滑点 (Slippage): 1-5 bps 正常, 5-20 bps 危机
      4. 资金费率 (Funding): 永续合约 0.01%/8h 基础

    业界基准:
      - 高频策略: 成本可吞噬毛利 30-50%
      - 单笔均利与"费+价差+滑点"相当时, 入场价微小恶化即由正转负
      - 永续合约: 93% 期货量在 perp, 资金费率影响显著
    """

    # 默认费率 (来源: 主流交易所 2026)
    DEFAULT_MAKER_FEE_BPS = 2.0   # 0.02%
    DEFAULT_TAKER_FEE_BPS = 5.0   # 0.05%
    DEFAULT_SLIPPAGE_BPS = 2.0    # 2 bps 正常市场
    DEFAULT_FUNDING_BPS = 1.0     # 0.01%/8h × 3次/天 / 3 = 1 bps/笔均摊

    @classmethod
    def compute(
        cls,
        original_sharpe: float,
        n_trades_per_year: int = 252,
        impact_bps: float = 0.0,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        funding_bps: Optional[float] = None,
        is_taker: bool = True,
        is_perpetual: bool = False,
        strategy_annual_return_pct: float = 0.20,
    ) -> CostAdjustedSharpe:
        """计算成本调整后的Sharpe

        Args:
            original_sharpe: 原始Sharpe (无成本)
            n_trades_per_year: 年交易笔数
            impact_bps: 市场冲击 (bps/笔)
            fee_bps: 手续费 (bps/笔), None=使用默认
            slippage_bps: 滑点 (bps/笔), None=使用默认
            funding_bps: 资金费率 (bps/笔), None=使用默认
            is_taker: 是否taker单 (True=0.05%, False=0.02%)
            is_perpetual: 是否永续合约 (计算资金费率)
            strategy_annual_return_pct: 策略预期年化收益率 (用于计算Sharpe退化)

        Returns:
            CostAdjustedSharpe
        """
        # 使用默认值
        if fee_bps is None:
            fee_bps = cls.DEFAULT_TAKER_FEE_BPS if is_taker else cls.DEFAULT_MAKER_FEE_BPS
        if slippage_bps is None:
            slippage_bps = cls.DEFAULT_SLIPPAGE_BPS
        if funding_bps is None:
            funding_bps = cls.DEFAULT_FUNDING_BPS if is_perpetual else 0.0

        # 总成本
        total_cost_bps = impact_bps + fee_bps + slippage_bps + funding_bps

        # 年化成本占比
        # 每笔成本 = total_cost_bps / 10000
        # 年化成本 = 每笔成本 × n_trades
        annual_cost_pct = (total_cost_bps / 10000.0) * n_trades_per_year

        # Sharpe退化
        # 假设: Sharpe ∝ 收益/波动率
        # 成本降低收益 → Sharpe 同比例下降
        # 但成本是固定的, 高频策略退化更严重
        # 经验公式: degradation = annual_cost_pct / expected_return
        expected_return = max(0.05, strategy_annual_return_pct)
        sharpe_degradation_pct = min(0.95, annual_cost_pct / expected_return)

        adjusted_sharpe = original_sharpe * (1.0 - sharpe_degradation_pct)

        # 判定
        if adjusted_sharpe <= 0.0:
            verdict = "BLOCK"
            detail = f"成本吞噬全部收益, 调整后Sharpe={adjusted_sharpe:.3f}≤0"
        elif sharpe_degradation_pct >= 0.50:
            verdict = "BLOCK"
            detail = f"Sharpe退化{sharpe_degradation_pct:.1%}≥50%, 调整后Sharpe={adjusted_sharpe:.3f}"
        elif sharpe_degradation_pct >= 0.30:
            verdict = "WARN"
            detail = f"Sharpe退化{sharpe_degradation_pct:.1%}≥30%, 调整后Sharpe={adjusted_sharpe:.3f}"
        elif adjusted_sharpe < 1.0:
            verdict = "WARN"
            detail = f"调整后Sharpe={adjusted_sharpe:.3f}<1.0, 边界策略"
        else:
            verdict = "PASS"
            detail = f"调整后Sharpe={adjusted_sharpe:.3f}, 退化{sharpe_degradation_pct:.1%}"

        return CostAdjustedSharpe(
            original_sharpe=original_sharpe,
            impact_cost_bps=impact_bps,
            fee_cost_bps=fee_bps,
            slippage_cost_bps=slippage_bps,
            funding_cost_bps=funding_bps,
            total_cost_bps=total_cost_bps,
            n_trades_per_year=n_trades_per_year,
            annual_cost_pct=annual_cost_pct,
            adjusted_sharpe=adjusted_sharpe,
            sharpe_degradation_pct=sharpe_degradation_pct,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 3. 综合市场冲击验证器 (Market Impact Validator)
# ============================================================================


class MarketImpactValidator:
    """综合市场冲击验证器

    集成:
      1. MarketImpactCalculator (单笔冲击)
      2. CostAdjustedSharpeCalculator (成本调整Sharpe)

    判定:
      - 任一 BLOCK → BLOCK
      - 任一 WARN → WARN
      - 全 PASS → PASS
    """

    def validate(
        self,
        order_size_usd: float,
        avg_daily_volume_usd: float,
        original_sharpe: float,
        volatility: float = 0.02,
        current_price: float = 50000.0,
        is_buy: bool = True,
        kyle_lambda: Optional[float] = None,
        ai_penetration: float = 0.0,
        is_illiquid_asset: bool = False,
        n_trades_per_year: int = 252,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        funding_bps: Optional[float] = None,
        is_taker: bool = True,
        is_perpetual: bool = False,
        strategy_annual_return_pct: float = 0.20,
    ) -> MarketImpactValidationResult:
        """综合验证

        Returns:
            MarketImpactValidationResult
        """
        result = MarketImpactValidationResult()
        block_reasons: List[str] = []
        warn_reasons: List[str] = []

        # 1. 计算市场冲击
        impact = MarketImpactCalculator.compute_impact(
            order_size_usd=order_size_usd,
            avg_daily_volume_usd=avg_daily_volume_usd,
            volatility=volatility,
            current_price=current_price,
            is_buy=is_buy,
            kyle_lambda=kyle_lambda,
            ai_penetration=ai_penetration,
            is_illiquid_asset=is_illiquid_asset,
        )
        result.impact_result = impact

        if impact.verdict == "BLOCK":
            block_reasons.append(f"市场冲击: {impact.detail}")
        elif impact.verdict == "WARN":
            warn_reasons.append(f"市场冲击: {impact.detail}")

        # 2. 计算成本调整Sharpe
        cost_adjusted = CostAdjustedSharpeCalculator.compute(
            original_sharpe=original_sharpe,
            n_trades_per_year=n_trades_per_year,
            impact_bps=impact.total_impact_bps,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            funding_bps=funding_bps,
            is_taker=is_taker,
            is_perpetual=is_perpetual,
            strategy_annual_return_pct=strategy_annual_return_pct,
        )
        result.cost_adjusted = cost_adjusted

        if cost_adjusted.verdict == "BLOCK":
            block_reasons.append(f"成本调整: {cost_adjusted.detail}")
        elif cost_adjusted.verdict == "WARN":
            warn_reasons.append(f"成本调整: {cost_adjusted.detail}")

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

    def generate_report(self, result: MarketImpactValidationResult) -> str:
        """生成人类可读报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("市场冲击验证报告 (Market Impact Report)")
        lines.append("=" * 70)
        lines.append(f"综合判定: {result.overall_verdict}")
        lines.append("")

        if result.impact_result:
            ir = result.impact_result
            lines.append("[市场冲击 Market Impact]")
            lines.append(f"  订单大小:        ${ir.order_size_usd:,.2f}")
            lines.append(f"  平均日成交量:    ${ir.avg_daily_volume_usd:,.2f}")
            lines.append(f"  参与率:          {ir.participation_rate:.4%}")
            lines.append(f"  资产波动率:      {ir.volatility:.2%}")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  平方根冲击:      {ir.sqrt_impact_pct:.4%} ({ir.sqrt_impact_pct*10000:.1f} bps)")
            lines.append(f"  Kyle λ 冲击:     {ir.kyle_impact_pct:.4%} ({ir.kyle_impact_pct*10000:.1f} bps)")
            lines.append(f"  临时冲击:        {ir.temporary_impact_pct:.4%} ({ir.temporary_impact_pct*10000:.1f} bps)")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  总冲击:          {ir.total_impact_pct:.4%} ({ir.total_impact_bps:.1f} bps)")
            lines.append(f"  原始价格:        ${ir.original_price:,.2f}")
            lines.append(f"  预期成交价:      ${ir.expected_fill_price:,.2f}")
            lines.append(f"  流动性不足:      {'是' if ir.is_illiquid else '否'}")
            lines.append(f"  判定:            {ir.verdict}")
            lines.append(f"  说明:            {ir.detail}")
            lines.append("")

        if result.cost_adjusted:
            ca = result.cost_adjusted
            lines.append("[成本调整Sharpe Cost Adjusted Sharpe]")
            lines.append(f"  原始 Sharpe:     {ca.original_sharpe:.4f}")
            lines.append(f"  市场冲击:        {ca.impact_cost_bps:.2f} bps/笔")
            lines.append(f"  手续费:          {ca.fee_cost_bps:.2f} bps/笔")
            lines.append(f"  滑点:            {ca.slippage_cost_bps:.2f} bps/笔")
            lines.append(f"  资金费率:        {ca.funding_cost_bps:.2f} bps/笔")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  总成本:          {ca.total_cost_bps:.2f} bps/笔")
            lines.append(f"  年交易笔数:      {ca.n_trades_per_year}")
            lines.append(f"  年化成本占比:    {ca.annual_cost_pct:.2%}")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  调整后 Sharpe:   {ca.adjusted_sharpe:.4f}")
            lines.append(f"  Sharpe 退化:     {ca.sharpe_degradation_pct:.2%}")
            lines.append(f"  判定:            {ca.verdict}")
            lines.append(f"  说明:            {ca.detail}")
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


def validate_market_impact(
    order_size_usd: float,
    avg_daily_volume_usd: float,
    original_sharpe: float,
    volatility: float = 0.02,
    current_price: float = 50000.0,
    is_buy: bool = True,
    kyle_lambda: Optional[float] = None,
    ai_penetration: float = 0.0,
    is_illiquid_asset: bool = False,
    n_trades_per_year: int = 252,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    funding_bps: Optional[float] = None,
    is_taker: bool = True,
    is_perpetual: bool = False,
    strategy_annual_return_pct: float = 0.20,
) -> MarketImpactValidationResult:
    """便利函数: 一键验证市场冲击"""
    validator = MarketImpactValidator()
    return validator.validate(
        order_size_usd=order_size_usd,
        avg_daily_volume_usd=avg_daily_volume_usd,
        original_sharpe=original_sharpe,
        volatility=volatility,
        current_price=current_price,
        is_buy=is_buy,
        kyle_lambda=kyle_lambda,
        ai_penetration=ai_penetration,
        is_illiquid_asset=is_illiquid_asset,
        n_trades_per_year=n_trades_per_year,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        funding_bps=funding_bps,
        is_taker=is_taker,
        is_perpetual=is_perpetual,
        strategy_annual_return_pct=strategy_annual_return_pct,
    )
