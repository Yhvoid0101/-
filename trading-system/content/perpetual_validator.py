# -*- coding: utf-8 -*-
"""
永续合约专项验证器 (Perpetual Validator) — v1.0

R16-2 核心新模块: 数字货币合约量化交易专项验证。

业界背景(2026年最新):
  - 80%+加密交易量在永续合约(decentralised.news 2026)
  - BingX 2026: 主流币波动3-8%, 中型币5-12%, 山寨币15-30%+
  - 资金费率: 0.01%/8h基础, 极端0.5%/8h(557%年化)
  - 强平是合约交易头号风险: 杠杆越高强平距离越近
  - delta-neutral资金费率套利: 15-40% APY(但非"无风险")

核心功能:
  1. 强平价格计算器
     - 多头: liq_price = entry × (1 - mmr/leverage)
     - 空头: liq_price = entry × (1 + mmr/leverage)
     - 来源: Binance Futures公式 + 维持保证金率表

  2. 杠杆合理性验证器
     - 基于波动率自适应杠杆建议
     - BTC(3-8%波动) → 3-5x
     - ETH(5-12%波动) → 2-3x
     - 山寨币(15-30%波动) → 1-2x
     - 来源: BingX 2026多币种合约风险管理

  3. 保证金管理验证器
     - 开仓保证金 = 仓位价值/杠杆
     - 维持保证金 = 仓位价值×mmr
     - 可用保证金 = 总余额 - 已用保证金
     - 维持保证金率<30% → WARN, <20% → BLOCK

  4. 资金费率对冲策略验证器
     - delta-neutral策略验证(现货多+永续空)
     - 费率反转风险评估
     - 极端场景压力测试

  5. 强平距离监测器
     - 当前价距强平价的%距离
     - 距离<5% → WARN, <2% → BLOCK(濒临强平)
     - 来源: myquant.gg 2026永续合约风险管理

铁律:
  - "数字货币合约量化交易领域" — 用户原话
  - 杠杆越高强平越近 → 必须验证强平距离
  - 资金费率可反转 → delta-neutral非"无风险"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.perpetual_validator")


# ============================================================================
# 维持保证金率表(Binance Futures 2026)
# 来源: Binance Futures文档, 不同币种不同档位
# ============================================================================

MAINTENANCE_MARGIN_RATES: Dict[str, float] = {
    # 主流币(低维持保证金率)
    "BTC": 0.004,    # 0.4%
    "ETH": 0.005,    # 0.5%
    "BNB": 0.006,    # 0.6%
    "SOL": 0.008,    # 0.8%
    "XRP": 0.008,    # 0.8%
    "ADA": 0.010,    # 1.0%
    "DOGE": 0.010,   # 1.0%
    "AVAX": 0.012,   # 1.2%
    # 中型币
    "LINK": 0.015,   # 1.5%
    "MATIC": 0.015,  # 1.5%
    "DOT": 0.015,    # 1.5%
    # 小型山寨币(高维持保证金率)
    "DEFAULT": 0.025,  # 2.5%默认
    "HIGH_RISK": 0.05,  # 5%高风险币种
}

# 资产波动率分类(基于2026年市场数据)
# 来源: BingX 2026 主流币3-8%, 中型币5-12%, 山寨币15-30%+
ASSET_VOLATILITY_TIERS: Dict[str, Tuple[float, float]] = {
    # symbol: (daily_vol_low_pct, daily_vol_high_pct)
    "BTC": (3.0, 8.0),
    "ETH": (4.0, 10.0),
    "BNB": (4.0, 10.0),
    "SOL": (5.0, 12.0),
    "XRP": (5.0, 12.0),
    "ADA": (5.0, 12.0),
    "DOGE": (8.0, 18.0),
    "AVAX": (6.0, 14.0),
    "LINK": (6.0, 14.0),
    "MATIC": (6.0, 14.0),
    "DOT": (5.0, 12.0),
    "DEFAULT": (10.0, 25.0),  # 默认山寨币
}


def get_maintenance_margin_rate(symbol: str) -> float:
    """获取维持保证金率"""
    base = symbol.split("/")[0].split("-")[0].upper() if symbol else ""
    return MAINTENANCE_MARGIN_RATES.get(base, MAINTENANCE_MARGIN_RATES["DEFAULT"])


def get_asset_volatility(symbol: str) -> Tuple[float, float]:
    """获取资产日波动率范围(%)"""
    base = symbol.split("/")[0].split("-")[0].upper() if symbol else ""
    return ASSET_VOLATILITY_TIERS.get(base, ASSET_VOLATILITY_TIERS["DEFAULT"])


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class LiquidationPriceResult:
    """强平价格计算结果"""
    entry_price: float
    liquidation_price: float
    leverage: int
    side: str  # "long" / "short"
    maintenance_margin_rate: float
    distance_to_liquidation_pct: float  # 当前价距强平价的%距离
    current_price: float
    verdict: str  # PASS / WARN / BLOCK
    detail: str


@dataclass(slots=True)
class LeverageValidationResult:
    """杠杆合理性验证结果"""
    symbol: str
    requested_leverage: int
    recommended_max_leverage: int
    asset_daily_volatility_pct: float
    is_excessive: bool
    verdict: str
    detail: str


@dataclass(slots=True)
class MarginStatusResult:
    """保证金状态验证结果"""
    position_notional_usd: float
    initial_margin_usd: float
    maintenance_margin_usd: float
    available_margin_usd: float
    total_balance_usd: float
    margin_ratio_pct: float  # 维持保证金/可用保证金
    leverage: int
    verdict: str
    detail: str


@dataclass(slots=True)
class FundingHedgeValidationResult:
    """资金费率对冲策略验证结果"""
    spot_position_usd: float
    perp_position_usd: float
    net_exposure_usd: float
    funding_rate_per_8h: float
    expected_funding_income_per_day_usd: float
    annual_yield_pct: float
    reversal_risk_score: float  # 0-1
    verdict: str
    detail: str
    stress_test_results: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PerpetualValidationResult:
    """永续合约综合验证结果"""
    liquidation_check: Optional[LiquidationPriceResult] = None
    leverage_check: Optional[LeverageValidationResult] = None
    margin_check: Optional[MarginStatusResult] = None
    funding_hedge_check: Optional[FundingHedgeValidationResult] = None
    overall_verdict: str = "PASS"
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)


# ============================================================================
# 1. 强平价格计算器
# ============================================================================


class LiquidationPriceCalculator:
    """强平价格计算器

    Binance Futures公式(简化版,忽略手续费):
      多头: liq_price = entry × (1 - mmr/leverage)
      空头: liq_price = entry × (1 + mmr/leverage)

    实际Binance公式更复杂(包含累计已实现盈亏),但简化版足够风险评估。

    来源: Binance Futures文档 + myquant.gg 2026
    """

    # 强平距离阈值
    LIQ_DISTANCE_WARN_PCT = 5.0   # 距强平价<5% → WARN
    LIQ_DISTANCE_BLOCK_PCT = 2.0  # 距强平价<2% → BLOCK(濒临强平)
    LIQ_DISTANCE_CRITICAL_PCT = 1.0  # 距强平价<1% → CRITICAL

    def calculate(
        self,
        entry_price: float,
        leverage: int,
        side: str,
        symbol: str = "BTC/USDT",
        current_price: Optional[float] = None,
    ) -> LiquidationPriceResult:
        """计算强平价格

        Args:
            entry_price: 开仓价格
            leverage: 杠杆倍数
            side: "long" or "short"
            symbol: 交易对(用于确定维持保证金率)
            current_price: 当前价格(可选,用于计算强平距离)

        Returns:
            LiquidationPriceResult
        """
        if leverage <= 0:
            return LiquidationPriceResult(
                entry_price=entry_price, liquidation_price=0.0,
                leverage=leverage, side=side,
                maintenance_margin_rate=0.0, distance_to_liquidation_pct=0.0,
                current_price=current_price or entry_price,
                verdict="BLOCK", detail="杠杆必须>0",
            )

        mmr = get_maintenance_margin_rate(symbol)

        # 强平价格计算
        if side.lower() == "long":
            liq_price = entry_price * (1.0 - mmr / leverage)
        else:  # short
            liq_price = entry_price * (1.0 + mmr / leverage)

        # 强平距离
        cur_price = current_price or entry_price
        if liq_price > 0:
            distance_pct = abs(cur_price - liq_price) / cur_price * 100.0
        else:
            distance_pct = 0.0

        # 判定
        if distance_pct < self.LIQ_DISTANCE_CRITICAL_PCT:
            verdict = "BLOCK"
            detail = "距强平价{:.2f}%<{:.1f}%, 濒临强平, 立即减仓或追加保证金".format(
                distance_pct, self.LIQ_DISTANCE_CRITICAL_PCT)
        elif distance_pct < self.LIQ_DISTANCE_BLOCK_PCT:
            verdict = "BLOCK"
            detail = "距强平价{:.2f}%<{:.1f}%, 强平风险极高".format(
                distance_pct, self.LIQ_DISTANCE_BLOCK_PCT)
        elif distance_pct < self.LIQ_DISTANCE_WARN_PCT:
            verdict = "WARN"
            detail = "距强平价{:.2f}%<{:.1f}%, 强平风险上升".format(
                distance_pct, self.LIQ_DISTANCE_WARN_PCT)
        else:
            verdict = "PASS"
            detail = "距强平价{:.2f}%, 强平风险可控".format(distance_pct)

        return LiquidationPriceResult(
            entry_price=entry_price,
            liquidation_price=liq_price,
            leverage=leverage,
            side=side,
            maintenance_margin_rate=mmr,
            distance_to_liquidation_pct=distance_pct,
            current_price=cur_price,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 2. 杠杆合理性验证器
# ============================================================================


class LeverageValidator:
    """杠杆合理性验证器

    基于资产波动率自适应推荐最大杠杆:
      - BTC(3-8%波动) → 推荐3-5x
      - ETH(4-10%波动) → 推荐2-3x
      - 山寨币(15-30%波动) → 推荐1-2x

    公式: max_leverage = floor(K / daily_vol_high)
      K=30 (经验常数, 1天最大容忍30%波动不强平)

    来源: BingX 2026多币种合约风险管理
    """

    K_VOLATILITY_CONSTANT = 30.0  # 1天最大容忍30%波动
    MIN_RECOMMENDED_LEVERAGE = 1
    MAX_RECOMMENDED_LEVERAGE = 20  # 永远不建议超过20x

    # 硬性杠杆上限(不同资产)
    LEVERAGE_HARD_LIMITS: Dict[str, int] = {
        "BTC": 20,
        "ETH": 15,
        "BNB": 10,
        "SOL": 10,
        "XRP": 8,
        "ADA": 8,
        "DOGE": 5,
        "AVAX": 5,
        "DEFAULT": 3,
        "HIGH_RISK": 2,
    }

    def validate(self, symbol: str, requested_leverage: int) -> LeverageValidationResult:
        """验证杠杆合理性

        Args:
            symbol: 交易对
            requested_leverage: 请求的杠杆倍数

        Returns:
            LeverageValidationResult
        """
        vol_low, vol_high = get_asset_volatility(symbol)
        base = symbol.split("/")[0].split("-")[0].upper() if symbol else ""

        # 计算推荐最大杠杆
        recommended = int(self.K_VOLATILITY_CONSTANT / vol_high)
        recommended = max(self.MIN_RECOMMENDED_LEVERAGE, recommended)

        # 硬性上限
        hard_limit = self.LEVERAGE_HARD_LIMITS.get(base, self.LEVERAGE_HARD_LIMITS["DEFAULT"])
        recommended = min(recommended, hard_limit)
        recommended = min(recommended, self.MAX_RECOMMENDED_LEVERAGE)

        is_excessive = requested_leverage > recommended

        if is_excessive:
            ratio = requested_leverage / recommended
            if ratio > 3.0:
                verdict = "BLOCK"
                detail = "杠杆{}x严重超过推荐{}x({:.1f}倍), 强平风险极高".format(
                    requested_leverage, recommended, ratio)
            elif ratio > 1.5:
                verdict = "BLOCK"
                detail = "杠杆{}x超过推荐{}x({:.1f}倍), 强平风险高".format(
                    requested_leverage, recommended, ratio)
            else:
                verdict = "WARN"
                detail = "杠杆{}x略超推荐{}x, 谨慎操作".format(
                    requested_leverage, recommended)
        else:
            verdict = "PASS"
            detail = "杠杆{}x<=推荐{}x, 合理".format(requested_leverage, recommended)

        return LeverageValidationResult(
            symbol=symbol,
            requested_leverage=requested_leverage,
            recommended_max_leverage=recommended,
            asset_daily_volatility_pct=vol_high,
            is_excessive=is_excessive,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 3. 保证金管理验证器
# ============================================================================


class MarginStatusValidator:
    """保证金管理验证器

    核心指标:
      - margin_ratio = maintenance_margin / available_margin
      - <30%: PASS, 30-50%: WARN, 50-80%: WARN, >80%: BLOCK

    来源: Binance Futures保证金管理 + production_hardening最佳实践
    """

    MARGIN_RATIO_WARN_PCT = 30.0   # 维持保证金/可用保证金>30% → WARN
    MARGIN_RATIO_BLOCK_PCT = 50.0  # >50% → BLOCK(强平风险)
    MARGIN_RATIO_CRITICAL_PCT = 80.0  # >80% → CRITICAL(濒临强平)

    def validate(
        self,
        position_notional_usd: float,
        leverage: int,
        total_balance_usd: float,
        symbol: str = "BTC/USDT",
    ) -> MarginStatusResult:
        """验证保证金状态

        Args:
            position_notional_usd: 仓位价值(USD)
            leverage: 杠杆
            total_balance_usd: 账户总余额
            symbol: 交易对

        Returns:
            MarginStatusResult
        """
        if leverage <= 0:
            return MarginStatusResult(
                position_notional_usd=position_notional_usd,
                initial_margin_usd=0, maintenance_margin_usd=0,
                available_margin_usd=total_balance_usd,
                total_balance_usd=total_balance_usd,
                margin_ratio_pct=0, leverage=leverage,
                verdict="BLOCK", detail="杠杆必须>0",
            )

        mmr = get_maintenance_margin_rate(symbol)
        initial_margin = position_notional_usd / leverage
        maintenance_margin = position_notional_usd * mmr
        available_margin = total_balance_usd - initial_margin

        if available_margin > 0:
            margin_ratio = maintenance_margin / available_margin * 100.0
        else:
            margin_ratio = 999.0  # 可用保证金<=0

        # 判定
        if margin_ratio > self.MARGIN_RATIO_CRITICAL_PCT:
            verdict = "BLOCK"
            detail = "保证金比率{:.1f}%>{:.0f}%, 濒临强平".format(
                margin_ratio, self.MARGIN_RATIO_CRITICAL_PCT)
        elif margin_ratio > self.MARGIN_RATIO_BLOCK_PCT:
            verdict = "BLOCK"
            detail = "保证金比率{:.1f}%>{:.0f}%, 强平风险高".format(
                margin_ratio, self.MARGIN_RATIO_BLOCK_PCT)
        elif margin_ratio > self.MARGIN_RATIO_WARN_PCT:
            verdict = "WARN"
            detail = "保证金比率{:.1f}%>{:.0f}%, 需关注".format(
                margin_ratio, self.MARGIN_RATIO_WARN_PCT)
        else:
            verdict = "PASS"
            detail = "保证金比率{:.1f}%, 健康".format(margin_ratio)

        return MarginStatusResult(
            position_notional_usd=position_notional_usd,
            initial_margin_usd=initial_margin,
            maintenance_margin_usd=maintenance_margin,
            available_margin_usd=available_margin,
            total_balance_usd=total_balance_usd,
            margin_ratio_pct=margin_ratio,
            leverage=leverage,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 4. 资金费率对冲策略验证器
# ============================================================================


class FundingHedgeValidator:
    """资金费率对冲策略验证器

    delta-neutral策略: 现货多 + 永续空 = 赚取资金费率
    风险:
      1. 资金费率反转(从收费变付费)
      2. 永续合约强平(现货无法对冲)
      3. 基差风险(现货与永续价差)

    来源: chartmini 2026 "Perpetual Futures Funding Rate Arbitrage"
          voiceofchain 2026 "Crypto Funding Rate Arbitrage Bot"
    """

    # 资金费率阈值
    FUNDING_NORMAL_RATE = 0.0001  # 0.01%/8h
    FUNDING_HIGH_RATE = 0.0005     # 0.05%/8h
    FUNDING_EXTREME_RATE = 0.005   # 0.5%/8h(极端)

    # 年化收益率阈值
    ANNUAL_YIELD_WARN_PCT = 50.0   # >50%年化 → WARN(不可持续)
    ANNUAL_YIELD_BLOCK_PCT = 100.0  # >100%年化 → BLOCK(必反转)

    # 反转风险阈值
    REVERSAL_RISK_WARN = 0.3
    REVERSAL_RISK_BLOCK = 0.6

    def validate(
        self,
        spot_position_usd: float,
        perp_position_usd: float,
        funding_rate_per_8h: float,
        perp_side: str = "short",  # delta-neutral: 现货多+永续空
    ) -> FundingHedgeValidationResult:
        """验证资金费率对冲策略

        Args:
            spot_position_usd: 现货仓位价值(USD)
            perp_position_usd: 永续仓位价值(USD)
            funding_rate_per_8h: 8h资金费率(如0.0001=0.01%)
            perp_side: 永续方向("short"=做空收funding, "long"=做多付funding)

        Returns:
            FundingHedgeValidationResult
        """
        # 净敞口(delta-neutral应为0)
        net_exposure = abs(spot_position_usd - perp_position_usd)

        # 计算资金费率收入
        # delta-neutral: 现货多+永续空, 正费率时空头收钱
        if perp_side.lower() == "short":
            # 空头在正费率下赚钱
            funding_income_per_8h = perp_position_usd * funding_rate_per_8h
        else:
            # 多头在正费率下付钱
            funding_income_per_8h = -perp_position_usd * funding_rate_per_8h

        funding_income_per_day = funding_income_per_8h * 3  # 24h/8h=3次
        annual_yield = (funding_income_per_day * 365 / max(spot_position_usd, 1.0)) * 100.0

        # 反转风险评估
        # 当前费率越高, 反转风险越大
        if funding_rate_per_8h > self.FUNDING_EXTREME_RATE:
            reversal_risk = 0.8
        elif funding_rate_per_8h > self.FUNDING_HIGH_RATE:
            reversal_risk = 0.5
        elif funding_rate_per_8h > self.FUNDING_NORMAL_RATE:
            reversal_risk = 0.3
        elif funding_rate_per_8h > 0:
            reversal_risk = 0.1
        else:
            # 负费率, 空头付钱
            reversal_risk = 0.9  # 已经反转

        # 压力测试: 极端场景
        stress_results = {
            "funding_reversal_negative_0.01": -perp_position_usd * 0.0001 * 3 * 365,
            "funding_reversal_negative_0.05": -perp_position_usd * 0.0005 * 3 * 365,
            "funding_extreme_positive_0.5": perp_position_usd * 0.005 * 3 * 365,
            "perp_liquidation_loss": -perp_position_usd * 0.05,  # 假设强平损失5%
        }

        # 判定
        if reversal_risk > self.REVERSAL_RISK_BLOCK:
            verdict = "BLOCK"
            detail = "费率反转风险{:.1f}>{:.1f}, 不可持续".format(
                reversal_risk, self.REVERSAL_RISK_BLOCK)
        elif annual_yield > self.ANNUAL_YIELD_BLOCK_PCT:
            verdict = "BLOCK"
            detail = "年化{:.1f}%>{:.0f}%, 必然反转".format(
                annual_yield, self.ANNUAL_YIELD_BLOCK_PCT)
        elif reversal_risk > self.REVERSAL_RISK_WARN:
            verdict = "WARN"
            detail = "费率反转风险{:.1f}, 需监控".format(reversal_risk)
        elif annual_yield > self.ANNUAL_YIELD_WARN_PCT:
            verdict = "WARN"
            detail = "年化{:.1f}%>{:.0f}%, 不可持续".format(
                annual_yield, self.ANNUAL_YIELD_WARN_PCT)
        else:
            verdict = "PASS"
            detail = "年化{:.1f}%, 反转风险{:.1f}, 健康".format(
                annual_yield, reversal_risk)

        return FundingHedgeValidationResult(
            spot_position_usd=spot_position_usd,
            perp_position_usd=perp_position_usd,
            net_exposure_usd=net_exposure,
            funding_rate_per_8h=funding_rate_per_8h,
            expected_funding_income_per_day_usd=funding_income_per_day,
            annual_yield_pct=annual_yield,
            reversal_risk_score=reversal_risk,
            verdict=verdict,
            detail=detail,
            stress_test_results=stress_results,
        )


# ============================================================================
# 5. 永续合约综合验证器
# ============================================================================


class PerpetualValidator:
    """永续合约综合验证器

    集成4大子验证器:
      1. LiquidationPriceCalculator
      2. LeverageValidator
      3. MarginStatusValidator
      4. FundingHedgeValidator
    """

    def __init__(self):
        self.liq_calculator = LiquidationPriceCalculator()
        self.leverage_validator = LeverageValidator()
        self.margin_validator = MarginStatusValidator()
        self.funding_validator = FundingHedgeValidator()

    def validate(
        self,
        symbol: str = "BTC/USDT",
        entry_price: float = 0.0,
        leverage: int = 1,
        side: str = "long",
        current_price: Optional[float] = None,
        position_notional_usd: float = 0.0,
        total_balance_usd: float = 0.0,
        spot_position_usd: float = 0.0,
        funding_rate_per_8h: float = 0.0,
        perp_side: str = "short",
        check_funding_hedge: bool = False,
    ) -> PerpetualValidationResult:
        """综合验证

        Args:
            symbol: 交易对
            entry_price: 开仓价格
            leverage: 杠杆
            side: 永续方向(long/short)
            current_price: 当前价格
            position_notional_usd: 仓位价值
            total_balance_usd: 账户总余额
            spot_position_usd: 现货仓位(对冲策略)
            funding_rate_per_8h: 8h资金费率
            perp_side: 永续方向(对冲策略)
            check_funding_hedge: 是否检查资金费率对冲

        Returns:
            PerpetualValidationResult
        """
        block_reasons: List[str] = []
        warn_reasons: List[str] = []

        # 1. 强平价格检查
        liq_result = None
        if entry_price > 0 and leverage > 0:
            liq_result = self.liq_calculator.calculate(
                entry_price=entry_price, leverage=leverage,
                side=side, symbol=symbol, current_price=current_price,
            )
            if liq_result.verdict == "BLOCK":
                block_reasons.append("强平检查: " + liq_result.detail)
            elif liq_result.verdict == "WARN":
                warn_reasons.append("强平检查: " + liq_result.detail)

        # 2. 杠杆合理性检查
        lev_result = self.leverage_validator.validate(symbol, leverage)
        if lev_result.verdict == "BLOCK":
            block_reasons.append("杠杆检查: " + lev_result.detail)
        elif lev_result.verdict == "WARN":
            warn_reasons.append("杠杆检查: " + lev_result.detail)

        # 3. 保证金状态检查
        margin_result = None
        if position_notional_usd > 0 and total_balance_usd > 0:
            margin_result = self.margin_validator.validate(
                position_notional_usd=position_notional_usd,
                leverage=leverage,
                total_balance_usd=total_balance_usd,
                symbol=symbol,
            )
            if margin_result.verdict == "BLOCK":
                block_reasons.append("保证金检查: " + margin_result.detail)
            elif margin_result.verdict == "WARN":
                warn_reasons.append("保证金检查: " + margin_result.detail)

        # 4. 资金费率对冲检查(可选)
        funding_result = None
        if check_funding_hedge and spot_position_usd > 0:
            funding_result = self.funding_validator.validate(
                spot_position_usd=spot_position_usd,
                perp_position_usd=position_notional_usd,
                funding_rate_per_8h=funding_rate_per_8h,
                perp_side=perp_side,
            )
            if funding_result.verdict == "BLOCK":
                block_reasons.append("资金费率对冲: " + funding_result.detail)
            elif funding_result.verdict == "WARN":
                warn_reasons.append("资金费率对冲: " + funding_result.detail)

        # 综合判定
        if block_reasons:
            overall = "BLOCK"
        elif warn_reasons:
            overall = "WARN"
        else:
            overall = "PASS"

        return PerpetualValidationResult(
            liquidation_check=liq_result,
            leverage_check=lev_result,
            margin_check=margin_result,
            funding_hedge_check=funding_result,
            overall_verdict=overall,
            block_reasons=block_reasons,
            warn_reasons=warn_reasons,
        )

    def generate_report(self, result: PerpetualValidationResult) -> str:
        """生成人类可读报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("R16-2 永续合约专项验证报告")
        lines.append("=" * 70)
        lines.append("综合判定: " + result.overall_verdict)
        lines.append("")

        if result.liquidation_check:
            lc = result.liquidation_check
            lines.append("[1] 强平价格检查")
            lines.append("    开仓价: {:.4f}".format(lc.entry_price))
            lines.append("    强平价: {:.4f}".format(lc.liquidation_price))
            lines.append("    杠杆: {}x ({})".format(lc.leverage, lc.side))
            lines.append("    维持保证金率: {:.2f}%".format(lc.maintenance_margin_rate * 100))
            lines.append("    距强平价: {:.2f}%".format(lc.distance_to_liquidation_pct))
            lines.append("    判定: {} - {}".format(lc.verdict, lc.detail))
            lines.append("")

        if result.leverage_check:
            lv = result.leverage_check
            lines.append("[2] 杠杆合理性检查")
            lines.append("    交易对: {}".format(lv.symbol))
            lines.append("    请求杠杆: {}x".format(lv.requested_leverage))
            lines.append("    推荐最大: {}x".format(lv.recommended_max_leverage))
            lines.append("    资产日波动率: {:.1f}%".format(lv.asset_daily_volatility_pct))
            lines.append("    判定: {} - {}".format(lv.verdict, lv.detail))
            lines.append("")

        if result.margin_check:
            mc = result.margin_check
            lines.append("[3] 保证金状态检查")
            lines.append("    仓位价值: ${:.2f}".format(mc.position_notional_usd))
            lines.append("    开仓保证金: ${:.2f}".format(mc.initial_margin_usd))
            lines.append("    维持保证金: ${:.2f}".format(mc.maintenance_margin_usd))
            lines.append("    可用保证金: ${:.2f}".format(mc.available_margin_usd))
            lines.append("    保证金比率: {:.1f}%".format(mc.margin_ratio_pct))
            lines.append("    判定: {} - {}".format(mc.verdict, mc.detail))
            lines.append("")

        if result.funding_hedge_check:
            fh = result.funding_hedge_check
            lines.append("[4] 资金费率对冲检查")
            lines.append("    现货仓位: ${:.2f}".format(fh.spot_position_usd))
            lines.append("    永续仓位: ${:.2f}".format(fh.perp_position_usd))
            lines.append("    净敞口: ${:.2f}".format(fh.net_exposure_usd))
            lines.append("    资金费率(8h): {:.4f}%".format(fh.funding_rate_per_8h * 100))
            lines.append("    日预期收入: ${:.2f}".format(fh.expected_funding_income_per_day_usd))
            lines.append("    年化收益: {:.1f}%".format(fh.annual_yield_pct))
            lines.append("    反转风险: {:.1f}".format(fh.reversal_risk_score))
            lines.append("    判定: {} - {}".format(fh.verdict, fh.detail))
            if fh.stress_test_results:
                lines.append("    压力测试:")
                for scenario, value in fh.stress_test_results.items():
                    lines.append("      {}: ${:.2f}".format(scenario, value))
            lines.append("")

        if result.block_reasons:
            lines.append("阻塞原因(BLOCK):")
            for r in result.block_reasons:
                lines.append("  - " + r)
        if result.warn_reasons:
            lines.append("警告原因(WARN):")
            for r in result.warn_reasons:
                lines.append("  - " + r)

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================


def validate_perpetual(
    symbol: str = "BTC/USDT",
    entry_price: float = 0.0,
    leverage: int = 1,
    side: str = "long",
    **kwargs,
) -> PerpetualValidationResult:
    """永续合约综合验证便捷函数"""
    validator = PerpetualValidator()
    return validator.validate(
        symbol=symbol, entry_price=entry_price, leverage=leverage,
        side=side, **kwargs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R16-2 永续合约专项验证器自测 ===\n")

    v = PerpetualValidator()

    # 测试1: BTC多头 10x杠杆, 距强平8%
    r1 = v.validate(symbol="BTC/USDT", entry_price=50000, leverage=10,
                    side="long", current_price=50000)
    print(v.generate_report(r1))
    print()
