# -*- coding: utf-8 -*-
"""
永续合约资金费率真实成本模型 (Funding Rate Cost Model) — R15 核心模块

解决"模拟牛逼实盘亏钱"在永续合约场景的核心根因:
  业界数据 (vc.ru 2026, coindaynow 2026):
    - 永续合约占期货总量 93%
    - 基础资金费率: 0.01%/8h ≈ 10.95% 年化
    - 极端过热期: 0.51%/8h ≈ 70% 年化 (不可持续)
    - Delta-neutral 策略: 10-30% 年化可行但费率反转是头号风险

资金费率机制:
  funding_payment = position_notional × funding_rate
  - 正费率: 多头付空头 (市场过热)
  - 负费率: 空头付多头 (市场过冷)
  - 结算频率: 多数所 8h, Hyperliquid/BingX 1h

费率构成:
  Funding = Premium Index + clamp(Interest - Premium, ±0.05%)
  Interest ≈ 0.01%/8h (基础利率)

策略影响 (pruviq 2026-02):
  - 现货+空永续 delta 中性, 0.02%/8h → $10k 名义日赚 $6, 月 $180
  - maker 入场 0 费 → 月净 $120 (扣滑点)
  - taker 双边 0.04-0.05% 入场费 → 首日 funding 全部吞噬

核心风险:
  1. 费率反转 (最大风险): 正费率转负, delta 中性策略亏损
  2. 极端费率: 0.51%/8h 不可持续, 必然回归
  3. 结算频率: 1h 结算所 vs 8h 结算所, 有效费率差异大
  4. 跨所套利: 不同所费率差异是套利机会也是风险

来源:
  - coindaynow 2026-05 "Funding Rate Arbitrage Strategy Guide"
  - pruviq 2026-02 "Funding Rate Arbitrage: A Practical Guide"
  - vc.ru 2026 "Funding Rate Arbitrage в 2026 году"
  - perpmate 2026 "Funding Rates Explained"
  - CoinGlass API V4 (聚合 30+ 所的 funding/OI/清算/多空比)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.funding_rate_cost")


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class FundingRateScenario:
    """资金费率场景

    基于业界 2026 实证数据
    """
    name: str
    funding_rate_per_8h: float   # 每 8h 费率
    annual_rate: float           # 年化费率
    description: str


# 业界实证场景 (来源: coindaynow 2026, vc.ru 2026, pruviq 2026)
FUNDING_SCENARIOS = {
    "normal": FundingRateScenario(
        name="正常市场",
        funding_rate_per_8h=0.0001,  # 0.01%/8h
        annual_rate=0.1095,          # 10.95% 年化
        description="基础利率, 市场均衡状态"
    ),
    "hot": FundingRateScenario(
        name="过热市场",
        funding_rate_per_8h=0.0003,  # 0.03%/8h
        annual_rate=0.3285,          # 32.85% 年化
        description="多头过度杠杆, 牛市中段"
    ),
    "extreme": FundingRateScenario(
        name="极端过热",
        funding_rate_per_8h=0.0051,  # 0.51%/8h
        annual_rate=5.5735,          # 557% 年化 (不可持续)
        description="市场极度过热, 必然回归 (2021牛市顶部)"
    ),
    "cold": FundingRateScenario(
        name="过冷市场",
        funding_rate_per_8h=-0.0002,  # -0.02%/8h
        annual_rate=-0.2190,          # -21.90% 年化
        description="空头过度杠杆, 熊市底部"
    ),
    "extreme_cold": FundingRateScenario(
        name="极端过冷",
        funding_rate_per_8h=-0.0010,  # -0.10%/8h
        annual_rate=-1.0950,          # -109.5% 年化
        description="市场恐慌, 空头挤兑"
    ),
}


@dataclass(slots=True)
class FundingCostResult:
    """资金费率成本计算结果"""

    position_notional_usd: float       # 持仓名义价值 (USD)
    position_side: str                 # "long" / "short"
    funding_rate_per_8h: float         # 每 8h 费率
    settlement_freq_hours: int         # 结算频率 (小时)
    n_settlements_per_year: int        # 年结算次数
    # 成本计算
    cost_per_settlement_usd: float     # 每次结算成本 (USD)
    cost_per_day_usd: float            # 每日成本 (USD)
    annual_cost_usd: float             # 年化成本 (USD)
    annual_cost_pct: float             # 年化成本占比
    # 风险评估
    is_reversal_risk: bool             # 费率反转风险
    is_extreme: bool                   # 极端费率
    breakeven_sharpe_drag: float       # 对Sharpe的拖累 (bps)
    verdict: str                       # PASS / WARN / BLOCK
    detail: str


@dataclass(slots=True)
class FundingRateValidationResult:
    """资金费率综合验证结果"""

    scenario_name: str
    funding_result: Optional[FundingCostResult] = None
    reversal_risk_score: float = 0.0    # 费率反转风险评分 [0, 1]
    extreme_scenario_check: Dict[str, Any] = field(default_factory=dict)
    overall_verdict: str = "PASS"
    block_reasons: List[str] = field(default_factory=list)
    warn_reasons: List[str] = field(default_factory=list)


# ============================================================================
# 1. 资金费率成本计算器 (Funding Rate Cost Calculator)
# ============================================================================


class FundingRateCostCalculator:
    """资金费率成本计算器

    公式:
      funding_payment = position_notional × funding_rate
      annual_cost = funding_payment × n_settlements_per_year

    业界基准 (2026):
      - 8h 结算: 3 次/天 × 365 = 1095 次/年
      - 1h 结算: 24 次/天 × 365 = 8760 次/年
      - 基础费率 0.01%/8h → 10.95% 年化
      - 极端费率 0.51%/8h → 557% 年化 (不可持续)
    """

    # 阈值 (来源: 业界实证 2026)
    # sharpe_drag_bps / 10000 = sharpe_degradation_pct
    # 5000 bps = 50% Sharpe退化 (多头永续长期持有不划算的边界)
    # 2000 bps = 20% Sharpe退化 (需关注的边界)
    EXTREME_FUNDING_THRESHOLD = 0.001  # 0.1%/8h → 极端
    REVERSAL_RISK_THRESHOLD = 0.0005   # 0.05%/8h → 反转风险
    SHARPE_DRAG_BLOCK_BPS = 5000       # 5000 bps (50%退化) → BLOCK
    SHARPE_DRAG_WARN_BPS = 2000        # 2000 bps (20%退化) → WARN

    @classmethod
    def compute(
        cls,
        position_notional_usd: float,
        position_side: str,
        funding_rate_per_8h: float,
        settlement_freq_hours: int = 8,
        strategy_sharpe: float = 1.0,
        strategy_annual_return_pct: float = 0.20,
    ) -> FundingCostResult:
        """计算资金费率成本

        Args:
            position_notional_usd: 持仓名义价值 (USD)
            position_side: "long" 或 "short"
            funding_rate_per_8h: 每 8h 费率 (如 0.0001 = 0.01%)
            settlement_freq_hours: 结算频率 (小时, 8 或 1)
            strategy_sharpe: 策略 Sharpe (用于计算 Sharpe 拖累)
            strategy_annual_return_pct: 策略预期年化收益

        Returns:
            FundingCostResult
        """
        if position_notional_usd <= 0:
            return FundingCostResult(
                position_notional_usd=0,
                position_side=position_side,
                funding_rate_per_8h=funding_rate_per_8h,
                settlement_freq_hours=settlement_freq_hours,
                n_settlements_per_year=0,
                cost_per_settlement_usd=0,
                cost_per_day_usd=0,
                annual_cost_usd=0,
                annual_cost_pct=0,
                is_reversal_risk=False,
                is_extreme=False,
                breakeven_sharpe_drag=0,
                verdict="PASS",
                detail="无持仓"
            )

        # 计算等价 8h 费率 (按结算频率调整)
        # 1h 结算所: 每次费率 = 8h费率 / 8
        # 但年化次数 = 24*365 = 8760 (vs 8h 的 1095)
        # 所以年化成本相同 (无套利空间, 仅现金流差异)
        rate_per_settlement = funding_rate_per_8h * (settlement_freq_hours / 8.0)
        n_settlements_per_year = int(24 * 365 / settlement_freq_hours)

        # 多头付正费率, 空头收正费率
        if position_side.lower() == "long":
            # 多头: 正费率=付费(成本), 负费率=收费(收益)
            cost_per_settlement = position_notional_usd * rate_per_settlement
        else:  # short
            # 空头: 正费率=收费(收益), 负费率=付费(成本)
            cost_per_settlement = -position_notional_usd * rate_per_settlement

        # 成本可以是负的 (空头在正费率市场赚钱 = 收益)
        cost_per_day = cost_per_settlement * (24 / settlement_freq_hours)
        annual_cost = cost_per_settlement * n_settlements_per_year
        annual_cost_pct = annual_cost / position_notional_usd

        # Sharpe 拖累 (bps)
        # 关键修复: 只有正成本(付费)才拖累Sharpe, 负成本(收费)反而提升Sharpe
        # 来源: pruviq 2026 空头正费率市场可盈利
        if annual_cost_pct > 0:
            # 真实成本: 拖累Sharpe
            expected_return = max(0.05, strategy_annual_return_pct)
            sharpe_drag_bps = annual_cost_pct / expected_return * 10000
        else:
            # 负成本(收费收益): 不拖累, 反而提升 (drag=0)
            sharpe_drag_bps = 0.0

        # 风险评估
        is_extreme = abs(funding_rate_per_8h) >= cls.EXTREME_FUNDING_THRESHOLD
        is_reversal = abs(funding_rate_per_8h) >= cls.REVERSAL_RISK_THRESHOLD

        # 判定
        # 关键: 极端费率对收费方是收益(但不可持续), 对付费方是成本
        if is_extreme:
            if annual_cost_pct > 0:
                # 极端费率 + 付费方 → BLOCK (成本极高)
                verdict = "BLOCK"
                detail = f"极端费率 {funding_rate_per_8h*100:.4f}%/8h, 年化成本{annual_cost_pct*100:.2f}%, 付费方灾难"
            else:
                # 极端费率 + 收费方 → WARN (收益不可持续, 必然回归)
                verdict = "WARN"
                detail = f"极端费率 {funding_rate_per_8h*100:.4f}%/8h, 年化收益{annual_cost_pct*100:.2f}%, 不可持续必然回归"
        elif sharpe_drag_bps >= cls.SHARPE_DRAG_BLOCK_BPS:
            verdict = "BLOCK"
            detail = f"Sharpe拖累 {sharpe_drag_bps:.0f}bps≥{cls.SHARPE_DRAG_BLOCK_BPS}bps, 成本过高"
        elif is_reversal or sharpe_drag_bps >= cls.SHARPE_DRAG_WARN_BPS:
            verdict = "WARN"
            detail = f"费率反转风险, Sharpe拖累 {sharpe_drag_bps:.0f}bps"
        else:
            verdict = "PASS"
            detail = f"年化成本{annual_cost_pct*100:.2f}%, Sharpe拖累{sharpe_drag_bps:.0f}bps"

        return FundingCostResult(
            position_notional_usd=position_notional_usd,
            position_side=position_side,
            funding_rate_per_8h=funding_rate_per_8h,
            settlement_freq_hours=settlement_freq_hours,
            n_settlements_per_year=n_settlements_per_year,
            cost_per_settlement_usd=cost_per_settlement,
            cost_per_day_usd=cost_per_day,
            annual_cost_usd=annual_cost,
            annual_cost_pct=annual_cost_pct,
            is_reversal_risk=is_reversal,
            is_extreme=is_extreme,
            breakeven_sharpe_drag=sharpe_drag_bps,
            verdict=verdict,
            detail=detail,
        )


# ============================================================================
# 2. 费率反转风险检测器 (Reversal Risk Detector)
# ============================================================================


class ReversalRiskDetector:
    """费率反转风险检测器

    来源: coindaynow 2026, pruviq 2026
    费率反转是 delta-neutral 策略的头号风险:
      - 正费率时: 空永续收资金费 → 费率转负时开始付费
      - 极端正费率不可持续, 必然回归均值
      - 反转可导致策略从盈利转为亏损

    检测方法:
      1. 当前费率 vs 历史均值 (z-score)
      2. 费率趋势 (上升/下降/震荡)
      3. 极端场景压力测试
    """

    # 历史均值 (业界 2026 实证)
    HISTORICAL_MEAN_FUNDING = 0.0001  # 0.01%/8h
    HISTORICAL_STD_FUNDING = 0.0002   # 0.02%/8h

    @classmethod
    def compute_risk_score(
        cls,
        current_funding_rate: float,
        funding_history: Optional[List[float]] = None,
    ) -> float:
        """计算费率反转风险评分 [0, 1]

        Args:
            current_funding_rate: 当前费率 (每 8h)
            funding_history: 历史费率序列 (可选)

        Returns:
            风险评分 [0, 1], 0=无风险, 1=极高风险
        """
        if funding_history and len(funding_history) > 10:
            mean = float(np.mean(funding_history))
            std = max(float(np.std(funding_history)), 1e-6)
            z_score = (current_funding_rate - mean) / std
            # z-score > 2 → 高反转风险
            risk = max(0.0, min(1.0, (z_score - 1.0) / 3.0))
        else:
            # 用历史基准
            z = (current_funding_rate - cls.HISTORICAL_MEAN_FUNDING) / cls.HISTORICAL_STD_FUNDING
            risk = max(0.0, min(1.0, (z - 1.0) / 3.0))

        return risk

    @classmethod
    def stress_test_extreme_scenarios(
        cls,
        position_notional_usd: float,
        position_side: str,
        strategy_sharpe: float = 1.0,
        strategy_annual_return_pct: float = 0.20,
    ) -> Dict[str, FundingCostResult]:
        """极端场景压力测试

        在 5 大业界场景下测试策略成本

        Returns:
            Dict[场景名, FundingCostResult]
        """
        results = {}
        for name, scenario in FUNDING_SCENARIOS.items():
            result = FundingRateCostCalculator.compute(
                position_notional_usd=position_notional_usd,
                position_side=position_side,
                funding_rate_per_8h=scenario.funding_rate_per_8h,
                settlement_freq_hours=8,
                strategy_sharpe=strategy_sharpe,
                strategy_annual_return_pct=strategy_annual_return_pct,
            )
            results[name] = result
        return results


# ============================================================================
# 3. 综合资金费率验证器 (Funding Rate Validator)
# ============================================================================


class FundingRateValidator:
    """综合资金费率验证器

    集成:
      1. FundingRateCostCalculator (成本计算)
      2. ReversalRiskDetector (反转风险)
      3. 极端场景压力测试

    综合判定:
      - 任一 BLOCK → BLOCK
      - 任一 WARN → WARN
      - 全 PASS → PASS
    """

    def validate(
        self,
        position_notional_usd: float,
        position_side: str,
        current_funding_rate: float = 0.0001,
        settlement_freq_hours: int = 8,
        funding_history: Optional[List[float]] = None,
        strategy_sharpe: float = 1.0,
        strategy_annual_return_pct: float = 0.20,
        run_stress_test: bool = True,
    ) -> FundingRateValidationResult:
        """综合验证

        Returns:
            FundingRateValidationResult
        """
        result = FundingRateValidationResult(scenario_name="综合验证")
        block_reasons: List[str] = []
        warn_reasons: List[str] = []

        # 1. 计算当前费率成本
        funding = FundingRateCostCalculator.compute(
            position_notional_usd=position_notional_usd,
            position_side=position_side,
            funding_rate_per_8h=current_funding_rate,
            settlement_freq_hours=settlement_freq_hours,
            strategy_sharpe=strategy_sharpe,
            strategy_annual_return_pct=strategy_annual_return_pct,
        )
        result.funding_result = funding

        if funding.verdict == "BLOCK":
            block_reasons.append(f"当前费率: {funding.detail}")
        elif funding.verdict == "WARN":
            warn_reasons.append(f"当前费率: {funding.detail}")

        # 2. 费率反转风险
        reversal_score = ReversalRiskDetector.compute_risk_score(
            current_funding_rate=current_funding_rate,
            funding_history=funding_history,
        )
        result.reversal_risk_score = reversal_score

        if reversal_score >= 0.7:
            block_reasons.append(f"费率反转风险评分 {reversal_score:.2f}≥0.7, 极高反转风险")
        elif reversal_score >= 0.4:
            warn_reasons.append(f"费率反转风险评分 {reversal_score:.2f}≥0.4, 需关注")

        # 3. 极端场景压力测试
        if run_stress_test:
            stress = ReversalRiskDetector.stress_test_extreme_scenarios(
                position_notional_usd=position_notional_usd,
                position_side=position_side,
                strategy_sharpe=strategy_sharpe,
                strategy_annual_return_pct=strategy_annual_return_pct,
            )
            result.extreme_scenario_check = {
                name: {
                    "verdict": r.verdict,
                    "annual_cost_pct": r.annual_cost_pct,
                    "sharpe_drag_bps": r.breakeven_sharpe_drag,
                }
                for name, r in stress.items()
            }

            # 检查极端场景
            for name, r in stress.items():
                if r.verdict == "BLOCK" and name in ("extreme", "extreme_cold"):
                    warn_reasons.append(f"极端场景 '{name}': {r.detail}")

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

    def generate_report(self, result: FundingRateValidationResult) -> str:
        """生成人类可读报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("资金费率验证报告 (Funding Rate Report)")
        lines.append("=" * 70)
        lines.append(f"综合判定: {result.overall_verdict}")
        lines.append("")

        if result.funding_result:
            fr = result.funding_result
            lines.append("[当前费率成本 Current Funding Cost]")
            lines.append(f"  持仓名义:        ${fr.position_notional_usd:,.2f}")
            lines.append(f"  持仓方向:        {fr.position_side}")
            lines.append(f"  费率 (8h):       {fr.funding_rate_per_8h*100:.4f}%")
            lines.append(f"  结算频率:        每 {fr.settlement_freq_hours} 小时")
            lines.append(f"  年结算次数:      {fr.n_settlements_per_year}")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  每次结算成本:    ${fr.cost_per_settlement_usd:,.2f}")
            lines.append(f"  每日成本:        ${fr.cost_per_day_usd:,.2f}")
            lines.append(f"  年化成本:        ${fr.annual_cost_usd:,.2f}")
            lines.append(f"  年化成本占比:    {fr.annual_cost_pct*100:.2f}%")
            lines.append(f"  ────────────────────────────────")
            lines.append(f"  极端费率:        {'是' if fr.is_extreme else '否'}")
            lines.append(f"  反转风险:        {'是' if fr.is_reversal_risk else '否'}")
            lines.append(f"  Sharpe 拖累:     {fr.breakeven_sharpe_drag:.0f} bps")
            lines.append(f"  判定:            {fr.verdict}")
            lines.append(f"  说明:            {fr.detail}")
            lines.append("")

        lines.append("[费率反转风险 Reversal Risk]")
        lines.append(f"  风险评分:        {result.reversal_risk_score:.2f} [0, 1]")
        lines.append("")

        if result.extreme_scenario_check:
            lines.append("[极端场景压力测试 Extreme Scenario Stress Test]")
            for name, info in result.extreme_scenario_check.items():
                scenario = FUNDING_SCENARIOS.get(name)
                if scenario:
                    lines.append(f"  {scenario.name:12s} ({scenario.funding_rate_per_8h*100:+.4f}%/8h):")
                    lines.append(f"    判定={info['verdict']}, 年化成本={info['annual_cost_pct']*100:+.2f}%, Sharpe拖累={info['sharpe_drag_bps']:.0f}bps")
                    lines.append(f"    {scenario.description}")
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


def validate_funding_rate(
    position_notional_usd: float,
    position_side: str,
    current_funding_rate: float = 0.0001,
    settlement_freq_hours: int = 8,
    funding_history: Optional[List[float]] = None,
    strategy_sharpe: float = 1.0,
    strategy_annual_return_pct: float = 0.20,
    run_stress_test: bool = True,
) -> FundingRateValidationResult:
    """便利函数: 一键验证资金费率"""
    validator = FundingRateValidator()
    return validator.validate(
        position_notional_usd=position_notional_usd,
        position_side=position_side,
        current_funding_rate=current_funding_rate,
        settlement_freq_hours=settlement_freq_hours,
        funding_history=funding_history,
        strategy_sharpe=strategy_sharpe,
        strategy_annual_return_pct=strategy_annual_return_pct,
        run_stress_test=run_stress_test,
    )
