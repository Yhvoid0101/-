# -*- coding: utf-8 -*-
"""
双层辩论架构 — P1-3 投资辩论+风控三方辩论

借鉴 TradingAgents (2024) 的双层辩论架构:
  第一层: 投资辩论 — 多空双方Agent辩论，决定交易方向
  第二层: 风控三方辩论 — 风控官/合规官/流动性官辩论，决定是否批准

设计原则:
  1. 辩论产生决策，而非单一Agent决定
  2. 每个辩论参与者有独立的评估维度和否决权
  3. 辩论记录可追溯，用于后续进化优化
  4. 风控层有硬性否决权（不受投资层影响）

参考:
  - TradingAgents: github.com/TauricResearch/TradingAgents
  - 辩论式AI决策: LLM Debate for Financial Decision Making
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("hermes.debate")


# ============================================================================
# 辩论数据结构
# ============================================================================


class DebateRole(Enum):
    """辩论角色"""
    # 第一层：投资辩论
    BULL = "bull"            # 多头分析师
    BEAR = "bear"            # 空头分析师
    TRADER = "trader"        # 交易员（综合多空观点）

    # 第二层：风控三方辩论
    RISK_OFFICER = "risk"       # 风控官
    COMPLIANCE = "compliance"   # 合规官
    LIQUIDITY = "liquidity"     # 流动性官


class DebateVerdict(Enum):
    """辩论裁决"""
    APPROVE = "approve"          # 批准
    APPROVE_WITH_CONDITIONS = "approve_conditions"  # 有条件批准
    REDUCE_SIZE = "reduce_size"  # 减仓批准
    REJECT = "reject"            # 否决
    STRONG_REJECT = "strong_reject"  # 强烈否决


@dataclass(slots=True)
class DebateArgument:
    """辩论论点"""
    role: DebateRole
    argument: str
    confidence: float  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class InvestmentDebateResult:
    """投资辩论结果"""
    symbol: str
    bull_score: float    # 多头得分 0-100
    bear_score: float    # 空头得分 0-100
    direction: str       # "long"/"short"/"neutral"
    confidence: float   # 0-1
    arguments: List[DebateArgument] = field(default_factory=list)
    trader_decision: str = ""  # 交易员综合决策


@dataclass(slots=True)
class RiskDebateResult:
    """风控辩论结果"""
    verdict: DebateVerdict
    risk_score: float        # 风控官评分 0-100
    compliance_score: float   # 合规官评分 0-100
    liquidity_score: float   # 流动性官评分 0-100
    conditions: List[str] = field(default_factory=list)
    max_position_size: float = 0.0  # 最大允许仓位
    arguments: List[DebateArgument] = field(default_factory=list)


@dataclass(slots=True)
class TwoLayerDebateResult:
    """双层辩论最终结果"""
    investment: InvestmentDebateResult
    risk: RiskDebateResult
    final_decision: str    # "execute"/"reject"/"modify"
    final_size: float = 0.0
    final_leverage: int = 1
    debate_log: str = ""   # 完整辩论记录


# ============================================================================
# 第一层：投资辩论
# ============================================================================


class InvestmentDebate:
    """投资辩论 — 多空双方辩论

    流程:
      1. 多头分析师提出做多论点
      2. 空头分析师提出做空论点
      3. 双方反驳对方论点
      4. 交易员综合双方观点做出决策
    """

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds

    def debate(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        signals: Dict[str, Any],
    ) -> InvestmentDebateResult:
        """执行投资辩论

        Args:
            symbol: 交易对
            market_data: 市场数据
            signals: 技术指标信号

        Returns:
            投资辩论结果
        """
        arguments: List[DebateArgument] = []

        # 多头论点
        bull_args = self._bull_analysis(symbol, market_data, signals)
        arguments.extend(bull_args)

        # 空头论点
        bear_args = self._bear_analysis(symbol, market_data, signals)
        arguments.extend(bear_args)

        # 多轮辩论
        for round_num in range(self.max_rounds):
            # 多头反驳空头
            bull_rebuttal = self._bull_rebuttal(bear_args, round_num)
            arguments.extend(bull_rebuttal)

            # 空头反驳多头
            bear_rebuttal = self._bear_rebuttal(bull_args, round_num)
            arguments.extend(bear_rebuttal)

        # 计算多空得分
        bull_score = self._score_arguments(
            [a for a in arguments if a.role == DebateRole.BULL]
        )
        bear_score = self._score_arguments(
            [a for a in arguments if a.role == DebateRole.BEAR]
        )

        # 交易员决策
        direction, confidence, trader_decision = self._trader_decision(
            bull_score, bear_score, market_data
        )

        return InvestmentDebateResult(
            symbol=symbol,
            bull_score=bull_score,
            bear_score=bear_score,
            direction=direction,
            confidence=confidence,
            arguments=arguments,
            trader_decision=trader_decision,
        )

    def _bull_analysis(
        self, symbol: str, market_data: Dict[str, Any], signals: Dict[str, Any]
    ) -> List[DebateArgument]:
        """多头分析"""
        args = []

        # 趋势信号
        if signals.get("trend") == "up":
            args.append(DebateArgument(
                role=DebateRole.BULL,
                argument=f"{symbol} 上升趋势明确，MA20>MA50，趋势跟踪策略建议做多",
                confidence=0.7,
                evidence={"trend": "up", "ma20": signals.get("ma20", 0)},
            ))

        # RSI超卖反弹
        rsi = signals.get("rsi", 50)
        if rsi < 30:
            args.append(DebateArgument(
                role=DebateRole.BULL,
                argument=f"RSI={rsi:.1f} 超卖区域，反弹概率高",
                confidence=0.6,
                evidence={"rsi": rsi},
            ))

        # 成交量放大
        vol_ratio = signals.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            args.append(DebateArgument(
                role=DebateRole.BULL,
                argument=f"成交量放大 {vol_ratio:.1f}x，买盘积极",
                confidence=0.5,
                evidence={"volume_ratio": vol_ratio},
            ))

        if not args:
            args.append(DebateArgument(
                role=DebateRole.BULL,
                argument=f"{symbol} 无明显做多信号",
                confidence=0.2,
            ))

        return args

    def _bear_analysis(
        self, symbol: str, market_data: Dict[str, Any], signals: Dict[str, Any]
    ) -> List[DebateArgument]:
        """空头分析"""
        args = []

        # 趋势下行
        if signals.get("trend") == "down":
            args.append(DebateArgument(
                role=DebateRole.BEAR,
                argument=f"{symbol} 下降趋势，MA20<MA50，做空信号",
                confidence=0.7,
                evidence={"trend": "down"},
            ))

        # RSI超买
        rsi = signals.get("rsi", 50)
        if rsi > 70:
            args.append(DebateArgument(
                role=DebateRole.BEAR,
                argument=f"RSI={rsi:.1f} 超买区域，回调风险高",
                confidence=0.6,
                evidence={"rsi": rsi},
            ))

        # 波动率过高
        volatility = signals.get("volatility", 0)
        if volatility > 0.05:
            args.append(DebateArgument(
                role=DebateRole.BEAR,
                argument=f"波动率 {volatility:.1%} 过高，下行风险大",
                confidence=0.5,
                evidence={"volatility": volatility},
            ))

        if not args:
            args.append(DebateArgument(
                role=DebateRole.BEAR,
                argument=f"{symbol} 无明显做空信号",
                confidence=0.2,
            ))

        return args

    def _bull_rebuttal(self, bear_args: List[DebateArgument], round_num: int) -> List[DebateArgument]:
        """多头反驳"""
        if not bear_args:
            return []
        return [DebateArgument(
            role=DebateRole.BULL,
            argument=f"反驳空头: {bear_args[0].argument[:50]}... 市场已price-in",
            confidence=max(0.3, bear_args[0].confidence - 0.1),
        )]

    def _bear_rebuttal(self, bull_args: List[DebateArgument], round_num: int) -> List[DebateArgument]:
        """空头反驳"""
        if not bull_args:
            return []
        return [DebateArgument(
            role=DebateRole.BEAR,
            argument=f"反驳多头: {bull_args[0].argument[:50]}... 信号可能滞后",
            confidence=max(0.3, bull_args[0].confidence - 0.1),
        )]

    def _score_arguments(self, args: List[DebateArgument]) -> float:
        """计算论点得分"""
        if not args:
            return 20.0
        total = sum(a.confidence for a in args)
        return min(100, total * 25)

    def _trader_decision(
        self, bull_score: float, bear_score: float, market_data: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """交易员综合决策"""
        diff = bull_score - bear_score

        if diff > 15:
            direction = "long"
            confidence = min(0.9, 0.5 + diff / 100)
            decision = f"多头优势 {diff:.1f}分，建议做多"
        elif diff < -15:
            direction = "short"
            confidence = min(0.9, 0.5 + abs(diff) / 100)
            decision = f"空头优势 {abs(diff):.1f}分，建议做空"
        else:
            direction = "neutral"
            confidence = 0.3
            decision = f"多空均衡(差{diff:.1f}分)，观望"

        return direction, confidence, decision


# ============================================================================
# 第二层：风控三方辩论
# ============================================================================


class RiskDebate:
    """风控三方辩论 — 风控官/合规官/流动性官

    流程:
      1. 风控官评估风险敞口
      2. 合规官检查合规性
      3. 流动性官评估流动性
      4. 三方综合裁决
    """

    def debate(
        self,
        investment_result: InvestmentDebateResult,
        account_state: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> RiskDebateResult:
        """执行风控辩论

        Args:
            investment_result: 投资辩论结果
            account_state: 账户状态（余额、持仓等）
            market_data: 市场数据

        Returns:
            风控辩论结果
        """
        arguments: List[DebateArgument] = []

        # 风控官评估
        risk_arg, risk_score = self._risk_officer_assess(
            investment_result, account_state, market_data
        )
        arguments.append(risk_arg)

        # 合规官评估
        compliance_arg, compliance_score = self._compliance_officer_assess(
            investment_result, account_state
        )
        arguments.append(compliance_arg)

        # 流动性官评估
        liquidity_arg, liquidity_score = self._liquidity_officer_assess(
            investment_result, market_data
        )
        arguments.append(liquidity_arg)

        # 综合裁决
        verdict, conditions, max_size = self._final_verdict(
            risk_score, compliance_score, liquidity_score,
            investment_result, account_state,
        )

        return RiskDebateResult(
            verdict=verdict,
            risk_score=risk_score,
            compliance_score=compliance_score,
            liquidity_score=liquidity_score,
            conditions=conditions,
            max_position_size=max_size,
            arguments=arguments,
        )

    def _risk_officer_assess(
        self, investment: InvestmentDebateResult,
        account: Dict[str, Any], market: Dict[str, Any],
    ) -> Tuple[DebateArgument, float]:
        """风控官评估"""
        balance = account.get("balance", 10000)
        existing_exposure = account.get("total_exposure", 0)
        max_drawdown = account.get("max_drawdown", 0)

        # 风险评分
        score = 100

        # 已有敞口过高
        if existing_exposure > balance * 0.5:
            score -= 30
        elif existing_exposure > balance * 0.3:
            score -= 15

        # 历史回撤过大
        if max_drawdown > 0.15:
            score -= 25
        elif max_drawdown > 0.10:
            score -= 10

        # 市场波动率
        vol = market.get("volatility", 0.02)
        if vol > 0.05:
            score -= 20
        elif vol > 0.03:
            score -= 10

        score = max(0, score)

        arg = DebateArgument(
            role=DebateRole.RISK_OFFICER,
            argument=f"风险评分 {score:.0f}/100: 敞口={existing_exposure:.0f} 回撤={max_drawdown:.1%} 波动={vol:.1%}",
            confidence=score / 100,
            evidence={"exposure": existing_exposure, "max_dd": max_drawdown, "vol": vol},
        )
        return arg, score

    def _compliance_officer_assess(
        self, investment: InvestmentDebateResult, account: Dict[str, Any],
    ) -> Tuple[DebateArgument, float]:
        """合规官评估"""
        score = 100

        # 检查杠杆是否超限
        leverage = account.get("leverage", 1)
        if leverage > 10:
            score -= 40
        elif leverage > 5:
            score -= 20

        # 检查单笔仓位是否过大
        position_pct = account.get("position_pct", 0)
        if position_pct > 0.3:
            score -= 30
        elif position_pct > 0.2:
            score -= 15

        # 检查日内交易频率
        daily_trades = account.get("daily_trades", 0)
        if daily_trades > 50:
            score -= 20

        score = max(0, score)

        arg = DebateArgument(
            role=DebateRole.COMPLIANCE,
            argument=f"合规评分 {score:.0f}/100: 杠杆={leverage}x 仓位={position_pct:.1%} 日交易={daily_trades}",
            confidence=score / 100,
        )
        return arg, score

    def _liquidity_officer_assess(
        self, investment: InvestmentDebateResult, market: Dict[str, Any],
    ) -> Tuple[DebateArgument, float]:
        """流动性官评估"""
        score = 100

        volume = market.get("volume", 0)
        spread = market.get("spread", 0.001)
        depth = market.get("depth", 0)

        # 价差过大
        if spread > 0.005:
            score -= 30
        elif spread > 0.002:
            score -= 15

        # 深度不足
        if depth < 10000:
            score -= 25
        elif depth < 50000:
            score -= 10

        # 成交量过低
        if volume < 100000:
            score -= 20

        score = max(0, score)

        arg = DebateArgument(
            role=DebateRole.LIQUIDITY,
            argument=f"流动性评分 {score:.0f}/100: 价差={spread:.2%} 深度={depth:.0f} 成交量={volume:.0f}",
            confidence=score / 100,
        )
        return arg, score

    def _final_verdict(
        self, risk_score: float, compliance_score: float, liquidity_score: float,
        investment: InvestmentDebateResult, account: Dict[str, Any],
    ) -> Tuple[DebateVerdict, List[str], float]:
        """最终裁决"""
        conditions: List[str] = []
        balance = account.get("balance", 10000)

        # 最低分决定
        min_score = min(risk_score, compliance_score, liquidity_score)

        # 计算最大仓位
        base_size = balance * 0.1  # 默认10%
        if min_score > 80:
            max_size = base_size * 1.5
            verdict = DebateVerdict.APPROVE
        elif min_score > 60:
            max_size = base_size
            verdict = DebateVerdict.APPROVE_WITH_CONDITIONS
            conditions.append("标准仓位")
        elif min_score > 40:
            max_size = base_size * 0.5
            verdict = DebateVerdict.REDUCE_SIZE
            conditions.append("减半仓位")
            conditions.append("设置止损")
        elif min_score > 20:
            max_size = base_size * 0.25
            verdict = DebateVerdict.REDUCE_SIZE
            conditions.append("最小仓位")
            conditions.append("严格止损")
            conditions.append("仅限试探")
        else:
            max_size = 0
            verdict = DebateVerdict.REJECT
            conditions.append("风险过高，否决交易")

        # 任一方极低分则否决
        if risk_score < 20 or compliance_score < 20:
            verdict = DebateVerdict.STRONG_REJECT
            max_size = 0
            conditions = ["硬性否决：风控或合规不达标"]

        return verdict, conditions, max_size


# ============================================================================
# 双层辩论协调器
# ============================================================================


class TwoLayerDebateCoordinator:
    """双层辩论协调器

    协调投资辩论和风控辩论，产生最终交易决策。
    """

    def __init__(self, max_investment_rounds: int = 2):
        self.investment_debate = InvestmentDebate(max_rounds=max_investment_rounds)
        self.risk_debate = RiskDebate()

    def execute_debate(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        signals: Dict[str, Any],
        account_state: Dict[str, Any],
    ) -> TwoLayerDebateResult:
        """执行完整的双层辩论

        Args:
            symbol: 交易对
            market_data: 市场数据
            signals: 技术指标信号
            account_state: 账户状态

        Returns:
            双层辩论最终结果
        """
        # 第一层：投资辩论
        investment_result = self.investment_debate.debate(symbol, market_data, signals)

        # 如果投资层决定neutral，跳过风控层
        if investment_result.direction == "neutral":
            risk_result = RiskDebateResult(
                verdict=DebateVerdict.REJECT,
                risk_score=100,
                compliance_score=100,
                liquidity_score=100,
                conditions=["投资层决定观望"],
                max_position_size=0,
            )
            return TwoLayerDebateResult(
                investment=investment_result,
                risk=risk_result,
                final_decision="reject",
                final_size=0,
                final_leverage=1,
                debate_log="投资层决定观望，跳过风控层",
            )

        # 第二层：风控辩论
        risk_result = self.risk_debate.debate(
            investment_result, account_state, market_data
        )

        # 综合裁决
        if risk_result.verdict in (DebateVerdict.APPROVE, DebateVerdict.APPROVE_WITH_CONDITIONS):
            final_decision = "execute"
            final_size = risk_result.max_position_size
            final_leverage = account_state.get("leverage", 1)
        elif risk_result.verdict == DebateVerdict.REDUCE_SIZE:
            final_decision = "modify"
            final_size = risk_result.max_position_size
            final_leverage = 1  # 降杠杆
        else:
            final_decision = "reject"
            final_size = 0
            final_leverage = 1

        # 生成辩论日志
        debate_log = self._generate_log(
            investment_result, risk_result, final_decision
        )

        return TwoLayerDebateResult(
            investment=investment_result,
            risk=risk_result,
            final_decision=final_decision,
            final_size=final_size,
            final_leverage=final_leverage,
            debate_log=debate_log,
        )

    def _generate_log(
        self, investment: InvestmentDebateResult,
        risk: RiskDebateResult, final: str,
    ) -> str:
        """生成辩论日志"""
        log = []
        log.append("=" * 60)
        log.append(f"投资辩论: {investment.symbol}")
        log.append(f"  多头得分: {investment.bull_score:.1f}")
        log.append(f"  空头得分: {investment.bear_score:.1f}")
        log.append(f"  方向: {investment.direction} (置信度={investment.confidence:.2f})")
        log.append(f"  交易员决策: {investment.trader_decision}")
        log.append("-" * 60)
        log.append("风控辩论:")
        log.append(f"  风控评分: {risk.risk_score:.1f}")
        log.append(f"  合规评分: {risk.compliance_score:.1f}")
        log.append(f"  流动性评分: {risk.liquidity_score:.1f}")
        log.append(f"  裁决: {risk.verdict.value}")
        if risk.conditions:
            log.append(f"  条件: {', '.join(risk.conditions)}")
        log.append(f"  最大仓位: {risk.max_position_size:.2f}")
        log.append("=" * 60)
        log.append(f"最终决策: {final}")
        return "\n".join(log)


__all__ = [
    "DebateRole",
    "DebateVerdict",
    "DebateArgument",
    "InvestmentDebateResult",
    "RiskDebateResult",
    "TwoLayerDebateResult",
    "InvestmentDebate",
    "RiskDebate",
    "TwoLayerDebateCoordinator",
]
