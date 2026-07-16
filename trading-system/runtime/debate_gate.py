# -*- coding: utf-8 -*-
"""
双层辩论决策门控 — P1-1优化

将debate_architecture.py的双层辩论架构集成到交易决策流程:
  第一层: 投资辩论(多头/空头/交易员) — 产生方向决策
  第二层: 风控三方辩论(风控官/合规官/流动性官) — 硬性否决权

设计原则:
  - 不破坏现有multi_agent_collaboration流程
  - 作为可选决策门控层叠加
  - 风控层有硬性否决权(REJECT/STRONG_REJECT)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.debate_gate")

try:
    _DEBATE_AVAILABLE = True
except ImportError:
    _DEBATE_AVAILABLE = False


@dataclass(slots=True)
class DebateGateResult:
    """辩论门控结果"""
    final_decision: str = "execute"  # execute/reject/modify
    final_size_multiplier: float = 1.0
    final_leverage: int = 1
    investment_direction: str = "neutral"  # long/short/neutral
    investment_confidence: float = 0.0
    risk_level: str = "unknown"  # APPROVE/APPROVE_WITH_CONDITIONS/REDUCE_SIZE/REJECT/STRONG_REJECT
    debate_log: list = field(default_factory=list)
    rejection_reason: str = ""


class DebateGate:
    """双层辩论决策门控

    在Agent决策后、订单提交前应用辩论门控:
      1. 投资层: 多空辩论产生方向和置信度
      2. 风控层: 三方辩论裁决(有硬性否决权)

    使用方式:
        gate = DebateGate(enabled=True)
        result = gate.review_decision(
            direction=1,  # 1=long, -1=short, 0=neutral
            confidence=0.7,
            quantity=0.5,
            leverage=3,
            market_state={...},
            agent_state={...},
        )
        if result.final_decision == "reject":
            # 风控否决,不执行
            pass
    """

    def __init__(
        self,
        enabled: bool = True,
        investment_enabled: bool = True,
        risk_enabled: bool = True,
    ):
        """
        Args:
            enabled: 总开关
            investment_enabled: 投资辩论层开关
            risk_enabled: 风控辩论层开关
        """
        self.enabled = enabled and _DEBATE_AVAILABLE
        self.investment_enabled = investment_enabled
        self.risk_enabled = risk_enabled

        if not _DEBATE_AVAILABLE:
            logger.warning("debate_architecture不可用,DebateGate将作为passthrough")
            self.enabled = False

        # 统计
        self._stats = {
            "total_reviews": 0,
            "approved": 0,
            "modified": 0,
            "rejected": 0,
            "strong_rejected": 0,
        }

    def review_decision(
        self,
        direction: int,
        confidence: float,
        quantity: float,
        leverage: int,
        market_state: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> DebateGateResult:
        """审查交易决策

        Args:
            direction: 方向 (1=long, -1=short, 0=neutral)
            confidence: 置信度 (0-1)
            quantity: 数量
            leverage: 杠杆
            market_state: 市场状态
            agent_state: Agent状态

        Returns:
            DebateGateResult 辩论结果
        """
        self._stats["total_reviews"] += 1

        if not self.enabled:
            # 未启用,passthrough
            result = DebateGateResult(
                final_decision="execute",
                final_size_multiplier=1.0,
                final_leverage=leverage,
                investment_direction="long" if direction > 0 else "short" if direction < 0 else "neutral",
                investment_confidence=confidence,
                risk_level="APPROVE",
            )
            self._stats["approved"] += 1
            return result

        market_state = market_state or {}
        agent_state = agent_state or {}
        debate_log = []

        try:
            # 第一层: 投资辩论
            inv_direction = "neutral"
            inv_confidence = confidence

            if self.investment_enabled and direction != 0:
                # 简化版投资辩论: 基于市场状态和方向
                trend = market_state.get("trend", "unknown")
                volatility = market_state.get("volatility", 0.01)
                rsi = market_state.get("rsi", 50)

                # 多头论点
                bull_score = 50
                if direction > 0:
                    bull_score += 20
                if trend == "up":
                    bull_score += 15
                if rsi < 30:
                    bull_score += 15  # 超卖反弹
                bull_score = min(100, bull_score)

                # 空头论点
                bear_score = 50
                if direction < 0:
                    bear_score += 20
                if trend == "down":
                    bear_score += 15
                if rsi > 70:
                    bear_score += 15  # 超买回落
                bear_score = min(100, bear_score)

                # 交易员综合
                diff = bull_score - bear_score
                if diff > 15:
                    inv_direction = "long"
                    inv_confidence = min(1.0, 0.5 + diff / 100)
                elif diff < -15:
                    inv_direction = "short"
                    inv_confidence = min(1.0, 0.5 + abs(diff) / 100)
                else:
                    inv_direction = "neutral"
                    inv_confidence = 0.3

                debate_log.append({
                    "layer": "investment",
                    "bull_score": bull_score,
                    "bear_score": bear_score,
                    "direction": inv_direction,
                    "confidence": inv_confidence,
                })

            # 投资层neutral则跳过风控层
            if inv_direction == "neutral":
                result = DebateGateResult(
                    final_decision="execute",
                    final_size_multiplier=0.0,  # neutral不交易
                    final_leverage=leverage,
                    investment_direction="neutral",
                    investment_confidence=inv_confidence,
                    risk_level="APPROVE",
                    debate_log=debate_log,
                )
                self._stats["approved"] += 1
                return result

            # 第二层: 风控三方辩论
            risk_level = "APPROVE"
            size_multiplier = 1.0
            final_leverage = leverage
            rejection_reason = ""

            if self.risk_enabled:
                # 风控官评估
                max_drawdown = agent_state.get("max_drawdown", 0.0)
                current_loss = agent_state.get("current_daily_loss", 0.0)
                risk_score = 100
                if max_drawdown > 0.15:
                    risk_score -= 30
                if current_loss < -0.03:
                    risk_score -= 25
                if leverage > 5:
                    risk_score -= 20
                risk_score = max(0, risk_score)

                # 合规官评估
                compliance_score = 100
                if leverage > 10:
                    compliance_score -= 40
                if quantity > 0.5:  # 大仓位
                    compliance_score -= 20
                if abs(current_loss) > 0.05:  # 日亏损超5%
                    compliance_score -= 30
                compliance_score = max(0, compliance_score)

                # 流动性官评估
                liquidity_score = 100
                volume = market_state.get("volume", 1000000)
                if volume < 100000:  # 低流动性
                    liquidity_score -= 40
                spread = market_state.get("spread", 0.001)
                if spread > 0.005:  # 宽价差
                    liquidity_score -= 30
                liquidity_score = max(0, liquidity_score)

                # 最终裁决
                min_score = min(risk_score, compliance_score, liquidity_score)

                if risk_score < 20 or compliance_score < 20:
                    risk_level = "STRONG_REJECT"
                    rejection_reason = f"硬性否决(风险={risk_score},合规={compliance_score})"
                    final_leverage = 0
                    size_multiplier = 0.0
                elif min_score > 80:
                    risk_level = "APPROVE"
                    size_multiplier = 1.5  # 高分加仓
                elif min_score > 60:
                    risk_level = "APPROVE_WITH_CONDITIONS"
                    size_multiplier = 1.0
                elif min_score > 40:
                    risk_level = "REDUCE_SIZE"
                    size_multiplier = 0.5
                elif min_score > 20:
                    risk_level = "REDUCE_SIZE"
                    size_multiplier = 0.25  # 仅试探
                else:
                    risk_level = "REJECT"
                    rejection_reason = f"风控否决(最低分={min_score})"
                    final_leverage = 0
                    size_multiplier = 0.0

                debate_log.append({
                    "layer": "risk",
                    "risk_officer_score": risk_score,
                    "compliance_officer_score": compliance_score,
                    "liquidity_officer_score": liquidity_score,
                    "min_score": min_score,
                    "risk_level": risk_level,
                })

            # 最终决策
            if risk_level in ("REJECT", "STRONG_REJECT"):
                final_decision = "reject"
                self._stats["strong_rejected" if risk_level == "STRONG_REJECT" else "rejected"] += 1
            elif risk_level == "REDUCE_SIZE":
                final_decision = "modify"
                self._stats["modified"] += 1
            else:
                final_decision = "execute"
                self._stats["approved"] += 1

            result = DebateGateResult(
                final_decision=final_decision,
                final_size_multiplier=size_multiplier,
                final_leverage=final_leverage,
                investment_direction=inv_direction,
                investment_confidence=inv_confidence,
                risk_level=risk_level,
                debate_log=debate_log,
                rejection_reason=rejection_reason,
            )

        except Exception as e:
            logger.warning("辩论门控异常,放行: %s", e)
            result = DebateGateResult(
                final_decision="execute",
                final_size_multiplier=1.0,
                final_leverage=leverage,
                investment_direction="long" if direction > 0 else "short" if direction < 0 else "neutral",
                investment_confidence=confidence,
                risk_level="APPROVE",
                debate_log=debate_log + [{"error": str(e)}],
            )
            self._stats["approved"] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取辩论统计"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """重置统计"""
        for key in self._stats:
            self._stats[key] = 0


__all__ = ["DebateGate", "DebateGateResult"]
