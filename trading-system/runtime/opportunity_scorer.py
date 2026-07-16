#!/usr/bin/env python3
"""Layer 3: Opportunity Scorer — 机会发现层

对每笔交易信号进行0-100置信度评分，决定是否值得开仓及仓位大小。

评分维度 (总分100):
  1. regime匹配度 (25分): 当前regime与信号方向的匹配度
  2. 资金费率信号 (20分): 资金费率极性是否支持信号方向
  3. OI变化趋势 (15分): OI增加+价格上涨=多头强势
  4. 波动率适宜度 (15分): 适中波动率最好(太低无机会,太高风险大)
  5. 信号置信度 (15分): decide()返回的confidence
  6. 微观结构 (10分): VPIN/流动性等

评分门控:
  - score < 70:  拒绝开仓 (decision = None)
  - 70 <= score < 85: 正常仓位
  - score >= 85: 加重仓位 (quantity × 1.2)

接入点: evolution_loop.py L2791之后 (decision确认后、submit_long/short前)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("hermes.opportunity_scorer")


@dataclass
class ScoreBreakdown:
    """Opportunity Score 各维度分数明细"""
    regime_match: float = 0.0        # /25
    funding_rate: float = 0.0        # /20
    oi_trend: float = 0.0            # /15
    volatility: float = 0.0          # /15
    signal_confidence: float = 0.0   # /15
    microstructure: float = 0.0      # /10
    total: float = 0.0               # /100
    rejected: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_match": round(self.regime_match, 2),
            "funding_rate": round(self.funding_rate, 2),
            "oi_trend": round(self.oi_trend, 2),
            "volatility": round(self.volatility, 2),
            "signal_confidence": round(self.signal_confidence, 2),
            "microstructure": round(self.microstructure, 2),
            "total": round(self.total, 2),
            "rejected": self.rejected,
            "reason": self.reason,
        }


class OpportunityScorer:
    """Layer 3: 机会发现层 — 信号置信度评分器

    在decide()返回后、submit_long/short执行前对决策进行评分。
    得分<70的信号被拒绝，得分>=85的信号仓位加重。
    """

    # 评分阈值
    REJECT_THRESHOLD = 45.0    # Phase 14.21: 55→45 沙盘探索期降低门槛让更多信号通过
    BOOST_THRESHOLD = 75.0     # >=75 加重仓位
    BOOST_MULTIPLIER = 1.2     # 加重倍数

    # 各维度最大分
    MAX_REGIME = 25.0
    MAX_FUNDING = 20.0
    MAX_OI = 15.0
    MAX_VOL = 15.0
    MAX_CONFIDENCE = 15.0
    MAX_MICRO = 10.0

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._score_count = 0
        self._reject_count = 0
        self._boost_count = 0
        logger.info("OpportunityScorer initialized: threshold reject<%s boost>=%s",
                    self.REJECT_THRESHOLD, self.BOOST_THRESHOLD)

    def score(
        self,
        decision: Dict[str, Any],
        tick: Dict[str, Any],
        agent_id: str = "",
        symbol: str = "",
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, ScoreBreakdown]:
        """对交易决策进行机会评分

        Args:
            decision: decide()返回的决策dict (含action/quantity/confidence等)
            tick: 当前tick数据 (含regime/adaptive_regime/atr/vpin等)
            agent_id: agent ID
            symbol: 交易对
            market_context: D1市场上下文 (含funding_rate/open_interest等)

        Returns:
            (total_score, ScoreBreakdown)
        """
        if not self.enabled:
            return 100.0, ScoreBreakdown(total=100.0)

        self._score_count += 1
        action = decision.get("action", "neutral")
        breakdown = ScoreBreakdown()

        # 1. regime匹配度 (25分)
        breakdown.regime_match = self._score_regime_match(action, tick)

        # 2. 资金费率信号 (20分)
        breakdown.funding_rate = self._score_funding_rate(action, market_context)

        # 3. OI变化趋势 (15分)
        breakdown.oi_trend = self._score_oi_trend(action, market_context)

        # 4. 波动率适宜度 (15分)
        breakdown.volatility = self._score_volatility(tick)

        # 5. 信号置信度 (15分)
        breakdown.signal_confidence = self._score_signal_confidence(decision)

        # 6. 微观结构 (10分)
        breakdown.microstructure = self._score_microstructure(tick)

        # 总分
        breakdown.total = (
            breakdown.regime_match + breakdown.funding_rate + breakdown.oi_trend +
            breakdown.volatility + breakdown.signal_confidence + breakdown.microstructure
        )

        # 门控判断
        if breakdown.total < self.REJECT_THRESHOLD:
            # Phase 14.21: 沙盘模式软拒绝 — 即使低分也允许交易 (探索期)
            _sandbox_soft = getattr(self, '_sandbox_mode', False)
            if _sandbox_soft and breakdown.total >= 30.0:
                # 沙盘模式: score>=30 不拒绝, 只记录
                breakdown.rejected = False
                breakdown.reason = f"SANDBOX_SOFT: score={breakdown.total:.1f}<{self.REJECT_THRESHOLD} (沙盘模式允许)"
                logger.info(
                    "OpportunityScorer SANDBOX_SOFT: agent=%s symbol=%s action=%s score=%.1f "
                "[regime=%.1f fr=%.1f oi=%.1f vol=%.1f conf=%.1f micro=%.1f]",
                agent_id[:12] if agent_id else "?", symbol, action, breakdown.total,
                breakdown.regime_match, breakdown.funding_rate, breakdown.oi_trend,
                breakdown.volatility, breakdown.signal_confidence, breakdown.microstructure,
            )
        elif breakdown.total >= self.BOOST_THRESHOLD:
            self._boost_count += 1
            logger.info(
                "OpportunityScorer BOOST: agent=%s symbol=%s action=%s score=%.1f>=%s "
                "[regime=%.1f fr=%.1f oi=%.1f vol=%.1f conf=%.1f micro=%.1f]",
                agent_id[:12] if agent_id else "?", symbol, action, breakdown.total,
                self.BOOST_THRESHOLD,
                breakdown.regime_match, breakdown.funding_rate, breakdown.oi_trend,
                breakdown.volatility, breakdown.signal_confidence, breakdown.microstructure,
            )

        return breakdown.total, breakdown

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        tick: Dict[str, Any],
        agent_id: str = "",
        symbol: str = "",
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """对decision应用Opportunity Score门控

        Returns:
            修改后的decision，或None(拒绝开仓)
        """
        if not self.enabled:
            return decision

        score, breakdown = self.score(decision, tick, agent_id, symbol, market_context)

        if breakdown.rejected:
            return None

        # 注入评分到decision
        decision["opportunity_score"] = score
        decision["opportunity_breakdown"] = breakdown.to_dict()

        # 高分加重仓位
        if score >= self.BOOST_THRESHOLD and decision.get("quantity"):
            decision["quantity"] = decision["quantity"] * self.BOOST_MULTIPLIER

        return decision

    # ------------------------------------------------------------------
    # 各维度评分实现
    # ------------------------------------------------------------------

    def _score_regime_match(self, action: str, tick: Dict[str, Any]) -> float:
        """regime匹配度评分 (0-25分)

        信号方向与当前regime的匹配度:
        - bull regime + long = 满分
        - bull regime + short = 低分
        - bear regime + short = 满分
        - bear regime + long = 低分
        - sideways regime = 中等分
        """
        regime_state = tick.get("regime", {})
        if not isinstance(regime_state, dict):
            return self.MAX_REGIME * 0.5  # 未知regime给一半分

        market_regime = regime_state.get("market_regime", "neutral")

        # adaptive_regime更精细
        ad_regime = tick.get("adaptive_regime", {})
        ad_label = ad_regime.get("regime", "") if isinstance(ad_regime, dict) else ""
        allowed_direction = ad_regime.get("allowed_direction", 0) if isinstance(ad_regime, dict) else 0

        # 如果adaptive_regime有allowed_direction，直接用它
        if allowed_direction != 0:
            d_dir = 1 if action == "long" else (-1 if action == "short" else 0)
            if d_dir == 0:
                return self.MAX_REGIME * 0.5

            # 方向匹配检查
            if allowed_direction == 2:  # 双向允许
                return self.MAX_REGIME * 0.8
            if (allowed_direction > 0 and d_dir > 0) or (allowed_direction < 0 and d_dir < 0):
                return self.MAX_REGIME  # 方向匹配，满分
            else:
                return self.MAX_REGIME * 0.2  # 方向不匹配，低分

        # 回退到market_regime匹配
        if market_regime == "bull":
            return self.MAX_REGIME if action == "long" else self.MAX_REGIME * 0.2
        elif market_regime == "bear":
            return self.MAX_REGIME if action == "short" else self.MAX_REGIME * 0.2
        elif market_regime == "sideways":
            return self.MAX_REGIME * 0.6  # 震荡市给中等分
        else:
            return self.MAX_REGIME * 0.5  # 未知

    def _score_funding_rate(self, action: str, market_context: Optional[Dict[str, Any]]) -> float:
        """资金费率信号评分 (0-20分)

        资金费率极性是否支持信号方向:
        - 正资金费率(多头付费) + short = 高分(逆势做空)
        - 负资金费率(空头付费) + long = 高分(逆势做多)
        - 正资金费率 + long = 中分(顺势但成本高)
        - 资金费率接近0 = 中等分
        """
        if not market_context:
            return self.MAX_FUNDING * 0.75  # 无数据给75% (不惩罚)

        # 从market_context获取funding_rate
        fr = None
        if isinstance(market_context, dict):
            fr = market_context.get("funding_rate")
            if fr is None:
                fr = market_context.get("fr")

        if fr is None:
            return self.MAX_FUNDING * 0.75  # 无数据给75%

        try:
            fr = float(fr)
        except (ValueError, TypeError):
            return self.MAX_FUNDING * 0.5

        # 资金费率绝对值越大，反向交易价值越高
        fr_abs = abs(fr)
        fr_magnitude = min(fr_abs / 0.001, 1.0)  # 0.1%费率为满分基准

        if action == "long":
            if fr < 0:  # 负费率+做多 = 高分(空头付费,多头收钱)
                return self.MAX_FUNDING * (0.6 + 0.4 * fr_magnitude)
            else:  # 正费率+做多 = 中分(多头付费)
                return self.MAX_FUNDING * (0.6 - 0.3 * fr_magnitude)
        elif action == "short":
            if fr > 0:  # 正费率+做空 = 高分(多头付费,空头收钱)
                return self.MAX_FUNDING * (0.6 + 0.4 * fr_magnitude)
            else:  # 负费率+做空 = 中分(空头付费)
                return self.MAX_FUNDING * (0.6 - 0.3 * fr_magnitude)
        else:
            return self.MAX_FUNDING * 0.5

    def _score_oi_trend(self, action: str, market_context: Optional[Dict[str, Any]]) -> float:
        """OI变化趋势评分 (0-15分)

        OI增加+价格上涨=多头强势(利好long)
        OI增加+价格下跌=空头强势(利好short)
        OI减少=平仓潮(中性偏负)
        """
        if not market_context:
            return self.MAX_OI * 0.8  # 无数据给80% (不惩罚)

        oi = None
        if isinstance(market_context, dict):
            oi = market_context.get("open_interest")
            if oi is None:
                oi = market_context.get("oi")

        if oi is None:
            return self.MAX_OI * 0.8  # 无数据给80%

        # OI是绝对值，无法判断趋势，给中等分
        # 如果有OI变化率数据会更好
        try:
            oi_val = float(oi) if oi else 0
            if oi_val > 0:
                return self.MAX_OI * 0.6  # 有OI数据，给中等偏上分数
            else:
                return self.MAX_OI * 0.3
        except (ValueError, TypeError):
            return self.MAX_OI * 0.5

    def _score_volatility(self, tick: Dict[str, Any]) -> float:
        """波动率适宜度评分 (0-15分)

        适中波动率最好(有交易机会但风险可控):
        - ATR/price 在 0.5%-2% = 高分
        - ATR/price < 0.3% = 低分(波动太小)
        - ATR/price > 3% = 低分(波动太大)
        """
        atr = tick.get("atr", 0.0)
        price = tick.get("price", 0) or tick.get("close", 0)

        if not price or not atr:
            # 尝试从market对象获取
            market = tick.get("market")
            if market:
                price = getattr(market, "close", 0)
                atr = tick.get("atr", getattr(market, "atr", 0))

        if not price or not atr:
            return self.MAX_VOL * 0.5

        try:
            atr_pct = float(atr) / float(price) if price > 0 else 0
        except (ValueError, TypeError, ZeroDivisionError):
            return self.MAX_VOL * 0.5

        # 波动率评分曲线
        if 0.005 <= atr_pct <= 0.02:  # 0.5%-2% 最佳
            return self.MAX_VOL
        elif 0.003 <= atr_pct < 0.005:  # 0.3%-0.5% 较好
            return self.MAX_VOL * 0.7
        elif 0.02 < atr_pct <= 0.03:  # 2%-3% 可接受
            return self.MAX_VOL * 0.6
        elif atr_pct < 0.003:  # 太低
            return self.MAX_VOL * 0.3
        else:  # >3% 太高
            return self.MAX_VOL * 0.2

    def _score_signal_confidence(self, decision: Dict[str, Any]) -> float:
        """信号置信度评分 (0-15分)

        直接映射decide()返回的confidence值
        最低给20%分 (避免confidence=0导致完全拒绝)
        """
        confidence = decision.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.5

        # Phase 14.22: 移除20%地板 — confidence=0 应该得0分, 不是3分
        # 根因: 20%地板让噪声级信号(confidence≈0)仍贡献3分到OpportunityScorer总分,
        #   推动总分超过REJECT_THRESHOLD/BOOST_THRESHOLD, 绕过信号质量门控
        # 修复: confidence 0-1 线性映射到 0-15分 (无地板)
        return self.MAX_CONFIDENCE * max(0.0, min(1.0, confidence))

    def _score_microstructure(self, tick: Dict[str, Any]) -> float:
        """微观结构评分 (0-10分)

        基于VPIN/流动性等微观结构指标
        """
        vpin = tick.get("vpin", {})
        if isinstance(vpin, dict):
            vpin_val = vpin.get("vpin", 0.0)
        else:
            vpin_val = 0.0

        liq_gap = tick.get("liquidity_gap", {})
        if isinstance(liq_gap, dict):
            low_liq_pct = liq_gap.get("low_pct", 0.0)
        else:
            low_liq_pct = 0.0

        # VPIN适中为好(有交易活动但不毒性)
        try:
            vpin_val = float(vpin_val)
        except (ValueError, TypeError):
            vpin_val = 0.0

        if vpin_val < 0.3:
            vpin_score = 0.8  # 低VPIN好
        elif vpin_val < 0.6:
            vpin_score = 0.6  # 中VPIN可接受
        else:
            vpin_score = 0.3  # 高VPIN差(毒性流动)

        # 流动性充足为好
        try:
            low_liq_pct = float(low_liq_pct)
        except (ValueError, TypeError):
            low_liq_pct = 0.0

        if low_liq_pct < 0.2:
            liq_score = 1.0  # 流动性充足
        elif low_liq_pct < 0.5:
            liq_score = 0.6
        else:
            liq_score = 0.3  # 流动性不足

        return self.MAX_MICRO * (0.5 * vpin_score + 0.5 * liq_score)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取评分器统计"""
        return {
            "enabled": self.enabled,
            "total_scored": self._score_count,
            "total_rejected": self._reject_count,
            "total_boosted": self._boost_count,
            "reject_rate": self._reject_count / max(1, self._score_count),
            "boost_rate": self._boost_count / max(1, self._score_count),
        }


# 便捷函数
def create_opportunity_scorer(enabled: bool = True) -> OpportunityScorer:
    """创建OpportunityScorer实例"""
    return OpportunityScorer(enabled=enabled)
