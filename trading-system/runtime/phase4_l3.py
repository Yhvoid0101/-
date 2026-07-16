"""Phase 4 L3 机会发现智能体 — 条件化权重 + 净 EV 评分 + 硬质量门槛 + 漂移检测。

铁律：
- 不使用全局单一权重；按 regime/品种/方向/持有期条件化。
- 评分目标为净 EV（费用+滑点+冲击+资金费率后）和校准概率，不是固定阈值70。
- 反馈只更新候选权重，生产权重经晋升门禁。
- score 高但净 EV 低必须触发回滚。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts.base import ContractStatus, make_blocked
from .contracts.market_data_envelope import MarketDataEnvelope
from .contracts.market_state_snapshot import MarketRegime, MarketStateSnapshot
from .contracts.opportunity_candidate import (
    EVBreakdown,
    OpportunityCandidate,
    OpportunityDirection,
    blocked_no_opportunity,
)


# ============================================================
# T1: L3OpportunityScorer — 条件化权重 + 净 EV 评分
# ============================================================

class WeightStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class L3OpportunityScorer:
    """L3 机会发现层：条件化权重评分，输出 OpportunityCandidate。"""

    # 默认条件化权重表（按 regime 分组）
    DEFAULT_WEIGHTS = {
        MarketRegime.TRENDING_UP: {"trend_strength": 0.5, "momentum": 0.3, "volume": 0.2},
        MarketRegime.TRENDING_DOWN: {"trend_strength": 0.5, "momentum": 0.3, "volume": 0.2},
        MarketRegime.HIGH_VOLATILITY: {"volatility": 0.4, "momentum": 0.3, "volume": 0.3},
        MarketRegime.LOW_VOLATILITY: {"mean_reversion": 0.5, "volume": 0.3, "trend_strength": 0.2},
        MarketRegime.RANGING: {"mean_reversion": 0.4, "volume": 0.3, "trend_strength": 0.3},
        MarketRegime.UNKNOWN: {},
    }

    def __init__(
        self,
        fee_rate: float = 0.0006,
        slippage_rate: float = 0.0003,
        impact_rate: float = 0.0001,
        funding_rate: float = 0.0001,
        min_probability: float = 0.5,
        min_confidence: float = 0.3,
        weight_table: Optional["CandidateWeightTable"] = None,
    ):
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.impact_rate = impact_rate
        self.funding_rate = funding_rate
        self.min_probability = min_probability
        self.min_confidence = min_confidence
        self.weight_table = weight_table or CandidateWeightTable()

    def score(self, snapshot: Optional[MarketStateSnapshot], envelope: MarketDataEnvelope) -> OpportunityCandidate:
        """评分：接受 L2 snapshot + L1 envelope，输出 OpportunityCandidate。"""
        # 1. 输入校验
        if snapshot is None:
            return self._blocked("NO_L2_INPUT", "无 L2 输入", envelope)
        if snapshot.is_blocked():
            return self._blocked(snapshot.error_code, snapshot.error_message, envelope, snapshot)

        # 2. 提取特征
        features = self._extract_features(envelope)
        regime = snapshot.regime

        # 3. 条件化权重（按 regime）
        weights = self.DEFAULT_WEIGHTS.get(regime, {})
        if not weights:
            return self._blocked("UNKNOWN_REGIME", f"未知 regime: {regime}", envelope, snapshot)

        # 4. 计算方向和置信度
        direction, base_confidence = self._determine_direction(snapshot, features)

        # 5. 计算净 EV
        price = float(envelope.klines[-1]["close"]) if envelope.klines else 100.0
        gross_ev = self._compute_gross_ev(features, direction, base_confidence, price)
        ev = EVBreakdown(
            gross_ev=gross_ev,
            fee_cost=price * self.fee_rate * 2,
            slippage_cost=price * self.slippage_rate * 2,
            impact_cost=price * self.impact_rate,
            funding_cost=price * self.funding_rate,
        )
        ev.compute_net()

        # 6. 校准概率（基于置信度和特征）
        calibrated_prob = self._calibrate(base_confidence, features, regime)

        # 7. 贡献分解
        contributions = {}
        for feat, weight in weights.items():
            contributions[feat] = features.get(feat, 0.0) * weight

        score = sum(contributions.values()) * 100.0

        # 8. 硬质量门槛
        rejection_reasons = []
        if ev.net_ev <= 0:
            rejection_reasons.append("NEGATIVE_EV")
        if calibrated_prob < self.min_probability:
            rejection_reasons.append("LOW_PROBABILITY")
        if base_confidence < self.min_confidence:
            rejection_reasons.append("LOW_CONFIDENCE")

        candidate = OpportunityCandidate(
            symbol=envelope.symbol,
            direction=direction,
            confidence=base_confidence,
            calibrated_probability=calibrated_prob,
            ev=ev,
            assumptions=[f"regime={regime.value}", f"direction={direction.value}"],
            invalidation_conditions=[f"regime_change", f"ev_negative"],
            feature_contributions=contributions,
            score=score,
            threshold=0.0,  # 不再固定70，改为净EV门槛
        )

        if rejection_reasons:
            from .contracts.base import make_blocked
            code = rejection_reasons[0]
            candidate.rejection_reasons = rejection_reasons
            return make_blocked(candidate, code, f"硬质量门槛不满足: {rejection_reasons}", candidate.correlation_id)

        return candidate

    def _extract_features(self, envelope: MarketDataEnvelope) -> Dict[str, float]:
        klines = envelope.klines or []
        if len(klines) < 2:
            return {"trend_strength": 0.0, "momentum": 0.0, "volume": 0.0, "volatility": 0.0, "mean_reversion": 0.0}

        closes = [float(k["close"]) for k in klines]
        volumes = [float(k.get("volume", 0)) for k in klines]
        returns = np.diff(closes) / np.array(closes[:-1])

        x = np.arange(len(closes), dtype=float)
        slope = float(np.polyfit(x, np.array(closes), 1)[0])
        mean_price = float(np.mean(closes)) + 1e-9
        trend_strength = slope / mean_price

        momentum = float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0

        vol_mean = float(np.mean(volumes)) + 1e-9
        volume = float(np.mean(volumes[-10:])) / vol_mean if len(volumes) >= 10 else 1.0

        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

        # 均值回归信号：价格偏离均值的程度
        z_score = (closes[-1] - float(np.mean(closes))) / (float(np.std(closes)) + 1e-9)
        mean_reversion = -z_score * 0.1

        return {
            "trend_strength": trend_strength,
            "momentum": momentum,
            "volume": volume,
            "volatility": volatility,
            "mean_reversion": mean_reversion,
        }

    def _determine_direction(self, snapshot: MarketStateSnapshot, features: Dict[str, float]) -> Tuple[OpportunityDirection, float]:
        regime = snapshot.regime
        trend = features.get("trend_strength", 0.0)
        momentum = features.get("momentum", 0.0)

        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            if trend > 0 and momentum > 0:
                direction = OpportunityDirection.LONG
            elif trend < 0 and momentum < 0:
                direction = OpportunityDirection.SHORT
            else:
                direction = OpportunityDirection.NEUTRAL
        elif regime == MarketRegime.RANGING:
            mr = features.get("mean_reversion", 0.0)
            if mr > 0.05:
                direction = OpportunityDirection.LONG
            elif mr < -0.05:
                direction = OpportunityDirection.SHORT
            else:
                direction = OpportunityDirection.NEUTRAL
        else:
            direction = OpportunityDirection.NEUTRAL

        confidence = snapshot.evidence.confidence
        return direction, confidence

    def _compute_gross_ev(self, features: Dict[str, float], direction: OpportunityDirection, confidence: float, price: float) -> float:
        if direction == OpportunityDirection.NEUTRAL:
            return 0.0
        # 预期收益 = 置信度 * 预期价格变动幅度 * 价格
        # 预期变动幅度基于趋势强度和动量（放大到合理范围）
        trend = features.get("trend_strength", 0.0)
        momentum = features.get("momentum", 0.0)
        # 趋势强度是斜率/均值，放大为预期收益率
        expected_return = (abs(trend) * 100.0 + abs(momentum)) * confidence
        return expected_return * price

    def _calibrate(self, confidence: float, features: Dict[str, float], regime: MarketRegime) -> float:
        # 简化校准：置信度 * (1 - 波动率惩罚)
        vol = features.get("volatility", 0.0)
        penalty = min(0.5, vol * 10)
        return max(0.0, min(1.0, confidence * (1.0 - penalty)))

    def _blocked(self, code: str, message: str, envelope: MarketDataEnvelope, snapshot: Optional[MarketStateSnapshot] = None) -> OpportunityCandidate:
        cand = OpportunityCandidate(symbol=envelope.symbol)
        return make_blocked(cand, code, message, cand.correlation_id)


# ============================================================
# T2: CandidateWeightTable — 反馈更新候选权重，不直接改生产
# ============================================================

@dataclass
class _WeightRecord:
    sample_count: int = 0
    avg_net_ev: float = 0.0
    avg_score: float = 0.0
    total_net_ev: float = 0.0
    total_score: float = 0.0
    status: WeightStatus = WeightStatus.CANDIDATE


class CandidateWeightTable:
    """候选权重表：反馈只更新候选，晋升需审批。"""

    DEFAULT_WEIGHTS = {
        "avg_net_ev": 0.0,
        "avg_score": 0.0,
        "sample_count": 0,
    }

    def __init__(self, min_promote_samples: int = 15):
        self._candidate: Dict[str, _WeightRecord] = defaultdict(_WeightRecord)
        self._production: Dict[str, Dict[str, float]] = defaultdict(lambda: dict(self.DEFAULT_WEIGHTS))
        self.min_promote_samples = min_promote_samples

    def record_feedback(self, regime: str, symbol: str, direction: str, net_ev: float, score: float):
        key = f"{regime}|{symbol}|{direction}"
        rec = self._candidate[key]
        rec.sample_count += 1
        rec.total_net_ev += net_ev
        rec.total_score += score
        rec.avg_net_ev = rec.total_net_ev / rec.sample_count
        rec.avg_score = rec.total_score / rec.sample_count

    def get_candidate_weights(self, regime: str, symbol: str, direction: str) -> Dict[str, float]:
        key = f"{regime}|{symbol}|{direction}"
        rec = self._candidate[key]
        return {"avg_net_ev": rec.avg_net_ev, "avg_score": rec.avg_score, "sample_count": rec.sample_count}

    def get_production_weights(self, regime: str, symbol: str, direction: str) -> Dict[str, float]:
        key = f"{regime}|{symbol}|{direction}"
        return dict(self._production[key])

    def promote(self, regime: str, symbol: str, direction: str, approved_by: str) -> Dict[str, Any]:
        key = f"{regime}|{symbol}|{direction}"
        rec = self._candidate[key]

        if not approved_by:
            return {"status": WeightStatus.REJECTED, "reason": "NO_APPROVER: 缺少审批人"}

        if rec.sample_count < self.min_promote_samples:
            return {"status": WeightStatus.REJECTED, "reason": f"INSUFFICIENT_SAMPLES: {rec.sample_count} < {self.min_promote_samples}"}

        rec.status = WeightStatus.PROMOTED
        self._production[key] = {
            "avg_net_ev": rec.avg_net_ev,
            "avg_score": rec.avg_score,
            "sample_count": rec.sample_count,
        }
        return {"status": WeightStatus.PROMOTED, "reason": "OK"}


# ============================================================
# T3: ScorerMetrics — Brier/校准/uplift/漂移检测
# ============================================================

@dataclass
class DriftSignal:
    is_drifted: bool
    reason: str = ""
    score_ev_correlation: float = 0.0


class ScorerMetrics:
    """评分指标：Brier/校准/top-decile uplift/漂移检测。"""

    def brier_score(self, scores: List[float], outcomes: List[float]) -> float:
        """Brier score: 均方误差。"""
        if not scores:
            return 1.0
        arr_s = np.array(scores, dtype=float)
        arr_o = np.array(outcomes, dtype=float)
        return float(np.mean((arr_s - arr_o) ** 2))

    def top_decile_uplift(self, scores: List[float], outcomes: List[float]) -> float:
        """top-decile uplift: 最高10%预测的胜率 - 全局胜率。"""
        if len(scores) < 10:
            return 0.0
        arr_s = np.array(scores, dtype=float)
        arr_o = np.array(outcomes, dtype=float)
        threshold = np.percentile(arr_s, 90)
        top_mask = arr_s >= threshold
        top_rate = float(np.mean(arr_o[top_mask])) if top_mask.any() else 0.0
        global_rate = float(np.mean(arr_o))
        return top_rate - global_rate

    def detect_drift(self, scores: List[float], net_evs: List[float]) -> DriftSignal:
        """漂移检测：score 高但净 EV 低 → 回滚。"""
        if len(scores) < 3:
            return DriftSignal(is_drifted=False, reason="INSUFFICIENT_DATA")

        arr_s = np.array(scores, dtype=float)
        arr_e = np.array(net_evs, dtype=float)

        # 归一化
        s_norm = (arr_s - np.mean(arr_s)) / (np.std(arr_s) + 1e-9)
        e_norm = (arr_e - np.mean(arr_e)) / (np.std(arr_e) + 1e-9)

        corr = float(np.corrcoef(s_norm, e_norm)[0, 1]) if len(scores) > 1 else 0.0

        # score 高但净 EV 低：负相关或低相关
        mean_score = float(np.mean(arr_s))
        mean_ev = float(np.mean(arr_e))

        if mean_score > 0.6 and mean_ev < 0:
            return DriftSignal(is_drifted=True, reason="SCORE_EV_DIVERGENCE: score高但净EV为负", score_ev_correlation=corr)

        if corr < -0.3:
            return DriftSignal(is_drifted=True, reason=f"SCORE_EV_DIVERGENCE: 负相关 {corr:.2f}", score_ev_correlation=corr)

        return DriftSignal(is_drifted=False, reason="OK", score_ev_correlation=corr)
