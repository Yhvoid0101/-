"""Phase 3 L2 市场理解智能体 — RegimeDetector 契约化包装 + OOD 检测 + 无监督状态发现。

铁律：
- 未知状态进入 L5 审核，不强行归类。
- 禁止标签泄漏：特征只使用当前及历史数据。
- 输出契约 MarketStateSnapshot，携带 regime/置信度/OOD距离/不确定性/证据特征。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts.base import ContractStatus, make_blocked
from .contracts.market_data_envelope import MarketDataEnvelope
from .contracts.market_state_snapshot import MarketRegime, MarketStateSnapshot, RegimeEvidence


# ============================================================
# T1: L2RegimeAnalyzer — 契约化 Regime 分析器
# ============================================================

class L2RegimeAnalyzer:
    """L2 市场理解层：分析 L1 K线，输出 MarketStateSnapshot。

    保留趋势/震荡/高/低波动基础标签，新增 OOD 检测和不确定性量化。
    所有特征只使用当前及历史数据，不使用未来收益率（无标签泄漏）。
    """

    def __init__(
        self,
        min_samples: int = 30,
        ood_threshold: float = 3.0,
        uncertainty_threshold: float = 0.5,
        lookback_window: int = 60,
    ):
        self.min_samples = min_samples
        self.ood_threshold = ood_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.lookback_window = lookback_window
        # 训练分布参考（用于 OOD 距离计算）
        self._ref_volatility: float = 0.01
        self._ref_trend: float = 0.0
        self._ref_vol_std: float = 0.005
        self._ref_trend_std: float = 0.05
        self._initialized = False

    def analyze(self, envelope: MarketDataEnvelope) -> MarketStateSnapshot:
        """分析 L1 数据，输出 MarketStateSnapshot。"""
        # 1. 输入校验
        if envelope is None:
            return self._blocked("INSUFFICIENT_DATA", "无输入")
        if envelope.is_blocked():
            return self._blocked(envelope.error_code, envelope.error_message, envelope)

        klines = envelope.klines or []
        if len(klines) < self.min_samples:
            return self._blocked("INSUFFICIENT_DATA", f"数据不足: {len(klines)} < {self.min_samples}", envelope)

        # 2. 提取特征（无泄漏：只用当前及历史）
        features = self._extract_features(klines)
        volatility = features["volatility"]
        trend_strength = features["trend_strength"]
        volume_profile = features["volume_profile"]
        change_point_prob = features["change_point_prob"]

        # 3. 初始化训练分布参考（首次调用）
        if not self._initialized:
            self._ref_volatility = max(volatility, 1e-6)
            self._ref_trend = abs(trend_strength)
            self._ref_vol_std = max(self._ref_volatility * 0.5, 1e-6)
            self._ref_trend_std = max(self._ref_trend * 0.5 + 0.01, 1e-6)
            self._initialized = True

        # 4. OOD 距离计算（马氏距离近似 + 绝对异常检测）
        returns = self._compute_returns(klines)
        max_abs_return = float(max(abs(r) for r in returns)) if len(returns) > 0 else 0.0
        # 绝对异常：单根收益率 > 50% 视为极端 OOD
        extreme_jump = max_abs_return > 0.5

        vol_z = abs(volatility - self._ref_volatility) / self._ref_vol_std
        trend_z = abs(trend_strength - self._ref_trend) / self._ref_trend_std
        ood_distance = math.sqrt(vol_z * vol_z + trend_z * trend_z)
        # 绝对异常贡献额外 OOD 距离
        if extreme_jump:
            ood_distance += max_abs_return * 10.0

        # 5. 不确定性（基于分类置信度反指标 + 分布分散度）
        regime, confidence = self._classify_regime(volatility, trend_strength, change_point_prob)
        if len(returns) > 2:
            ret_std = float(np.std(returns))
            dispersion = min(1.0, abs(ret_std - self._ref_volatility) / (self._ref_volatility + 1e-6))
            uncertainty = (1.0 - confidence) * 0.7 + dispersion * 0.3
        else:
            uncertainty = 1.0

        # 6. OOD 阻断（不强行归类）
        if ood_distance > self.ood_threshold:
            evidence = RegimeEvidence(
                label="ood",
                confidence=0.0,
                ood_distance=ood_distance,
                supporting_features=features,
                duration_bars=len(klines),
            )
            snap = MarketStateSnapshot(
                symbol=envelope.symbol,
                timeframe=envelope.interval,
                regime=MarketRegime.UNKNOWN,
                evidence=evidence,
                is_ood=True,
                uncertainty=uncertainty,
                is_stable=False,
            )
            return make_blocked(snap, "OOD_UNKNOWN", f"OOD 距离 {ood_distance:.2f} > {self.ood_threshold}", snap.correlation_id)

        # 7. 高不确定性阻断
        if uncertainty > self.uncertainty_threshold:
            evidence = RegimeEvidence(
                label="uncertain",
                confidence=1.0 - uncertainty,
                ood_distance=ood_distance,
                supporting_features=features,
                duration_bars=len(klines),
            )
            snap = MarketStateSnapshot(
                symbol=envelope.symbol,
                timeframe=envelope.interval,
                regime=MarketRegime.UNKNOWN,
                evidence=evidence,
                is_ood=False,
                uncertainty=uncertainty,
                is_stable=False,
            )
            return make_blocked(snap, "HIGH_UNCERTAINTY", f"不确定性 {uncertainty:.2f} > {self.uncertainty_threshold}", snap.correlation_id)

        # 8. Regime 判定已在步骤5完成

        evidence = RegimeEvidence(
            label=regime.value,
            confidence=confidence,
            ood_distance=ood_distance,
            change_points=[change_point_prob],
            supporting_features=features,
            duration_bars=len(klines),
            cross_symbol_consistency=1.0,
        )

        return MarketStateSnapshot(
            symbol=envelope.symbol,
            timeframe=envelope.interval,
            regime=regime,
            evidence=evidence,
            is_ood=False,
            uncertainty=uncertainty,
            is_stable=confidence > 0.5,
        )

    def _extract_features(self, klines: List[Dict[str, Any]]) -> Dict[str, float]:
        """提取特征（无泄漏：只用当前及历史数据）。"""
        closes = [float(k["close"]) for k in klines]
        highs = [float(k["high"]) for k in klines]
        lows = [float(k["low"]) for k in klines]
        volumes = [float(k.get("volume", 0)) for k in klines]

        returns = self._compute_returns(klines)
        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

        # 趋势强度：线性回归斜率 / 均值
        if len(closes) >= 2:
            x = np.arange(len(closes), dtype=float)
            y = np.array(closes, dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            mean_price = float(np.mean(closes)) + 1e-9
            trend_strength = slope / mean_price
        else:
            trend_strength = 0.0

        # 成交量特征：最近窗口均值 / 全局均值
        if len(volumes) >= 10:
            recent = float(np.mean(volumes[-10:]))
            global_v = float(np.mean(volumes)) + 1e-9
            volume_profile = recent / global_v
        else:
            volume_profile = 1.0

        # 变化点概率：最近收益率跳变
        if len(returns) > 5:
            recent_ret = returns[-5:]
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns)) + 1e-9
            change_point_prob = min(1.0, abs(float(np.mean(recent_ret)) - mean_ret) / std_ret)
        else:
            change_point_prob = 0.0

        return {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "volume_profile": volume_profile,
            "change_point_prob": change_point_prob,
        }

    def _compute_returns(self, klines: List[Dict[str, Any]]) -> np.ndarray:
        closes = [float(k["close"]) for k in klines]
        if len(closes) < 2:
            return np.array([])
        return np.diff(closes) / np.array(closes[:-1])

    def _classify_regime(self, volatility: float, trend_strength: float, change_point_prob: float) -> Tuple[MarketRegime, float]:
        """基础四分类（无泄漏）。"""
        vol_threshold = self._ref_volatility * 2.0
        trend_threshold = 0.001

        if volatility > vol_threshold:
            if abs(trend_strength) > trend_threshold:
                regime = MarketRegime.HIGH_VOLATILITY
            else:
                regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, volatility / (vol_threshold + 1e-9))
        elif abs(trend_strength) > trend_threshold:
            regime = MarketRegime.TRENDING_UP if trend_strength > 0 else MarketRegime.TRENDING_DOWN
            confidence = min(1.0, abs(trend_strength) / (trend_threshold * 10))
        elif volatility < self._ref_volatility * 0.5:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, 1.0 - volatility / (self._ref_volatility + 1e-9))
        else:
            regime = MarketRegime.RANGING
            confidence = 0.5

        return regime, max(0.1, min(0.99, confidence))

    def _blocked(self, code: str, message: str, envelope: Optional[MarketDataEnvelope] = None) -> MarketStateSnapshot:
        snap = MarketStateSnapshot(
            symbol=envelope.symbol if envelope else "",
            timeframe=envelope.interval if envelope else "",
        )
        return make_blocked(snap, code, message, snap.correlation_id)


# ============================================================
# T2: 无监督状态发现候选 + 稳定性/经济意义评估
# ============================================================

class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    SHADOW = "SHADOW"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class RegimeCandidate:
    """无监督状态发现候选。"""
    label: str
    features: Dict[str, float]
    sample_count: int
    stability_score: float
    min_duration_bars: int
    economic_meaning: str
    cross_window_reproducibility: float
    future_behavior_diff: float
    candidate_id: str = ""
    status: CandidateStatus = CandidateStatus.PROPOSED
    rejection_reason: str = ""


@dataclass
class CandidateResult:
    candidate: RegimeCandidate
    status: CandidateStatus
    rejection_reason: str = ""


class RegimeCandidateEvaluator:
    """候选状态评估器 — 稳定性/经济意义/样本数/跨窗口复现/未来行为差异。"""

    def __init__(
        self,
        min_samples: int = 100,
        min_stability: float = 0.6,
        min_duration: int = 5,
        min_cross_window: float = 0.5,
        min_future_diff: float = 0.15,
    ):
        self.min_samples = min_samples
        self.min_stability = min_stability
        self.min_duration = min_duration
        self.min_cross_window = min_cross_window
        self.min_future_diff = min_future_diff

    def evaluate(self, candidate: RegimeCandidate) -> CandidateResult:
        """评估候选。不满足条件的直接 REJECTED。"""
        self._counter = getattr(self, "_counter", 0) + 1
        candidate.candidate_id = f"rc-{self._counter:04d}"

        if candidate.sample_count < self.min_samples:
            return self._reject(candidate, f"SAMPLES: {candidate.sample_count} < {self.min_samples}")
        if candidate.stability_score < self.min_stability:
            return self._reject(candidate, f"STABILITY: {candidate.stability_score:.2f} < {self.min_stability}")
        if candidate.min_duration_bars < self.min_duration:
            return self._reject(candidate, f"DURATION: {candidate.min_duration_bars} < {self.min_duration}")
        if candidate.cross_window_reproducibility < self.min_cross_window:
            return self._reject(candidate, f"CROSS_WINDOW: {candidate.cross_window_reproducibility:.2f} < {self.min_cross_window}")
        if candidate.future_behavior_diff < self.min_future_diff:
            return self._reject(candidate, f"FUTURE_DIFF: {candidate.future_behavior_diff:.2f} < {self.min_future_diff}")
        if not candidate.economic_meaning:
            return self._reject(candidate, "ECONOMIC: 缺少经济意义解释")

        candidate.status = CandidateStatus.SHADOW
        return CandidateResult(candidate=candidate, status=CandidateStatus.SHADOW)

    def _reject(self, candidate: RegimeCandidate, reason: str) -> CandidateResult:
        candidate.status = CandidateStatus.REJECTED
        candidate.rejection_reason = reason
        return CandidateResult(candidate=candidate, status=CandidateStatus.REJECTED, rejection_reason=reason)
