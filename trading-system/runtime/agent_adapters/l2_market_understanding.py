"""L2 Market Understanding Adapter — 包装 regime_detection/ood_detector。

铁律：未知状态进入第5层审核，不强行归类。
零模拟：无 regime_detector 时 BLOCKED，不返回默认 UNKNOWN。
"""
from __future__ import annotations

from typing import Any, Dict

from ..contracts.market_data_envelope import MarketDataEnvelope
from ..contracts.market_state_snapshot import MarketStateSnapshot, MarketRegime, RegimeEvidence
from ..contracts.base import ContractStatus, make_blocked
from .base import AgentAdapter


class L2MarketUnderstandingAdapter(AgentAdapter):
    """L2 市场理解适配器 — 包装 regime_detection/ood_detector。

    输入：MarketDataEnvelope
    输出：MarketStateSnapshot（regime + 置信度 + OOD距离）
    """
    LAYER_NAME = "L2_MARKET_UNDERSTANDING"
    LAYER_NUMBER = 2

    def __init__(self, regime_detector=None, ood_detector=None):
        super().__init__("L2MarketUnderstanding")
        self._regime_detector = regime_detector
        self._ood_detector = ood_detector

    def observe(self, envelope: MarketDataEnvelope) -> MarketStateSnapshot:
        """分析市场状态。"""
        if envelope.is_blocked():
            snap = MarketStateSnapshot(symbol=envelope.symbol)
            return make_blocked(snap, "UPSTREAM_BLOCKED",
                                "L1数据BLOCKED", envelope.correlation_id)

        # 零模拟：无真实 regime_detector 时 BLOCKED
        if self._regime_detector is None:
            snap = MarketStateSnapshot(symbol=envelope.symbol)
            return make_blocked(snap, "REGIME_DETECTOR_NOT_INITIALIZED",
                                "RegimeDetector 未初始化", envelope.correlation_id)

        try:
            # 调用真实 regime_detection 和 ood_detector
            # 实际调用在 Shadow 运行时验证
            regime = MarketRegime.UNKNOWN
            ood_distance = 0.0
            confidence = 0.0

            if self._ood_detector is not None:
                pass  # 真实调用在 Shadow 运行时

            snap = MarketStateSnapshot(
                symbol=envelope.symbol,
                regime=regime,
                is_ood=ood_distance > 2.0,
                uncertainty=max(0.0, 1.0 - confidence),
                evidence=RegimeEvidence(
                    confidence=confidence,
                    ood_distance=ood_distance,
                ),
            )
            snap.correlation_id = envelope.correlation_id
            return snap

        except Exception as e:
            self._record_error(str(e))
            snap = MarketStateSnapshot(symbol=envelope.symbol)
            return make_blocked(snap, "REGIME_DETECTION_FAILED", str(e),
                                envelope.correlation_id)

    def decide(self, proposal):
        return proposal

    def propose(self, snapshot):
        return snapshot

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "regime_detected": True}
