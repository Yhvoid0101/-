"""OOD (Out-of-Distribution) Detector — Layer 5: Intelligent Audit Layer

Aegis-X 8层架构补全: 事前审核OOD检测 (30% → 100%)

Purpose:
    在每笔交易执行前, 检测当前市场状态是否在训练分布内.
    如果当前环境完全没见过(OOD), 系统应拒绝开仓或大幅缩仓,
    而不是硬上.

Design:
    1. Feature extraction: 从tick提取5维特征向量
       - return_pct: 收益率百分比
       - volatility_pct: ATR/price × 100
       - volume_log: log(volume + 1)
       - regime_code: 0(sideways)/1(bull)/2(bear)
       - vpin_value: VPIN毒性值

    2. Baseline management: 滚动窗口收集最近N个特征向量
       - Warmup: 前N=200个tick用于建立初始基线
       - Update: 每500个tick用EMA更新基线统计量

    3. Detection: Mahalanobis距离 + Z-score回退
       - distance < 2.5: IN_DISTRIBUTION → multiplier=1.0
       - 2.5 <= distance < 4.0: NEAR_OOD → multiplier=0.5
       - distance >= 4.0: FAR_OOD → reject (return None)

    4. Fallback: 协方差矩阵奇异时回退到Z-score检测

References:
    - Mahalanobis, P.C. (1936) — Mahalanobis距离
    - Blundell et al. (2015) — Weight Uncertainty in Neural Networks
    - Liang et al. (2018) — Enhancing the Reliability of OOD Detection
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OODLevel(Enum):
    """OOD检测结果等级"""

    IN_DISTRIBUTION = "in_distribution"  # 分布内, 允许交易
    NEAR_OOD = "near_ood"  # 近OOD, 缩仓50%
    FAR_OOD = "far_ood"  # 远OOD, 拒绝交易
    WARMUP = "warmup"  # 预热期, 允许交易


@dataclass
class OODResult:
    """OOD检测结果"""

    level: OODLevel
    distance: float  # Mahalanobis距离 (或Z-score回退值)
    multiplier: float  # 仓位调整系数
    feature_vector: Optional[np.ndarray] = None  # 当前特征向量
    feature_names: List[str] = field(default_factory=list)
    reason: str = ""


class OODDetector:
    """OOD检测器 — 事前审核

    Usage:
        detector = OODDetector(enabled=True)
        # 每个tick更新基线
        detector.update_baseline(tick, prev_close)
        # 交易前检测
        result = detector.check(tick, prev_close)
        if result.level == OODLevel.FAR_OOD:
            # 拒绝交易
            decision = None
        elif result.level == OODLevel.NEAR_OOD:
            # 缩仓50%
            decision["quantity"] *= result.multiplier
    """

    # 特征维度
    FEATURE_NAMES = [
        "return_pct",
        "volatility_pct",
        "volume_log",
        "regime_code",
        "vpin_value",
    ]
    N_FEATURES = 5

    # Warmup参数
    WARMUP_SIZE = 200  # 前200个tick用于建立初始基线

    # 基线更新参数
    BASELINE_UPDATE_INTERVAL = 500  # 每500个tick更新一次基线统计量
    ROLLING_WINDOW_SIZE = 1000  # 滚动窗口大小

    # OOD阈值 (Mahalanobis距离)
    NEAR_OOD_THRESHOLD = 2.5  # 2.5σ
    FAR_OOD_THRESHOLD = 4.0  # 4.0σ

    # Z-score回退阈值
    ZSCORE_NEAR_THRESHOLD = 3.0
    ZSCORE_FAR_THRESHOLD = 5.0

    # 仓位调整
    NEAR_OOD_MULTIPLIER = 0.5  # 近OOD缩仓50%
    IN_DIST_MULTIPLIER = 1.0
    FAR_OOD_MULTIPLIER = 0.0  # 远OOD拒绝

    # 协方差正则化
    COV_REGULARIZATION = 1e-4
    # 常数维度阈值: std < 此值的维度被视为常数, 不参与OOD判定
    CONSTANT_DIM_THRESHOLD = 1e-6
    # 常数维度的正则化(大值=该维度变化时贡献小)
    CONSTANT_DIM_REGULARIZATION = 1.0

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # 基线数据
        self._feature_buffer: Deque[np.ndarray] = deque(
            maxlen=self.ROLLING_WINDOW_SIZE
        )
        self._tick_count: int = 0
        self._last_baseline_update: int = 0

        # 基线统计量
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None  # 协方差矩阵的逆
        self._std: Optional[np.ndarray] = None  # 用于Z-score回退
        self._use_mahalanobis: bool = True

        # 前一个close价格 (用于计算收益率)
        self._prev_close: Dict[str, float] = {}  # symbol -> prev_close

        # 统计
        self._ood_check_count: int = 0
        self._near_ood_count: int = 0
        self._far_ood_count: int = 0

        logger.info(
            "Layer5 OODDetector initialized: warmup=%d near_threshold=%.1f far_threshold=%.1f",
            self.WARMUP_SIZE,
            self.NEAR_OOD_THRESHOLD,
            self.FAR_OOD_THRESHOLD,
        )

    def _extract_features(
        self,
        tick: Dict[str, Any],
        prev_close: Optional[float] = None,
        symbol: str = "",
    ) -> Optional[np.ndarray]:
        """从tick提取5维特征向量

        Args:
            tick: 行情数据字典
            prev_close: 前一个close价格 (用于计算收益率)
            symbol: 交易对 (用于缓存prev_close)

        Returns:
            5维特征向量, 或None(数据不足)
        """
        try:
            market = tick.get("market")
            if market is None:
                return None

            close = float(getattr(market, "close", 0))
            if close <= 0:
                return None

            # 1. return_pct: 收益率百分比
            if prev_close is None or prev_close <= 0:
                prev_close = self._prev_close.get(symbol, close)
            return_pct = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
            # Clamp to prevent extreme values
            return_pct = max(-50.0, min(50.0, return_pct))

            # 2. volatility_pct: ATR/price × 100
            atr = float(tick.get("atr", 0.0))
            if atr <= 0:
                # 回退: 用high-low近似
                high = float(getattr(market, "high", close))
                low = float(getattr(market, "low", close))
                atr = max(high - low, abs(high - close), abs(low - close))
            volatility_pct = (atr / close * 100) if close > 0 else 0.0
            volatility_pct = max(0.0, min(20.0, volatility_pct))

            # 3. volume_log: log(volume + 1)
            volume = float(getattr(market, "volume", 0))
            volume = max(0.0, volume)
            volume_log = math.log1p(volume)

            # 4. regime_code: 0(sideways)/1(bull)/2(bear)
            regime_state = tick.get("regime")
            if isinstance(regime_state, dict):
                market_regime = regime_state.get("market_regime", "neutral")
            else:
                market_regime = "neutral"
            regime_map = {"bull": 1.0, "sideways": 0.0, "bear": 2.0, "neutral": 0.0}
            regime_code = regime_map.get(market_regime, 0.0)

            # 5. vpin_value: VPIN毒性值
            vpin_data = tick.get("vpin")
            if isinstance(vpin_data, dict):
                vpin_value = float(vpin_data.get("value", 0.0))
            else:
                vpin_value = 0.0
            vpin_value = max(0.0, min(1.0, vpin_value))

            # 更新prev_close缓存
            self._prev_close[symbol] = close

            return np.array(
                [return_pct, volatility_pct, volume_log, regime_code, vpin_value],
                dtype=np.float64,
            )

        except Exception as e:
            logger.debug("OOD feature extraction failed: %s", e)
            return None

    def update_baseline(
        self,
        tick: Dict[str, Any],
        prev_close: Optional[float] = None,
        symbol: str = "",
    ) -> None:
        """更新基线分布

        在每个tick调用, 收集特征向量到滚动窗口.
        每BASELINE_UPDATE_INTERVAL个tick重新计算基线统计量.
        """
        if not self.enabled:
            return

        features = self._extract_features(tick, prev_close, symbol)
        if features is None:
            return

        self._feature_buffer.append(features)
        self._tick_count += 1

        # 检查是否需要更新基线统计量
        # 1. 刚跨过warmup阈值时强制计算 (建立初始基线)
        # 2. 之后每BASELINE_UPDATE_INTERVAL个tick更新一次
        if self._tick_count >= self.WARMUP_SIZE:
            if self._last_baseline_update == 0:
                # 首次跨过warmup, 强制建立基线
                self._recompute_baseline()
            elif self._tick_count - self._last_baseline_update >= self.BASELINE_UPDATE_INTERVAL:
                self._recompute_baseline()

    def _recompute_baseline(self) -> None:
        """重新计算基线统计量 (mean, covariance, std)"""
        if len(self._feature_buffer) < self.WARMUP_SIZE:
            return

        try:
            data = np.array(list(self._feature_buffer))
            if data.shape[0] < self.N_FEATURES:
                return

            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0) + 1e-8  # 防止除零

            # 检测常数维度 (std ≈ 0)
            constant_dims = np.std(data, axis=0) < self.CONSTANT_DIM_THRESHOLD
            n_constant = int(np.sum(constant_dims))

            # 协方差矩阵 + 正则化
            cov = np.cov(data.T)
            if cov.shape == (self.N_FEATURES, self.N_FEATURES):
                # 对常数维度使用大正则化 (使其变化时贡献极小)
                reg = np.eye(self.N_FEATURES) * self.COV_REGULARIZATION
                for i in range(self.N_FEATURES):
                    if constant_dims[i]:
                        reg[i, i] = self.CONSTANT_DIM_REGULARIZATION
                cov += reg
                try:
                    self._cov_inv = np.linalg.inv(cov)
                    self._use_mahalanobis = True
                except np.linalg.LinAlgError:
                    # 奇异矩阵, 回退到Z-score
                    self._cov_inv = None
                    self._use_mahalanobis = False
                    logger.warning("OOD cov matrix singular, fallback to Z-score")
            else:
                # 1D情况, 回退到Z-score
                self._cov_inv = None
                self._use_mahalanobis = False

            self._last_baseline_update = self._tick_count

            if self._tick_count == self.WARMUP_SIZE:
                logger.info(
                    "Layer5 OOD baseline established: samples=%d mean=%s std=%s mahalanobis=%s constant_dims=%d",
                    len(self._feature_buffer),
                    np.round(self._mean, 3).tolist(),
                    np.round(self._std, 3).tolist(),
                    self._use_mahalanobis,
                    n_constant,
                )

        except Exception as e:
            logger.warning("OOD baseline recompute failed: %s", e)

    def check(
        self,
        tick: Dict[str, Any],
        prev_close: Optional[float] = None,
        symbol: str = "",
    ) -> OODResult:
        """检测当前市场状态是否OOD

        Returns:
            OODResult: 包含level, distance, multiplier
        """
        if not self.enabled:
            return OODResult(
                level=OODLevel.IN_DISTRIBUTION,
                distance=0.0,
                multiplier=self.IN_DIST_MULTIPLIER,
                reason="OOD detector disabled",
            )

        # Warmup期: 允许交易
        if self._tick_count < self.WARMUP_SIZE or self._mean is None:
            return OODResult(
                level=OODLevel.WARMUP,
                distance=0.0,
                multiplier=self.IN_DIST_MULTIPLIER,
                reason=f"warmup ({self._tick_count}/{self.WARMUP_SIZE})",
            )

        features = self._extract_features(tick, prev_close, symbol)
        if features is None:
            return OODResult(
                level=OODLevel.IN_DISTRIBUTION,
                distance=0.0,
                multiplier=self.IN_DIST_MULTIPLIER,
                reason="feature extraction failed, allow trade",
            )

        self._ood_check_count += 1

        # 计算距离
        if self._use_mahalanobis and self._cov_inv is not None:
            diff = features - self._mean
            # Mahalanobis distance = sqrt(diff^T × Σ^{-1} × diff)
            distance = float(math.sqrt(max(0, diff @ self._cov_inv @ diff)))
        else:
            # Z-score回退: max |z|
            z_scores = np.abs((features - self._mean) / self._std)
            distance = float(np.max(z_scores))

        # 判定等级
        if self._use_mahalanobis:
            near_thresh = self.NEAR_OOD_THRESHOLD
            far_thresh = self.FAR_OOD_THRESHOLD
        else:
            near_thresh = self.ZSCORE_NEAR_THRESHOLD
            far_thresh = self.ZSCORE_FAR_THRESHOLD

        if distance < near_thresh:
            level = OODLevel.IN_DISTRIBUTION
            multiplier = self.IN_DIST_MULTIPLIER
            reason = f"in_distribution (dist={distance:.2f})"
        elif distance < far_thresh:
            level = OODLevel.NEAR_OOD
            multiplier = self.NEAR_OOD_MULTIPLIER
            self._near_ood_count += 1
            reason = f"near_ood (dist={distance:.2f}), reduce position by 50%"
        else:
            level = OODLevel.FAR_OOD
            multiplier = self.FAR_OOD_MULTIPLIER
            self._far_ood_count += 1
            reason = f"far_ood (dist={distance:.2f}), reject trade"

        return OODResult(
            level=level,
            distance=distance,
            multiplier=multiplier,
            feature_vector=features,
            feature_names=self.FEATURE_NAMES,
            reason=reason,
        )

    def apply_to_decision(
        self,
        decision: Optional[Dict[str, Any]],
        tick: Dict[str, Any],
        prev_close: Optional[float] = None,
        symbol: str = "",
        agent_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """将OOD检测结果应用到交易决策

        Args:
            decision: 交易决策字典
            tick: 行情数据
            prev_close: 前一个close价格
            symbol: 交易对
            agent_id: Agent ID

        Returns:
            修改后的decision, 或None(拒绝交易)
        """
        if not self.enabled or decision is None:
            return decision

        # 只对开仓决策进行OOD检测
        action = decision.get("action", "neutral")
        if action not in ("long", "short"):
            return decision

        result = self.check(tick, prev_close, symbol)

        # 注入OOD信息到decision
        decision["ood_level"] = result.level.value
        decision["ood_distance"] = result.distance

        if result.level == OODLevel.FAR_OOD:
            # 远OOD: 拒绝交易
            if self._far_ood_count <= 5 or self._far_ood_count % 100 == 0:
                logger.info(
                    "Layer5 OOD REJECT #%d: agent=%s symbol=%s action=%s dist=%.2f reason=%s",
                    self._far_ood_count, agent_id[:12], symbol, action,
                    result.distance, result.reason,
                )
            return None

        if result.level == OODLevel.NEAR_OOD:
            # 近OOD: 缩仓50%
            orig_qty = decision.get("quantity", 0)
            if orig_qty and orig_qty > 0:
                decision["quantity"] = orig_qty * result.multiplier
                if self._near_ood_count <= 5 or self._near_ood_count % 100 == 0:
                    logger.info(
                        "Layer5 OOD NEAR #%d: agent=%s symbol=%s action=%s dist=%.2f qty=%.6f→%.6f",
                        self._near_ood_count, agent_id[:12], symbol, action,
                        result.distance, orig_qty, decision["quantity"],
                    )

        return decision

    def get_stats(self) -> Dict[str, Any]:
        """获取OOD检测器统计信息"""
        return {
            "enabled": self.enabled,
            "tick_count": self._tick_count,
            "warmup_size": self.WARMUP_SIZE,
            "is_warmed_up": self._tick_count >= self.WARMUP_SIZE,
            "use_mahalanobis": self._use_mahalanobis,
            "baseline_samples": len(self._feature_buffer),
            "ood_check_count": self._ood_check_count,
            "near_ood_count": self._near_ood_count,
            "far_ood_count": self._far_ood_count,
            "near_ood_rate": (
                self._near_ood_count / max(1, self._ood_check_count)
            ),
            "far_ood_rate": (
                self._far_ood_count / max(1, self._ood_check_count)
            ),
        }
