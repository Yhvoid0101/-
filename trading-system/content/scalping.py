# -*- coding: utf-8 -*-
"""
scalping.py — 高频刮头皮策略 v1.0

Phase 8.6: 从抽象世界观升级为具体策略实现

算法：
  1. 短期ROC (Rate of Change) - 极短窗口价格动量
  2. Williams %R - 超买超卖检测
  3. 价差估计 - 从high/low推算bid-ask spread
  4. 成交量冲击 - 短期成交量异常检测
  5. 综合信号融合 - 4分量加权

信号类型：
  - SCALP_LONG: 短期超卖+正动量 → 买入
  - SCALP_SHORT: 短期超买+负动量 → 卖出
  - SCALP_FLAT: 信号不足
"""

from __future__ import annotations
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np


class ScalpingStrategy:
    """高频刮头皮策略

    极短时间内的微小利差交易，依赖短期动量反转。
    """

    def __init__(
        self,
        max_trades_per_day: int = 200,
        roc_window: int = 5,
        willr_window: int = 14,
        spread_threshold: float = 0.001,
        volume_spiake_mult: float = 2.0,
        **kwargs: Any,
    ):
        self.max_trades_per_day = max_trades_per_day
        self.roc_window = roc_window
        self.willr_window = willr_window
        self.spread_threshold = spread_threshold
        self.volume_spike_mult = volume_spiake_mult

        self._prices: deque = deque(maxlen=max(roc_window, willr_window) * 3)
        self._highs: deque = deque(maxlen=willr_window * 3)
        self._lows: deque = deque(maxlen=willr_window * 3)
        self._volumes: deque = deque(maxlen=roc_window * 3)
        self._timestamps: deque = deque(maxlen=roc_window * 3)
        self._trade_count: int = 0

    def update_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收价格更新"""
        self._prices.append(price)
        self._timestamps.append(timestamp or time.time())

    def update_high_low(self, high: float, low: float) -> None:
        """Feeder: 接收高低价"""
        self._highs.append(high)
        self._lows.append(low)

    def update_volume(self, volume: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收成交量"""
        self._volumes.append(volume)

    def update_price_volume(
        self, price: float, volume: float, timestamp: Optional[float] = None
    ) -> None:
        """Feeder: 同时接收价格和成交量"""
        self.update_price(price, timestamp)
        self.update_volume(volume, timestamp)

    def _compute_roc(self) -> float:
        """短期ROC"""
        if len(self._prices) < self.roc_window + 1:
            return 0.0
        current = self._prices[-1]
        past = self._prices[-(self.roc_window + 1)]
        if past == 0:
            return 0.0
        return (current - past) / past

    def _compute_willr(self) -> float:
        """Williams %R (-100 to 0)"""
        if len(self._highs) < self.willr_window or len(self._lows) < self.willr_window:
            return -50.0
        recent_high = max(list(self._highs)[-self.willr_window:])
        recent_low = min(list(self._lows)[-self.willr_window:])
        current = self._prices[-1] if self._prices else 0.0
        if recent_high == recent_low:
            return -50.0
        return -100.0 * (recent_high - current) / (recent_high - recent_low)

    def _estimate_spread(self) -> float:
        """从high/low估计价差"""
        if len(self._highs) < 2 or len(self._lows) < 2 or not self._prices:
            return 0.0
        spread = (self._highs[-1] - self._lows[-1]) / self._prices[-1]
        return abs(spread)

    def _compute_volume_spike(self) -> float:
        """成交量异常倍数"""
        if len(self._volumes) < self.roc_window + 1:
            return 1.0
        recent = list(self._volumes)[-self.roc_window:]
        avg_vol = np.mean(recent[:-1]) if len(recent) > 1 else recent[0]
        if avg_vol == 0:
            return 1.0
        return recent[-1] / avg_vol

    def analyze(self) -> Dict[str, Any]:
        """主分析函数 — 返回信号dict"""
        result: Dict[str, Any] = {
            "signal_type": "SCALP_FLAT",
            "direction": "flat",
            "confidence": 0.0,
            "strength": 0.0,
            "description": "",
        }

        if len(self._prices) < self.roc_window + 1:
            result["description"] = "数据不足"
            return result

        roc = self._compute_roc()
        willr = self._compute_willr()
        spread = self._estimate_spread()
        vol_spike = self._compute_volume_spike()

        # 交易频率限制
        if self._trade_count >= self.max_trades_per_day:
            result["description"] = f"达到每日交易上限 {self.max_trades_per_day}"
            return result

        # 价差过小不值得刮头皮
        if spread < self.spread_threshold:
            result["description"] = f"价差过小 {spread:.6f}"
            return result

        # Williams %R 超卖 (-80 to -100) + 正ROC → 买入
        if willr < -80 and roc > 0:
            confidence = min(1.0, abs(willr + 50) / 50.0 * (1 + vol_spike * 0.1))
            result.update({
                "signal_type": "SCALP_LONG",
                "direction": "long",
                "confidence": confidence,
                "strength": abs(roc),
                "description": f"超卖反弹 willr={willr:.1f} roc={roc:.4f} vol={vol_spike:.1f}x",
            })
            self._trade_count += 1
        # Williams %R 超买 (0 to -20) + 负ROC → 卖出
        elif willr > -20 and roc < 0:
            confidence = min(1.0, abs(willr + 50) / 50.0 * (1 + vol_spike * 0.1))
            result.update({
                "signal_type": "SCALP_SHORT",
                "direction": "short",
                "confidence": confidence,
                "strength": abs(roc),
                "description": f"超买回落 willr={willr:.1f} roc={roc:.4f} vol={vol_spike:.1f}x",
            })
            self._trade_count += 1
        else:
            result["description"] = f"无信号 willr={willr:.1f} roc={roc:.4f}"

        return result

    def reset_daily(self) -> None:
        """重置每日交易计数"""
        self._trade_count = 0
