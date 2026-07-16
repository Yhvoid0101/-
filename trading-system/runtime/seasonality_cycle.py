# -*- coding: utf-8 -*-
"""
seasonality_cycle.py — 季节性周期策略 v1.0

Phase 8.6: 从抽象世界观升级为具体策略实现

算法：
  1. 比特币4年减半周期定位 (2012/2016/2020/2024 halving dates)
  2. MVRV近似 (价格/长期移动平均 → 周期顶底估值)
  3. ReserveRisk近似 (波动率倒数 → 风险偏好)
  4. 月末/季末/月初效应 (turn-of-month anomaly)
  5. 周末波动率溢价

信号类型：
  - CYCLE_ACCUMULATE: 减半前+低估值 → 逢低买入
  - CYCLE_DISTRIBUTE: 减半后高位+高估值 → 逢高卖出
  - CYCLE_HOLD: 周期中性
  - TURN_OF_MONTH: 月初效应买入
"""

from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


class SeasonalityCycleStrategy:
    """季节性周期策略

    比特币4年减半周期 + MVRV + 月末季末效应。
    """

    # 比特币减半日期 (UTC)
    HALVING_DATES = [
        datetime(2012, 11, 28, tzinfo=timezone.utc),
        datetime(2016, 7, 9, tzinfo=timezone.utc),
        datetime(2020, 5, 11, tzinfo=timezone.utc),
        datetime(2024, 4, 20, tzinfo=timezone.utc),
        # 预计下一次: 2028
        datetime(2028, 4, 1, tzinfo=timezone.utc),
    ]

    def __init__(
        self,
        pre_halving_buy_days: int = 500,
        post_halving_peak_min: int = 480,
        post_halving_peak_max: int = 550,
        peak_to_bottom_days: int = 365,
        mvrv_cycle_top: float = 3.5,
        mvrv_caution: float = 3.0,
        mvrv_undervalued: float = 1.5,
        reserve_risk_top_warning: float = 0.002,
        reserve_risk_bullish: float = 0.008,
        month_end_window_days: int = 5,
        turn_of_month_days: int = 4,
        weekend_vol_premium: float = 0.30,
        long_ma_window: int = 200,
        **kwargs: Any,
    ):
        self.pre_halving_buy_days = pre_halving_buy_days
        self.post_halving_peak_min = post_halving_peak_min
        self.post_halving_peak_max = post_halving_peak_max
        self.peak_to_bottom_days = peak_to_bottom_days
        self.mvrv_cycle_top = mvrv_cycle_top
        self.mvrv_caution = mvrv_caution
        self.mvrv_undervalued = mvrv_undervalued
        self.reserve_risk_top_warning = reserve_risk_top_warning
        self.reserve_risk_bullish = reserve_risk_bullish
        self.month_end_window_days = month_end_window_days
        self.turn_of_month_days = turn_of_month_days
        self.weekend_vol_premium = weekend_vol_premium
        self.long_ma_window = long_ma_window

        self._prices: deque = deque(maxlen=long_ma_window * 2)
        self._volumes: deque = deque(maxlen=long_ma_window * 2)
        self._current_date: Optional[datetime] = None
        self._mvrv: float = 1.0
        self._reserve_risk: float = 0.005
        self._current_price: float = 0.0

    def update_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收价格"""
        self._prices.append(price)
        self._current_price = price
        if timestamp is not None:
            self._current_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            self._current_date = datetime.now(timezone.utc)
        self._update_mvrv_approx()

    def update_volume(self, volume: float) -> None:
        """Feeder: 接收成交量"""
        self._volumes.append(volume)

    def update_onchain_metrics(
        self, mvrv: Optional[float] = None, reserve_risk: Optional[float] = None
    ) -> None:
        """Feeder: 接收链上指标 (可选，无数据时用近似值)"""
        if mvrv is not None:
            self._mvrv = mvrv
        if reserve_risk is not None:
            self._reserve_risk = reserve_risk

    def set_date(self, date: Any) -> None:
        """Feeder: 设置当前日期"""
        if isinstance(date, datetime):
            self._current_date = date if date.tzinfo else date.replace(tzinfo=timezone.utc)
        elif isinstance(date, str):
            try:
                self._current_date = datetime.fromisoformat(date)
                if self._current_date.tzinfo is None:
                    self._current_date = self._current_date.replace(tzinfo=timezone.utc)
            except Exception:
                pass

    def _update_mvrv_approx(self) -> None:
        """用价格/长期均线近似MVRV"""
        if len(self._prices) < self.long_ma_window:
            self._mvrv = 1.0
            return
        long_ma = float(np.mean(list(self._prices)[-self.long_ma_window:]))
        if long_ma > 0:
            self._mvrv = self._current_price / long_ma
        # 近似ReserveRisk: 波动率的倒数 (低波动=高信心=低风险)
        if len(self._prices) >= 30:
            rets = np.diff(list(self._prices)[-30:])
            vol = float(np.std(rets) / np.mean(np.abs(rets))) if np.mean(np.abs(rets)) > 0 else 0.02
            self._reserve_risk = 1.0 / (1.0 + 100.0 * vol)

    def _get_halving_position(self) -> Tuple[str, int]:
        """获取当前在减半周期中的位置

        Returns:
            (phase, days_since_last_halving)
            phase: "pre_halving" | "post_halving_peak" | "bear" | "accumulation"
        """
        if self._current_date is None:
            return ("accumulation", 0)

        # 找到最近的减半
        last_halving = self.HALVING_DATES[0]
        next_halving = self.HALVING_DATES[-1]
        for i, h in enumerate(self.HALVING_DATES):
            if h <= self._current_date:
                last_halving = h
                if i + 1 < len(self.HALVING_DATES):
                    next_halving = self.HALVING_DATES[i + 1]
            else:
                next_halving = h
                break

        days_since = (self._current_date - last_halving).days
        days_to_next = (next_halving - self._current_date).days

        if days_to_next < self.pre_halving_buy_days and days_to_next > 0:
            return ("pre_halving", days_to_next)
        elif self.post_halving_peak_min <= days_since <= self.post_halving_peak_max:
            return ("post_halving_peak", days_since)
        elif days_since > self.post_halving_peak_max and days_since < self.peak_to_bottom_days + self.post_halving_peak_max:
            return ("bear", days_since)
        else:
            return ("accumulation", days_since)

    def _check_turn_of_month(self) -> bool:
        """检查是否在月初效应窗口"""
        if self._current_date is None:
            return False
        day = self._current_date.day
        # 月初 + 月末窗口
        return day <= self.turn_of_month_days or day >= 28 - self.month_end_window_days + 1

    def _is_weekend(self) -> bool:
        """是否周末"""
        if self._current_date is None:
            return False
        return self._current_date.weekday() >= 5

    def analyze(self) -> Dict[str, Any]:
        """主分析函数"""
        result: Dict[str, Any] = {
            "signal_type": "CYCLE_HOLD",
            "direction": "flat",
            "confidence": 0.0,
            "strength": 0.0,
            "description": "",
        }

        if len(self._prices) < 30:
            result["description"] = "数据不足"
            return result

        phase, days = self._get_halving_position()
        turn_of_month = self._check_turn_of_month()
        is_weekend = self._is_weekend()

        signals: List[Tuple[float, str, str]] = []

        # 减半前买入信号
        if phase == "pre_halving":
            strength = 1.0 - (days / self.pre_halving_buy_days)  # 越接近减半越强
            signals.append((strength, "long", f"减半前{days}天"))

        # 减半后高位减仓
        if phase == "post_halving_peak":
            signals.append((0.5, "short", f"减半后{days}天(高位区间)"))

        # 熊市阶段逢低买入
        if phase == "bear":
            signals.append((0.3, "long", f"熊市{days}天(逢低布局)"))

        # MVRV估值信号
        if self._mvrv < self.mvrv_undervalued:
            signals.append((0.7, "long", f"MVRV低估={self._mvrv:.2f}"))
        elif self._mvrv > self.mvrv_cycle_top:
            signals.append((0.8, "short", f"MVRV高估={self._mvrv:.2f}"))
        elif self._mvrv > self.mvrv_caution:
            signals.append((0.4, "short", f"MVRV谨慎={self._mvrv:.2f}"))

        # ReserveRisk信号
        if self._reserve_risk > self.reserve_risk_bullish:
            signals.append((0.4, "long", f"ReserveRisk看涨={self._reserve_risk:.4f}"))
        elif self._reserve_risk < self.reserve_risk_top_warning:
            signals.append((0.6, "short", f"ReserveRisk顶部警告={self._reserve_risk:.4f}"))

        # 月初效应
        if turn_of_month:
            signals.append((0.3, "long", "月初/月末效应"))

        # 周末溢价
        if is_weekend:
            signals.append((0.2, "long", f"周末波动率溢价{self.weekend_vol_premium:.0%}"))

        if not signals:
            result["description"] = f"周期中性 phase={phase} mvrv={self._mvrv:.2f}"
            return result

        # 加权融合
        long_score = sum(s for s, d, _ in signals if d == "long")
        short_score = sum(s for s, d, _ in signals if d == "short")

        if long_score > short_score + 0.1:
            net = long_score - short_score
            result.update({
                "signal_type": "CYCLE_ACCUMULATE" if phase == "pre_halving" else "CYCLE_LONG",
                "direction": "long",
                "confidence": min(1.0, net),
                "strength": net,
                "description": "; ".join(desc for _, _, desc in signals if True),
            })
        elif short_score > long_score + 0.1:
            net = short_score - long_score
            result.update({
                "signal_type": "CYCLE_DISTRIBUTE",
                "direction": "short",
                "confidence": min(1.0, net),
                "strength": net,
                "description": "; ".join(desc for _, _, desc in signals),
            })
        else:
            result["description"] = f"多空平衡 long={long_score:.2f} short={short_score:.2f}"

        return result
