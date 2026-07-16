# -*- coding: utf-8 -*-
"""
microstructure_alpha_trading.py — 订单簿微观结构Alpha策略 v1.0

Phase 8.6: 从抽象世界观升级为具体策略实现

算法:
  1. OBI (Order Book Imbalance) - Cont 2014深度加权
     从K线high/low近似bid/ask深度
  2. Hawkes爆发检测 - 自激发强度λ(t)
     基于成交量到达序列检测交易爆发
  3. Kyle lambda价格冲击 - 在线滚动估计
     λ = Δprice / Δvolume
  4. VPIN毒性过滤 - Easley Lopez de Prado
     从成交量买卖分类估计毒性
  5. 队列位置估计 - Foucault 2005
     从价格反转频率估计队列消散
  6. 综合Alpha融合 - 5分量加权

信号类型:
  - MICRO_LONG / MICRO_SHORT / MICRO_FLAT / MICRO_TOXIC
"""

from __future__ import annotations
import math
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np


class MicrostructureAlphaStrategy:
    """订单簿微观结构Alpha策略

    用K线数据近似订单簿指标, 捕捉微观结构alpha。
    """

    def __init__(
        self,
        obi_n_levels: int = 5,
        obi_decay: float = 0.8,
        obi_strong_threshold: float = 0.5,
        obi_medium_threshold: float = 0.3,
        obi_weak_threshold: float = 0.1,
        kyle_window: int = 50,
        kyle_min_history: int = 30,
        kyle_impact_max: float = 0.05,
        hawkes_baseline: float = 1.0,
        hawkes_alpha: float = 0.5,
        hawkes_beta: float = 1.0,
        hawkes_burst_threshold: float = 2.5,
        hawkes_memory: int = 30,
        queue_arrival_window: int = 20,
        queue_time_max: int = 50,
        queue_pressure_threshold: float = 0.3,
        queue_front_threshold: float = 0.7,
        resilience_window: int = 30,
        resilience_shock_threshold: float = 0.5,
        resilience_recovery_time: int = 10,
        resilience_healthy: float = 0.7,
        resilience_breakdown: float = 0.3,
        vpin_window: int = 50,
        vpin_toxic_threshold: float = 0.7,
        vpin_warning_threshold: float = 0.5,
        pin_high_threshold: float = 0.3,
        adverse_cost_max_bps: float = 5.0,
        w_obi: float = 0.25,
        w_hawkes: float = 0.15,
        **kwargs: Any,
    ):
        self.obi_n_levels = obi_n_levels
        self.obi_decay = obi_decay
        self.obi_strong_threshold = obi_strong_threshold
        self.obi_medium_threshold = obi_medium_threshold
        self.obi_weak_threshold = obi_weak_threshold
        self.kyle_window = kyle_window
        self.kyle_min_history = kyle_min_history
        self.kyle_impact_max = kyle_impact_max
        self.hawkes_baseline = hawkes_baseline
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        self.hawkes_burst_threshold = hawkes_burst_threshold
        self.hawkes_memory = hawkes_memory
        self.queue_arrival_window = queue_arrival_window
        self.queue_time_max = queue_time_max
        self.queue_pressure_threshold = queue_pressure_threshold
        self.queue_front_threshold = queue_front_threshold
        self.resilience_window = resilience_window
        self.resilience_shock_threshold = resilience_shock_threshold
        self.resilience_recovery_time = resilience_recovery_time
        self.resilience_healthy = resilience_healthy
        self.resilience_breakdown = resilience_breakdown
        self.vpin_window = vpin_window
        self.vpin_toxic_threshold = vpin_toxic_threshold
        self.vpin_warning_threshold = vpin_warning_threshold
        self.pin_high_threshold = pin_high_threshold
        self.adverse_cost_max_bps = adverse_cost_max_bps
        self.w_obi = w_obi
        self.w_hawkes = w_hawkes

        # 从key_params提取剩余权重
        self.w_kyle = kwargs.get("w_kyle", 0.20)
        self.w_vpin = kwargs.get("w_vpin", 0.20)
        self.w_queue = kwargs.get("w_queue", 0.20)

        self._prices: deque = deque(maxlen=max(kyle_window, vpin_window) * 2)
        self._highs: deque = deque(maxlen=kyle_window * 2)
        self._lows: deque = deque(maxlen=kyle_window * 2)
        self._volumes: deque = deque(maxlen=max(kyle_window, vpin_window) * 2)
        self._timestamps: deque = deque(maxlen=hawkes_memory * 2)

        # Hawkes强度
        self._hawkes_intensity: float = hawkes_baseline

        # Kyle lambda
        self._kyle_lambda: float = 0.0

        # VPIN
        self._vpin: float = 0.0

        # OBI
        self._obi: float = 0.0

    def update_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收价格"""
        self._prices.append(price)
        self._timestamps.append(timestamp or time.time())
        self._update_hawkes(timestamp)

    def update_high_low(self, high: float, low: float) -> None:
        """Feeder: 接收高低价 (用于OBI近似)"""
        self._highs.append(high)
        self._lows.append(low)

    def update_volume(self, volume: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收成交量"""
        self._volumes.append(volume)

    def update_price_volume(
        self, price: float, volume: float, timestamp: Optional[float] = None
    ) -> None:
        """Feeder: 价格+成交量"""
        self.update_price(price, timestamp)
        self.update_volume(volume, timestamp)

    def update_book_depth(self, bids: Any = None, asks: Any = None) -> None:
        """Feeder: 接收订单簿深度 (可选, 无数据时用K线近似)"""
        # 如果有真实订单簿数据, 可以在这里更新

    def update_market_data(self, **kwargs: Any) -> None:
        """Feeder: 通用市场数据更新"""
        if "price" in kwargs:
            self.update_price(kwargs["price"])
        if "high" in kwargs and "low" in kwargs:
            self.update_high_low(kwargs["high"], kwargs["low"])
        if "volume" in kwargs:
            self.update_volume(kwargs["volume"])

    def _approximate_obi(self) -> float:
        """从K线近似OBI (Order Book Imbalance)

        用 (close-low)/(high-low) 近似买方力量比例
        OBI = (buy_pressure - 0.5) * 2  → [-1, 1]
        """
        if len(self._highs) < 2 or len(self._lows) < 2 or not self._prices:
            return 0.0
        recent_n = min(self.obi_n_levels, len(self._highs))
        obi_sum = 0.0
        weight_sum = 0.0
        for i in range(-recent_n, 0):
            h = self._highs[i]
            l = self._lows[i]
            c = self._prices[i] if abs(i) <= len(self._prices) else (h + l) / 2
            weight = self.obi_decay ** (recent_n + i)
            if h > l:
                buy_pressure = (c - l) / (h - l)
                obi_sum += (buy_pressure - 0.5) * 2 * weight
                weight_sum += weight
        return float(obi_sum / weight_sum) if weight_sum > 0 else 0.0

    def _update_hawkes(self, timestamp: Optional[float] = None) -> None:
        """更新Hawkes强度"""
        # 时间衰减
        self._hawkes_intensity *= math.exp(-self.hawkes_beta * 0.1)  # 假设0.1s间隔
        self._hawkes_intensity += self.hawkes_baseline * 0.1
        # 自激发: 价格变动触发
        if len(self._prices) >= 2:
            price_change = abs(self._prices[-1] - self._prices[-2]) / self._prices[-2] if self._prices[-2] > 0 else 0
            if price_change > 0.001:  # 显著价格变动
                self._hawkes_intensity += self.hawkes_alpha * price_change * 100

    def _compute_kyle_lambda(self) -> float:
        """计算Kyle lambda (价格冲击系数)"""
        if len(self._prices) < self.kyle_min_history or len(self._volumes) < self.kyle_min_history:
            return 0.0
        n = min(self.kyle_window, len(self._prices) - 1, len(self._volumes) - 1)
        price_changes = np.diff(list(self._prices)[-n - 1:])
        volumes = np.array(list(self._volumes)[-n:], dtype=float)
        # λ = |Δprice| / volume
        valid = volumes > 0
        if not np.any(valid):
            return 0.0
        lambdas = np.abs(price_changes[valid][:len(volumes[valid])]) / volumes[valid][:len(price_changes[valid])]
        if len(lambdas) == 0:
            return 0.0
        return float(np.mean(lambdas))

    def _compute_vpin(self) -> float:
        """计算VPIN (Volume-synchronized Probability of Informed Trading)"""
        if len(self._volumes) < self.vpin_window or len(self._prices) < 2:
            return 0.0
        # 近似: 用价格方向分类买卖单
        recent_vols = list(self._volumes)[-self.vpin_window:]
        recent_prices = list(self._prices)[-self.vpin_window - 1:]
        if len(recent_prices) < 2:
            return 0.0
        price_changes = np.diff(recent_prices)
        # 价格上涨 → 买单, 下跌 → 卖单
        buy_vols = []
        sell_vols = []
        for i, change in enumerate(price_changes[:len(recent_vols)]):
            if change > 0:
                buy_vols.append(recent_vols[i])
                sell_vols.append(0.0)
            elif change < 0:
                buy_vols.append(0.0)
                sell_vols.append(recent_vols[i])
            else:
                # 价格不变, 50/50分类
                buy_vols.append(recent_vols[i] * 0.5)
                sell_vols.append(recent_vols[i] * 0.5)
        total_vol = sum(buy_vols) + sum(sell_vols)
        if total_vol == 0:
            return 0.0
        order_imbalance = sum(abs(b - s) for b, s in zip(buy_vols, sell_vols))
        return float(order_imbalance / total_vol)

    def _compute_queue_pressure(self) -> float:
        """估计队列压力 (价格反转频率)"""
        if len(self._prices) < self.queue_arrival_window:
            return 0.0
        recent = list(self._prices)[-self.queue_arrival_window:]
        reversals = 0
        for i in range(1, len(recent) - 1):
            if (recent[i] > recent[i - 1] and recent[i] > recent[i + 1]) or \
               (recent[i] < recent[i - 1] and recent[i] < recent[i + 1]):
                reversals += 1
        return float(reversals / (len(recent) - 2)) if len(recent) > 2 else 0.0

    def _compute_resilience(self) -> float:
        """订单簿韧性 (冲击恢复率)"""
        if len(self._prices) < self.resilience_window:
            return self.resilience_healthy
        recent = list(self._prices)[-self.resilience_window:]
        # 波动率衰减率作为韧性近似
        half = len(recent) // 2
        vol_first = float(np.std(np.diff(recent[:half]))) if half > 1 else 0.0
        vol_second = float(np.std(np.diff(recent[half:]))) if half > 1 else 0.0
        if vol_first == 0:
            return self.resilience_healthy
        recovery = 1.0 - min(1.0, vol_second / vol_first)
        return float(recovery)

    def analyze(self) -> Dict[str, Any]:
        """主分析函数"""
        result: Dict[str, Any] = {
            "signal_type": "MICRO_FLAT",
            "direction": "flat",
            "confidence": 0.0,
            "strength": 0.0,
            "description": "",
        }

        if len(self._prices) < self.kyle_min_history:
            result["description"] = f"数据不足 {len(self._prices)}/{self.kyle_min_history}"
            return result

        # 计算5个微观结构指标
        obi = self._approximate_obi()
        hawkes_burst = self._hawkes_intensity / self.hawkes_baseline  # 爆发倍数
        kyle = self._compute_kyle_lambda()
        vpin = self._compute_vpin()
        queue_pressure = self._compute_queue_pressure()
        resilience = self._compute_resilience()

        # VPIN毒性过滤
        if vpin > self.vpin_toxic_threshold:
            result.update({
                "signal_type": "MICRO_TOXIC",
                "direction": "flat",
                "confidence": 0.9,
                "strength": 0.9,
                "description": f"VPIN毒性={vpin:.3f}>{self.vpin_toxic_threshold}, 停止交易",
            })
            return result

        # 综合Alpha融合
        # OBI: 正值=买方力量 → long
        obi_signal = float(np.clip(obi, -1, 1))
        # Hawkes: 爆发>阈值 → 跟随方向 (用OBI方向)
        hawkes_signal = obi_signal * min(1.0, hawkes_burst / self.hawkes_burst_threshold) if hawkes_burst > 1.0 else 0.0
        # Kyle: 高lambda + OBI一致 → 增强信号
        kyle_signal = obi_signal * min(1.0, kyle / self.kyle_impact_max) if kyle > 0 else 0.0
        # VPIN: 中等毒性 → 减弱信号
        vpin_signal = -obi_signal * max(0, (vpin - self.vpin_warning_threshold) / (self.vpin_toxic_threshold - self.vpin_warning_threshold))
        # Queue: 高队列压力 + OBI一致 → 增强信号
        queue_signal = obi_signal * min(1.0, queue_pressure / self.queue_pressure_threshold) if queue_pressure > self.queue_pressure_threshold else 0.0

        alpha = (
            self.w_obi * obi_signal
            + self.w_hawkes * hawkes_signal
            + self.w_kyle * kyle_signal
            + self.w_vpin * vpin_signal
            + self.w_queue * queue_signal
        )

        # 韧性过滤: 韧性低 → 减弱信号
        if resilience < self.resilience_breakdown:
            alpha *= 0.3
            result["description"] = f"订单簿韧性低={resilience:.2f}"

        # 逆向选择成本检查
        adverse_cost = kyle * 10000  # 转为bps
        if adverse_cost > self.adverse_cost_max_bps:
            alpha *= 0.5
            result["description"] = f"逆向选择成本高={adverse_cost:.1f}bps"

        # 生成信号
        if alpha > self.obi_weak_threshold:
            result.update({
                "signal_type": "MICRO_LONG",
                "direction": "long",
                "confidence": min(1.0, abs(alpha)),
                "strength": abs(alpha),
                "description": f"微观Alpha做多 alpha={alpha:.3f} obi={obi:.2f} hawkes={hawkes_burst:.1f}x vpin={vpin:.3f}",
            })
        elif alpha < -self.obi_weak_threshold:
            result.update({
                "signal_type": "MICRO_SHORT",
                "direction": "short",
                "confidence": min(1.0, abs(alpha)),
                "strength": abs(alpha),
                "description": f"微观Alpha做空 alpha={alpha:.3f} obi={obi:.2f} hawkes={hawkes_burst:.1f}x vpin={vpin:.3f}",
            })
        else:
            desc = result.get("description", "")
            result["description"] = f"无信号 alpha={alpha:.3f} obi={obi:.2f} vpin={vpin:.3f}" + (f" | {desc}" if desc else "")

        return result
