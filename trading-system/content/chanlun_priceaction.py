# -*- coding: utf-8 -*-
"""
缠论+裸K价格行为模块 (ChanLun + Price Action)

精选注入3大客观性强、参数少、过拟合风险低的技术分析信号：

1. 缠论背驰量化 (ChanLun Divergence)
   来源：缠中说禅《缠论》核心概念
   原理：趋势力度衰减，价格创新高/低但MACD柱状图面积减小
   量化：用MACD柱状图面积对比替代主观判断，客观可验证
   信号：顶背驰=做空，底背驰=做多
   参数：1个（波数窗口，默认5）

2. 裸K形态识别 (Price Action Patterns)
   来源：Al Brooks《Price Action》系列
   原理：直接读K线，无指标滞后
   形态：Pin Bar（锤子/射击之星）、Engulfing（吞没）
   量化：影线/实体比例，客观可计算
   参数：1个（Pin Bar影线比例，默认2.0）

3. 支撑阻力位 (Support/Resistance)
   来源：经典技术分析+Wyckoff理论
   原理：价格在关键水平有记忆效应
   量化：近期N根K线高低点聚类
   信号：支撑位做多，阻力位做空
   参数：1个（回溯窗口，默认20）

设计原则（反过拟合）：
  - 总共仅3个参数，远低于过拟合警戒线（10+参数）
  - 每个信号有明确经济逻辑，非数据挖掘
  - 所有信号必须通过PBO<0.3 + 蒙特卡洛p<0.05验证
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("hermes.chanlun_priceaction")


# ============================================================================
# 1. 缠论背驰量化
# ============================================================================


class ChanLunDivergence:
    """缠论背驰量化检测

    缠论核心概念：
      - 走势必完美：任何走势必然包含至少一个中枢
      - 背驰：趋势力度衰减，价格创新高/低但力度指标减弱
      - 背驰是趋势反转的必要条件（非充分）

    量化方法（客观化）：
      用MACD柱状图面积（积分）衡量趋势力度。
      - 顶背驰：价格创新高，但MACD柱状图面积 < 前一波面积 × 衰减阈值
      - 底背驰：价格创新低，但MACD柱状图面积 < 前一波面积 × 衰减阈值

    参数：仅1个（wave_window，背驰判断的波数窗口）
    """

    def __init__(
        self,
        wave_window: int = 5,        # 判断新高/新低的回溯窗口
        decay_threshold: float = 0.8,  # 力度衰减阈值（面积比<0.8=背驰）
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        """
        Args:
            wave_window: 判断价格新高/新低的回溯K线数
            decay_threshold: 力度衰减阈值（当前面积/前波面积<此值=背驰）
            fast_period: MACD快线周期
            slow_period: MACD慢线周期
            signal_period: MACD信号线周期
        """
        self.wave_window = wave_window
        self.decay_threshold = decay_threshold
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # 历史数据缓存
        self._closes: deque = deque(maxlen=200)
        self._macd_hist: deque = deque(maxlen=200)
        self._ema_fast = 0.0
        self._ema_slow = 0.0
        self._ema_signal = 0.0
        self._initialized = False

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新K线数据并检测背驰

        Args:
            kline: {"open", "high", "low", "close", "volume"}

        Returns:
            {
                "signal": "top_divergence"/"bottom_divergence"/"none",
                "direction": "short"/"long"/"neutral",
                "strength": float,  # 0-1
                "macd_hist": float,
                "area_ratio": float,  # 当前面积/前波面积
            }
        """
        close = kline.get("close", 0)
        if close <= 0:
            return self._no_signal()

        self._closes.append(close)

        # 计算MACD
        if not self._initialized:
            self._ema_fast = close
            self._ema_slow = close
            self._initialized = True
        else:
            alpha_fast = 2.0 / (self.fast_period + 1)
            alpha_slow = 2.0 / (self.slow_period + 1)
            alpha_signal = 2.0 / (self.signal_period + 1)
            self._ema_fast = alpha_fast * close + (1 - alpha_fast) * self._ema_fast
            self._ema_slow = alpha_slow * close + (1 - alpha_slow) * self._ema_slow
            macd_line = self._ema_fast - self._ema_slow
            self._ema_signal = alpha_signal * macd_line + (1 - alpha_signal) * self._ema_signal
            macd_hist = macd_line - self._ema_signal
            self._macd_hist.append(macd_hist)

        # 数据不足，无法判断
        if len(self._closes) < self.wave_window * 2 + 5:
            return self._no_signal()

        return self._detect_divergence()

    def _detect_divergence(self) -> Dict[str, Any]:
        """检测背驰"""
        closes = list(self._closes)
        hist = list(self._macd_hist)

        if len(hist) < self.wave_window * 2:
            return self._no_signal()

        # 当前波和前一波的窗口
        current_window = closes[-self.wave_window:]
        prev_window = closes[-self.wave_window * 2:-self.wave_window]

        current_hist = hist[-self.wave_window:]
        prev_hist = hist[-self.wave_window * 2:-self.wave_window]

        # 计算MACD柱状图面积（绝对值积分）
        current_area = sum(abs(h) for h in current_hist)
        prev_area = sum(abs(h) for h in prev_hist)

        if prev_area < 1e-10:
            return self._no_signal()

        area_ratio = current_area / prev_area

        # 顶背驰检测：价格创新高，但MACD面积减小
        current_high = max(current_window)
        prev_high = max(prev_window)
        if current_high > prev_high and area_ratio < self.decay_threshold:
            strength = min(1.0, (1.0 - area_ratio) * 2)
            return {
                "signal": "top_divergence",
                "direction": "short",
                "strength": strength,
                "macd_hist": current_hist[-1],
                "area_ratio": area_ratio,
            }

        # 底背驰检测：价格创新低，但MACD面积减小
        current_low = min(current_window)
        prev_low = min(prev_window)
        if current_low < prev_low and area_ratio < self.decay_threshold:
            strength = min(1.0, (1.0 - area_ratio) * 2)
            return {
                "signal": "bottom_divergence",
                "direction": "long",
                "strength": strength,
                "macd_hist": current_hist[-1],
                "area_ratio": area_ratio,
            }

        return self._no_signal()

    def _no_signal(self) -> Dict[str, Any]:
        return {
            "signal": "none",
            "direction": "neutral",
            "strength": 0.0,
            "macd_hist": 0.0,
            "area_ratio": 1.0,
        }


# ============================================================================
# 2. 裸K形态识别
# ============================================================================


class PriceActionPatterns:
    """裸K形态识别

    来源：Al Brooks《Price Action》系列

    识别3种高概率形态：
      1. Pin Bar（锤子线/射击之星）：长影线+小实体，反转信号
      2. Bullish/Bearish Engulfing（看涨/看跌吞没）：强反转信号
      3. Inside Bar（内包线）：整理形态，突破方向信号

    参数：仅1个（pin_bar_shadow_ratio，Pin Bar影线/实体比例）
    """

    def __init__(
        self,
        pin_bar_shadow_ratio: float = 2.0,  # 影线长度≥实体2倍=Pin Bar
        min_body_pct: float = 0.0005,       # 最小实体占比（避免十字星误判）
    ):
        """
        Args:
            pin_bar_shadow_ratio: Pin Bar影线/实体比例阈值
            min_body_pct: 最小实体占比（价格×此值），低于此值不判断
        """
        self.pin_bar_shadow_ratio = pin_bar_shadow_ratio
        self.min_body_pct = min_body_pct
        self._klines: deque = deque(maxlen=10)

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新K线并识别形态

        Returns:
            {
                "pattern": "pin_bar_bull"/"pin_bar_bear"/"engulfing_bull"/"engulfing_bear"/"inside_bar"/"none",
                "direction": "long"/"short"/"neutral",
                "strength": float,  # 0-1
            }
        """
        self._klines.append(kline)

        if len(self._klines) < 2:
            return {"pattern": "none", "direction": "neutral", "strength": 0.0}

        current = self._klines[-1]
        prev = self._klines[-2]

        # 1. Pin Bar检测
        pin_result = self._detect_pin_bar(current)
        if pin_result["pattern"] != "none":
            return pin_result

        # 2. Engulfing检测（需要前一根K线）
        engulf_result = self._detect_engulfing(current, prev)
        if engulf_result["pattern"] != "none":
            return engulf_result

        # 3. Inside Bar检测
        inside_result = self._detect_inside_bar(current, prev)
        if inside_result["pattern"] != "none":
            return inside_result

        return {"pattern": "none", "direction": "neutral", "strength": 0.0}

    def _detect_pin_bar(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """检测Pin Bar（锤子线/射击之星）

        Pin Bar特征：
          - 实体小（占总长度<30%）
          - 一端有长影线（≥实体×shadow_ratio）
          - 另一端影线短（<实体）

        看涨Pin Bar（锤子线）：长下影线，下影线≥实体×2，上影线<实体
        看跌Pin Bar（射击之星）：长上影线，上影线≥实体×2，下影线<实体
        """
        open_p = kline.get("open", 0)
        close = kline.get("close", 0)
        high = kline.get("high", close)
        low = kline.get("low", close)

        body = abs(close - open_p)
        total_range = high - low

        if total_range <= 0 or body < self.min_body_pct * close:
            return {"pattern": "none", "direction": "neutral", "strength": 0.0}

        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low

        # 看涨Pin Bar（锤子线）：长下影线
        if lower_shadow >= body * self.pin_bar_shadow_ratio and upper_shadow < body:
            strength = min(1.0, lower_shadow / (body * self.pin_bar_shadow_ratio) * 0.5)
            return {
                "pattern": "pin_bar_bull",
                "direction": "long",
                "strength": strength,
            }

        # 看跌Pin Bar（射击之星）：长上影线
        if upper_shadow >= body * self.pin_bar_shadow_ratio and lower_shadow < body:
            strength = min(1.0, upper_shadow / (body * self.pin_bar_shadow_ratio) * 0.5)
            return {
                "pattern": "pin_bar_bear",
                "direction": "short",
                "strength": strength,
            }

        return {"pattern": "none", "direction": "neutral", "strength": 0.0}

    def _detect_engulfing(
        self, current: Dict[str, float], prev: Dict[str, float],
    ) -> Dict[str, Any]:
        """检测吞没形态

        看涨吞没：前阴后阳，当前K线实体完全包住前一根实体
        看跌吞没：前阳后阴，当前K线实体完全包住前一根实体
        """
        prev_open = prev.get("open", 0)
        prev_close = prev.get("close", 0)
        curr_open = current.get("open", 0)
        curr_close = current.get("close", 0)

        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)

        if prev_body < 1e-10 or curr_body < 1e-10:
            return {"pattern": "none", "direction": "neutral", "strength": 0.0}

        # 看涨吞没：前跌后涨，当前实体包住前实体
        if prev_close < prev_open and curr_close > curr_open:
            if curr_open <= prev_close and curr_close >= prev_open:
                strength = min(1.0, curr_body / prev_body * 0.4)
                return {
                    "pattern": "engulfing_bull",
                    "direction": "long",
                    "strength": strength,
                }

        # 看跌吞没：前涨后跌，当前实体包住前实体
        if prev_close > prev_open and curr_close < curr_open:
            if curr_open >= prev_close and curr_close <= prev_open:
                strength = min(1.0, curr_body / prev_body * 0.4)
                return {
                    "pattern": "engulfing_bear",
                    "direction": "short",
                    "strength": strength,
                }

        return {"pattern": "none", "direction": "neutral", "strength": 0.0}

    def _detect_inside_bar(
        self, current: Dict[str, float], prev: Dict[str, float],
    ) -> Dict[str, Any]:
        """检测内包线（Inside Bar）

        Inside Bar：当前K线完全在前一根K线范围内
        信号：整理形态，突破方向待定（返回neutral，强度低）
        """
        prev_high = prev.get("high", 0)
        prev_low = prev.get("low", 0)
        curr_high = current.get("high", 0)
        curr_low = current.get("low", 0)

        if curr_high <= prev_high and curr_low >= prev_low:
            return {
                "pattern": "inside_bar",
                "direction": "neutral",
                "strength": 0.2,  # 低强度，表示整理中
            }

        return {"pattern": "none", "direction": "neutral", "strength": 0.0}


# ============================================================================
# 3. 支撑阻力位
# ============================================================================


class SupportResistance:
    """支撑阻力位检测

    来源：经典技术分析 + Wyckoff理论

    原理：
      - 价格在关键水平有"记忆效应"
      - 支撑位：多次触及未破的低点
      - 阻力位：多次触及未破的高点
      - 价格接近支撑位=买入信号，接近阻力位=卖出信号

    量化方法：
      - 取近N根K线的最高点/最低点
      - 聚类识别密集区（多个相近的高点/低点）
      - 计算价格距支撑/阻力的偏离度

    参数：仅1个（lookback_window，回溯窗口）
    """

    def __init__(
        self,
        lookback_window: int = 20,       # 回溯K线数
        proximity_threshold: float = 0.01,  # 接近阈值（1%以内=接近）
        min_touches: int = 2,            # 最少触及次数
    ):
        """
        Args:
            lookback_window: 回溯K线数
            proximity_threshold: 接近阈值（价格距支撑/阻力<此比例=接近）
            min_touches: 形成有效支撑/阻力的最少触及次数
        """
        self.lookback_window = lookback_window
        self.proximity_threshold = proximity_threshold
        self.min_touches = min_touches
        self._klines: deque = deque(maxlen=lookback_window + 5)

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新K线并检测支撑阻力

        Returns:
            {
                "signal": "near_support"/"near_resistance"/"none",
                "direction": "long"/"short"/"neutral",
                "strength": float,
                "support": float,   # 支撑位
                "resistance": float, # 阻力位
                "current_price": float,
            }
        """
        self._klines.append(kline)

        if len(self._klines) < self.lookback_window:
            return self._no_signal(kline.get("close", 0))

        return self._detect_levels(kline.get("close", 0))

    def _detect_levels(self, current_price: float) -> Dict[str, Any]:
        """检测支撑阻力位 (v596 Bug#3修复: 局部极值+聚类替代max/min)

        旧版bug: resistance=max(highs), support=min(lows) 只取单一极值，无聚类
        新版方法:
          1. 检测局部极值（高低点）
          2. 按价格聚类（差异<1%的极值合并为一组）
          3. 每组取均值作为支撑/阻力位，触碰次数=组内极值数
          4. 选择距当前价格最近的支撑/阻力位

        科学依据:
          - 支撑阻力位是价格"记忆"的体现，需要多次触及才有效
          - max/min只反映单次极端值，不代表价格密集区
          - 聚类方法识别真正的供需集中区
        """
        klines = list(self._klines)[-self.lookback_window:]

        # 收集所有高点低点
        highs = [k.get("high", k.get("close", 0)) for k in klines]
        lows = [k.get("low", k.get("close", 0)) for k in klines]

        # v596 Bug#3修复: 局部极值检测 + 聚类
        resistance_levels = self._find_price_levels(highs, is_high=True)
        support_levels = self._find_price_levels(lows, is_high=False)

        # 选择距当前价格最近的支撑/阻力位
        if resistance_levels:
            # 阻力位：选择当前价格上方的最近阻力
            above_resistances = [r for r in resistance_levels if r["price"] > current_price]
            if above_resistances:
                resistance = min(above_resistances, key=lambda x: x["price"])["price"]
                resistance_touches = min(above_resistances, key=lambda x: x["price"])["touches"]
            else:
                # 没有上方阻力，取最高阻力位
                resistance = max(r["price"] for r in resistance_levels)
                resistance_touches = max(resistance_levels, key=lambda x: x["price"])["touches"]
        else:
            resistance = current_price * 1.02  # 默认2%上方
            resistance_touches = 0

        if support_levels:
            # 支撑位：选择当前价格下方的最近支撑
            below_supports = [s for s in support_levels if s["price"] < current_price]
            if below_supports:
                support = max(below_supports, key=lambda x: x["price"])["price"]
                support_touches = max(below_supports, key=lambda x: x["price"])["touches"]
            else:
                # 没有下方支撑，取最低支撑位
                support = min(s["price"] for s in support_levels)
                support_touches = min(support_levels, key=lambda x: x["price"])["touches"]
        else:
            support = current_price * 0.98  # 默认2%下方
            support_touches = 0

        # 计算偏离度
        if resistance > 0:
            resistance_distance = (resistance - current_price) / resistance
        else:
            resistance_distance = 1.0

        if support > 0 and current_price > 0:
            support_distance = (current_price - support) / current_price
        else:
            support_distance = 1.0

        # 接近支撑位=做多信号（触碰次数越多，强度越大）
        if support_distance < self.proximity_threshold and support_touches >= self.min_touches:
            base_strength = min(1.0, (self.proximity_threshold - support_distance) / self.proximity_threshold)
            touch_bonus = min(0.3, support_touches * 0.1)  # 触碰次数加成，最多+0.3
            return {
                "signal": "near_support",
                "direction": "long",
                "strength": min(1.0, base_strength * 0.7 + touch_bonus),
                "support": support,
                "resistance": resistance,
                "support_touches": support_touches,
                "resistance_touches": resistance_touches,
                "all_supports": [s["price"] for s in support_levels],
                "all_resistances": [r["price"] for r in resistance_levels],
                "current_price": current_price,
            }

        # 接近阻力位=做空信号
        if resistance_distance < self.proximity_threshold and resistance_touches >= self.min_touches:
            base_strength = min(1.0, (self.proximity_threshold - resistance_distance) / self.proximity_threshold)
            touch_bonus = min(0.3, resistance_touches * 0.1)
            return {
                "signal": "near_resistance",
                "direction": "short",
                "strength": min(1.0, base_strength * 0.7 + touch_bonus),
                "support": support,
                "resistance": resistance,
                "support_touches": support_touches,
                "resistance_touches": resistance_touches,
                "all_supports": [s["price"] for s in support_levels],
                "all_resistances": [r["price"] for r in resistance_levels],
                "current_price": current_price,
            }

        return {
            "signal": "none",
            "direction": "neutral",
            "strength": 0.0,
            "support": support,
            "resistance": resistance,
            "current_price": current_price,
        }

    def _no_signal(self, current_price: float) -> Dict[str, Any]:
        return {
            "signal": "none",
            "direction": "neutral",
            "strength": 0.0,
            "support": current_price,
            "resistance": current_price,
            "current_price": current_price,
        }

    def _find_price_levels(
        self,
        prices: List[float],
        is_high: bool = True,
        cluster_threshold: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """v596 Bug#3修复: 局部极值检测 + 价格聚类

        替代旧版 max(highs)/min(lows) 的单一极值方法。

        方法:
          1. 检测局部极值（高点或低点）
          2. 按价格排序
          3. 将价格差异<cluster_threshold(1%)的极值合并为一组
          4. 每组取价格均值作为支撑/阻力位，touches=组内极值数

        Args:
            prices: 价格序列（highs或lows）
            is_high: True=检测高点(阻力), False=检测低点(支撑)
            cluster_threshold: 聚类阈值（价格差异<此比例合并）

        Returns:
            [{"price": float, "touches": int}, ...] 按价格排序
        """
        if len(prices) < 3:
            return []

        # 1. 检测局部极值
        extremes: List[float] = []
        for i in range(1, len(prices) - 1):
            if is_high:
                # 局部高点: prices[i] > 前后
                if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                    extremes.append(prices[i])
            else:
                # 局部低点: prices[i] < 前后
                if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                    extremes.append(prices[i])

        if not extremes:
            # 无局部极值时，取最大/最小值作为单一极值
            if is_high:
                extremes = [max(prices)]
            else:
                extremes = [min(prices)]

        # 2. 按价格排序
        extremes.sort()

        # 3. 聚类：将价格差异<1%的极值合并
        clusters: List[List[float]] = []
        for price in extremes:
            if clusters:
                last_cluster = clusters[-1]
                last_price = last_cluster[-1]
                # 价格差异 < cluster_threshold 则合并
                if last_price > 0 and abs(price - last_price) / last_price < cluster_threshold:
                    last_cluster.append(price)
                    continue
            # 创建新组
            clusters.append([price])

        # 4. 每组取均值，touches=组内极值数
        levels: List[Dict[str, Any]] = []
        for cluster in clusters:
            avg_price = sum(cluster) / len(cluster)
            levels.append({
                "price": avg_price,
                "touches": len(cluster),
            })

        # 按触碰次数降序排序（触碰次数多的更重要）
        levels.sort(key=lambda x: x["touches"], reverse=True)
        return levels


# ============================================================================
# 4. 综合价格行为系统
# ============================================================================


class ChanLunPriceActionSystem:
    """综合缠论+裸K价格行为系统

    整合3大信号源，提供统一的交易信号评估接口。

    使用方式：
        system = ChanLunPriceActionSystem()
        result = system.evaluate(kline)
        # result["signal"] = "long"/"short"/"neutral"
        # result["strength"] = 0-1
        # result["sources"] = ["chanlun_divergence", "pin_bar_bull", ...]
    """

    def __init__(
        self,
        wave_window: int = 5,
        pin_bar_shadow_ratio: float = 2.0,
        lookback_window: int = 20,
        swing_lookback: int = 5,
        phase_window: int = 20,
        minor_period: int = 5,
        major_period: int = 20,
    ):
        self.chanlun = ChanLunDivergence(wave_window=wave_window)
        self.price_action = PriceActionPatterns(pin_bar_shadow_ratio=pin_bar_shadow_ratio)
        self.sr = SupportResistance(lookback_window=lookback_window)
        # v3.33: 裸K四维分析模块 — 结构维度+趋势维度
        # 用户诉求: "就拿裸K而言，就有结构，趋势，动能，价格等知识点"
        self.structure = MarketStructureAnalyzer(swing_lookback=swing_lookback)
        self.trend_phase = TrendPhaseAnalyzer(phase_window=phase_window)
        # v3.34: 道氏理论模块 — 多周期趋势分级+跨资产互证
        # 用户诉求: "其次什么道氏理论，也就是我说的初始基因的知识储备不够丰富不够全面"
        self.dow_theory = DowTheoryAnalyzer(
            minor_period=minor_period, major_period=major_period,
        )

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新所有信号并综合评估 (v3.33: 裸K四维融合)

        四维融合逻辑（用户要求"真正的懂，才会灵活的用"）：
          - 结构维度 (BOS/CHoCH): 强独立信号，权重1.0，机构资金行为直接证据
          - 趋势维度 (三阶段): 方向过滤，只在markup/markdown阶段允许顺势入场
          - 动能维度 (背驰): 独立信号，权重0.5，趋势力度衰减预警
          - 价格维度 (PinBar/支撑阻力): 独立信号，权重0.5，关键位置反应

        融合规则：
          1. 趋势阶段=markup → 只允许做多信号通过（做空信号置neutral）
          2. 趋势阶段=markdown → 只允许做空信号通过
          3. 趋势阶段=distribution/re_accumulation → 信号强度×0.5（衰竭期谨慎）
          4. 趋势阶段=accumulation/ranging → 信号强度×0.7（方向未明，降低权重）
          5. BOS/CHoCH信号不受趋势阶段过滤（结构突破是最高优先级信号）

        Returns:
            {
                "signal": "long"/"short"/"neutral",
                "strength": float,  # 0-1，综合强度
                "sources": List[str],  # 触发的信号源列表
                "details": Dict,  # 各信号详细结果（含四维）
                "phase": str,  # v3.33: 当前趋势阶段
                "structure_bias": str,  # v3.33: 当前结构偏向
            }
        """
        # 各信号独立评估（四维并行）
        div_result = self.chanlun.update(kline)         # 动能维度
        pa_result = self.price_action.update(kline)     # 价格维度
        sr_result = self.sr.update(kline)               # 价格维度
        struct_result = self.structure.update(kline)    # 结构维度 v3.33
        phase_result = self.trend_phase.update(kline)   # 趋势维度 v3.33
        dow_result = self.dow_theory.update(kline)      # 道氏理论 v3.34

        # 综合信号
        long_scores = []
        short_scores = []
        sources = []

        # 动能维度 (背驰) — 权重1.0 (恢复原始权重)
        if div_result["direction"] == "long":
            long_scores.append(div_result["strength"] * 1.0)
            sources.append(f"chanlun_{div_result['signal']}")
        elif div_result["direction"] == "short":
            short_scores.append(div_result["strength"] * 1.0)
            sources.append(f"chanlun_{div_result['signal']}")

        # 价格维度 (PinBar/Engulfing) — 权重1.0 (恢复原始权重)
        if pa_result["direction"] == "long":
            long_scores.append(pa_result["strength"] * 1.0)
            sources.append(pa_result["pattern"])
        elif pa_result["direction"] == "short":
            short_scores.append(pa_result["strength"] * 1.0)
            sources.append(pa_result["pattern"])

        # 价格维度 (支撑阻力) — 权重1.0 (恢复原始权重)
        if sr_result["direction"] == "long":
            long_scores.append(sr_result["strength"] * 1.0)
            sources.append(sr_result["signal"])
        elif sr_result["direction"] == "short":
            short_scores.append(sr_result["strength"] * 1.0)
            sources.append(sr_result["signal"])

        # 结构维度 (BOS/CHoCH) — 权重1.5，强信号
        # 经济逻辑: 机构资金行为是最强证据，不受趋势阶段过滤
        if struct_result["direction"] == "long":
            long_scores.append(struct_result["strength"] * 1.5)
            sources.append(struct_result["signal"])
        elif struct_result["direction"] == "short":
            short_scores.append(struct_result["strength"] * 1.5)
            sources.append(struct_result["signal"])

        # 趋势维度 (三阶段) — 仅作为方向过滤，不直接贡献信号
        # 底层逻辑修复: 趋势阶段是上下文，不是入场触发
        # v3.33实验教训: 让趋势阶段直接贡献信号导致markup阶段过度交易(50笔32%胜率)
        # 正确用法: 趋势阶段过滤其他信号，而非自己产生信号
        current_phase = phase_result["phase"]
        phase_direction = phase_result["direction"]
        phase_strength = phase_result["strength"]
        # 注: phase_direction/phase_strength 保留用于下方过滤逻辑，不再贡献scores

        # 综合决策：多信号共振增强置信度
        long_score = sum(long_scores) * (1 + len(long_scores) * 0.2) if long_scores else 0
        short_score = sum(short_scores) * (1 + len(short_scores) * 0.2) if short_scores else 0

        # 趋势阶段过滤（BOS/CHoCH结构信号除外）
        has_struct_signal = struct_result["direction"] in ("long", "short")
        if not has_struct_signal:
            # 非结构信号需要受趋势阶段过滤
            if current_phase == "markup" and short_score > long_score:
                # 加速期上涨，过滤做空信号
                short_score = 0
            elif current_phase == "markdown" and long_score > short_score:
                # 加速期下跌，过滤做多信号
                long_score = 0
            elif current_phase in ("distribution", "re_accumulation"):
                # 衰竭期：信号强度减半
                long_score *= 0.5
                short_score *= 0.5
            elif current_phase in ("accumulation", "ranging"):
                # 启动期/震荡期：信号强度降低
                long_score *= 0.7
                short_score *= 0.7

        # v3.34: 道氏理论大方向过滤（最强约束）
        # 经济逻辑: 大级别趋势约束小级别，逆大势做小波动=逆水行舟
        # 用户诉求: "其次什么道氏理论，初始基因的知识储备不够丰富不够全面"
        # 道氏原则2: 三级趋势分级，主要趋势决定交易大背景
        dow_direction = dow_result["direction"]
        dow_strength = dow_result["strength"]
        dow_confirmed = dow_result["confirmed"]
        if dow_confirmed and dow_strength > 0.5:
            # 道氏确认的大方向：逆大势信号强度×0.3（大幅衰减但不完全禁止）
            # 不完全禁止是为了保留极端结构信号(CHoCH)的反转捕捉能力
            if dow_direction == "long" and short_score > long_score:
                short_score *= 0.3
                sources.append("dow_filter_short")
            elif dow_direction == "short" and long_score > short_score:
                long_score *= 0.3
                sources.append("dow_filter_long")

        if long_score > short_score and long_score > 0:
            signal = "long"
            strength = min(1.0, long_score)
        elif short_score > long_score and short_score > 0:
            signal = "short"
            strength = min(1.0, short_score)
        else:
            signal = "neutral"
            strength = 0.0

        return {
            "signal": signal,
            "strength": strength,
            "sources": sources,
            "details": {
                "chanlun": div_result,
                "price_action": pa_result,
                "support_resistance": sr_result,
                "structure": struct_result,      # v3.33 新增
                "trend_phase": phase_result,     # v3.33 新增
                "dow_theory": dow_result,        # v3.34 新增
            },
            "phase": current_phase,                              # v3.33 新增
            "structure_bias": struct_result.get("structure_bias", "neutral"),  # v3.33 新增
            "dow_confirmed": dow_result.get("confirmed", False),  # v3.34 新增
            "dow_direction": dow_result.get("direction", "neutral"),  # v3.34 新增
        }

    def get_support_resistance(self) -> Tuple[float, float]:
        """获取当前支撑阻力位（供止损止盈参考）"""
        if len(self.sr._klines) > 0:
            last = self.sr._klines[-1]
            result = self.sr._detect_levels(last.get("close", 0))
            return result["support"], result["resistance"]
        return 0.0, 0.0


# ============================================================================
# v3.33 裸K四维分析模块 (结构维度 + 趋势维度)
# ============================================================================
#
# 用户核心诉求：
#   "就拿裸K而言，就有结构，趋势，动能，价格等知识点"
#   "已有的知识没去搞懂底层逻辑，真正的懂，才会灵活的用！"
#
# 设计原则（反过拟合）：
#   1. 每个维度独立计算，避免参数耦合
#   2. 每个判断都有明确经济逻辑（机构资金行为/趋势生命周期）
#   3. 参数极少（每模块≤2个），远低于过拟合警戒线
#   4. 信号可独立验证（PBO<0.3 + 蒙特卡洛p<0.05）
#
# 四维对应关系：
#   - 结构维度 → MarketStructureAnalyzer (BOS/CHoCH)  ← 本模块新增
#   - 趋势维度 → TrendPhaseAnalyzer (三阶段)          ← 本模块新增
#   - 动能维度 → ChanLunDivergence (背驰)             ← 已有，增强
#   - 价格维度 → PriceActionPatterns + SupportResistance ← 已有
# ============================================================================


class MarketStructureAnalyzer:
    """裸K结构维度分析 — ICT/SMC智能资金概念

    底层逻辑（用户要求"真正懂底层逻辑"）：
      市场不是随机的，而是由机构资金（Smart Money）驱动。
      机构资金体量庞大，其建仓/出货必然在价格结构上留下可识别的足迹。

    核心概念：
      1. 市场结构序列（HH/HL/LH/LL）：
         - HH (Higher High) 更高高点：多头力量增强
         - HL (Higher Low)  更高低点：多头支撑上移
         - LH (Lower High)  更低高点：空头力量增强
         - LL (Lower Low)   更低低点：空头压力下移
         - HH+HL序列 = 多头结构（上涨趋势）
         - LH+LL序列 = 空头结构（下跌趋势）

      2. BOS (Break of Structure) 结构突破 — 趋势延续信号：
         - 多头BOS：价格突破前一个HH → 机构继续加仓做多
         - 空头BOS：价格跌破前一个LL → 机构继续加仓做空
         - 经济含义：机构资金延续其方向操作，趋势未结束

      3. CHoCH (Change of Character) 性格变化 — 趋势反转预警：
         - 多头→空头CHoCH：多头趋势中价格跌破前一个HL
         - 空头→多头CHoCH：空头趋势中价格突破前一个LH
         - 经济含义：机构资金开始反向操作，趋势可能反转

    与传统突破的区别：
      传统突破只看价格穿越某条线，而BOS/CHoCH看的是结构序列的变化，
      即"机构资金的行为模式"而非"单一价格事件"。

    参数：仅2个（swing_lookback, break_threshold）
    """

    def __init__(
        self,
        swing_lookback: int = 5,        # 检测波段高低点的窗口
        break_threshold: float = 0.001,  # 突破确认阈值（0.1%）
    ):
        """
        Args:
            swing_lookback: 识别波段高低点的K线回溯数。
                            过小=噪声多，过大=迟钝。5根是日内交易的平衡点。
            break_threshold: 突破确认阈值（价格穿越比例）。
                             避免假突破，0.1%是BTC/ETH日内常见噪声水平。
        """
        self.swing_lookback = swing_lookback
        self.break_threshold = break_threshold
        self._klines: deque = deque(maxlen=swing_lookback * 4 + 10)
        # 结构状态追踪
        self._last_swing_high: Optional[float] = None
        self._last_swing_low: Optional[float] = None
        self._prev_swing_high: Optional[float] = None
        self._prev_swing_low: Optional[float] = None
        self._structure_bias: str = "neutral"  # bullish/bearish/neutral

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新K线并检测结构变化

        Returns:
            {
                "signal": "bos_bull"/"bos_bear"/"choch_bull"/"choch_bear"/"none",
                "direction": "long"/"short"/"neutral",
                "strength": float,  # 0-1
                "structure_bias": str,  # 当前结构偏向 bullish/bearish/neutral
                "last_swing_high": float,
                "last_swing_low": float,
            }
        """
        self._klines.append(kline)
        if len(self._klines) < self.swing_lookback * 2 + 2:
            return self._no_signal()

        # v3.36 根因修复: BOS/CHoCH 0触发死锁
        # 原BUG: line 802-803 "if new_high is None and new_low is None: return"
        #   → 价格突破时无新波段点 → 直接return → 不检测突破
        #   → 等新波段点确认时 _last_swing_high 已更新为更高值 → close无法突破
        #   → 完美死锁, BOS/CHoCH永远0触发
        # 修复原则(根治非妥协):
        #   BOS/CHoCH是"价格突破事件", 与"波段点确认事件"是独立事件
        #   波段点检测负责维护参考点, 突破检测每根K线都执行
        # 来源: ICT/SMC理论 — BOS=Break of Structure, CHoCH=Change of Character
        #       两者都是价格行为事件, 不依赖新波段点形成
        new_high, new_low = self._detect_swings()
        # 更新波段序列(如有新波段点)
        if new_high is not None:
            self._prev_swing_high = self._last_swing_high
            self._last_swing_high = new_high
        if new_low is not None:
            self._prev_swing_low = self._last_swing_low
            self._last_swing_low = new_low

        # 无论是否检测到新波段点, 都检查当前close是否突破已有波段点
        if self._last_swing_high is None or self._last_swing_low is None:
            return self._no_signal()

        close = kline.get("close", 0)
        if close <= 0:
            return self._no_signal()

        # 结构突破检测(每根K线都执行)
        return self._detect_structure_event(close)

    def _detect_swings(self) -> Tuple[Optional[float], Optional[float]]:
        """检测波段高低点

        波段高点：lookback窗口内的最高点，且该点位于窗口中央位置
        波段低点：lookback窗口内的最低点，且该点位于窗口中央位置
        （中央位置确保该点左右两侧都有K线，确认其为有效波段）
        """
        klines = list(self._klines)
        n = len(klines)
        if n < self.swing_lookback * 2 + 1:
            return None, None

        # 检查最近的K线是否构成波段点（以最后第swing_lookback根为中心）
        center_idx = n - self.swing_lookback - 1
        if center_idx < self.swing_lookback:
            return None, None

        center = klines[center_idx]
        center_high = center.get("high", center.get("close", 0))
        center_low = center.get("low", center.get("close", 0))

        window = klines[center_idx - self.swing_lookback: center_idx + self.swing_lookback + 1]

        highs = [k.get("high", k.get("close", 0)) for k in window]
        lows = [k.get("low", k.get("close", 0)) for k in window]

        new_high = None
        new_low = None

        # 中心点是窗口最高点 = 波段高点
        if center_high >= max(highs) and center_high > 0:
            new_high = center_high
        # 中心点是窗口最低点 = 波段低点
        if center_low <= min(lows) and center_low > 0:
            new_low = center_low

        return new_high, new_low

    def _detect_structure_event(self, close: float) -> Dict[str, Any]:
        """检测BOS或CHoCH事件

        判定逻辑：
          - 价格突破last_swing_high（多头BOS）→ 趋势延续做多
          - 价格跌破last_swing_low（空头BOS）→ 趋势延续做空
          - 多头结构中价格跌破last_swing_low（CHoCH）→ 反转做空
          - 空头结构中价格突破last_swing_high（CHoCH）→ 反转做多
        """
        if self._last_swing_high is None or self._last_swing_low is None:
            return self._no_signal()

        # 突破确认（避免假突破）
        break_high = close > self._last_swing_high * (1 + self.break_threshold)
        break_low = close < self._last_swing_low * (1 - self.break_threshold)

        if not break_high and not break_low:
            return self._no_signal()

        # 计算突破强度（突破幅度越大，信号越强）
        if break_high:
            break_pct = (close - self._last_swing_high) / self._last_swing_high
            strength = min(1.0, break_pct / (self.break_threshold * 10) * 0.5 + 0.3)
        else:
            break_pct = (self._last_swing_low - close) / self._last_swing_low
            strength = min(1.0, break_pct / (self.break_threshold * 10) * 0.5 + 0.3)

        # 根据当前结构偏向判断BOS还是CHoCH
        # v3.36 次修复: 触发后消费对应波段点防止重复触发
        # 经济逻辑: 突破是一次性事件, 旧波段点失去参考意义
        # ICT/SMC理论: 每次BOS后市场进入新结构区间, 需等待新波段点形成
        if break_high:
            if self._structure_bias == "bearish":
                # 空头结构中突破前高 = CHoCH（反转做多）
                self._structure_bias = "bullish"
                # 消费已突破的波段高点, 等待新波段点形成
                self._last_swing_high = None
                return {
                    "signal": "choch_bull",
                    "direction": "long",
                    "strength": strength,
                    "structure_bias": self._structure_bias,
                    "last_swing_high": self._last_swing_high,
                    "last_swing_low": self._last_swing_low,
                }
            else:
                # 多头结构中突破前高 = BOS（延续做多）
                self._structure_bias = "bullish"
                # 消费已突破的波段高点, 防止重复触发
                self._last_swing_high = None
                return {
                    "signal": "bos_bull",
                    "direction": "long",
                    "strength": strength * 0.85,  # v3.36: 0.8→0.85 BOS延续强证据
                    "structure_bias": self._structure_bias,
                    "last_swing_high": self._last_swing_high,
                    "last_swing_low": self._last_swing_low,
                }

        if break_low:
            if self._structure_bias == "bullish":
                # 多头结构中跌破前低 = CHoCH（反转做空）
                self._structure_bias = "bearish"
                # 消费已突破的波段低点, 等待新波段点形成
                self._last_swing_low = None
                return {
                    "signal": "choch_bear",
                    "direction": "short",
                    "strength": strength,
                    "structure_bias": self._structure_bias,
                    "last_swing_high": self._last_swing_high,
                    "last_swing_low": self._last_swing_low,
                }
            else:
                # 空头结构中跌破前低 = BOS（延续做空）
                self._structure_bias = "bearish"
                # 消费已突破的波段低点, 防止重复触发
                self._last_swing_low = None
                return {
                    "signal": "bos_bear",
                    "direction": "short",
                    "strength": strength * 0.85,  # v3.36: 0.8→0.85
                    "structure_bias": self._structure_bias,
                    "last_swing_high": self._last_swing_high,
                    "last_swing_low": self._last_swing_low,
                }

        return self._no_signal()

    def _no_signal(self) -> Dict[str, Any]:
        return {
            "signal": "none",
            "direction": "neutral",
            "strength": 0.0,
            "structure_bias": self._structure_bias,
            "last_swing_high": self._last_swing_high if self._last_swing_high else 0.0,
            "last_swing_low": self._last_swing_low if self._last_swing_low else 0.0,
        }


class TrendPhaseAnalyzer:
    """裸K趋势维度分析 — 趋势生命周期三阶段

    底层逻辑（用户要求"真正懂底层逻辑"）：
      趋势不是瞬时事件，而是有完整生命周期的有机体。
      道氏理论+Wyckoff理论都揭示了趋势的三个阶段：

    三阶段模型：
      1. 启动期 (Accumulation/Distribution):
         - 聪明钱开始建仓/出货，但价格尚未明确突破
         - 特征：价格震荡、波动率收缩、量能温和
         - 此时入场：风险高（方向未明），回报高（如果方向正确）
         - 策略含义：等待突破确认，不在此阶段追单

      2. 加速期 (Markup/Markdown):
         - 大众资金跟进，价格趋势明确
         - 特征：趋势斜率陡峭、量能放大、回调浅
         - 此时入场：风险低（趋势明确），回报中等（已错过启动段）
         - 策略含义：顺势入场的最佳阶段，回调买入/反弹做空

      3. 衰竭期 (Re-distribution/Re-accumulation):
         - 趋势力度衰减，聪明钱开始平仓
         - 特征：趋势斜率放缓、量能递减、回调深、出现背离
         - 此时入场：风险高（反转在即），回报低（趋势末端）
         - 策略含义：减仓或离场，等待新趋势形成

    量化方法（客观化）：
      - 趋势斜率：线性回归斜率，衡量方向与强度
      - 量能变化：近期量能vs远期量能的比值，衡量参与度
      - 波动率：ATR比值，衡量市场状态
      - 综合三者判断当前所处阶段

    参数：仅2个（phase_window, slope_threshold）
    """

    def __init__(
        self,
        phase_window: int = 20,         # 判断阶段的回溯窗口
        slope_threshold: float = 0.0005,  # 趋势斜率阈值（0.05%/根K线）
    ):
        """
        Args:
            phase_window: 判断趋势阶段的K线回溯数。
                          20根=1小时(3min K线)≈日内交易有效周期
            slope_threshold: 区分"有趋势"和"无趋势"的斜率阈值。
                             BTC日内波动0.5%-2%，0.05%/根是有效趋势下限。
        """
        self.phase_window = phase_window
        self.slope_threshold = slope_threshold
        self._klines: deque = deque(maxlen=phase_window * 2 + 10)
        self._volumes: deque = deque(maxlen=phase_window * 2 + 10)

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新K线并判断趋势阶段

        Returns:
            {
                "phase": "accumulation"/"markup"/"distribution"/"markdown"/"ranging",
                "direction": "long"/"short"/"neutral",
                "strength": float,  # 0-1，阶段信号强度
                "trend_slope": float,  # 趋势斜率（正=上涨，负=下跌）
                "volume_ratio": float,  # 量能比值（>1=放量，<1=缩量）
                "phase_confidence": float,  # 阶段判断置信度
            }
        """
        close = kline.get("close", 0)
        volume = kline.get("volume", 0)
        if close <= 0:
            return self._no_signal()

        self._klines.append(close)
        self._volumes.append(volume if volume > 0 else 0)

        if len(self._klines) < self.phase_window:
            return self._no_signal()

        # 计算3个核心指标
        slope = self._compute_trend_slope()
        volume_ratio = self._compute_volume_ratio()
        volatility = self._compute_volatility_ratio()

        # 综合判断阶段
        return self._classify_phase(slope, volume_ratio, volatility)

    def _compute_trend_slope(self) -> float:
        """计算趋势斜率（线性回归）

        使用最小二乘法拟合close价格的线性趋势。
        斜率>0=上涨趋势，斜率<0=下跌趋势，斜率≈0=无趋势。
        """
        closes = list(self._klines)[-self.phase_window:]
        n = len(closes)
        if n < 5:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(closes) / n
        numerator = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(closes))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-10 or y_mean < 1e-10:
            return 0.0

        # 归一化斜率（相对于价格）
        slope = numerator / denominator / y_mean
        return slope

    def _compute_volume_ratio(self) -> float:
        """计算量能比值（近期/远期）

        volume_ratio > 1 = 放量（参与度增强）
        volume_ratio < 1 = 缩量（参与度减弱）
        """
        vols = list(self._volumes)[-self.phase_window:]
        n = len(vols)
        if n < 10:
            return 1.0

        half = n // 2
        recent_vol = sum(vols[half:]) / half if half > 0 else 0
        distant_vol = sum(vols[:half]) / half if half > 0 else 0

        if distant_vol < 1e-10:
            return 1.0

        return recent_vol / distant_vol

    def _compute_volatility_ratio(self) -> float:
        """计算波动率比值（近期/远期）

        波动率收缩=启动期特征
        波动率扩张=加速期特征
        波动率再次扩张=衰竭期特征（伴随量能递减）
        """
        klines = list(self._klines)[-self.phase_window:]
        n = len(klines)
        if n < 10:
            return 1.0

        half = n // 2
        recent_klines = klines[half:]
        distant_klines = klines[:half]

        # 用价格变化幅度近似波动率
        recent_ranges = [
            abs(recent_klines[i] - recent_klines[i - 1])
            for i in range(1, len(recent_klines))
        ]
        distant_ranges = [
            abs(distant_klines[i] - distant_klines[i - 1])
            for i in range(1, len(distant_klines))
        ]

        if not recent_ranges or not distant_ranges:
            return 1.0

        recent_vol = sum(recent_ranges) / len(recent_ranges)
        distant_vol = sum(distant_ranges) / len(distant_ranges)

        if distant_vol < 1e-10:
            return 1.0

        return recent_vol / distant_vol

    def _classify_phase(
        self, slope: float, volume_ratio: float, volatility: float,
    ) -> Dict[str, Any]:
        """根据3个指标综合判断趋势阶段

        判定规则（基于经济逻辑，非数据挖掘）：
          |slope| < threshold 且 volatility < 1.0 → accumulation (启动期，震荡+收缩)
          slope > threshold 且 volume_ratio > 1.0 → markup (加速期上涨，放量)
          slope > threshold 且 volume_ratio < 1.0 → distribution (衰竭期上涨，缩量)
          slope < -threshold 且 volume_ratio > 1.0 → markdown (加速期下跌，放量)
          slope < -threshold 且 volume_ratio < 1.0 → re-accumulation (衰竭期下跌，缩量)
          其他 → ranging (无明显趋势)
        """
        abs_slope = abs(slope)
        has_trend = abs_slope > self.slope_threshold

        if not has_trend:
            # 无明显趋势
            if volatility < 0.9:
                # 波动率收缩 = 启动期
                phase = "accumulation"
                direction = "neutral"
                strength = min(1.0, (0.9 - volatility) * 2)
            else:
                phase = "ranging"
                direction = "neutral"
                strength = 0.0
        else:
            # 有趋势
            is_up = slope > 0
            is_volume_up = volume_ratio > 1.0

            if is_up and is_volume_up:
                # 上涨+放量 = 加速期
                phase = "markup"
                direction = "long"
                strength = min(1.0, abs_slope / self.slope_threshold * 0.3 + 0.4)
            elif is_up and not is_volume_up:
                # 上涨+缩量 = 衰竭期
                phase = "distribution"
                direction = "neutral"  # 衰竭期不建议追多
                strength = min(1.0, (1.0 - volume_ratio) * 2)
            elif not is_up and is_volume_up:
                # 下跌+放量 = 加速期下跌
                phase = "markdown"
                direction = "short"
                strength = min(1.0, abs_slope / self.slope_threshold * 0.3 + 0.4)
            else:
                # 下跌+缩量 = 衰竭期下跌
                phase = "re_accumulation"
                direction = "neutral"
                strength = min(1.0, (1.0 - volume_ratio) * 2)

        # 阶段置信度：基于指标的一致性
        confidence = 0.5
        if has_trend:
            confidence += 0.2
        if abs(volume_ratio - 1.0) > 0.2:
            confidence += 0.15
        if abs(volatility - 1.0) > 0.2:
            confidence += 0.15
        confidence = min(1.0, confidence)

        return {
            "phase": phase,
            "direction": direction,
            "strength": strength,
            "trend_slope": slope,
            "volume_ratio": volume_ratio,
            "volatility_ratio": volatility,
            "phase_confidence": confidence,
        }

    def _no_signal(self) -> Dict[str, Any]:
        return {
            "phase": "unknown",
            "direction": "neutral",
            "strength": 0.0,
            "trend_slope": 0.0,
            "volume_ratio": 1.0,
            "volatility_ratio": 1.0,
            "phase_confidence": 0.0,
        }


# ============================================================================
# v3.34 道氏理论模块 (Dow Theory Analyzer)
# ============================================================================
#
# 用户核心诉求：
#   "其次什么道氏理论，也就是我说的初始基因的知识储备不够丰富不够全面"
#   "已有的知识没去搞懂底层逻辑，真正的懂，才会灵活的用！"
#
# 道氏理论六大基本原则（Charles Dow 1884, William Hamilton发展）：
#   1. 市场反映一切 — 所有信息已反映在价格中（已隐含在系统中）
#   2. 三级趋势分级 — 主要/次要/日间趋势 ← 本模块核心职责
#   3. 主要趋势三阶段 — 积累/加速/分配 ← 已在TrendPhaseAnalyzer实现
#   4. 趋势相互确认 — 道氏原始版本:工业指数+铁路指数互证
#                     现代版本:BTC+ETH互证 ← 本模块实现
#   5. 量能确认趋势 — 放量=趋势有效 ← 已在TrendPhaseAnalyzer实现
#   6. 趋势持续直到反转 — ← 已在MarketStructureAnalyzer实现(BOS/CHoCH)
#
# 本模块与现有模块的职责划分（避免功能重叠）：
#   - TrendPhaseAnalyzer: 单周期内判断趋势阶段(启动/加速/衰竭)
#   - MarketStructureAnalyzer: 单周期内判断结构突破(BOS/CHoCH)
#   - DowTheoryAnalyzer: 多周期趋势方向一致性 + 跨资产互证 ← 独特职责
#
# 设计原则（反过拟合）：
#   1. 仅3个参数，远低于过拟合警戒线
#   2. 趋势方向判断用简单MA比较，非复杂模型
#   3. 一致性判断基于明确经济逻辑：大级别趋势约束小级别
# ============================================================================


class DowTheoryAnalyzer:
    """道氏理论多周期趋势分级分析

    底层逻辑（用户要求"真正懂底层逻辑"）：
      道氏理论的核心洞察：市场同时存在三个级别的趋势，它们相互制约。

    三级趋势模型：
      1. 主要趋势 (Primary Trend, 周期-数月):
         - 大方向，像潮汐
         - 决定交易的"大背景"：主要趋势向上时，做多为主；向下时，做空为主
         - 量化：1小时K线趋势（20根3min聚合）

      2. 次要趋势 (Secondary Trend, 3周-数月):
         - 主要趋势的回调/反弹，像波浪
         - 持续数周到数月，回调幅度为主要趋势的1/3-2/3
         - 量化：15分钟K线趋势（5根3min聚合）

      3. 日间趋势 (Daily Trend, 日内):
         - 日常波动，像涟漪
         - 噪声较多，但提供精确入场时机
         - 量化：3分钟K线趋势（原始数据）

    三级一致性判断（核心信号）：
      - 三级同向（都上涨）= 强多头，信号强度1.0
        经济含义：大方向、中周期、短期都向上，机构资金持续买入
      - 两级同向（主要+次要）= 中等多头，信号强度0.6
        经济含义：大方向和中周期向上，短期回调提供入场机会
      - 三级分歧 = 无趋势/转折，信号强度0.0
        经济含义：趋势可能反转，不应入场

    跨资产互证（道氏原则4的现代版本）：
      道氏原始版本要求工业指数和铁路指数互证。
      现代加密货币版本：BTC和ETH应同向（BTC领涨，ETH确认）。
      如果BTC上涨但ETH下跌 = 分歧，趋势不可靠。

    参数：仅3个（minor_period, major_period, confirmation_threshold）
    """

    def __init__(
        self,
        minor_period: int = 5,    # 次要趋势聚合K线数（5根3min=15min）
        major_period: int = 20,   # 主要趋势聚合K线数（20根3min=1h）
        confirmation_threshold: float = 0.6,  # 一致性阈值
    ):
        """
        Args:
            minor_period: 次要趋势的聚合周期。
                          5根3min=15min，捕捉中周期趋势。
            major_period: 主要趋势的聚合周期。
                          20根3min=1h，捕捉大方向。
            confirmation_threshold: 三级一致性的判定阈值。
                                    0.6=至少2/3同向才算确认。
        """
        self.minor_period = minor_period
        self.major_period = major_period
        self.confirmation_threshold = confirmation_threshold

        # 原始K线缓存（日间趋势）
        self._klines: deque = deque(maxlen=major_period * 3 + 10)
        # 跨资产趋势缓存（道氏互证）
        self._cross_asset_trends: Dict[str, str] = {}

    def update(self, kline: Dict[str, float]) -> Dict[str, Any]:
        """更新K线并进行多周期趋势分级分析

        Returns:
            {
                "signal": "strong_bull"/"strong_bear"/"weak_bull"/"weak_bear"/"neutral",
                "direction": "long"/"short"/"neutral",
                "strength": float,  # 0-1
                "primary_trend": str,   # 主要趋势方向 up/down/flat
                "secondary_trend": str, # 次要趋势方向
                "daily_trend": str,     # 日间趋势方向
                "consistency": float,   # 三级一致性 0-1
                "confirmed": bool,      # 是否通过道氏确认
            }
        """
        self._klines.append(kline)

        if len(self._klines) < self.major_period + 5:
            return self._no_signal()

        # 计算三级趋势方向
        daily_trend = self._compute_trend_direction(list(self._klines), 1)
        secondary_trend = self._compute_trend_direction(
            list(self._klines), self.minor_period
        )
        primary_trend = self._compute_trend_direction(
            list(self._klines), self.major_period
        )

        # 三级一致性计算
        trends = [primary_trend, secondary_trend, daily_trend]
        up_count = sum(1 for t in trends if t == "up")
        down_count = sum(1 for t in trends if t == "down")
        flat_count = sum(1 for t in trends if t == "flat")

        # 一致性 = 最大同向比例
        max_consensus = max(up_count, down_count, flat_count) / 3.0

        # 信号判定
        if up_count >= 2 and max_consensus >= self.confirmation_threshold:
            if up_count == 3:
                signal = "strong_bull"
                strength = 1.0
            else:
                signal = "weak_bull"
                strength = 0.6
            direction = "long"
        elif down_count >= 2 and max_consensus >= self.confirmation_threshold:
            if down_count == 3:
                signal = "strong_bear"
                strength = 1.0
            else:
                signal = "weak_bear"
                strength = 0.6
            direction = "short"
        else:
            signal = "neutral"
            direction = "neutral"
            strength = 0.0

        return {
            "signal": signal,
            "direction": direction,
            "strength": strength,
            "primary_trend": primary_trend,
            "secondary_trend": secondary_trend,
            "daily_trend": daily_trend,
            "consistency": max_consensus,
            "confirmed": max_consensus >= self.confirmation_threshold,
        }

    def _compute_trend_direction(
        self, klines: List[Dict[str, float]], period: int,
    ) -> str:
        """计算指定周期的趋势方向

        方法：聚合K线到指定周期，比较近期MA vs 远期MA
        - 近期MA > 远期MA = 上涨趋势
        - 近期MA < 远期MA = 下跌趋势
        - 差异<阈值 = 无趋势

        Args:
            klines: 原始K线列表
            period: 聚合周期（1=日间, 5=次要, 20=主要）
        """
        if len(klines) < period * 4:
            return "flat"

        # 聚合K线：每period根合并为1根
        aggregated = []
        for i in range(0, len(klines) - period + 1, period):
            chunk = klines[i:i + period]
            if not chunk:
                continue
            closes = [k.get("close", 0) for k in chunk]
            avg_close = sum(closes) / len(closes) if closes else 0
            aggregated.append(avg_close)

        if len(aggregated) < 4:
            return "flat"

        # 近期MA vs 远期MA
        half = len(aggregated) // 2
        if half < 2:
            return "flat"

        recent_ma = sum(aggregated[half:]) / (len(aggregated) - half)
        distant_ma = sum(aggregated[:half]) / half

        if distant_ma < 1e-10:
            return "flat"

        ma_ratio = (recent_ma - distant_ma) / distant_ma

        # 阈值：0.1%差异才算有趋势（避免噪声）
        if ma_ratio > 0.001:
            return "up"
        elif ma_ratio < -0.001:
            return "down"
        else:
            return "flat"

    def update_cross_asset(self, symbol: str, trend: str) -> None:
        """更新跨资产趋势（道氏互证）

        道氏原则4的现代版本：BTC和ETH应同向。
        如果BTC上涨但ETH下跌 = 趋势不可靠。

        Args:
            symbol: 资产符号（如"BTC-USDT"）
            trend: 该资产的主要趋势方向 "up"/"down"/"flat"
        """
        self._cross_asset_trends[symbol] = trend

    def get_cross_asset_confirmation(self) -> Dict[str, Any]:
        """获取跨资产互证结果

        Returns:
            {
                "confirmed": bool,  # 是否互证
                "agree_count": int,  # 同向资产数
                "total_count": int,
                "agreement_ratio": float,
            }
        """
        total = len(self._cross_asset_trends)
        if total < 2:
            return {"confirmed": False, "agree_count": 0, "total_count": total, "agreement_ratio": 0.0}

        up_count = sum(1 for t in self._cross_asset_trends.values() if t == "up")
        down_count = sum(1 for t in self._cross_asset_trends.values() if t == "down")
        max_agree = max(up_count, down_count)
        agreement_ratio = max_agree / total

        return {
            "confirmed": agreement_ratio >= self.confirmation_threshold,
            "agree_count": max_agree,
            "total_count": total,
            "agreement_ratio": agreement_ratio,
        }

    def _no_signal(self) -> Dict[str, Any]:
        return {
            "signal": "neutral",
            "direction": "neutral",
            "strength": 0.0,
            "primary_trend": "flat",
            "secondary_trend": "flat",
            "daily_trend": "flat",
            "consistency": 0.0,
            "confirmed": False,
        }
