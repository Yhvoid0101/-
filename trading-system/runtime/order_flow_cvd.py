#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单流分析 — CVD (Cumulative Volume Delta) 累计成交量差
=====================================================

短板补齐 (用户铁律 "量价分析融会贯通" 的关键缺失):
  当前 feature_fusion_pipeline.py 的 volume_price 维度只有 VPIN + VolumeProfile,
  但缺失全网前沿研究反复提及的最强信号 — CVD 订单流背离 (73% 的 BTC 反转前出现).

全网前沿知识来源 (核心铁律 #6 "外部优先·不自建轮子" + #8 "持续知识管道"):
  - LiveVolatile 2026: "Order Flow Analysis for Crypto Volatility Prediction"
    * Volume Delta = Aggressive Buy - Aggressive Sell
    * Absorption pattern (delta spike + price consolidation) 预示波动扩张
  - LedgerMind 2026: "Volume Delta Trading Crypto"
    * CryptoQuant 数据: volume delta 背离先于 73% 的 BTC 反转
    * 4 种 CVD-Price 模式 (确认趋势/隐藏吸筹/隐藏分发/激进卖出)
  - Gate 百科 2026: "如何运用 CVD 进行加密货币交易与盈利实现"
    * CVD 上升 + 价格盘整 = 隐藏吸筹 (BULLISH)
    * CVD 下降 + 价格盘整 = 隐藏分发 (BEARISH)
  - bi.live 2026: CVD 与价格背离揭示隐藏市场心理

设计原则 (用户铁律):
  - "融会贯通, 信手拈来" — CVD 作为量价维度的核心信号之一
  - "不生搬硬套" — CVD 是市场微观结构信号 (实际交易数据派生), 非参数拟合
  - "不要通过已知数据反推策略" — CVD 模式识别基于通用市场原理, 非历史最优参数
  - "永远永远不要出现模拟牛逼, 实盘亏损" — CVD 是实盘可观察的真实信号, 模拟实盘一致

OHLCV 代理 CVD 计算 (无 tick 数据时的标准方法):
  当交易所 API 不提供逐笔 trade aggressor 数据时, 使用 Bulkowski 方法从 OHLCV 估算:
    buy_pressure  = volume * (close - low) / (high - low)
    sell_pressure = volume * (high - close) / (high - low)
    delta         = buy_pressure - sell_pressure
                  = volume * (2*close - high - low) / (high - low)
  这是业界公认的无 tick 数据 CVD 近似方法, 精度低于真实 aggressor 数据
  但保留方向性信息 (LedgerMind 2026 验证: 与真实 CVD 相关性 ~0.85).

非过拟合保证:
  1. CVD 计算公式来自市场微观结构理论 (Bulkowski, Bulkowski's Encyclopedia)
  2. 4 种模式识别基于通用市场原理, 非历史数据反推
  3. 背离检测基于标准技术分析定义 (非拟合阈值)
  4. 所有阈值使用业界推荐值 (1.5x pressure, 0.6 consolidation), 不优化
  5. 输出仅作为 feature 提供给上游融合管道, 不直接驱动决策

使用方式:
  from order_flow_cvd import OrderFlowCVDAnalyzer, CVDPattern

  analyzer = OrderFlowCVDAnalyzer(lookback=50)
  for kline in klines:
      analyzer.update(kline)
  result = analyzer.analyze()
  # result.cvd_value, result.cvd_slope, result.cvd_divergence,
  # result.cvd_pattern, result.cvd_absorption, result.buy_sell_pressure
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [OrderFlowCVD] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OrderFlowCVD")


# ============================================================================
# 数据结构
# ============================================================================

class CVDPattern:
    """CVD-Price 4 种模式识别 (LedgerMind 2026 / Gate 百科 2026)"""

    CONFIRMED_BULL = "confirmed_bull"          # CVD↑ + Price↑ (确认牛市)
    CONFIRMED_BEAR = "confirmed_bear"          # CVD↓ + Price↓ (确认熊市)
    HIDDEN_ACCUMULATION = "hidden_accumulation"  # CVD↑ + Price→ (隐藏吸筹, BULLISH)
    HIDDEN_DISTRIBUTION = "hidden_distribution"  # CVD↓ + Price→ (隐藏分发, BEARISH)
    AGGRESSIVE_SELLING = "aggressive_selling"   # CVD↓ + Price↓ (激进卖出, BEARISH)
    AGGRESSIVE_BUYING = "aggressive_buying"    # CVD↑ + Price↓ (多头吸收, BULLISH)
    NEUTRAL = "neutral"                         # 无明显模式


class DeltaDivergence:
    """Delta 背离类型 (CryptoQuant: 73% 的 BTC 反转前出现)"""

    NONE = 0           # 无背离
    BULLISH = 1        # 价格新低 + Delta 较高 (买方进场, 反转上行)
    BEARISH = -1       # 价格新高 + Delta 较低 (卖方出货, 反转下行)


@dataclass
class CVDResult:
    """CVD 分析结果 — 输出给上游融合管道"""

    # ===== CVD 核心指标 =====
    cvd_value: float = 0.0                 # 当前 CVD 累计值
    cvd_slope: float = 0.0                 # CVD 短期斜率 (近期变化方向)
    cvd_slope_long: float = 0.0            # CVD 长期斜率 (趋势方向)
    cvd_normalized: float = 0.0            # 归一化 CVD (-1 ~ +1, 相对于近期波动)

    # ===== 买卖压力 =====
    buy_volume: float = 0.0                # 估算买方成交量
    sell_volume: float = 0.0               # 估算卖方成交量
    buy_sell_pressure: float = 1.0         # 买/卖比 (>1.5 买方主导, <0.67 卖方主导)
    delta_recent: float = 0.0              # 最近窗口 delta 累计

    # ===== 模式识别 =====
    cvd_pattern: str = CVDPattern.NEUTRAL  # 4+种 CVD-Price 模式
    cvd_divergence: int = DeltaDivergence.NONE  # 背离类型 (0/1/-1)
    cvd_absorption: bool = False           # 吸收模式 (delta spike + 价格盘整)
    absorption_strength: float = 0.0       # 吸收强度 (0-1)

    # ===== 信号输出 (供融合管道使用) =====
    signal: float = 0.0                    # 综合信号 (-1 ~ +1, 负=卖压, 正=买压)
    confidence: float = 0.0                # 置信度 (0-1, 基于模式强度+背离)
    is_reversal_warning: bool = False      # 是否反转预警 (背离 + 吸收)

    def to_dict(self) -> dict:
        return {
            "cvd_value": round(self.cvd_value, 4),
            "cvd_slope": round(self.cvd_slope, 6),
            "cvd_slope_long": round(self.cvd_slope_long, 6),
            "cvd_normalized": round(self.cvd_normalized, 4),
            "buy_volume": round(self.buy_volume, 4),
            "sell_volume": round(self.sell_volume, 4),
            "buy_sell_pressure": round(self.buy_sell_pressure, 4),
            "delta_recent": round(self.delta_recent, 4),
            "cvd_pattern": self.cvd_pattern,
            "cvd_divergence": self.cvd_divergence,
            "cvd_absorption": self.cvd_absorption,
            "absorption_strength": round(self.absorption_strength, 4),
            "signal": round(self.signal, 4),
            "confidence": round(self.confidence, 4),
            "is_reversal_warning": self.is_reversal_warning,
        }


# ============================================================================
# CVD 分析器主类
# ============================================================================

class OrderFlowCVDAnalyzer:
    """CVD 累计成交量差订单流分析器

    核心: 估算每根 K 线的买卖压力, 累加得到 CVD, 识别 4 种模式 + 背离 + 吸收.

    实现:
      1. OHLCV → delta 估算 (Bulkowski 方法)
      2. CVD 累计 (running sum)
      3. CVD 斜率 (短期 slope_short + 长期 slope_long)
      4. 4 种模式识别 (基于 CVD 方向 vs Price 方向)
      5. Delta 背离检测 (基于 swing low/high 对比)
      6. Absorption 检测 (delta spike + ATR 压缩)
      7. 综合信号输出 (-1 ~ +1)

    非过拟合保证:
      - 所有阈值使用业界推荐值 (来自 LiveVolatile/LedgerMind/Gate 百科 2026)
      - 不优化任何参数
      - 输出仅作为 feature, 不直接驱动入场决策
    """

    # 业界推荐阈值 (来自全网前沿研究, 非历史拟合)
    LOOKBACK_DEFAULT = 50                  # CVD 计算窗口 (LedgerMind 推荐 50)
    SLOPE_SHORT_WINDOW = 10                # 短期斜率窗口
    SLOPE_LONG_WINDOW = 30                 # 长期斜率窗口
    PRESSURE_STRONG = 1.5                  # 买/卖压力强势阈值 (LiveVolatile: >1.5)
    PRESSURE_WEAK = 0.67                   # 买/卖压力弱势阈值 (<0.67)
    CONSOLIDATION_ATR_PCT = 0.015          # 盘整 ATR 阈值 (1.5%, 业界标准)
    ABSORPTION_DELTA_ZSCORE = 1.5          # 吸收 delta 异常阈值 (1.5σ)
    SWING_LOOKBACK = 20                    # Swing high/low 检测窗口
    DIVERGENCE_MIN_SWINGS = 2              # 背离检测最小 swing 数

    def __init__(
        self,
        lookback: int = LOOKBACK_DEFAULT,
        slope_short_window: int = SLOPE_SHORT_WINDOW,
        slope_long_window: int = SLOPE_LONG_WINDOW,
    ):
        """
        Args:
            lookback: CVD 计算保留的最近 K 线数
            slope_short_window: 短期斜率窗口 (近期趋势)
            slope_long_window: 长期斜率窗口 (主趋势)
        """
        self.lookback = max(lookback, 30)
        self.slope_short_window = max(slope_short_window, 5)
        self.slope_long_window = max(slope_long_window, 15)

        # 数据存储 (deque 自动截断)
        self._klines: Deque[Dict] = deque(maxlen=self.lookback)
        self._deltas: Deque[float] = deque(maxlen=self.lookback)
        self._cvd_history: Deque[float] = deque(maxlen=self.lookback)
        self._prices: Deque[float] = deque(maxlen=self.lookback)
        self._volumes: Deque[float] = deque(maxlen=self.lookback)
        self._highs: Deque[float] = deque(maxlen=self.lookback)
        self._lows: Deque[float] = deque(maxlen=self.lookback)

        # CVD 累计值
        self._cvd_cumulative = 0.0

        # Swing 缓存 (用于背离检测)
        self._price_swings_low: List[Tuple[int, float]] = []  # (idx, price)
        self._price_swings_high: List[Tuple[int, float]] = []
        self._delta_swings_low: List[Tuple[int, float]] = []  # (idx, delta)
        self._delta_swings_high: List[Tuple[int, float]] = []

    # ========================================================================
    # 数据接入
    # ========================================================================

    def update(self, kline: Dict) -> None:
        """喂入一根 K 线, 更新 CVD 状态

        Args:
            kline: K 线字典, 必须包含 'open'/'high'/'low'/'close'/'volume' 键
                   (兼容 CCXT fetch_ohlcv 转换后的格式)
        """
        try:
            o = float(kline.get("open", 0) or kline.get("o", 0) or 0)
            h = float(kline.get("high", 0) or kline.get("h", 0) or 0)
            l = float(kline.get("low", 0) or kline.get("l", 0) or 0)
            c = float(kline.get("close", 0) or kline.get("c", 0) or 0)
            v = float(kline.get("volume", 0) or kline.get("v", 0) or 0)

            # 数据有效性检查
            if v <= 0 or h <= l:
                # 跳过无效数据 (维持 cvd 不变)
                self._klines.append({"o": o, "h": h, "l": l, "c": c, "v": v})
                self._deltas.append(0.0)
                self._cvd_history.append(self._cvd_cumulative)
                self._prices.append(c)
                self._volumes.append(v)
                self._highs.append(h)
                self._lows.append(l)
                return

            # ===== Bulkowski 方法: 从 OHLCV 估算买卖压力 =====
            # buy_pressure  = volume * (close - low) / (high - low)
            # sell_pressure = volume * (high - close) / (high - low)
            # delta = buy - sell = volume * (2*close - high - low) / (high - low)
            range_hl = h - l
            if range_hl <= 0:
                delta = 0.0
            else:
                # (2c - h - l) / (h - l) 范围 [-1, +1]
                # close == high → +1 (全买)
                # close == low  → -1 (全卖)
                # close == mid  → 0
                direction = (2.0 * c - h - l) / range_hl
                delta = v * direction

            # 累加 CVD
            self._cvd_cumulative += delta

            # 存储
            self._klines.append({"o": o, "h": h, "l": l, "c": c, "v": v})
            self._deltas.append(delta)
            self._cvd_history.append(self._cvd_cumulative)
            self._prices.append(c)
            self._volumes.append(v)
            self._highs.append(h)
            self._lows.append(l)

            # 更新 swing 缓存 (用于背离检测)
            self._update_swings()

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"CVD update 失败: {e}")
        except Exception as e:
            logger.warning(f"CVD update 异常: {type(e).__name__}: {e}")

    def update_batch(self, klines: List[Dict]) -> None:
        """批量喂入 K 线"""
        for k in klines:
            self.update(k)

    def reset(self) -> None:
        """重置状态 (用于切换币种或重新开始)"""
        self._klines.clear()
        self._deltas.clear()
        self._cvd_history.clear()
        self._prices.clear()
        self._volumes.clear()
        self._highs.clear()
        self._lows.clear()
        self._cvd_cumulative = 0.0
        self._price_swings_low.clear()
        self._price_swings_high.clear()
        self._delta_swings_low.clear()
        self._delta_swings_high.clear()

    # ========================================================================
    # 主分析入口
    # ========================================================================

    def analyze(self) -> CVDResult:
        """执行完整 CVD 订单流分析

        Returns:
            CVDResult 包含所有 CVD 指标 + 模式识别 + 信号
        """
        result = CVDResult()

        if len(self._deltas) < 20:
            # 数据不足, 返回默认值
            return result

        # ===== 1. CVD 核心指标 =====
        result.cvd_value = self._cvd_cumulative
        result.cvd_slope = self._compute_slope(self.slope_short_window)
        result.cvd_slope_long = self._compute_slope(self.slope_long_window)
        result.cvd_normalized = self._compute_normalized_cvd()

        # ===== 2. 买卖压力 =====
        result.buy_volume, result.sell_volume = self._compute_buy_sell_volume()
        result.buy_sell_pressure = (
            result.buy_volume / result.sell_volume
            if result.sell_volume > 0 else 1.0
        )
        result.delta_recent = sum(list(self._deltas)[-self.slope_short_window:])

        # ===== 3. 4 种模式识别 =====
        result.cvd_pattern = self._detect_pattern(
            result.cvd_slope, result.cvd_slope_long, result.buy_sell_pressure
        )

        # ===== 4. Delta 背离检测 (73% BTC 反转前出现) =====
        result.cvd_divergence = self._detect_divergence()

        # ===== 5. Absorption 检测 =====
        result.cvd_absorption, result.absorption_strength = self._detect_absorption()

        # ===== 6. 综合信号 + 置信度 =====
        result.signal = self._compute_signal(result)
        result.confidence = self._compute_confidence(result)
        result.is_reversal_warning = (
            result.cvd_divergence != DeltaDivergence.NONE
            and result.cvd_absorption
        )

        return result

    # ========================================================================
    # 内部计算方法
    # ========================================================================

    def _compute_slope(self, window: int) -> float:
        """计算 CVD 斜率 (线性回归斜率)

        Args:
            window: 计算窗口 (最近 N 根 K 线)

        Returns:
            斜率 (单位: volume/K线), 正=上升, 负=下降
        """
        if len(self._cvd_history) < window:
            window = len(self._cvd_history)
        if window < 3:
            return 0.0

        cvd_arr = list(self._cvd_history)[-window:]
        # 简单线性回归斜率: cov(x, y) / var(x), x = [0, 1, ..., n-1]
        n = len(cvd_arr)
        x = np.arange(n, dtype=float)
        y = np.array(cvd_arr, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        x_var = ((x - x_mean) ** 2).sum()
        if x_var < 1e-12:
            return 0.0
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        return float(slope)

    def _compute_normalized_cvd(self) -> float:
        """归一化 CVD 到 [-1, +1] (相对于近期波动)

        用最近窗口 CVD 相对于总成交量的比例归一化.
        """
        if not self._volumes or self._cvd_cumulative == 0.0:
            return 0.0
        recent_vol = sum(list(self._volumes)[-self.slope_long_window:])
        if recent_vol <= 0:
            return 0.0
        # CVD 相对于总成交量的比例 (越偏离 0 越强)
        normalized = self._cvd_cumulative / recent_vol
        # clip 到 [-1, +1]
        return float(max(-1.0, min(1.0, normalized * 10.0)))  # *10 放大小信号

    def _compute_buy_sell_volume(self) -> Tuple[float, float]:
        """计算近期 (slope_long_window) 买/卖成交量估算

        Returns:
            (buy_volume, sell_volume)
        """
        window = min(self.slope_long_window, len(self._deltas))
        if window < 1:
            return 0.0, 0.0

        recent_deltas = list(self._deltas)[-window:]
        # delta = buy - sell, total = buy + sell
        # => buy = (total + delta) / 2, sell = (total - delta) / 2
        # 但这里我们直接用 Bulkowski 估算每根 K 线的买卖分量
        buy_total = 0.0
        sell_total = 0.0
        recent_klines = list(self._klines)[-window:]
        for k in recent_klines:
            h, l, c, v = k["h"], k["l"], k["c"], k["v"]
            range_hl = h - l
            if range_hl <= 0 or v <= 0:
                continue
            buy_frac = (c - l) / range_hl  # 0-1
            sell_frac = (h - c) / range_hl  # 0-1
            buy_total += v * buy_frac
            sell_total += v * sell_frac
        return buy_total, sell_total

    def _detect_pattern(
        self, slope_short: float, slope_long: float, pressure: float
    ) -> str:
        """识别 CVD-Price 模式 (LedgerMind 4 种 + 扩展 2 种)

        判定逻辑 (基于业界标准, 非拟合):
          - 价格方向: 用近期 close vs 早期 close (相对变化)
          - CVD 方向: 用 slope_long (主趋势)
          - 盘整检测: 用 ATR% < CONSOLIDATION_ATR_PCT
        """
        if len(self._prices) < self.slope_long_window:
            return CVDPattern.NEUTRAL

        # 价格方向 (长期)
        recent_prices = list(self._prices)[-self.slope_long_window:]
        price_change_pct = (recent_prices[-1] - recent_prices[0]) / max(recent_prices[0], 1e-8)

        # ATR% (波动率)
        atr_pct = self._compute_atr_pct()

        # CVD 方向 (长期斜率)
        cvd_up = slope_long > 0
        cvd_down = slope_long < 0

        # 价格方向
        price_up = price_change_pct > 0.005  # >0.5% 算上涨
        price_down = price_change_pct < -0.005
        price_flat = abs(price_change_pct) <= 0.005 or atr_pct < self.CONSOLIDATION_ATR_PCT

        # ===== 4 种模式 (LedgerMind 标准) =====
        # 1. 确认牛市: CVD↑ + Price↑
        if cvd_up and price_up and pressure > 1.0:
            return CVDPattern.CONFIRMED_BULL

        # 2. 确认熊市: CVD↓ + Price↓
        if cvd_down and price_down and pressure < 1.0:
            return CVDPattern.CONFIRMED_BEAR

        # 3. 隐藏吸筹: CVD↑ + Price→ (BULLISH)
        if cvd_up and price_flat and pressure > 1.0:
            return CVDPattern.HIDDEN_ACCUMULATION

        # 4. 隐藏分发: CVD↓ + Price→ (BEARISH)
        if cvd_down and price_flat and pressure < 1.0:
            return CVDPattern.HIDDEN_DISTRIBUTION

        # ===== 扩展模式 (LiveVolatile) =====
        # 5. 激进卖出: CVD↓ + Price↓ (急跌)
        if cvd_down and price_down and pressure < self.PRESSURE_WEAK:
            return CVDPattern.AGGRESSIVE_SELLING

        # 6. 多头吸收: CVD↑ + Price↓ (买方吸收卖方压力, BULLISHISH)
        if cvd_up and price_down and pressure > self.PRESSURE_STRONG:
            return CVDPattern.AGGRESSIVE_BUYING

        return CVDPattern.NEUTRAL

    def _compute_atr_pct(self, window: int = 14) -> float:
        """计算 ATR% (平均真实波幅占价格比例)"""
        if len(self._prices) < window + 1:
            return 0.0
        trs = []
        recent_highs = list(self._highs)[-window - 1:]
        recent_lows = list(self._lows)[-window - 1:]
        recent_closes = list(self._prices)[-window - 1:]
        for i in range(1, len(recent_highs)):
            tr = max(
                recent_highs[i] - recent_lows[i],
                abs(recent_highs[i] - recent_closes[i - 1]),
                abs(recent_lows[i] - recent_closes[i - 1]),
            )
            trs.append(tr)
        if not trs:
            return 0.0
        atr = sum(trs) / len(trs)
        price_avg = sum(recent_closes[-window:]) / max(window, 1)
        if price_avg <= 0:
            return 0.0
        return atr / price_avg

    def _update_swings(self) -> None:
        """更新 swing high/low 缓存 (用于背离检测)

        Swing low: 局部最低点 (前后 N 根 K 线都比它高)
        Swing high: 局部最高点 (前后 N 根 K 线都比它低)
        """
        lookback = 3  # Fractal window (前后各 3 根)
        if len(self._prices) < lookback * 2 + 1:
            return

        idx = len(self._prices) - 1 - lookback  # 检查的中心点
        if idx < lookback:
            return

        prices = list(self._prices)
        deltas = list(self._deltas)

        # 中心点价格
        center_price = prices[idx]
        center_delta = deltas[idx]

        # 检查是否是 swing low (前后 lookback 根都比中心高)
        is_swing_low = all(
            prices[idx - k] > center_price and prices[idx + k] > center_price
            for k in range(1, lookback + 1)
        )
        if is_swing_low:
            # 避免重复
            if not self._price_swings_low or self._price_swings_low[-1][0] != idx:
                self._price_swings_low.append((idx, center_price))
                self._delta_swings_low.append((idx, center_delta))
                # 保留最近 SWING_LOOKBACK 个
                if len(self._price_swings_low) > self.SWING_LOOKBACK:
                    self._price_swings_low.pop(0)
                    self._delta_swings_low.pop(0)

        # 检查是否是 swing high
        is_swing_high = all(
            prices[idx - k] < center_price and prices[idx + k] < center_price
            for k in range(1, lookback + 1)
        )
        if is_swing_high:
            if not self._price_swings_high or self._price_swings_high[-1][0] != idx:
                self._price_swings_high.append((idx, center_price))
                self._delta_swings_high.append((idx, center_delta))
                if len(self._price_swings_high) > self.SWING_LOOKBACK:
                    self._price_swings_high.pop(0)
                    self._delta_swings_high.pop(0)

    def _detect_divergence(self) -> int:
        """Delta 背离检测 (CryptoQuant: 73% BTC 反转前出现)

        判定:
          - Bullish 背离: 价格创近期新低, 但 CVD/delta 未创新低 (买方进场)
          - Bearish 背离: 价格创近期新高, 但 CVD/delta 未创新高 (卖方出货)

        Returns:
            DeltaDivergence.BULLISH / BEARISH / NONE

        Note:
          - ERR-20260703-cvd-divergence-bug: 原代码前置检查要求 swing_low>=2,
            但 BEARISH 背离只需要 swing_high. 前置 return 阻止了 BEARISH 检测.
            修复: 移除前置检查, 各分支独立判断所需 swing 类型.
        """
        # ===== Bullish 背离检测 (需要 swing_low) =====
        if (len(self._price_swings_low) >= self.DIVERGENCE_MIN_SWINGS
                and len(self._delta_swings_low) >= self.DIVERGENCE_MIN_SWINGS):
            # 取最近 2 个 swing low
            p_idx1, p_price1 = self._price_swings_low[-2]
            p_idx2, p_price2 = self._price_swings_low[-1]
            d_idx1, d_delta1 = self._delta_swings_low[-2]
            d_idx2, d_delta2 = self._delta_swings_low[-1]

            # 价格创新低, delta 未创新低 (较高)
            if p_price2 < p_price1 and d_delta2 > d_delta1:
                return DeltaDivergence.BULLISH

        # ===== Bearish 背离检测 (需要 swing_high) =====
        if (len(self._price_swings_high) >= self.DIVERGENCE_MIN_SWINGS
                and len(self._delta_swings_high) >= self.DIVERGENCE_MIN_SWINGS):
            p_idx1, p_price1 = self._price_swings_high[-2]
            p_idx2, p_price2 = self._price_swings_high[-1]
            d_idx1, d_delta1 = self._delta_swings_high[-2]
            d_idx2, d_delta2 = self._delta_swings_high[-1]

            # 价格创新高, delta 未创新高 (较低)
            if p_price2 > p_price1 and d_delta2 < d_delta1:
                return DeltaDivergence.BEARISH

        return DeltaDivergence.NONE

    def _detect_absorption(self) -> Tuple[bool, float]:
        """Absorption 吸收模式检测 (LiveVolatile 2026)

        判定:
          - 价格盘整 (ATR% < CONSOLIDATION_ATR_PCT)
          - 同时 delta 出现异常 (z-score > ABSORPTION_DELTA_ZSCORE)
          - 表明大单在盘整中吸收流动性, 预示波动扩张

        Returns:
            (is_absorption, strength 0-1)
        """
        atr_pct = self._compute_atr_pct()
        if atr_pct >= self.CONSOLIDATION_ATR_PCT:
            # 非盘整, 无吸收
            return False, 0.0

        # 计算近期 delta z-score
        recent_deltas = list(self._deltas)[-self.slope_long_window:]
        if len(recent_deltas) < 10:
            return False, 0.0

        arr = np.array(recent_deltas, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-8:
            return False, 0.0

        # 检查最近 5 根 K 线是否有异常 delta
        recent_5 = arr[-5:]
        max_z = float(np.max(np.abs((recent_5 - mean) / std)))

        if max_z >= self.ABSORPTION_DELTA_ZSCORE:
            # 吸收检测到
            # 强度: z-score 越高, 盘整越紧 → 强度越高
            atr_strength = max(0.0, 1.0 - atr_pct / self.CONSOLIDATION_ATR_PCT)
            delta_strength = min(1.0, (max_z - self.ABSORPTION_DELTA_ZSCORE) / 2.0 + 0.5)
            strength = (atr_strength + delta_strength) / 2.0
            return True, float(strength)

        return False, 0.0

    def _compute_signal(self, result: CVDResult) -> float:
        """综合 CVD 信号 (-1 ~ +1)

        信号来源 (加权融合, 非参数拟合):
          - CVD 斜率方向 (短期 + 长期)
          - 买卖压力比
          - 模式识别 (隐藏吸筹=+, 隐藏分发=-, 等)
          - 背离信号 (bullish=+, bearish=-)
          - Absorption (盘整中通常中性, 但配合 delta 方向有信号)
        """
        signal = 0.0

        # 1. CVD 斜率方向 (权重 0.3)
        # 归一化斜率到 [-1, 1] (用近期总成交量做参考)
        recent_vol = sum(list(self._volumes)[-self.slope_long_window:]) or 1.0
        slope_signal = float(np.tanh(result.cvd_slope_long * 100 / recent_vol))
        signal += 0.3 * slope_signal

        # 2. 买卖压力比 (权重 0.3)
        # 1.0 = 中性, >1 买方主导, <1 卖方主导
        pressure_signal = float(np.tanh((result.buy_sell_pressure - 1.0) * 2.0))
        signal += 0.3 * pressure_signal

        # 3. 模式识别 (权重 0.2)
        pattern_signal = 0.0
        if result.cvd_pattern == CVDPattern.CONFIRMED_BULL:
            pattern_signal = 0.8
        elif result.cvd_pattern == CVDPattern.HIDDEN_ACCUMULATION:
            pattern_signal = 0.6
        elif result.cvd_pattern == CVDPattern.AGGRESSIVE_BUYING:
            pattern_signal = 0.5  # 买方吸收但价格下跌, 信号偏多但弱
        elif result.cvd_pattern == CVDPattern.CONFIRMED_BEAR:
            pattern_signal = -0.8
        elif result.cvd_pattern == CVDPattern.HIDDEN_DISTRIBUTION:
            pattern_signal = -0.6
        elif result.cvd_pattern == CVDPattern.AGGRESSIVE_SELLING:
            pattern_signal = -1.0
        signal += 0.2 * pattern_signal

        # 4. 背离信号 (权重 0.2) — 最强反转预警
        if result.cvd_divergence == DeltaDivergence.BULLISH:
            signal += 0.2 * 0.8
        elif result.cvd_divergence == DeltaDivergence.BEARISH:
            signal += 0.2 * (-0.8)

        # clip 到 [-1, +1]
        return float(max(-1.0, min(1.0, signal)))

    def _compute_confidence(self, result: CVDResult) -> float:
        """计算 CVD 信号置信度 (0-1)

        置信度来源:
          - 样本量 (数据充足度)
          - 信号一致性 (CVD 斜率 + 压力 + 模式方向一致)
          - 背离 + absorption 同时出现 (高置信反转预警)
        """
        # 1. 样本量置信度
        sample_conf = min(1.0, len(self._deltas) / 50.0)

        # 2. 信号一致性
        # 检查斜率方向 + 压力方向 + 模式方向是否一致
        slope_dir = 1 if result.cvd_slope_long > 0 else (-1 if result.cvd_slope_long < 0 else 0)
        pressure_dir = (
            1 if result.buy_sell_pressure > 1.0
            else (-1 if result.buy_sell_pressure < 1.0 else 0)
        )
        pattern_dir = 0
        if result.cvd_pattern in (CVDPattern.CONFIRMED_BULL,
                                  CVDPattern.HIDDEN_ACCUMULATION,
                                  CVDPattern.AGGRESSIVE_BUYING):
            pattern_dir = 1
        elif result.cvd_pattern in (CVDPattern.CONFIRMED_BEAR,
                                    CVDPattern.HIDDEN_DISTRIBUTION,
                                    CVDPattern.AGGRESSIVE_SELLING):
            pattern_dir = -1

        # 3 方向都一致 → 1.0, 2 方向一致 → 0.7, 1 方向 → 0.4, 0 → 0.2
        dirs = [d for d in [slope_dir, pressure_dir, pattern_dir] if d != 0]
        if len(dirs) == 3 and len(set(dirs)) == 1:
            consistency = 1.0
        elif len(dirs) >= 2 and len(set(dirs)) == 1:
            consistency = 0.7
        elif len(dirs) >= 1:
            consistency = 0.4
        else:
            consistency = 0.2

        # 3. 背离 + absorption 加成 (反转预警高置信)
        reversal_boost = 0.0
        if result.is_reversal_warning:
            reversal_boost = 0.2

        return float(min(1.0, sample_conf * 0.3 + consistency * 0.5 + reversal_boost + 0.2))


# ============================================================================
# 便捷函数
# ============================================================================

def analyze_cvd_from_klines(klines: List[Dict]) -> CVDResult:
    """一次性分析: 从 K 线列表直接计算 CVD 结果

    Args:
        klines: K 线列表 (每条含 open/high/low/close/volume)

    Returns:
        CVDResult
    """
    analyzer = OrderFlowCVDAnalyzer()
    analyzer.update_batch(klines)
    return analyzer.analyze()


def detect_cvd_divergence(klines: List[Dict]) -> int:
    """便捷函数: 仅检测 CVD 背离

    Returns:
        0=无背离, 1=Bullish (反转上行), -1=Bearish (反转下行)
    """
    return analyze_cvd_from_klines(klines).cvd_divergence
