# -*- coding: utf-8 -*-
"""
加密货币日内交易强化模块 (Crypto Intraday Enhancement)

基于2025-2026前沿数字货币日内交易最佳实践：
  - VWAP（成交量加权平均价）+ 标准差带（来自Solyzer/Obside实战指南）
  - 资金费率驱动策略（来自Thrive.fi衍生品交易手册）
  - 清算级联检测器 LVI（来自LiveVolatile 2026年2月$28亿清算事件分析）
  - 未平仓合约(OI)背离分析（来自Thrive.fi）
  - 日内时段流动性感知（24/7市场的UTC时段特性）
  - ATR-based动态止损（替代固定百分比，来自Obside风险管理）

核心组件：
  1. VWAPCalculator — VWAP + 标准差带（1x/2x/3x）
  2. FundingRateSimulator — 永续合约资金费率模拟
  3. LiquidationCascadeDetector — 清算级联检测（LVI指标）
  4. OpenInterestTracker — 未平仓合约跟踪 + 背离检测
  5. IntradaySessionProfile — 日内时段流动性画像
  6. ATRStopCalculator — ATR-based动态止损
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.crypto_intraday")


# ============================================================================
# 1. VWAP计算器（成交量加权平均价 + 标准差带）
# ============================================================================


class VWAPCalculator:
    """VWAP计算器 — 日内交易的核心指标

    来自Solyzer 2026 VWAP策略指南：
      - VWAP = Sum(Price * Volume) / Sum(Volume)
      - 标准差带：1x/2x/3x，识别超买超卖区域
      - 24小时VWAP从UTC午夜重置
      - 价格偏离VWAP 2+标准差 = 均值回归信号

    策略应用：
      - VWAP Mean Reversion：价格偏离2+σ时入场，目标回归VWAP
      - VWAP Pullback in Trend：趋势中回踩VWAP时入场
      - VWAP as Support/Resistance：上升趋势中VWAP为支撑，下降趋势中为阻力
    """

    def __init__(self, session_reset_hours: int = 24):
        """初始化VWAP计算器

        Args:
            session_reset_hours: VWAP重置周期（小时），默认24小时
        """
        self._session_reset_sec = session_reset_hours * 3600
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def update(self, symbol: str, price: float, volume: float,
               timestamp: Optional[float] = None) -> Dict[str, float]:
        """更新VWAP计算

        Args:
            symbol: 交易对
            price: 当前价格
            volume: 当前成交量
            timestamp: 时间戳（默认当前时间）

        Returns:
            {"vwap": float, "upper_1": float, "lower_1": float,
             "upper_2": float, "lower_2": float, "upper_3": float, "lower_3": float}
        """
        ts = timestamp or time.time()

        if symbol not in self._sessions:
            self._sessions[symbol] = {
                "start_ts": ts,
                "cum_pv": 0.0,  # 累计价格×成交量
                "cum_vol": 0.0,  # 累计成交量
                "prices": [],  # 价格历史（用于计算标准差）
                "volumes": [],
            }

        session = self._sessions[symbol]

        # 检查是否需要重置（超过会话周期）
        if ts - session["start_ts"] > self._session_reset_sec:
            session["start_ts"] = ts
            session["cum_pv"] = 0.0
            session["cum_vol"] = 0.0
            session["prices"] = []
            session["volumes"] = []

        # 更新累计值
        session["cum_pv"] += price * volume
        session["cum_vol"] += volume
        session["prices"].append(price)
        session["volumes"].append(volume)

        # 保持窗口大小（最近1000根K线）
        if len(session["prices"]) > 1000:
            session["prices"] = session["prices"][-1000:]
            session["volumes"] = session["volumes"][-1000:]

        # 计算VWAP
        if session["cum_vol"] > 0:
            vwap = session["cum_pv"] / session["cum_vol"]
        else:
            vwap = price

        # 计算成交量加权标准差
        if len(session["prices"]) > 2:
            prices = session["prices"]
            volumes = session["volumes"]
            total_vol = sum(volumes)
            if total_vol > 0:
                weighted_mean = sum(p * v for p, v in zip(prices, volumes)) / total_vol
                variance = sum(v * (p - weighted_mean) ** 2 for p, v in zip(prices, volumes)) / total_vol
                std = math.sqrt(max(variance, 0))
            else:
                std = 0.0
        else:
            std = 0.0

        return {
            "vwap": vwap,
            "upper_1": vwap + 1 * std,
            "lower_1": vwap - 1 * std,
            "upper_2": vwap + 2 * std,
            "lower_2": vwap - 2 * std,
            "upper_3": vwap + 3 * std,
            "lower_3": vwap - 3 * std,
            "std_dev": std,
        }

    def get_signal(self, symbol: str, price: float) -> Dict[str, Any]:
        """获取VWAP交易信号

        返回基于VWAP的均值回归和趋势信号：
          - "vwap_reversion_long": 价格低于lower_2，做多（回归VWAP）
          - "vwap_reversion_short": 价格高于upper_2，做空（回归VWAP）
          - "vwap_support": 价格在VWAP上方，VWAP为支撑
          - "vwap_resistance": 价格在VWAP下方，VWAP为阻力
          - "neutral": 价格在1σ范围内
        """
        session = self._sessions.get(symbol)
        if not session or session["cum_vol"] == 0:
            return {"signal": "neutral", "strength": 0.0, "vwap": price}

        vwap = session["cum_pv"] / session["cum_vol"]
        prices = session["prices"]
        volumes = session["volumes"]

        if len(prices) < 2:
            return {"signal": "neutral", "strength": 0.0, "vwap": vwap}

        total_vol = sum(volumes)
        if total_vol == 0:
            return {"signal": "neutral", "strength": 0.0, "vwap": vwap}

        weighted_mean = sum(p * v for p, v in zip(prices, volumes)) / total_vol
        variance = sum(v * (p - weighted_mean) ** 2 for p, v in zip(prices, volumes)) / total_vol
        std = math.sqrt(max(variance, 0))

        if std == 0:
            return {"signal": "neutral", "strength": 0.0, "vwap": vwap}

        # Z-score: 价格偏离VWAP的标准差倍数
        z_score = (price - vwap) / std

        # 均值回归信号（偏离1.5+σ，降低阈值增加交易频率）
        if z_score < -1.5:
            strength = min(1.0, abs(z_score) / 3.0)
            return {"signal": "vwap_reversion_long", "strength": strength,
                    "vwap": vwap, "z_score": z_score}
        elif z_score > 1.5:
            strength = min(1.0, abs(z_score) / 3.0)
            return {"signal": "vwap_reversion_short", "strength": strength,
                    "vwap": vwap, "z_score": z_score}

        # 趋势信号
        if price > vwap:
            return {"signal": "vwap_support", "strength": min(1.0, z_score / 2.0),
                    "vwap": vwap, "z_score": z_score}
        elif price < vwap:
            return {"signal": "vwap_resistance", "strength": min(1.0, abs(z_score) / 2.0),
                    "vwap": vwap, "z_score": z_score}

        return {"signal": "neutral", "strength": 0.0, "vwap": vwap, "z_score": z_score}


# ============================================================================
# 2. 资金费率模拟器（Funding Rate Simulator）
# ============================================================================


@dataclass(slots=True)
class FundingRateState:
    """资金费率状态"""
    current_rate: float = 0.0  # 当前资金费率（每8小时）
    cumulative_rate: float = 0.0  # 累计资金费率
    next_funding_ts: float = 0.0  # 下次资金费率结算时间
    rate_history: List[float] = field(default_factory=list)
    extreme_count: int = 0  # 极端费率连续次数（>0.05%）


class FundingRateSimulator:
    """永续合约资金费率模拟器

    来自Thrive.fi 2026衍生品交易手册：
      - 资金费率>0.05%/8h = 极端，市场过度做多，寻找做空机会
      - 资金费率<-0.05%/8h = 极端，市场过度做空，寻找做多机会
      - 资金费率翻转可以移动现货价格
      - 极端费率持续多个周期 = 高概率均值回归

    策略应用：
      - Funding-driven contrarian：费率>0.05%+失败反弹 → 做空
      - Funding rate fade：极端费率预测均值回归
    """

    FUNDING_INTERVAL_SEC = 8 * 3600  # 8小时结算一次
    EXTREME_THRESHOLD = 0.0005  # 0.05% = 极端
    MAX_RATE = 0.001  # 0.1% 上限
    MIN_RATE = -0.001  # -0.1% 下限

    def __init__(self):
        self._states: Dict[str, FundingRateState] = {}

    def update(self, symbol: str, price_change: float, timestamp: Optional[float] = None) -> float:
        """更新资金费率

        资金费率受市场情绪和价格变动影响：
          - 价格上涨 → 多头支付 → 费率变正
          - 价格下跌 → 空头支付 → 费率变负

        Args:
            symbol: 交易对
            price_change: 价格变动百分比
            timestamp: 时间戳

        Returns:
            当前资金费率
        """
        ts = timestamp or time.time()

        if symbol not in self._states:
            self._states[symbol] = FundingRateState(
                current_rate=0.0,
                next_funding_ts=ts + self.FUNDING_INTERVAL_SEC,
            )

        state = self._states[symbol]

        # 检查是否到结算时间
        if ts >= state.next_funding_ts:
            # 结算资金费率
            state.cumulative_rate += state.current_rate
            state.rate_history.append(state.current_rate)
            if len(state.rate_history) > 100:
                state.rate_history = state.rate_history[-100:]
            state.next_funding_ts = ts + self.FUNDING_INTERVAL_SEC

            # 更新极端费率计数
            if abs(state.current_rate) > self.EXTREME_THRESHOLD:
                state.extreme_count += 1
            else:
                state.extreme_count = 0

        # 资金费率跟随价格变动（带均值回归）
        # 费率 = 0.7 * 前值 + 0.3 * (价格变动 * 缩放因子) + 噪声
        target_rate = price_change * 0.5  # 价格变动影响
        target_rate = max(self.MIN_RATE, min(self.MAX_RATE, target_rate))

        # 均值回归（费率倾向于回归0）
        mean_reversion = -state.current_rate * 0.1
        noise = random.gauss(0, 0.0001)  # 小随机扰动

        state.current_rate = (0.7 * state.current_rate +
                              0.3 * target_rate +
                              mean_reversion +
                              noise)
        state.current_rate = max(self.MIN_RATE, min(self.MAX_RATE, state.current_rate))

        return state.current_rate

    def get_signal(self, symbol: str) -> Dict[str, Any]:
        """获取资金费率交易信号

        Returns:
            {"signal": "funding_short"/"funding_long"/"neutral",
             "rate": float, "extreme_count": int, "strength": float}
        """
        state = self._states.get(symbol)
        if not state:
            return {"signal": "neutral", "rate": 0.0, "extreme_count": 0, "strength": 0.0}

        # 极端正费率 → 做空信号（市场过度做多）
        if state.current_rate > self.EXTREME_THRESHOLD:
            strength = min(1.0, state.current_rate / self.MAX_RATE)
            return {"signal": "funding_short", "rate": state.current_rate,
                    "extreme_count": state.extreme_count, "strength": strength}

        # 极端负费率 → 做多信号（市场过度做空）
        if state.current_rate < -self.EXTREME_THRESHOLD:
            strength = min(1.0, abs(state.current_rate) / self.MAX_RATE)
            return {"signal": "funding_long", "rate": state.current_rate,
                    "extreme_count": state.extreme_count, "strength": strength}

        return {"signal": "neutral", "rate": state.current_rate,
                "extreme_count": state.extreme_count, "strength": 0.0}


# ============================================================================
# 3. 清算级联检测器（Liquidation Cascade Detector）
# ============================================================================


@dataclass(slots=True)
class LiquidationCluster:
    """清算集群"""
    price_level: float
    side: str  # "long" or "short"
    size: float  # 清算量（美元）
    distance_pct: float  # 距当前价格百分比


class LiquidationCascadeDetector:
    """清算级联检测器 — LVI指标

    来自LiveVolatile 2026年2月$28亿清算事件分析：
      LVI = (多头清算总量 / 未平仓合约) × 波动率乘数

      LVI < 15: 低风险
      LVI 15-30: 中等风险
      LVI 30-50: 高风险（24小时内可能级联）
      LVI > 50: 极端风险（级联迫在眉睫）

    关键发现：
      - 杠杆超过未平仓合约60%时，48小时内级联概率>78%
      - 清算集群是价格磁铁
      - 级联产生V型恢复
    """

    def __init__(self):
        self._clusters: Dict[str, List[LiquidationCluster]] = {}
        self._open_interest: Dict[str, float] = {}
        self._lvi_history: Dict[str, List[float]] = {}

    def update_open_interest(self, symbol: str, oi: float) -> None:
        """更新未平仓合约"""
        self._open_interest[symbol] = oi

    def update_clusters(self, symbol: str, current_price: float,
                        long_leverage_ratio: float = 0.6,
                        short_leverage_ratio: float = 0.4) -> List[LiquidationCluster]:
        """更新清算集群

        模拟杠杆仓位在不同价格水平的清算分布

        Args:
            symbol: 交易对
            current_price: 当前价格
            long_leverage_ratio: 多头杠杆占比（默认60%）
            short_leverage_ratio: 空头杠杆占比（默认40%）
        """
        oi = self._open_interest.get(symbol, 0)
        if oi <= 0:
            return []

        clusters = []

        # 多头清算集群（价格下方）
        # 典型杠杆：5x, 10x, 25x, 50x, 100x
        long_leverages = [(5, 0.20), (10, 0.30), (25, 0.25), (50, 0.15), (100, 0.10)]
        for leverage, fraction in long_leverages:
            # 清算价格 ≈ 当前价格 * (1 - 1/杠杆)
            liq_price = current_price * (1 - 1.0 / leverage)
            size = oi * long_leverage_ratio * fraction
            distance = (current_price - liq_price) / current_price
            clusters.append(LiquidationCluster(
                price_level=liq_price,
                side="long",
                size=size,
                distance_pct=distance,
            ))

        # 空头清算集群（价格上方）
        short_leverages = [(5, 0.20), (10, 0.30), (25, 0.25), (50, 0.15), (100, 0.10)]
        for leverage, fraction in short_leverages:
            liq_price = current_price * (1 + 1.0 / leverage)
            size = oi * short_leverage_ratio * fraction
            distance = (liq_price - current_price) / current_price
            clusters.append(LiquidationCluster(
                price_level=liq_price,
                side="short",
                size=size,
                distance_pct=distance,
            ))

        self._clusters[symbol] = clusters
        return clusters

    def calculate_lvi(self, symbol: str, volatility: float) -> float:
        """计算Liquidation Volatility Index (LVI)

        LVI = (多头清算总量 / 未平仓合约) × 波动率乘数

        Args:
            symbol: 交易对
            volatility: 当前波动率（ATR/价格）

        Returns:
            LVI值
        """
        clusters = self._clusters.get(symbol, [])
        oi = self._open_interest.get(symbol, 0)

        if oi <= 0 or not clusters:
            return 0.0

        # 多头清算总量（价格下方5%内）
        current_price = clusters[0].price_level if clusters else 0
        long_liq_total = sum(
            c.size for c in clusters
            if c.side == "long" and c.distance_pct < 0.05
        )

        # LVI = (多头清算总量 / OI) × 波动率乘数
        # 波动率乘数 = 波动率 * 100（将0.04的波动率转换为4）
        vol_multiplier = max(1.0, volatility * 100)
        lvi = (long_liq_total / oi) * vol_multiplier * 100

        # 记录历史
        if symbol not in self._lvi_history:
            self._lvi_history[symbol] = []
        self._lvi_history[symbol].append(lvi)
        if len(self._lvi_history[symbol]) > 100:
            self._lvi_history[symbol] = self._lvi_history[symbol][-100:]

        return lvi

    def get_risk_level(self, symbol: str, volatility: float) -> Dict[str, Any]:
        """获取清算级联风险等级

        Returns:
            {"lvi": float, "level": "low"/"moderate"/"high"/"extreme",
             "cascade_probability": float, "nearest_cluster": LiquidationCluster}
        """
        lvi = self.calculate_lvi(symbol, volatility)

        if lvi < 15:
            level = "low"
            probability = 0.05
        elif lvi < 30:
            level = "moderate"
            probability = 0.25
        elif lvi < 50:
            level = "high"
            probability = 0.60
        else:
            level = "extreme"
            probability = 0.90

        # 找最近的清算集群
        clusters = self._clusters.get(symbol, [])
        nearest = min(clusters, key=lambda c: c.distance_pct) if clusters else None

        return {
            "lvi": lvi,
            "level": level,
            "cascade_probability": probability,
            "nearest_cluster": nearest,
            "signal": "cascade_warning" if lvi > 30 else "normal",
        }


# ============================================================================
# 4. 未平仓合约跟踪器（Open Interest Tracker）
# ============================================================================


class OpenInterestTracker:
    """未平仓合约跟踪器 — OI背离检测

    来自Thrive.fi 2026衍生品交易手册：
      - OI上升 + 价格上升 = 新多头开仓，趋势健康
      - OI上升 + 价格下降 = 新空头开仓，下跌趋势健康
      - OI下降 + 价格上升 = 空头回补，上涨趋势疲软（背离）
      - OI下降 + 价格下降 = 多头平仓，下跌趋势疲软（背离）

    OI背离是最可靠的反转信号之一
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._oi_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, oi)
        self._price_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, price)

    def update(self, symbol: str, oi: float, price: float,
               timestamp: Optional[float] = None) -> None:
        """更新OI和价格历史"""
        ts = timestamp or time.time()

        if symbol not in self._oi_history:
            self._oi_history[symbol] = []
            self._price_history[symbol] = []

        self._oi_history[symbol].append((ts, oi))
        self._price_history[symbol].append((ts, price))

        if len(self._oi_history[symbol]) > self._window:
            self._oi_history[symbol] = self._oi_history[symbol][-self._window:]
            self._price_history[symbol] = self._price_history[symbol][-self._window:]

    def detect_divergence(self, symbol: str, lookback: int = 20) -> Dict[str, Any]:
        """检测OI背离

        Returns:
            {"divergence": "bullish"/"bearish"/"none",
             "strength": float, "description": str}
        """
        oi_hist = self._oi_history.get(symbol, [])
        price_hist = self._price_history.get(symbol, [])

        if len(oi_hist) < lookback or len(price_hist) < lookback:
            return {"divergence": "none", "strength": 0.0, "description": "insufficient data"}

        # 取lookback周期的数据
        recent_oi = oi_hist[-lookback:]
        recent_price = price_hist[-lookback:]

        # 计算趋势（线性回归斜率简化版）
        oi_start = recent_oi[0][1]
        oi_end = recent_oi[-1][1]
        price_start = recent_price[0][1]
        price_end = recent_price[-1][1]

        oi_change = (oi_end - oi_start) / oi_start if oi_start > 0 else 0
        price_change = (price_end - price_start) / price_start if price_start > 0 else 0

        # 背离检测
        # OI下降 + 价格上升 = 看跌背离（空头回补，趋势疲软）
        if oi_change < -0.02 and price_change > 0.02:
            strength = min(1.0, abs(oi_change) + price_change)
            return {"divergence": "bearish", "strength": strength,
                    "description": f"OI下降{oi_change:.1%}但价格上升{price_change:.1%}，空头回补，上涨趋势疲软"}

        # OI下降 + 价格下降 = 看涨背离（多头平仓，下跌趋势疲软）
        if oi_change < -0.02 and price_change < -0.02:
            strength = min(1.0, abs(oi_change) + abs(price_change))
            return {"divergence": "bullish", "strength": strength,
                    "description": f"OI下降{oi_change:.1%}且价格下降{price_change:.1%}，多头平仓，下跌趋势疲软"}

        return {"divergence": "none", "strength": 0.0,
                "description": f"OI变化{oi_change:.1%}，价格变化{price_change:.1%}，趋势健康"}


# ============================================================================
# 5. 日内时段流动性画像（Intraday Session Profile）
# ============================================================================


class IntradaySessionProfile:
    """日内时段流动性画像 — 24/7市场的UTC时段特性

    来自Obside 2026日内交易手册：
      - 09:00 UTC: 欧洲开盘，流动性上升
      - 13:30 UTC: 美国开盘，流动性最高（美欧重叠）
      - 21:00 UTC: 美国收盘，流动性下降
      - 00:00-08:00 UTC: 亚洲时段，流动性较低（但周末更低）

    流动性影响：
      - 高流动性时段：点差小，滑点低，适合突破策略
      - 低流动性时段：点差大，滑点高，适合均值回归
    """

    # 时段定义（UTC小时）
    SESSIONS = {
        "asian": (0, 8),      # 亚洲时段：00:00-08:00 UTC
        "european": (8, 13),  # 欧洲时段：08:00-13:00 UTC
        "overlap": (13, 17),  # 美欧重叠：13:00-17:00 UTC（流动性最高）
        "us": (17, 21),       # 美国时段：17:00-21:00 UTC
        "late": (21, 24),     # 晚间时段：21:00-24:00 UTC
    }

    # 时段流动性乘数（基于历史交易量数据）
    LIQUIDITY_MULTIPLIER = {
        "asian": 0.7,
        "european": 1.0,
        "overlap": 1.5,  # 美欧重叠流动性最高
        "us": 1.2,
        "late": 0.8,
    }

    # 时段波动率乘数
    VOLATILITY_MULTIPLIER = {
        "asian": 0.8,
        "european": 1.0,
        "overlap": 1.3,  # 重叠时段波动率也高
        "us": 1.1,
        "late": 0.9,
    }

    def get_session(self, timestamp: Optional[float] = None) -> str:
        """获取当前时段

        Args:
            timestamp: 时间戳（默认当前时间）

        Returns:
            时段名称：asian/european/overlap/us/late
        """
        ts = timestamp or time.time()
        hour_utc = (ts % 86400) // 3600  # UTC小时

        for session, (start, end) in self.SESSIONS.items():
            if start <= hour_utc < end:
                return session

        return "asian"  # 默认

    def get_liquidity(self, timestamp: Optional[float] = None) -> float:
        """获取当前流动性乘数"""
        session = self.get_session(timestamp)
        return self.LIQUIDITY_MULTIPLIER.get(session, 1.0)

    def get_volatility(self, timestamp: Optional[float] = None) -> float:
        """获取当前波动率乘数"""
        session = self.get_session(timestamp)
        return self.VOLATILITY_MULTIPLIER.get(session, 1.0)

    def get_profile(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """获取完整时段画像"""
        session = self.get_session(timestamp)
        return {
            "session": session,
            "liquidity": self.LIQUIDITY_MULTIPLIER.get(session, 1.0),
            "volatility": self.VOLATILITY_MULTIPLIER.get(session, 1.0),
            "is_high_liquidity": session in ("overlap", "us"),
            "is_low_liquidity": session in ("asian", "late"),
            "suitable_for_breakout": session in ("overlap", "us"),
            "suitable_for_mean_reversion": session in ("asian", "late"),
        }


# ============================================================================
# 6. ATR-based动态止损计算器
# ============================================================================


class ATRStopCalculator:
    """ATR-based动态止损计算器

    来自Obside 2026风险管理最佳实践：
      - ATR-based止损，不用固定百分比
      - 2%的止损在BTC不同波动率下含义不同
      - 标准方法：止损 = 入场价 ± (ATR × 乘数)

    常用乘数：
      - 保守：1.5x ATR
      - 标准：2.0x ATR
      - 激进：3.0x ATR
    """

    @staticmethod
    def calculate_stop(entry_price: float, atr: float, direction: str,
                       multiplier: float = 2.0) -> float:
        """计算ATR-based止损

        Args:
            entry_price: 入场价格
            atr: 当前ATR值
            direction: "long" or "short"
            multiplier: ATR乘数（默认2.0）

        Returns:
            止损价格
        """
        if atr <= 0:
            # ATR无效时回退到固定百分比
            atr = entry_price * 0.02

        stop_distance = atr * multiplier

        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    @staticmethod
    def calculate_take_profit(entry_price: float, atr: float, direction: str,
                               risk_reward_ratio: float = 2.0,
                               multiplier: float = 2.0) -> float:
        """计算ATR-based止盈

        Args:
            entry_price: 入场价格
            atr: 当前ATR值
            direction: "long" or "short"
            risk_reward_ratio: 风险回报比（默认2:1）
            multiplier: ATR乘数（用于计算止损距离）

        Returns:
            止盈价格
        """
        if atr <= 0:
            atr = entry_price * 0.02

        stop_distance = atr * multiplier
        tp_distance = stop_distance * risk_reward_ratio

        if direction == "long":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    @staticmethod
    def calculate_position_size(account_equity: float, entry_price: float,
                                 atr: float, risk_pct: float = 0.005,
                                 multiplier: float = 2.0) -> float:
        """计算基于ATR的仓位大小

        来自Obside风险管理：加密货币每笔交易风险0.25-0.5%（不是1-2%）

        Args:
            account_equity: 账户权益
            entry_price: 入场价格
            atr: 当前ATR值
            risk_pct: 每笔风险百分比（默认0.5%）
            multiplier: ATR乘数

        Returns:
            仓位数量
        """
        if atr <= 0 or entry_price <= 0:
            return 0.0

        risk_amount = account_equity * risk_pct
        stop_distance = atr * multiplier
        quantity = risk_amount / stop_distance

        return quantity


# ============================================================================
# 7. 加密货币日内交易强化系统（统一管理）
# ============================================================================


class CryptoIntradaySystem:
    """加密货币日内交易强化系统 — 统一管理所有日内交易组件

    集成：
      - VWAPCalculator（VWAP + 标准差带）
      - FundingRateSimulator（资金费率）
      - LiquidationCascadeDetector（清算级联LVI）
      - OpenInterestTracker（OI背离）
      - IntradaySessionProfile（时段流动性）
      - ATRStopCalculator（ATR动态止损）
    """

    def __init__(self):
        self.vwap = VWAPCalculator()
        self.funding = FundingRateSimulator()
        self.cascade = LiquidationCascadeDetector()
        self.oi_tracker = OpenInterestTracker()
        self.session = IntradaySessionProfile()
        self.atr_stop = ATRStopCalculator

        logger.info("CryptoIntradaySystem initialized (2026 best practices)")

    def update(self, symbol: str, price: float, volume: float,
               price_change: float, atr: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """更新所有日内交易指标

        Args:
            symbol: 交易对
            price: 当前价格
            volume: 当前成交量
            price_change: 价格变动百分比
            atr: 当前ATR值
            timestamp: 时间戳

        Returns:
            所有指标的当前状态
        """
        ts = timestamp or time.time()

        # 1. 更新VWAP
        vwap_data = self.vwap.update(symbol, price, volume, ts)

        # 2. 更新资金费率
        funding_rate = self.funding.update(symbol, price_change, ts)

        # 3. 更新OI（模拟：OI与价格负相关，价格涨OI降=背离）
        base_oi = 1e9  # 基准OI 10亿美元
        oi_change = -price_change * 0.5 + random.gauss(0, 0.02)  # OI变化
        current_oi = base_oi * (1 + oi_change)
        self.cascade.update_open_interest(symbol, current_oi)
        self.oi_tracker.update(symbol, current_oi, price, ts)

        # 4. 更新清算集群
        self.cascade.update_clusters(symbol, price)

        # 5. 计算LVI
        volatility = atr / price if price > 0 else 0
        cascade_risk = self.cascade.get_risk_level(symbol, volatility)

        # 6. 获取时段画像
        session_profile = self.session.get_profile(ts)

        # 7. 检测OI背离
        oi_divergence = self.oi_tracker.detect_divergence(symbol)

        return {
            "vwap": vwap_data,
            "funding_rate": funding_rate,
            "funding_signal": self.funding.get_signal(symbol),
            "cascade_risk": cascade_risk,
            "session": session_profile,
            "oi_divergence": oi_divergence,
            "vwap_signal": self.vwap.get_signal(symbol, price),
        }

    def get_trading_signals(self, symbol: str, price: float) -> Dict[str, Any]:
        """获取所有日内交易信号

        综合所有指标给出交易建议
        """
        vwap_sig = self.vwap.get_signal(symbol, price)
        funding_sig = self.funding.get_signal(symbol)
        session_profile = self.session.get_profile()
        oi_divergence = self.oi_tracker.detect_divergence(symbol)

        # 综合信号强度
        signals = []
        total_strength = 0.0

        # VWAP信号
        if vwap_sig["signal"] != "neutral":
            signals.append({
                "source": "vwap",
                "signal": vwap_sig["signal"],
                "strength": vwap_sig["strength"],
            })
            total_strength += vwap_sig["strength"]

        # 资金费率信号
        if funding_sig["signal"] != "neutral":
            signals.append({
                "source": "funding",
                "signal": funding_sig["signal"],
                "strength": funding_sig["strength"],
            })
            total_strength += funding_sig["strength"]

        # OI背离信号
        if oi_divergence["divergence"] != "none":
            signals.append({
                "source": "oi_divergence",
                "signal": oi_divergence["divergence"],
                "strength": oi_divergence["strength"],
            })
            total_strength += oi_divergence["strength"]

        return {
            "signals": signals,
            "total_strength": min(1.0, total_strength),
            "session": session_profile["session"],
            "is_high_liquidity": session_profile["is_high_liquidity"],
            "signal_count": len(signals),
        }
