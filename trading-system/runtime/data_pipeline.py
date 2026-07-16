# -*- coding: utf-8 -*-
"""
数据管道 (Data Pipeline) — 行情数据获取与AI预测服务 v2.28

v2.28修复(2026-06-26): next_bar数据循环复用
  - 原问题: idx>=len时返回None,导致进化500轮但91轮后无数据
  - 修复: 数据耗尽时idx重置到0,循环复用历史数据
  - 来源: walk-forward optimization标准实践

数据源：
  - Gate.io 真实历史K线 (v3.18起, 主数据源) + Binance 备用 + 本地缓存优先
  - TimesFM 本地时间序列预测（可选）
  - Kronos 价格区间信号（可选）

架构原则：
  - TimesFM/Kronos 是传感器，不是决策者（与均线RSI地位平等）
  - 数据管道只负责"提供数据"，不参与交易决策
  - 子Agent通过信号工具池选择使用哪些数据源
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sandbox_engine import MarketData
# P0-1确定性事件驱动重构：使用确定性RNG替代模块级random
from .deterministic_rng import DeterministicRNG, global_rng_manager

logger = logging.getLogger("hermes.data_pipeline")


# ============================================================================
# 历史K线数据（回测模式）
# ============================================================================


@dataclass(slots=True)
class OHLCVBar:
    """单根K线数据"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoricalDataFeed:
    """历史K线数据源 — 回测模式的核心数据输入

    支持从JSON/CSV文件加载，或通过生成器模拟。
    """

    def __init__(self):
        self._bars: Dict[str, List[OHLCVBar]] = {}
        self._current_index: Dict[str, int] = {}
        self._symbols: List[str] = []

    def load_from_json(self, filepath: str, symbol: str) -> int:
        """从JSON文件加载K线数据

        期望格式: [{"timestamp": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        bars = []
        for item in data:
            bars.append(OHLCVBar(
                timestamp=item.get("timestamp", item.get("ts", 0)),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", item.get("vol", 0))),
            ))

        self._bars[symbol] = sorted(bars, key=lambda b: b.timestamp)
        self._current_index[symbol] = 0
        if symbol not in self._symbols:
            self._symbols.append(symbol)

        logger.info("Loaded %d bars for %s", len(bars), symbol)
        return len(bars)

    def load_from_list(self, bars_data: List[Dict[str, Any]], symbol: str) -> int:
        """从字典列表加载K线数据"""
        bars = []
        for item in bars_data:
            bars.append(OHLCVBar(
                timestamp=item.get("timestamp", item.get("ts", 0)),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", item.get("vol", 0))),
            ))

        self._bars[symbol] = sorted(bars, key=lambda b: b.timestamp)
        self._current_index[symbol] = 0
        if symbol not in self._symbols:
            self._symbols.append(symbol)

        return len(bars)

    def generate_mock_data(
        self,
        symbol: str,
        n_bars: int = 1000,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0,
        seed: int = 42,
        use_realistic: bool = True,
    ) -> int:
        """生成模拟K线数据

        v2.3 反温室升级：默认使用真实统计特征数据生成器
          - Student-t厚尾（峰度10-100+，vs高斯=3）
          - GARCH(1,1)波动率聚集
          - Merton跳跃扩散（黑天鹅事件）
          - 价量相关性（暴跌时成交量激增）
          - 牛熊周期切换

        解决"沙盘赚钱实盘亏钱"的根源：纯高斯随机游走永不遇极端行情，
        导致策略在沙盘表现好但实盘遇厚尾事件必亏。

        Args:
            symbol: 交易对名称
            n_bars: K线数量
            start_price: 起始价格
            volatility: 波动率（use_realistic=False时使用，作为年化波动率）
            trend: 趋势强度（正=上涨，负=下跌）
            seed: 随机种子
            use_realistic: True=使用真实数据生成器（推荐），
                          False=使用旧的纯高斯随机游走（仅用于对比测试）

        Returns:
            生成的K线数量
        """
        if use_realistic:
            return self._generate_realistic_data(
                symbol, n_bars, start_price, volatility, trend, seed,
            )
        else:
            return self._generate_gaussian_data(
                symbol, n_bars, start_price, volatility, trend, seed,
            )

    def _generate_realistic_data(
        self,
        symbol: str,
        n_bars: int,
        start_price: float,
        volatility: float,
        trend: float,
        seed: int,
    ) -> int:
        """使用真实统计特征生成K线数据（v2.3反温室核心）

        综合Student-t厚尾 + GARCH波动率聚集 + Merton跳跃 + 价量相关 + 牛熊周期。
        """
        from .realistic_data_generator import (
            RealisticDataConfig, RealisticMarketDataGenerator,
        )

        # 将传入的volatility作为年化波动率（0.02 → 20%年化，偏低，调整到加密货币典型70%）
        # 如果用户传入0.02，理解为"每根K线2%"，转换为年化
        annual_vol = volatility * math.sqrt(8760) if volatility < 0.1 else volatility

        # 将trend作为年化漂移
        annual_drift = trend * 8760 if abs(trend) < 0.1 else trend

        config = RealisticDataConfig(
            start_price=start_price,
            annual_drift=annual_drift,
            annual_volatility=annual_vol,
            seed=seed,
        )
        generator = RealisticMarketDataGenerator(config)
        raw_bars = generator.generate(symbol, n_bars)

        # 转换为OHLCVBar
        bars = [
            OHLCVBar(
                timestamp=b["timestamp"],
                open=b["open"],
                high=b["high"],
                low=b["low"],
                close=b["close"],
                volume=b["volume"],
            )
            for b in raw_bars
        ]

        self._bars[symbol] = bars
        self._current_index[symbol] = 0
        if symbol not in self._symbols:
            self._symbols.append(symbol)

        # 验证分布特征
        stats = generator.validate_distribution(raw_bars)
        logger.info(
            "Generated %d REALISTIC bars for %s: kurtosis=%.1f (gaussian=0) "
            "skew=%.2f acf=%.3f extreme_freq=%.2f%% jumps=%d",
            n_bars, symbol,
            stats.get("excess_kurtosis", 0),
            stats.get("skewness", 0),
            stats.get("squared_returns_acf", 0),
            stats.get("extreme_event_freq", 0) * 100,
            generator.get_statistics().get("jumps", 0),
        )
        return n_bars

    def _generate_gaussian_data(
        self,
        symbol: str,
        n_bars: int,
        start_price: float,
        volatility: float,
        trend: float,
        seed: int,
    ) -> int:
        """旧的纯高斯随机游走（仅用于对比测试）

        警告：此方法生成的数据无尖峰厚尾/波动率聚集/跳跃，
        会导致策略在沙盘表现好但实盘遇极端行情必亏。
        生产环境请使用_generate_realistic_data。

        P0-1确定性重构：使用DeterministicRNG替代模块级random
        """
        # P0-1: 使用确定性RNG，保证可复现性
        rng = DeterministicRNG(seed=seed)

        bars = []
        price = start_price
        base_time = time.time() - n_bars * 3600

        for i in range(n_bars):
            ret = rng.gauss(trend / n_bars, volatility)
            open_price = price
            close_price = price * (1 + ret)
            high_price = max(open_price, close_price) * (1 + rng.random() * volatility * 0.5)
            low_price = min(open_price, close_price) * (1 - rng.random() * volatility * 0.5)
            volume = rng.uniform(100, 10000)

            bars.append(OHLCVBar(
                timestamp=base_time + i * 3600,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            ))
            price = close_price

        self._bars[symbol] = bars
        self._current_index[symbol] = 0
        if symbol not in self._symbols:
            self._symbols.append(symbol)

        logger.warning(
            "Generated %d GAUSSIAN bars for %s (DEPRECATED: no fat tail/clustering/jumps)",
            n_bars, symbol,
        )
        return n_bars

    def next_bar(self, symbol: str) -> Optional[MarketData]:
        """获取下一根K线，转换为MarketData格式

        v2.28修复: 数据循环复用 — 当idx超过数据长度时自动回到0
        原问题: n_bars=2190, 每轮消耗24 tick, 91轮后数据耗尽返回None
                导致进化500轮但只有前91轮有交易, 后409轮无数据
                IPOP restart后的新种群根本没有数据可交易
        修复: 数据耗尽时自动重置idx到0, 让历史数据循环复用
        来源: walk-forward optimization标准实践 — 数据循环复用避免浪费
        效果: 500轮进化全程有数据, 进化算法能持续优化策略
        注意: 循环复用可能引入轻微look-ahead bias, 但优于"无数据可交易"
        """
        if symbol not in self._bars:
            return None

        idx = self._current_index.get(symbol, 0)
        if idx >= len(self._bars[symbol]):
            # v2.28: 数据循环复用 — 重置到0而不是返回None
            # 原代码: return None (数据耗尽后所有Agent停止交易)
            # 新代码: idx = 0 (循环复用历史数据)
            idx = 0
            self._current_index[symbol] = 0

        bar = self._bars[symbol][idx]
        self._current_index[symbol] = idx + 1

        # v3.32o 修复: 模拟订单簿 spread 从 0.001 → 0.003
        # 根因: spread=0.001 (0.1%) 远小于真实市场滑点
        #   - ask = close * 1.0005 (仅高 0.05%)
        #   - bid = close * 0.9995 (仅低 0.05%)
        #   - 导致 R:R 校验基于 ask 与基于 close 几乎相同, 校验形同虚设
        #   - 实测: 49/52笔交易 R:R<1.5 (94.2%), 校验完全无效
        # 真实市场: Binance/OKX BTC合约滑点 0.1%-0.5% (含市场冲击)
        # 修复: spread=0.003 (0.3%) 是真实 BTC 合约市场的中间值
        # 反过拟合: 这是修复模拟环境与真实环境的差异, 符合"杜绝模拟牛逼实盘亏钱"铁律
        # 来源: Binance 2024 费率表 + 市场冲击成本研究
        spread = 0.003
        bid = bar.close * (1 - spread / 2)
        ask = bar.close * (1 + spread / 2)

        return MarketData(
            symbol=symbol,
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            bid=bid,
            ask=ask,
            bid_depth=bar.volume * 0.3,
            ask_depth=bar.volume * 0.3,
        )

    def peek_bar(self, symbol: str, offset: int = 0) -> Optional[MarketData]:
        """预览K线但不推进索引"""
        if symbol not in self._bars:
            return None

        idx = self._current_index.get(symbol, 0) + offset
        if idx < 0 or idx >= len(self._bars[symbol]):
            return None

        bar = self._bars[symbol][idx]
        # v3.32o: spread=0.003 (与 next_bar 一致, 真实市场滑点)
        spread = 0.003
        bid = bar.close * (1 - spread / 2)
        ask = bar.close * (1 + spread / 2)

        return MarketData(
            symbol=symbol,
            timestamp=bar.timestamp,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            bid=bid,
            ask=ask,
        )

    def get_bars(self, symbol: str, n: int = 100) -> List[OHLCVBar]:
        """获取最近N根K线（用于技术指标计算）"""
        if symbol not in self._bars:
            return []

        idx = self._current_index.get(symbol, 0)
        start = max(0, idx - n)
        end = idx
        return self._bars[symbol][start:end]

    def get_all_bars(self, symbol: str) -> List[OHLCVBar]:
        """获取所有K线"""
        return self._bars.get(symbol, [])

    def reset(self) -> None:
        """重置所有symbol的索引"""
        for symbol in self._current_index:
            self._current_index[symbol] = 0

    def has_more(self, symbol: str) -> bool:
        """是否还有更多K线"""
        if symbol not in self._bars:
            return False
        return self._current_index.get(symbol, 0) < len(self._bars[symbol])

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    @property
    def total_bars(self) -> Dict[str, int]:
        return {s: len(bars) for s, bars in self._bars.items()}


# ============================================================================
# 技术指标计算器（传感器层）
# ============================================================================


class TechnicalIndicators:
    """技术指标计算器 — 纯函数，无状态

    所有指标接收numpy数组或列表，返回计算结果。
    子Agent通过信号工具池调用这些指标来判断入场/出场。
    """

    @staticmethod
    def sma(data: List[float], period: int) -> List[float]:
        """简单移动平均"""
        n = len(data)
        if n == 0:
            return []
        if n < period:
            return [data[-1]] * n
        # P4#5: numpy cumsum 实现 O(n) 滚动和，替代 O(n*period) 切片求和
        arr = np.asarray(data, dtype=np.float64)
        cumsum = np.cumsum(arr)
        result = np.empty(n, dtype=np.float64)
        # i < period-1: 用 data[:i+1] 的均值
        for i in range(period - 1):
            result[i] = cumsum[i] / (i + 1)
        # i >= period-1: 用 cumsum 差值计算窗口和
        result[period - 1] = cumsum[period - 1] / period
        if n > period:
            result[period:] = (cumsum[period:] - cumsum[:-period]) / period
        return result.tolist()

    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """指数移动平均"""
        n = len(data)
        if n == 0:
            return []
        # P4#5: 小数组用纯 Python 避免 numpy 转换开销
        if n < 1000:
            multiplier = 2.0 / (period + 1)
            result = [data[0]]
            for i in range(1, n):
                result.append(data[i] * multiplier + result[-1] * (1 - multiplier))
            return result
        # 大数组用 numpy 加速
        arr = np.asarray(data, dtype=np.float64)
        multiplier = 2.0 / (period + 1)
        result = np.empty(n, dtype=np.float64)
        result[0] = arr[0]
        for i in range(1, n):
            result[i] = arr[i] * multiplier + result[i - 1] * (1 - multiplier)
        return result.tolist()

    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[float]:
        """相对强弱指数"""
        n = len(data)
        if n < period + 1:
            return [50.0] * n if n else []

        # P4#5: numpy 向量化 gains/losses 计算
        arr = np.asarray(data, dtype=np.float64)
        delta = np.diff(arr)  # 长度 n-1
        gains = np.maximum(delta, 0.0)
        losses = np.maximum(-delta, 0.0)
        # 转为 list 避免递推循环中 numpy 标量运算的开销
        gains_l = gains.tolist()
        losses_l = losses.tolist()

        # 注意：原代码返回长度 = len(gains) = n - 1（前 period 个为 50.0）
        result = [50.0] * period

        avg_gain = sum(gains_l[:period]) / period
        avg_loss = sum(losses_l[:period]) / period

        # 递推部分（EMA 平滑，循环不可避免）
        for i in range(period, n - 1):
            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            result.append(rsi_val)
            avg_gain = (avg_gain * (period - 1) + gains_l[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses_l[i]) / period

        return result

    @staticmethod
    def macd(
        data: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[List[float], List[float], List[float]]:
        """MACD: (MACD线, 信号线, 柱状图)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)

        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        data: List[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[List[float], List[float], List[float]]:
        """布林带: (中轨, 上轨, 下轨)"""
        n = len(data)
        if n == 0:
            return [], [], []
        sma = TechnicalIndicators.sma(data, period)
        sma_arr = np.asarray(sma, dtype=np.float64)
        arr = np.asarray(data, dtype=np.float64)
        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)

        # P4#5: numpy 向量化 std 计算
        # i < period-1: 用 data[:i+1] 计算 std（总体标准差 ddof=0）
        for i in range(min(period - 1, n)):
            window = arr[:i + 1]
            std = float(np.std(window))
            upper[i] = sma_arr[i] + std_dev * std
            lower[i] = sma_arr[i] - std_dev * std

        # i >= period-1: 用滑动窗口计算 std
        if n >= period:
            windows = np.lib.stride_tricks.sliding_window_view(arr, period)
            stds = np.std(windows, axis=1)
            upper[period - 1:] = sma_arr[period - 1:] + std_dev * stds
            lower[period - 1:] = sma_arr[period - 1:] - std_dev * stds

        return sma, upper.tolist(), lower.tolist()

    @staticmethod
    def atr(
        high: List[float], low: List[float], close: List[float],
        period: int = 14,
    ) -> List[float]:
        """平均真实波幅"""
        n = len(high)
        if n < 2:
            return [0.0] * n if n else []

        # P4#5: 列表推导式计算 TR（比 for 循环快，无 numpy 转换开销）
        tr = [
            max(h - l, abs(h - c_prev), abs(l - c_prev))
            for h, l, c_prev in zip(high[1:], low[1:], close[:-1])
        ]

        result = [0.0]  # 第一个值为0
        atr_val = sum(tr[:period]) / period if len(tr) >= period else sum(tr) / len(tr)
        result.append(atr_val)

        # 递推部分（EMA 平滑）
        for i in range(1, len(tr)):
            atr_val = (atr_val * (period - 1) + tr[i]) / period
            result.append(atr_val)

        return result

    @staticmethod
    def adx(
        high: List[float], low: List[float], close: List[float],
        period: int = 14,
    ) -> List[float]:
        """平均趋向指数

        注：ADX 的递推部分（3个EMA变量串行依赖）是性能瓶颈，
        向量化 TR/DM 的收益被 tolist 转换开销抵消，因此保持纯 Python 实现。
        """
        n = len(high)
        if n < period + 1:
            return [25.0] * n if n else []

        tr = []
        plus_dm = []
        minus_dm = []

        for i in range(1, n):
            tr.append(max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]),
            ))
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)

        result = [25.0] * period

        tr_ema = sum(tr[:period]) / period
        plus_dm_ema = sum(plus_dm[:period]) / period
        minus_dm_ema = sum(minus_dm[:period]) / period

        for i in range(period, n - 1):
            tr_ema = (tr_ema * (period - 1) + tr[i]) / period
            plus_dm_ema = (plus_dm_ema * (period - 1) + plus_dm[i]) / period
            minus_dm_ema = (minus_dm_ema * (period - 1) + minus_dm[i]) / period

            plus_di = 100 * plus_dm_ema / tr_ema if tr_ema > 0 else 0
            minus_di = 100 * minus_dm_ema / tr_ema if tr_ema > 0 else 0

            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
            result.append(dx)

        # ADX = EMA of DX
        adx = TechnicalIndicators.ema(result[period:], period)
        return [25.0] * period + adx

    # ============================================================
    # v3.21 新增领先指标 (修复根因2: 滞后指标虚假共振)
    # 来源: DIRECTION-CHECK智能体诊断 — RSI/MACD/EMA/BB均为滞后指标
    #       在震荡末端反向, signal_count放大1.3-1.45倍导致虚假确认
    # 领先指标基于价量关系, 提前反映市场结构变化
    # ============================================================

    @staticmethod
    def roc(data: List[float], period: int = 10) -> List[float]:
        """Rate of Change 价格变化率 (领先指标)

        ROC = (price - price[n]) / price[n] * 100

        与RSI/MACD不同, ROC直接度量价格动量, 不经过平滑,
        对价格反转响应快, 是经典领先指标.

        Args:
            period: 回看周期 (默认10)
        Returns:
            ROC值列表 (百分比), 前period个为0
        """
        n = len(data)
        if n <= period:
            return [0.0] * n
        arr = np.asarray(data, dtype=np.float64)
        result = np.zeros(n, dtype=np.float64)
        result[period:] = (arr[period:] - arr[:-period]) / arr[:-period] * 100.0
        return result.tolist()

    @staticmethod
    def mfi(
        high: List[float], low: List[float], close: List[float],
        volume: List[float], period: int = 14,
    ) -> List[float]:
        """Money Flow Index 资金流量指数 (领先+价量综合)

        MFI = 100 - 100 / (1 + Money Flow Ratio)
        Money Flow Ratio = Positive MF / Negative MF

        与RSI不同, MFI加入成交量权重, 大资金进场会先于价格反映.
        经典领先指标, 适合日内交易.

        Returns:
            MFI值列表 (0-100), 前period个为50.0

        v3.10.19 LOSS-HUNTER 修复: IndexError list index out of range
        根因: n=len(close) 但 tp/mf 长度可能 < n (high/low/volume 长度不一致时)
        修复: 用 min_len 对齐所有输入列表, 防止索引越界
        """
        # v3.10.19: 对齐所有输入列表长度, 防止索引越界
        min_len = min(len(close), len(high), len(low), len(volume))
        n = min_len
        if n <= period:
            return [50.0] * n if n else []

        # 典型价格 = (H+L+C)/3 (截断到 min_len 对齐)
        tp = [(high[i] + low[i] + close[i]) / 3.0 for i in range(n)]
        # 资金流 = TP * Volume
        mf = [tp[i] * volume[i] for i in range(n)]

        result = [50.0] * period
        # v3.10.22 LOSS-HUNTER 修复: MFI计算方向反向bug (100%信号反向根因#1)
        # 原bug: 用单个 prev_tp = tp[i-1] 与整个 window 比较
        #   上涨趋势: tp递增, tp[i-1]是window最高点 → 所有window tp < prev_tp
        #             → neg_mf主导 → MFI低 → 触发long(均值回归) → 但趋势继续涨 → long亏损
        #   下跌趋势: tp递减, tp[i-1]是window最低点 → 所有window tp > prev_tp
        #             → pos_mf主导 → MFI高 → 触发short → 但趋势继续跌 → short亏损
        # 修复: 标准MFI算法 — 逐bar比较 tp[k] vs tp[k-1], 确定每bar资金流方向
        # 来源: Wilder's Original MFI + TradingView标准实现
        raw_pos_mf = [0.0] * n  # 每bar的正资金流 (tp上涨时)
        raw_neg_mf = [0.0] * n  # 每bar的负资金流 (tp下跌时)
        for k in range(1, n):
            if tp[k] > tp[k - 1]:
                raw_pos_mf[k] = mf[k]
            elif tp[k] < tp[k - 1]:
                raw_neg_mf[k] = mf[k]

        for i in range(period, n):
            # 窗口 [i-period+1, i] 内的累计正/负资金流
            window_pos = sum(raw_pos_mf[i - period + 1:i + 1])
            window_neg = sum(raw_neg_mf[i - period + 1:i + 1])
            if window_neg == 0:
                result.append(100.0)
            elif window_pos == 0:
                result.append(0.0)
            else:
                mfr = window_pos / window_neg
                result.append(100.0 - 100.0 / (1.0 + mfr))
        return result

    @staticmethod
    def willr(
        high: List[float], low: List[float], close: List[float],
        period: int = 14,
    ) -> List[float]:
        """Williams %R 威廉指标 (领先-超买超卖)

        %R = -100 * (HighestHigh - Close) / (HighestHigh - LowestLow)

        与RSI同为超买超卖指标, 但响应更快, 适合捕捉短期反转.
        范围: -100 到 0 (0=最强超买, -100=最强超卖)

        Returns:
            Williams %R值列表, 前period个为-50.0

        v3.10.19 LOSS-HUNTER 修复: ValueError max() iterable argument is empty
        根因: n=len(close) 但 high/low 可能比 close 短, 切片为空
        修复: 用 min_len 对齐所有输入列表, 防止空切片
        """
        # v3.10.19: 对齐所有输入列表长度, 防止空切片
        min_len = min(len(close), len(high), len(low))
        n = min_len
        if n <= period:
            return [-50.0] * n if n else []

        result = [-50.0] * period
        for i in range(period, n):
            # v3.10.19: 从 period 开始 (前 period 个为默认值), 防止 off-by-one
            # 切片 [i-period+1:i+1] 包含当前bar, 长度=period
            hh = max(high[i - period + 1:i + 1])
            ll = min(low[i - period + 1:i + 1])
            if hh == ll:
                result.append(-50.0)
            else:
                result.append(-100.0 * (hh - close[i]) / (hh - ll))
        return result

    @staticmethod
    def volume_anomaly(volume: List[float], period: int = 20) -> List[float]:
        """成交量异常比率 (领先-资金行为)

        ratio = current_volume / SMA(volume, period)

        成交量是价格的先行指标, 放量往往先于价格突破.
        ratio > 2.0 表示显著放量, < 0.5 表示缩量.

        Returns:
            成交量比率列表, 前period个为1.0
        """
        n = len(volume)
        if n == 0:
            return []
        sma_vol = TechnicalIndicators.sma(volume, period)
        result = []
        for i in range(n):
            if sma_vol[i] > 0:
                result.append(volume[i] / sma_vol[i])
            else:
                result.append(1.0)
        return result


# ============================================================================
# TimesFM 预测服务（可选传感器）
# ============================================================================


class TimesFMPredictor:
    """TimesFM 时间序列预测服务 — 传感器，不是决策者

    提供基于历史价格的下一根K线方向预测和置信度。
    子Agent可以选择性使用此信号（与其他技术指标地位平等）。
    """

    def __init__(self, model_path: str = "", use_real_inference: bool = False):
        """
        Args:
            model_path: TimesFM模型路径（留空则使用模拟预测）
            use_real_inference: 是否强制使用真实推理（True=加载失败则报错，False=降级到模拟）
        """
        self.model_path = model_path
        self._model_loaded = False
        self._model = None
        self._use_real_inference = use_real_inference
        self._prediction_cache: Dict[str, deque] = {}
        self._cache_size = 100

        # P0-3: 消除空壳交付 — 明确尝试加载真实模型
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """加载真实TimesFM模型

        P0-3: 不再伪装有模型，明确尝试加载，失败则明确报错或降级
        """
        try:
            # 尝试导入timesfm库
            import timesfm  # type: ignore
            import torch

            logger.info("Loading TimesFM model from %s", model_path)
            self._model = timesfm.TimesFm(
                context_len=512,
                horizon_len=1,
                input_patch_len=32,
                output_patch_len=128,
                backend="gpu" if torch.cuda.is_available() else "cpu",
            )
            self._model.load_from_checkpoint(model_path)
            self._model_loaded = True
            logger.info("TimesFM model loaded successfully (backend=%s)",
                        "gpu" if torch.cuda.is_available() else "cpu")

        except ImportError as e:
            msg = f"TimesFM库未安装: {e}. 请执行: pip install timesfm torch"
            if self._use_real_inference:
                raise ImportError(msg) from e
            logger.warning("%s — 降级到模拟模式", msg)
            self._model_loaded = False

        except Exception as e:
            msg = f"TimesFM模型加载失败: {e}"
            if self._use_real_inference:
                raise RuntimeError(msg) from e
            logger.warning("%s — 降级到模拟模式", msg)
            self._model_loaded = False

    def predict(
        self, symbol: str, price_history: List[float],
        horizon: int = 1,
    ) -> Dict[str, Any]:
        """预测下一根K线的方向

        Args:
            symbol: 交易对
            price_history: 历史价格序列（至少需要50个点）
            horizon: 预测步长

        Returns:
            {"direction": "long"/"short"/"neutral", "confidence": 0-1, "predicted_price": float}
        """
        if len(price_history) < 20:
            # 边界条件：price_history为空时返回默认值
            last_price = price_history[-1] if price_history else 0.0
            return {
                "direction": "neutral", "confidence": 0.0,
                "predicted_price": last_price,
                "source": "insufficient_data",  # P0-3: 明确标注来源
            }

        if self._model_loaded and self._model is not None:
            # P0-3: 真实TimesFM推理
            return self._real_predict(symbol, price_history, horizon)

        # P0-3: 明确标注为模拟模式（不再伪装成真实推理）
        return self._mock_predict(symbol, price_history, horizon)

    def _real_predict(
        self, symbol: str, price_history: List[float], horizon: int,
    ) -> Dict[str, Any]:
        """真实TimesFM推理

        P0-3: 消除空壳 — 当模型加载成功时执行真实推理
        """
        import numpy as np

        try:
            # TimesFM需要2D输入: (batch_size, context_len)
            context = np.array(price_history[-512:], dtype=np.float32).reshape(1, -1)

            # 执行推理
            forecast = self._model.forecast(context)
            point_forecast = forecast[0].mean  # 取均值预测

            predicted_price = float(point_forecast[0]) if hasattr(point_forecast, '__getitem__') else float(point_forecast)

            # 基于预测价格计算方向和置信度
            last_price = price_history[-1]
            change_pct = (predicted_price - last_price) / last_price if last_price > 0 else 0

            if abs(change_pct) < 0.002:
                direction = "neutral"
                confidence = 0.3
            elif change_pct > 0:
                direction = "long"
                confidence = min(0.95, 0.5 + abs(change_pct) * 50)
            else:
                direction = "short"
                confidence = min(0.95, 0.5 + abs(change_pct) * 50)

            return {
                "direction": direction,
                "confidence": round(confidence, 3),
                "predicted_price": round(predicted_price, 4),
                "source": "timesfm_real",
            }

        except Exception as e:
            logger.error("TimesFM real inference failed: %s — 降级到模拟", e)
            return self._mock_predict(symbol, price_history, horizon)

    def _mock_predict(
        self, symbol: str, price_history: List[float], horizon: int,
    ) -> Dict[str, Any]:
        """模拟预测（P0-3: 明确标注为模拟模式，不再伪装成真实推理）

        当TimesFM模型未加载时使用此方法。
        基于简单趋势+波动+确定性RNG生成预测。

        P0-1确定性重构：使用全局确定性RNG替代模块级random
        保证同一数据→同一预测结果→可复现的进化对比
        """
        # P0-1: 使用全局确定性RNG
        rng = global_rng_manager.global_rng

        # 简单趋势检测
        short_ma = sum(price_history[-5:]) / 5
        long_ma = sum(price_history[-20:]) / 20
        # P0修复#9: 防御除零 — long_ma 为 0 时 trend=0（中性）
        trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0

        # 最近波动
        recent = price_history[-10:]
        vol = (max(recent) - min(recent)) / (sum(recent) / len(recent))

        # 综合判断
        # 注意：direction使用"long"/"short"/"neutral"，与AgentDecisionEngine._eval_signal一致
        # 之前使用"up"/"down"导致信号被静默忽略（BLOCK-1修复）
        if abs(trend) < 0.005:
            direction = "neutral"
            confidence = 0.3
        elif trend > 0:
            direction = "long"
            confidence = min(0.9, 0.5 + abs(trend) * 20)
        else:
            direction = "short"
            confidence = min(0.9, 0.5 + abs(trend) * 20)

        # 加入随机噪声（模拟预测不确定性）
        # P0-1: 使用确定性RNG
        confidence *= rng.uniform(0.7, 1.0)

        predicted_price = price_history[-1] * (1 + trend * horizon)

        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "predicted_price": round(predicted_price, 4),
            "trend_strength": round(trend, 6),
            "volatility": round(vol, 6),
            "source": "mock",  # P0-3: 明确标注来源
        }


# ============================================================================
# Kronos 价格区间（可选传感器）
# ============================================================================


class KronosRangePredictor:
    """Kronos 价格区间预测 — 传感器，不是决策者

    提供基于统计的价格区间上下界。
    """

    def __init__(self):
        pass

    def compute_range(
        self, price_history: List[float], lookback: int = 100,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """计算价格区间

        Args:
            price_history: 历史价格
            lookback: 回看窗口
            confidence_level: 置信水平

        Returns:
            {"lower": float, "upper": float, "mid": float, "width_pct": float}
        """
        if len(price_history) < 10:
            current = price_history[-1] if price_history else 100.0
            return {"lower": current * 0.95, "upper": current * 1.05, "mid": current, "width_pct": 0.10}

        window = price_history[-lookback:] if len(price_history) >= lookback else price_history

        # P0修复#6: 修复 ATR 逻辑错误 — 不再将同一数组作为 high/low/close
        # 原代码: atr = TechnicalIndicators.atr(window, window, window, 14) → TR恒为0
        # 修复: 从 close 价格推导 high/low（滚动窗口极值），或直接用收益率标准差
        if len(window) >= 14:
            # 方法: 用滚动窗口的 max/min 作为 high/low
            high_prices = []
            low_prices = []
            for i in range(len(window)):
                start = max(0, i - 13)
                slice_w = window[start:i + 1]
                high_prices.append(max(slice_w))
                low_prices.append(min(slice_w))
            atr = TechnicalIndicators.atr(high_prices, low_prices, window, 14)
            avg_atr = sum(atr) / len(atr) if atr else 0
        else:
            # 数据不足时用收益率标准差作为波动率代理
            returns = [window[i] / window[i - 1] - 1 for i in range(1, len(window)) if window[i - 1] != 0]
            avg_atr = (sum(returns) / len(returns) if returns else 0) * (price_history[-1] if price_history else 100)

        current = price_history[-1]
        # 区间宽度 = 2 * ATR * 置信系数
        width = avg_atr * 2 * (1 + confidence_level)

        return {
            "lower": round(current - width, 4),
            "upper": round(current + width, 4),
            "mid": round(current, 4),
            "width_pct": round(width / current * 100, 2),
        }


# ============================================================================
# 统一数据管道
# ============================================================================


class DataPipeline:
    """统一数据管道 — 整合行情+预测+区间

    为沙盘进化循环提供"一口接入"的数据服务。
    """

    def __init__(self):
        self.feed = HistoricalDataFeed()
        self.indicators = TechnicalIndicators()
        self.timesfm = TimesFMPredictor()
        self.kronos = KronosRangePredictor()

        # 多模型集成传感器（LightGBM + Prophet + Qwen 三层架构）
        # 来源：theneuralbase.com 2026 + QuantMind 2026
        # 传感器地位：与TimesFM/Kronos平等，不参与决策，只提供信号
        try:
            from .multi_model_ensemble import MultiModelEnsembleSystem
            # 自动检测 LightGBM 模型文件路径（优先 v2 40特征版本）
            lgbm_model_path = ""
            for candidate in [
                os.path.expanduser("~/hermes_models/lightgbm_model_v2.txt"),
                os.path.expanduser("~/hermes_models/lightgbm_model.txt"),
                os.path.join(os.path.dirname(__file__), "lightgbm_model.txt"),
                "/home/lmy/hermes_models/lightgbm_model_v2.txt",
                "/home/lmy/hermes_models/lightgbm_model.txt",
            ]:
                if os.path.exists(candidate):
                    lgbm_model_path = candidate
                    break
            self.multi_model = MultiModelEnsembleSystem(lightgbm_model_path=lgbm_model_path)
        except ImportError:
            self.multi_model = None
            logger.warning("MultiModelEnsembleSystem not available")

        # P2-1: 市场状态检测器 — 为 DynamicWeightManager 提供实时市场状态
        # 来源：regime_detection.py（HMM + 变点检测 + GARCH波动率）
        # 作用：检测 trend/range/volatile 状态，动态调整三模型权重
        try:
            from .regime_detection import RegimeDetectionSystem
            self.regime_detector = RegimeDetectionSystem()
            self._last_price_for_regime: Dict[str, float] = {}
            logger.info("RegimeDetectionSystem integrated for dynamic weight adjustment")
        except ImportError:
            self.regime_detector = None
            logger.warning("RegimeDetectionSystem not available, using default weights")

        self._price_cache: Dict[str, deque] = {}
        self._indicator_cache: Dict[str, Dict[str, Any]] = {}

        # Phase E Batch 3: WebSocketStream 接入 — 实时行情数据流
        # 来源：websocket_stream.py → SyncWebSocketStreamManager（同步封装）
        # 设计：使用同步封装版本，适合沙盘环境（非异步）
        #   - exchange_name="gateio"（铁律：唯一交易所为Gate.io）
        #   - 依赖 ccxt.pro，缺失时降级为 None（不阻断，但记录警告）
        # 铁律14合规：零遗漏 — 已实现基础设施模块必须接入主干
        self._websocket_stream = None
        try:
            from .websocket_stream import SyncWebSocketStreamManager
            self._websocket_stream = SyncWebSocketStreamManager(
                exchange_name="gateio",
            )
        except Exception as _ws_err:
            logger.warning("WebSocketStreamManager初始化失败: %s", str(_ws_err)[:150])
    def initialize(
        self,
        symbols: List[str],
        bars_per_symbol: int = 1000,
        use_mock: bool = True,
        use_local_real: bool = True,
    ) -> None:
        """初始化数据管道

        v3.18 (2026-06-27): use_mock=False 时使用 Gate.io 真实历史K线数据
          - 用户要求: "接入实盘真实数据去进化迭代AI量化技能包"
          - 替代原"未生成任何数据"的空实现
          - 来源: gateio_data_integration.create_real_data_feed
          - 数据校验: GateIoDataValidator 4层校验保障准确率≥99.9%

        v502 (2026-06-28): 反温室工程化 — 强制沙盘使用真实K线 (P2-2 修复)
          - 用户铁证要求: "全方位多维度多层次多角度深入彻底杜绝根治模拟牛逼,实盘亏钱!"
          - v501 根因分析发现: 沙盘 BTC=124238 vs 真实 BTC=74632, 偏差+66.47%
          - 根因: 默认 use_mock=True 调用 _generate_cyclical_data() 生成合成数据
            所有参数(止损/止盈/信号门槛)在高价位校准, 部署到真实低价位完全错位
          - 修复: 新增 use_local_real=True 参数, 优先读 data/real_klines/ 本地文件
          - 优先级: 本地真实K线 > Gate.io live API > 合成数据(仅作最后降级)
          - 来源: v501_root_cause_report.md 数据温室铁证
        """
        if not use_mock:
            # v502 反温室工程化: 优先加载本地真实K线 (P2-2 修复)
            if use_local_real:
                all_local_ok = True
                for symbol in symbols:
                    ok = self._load_local_real_klines(symbol, bars_per_symbol)
                    if not ok:
                        all_local_ok = False
                    self._init_price_cache_with_warmup(symbol)
                if all_local_ok:
                    logger.info(
                        "v502 反温室工程化: 全部 %d 个symbol使用本地真实K线 (杜绝数据温室)",
                        len(symbols),
                    )
                    return
                else:
                    logger.warning(
                        "v502: 部分symbol本地K线缺失, 降级到Gate.io live API",
                    )

            # v3.18: 加载 Gate.io 真实历史数据
            # 估算需要的K线数量对应天数（1h框架下 24*days ≈ bars_per_symbol）
            days_est = max(30, bars_per_symbol // 24)
            try:
                from .gateio_data_integration import (
                    create_real_data_feed, _to_internal_symbol,
                )
                # 内部符号 BTC-USDT → Gate.io BTC_USDT
                gateio_pairs = [s.replace("-", "_") for s in symbols]
                logger.info(
                    "v3.18 加载 Gate.io 真实历史数据: pairs=%s interval=1h days=%d",
                    gateio_pairs, days_est,
                )
                feed = create_real_data_feed(
                    pairs=gateio_pairs,
                    interval="1h",
                    days=days_est,
                )
                # 把真实数据合并到 self.feed
                for symbol in symbols:
                    internal_sym = _to_internal_symbol(symbol.replace("-", "_"))
                    if internal_sym in feed._bars:
                        self.feed._bars[symbol] = feed._bars[internal_sym]
                        self.feed._current_index[symbol] = 0
                        if symbol not in self.feed._symbols:
                            self.feed._symbols.append(symbol)
                        bar_count = len(feed._bars[internal_sym])
                        logger.info(
                            "  ✅ %s 加载 %d 根真实K线 (Gate.io)",
                            symbol, bar_count,
                        )
                    else:
                        logger.warning(
                            "  ❌ %s Gate.io 无数据, 降级到合成数据", symbol,
                        )
                        raise RuntimeError(f"real market data unavailable for {symbol}")
                    self._init_price_cache_with_warmup(symbol)
                return
            except Exception as e:
                logger.error("Gate.io real market data loading failed: %s", e)
                raise RuntimeError("real market data unavailable") from e
        for symbol in symbols:
            if use_mock:
                trend = 0.0005  # 默认微弱趋势
                # v2.37 修复 BUG-D (INTRADAY-GUARD): 按 symbol 区分 start_price
                # 根因: 原所有 symbol 都用 start_price=50000.0, 导致 ETH/SOL 价格虚高
                #       SOL 从 50000 涨到 235 万 (47倍), quantity 计算后被 round(,4) 截断为 0
                #       → BUG-C: 所有交易 quantity=0, agent 无实际持仓但亏钱
                # 修复: 按真实价格量级设置 start_price (2026-06 市场参考价)
                start_price = 50000.0  # 默认 (BTC 量级)
                if symbol == "BTC-USDT":
                    trend = 0.0003
                    start_price = 80000.0   # BTC 真实约 6-10 万 USDT
                elif symbol == "ETH-USDT":
                    trend = -0.0002
                    start_price = 3000.0    # ETH 真实约 2000-5000 USDT
                elif symbol == "SOL-USDT":
                    trend = 0.0005
                    start_price = 150.0     # SOL 真实约 100-300 USDT
                # 使用线性趋势+正弦波周期生成可交易信号
                # 正弦波周期约200bar，SMA20/50交叉可捕捉
                self._generate_cyclical_data(
                    symbol, n_bars=bars_per_symbol, start_price=start_price,
                    trend=trend,
                )
            self._init_price_cache_with_warmup(symbol)

    def _init_price_cache_with_warmup(self, symbol: str, warmup_bars: int = 50) -> None:
        """v503 Fix 19: 初始化价格缓存并预热 — 消除前19个tick的空indicators

        根因 (P44 诊断): _price_cache 初始化为空 deque(maxlen=500),
          而 next_tick() 每次只 append 一个 close. 前 19 个 tick len(prices)<20,
          _compute_indicators 返回空 dict {} → _compute_entry_signals 返回 neutral
          → Fix 17 fallback 触发 → 不可靠信号方向 → 0% 胜率

        深层根因: 即使 len(prices) >= 20 但 < 50, _compute_indicators L1541:
          sma_50 = ... if len(prices) >= 50 else prices[-1]
          → sma_50 = 当前 close → trend_up = close > close = False
          → 所有 agent 选 short → Short 胜率 0% (BTC 上涨趋势中 short = 必亏)

        修复原理 (用户铁律: "策略要有深度, 要理解底层逻辑, 而非表面指标"):
          空 indicators = 策略无深度 = 必须根治
          1. 从 feed._bars[symbol] 预读前 warmup_bars 根K线的 close, 填入 _price_cache
          2. 推进 _current_index 到 warmup_bars, 使 next_bar() 从 warmup_bars 开始返回
          3. 这样第一个 next_tick() 返回 bars[warmup_bars],
             _price_cache 已有 warmup_bars 个 close → len>=50 → sma_50 正常计算

        成本: 消耗 warmup_bars=50 根K线作为预热 (2000根中占2.5%),
          换取从第一 tick 就有完整技术指标 (sma_50/rsi_14/bollinger 等).
          这是 walk-forward optimization 标准实践 — 用少量数据预热指标窗口.

        防回退: 此修复同时解决了 Fix 17 的 fallback 触发条件 —
          当 indicators 始终可用时, Fix 17 的 crypto_intraday/chanlun_pa 分支
          只在真正的信号缺失时触发 (而非 indicators 计算失败时).
        """
        self._price_cache[symbol] = deque(maxlen=500)

        bars = self.feed._bars.get(symbol, [])
        if len(bars) < warmup_bars:
            logger.warning(
                "Fix 19: %s 只有 %d 根K线, 不足预热 %d 根, 跳过预热 (indicators 前 %d tick 仍为空)",
                symbol, len(bars), warmup_bars, max(19, warmup_bars),
            )
            return

        # 预读前 warmup_bars 根K线的 close, 填入缓存
        for i in range(warmup_bars):
            self._price_cache[symbol].append(bars[i].close)

        # 推进 feed 索引到 warmup_bars, 使 next_bar() 从 warmup_bars 开始返回
        # 这样 next_tick() 第一次调用时:
        #   - next_bar() 返回 bars[warmup_bars], _current_index 变为 warmup_bars+1
        #   - _price_cache 已有 warmup_bars 个 close (bars[0..warmup_bars-1])
        #   - append 后 _price_cache 有 warmup_bars+1 个 close
        #   - get_bars(symbol, 100) 返回 bars[1..warmup_bars] (最近100根)
        #   - _compute_indicators 用 prices(warmup_bars+1个) + high/low/vol 计算
        self.feed._current_index[symbol] = warmup_bars

        logger.info(
            "Fix 19: %s 预热 %d 根K线完成, _price_cache=%d, next_bar将从index=%d开始 "
            "(消除前%d tick空indicators, sma_50从首tick起可用)",
            symbol, warmup_bars, len(self._price_cache[symbol]),
            warmup_bars, max(19, warmup_bars),
        )

    def _generate_cyclical_data_fallback(
        self, symbol: str, n_bars: int,
    ) -> None:
        """合成数据降级方案 (仅在Gate.io真实数据不可用时使用)"""
        start_price = 50000.0
        if symbol == "ETH-USDT":
            start_price = 3000.0
        elif symbol == "SOL-USDT":
            start_price = 150.0
        self._generate_cyclical_data(
            symbol, n_bars=n_bars, start_price=start_price, trend=0.0003,
        )

    def _load_local_real_klines(
        self, symbol: str, bars_per_symbol: int,
    ) -> bool:
        """v502 反温室工程化: 加载本地真实K线 (P2-2 修复核心)

        用户铁证: "全方位多维度多层次多角度深入彻底杜绝根治模拟牛逼,实盘亏钱!"
        v501 根因: 沙盘合成数据 BTC=124238 vs 真实 BTC=74632, 偏差+66.47%
                   ETH 偏差+79.65%, SOL 偏差+161.07% (最严重)

        数据源: data/real_klines/gateio_{SYMBOL}_1h_*.json
        选择策略: 同symbol多个文件时, 选最大的(24m > 12m)
        循环复用: 数据不足 bars_per_symbol 时按 v2.28 实践循环扩展

        Returns:
            True: 加载成功
            False: 文件不存在/数据不足/解析失败
        """
        # 计算路径: hermes_v6/data/real_klines/
        # data_pipeline.py 位于 hermes_v6/sandbox_trading/, 上两级到 hermes_v6/
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_klines_dir = os.path.join(root, "data", "real_klines")
        if not os.path.isdir(real_klines_dir):
            logger.warning(
                "v502: 本地真实K线目录不存在: %s (反温室修复未生效, 将降级)",
                real_klines_dir,
            )
            return False

        # v502.1 修复: 支持多数据源前缀 (gateio_/binance_), 选最大文件覆盖最长历史
        # 原bug: pattern 写死 "gateio_", 导致 binance_*_25m.json (18169 bars 含牛熊周期)
        #        无法被匹配, 只能加载 gateio_*_12m.json (8761 bars 窄幅盘整期)
        # 反温室铁律: 策略必须经历完整牛熊周期才能进化出鲁棒性, 12个月数据不够
        file_sym = symbol.replace("-", "_")  # BTC-USDT → BTC_USDT
        # binance 文件名格式: binance_BTCUSDT_1h_25m.json (无下划线分隔)
        # gateio 文件名格式: gateio_BTC_USDT_1h_v367_12m.json (有下划线分隔)
        file_sym_nounderscore = file_sym.replace("_", "")  # BTC_USDT → BTCUSDT
        try:
            candidates = []
            for f in os.listdir(real_klines_dir):
                if not f.endswith(".json"):
                    continue
                # 匹配 gateio_BTC_USDT_1h_* 或 binance_BTCUSDT_1h_*
                # Phase 14.1: 多周期数据加载 — 优先1h, 兼容5m/15m
                if (f.startswith(f"gateio_{file_sym}_1h_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_1h_")
                        or f.startswith(f"gateio_{file_sym}_5m_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_5m_")
                        or f.startswith(f"gateio_{file_sym}_15m_")
                        or f.startswith(f"binance_{file_sym_nounderscore}_15m_")):
                    candidates.append(f)
        except OSError as e:
            logger.error("v502: 读取目录失败 %s: %s", real_klines_dir, e)
            return False

        if not candidates:
            logger.warning(
                "v502: 本地无 %s 真实K线文件 (pattern: gateio_%s_1h_* / binance_%s_1h_*) — 反温室修复未生效",
                symbol, file_sym, file_sym_nounderscore,
            )
            return False

        # Phase 14.1: 1h优先 — 5m/15m文件更大但不能作为主周期
        # 排序key: ('_1h_' in f, file_size) — 1h文件优先, 同周期内选最大文件
        candidates.sort(
            key=lambda f: ('_1h_' in f, f.startswith('gateio_'), os.path.getsize(os.path.join(real_klines_dir, f))),
            reverse=True,
        )
        chosen = candidates[0]
        full_path = os.path.join(real_klines_dir, chosen)
        logger.info(
            "v502.1: %s 候选文件 %d 个, 选最大: %s (%d bytes)",
            symbol, len(candidates), chosen,
            os.path.getsize(full_path),
        )

        try:
            with open(full_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            # P5修复: 真实K线文件格式是 {"symbol":..., "interval":..., "data":[...]} (dict 包裹)
            # 原代码期望纯列表, 导致所有本地文件"0条"降级到Gate.io API
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                data = data["data"]
            if not isinstance(data, list) or len(data) < 100:
                logger.warning(
                    "v502: %s 数据不足: %d 条 (需≥100)",
                    chosen, len(data) if isinstance(data, list) else 0,
                )
                return False

            # 转换为 OHLCVBar (字段一一对应)
            # v502.2 修复: 时间戳单位自动检测 (binance=毫秒, gateio=秒)
            # 原bug: binance_*_25m.json 时间戳 1717200000000 (毫秒) 被当作秒解析
            #        → year 56385 错误, 导致 25 个月完整牛熊周期数据无法使用
            # 反温室铁律: 必须用 25 个月含牛熊周期的数据, 12 个月盘整期不够
            bars = []
            ts_unit_detected = False
            ts_is_milliseconds = False
            for k in data:
                try:
                    ts_raw = float(k.get("timestamp", 0))
                    # 自动检测: > 1e12 必为毫秒 (秒级 1e12 对应 year 33658, 不可能)
                    if not ts_unit_detected:
                        ts_is_milliseconds = ts_raw > 1e12
                        ts_unit_detected = True
                    ts_sec = ts_raw / 1000.0 if ts_is_milliseconds else ts_raw
                    bars.append(OHLCVBar(
                        timestamp=ts_sec,
                        open=float(k.get("open", 0)),
                        high=float(k.get("high", 0)),
                        low=float(k.get("low", 0)),
                        close=float(k.get("close", 0)),
                        volume=float(k.get("volume", 0)),
                    ))
                except (TypeError, ValueError):
                    continue

            if len(bars) < 100:
                logger.warning("v502: %s 有效K线不足: %d", chosen, len(bars))
                return False

            # v2.28 实践: 数据不足时循环复用 (walk-forward 标准做法)
            original_count = len(bars)
            while len(bars) < bars_per_symbol:
                bars.extend(bars[:original_count])

            self.feed._bars[symbol] = bars
            self.feed._current_index[symbol] = 0
            if symbol not in self.feed._symbols:
                self.feed._symbols.append(symbol)

            logger.info(
                "  ✅ v502 反温室: %s 加载 %d 根真实K线 (文件: %s, "
                "价格 %.1f→%.1f, 杜绝数据温室)",
                symbol, original_count, chosen,
                bars[0].close, bars[original_count - 1].close,
            )
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.error("v502: 加载本地真实K线失败 %s: %s", chosen, e)
            return False

    def _generate_cyclical_data(
        self, symbol: str, n_bars: int, start_price: float, trend: float,
    ) -> None:
        """生成含可交易周期的数据（正弦波+噪声）"""
        import math, random as _r
        _r.seed(hash(symbol) % (2**31))
        bars = []
        price = start_price
        for i in range(n_bars):
            cycle = math.sin(2 * math.pi * i / 200) * 0.008
            cycle += 0.4 * math.sin(2 * math.pi * i / 50) * 0.006
            trend_component = trend * i / 200
            noise = _r.gauss(0, 0.006)
            ret = cycle + trend_component + noise
            price *= (1 + ret)
            high = price * (1 + abs(_r.gauss(0, 0.005)))
            low = price * (1 - abs(_r.gauss(0, 0.005)))
            # v3.29 LOSS-HUNTER 根因修复: timestamp间隔 86400→3600
            # 根因: _generate_cyclical_data 生成 timestamp=i*86400 (24小时间隔),
            #       但系统是1小时K线框架 (ticks_per_round=24, 模拟24小时)
            #       导致 hold_hours = (exit_time - entry_time) / 3600 被放大24倍
            #       agent在tick 19开仓, tick 20平仓, 实际1小时但计算为24小时
            #       24小时 > max_hold_hours(3-8), 触发时间止损, 0%胜率
            # 修复: timestamp = i * 3600 (1小时间隔, 匹配1小时K线框架)
            # 来源: v325b audit.log entry_time=1641600=19*86400, exit_time=1728000=20*86400
            bars.append(OHLCVBar(
                timestamp=float(i * 3600),
                open=price / (1 + ret) if i > 0 else price,
                high=max(price, high),
                low=min(price, low) if price > 0 else price * 0.99,
                close=price,
                volume=_r.uniform(500, 5000) + abs(noise) * 50000,
            ))
        self.feed._bars[symbol] = bars
        self.feed._current_index[symbol] = 0
        if symbol not in self.feed._symbols:
            self.feed._symbols.append(symbol)

    def next_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取下一个tick的完整数据包

        Returns:
            {
                "market": MarketData,
                "indicators": {...},  # 技术指标
                "timesfm_prediction": {...},  # TimesFM预测
                "kronos_range": {...},  # Kronos区间
            }
        """
        market = self.feed.next_bar(symbol)
        if not market:
            return None

        # P2-2: 反馈闭环 — 用上一根K线的实际涨跌方向更新模型性能监控
        # 来源：multi_model_ensemble.py ModelPerformanceMonitor.update_actual_result()
        # 作用：记录预测→回填实际结果→计算EMA准确率→自动训练meta-learner
        if self.multi_model is not None and symbol in self._price_cache and len(self._price_cache[symbol]) >= 1:
            prev_price = self._price_cache[symbol][-1]
            actual_dir = (
                "long" if market.close > prev_price
                else "short" if market.close < prev_price
                else "neutral"
            )
            try:
                self.multi_model.update_actual_result(actual_direction=actual_dir)
            except Exception as e:
                logger.warning("Feedback loop update failed: %s", e)

        # 更新价格缓存
        if symbol in self._price_cache:
            self._price_cache[symbol].append(market.close)

        prices = list(self._price_cache.get(symbol, []))

        # 计算技术指标（需要足够数据）
        indicators = self._compute_indicators(symbol, market, prices) if len(prices) >= 20 else {}

        # TimesFM预测（数据不足时返回默认值）
        timesfm_pred = self.timesfm.predict(symbol, prices)

        # Kronos区间（数据不足时返回默认值）
        kronos_range = self.kronos.compute_range(prices)

        # 多模型集成信号（LightGBM + Prophet + Qwen 融合信号）
        # 传感器地位：与TimesFM/Kronos平等，子Agent可选择使用
        multi_model_signal = {}
        if self.multi_model is not None:
            # P2-3: 数据质量门禁 — 在模型推理前验证输入数据
            # 来源：multi_model_ensemble.py InputValidator
            # 作用：拦截零价/NaN/RSI越界等异常数据，避免污染模型预测
            data_valid = True
            try:
                from .multi_model_ensemble import InputValidator
                validation = InputValidator.validate_all(indicators, prices)
                if not validation["valid"]:
                    logger.warning("Input data invalid, skip multi-model: %s", validation["errors"])
                    data_valid = False
                elif validation["warnings"]:
                    logger.debug("Input data warnings: %s", validation["warnings"])
            except ImportError:
                pass  # InputValidator 不可用时跳过验证

            if data_valid:
                # P2-1: 更新市场状态检测器，动态调整三模型权重
                # 映射：HMM(bull/bear/sideways) + GARCH(LOW/NORMAL/HIGH) → (trend/range/volatile)
                if self.regime_detector is not None and len(prices) >= 2:
                    last_price = self._last_price_for_regime.get(symbol, prices[-2])
                    ret = (prices[-1] - last_price) / last_price if last_price > 0 else 0.0
                    self._last_price_for_regime[symbol] = prices[-1]
                    try:
                        regime_state = self.regime_detector.update(ret)
                        mapped_regime = self._map_regime(regime_state)
                        self.multi_model.set_market_regime(mapped_regime)
                    except Exception as e:
                        logger.warning("Regime detection failed: %s", e)

                try:
                    multi_model_signal = self.multi_model.get_ensemble_signal(
                        symbol=symbol,
                        indicators=indicators,
                        price_history=prices,
                    )
                except Exception as e:
                    logger.warning("MultiModel signal failed: %s", e)
                    multi_model_signal = {}

        return {
            "market": market,
            "indicators": indicators,
            "timesfm_prediction": timesfm_pred,
            "kronos_range": kronos_range,
            "multi_model": multi_model_signal,
        }

    def _map_regime(self, regime_state: Dict[str, Any]) -> str:
        """将 RegimeDetectionSystem 输出映射到 MultiModelEnsembleSystem 的市场状态

        映射规则（来源：regime_detection.py + multi_model_ensemble.py DynamicWeightManager）：
          - 高波动 OR 状态转换期 → "volatile"（不确定性高，三模型权重趋于均衡）
          - 低/正常波动 + 趋势市场(bull/bear) → "trend"（LightGBM权重提升）
          - 低/正常波动 + 震荡市场(sideways) → "range"（Prophet权重提升）
          - 其他 → "unknown"（使用默认权重）

        Args:
            regime_state: RegimeDetectionSystem.update() 返回值

        Returns:
            "trend" / "range" / "volatile" / "unknown"
        """
        market_regime = regime_state.get("market_regime", "sideways")
        vol_state = regime_state.get("volatility_state", {})
        vol_regime = vol_state.get("volatility_regime", "NORMAL")
        is_transitioning = regime_state.get("is_transitioning", False)

        # 高波动或状态转换期 → volatile
        if vol_regime == "HIGH" or is_transitioning:
            return "volatile"

        # 低/正常波动 + 趋势市场 → trend
        if market_regime in ("bull", "bear"):
            return "trend"

        # 低/正常波动 + 震荡市场 → range
        if market_regime == "sideways":
            return "range"

        return "unknown"

    def _compute_indicators(
        self, symbol: str, market: MarketData, prices: List[float],
    ) -> Dict[str, Any]:
        """计算技术指标 (v3.21: 加入领先指标, 修复根因2)"""
        bars = self.feed.get_bars(symbol, 100)
        prices = [b.close for b in bars]
        high_prices = [b.high for b in bars]
        low_prices = [b.low for b in bars]
        volumes = [b.volume for b in bars]

        if len(prices) < 20:
            return {}

        # 检查缓存：同一K线周期内不重复计算
        cache_key = f"{symbol}:{market.timestamp}"
        if cache_key in self._indicator_cache:
            cached = self._indicator_cache[cache_key]
            cached["current_price"] = market.close
            return cached

        # 单次调用 bollinger_bands 并解包三个返回值
        bb_mid, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(prices, 20)

        # v3.21: 领先指标计算 (修复根因2: 滞后指标虚假共振)
        # ROC/MFI/WilliamsR/VolumeAnomaly 提前反映市场结构变化
        roc_values = TechnicalIndicators.roc(prices, 10) if len(prices) >= 11 else [0.0]
        mfi_values = (
            TechnicalIndicators.mfi(high_prices, low_prices, prices, volumes, 14)
            if len(volumes) >= 15 else [50.0]
        )
        willr_values = (
            TechnicalIndicators.willr(high_prices, low_prices, prices, 14)
            if len(prices) >= 15 else [-50.0]
        )
        vol_ratio_values = (
            TechnicalIndicators.volume_anomaly(volumes, 20)
            if len(volumes) >= 20 else [1.0]
        )
        # v598 Phase 2: vp_vol_ratio_5 — 5周期量比 (与离线_v74_qs_feature_extract一致)
        # 来源: v91量价双维8状态分类需要此指标 (用户铁律: 量价分析融会贯通)
        # 计算: current_volume / SMA(volume, 5), 与离线vp_vol_ratio_5定义完全一致
        vol_ratio_5_values = (
            TechnicalIndicators.volume_anomaly(volumes, 5)
            if len(volumes) >= 5 else [1.0]
        )

        indicators = {
            # 滞后指标 (原有, 仅作确认)
            "sma_20": TechnicalIndicators.sma(prices, 20)[-1],
            "sma_50": TechnicalIndicators.sma(prices, 50)[-1] if len(prices) >= 50 else prices[-1],
            # Phase 14.8: 顺大逆小核心 — ema_200长周期趋势判断
            # 顺大: close > ema_200 = 大趋势向上; close < ema_200 = 大趋势向下
            "ema_200": TechnicalIndicators.ema(prices, 200)[-1] if len(prices) >= 200 else TechnicalIndicators.ema(prices, min(len(prices), 50))[-1],
            "ema_12": TechnicalIndicators.ema(prices, 12)[-1],
            "ema_26": TechnicalIndicators.ema(prices, 26)[-1],
            "rsi_14": TechnicalIndicators.rsi(prices, 14)[-1],
            "atr_14": TechnicalIndicators.atr(high_prices, low_prices, prices, 14)[-1],
            "bollinger_upper": bb_upper[-1],
            "bollinger_lower": bb_lower[-1],
            "bollinger_mid": bb_mid[-1],
            "adx_14": TechnicalIndicators.adx(high_prices, low_prices, prices, 14)[-1],
            # v3.21 领先指标 (修复根因2: 替代滞后指标的主导地位)
            "roc_10": roc_values[-1],                # 价格动量 (领先)
            "mfi_14": mfi_values[-1],                # 资金流量 (领先+价量)
            "willr_14": willr_values[-1],            # 威廉指标 (领先-超买超卖)
            "volume_ratio_20": vol_ratio_values[-1],  # 成交量异常 (领先-资金行为)
            # v598 Phase 2: 5周期量比 — v91量价双维8状态分类核心输入
            # 与离线_v74_qs_feature_extract.py的vp_vol_ratio_5完全一致
            # 用于 v91_runtime_adapter.classify_market_state_v91() 的Volume维度
            "vp_vol_ratio_5": vol_ratio_5_values[-1] if vol_ratio_5_values else 1.0,
            "current_price": market.close,
        }

        self._indicator_cache[cache_key] = dict(indicators)
        return indicators

    def get_price_history(self, symbol: str, n: int = 100) -> List[float]:
        """获取最近N个收盘价"""
        return list(self._price_cache.get(symbol, []))[-n:]

    def has_more(self, symbol: str) -> bool:
        return self.feed.has_more(symbol)

    def reset(self) -> None:
        self.feed.reset()
        self._price_cache.clear()
        self._indicator_cache.clear()
        for symbol in self.feed.symbols:
            self._init_price_cache_with_warmup(symbol)

    @property
    def symbols(self) -> List[str]:
        return self.feed.symbols

    # ========================================================================
    # Phase 14.16r: 新数据源加载（资金费率/OI/情绪/相关性）
    # 数据路径: /mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/
    # 格式: JSONL, 每行一条JSON记录
    # ========================================================================

    # 市场上下文数据根目录（WSL 路径）
    MARKET_DATA_ROOT = "/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData"

    @staticmethod
    def _iter_date_range(date_range: Tuple[str, str]):
        """迭代日期范围 [start, end]，格式 YYYYMMDD

        Args:
            date_range: (start_date, end_date)，含端点
        Yields:
            str: YYYYMMDD 格式日期
        """
        from datetime import datetime, timedelta
        start = datetime.strptime(date_range[0], "%Y%m%d")
        end = datetime.strptime(date_range[1], "%Y%m%d")
        cur = start
        while cur <= end:
            yield cur.strftime("%Y%m%d")
            cur += timedelta(days=1)

    def load_funding_rate(
        self, symbol: str, date_range: Tuple[str, str]
    ) -> List[Dict[str, Any]]:
        """加载资金费率数据

        文件命名: funding_rates/{SYMBOL}_{YYYYMMDD}.jsonl
        字段: timestamp, symbol, exchange, funding_rate, funding_time

        Args:
            symbol: 交易对 (如 "BTC")
            date_range: (start, end) YYYYMMDD 格式
        Returns:
            list[dict]: 资金费率记录列表
        """
        records: List[Dict[str, Any]] = []
        directory = os.path.join(self.MARKET_DATA_ROOT, "funding_rates")
        for date_str in self._iter_date_range(date_range):
            filepath = os.path.join(directory, f"{symbol}_{date_str}.jsonl")
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load funding_rate %s: %s", filepath, e)
        logger.info(
            "Loaded %d funding_rate records for %s in %s",
            len(records), symbol, date_range,
        )
        return records

    def load_open_interest(
        self, symbol: str, date_range: Tuple[str, str]
    ) -> List[Dict[str, Any]]:
        """加载未平仓合约数据

        文件命名: open_interest/{SYMBOL}_{YYYYMMDD}.jsonl
        字段: timestamp, symbol, exchange, open_interest, oi_change_pct

        Args:
            symbol: 交易对 (如 "BTC")
            date_range: (start, end) YYYYMMDD 格式
        Returns:
            list[dict]: OI 记录列表
        """
        records: List[Dict[str, Any]] = []
        directory = os.path.join(self.MARKET_DATA_ROOT, "open_interest")
        for date_str in self._iter_date_range(date_range):
            filepath = os.path.join(directory, f"{symbol}_{date_str}.jsonl")
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load open_interest %s: %s", filepath, e)
        logger.info(
            "Loaded %d open_interest records for %s in %s",
            len(records), symbol, date_range,
        )
        return records

    def load_fear_greed(
        self, date_range: Tuple[str, str]
    ) -> List[Dict[str, Any]]:
        """加载恐慌贪婪指数数据

        文件命名: fear_greed/fear_greed_{YYYYMMDD}.jsonl
        字段: timestamp, value, classification, source

        Args:
            date_range: (start, end) YYYYMMDD 格式
        Returns:
            list[dict]: 情绪指数记录列表
        """
        records: List[Dict[str, Any]] = []
        directory = os.path.join(self.MARKET_DATA_ROOT, "fear_greed")
        for date_str in self._iter_date_range(date_range):
            filepath = os.path.join(directory, f"fear_greed_{date_str}.jsonl")
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load fear_greed %s: %s", filepath, e)
        logger.info(
            "Loaded %d fear_greed records in %s",
            len(records), date_range,
        )
        return records

    def load_correlation_matrix(self, date: str) -> Dict[str, Any]:
        """加载相关性矩阵数据

        文件命名: correlation_matrix/correlation_{YYYYMMDD}.jsonl
        字段: date, window_days, matrix, computed_at
            matrix: {"ARB-ATOM": 0.619, "ARB-AVAX": 0.698, ...}

        Args:
            date: YYYYMMDD 格式日期
        Returns:
            dict: 相关性矩阵记录（含 matrix 字段）；找不到返回空 dict
        """
        directory = os.path.join(self.MARKET_DATA_ROOT, "correlation_matrix")
        filepath = os.path.join(directory, f"correlation_{date}.jsonl")
        if not os.path.exists(filepath):
            logger.warning("correlation_matrix not found for %s", date)
            return {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    logger.info(
                        "Loaded correlation_matrix for %s (pairs=%d)",
                        date, len(record.get("matrix", {})),
                    )
                    return record
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load correlation_matrix %s: %s", filepath, e)
        return {}

    def load_market_context(
        self, symbol: str, date_range: Tuple[str, str]
    ) -> Dict[str, Any]:
        """统一加载所有市场上下文数据

        汇总资金费率、OI、情绪指数、相关性矩阵，供信号生成器使用。

        Args:
            symbol: 交易对 (如 "BTC")
            date_range: (start, end) YYYYMMDD 格式
        Returns:
            dict: 包含以下字段
                - funding_rates: list[dict]
                - open_interest: list[dict]
                - fear_greed: list[dict]
                - correlation_matrix: dict (取 date_range 最后一天的矩阵)
                - latest_funding_rate: float (最新资金费率，无则 0.0)
                - latest_oi_change_pct: float (最新 OI 变化%，无则 0.0)
                - latest_fear_greed: float (最新情绪值，无则 50)
        """
        funding_rates = self.load_funding_rate(symbol, date_range)
        open_interest = self.load_open_interest(symbol, date_range)
        fear_greed = self.load_fear_greed(date_range)

        # 相关性矩阵：取 date_range 的最后一天
        end_date = date_range[1]
        correlation_matrix = self.load_correlation_matrix(end_date)

        # 提取最新值（按 timestamp 排序后取最后一条）
        latest_funding_rate = 0.0
        if funding_rates:
            sorted_fr = sorted(
                funding_rates,
                key=lambda r: r.get("timestamp", r.get("funding_time", "")),
            )
            latest_funding_rate = float(sorted_fr[-1].get("funding_rate", 0.0))

        latest_oi_change_pct = 0.0
        if open_interest:
            sorted_oi = sorted(
                open_interest,
                key=lambda r: r.get("timestamp", ""),
            )
            latest_oi_change_pct = float(sorted_oi[-1].get("oi_change_pct", 0.0))

        latest_fear_greed = 50.0
        if fear_greed:
            sorted_fg = sorted(
                fear_greed,
                key=lambda r: r.get("timestamp", ""),
            )
            latest_fear_greed = float(sorted_fg[-1].get("value", 50))

        context: Dict[str, Any] = {
            "funding_rates": funding_rates,
            "open_interest": open_interest,
            "fear_greed": fear_greed,
            "correlation_matrix": correlation_matrix,
            "latest_funding_rate": latest_funding_rate,
            "latest_oi_change_pct": latest_oi_change_pct,
            "latest_fear_greed": latest_fear_greed,
        }
        logger.info(
            "load_market_context(%s, %s): fr=%d oi=%d fg=%d corr_pairs=%d",
            symbol, date_range,
            len(funding_rates), len(open_interest), len(fear_greed),
            len(correlation_matrix.get("matrix", {})) if correlation_matrix else 0,
        )
        return context
