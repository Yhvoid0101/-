#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v515 高保真市场仿真器 — Agent-5 核心交付物
================================================================
目标: 消除模拟环境与实盘交易的表现差异, 将差异率降低至 15% 以内。

复现的微观结构因素:
  1. 点差 (spread): 基于历史波动率动态计算, 非固定值
     - 基准: BTC~0.02%, ETH~0.04%, SOL~0.08%, BNB~0.05%, XRP~0.10%
     - 动态: spread = base_spread * (1 + vol_multiplier * (cur_vol / avg_vol - 1))
  2. 滑点 (slippage): 基于订单大小 vs 流动性
     - slippage = base_slippage * (order_size / avg_volume)^0.5
  3. 流动性 (liquidity): 基于交易时段 (亚/欧/美) 和波动率
     - 亚洲 (00-08 UTC, 低), 欧洲 (08-16 UTC, 中), 美国 (16-24 UTC, 高)
  4. 订单执行延迟: 基于 Agent-1 WebSocket 实测数据
     - WS 中位数 222.646ms, p95 1134.866ms (来自 _v514_ws_test_result.json)
     - REST 中位数 1056ms (来自 INTERFACE_SPEC.md)
     - 延迟导致入场价格偏移: price_drift = price * vol_per_second * (delay / 3600)
  5. 部分成交 (partial fills): 基于订单大小 vs 可用流动性
  6. 资金费率 (永续合约): 8h 周期, 影响持仓成本

设计原理:
  - 复用 v515 策略的指标计算和信号生成 (确保策略一致性)
  - 在 v515 回测引擎之上叠加微观结构成本
  - 可独立运行, 也可被对比工具调用
  - 输出与 v515 相同格式的 metrics, 便于直接对比
"""
import json
import sys
import math
import time
import os
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# ============================================================
# 路径配置
# ============================================================
HERE = Path(__file__).parent.resolve()
SANDBOX_DIR = HERE.parent.parent  # sandbox_trading/
V515_STRATEGY_PATH = SANDBOX_DIR / "_v515_ensemble_strategy.py"
DATA_CACHE_GATEIO = SANDBOX_DIR / "data_cache_gateio"
WS_TEST_RESULT = HERE.parent / "agent_1_pipeline" / "_v514_ws_test_result.json"
SIM_RESULT_PATH = HERE / "_v515_hifi_sim_result.json"
SIM_DETAIL_PATH = HERE / "_v515_hifi_sim_detail.json"

# 导入 v515 策略模块 (同一进程, 确保策略一致)
sys.path.insert(0, str(SANDBOX_DIR))
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("v515_ensemble_strategy", V515_STRATEGY_PATH)
    v515 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(v515)
    V515_AVAILABLE = True
except Exception as e:
    print(f"[WARN] 无法导入 v515 策略模块: {e}")
    V515_AVAILABLE = False

# ============================================================
# 符号配置 (与 v515 一致)
# ============================================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# Gate.io 数据文件中符号带下划线 (BTC_USDT)
SYMBOL_TO_GATEIO = {
    "BTCUSDT": "BTC_USDT",
    "ETHUSDT": "ETH_USDT",
    "SOLUSDT": "SOL_USDT",
    "BNBUSDT": "BNB_USDT",
    "XRPUSDT": "XRP_USDT",
}

# ============================================================
# 微观结构参数 (基于真实市场数据校准)
# ============================================================

# 基准点差 (bps = 0.01%), 来自 Gate.io 实盘观察
BASE_SPREAD_BPS = {
    "BTCUSDT": 2.0,    # 0.02%
    "ETHUSDT": 4.0,    # 0.04%
    "SOLUSDT": 8.0,    # 0.08%
    "BNBUSDT": 5.0,    # 0.05%
    "XRPUSDT": 10.0,   # 0.10%
}

# 基准滑点系数 (订单大小冲击)
BASE_SLIPPAGE_COEFF = {
    "BTCUSDT": 0.00005,   # 流动性最好
    "ETHUSDT": 0.00008,
    "SOLUSDT": 0.00015,
    "BNBUSDT": 0.00012,
    "XRPUSDT": 0.00020,   # 流动性最差
}

# 交易时段流动性倍数 (UTC 时间)
# 亚洲 (00-08 UTC): 低流动性, 大户稀少
# 欧洲 (08-16 UTC): 中流动性
# 美国 (16-24 UTC): 高流动性, 美股开盘
SESSION_LIQUIDITY_MULT = {
    "asia": 0.65,    # 亚洲时段流动性 65%
    "europe": 1.00,  # 欧洲时段基准
    "us": 1.35,      # 美国时段流动性 135%
}

# Agent-1 WebSocket 实测延迟 (来自 _v514_ws_test_result.json)
# median=222.646ms, p95=1134.866ms, mean=423.166ms
# REST API 中位数 1056ms (INTERFACE_SPEC.md)
DEFAULT_WS_LATENCY_MS = 222.646
DEFAULT_REST_LATENCY_MS = 1056.0
DEFAULT_EXEC_LATENCY_MS = 818.0  # 任务书指定基准 (混合 WS/REST 路径)

# 资金费率 (8h 周期, 永续合约)
# 历史平均: BTC ~0.01%/8h, ETH ~0.015%/8h, 山寨币 ~0.02%/8h
FUNDING_RATE_8H = {
    "BTCUSDT": 0.0001,   # 0.01% per 8h
    "ETHUSDT": 0.00015,
    "SOLUSDT": 0.00020,
    "BNBUSDT": 0.00018,
    "XRPUSDT": 0.00025,
}
FUNDING_PERIOD_HOURS = 8

# v515 基础成本 (理想回测假设)
V515_FEE_RATE = 0.0004
V515_SLIPPAGE = 0.0002
V515_ROUND_TRIP_COST = 2 * (V515_FEE_RATE + V515_SLIPPAGE)  # 0.12%

# 部分成交阈值: 订单大小超过该比例的可用流动性时开始部分成交
PARTIAL_FILL_THRESHOLD = 0.05  # 订单 > 5% 可用流动性时部分成交


# ============================================================
# 1. 数据加载 (兼容 Gate.io 永续 K 线格式)
# ============================================================
def load_gateio_klines(symbol: str, max_bars: int = 18169) -> List[Dict]:
    """
    加载 Gate.io 永续合约 K 线数据, 转换为 v515 兼容格式。

    v515 期望字段: ts (毫秒), open, high, low, close, volume
    Gate.io 提供: timestamp (秒, float), open, high, low, close, volume, quote_volume
    """
    gateio_sym = SYMBOL_TO_GATEIO.get(symbol, symbol)
    candidates = [
        DATA_CACHE_GATEIO / f"perp_kline_{gateio_sym}_1h_1500.json",
        DATA_CACHE_GATEIO / f"perp_kline_{gateio_sym}_1h_1000.json",
        DATA_CACHE_GATEIO / f"perp_kline_{gateio_sym}_1h_500.json",
        DATA_CACHE_GATEIO / f"perp_history_{gateio_sym}_1h_540d.json",
    ]

    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 200:
                    # 转换为 v515 兼容格式
                    bars = []
                    for b in data[:max_bars]:
                        ts_sec = b.get("timestamp", 0)
                        # Gate.io timestamp 是秒 (float), v515 期望毫秒
                        ts_ms = int(float(ts_sec) * 1000) if ts_sec > 0 else 0
                        bars.append({
                            "ts": ts_ms,
                            "open": float(b["open"]),
                            "high": float(b["high"]),
                            "low": float(b["low"]),
                            "close": float(b["close"]),
                            "volume": float(b.get("volume", 0)),
                            "quote_volume": float(b.get("quote_volume", 0)),
                        })
                    return bars
            except Exception as e:
                print(f"  [load_gateio_klines] {p} error: {e}")

    # 回退到 v515 的加载函数
    if V515_AVAILABLE:
        return v515.load_klines(symbol, max_bars)
    return []


# ============================================================
# 2. 微观结构模型
# ============================================================
@dataclass
class MarketMicrostructure:
    """单根 K 线的微观结构参数 (动态计算)"""
    spread_bps: float = 0.0          # 当前点差 (bps)
    slippage_bps: float = 0.0        # 当前滑点 (bps)
    liquidity_mult: float = 1.0      # 流动性倍数
    session: str = "europe"          # 当前交易时段
    exec_delay_ms: float = 0.0       # 执行延迟 (ms)
    funding_rate: float = 0.0        # 当前资金费率
    partial_fill_ratio: float = 1.0  # 成交比例 (1.0 = 全部成交)


def get_trading_session(ts_ms: int) -> str:
    """根据时间戳 (毫秒) 判断交易时段 (UTC)"""
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
    hour = dt.hour
    if 0 <= hour < 8:
        return "asia"
    elif 8 <= hour < 16:
        return "europe"
    else:
        return "us"


def calc_volatility(closes: np.ndarray, window: int = 24) -> float:
    """计算短期波动率 (24h 收益率标准差)"""
    if len(closes) < window + 1:
        return 0.02  # 默认 2%
    returns = np.diff(np.log(closes[-window - 1:]))
    return float(np.std(returns))


def calc_avg_volatility(closes: np.ndarray, window: int = 240) -> float:
    """计算长期平均波动率 (用于归一化)"""
    if len(closes) < window + 1:
        return 0.02
    returns = np.diff(np.log(closes[-window - 1:]))
    return float(np.std(returns))


def calc_dynamic_spread(symbol: str, cur_vol: float, avg_vol: float) -> float:
    """
    动态点差计算:
      - 基准点差 (按符号)
      - 波动率放大: spread = base * (1 + 0.5 * (cur_vol/avg_vol - 1))
      - 高波动时点差扩大 (做市商风险补偿)
    """
    base = BASE_SPREAD_BPS.get(symbol, 5.0)
    if avg_vol > 0:
        vol_ratio = cur_vol / avg_vol
        # 波动率越高, 点差越大 (非线性)
        multiplier = 1.0 + 0.5 * max(0, vol_ratio - 1)
        multiplier = min(multiplier, 3.0)  # 上限 3x
    else:
        multiplier = 1.0
    return base * multiplier


def calc_slippage(symbol: str, order_size_quote: float, avg_quote_volume: float) -> float:
    """
    滑点计算 (平方根模型):
      slippage = base_slippage * sqrt(order_size / avg_volume)

    order_size_quote: 订单金额 (USDT)
    avg_quote_volume: 平均成交额 (USDT/小时)
    """
    base = BASE_SLIPPAGE_COEFF.get(symbol, 0.0001)
    if avg_quote_volume > 0 and order_size_quote > 0:
        size_ratio = order_size_quote / avg_quote_volume
        # 平方根模型 (Almgren-Chriss 简化版)
        slippage = base * math.sqrt(max(size_ratio, 0))
        # 上限 50bps (防止极端值)
        return min(slippage, 0.005)
    return 0.0


def calc_partial_fill_ratio(order_size_quote: float, available_liquidity: float) -> float:
    """
    部分成交比例:
      - 订单 < 5% 可用流动性: 全部成交 (1.0)
      - 订单 > 5%: 线性衰减
      - 订单 > 50%: 至少成交 30%
    """
    if available_liquidity <= 0:
        return 1.0
    ratio = order_size_quote / available_liquidity
    if ratio <= PARTIAL_FILL_THRESHOLD:
        return 1.0
    # 线性衰减: 5% → 100%成交, 50% → 30%成交
    decay = (ratio - PARTIAL_FILL_THRESHOLD) / (0.50 - PARTIAL_FILL_THRESHOLD)
    fill_ratio = 1.0 - 0.7 * min(decay, 1.0)
    return max(fill_ratio, 0.30)


def calc_exec_delay(use_ws: bool = True, volatility: float = 0.02) -> float:
    """
    执行延迟 (ms):
      - WS 路径: 中位数 222.646ms (Agent-1 实测)
      - REST 路径: 中位数 1056ms
      - 高波动时延迟增加 (网络拥塞)
    """
    base_delay = DEFAULT_WS_LATENCY_MS if use_ws else DEFAULT_REST_LATENCY_MS
    # 高波动时延迟增加 20-50%
    vol_mult = 1.0 + max(0, (volatility - 0.02) / 0.02) * 0.3
    return base_delay * min(vol_mult, 2.0)


def calc_price_drift(price: float, delay_ms: float, volatility: float) -> float:
    """
    延迟导致的价格漂移:
      - 假设价格服从几何布朗运动
      - 1 小时的波动率 = volatility
      - delay 秒的漂移标准差 = price * volatility * sqrt(delay/3600)
    返回漂移量的标准差 (绝对值)
    """
    delay_hours = delay_ms / 1000.0 / 3600.0
    return price * volatility * math.sqrt(delay_hours)


def compute_microstructure(symbol: str, bar: Dict, closes: np.ndarray,
                            order_size_quote: float = 10000.0,
                            use_ws: bool = True) -> MarketMicrostructure:
    """计算单根 K 线的完整微观结构"""
    ts_ms = bar["ts"]
    session = get_trading_session(ts_ms)
    cur_vol = calc_volatility(closes)
    avg_vol = calc_avg_volatility(closes)

    spread_bps = calc_dynamic_spread(symbol, cur_vol, avg_vol)

    avg_quote_vol = bar.get("quote_volume", 1000000)
    liquidity_mult = SESSION_LIQUIDITY_MULT[session]
    effective_liquidity = avg_quote_vol * liquidity_mult

    slippage_bps = calc_slippage(symbol, order_size_quote, effective_liquidity) * 10000
    partial_fill = calc_partial_fill_ratio(order_size_quote, effective_liquidity)
    exec_delay = calc_exec_delay(use_ws, cur_vol)
    funding = FUNDING_RATE_8H.get(symbol, 0.0001)

    return MarketMicrostructure(
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        liquidity_mult=liquidity_mult,
        session=session,
        exec_delay_ms=exec_delay,
        funding_rate=funding,
        partial_fill_ratio=partial_fill,
    )


# ============================================================
# 3. 高保真回测引擎 (在 v515 之上叠加微观结构)
# ============================================================
def high_fidelity_backtest(bars: List[Dict], indicators: Dict[str, np.ndarray],
                            start: int, end: int, genes: Dict,
                            symbol: str,
                            order_size_quote: float = 10000.0,
                            use_ws: bool = True,
                            enable_spread: bool = True,
                            enable_slippage: bool = True,
                            enable_delay: bool = True,
                            enable_partial_fill: bool = True,
                            enable_funding: bool = True) -> Dict:
    """
    高保真回测: 在 v515 策略信号之上叠加真实微观结构成本。

    与 v515 backtest_window 的差异:
      1. 入场价格叠加: 点差 + 滑点 + 延迟漂移
      2. 出场价格叠加: 点差 + 滑点
      3. 持仓期间扣除: 资金费率
      4. 大单部分成交: 降低实际成交仓位
      5. 所有成本累加到 pnl

    返回与 v515 calc_metrics 兼容的指标 + 微观结构统计
    """
    closes = indicators["closes"]
    highs = indicators["highs"]
    lows = indicators["lows"]
    atr = indicators["atr"]

    position = None
    trades = []
    cooldown_until = -1
    consecutive_losses = 0
    max_consecutive_losses = 0
    max_hold_hours = 240
    atr_trail_mult = genes.get("atr_trail_mult", 2.241)

    # 微观结构统计
    ms_stats = {
        "spread_bps_sum": 0.0,
        "slippage_bps_sum": 0.0,
        "delay_drift_bps_sum": 0.0,
        "funding_cost_bps_sum": 0.0,
        "partial_fill_count": 0,
        "session_counts": {"asia": 0, "europe": 0, "us": 0},
        "total_entries": 0,
        "total_exits": 0,
        "avg_delay_ms": 0.0,
        "delay_samples": [],
    }

    for i in range(max(start, 30), end):
        if i < cooldown_until:
            continue

        bar = bars[i] if i < len(bars) else None
        if bar is None:
            continue

        # 更新时段统计
        session = get_trading_session(bar["ts"])
        ms_stats["session_counts"][session] += 1

        if position is not None:
            hold_hours = i - position["entry_idx"]
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]

            # 计算当前微观结构 (出场时)
            cur_closes = closes[max(0, i - 240):i + 1]
            ms_exit = compute_microstructure(
                symbol, bar, cur_closes,
                order_size_quote=order_size_quote,
                use_ws=use_ws,
            ) if (enable_spread or enable_slippage) else None

            # 资金费率成本 (每 8 小时结算一次)
            if enable_funding and hold_hours > 0:
                funding_periods = hold_hours / FUNDING_PERIOD_HOURS
                funding_rate = FUNDING_RATE_8H.get(symbol, 0.0001)
                funding_cost_bps = funding_rate * funding_periods * 10000
                ms_stats["funding_cost_bps_sum"] += funding_cost_bps

            if position["dir"] == "LONG":
                # 移动止损
                if hold_hours > max_hold_hours / 3 and atr[i] > 0:
                    new_trail = bar_close - atr_trail_mult * atr[i]
                    if new_trail > position["stop_loss"]:
                        position["stop_loss"] = new_trail

                exit_price = None
                exit_reason = ""

                if bar_low <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "stop"
                elif bar_high >= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "tp"
                elif hold_hours >= max_hold_hours:
                    exit_price = bar_close
                    exit_reason = "timeout"

                if exit_price is not None:
                    # 叠加出场微观结构成本
                    actual_exit = exit_price
                    if ms_exit and enable_spread:
                        spread_cost = ms_exit.spread_bps / 10000 * exit_price
                        actual_exit -= spread_cost  # 卖出时点差不利
                        ms_stats["spread_bps_sum"] += ms_exit.spread_bps
                    if ms_exit and enable_slippage:
                        slip_cost = ms_exit.slippage_bps / 10000 * exit_price
                        actual_exit -= slip_cost
                        ms_stats["slippage_bps_sum"] += ms_exit.slippage_bps

                    # 延迟导致出场价格偏移
                    if enable_delay and ms_exit:
                        drift = calc_price_drift(exit_price, ms_exit.exec_delay_ms, calc_volatility(closes[max(0, i-240):i+1]))
                        # 出场时延迟对多头不利 (可能滑得更低)
                        actual_exit -= drift * 0.5
                        ms_stats["delay_drift_bps_sum"] += (drift * 0.5 / exit_price) * 10000
                        ms_stats["delay_samples"].append(ms_exit.exec_delay_ms)

                    pnl = (actual_exit - position["entry_price"]) / position["entry_price"]

                    # 扣除资金费率
                    if enable_funding:
                        funding_periods = hold_hours / FUNDING_PERIOD_HOURS
                        funding_rate = FUNDING_RATE_8H.get(symbol, 0.0001)
                        pnl -= funding_rate * funding_periods

                    # 基础手续费 (v515 假设)
                    pnl -= V515_ROUND_TRIP_COST

                    # 部分成交调整: 实际仓位可能小于满仓
                    fill_ratio = position.get("fill_ratio", 1.0)
                    if fill_ratio < 1.0:
                        pnl *= fill_ratio

                    trades.append({
                        "pnl": pnl, "dir": "LONG",
                        "ts": bar["ts"], "hold": hold_hours,
                        "strategy": position.get("strategy", "?"),
                        "exit_reason": exit_reason,
                        "fill_ratio": fill_ratio,
                        "microstructure": {
                            "spread_bps": ms_exit.spread_bps if ms_exit else 0,
                            "slippage_bps": ms_exit.slippage_bps if ms_exit else 0,
                            "delay_ms": ms_exit.exec_delay_ms if ms_exit else 0,
                        }
                    })
                    position = None
                    cooldown_until = i + 4
                    ms_stats["total_exits"] += 1
                    if pnl < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
                    continue
            else:  # SHORT
                if hold_hours > max_hold_hours / 3 and atr[i] > 0:
                    new_trail = bar_close + atr_trail_mult * atr[i]
                    if new_trail < position["stop_loss"] or position["stop_loss"] == 0:
                        position["stop_loss"] = new_trail

                exit_price = None
                exit_reason = ""

                if position["stop_loss"] > 0 and bar_high >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                    exit_reason = "stop"
                elif bar_low <= position["take_profit"]:
                    exit_price = position["take_profit"]
                    exit_reason = "tp"
                elif hold_hours >= max_hold_hours:
                    exit_price = bar_close
                    exit_reason = "timeout"

                if exit_price is not None:
                    actual_exit = exit_price
                    if ms_exit and enable_spread:
                        spread_cost = ms_exit.spread_bps / 10000 * exit_price
                        actual_exit += spread_cost  # 买入平仓时点差不利
                        ms_stats["spread_bps_sum"] += ms_exit.spread_bps
                    if ms_exit and enable_slippage:
                        slip_cost = ms_exit.slippage_bps / 10000 * exit_price
                        actual_exit += slip_cost
                        ms_stats["slippage_bps_sum"] += ms_exit.slippage_bps

                    if enable_delay and ms_exit:
                        drift = calc_price_drift(exit_price, ms_exit.exec_delay_ms, calc_volatility(closes[max(0, i-240):i+1]))
                        actual_exit += drift * 0.5  # 空头延迟不利
                        ms_stats["delay_drift_bps_sum"] += (drift * 0.5 / exit_price) * 10000
                        ms_stats["delay_samples"].append(ms_exit.exec_delay_ms)

                    pnl = (position["entry_price"] - actual_exit) / position["entry_price"]

                    if enable_funding:
                        funding_periods = hold_hours / FUNDING_PERIOD_HOURS
                        funding_rate = FUNDING_RATE_8H.get(symbol, 0.0001)
                        # 空头收取资金费率 (相反方向)
                        pnl += funding_rate * funding_periods * 0.5  # 部分对冲

                    pnl -= V515_ROUND_TRIP_COST

                    fill_ratio = position.get("fill_ratio", 1.0)
                    if fill_ratio < 1.0:
                        pnl *= fill_ratio

                    trades.append({
                        "pnl": pnl, "dir": "SHORT",
                        "ts": bar["ts"], "hold": hold_hours,
                        "strategy": position.get("strategy", "?"),
                        "exit_reason": exit_reason,
                        "fill_ratio": fill_ratio,
                        "microstructure": {
                            "spread_bps": ms_exit.spread_bps if ms_exit else 0,
                            "slippage_bps": ms_exit.slippage_bps if ms_exit else 0,
                            "delay_ms": ms_exit.exec_delay_ms if ms_exit else 0,
                        }
                    })
                    position = None
                    cooldown_until = i + 4
                    ms_stats["total_exits"] += 1
                    if pnl < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0
                    continue
            continue

        # 生成信号 (复用 v515)
        if V515_AVAILABLE:
            direction, stop, tp, strength, strategy = v515.combined_signal(indicators, i, genes)
        else:
            direction, stop, tp, strength, strategy = ("flat", 0.0, 0.0, 0.0, "none")

        if direction == "flat":
            continue

        # 入场微观结构
        entry = closes[i]
        cur_closes = closes[max(0, i - 240):i + 1]
        ms_entry = compute_microstructure(
            symbol, bar, cur_closes,
            order_size_quote=order_size_quote,
            use_ws=use_ws,
        )

        actual_entry = entry
        if enable_spread:
            spread_cost = ms_entry.spread_bps / 10000 * entry
            if direction == "long":
                actual_entry += spread_cost  # 买入时点差不利
            else:
                actual_entry -= spread_cost  # 卖出时点差不利
            ms_stats["spread_bps_sum"] += ms_entry.spread_bps

        if enable_slippage:
            slip_cost = ms_entry.slippage_bps / 10000 * entry
            if direction == "long":
                actual_entry += slip_cost
            else:
                actual_entry -= slip_cost
            ms_stats["slippage_bps_sum"] += ms_entry.slippage_bps

        if enable_delay:
            drift = calc_price_drift(entry, ms_entry.exec_delay_ms, calc_volatility(cur_closes))
            # 延迟方向不确定, 用随机方向 (期望 0, 但增加波动)
            # 这里用确定性偏移: 假设延迟导致入场更差
            if direction == "long":
                actual_entry += drift * 0.5
            else:
                actual_entry -= drift * 0.5
            ms_stats["delay_drift_bps_sum"] += (drift * 0.5 / entry) * 10000
            ms_stats["delay_samples"].append(ms_entry.exec_delay_ms)

        # 部分成交
        fill_ratio = 1.0
        if enable_partial_fill:
            fill_ratio = ms_entry.partial_fill_ratio
            if fill_ratio < 1.0:
                ms_stats["partial_fill_count"] += 1

        atr_val = atr[i] if atr[i] > 0 else entry * 0.02

        if direction == "long":
            position = {
                "dir": "LONG", "entry_idx": i, "entry_price": actual_entry,
                "stop_loss": stop, "take_profit": tp,
                "signal_strength": strength, "strategy": strategy,
                "fill_ratio": fill_ratio,
            }
        else:
            position = {
                "dir": "SHORT", "entry_idx": i, "entry_price": actual_entry,
                "stop_loss": stop, "take_profit": tp,
                "signal_strength": strength, "strategy": strategy,
                "fill_ratio": fill_ratio,
            }
        ms_stats["total_entries"] += 1

    # 计算指标
    metrics = calc_hifi_metrics(trades, bars, start, end, max_consecutive_losses)
    metrics["microstructure_stats"] = ms_stats
    metrics["symbol"] = symbol
    return metrics


# ============================================================
# 4. 指标计算 (与 v515 兼容 + 微观结构扩展)
# ============================================================
def calc_hifi_metrics(trades: List[Dict], bars: List[Dict], start: int, end: int,
                       max_consec_loss: int) -> Dict:
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "profit_loss_ratio": 0,
            "sharpe": 0, "max_dd": 0, "annual_return": 0,
            "max_consecutive_losses": 0, "total_return": 0,
            "monthly_loss_prob": 0,
            "win_rate_ci_lo": 0.0, "win_rate_ci_hi": 1.0,
            "avg_fill_ratio": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    pf = avg_win / avg_loss if avg_loss > 0 else float('inf')

    cum = np.cumprod([1 + p for p in pnls])
    total_return = cum[-1] - 1 if len(cum) > 0 else 0

    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * math.sqrt(8760 / max(1, len(trades) / max(1, (end - start) / 8760)))
    else:
        sharpe = 0

    peak = cum[0]
    max_dd = 0
    for c in cum:
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    hours = end - start
    years = hours / 8760
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else 0

    monthly_loss_prob = calc_monthly_loss_prob(trades, bars, start, end)
    win_rate_ci_lo, win_rate_ci_hi = bootstrap_win_rate_ci(pnls)

    fill_ratios = [t.get("fill_ratio", 1.0) for t in trades]

    return {
        "n_trades": len(trades), "win_rate": win_rate,
        "profit_loss_ratio": pf, "sharpe": sharpe,
        "max_dd": max_dd, "annual_return": annual_return,
        "max_consecutive_losses": max_consec_loss,
        "total_return": total_return,
        "monthly_loss_prob": monthly_loss_prob,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "win_rate_ci_lo": win_rate_ci_lo,
        "win_rate_ci_hi": win_rate_ci_hi,
        "avg_fill_ratio": float(np.mean(fill_ratios)),
    }


def bootstrap_win_rate_ci(pnls: List[float], n_boot: int = 1000, seed: int = 515) -> Tuple[float, float]:
    """Bootstrap 95% CI for win rate"""
    if len(pnls) < 5:
        return (0.0, 1.0)
    rng = np.random.default_rng(seed)
    n = len(pnls)
    pnls_arr = np.array(pnls)
    boot_rates = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pnls_arr[idx]
        boot_rates[b] = np.mean(sample > 0)
    return (float(np.percentile(boot_rates, 2.5)), float(np.percentile(boot_rates, 97.5)))


def calc_monthly_loss_prob(trades: List[Dict], bars: List[Dict], start: int, end: int) -> float:
    if not trades:
        return 0.0
    monthly_pnl = {}
    for t in trades:
        ts = t["ts"]
        dt = datetime.datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=datetime.timezone.utc)
        month_key = f"{dt.year}-{dt.month:02d}"
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t["pnl"]
    if not monthly_pnl:
        return 0.0
    loss_months = sum(1 for v in monthly_pnl.values() if v < 0)
    months_with_trades = sum(1 for v in monthly_pnl.values() if v != 0)
    if months_with_trades == 0:
        return 0.0
    return loss_months / months_with_trades


# ============================================================
# 5. 单符号高保真仿真 (Walk-Forward)
# ============================================================
def simulate_symbol(symbol: str, genes: Optional[Dict] = None,
                     order_size_quote: float = 10000.0,
                     use_ws: bool = True) -> Dict:
    """
    对单个符号执行高保真仿真 (Walk-Forward)。

    返回:
      - ideal_metrics: 理想回测 (v515 默认, 无微观结构)
      - hifi_metrics: 高保真仿真 (含全部微观结构)
      - factor_isolation: 各因素单独影响
    """
    print(f"\n[Agent-5] 仿真 {symbol}...")

    bars = load_gateio_klines(symbol)
    if not bars or len(bars) < 500:
        print(f"  [WARN] {symbol} 数据不足 ({len(bars)} bars), 跳过")
        return {"symbol": symbol, "status": "no_data", "bars": len(bars)}

    print(f"  数据: {len(bars)} bars")

    if V515_AVAILABLE:
        indicators = v515.calc_indicators(bars)
    else:
        return {"symbol": symbol, "status": "v515_unavailable"}

    # 默认基因 (v515 中等表现)
    if genes is None:
        genes = {
            "breakout_n": 40.0, "atr_stop_mult": 3.0, "take_profit_r": 3.0,
            "atr_trail_mult": 2.241,
            "rsi_oversold": 32.0, "rsi_overbought": 68.0,
            "z_threshold": 1.5, "mr_atr_stop_mult": 2.0, "mr_take_profit_r": 1.5,
            "bo_atr_stop_mult": 2.0, "bo_take_profit_r": 3.0,
            "adx_threshold": 20.0, "bb_width_threshold": 0.04,
            "high_vol_threshold": 0.04,
        }

    n = len(bars)
    window_size = n // 4  # 4 个 Walk-Forward 窗口
    all_ideal = []
    all_hifi = []

    # 因素隔离测试 (每次只开一个因素)
    factor_tests = {
        "spread_only": dict(enable_spread=True, enable_slippage=False, enable_delay=False,
                            enable_partial_fill=False, enable_funding=False),
        "slippage_only": dict(enable_spread=False, enable_slippage=True, enable_delay=False,
                              enable_partial_fill=False, enable_funding=False),
        "delay_only": dict(enable_spread=False, enable_slippage=False, enable_delay=True,
                           enable_partial_fill=False, enable_funding=False),
        "funding_only": dict(enable_spread=False, enable_slippage=False, enable_delay=False,
                             enable_partial_fill=False, enable_funding=True),
        "partial_fill_only": dict(enable_spread=False, enable_slippage=False, enable_delay=False,
                                  enable_partial_fill=True, enable_funding=False),
        "all_enabled": dict(enable_spread=True, enable_slippage=True, enable_delay=True,
                            enable_partial_fill=True, enable_funding=True),
        "all_disabled": dict(enable_spread=False, enable_slippage=False, enable_delay=False,
                             enable_partial_fill=False, enable_funding=False),
    }

    factor_metrics = {}

    for w in range(4):
        ws = w * window_size
        we = (w + 1) * window_size
        oos_start = int(ws + (we - ws) * 0.65)  # 65% IS, 35% OOS

        # 理想回测 (v515 原生, 无微观结构)
        if V515_AVAILABLE:
            ideal_m = v515.backtest_window(bars, indicators, oos_start, we, genes)
        else:
            ideal_m = {"n_trades": 0, "sharpe": 0, "win_rate": 0, "annual_return": 0,
                       "max_dd": 0, "total_return": 0, "monthly_loss_prob": 0,
                       "max_consecutive_losses": 0}
        all_ideal.append(ideal_m)

        # 高保真仿真 (全部微观结构)
        hifi_m = high_fidelity_backtest(
            bars, indicators, oos_start, we, genes, symbol,
            order_size_quote=order_size_quote,
            use_ws=use_ws,
            **factor_tests["all_enabled"],
        )
        all_hifi.append(hifi_m)

        # 因素隔离 (只在第 1 个窗口做, 节省时间)
        if w == 0:
            for fname, fconfig in factor_tests.items():
                fm = high_fidelity_backtest(
                    bars, indicators, oos_start, we, genes, symbol,
                    order_size_quote=order_size_quote,
                    use_ws=use_ws,
                    **fconfig,
                )
                factor_metrics[fname] = {
                    "n_trades": fm["n_trades"],
                    "sharpe": fm["sharpe"],
                    "win_rate": fm["win_rate"],
                    "annual_return": fm["annual_return"],
                    "max_dd": fm["max_dd"],
                    "total_return": fm["total_return"],
                }

    # 聚合 Walk-Forward 结果
    ideal_agg = aggregate_metrics(all_ideal)
    hifi_agg = aggregate_metrics(all_hifi)

    # 合并微观结构统计
    hifi_agg["microstructure_stats"] = aggregate_ms_stats([m.get("microstructure_stats", {}) for m in all_hifi])
    hifi_agg["factor_isolation"] = factor_metrics

    return {
        "symbol": symbol,
        "status": "ok",
        "bars": len(bars),
        "ideal_metrics": ideal_agg,
        "hifi_metrics": hifi_agg,
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """聚合多个 Walk-Forward 窗口的指标"""
    if not metrics_list:
        return {}
    valid = [m for m in metrics_list if m.get("n_trades", 0) > 0]
    if not valid:
        return {"n_trades": 0, "sharpe": 0, "win_rate": 0, "annual_return": 0,
                "max_dd": 0, "total_return": 0, "monthly_loss_prob": 0,
                "max_consecutive_losses": 0}
    return {
        "n_trades": float(np.mean([m["n_trades"] for m in valid])),
        "sharpe": float(np.mean([m["sharpe"] for m in valid])),
        "win_rate": float(np.mean([m["win_rate"] for m in valid])),
        "annual_return": float(np.mean([m["annual_return"] for m in valid])),
        "max_dd": float(np.mean([m["max_dd"] for m in valid])),
        "total_return": float(np.mean([m["total_return"] for m in valid])),
        "monthly_loss_prob": float(np.mean([m["monthly_loss_prob"] for m in valid])),
        "max_consecutive_losses": float(np.mean([m["max_consecutive_losses"] for m in valid])),
        "profit_loss_ratio": float(np.mean([m.get("profit_loss_ratio", 0) for m in valid])),
    }


def aggregate_ms_stats(stats_list: List[Dict]) -> Dict:
    """聚合微观结构统计"""
    if not stats_list:
        return {}
    total_spread = sum(s.get("spread_bps_sum", 0) for s in stats_list)
    total_slippage = sum(s.get("slippage_bps_sum", 0) for s in stats_list)
    total_delay_drift = sum(s.get("delay_drift_bps_sum", 0) for s in stats_list)
    total_funding = sum(s.get("funding_cost_bps_sum", 0) for s in stats_list)
    total_partial = sum(s.get("partial_fill_count", 0) for s in stats_list)
    total_entries = sum(s.get("total_entries", 0) for s in stats_list)
    total_exits = sum(s.get("total_exits", 0) for s in stats_list)
    all_delays = []
    for s in stats_list:
        all_delays.extend(s.get("delay_samples", []))

    sessions = {"asia": 0, "europe": 0, "us": 0}
    for s in stats_list:
        for k in sessions:
            sessions[k] += s.get("session_counts", {}).get(k, 0)

    return {
        "total_spread_bps": total_spread,
        "total_slippage_bps": total_slippage,
        "total_delay_drift_bps": total_delay_drift,
        "total_funding_bps": total_funding,
        "partial_fill_count": total_partial,
        "total_entries": total_entries,
        "total_exits": total_exits,
        "avg_delay_ms": float(np.mean(all_delays)) if all_delays else 0.0,
        "p95_delay_ms": float(np.percentile(all_delays, 95)) if all_delays else 0.0,
        "session_distribution": sessions,
    }


# ============================================================
# 6. 主函数
# ============================================================
def main():
    print("=" * 70)
    print("Agent-5: 高保真市场仿真器")
    print("目标: 消除模拟 vs 实盘差异 (差异率 ≤ 15%)")
    print("=" * 70)
    print(f"延迟来源: Agent-1 WS 实测 (median={DEFAULT_WS_LATENCY_MS}ms, p95=1134.866ms)")
    print(f"REST 延迟: {DEFAULT_REST_LATENCY_MS}ms (INTERFACE_SPEC)")
    print(f"执行延迟基准: {DEFAULT_EXEC_LATENCY_MS}ms (任务书)")
    print(f"点差模型: 动态 (BTC~2bps, ETH~4bps, SOL~8bps, BNB~5bps, XRP~10bps)")
    print(f"滑点模型: sqrt(order_size / avg_volume)")
    print(f"流动性时段: 亚洲(低)/欧洲(中)/美国(高)")
    print(f"部分成交: 订单 > 5% 可用流动性时触发")
    print(f"资金费率: 8h 周期 (BTC~0.01%, 山寨~0.02%)")
    print()

    # 加载 Agent-1 延迟数据
    if WS_TEST_RESULT.exists():
        try:
            ws_data = json.loads(WS_TEST_RESULT.read_text(encoding="utf-8"))
            lat = ws_data.get("latency_stats", {})
            print(f"Agent-1 WS 延迟数据已加载:")
            print(f"  median={lat.get('median_ms', '?')}ms, p95={lat.get('p95_ms', '?')}ms, n={lat.get('n', '?')}")
        except Exception as e:
            print(f"[WARN] 无法加载 WS 延迟数据: {e}")
    else:
        print(f"[WARN] WS 延迟数据文件不存在: {WS_TEST_RESULT}")

    # 对每个符号执行仿真
    all_results = {}
    for sym in SYMBOLS:
        result = simulate_symbol(sym, order_size_quote=10000.0, use_ws=True)
        all_results[sym] = result

    # 保存结果
    output = {
        "version": "v515_hifi_simulator",
        "agent_id": "agent_5_simulator",
        "timestamp": time.time(),
        "latency_source": {
            "ws_median_ms": DEFAULT_WS_LATENCY_MS,
            "ws_p95_ms": 1134.866,
            "rest_median_ms": DEFAULT_REST_LATENCY_MS,
            "exec_baseline_ms": DEFAULT_EXEC_LATENCY_MS,
        },
        "symbols": all_results,
    }

    SIM_RESULT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[Agent-5] 仿真结果已保存: {SIM_RESULT_PATH}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("仿真汇总 (理想 vs 高保真)")
    print("=" * 70)
    print(f"{'符号':<10} {'模式':<10} {'交易数':<8} {'夏普':<8} {'胜率':<8} {'年化':<10} {'回撤':<8}")
    print("-" * 70)
    for sym, r in all_results.items():
        if r.get("status") != "ok":
            print(f"{sym:<10} {r.get('status', '?')}")
            continue
        ideal = r["ideal_metrics"]
        hifi = r["hifi_metrics"]
        print(f"{sym:<10} {'理想':<10} {ideal['n_trades']:<8.1f} {ideal['sharpe']:<8.3f} {ideal['win_rate']*100:<8.1f} {ideal['annual_return']*100:<10.2f} {ideal['max_dd']*100:<8.2f}")
        print(f"{'':<10} {'高保真':<10} {hifi['n_trades']:<8.1f} {hifi['sharpe']:<8.3f} {hifi['win_rate']*100:<8.1f} {hifi['annual_return']*100:<10.2f} {hifi['max_dd']*100:<8.2f}")

    return output


if __name__ == "__main__":
    main()
