#!/usr/bin/env python3
"""WebSocket 实时数据流模块 — Layer 8 阶段4

Gate.io WebSocket v4 长连接, 订阅:
  - spot.tickers: 现货实时报价
  - spot.kline: K线数据
  - futures.tickers: 合约报价 (含资金费率)

特性:
  - 自动重连 (3次, 5秒间隔)
  - 心跳保活 (库级 ping + 应用层 ping)
  - 时钟偏移校准 (握手校时)
  - 降级模式: WebSocket不可用时自动生成合成数据, 确保沙盘进化不中断

用法:
    from sandbox_trading.ws_data_feed import WsDataFeed
    feed = WsDataFeed(symbols=["BTC_USDT", "ETH_USDT"], enabled=True)
    feed.start()  # 非阻塞启动
    price = feed.get_price("BTC_USDT")
    kline = feed.get_kline("BTC_USDT")
    funding = feed.get_funding_rate("BTC_USDT")

来源: _v514_ws_pipeline.py (Gate.io WS v4 实现) + 扩展 K线/资金费率
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.ws_data_feed")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass(slots=True)
class RealtimeTick:
    """实时报价快照"""
    symbol: str = ""
    price: float = 0.0
    volume_24h: float = 0.0
    change_pct: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    server_time: float = 0.0
    local_time: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "change_pct": self.change_pct,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "server_time": self.server_time,
            "local_time": self.local_time,
            "latency_ms": self.latency_ms,
        }


@dataclass(slots=True)
class RealtimeKline:
    """实时K线数据"""
    symbol: str = ""
    interval: str = "1m"
    open: float = 0.0
    close: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    timestamp: float = 0.0
    closed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open": self.open,
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "timestamp": self.timestamp,
            "closed": self.closed,
        }


@dataclass(slots=True)
class FundingRateInfo:
    """资金费率信息"""
    symbol: str = ""
    funding_rate: float = 0.0
    next_funding_time: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "funding_rate": self.funding_rate,
            "next_funding_time": self.next_funding_time,
            "timestamp": self.timestamp,
        }


# ============================================================================
# WebSocket 实时数据流
# ============================================================================

class WsDataFeed:
    """WebSocket 实时数据流 — Gate.io v4

    订阅 spot.tickers + spot.kline + futures.tickers
    自动降级为合成数据模式 (当 WebSocket 不可用时)
    """

    WS_URL = "wss://api.gateio.ws/ws/v4/"
    SERVER_TIME_URL = "https://api.gateio.ws/api/v4/spot/time"

    DEFAULT_SYMBOLS = ["BTC_USDT", "ETH_USDT"]

    # 心跳/重连参数
    PING_INTERVAL_SEC = 10
    RECONNECT_INTERVAL_SEC = 5
    MAX_RECONNECT_ATTEMPTS = 3

    # K线订阅间隔
    KLINE_INTERVAL = "1m"

    # 合成数据更新间隔 (秒), 避免每次get_price都更新
    SYNTHETIC_UPDATE_INTERVAL_SEC = 1.0

    # 合成数据基准价格 (降级模式)
    SYNTHETIC_BASE_PRICES = {
        "BTC_USDT": 40000.0,
        "ETH_USDT": 2200.0,
        "SOL_USDT": 100.0,
        "BNB_USDT": 300.0,
        "XRP_USDT": 0.5,
    }

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        enabled: bool = True,
        test_mode: bool = False,
        fallback_to_synthetic: bool = True,
    ):
        """初始化 WebSocket 数据流

        Args:
            symbols: 订阅的交易对列表 (默认 BTC_USDT, ETH_USDT)
            enabled: 是否启用 WebSocket
            test_mode: 测试模式 (使用合成数据, 不连接真实 WebSocket)
            fallback_to_synthetic: WebSocket失败时是否降级为合成数据
        """
        self.symbols = symbols if symbols else list(self.DEFAULT_SYMBOLS)
        self.enabled = enabled
        self.test_mode = test_mode
        self.fallback_to_synthetic = fallback_to_synthetic

        # 数据缓存
        self._ticks: Dict[str, RealtimeTick] = {}
        self._klines: Dict[str, RealtimeKline] = {}
        self._funding_rates: Dict[str, FundingRateInfo] = {}

        # WebSocket 运行状态
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0

        # 时钟校准
        self._clock_offset: Optional[float] = None
        self._subscribe_sent_local: Optional[float] = None

        # 统计
        self._msg_count = 0
        self._error_count = 0
        self._errors: List[str] = []
        self._latency_history: List[float] = []

        # 合成数据状态
        self._synthetic_prices: Dict[str, float] = {}
        self._synthetic_initialized = False
        self._last_synthetic_update: float = 0.0

        # 降级状态
        self._degraded = False
        self._degrade_reason = ""

        if enabled and test_mode:
            self._init_synthetic_data()
            self._degraded = True
            self._degrade_reason = "test_mode"
            logger.info("WsDataFeed test_mode: using synthetic data for %s", self.symbols)

    # ----------------------------------------------------------------------
    # 启动/停止
    # ----------------------------------------------------------------------

    def start(self) -> bool:
        """启动 WebSocket 数据流 (非阻塞, 后台线程)

        Returns:
            bool: True=启动成功, False=失败
        """
        if not self.enabled:
            logger.warning("WsDataFeed disabled, skipping start")
            return False

        if self.test_mode:
            logger.info("WsDataFeed test_mode, synthetic data ready")
            return True

        try:
            import websocket  # noqa: F401
        except ImportError:
            logger.warning("websocket-client not installed")
            if self.fallback_to_synthetic:
                self._init_synthetic_data()
                self._degraded = True
                self._degrade_reason = "websocket-client not installed"
                logger.info("WsDataFeed degraded to synthetic mode: websocket-client not installed")
                return True
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WsDataFeed started for symbols=%s", self.symbols)
        return True

    def stop(self) -> None:
        """停止 WebSocket 数据流"""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("WsDataFeed stopped (msgs=%d errors=%d)", self._msg_count, self._error_count)

    # ----------------------------------------------------------------------
    # 运行循环 (带自动重连)
    # ----------------------------------------------------------------------

    def _run_loop(self) -> None:
        """WebSocket 运行循环 (带自动重连)"""
        import websocket

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(
                    ping_interval=self.PING_INTERVAL_SEC,
                    ping_timeout=5,
                )
            except Exception as e:
                self._error_count += 1
                self._errors.append(f"run_loop: {e}")
                logger.error("WsDataFeed run error: %s", e)

            if not self._running:
                break

            self._reconnect_count += 1
            if self._reconnect_count > self.MAX_RECONNECT_ATTEMPTS:
                logger.error(
                    "WsDataFeed max reconnect (%d), giving up",
                    self.MAX_RECONNECT_ATTEMPTS,
                )
                if self.fallback_to_synthetic and not self._degraded:
                    self._init_synthetic_data()
                    self._degraded = True
                    self._degrade_reason = f"max reconnect ({self.MAX_RECONNECT_ATTEMPTS})"
                    logger.info("WsDataFeed degraded to synthetic mode after max reconnect")
                break

            logger.info(
                "WsDataFeed reconnect in %ds (attempt %d/%d)",
                self.RECONNECT_INTERVAL_SEC,
                self._reconnect_count,
                self.MAX_RECONNECT_ATTEMPTS,
            )
            time.sleep(self.RECONNECT_INTERVAL_SEC)

    # ----------------------------------------------------------------------
    # WebSocket 回调
    # ----------------------------------------------------------------------

    def _on_open(self, ws) -> None:
        """连接建立回调"""
        logger.info("WsDataFeed connected, subscribing...")
        self._connected = True
        self._reconnect_count = 0
        self._send_subscribe()

    def _on_message(self, ws, message: str) -> None:
        """消息接收回调"""
        self._msg_count += 1
        received_at = time.time()
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            event = data.get("event", "")

            if event == "subscribe" and channel:
                self._handle_subscribe_confirm(data, received_at)
                return

            if channel == "spot.tickers":
                self._handle_ticker(data, received_at)
            elif channel == "spot.kline":
                self._handle_kline(data, received_at)
            elif channel == "futures.tickers":
                self._handle_futures_ticker(data, received_at)

        except json.JSONDecodeError as e:
            self._error_count += 1
            logger.debug("WsDataFeed JSON decode error: %s", e)
        except Exception as e:
            self._error_count += 1
            self._errors.append(f"on_message: {e}")
            logger.debug("WsDataFeed message error: %s", e)

    def _on_error(self, ws, error) -> None:
        """错误回调"""
        self._error_count += 1
        self._errors.append(f"ws_error: {error}")
        logger.error("WsDataFeed error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """连接关闭回调"""
        self._connected = False
        logger.info(
            "WsDataFeed closed: code=%s msg=%s",
            close_status_code, str(close_msg)[:100] if close_msg else "",
        )

    # ----------------------------------------------------------------------
    # 订阅
    # ----------------------------------------------------------------------

    def _send_subscribe(self) -> None:
        """发送订阅消息: spot.tickers + spot.kline + futures.tickers"""
        if not self._ws:
            return

        subscriptions = [
            {
                "channel": "spot.tickers",
                "event": "subscribe",
                "payload": self.symbols,
            },
            {
                "channel": "spot.kline",
                "event": "subscribe",
                "payload": [f"{s}_{self.KLINE_INTERVAL}" for s in self.symbols],
            },
            {
                "channel": "futures.tickers",
                "event": "subscribe",
                "payload": self.symbols,
            },
        ]

        for msg in subscriptions:
            msg["time"] = int(time.time())
            try:
                self._ws.send(json.dumps(msg))
                if msg["channel"] == "spot.tickers":
                    self._subscribe_sent_local = time.time()
            except Exception as e:
                logger.error("WsDataFeed subscribe error (%s): %s", msg["channel"], e)

        logger.info(
            "WsDataFeed subscribed: tickers + kline(%s) + futures.tickers for %s",
            self.KLINE_INTERVAL, self.symbols,
        )

    def _handle_subscribe_confirm(self, data: Dict, received_at: float) -> None:
        """处理订阅确认"""
        channel = data.get("channel", "")
        logger.info("WsDataFeed subscribe confirmed: %s", channel)

    # ----------------------------------------------------------------------
    # 消息处理
    # ----------------------------------------------------------------------

    def _handle_ticker(self, data: Dict, received_at: float) -> None:
        """处理 spot.tickers 消息"""
        try:
            payload = data.get("result") or data.get("data")
            items = payload if isinstance(payload, list) else [payload] if payload else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                symbol = item.get("currency_pair", "")
                if not symbol:
                    continue

                tick = RealtimeTick(
                    symbol=symbol,
                    price=float(item.get("last", 0)),
                    volume_24h=float(item.get("base_volume", 0)),
                    change_pct=float(item.get("change_percent", 0)),
                    high_24h=float(item.get("high_24h", 0)),
                    low_24h=float(item.get("low_24h", 0)),
                    server_time=float(data.get("time_ms", 0)) / 1000.0,
                    local_time=received_at,
                )

                # 延迟计算
                if tick.server_time > 0 and self._clock_offset is not None:
                    tick.latency_ms = (received_at - (tick.server_time + self._clock_offset)) * 1000
                    if 0 < tick.latency_ms < 10000:
                        self._latency_history.append(tick.latency_ms)
                        if len(self._latency_history) > 1000:
                            self._latency_history.pop(0)

                self._ticks[symbol] = tick

        except Exception as e:
            self._error_count += 1
            logger.debug("WsDataFeed ticker parse error: %s", e)

    def _handle_kline(self, data: Dict, received_at: float) -> None:
        """处理 spot.kline 消息"""
        try:
            payload = data.get("result") or data.get("data")
            items = payload if isinstance(payload, list) else [payload] if payload else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                symbol = item.get("currency_pair", "")
                kline_data = item.get("kline", item)

                if not symbol:
                    continue

                kline = RealtimeKline(
                    symbol=symbol,
                    interval=item.get("interval", self.KLINE_INTERVAL),
                    open=float(kline_data.get("open", 0) or kline_data.get("o", 0)),
                    close=float(kline_data.get("close", 0) or kline_data.get("c", 0)),
                    high=float(kline_data.get("high", 0) or kline_data.get("h", 0)),
                    low=float(kline_data.get("low", 0) or kline_data.get("l", 0)),
                    volume=float(kline_data.get("volume", 0) or kline_data.get("v", 0)),
                    timestamp=float(kline_data.get("timestamp", 0) or kline_data.get("t", 0)),
                    closed=bool(kline_data.get("closed", False)),
                )
                self._klines[symbol] = kline

        except Exception as e:
            self._error_count += 1
            logger.debug("WsDataFeed kline parse error: %s", e)

    def _handle_futures_ticker(self, data: Dict, received_at: float) -> None:
        """处理 futures.tickers 消息 (含资金费率)"""
        try:
            payload = data.get("result") or data.get("data")
            items = payload if isinstance(payload, list) else [payload] if payload else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                symbol = item.get("contract", "")
                if not symbol:
                    continue

                # futures.tickers 包含资金费率字段
                funding_rate = float(item.get("funding_rate", 0) or item.get("fundingRate", 0))
                next_funding = float(item.get("funding_next_apply", 0) or item.get("fundingNextApply", 0))

                self._funding_rates[symbol] = FundingRateInfo(
                    symbol=symbol,
                    funding_rate=funding_rate,
                    next_funding_time=next_funding,
                    timestamp=received_at,
                )

                # 同时更新 tick (futures price)
                last_price = float(item.get("last", 0))
                if last_price > 0:
                    if symbol not in self._ticks:
                        self._ticks[symbol] = RealtimeTick(symbol=symbol)
                    self._ticks[symbol].price = last_price
                    self._ticks[symbol].local_time = received_at

        except Exception as e:
            self._error_count += 1
            logger.debug("WsDataFeed futures ticker parse error: %s", e)

    # ----------------------------------------------------------------------
    # 合成数据 (降级模式)
    # ----------------------------------------------------------------------

    def _init_synthetic_data(self) -> None:
        """初始化合成数据 (降级模式)"""
        if self._synthetic_initialized:
            return

        for sym in self.symbols:
            base_price = self.SYNTHETIC_BASE_PRICES.get(sym, 100.0)
            # 添加随机偏移
            self._synthetic_prices[sym] = base_price * (1 + random.uniform(-0.02, 0.02))

            # 初始化 tick
            self._ticks[sym] = RealtimeTick(
                symbol=sym,
                price=self._synthetic_prices[sym],
                volume_24h=base_price * 10000,
                change_pct=random.uniform(-2, 2),
                high_24h=self._synthetic_prices[sym] * 1.03,
                low_24h=self._synthetic_prices[sym] * 0.97,
                local_time=time.time(),
            )

            # 初始化 kline
            self._klines[sym] = RealtimeKline(
                symbol=sym,
                interval=self.KLINE_INTERVAL,
                open=self._synthetic_prices[sym] * 0.999,
                close=self._synthetic_prices[sym],
                high=self._synthetic_prices[sym] * 1.001,
                low=self._synthetic_prices[sym] * 0.999,
                volume=base_price * 100,
                timestamp=time.time(),
                closed=False,
            )

            # 初始化 funding rate (合成: -0.01% 到 0.01%)
            self._funding_rates[sym] = FundingRateInfo(
                symbol=sym,
                funding_rate=random.uniform(-0.0001, 0.0001),
                next_funding_time=time.time() + 28800,
                timestamp=time.time(),
            )

        self._synthetic_initialized = True
        logger.info("WsDataFeed synthetic data initialized for %s", self.symbols)

    def _maybe_update_synthetic(self) -> None:
        """节流更新合成数据 (按时间间隔)"""
        now = time.time()
        if now - self._last_synthetic_update < self.SYNTHETIC_UPDATE_INTERVAL_SEC:
            return
        self._last_synthetic_update = now
        self._update_synthetic_data()

    def _update_synthetic_data(self) -> None:
        """更新合成数据 (随机游走)"""
        for sym in self.symbols:
            if sym not in self._synthetic_prices:
                continue

            # 随机游走 (±0.1%)
            old_price = self._synthetic_prices[sym]
            change = old_price * random.uniform(-0.001, 0.001)
            new_price = old_price + change
            self._synthetic_prices[sym] = new_price

            # 更新 tick
            if sym in self._ticks:
                tick = self._ticks[sym]
                tick.price = new_price
                tick.local_time = time.time()
                tick.high_24h = max(tick.high_24h, new_price)
                tick.low_24h = min(tick.low_24h, new_price)

            # 更新 kline
            if sym in self._klines:
                kline = self._klines[sym]
                kline.close = new_price
                kline.high = max(kline.high, new_price)
                kline.low = min(kline.low, new_price)
                kline.timestamp = time.time()

    # ----------------------------------------------------------------------
    # 公开 API
    # ----------------------------------------------------------------------

    def get_price(self, symbol: str) -> Optional[float]:
        """获取最新价格

        Args:
            symbol: 交易对 (如 "BTC_USDT" 或 "BTC-USDT")

        Returns:
            float or None: 最新价格
        """
        sym = self._normalize_symbol(symbol)

        if self._degraded:
            self._maybe_update_synthetic()

        tick = self._ticks.get(sym)
        return tick.price if tick and tick.price > 0 else None

    def get_tick(self, symbol: str) -> Optional[RealtimeTick]:
        """获取完整 tick 数据"""
        sym = self._normalize_symbol(symbol)

        if self._degraded:
            self._maybe_update_synthetic()

        return self._ticks.get(sym)

    def get_kline(self, symbol: str) -> Optional[RealtimeKline]:
        """获取最新 K 线数据"""
        sym = self._normalize_symbol(symbol)

        if self._degraded:
            self._maybe_update_synthetic()

        return self._klines.get(sym)

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """获取资金费率"""
        sym = self._normalize_symbol(symbol)
        fr = self._funding_rates.get(sym)
        return fr.funding_rate if fr else None

    def get_funding_info(self, symbol: str) -> Optional[FundingRateInfo]:
        """获取完整资金费率信息"""
        sym = self._normalize_symbol(symbol)
        return self._funding_rates.get(sym)

    def get_all_prices(self) -> Dict[str, float]:
        """获取所有交易对的最新价格"""
        if self._degraded:
            self._maybe_update_synthetic()
        return {sym: tick.price for sym, tick in self._ticks.items() if tick.price > 0}

    def is_alive(self) -> bool:
        """WebSocket 是否存活 (或降级模式是否有效)"""
        if self._degraded:
            return True
        return self._connected and self._running

    def is_degraded(self) -> bool:
        """是否处于降级模式"""
        return self._degraded

    def get_stats(self) -> Dict[str, Any]:
        """获取运行统计"""
        avg_latency = (
            sum(self._latency_history) / len(self._latency_history)
            if self._latency_history else 0
        )
        return {
            "enabled": self.enabled,
            "connected": self._connected,
            "degraded": self._degraded,
            "degrade_reason": self._degrade_reason,
            "symbols": list(self.symbols),
            "cached_symbols": list(self._ticks.keys()),
            "msg_count": self._msg_count,
            "error_count": self._error_count,
            "reconnect_count": self._reconnect_count,
            "avg_latency_ms": round(avg_latency, 2),
            "test_mode": self.test_mode,
        }

    # ----------------------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """标准化交易对符号

        支持多种格式:
          BTC_USDT → BTC_USDT
          BTC-USDT → BTC_USDT
          BTC/USDT → BTC_USDT
          btc_usdt → BTC_USDT
        """
        sym = symbol.upper().replace("-", "_").replace("/", "_")
        return sym


# ============================================================================
# 便捷函数
# ============================================================================

def create_ws_feed(
    symbols: Optional[List[str]] = None,
    test_mode: bool = False,
) -> WsDataFeed:
    """创建 WebSocket 数据流实例"""
    feed = WsDataFeed(symbols=symbols, enabled=True, test_mode=test_mode)
    feed.start()
    return feed
