# -*- coding: utf-8 -*-
"""
Gate.io WebSocket 实时行情接入 v3.31

解决v347报告的关键瓶颈:
  - REST API延迟: mean=1141ms, p95=2133ms (超目标11倍)
  - 目标: ≤100ms (用户要求)

Gate.io WebSocket v4:
  - 现货: wss://api.gateio.ws/ws/v4/
  - 订阅 spot.tickers 实时ticker推送
  - 订阅 spot.trades 实时成交推送
  - 订阅 spot.candlesticks 实时K线推送

架构:
  - WebSocketStreamer: 异步WebSocket行情流
  - 实时缓存最新行情 (in-memory)
  - 延迟监控 (≤100ms目标)
  - 自动重连机制

模块职责: LIVE-DATA-CONNECTOR + LOSS-HUNTER (本智能体协助)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("hermes.gateio_ws")


# ============================================================================
# 1. 数据结构
# ============================================================================


@dataclass
class RealtimeTick:
    """实时tick数据"""
    symbol: str
    timestamp: float
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    received_at: float = field(default_factory=time.time)

    @property
    def latency_ms(self) -> float:
        """接收延迟(ms)"""
        return (self.received_at - self.timestamp) * 1000


# ============================================================================
# 2. WebSocket 行情流
# ============================================================================


class GateIoWebSocketStreamer:
    """Gate.io WebSocket 实时行情流

    使用WebSocket订阅实时行情,延迟目标≤100ms (vs REST 1141ms)

    使用方式:
        streamer = GateIoWebSocketStreamer()
        streamer.subscribe(["BTC_USDT", "ETH_USDT"])
        streamer.start()
        # 获取最新行情
        tick = streamer.get_latest("BTC_USDT")
        print(f"BTC价格: {tick.price}, 延迟: {tick.latency_ms:.1f}ms")
    """

    WS_URL = "wss://api.gateio.ws/ws/v4/"

    def __init__(self):
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscribed_symbols: List[str] = []
        self._latest: Dict[str, RealtimeTick] = {}
        self._lock = threading.Lock()
        self._latency_history: deque = deque(maxlen=1000)
        self._message_count = 0
        self._error_count = 0
        self._last_heartbeat = time.time()
        self._callbacks: List[Callable[[RealtimeTick], None]] = []

    # ------------------------------------------------------------------
    # 订阅管理
    # ------------------------------------------------------------------

    def subscribe(self, symbols: List[str]) -> None:
        """订阅交易对行情"""
        self._subscribed_symbols.extend(symbols)
        if self._ws:
            self._send_subscribe(symbols)

    def add_callback(self, callback: Callable[[RealtimeTick], None]) -> None:
        """添加行情回调"""
        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """启动WebSocket连接"""
        try:
            import websocket  # noqa: F401  # 检查库可用性
        except ImportError:
            logger.error("websocket-client库未安装,无法启动WebSocket流")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("GateIoWebSocketStreamer启动, 订阅: %s", self._subscribed_symbols)
        return True

    def stop(self) -> None:
        """停止WebSocket连接"""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("GateIoWebSocketStreamer已停止")

    # ------------------------------------------------------------------
    # 内部运行循环
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """WebSocket运行循环(带自动重连)"""
        import websocket

        backoff = 1
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=10, ping_timeout=5)
                backoff = 1  # 重置backoff
            except Exception as e:
                self._error_count += 1
                logger.error("WebSocket连接异常: %s", e)

            if self._running:
                logger.info("WebSocket断开, %ds后重连...", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)  # 指数backoff, 最大60s

    def _on_open(self, ws) -> None:
        """连接建立回调"""
        logger.info("WebSocket连接建立")
        if self._subscribed_symbols:
            self._send_subscribe(self._subscribed_symbols)

    def _on_message(self, ws, message: str) -> None:
        """消息接收回调

        v3.31.1 修复: 提取消息头部的server time用于真实延迟测量
        根因: 原实现用time.time()覆盖timestamp, 导致latency_ms永远~0ms (假数据)
        正确做法: 使用Gate.io消息头的time字段(服务器时间戳)计算网络延迟
        """
        try:
            self._message_count += 1
            received_at = time.time()
            data = json.loads(message)
            self._last_heartbeat = received_at

            # v3.31.1: 提取服务器时间戳 (消息头部的time字段, 单位:秒)
            server_time = data.get("time")
            if server_time is not None:
                try:
                    server_time = float(server_time)
                except (TypeError, ValueError):
                    server_time = None

            # 处理ticker推送
            channel = data.get("channel", "")
            if channel == "spot.tickers":
                self._handle_ticker(data, server_time, received_at)
            elif channel == "spot.trades":
                self._handle_trade(data, server_time, received_at)
            elif channel == "heartbeat":
                pass  # 心跳
            elif "result" in data:
                pass  # 订阅确认
        except Exception as e:
            self._error_count += 1
            logger.debug("消息处理异常: %s", e)

    def _on_error(self, ws, error) -> None:
        """错误回调"""
        self._error_count += 1
        logger.error("WebSocket错误: %s", error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """连接关闭回调"""
        logger.info("WebSocket关闭: code=%s msg=%s", close_status_code, close_msg)

    # ------------------------------------------------------------------
    # 订阅与消息处理
    # ------------------------------------------------------------------

    def _send_subscribe(self, symbols: List[str]) -> None:
        """发送订阅消息"""
        if not self._ws:
            return
        msg = {
            "time": int(time.time()),
            "channel": "spot.tickers",
            "event": "subscribe",
            "payload": symbols,
        }
        try:
            self._ws.send(json.dumps(msg))
            logger.info("已订阅: %s", symbols)
        except Exception as e:
            logger.error("订阅失败: %s", e)

    def _handle_ticker(self, data: Dict[str, Any], server_time: Optional[float], received_at: float) -> None:
        """处理ticker推送

        v3.31.1: 使用server_time计算真实网络延迟
        """
        try:
            payload = data.get("result", data.get("data", {}))
            if isinstance(payload, list):
                for item in payload:
                    self._parse_ticker(item, server_time, received_at)
            elif isinstance(payload, dict):
                self._parse_ticker(payload, server_time, received_at)
        except Exception as e:
            logger.debug("ticker处理异常: %s", e)

    def _parse_ticker(self, item: Dict[str, Any], server_time: Optional[float], received_at: float) -> None:
        """解析单个ticker

        v3.31.1 修复延迟测量根因:
          - 原: timestamp=time.time() → latency_ms ≈ 0ms (假数据, 误导优化决策)
          - 新: timestamp=server_time (Gate.io消息头) → latency_ms = 真实网络延迟
          - fallback: 若无server_time, latency_ms=0并标记为不可信
        """
        try:
            symbol = item.get("currency_pair", "").replace("_", "-")
            if not symbol:
                return

            # v3.31.1: 优先使用服务器时间戳, 无则用接收时间(fallback)
            ts = server_time if server_time is not None else received_at
            tick = RealtimeTick(
                symbol=symbol,
                timestamp=ts,
                price=float(item.get("last", 0)) or 0.0,
                volume=float(item.get("base_volume", 0)) or 0.0,
                bid=float(item.get("highest_bid", 0)) or 0.0,
                ask=float(item.get("lowest_ask", 0)) or 0.0,
                received_at=received_at,
            )

            with self._lock:
                self._latest[symbol] = tick
                # v3.31.1: 只在server_time有效时记录延迟 (避免假0ms污染统计)
                if server_time is not None:
                    self._latency_history.append(tick.latency_ms)

            # 触发回调
            for cb in self._callbacks:
                try:
                    cb(tick)
                except Exception as e:
                    logger.debug("回调异常: %s", e)
        except Exception as e:
            logger.debug("ticker解析异常: %s", e)

    def _handle_trade(self, data: Dict[str, Any], server_time: Optional[float], received_at: float) -> None:
        """处理成交推送

        v3.31.1: trade消息的create_time_ms是毫秒级服务器时间戳, 精度更高
        """
        try:
            payload = data.get("result", data.get("data", {}))
            if isinstance(payload, list):
                for item in payload:
                    self._parse_trade(item, server_time, received_at)
            elif isinstance(payload, dict):
                self._parse_trade(payload, server_time, received_at)
        except Exception as e:
            logger.debug("trade处理异常: %s", e)

    def _parse_trade(self, item: Dict[str, Any], server_time: Optional[float], received_at: float) -> None:
        """解析单个成交

        v3.31.1: trade消息的create_time_ms是毫秒级服务器时间戳, 精度更高
        Gate.io trades 推送字段:
          - create_time: 秒级时间戳
          - create_time_ms: 毫秒级时间戳 (优先使用, 精度更高)
        """
        try:
            symbol = item.get("currency_pair", "").replace("_", "-")
            if not symbol:
                return

            # v3.31.1: 优先使用毫秒级时间戳 (精度更高), 其次秒级, 最后server_time
            ts_ms = item.get("create_time_ms")
            ts_s = item.get("create_time")
            if ts_ms is not None:
                ts = float(ts_ms) / 1000.0
            elif ts_s is not None:
                ts = float(ts_s)
            elif server_time is not None:
                ts = server_time
            else:
                ts = received_at  # fallback (无法计算真实延迟)

            tick = RealtimeTick(
                symbol=symbol,
                timestamp=ts,
                price=float(item.get("price", 0)) or 0.0,
                volume=float(item.get("amount", 0)) or 0.0,
                received_at=received_at,
            )
            with self._lock:
                self._latest[symbol] = tick
                # v3.31.1: 只在ts来自服务器时记录延迟 (避免假0ms)
                if ts_ms is not None or ts_s is not None or server_time is not None:
                    self._latency_history.append(tick.latency_ms)
        except Exception as e:
            logger.debug("trade解析异常: %s", e)

    # ------------------------------------------------------------------
    # 公开查询接口
    # ------------------------------------------------------------------

    def get_latest(self, symbol: str) -> Optional[RealtimeTick]:
        """获取最新tick"""
        symbol = symbol.replace("_", "-")
        with self._lock:
            return self._latest.get(symbol)

    def get_all_latest(self) -> Dict[str, RealtimeTick]:
        """获取所有最新tick"""
        with self._lock:
            return dict(self._latest)

    def get_latency_stats(self) -> Dict[str, Any]:
        """获取延迟统计"""
        with self._lock:
            if not self._latency_history:
                return {"available": False}
            latencies = list(self._latency_history)

        import statistics
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        return {
            "available": True,
            "n": n,
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p50_ms": latencies_sorted[n // 2],
            "p95_ms": latencies_sorted[int(n * 0.95)] if n >= 20 else max(latencies),
            "p99_ms": latencies_sorted[int(n * 0.99)] if n >= 100 else max(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": statistics.stdev(latencies) if n > 1 else 0.0,
            "meets_target_100ms": latencies_sorted[int(n * 0.95)] <= 100 if n >= 20 else False,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "last_heartbeat_age_s": time.time() - self._last_heartbeat,
        }

    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self._running and (time.time() - self._last_heartbeat) < 30


# ============================================================================
# 3. 全局实例管理
# ============================================================================


# v598 Phase H: 默认订阅 10 个主要交易对 (按现货成交量排名)
# 来源: CoinGecko 2026 Q2 现货成交量 Top 10 (排除稳定币对 USDC/TUSD/FDUSD)
# 用户铁律 "理解各指标的底层逻辑和市场含义": 主流币种流动性高, 价差小, 滑点低
# 格式: Gate.io 使用下划线 (BTC_USDT), CCXT 使用横线 (BTC-USDT)
#       _parse_ticker 内部已处理 _ → - 转换 (L259)
DEFAULT_TOP_10_SYMBOLS: List[str] = [
    "BTC_USDT",   # 比特币 — 市值第一, 流动性最佳
    "ETH_USDT",   # 以太坊 — 智能合约平台基石
    "SOL_USDT",   # Solana — 高 TPS 公链代表
    "BNB_USDT",   # 币安币 — 交易所生态代币
    "DOGE_USDT",  # 狗狗币 — 社区驱动高波动
    "XRP_USDT",   # 瑞波币 — 跨境支付
    "ADA_USDT",   # Cardano — 学术派公链
    "AVAX_USDT",  # Avalanche — 子网架构
    "LINK_USDT",  # Chainlink — 预言机龙头
    "DOT_USDT",   # Polkadot — 跨链协议
]


_global_streamer: Optional[GateIoWebSocketStreamer] = None


def get_global_streamer() -> GateIoWebSocketStreamer:
    """获取全局WebSocket流实例(单例)"""
    global _global_streamer
    if _global_streamer is None:
        _global_streamer = GateIoWebSocketStreamer()
    return _global_streamer


def start_realtime_stream(symbols: Optional[List[str]] = None) -> bool:
    """启动实时行情流(便捷接口)

    v598 Phase H: symbols 参数改为可选, 默认使用 DEFAULT_TOP_10_SYMBOLS (10 个主要交易对)
    解决 gateio_websocket_provider.py:6 注释自承 "REST API延迟: mean=1141ms, p95=2133ms
    (超目标11倍)" — WS 长连接 p95 目标 ≤100ms

    Args:
        symbols: 交易对列表, 如 ["BTC_USDT", "ETH_USDT"]
                 None 时使用 DEFAULT_TOP_10_SYMBOLS (10 个主要交易对)

    Returns:
        True=启动成功
    """
    if symbols is None:
        symbols = list(DEFAULT_TOP_10_SYMBOLS)
    streamer = get_global_streamer()
    streamer.subscribe(symbols)
    return streamer.start()


def start_default_realtime_stream() -> bool:
    """v598 Phase H: 启动默认 10 交易对实时行情流 (便捷接口)

    等价于 start_realtime_stream(DEFAULT_TOP_10_SYMBOLS)
    用于快速启动主流币种行情监控, 延迟目标 ≤100ms (vs REST 1141ms)
    """
    return start_realtime_stream(DEFAULT_TOP_10_SYMBOLS)


def stop_realtime_stream() -> None:
    """停止实时行情流"""
    global _global_streamer
    if _global_streamer:
        _global_streamer.stop()
        _global_streamer = None


def get_realtime_price(symbol: str) -> Optional[float]:
    """获取实时价格(便捷接口)"""
    streamer = get_global_streamer()
    tick = streamer.get_latest(symbol)
    return tick.price if tick else None
