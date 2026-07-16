# -*- coding: utf-8 -*-
"""
v514 Agent-1 数据管道: Gate.io WebSocket v4 长连接行情订阅

目标:
  - 替代 v510/v511 的 REST 轮询 (urllib.request)
  - 将行情延迟从 REST 1056ms 降至 WebSocket ≤300ms (现实目标)
  - 用户硬指标 ≤100ms 在国内物理不可达 (Binance 被封锁, Gate.io 是国内最佳)

实现要点 (吸取 v3.31.1 教训 + 外部最佳实践):
  1. 时钟偏移修正 (核心): 启动时通过 REST /api/v4/spot/time 获取服务器时间,
     计算 clock_offset = local_clock - server_clock。所有延迟测量都减去 offset,
     否则本地时钟与服务器时钟不同步会导致延迟测量失真 (负值或虚高)。
  2. 双路延迟测量:
     a) 行情延迟 = local_receive - (server_event_time + clock_offset)  [主指标]
     b) 应用层 RTT = 通过 spot.ping/spot.pong 往返时间  [交叉验证]
  3. v3.31.1 教训: 不用 time.time() 覆盖 timestamp 导致假 0ms。只用服务器时间戳,
     且仅在 clock_offset 有效时才记录延迟, 避免污染统计。
  4. 自动重连: 5 秒间隔, 最多 3 次 (任务硬要求)。
  5. 心跳: websocket-client 库级 ping_interval=10s + 应用层 spot.ping 每 10s。
  6. 行情缓存: 5 符号 (BTC/ETH/SOL/BNB/XRP_USDT) 最新价格字典 (线程安全)。
  7. 接口: get_latest_price(symbol) 同步方法返回缓存价格。
  8. CLI: --hours (默认 1 小时运行), --test (60 秒测试 + 延迟统计输出)。

依赖:
  - websocket-client (已验证 v1.9.0 可用)
  - urllib.request (标准库, 用于获取服务器时间)

模块职责: 数据管道优化 (Agent-1)
来源: v514 多智能体协作 / v508 延迟测试 / v3.31 WebSocket 实现经验
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v514.ws_pipeline")


# ============================================================================
# 1. 数据结构
# ============================================================================


@dataclass
class TickSnapshot:
    """单个交易对的最新行情快照 (线程安全访问由外层锁保证)"""

    symbol: str
    price: float
    server_time: float          # 服务器事件时间 (秒, 已修正 clock_offset)
    received_at: float          # 本地接收时间 (秒, time.time())
    volume_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

    @property
    def latency_ms(self) -> float:
        """网络+处理延迟 (ms) = 本地接收 - 服务器事件 (已修正时钟偏移)"""
        return max(0.0, (self.received_at - self.server_time) * 1000.0)


# ============================================================================
# 2. Gate.io WebSocket v4 行情管道
# ============================================================================


class GateIoWsV514Pipeline:
    """Gate.io WebSocket v4 长连接行情管道

    使用方式 (作为库):
        pipe = GateIoWsV514Pipeline(symbols=["BTC_USDT", "ETH_USDT"])
        pipe.start()                       # 非阻塞, 启动后台线程
        time.sleep(2)
        price = pipe.get_latest_price("BTC_USDT")
        stats = pipe.get_latency_stats()
        pipe.stop()

    使用方式 (CLI):
        python _v514_ws_pipeline.py --test        # 60 秒测试
        python _v514_ws_pipeline.py --hours 2     # 运行 2 小时
    """

    WS_URL = "wss://api.gateio.ws/ws/v4/"
    SERVER_TIME_URL = "https://api.gateio.ws/api/v4/spot/time"

    # 默认订阅的 5 个交易对 (任务硬要求)
    DEFAULT_SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT"]

    # 心跳/重连参数 (任务硬要求)
    PING_INTERVAL_SEC = 10          # 库级 ping
    APP_PING_INTERVAL_SEC = 10      # 应用层 spot.ping (用于 RTT 测量)
    RECONNECT_INTERVAL_SEC = 5      # 断线后 5 秒重连
    MAX_RECONNECT_ATTEMPTS = 3      # 最多 3 次

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        test_mode: bool = False,
        test_duration_sec: int = 60,
    ):
        """
        Args:
            symbols: 订阅的交易对列表 (默认 5 符号)
            test_mode: True=测试模式 (60 秒后自动停止)
            test_duration_sec: 测试模式运行时长 (秒)
        """
        self.symbols = symbols if symbols else list(self.DEFAULT_SYMBOLS)

        # WebSocket 运行状态
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._app_ping_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # 时钟偏移: local_clock - server_clock (秒)
        # 修正后: true_server_time = local_time - clock_offset
        # 延迟 = local_receive - (server_event + clock_offset)... 实际上:
        # 若 server_event 是服务器钟时间, local_receive 是本地钟时间,
        # 则 真实延迟 = local_receive - server_event - clock_offset
        self._clock_offset: Optional[float] = None
        self._clock_offset_fetched_at: float = 0.0

        # 行情缓存 (线程安全)
        self._latest: Dict[str, TickSnapshot] = {}
        self._lock = threading.Lock()

        # 延迟统计 (主指标: 基于 server time + clock_offset)
        self._latency_history: deque = deque(maxlen=10000)
        # RTT 统计 (交叉验证: 应用层 ping/pong)
        self._rtt_history: deque = deque(maxlen=1000)

        # 计数器
        self._message_count = 0
        self._ticker_count = 0
        self._error_count = 0
        self._reconnect_count = 0
        self._negative_latency_count = 0  # offset 偏高导致的负延迟计数 (诊断用)
        self._last_heartbeat = time.time()

        # 测试模式
        self._test_mode = test_mode
        self._test_duration_sec = test_duration_sec
        self._start_time: float = 0.0
        self._errors: List[str] = []

        # 待匹配的应用层 ping 时间戳 (key=发送时本地时间, value=同)
        # 用于 spot.ping/spot.pong RTT 匹配
        self._pending_ping_ts: Dict[float, float] = {}

        # 订阅握手时间戳 (用于无 TLS 开销的时钟校准)
        # subscribe 发送时的本地时间; 收到 subscribe 确认时计算 RTT+offset
        self._subscribe_sent_local: Optional[float] = None
        self._subscribe_confirmed: bool = False

    # ----------------------------------------------------------------------
    # 时钟偏移修正 (核心: 确保延迟测量真实)
    # ----------------------------------------------------------------------

    def _fetch_clock_offset(self) -> Optional[float]:
        """通过 REST 获取 Gate.io 服务器时间, 计算时钟偏移

        Gate.io /api/v4/spot/time 返回 {"server_time": <ms>}, 单位是毫秒 (13 位整数)。
        本地 time.time() 返回秒 (10 位浮点)。必须统一到秒。

        Returns:
            clock_offset (秒) = local_clock - server_clock, 失败返回 None
        """
        try:
            t0 = time.time()
            req = urllib.request.Request(
                self.SERVER_TIME_URL,
                headers={"User-Agent": "v514-ws-pipeline/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                t1 = time.time()
                data = json.loads(resp.read().decode("utf-8"))
            # 本地时间取请求往返中点, 抵消网络延迟
            local_mid = (t0 + t1) / 2.0
            server_time_raw = float(data.get("server_time", 0))
            if server_time_raw <= 0:
                return None
            # 单位归一化: Gate.io REST server_time 是毫秒, WS 消息 time 字段是秒
            # 若值 > 1e12 (13 位) 视为毫秒, 除以 1000 转秒
            if server_time_raw > 1e12:
                server_time_sec = server_time_raw / 1000.0
            else:
                server_time_sec = server_time_raw
            offset = local_mid - server_time_sec
            self._clock_offset_fetched_at = local_mid
            logger.info(
                "时钟偏移修正: offset=%.3fs (local=%.3f server=%.3f, rest_rtt=%.0fms)",
                offset, local_mid, server_time_sec, (t1 - t0) * 1000,
            )
            return offset
        except Exception as e:
            self._error_count += 1
            self._errors.append(f"clock_offset_fetch_failed: {e}")
            logger.warning("时钟偏移获取失败 (延迟测量可能不准): %s", e)
            return None

    # ----------------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------------

    def start(self) -> bool:
        """启动 WebSocket 管道 (非阻塞, 后台线程)"""
        try:
            import websocket  # noqa: F401
        except ImportError:
            self._errors.append("websocket-client not installed")
            logger.error("websocket-client 库未安装, 无法启动")
            return False

        # 先获取时钟偏移 (延迟测量准确性的前提)
        self._clock_offset = self._fetch_clock_offset()

        self._running = True
        self._stop_event.clear()
        self._start_time = time.time()

        # WebSocket 主线程
        self._thread = threading.Thread(
            target=self._run_loop, name="v514-ws-main", daemon=True
        )
        self._thread.start()

        # 应用层 ping 线程 (RTT 测量)
        self._app_ping_thread = threading.Thread(
            target=self._app_ping_loop, name="v514-ws-ping", daemon=True
        )
        self._app_ping_thread.start()

        # 测试模式定时器
        if self._test_mode:
            t = threading.Thread(
                target=self._test_timer, name="v514-ws-test-timer", daemon=True
            )
            t.start()

        logger.info(
            "v514 WebSocket 管道启动 (test_mode=%s, symbols=%s)",
            self._test_mode, self.symbols,
        )
        return True

    def stop(self) -> None:
        """停止管道"""
        logger.info("停止 v514 WebSocket 管道...")
        self._running = False
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        if self._app_ping_thread:
            self._app_ping_thread.join(timeout=2)
        logger.info(
            "管道已停止 (messages=%d, tickers=%d, reconnects=%d, errors=%d)",
            self._message_count, self._ticker_count,
            self._reconnect_count, self._error_count,
        )

    def _test_timer(self) -> None:
        """测试模式: 到时自动停止"""
        self._stop_event.wait(timeout=self._test_duration_sec)
        if self._running:
            logger.info("测试时长 %ds 到达, 自动停止", self._test_duration_sec)
            self._running = False
            # 给主线程一点时间收尾
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass

    # ----------------------------------------------------------------------
    # WebSocket 运行循环 (带自动重连)
    # ----------------------------------------------------------------------

    def _run_loop(self) -> None:
        """WebSocket 运行循环 (带自动重连, 最多 3 次)"""
        import websocket

        attempts = 0
        while self._running and attempts < self.MAX_RECONNECT_ATTEMPTS:
            try:
                self._ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # 库级 ping_interval=10s 自动维持心跳
                self._ws.run_forever(
                    ping_interval=self.PING_INTERVAL_SEC,
                    ping_timeout=5,
                )
                # 正常退出 (stop() 触发) 或异常断开
                if not self._running:
                    break
                # 仍在运行说明是异常断开, 尝试重连
                attempts += 1
                self._reconnect_count += 1
                logger.warning(
                    "WebSocket 断开, %ds 后重连 (尝试 %d/%d)",
                    self.RECONNECT_INTERVAL_SEC, attempts, self.MAX_RECONNECT_ATTEMPTS,
                )
                if self._stop_event.wait(timeout=self.RECONNECT_INTERVAL_SEC):
                    break  # stop() 被调用
            except Exception as e:
                self._error_count += 1
                self._errors.append(f"run_loop_exception: {e}")
                logger.error("WebSocket 运行异常: %s", e)
                attempts += 1
                self._reconnect_count += 1
                if self._stop_event.wait(timeout=self.RECONNECT_INTERVAL_SEC):
                    break

        if attempts >= self.MAX_RECONNECT_ATTEMPTS and self._running:
            msg = (
                f"重连已达上限 {self.MAX_RECONNECT_ATTEMPTS} 次, 放弃. "
                f"messages={self._message_count}"
            )
            self._errors.append(msg)
            logger.error(msg)
            self._running = False

    # ----------------------------------------------------------------------
    # WebSocket 回调
    # ----------------------------------------------------------------------

    def _on_open(self, ws) -> None:
        """连接建立: 发送订阅"""
        logger.info("WebSocket 连接建立, 订阅 %s", self.symbols)
        # 重连成功后重新获取时钟偏移 (可能时钟漂移)
        if self._clock_offset is None or (
            time.time() - self._clock_offset_fetched_at > 300
        ):
            self._clock_offset = self._fetch_clock_offset()
        self._send_subscribe()

    def _on_message(self, ws, message: str) -> None:
        """消息接收回调 (延迟测量的核心入口)"""
        try:
            self._message_count += 1
            received_at = time.time()
            data = json.loads(message)
            self._last_heartbeat = received_at

            channel = data.get("channel", "")
            event = data.get("event", "")

            # 提取服务器时间 (消息头 time 字段, 秒级)
            server_time = data.get("time")
            if server_time is not None:
                try:
                    server_time = float(server_time)
                except (TypeError, ValueError):
                    server_time = None

            if channel == "spot.tickers" and event == "update":
                self._handle_ticker(data, server_time, received_at)
            elif channel == "spot.pong":
                self._handle_pong(data, received_at)
            elif channel == "heartbeat":
                pass
            elif event == "subscribe":
                self._handle_subscribe_confirm(data, received_at)
            else:
                # 其他消息 (如订阅结果) 忽略
                pass
        except Exception as e:
            self._error_count += 1
            self._errors.append(f"on_message_exception: {e}")
            logger.debug("消息处理异常: %s", e)

    def _on_error(self, ws, error) -> None:
        """错误回调"""
        self._error_count += 1
        self._errors.append(f"ws_error: {error}")
        logger.error("WebSocket 错误: %s", error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """连接关闭回调"""
        logger.info(
            "WebSocket 关闭: code=%s msg=%s",
            close_status_code, str(close_msg)[:100] if close_msg else "",
        )

    # ----------------------------------------------------------------------
    # 订阅与消息处理
    # ----------------------------------------------------------------------

    def _send_subscribe(self) -> None:
        """发送 spot.tickers 订阅消息 (同时记录发送时间用于握手校时)"""
        if not self._ws:
            return
        msg = {
            "time": int(time.time()),
            "channel": "spot.tickers",
            "event": "subscribe",
            "payload": self.symbols,
        }
        try:
            send_local = time.time()
            self._ws.send(json.dumps(msg))
            self._subscribe_sent_local = send_local  # 记录用于握手校时
            logger.info("已发送订阅: %s (local_ts=%.3f)", self.symbols, send_local)
        except Exception as e:
            self._error_count += 1
            self._errors.append(f"subscribe_failed: {e}")
            logger.error("订阅发送失败: %s", e)

    def _handle_subscribe_confirm(self, data: Dict[str, Any], received_at: float) -> None:
        """处理订阅确认, 用握手 RTT 精化时钟偏移 (无 TLS 开销, 最准确)

        订阅确认消息含 time (秒) 和 time_ms (毫秒) 字段, 是服务器处理订阅请求的时刻。
        我们已知订阅发送的本地时间, 故可计算无 TLS 开销的 RTT:
          RTT = received_at - subscribe_sent_local
          单向延迟 ≈ RTT/2
          服务器处理订阅时刻 (服务器钟) = time_ms / 1000
          同一瞬间本地钟读数 ≈ received_at - RTT/2
          offset = (received_at - RTT/2) - (time_ms / 1000)
        """
        logger.info("订阅确认: %s", data)
        if self._subscribe_confirmed or self._subscribe_sent_local is None:
            return
        try:
            time_ms = data.get("time_ms")
            if time_ms is None:
                # 回退到秒级 time 字段
                time_s = data.get("time")
                if time_s is None:
                    return
                server_time_sec = float(time_s)
            else:
                server_time_sec = float(time_ms) / 1000.0

            rtt_sec = received_at - self._subscribe_sent_local
            rtt_ms = rtt_sec * 1000.0
            # 用 RTT/2 估算单向延迟, 计算时钟偏移
            refined_offset = (received_at - rtt_sec / 2.0) - server_time_sec
            prev_offset = self._clock_offset
            self._clock_offset = refined_offset
            self._clock_offset_fetched_at = received_at
            self._subscribe_confirmed = True
            # 同时记录这次握手的 RTT (作为最可靠的单向延迟估计)
            self._rtt_history.append(rtt_ms)
            logger.info(
                "握手校时: offset=%.3fs -> %.3fs (握手RTT=%.0fms, 单向≈%.0fms, "
                "server_time_ms=%s)",
                prev_offset if prev_offset is not None else float("nan"),
                refined_offset, rtt_ms, rtt_ms / 2.0, time_ms,
            )
        except Exception as e:
            self._error_count += 1
            self._errors.append(f"subscribe_confirm_parse_failed: {e}")
            logger.debug("订阅确认解析异常: %s", e)

    def _handle_ticker(
        self,
        data: Dict[str, Any],
        server_time: Optional[float],
        received_at: float,
    ) -> None:
        """处理 spot.tickers update 消息

        延迟测量 (v3.31.1 教训 + 时钟偏移修正):
          - 仅当 server_time 有效 AND clock_offset 有效时才记录延迟
          - 修正后延迟 = received_at - (server_time + clock_offset)
            解释: server_time 是服务器钟读数, 加上 clock_offset 转换到本地钟读数,
                   received_at - 该读数 = 真实网络+处理延迟
          - 若 clock_offset 无效, 只更新价格不记录延迟 (避免假数据污染统计)
        """
        try:
            payload = data.get("result")
            if payload is None:
                payload = data.get("data")
            items = payload if isinstance(payload, list) else [payload] if payload else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                symbol = item.get("currency_pair", "")
                if not symbol:
                    continue

                price_str = item.get("last")
                if price_str is None or price_str == "":
                    continue
                try:
                    price = float(price_str)
                except (TypeError, ValueError):
                    continue

                # 计算修正后的服务器事件时间 (转换到本地钟读数)
                corrected_server_time = None
                latency_ms: Optional[float] = None
                if server_time is not None and self._clock_offset is not None:
                    corrected_server_time = server_time + self._clock_offset
                    # 不钳位到 0: 负值说明 offset 估计偏高, 记录原始值用于诊断
                    # (v514 教训: 钳位到 0 会制造假 0ms, 使中位数失真)
                    latency_ms = (received_at - corrected_server_time) * 1000.0

                snapshot = TickSnapshot(
                    symbol=symbol,
                    price=price,
                    server_time=corrected_server_time if corrected_server_time else server_time or received_at,
                    received_at=received_at,
                    volume_24h=float(item.get("base_volume", 0) or 0),
                    bid=float(item.get("highest_bid", 0) or 0),
                    ask=float(item.get("lowest_ask", 0) or 0),
                )

                with self._lock:
                    self._latest[symbol] = snapshot
                    self._ticker_count += 1
                    # 仅在延迟可信任时记录 (v3.31.1: 避免假 0ms)
                    if latency_ms is not None:
                        self._latency_history.append(latency_ms)
                        if latency_ms < 0:
                            self._negative_latency_count += 1
        except Exception as e:
            self._error_count += 1
            self._errors.append(f"ticker_handle_exception: {e}")
            logger.debug("ticker 处理异常: %s", e)

    # ----------------------------------------------------------------------
    # 应用层 ping/pong (RTT 交叉验证)
    # ----------------------------------------------------------------------

    def _app_ping_loop(self) -> None:
        """应用层 ping 循环 (每 10s 发送 spot.ping, 用于 RTT 测量)

        首个 ping 在连接后 1s 发送 (尽快精化时钟偏移), 后续每 10s 一次。
        """
        # 首个 ping: 等待 1s 让 WebSocket 完成订阅握手
        if self._stop_event.wait(timeout=1.0):
            return
        first = True
        while self._running and not self._stop_event.is_set():
            if not self._running or self._ws is None:
                if self._stop_event.wait(timeout=1.0):
                    break
                continue
            try:
                ping_ts = time.time()
                msg = {
                    "time": int(ping_ts),
                    "channel": "spot.ping",
                    "event": "ping",
                }
                self._ws.send(json.dumps(msg))
                # 记录待匹配的 ping (key=发送时本地时间)
                self._pending_ping_ts[ping_ts] = ping_ts
                # 清理超过 30s 未匹配的 ping
                cutoff = ping_ts - 30
                stale = [t for t in self._pending_ping_ts if t < cutoff]
                for t in stale:
                    self._pending_ping_ts.pop(t, None)
            except Exception as e:
                logger.debug("应用层 ping 发送失败: %s", e)
            # 首个 ping 后, 后续按正常间隔
            if first:
                first = False
            if self._stop_event.wait(timeout=self.APP_PING_INTERVAL_SEC):
                break

    def _handle_pong(self, data: Dict[str, Any], received_at: float) -> None:
        """处理 spot.pong 响应, 仅记录 RTT (不精化时钟偏移)

        重要教训 (v514 实测): 用单个 pong 的 RTT 精化 offset 会因网络抖动导致
        offset 大幅波动 (实测 RTT 从 193ms 跳到 1478ms, offset 从 0.143s 跳到 1.523s),
        进而使大量延迟样本被错误钳位到 0ms, 中位数失真为 0.00ms。
        正确做法: offset 只用订阅握手的单次稳定测量, pong 仅用于 RTT 监控/网络健康。
        """
        try:
            orig_time = data.get("time")
            if orig_time is None:
                return
            orig_ts = float(orig_time)
            best_key = None
            best_diff = float("inf")
            for k in list(self._pending_ping_ts.keys()):
                d = abs(k - orig_ts)
                if d < best_diff:
                    best_diff = d
                    best_key = k
            if best_key is not None and best_diff < 5.0:
                ping_sent_local = self._pending_ping_ts[best_key]
                rtt_sec = received_at - ping_sent_local
                rtt_ms = rtt_sec * 1000.0
                self._rtt_history.append(rtt_ms)
                self._pending_ping_ts.pop(best_key, None)
                # 仅当日志级别为 DEBUG 时打印, 避免刷屏
                if logger.isEnabledFor(10):  # DEBUG=10
                    logger.debug(
                        "pong RTT=%.0fms (单向≈%.0fms)", rtt_ms, rtt_ms / 2.0,
                    )
        except Exception as e:
            logger.debug("pong 处理异常: %s", e)

    # ----------------------------------------------------------------------
    # 公开查询接口
    # ----------------------------------------------------------------------

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取指定交易对的最新价格 (同步, 线程安全)

        Args:
            symbol: 交易对, 如 "BTC_USDT" (也接受 "BTC-USDT")

        Returns:
            最新价格 (float), 无数据返回 None
        """
        symbol = symbol.replace("-", "_")
        with self._lock:
            snap = self._latest.get(symbol)
            return snap.price if snap else None

    def get_latest(self, symbol: str) -> Optional[TickSnapshot]:
        """获取完整快照 (含延迟信息)"""
        symbol = symbol.replace("-", "_")
        with self._lock:
            return self._latest.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """获取所有已缓存交易对的最新价格"""
        with self._lock:
            return {sym: snap.price for sym, snap in self._latest.items()}

    def get_latency_stats(self) -> Dict[str, Any]:
        """获取延迟统计 (主指标: 基于 server time + clock_offset)

        负延迟处理: 负值是 offset 估计偏高的产物 (非真实延迟), 从主统计中排除,
        但单独报告 negative_count 用于数据质量诊断。
        """
        with self._lock:
            latencies_raw = list(self._latency_history)
            rtts = list(self._rtt_history)
            neg_count = self._negative_latency_count

        # 分离正/负延迟 (负值是 offset 误差产物, 不计入主统计)
        latencies = [x for x in latencies_raw if x >= 0]

        if not latencies:
            latency_stats = {
                "available": False, "n": 0,
                "min_ms": 0.0, "median_ms": 0.0, "mean_ms": 0.0,
                "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0,
            }
        else:
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            latency_stats = {
                "available": True,
                "n": n,
                "min_ms": round(lat_sorted[0], 3),
                "median_ms": round(statistics.median(latencies), 3),
                "mean_ms": round(statistics.mean(latencies), 3),
                "p95_ms": round(lat_sorted[min(n - 1, int(n * 0.95))], 3),
                "p99_ms": round(lat_sorted[min(n - 1, int(n * 0.99))], 3),
                "max_ms": round(lat_sorted[-1], 3),
                "std_ms": round(statistics.stdev(latencies), 3) if n > 1 else 0.0,
            }

        # 数据质量指标
        latency_stats["raw_total"] = len(latencies_raw)
        latency_stats["negative_latency_count"] = neg_count
        latency_stats["offset_source"] = (
            "subscribe_handshake" if self._subscribe_confirmed
            else ("rest_fallback" if self._clock_offset is not None else "none")
        )

        if rtts:
            rtt_sorted = sorted(rtts)
            latency_stats["rtt_n"] = len(rtts)
            latency_stats["rtt_mean_ms"] = round(statistics.mean(rtts), 3)
            latency_stats["rtt_median_ms"] = round(statistics.median(rtts), 3)
            latency_stats["rtt_min_ms"] = round(rtt_sorted[0], 3)
            latency_stats["rtt_max_ms"] = round(rtt_sorted[-1], 3)

        latency_stats["clock_offset_sec"] = (
            round(self._clock_offset, 3) if self._clock_offset is not None else None
        )
        latency_stats["clock_offset_valid"] = self._clock_offset is not None
        return latency_stats

    @property
    def is_alive(self) -> bool:
        """管道是否存活 (有心跳且在运行)"""
        return self._running and (time.time() - self._last_heartbeat) < 30

    # ----------------------------------------------------------------------
    # 测试结果导出
    # ----------------------------------------------------------------------

    def export_test_result(self, test_duration_sec: int) -> Dict[str, Any]:
        """导出测试结果 JSON (符合任务要求的结构)"""
        stats = self.get_latency_stats()
        return {
            "agent_id": "agent_1_pipeline",
            "timestamp": int(time.time()),
            "test_duration_sec": test_duration_sec,
            "symbols_subscribed": list(self.symbols),
            "messages_received": self._message_count,
            "ticker_updates": self._ticker_count,
            "latency_stats": {
                "min_ms": stats.get("min_ms", 0.0),
                "median_ms": stats.get("median_ms", 0.0),
                "mean_ms": stats.get("mean_ms", 0.0),
                "p95_ms": stats.get("p95_ms", 0.0),
                "p99_ms": stats.get("p99_ms", 0.0),
                "max_ms": stats.get("max_ms", 0.0),
                "n": stats.get("n", 0),
                "available": stats.get("available", False),
                "clock_offset_sec": stats.get("clock_offset_sec"),
                "offset_source": stats.get("offset_source", "none"),
                "negative_latency_count": stats.get("negative_latency_count", 0),
                "rtt_mean_ms": stats.get("rtt_mean_ms"),
                "rtt_median_ms": stats.get("rtt_median_ms"),
                "rtt_min_ms": stats.get("rtt_min_ms"),
            },
            "reconnect_count": self._reconnect_count,
            "errors": self._errors[-20:],  # 最多保留 20 条
            "prices_at_end": self.get_all_prices(),
        }


# ============================================================================
# 3. CLI 入口
# ============================================================================


def run_test_mode(duration_sec: int = 60, output_path: Optional[str] = None) -> Dict[str, Any]:
    """运行测试模式: 指定秒数后停止, 输出延迟统计

    Args:
        duration_sec: 测试时长 (秒)
        output_path: 结果 JSON 输出路径 (None=不写文件)

    Returns:
        测试结果字典
    """
    if output_path is None:
        output_path = (
            "/mnt/c/Users/Administrator.SK-20260514VARK/Documents/trae_projects/2026/"
            "hermes_v6/sandbox_trading/_v514_multi_agent/agent_1_pipeline/"
            "_v514_ws_test_result.json"
        )

    pipe = GateIoWsV514Pipeline(test_mode=True, test_duration_sec=duration_sec)
    if not pipe.start():
        logger.error("管道启动失败")
        return {"error": "start_failed", "errors": pipe._errors}

    # 等待测试完成
    start = time.time()
    while pipe._running and (time.time() - start) < duration_sec + 5:
        time.sleep(1)
        elapsed = int(time.time() - start)
        if elapsed % 10 == 0 and elapsed > 0:
            stats = pipe.get_latency_stats()
            prices = pipe.get_all_prices()
            logger.info(
                "[%ds] msgs=%d tickers=%d latency_n=%s median=%.1fms prices=%s",
                elapsed, pipe._message_count, pipe._ticker_count,
                stats.get("n", 0), stats.get("median_ms", 0.0), prices,
            )

    pipe.stop()

    result = pipe.export_test_result(duration_sec)

    # 写入 JSON 结果文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("测试结果已写入: %s", output_path)
    except Exception as e:
        logger.error("结果文件写入失败: %s", e)
        result["file_write_error"] = str(e)

    # 打印摘要
    print("\n" + "=" * 60)
    print("v514 WebSocket 管道测试结果")
    print("=" * 60)
    print(f"测试时长:       {duration_sec}s")
    print(f"订阅符号:       {result['symbols_subscribed']}")
    print(f"消息总数:       {result['messages_received']}")
    print(f"Ticker 更新:    {result['ticker_updates']}")
    print(f"重连次数:       {result['reconnect_count']}")
    ls = result["latency_stats"]
    print(f"延迟样本数:     {ls.get('n', 0)}")
    print(f"时钟偏移:       {ls.get('clock_offset_sec')}s (valid={ls.get('available')})")
    if ls.get("available"):
        print(f"  min:    {ls['min_ms']:.2f} ms")
        print(f"  median: {ls['median_ms']:.2f} ms")
        print(f"  mean:   {ls['mean_ms']:.2f} ms")
        print(f"  p95:    {ls['p95_ms']:.2f} ms")
        print(f"  p99:    {ls['p99_ms']:.2f} ms")
        print(f"  max:    {ls['max_ms']:.2f} ms")
    if ls.get("rtt_mean_ms") is not None:
        print(f"  RTT均值: {ls['rtt_mean_ms']:.2f} ms")
    print(f"错误数:         {len(result['errors'])}")
    if result["errors"]:
        for e in result["errors"][:5]:
            print(f"  - {e}")
    print(f"结束价格:       {result['prices_at_end']}")
    print("=" * 60)

    return result


def run_long_mode(hours: float) -> None:
    """长时间运行模式"""
    duration_sec = int(hours * 3600)
    logger.info("长时间运行模式: %.2f 小时 (%ds)", hours, duration_sec)
    pipe = GateIoWsV514Pipeline(test_mode=False)
    if not pipe.start():
        logger.error("管道启动失败")
        return
    try:
        while pipe._running:
            time.sleep(60)
            stats = pipe.get_latency_stats()
            logger.info(
                "运行中: msgs=%d tickers=%d median=%.1fms p95=%.1fms alive=%s",
                pipe._message_count, pipe._ticker_count,
                stats.get("median_ms", 0.0), stats.get("p95_ms", 0.0),
                pipe.is_alive,
            )
    except KeyboardInterrupt:
        logger.info("收到中断信号, 停止...")
    finally:
        pipe.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v514 Gate.io WebSocket v4 行情管道 (Agent-1)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="测试模式: 运行 60 秒并输出延迟统计",
    )
    parser.add_argument(
        "--hours", type=float, default=1.0,
        help="运行时长 (小时, 默认 1)",
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="测试模式时长 (秒, 默认 60)",
    )
    args = parser.parse_args()

    if args.test:
        run_test_mode(duration_sec=args.duration)
    else:
        run_long_mode(hours=args.hours)
    return 0


if __name__ == "__main__":
    sys.exit(main())
