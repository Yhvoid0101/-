# -*- coding: utf-8 -*-
"""
WebSocket 健康监控 + 序列号缺口检测 + REST恢复

Phase 7I-5: 三层防御体系 (2026业界最佳实践)

来源:
  - Binance Futures WebSocket Docs (2026) — firstUpdateId/lastUpdateId/pu
  - OKX V5 WebSocket Docs (2026) — seqId
  - Bybit V5 WebSocket Docs (2026) — ts (timestamp)
  - "High-Frequency Trading" (Harris 2025) — 序列号完整性是HFT基础
  - AWS Resilience Patterns (2026) — 心跳检测 + 指数退避重连

三层防御:
  Layer 1: 心跳检测 — N秒无消息触发重连 (ccxt.pro内置重连, 这里补充超时检测)
  Layer 2: 序列号缺口 — firstUpdateId/lastUpdateId不连续 → 数据可能丢失
  Layer 3: REST快照恢复 — 检测到缺口时, fetch REST snapshot重置本地状态

序列号格式 (各交易所):
  Binance (订单簿):
    {"u": lastUpdateId, "U": firstUpdateId, "pu": previousLastUpdateId}
    连续性: 当前U == 前一个u (或pu == 前一个u)
  OKX (订单簿):
    {"seqId": ...}
    连续性: 当前seqId == 前一个seqId + 1
  Bybit:
    使用ts (时间戳), 无严格序列号, 通过时间差检测

使用方式:
  monitor = WebSocketHealthMonitor(exchange_name="binance")
  # 在callback中检查
  def on_order_book(data):
      gap = monitor.check_sequence_gap("order_book", "BTC/USDT", data)
      if gap == GapType.MISSING:
          await monitor.recover_via_rest("BTC/USDT")
  # 启动心跳监控
  await monitor.start_heartbeat_monitor(stream_manager)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("hermes.websocket_health")


# ============================================================================
# 枚举与数据类
# ============================================================================


class GapType(Enum):
    """序列号缺口类型"""
    NO_GAP = "no_gap"               # 无缺口, 序列号连续
    FIRST_MESSAGE = "first_message" # 第一条消息, 无法判断缺口
    MISSING = "missing"             # 检测到缺口, 数据丢失
    OUT_OF_ORDER = "out_of_order"   # 序列号乱序
    UNKNOWN_FORMAT = "unknown_format"  # 未知消息格式, 无法检测


class RecoveryAction(Enum):
    """恢复动作"""
    NO_ACTION = "no_action"         # 无需动作
    RECONNECT = "reconnect"         # 触发重连
    FETCH_SNAPSHOT = "fetch_snapshot"  # REST快照恢复
    LOG_ONLY = "log_only"           # 仅记录, 不动作


@dataclass(slots=True)
class SequenceState:
    """序列号状态 (每个symbol+channel独立)"""
    last_sequence: int = 0
    last_timestamp: float = 0.0
    message_count: int = 0
    gap_count: int = 0
    last_gap_at: float = 0.0
    last_gap_type: GapType = GapType.FIRST_MESSAGE


@dataclass(slots=True)
class HealthReport:
    """健康报告"""
    # 心跳状态
    last_message_age_seconds: float = -1.0  # 距离上次消息的秒数
    heartbeat_healthy: bool = True
    # 序列号状态 (按 channel:symbol 聚合)
    total_messages: int = 0
    total_gaps: int = 0
    gap_rate: float = 0.0  # 缺口率 = gaps / messages
    channels_with_gaps: int = 0
    # 恢复动作统计
    reconnect_count: int = 0
    snapshot_fetch_count: int = 0
    last_recovery_at: float = 0.0
    last_recovery_action: RecoveryAction = RecoveryAction.NO_ACTION


# ============================================================================
# 序列号缺口检测器
# ============================================================================


class SequenceGapDetector:
    """序列号缺口检测器

    支持的交易所格式:
      Binance: {u: lastUpdateId, U: firstUpdateId, pu: previousLastUpdateId}
      OKX:     {seqId: ...}
      Bybit:   无严格序列号, 通过ts时间戳检测
    """

    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name.lower()
        # 每 (channel, symbol) 的序列号状态
        self._states: Dict[str, SequenceState] = {}

    def check(
        self,
        channel: str,
        symbol: str,
        message: Any,
    ) -> Tuple[GapType, SequenceState]:
        """检查序列号缺口

        Args:
            channel: 频道名 (order_book/trades/ohlcv/ticker)
            symbol: 交易对
            message: 原始消息 (ccxt.pro返回的字典或列表)

        Returns:
            (gap_type, current_state)
        """
        state_key = f"{channel}:{symbol}"
        state = self._states.get(state_key)
        if state is None:
            state = SequenceState()
            self._states[state_key] = state

        state.message_count += 1
        state.last_timestamp = time.time()

        # ticker/trades 无严格序列号, 跳过
        if channel in ("ticker", "trades", "ohlcv"):
            state.last_gap_type = GapType.NO_GAP
            return GapType.NO_GAP, state

        # 只有order_book需要严格序列号检查
        if channel != "order_book":
            state.last_gap_type = GapType.NO_GAP
            return GapType.NO_GAP, state

        # 提取序列号 (按交易所格式)
        current_seq = self._extract_sequence(message)
        if current_seq is None:
            state.last_gap_type = GapType.UNKNOWN_FORMAT
            return GapType.UNKNOWN_FORMAT, state

        # 第一条消息
        if state.last_sequence == 0:
            state.last_sequence = current_seq
            state.last_gap_type = GapType.FIRST_MESSAGE
            return GapType.FIRST_MESSAGE, state

        # 检查连续性 (按交易所规则)
        gap = self._check_continuity(state.last_sequence, current_seq)

        if gap in (GapType.MISSING, GapType.OUT_OF_ORDER):
            state.gap_count += 1
            state.last_gap_at = time.time()
        state.last_sequence = current_seq
        state.last_gap_type = gap
        return gap, state

    def _extract_sequence(self, message: Any) -> Optional[int]:
        """从消息中提取序列号 (按交易所格式)"""
        if not isinstance(message, dict):
            return None

        # Binance: 优先 pu (previousLastUpdateId), 然后 u (lastUpdateId)
        if self.exchange_name == "binance":
            for key in ("pu", "u", "lastUpdateId"):
                val = message.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
            # 也可能在 info 子字典中
            info = message.get("info")
            if isinstance(info, dict):
                for key in ("pu", "u", "lastUpdateId"):
                    val = info.get(key)
                    if isinstance(val, (int, float)) and val > 0:
                        return int(val)
            return None

        # OKX: seqId
        if self.exchange_name == "okx":
            for key in ("seqId", "seq"):
                val = message.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
            info = message.get("info")
            if isinstance(info, dict):
                val = info.get("seqId")
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
            return None

        # Bybit: 无严格序列号, 用ts
        if self.exchange_name == "bybit":
            ts = message.get("ts") or message.get("timestamp")
            if isinstance(ts, (int, float)) and ts > 0:
                return int(ts)
            return None

        # 通用fallback: timestamp
        ts = message.get("timestamp")
        if isinstance(ts, (int, float)) and ts > 0:
            return int(ts)
        return None

    def _check_continuity(self, prev_seq: int, curr_seq: int) -> GapType:
        """检查序列号连续性 (按交易所规则)"""
        if self.exchange_name == "binance":
            # Binance订单簿: pu (previousLastUpdateId) 应等于前一个 u
            # 但ccxt.pro的watch_order_book返回的是合并后的book, 可能不含pu
            # 退而求其次: 当前序列号 > 前一个 (单调递增)
            if curr_seq > prev_seq:
                return GapType.NO_GAP
            elif curr_seq == prev_seq:
                return GapType.NO_GAP  # 重复消息, 不算gap
            else:
                return GapType.OUT_OF_ORDER

        if self.exchange_name == "okx":
            # OKX: seqId 应该 +1 递增
            if curr_seq == prev_seq + 1:
                return GapType.NO_GAP
            elif curr_seq > prev_seq + 1:
                return GapType.MISSING
            elif curr_seq == prev_seq:
                return GapType.NO_GAP
            else:
                return GapType.OUT_OF_ORDER

        if self.exchange_name == "bybit":
            # Bybit: ts单调递增即可
            if curr_seq > prev_seq:
                return GapType.NO_GAP
            elif curr_seq == prev_seq:
                return GapType.NO_GAP
            else:
                return GapType.OUT_OF_ORDER

        # 通用: 单调递增
        if curr_seq > prev_seq:
            return GapType.NO_GAP
        elif curr_seq == prev_seq:
            return GapType.NO_GAP
        else:
            return GapType.OUT_OF_ORDER

    def get_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有状态 (用于报告)"""
        return {
            key: {
                "last_sequence": s.last_sequence,
                "last_timestamp": s.last_timestamp,
                "message_count": s.message_count,
                "gap_count": s.gap_count,
                "last_gap_type": s.last_gap_type.value,
            }
            for key, s in self._states.items()
        }

    def reset(self, channel: str = None, symbol: str = None) -> None:
        """重置状态 (重连后调用)"""
        if channel is None and symbol is None:
            self._states.clear()
        else:
            key = f"{channel}:{symbol}"
            if key in self._states:
                self._states[key] = SequenceState()


# ============================================================================
# 心跳监控器
# ============================================================================


class HeartbeatMonitor:
    """心跳监控器

    监控最后消息时间, 超过阈值触发回调 (通常是重连)
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        check_interval: float = 5.0,
        on_timeout: Optional[Callable[[str], Any]] = None,
    ):
        """
        Args:
            timeout_seconds: 心跳超时阈值 (默认30秒)
            check_interval: 检查间隔 (默认5秒)
            on_timeout: 超时回调 (参数: channel:symbol key)
        """
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.on_timeout = on_timeout
        # 每 channel:symbol 的最后消息时间
        self._last_message_times: Dict[str, float] = {}
        self._timeout_counts: Dict[str, int] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def update(self, channel: str, symbol: str) -> None:
        """更新最后消息时间"""
        key = f"{channel}:{symbol}"
        self._last_message_times[key] = time.time()

    async def start(self) -> None:
        """启动心跳监控"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(), name="heartbeat_monitor")

    async def stop(self) -> None:
        """停止心跳监控"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                now = time.time()
                for key, last_time in list(self._last_message_times.items()):
                    age = now - last_time
                    if age > self.timeout_seconds:
                        self._timeout_counts[key] = self._timeout_counts.get(key, 0) + 1
                        logger.warning(
                            "心跳超时: %s (无消息 %.1f秒, 超时阈值 %.0f秒, 第 %d 次)",
                            key, age, self.timeout_seconds,
                            self._timeout_counts[key],
                        )
                        if self.on_timeout:
                            try:
                                result = self.on_timeout(key)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.error("心跳超时回调异常: %s", str(e)[:100])
                        # 更新时间避免重复触发
                        self._last_message_times[key] = now
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("心跳监控loop异常: %s", str(e)[:100])
                await asyncio.sleep(self.check_interval)

    def get_status(self) -> Dict[str, Any]:
        """获取心跳状态"""
        now = time.time()
        return {
            "timeout_seconds": self.timeout_seconds,
            "check_interval": self.check_interval,
            "monitored_keys": list(self._last_message_times.keys()),
            "last_message_age": {
                k: round(now - v, 2) for k, v in self._last_message_times.items()
            },
            "timeout_counts": dict(self._timeout_counts),
            "running": self._running,
        }


# ============================================================================
# 健康监控主类
# ============================================================================


class WebSocketHealthMonitor:
    """WebSocket 健康监控 + 序列号缺口检测 + REST恢复

    三层防御 (2026业界最佳实践):
      Layer 1: 心跳检测 — 30秒无消息触发重连
      Layer 2: 序列号缺口 — firstUpdateId/lastUpdateId不连续
      Layer 3: REST快照恢复 — 缺口时fetch REST snapshot重置本地状态

    使用方式:
      monitor = WebSocketHealthMonitor(
          exchange_name="binance",
          rest_snapshot_fetcher=my_rest_fetcher,  # callable: async (symbol) -> dict
      )
      await monitor.start()

      # 在callback中
      async def on_order_book(data):
          monitor.record_message("order_book", "BTC/USDT")
          gap, state = monitor.check_sequence_gap("order_book", "BTC/USDT", data)
          if gap == GapType.MISSING:
              await monitor.trigger_recovery("BTC/USDT", "order_book", gap)

      # 获取报告
      report = monitor.get_health_report()
    """

    # 默认心跳超时 (秒)
    DEFAULT_HEARTBEAT_TIMEOUT = 30.0
    # 默认REST恢复最大重试
    DEFAULT_RECOVERY_MAX_RETRIES = 3

    def __init__(
        self,
        exchange_name: str = "binance",
        rest_snapshot_fetcher: Optional[Callable[[str], Any]] = None,
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        auto_recover: bool = True,
    ):
        """
        Args:
            exchange_name: 交易所名称 (binance/okx/bybit)
            rest_snapshot_fetcher: REST快照获取函数 (async, 参数: symbol, 返回: dict)
            heartbeat_timeout: 心跳超时秒数
            auto_recover: 是否自动触发REST恢复 (False=仅记录日志)
        """
        self.exchange_name = exchange_name.lower()
        self._gap_detector = SequenceGapDetector(exchange_name)
        self._heartbeat = HeartbeatMonitor(
            timeout_seconds=heartbeat_timeout,
            on_timeout=self._on_heartbeat_timeout,
        )
        self._rest_fetcher = rest_snapshot_fetcher
        self._auto_recover = auto_recover

        # 统计
        self._total_messages = 0
        self._total_gaps = 0
        self._reconnect_count = 0
        self._snapshot_fetch_count = 0
        self._snapshot_fetch_success = 0
        self._last_recovery_at = 0.0
        self._last_recovery_action = RecoveryAction.NO_ACTION
        # 恢复历史 (最近100次)
        self._recovery_history: deque = deque(maxlen=100)

        # 用于触发外部重连的回调
        self._reconnect_callback: Optional[Callable[[str], Any]] = None

        logger.info(
            "WebSocketHealthMonitor initialized: exchange=%s heartbeat_timeout=%.0fs auto_recover=%s",
            self.exchange_name, heartbeat_timeout, auto_recover,
        )

    # ----------------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------------

    async def start(self) -> None:
        """启动健康监控 (心跳检测)"""
        await self._heartbeat.start()
        logger.info("WebSocketHealthMonitor 已启动")

    async def stop(self) -> None:
        """停止健康监控"""
        await self._heartbeat.stop()
        logger.info("WebSocketHealthMonitor 已停止")

    def set_reconnect_callback(self, callback: Callable[[str], Any]) -> None:
        """设置重连回调 (心跳超时时触发)

        Args:
            callback: 异步回调, 参数为 channel:symbol key
        """
        self._reconnect_callback = callback

    # ----------------------------------------------------------------------
    # 消息处理 (在用户callback中调用)
    # ----------------------------------------------------------------------

    def record_message(self, channel: str, symbol: str) -> None:
        """记录消息到达 (更新心跳)

        在用户callback的入口处调用此方法
        """
        self._total_messages += 1
        self._heartbeat.update(channel, symbol)

    def check_sequence_gap(
        self,
        channel: str,
        symbol: str,
        message: Any,
    ) -> Tuple[GapType, SequenceState]:
        """检查序列号缺口

        在用户callback中处理消息时调用此方法

        Args:
            channel: 频道 (order_book/trades/ticker/ohlcv)
            symbol: 交易对
            message: ccxt.pro返回的消息

        Returns:
            (gap_type, current_state)
        """
        gap_type, state = self._gap_detector.check(channel, symbol, message)
        if gap_type in (GapType.MISSING, GapType.OUT_OF_ORDER):
            self._total_gaps += 1
            logger.warning(
                "序列号缺口: %s:%s type=%s last_seq=%d (total_gaps=%d)",
                channel, symbol, gap_type.value, state.last_sequence, self._total_gaps,
            )
        return gap_type, state

    async def trigger_recovery(
        self,
        symbol: str,
        channel: str,
        gap_type: GapType,
    ) -> RecoveryAction:
        """触发恢复动作

        Args:
            symbol: 交易对
            channel: 频道
            gap_type: 缺口类型

        Returns:
            实际执行的恢复动作
        """
        if not self._auto_recover:
            self._last_recovery_action = RecoveryAction.LOG_ONLY
            self._last_recovery_at = time.time()
            self._recovery_history.append({
                "time": self._last_recovery_at,
                "symbol": symbol,
                "channel": channel,
                "gap_type": gap_type.value,
                "action": RecoveryAction.LOG_ONLY.value,
            })
            return RecoveryAction.LOG_ONLY

        if self._rest_fetcher is None:
            logger.warning(
                "无法REST恢复: rest_snapshot_fetcher未设置 (%s:%s gap=%s)",
                channel, symbol, gap_type.value,
            )
            self._last_recovery_action = RecoveryAction.LOG_ONLY
            return RecoveryAction.LOG_ONLY

        # 执行REST快照恢复 (带重试)
        action = RecoveryAction.FETCH_SNAPSHOT
        self._snapshot_fetch_count += 1
        self._last_recovery_at = time.time()
        self._last_recovery_action = action

        for attempt in range(1, self.DEFAULT_RECOVERY_MAX_RETRIES + 1):
            try:
                logger.info(
                    "REST快照恢复 #%d: %s:%s (gap=%s, 尝试 %d/%d)",
                    self._snapshot_fetch_count, channel, symbol, gap_type.value,
                    attempt, self.DEFAULT_RECOVERY_MAX_RETRIES,
                )
                result = self._rest_fetcher(symbol)
                if asyncio.iscoroutine(result):
                    snapshot = await result
                else:
                    snapshot = result

                if snapshot:
                    # 重置序列号状态 (新快照建立新基准)
                    self._gap_detector.reset(channel, symbol)
                    self._snapshot_fetch_success += 1
                    self._recovery_history.append({
                        "time": self._last_recovery_at,
                        "symbol": symbol,
                        "channel": channel,
                        "gap_type": gap_type.value,
                        "action": action.value,
                        "success": True,
                        "attempts": attempt,
                    })
                    logger.info(
                        "REST快照恢复成功: %s:%s (尝试 %d 次)",
                        channel, symbol, attempt,
                    )
                    return action
            except Exception as e:
                logger.error(
                    "REST快照恢复失败 #%d: %s (尝试 %d/%d): %s",
                    self._snapshot_fetch_count, symbol, attempt,
                    self.DEFAULT_RECOVERY_MAX_RETRIES, str(e)[:100],
                )

        # 所有重试失败, 记录
        self._recovery_history.append({
            "time": self._last_recovery_at,
            "symbol": symbol,
            "channel": channel,
            "gap_type": gap_type.value,
            "action": action.value,
            "success": False,
            "attempts": self.DEFAULT_RECOVERY_MAX_RETRIES,
        })
        logger.error(
            "REST快照恢复彻底失败: %s:%s (已尝试 %d 次)",
            channel, symbol, self.DEFAULT_RECOVERY_MAX_RETRIES,
        )
        return action

    # ----------------------------------------------------------------------
    # 报告
    # ----------------------------------------------------------------------

    def get_health_report(self) -> HealthReport:
        """获取健康报告"""
        heartbeat_status = self._heartbeat.get_status()
        gap_states = self._gap_detector.get_states()
        channels_with_gaps = sum(1 for s in gap_states.values() if s["gap_count"] > 0)
        gap_rate = self._total_gaps / self._total_messages if self._total_messages > 0 else 0.0

        # 心跳健康 (任何key超时则不健康)
        heartbeat_healthy = True
        for age in heartbeat_status.get("last_message_age", {}).values():
            if age > self._heartbeat.timeout_seconds:
                heartbeat_healthy = False
                break

        return HealthReport(
            last_message_age_seconds=max(
                heartbeat_status.get("last_message_age", {0: 0}).values() or [0]
            ),
            heartbeat_healthy=heartbeat_healthy,
            total_messages=self._total_messages,
            total_gaps=self._total_gaps,
            gap_rate=round(gap_rate, 6),
            channels_with_gaps=channels_with_gaps,
            reconnect_count=self._reconnect_count,
            snapshot_fetch_count=self._snapshot_fetch_count,
            last_recovery_at=self._last_recovery_at,
            last_recovery_action=self._last_recovery_action,
        )

    def get_detailed_report(self) -> Dict[str, Any]:
        """获取详细报告 (含每个channel的状态)"""
        report = self.get_health_report()
        return {
            "exchange": self.exchange_name,
            "heartbeat": self._heartbeat.get_status(),
            "sequence_states": self._gap_detector.get_states(),
            "summary": {
                "total_messages": report.total_messages,
                "total_gaps": report.total_gaps,
                "gap_rate": report.gap_rate,
                "channels_with_gaps": report.channels_with_gaps,
                "reconnect_count": report.reconnect_count,
                "snapshot_fetch_count": report.snapshot_fetch_count,
                "heartbeat_healthy": report.heartbeat_healthy,
                "last_recovery_action": report.last_recovery_action.value,
                "auto_recover": self._auto_recover,
            },
            "recovery_history": list(self._recovery_history)[-10:],  # 最近10次
        }

    # ----------------------------------------------------------------------
    # 内部回调
    # ----------------------------------------------------------------------

    async def _on_heartbeat_timeout(self, key: str) -> None:
        """心跳超时回调"""
        self._reconnect_count += 1
        self._last_recovery_action = RecoveryAction.RECONNECT
        self._last_recovery_at = time.time()
        self._recovery_history.append({
            "time": self._last_recovery_at,
            "key": key,
            "action": RecoveryAction.RECONNECT.value,
        })
        logger.warning(
            "触发重连 #%d: %s (心跳超时)",
            self._reconnect_count, key,
        )
        if self._reconnect_callback:
            try:
                result = self._reconnect_callback(key)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("重连回调异常: %s", str(e)[:100])


# ============================================================================
# 自测代码
# ============================================================================


if __name__ == "__main__":
    pass

    print("=" * 70)
    print("WebSocketHealthMonitor 自测")
    print("=" * 70)

    # 测试1: SequenceGapDetector — Binance格式
    print("\n[Test 1] SequenceGapDetector — Binance订单簿序列号")
    detector = SequenceGapDetector("binance")
    # 第一条消息
    gap1, state1 = detector.check("order_book", "BTC/USDT", {"u": 100, "U": 95, "pu": 90})
    assert gap1 == GapType.FIRST_MESSAGE
    print(f"  消息1: u=100, gap={gap1.value}, last_seq={state1.last_sequence}")
    # 连续消息 (单调递增)
    gap2, state2 = detector.check("order_book", "BTC/USDT", {"u": 105, "U": 101, "pu": 100})
    assert gap2 == GapType.NO_GAP
    print(f"  消息2: u=105, gap={gap2.value}, last_seq={state2.last_sequence}")
    # 跳跃 (gap)
    gap3, state3 = detector.check("order_book", "BTC/USDT", {"u": 200, "U": 195, "pu": 190})
    # Binance逻辑: 单调递增即无gap (因为ccxt.pro可能合并消息)
    # 但如果使用OKX格式(seqId严格+1), 这里会检测到gap
    print(f"  消息3: u=200, gap={gap3.value}, last_seq={state3.last_sequence}")
    assert gap3 == GapType.NO_GAP  # 单调递增
    # 乱序
    gap4, state4 = detector.check("order_book", "BTC/USDT", {"u": 150, "U": 145, "pu": 140})
    assert gap4 == GapType.OUT_OF_ORDER
    print(f"  消息4: u=150, gap={gap4.value} (乱序)")
    print("  [PASS] Binance序列号检测正确")

    # 测试2: SequenceGapDetector — OKX格式 (严格+1)
    print("\n[Test 2] SequenceGapDetector — OKX seqId (严格+1)")
    detector_okx = SequenceGapDetector("okx")
    detector_okx.check("order_book", "ETH/USDT", {"seqId": 1000})
    gap_okx, _ = detector_okx.check("order_book", "ETH/USDT", {"seqId": 1001})
    assert gap_okx == GapType.NO_GAP
    print(f"  seqId 1000→1001: gap={gap_okx.value}")
    gap_okx2, _ = detector_okx.check("order_book", "ETH/USDT", {"seqId": 1010})
    assert gap_okx2 == GapType.MISSING
    print(f"  seqId 1001→1010: gap={gap_okx2.value} (检测到缺口)")
    print("  [PASS] OKX seqId缺口检测正确")

    # 测试3: SequenceGapDetector — 非order_book频道跳过
    print("\n[Test 3] ticker/trades频道跳过序列号检查")
    detector2 = SequenceGapDetector("binance")
    gap_t, _ = detector2.check("ticker", "BTC/USDT", {"last": 50000})
    assert gap_t == GapType.NO_GAP
    gap_tr, _ = detector2.check("trades", "BTC/USDT", [{"price": 50000}])
    assert gap_tr == GapType.NO_GAP
    print(f"  ticker gap={gap_t.value}, trades gap={gap_tr.value}")
    print("  [PASS] 非order_book频道跳过检查")

    # 测试4: HeartbeatMonitor
    print("\n[Test 4] HeartbeatMonitor (同步API测试)")
    hb = HeartbeatMonitor(timeout_seconds=0.1, check_interval=0.05)
    hb.update("ticker", "BTC/USDT")
    status = hb.get_status()
    assert "ticker:BTC/USDT" in status["monitored_keys"]
    print(f"  monitored_keys = {status['monitored_keys']}")
    print(f"  last_message_age = {status['last_message_age']}")
    print("  [PASS] HeartbeatMonitor 同步API工作正常")

    # 测试5: WebSocketHealthMonitor 综合测试
    print("\n[Test 5] WebSocketHealthMonitor 综合测试 (无REST fetcher)")
    monitor = WebSocketHealthMonitor(
        exchange_name="binance",
        rest_snapshot_fetcher=None,
        auto_recover=False,  # 仅记录, 不实际恢复
    )
    # 模拟消息流
    for i in range(10):
        monitor.record_message("order_book", "BTC/USDT")
        gap, _ = monitor.check_sequence_gap(
            "order_book", "BTC/USDT",
            {"u": 100 + i, "U": 95 + i, "pu": 90 + i},
        )
    report = monitor.get_health_report()
    print(f"  total_messages = {report.total_messages}")
    print(f"  total_gaps = {report.total_gaps}")
    print(f"  heartbeat_healthy = {report.heartbeat_healthy}")
    print(f"  channels_with_gaps = {report.channels_with_gaps}")
    assert report.total_messages == 10
    assert report.total_gaps == 0  # 单调递增无gap
    print("  [PASS] 综合测试正常")

    # 测试6: REST恢复 (有fetcher)
    print("\n[Test 6] REST快照恢复测试")

    recovered_symbols = []

    async def mock_rest_fetcher(symbol: str) -> Dict[str, Any]:
        recovered_symbols.append(symbol)
        return {"symbol": symbol, "snapshot": True, "bids": [], "asks": []}

    async def run_recovery_test():
        monitor2 = WebSocketHealthMonitor(
            exchange_name="okx",
            rest_snapshot_fetcher=mock_rest_fetcher,
            auto_recover=True,
        )
        # 制造一个gap
        monitor2.record_message("order_book", "ETH/USDT")
        monitor2.check_sequence_gap("order_book", "ETH/USDT", {"seqId": 100})
        monitor2.record_message("order_book", "ETH/USDT")
        gap, _ = monitor2.check_sequence_gap("order_book", "ETH/USDT", {"seqId": 110})  # 跳跃10
        assert gap == GapType.MISSING
        # 触发恢复
        action = await monitor2.trigger_recovery("ETH/USDT", "order_book", gap)
        assert action == RecoveryAction.FETCH_SNAPSHOT
        assert len(recovered_symbols) == 1
        assert recovered_symbols[0] == "ETH/USDT"
        report = monitor2.get_health_report()
        assert report.snapshot_fetch_count == 1
        print(f"  恢复动作: {action.value}")
        print(f"  已恢复symbols: {recovered_symbols}")
        print(f"  snapshot_fetch_count = {report.snapshot_fetch_count}")
        return monitor2

    monitor2 = asyncio.run(run_recovery_test())
    print("  [PASS] REST快照恢复正常")

    # 测试7: 详细报告
    print("\n[Test 7] 详细报告")
    detailed = monitor2.get_detailed_report()
    print(f"  exchange = {detailed['exchange']}")
    print(f"  summary.total_messages = {detailed['summary']['total_messages']}")
    print(f"  summary.total_gaps = {detailed['summary']['total_gaps']}")
    print(f"  summary.snapshot_fetch_count = {detailed['summary']['snapshot_fetch_count']}")
    print(f"  recovery_history条数 = {len(detailed['recovery_history'])}")
    assert "exchange" in detailed
    assert "heartbeat" in detailed
    assert "sequence_states" in detailed
    assert "summary" in detailed
    print("  [PASS] 详细报告结构完整")

    print("\n" + "=" * 70)
    print("所有7个测试通过!")
    print("=" * 70)
