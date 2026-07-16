# -*- coding: utf-8 -*-
"""
v598 Phase H: 实盘延迟 SLA 监控 + 熔断器 (Latency SLA Monitor + Circuit Breaker)

用户硬约束:
  - 行情数据更新延迟 ≤ 100ms (P99)
  - 订单执行响应时间 ≤ 500ms (P99)

ERR-20260702-v598p5-latency-11x:
  原 gateio_websocket_provider.py:6 注释自承 "REST API延迟: mean=1141ms, p95=2133ms
  (超目标11倍)"。本模块提供独立 SLA 监控层, 在行情或订单延迟突破阈值时熔断下单,
  从源头杜绝"模拟牛逼实盘亏损"——延迟超阈值时, 信号已过期, 继续下单等于赌博.

设计原则 (用户铁律):
  - "永远永远不要出现模拟牛逼，实盘亏损的情况": 延迟突破时 BLOCK 下单
  - "理解各指标的底层逻辑和市场含义": P99 (而非均值) 才是尾部风险代理
  - "打破瓶颈，打破极限": 熔断+冷却+自动恢复, 不是简单 BOOL flag

核心组件:
  1. LatencySLAMonitor: 行情/订单延迟 P99 监控, 违反时 trip 熔断
  2. CircuitBreaker: 三态熔断器 (CLOSED / OPEN / HALF_OPEN)
  3. SLAReport: 每日 p50/p95/p99 统计 + 违反次数

熔断状态机:
  CLOSED  ──延迟>P99阈值──>  OPEN  (拒绝下单)
    ↑                            │
    └──冷却时间结束──  HALF_OPEN (允许1次试探)
                              │
                       ┌──────┴──────┐
                  试探成功          试探失败
                       │               │
                    CLOSED          OPEN (重启冷却)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("hermes.latency_sla")


# ============================================================================
# 1. 熔断器状态机
# ============================================================================


class CircuitState(Enum):
    """熔断器三态"""

    CLOSED = "closed"          # 正常, 允许下单
    OPEN = "open"               # 熔断, 拒绝下单
    HALF_OPEN = "half_open"     # 半开, 允许 1 次试探


class CircuitBreaker:
    """三态熔断器 (CLOSED ↔ OPEN ↔ HALF_OPEN)

    用户铁律 "杜绝系统性亏损风险": 延迟超阈值时, 必须真正 BLOCK 下单, 不能仅打日志.

    状态转换:
      CLOSED → OPEN: 当 trip_count >= failure_threshold (默认 3 次)
      OPEN → HALF_OPEN: 冷却时间 cooldown_s 后
      HALF_OPEN → CLOSED: 探试成功
      HALF_OPEN → OPEN: 探试失败 (重启冷却)
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_s: float = 60.0,
        name: str = "default",
    ):
        """
        Args:
            failure_threshold: 连续违反阈值次数 → trip OPEN
            cooldown_s: OPEN 状态冷却时间 (秒)
            name: 熔断器名称 (用于日志)
        """
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_s = max(1.0, float(cooldown_s))
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0           # 连续违反次数
        self._trip_count = 0              # 累计 trip 次数
        self._last_trip_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # OPEN → HALF_OPEN 自动转换 (冷却时间到)
            if (self._state == CircuitState.OPEN
                    and time.time() - self._last_trip_time >= self.cooldown_s):
                self._state = CircuitState.HALF_OPEN
                logger.warning(
                    "CircuitBreaker[%s] OPEN→HALF_OPEN (冷却 %ds 结束, 允许试探)",
                    self.name, int(self.cooldown_s),
                )
            return self._state

    @property
    def is_blocked(self) -> bool:
        """是否 BLOCK 下单"""
        return self.state == CircuitState.OPEN

    @property
    def trip_count(self) -> int:
        with self._lock:
            return self._trip_count

    def trip(self, reason: str = "") -> None:
        """违反 SLA, 记录一次失败"""
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                # 试探失败 → 重启冷却
                self._state = CircuitState.OPEN
                self._last_trip_time = time.time()
                self._trip_count += 1
                logger.error(
                    "CircuitBreaker[%s] HALF_OPEN→OPEN (试探失败: %s, 冷却 %ds)",
                    self.name, reason, int(self.cooldown_s),
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_trip_time = time.time()
                    self._trip_count += 1
                    logger.error(
                        "CircuitBreaker[%s] CLOSED→OPEN (连续违反 %d 次, 原因: %s, 冷却 %ds)",
                        self.name, self._failure_count, reason, int(self.cooldown_s),
                    )
                else:
                    logger.warning(
                        "CircuitBreaker[%s] 违反记录 %d/%d (原因: %s)",
                        self.name, self._failure_count, self.failure_threshold, reason,
                    )

    def reset(self) -> None:
        """探试成功 (HALF_OPEN → CLOSED)"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(
                    "CircuitBreaker[%s] HALF_OPEN→CLOSED (试探成功, 恢复下单)",
                    self.name,
                )
            elif self._state == CircuitState.CLOSED:
                # 正常状态下的成功 → 重置失败计数
                self._failure_count = 0

    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态摘要"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "trip_count": self._trip_count,
                "cooldown_s": self.cooldown_s,
                "last_trip_age_s": time.time() - self._last_trip_time if self._last_trip_time else None,
                "is_blocked": self._state == CircuitState.OPEN,
            }


# ============================================================================
# 2. 延迟样本缓冲
# ============================================================================


class LatencyBuffer:
    """线程安全的延迟样本环形缓冲

    用 numpy 计算 P50/P95/P99, 比 sorted() 快 ~10x (大样本场景)
    """

    def __init__(self, maxlen: int = 1000):
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, latency_ms: float) -> None:
        if latency_ms < 0 or not math.isfinite(latency_ms):
            return  # 跳过无效样本
        with self._lock:
            self._buffer.append(float(latency_ms))

    def percentile(self, q: float) -> Optional[float]:
        """计算百分位数 (q ∈ [0, 100])"""
        with self._lock:
            if not self._buffer:
                return None
            arr = np.fromiter(self._buffer, dtype=np.float64)
        return float(np.percentile(arr, q))

    def stats(self) -> Dict[str, float]:
        with self._lock:
            if not self._buffer:
                return {"n": 0}
            arr = np.fromiter(self._buffer, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()) if len(arr) > 1 else 0.0,
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


# ============================================================================
# 3. SLA 报告
# ============================================================================


@dataclass
class SLAReport:
    """SLA 监控报告 (供 loop_end 摘要输出)"""
    timestamp: float = field(default_factory=time.time)
    market_p50_ms: float = 0.0
    market_p95_ms: float = 0.0
    market_p99_ms: float = 0.0
    market_n_samples: int = 0
    market_violations: int = 0
    order_p50_ms: float = 0.0
    order_p95_ms: float = 0.0
    order_p99_ms: float = 0.0
    order_n_samples: int = 0
    order_violations: int = 0
    circuit_state: str = "closed"
    circuit_trip_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "market": {
                "p50_ms": round(self.market_p50_ms, 3),
                "p95_ms": round(self.market_p95_ms, 3),
                "p99_ms": round(self.market_p99_ms, 3),
                "n_samples": self.market_n_samples,
                "violations": self.market_violations,
                "meets_100ms_target": self.market_p99_ms <= 100.0 if self.market_n_samples > 0 else False,
            },
            "order": {
                "p50_ms": round(self.order_p50_ms, 3),
                "p95_ms": round(self.order_p95_ms, 3),
                "p99_ms": round(self.order_p99_ms, 3),
                "n_samples": self.order_n_samples,
                "violations": self.order_violations,
                "meets_500ms_target": self.order_p99_ms <= 500.0 if self.order_n_samples > 0 else False,
            },
            "circuit_breaker": {
                "state": self.circuit_state,
                "trip_count": self.circuit_trip_count,
            },
        }


# ============================================================================
# 4. 核心监控器
# ============================================================================


class LatencySLAMonitor:
    """实盘 SLA 监控 + 熔断器

    用户硬约束:
      - 行情数据更新延迟 ≤ 100ms (P99)
      - 订单执行响应时间 ≤ 500ms (P99)

    使用方式:
        monitor = LatencySLAMonitor()
        # 行情接收到时
        monitor.record_market_latency(latency_ms=tick.latency_ms)
        # 下单前后
        if not monitor.can_place_order():
            logger.warning("SLA熔断, 跳过下单")
            return
        t0 = time.time()
        order_result = connector.submit_order(order)
        monitor.record_order_latency(latency_ms=(time.time()-t0)*1000)
        # 每日报告
        report = monitor.daily_report()
    """

    # 用户硬约束阈值 (来自 project_memory Hard Constraints)
    MARKET_P99_THRESHOLD_MS = 100.0
    ORDER_P99_THRESHOLD_MS = 500.0

    def __init__(
        self,
        market_p99_threshold_ms: float = MARKET_P99_THRESHOLD_MS,
        order_p99_threshold_ms: float = ORDER_P99_THRESHOLD_MS,
        min_samples_for_eval: int = 20,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_s: float = 60.0,
    ):
        """
        Args:
            market_p99_threshold_ms: 行情 P99 阈值 (ms), 默认 100ms
            order_p99_threshold_ms: 订单 P99 阈值 (ms), 默认 500ms
            min_samples_for_eval: 评估所需最小样本数 (少于则不评估)
            circuit_failure_threshold: 熔断器连续违反次数阈值
            circuit_cooldown_s: 熔断冷却时间 (秒)
        """
        self.market_p99_threshold = float(market_p99_threshold_ms)
        self.order_p99_threshold = float(order_p99_threshold_ms)
        self.min_samples_for_eval = max(5, int(min_samples_for_eval))

        self._market_buffer = LatencyBuffer(maxlen=10000)
        self._order_buffer = LatencyBuffer(maxlen=1000)

        # 双熔断器: 行情 + 订单 (任一熔断都 BLOCK 下单)
        self.market_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            cooldown_s=circuit_cooldown_s,
            name="market_latency",
        )
        self.order_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            cooldown_s=circuit_cooldown_s,
            name="order_latency",
        )

        self._market_violations = 0
        self._order_violations = 0
        self._lock = threading.Lock()

        # v598 Phase H: HALF_OPEN 转换时清空缓冲, 让试探基于全新样本
        # 否则旧的高延迟样本会污染 p99, 导致试探永远失败 (违反 "冷却后试探" 语义)
        # 跟踪上次观察到的 breaker state, 用于检测 OPEN→HALF_OPEN 转换
        self._market_last_state: str = "closed"
        self._order_last_state: str = "closed"

    def _maybe_clear_on_half_open(self, breaker: CircuitBreaker,
                                  buffer: LatencyBuffer,
                                  last_state_attr: str) -> bool:
        """检测 OPEN→HALF_OPEN 转换, 清空缓冲让试探基于全新样本

        Returns:
            True=本次是 HALF_OPEN 后的首个样本 (已清空缓冲)
        """
        current_state = breaker.state.value  # 触发 OPEN→HALF_OPEN 自动转换
        last_state = getattr(self, last_state_attr)

        # 状态转换: OPEN → HALF_OPEN
        if last_state == "open" and current_state == "half_open":
            buffer.clear()
            logger.info(
                "CircuitBreaker[%s] OPEN→HALF_OPEN 转换检测, 清空延迟缓冲让试探基于全新样本",
                breaker.name,
            )
            setattr(self, last_state_attr, current_state)
            return True
        setattr(self, last_state_attr, current_state)
        return False

    # ------------------------------------------------------------------
    # 行情延迟
    # ------------------------------------------------------------------

    def record_market_latency(self, latency_ms: float) -> bool:
        """记录行情延迟样本

        Args:
            latency_ms: 行情延迟 (ms)

        Returns:
            True=未违反 SLA, False=违反 (P99 > 阈值)
        """
        # v598 Phase H: OPEN→HALF_OPEN 时清空缓冲让试探基于全新样本
        # 读取当前 state (会触发 OPEN→HALF_OPEN 自动转换如果冷却结束)
        self._maybe_clear_on_half_open(
            self.market_breaker, self._market_buffer, "_market_last_state"
        )
        self._market_buffer.append(latency_ms)

        # 仅在样本充足时评估 (避免早期误判)
        stats = self._market_buffer.stats()
        if stats.get("n", 0) < self.min_samples_for_eval:
            # 更新 last_state (供下次 _maybe_clear_on_half_open 比较)
            self._market_last_state = self.market_breaker.state.value
            return True

        p99 = stats.get("p99_ms", 0.0)
        if p99 > self.market_p99_threshold:
            with self._lock:
                self._market_violations += 1
            self.market_breaker.trip(
                reason=f"market_p99={p99:.1f}ms > {self.market_p99_threshold:.0f}ms"
            )
            # trip 后状态可能 CLOSED→OPEN 或 HALF_OPEN→OPEN, 更新 last_state
            self._market_last_state = self.market_breaker.state.value
            return False
        # 探试成功 (HALF_OPEN → CLOSED)
        self.market_breaker.reset()
        # 更新 last_state (HALF_OPEN→CLOSED 或维持 CLOSED)
        self._market_last_state = self.market_breaker.state.value
        return True

    # ------------------------------------------------------------------
    # 订单延迟
    # ------------------------------------------------------------------

    def record_order_latency(self, latency_ms: float) -> bool:
        """记录订单延迟样本

        Args:
            latency_ms: 订单执行延迟 (ms)

        Returns:
            True=未违反 SLA, False=违反
        """
        # v598 Phase H: OPEN→HALF_OPEN 时清空缓冲让试探基于全新样本
        self._maybe_clear_on_half_open(
            self.order_breaker, self._order_buffer, "_order_last_state"
        )
        self._order_buffer.append(latency_ms)

        stats = self._order_buffer.stats()
        if stats.get("n", 0) < self.min_samples_for_eval:
            self._order_last_state = self.order_breaker.state.value
            return True

        p99 = stats.get("p99_ms", 0.0)
        if p99 > self.order_p99_threshold:
            with self._lock:
                self._order_violations += 1
            self.order_breaker.trip(
                reason=f"order_p99={p99:.1f}ms > {self.order_p99_threshold:.0f}ms"
            )
            self._order_last_state = self.order_breaker.state.value
            return False
        self.order_breaker.reset()
        self._order_last_state = self.order_breaker.state.value
        return True

    # ------------------------------------------------------------------
    # 熔断查询
    # ------------------------------------------------------------------

    def can_place_order(self) -> bool:
        """是否允许下单 (任一熔断器 OPEN 则 BLOCK)

        用户铁律 "杜绝系统性亏损风险": 延迟超阈值时, 信号已过期, 继续下单等于赌博.
        必须真正 BLOCK 下单而非仅打日志.
        """
        # 触发 OPEN→HALF_OPEN 自动转换 (冷却时间到)
        market_blocked = self.market_breaker.is_blocked
        order_blocked = self.order_breaker.is_blocked
        if market_blocked or order_blocked:
            return False
        return True

    @property
    def is_blocked(self) -> bool:
        """是否被 BLOCK (任一熔断器 OPEN)"""
        return not self.can_place_order()

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def daily_report(self) -> SLAReport:
        """生成每日 SLA 报告"""
        market_stats = self._market_buffer.stats()
        order_stats = self._order_buffer.stats()

        # 熔断器主状态 (优先 market)
        if self.market_breaker.is_blocked:
            cb_state = self.market_breaker.state.value
        elif self.order_breaker.is_blocked:
            cb_state = self.order_breaker.state.value
        else:
            # 取两者中较严重的 (HALF_OPEN > CLOSED)
            cb_state = max(
                self.market_breaker.state.value,
                self.order_breaker.state.value,
                key=lambda s: {"closed": 0, "half_open": 1, "open": 2}.get(s, 0),
            )

        return SLAReport(
            timestamp=time.time(),
            market_p50_ms=market_stats.get("p50_ms", 0.0),
            market_p95_ms=market_stats.get("p95_ms", 0.0),
            market_p99_ms=market_stats.get("p99_ms", 0.0),
            market_n_samples=market_stats.get("n", 0),
            market_violations=self._market_violations,
            order_p50_ms=order_stats.get("p50_ms", 0.0),
            order_p95_ms=order_stats.get("p95_ms", 0.0),
            order_p99_ms=order_stats.get("p99_ms", 0.0),
            order_n_samples=order_stats.get("n", 0),
            order_violations=self._order_violations,
            circuit_state=cb_state,
            circuit_trip_count=(
                self.market_breaker.trip_count + self.order_breaker.trip_count
            ),
        )

    def get_state(self) -> Dict[str, Any]:
        """获取完整状态 (供 evolution_loop 暴露)"""
        return {
            "market_breaker": self.market_breaker.get_state(),
            "order_breaker": self.order_breaker.get_state(),
            "market_latency": self._market_buffer.stats(),
            "order_latency": self._order_buffer.stats(),
            "market_violations": self._market_violations,
            "order_violations": self._order_violations,
            "market_p99_threshold_ms": self.market_p99_threshold,
            "order_p99_threshold_ms": self.order_p99_threshold,
            "is_blocked": self.is_blocked,
            "can_place_order": self.can_place_order(),
        }

    # ------------------------------------------------------------------
    # 重置 (用于测试或人工干预)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置所有熔断器 + 清空缓冲 (谨慎使用)"""
        self.market_breaker.reset()
        self.order_breaker.reset()
        self._market_buffer.clear()
        self._order_buffer.clear()
        with self._lock:
            self._market_violations = 0
            self._order_violations = 0
            self._market_last_state = "closed"
            self._order_last_state = "closed"
        logger.warning("LatencySLAMonitor 重置 (人工干预)")


# ============================================================================
# 5. 全局单例 (供 exchange_connector 使用)
# ============================================================================


_global_monitor: Optional[LatencySLAMonitor] = None
_global_lock = threading.Lock()


def get_global_sla_monitor() -> LatencySLAMonitor:
    """获取全局 SLA 监控器 (单例)"""
    global _global_monitor
    if _global_monitor is None:
        with _global_lock:
            if _global_monitor is None:
                _global_monitor = LatencySLAMonitor()
                logger.info("LatencySLAMonitor 全局实例已创建 (market≤100ms, order≤500ms)")
    return _global_monitor


def reset_global_sla_monitor() -> None:
    """重置全局 SLA 监控器 (主要供测试使用)"""
    global _global_monitor
    with _global_lock:
        if _global_monitor is not None:
            _global_monitor.reset()
        _global_monitor = None


# ============================================================================
# 6. 自检
# ============================================================================


def _self_test() -> bool:
    """自检: 验证 SLA 监控器核心功能"""
    monitor = LatencySLAMonitor(
        market_p99_threshold_ms=100.0,
        order_p99_threshold_ms=500.0,
        min_samples_for_eval=5,
        circuit_failure_threshold=2,
        circuit_cooldown_s=1.0,
    )

    # T1: 初始状态应允许下单
    assert monitor.can_place_order(), "T1 FAIL: 初始应允许下单"
    print("  [T1] PASS: 初始 can_place_order=True")

    # T2: 少量样本 (低于 min_samples) 不应触发违反
    for ms in [50.0, 60.0, 70.0]:
        monitor.record_market_latency(ms)
    assert monitor.can_place_order(), "T2 FAIL: 少量样本不应触发熔断"
    print("  [T2] PASS: 少量样本不评估 SLA (min_samples_for_eval=5)")

    # T3: 大量低延迟样本 → 未违反
    for ms in [50.0] * 20:
        monitor.record_market_latency(ms)
    assert monitor.can_place_order(), "T3 FAIL: 低延迟不应触发熔断"
    print("  [T3] PASS: 低延迟样本不触发熔断")

    # T4: 高延迟样本 (P99 > 100ms) → 违反 + 熔断
    # 用新的 monitor 避免污染 T3
    monitor2 = LatencySLAMonitor(
        market_p99_threshold_ms=100.0,
        min_samples_for_eval=5,
        circuit_failure_threshold=2,
        circuit_cooldown_s=1.0,
    )
    # 先填充 20 笔低延迟 (达到 min_samples)
    for ms in [50.0] * 20:
        monitor2.record_market_latency(ms)
    # 第 1 笔高延迟 → buffer 已有足够样本 → P99 > 100 → 1 次违反, 仍 CLOSED
    monitor2.record_market_latency(200.0)
    assert monitor2.market_breaker.state.value == "closed", \
        f"T4a FAIL: 1 次违反不应触发熔断 (threshold=2), 实际: {monitor2.market_breaker.state.value}"
    assert monitor2.market_breaker.trip_count == 0, \
        f"T4a FAIL: 1 次违反不应 trip, trip_count={monitor2.market_breaker.trip_count}"
    print("  [T4a] PASS: 1 次违反未触发熔断 (failure_count=1 < threshold=2)")

    # 第 2 笔高延迟 → 第 2 次违反 → 熔断 OPEN (failure_count=2 ≥ threshold=2)
    monitor2.record_market_latency(250.0)
    assert monitor2.market_breaker.is_blocked, \
        f"T4b FAIL: 2 次违反应触发熔断, 状态: {monitor2.market_breaker.state.value}"
    assert not monitor2.can_place_order(), \
        "T4b FAIL: 熔断后 can_place_order 应为 False"
    assert monitor2.market_breaker.trip_count == 1, \
        f"T4b FAIL: trip_count 应=1, 实际={monitor2.market_breaker.trip_count}"
    print("  [T4b] PASS: 连续 2 次违反触发熔断 OPEN (BLOCK 下单, trip_count=1)")

    # T5: 冷却时间到 → HALF_OPEN, 探试成功 → CLOSED
    time.sleep(1.1)  # cooldown_s=1.0
    # 调用 state 触发 OPEN→HALF_OPEN 自动转换
    state = monitor2.market_breaker.state
    assert state.value == "half_open", \
        f"T5a FAIL: 冷却后应 HALF_OPEN, 实际: {state.value}"
    print("  [T5a] PASS: 冷却时间到自动 OPEN→HALF_OPEN")

    # 探试成功 (低延迟) → CLOSED
    for ms in [50.0] * 6:
        monitor2.record_market_latency(ms)
    assert monitor2.market_breaker.state.value == "closed", \
        f"T5b FAIL: 探试成功应 CLOSED, 实际: {monitor2.market_breaker.state.value}"
    print("  [T5b] PASS: 探试成功 HALF_OPEN→CLOSED (恢复下单)")

    # T6: 订单延迟熔断 (P99 > 500ms)
    monitor3 = LatencySLAMonitor(
        order_p99_threshold_ms=500.0,
        min_samples_for_eval=5,
        circuit_failure_threshold=2,
        circuit_cooldown_s=1.0,
    )
    for ms in [1000.0, 1100.0, 1200.0]:
        monitor3.record_order_latency(ms)
    for ms in [1000.0, 1100.0, 1200.0]:
        monitor3.record_order_latency(ms)
    assert monitor3.order_breaker.is_blocked, "T6 FAIL: 订单延迟应触发熔断"
    assert not monitor3.can_place_order(), "T6 FAIL: 熔断后应 BLOCK"
    print("  [T6] PASS: 订单延迟 P99>500ms 触发熔断")

    # T7: 日报告生成
    report = monitor3.daily_report()
    assert report.order_violations >= 1, "T7 FAIL: 应有订单违反记录"
    assert report.circuit_state in ("open", "half_open", "closed"), "T7 FAIL: 状态值无效"
    print(f"  [T7] PASS: 日报告 order_violations={report.order_violations}, "
          f"circuit_state={report.circuit_state}")

    # T8: 全局单例
    g1 = get_global_sla_monitor()
    g2 = get_global_sla_monitor()
    assert g1 is g2, "T8 FAIL: 全局单例应一致"
    print("  [T8] PASS: 全局单例一致")
    reset_global_sla_monitor()

    print("\n[SelfTest] LatencySLAMonitor 8/8 自检全部通过 ✅")
    return True


if __name__ == "__main__":
    _self_test()
