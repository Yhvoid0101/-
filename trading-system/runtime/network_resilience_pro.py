# -*- coding: utf-8 -*-
"""
网络弹性 Pro (Network Resilience Pro)
====================================================================
来源: R17 — 用户铁律 "网络限制根治方案"
功能: 多节点API访问策略 + 自动重连机制 + 网络延迟监测 + 智能路由选择

4大组件:
    1. NetworkResiliencePro: 主控协调器
    2. LatencyMonitor: 延迟监测 (p50/p95/p99)
    3. SmartRouter: 智能路由 (选择最优节点)
    4. AutoReconnectManager: 自动重连 (指数退避)

用户原话: "多节点API访问策略、自动重连机制、网络延迟监测与智能路由选择"
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.network_resilience_pro")


class LatencyMonitor:
    """延迟监测器 — 记录每个源的 p50/p95/p99 延迟"""

    def __init__(self, window_size: int = 100):
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def record(self, source: str, latency_ms: float) -> None:
        """记录一次延迟"""
        try:
            latency_ms = float(latency_ms)
            if latency_ms >= 0:
                self._latencies[source].append(latency_ms)
        except (TypeError, ValueError):
            pass

    def get_stats(self, source: str) -> Dict[str, float]:
        """获取延迟统计"""
        lats = list(self._latencies.get(source, []))
        if not lats:
            return {"p50_ms": 99999, "p95_ms": 99999, "p99_ms": 99999, "count": 0}

        lats_sorted = sorted(lats)
        n = len(lats_sorted)
        return {
            "p50_ms": lats_sorted[n // 2],
            "p95_ms": lats_sorted[int(n * 0.95)],
            "p99_ms": lats_sorted[int(n * 0.99)],
            "count": n,
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有源的延迟统计"""
        return {src: self.get_stats(src) for src in self._latencies}


class SmartRouter:
    """智能路由器 — 选择最优节点"""

    def __init__(self, latency_monitor: LatencyMonitor):
        self._latency_monitor = latency_monitor
        self._sources: List[str] = []
        self._current_source: Optional[str] = None
        self._switch_count: int = 0

    def register_source(self, source: str) -> None:
        """注册数据源"""
        if source not in self._sources:
            self._sources.append(source)
            if self._current_source is None:
                self._current_source = source

    def select_best(self) -> Optional[str]:
        """选择最优源 (p95 最低)"""
        if not self._sources:
            return None

        best = None
        best_p95 = 99999
        for src in self._sources:
            stats = self._latency_monitor.get_stats(src)
            if stats["p95_ms"] < best_p95:
                best_p95 = stats["p95_ms"]
                best = src

        if best and best != self._current_source:
            self._switch_count += 1
            self._current_source = best
            logger.info("SmartRouter: 切换到 %s (p95=%dms)", best, best_p95)

        return self._current_source

    @property
    def current_source(self) -> Optional[str]:
        return self._current_source

    @property
    def switch_count(self) -> int:
        return self._switch_count


class AutoReconnectManager:
    """自动重连管理器 — 指数退避"""

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._retry_count: int = 0
        self._is_connected: bool = True
        self._last_connect_time: float = time.time()
        self._should_degrade_to_rest: bool = False

    def on_disconnect(self) -> None:
        """断开连接"""
        self._is_connected = False
        logger.warning("AutoReconnectManager: 连接断开, 开始重连")

    def on_connect(self) -> None:
        """重连成功"""
        self._is_connected = True
        self._retry_count = 0
        self._should_degrade_to_rest = False
        self._last_connect_time = time.time()
        logger.info("AutoReconnectManager: 重连成功")

    def should_retry(self) -> bool:
        """是否应该重试"""
        if self._retry_count >= self._max_retries:
            self._should_degrade_to_rest = True
            return False
        return not self._is_connected

    def get_retry_delay(self) -> float:
        """获取重试延迟 (指数退避)"""
        delay = self._base_delay * (2 ** self._retry_count)
        self._retry_count += 1
        return min(delay, self._max_delay)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def should_degrade_to_rest(self) -> bool:
        return self._should_degrade_to_rest

    @property
    def retry_count(self) -> int:
        return self._retry_count


class NetworkResiliencePro:
    """网络弹性 Pro — 主控协调器

    集成 LatencyMonitor + SmartRouter + AutoReconnectManager
    API: get_full_report() + check_health()
    """

    def __init__(self):
        self._latency_monitor = LatencyMonitor()
        self._smart_router = SmartRouter(self._latency_monitor)
        self._reconnect_manager = AutoReconnectManager()

        # 注册默认数据源
        for src in ["binance", "okx", "gateio", "bybit"]:
            self._smart_router.register_source(src)

        logger.info("NetworkResiliencePro initialized: %d sources", len(self._smart_router._sources))

    def record_latency(self, source: str, latency_ms: float) -> None:
        """记录延迟"""
        self._latency_monitor.record(source, latency_ms)

    def get_full_report(self) -> Dict[str, Any]:
        """获取完整报告

        返回:
            {
                sources: [数据源列表],
                latency: {source: {p50_ms, p95_ms, p99_ms, count}},
                reconnect: {is_connected, should_degrade_to_rest, retry_count},
                current_source: 当前源,
                switch_count: 切换次数,
            }
        """
        # 选择最优源
        best = self._smart_router.select_best()

        latency_stats = self._latency_monitor.get_all_stats()

        return {
            "sources": self._smart_router._sources,
            "latency": latency_stats,
            "reconnect": {
                "is_connected": self._reconnect_manager.is_connected,
                "should_degrade_to_rest": self._reconnect_manager.should_degrade_to_rest,
                "retry_count": self._reconnect_manager.retry_count,
            },
            "current_source": best,
            "switch_count": self._smart_router.switch_count,
        }

    def check_health(self) -> Tuple[bool, List[str]]:
        """健康检查

        返回:
            (is_healthy, issues): 健康状态和问题列表
        """
        issues = []
        report = self.get_full_report()

        # 检查连接状态
        if not report["reconnect"]["is_connected"]:
            issues.append("not_connected")

        # 检查是否需要降级
        if report["reconnect"]["should_degrade_to_rest"]:
            issues.append("should_degrade_to_rest")

        # 检查延迟
        latency_dict = report["latency"]
        if latency_dict:
            healthy_sources = sum(
                1 for stats in latency_dict.values()
                if isinstance(stats, dict) and stats.get("p95_ms", 99999) < 2000
            )
            if healthy_sources == 0:
                issues.append("all_sources_high_latency")

        is_healthy = len(issues) == 0
        return is_healthy, issues
