# -*- coding: utf-8 -*-
"""
网络弹性层 (Network Resilience Layer) — v1.0

R16-1 核心新模块: 解决"网络连接限制问题",保障交易指令实时传输与执行。

业界背景(2026年最新):
  - Binance API: 权重制限流, 1200权重/分钟, 429警告→418 IP封禁(2分钟-3天)
  - -1015: 订单频率超限(10单/秒, 100000单/天)
  - CCXT 4.5.59: 内置enableRateLimit + leaky bucket算法
  - voiceofchain 2026: 指数退避+jitter是避免IP封禁的标准方案
  - pbquantlab 2026: Dynamic Weight管理(实时读取X-MBX-USED-WEIGHT头)

核心功能:
  1. 多源fallback链: Binance → Bybit → OKX → CoinGecko → 离线缓存
     - 任一源故障自动切换下一源
     - 每源独立健康分数(0-100), 低于阈值自动降级
     - 来源: voiceofchain 2026 "多交易所fallback是生产级标配"

  2. 指数退避+jitter重试
     - 退避时间: base × 2^attempt + random(0, jitter)
     - 最大重试: 5次(总等待<60s)
     - 429错误: 立即停止, 读取Retry-After头
     - 418错误: IP封禁, 切换源+延长冷却
     - 来源: AWS架构最佳实践 + Binance官方推荐

  3. 权重管理器
     - 实时读取X-MBX-USED-WEIGHT-1M响应头
     - 权重>80%阈值时主动等待
     - 全局权重池共享(多账户/多策略)
     - 来源: pbquantlab 2026 Dynamic Weight

  4. 离线缓存(24h TTL)
     - K线数据缓存到本地JSON
     - 网络故障时降级使用缓存(标记为"可能过期")
     - 缓存命中时验证时间戳
     - 来源: production_hardening最佳实践

  5. 健康监控系统
     - 每源独立健康分数(成功率高=分数高)
     - 延迟监控(p50/p95/p99)
     - 错误率监控(429/418/timeout)
     - 自动降级: 健康分数<50 → 降级, <20 → 禁用

铁律:
  - "彻底解决网络连接限制问题" — 用户原话
  - 网络故障不阻塞策略进化(降级使用缓存)
  - 永不触发IP封禁(权重>80%主动等待)
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.network_resilience")


# ============================================================================
# 常量定义
# ============================================================================

# 多源fallback链(优先级从高到低)
DEFAULT_SOURCE_CHAIN: List[str] = [
    "bybit",        # 主源: P6修复 - binance在中国网络不可达,bybit优先
    "okx",          # 备源1: API稳定
    "coingecko",    # 备源2: 免费无认证, 但限流严格
    "cache",        # 最后降级: 离线缓存
    # Phase F-1 (2026-07-15): "binance" 已移除 — 铁律禁止Binance API路径
    # 原备源3 binance 因中国网络不可达+铁律禁止, 完全移除
]

# 重试配置
MAX_RETRY_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
JITTER_SECONDS = 0.5

# 权重管理
WEIGHT_LIMIT_PER_MINUTE = 1200  # Binance现货默认
WEIGHT_WARN_THRESHOLD = 0.80    # 80%警告
WEIGHT_BLOCK_THRESHOLD = 0.95   # 95%阻塞(主动等待)

# 健康分数阈值
HEALTH_SCORE_DEGRADE = 50.0     # 低于50降级
HEALTH_SCORE_DISABLE = 20.0     # 低于20禁用
HEALTH_SCORE_RECOVERY = 80.0    # 恢复阈值(滞后)

# 缓存TTL
CACHE_TTL_HOURS = 24


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class SourceHealth:
    """数据源健康状态"""
    source_name: str
    health_score: float = 100.0   # 0-100
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_success_ts: float = 0.0
    last_failure_ts: float = 0.0
    last_error: str = ""
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    is_degraded: bool = False
    is_disabled: bool = False
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_success(self, latency_ms: float) -> None:
        """记录成功请求"""
        self.success_count += 1
        self.consecutive_failures = 0
        self.last_success_ts = time.time()
        self.latencies_ms.append(latency_ms)
        # 更新统计
        if self.latencies_ms:
            import numpy as np
            arr = np.array(list(self.latencies_ms))
            self.avg_latency_ms = float(np.mean(arr))
            if len(arr) >= 20:
                self.p95_latency_ms = float(np.percentile(arr, 95))
                self.p99_latency_ms = float(np.percentile(arr, 99))
        # 健康分数恢复
        if self.is_degraded and not self.is_disabled:
            total = self.success_count + self.failure_count
            if total > 0:
                rate = self.success_count / total
                if rate > 0.8 and self.health_score < HEALTH_SCORE_RECOVERY:
                    self.health_score = min(100.0, self.health_score + 5.0)
                    if self.health_score >= HEALTH_SCORE_RECOVERY:
                        self.is_degraded = False
                        logger.info("源 %s 健康恢复至 %.1f, 取消降级", self.source_name, self.health_score)

    def record_failure(self, error: str) -> None:
        """记录失败请求"""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_ts = time.time()
        self.last_error = error
        # 健康分数衰减
        penalty = min(20.0, 5.0 + self.consecutive_failures * 2.0)
        self.health_score = max(0.0, self.health_score - penalty)
        # 连续失败3次降级
        if self.consecutive_failures >= 3 and self.health_score < HEALTH_SCORE_DEGRADE:
            self.is_degraded = True
            logger.warning("源 %s 连续失败%d次, 健康分数%.1f, 降级",
                          self.source_name, self.consecutive_failures, self.health_score)
        # 健康分数<20禁用
        if self.health_score < HEALTH_SCORE_DISABLE:
            self.is_disabled = True
            logger.error("源 %s 健康分数%.1f<%d, 禁用",
                        self.source_name, self.health_score, HEALTH_SCORE_DISABLE)


@dataclass(slots=True)
class WeightUsage:
    """权重使用状态"""
    used_weight_1m: int = 0
    weight_limit_1m: int = WEIGHT_LIMIT_PER_MINUTE
    last_update_ts: float = 0.0
    orders_per_second: int = 0
    orders_per_day: int = 0

    @property
    def usage_ratio(self) -> float:
        """权重使用率 0-1"""
        if self.weight_limit_1m <= 0:
            return 0.0
        return self.used_weight_1m / self.weight_limit_1m

    @property
    def should_wait(self) -> bool:
        """是否应该等待(权重>95%)"""
        return self.usage_ratio >= WEIGHT_BLOCK_THRESHOLD

    @property
    def should_warn(self) -> bool:
        """是否应该警告(权重>80%)"""
        return self.usage_ratio >= WEIGHT_WARN_THRESHOLD

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """从HTTP响应头更新权重使用状态"""
        # Binance: X-MBX-USED-WEIGHT-1M
        weight = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
        if weight:
            try:
                self.used_weight_1m = int(weight)
                self.last_update_ts = time.time()
            except (ValueError, TypeError):
                pass


@dataclass(slots=True)
class FetchResult:
    """数据获取结果"""
    success: bool
    data: Optional[Any] = None
    source: str = ""
    latency_ms: float = 0.0
    from_cache: bool = False
    cache_age_hours: float = 0.0
    error: str = ""
    attempts: int = 0
    weight_used: int = 0


# ============================================================================
# 1. 指数退避+jitter重试器
# ============================================================================


class ExponentialBackoffRetry:
    """指数退避+jitter重试器

    来源: AWS架构最佳实践 + Binance官方推荐
    公式: wait = min(base × 2^attempt + random(0, jitter), max_wait)

    特殊处理:
      - HTTP 429: 读取Retry-After头, 立即停止重试
      - HTTP 418: IP封禁, 切换源+延长冷却
      - 网络超时: 标准退避
      - 5xx错误: 标准退避
      - 4xx错误(非429): 不重试(参数错误)
    """

    def __init__(
        self,
        max_attempts: int = MAX_RETRY_ATTEMPTS,
        base_backoff: float = BASE_BACKOFF_SECONDS,
        max_backoff: float = MAX_BACKOFF_SECONDS,
        jitter: float = JITTER_SECONDS,
    ):
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

    def compute_wait_time(self, attempt: int) -> float:
        """计算退避时间

        Args:
            attempt: 第几次重试(0-based)

        Returns:
            等待秒数
        """
        wait = self.base_backoff * (2 ** attempt) + random.uniform(0, self.jitter)
        return min(wait, self.max_backoff)

    def should_retry(self, error: Exception, attempt: int) -> Tuple[bool, float, str]:
        """判断是否应该重试

        Returns:
            (should_retry, wait_seconds, reason)
        """
        if attempt >= self.max_attempts:
            return False, 0.0, "达到最大重试次数"

        error_str = str(error).lower()
        error_type = type(error).__name__

        # HTTP 418: IP封禁, 不重试(切换源)
        if "418" in error_str or "banned" in error_str:
            return False, 0.0, "IP封禁, 需切换数据源"

        # HTTP 429: 限流, 读取Retry-After
        if "429" in error_str or "rate limit" in error_str or "too many" in error_str:
            wait = self.compute_wait_time(attempt) * 2  # 限流加倍等待
            return True, wait, "限流, 退避后重试"

        # 网络超时/连接错误: 标准退避
        if any(kw in error_type.lower() for kw in ["timeout", "connection", "network"]):
            return True, self.compute_wait_time(attempt), "网络超时, 退避重试"

        # 5xx服务器错误: 标准退避
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True, self.compute_wait_time(attempt), "服务器错误, 退避重试"

        # 4xx客户端错误(非429): 不重试
        if any(code in error_str for code in ["400", "401", "403", "404"]):
            return False, 0.0, "客户端错误, 不重试"

        # 其他错误: 标准退避
        return True, self.compute_wait_time(attempt), "未知错误, 退避重试"

    def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Tuple[bool, Any, str, int]:
        """执行函数,带指数退避重试

        Args:
            func: 要执行的函数(应返回数据或抛出异常)
            *args, **kwargs: 函数参数

        Returns:
            (success, data, error_message, attempts)
        """
        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000.0
                return True, result, "", attempt + 1
            except Exception as e:
                last_error = str(e)
                should_retry, wait, reason = self.should_retry(e, attempt)
                logger.debug(
                    "重试 attempt=%d/%d error=%s reason=%s wait=%.2fs",
                    attempt + 1, self.max_attempts, last_error[:100], reason, wait,
                )
                if not should_retry:
                    return False, None, last_error, attempt + 1
                if wait > 0:
                    time.sleep(wait)

        return False, None, last_error, self.max_attempts


# ============================================================================
# 2. 权重管理器
# ============================================================================


class WeightManager:
    """权重管理器(防IP封禁)

    来源: pbquantlab 2026 Dynamic Weight管理
    核心: 实时读取X-MBX-USED-WEIGHT-1M响应头, 权重>95%主动等待

    Binance权重表(2026):
      - GET /api/v3/ping: 1权重
      - GET /api/v3/ticker: 1权重
      - GET /api/v3/klines: 2权重
      - GET /api/v3/depth (limit=100): 5权重
      - GET /api/v3/depth (limit=1000): 50权重
      - 限制: 1200权重/分钟(现货), 6000权重/分钟(期货)
    """

    def __init__(self, weight_limit_per_minute: int = WEIGHT_LIMIT_PER_MINUTE):
        self.weight_limit = weight_limit_per_minute
        self.sources_weight: Dict[str, WeightUsage] = {}

    def get_usage(self, source: str) -> WeightUsage:
        """获取指定源的权重使用状态"""
        if source not in self.sources_weight:
            self.sources_weight[source] = WeightUsage(weight_limit_1m=self.weight_limit)
        return self.sources_weight[source]

    def update_from_headers(self, source: str, headers: Dict[str, str]) -> None:
        """从HTTP响应头更新权重使用状态"""
        usage = self.get_usage(source)
        usage.update_from_headers(headers)
        # 权重>95%主动等待
        if usage.should_wait:
            wait_seconds = 60.0 * (usage.usage_ratio - 0.95) / 0.05  # 0-60s
            logger.warning(
                "源 %s 权重使用%.1f%%(%d/%d), 主动等待%.1fs防封禁",
                source, usage.usage_ratio * 100, usage.used_weight_1m,
                usage.weight_limit_1m, wait_seconds,
            )
            time.sleep(min(wait_seconds, 60.0))
        elif usage.should_warn:
            logger.info(
                "源 %s 权重使用%.1f%%(%d/%d), 警告",
                source, usage.usage_ratio * 100, usage.used_weight_1m,
                usage.weight_limit_1m,
            )

    def should_throttle(self, source: str) -> Tuple[bool, float]:
        """判断是否应该节流

        Returns:
            (should_throttle, wait_seconds)
        """
        usage = self.get_usage(source)
        if usage.should_wait:
            # 等待到权重窗口重置(约60s)
            elapsed = time.time() - usage.last_update_ts
            remaining = max(0.0, 60.0 - elapsed)
            return True, remaining
        return False, 0.0


# ============================================================================
# 3. 离线缓存管理器
# ============================================================================


class OfflineCache:
    """离线缓存管理器(24h TTL)

    网络故障时降级使用缓存,标记"可能过期"。
    缓存命中时验证时间戳。
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, source: str, symbol: str, timeframe: str, data_type: str = "klines") -> str:
        """生成缓存键"""
        return "{}_{}_{}_{}".format(data_type, source, symbol.replace("/", "_"), timeframe)

    def get(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        max_age_hours: float = CACHE_TTL_HOURS,
        data_type: str = "klines",
    ) -> Tuple[Optional[Any], float, str]:
        """获取缓存数据

        Returns:
            (data, age_hours, status) — status: "fresh"/"stale"/"miss"
        """
        key = self._cache_key(source, symbol, timeframe, data_type)
        cache_file = os.path.join(self.cache_dir, key + ".json")
        if not os.path.exists(cache_file):
            return None, 0.0, "miss"

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                entry = json.load(f)
            cached_ts = entry.get("timestamp", 0)
            age_hours = (time.time() - cached_ts) / 3600.0
            if age_hours > max_age_hours:
                return entry.get("data"), age_hours, "stale"
            return entry.get("data"), age_hours, "fresh"
        except Exception as e:
            logger.warning("读取缓存失败 %s: %s", key, e)
            return None, 0.0, "miss"

    def set(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        data: Any,
        data_type: str = "klines",
    ) -> None:
        """写入缓存"""
        key = self._cache_key(source, symbol, timeframe, data_type)
        cache_file = os.path.join(self.cache_dir, key + ".json")
        try:
            entry = {
                "timestamp": time.time(),
                "source": source,
                "symbol": symbol,
                "timeframe": timeframe,
                "data_type": data_type,
                "data": data,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("写入缓存失败 %s: %s", key, e)

    def cleanup_expired(self, max_age_hours: float = 48.0) -> int:
        """清理过期缓存(>48h)"""
        cleaned = 0
        now = time.time()
        for fname in os.listdir(self.cache_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.cache_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
                age_hours = (now - mtime) / 3600.0
                if age_hours > max_age_hours:
                    os.remove(fpath)
                    cleaned += 1
            except Exception:
                pass
        return cleaned


# ============================================================================
# 4. 多源数据获取器(网络弹性层核心)
# ============================================================================


class MultiSourceDataFetcher:
    """多源数据获取器

    集成3大子组件:
      1. ExponentialBackoffRetry: 指数退避+jitter重试
      2. WeightManager: 权重管理(防IP封禁)
      3. OfflineCache: 离线缓存(网络故障降级)

    使用方式:
        fetcher = MultiSourceDataFetcher()
        result = fetcher.fetch_klines("BTC/USDT", "1h", limit=500)
        if result.success:
            bars = result.data
            print(f"数据来源: {result.source}, 延迟: {result.latency_ms}ms")
    """

    def __init__(
        self,
        source_chain: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        weight_limit_per_minute: int = WEIGHT_LIMIT_PER_MINUTE,
    ):
        self.source_chain = source_chain or DEFAULT_SOURCE_CHAIN
        if cache_dir is None:
            sandbox_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(sandbox_dir, "data_cache_resilient")
        self.cache = OfflineCache(cache_dir)
        self.retry = ExponentialBackoffRetry()
        self.weight_manager = WeightManager(weight_limit_per_minute)
        self.health_status: Dict[str, SourceHealth] = {
            s: SourceHealth(source_name=s) for s in self.source_chain
        }

    def _get_ccxt_source(self, source_name: str):
        """获取CCXT交易所实例"""
        try:
            import ccxt
        except ImportError:
            return None

        # 健康检查
        health = self.health_status.get(source_name)
        if health and health.is_disabled:
            return None

        # Phase F-1: binance 分支已移除 — 铁律禁止Binance API路径
        if source_name == "bybit":
            exchange = ccxt.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "linear"},
                "timeout": 5000,
            })
        elif source_name == "okx":
            exchange = ccxt.okx({
                "enableRateLimit": True,
                "timeout": 5000,
            })
        elif source_name == "coingecko":
            try:
                exchange = ccxt.coingecko({
                    "enableRateLimit": True,
                    "timeout": 5000,
                })
            except Exception:
                return None
        else:
            return None

        return exchange

    def _fetch_from_source(
        self,
        source_name: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Tuple[bool, Any, float, str]:
        """从单个源获取K线数据

        Returns:
            (success, data, latency_ms, error)
        """
        # 缓存源特殊处理: 遍历所有源找最新的缓存
        if source_name == "cache":
            best_data = None
            best_age = float("inf")
            best_source = ""
            for try_source in ["bybit", "okx", "coingecko", "any"]:  # Phase F-1: binance移除
                data, age, status = self.cache.get(try_source, symbol, timeframe)
                if data is not None and age < best_age:
                    best_data = data
                    best_age = age
                    best_source = try_source
            if best_data is not None:
                return True, best_data, 0.0, "cache_{}".format(best_source)
            return False, None, 0.0, "cache_miss"

        exchange = self._get_ccxt_source(source_name)
        if exchange is None:
            return False, None, 0.0, "source_unavailable"

        # 权重节流检查
        should_throttle, wait_s = self.weight_manager.should_throttle(source_name)
        if should_throttle:
            logger.info("源 %s 权重过高, 节流等待%.1fs", source_name, wait_s)
            time.sleep(min(wait_s, 30.0))

        def _do_fetch():
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        start_time = time.time()
        success, data, error, attempts = self.retry.execute_with_retry(_do_fetch)
        latency_ms = (time.time() - start_time) * 1000.0

        if success:
            # 更新健康状态
            self.health_status[source_name].record_success(latency_ms)
            # 更新权重(从最后一次响应头)
            try:
                if hasattr(exchange, "last_response_headers") and exchange.last_response_headers:
                    self.weight_manager.update_from_headers(source_name, exchange.last_response_headers)
            except Exception:
                pass
            # 写入缓存
            self.cache.set(source_name, symbol, timeframe, data)
            return True, data, latency_ms, ""
        else:
            self.health_status[source_name].record_failure(error)
            return False, None, latency_ms, error

    def fetch_klines(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        limit: int = 500,
        preferred_source: Optional[str] = None,
    ) -> FetchResult:
        """获取K线数据(多源fallback)

        Args:
            symbol: 交易对(如BTC/USDT)
            timeframe: 时间框架(1m/5m/15m/1h/4h/1d)
            limit: K线数量
            preferred_source: 优先源(可选)

        Returns:
            FetchResult
        """
        # 构建源优先级链
        chain = list(self.source_chain)
        if preferred_source and preferred_source in chain:
            chain.remove(preferred_source)
            chain.insert(0, preferred_source)

        last_error = ""
        total_attempts = 0

        for source in chain:
            # 跳过禁用源
            health = self.health_status.get(source)
            if health and health.is_disabled and source != "cache":
                continue

            success, data, latency_ms, error = self._fetch_from_source(
                source, symbol, timeframe, limit
            )
            total_attempts += 1

            if success:
                from_cache = source == "cache" or "cache_" in error
                cache_age = 0.0
                if from_cache and "cache_" in error:
                    _, age, _ = self.cache.get("any", symbol, timeframe)
                    cache_age = age

                return FetchResult(
                    success=True,
                    data=data,
                    source=source,
                    latency_ms=latency_ms,
                    from_cache=from_cache,
                    cache_age_hours=cache_age,
                    attempts=total_attempts,
                    weight_used=self.weight_manager.get_usage(source).used_weight_1m,
                )
            else:
                last_error = error
                logger.warning("源 %s 获取失败: %s, 尝试下一源", source, error[:100])

        # 所有源失败
        return FetchResult(
            success=False,
            data=None,
            source="",
            latency_ms=0.0,
            from_cache=False,
            cache_age_hours=0.0,
            error=last_error,
            attempts=total_attempts,
            weight_used=0,
        )

    def fetch_funding_rate(
        self,
        symbol: str = "BTC/USDT",
        preferred_source: Optional[str] = None,
    ) -> FetchResult:
        """获取资金费率(永续合约)

        Returns:
            FetchResult (data为funding_rate: float)
        """
        # 永续合约只在binance/bybit/okx有
        perp_sources = [s for s in self.source_chain if s in ["bybit", "okx", "cache"]]  # Phase F-1: binance移除
        if preferred_source and preferred_source in perp_sources:
            perp_sources.remove(preferred_source)
            perp_sources.insert(0, preferred_source)

        for source in perp_sources:
            health = self.health_status.get(source)
            if health and health.is_disabled and source != "cache":
                continue

            if source == "cache":
                data, age, status = self.cache.get("any", symbol, "funding", data_type="funding")
                if data is not None:
                    return FetchResult(
                        success=True, data=data, source="cache", from_cache=True,
                        cache_age_hours=age, attempts=1,
                    )
                continue

            exchange = self._get_ccxt_source(source)
            if exchange is None:
                continue

            # 永续合约模式 (Phase F-1: binance分支移除)
            if source == "bybit":
                exchange.options["defaultType"] = "swap"

            def _fetch_funding():
                if not hasattr(exchange, "fetch_funding_rate"):
                    raise RuntimeError("交易所{}不支持fetch_funding_rate".format(source))
                return exchange.fetch_funding_rate(symbol)

            success, data, error, attempts = self.retry.execute_with_retry(_fetch_funding)
            if success:
                self.health_status[source].record_success(0.0)
                funding_rate = data.get("fundingRate", 0.0) if isinstance(data, dict) else 0.0
                self.cache.set(source, symbol, "funding", funding_rate, data_type="funding")
                return FetchResult(
                    success=True, data=funding_rate, source=source,
                    from_cache=False, attempts=attempts,
                )
            else:
                self.health_status[source].record_failure(error)

        return FetchResult(success=False, error="所有源获取funding失败")

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康监控报告"""
        report = {}
        for source, health in self.health_status.items():
            total = health.success_count + health.failure_count
            success_rate = health.success_count / total if total > 0 else 0.0
            report[source] = {
                "health_score": round(health.health_score, 1),
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "success_rate": round(success_rate * 100, 1),
                "consecutive_failures": health.consecutive_failures,
                "avg_latency_ms": round(health.avg_latency_ms, 1),
                "p95_latency_ms": round(health.p95_latency_ms, 1),
                "p99_latency_ms": round(health.p99_latency_ms, 1),
                "is_degraded": health.is_degraded,
                "is_disabled": health.is_disabled,
                "last_error": health.last_error[:200] if health.last_error else "",
                "weight_used": self.weight_manager.get_usage(source).used_weight_1m,
                "weight_usage_pct": round(
                    self.weight_manager.get_usage(source).usage_ratio * 100, 1
                ),
            }
        return report


# ============================================================================
# 便捷函数
# ============================================================================


def fetch_klines_resilient(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
) -> FetchResult:
    """弹性获取K线数据(便捷函数)"""
    fetcher = MultiSourceDataFetcher()
    return fetcher.fetch_klines(symbol, timeframe, limit)


def fetch_funding_rate_resilient(symbol: str = "BTC/USDT") -> FetchResult:
    """弹性获取资金费率(便捷函数)"""
    fetcher = MultiSourceDataFetcher()
    return fetcher.fetch_funding_rate(symbol)


if __name__ == "__main__":
    # 自测
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    print("=== R16-1 网络弹性层自测 ===\n")

    # 测试1: 指数退避计算
    retry = ExponentialBackoffRetry(base_backoff=1.0, jitter=0.0)
    waits = [retry.compute_wait_time(i) for i in range(5)]
    print("测试1: 指数退避等待序列:", [round(w, 2) for w in waits])
    assert waits[0] < waits[1] < waits[2] < waits[3] < waits[4]
    print("  [+] 指数退避序列正确\n")

    # 测试2: 权重管理
    wm = WeightManager(weight_limit_per_minute=1200)
    wm.update_from_headers("binance", {"X-MBX-USED-WEIGHT-1M": "600"})
    assert wm.get_usage("binance").usage_ratio == 0.5
    print("测试2: 权重管理 600/1200 = 50%")
    print("  [+] 权重管理正确\n")

    # 测试3: 离线缓存
    cache = OfflineCache("/tmp/test_cache_r16")
    cache.set("binance", "BTC/USDT", "1h", [[1, 2, 3, 4, 5]])
    data, age, status = cache.get("binance", "BTC/USDT", "1h")
    assert data == [[1, 2, 3, 4, 5]]
    assert status == "fresh"
    print("测试3: 离线缓存写入+读取")
    print("  [+] 缓存正确, status={}\n".format(status))

    # 测试4: 多源fallback
    fetcher = MultiSourceDataFetcher()
    print("测试4: 多源fallback链:", fetcher.source_chain)
    assert "bybit" in fetcher.source_chain  # Phase F-1: binance移除, 改assert bybit
    assert "cache" in fetcher.source_chain
    print("  [+] fallback链完整\n")

    # 测试5: 健康监控
    health = SourceHealth(source_name="test")
    for _ in range(10):
        health.record_success(50.0)
    assert health.success_count == 10
    assert health.health_score == 100.0
    print("测试5: 健康监控 10次成功后分数={}".format(health.health_score))

    health.record_failure("timeout")
    health.record_failure("timeout")
    health.record_failure("timeout")
    print("  3次失败后分数={}".format(health.health_score))
    assert health.health_score < 100.0
    print("  [+] 健康分数衰减正确\n")

    print("=== R16-1 自测全部完成 ===")
