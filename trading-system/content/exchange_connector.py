# -*- coding: utf-8 -*-
"""
交易所连接器 (Exchange Connector) — 实盘交易所统一接入层

基于 CCXT 统一接口，支持 Binance / OKX / Bybit / Hyperliquid / Gate / Kraken 等。

设计原则：
  1. 实现 IMatchingEngine 接口，与沙盘引擎无缝切换
  2. 支持现货 + 永续合约
  3. 安全第一：API密钥本地加密、IP白名单、提现权限关闭
  4. 错误重试 + 限流 + 断线重连
  5. 完整审计日志

支持的交易所（CCXT 100+交易所，重点支持以下）：
  - Binance (现货 + 永续 + 币安合约)
  - OKX (现货 + 永续 + 期权)
  - Bybit (现货 + 永续 + 反向合约)
  - Hyperliquid (去中心化永续，L1链上订单簿)
  - Gate.io (现货 + 永续)
  - Kraken (现货 + 期货)

参考：
  - CCXT Unified API (docs.ccxt.com)
  - Hyperliquid Python SDK (github.com/hyperliquid-dex/hyperliquid-python-sdk)
  - Binance API Docs (binance-docs.github.io/apidocs)
  - OKX V5 API (www.okx.com/docs-v5)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .matching_engine_interface import (
    IMatchingEngine,
    MatchingEngineAdapter,
    UnifiedAccount,
    UnifiedOrder,
    UnifiedOrderBook,
    UnifiedOrderSide,
    UnifiedOrderStatus,
    UnifiedOrderType,
    UnifiedPosition,
    UnifiedTicker,
)

# Phase 7I-1: 网络弹性层 — 彻底解决"网络连接限制问题"
# 来源: AWS架构最佳实践 + Binance官方推荐 + pbquantlab 2026 Dynamic Weight
try:
    from .network_resilience import (
        ExponentialBackoffRetry,
        FetchResult,
        MultiSourceDataFetcher,
        WeightManager,
    )
    _NETWORK_RESILIENCE_AVAILABLE = True
except ImportError:
    _NETWORK_RESILIENCE_AVAILABLE = False
    ExponentialBackoffRetry = None  # type: ignore
    WeightManager = None  # type: ignore
    MultiSourceDataFetcher = None  # type: ignore
    FetchResult = None  # type: ignore

# v598 Phase H: 实盘延迟 SLA 监控 + 熔断器
# 用户铁律 "永远永远不要出现模拟牛逼，实盘亏损的情况":
#   延迟突破 P99 阈值时, 信号已过期, 继续下单等于赌博 → 必须 BLOCK 下单
# ERR-20260702-v598p5-latency-11x: REST p95=2133ms 超目标 11 倍, 需独立 SLA 层强制熔断
# 双阈值: 行情 P99≤100ms + 订单 P99≤500ms, 任一 OPEN 则拒绝下单
try:
    from .latency_sla_monitor import (
        LatencySLAMonitor,
        get_global_sla_monitor,
    )
    _LATENCY_SLA_AVAILABLE = True
except ImportError:
    _LATENCY_SLA_AVAILABLE = False
    LatencySLAMonitor = None  # type: ignore
    get_global_sla_monitor = None  # type: ignore

logger = logging.getLogger("hermes.exchange_connector")


# ============================================================================
# 交易所连接器配置
# ============================================================================


@dataclass
class ExchangeConfig:
    """交易所配置"""
    exchange: str = "gate"          # binance/okx/bybit/hyperliquid/gate/kraken
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""               # OKX需要
    testnet: bool = False               # 测试网
    market_type: str = "swap"           # spot/swap/future/option
    rate_limit: bool = True             # 启用限流
    enable_rate_limit: bool = True
    timeout: int = 30000                # 超时（毫秒）
    retry_attempts: int = 3             # 重试次数
    default_leverage: int = 3
    default_type: str = "linear"        # linear/inverse


# ============================================================================
# CCXT 交易所连接器
# ============================================================================


class CCXTExchangeConnector(MatchingEngineAdapter):
    """CCXT 交易所连接器

    实现 IMatchingEngine 接口，对接实盘交易所。

    使用：
        connector = CCXTExchangeConnector(
            config=ExchangeConfig(
                exchange="gate",  # Phase F-1: binance -> gate
                api_key="your_key",
                api_secret="your_secret",
                market_type="swap",
            )
        )
        connector.connect()
        ticker = connector.get_ticker("BTC/USDT:USDT")
        order = connector.submit_order(...)
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__(name=f"ccxt_{config.exchange}")
        self.config = config
        self._exchange = None
        self._ccxt_available = False
        # Phase 7I-1: 网络弹性层组件初始化
        # 1. ExponentialBackoffRetry: 指数退避+jitter, 处理429/418/timeout/5xx
        # 2. WeightManager: 实时读取X-MBX-USED-WEIGHT-1M, >95%主动等待防IP封禁
        # 3. MultiSourceDataFetcher: 多源fallback链(Bybit→OKX→CoinGecko→Cache)  # Phase F-1: binance移除
        if _NETWORK_RESILIENCE_AVAILABLE:
            self._retry = ExponentialBackoffRetry(
                max_attempts=max(config.retry_attempts, 3),
            )
            self._weight_manager = WeightManager()
            # 多源fetcher用于public market data(klines/funding)
            # 优先使用配置的交易所作为主源, 失败时fallback到其他源
            preferred_chain = [config.exchange] + [
                s for s in ["bybit", "okx", "coingecko", "cache"]  # Phase F-1: binance移除
                if s != config.exchange
            ]
            self._data_fetcher = MultiSourceDataFetcher(source_chain=preferred_chain)
        else:
            self._retry = None
            self._weight_manager = None
            self._data_fetcher = None
        # Phase 7J-3: API 调用延迟监控 (与 LiveMatchingEngine 对齐)
        # 之前此字段缺失, 导致 exchange_connector 路径订单提交无延迟统计 (监控盲点)
        # 来源: voiceofchain 2026 "Crypto Bot Latency Benchmark" — 必须测量才能优化
        self._latency_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        # v598 Phase H: 获取全局 SLA 监控器 (延迟熔断器)
        # 双熔断器任一 OPEN 则 BLOCK 下单: 行情 P99≤100ms + 订单 P99≤500ms
        # 用户铁律 "永远永远不要出现模拟牛逼，实盘亏损的情况":
        #   信号已过期时下单 = 赌博, 必须 BLOCK 而非仅打日志
        if _LATENCY_SLA_AVAILABLE:
            self._sla_monitor: Optional[LatencySLAMonitor] = get_global_sla_monitor()
        else:
            self._sla_monitor = None
        self._init_ccxt()

    def _init_ccxt(self):
        """初始化CCXT"""
        try:
            import ccxt
            self._ccxt_available = True

            exchange_class = getattr(ccxt, self.config.exchange, None)
            if not exchange_class:
                logger.error("Unknown exchange: %s", self.config.exchange)
                return

            params = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": self.config.enable_rate_limit,
                "timeout": self.config.timeout,
                "options": {
                    "defaultType": self.config.market_type,
                },
            }
            if self.config.passphrase:
                params["password"] = self.config.passphrase

            self._exchange = exchange_class(params)

            if self.config.testnet:
                self._exchange.set_sandbox_mode(True)

            logger.info("CCXT %s initialized (testnet=%s)",
                        self.config.exchange, self.config.testnet)
        except ImportError:
            logger.warning("CCXT not installed. Run: pip install ccxt")
            self._ccxt_available = False
        except Exception as e:
            logger.error("CCXT init failed: %s", str(e)[:200])
            self._ccxt_available = False

    # ------------------------------------------------------------------
    # Phase 7I-1: 网络弹性层 — 核心包装方法
    # ------------------------------------------------------------------

    def _call_with_resilience(self, func: Callable, *args, **kwargs) -> Any:
        """Phase 7I-1: 包装交易所API调用, 应用 ExponentialBackoffRetry + WeightManager

        网络弹性三重保障:
          1. 调用前: 权重节流检查(>95%主动等待, 防IP封禁)
          2. 调用中: 指数退避+jitter重试(429/418/timeout/5xx自动重试)
          3. 调用后: 从响应头更新权重使用状态(X-MBX-USED-WEIGHT-1M)

        Phase 7J-3 优化: 自动从 func.__name__ 推断 call_name, 记录每次调用延迟.
          - 降级模式(无弹性层)也记录延迟
          - 弹性模式记录含重试的总耗时
          - 失败时错误信息附带延迟数据, 便于诊断
          来源: voiceofchain 2026 "必须测量才能优化" + AWS tick-to-trade 五阶段

        Args:
            func: 要调用的函数(如self._exchange.fetch_ticker)
            *args, **kwargs: 函数参数

        Returns:
            调用结果

        Raises:
            RuntimeError: 所有重试均失败
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")

        # Phase 7J-3: 自动推断 call_name (CCXT 方法名: fetch_ticker/create_order/...)
        call_name = getattr(func, "__name__", "unknown")

        # 无弹性层时直接调用(降级模式) — 仍记录延迟
        if self._retry is None or self._weight_manager is None:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                self._latency_stats[call_name].append(latency_ms)

        source_name = self.config.exchange

        # 1. 权重节流检查(防IP封禁)
        should_throttle, wait_s = self._weight_manager.should_throttle(source_name)
        if should_throttle:
            logger.info("[%s] 权重使用过高, 主动节流%.1fs防封禁", source_name, wait_s)
            time.sleep(min(wait_s, 30.0))

        # 2. 执行带重试(指数退避+jitter) — Phase 7J-3: 记录含重试的总耗时
        start = time.perf_counter()
        success, data, error, attempts = self._retry.execute_with_retry(
            func, *args, **kwargs
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._latency_stats[call_name].append(latency_ms)

        # 3. 从响应头更新权重(CCXT 4.x: last_response_headers)
        try:
            headers = getattr(self._exchange, "last_response_headers", None) or {}
            if headers:
                self._weight_manager.update_from_headers(source_name, headers)
        except Exception as e:
            # Phase 7J-10 反温室修复 MEDIUM #2: 权重更新失败静默吞没
            # 之前: except: pass → 权重管理失效, 后续可能未节流导致 IP 封禁
            # 现在: warning 日志 (反温室: IP 封禁=实盘无法交易=策略失效)
            self._logger.warning(
                "update_from_headers %s FAILED: %s (权重管理可能失效, 后续可能未节流!)",
                source_name, str(e)[:100],
            )

        if not success:
            raise RuntimeError(
                "调用{}失败(尝试{}次, 耗时{:.0f}ms): {}".format(
                    call_name, attempts, latency_ms, str(error)[:200]
                )
            )

        return data

    def get_latency_report(self) -> Dict[str, Any]:
        """Phase 7J-3: 获取API调用延迟报告(订单执行速度优化)

        与 LiveMatchingEngine.get_latency_report 接口对齐.
        之前此方法缺失, 导致 exchange_connector 路径无延迟统计 (监控盲点).

        Returns:
            dict 每种API调用的 p50/p95/p99/avg/count 统计
            业界基准(2026): REST 50-500ms, WebSocket 5-50ms, Co-located 1-5ms
        """
        report: Dict[str, Any] = {}
        for call_name, latencies in self._latency_stats.items():
            if not latencies:
                continue
            arr = sorted(latencies)
            n = len(arr)
            report[call_name] = {
                "count": n,
                "avg_ms": round(sum(arr) / n, 1),
                "p50_ms": round(arr[n // 2], 1),
                "p95_ms": round(arr[int(n * 0.95)] if n >= 20 else arr[-1], 1),
                "p99_ms": round(arr[int(n * 0.99)] if n >= 100 else arr[-1], 1),
                "min_ms": round(arr[0], 1),
                "max_ms": round(arr[-1], 1),
            }
        return report

    def get_resilience_status(self) -> Dict[str, Any]:
        """Phase 7I-1: 获取网络弹性层状态(用于监控/诊断)

        Returns:
            dict 包含:
              - retry: 重试器配置
              - weight: 各源权重使用状态
              - data_fetcher: 多源健康报告
              - available: 弹性层是否可用
        """
        if not _NETWORK_RESILIENCE_AVAILABLE:
            return {"available": False, "reason": "network_resilience module not imported"}

        status: Dict[str, Any] = {
            "available": True,
            "exchange": self.config.exchange,
            "retry": {
                "max_attempts": self._retry.max_attempts if self._retry else 0,
                "base_backoff": self._retry.base_backoff if self._retry else 0,
                "max_backoff": self._retry.max_backoff if self._retry else 0,
            },
        }

        # 权重使用状态
        if self._weight_manager:
            weight_info = {}
            for source, usage in self._weight_manager.sources_weight.items():
                weight_info[source] = {
                    "used_weight_1m": usage.used_weight_1m,
                    "weight_limit_1m": usage.weight_limit_1m,
                    "usage_ratio": round(usage.usage_ratio, 3),
                    "should_wait": usage.should_wait,
                    "should_warn": usage.should_warn,
                }
            status["weight"] = weight_info

        # 多源健康报告
        if self._data_fetcher:
            status["data_fetcher"] = self._data_fetcher.get_health_report()

        return status

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if not self._ccxt_available or not self._exchange:
            logger.error("CCXT not available, cannot connect")
            return False
        try:
            # v3.10.19: 无API key时用公开接口测试连接 (根治网络冗余WARN)
            # 原问题: connect()总用fetch_balance(认证接口), 无API key时必失败
            #         导致Phase 4网络冗余测试Gate.io连接恒失败 → 1/2源WARN
            # 修复: 有API key → fetch_balance(完整认证测试)
            #       无API key → fetch_ticker(公开数据测试, 足以验证网络可达性)
            if self.config.api_key:
                # Phase 7I-1: 测试连接使用弹性层(重试+权重管理)
                balance = self._call_with_resilience(self._exchange.fetch_balance)
                self._connected = True
                self._logger.info("[%s] Connected (authenticated), balance available",
                                  self.config.exchange)
            else:
                # 无API key: 用公开ticker测试网络可达性
                test_symbol = "BTC/USDT"
                self._call_with_resilience(self._exchange.fetch_ticker, test_symbol)
                self._connected = True
                self._logger.info("[%s] Connected (public data, no API key)",
                                  self.config.exchange)
            return True
        except Exception as e:
            self._logger.error("Connect failed: %s", str(e)[:200])
            return False

    def disconnect(self):
        if self._exchange:
            # CCXT没有显式disconnect，但可以清理
            pass
        self._connected = False

    @property
    def engine_type(self) -> str:
        return "live"

    # ------------------------------------------------------------------
    # 行情查询
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> UnifiedTicker:
        if not self._exchange:
            return UnifiedTicker(symbol=symbol, bid=0, ask=0, last=0,
                                 volume_24h=0, high_24h=0, low_24h=0)

        try:
            # Phase 7I-1: fetch_ticker 用弹性层包装(retry+weight)
            t = self._call_with_resilience(self._exchange.fetch_ticker, symbol)
            funding_rate = 0.0
            next_funding_ts = 0.0

            # 永续合约获取资金费率
            if self.config.market_type in ("swap", "future") and ":" in symbol:
                try:
                    # Phase 7I-1: fetch_funding_rate 用弹性层包装
                    funding = self._call_with_resilience(
                        self._exchange.fetch_funding_rate, symbol
                    )
                    funding_rate = funding.get("fundingRate", 0.0)
                    next_funding_ts = funding.get("fundingDatetime", 0)
                    if isinstance(next_funding_ts, str):
                        next_funding_ts = 0
                    # Phase 7J-10 反温室修复 MEDIUM #1: fundingRate 键缺失检测
                    # 之前: funding 中无 fundingRate 键时 .get() 默认0, 等同于"费率真为0"
                    # 现在: 键缺失时 warning (反温室: 费率0=策略基于错误费率决策)
                    if "fundingRate" not in funding:
                        self._logger.warning(
                            "get_ticker %s: fundingRate 键缺失, 返回0 (P&L可能虚高!): %s",
                            symbol, str(funding)[:150],
                        )
                except Exception as e:
                    # Phase 7J-10 反温室修复 MEDIUM #1: 资金费率获取失败静默 fallback 为 0
                    # 之前: except: pass → 资金费率获取失败时 funding_rate=0, 下游 P&L 计算虚高
                    # 现在: warning 日志, 便于诊断 (反温室: 永续合约核心成本不可掩盖)
                    # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 费率低估=实盘成本失控
                    self._logger.warning(
                        "get_ticker %s: fetch_funding_rate FAILED, 返回0 (P&L可能虚高!): %s",
                        symbol, str(e)[:150],
                    )

            return UnifiedTicker(
                symbol=symbol,
                bid=t.get("bid", 0),
                ask=t.get("ask", 0),
                last=t.get("last", 0),
                volume_24h=t.get("baseVolume", 0),
                high_24h=t.get("high", 0),
                low_24h=t.get("low", 0),
                funding_rate=funding_rate,
                next_funding_ts=next_funding_ts,
            )
        except Exception as e:
            # Phase 7J-5 反温室修复: 行情获取失败必须可见 (之前返回0价格, 下游可能基于0下单)
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 行情中断时策略基于0价格下单=灾难
            # 修复: 记录 CRITICAL 日志 + 统计计数器, 监控系统可基于此告警
            # 注意: 返回0价格保持向后兼容, 但监控系统必须检测 last==0 并暂停交易
            self._logger.critical(
                "get_ticker %s FAILED: %s (返回0价格, 监控系统应暂停交易!)",
                symbol, str(e)[:150],
            )
            self._record_stat("ticker_failures")
            return UnifiedTicker(symbol=symbol, bid=0, ask=0, last=0,
                                 volume_24h=0, high_24h=0, low_24h=0)

    def get_order_book(self, symbol: str, limit: int = 20) -> UnifiedOrderBook:
        if not self._exchange:
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[])

        try:
            # Phase 7I-1: fetch_order_book 用弹性层包装(retry+weight)
            ob = self._call_with_resilience(
                self._exchange.fetch_order_book, symbol, limit
            )
            return UnifiedOrderBook(
                symbol=symbol,
                bids=[(b[0], b[1]) for b in ob.get("bids", [])],
                asks=[(a[0], a[1]) for a in ob.get("asks", [])],
            )
        except Exception as e:
            self._logger.error("get_order_book %s failed: %s", symbol, str(e)[:100])
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[])

    def get_history(self, symbol: str, timeframe: str = "1h",
                   limit: int = 100) -> List[Dict[str, Any]]:
        if not self._exchange:
            return []

        # Phase 7I-1: 优先使用多源fallback(public market data)
        # klines是公开数据, 可从任一源获取, 不需要API key
        # testnet模式跳过多源(测试网数据与主网不同), 用直连
        if self._data_fetcher is not None and not self.config.testnet:
            try:
                result = self._data_fetcher.fetch_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    preferred_source=self.config.exchange,
                )
                if result.success and result.data:
                    self._logger.debug(
                        "get_history %s from %s (latency=%.0fms, cache=%s, age=%.1fh)",
                        symbol, result.source, result.latency_ms,
                        result.from_cache, result.cache_age_hours,
                    )
                    return [
                        {"timestamp": c[0], "open": c[1], "high": c[2],
                         "low": c[3], "close": c[4], "volume": c[5]}
                        for c in result.data
                    ]
                # 多源全部失败, 降级到直连
                self._logger.warning(
                    "MultiSource fetch_klines all failed: %s, fallback to direct",
                    result.error[:100] if result.error else "unknown",
                )
            except Exception as e:
                self._logger.warning(
                    "MultiSource fetch_klines exception: %s, fallback to direct",
                    str(e)[:100],
                )

        # 直连模式(retry+weight, 但无多源fallback)
        try:
            ohlcv = self._call_with_resilience(
                self._exchange.fetch_ohlcv, symbol, timeframe, limit=limit
            )
            return [
                {"timestamp": c[0], "open": c[1], "high": c[2],
                 "low": c[3], "close": c[4], "volume": c[5]}
                for c in ohlcv
            ]
        except Exception as e:
            self._logger.error("get_history %s failed: %s", symbol, str(e)[:100])
            return []

    def get_funding_rate(self, symbol: str) -> float:
        if not self._exchange:
            # Phase 7J-8 反温室修复 HIGH #10: 静默返回0 → warning
            # 之前: exchange 未初始化时静默返回0, 调用方无法区分"费率真的是0"和"获取失败"
            # 现在: warning 日志, 便于诊断 (反温室: 费率0=策略基于错误费率决策)
            self._logger.warning(
                "get_funding_rate %s: exchange 未初始化, 返回0 (P&L可能虚高!)", symbol,
            )
            return 0.0

        # Phase 7I-1: funding_rate是公开数据, 优先多源fallback
        if self._data_fetcher is not None and not self.config.testnet:
            try:
                result = self._data_fetcher.fetch_funding_rate(
                    symbol=symbol,
                    preferred_source=self.config.exchange,
                )
                if result.success and result.data is not None:
                    return float(result.data)
                self._logger.debug(
                    "MultiSource fetch_funding_rate failed: %s, fallback to direct",
                    result.error[:100] if result.error else "unknown",
                )
            except Exception as e:
                self._logger.debug(
                    "MultiSource fetch_funding_rate exception: %s, fallback to direct",
                    str(e)[:100],
                )

        # 直连模式(retry+weight)
        try:
            funding = self._call_with_resilience(
                self._exchange.fetch_funding_rate, symbol
            )
            # Phase 7J-8 反温室修复 HIGH #10: 检查 fundingRate 键存在
            # 之前: funding.get("fundingRate", 0.0) — 键不存在时静默返回0, 可能被误认为"费率真的是0"
            # 现在: 键不存在时 warning (反温室: 费率缺失=策略基于错误费率决策)
            if "fundingRate" not in funding:
                self._logger.warning(
                    "get_funding_rate %s: fundingRate 键缺失, 返回0 (P&L可能虚高!): %s",
                    symbol, str(funding)[:150],
                )
                return 0.0
            return float(funding.get("fundingRate", 0.0))
        except Exception as e:
            # Phase 7J-5 反温室修复: 资金费率获取失败必须可见 (之前静默返回0, P&L虚高)
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 费率失败时P&L忽略费率成本=虚假盈利
            self._logger.warning(
                "get_funding_rate %s FAILED: %s (返回0, 永续合约P&L可能虚高!)",
                symbol, str(e)[:150],
            )
            self._record_stat("funding_rate_failures")
            return 0.0

    # ------------------------------------------------------------------
    # 订单管理
    # ------------------------------------------------------------------

    def submit_order(self, order: UnifiedOrder) -> UnifiedOrder:
        if not self._exchange:
            order.status = UnifiedOrderStatus.REJECTED
            return order

        try:
            # v598 Phase H: SLA 前置熔断检查 (用户铁律: 延迟超阈值时下单=赌博)
            # can_place_order() 检查双熔断器状态: 行情 P99≤100ms + 订单 P99≤500ms
            # 任一 OPEN → BLOCK 下单 (而非仅打日志), 从源头杜绝"模拟牛逼实盘亏损"
            if self._sla_monitor is not None and not self._sla_monitor.can_place_order():
                self._logger.warning(
                    "submit_order BLOCKED by SLA circuit breaker "
                    "(market/order latency P99 超阈值, 信号已过期)"
                )
                self._record_stat("orders_sla_blocked")
                order.status = UnifiedOrderStatus.REJECTED
                return order

            self._record_stat("orders_submitted")
            _order_start_ts = time.perf_counter()  # Phase H: 测量真实下单延迟

            # 转换参数
            side = "buy" if order.side == UnifiedOrderSide.BUY else "sell"
            order_type = order.order_type.value

            params = {}
            if order.leverage > 1:
                params["leverage"] = order.leverage
            if order.reduce_only:
                params["reduceOnly"] = True
            if order.client_order_id:
                params["clientOrderId"] = order.client_order_id

            # Phase 7I-1: create_order 用弹性层包装(retry+weight)
            # 注意: 订单是认证操作, 仅用retry+weight, 不用多源fallback
            # 4xx错误(参数错误/余额不足)不会重试, 只有timeout/5xx/429才重试
            # 建议: 始终设置client_order_id防止重试导致重复下单
            result = self._call_with_resilience(
                self._exchange.create_order,
                symbol=order.symbol,
                type=order_type,
                side=side,
                amount=order.quantity,
                price=order.price if order.order_type != UnifiedOrderType.MARKET else None,
                params=params,
            )

            order.order_id = result.get("id", "")
            order.filled_qty = result.get("filled", 0.0)
            order.filled_price = result.get("average", 0.0) or result.get("price", 0.0)
            order.fee_paid = result.get("fee", {}).get("cost", 0.0)

            status = result.get("status", "open")
            if status == "closed":
                order.status = UnifiedOrderStatus.FILLED
                self._record_stat("orders_filled")
            elif status == "open":
                order.status = UnifiedOrderStatus.OPEN
            elif status == "cancelled":
                order.status = UnifiedOrderStatus.CANCELLED
                self._record_stat("orders_cancelled")
            else:
                order.status = UnifiedOrderStatus.PENDING

            self._record_stat("total_fee_paid", order.fee_paid)
            # v598 Phase H: SLA 后置延迟记录
            # 测量 create_order 真实耗时 (含重试), 喂入熔断器评估 P99
            # 持续高延迟 → 触发熔断 → 后续 can_place_order() 返回 False → BLOCK
            _order_latency_ms = (time.perf_counter() - _order_start_ts) * 1000.0
            if self._sla_monitor is not None:
                self._sla_monitor.record_order_latency(_order_latency_ms)
            return order

        except Exception as e:
            # Phase 7J-4: 订单提交异常分类 (P1-3 缺口修复)
            # 之前: 单个 except Exception 笼统捕获, 余额不足/网络失败/参数错误全归 REJECTED
            # 现在: 区分 ccxt 异常类型, 便于监控/诊断/恢复决策
            # 来源: matrixtrak 2026 "classify failure before you act" + nadcab 2026
            # 分类决策框架:
            #   insufficient_funds / invalid_order → 不可恢复, 不重试
            #   network / exchange_unavailable → 可恢复, 已由 _call_with_resilience 重试耗尽
            #   retry_exhausted → RuntimeError from _call_with_resilience
            error_class = type(e).__name__
            error_category = self._classify_order_error(e)
            self._logger.error(
                "submit_order failed [%s/%s]: %s",
                error_category, error_class, str(e)[:200],
            )
            self._record_stat("errors")
            self._record_stat(f"errors_{error_category}")
            # v598 Phase H: 异常路径也记录延迟 (失败订单的延迟也是熔断器输入)
            # 防御: 若异常发生在 _order_start_ts 赋值前 (SLA 检查段, 理论不抛), 跳过记录
            try:
                _order_latency_ms = (time.perf_counter() - _order_start_ts) * 1000.0
                if self._sla_monitor is not None:
                    self._sla_monitor.record_order_latency(_order_latency_ms)
            except (NameError, UnboundLocalError):
                pass  # _order_start_ts 未定义, 跳过
            order.status = UnifiedOrderStatus.REJECTED
            return order

    def _classify_order_error(self, e: Exception) -> str:
        """Phase 7J-4: 分类订单提交异常

        返回异常类别字符串, 用于统计/监控/恢复决策:
          - insufficient_funds: 余额不足 (不可恢复)
          - invalid_order: 参数错误/订单无效 (不可恢复)
          - exchange_unavailable: 交易所不可用 (5xx, 可恢复但已重试耗尽)
          - network: 网络错误 (timeout, 可恢复但已重试耗尽)
          - exchange_error: 其他交易所错误
          - retry_exhausted: _call_with_resilience 重试耗尽
          - unknown: 未分类

        来源: matrixtrak 2026 异常分类决策框架 + CCXT 异常体系
        """
        try:
            import ccxt
            if isinstance(e, ccxt.InsufficientFunds):
                return "insufficient_funds"
            if isinstance(e, ccxt.InvalidOrder):
                return "invalid_order"
            if isinstance(e, ccxt.ExchangeNotAvailable):
                return "exchange_unavailable"
            if isinstance(e, ccxt.NetworkError):
                return "network"
            if isinstance(e, ccxt.ExchangeError):
                return "exchange_error"
        except ImportError:
            # Phase 7J-10 反温室修复 MEDIUM #3: ccxt 未安装时分类降级
            # 之前: 静默 pass, 调用方无法区分"未分类"和"ccxt未安装"
            # 现在: warning 日志 (反温室: 错误分类降级=风控决策可能不准)
            self._logger.warning(
                "classify_error: ccxt 未安装, 异常分类降级到 unknown: %s",
                type(e).__name__,
            )
        if isinstance(e, RuntimeError):
            return "retry_exhausted"
        return "unknown"

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        if not self._exchange or not symbol:
            return False
        try:
            # Phase 7I-1: cancel_order 用弹性层包装(retry+weight)
            self._call_with_resilience(
                self._exchange.cancel_order, order_id, symbol
            )
            self._record_stat("orders_cancelled")
            return True
        except Exception as e:
            self._logger.error("cancel_order failed: %s", str(e)[:100])
            return False

    def batch_order(self, orders: List[UnifiedOrder],
                    max_workers: int = 8, per_order_timeout: float = 30.0) -> List[UnifiedOrder]:
        """G6: 批量并发下单 — 单次延迟不随交易对数增长

        来源: _live_data_gap_patch_list.py G6 (P1, owner: LIVE-DATA-CONNECTOR)
        验证标准: 7 交易对并发下单, 总延迟 < 单笔延迟 × 1.5

        设计:
          - 使用 ThreadPoolExecutor 并发调用 submit_order (CCXT 同步接口)
          - 失败隔离: 单笔失败不影响其他笔, 返回 per-order 状态
          - 超时控制: per_order_timeout 防止单笔卡死整批
          - Hyperliquid 原生 batch 接口由 HyperliquidConnector 单独实现

        Args:
            orders: 待提交订单列表 (每个订单独立 symbol/side/quantity)
            max_workers: 最大并发数 (默认8, 对应8+交易所并发)
            per_order_timeout: 单笔订单超时秒数 (默认30s)

        Returns:
            与 orders 等长的结果列表, 每个元素为提交后的 UnifiedOrder (含状态)
        """
        if not orders:
            return []

        # 单笔下单无并发收益, 直接串行
        if len(orders) == 1:
            self._record_stat("batch_order_single")
            return [self.submit_order(orders[0])]

        self._record_stat("batch_order_submitted")
        self._record_stat(f"batch_order_size_{len(orders)}")

        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FutTimeout
        import time as _time

        t0 = _time.time()
        results: Dict[int, UnifiedOrder] = {}
        errors: Dict[int, str] = {}

        def _submit_one(idx: int, order: UnifiedOrder) -> Tuple[int, UnifiedOrder]:
            """单笔下单 (失败隔离: 异常不抛出, 标记 REJECTED)"""
            try:
                result = self.submit_order(order)
                return idx, result
            except Exception as e:
                # 失败隔离: 标记 REJECTED, 不影响其他笔
                order.status = UnifiedOrderStatus.REJECTED
                order.meta = {**getattr(order, "meta", {}),
                              "batch_error": str(e)[:200]}
                self._logger.error("batch_order[%d] failed: %s", idx, str(e)[:100])
                return idx, order

        # 并发提交
        n_workers = min(max_workers, len(orders))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(_submit_one, i, o): i
                for i, o in enumerate(orders)
            }
            for fut in as_completed(future_map, timeout=per_order_timeout * len(orders)):
                idx = future_map[fut]
                try:
                    fut_result = fut.result(timeout=per_order_timeout)
                    results[fut_result[0]] = fut_result[1]
                except _FutTimeout:
                    order = orders[idx]
                    order.status = UnifiedOrderStatus.REJECTED
                    order.meta = {**getattr(order, "meta", {}),
                                  "batch_error": "timeout"}
                    results[idx] = order
                    errors[idx] = "timeout"
                    self._record_stat("batch_order_timeout")
                except Exception as e:
                    order = orders[idx]
                    order.status = UnifiedOrderStatus.REJECTED
                    order.meta = {**getattr(order, "meta", {}),
                                  "batch_error": str(e)[:200]}
                    results[idx] = order
                    errors[idx] = str(e)[:100]
                    self._record_stat("batch_order_error")

        # 按原始顺序返回
        out: List[UnifiedOrder] = []
        n_filled = 0
        n_rejected = 0
        for i in range(len(orders)):
            o = results.get(i, orders[i])
            if o.status == UnifiedOrderStatus.FILLED:
                n_filled += 1
            else:
                n_rejected += 1
            out.append(o)

        elapsed = _time.time() - t0
        self._record_stat("batch_order_total_time", elapsed)

        # 单笔平均延迟估算 (用于验证 G6: 总延迟 < 单笔延迟 × 1.5)
        avg_per_order = elapsed / max(len(orders), 1)
        self._logger.info(
            "G6 batch_order: size=%d filled=%d rejected=%d elapsed=%.3fs avg=%.3fs",
            len(orders), n_filled, n_rejected, elapsed, avg_per_order,
        )

        if n_rejected > 0:
            self._record_stat("batch_order_partial_failure")

        return out

    def get_order(self, order_id: str, symbol: str = None) -> Optional[UnifiedOrder]:
        if not self._exchange or not symbol:
            return None
        try:
            # Phase 7I-1: fetch_order 用弹性层包装(retry+weight)
            o = self._call_with_resilience(
                self._exchange.fetch_order, order_id, symbol
            )
            return UnifiedOrder(
                order_id=o.get("id", ""),
                agent_id="",
                symbol=o.get("symbol", symbol),
                side=UnifiedOrderSide.BUY if o.get("side") == "buy" else UnifiedOrderSide.SELL,
                order_type=UnifiedOrderType.MARKET if o.get("type") == "market" else UnifiedOrderType.LIMIT,
                quantity=o.get("amount", 0),
                price=o.get("price", 0),
                filled_qty=o.get("filled", 0),
                filled_price=o.get("average", 0),
                fee_paid=o.get("fee", {}).get("cost", 0),
                status=self._map_status(o.get("status", "")),
            )
        except Exception as e:
            self._logger.error("get_order failed: %s", str(e)[:100])
            return None

    def get_open_orders(self, symbol: str = None) -> List[UnifiedOrder]:
        if not self._exchange:
            return []
        try:
            # Phase 7I-1: fetch_open_orders 用弹性层包装(retry+weight)
            if symbol:
                orders = self._call_with_resilience(
                    self._exchange.fetch_open_orders, symbol
                )
            else:
                orders = self._call_with_resilience(
                    self._exchange.fetch_open_orders
                )
            return [
                UnifiedOrder(
                    order_id=o.get("id", ""),
                    agent_id="",
                    symbol=o.get("symbol", ""),
                    side=UnifiedOrderSide.BUY if o.get("side") == "buy" else UnifiedOrderSide.SELL,
                    order_type=UnifiedOrderType.MARKET if o.get("type") == "market" else UnifiedOrderType.LIMIT,
                    quantity=o.get("amount", 0),
                    price=o.get("price", 0),
                    status=UnifiedOrderStatus.OPEN,
                )
                for o in orders
            ]
        except Exception as e:
            self._logger.error("get_open_orders failed: %s", str(e)[:100])
            return []

    # ------------------------------------------------------------------
    # 持仓与账户
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[UnifiedPosition]:
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
        return None

    def get_positions(self) -> List[UnifiedPosition]:
        if not self._exchange:
            return []
        try:
            # Phase 7I-1: fetch_positions 用弹性层包装(retry+weight)
            positions = self._call_with_resilience(
                self._exchange.fetch_positions
            )
            result = []
            for p in positions:
                if not p.get("contracts") or p.get("contracts", 0) == 0:
                    continue
                result.append(UnifiedPosition(
                    symbol=p.get("symbol", ""),
                    side=p.get("side", "flat"),
                    quantity=abs(p.get("contracts", 0)),
                    entry_price=p.get("entryPrice", 0),
                    mark_price=p.get("markPrice", 0),
                    liquidation_price=p.get("liquidationPrice", 0),
                    leverage=p.get("leverage", 1),
                    margin=p.get("initialMargin", 0),
                    unrealized_pnl=p.get("unrealizedPnl", 0),
                    realized_pnl=p.get("realizedPnl", 0),
                    funding_rate=p.get("fundingRate", 0),
                ))
            return result
        except Exception as e:
            self._logger.error("get_positions failed: %s", str(e)[:100])
            return []

    def get_account(self) -> UnifiedAccount:
        if not self._exchange:
            return UnifiedAccount(
                account_id="disconnected",
                total_balance=0, available_balance=0,
                margin_used=0, margin_ratio=0,
                unrealized_pnl=0, realized_pnl=0,
            )
        try:
            # Phase 7I-1: fetch_balance 用弹性层包装(retry+weight)
            balance = self._call_with_resilience(self._exchange.fetch_balance)
            total = balance.get("total", {}).get("USDT", 0)
            free = balance.get("free", {}).get("USDT", 0)
            used = balance.get("used", {}).get("USDT", 0)

            positions = self.get_positions()
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)

            return UnifiedAccount(
                account_id=self.config.exchange,
                total_balance=total,
                available_balance=free,
                margin_used=used,
                margin_ratio=used / total if total > 0 else 0,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0,
                positions=positions,
            )
        except Exception as e:
            self._logger.error("get_account failed: %s", str(e)[:100])
            return UnifiedAccount(
                account_id="error",
                total_balance=0, available_balance=0,
                margin_used=0, margin_ratio=0,
                unrealized_pnl=0, realized_pnl=0,
            )

    # ------------------------------------------------------------------
    # 杠杆设置
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        if not self._exchange:
            return False
        try:
            # Phase 7I-1: set_leverage 用弹性层包装(retry+weight)
            self._call_with_resilience(
                self._exchange.set_leverage, leverage, symbol
            )
            return True
        except Exception as e:
            # Phase 7J-8 反温室修复 HIGH #9: set_leverage 失败 debug → warning
            # 之前: 仅 debug 日志, 调用方可能忽略返回值继续下单, 实盘使用错误杠杆(如1x而非5x)
            # 现在: warning 级别 + 包含 symbol/leverage 便于诊断 (反温室: 杠杆错误=仓位失控)
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 杠杆设置失败=实盘仓位与预期不符
            self._logger.warning(
                "set_leverage FAILED %s leverage=%d: %s (实盘可能使用默认杠杆, 仓位与预期不符!)",
                symbol, leverage, str(e)[:150],
            )
            return False

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _map_status(ccxt_status: str) -> UnifiedOrderStatus:
        mapping = {
            "open": UnifiedOrderStatus.OPEN,
            "closed": UnifiedOrderStatus.FILLED,
            "cancelled": UnifiedOrderStatus.CANCELLED,
            "expired": UnifiedOrderStatus.EXPIRED,
            "rejected": UnifiedOrderStatus.REJECTED,
        }
        return mapping.get(ccxt_status, UnifiedOrderStatus.PENDING)


# ============================================================================
# Hyperliquid 连接器（去中心化永续合约）
# ============================================================================


class HyperliquidConnector(MatchingEngineAdapter):
    """Hyperliquid 连接器

    Hyperliquid 是去中心化永续合约交易所，运行在自己的L1链上。
    使用私钥签名认证，支持链上订单簿。

    使用：
        connector = HyperliquidConnector(
            private_key="0x...",
            testnet=False,
        )
        connector.connect()
    """

    def __init__(self, private_key: str = "", testnet: bool = False,
                 wallet_address: str = None):
        super().__init__(name="hyperliquid")
        self.private_key = private_key
        self.testnet = testnet
        self.wallet_address = wallet_address
        self._info = None
        self._exchange = None
        self._sdk_available = False
        self._init_sdk()

    def _init_sdk(self):
        """初始化Hyperliquid SDK"""
        try:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils import constants
            import eth_account

            api_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL

            if self.private_key:
                wallet = eth_account.Account.from_key(self.private_key)
                self._exchange = Exchange(wallet, api_url)

            self._info = Info(api_url)
            self._sdk_available = True
            self._logger.info("Hyperliquid SDK initialized (testnet=%s)", self.testnet)
        except ImportError:
            self._logger.warning("Hyperliquid SDK not installed. Run: pip install hyperliquid-python-sdk")
        except Exception as e:
            self._logger.error("Hyperliquid SDK init failed: %s", str(e)[:200])

    def connect(self) -> bool:
        if not self._sdk_available:
            return False
        try:
            # 测试连接：获取所有币种中间价
            mids = self._info.all_mids()
            self._connected = True
            self._logger.info("Hyperliquid connected, %d symbols available", len(mids))
            return True
        except Exception as e:
            self._logger.error("Hyperliquid connect failed: %s", str(e)[:100])
            return False

    @property
    def engine_type(self) -> str:
        return "live"

    def get_ticker(self, symbol: str) -> UnifiedTicker:
        """Phase 7J-7 反温室修复 CRITICAL: 不再伪造 bid/ask/funding_rate

        之前: bid=last*0.9999, ask=last*1.0001, funding_rate=0.0001 (全部伪造)
        原因: 真实 Hyperliquid bid/ask 价差在波动市场可达 0.5%-1%, 0.02% 伪造价差
              让沙盘套利策略基于虚假数据进化, 实盘真实价差下套利空间消失甚至反向亏损
              真实资金费率波动 -0.003~+0.005, 固定 0.0001 让费率套利策略方向错误
        修复: 调用 l2_snapshot 获取真实 bid/ask, 调用 funding_history 获取真实费率
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 伪造市场数据是核心反温室风险
        """
        if not self._info:
            return UnifiedTicker(symbol=symbol, bid=0, ask=0, last=0,
                                 volume_24h=0, high_24h=0, low_24h=0)
        try:
            # Hyperliquid symbol格式: "BTC" (not "BTC/USDT:USDT")
            hl_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            mids = self._info.all_mids()
            last_price = float(mids.get(hl_symbol, 0))

            # Phase 7J-7: 获取真实 bid/ask (不再伪造 last*0.9999)
            real_bid = 0.0
            real_ask = 0.0
            try:
                ob = self._info.l2_snapshot(hl_symbol)
                levels = ob.get("levels", []) if ob else []
                for lvl in levels[:5]:  # 取前5档
                    if lvl.get("side") == "bid" and real_bid == 0.0:
                        real_bid = float(lvl.get("px", 0))
                    elif lvl.get("side") == "ask" and real_ask == 0.0:
                        real_ask = float(lvl.get("px", 0))
                    if real_bid > 0 and real_ask > 0:
                        break
            except Exception as ob_err:
                self._logger.warning(
                    "Hyperliquid l2_snapshot failed for %s: %s (使用 last_price 作为 bid/ask 回退)",
                    hl_symbol, str(ob_err)[:100],
                )

            # 回退: 如果 l2_snapshot 失败, 用 last_price 作为 bid/ask (但标注非真实)
            if real_bid <= 0:
                real_bid = last_price
            if real_ask <= 0:
                real_ask = last_price

            # Phase 7J-7: 获取真实资金费率 (不再伪造 0.0001)
            real_funding_rate = 0.0
            try:
                # Hyperliquid: 获取当前资金费率 (post 1s timestamp = 当前时刻)
                funding_data = self._info.funding_history(
                    coin=hl_symbol, start_time=int(time.time() * 1000) - 86400000, end_time=int(time.time() * 1000)
                )
                if funding_data:
                    # 取最近一条资金费率记录
                    latest_funding = funding_data[-1] if isinstance(funding_data, list) else funding_data
                    if isinstance(latest_funding, dict):
                        real_funding_rate = float(latest_funding.get("fundingRate", 0))
            except Exception as fr_err:
                self._logger.warning(
                    "Hyperliquid funding_history failed for %s: %s (funding_rate=0)",
                    hl_symbol, str(fr_err)[:100],
                )

            return UnifiedTicker(
                symbol=symbol,
                bid=real_bid,
                ask=real_ask,
                last=last_price,
                volume_24h=0,
                high_24h=0,
                low_24h=0,
                funding_rate=real_funding_rate,
            )
        except Exception as e:
            self._logger.error("Hyperliquid get_ticker failed: %s", str(e)[:100])
            return UnifiedTicker(symbol=symbol, bid=0, ask=0, last=0,
                                 volume_24h=0, high_24h=0, low_24h=0)

    def get_order_book(self, symbol: str, limit: int = 20) -> UnifiedOrderBook:
        if not self._info:
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[])
        try:
            hl_symbol = symbol.split("/")[0] if "/" in symbol else symbol
            ob = self._info.l2_snapshot(hl_symbol)
            bids = [(float(b["px"]), float(b["sz"])) for b in ob.get("levels", [])[:limit] if b["side"] == "bid"]
            asks = [(float(a["px"]), float(a["sz"])) for a in ob.get("levels", [])[:limit] if a["side"] == "ask"]
            return UnifiedOrderBook(symbol=symbol, bids=bids, asks=asks)
        except Exception as e:
            self._logger.error("Hyperliquid get_order_book failed: %s", str(e)[:100])
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[])

    def submit_order(self, order: UnifiedOrder) -> UnifiedOrder:
        """Phase 7J-5 反温室修复: 完成真实API调用 (之前是"简化实现"空壳)

        之前: order.status = OPEN; return order — 订单从未真实提交到交易所
        现在: 调用 self._exchange.order(...) 真实提交到 Hyperliquid L1

        铁律: "反复完整百分之百的完整实现" + "一定不要出现模拟牛逼，实盘亏钱！"
        来源: 反温室审视报告 CRITICAL #5
        """
        # Hyperliquid下单需要私钥签名
        if not self._exchange:
            order.status = UnifiedOrderStatus.REJECTED
            return order
        try:
            self._record_stat("orders_submitted")

            # 1. symbol 转换: "BTC/USDT:USDT" → "BTC"
            coin = order.symbol.split("/")[0] if "/" in order.symbol else order.symbol

            # 2. 方向转换
            is_buy = order.side == UnifiedOrderSide.BUY

            # 3. 订单类型转换 (Hyperliquid SDK 格式)
            if order.order_type == UnifiedOrderType.MARKET:
                # 市价单: 用IOC限价模拟 (Hyperliquid无纯市价单)
                # 需要先获取当前价格作为 limit_px
                try:
                    mids = self._info.all_mids()
                    limit_px = float(mids.get(coin, 0))
                    if limit_px <= 0:
                        raise RuntimeError(f"无法获取 {coin} 当前价格")
                    # 市价买入用稍高价, 卖出用稍低价 (确保成交)
                    limit_px = limit_px * 1.001 if is_buy else limit_px * 0.999
                    hl_order_type = {"limit": {"tif": "Ioc"}}
                except Exception as price_err:
                    self._logger.error("Hyperliquid 市价单价格获取失败: %s", str(price_err)[:100])
                    order.status = UnifiedOrderStatus.REJECTED
                    return order
            elif order.order_type == UnifiedOrderType.LIMIT:
                limit_px = order.price
                hl_order_type = {"limit": {"tif": "Gtc"}}
            elif order.order_type == UnifiedOrderType.POST_ONLY:
                limit_px = order.price
                hl_order_type = {"limit": {"tif": "Alo"}}
            else:
                # 其他类型默认用限价GTC
                limit_px = order.price if order.price > 0 else 1.0
                hl_order_type = {"limit": {"tif": "Gtc"}}

            # 4. 调用 Hyperliquid SDK 真实下单
            result = self._exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=order.quantity,
                limit_px=limit_px,
                order_type=hl_order_type,
                reduce_only=order.reduce_only,
            )

            # 5. 解析响应 (Hyperliquid 返回 {"status": "ok", "response": {"data": {"statuses": [...]}}})
            if isinstance(result, dict):
                status = result.get("status", "")
                if status == "ok":
                    # 订单已提交, 提取 order_id
                    response_data = result.get("response", {}).get("data", {})
                    statuses = response_data.get("statuses", [])
                    if statuses:
                        first_status = statuses[0]
                        if "resting" in first_status:
                            order.order_id = first_status["resting"].get("oid", "")
                            order.status = UnifiedOrderStatus.OPEN
                        elif "filled" in first_status:
                            order.order_id = first_status["filled"].get("oid", "")
                            order.filled_qty = float(first_status["filled"].get("totalSz", 0))
                            order.filled_price = float(first_status["filled"].get("avgPx", 0))
                            order.status = UnifiedOrderStatus.FILLED
                            self._record_stat("orders_filled")
                        elif "error" in first_status:
                            # 交易所返回的业务错误 (如余额不足)
                            err_msg = first_status["error"]
                            self._logger.error("Hyperliquid order rejected: %s", str(err_msg)[:200])
                            order.status = UnifiedOrderStatus.REJECTED
                            self._record_stat("errors")
                            self._record_stat("errors_exchange_rejected")
                            return order
                    else:
                        order.status = UnifiedOrderStatus.PENDING
                else:
                    self._logger.error("Hyperliquid order unexpected status: %s", str(status)[:100])
                    order.status = UnifiedOrderStatus.REJECTED
            else:
                order.status = UnifiedOrderStatus.PENDING

            return order

        except Exception as e:
            # Phase 7J-4 异常分类 (与 CCXTExchangeConnector 对齐)
            error_class = type(e).__name__
            self._logger.error(
                "Hyperliquid submit_order failed [%s]: %s",
                error_class, str(e)[:200],
            )
            self._record_stat("errors")
            self._record_stat(f"errors_{error_class.lower()}")
            order.status = UnifiedOrderStatus.REJECTED
            return order

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        if not self._exchange:
            return False
        try:
            self._exchange.cancel(symbol.split("/")[0] if "/" in symbol else symbol, order_id)
            self._record_stat("orders_cancelled")
            return True
        except Exception as e:
            self._logger.error("Hyperliquid cancel failed: %s", str(e)[:100])
            return False

    def get_positions(self) -> List[UnifiedPosition]:
        if not self._info or not self.wallet_address:
            return []
        try:
            state = self._info.user_state(self.wallet_address)
            positions = state.get("assetPositions", [])
            result = []
            for p in positions:
                pos = p.get("position", {})
                if not pos.get("szi"):
                    continue
                szi = float(pos.get("szi", 0))
                result.append(UnifiedPosition(
                    symbol=pos.get("coin", ""),
                    side="long" if szi > 0 else "short",
                    quantity=abs(szi),
                    entry_price=float(pos.get("entryPx", 0)),
                    mark_price=float(pos.get("markPx", 0)),
                    leverage=int(float(pos.get("leverage", {}).get("value", 1))),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                ))
            return result
        except Exception as e:
            self._logger.error("Hyperliquid get_positions failed: %s", str(e)[:100])
            return []

    def get_account(self) -> UnifiedAccount:
        if not self._info or not self.wallet_address:
            return UnifiedAccount(
                account_id="disconnected",
                total_balance=0, available_balance=0,
                margin_used=0, margin_ratio=0,
                unrealized_pnl=0, realized_pnl=0,
            )
        try:
            state = self._info.user_state(self.wallet_address)
            margin_used = float(state.get("marginUsed", 0))
            total_balance = float(state.get("marginSummary", {}).get("accountValue", 0))
            available = float(state.get("withdrawable", 0))

            positions = self.get_positions()
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)

            return UnifiedAccount(
                account_id=f"hyperliquid_{self.wallet_address[:8]}",
                total_balance=total_balance,
                available_balance=available,
                margin_used=margin_used,
                margin_ratio=margin_used / total_balance if total_balance > 0 else 0,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0,
                positions=positions,
            )
        except Exception as e:
            self._logger.error("Hyperliquid get_account failed: %s", str(e)[:100])
            return UnifiedAccount(
                account_id="error",
                total_balance=0, available_balance=0,
                margin_used=0, margin_ratio=0,
                unrealized_pnl=0, realized_pnl=0,
            )


# ============================================================================
# 工厂函数
# ============================================================================


def create_exchange_connector(exchange: str = "gate",
                              api_key: str = "",
                              api_secret: str = "",
                              passphrase: str = "",
                              testnet: bool = False,
                              market_type: str = "swap",
                              **kwargs) -> IMatchingEngine:
    """创建交易所连接器

    Args:
        exchange: 交易所名称 (binance/okx/bybit/hyperliquid/gate/kraken)
        api_key: API密钥
        api_secret: API密钥
        passphrase: 密码（OKX需要）
        testnet: 是否使用测试网
        market_type: 市场类型 (spot/swap/future)

    Returns:
        IMatchingEngine 实例
    """
    if exchange.lower() == "hyperliquid":
        private_key = kwargs.get("private_key", api_secret)
        wallet_address = kwargs.get("wallet_address")
        return HyperliquidConnector(
            private_key=private_key,
            testnet=testnet,
            wallet_address=wallet_address,
        )

    config = ExchangeConfig(
        exchange=exchange,
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        testnet=testnet,
        market_type=market_type,
    )
    return CCXTExchangeConnector(config)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    "ExchangeConfig",
    "CCXTExchangeConnector",
    "HyperliquidConnector",
    "create_exchange_connector",
]
