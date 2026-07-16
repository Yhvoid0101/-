# -*- coding: utf-8 -*-
"""
实盘撮合引擎 (Live Matching Engine) — P0-2 替换边界模式实盘路径

实现 IMatchingEngine 接口，对接真实交易所API。
策略代码通过 MatchingEngineFactory.create(EngineMode.LIVE) 获取实例，
与沙盘模式零修改切换。

设计原则(来自 NautilusTrader LiveClient):
  1. 所有交易所交互通过ccxt统一API
  2. 订单状态通过WebSocket实时更新
  3. 错误处理: 网络重试 + 降级策略
  4. 风控前置: 实盘下单前强制风控检查

依赖:
  pip install ccxt
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from .matching_engine_interface import (
    MatchingEngineAdapter,
    UnifiedOrder,
    UnifiedOrderBook,
    UnifiedOrderSide,
    UnifiedOrderStatus,
    UnifiedOrderType,
    UnifiedPosition,
    UnifiedTicker,
    UnifiedAccount,
)

# Phase 7I-2: 网络弹性层 — 彻底解决"网络连接限制问题"
# 与 exchange_connector.py 共用 network_resilience 模块
try:
    from .network_resilience import (
        ExponentialBackoffRetry,
        MultiSourceDataFetcher,
        WeightManager,
    )
    _NETWORK_RESILIENCE_AVAILABLE = True
except ImportError:
    _NETWORK_RESILIENCE_AVAILABLE = False
    ExponentialBackoffRetry = None  # type: ignore
    WeightManager = None  # type: ignore
    MultiSourceDataFetcher = None  # type: ignore

logger = logging.getLogger("hermes.live_matching_engine")


class LiveMatchingEngine(MatchingEngineAdapter):
    """实盘撮合引擎 — 对接真实交易所

    通过ccxt库对接交易所(OKX/Binance/Bybit等)，
    实现与SandboxMatchingEngine相同的接口。

    用法:
      engine = LiveMatchingEngine(
          exchange="okx",
          api_key="xxx",
          api_secret="xxx",
      )
      # 策略代码与沙盘模式完全相同
      order = engine.submit_order(unified_order)
    """

    def __init__(
        self,
        exchange: str = "okx",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = False,
        default_leverage: int = 1,
        **kwargs: Any,
    ):
        """
        Args:
            exchange: 交易所名称(okx/binance/bybit等)
            api_key: API Key
            api_secret: API Secret
            passphrase: 交易所密码(OKX需要)
            testnet: 是否使用测试网
            default_leverage: 默认杠杆
        """
        # P0修复#3: super().__init__ 只接受 name 参数，不透传 kwargs
        super().__init__(name=f"live_{exchange}")

        self._exchange_name = exchange
        self._testnet = testnet
        self._default_leverage = default_leverage

        # P0修复#2: 初始化缓存属性，避免首次调用时 AttributeError
        self._market_cache: Dict[str, Any] = {}
        self._order_history: List[UnifiedOrder] = []

        # 延迟导入ccxt，避免未安装时影响沙盘模式
        try:
            import ccxt
        except ImportError as e:
            raise ImportError(
                f"实盘模式需要ccxt库: pip install ccxt. 错误: {e}"
            ) from e

        # 创建交易所实例
        exchange_class = getattr(ccxt, exchange)
        config: Dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
        if passphrase:
            config["password"] = passphrase

        self._exchange = exchange_class(config)

        if testnet:
            self._exchange.set_sandbox_mode(True)

        # Phase 7I-2: 网络弹性层组件初始化
        # 1. ExponentialBackoffRetry: 指数退避+jitter, 处理429/418/timeout/5xx
        # 2. WeightManager: 实时读取X-MBX-USED-WEIGHT-1M, >95%主动等待防IP封禁
        # 3. MultiSourceDataFetcher: 多源fallback链(公开数据专用)
        # 4. _latency_stats: API调用延迟监控(p50/p95/p99, 用于"订单执行速度优化")
        if _NETWORK_RESILIENCE_AVAILABLE:
            self._retry = ExponentialBackoffRetry(max_attempts=5)
            self._weight_manager = WeightManager()
            preferred_chain = [exchange] + [
                s for s in ["binance", "bybit", "okx", "coingecko", "cache"]
                if s != exchange
            ]
            self._data_fetcher = MultiSourceDataFetcher(source_chain=preferred_chain)
        else:
            self._retry = None
            self._weight_manager = None
            self._data_fetcher = None

        # 延迟监控: 每种API调用类型记录最近1000次延迟
        self._latency_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Phase 7J-7 反温室修复: 数据缓存 + stale 标志
        # 原因: API故障时返回0/空列表→下游误判→重复建仓/异常订单→爆仓
        # 修复: 缓存上次有效数据, 故障时返回缓存+stale标志, 下游检测stale拒绝下单
        self._last_valid_positions: Optional[List[UnifiedPosition]] = None
        self._last_valid_balance: Optional[Tuple[float, float, float]] = None
        self._last_valid_ticker: Dict[str, Dict[str, float]] = {}
        self._positions_stale: bool = False
        self._account_stale: bool = False
        self._ticker_stale: Dict[str, bool] = {}

        logger.info(
            "LiveMatchingEngine initialized: exchange=%s testnet=%s resilience=%s",
            exchange, testnet, _NETWORK_RESILIENCE_AVAILABLE,
        )

    # ========================================================================
    # Phase 7I-2: 网络弹性层 + 延迟监控
    # ========================================================================

    def _call_with_resilience(self, func: Callable, call_name: str = "", *args, **kwargs) -> Any:
        """Phase 7I-2: 包装交易所API调用, 应用 ExponentialBackoffRetry + WeightManager + 延迟监控

        三重保障 + 延迟追踪:
          1. 调用前: 权重节流检查(>95%主动等待, 防IP封禁)
          2. 调用中: 指数退避+jitter重试(429/418/timeout/5xx自动重试)
          3. 调用后: 从响应头更新权重 + 记录延迟到_latency_stats
          4. 延迟监控: 每次调用记录耗时, 用于p50/p95/p99统计(订单执行速度优化)

        Args:
            func: 要调用的函数(如self._exchange.fetch_ticker)
            call_name: API调用名称(用于延迟统计, 如"fetch_ticker"/"create_order")
            *args, **kwargs: 函数参数

        Returns:
            调用结果

        Raises:
            RuntimeError: 所有重试均失败
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")

        # 无弹性层时直接调用(降级模式)
        if self._retry is None or self._weight_manager is None:
            start = time.time()
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000.0
            if call_name:
                self._latency_stats[call_name].append(latency_ms)
            return result

        source_name = self._exchange_name

        # 1. 权重节流检查(防IP封禁)
        should_throttle, wait_s = self._weight_manager.should_throttle(source_name)
        if should_throttle:
            logger.info("[%s] 权重过高, 节流%.1fs防封禁", source_name, wait_s)
            time.sleep(min(wait_s, 30.0))

        # 2. 执行带重试(指数退避+jitter) + 延迟追踪
        start = time.time()
        success, data, error, attempts = self._retry.execute_with_retry(
            func, *args, **kwargs
        )
        latency_ms = (time.time() - start) * 1000.0

        # 记录延迟(含重试的总耗时)
        if call_name:
            self._latency_stats[call_name].append(latency_ms)

        # 3. 从响应头更新权重
        try:
            headers = getattr(self._exchange, "last_response_headers", None) or {}
            if headers:
                self._weight_manager.update_from_headers(source_name, headers)
        except Exception as e:
            # Phase 7J-10 反温室修复 MEDIUM #4: 权重更新失败静默吞没 (与 exchange_connector 对齐)
            # 之前: except: pass → 权重管理失效, 后续可能未节流导致 IP 封禁
            # 现在: warning 日志 (反温室: IP 封禁=实盘无法交易=策略失效)
            logger.warning(
                "update_from_headers %s FAILED: %s (权重管理可能失效, 后续可能未节流!)",
                source_name, str(e)[:100],
            )

        if not success:
            raise RuntimeError(
                "调用{}失败(尝试{}次, 耗时{:.0f}ms): {}".format(
                    call_name or "api", attempts, latency_ms, str(error)[:200]
                )
            )

        return data

    def get_resilience_status(self) -> Dict[str, Any]:
        """Phase 7I-2: 获取网络弹性层状态(用于监控/诊断)"""
        if not _NETWORK_RESILIENCE_AVAILABLE:
            return {"available": False, "reason": "network_resilience module not imported"}

        status: Dict[str, Any] = {
            "available": True,
            "exchange": self._exchange_name,
            "retry": {
                "max_attempts": self._retry.max_attempts if self._retry else 0,
                "base_backoff": self._retry.base_backoff if self._retry else 0,
                "max_backoff": self._retry.max_backoff if self._retry else 0,
            },
        }

        if self._weight_manager:
            weight_info = {}
            for source, usage in self._weight_manager.sources_weight.items():
                weight_info[source] = {
                    "used_weight_1m": usage.used_weight_1m,
                    "usage_ratio": round(usage.usage_ratio, 3),
                    "should_wait": usage.should_wait,
                }
            status["weight"] = weight_info

        if self._data_fetcher:
            status["data_fetcher"] = self._data_fetcher.get_health_report()

        return status

    def get_latency_report(self) -> Dict[str, Any]:
        """Phase 7I-2: 获取API调用延迟报告(订单执行速度优化)

        Phase 7J-9 反温室修复 HIGH #12: 添加端到端延迟估算
        之前: 只报告单点 API 延迟 (get_ticker/submit_order 各自统计)
              沙盘只看单点延迟, 实盘需端到端 (信号→下单→成交), 策略可能因总延迟超时失效
        现在: 添加 e2e_estimate_ms (所有 API 调用 avg_ms 之和, 端到端下限估计)
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 单点延迟OK≠端到端OK

        Returns:
            dict 每种API调用的 p50/p95/p99/avg/count 统计 + e2e_estimate_ms
            业界基准(2026): REST 50-500ms, WebSocket 5-50ms, Co-located 1-5ms
        """
        report: Dict[str, Any] = {}
        e2e_total_ms = 0.0
        for call_name, latencies in self._latency_stats.items():
            if not latencies:
                continue
            arr = sorted(latencies)
            n = len(arr)
            avg_ms = round(sum(arr) / n, 1)
            report[call_name] = {
                "count": n,
                "avg_ms": avg_ms,
                "p50_ms": round(arr[n // 2], 1),
                "p95_ms": round(arr[int(n * 0.95)] if n >= 20 else arr[-1], 1),
                "p99_ms": round(arr[int(n * 0.99)] if n >= 100 else arr[-1], 1),
                "min_ms": round(arr[0], 1),
                "max_ms": round(arr[-1], 1),
            }
            e2e_total_ms += avg_ms
        # Phase 7J-9: 端到端延迟估算 (所有 API 调用 avg 之和, 下限估计)
        if e2e_total_ms > 0:
            report["e2e_estimate_ms"] = round(e2e_total_ms, 1)
            report["e2e_note"] = "所有API调用avg之和(下限估计, 不含策略计算时间)"
        return report

    # ========================================================================
    # IMatchingEngine 接口实现
    # ========================================================================

    @property
    def engine_type(self) -> str:
        """P0修复#4: 实现 engine_type property"""
        return "live"

    def submit_order(self, order: UnifiedOrder) -> UnifiedOrder:
        """P0修复#1: 接口契约 — 接受 UnifiedOrder 参数，返回 UnifiedOrder

        Phase 7J-7 反温室修复 CRITICAL: 强制 client_order_id 实现幂等去重
        原因: 网络抖动导致 create_order 请求超时但交易所实际已收到→重试→
              交易所收到2笔相同订单→实盘仓位翻倍→爆仓
        修复: 入口强制生成 client_order_id, 重试时复用同一 ID 让交易所幂等去重
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 重复下单是日内合约致命风险

        Args:
            order: 统一订单数据结构

        Returns:
            更新后的 UnifiedOrder（含交易所返回的订单ID和状态）
        """
        try:
            ccxt_type = self._to_ccxt_order_type(order.order_type)
            ccxt_side = "buy" if order.side == UnifiedOrderSide.BUY else "sell"

            # Phase 7J-7 反温室修复: 强制生成 client_order_id 实现幂等去重
            # 格式: agent_id-symbol-timestamp-uuid8 (保证全局唯一 + 可追溯)
            # 重试时复用同一 client_order_id, 交易所识别为重复请求返回原订单
            if not order.client_order_id:
                order.client_order_id = "{}-{}-{}-{}".format(
                    order.agent_id or "unknown",
                    order.symbol.replace("/", "").replace(":", ""),
                    int(time.time() * 1000),
                    uuid.uuid4().hex[:8],
                )

            params: Dict[str, Any] = {
                "clientOrderId": order.client_order_id,  # 幂等去重关键
            }
            if order.leverage > 1:
                params["leverage"] = order.leverage
            if order.reduce_only:
                params["reduceOnly"] = True

            # G2修复 (LIVE-DATA-CONNECTOR): STOP_LOSS/TAKE_PROFIT 传递触发价
            # 之前: 条件单被降级为 market 单, 无触发价, 止损变市价 → 实盘爆仓
            # 修复: 显式传递 stopPrice, workingType=MARK_PRICE (防止插针误触发)
            is_stop_order = order.order_type in (UnifiedOrderType.STOP_LOSS, UnifiedOrderType.TAKE_PROFIT)
            if is_stop_order:
                trigger_price = order.stop_price if order.stop_price > 0 else order.price
                if trigger_price <= 0:
                    logger.error(
                        "G2修复: STOP订单缺少触发价 order_type=%s stop_price=%s price=%s → REJECTED",
                        order.order_type, order.stop_price, order.price,
                    )
                    order.status = UnifiedOrderStatus.REJECTED
                    self._record_stat("orders_rejected")
                    self._order_history.append(order)
                    return order
                params["stopPrice"] = trigger_price
                params["workingType"] = "MARK_PRICE"  # 用标记价格触发, 防插针
                params["reduceOnly"] = True  # 止损/止盈单强制仅减仓
                logger.info(
                    "G2修复: 提交条件单 %s trigger=%s stopPrice=%s",
                    order.order_type, trigger_price, params["stopPrice"],
                )

            # G2修复: LIMIT 单传 price, STOP 单不传 price (用 stopPrice), 其他类型不传
            if order.order_type == UnifiedOrderType.LIMIT:
                ccxt_price = order.price
            elif is_stop_order:
                ccxt_price = None  # STOP_MARKET 不需要 price, 用 stopPrice
            else:
                ccxt_price = None

            result = self._call_with_resilience(
                self._exchange.create_order, "create_order",
                symbol=order.symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=order.quantity,
                price=ccxt_price,
                params=params,
            )

            order.order_id = result.get("id", order.order_id)
            order.status = UnifiedOrderStatus.OPEN
            order.filled_qty = float(result.get("filled", 0))
            order.filled_price = float(result.get("average", 0))
            order.fee_paid = float(result.get("fee", {}).get("cost", 0))

            self._record_stat("orders_submitted")
            if order.filled_qty > 0:
                self._record_stat("orders_filled")

            logger.info(
                "Live order submitted: %s %s %s qty=%s price=%s -> id=%s",
                order.symbol, ccxt_side, ccxt_type, order.quantity,
                order.price, order.order_id,
            )

        except Exception as e:
            order.status = UnifiedOrderStatus.REJECTED
            self._record_stat("errors")
            self._record_stat("orders_rejected")
            logger.error("Live order failed: %s %s", order.symbol, e)

        self._order_history.append(order)
        return order

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """取消实盘订单"""
        try:
            self._call_with_resilience(
                self._exchange.cancel_order, "cancel_order", order_id, symbol or ""
            )
            self._record_stat("orders_cancelled")
            logger.info("Live order cancelled: %s", order_id)
            return True
        except Exception as e:
            self._record_stat("errors")
            logger.error("Cancel order failed: %s: %s", order_id, e)
            return False

    def get_order(self, order_id: str, symbol: str = None) -> Optional[UnifiedOrder]:
        """查询订单状态"""
        try:
            result = self._call_with_resilience(
                self._exchange.fetch_order, "fetch_order", order_id, symbol or ""
            )
            return self._convert_order(result)
        except Exception as e:
            logger.error("Fetch order failed: %s: %s", order_id, e)
            return None

    def get_open_orders(self, symbol: str = None) -> List[UnifiedOrder]:
        """获取未成交订单"""
        try:
            if symbol:
                results = self._call_with_resilience(
                    self._exchange.fetch_open_orders, "fetch_open_orders", symbol
                )
            else:
                results = self._call_with_resilience(
                    self._exchange.fetch_open_orders, "fetch_open_orders"
                )
            return [self._convert_order(r) for r in results]
        except Exception as e:
            logger.error("Fetch open orders failed: %s", e)
            return []

    def get_position(self, symbol: str) -> Optional[UnifiedPosition]:
        """查询单个持仓"""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_positions(self) -> List[UnifiedPosition]:
        """P0修复#4: 接口契约 — 返回 List[UnifiedPosition]"""
        try:
            positions = self._call_with_resilience(
                self._exchange.fetch_positions, "fetch_positions"
            )
            result: List[UnifiedPosition] = []
            for pos in positions:
                symbol = pos.get("symbol", "")
                if not symbol:
                    continue
                result.append(UnifiedPosition(
                    symbol=symbol,
                    side="long" if pos.get("side") == "long" else "short",
                    quantity=abs(float(pos.get("contracts", 0))),
                    entry_price=float(pos.get("entryPrice", 0)),
                    mark_price=float(pos.get("markPrice", 0)),
                    liquidation_price=float(pos.get("liquidationPrice", 0)),
                    leverage=int(pos.get("leverage", 1)),
                    margin=float(pos.get("initialMargin", 0)),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                    funding_rate=float(pos.get("fundingRate", 0)),
                ))
            # Phase 7J-7: 成功查询时更新缓存, 清除 stale 标志
            self._last_valid_positions = result
            self._positions_stale = False
            return result
        except Exception as e:
            # Phase 7J-7 反温室修复 CRITICAL: 持仓查询失败不再返回空列表
            # 原因: 空列表语义=无持仓, 下游策略/OMS误以为"无持仓"→重复建仓→爆仓
            # 合约杠杆5x下, 重复建仓=10x敞口, 反向波动10%即爆仓
            # 修复: 返回上一次有效持仓 + stale=True 标志, 下游检测到 stale 时拒绝下单
            # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 重复建仓是日内合约致命风险
            logger.error(
                "Fetch positions FAILED: %s — 返回上次缓存持仓(stale), 下游必须检测stale拒绝下单",
                str(e)[:200],
            )
            self._record_stat("positions_fetch_failures")
            if self._last_valid_positions is not None:
                # 返回缓存持仓, 标记 stale=True 让下游显式处理
                stale_positions: List[UnifiedPosition] = []
                for pos in self._last_valid_positions:
                    stale_pos = UnifiedPosition(
                        symbol=pos.symbol, side=pos.side, quantity=pos.quantity,
                        entry_price=pos.entry_price, mark_price=pos.mark_price,
                        liquidation_price=pos.liquidation_price, leverage=pos.leverage,
                        margin=pos.margin, unrealized_pnl=pos.unrealized_pnl,
                        funding_rate=pos.funding_rate,
                    )
                    stale_positions.append(stale_pos)
                # 用 extra 字段标记 stale (UnifiedPosition 无 stale 字段, 通过日志告知)
                logger.warning(
                    "返回 stale 持仓缓存 (%d positions) — 下游应调用 has_stale_positions() 检测",
                    len(stale_positions),
                )
                self._positions_stale = True
                return stale_positions
            # 无缓存: 返回空列表但设置 stale 标志 (区分"真无持仓"和"查询失败")
            self._positions_stale = True
            logger.warning("无缓存持仓, 返回空列表(stale=True) — 下游必须检测stale拒绝下单")
            return []

    def has_stale_positions(self) -> bool:
        """Phase 7J-7 反温室修复: 检测持仓数据是否过期 (查询失败时返回 True)

        下游策略/OMS 调用此方法, 若返回 True 则拒绝新建仓 (防重复建仓爆仓).
        """
        return getattr(self, "_positions_stale", False)

    def clear_positions_stale(self) -> None:
        """Phase 7J-7: 清除 stale 标志 (成功刷新持仓后调用)"""
        self._positions_stale = False

    def get_account(self) -> UnifiedAccount:
        """P0修复#4: 实现 get_account() 返回 UnifiedAccount"""
        try:
            balance = self._call_with_resilience(
                self._exchange.fetch_balance, "fetch_balance"
            )
            total = float(balance.get("total", {}).get("USDT", 0))
            free = float(balance.get("free", {}).get("USDT", 0))
            used = float(balance.get("used", {}).get("USDT", 0))
            # Phase 7J-7: 成功查询时清除 stale 标志
            self._account_stale = False
            self._last_valid_balance = (total, free, used)
            return UnifiedAccount(
                account_id=self._exchange_name,
                total_balance=total,
                available_balance=free,
                margin_used=used,
                margin_ratio=used / total if total > 0 else 0.0,
                unrealized_pnl=float(balance.get("info", {}).get("upl", 0)),
                realized_pnl=float(balance.get("info", {}).get("realisedPnl", 0)),
                positions=self.get_positions(),
            )
        except Exception as e:
            # Phase 7J-7 反温室修复 CRITICAL: 余额查询失败不再返回0
            # 原因: 返回0余额→策略以为账户清零→触发止损平仓→反向市价单→真实亏损
            #       或 position_size=0*kelly=0→跳过交易→错过机会
            # 修复: 返回上次缓存余额 + stale 标志, 下游检测 stale 时拒绝下单
            logger.error(
                "Fetch account FAILED: %s — 返回上次缓存余额(stale), 下游必须检测stale拒绝下单",
                str(e)[:200],
            )
            self._record_stat("account_fetch_failures")
            self._account_stale = True
            if hasattr(self, "_last_valid_balance") and self._last_valid_balance is not None:
                total, free, used = self._last_valid_balance
                logger.warning("返回 stale 余额缓存: total=%.2f free=%.2f", total, free)
                return UnifiedAccount(
                    account_id=self._exchange_name,
                    total_balance=total, available_balance=free,
                    margin_used=used,
                    margin_ratio=used / total if total > 0 else 0.0,
                    unrealized_pnl=0.0, realized_pnl=0.0,
                )
            # 无缓存: 返回0但标记 stale
            logger.warning("无缓存余额, 返回0(stale=True) — 下游必须检测stale拒绝下单")
            return UnifiedAccount(
                account_id=self._exchange_name,
                total_balance=0.0, available_balance=0.0,
                margin_used=0.0, margin_ratio=0.0,
                unrealized_pnl=0.0, realized_pnl=0.0,
            )

    def has_stale_account(self) -> bool:
        """Phase 7J-7 反温室修复: 检测账户余额数据是否过期 (查询失败时返回 True)"""
        return getattr(self, "_account_stale", False)

    def get_ticker(self, symbol: str) -> UnifiedTicker:
        """查询行情"""
        try:
            ticker = self._call_with_resilience(
                self._exchange.fetch_ticker, "fetch_ticker", symbol
            )
            last_price = float(ticker.get("last", 0))
            # Phase 7J-7 反温室修复 CRITICAL: 0价格检测
            # 原因: 行情API故障返回last=0→下游entry_price=0→position_size=inf→爆仓
            # 修复: 0价格时返回上次缓存价格 + stale标志, 下游检测stale拒绝下单
            if last_price <= 0:
                logger.error(
                    "get_ticker %s returned last=0 (API异常), 返回缓存价格(stale)",
                    symbol,
                )
                self._record_stat("ticker_zero_price")
                if symbol in self._last_valid_ticker:
                    cached = self._last_valid_ticker[symbol]
                    self._ticker_stale[symbol] = True
                    return UnifiedTicker(
                        symbol=symbol, bid=cached["bid"], ask=cached["ask"],
                        last=cached["last"], volume_24h=0, high_24h=0, low_24h=0,
                        funding_rate=cached.get("funding_rate", 0),
                        timestamp=time.time(),
                    )
                # 无缓存: 返回0但标记 stale
                self._ticker_stale[symbol] = True
                return UnifiedTicker(symbol=symbol, bid=0, ask=0, last=0, volume_24h=0,
                                     high_24h=0, low_24h=0)
            # 成功: 更新缓存, 清除 stale
            self._last_valid_ticker[symbol] = {
                "bid": float(ticker.get("bid", 0)),
                "ask": float(ticker.get("ask", 0)),
                "last": last_price,
                "funding_rate": float(ticker.get("fundingRate", 0)),
            }
            self._ticker_stale[symbol] = False
            return UnifiedTicker(
                symbol=symbol,
                bid=float(ticker.get("bid", 0)),
                ask=float(ticker.get("ask", 0)),
                last=last_price,
                volume_24h=float(ticker.get("baseVolume", 0)),
                high_24h=float(ticker.get("high", 0)),
                low_24h=float(ticker.get("low", 0)),
                funding_rate=float(ticker.get("fundingRate", 0)),
                timestamp=ticker.get("timestamp", time.time() * 1000) / 1000,
            )
        except Exception as e:
            # Phase 7J-7 反温室修复 CRITICAL: 行情查询失败不再返回0价格
            # 原因: 返回last=0→entry_price=0→position_size=inf→爆仓
            # 修复: 返回上次缓存价格 + stale标志, 下游检测stale拒绝下单
            logger.error(
                "Fetch ticker FAILED: %s: %s — 返回缓存价格(stale), 下游必须检测stale拒绝下单",
                symbol, str(e)[:150],
            )
            self._record_stat("ticker_failures")
            self._ticker_stale[symbol] = True
            if symbol in self._last_valid_ticker:
                cached = self._last_valid_ticker[symbol]
                logger.warning("返回 stale 行情缓存: %s last=%.2f", symbol, cached["last"])
                return UnifiedTicker(
                    symbol=symbol, bid=cached["bid"], ask=cached["ask"],
                    last=cached["last"], volume_24h=0, high_24h=0, low_24h=0,
                    funding_rate=cached.get("funding_rate", 0),
                    timestamp=time.time(),
                )
            # 无缓存: 返回0但标记 stale
            logger.warning("无缓存行情 %s, 返回0(stale=True) — 下游必须检测stale拒绝下单", symbol)
            return UnifiedTicker(symbol=symbol, bid=0, ask=0, last=0, volume_24h=0,
                                 high_24h=0, low_24h=0)

    def has_stale_ticker(self, symbol: str) -> bool:
        """Phase 7J-7 反温室修复: 检测行情数据是否过期 (查询失败或0价格时返回 True)"""
        return self._ticker_stale.get(symbol, False)

    def get_order_book(self, symbol: str, limit: int = 20) -> UnifiedOrderBook:
        """查询订单簿"""
        try:
            ob = self._call_with_resilience(
                self._exchange.fetch_order_book, "fetch_order_book", symbol, limit
            )
            bids = [(float(p), float(q)) for p, q in ob.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in ob.get("asks", [])]
            return UnifiedOrderBook(symbol=symbol, bids=bids, asks=asks)
        except Exception as e:
            logger.error("Fetch order book failed: %s: %s", symbol, e)
            return UnifiedOrderBook(symbol=symbol, bids=[], asks=[])

    def get_history(self, symbol: str, timeframe: str = "1h",
                   limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史K线

        Phase 7I-2: 优先使用MultiSourceDataFetcher多源fallback(公开数据)
        testnet模式用直连(测试网数据与主网不同)
        """
        # Phase 7I-2: 多源fallback(公开数据, 无需API key)
        if self._data_fetcher is not None and not self._testnet:
            try:
                result = self._data_fetcher.fetch_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    preferred_source=self._exchange_name,
                )
                if result.success and result.data:
                    logger.debug(
                        "get_history %s from %s (latency=%.0fms, cache=%s)",
                        symbol, result.source, result.latency_ms, result.from_cache,
                    )
                    return [
                        {"timestamp": row[0] / 1000, "open": row[1], "high": row[2],
                         "low": row[3], "close": row[4], "volume": row[5]}
                        for row in result.data
                    ]
                logger.warning(
                    "MultiSource fetch_klines failed: %s, fallback to direct",
                    result.error[:100] if result.error else "unknown",
                )
            except Exception as e:
                logger.warning(
                    "MultiSource fetch_klines exception: %s, fallback to direct",
                    str(e)[:100],
                )

        # 直连模式(retry+weight, 无多源fallback)
        try:
            ohlcv = self._call_with_resilience(
                self._exchange.fetch_ohlcv, "fetch_ohlcv",
                symbol, timeframe, limit=limit
            )
            return [
                {"timestamp": row[0] / 1000, "open": row[1], "high": row[2],
                 "low": row[3], "close": row[4], "volume": row[5]}
                for row in ohlcv
            ]
        except Exception as e:
            logger.error("Fetch history failed: %s: %s", symbol, e)
            return []

    # ========================================================================
    # 兼容方法（非接口，保留向后兼容）
    # ========================================================================

    def update_market(self, market_data: Any) -> None:
        """实盘模式: 行情通过WebSocket实时推送，此方法为兼容接口"""
        if market_data is not None:
            symbol = getattr(market_data, "symbol", None)
            if symbol:
                self._market_cache[symbol] = market_data

    def get_market(self, symbol: str) -> Optional[Any]:
        """获取缓存的实盘行情"""
        if symbol in self._market_cache:
            return self._market_cache[symbol]
        return self.get_ticker(symbol)

    def close_position(self, symbol: str) -> Optional[UnifiedOrder]:
        """平仓实盘持仓"""
        pos = self.get_position(symbol)
        if pos is None or pos.quantity == 0:
            return None

        close_side = UnifiedOrderSide.SELL if pos.side == "long" else UnifiedOrderSide.BUY
        order = UnifiedOrder(
            order_id=f"close-{int(time.time()*1000)}",
            agent_id="system",
            symbol=symbol,
            side=close_side,
            order_type=UnifiedOrderType.MARKET,
            quantity=pos.quantity,
            leverage=pos.leverage,
            reduce_only=True,
        )
        return self.submit_order(order)

    # ========================================================================
    # 内部工具方法
    # ========================================================================

    def _to_ccxt_order_type(self, order_type: UnifiedOrderType) -> str:
        """转换订单类型到ccxt格式

        G2修复 (LIVE-DATA-CONNECTOR): STOP_LOSS/TAKE_PROFIT 不再静默降级为 market
        之前: STOP_LOSS/TAKE_PROFIT 落入 default "market" → 止损变市价 → 实盘爆仓风险
        修复: 显式映射为条件单类型, 由 submit_order 传递 stopPrice 参数
        若交易所不支持条件单, ccxt 会抛出异常, 由上层 catch 记录 REJECTED
        """
        mapping = {
            UnifiedOrderType.MARKET: "market",
            UnifiedOrderType.LIMIT: "limit",
            UnifiedOrderType.POST_ONLY: "post_only",
            UnifiedOrderType.IOC: "ioc",
            UnifiedOrderType.FOK: "fok",
            # G2修复: 显式映射条件单, 不再 default 到 "market"
            UnifiedOrderType.STOP_LOSS: "STOP_MARKET",       # Binance Futures 风格
            UnifiedOrderType.TAKE_PROFIT: "TAKE_PROFIT_MARKET",
        }
        result = mapping.get(order_type)
        if result is None:
            # 未知类型显式报错, 杜绝静默降级
            logger.error(
                "G2修复: 未知订单类型 %s 拒绝降级为 market (原bug: 止损变市价导致爆仓)",
                order_type,
            )
            raise ValueError(f"Unsupported order type: {order_type} (G2: 拒绝静默降级)")
        return result

    def _convert_order(self, result: Dict[str, Any]) -> UnifiedOrder:
        """转换ccxt订单结果到UnifiedOrder"""
        side_str = result.get("side", "buy")
        type_str = result.get("type", "market")
        status_str = result.get("status", "open")

        status_map = {
            "open": UnifiedOrderStatus.OPEN,
            "closed": UnifiedOrderStatus.FILLED,
            "canceled": UnifiedOrderStatus.CANCELLED,
            "cancelled": UnifiedOrderStatus.CANCELLED,
            "expired": UnifiedOrderStatus.EXPIRED,
            "rejected": UnifiedOrderStatus.REJECTED,
        }

        return UnifiedOrder(
            order_id=str(result.get("id", "")),
            agent_id="live",
            symbol=result.get("symbol", ""),
            side=UnifiedOrderSide(side_str) if side_str in ("buy", "sell") else UnifiedOrderSide.BUY,
            # G2修复: 反向转换支持条件单类型 (stop_market/take_profit_market 等)
            order_type=self._ccxt_type_to_unified(type_str),
            quantity=float(result.get("amount", 0)),
            price=float(result.get("price", 0)),
            filled_qty=float(result.get("filled", 0)),
            filled_price=float(result.get("average", 0)),
            fee_paid=float(result.get("fee", {}).get("cost", 0)),
            status=status_map.get(status_str, UnifiedOrderStatus.PENDING),
            timestamp=result.get("timestamp", time.time() * 1000) / 1000 if result.get("timestamp") else time.time(),
        )

    def _ccxt_type_to_unified(self, type_str: str) -> UnifiedOrderType:
        """G2修复: ccxt订单类型反向转换, 支持条件单

        之前: 只识别 market/limit/post_only/ioc/fok, 其他全部 fallback MARKET
        修复: 识别 stop_market/take_profit_market/stop_loss/take_profit 等条件单类型
        """
        type_lower = (type_str or "").lower()
        direct_map = {
            "market": UnifiedOrderType.MARKET,
            "limit": UnifiedOrderType.LIMIT,
            "post_only": UnifiedOrderType.POST_ONLY,
            "ioc": UnifiedOrderType.IOC,
            "fok": UnifiedOrderType.FOK,
        }
        if type_lower in direct_map:
            return direct_map[type_lower]
        # G2修复: 条件单类型识别 (Binance/Gate/OKX 命名差异)
        if "stop_loss" in type_lower or "stop_market" in type_lower or type_lower == "stop":
            return UnifiedOrderType.STOP_LOSS
        if "take_profit" in type_lower or "take_profit_market" in type_lower:
            return UnifiedOrderType.TAKE_PROFIT
        # 未知类型不再默认 MARKET, 记录警告后 fallback (向后兼容)
        logger.warning("G2修复: 未知ccxt订单类型 %s fallback MARKET", type_str)
        return UnifiedOrderType.MARKET


__all__ = ["LiveMatchingEngine"]
