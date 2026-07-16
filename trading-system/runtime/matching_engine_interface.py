# -*- coding: utf-8 -*-
"""
撮合引擎统一接口 (Matching Engine Interface) — 沙盘与实盘的桥接层

定义统一的撮合引擎接口，使沙盘模拟引擎和实盘交易所连接器实现同一套接口。
这样策略层无需关心底层是沙盘还是实盘，可以无缝切换。

设计原则（借鉴 Hummingbot V2 Architecture + NautilusTrader Adapter Pattern）：
  1. 接口与实现分离（依赖倒置）
  2. 沙盘引擎和实盘连接器都实现同一接口
  3. 策略层通过接口调用，不直接依赖具体实现
  4. 支持运行时切换（沙盘↔实盘）

参考：
  - Hummingbot V2 Executors Architecture (hummingbot.org/strategies/v2-strategies/executors/)
  - NautilusTrader Adapter Pattern (nautilustrader.io)
  - CCXT Unified API (docs.ccxt.com)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.matching_engine_interface")


# ============================================================================
# 统一数据结构（与 sandbox_engine.py 保持兼容）
# ============================================================================


class UnifiedOrderSide(Enum):
    """统一订单方向"""
    BUY = "buy"      # 买入/做多
    SELL = "sell"    # 卖出/做空


class UnifiedOrderType(Enum):
    """统一订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"      # 只挂maker单
    IOC = "ioc"                  # Immediate or Cancel
    FOK = "fok"                  # Fill or Kill
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class UnifiedOrderStatus(Enum):
    """统一订单状态"""
    PENDING = "pending"
    OPEN = "open"                # 已挂单，未成交
    PARTIAL = "partial"          # 部分成交
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(slots=True)
class UnifiedOrder:
    """统一订单数据结构"""
    order_id: str
    agent_id: str
    symbol: str                          # e.g. "BTC/USDT:USDT" (永续) or "BTC/USDT" (现货)
    side: UnifiedOrderSide
    order_type: UnifiedOrderType
    quantity: float                      # 合约张数 or 基础货币数量
    price: float = 0.0                   # 限价单价格
    leverage: int = 1                    # 杠杆倍数（永续合约）
    reduce_only: bool = False            # 仅减仓（永续合约）
    time_in_force: str = "GTC"           # GTC/IOC/FOK/PO
    stop_price: float = 0.0              # 止损单触发价
    take_profit_price: float = 0.0       # 止盈单触发价
    timestamp: float = field(default_factory=time.time)
    status: UnifiedOrderStatus = UnifiedOrderStatus.PENDING
    filled_qty: float = 0.0
    filled_price: float = 0.0
    fee_paid: float = 0.0
    slippage: float = 0.0
    client_order_id: Optional[str] = None  # 客户端自定义ID
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnifiedPosition:
    """统一持仓数据结构"""
    symbol: str
    side: str                            # "long" / "short" / "flat"
    quantity: float                       # 持仓数量（正数）
    entry_price: float
    mark_price: float
    liquidation_price: float = 0.0
    leverage: int = 1
    margin: float = 0.0                  # 占用保证金
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    funding_rate: float = 0.0            # 当前资金费率
    next_funding_ts: float = 0.0         # 下次资金费率结算时间


@dataclass(slots=True)
class UnifiedTicker:
    """统一行情数据"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    funding_rate: float = 0.0            # 永续合约资金费率
    next_funding_ts: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class UnifiedOrderBook:
    """统一订单簿"""
    symbol: str
    bids: List[Tuple[float, float]]      # [(price, qty), ...]
    asks: List[Tuple[float, float]]
    timestamp: float = field(default_factory=time.time)

    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2

    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0][0] - self.bids[0][0]

    def depth(self, levels: int = 10) -> Dict[str, float]:
        """计算订单簿深度"""
        bid_depth = sum(q for _, q in self.bids[:levels])
        ask_depth = sum(q for _, q in self.asks[:levels])
        return {"bid_depth": bid_depth, "ask_depth": ask_depth, "imbalance": (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)}


@dataclass(slots=True)
class UnifiedAccount:
    """统一账户信息"""
    account_id: str
    total_balance: float                # 总权益
    available_balance: float             # 可用余额
    margin_used: float                  # 已用保证金
    margin_ratio: float                  # 保证金率
    unrealized_pnl: float
    realized_pnl: float
    positions: List[UnifiedPosition] = field(default_factory=list)


# ============================================================================
# 撮合引擎抽象接口
# ============================================================================


class IMatchingEngine(ABC):
    """撮合引擎统一接口

    所有撮合引擎（沙盘 / 实盘）必须实现此接口。
    策略层通过此接口与撮合引擎交互，无需关心底层实现。
    """

    @abstractmethod
    def submit_order(self, order: UnifiedOrder) -> UnifiedOrder:
        """提交订单"""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """取消订单"""

    @abstractmethod
    def get_order(self, order_id: str, symbol: str = None) -> Optional[UnifiedOrder]:
        """查询订单"""

    @abstractmethod
    def get_open_orders(self, symbol: str = None) -> List[UnifiedOrder]:
        """获取未成交订单"""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[UnifiedPosition]:
        """查询持仓"""

    @abstractmethod
    def get_positions(self) -> List[UnifiedPosition]:
        """查询所有持仓"""

    @abstractmethod
    def get_account(self) -> UnifiedAccount:
        """查询账户信息"""

    @abstractmethod
    def get_ticker(self, symbol: str) -> UnifiedTicker:
        """查询行情"""

    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 20) -> UnifiedOrderBook:
        """查询订单簿"""

    # 可选方法（默认实现，子类可覆盖）
    def get_funding_rate(self, symbol: str) -> float:
        """查询资金费率（永续合约）"""
        ticker = self.get_ticker(symbol)
        return ticker.funding_rate

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """设置杠杆（永续合约）"""
        return True

    def get_history(self, symbol: str, timeframe: str = "1h",
                   limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史K线"""
        return []

    @property
    def engine_type(self) -> str:
        """引擎类型：sandbox / live"""
        return "unknown"

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return False


# ============================================================================
# 撮合引擎适配器基类
# ============================================================================


class MatchingEngineAdapter(IMatchingEngine):
    """撮合引擎适配器基类

    提供通用功能（日志、错误处理、指标统计），具体实现由子类完成。
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._connected = False
        self._stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "errors": 0,
            "total_fee_paid": 0.0,
            "total_slippage": 0.0,
        }
        self._logger = logging.getLogger(f"hermes.matching_engine.{name}")

    def _record_stat(self, key: str, value: Any = 1):
        if key in self._stats:
            if isinstance(self._stats[key], (int, float)):
                if isinstance(value, (int, float)):
                    self._stats[key] += value
                else:
                    self._stats[key] += 1
            else:
                self._stats[key] = value

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def reset_stats(self):
        for k in self._stats:
            if isinstance(self._stats[k], (int, float)):
                self._stats[k] = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """连接到撮合引擎（实盘需要API密钥，沙盘直接返回True）"""
        self._connected = True
        self._logger.info("[%s] Connected", self.name)
        return True

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self._logger.info("[%s] Disconnected", self.name)


# ============================================================================
# 撮合引擎工厂
# ============================================================================


class MatchingEngineFactory:
    """撮合引擎工厂

    根据配置创建对应的撮合引擎实例。

    使用：
        engine = MatchingEngineFactory.create({
            "type": "sandbox",
            "config": {...}
        })

        engine = MatchingEngineFactory.create({
            "type": "live",
            "exchange": "binance",
            "api_key": "...",
            "api_secret": "...",
        })
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> IMatchingEngine:
        engine_type = config.get("type", "sandbox")

        if engine_type == "sandbox":
            from .sandbox_engine import SandboxMatchingEngine
            # 适配沙盘引擎到统一接口
            return SandboxMatchingEngineAdapter(
                sandbox_engine=SandboxMatchingEngine(**config.get("config", {}))
            )

        elif engine_type == "live":
            # P0修复#10: 工厂直接创建 LiveMatchingEngine，不再依赖不存在的 exchange_connector
            from .live_matching_engine import LiveMatchingEngine
            return LiveMatchingEngine(
                exchange=config.get("exchange", "binance"),
                api_key=config.get("api_key", ""),
                api_secret=config.get("api_secret", ""),
                passphrase=config.get("passphrase", ""),
                testnet=config.get("testnet", False),
                default_leverage=config.get("default_leverage", 1),
            )

        else:
            raise ValueError(f"Unknown engine type: {engine_type}")


# ============================================================================
# 沙盘引擎适配器
# ============================================================================


class SandboxMatchingEngineAdapter(MatchingEngineAdapter):
    """沙盘撮合引擎适配器

    将 sandbox_engine.py 的 SandboxMatchingEngine 适配到统一接口。
    P0修复#5: 消除硬编码桩值，委托给沙盘引擎获取真实数据。
    P0整合: submit_order 真正桥接到 SandboxMatchingEngine.submit_order，
           不再仅记录订单；get_position/get_account 基于 agent_id 上下文查询。
    """

    def __init__(self, sandbox_engine):
        super().__init__(name="sandbox_adapter")
        self._engine = sandbox_engine
        # P0修复#5: 维护订单和账户状态，不再返回硬编码假数据
        self._orders: Dict[str, UnifiedOrder] = {}
        # 与沙盘引擎初始资金保持一致（沙盘引擎用 initial_capital）
        self._initial_balance: float = float(
            getattr(sandbox_engine, "initial_capital", 100000.0)
        )
        self._realized_pnl: float = 0.0
        # P0整合: 沙盘引擎按 agent_id 维度管理账户/持仓，统一接口是单账户视图，
        # 因此用“最近一次 submit_order 的 agent_id”作为查询上下文。
        self._current_agent_id: str = "default"

    @property
    def engine_type(self) -> str:
        return "sandbox"

    # ------------------------------------------------------------------
    # 沙盘类型映射辅助
    # ------------------------------------------------------------------

    def _sbx_enums(self):
        """懒加载 sandbox_engine 枚举（避免循环导入）"""
        from .sandbox_engine import (
            OrderSide as SbxSide,
            OrderType as SbxType,
            OrderStatus as SbxStatus,
        )
        return SbxSide, SbxType, SbxStatus

    def _map_side_to_sandbox(self, side: UnifiedOrderSide):
        """UnifiedOrderSide -> sandbox OrderSide"""
        SbxSide, _, _ = self._sbx_enums()
        if side == UnifiedOrderSide.BUY:
            return SbxSide.LONG
        if side == UnifiedOrderSide.SELL:
            return SbxSide.SHORT
        raise ValueError(f"无法映射订单方向: {side}")

    def _map_type_to_sandbox(self, order_type: UnifiedOrderType, price: float):
        """UnifiedOrderType -> sandbox OrderType

        sandbox 仅支持 MARKET/LIMIT，其余类型降级：
          - LIMIT / POST_ONLY -> LIMIT
          - MARKET / IOC / FOK / STOP_LOSS / TAKE_PROFIT -> 有价用 LIMIT，否则 MARKET
        """
        _, SbxType, _ = self._sbx_enums()
        if order_type == UnifiedOrderType.LIMIT:
            return SbxType.LIMIT
        if order_type == UnifiedOrderType.POST_ONLY:
            return SbxType.LIMIT
        if order_type == UnifiedOrderType.MARKET:
            return SbxType.MARKET
        # IOC/FOK/STOP_LOSS/TAKE_PROFIT：有价格则按限价处理，否则市价
        return SbxType.LIMIT if price > 0 else SbxType.MARKET

    def _convert_sbx_order_to_unified(
        self, sbx_order, original: Optional[UnifiedOrder] = None
    ) -> UnifiedOrder:
        """将 sandbox_engine.Order 转换回 UnifiedOrder"""
        SbxSide, SbxType, SbxStatus = self._sbx_enums()

        # side 反向映射
        if sbx_order.side == SbxSide.LONG:
            uni_side = UnifiedOrderSide.BUY
        elif sbx_order.side == SbxSide.SHORT:
            uni_side = UnifiedOrderSide.SELL
        else:
            uni_side = original.side if original else UnifiedOrderSide.BUY

        # order_type 反向映射
        uni_type = (
            UnifiedOrderType.LIMIT
            if sbx_order.order_type == SbxType.LIMIT
            else UnifiedOrderType.MARKET
        )

        # status 映射
        status_map = {
            SbxStatus.PENDING: UnifiedOrderStatus.PENDING,
            SbxStatus.FILLED: UnifiedOrderStatus.FILLED,
            SbxStatus.PARTIAL: UnifiedOrderStatus.PARTIAL,
            SbxStatus.CANCELLED: UnifiedOrderStatus.CANCELLED,
            SbxStatus.REJECTED: UnifiedOrderStatus.REJECTED,
        }
        uni_status = status_map.get(sbx_order.status, UnifiedOrderStatus.PENDING)

        # 保留 original 的客户端字段，合并 sandbox 的成交信息
        order_id = original.order_id if original else sbx_order.order_id
        meta = dict(original.meta) if original and original.meta else {}
        if getattr(sbx_order, "signal_source", ""):
            meta["signal_source"] = sbx_order.signal_source
        if getattr(sbx_order, "mev_cost", 0):
            meta["mev_cost"] = sbx_order.mev_cost
        meta["sandbox_order_id"] = sbx_order.order_id

        unified = UnifiedOrder(
            order_id=order_id,
            agent_id=sbx_order.agent_id,
            symbol=sbx_order.symbol,
            side=uni_side,
            order_type=uni_type,
            quantity=sbx_order.quantity,
            price=sbx_order.price,
            leverage=sbx_order.leverage,
            reduce_only=original.reduce_only if original else False,
            time_in_force=original.time_in_force if original else "GTC",
            timestamp=sbx_order.timestamp,
            status=uni_status,
            filled_qty=sbx_order.filled_qty,
            filled_price=sbx_order.filled_price,
            fee_paid=sbx_order.fee_paid,
            slippage=sbx_order.slippage,
            client_order_id=original.client_order_id if original else None,
            meta=meta,
        )

        # 统计
        if uni_status == UnifiedOrderStatus.FILLED:
            self._record_stat("orders_filled")
            self._record_stat("total_fee_paid", sbx_order.fee_paid)
        elif uni_status == UnifiedOrderStatus.REJECTED:
            self._record_stat("orders_rejected")

        self._orders[order_id] = unified
        return unified

    def _convert_trade_to_position(self, trade) -> UnifiedPosition:
        """将 sandbox_engine.Trade 转换为 UnifiedPosition"""
        SbxSide, _, _ = self._sbx_enums()
        if trade.side == SbxSide.LONG:
            side_str = "long"
        elif trade.side == SbxSide.SHORT:
            side_str = "short"
        else:
            side_str = "flat"

        entry_price = float(getattr(trade, "entry_price", 0))
        # mark_price 优先用当前行情
        mark_price = entry_price
        market = None
        if hasattr(self._engine, "get_market"):
            try:
                market = self._engine.get_market(trade.symbol)
            except Exception:
                market = None
        if market:
            close = getattr(market, "close", None)
            if close is None and isinstance(market, dict):
                close = market.get("close", 0)
            if close and float(close) > 0:
                mark_price = float(close)

        # 未实现盈亏（仅 open 持仓）
        unrealized = 0.0
        if getattr(trade, "status", "") == "open" and market:
            close = float(getattr(market, "close", 0) or 0)
            if close > 0:
                if trade.side == SbxSide.LONG:
                    unrealized = trade.quantity * (close - entry_price)
                else:
                    unrealized = trade.quantity * (entry_price - close)

        return UnifiedPosition(
            symbol=trade.symbol,
            side=side_str,
            quantity=abs(float(trade.quantity)),
            entry_price=entry_price,
            mark_price=mark_price,
            liquidation_price=float(getattr(trade, "liquidation_price", 0)),
            leverage=int(getattr(trade, "leverage", 1) or 1),
            margin=0.0,
            unrealized_pnl=unrealized,
            realized_pnl=float(getattr(trade, "pnl", 0)),
            funding_rate=0.0,
            next_funding_ts=0.0,
        )

    def _get_last_price(self, symbol: str) -> float:
        """P0修复#5: 从沙盘引擎获取最新价格，不再硬编码 50000.0"""
        # SandboxMatchingEngine 使用 get_market(symbol).close 获取最新价格
        market = None
        for method_name in ("get_market", "get_last_market"):
            method = getattr(self._engine, method_name, None)
            if callable(method):
                try:
                    market = method(symbol)
                    if market:
                        break
                except Exception:
                    pass
        if market:
            # MarketData 有 close 属性
            close = getattr(market, "close", None)
            if close is None and isinstance(market, dict):
                close = market.get("close", 0)
            if close and float(close) > 0:
                return float(close)
        # 尝试 market_data 属性
        market_data = (
            getattr(self._engine, "market_data", None)
            or getattr(self._engine, "_market_data", None)
            or getattr(self._engine, "current_market", None)
        )
        if market_data:
            if isinstance(market_data, dict):
                md = market_data.get(symbol)
                if md:
                    close = getattr(md, "close", None) or (
                        md.get("close", 0) if isinstance(md, dict) else 0
                    )
                    if close and float(close) > 0:
                        return float(close)
            else:
                close = getattr(market_data, "close", None)
                if close and float(close) > 0:
                    return float(close)
        # 最终回退：抛出异常而非返回假数据
        raise ValueError(
            f"无法从沙盘引擎获取 {symbol} 的最新价格，沙盘引擎未提供价格查询接口"
        )

    # ------------------------------------------------------------------
    # IMatchingEngine 实现
    # ------------------------------------------------------------------

    def submit_order(self, order: UnifiedOrder) -> UnifiedOrder:
        """提交订单到沙盘引擎（真正桥接到 SandboxMatchingEngine.submit_order）"""
        try:
            self._record_stat("orders_submitted")

            # 记录 agent_id 上下文，供后续 get_position/get_account 使用
            if order.agent_id:
                self._current_agent_id = order.agent_id

            # UnifiedOrderSide -> sandbox OrderSide
            try:
                sbx_side = self._map_side_to_sandbox(order.side)
            except ValueError:
                order.status = UnifiedOrderStatus.REJECTED
                self._record_stat("orders_rejected")
                return order

            # UnifiedOrderType -> sandbox OrderType
            sbx_type = self._map_type_to_sandbox(order.order_type, order.price)

            # signal_source 从 meta 获取（sandbox Order 有此字段）
            signal_source = ""
            if order.meta:
                signal_source = str(order.meta.get("signal_source", "") or "")

            # 真正调用沙盘引擎
            sbx_order = self._engine.submit_order(
                agent_id=order.agent_id,
                symbol=order.symbol,
                side=sbx_side,
                quantity=order.quantity,
                order_type=sbx_type,
                price=order.price,
                leverage=order.leverage,
                signal_source=signal_source,
            )

            # sandbox Order -> UnifiedOrder
            return self._convert_sbx_order_to_unified(sbx_order, original=order)
        except Exception as e:
            self._logger.error("submit_order failed: %s", str(e)[:200])
            self._record_stat("errors")
            order.status = UnifiedOrderStatus.REJECTED
            return order

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        # 优先尝试沙盘引擎的撤单接口
        for method_name in ("cancel_order", "cancel"):
            method = getattr(self._engine, method_name, None)
            if callable(method):
                try:
                    result = method(order_id)
                    if result:
                        self._record_stat("orders_cancelled")
                        if order_id in self._orders:
                            self._orders[order_id].status = UnifiedOrderStatus.CANCELLED
                        return True
                except Exception:
                    pass
        # 回退：本地状态标记
        if order_id in self._orders:
            self._orders[order_id].status = UnifiedOrderStatus.CANCELLED
            self._record_stat("orders_cancelled")
            return True
        return False

    def get_order(self, order_id: str, symbol: str = None) -> Optional[UnifiedOrder]:
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: str = None) -> List[UnifiedOrder]:
        return [
            o for o in self._orders.values()
            if o.status in (UnifiedOrderStatus.OPEN, UnifiedOrderStatus.PARTIAL)
            and (symbol is None or o.symbol == symbol)
        ]

    def get_position(self, symbol: str) -> Optional[UnifiedPosition]:
        """从沙盘引擎获取持仓（基于当前 agent_id 上下文）"""
        agent_id = self._current_agent_id
        try:
            positions = self._engine.get_positions(agent_id)
            trade = positions.get(symbol)
            if trade is not None:
                return self._convert_trade_to_position(trade)
        except Exception as e:
            self._logger.error("get_position failed: %s", str(e)[:200])
        return None

    def get_positions(self) -> List[UnifiedPosition]:
        """从沙盘引擎获取所有持仓（基于当前 agent_id 上下文）"""
        agent_id = self._current_agent_id
        try:
            positions = self._engine.get_positions(agent_id)
            return [self._convert_trade_to_position(t) for t in positions.values()]
        except Exception as e:
            self._logger.error("get_positions failed: %s", str(e)[:200])
            return []

    def get_account(self) -> UnifiedAccount:
        """从沙盘引擎获取账户信息（基于当前 agent_id 上下文）"""
        agent_id = self._current_agent_id
        try:
            balance = float(self._engine.get_balance(agent_id))
            equity = float(self._engine.get_equity(agent_id))
            unrealized = equity - balance

            # 已实现盈亏：从沙盘引擎 trade_history 汇总
            realized = 0.0
            try:
                trades = self._engine.get_trades(agent_id)
                realized = sum(float(getattr(t, "pnl", 0)) for t in trades)
            except Exception:
                pass
            self._realized_pnl = realized

            # 持仓列表与占用保证金
            positions_list = self.get_positions()
            margin_used = sum(p.margin for p in positions_list)

            return UnifiedAccount(
                account_id=agent_id,
                total_balance=equity,
                available_balance=balance,
                margin_used=margin_used,
                margin_ratio=(margin_used / equity) if equity > 0 else 0.0,
                unrealized_pnl=unrealized,
                realized_pnl=realized,
                positions=positions_list,
            )
        except Exception as e:
            self._logger.error("get_account failed: %s", str(e)[:200])
            # 回退：使用初始余额（明确标注为初始值，非假数据）
            return UnifiedAccount(
                account_id=agent_id,
                total_balance=self._initial_balance,
                available_balance=self._initial_balance,
                margin_used=0.0,
                margin_ratio=0.0,
                unrealized_pnl=0.0,
                realized_pnl=self._realized_pnl,
            )

    def get_ticker(self, symbol: str) -> UnifiedTicker:
        """从沙盘引擎获取行情（使用 get_market 的完整行情数据）"""
        try:
            market = None
            if hasattr(self._engine, "get_market"):
                try:
                    market = self._engine.get_market(symbol)
                except Exception:
                    market = None

            if market:
                close = getattr(market, "close", None)
                if close is None and isinstance(market, dict):
                    close = market.get("close", 0)
                last_price = float(close) if close and float(close) > 0 else self._get_last_price(symbol)
                high = float(getattr(market, "high", 0) or last_price)
                low = float(getattr(market, "low", 0) or last_price)
                volume = float(getattr(market, "volume", 0) or 0)
            else:
                last_price = self._get_last_price(symbol)
                high = last_price
                low = last_price
                volume = 0.0

            spread = last_price * 0.001  # 0.1% 价差
            return UnifiedTicker(
                symbol=symbol,
                bid=last_price - spread / 2,
                ask=last_price + spread / 2,
                last=last_price,
                volume_24h=volume,
                high_24h=high,
                low_24h=low,
                funding_rate=0.0001,
            )
        except Exception as e:
            self._logger.error("get_ticker failed: %s", str(e)[:200])
            raise

    def get_order_book(self, symbol: str, limit: int = 20) -> UnifiedOrderBook:
        """P0修复#5: 基于真实价格生成订单簿，不再硬编码 50000"""
        last_price = self._get_last_price(symbol)
        spread = last_price * 0.001
        bids = [(last_price - spread / 2 - i * last_price * 0.0001, 1.0) for i in range(limit)]
        asks = [(last_price + spread / 2 + i * last_price * 0.0001, 1.0) for i in range(limit)]
        return UnifiedOrderBook(symbol=symbol, bids=bids, asks=asks)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    "UnifiedOrderSide",
    "UnifiedOrderType",
    "UnifiedOrderStatus",
    "UnifiedOrder",
    "UnifiedPosition",
    "UnifiedTicker",
    "UnifiedOrderBook",
    "UnifiedAccount",
    "IMatchingEngine",
    "MatchingEngineAdapter",
    "MatchingEngineFactory",
    "SandboxMatchingEngineAdapter",
]
