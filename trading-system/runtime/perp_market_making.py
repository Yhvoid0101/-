# -*- coding: utf-8 -*-
"""
永续合约做市模块 (Perpetual Market Making) — Avellaneda-Stoikov做市策略

全网顶级成熟合约交易技术 #2：在永续合约订单簿两侧挂限价单，赚取买卖价差+资金费率

核心原理（来源：Hummingbot PMM、Avellaneda-Stoikov 2008、Hyperliquid Bot 2026）：
  1. 在买一价和卖一价两侧挂限价单（maker单）
  2. 当两侧都成交时，赚取买卖价差（spread）
  3. 同时收取永续合约资金费率（如果持仓为空方）
  4. 库存管理：偏离中性时调整报价（Avellaneda-Stoikov模型）

Avellaneda-Stoikov模型核心公式：
  - 保留价（Reservation Price）：
    r(t) = s - q * γ * σ² * (T - t)
    其中 s=中间价, q=库存, γ=风险厌恶系数, σ=波动率, T=时间窗口
  - 最优报价价差：
    δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
    其中 k=订单到达率参数

策略变体：
  - 纯做市（Pure MM）：仅赚取价差
  - 永续做市（Perp MM）：价差+资金费率（推荐）
  - 跨所做市（XEMM）：A所买入，B所卖出（库存对冲）

风险控制：
  - 库存风险：偏离中性时调整报价偏移
  - 逆向选择风险：被信息交易者套利
  - 资金费率反转风险
  - 清算风险

参考：
  - Hummingbot perpetual_market_making (hummingbot.org/strategies/v1-strategies/perpetual-market-making/)
  - Avellaneda & Stoikov 2008 "High-frequency trading in a limit order book"
  - Hyperliquid Bot Automated Trading (dailycoin.com/hyperliquid-bot-automated-trading)
  - Hummingbot Avellaneda Market Making (hummingbot.org/strategies/v1-strategies/avellaneda-market-making/)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("hermes.perp_market_making")


# ============================================================================
# 数据结构
# ============================================================================


class MMStatus(Enum):
    """做市状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class OrderSide(Enum):
    BID = "bid"   # 买单
    ASK = "ask"   # 卖单


@dataclass(slots=True)
class MMConfig:
    """做市配置"""
    symbol: str
    spread: float = 0.001                    # 报价价差（0.1%）
    order_amount: float = 0.01               # 单笔订单数量
    max_inventory: float = 0.1               # 最大库存（基础货币）
    inventory_skew_factor: float = 1.0       # 库存偏斜因子
    leverage: int = 3                         # 杠杆倍数
    # Avellaneda-Stoikov参数
    gamma: float = 0.1                        # 风险厌恶系数
    sigma_window: int = 60                    # 波动率计算窗口（K线数）
    T: float = 1.0                            # 时间窗口（天）
    k: float = 1.5                            # 订单到达率参数
    # 资金费率
    collect_funding: bool = True              # 是否收取资金费率
    min_funding_rate: float = 0.00005         # 最低费率阈值
    # 风控
    max_drawdown: float = 0.05                # 最大回撤（5%）
    stop_loss_threshold: float = 0.02         # 止损阈值（2%）
    refresh_interval: float = 5.0             # 报价刷新间隔（秒）


@dataclass(slots=True)
class MMQuote:
    """做市报价"""
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    mid_price: float
    spread: float
    inventory: float                          # 当前库存（正数=多头，负数=空头）
    reservation_price: float                  # 保留价（Avellaneda-Stoikov）
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class MMStats:
    """做市统计"""
    orders_placed: int = 0
    orders_filled_bid: int = 0
    orders_filled_ask: int = 0
    orders_cancelled: int = 0
    spread_captured: float = 0.0              # 累计价差收益
    funding_collected: float = 0.0            # 累计资金费率
    inventory_pnl: float = 0.0                # 库存盈亏
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    uptime_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)


# ============================================================================
# Avellaneda-Stoikov 做市引擎
# ============================================================================


class PerpetualMarketMaker:
    """永续合约做市引擎

    核心流程：
      1. 计算保留价（Reservation Price）— 考虑库存偏移
      2. 计算最优报价价差 — Avellaneda-Stoikov公式
      3. 挂买卖限价单 — maker单
      4. 监控成交 — 更新库存
      5. 库存管理 — 偏离中性时调整报价
      6. 资金费率收取 — 每8小时结算

    使用：
        mm = PerpetualMarketMaker(
            engine=engine,
            config=MMConfig(
                symbol="BTC/USDT:USDT",
                spread=0.001,
                order_amount=0.01,
                max_inventory=0.1,
                leverage=3,
            )
        )
        mm.start()
        # 每5秒刷新报价
        while mm.is_running():
            quote = mm.refresh_quotes()
            time.sleep(5)
    """

    def __init__(self, engine=None, config: MMConfig = None):
        self.engine = engine
        self.config = config or MMConfig(symbol="BTC/USDT:USDT")
        self.status = MMStatus.IDLE
        self.inventory = 0.0                          # 当前库存（基础货币）
        self.realized_pnl = 0.0
        self._stats = MMStats()
        self._price_history: deque = deque(maxlen=self.config.sigma_window * 10)
        self._active_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order_info
        self._funding_collected_total = 0.0

    # ------------------------------------------------------------------
    # 启动/停止
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """启动做市"""
        if self.status == MMStatus.RUNNING:
            return True
        self.status = MMStatus.RUNNING
        self._stats = MMStats()
        logger.info(
            "MM started: %s spread=%.4f%% amount=%.4f max_inv=%.4f lev=%dx",
            self.config.symbol, self.config.spread * 100,
            self.config.order_amount, self.config.max_inventory, self.config.leverage,
        )
        return True

    def stop(self, reason: str = "manual"):
        """停止做市"""
        self.status = MMStatus.STOPPED
        self._cancel_all_orders()
        logger.info("MM stopped: %s reason=%s", self.config.symbol, reason)

    def pause(self):
        """暂停做市"""
        self.status = MMStatus.PAUSED
        self._cancel_all_orders()

    def resume(self):
        """恢复做市"""
        if self.status == MMStatus.PAUSED:
            self.status = MMStatus.RUNNING

    def is_running(self) -> bool:
        return self.status == MMStatus.RUNNING

    # ------------------------------------------------------------------
    # Phase 1: 计算保留价（Avellaneda-Stoikov）
    # ------------------------------------------------------------------

    def _calculate_reservation_price(self, mid_price: float,
                                     volatility: float,
                                     time_remaining: float = None) -> float:
        """计算保留价

        r(t) = s - q * γ * σ² * (T - t)

        Args:
            mid_price: 中间价
            volatility: 波动率
            time_remaining: 剩余时间（默认=T）

        Returns:
            保留价
        """
        if time_remaining is None:
            time_remaining = self.config.T

        # q = inventory / max_inventory (归一化到[-1, 1])
        q = max(-1.0, min(1.0, self.inventory / max(self.config.max_inventory, 1e-9)))

        # 保留价偏移
        reservation_offset = -q * self.config.gamma * (volatility ** 2) * time_remaining

        return mid_price * (1 + reservation_offset)

    def _calculate_optimal_spread(self, volatility: float,
                                  time_remaining: float = None) -> float:
        """计算最优报价价差（Avellaneda-Stoikov）

        δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
        """
        if time_remaining is None:
            time_remaining = self.config.T

        term1 = self.config.gamma * (volatility ** 2) * time_remaining
        term2 = (2 / self.config.gamma) * math.log(1 + self.config.gamma / self.config.k)
        return term1 + term2

    # ------------------------------------------------------------------
    # Phase 2: 计算报价
    # ------------------------------------------------------------------

    def calculate_quote(self) -> Optional[MMQuote]:
        """计算做市报价"""
        if not self.engine or not self.is_running():
            return None

        try:
            # 获取订单簿
            order_book = self.engine.get_order_book(self.config.symbol, limit=20)
            if not order_book.bids or not order_book.asks:
                return None

            mid_price = order_book.mid_price()
            spread = order_book.spread()

            # 更新价格历史
            self._price_history.append(mid_price)

            # 计算波动率
            volatility = self._calculate_volatility()

            # 计算保留价
            reservation = self._calculate_reservation_price(mid_price, volatility)

            # 计算最优价差
            optimal_spread = self._calculate_optimal_spread(volatility)

            # 实际价差 = max(配置价差, 最优价差, 市场价差)
            actual_spread = max(self.config.spread, optimal_spread, spread / mid_price)

            # 计算买卖价
            bid_price = reservation * (1 - actual_spread / 2)
            ask_price = reservation * (1 + actual_spread / 2)

            # 库存偏斜：偏离中性时调整订单数量
            inventory_ratio = self.inventory / max(self.config.max_inventory, 1e-9)
            bid_qty = self.config.order_amount * (1 + inventory_ratio * self.config.inventory_skew_factor)
            ask_qty = self.config.order_amount * (1 - inventory_ratio * self.config.inventory_skew_factor)

            # 确保数量为正
            bid_qty = max(0.001, bid_qty)
            ask_qty = max(0.001, ask_qty)

            return MMQuote(
                bid_price=bid_price,
                ask_price=ask_price,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                mid_price=mid_price,
                spread=actual_spread,
                inventory=self.inventory,
                reservation_price=reservation,
            )
        except Exception as e:
            logger.error("calculate_quote failed: %s", str(e)[:200])
            return None

    def _calculate_volatility(self) -> float:
        """计算波动率（基于价格历史）"""
        if len(self._price_history) < 10:
            return 0.02  # 默认2%波动率

        prices = np.array(list(self._price_history))
        returns = np.diff(np.log(prices))
        return float(np.std(returns) * math.sqrt(365))  # 年化波动率

    # ------------------------------------------------------------------
    # Phase 3: 挂单与撤单
    # ------------------------------------------------------------------

    def refresh_quotes(self) -> Optional[MMQuote]:
        """刷新报价：撤旧单 + 挂新单"""
        if not self.is_running():
            return None

        quote = self.calculate_quote()
        if not quote:
            return None

        # 撤销旧单
        self._cancel_all_orders()

        # 挂新买单
        bid_order = self._place_order(
            side=OrderSide.BID,
            price=quote.bid_price,
            qty=quote.bid_qty,
        )

        # 挂新卖单
        ask_order = self._place_order(
            side=OrderSide.ASK,
            price=quote.ask_price,
            qty=quote.ask_qty,
        )

        return quote

    def _place_order(self, side: OrderSide, price: float, qty: float) -> Optional[str]:
        """挂限价单"""
        try:
            # 这里应该调用engine.submit_order
            # 简化实现：返回模拟order_id
            order_id = f"mm_{side.value}_{int(time.time() * 1000)}"
            self._active_orders[order_id] = {
                "side": side,
                "price": price,
                "qty": qty,
                "timestamp": time.time(),
            }
            self._stats.orders_placed += 1
            return order_id
        except Exception as e:
            logger.error("place_order failed: %s", str(e)[:100])
            return None

    def _cancel_all_orders(self):
        """撤销所有挂单"""
        for order_id in list(self._active_orders.keys()):
            try:
                # 调用engine.cancel_order
                self._stats.orders_cancelled += 1
            except Exception:
                pass
        self._active_orders.clear()

    # ------------------------------------------------------------------
    # Phase 4: 库存管理与成交处理
    # ------------------------------------------------------------------

    def on_order_filled(self, order_id: str, filled_qty: float, filled_price: float):
        """订单成交回调"""
        order = self._active_orders.get(order_id)
        if not order:
            return

        side = order["side"]
        if side == OrderSide.BID:
            self.inventory += filled_qty
            self._stats.orders_filled_bid += 1
        else:
            self.inventory -= filled_qty
            self._stats.orders_filled_ask += 1

        # 计算价差收益
        if side == OrderSide.ASK and self.inventory < 0:
            # 卖单成交，库存为负，说明之前有买单
            self._stats.spread_captured += filled_price * filled_qty * self.config.spread

        # 移除已成交订单
        del self._active_orders[order_id]

        # 检查库存是否超限
        if abs(self.inventory) > self.config.max_inventory:
            logger.warning(
                "Inventory %.4f exceeds max %.4f, adjusting quotes",
                self.inventory, self.config.max_inventory,
            )

    # ------------------------------------------------------------------
    # Phase 5: 资金费率收取
    # ------------------------------------------------------------------

    def settle_funding(self) -> float:
        """结算资金费率（每8小时调用）"""
        if not self.config.collect_funding or not self.engine:
            return 0.0

        try:
            ticker = self.engine.get_ticker(self.config.symbol)
            funding_rate = ticker.funding_rate

            # 持有空头库存时收取资金费率（费率为正）
            if self.inventory < 0 and funding_rate > 0:
                funding_payment = abs(self.inventory) * ticker.last * funding_rate
                self._stats.funding_collected += funding_payment
                self._funding_collected_total += funding_payment
                return funding_payment
            # 持有多头库存时支付资金费率
            elif self.inventory > 0 and funding_rate > 0:
                funding_payment = -self.inventory * ticker.last * funding_rate
                self._stats.funding_collected += funding_payment
                return funding_payment

            return 0.0
        except Exception as e:
            logger.error("settle_funding failed: %s", str(e)[:100])
            return 0.0

    # ------------------------------------------------------------------
    # 风控
    # ------------------------------------------------------------------

    def check_risk(self) -> Dict[str, Any]:
        """检查风控"""
        current_pnl = self.realized_pnl + self._stats.spread_captured + self._stats.funding_collected

        # 更新峰值
        if current_pnl > self._stats.peak_pnl:
            self._stats.peak_pnl = current_pnl

        # 计算回撤
        drawdown = (self._stats.peak_pnl - current_pnl) / max(self._stats.peak_pnl, 1.0)
        if drawdown > self._stats.max_drawdown:
            self._stats.max_drawdown = drawdown

        # 风控检查
        should_stop = False
        stop_reason = ""

        if drawdown > self.config.max_drawdown:
            should_stop = True
            stop_reason = "max_drawdown_exceeded"

        if abs(self.inventory) > self.config.max_inventory * 2:
            should_stop = True
            stop_reason = "inventory_overflow"

        return {
            "current_pnl": current_pnl,
            "drawdown": drawdown,
            "max_drawdown": self._stats.max_drawdown,
            "inventory": self.inventory,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        }

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        self._stats.total_pnl = (
            self.realized_pnl
            + self._stats.spread_captured
            + self._stats.funding_collected
        )
        self._stats.uptime_seconds = time.time() - self._stats.started_at
        return {
            "status": self.status.value,
            "symbol": self.config.symbol,
            "inventory": self.inventory,
            "stats": {
                "orders_placed": self._stats.orders_placed,
                "orders_filled_bid": self._stats.orders_filled_bid,
                "orders_filled_ask": self._stats.orders_filled_ask,
                "orders_cancelled": self._stats.orders_cancelled,
                "spread_captured": self._stats.spread_captured,
                "funding_collected": self._stats.funding_collected,
                "total_pnl": self._stats.total_pnl,
                "max_drawdown": self._stats.max_drawdown,
                "uptime_seconds": self._stats.uptime_seconds,
            },
        }


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    "MMStatus",
    "OrderSide",
    "MMConfig",
    "MMQuote",
    "MMStats",
    "PerpetualMarketMaker",
]
