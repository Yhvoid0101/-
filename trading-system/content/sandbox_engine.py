# -*- coding: utf-8 -*-
"""
沙盘撮合引擎 (Sandbox Matching Engine) — 模拟交易所的真实交易环境

模拟真实交易所的撮合逻辑，包括：
  - 市价单/限价单撮合
  - 手续费（maker/taker费率）
  - 滑点（基于订单簿深度模拟）
  - MEV（矿工可提取价值）模拟
  - 资金费率（永续合约）
  - 订单生命周期管理

核心理念：让子Agent在尽可能接近真实的环境中进化，避免"实验室过拟合"。
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# P0-1确定性事件驱动重构：替代模块级 random，保证可复现性
# 借鉴 NautilusTrader "Parity is built into the architecture"
from .deterministic_rng import (
    DeterministicRNG,
    global_rng_manager,
)

logger = logging.getLogger("hermes.sandbox_engine")


# ============================================================================
# 订单与交易数据结构
# ============================================================================


class OrderSide(Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class Order:
    """交易订单"""

    order_id: str
    agent_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float  # 合约张数
    price: float = 0.0  # 限价单价格，市价单=0
    leverage: int = 1
    timestamp: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_price: float = 0.0
    fee_paid: float = 0.0
    slippage: float = 0.0
    mev_cost: float = 0.0
    # v3.22修复: 添加signal_source字段, 记录入场信号来源 (BLOCK-3未完成部分)
    # 根因: Trade.signal_source 字段一直为空, 无法诊断哪个信号导致亏损
    signal_source: str = ""

    @property
    def notional(self) -> float:
        """名义价值"""
        return self.quantity * (self.filled_price or self.price)


@dataclass(slots=True)
class Trade:
    """成交记录"""

    trade_id: str
    agent_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    leverage: int = 1
    exit_price: float = 0.0
    entry_time: float = field(default_factory=time.time)
    exit_time: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fee_total: float = 0.0
    slippage_total: float = 0.0
    mev_total: float = 0.0
    status: str = "open"  # open, closed, liquidated
    liquidation_price: float = 0.0
    # BLOCK-5修复：添加funding_payment字段，记录资金费率结算
    funding_payment: float = 0.0
    # BLOCK-3修复：添加signal_source字段，记录入场信号来源
    signal_source: str = ""
    # BLOCK-4修复：添加cascade_alerted字段，记录清算级联预警
    cascade_alerted: bool = False
    # BLOCK-2修复：添加stop_loss/take_profit字段，记录决策时计算的止损止盈
    stop_loss: float = 0.0
    take_profit: float = 0.0
    # v2.35 修复: 添加 shortfall_bps 和 execution_algorithm 字段
    # 根因: evolution_loop.py:1754 试图设置 trade.shortfall_bps, 但 Trade 用了 @dataclass(slots=True)
    #       slots=True 不允许动态添加属性, 抛出 AttributeError
    # 影响: GT-Score 的 shortfall_penalty 永远为0, 执行质量差的 agent 不被惩罚
    # 来源: 500轮 intraday 进化日志大量重复 WARNING: 'Trade' object has no attribute 'shortfall_bps'
    shortfall_bps: float = 0.0
    execution_algorithm: str = "market"
    # v3.17 修复: 添加 bars_held 计数器 (解决 3.44% 胜率根因)
    # 根因: B2 修复(intrabar high/low 检查止损) + 1-3% 止损过紧 = 89.78% 交易在 <1s 内被止损
    # 修复: 前 2 根K线只用 close 价检查 (给策略呼吸空间), 第 3 根起恢复 intrabar 检查
    bars_held: int = 0
    # v3.14 修复 (ME协调): 添加 max_pnl_pct 字段 (trailing_stop 追踪最高盈利)
    # 根因: evolution_loop.py:3197 试图设置 trade.max_pnl_pct, 但 Trade 用了 @dataclass(slots=True)
    #       slots=True 不允许动态添加属性, 抛出 AttributeError
    # 影响: trailing_stop 逻辑崩溃, 进化任务在第一次平仓时退出
    # 之前此 bug 被掩盖: feeder 调用链断裂导致策略不产出信号, 不触发平仓逻辑
    # v3.14 feeder 修复后策略开始交易, 暴露了此隐藏 bug
    max_pnl_pct: float = 0.0
    # v3.32l 修复: 添加 forced_exit_price 字段 (根治 SL 滑点 + 时间止损 exit_price)
    # 根因 (v3.32k 深度分析):
    #   - SL 触发用 intraday_low 检查 (evolution_loop.py:3334), 但成交用 close 价
    #     (sandbox_engine.py:412 base_price=market.ask or market.close)
    #   - 实测: SL=$136004 但 exit=$132154, 滑点 2.72% (close 远低于 SL = gap down)
    #   - 真实交易所: SL 单触发时立即在 SL 价位成交 (或略差), 不会等到 close
    #   - 之前 close 价成交 = 过度悲观 (假设 SL 在 gap down 中填充到收盘价)
    # 修复 (基于 v3.55 历史修复 _v355_timestop_rootfix_report.json):
    #   - SL/TP 触发: forced_exit_price = trigger price (SL 或 TP)
    #   - 时间止损触发: forced_exit_price = max(close, stop) 做多 / min(close, stop) 做空
    #     (v3.55: "time_stop exit_price=max(close,stop)做多 / min(close,stop)做空")
    # 铁律: "杜绝模拟牛逼, 实盘亏钱!" — SL 滑点过度悲观会让模拟亏太多, 但 close 成交也
    #       不真实; 真实 SL 单在 trigger price 成交, 这是 backtest 标准做法
    forced_exit_price: float = 0.0
    exit_reason: str = ""
    # v598 Phase J: 添加 regime 字段 (记录 trade 关闭时的市场状态)
    # 用途: FailureModeMatcher.on_trade_closed(pnl, regime) 检测同 regime 连续亏损
    # 设置位置: evolution_loop.py loop_end 在调用 notify_trade_closed 前,
    #          从 regime_detector.get_regime(symbol) 查询并附加到 trade.regime
    # 默认 "": 未设置时 matcher 归一化为 "ranging" (向后兼容)
    regime: str = ""
    # v699 Phase O (Task #42): 添加 features 字段 (持久化决策时特征上下文)
    # 根因: PerTradeAttributionEngine 需要 features 字典进行 5 维归因, 但 v97 detail
    #       只有 18 字段无 features, 导致 data_quality="minimal", confidence=0.104,
    #       98.4% root_cause unexplained — 归因引擎"功能可用但产出无价值"
    # 用途: 入场决策时捕获 indicators/特征子集 (atr_abs/rsi_14/adx/confluence_score/
    #       mtf_*/pa_*/vp_*/kelly_* 等 19+ 字段), 持久化到 trade, 供复盘归因分析
    # 设置位置: evolution_loop.py L3419 之后 trade.features = decision.get("features", {})
    # 默认 {}: 向后兼容, 旧 trade 无此字段时归因引擎降级为 minimal 模式
    # 铁律: "复盘不是摆设" — 数据缺口 = 归因失效 = 复盘闭环空转
    features: dict = field(default_factory=dict)


@dataclass(slots=True)
class MarketData:
    """市场行情数据"""

    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = 0.0  # 买一价
    ask: float = 0.0  # 卖一价
    bid_depth: float = 0.0  # 买盘深度
    ask_depth: float = 0.0  # 卖盘深度
    funding_rate: float = 0.0  # 资金费率
    open_interest: float = 0.0  # 持仓量

    @property
    def mid_price(self) -> float:
        """中间价"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.close

    @property
    def spread(self) -> float:
        """买卖价差"""
        if self.bid and self.ask:
            return (self.ask - self.bid) / self.mid_price
        return 0.001  # 默认0.1%


# ============================================================================
# 沙盘撮合引擎
# ============================================================================


class SandboxMatchingEngine:
    """沙盘撮合引擎 — 模拟真实交易所的撮合逻辑

    使用方式:
      engine = SandboxMatchingEngine(initial_capital=10000)
      engine.update_market(market_data)  # 每根K线更新行情
      order = engine.submit_order(agent_id, "BTC-USDT", OrderSide.LONG, 1.0)
      trades = engine.get_trades(agent_id)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        # v2.27: 提高手续费到真实Binance费率
        # 原值: maker=0.02%, taker=0.05% (偏低,导致策略"模拟牛逼")
        # 真实: Binance VIP0 maker=0.1%, taker=0.1% (无BNB折扣)
        # 来源: Binance 2024费率表 + "不要模拟牛逼实盘亏钱"原则
        maker_fee: float = 0.001,   # 0.1% (真实Binance maker费率)
        taker_fee: float = 0.001,   # 0.1% (真实Binance taker费率)
        min_slippage: float = 0.0001,  # 最小滑点0.01%
        max_slippage: float = 0.05,  # v2.3修复：最大滑点5%（原0.5%严重低估大单冲击）
        mev_probability: float = 0.05,  # MEV发生概率
        mev_max_impact: float = 0.003,  # MEV最大影响0.3%
        funding_interval_hours: int = 8,  # 资金费率结算间隔
        max_leverage: int = 10,
        latency_ms: float = 100.0,  # B6修复：网络延迟（毫秒）
        rng: Optional[DeterministicRNG] = None,  # P0-1: 确定性RNG
        deterministic: bool = True,  # P0-1: 是否启用确定性模式
    ):
        """
        Args:
            initial_capital: 初始资金
            maker_fee: Maker手续费率
            taker_fee: Taker手续费率
            min_slippage: 最小滑点
            max_slippage: 最大滑点
            mev_probability: MEV发生概率
            mev_max_impact: MEV最大价格影响
            funding_interval_hours: 资金费率结算间隔（小时）
            max_leverage: 最大杠杆
            latency_ms: 网络延迟（毫秒），B6修复：延迟导致价格漂移成本
            rng: 确定性随机数生成器（None=使用全局管理器）
            deterministic: 是否启用确定性模式（True=可复现，False=兼容旧行为）
        """
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.min_slippage = min_slippage
        self.max_slippage = max_slippage
        self.mev_probability = mev_probability
        self.mev_max_impact = mev_max_impact
        self.funding_interval_hours = funding_interval_hours
        self.max_leverage = max_leverage
        self.latency_ms = latency_ms  # B6: 网络延迟

        # P0-1确定性事件驱动重构：可种子化RNG，保证可复现性
        # 借鉴 NautilusTrader "单线程确定性+双纳秒时间戳"
        # 核心铁律：同一基因编码+同一历史数据→完全相同的交易结果
        self.deterministic = deterministic
        if rng is not None:
            self._rng = rng
        else:
            # 使用全局RNG管理器的全局RNG（可被外部重置以保证可复现性）
            self._rng = global_rng_manager.global_rng

        # 账户状态
        self.balances: Dict[str, float] = defaultdict(lambda: initial_capital)
        self.positions: Dict[str, Dict[str, Trade]] = defaultdict(dict)  # {agent_id: {symbol: Trade}}
        self.open_orders: List[Order] = []
        self.trade_history: Dict[str, List[Trade]] = defaultdict(list)
        self.order_history: List[Order] = []

        # 市场状态
        self.current_market: Dict[str, MarketData] = {}
        self._order_counter: int = 0
        self._trade_counter: int = 0
        self._last_funding_time: float = 0.0

        # 统计
        self.stats: Dict[str, Any] = {
            "total_orders": 0,
            "total_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "total_mev": 0.0,
            "liquidations": 0,
        }

    # ------------------------------------------------------------------
    # 行情更新
    # ------------------------------------------------------------------

    def update_market(self, market_data: MarketData) -> None:
        """更新市场行情

        每根K线调用一次，驱动：
        1. 撮合挂单
        2. 检查止损/止盈
        3. 检查清算
        4. 结算资金费率
        """
        self.current_market[market_data.symbol] = market_data

        # 撮合限价单
        self._match_limit_orders(market_data)

        # 检查所有持仓的止损/止盈/清算
        self._check_positions(market_data)

        # 资金费率结算
        self._settle_funding(market_data)

    def get_market(self, symbol: str) -> Optional[MarketData]:
        """获取当前行情"""
        return self.current_market.get(symbol)

    # ------------------------------------------------------------------
    # 订单提交
    # ------------------------------------------------------------------

    def submit_order(
        self,
        agent_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        leverage: int = 1,
        signal_source: str = "",
    ) -> Order:
        """提交订单

        Args:
            agent_id: 子Agent ID
            symbol: 交易对
            side: 多/空
            quantity: 数量（合约张数）
            order_type: 市价/限价
            price: 限价（市价单不传）
            leverage: 杠杆倍数

        Returns:
            Order对象（含成交信息）
        """
        self._order_counter += 1
        order_id = f"ord-{self._order_counter:06d}"

        # 杠杆限制
        leverage = min(leverage, self.max_leverage)

        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            leverage=leverage,
            signal_source=signal_source,
        )

        market = self.current_market.get(symbol)
        if not market:
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return order

        if order_type == OrderType.MARKET:
            self._execute_market_order(order, market)
        else:
            self._place_limit_order(order, market)

        self.order_history.append(order)
        self.stats["total_orders"] += 1

        return order

    def submit_long(
        self, agent_id: str, symbol: str, quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0, leverage: int = 1,
        signal_source: str = "",
    ) -> Order:
        """便捷方法：做多"""
        return self.submit_order(agent_id, symbol, OrderSide.LONG, quantity, order_type, price, leverage, signal_source=signal_source)

    def submit_short(
        self, agent_id: str, symbol: str, quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0, leverage: int = 1,
        signal_source: str = "",
    ) -> Order:
        """便捷方法：做空"""
        return self.submit_order(agent_id, symbol, OrderSide.SHORT, quantity, order_type, price, leverage, signal_source=signal_source)

    def close_position(
        self, agent_id: str, symbol: str,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
    ) -> Optional[Order]:
        """平仓：关闭指定交易对的持仓

        v3.32l 修复: 优先使用 trade.forced_exit_price (由 _check_exit 设置)
        根因: SL/TP/时间止损触发时, exit_price 应为 trigger price, 而非 market.close
        """
        trade = self.positions.get(agent_id, {}).get(symbol)
        if not trade or trade.status != "open":
            return None

        # v3.32l: 优先使用 forced_exit_price (SL/TP/时间止损场景)
        # 若调用方未传 price, 但 trade 上有 forced_exit_price, 则使用之
        forced_price = price
        if forced_price <= 0:
            forced_price = getattr(trade, 'forced_exit_price', 0.0)

        close_side = OrderSide.SHORT if trade.side == OrderSide.LONG else OrderSide.LONG
        order = self.submit_order(
            agent_id, symbol, close_side,
            trade.quantity, order_type, forced_price, trade.leverage if hasattr(trade, 'leverage') else 1,
        )
        # 平仓后重置 forced_exit_price (避免影响下次开仓)
        if hasattr(trade, 'forced_exit_price'):
            trade.forced_exit_price = 0.0
        return order

    # ------------------------------------------------------------------
    # 市价单执行
    # ------------------------------------------------------------------

    def _execute_market_order(self, order: Order, market: MarketData) -> None:
        """执行市价单

        B5修复：基于订单簿深度模拟部分成交
        - 大单超过可用深度时，分层成交（walk the book）
        - 深度不足时部分成交，剩余进入下一轮
        - 加权平均成交价反映真实执行成本
        """
        # 计算成交价（含滑点）
        # v3.32l 修复: 当 order.price > 0 (forced_exit_price 场景), 用它作为 base_price
        # 根因: SL/TP/时间止损触发时, _check_exit 设置 trade.forced_exit_price,
        #       close_position 将其传入 order.price, 这里读取作为基准成交价
        # 场景1 (SL触发): forced_exit_price = stop_loss → 在止损价成交 (非 close 价)
        # 场景2 (TP触发): forced_exit_price = take_profit → 在止盈价成交
        # 场景3 (时间止损): forced_exit_price = max/min(close, stop) → 受 SL 约束
        # 普通开仓: order.price = 0 → 走原逻辑 (market.ask/bid or close)
        forced_exit = order.price > 0
        if forced_exit:
            base_price = order.price
            # 深度仍按订单簿取 (SL/TP 成交也受深度影响)
            if order.side == OrderSide.LONG:
                available_depth = market.ask_depth
            else:
                available_depth = market.bid_depth
        elif order.side == OrderSide.LONG:
            base_price = market.ask or market.close
            available_depth = market.ask_depth
        else:
            base_price = market.bid or market.close
            available_depth = market.bid_depth

        # B5修复：基于订单簿深度决定成交量
        # 如果订单量超过可用深度，只成交可用部分（部分成交）
        # 深度为0或负值时，使用订单量作为回退（避免完全无法成交）
        if available_depth <= 0:
            available_depth = max(order.quantity * 0.5, 1.0)  # 回退：至少成交50%

        fill_qty = min(order.quantity, available_depth)
        remaining_qty = order.quantity - fill_qty

        # B5修复：分层成交（walk the book）
        # 第一层：在最优价格成交fill_qty
        # 如果有remaining_qty，在更差的价格成交（模拟吃单深度）
        slippage_pct = self._compute_slippage(fill_qty, market, order.side)
        latency_cost_pct = self._compute_latency_cost(market)
        # v2.19 P2: K线跳变成本 — 模拟从close到下一根open的look-ahead成本
        # 真实交易：看到K线收盘→下单→下一根K线开盘成交
        # 沙盘缺陷：同根K线决策+成交，获得系统性价格优势
        # 修复：加入|open-close|/close作为额外成本，总是不利于交易者(adverse selection)
        # 来源：实盘执行经验 + Almgren-Chriss最优执行框架
        if forced_exit:
            bar_gap_pct = 0.0
        elif hasattr(market, 'open') and market.open > 0 and market.close > 0:
            bar_gap_pct = abs(market.open - market.close) / market.close
        else:
            bar_gap_pct = 0.0
        total_cost_pct = slippage_pct + latency_cost_pct + bar_gap_pct

        if order.side == OrderSide.LONG:
            fill_price_level1 = base_price * (1 + total_cost_pct)
        else:
            fill_price_level1 = base_price * (1 - total_cost_pct)

        # 如果有剩余量，在更差的价格成交（价格冲击）
        if remaining_qty > 0:
            # 第二层价格：额外0.1%价格冲击（模拟吃掉更深的订单簿）
            level2_impact = 0.001  # 0.1%额外成本
            if order.side == OrderSide.LONG:
                fill_price_level2 = fill_price_level1 * (1 + level2_impact)
            else:
                fill_price_level2 = fill_price_level1 * (1 - level2_impact)
            # 加权平均成交价
            total_notional = fill_qty * fill_price_level1 + remaining_qty * fill_price_level2
            fill_price = total_notional / order.quantity
            # 部分成交标记
            order.filled_qty = fill_qty  # 实际成交的量
            order.status = OrderStatus.PARTIAL
        else:
            fill_price = fill_price_level1
            order.filled_qty = order.quantity
            order.status = OrderStatus.FILLED

        # MEV模拟
        mev_impact = self._compute_mev(order, market)

        # 手续费
        fee_rate = self.taker_fee
        notional = order.filled_qty * fill_price
        fee = notional * fee_rate

        order.filled_price = fill_price
        order.slippage = total_cost_pct  # 包含延迟成本
        order.mev_cost = mev_impact * notional
        order.fee_paid = fee

        # 更新持仓（用实际成交量）
        self._update_position(order, market, fill_price, fee)

        self.stats["total_volume"] += notional
        self.stats["total_fees"] += fee
        self.stats["total_mev"] += order.mev_cost

    def _place_limit_order(self, order: Order, market: MarketData) -> None:
        """挂限价单"""
        # 检查限价是否合理
        if order.side == OrderSide.LONG and order.price >= (market.ask or market.close):
            # 限价高于卖一价，按市价成交
            order.order_type = OrderType.MARKET
            self._execute_market_order(order, market)
            return
        elif order.side == OrderSide.SHORT and order.price <= (market.bid or market.close):
            order.order_type = OrderType.MARKET
            self._execute_market_order(order, market)
            return

        self.open_orders.append(order)

    def _match_limit_orders(self, market: MarketData) -> None:
        """撮合挂单"""
        remaining = []
        for order in self.open_orders:
            if order.symbol != market.symbol:
                remaining.append(order)
                continue

            should_fill = False
            if order.side == OrderSide.LONG and market.low <= order.price:
                should_fill = True
            elif order.side == OrderSide.SHORT and market.high >= order.price:
                should_fill = True

            if should_fill:
                order.filled_price = order.price
                order.filled_qty = order.quantity
                order.slippage = 0.0  # 限价单无滑点

                fee = order.quantity * order.price * self.maker_fee
                order.fee_paid = fee
                order.status = OrderStatus.FILLED

                self._update_position(order, market, order.price, fee)
                self.stats["total_volume"] += order.quantity * order.price
                self.stats["total_fees"] += fee
            else:
                remaining.append(order)

        self.open_orders = remaining

    # ------------------------------------------------------------------
    # 持仓管理
    # ------------------------------------------------------------------

    def _update_position(
        self, order: Order, market: MarketData,
        fill_price: float, fee: float,
    ) -> None:
        """更新持仓状态"""
        agent_id = order.agent_id
        symbol = order.symbol
        existing = self.positions.get(agent_id, {}).get(symbol)

        if existing and existing.status == "open":
            # 平仓
            existing.exit_price = fill_price
            existing.exit_time = market.timestamp  # 使用市场时间戳，保持与entry_time一致
            existing.fee_total += fee

            # 计算PnL
            if existing.side == OrderSide.LONG:
                pnl_pct = (fill_price - existing.entry_price) / existing.entry_price
            else:
                pnl_pct = (existing.entry_price - fill_price) / existing.entry_price

            existing.pnl = existing.quantity * existing.entry_price * pnl_pct - existing.fee_total
            existing.pnl_pct = pnl_pct
            existing.status = "closed"

            self.trade_history[agent_id].append(existing)
            self.stats["total_trades"] += 1

            # 更新余额
            released_margin = existing.quantity * existing.entry_price / existing.leverage
            self.balances[agent_id] += released_margin + existing.pnl
            del self.positions[agent_id][symbol]

            logger.debug(
                "Trade closed: %s %s %s PnL=%.2f (%.2f%%)",
                agent_id[:8], symbol, existing.side.value, existing.pnl, pnl_pct * 100,
            )
        else:
            # 开仓
            self._trade_counter += 1
            trade = Trade(
                trade_id=f"trade-{self._trade_counter:06d}",
                agent_id=agent_id,
                order_id=order.order_id,
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                leverage=order.leverage,
                entry_time=market.timestamp,  # 使用市场时间戳，非真实时间
                fee_total=fee,
                slippage_total=order.slippage,
                mev_total=order.mev_cost,
                liquidation_price=self._calc_liquidation_price(
                    fill_price, order.side, order.leverage,
                ),
                # v3.22修复: 传入signal_source (从Order继承)
                signal_source=getattr(order, "signal_source", ""),
            )

            if agent_id not in self.positions:
                self.positions[agent_id] = {}
            self.positions[agent_id][symbol] = trade

            # 扣除保证金
            margin = order.quantity * fill_price / order.leverage
            self.balances[agent_id] -= margin

            logger.debug(
                "Trade opened: %s %s %s qty=%.4f price=%.2f",
                agent_id[:8], symbol, order.side.value, order.quantity, fill_price,
            )

    def _check_positions(self, market: MarketData) -> None:
        """检查持仓的止损/止盈/清算"""
        for agent_id, agent_positions in list(self.positions.items()):
            for symbol, trade in list(agent_positions.items()):
                if trade.symbol != market.symbol:
                    continue

                current_price = market.close
                if trade.side == OrderSide.LONG:
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                else:
                    pnl_pct = (trade.entry_price - current_price) / trade.entry_price

                # 检查清算
                if trade.liquidation_price > 0:
                    if trade.side == OrderSide.LONG and current_price <= trade.liquidation_price:
                        self._liquidate(agent_id, symbol, trade, current_price, market)
                        continue
                    elif trade.side == OrderSide.SHORT and current_price >= trade.liquidation_price:
                        self._liquidate(agent_id, symbol, trade, current_price, market)
                        continue

    def _liquidate(
        self, agent_id: str, symbol: str,
        trade: Trade, price: float, market: MarketData,
    ) -> None:
        """清算爆仓"""
        trade.exit_price = price
        trade.exit_time = market.timestamp
        trade.status = "liquidated"

        # 清算费
        liquidation_fee = trade.quantity * trade.entry_price * 0.01
        trade.fee_total += liquidation_fee

        # 全部亏损
        if trade.side == OrderSide.LONG:
            pnl_pct = (price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - price) / trade.entry_price

        trade.pnl = trade.quantity * trade.entry_price * pnl_pct - trade.fee_total
        trade.pnl_pct = pnl_pct

        self.trade_history[agent_id].append(trade)
        self.stats["total_trades"] += 1
        self.stats["liquidations"] += 1

        self.balances[agent_id] += trade.pnl
        del self.positions[agent_id][symbol]

        logger.warning(
            "LIQUIDATION: %s %s at %.2f, PnL=%.2f",
            agent_id[:8], symbol, price, trade.pnl,
        )

    def _settle_funding(self, market: MarketData) -> None:
        """结算资金费率"""
        if market.funding_rate == 0:
            return

        hours_since = (market.timestamp - self._last_funding_time) / 3600
        if hours_since < self.funding_interval_hours:
            return

        for agent_id, agent_positions in self.positions.items():
            for symbol, trade in agent_positions.items():
                if trade.symbol == market.symbol:
                    funding_payment = (
                        trade.quantity * trade.entry_price * market.funding_rate
                    )
                    self.balances[agent_id] -= funding_payment
                    trade.fee_total += abs(funding_payment)
                    # BLOCK-5修复：单独记录funding_payment到trade对象
                    trade.funding_payment += funding_payment

        self._last_funding_time = market.timestamp

    # ------------------------------------------------------------------
    # 滑点与MEV模拟
    # ------------------------------------------------------------------

    def _compute_slippage(
        self, quantity: float, market: MarketData, side: OrderSide,
    ) -> float:
        """计算滑点

        B4修复：从线性深度模型改为平方根法则（Square-Root Law）
        来源：Almgren-Chriss最优执行框架
        公式：Market Impact = k × √(Order_Size / ADV)

        实证：平方根法则与实盘误差<3%，线性模型在大单场景严重低估滑点

        - 小单：滑点接近最小值
        - 大单：滑点按平方根增长（比线性慢，但比固定值更真实）
        - 波动大时滑点增加
        """
        # B4修复：使用平方根法则计算滑点
        # ADV近似 = (bid_depth + ask_depth) / 2 × 某个倍数
        # 沙盘中bid_depth/ask_depth是单笔K线成交量的30%，作为ADV代理
        depth = market.bid_depth if side == OrderSide.SHORT else market.ask_depth
        adv_proxy = max(market.volume, depth * 3.33, 1.0)  # ADV代理值

        if quantity > 0 and adv_proxy > 0:
            # 平方根法则：impact = k × √(size / ADV)
            # k=0.1（流动性系数），结果转换为滑点百分比
            size_ratio = quantity / adv_proxy
            k = 0.1  # 流动性系数（0.1=高流动性，0.5=低流动性）
            sqrt_impact = k * math.sqrt(max(size_ratio, 0))

            # v2.3修复：移除0.5%硬上限，大单冲击真实反映
            # 原问题：max_slippage=0.5%截断大单冲击，size_ratio=1时10%冲击被截断为0.5%
            #         大单冲击被低估20倍，策略在沙盘不惧大单，实盘遇流动性枯竭必亏
            # 修复方案：
            #   - 小单（size_ratio < 0.5）：平方根法则，上限max_slippage
            #   - 大单（size_ratio >= 0.5）：允许超过max_slippage，对数压缩防止无限增长
            if size_ratio < 0.5:
                # 小单：平方根法则 + 软上限
                base_slippage = max(self.min_slippage, min(self.max_slippage, sqrt_impact))
            else:
                # 大单：允许冲击超过max_slippage，对数压缩
                # 公式：base = max_slippage + log(1 + size_ratio) * 0.02
                # size_ratio=1 → base ≈ 5% + 1.4% = 6.4%
                # size_ratio=5 → base ≈ 5% + 3.6% = 8.6%
                # size_ratio=10 → base ≈ 5% + 4.8% = 9.8%
                base_slippage = self.max_slippage + math.log(1 + size_ratio) * 0.02
                base_slippage = max(self.min_slippage, base_slippage)
        else:
            base_slippage = self.max_slippage * 0.5

        # 波动率加成
        vol_boost = 1.0 + market.spread * 10

        # P0-1确定性重构：使用可种子化RNG替代模块级random.gauss
        # 保证同一基因+同一数据→同一滑点噪声→可复现的进化对比
        noise = self._rng.gauss(0, base_slippage * 0.2)

        return max(0, base_slippage * vol_boost + noise)

    def _compute_latency_cost(self, market: MarketData) -> float:
        """B6修复：计算网络延迟导致的价格漂移成本

        延迟期间价格会随机漂移，对交易者总是不利的（adverse selection）。
        买入时价格可能上涨，卖出时价格可能下跌。

        模型：
          - 基础延迟成本 = latency_ms / 1000 × 5.0 bps（来自RealisticSlippageModel）
          - 波动率加成：高波动时延迟成本更高
          - 随机漂移：模拟延迟期间的价格随机游走

        来源：Almgren-Chriss最优执行框架
        实证：100ms延迟 ≈ 0.5bps，500ms ≈ 2.5bps（与实盘一致）

        Args:
            market: 市场数据

        Returns:
            延迟成本百分比（总是正值，对交易者不利）
        """
        if self.latency_ms <= 0:
            return 0.0

        # 基础延迟成本（bps转百分比）
        # 100ms → 0.5bps → 0.00005 (0.005%)
        base_latency_bps = (self.latency_ms / 1000.0) * 5.0
        base_latency_pct = base_latency_bps / 10000.0

        # 波动率加成：用spread作为波动率代理
        # spread越大，延迟期间价格漂移越大
        vol_factor = 1.0 + market.spread * 5.0

        # 随机漂移：延迟期间价格随机游走
        # 使用半正态分布（总是不利方向）
        # P0-1确定性重构：使用可种子化RNG替代模块级random.gauss
        drift_noise = abs(self._rng.gauss(0, base_latency_pct * 0.5))

        total_latency_cost = base_latency_pct * vol_factor + drift_noise

        # 上限：延迟成本不超过max_slippage的50%
        return min(total_latency_cost, self.max_slippage * 0.5)

    def _compute_mev(self, order: Order, market: MarketData) -> float:
        """模拟MEV（矿工可提取价值）

        MEV场景：
        - 三明治攻击：大单被前后夹击
        - 抢跑：订单被抢先执行
        - 尾随：订单成交后价格被推高/压低

        返回MEV造成的额外成本比率
        """
        if self._rng.random() > self.mev_probability:
            return 0.0

        # MEV影响与大单成正比
        size_factor = min(order.quantity / 100.0, 1.0)
        # P0-1确定性重构：使用可种子化RNG替代模块级random.uniform
        mev = self._rng.uniform(0, self.mev_max_impact * size_factor)

        return mev

    def _calc_liquidation_price(
        self, entry_price: float, side: OrderSide, leverage: int,
    ) -> float:
        """计算清算价格"""
        if leverage <= 1:
            return 0.0  # 无杠杆不爆仓

        maintenance_margin = 0.005  # 0.5% 维持保证金

        if side == OrderSide.LONG:
            return entry_price * (1 - 1 / leverage + maintenance_margin)
        else:
            return entry_price * (1 + 1 / leverage - maintenance_margin)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_positions(self, agent_id: str) -> Dict[str, Trade]:
        """获取Agent当前持仓"""
        return self.positions.get(agent_id, {})

    def get_trades(self, agent_id: str) -> List[Trade]:
        """获取Agent已完成交易"""
        return self.trade_history.get(agent_id, [])

    def get_balance(self, agent_id: str) -> float:
        """获取Agent余额"""
        return self.balances.get(agent_id, self.initial_capital)

    def get_equity(self, agent_id: str) -> float:
        """获取Agent总权益（余额+未实现盈亏）"""
        balance = self.get_balance(agent_id)
        unrealized = 0.0
        for symbol, trade in self.positions.get(agent_id, {}).items():
            if trade.status == "open":
                market = self.current_market.get(symbol)
                if market:
                    if trade.side == OrderSide.LONG:
                        unrealized += trade.quantity * (market.close - trade.entry_price)
                    else:
                        unrealized += trade.quantity * (trade.entry_price - market.close)
        return balance + unrealized

    def get_all_trades(self) -> Dict[str, List[Trade]]:
        """获取所有Agent的交易记录"""
        return dict(self.trade_history)

    def reset(self) -> None:
        """重置引擎状态"""
        self.balances.clear()
        self.positions.clear()
        self.open_orders.clear()
        self.trade_history.clear()
        self.order_history.clear()
        self.current_market.clear()
        self._order_counter = 0
        self._trade_counter = 0
        self._last_funding_time = 0.0
        self.stats = {
            "total_orders": 0,
            "total_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "total_mev": 0.0,
            "liquidations": 0,
        }

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """P0-1: 重置确定性RNG，保证可复现性

        在新一轮进化或需要复现特定场景时调用。

        Args:
            seed: 新种子（None=使用当前种子重置）
        """
        if seed is not None:
            self._rng.reseed(seed)
        else:
            self._rng.reset()

    def get_rng_state(self) -> tuple:
        """P0-1: 获取RNG内部状态（用于检查点）"""
        return self._rng.get_state()

    def set_rng_state(self, state: tuple) -> None:
        """P0-1: 恢复RNG内部状态（用于检查点恢复）"""
        self._rng.set_state(state)

    def verify_determinism(self, n_runs: int = 2) -> bool:
        """P0-1: 验证引擎确定性

        运行多次相同的操作序列，验证结果完全一致。

        Args:
            n_runs: 验证运行次数

        Returns:
            True 如果所有运行结果完全一致
        """
        # 保存当前状态
        saved_state = self.get_rng_state()
        saved_balances = dict(self.balances)
        saved_positions = {k: dict(v) for k, v in self.positions.items()}

        results = []
        for _ in range(n_runs):
            # 重置RNG
            self._rng.reset()
            # 生成一个测试随机数序列
            test_seq = [self._rng.gauss(0, 1) for _ in range(10)]
            results.append(tuple(test_seq))

        # 恢复状态
        self.set_rng_state(saved_state)
        self.balances = defaultdict(lambda: self.initial_capital, saved_balances)
        self.positions = defaultdict(dict, saved_positions)

        # 验证所有运行结果一致
        is_deterministic = all(r == results[0] for r in results)
        if is_deterministic:
            logger.info("P0-1 确定性验证通过：%d次运行结果完全一致", n_runs)
        else:
            logger.error("P0-1 确定性验证失败：运行结果不一致")
        return is_deterministic

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        stats = dict(self.stats)
        stats["open_positions"] = sum(
            len(positions) for positions in self.positions.values()
        )
        stats["open_orders"] = len(self.open_orders)
        return stats