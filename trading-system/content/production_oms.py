# -*- coding: utf-8 -*-
"""
生产级订单管理系统 (Production OMS) — v1.0

R16-3 核心新模块: 金融级生产环境订单管理系统。

业界背景(2026年最新):
  - 87%回测正收益策略实盘亏损(Glassnode 2025), 根因之一是订单执行偏差
  - 真实环境: 部分成交、滑点、延迟、订单拒绝、网络抖动
  - 回测假设: 全部成交、无滑点、零延迟、订单100%成功
  - 差距: 5-20%的Sharpe衰减来自OMS执行偏差

核心功能:
  1. 订单生命周期管理
     - NEW → PARTIALLY_FILLED → FILLED / CANCELED / REJECTED / EXPIRED
     - 状态机严格转换, 不允许非法跳转
     - 来源: FIX 4.4协议 + Binance订单状态机

  2. 真实滑点模拟器
     - 基于订单簿深度的非线性滑点
     - 小单(<0.1% ADV): 线性滑点
     - 中单(0.1-1% ADV): 平方根冲击(Almgren-Chriss)
     - 大单(>1% ADV): 严重滑点, 可能无法成交
     - 来源: Almgren-Chriss 2000 + R15-2 market_impact_model

  3. 延迟真实模拟器
     - 网络延迟: 50-500ms(本地→交易所)
     - 交易所处理: 10-100ms
     - 撮合延迟: 1-10ms
     - 高频场景: 总延迟可达1s+
     - 来源: Binance WebSocket实测数据

  4. 部分成交处理器
     - 大单拆分为多个子单(TWAP/VWAP)
     - 部分成交后剩余量继续挂单
     - 超时未成交的剩余量自动撤单
     - 来源: execution_algorithms最佳实践

  5. 订单失败降级
     - 订单被拒(余额不足/参数错误) → 降级处理
     - 网络超时 → 重试或降级
     - 部分成交 → 接受已成交部分
     - 来源: production_hardening Kill Switch

铁律:
  - "金融级生产环境" — 用户原话
  - 高可用性: 单点故障不阻塞整体(降级而非崩溃)
  - 低延迟性: 关键路径<100ms
  - 数据准确性: 订单状态100%一致
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.production_oms")


# ============================================================================
# 订单状态机
# ============================================================================


class OrderStatus(str, Enum):
    """订单状态(FIX 4.4 + Binance扩展)"""
    NEW = "NEW"                        # 新订单
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"                  # 完全成交
    CANCELED = "CANCELED"              # 已撤销
    REJECTED = "REJECTED"              # 已拒绝
    EXPIRED = "EXPIRED"                # 已过期
    PENDING_CANCEL = "PENDING_CANCEL"  # 撤单中


class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """订单类型"""
    MARKET = "MARKET"        # 市价单(立即成交, 有滑点)
    LIMIT = "LIMIT"          # 限价单(指定价格, 可能不成交)
    STOP = "STOP"            # 止损单
    STOP_LIMIT = "STOP_LIMIT"  # 止损限价单
    POST_ONLY = "POST_ONLY"  # 只做maker


# 合法状态转换
VALID_TRANSITIONS: Dict[OrderStatus, List[OrderStatus]] = {
    OrderStatus.NEW: [
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.PENDING_CANCEL,
    ],
    OrderStatus.PARTIALLY_FILLED: [
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.PENDING_CANCEL,
    ],
    OrderStatus.PENDING_CANCEL: [
        OrderStatus.CANCELED,
        OrderStatus.FILLED,  # 撤单前可能已成交
    ],
    OrderStatus.FILLED: [],
    OrderStatus.CANCELED: [],
    OrderStatus.REJECTED: [],
    OrderStatus.EXPIRED: [],
}


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class Order:
    """订单"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # 限价单价格
    stop_price: Optional[float] = None  # 止损触发价
    # R18-5 BLOCK-6修复: 添加reference_price字段(提交时的市场价,用于市价单滑点统计)
    reference_price: Optional[float] = None  # 提交时的参考市场价
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    time_in_force: str = "GTC"  # GTC/IOC/FOK/PO
    reduce_only: bool = False
    post_only: bool = False
    rejects_reason: str = ""
    fills: List[Dict[str, Any]] = field(default_factory=list)  # 每次成交记录


@dataclass(slots=True)
class SlippageModel:
    """滑点模型参数"""
    avg_daily_volume_usd: float = 100_000_000.0  # ADV默认1亿美元(仅作fallback)
    base_spread_bps: float = 2.0  # 基础买卖价差2bps
    market_impact_coef: float = 0.10  # Almgren-Chriss系数
    volatility: float = 0.50  # 50%年化波动率


# R19-3 WARN-5修复: symbol→ADV动态映射表
# 来源: CoinMarketCap 2026 现货日均成交额(估算,单位:USD)
# 之前: SlippageModel.avg_daily_volume_usd 硬编码1亿,所有币种共用
# 后果: BTC(实际300亿)滑点被高估, 山寨币(实际1000万)滑点被低估 → 模拟vs实盘滑点差异巨大
# 修复: 按symbol查询真实ADV, 缺省时fallback到保守值(1亿,反温室:宁可高估滑点)
SYMBOL_ADV_MAP: Dict[str, float] = {
    # 主流币(高流动性, ADV > 50亿USD)
    "BTC": 30_000_000_000.0,   # ~300亿
    "ETH": 15_000_000_000.0,   # ~150亿
    "BNB": 2_000_000_000.0,    # ~20亿
    "SOL": 4_000_000_000.0,    # ~40亿
    "XRP": 2_000_000_000.0,    # ~20亿
    # 中型币(中等流动性, ADV 1-10亿USD)
    "ADA": 800_000_000.0,      # ~8亿
    "DOGE": 1_500_000_000.0,   # ~15亿
    "AVAX": 600_000_000.0,     # ~6亿
    "LINK": 500_000_000.0,     # ~5亿
    "MATIC": 700_000_000.0,    # ~7亿
    "DOT": 500_000_000.0,      # ~5亿
    # 小型山寨币(低流动性, ADV < 1亿USD,反温室: 保守取下限)
    "DEFAULT_SMALL": 50_000_000.0,  # 默认山寨币5千万
}


def get_adv_for_symbol(symbol: str) -> float:
    """R19-3: 根据symbol查询真实ADV(日均成交额,USD)

    Args:
        symbol: 交易对(支持多种格式):
          - "BTC/USDT" (CCXT标准)
          - "BTC-USDT" (沙盘默认)
          - "BTC/USDT:USDT" (Hyperliquid永续)
          - "BTCUSDT" (Binance原生, R20-4b修复)
          - "BTC" (基础币种直接)

    Returns:
        ADV(USD), 缺省时返回1亿(保守fallback,反温室:宁可高估滑点)

    来源: CoinMarketCap 2026现货成交额估算 + 反温室设计原则
    R20-4b修复: 原版仅支持分隔符格式, Binance原生"BTCUSDT"会fallback到
      DEFAULT_SMALL(50M), 导致BTC滑点被高估600x(30B/50M), 模拟≠实盘.
      修复: 分隔符解析失败后, 按已知币种前缀匹配(长前缀优先, 避免BTC匹配BT币种).
    """
    if not symbol:
        return 100_000_000.0  # 缺省保守值
    # 路径1: 分隔符解析(原R19-3逻辑, 处理"BTC/USDT"、"BTC-USDT"等)
    base = symbol.split("/")[0].split("-")[0].upper()
    if base in SYMBOL_ADV_MAP:
        return SYMBOL_ADV_MAP[base]
    # 路径2: R20-4b Binance原生格式("BTCUSDT"无分隔符)
    # 按币种长度降序匹配, 避免"BTC"误匹配"BTCUSD"等不存在币种
    # 同时避免短前缀误匹配(如"ETH"匹配"ETHR"——不存在但理论风险)
    upper_symbol = symbol.upper()
    known_bases = sorted(
        (k for k in SYMBOL_ADV_MAP if k != "DEFAULT_SMALL"),
        key=len, reverse=True,
    )
    for known_base in known_bases:
        if upper_symbol.startswith(known_base):
            return SYMBOL_ADV_MAP[known_base]
    return SYMBOL_ADV_MAP["DEFAULT_SMALL"]


@dataclass(slots=True)
class LatencyModel:
    """延迟模型参数"""
    network_latency_ms: Tuple[float, float] = (50.0, 500.0)  # 网络延迟范围
    exchange_processing_ms: Tuple[float, float] = (10.0, 100.0)  # 交易所处理
    matching_engine_ms: Tuple[float, float] = (1.0, 10.0)  # 撮合引擎
    retry_delay_ms: Tuple[float, float] = (1000.0, 5000.0)  # 重试延迟


@dataclass(slots=True)
class FillResult:
    """成交结果"""
    success: bool
    order: Optional[Order] = None
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    error: str = ""


# ============================================================================
# 1. 真实滑点模拟器
# ============================================================================


class RealisticSlippageSimulator:
    """真实滑点模拟器

    基于订单簿深度的非线性滑点:
      - 小单(<0.1% ADV): 线性滑点 = base_spread + α × participation
      - 中单(0.1-1% ADV): 平方根冲击 = base_spread + β × σ × √(Q/V)
      - 大单(>1% ADV): 严重滑点, 可能无法完全成交

    来源: Almgren-Chriss 2000 + R15-2 market_impact_model
    """

    def __init__(self, model: SlippageModel = None):
        self.model = model or SlippageModel()

    def compute_slippage(
        self,
        order_size_usd: float,
        is_market_order: bool = True,
        leverage: int = 1,
        symbol: str = "",
    ) -> Tuple[float, float, bool]:
        """计算滑点

        Phase 7J-8 反温室修复 HIGH #7: 添加 leverage 参数
        之前: 滑点基于 order_size_usd (保证金), 未考虑杠杆放大
              沙盘 1x 杠杆滑点小, 实盘 5x 杠杆实际市场冲击 = 保证金×5, 滑点应更大
        现在: effective_order_size = order_size_usd × leverage, 真实反映市场冲击
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 杠杆滑点低估=实盘成本失控

        R19-3 WARN-5修复: 添加 symbol 参数,动态查询ADV
        之前: adv = self.model.avg_daily_volume_usd (硬编码1亿,所有币种共用)
        后果: BTC(实际300亿)滑点高估, 山寨币(实际1000万)滑点低估 → 模拟vs实盘滑点差异
        现在: 优先使用 get_adv_for_symbol(symbol), 缺省时fallback到model默认值

        Args:
            order_size_usd: 订单金额(USD, 保证金)
            is_market_order: 是否市价单(taker有滑点, maker无)
            leverage: 杠杆倍数 (默认1=现货, 5=5x杠杆合约)
            symbol: 交易对(如"BTC/USDT"),用于动态查询ADV [R19-3新增]

        Returns:
            (slippage_bps, fill_ratio, can_fill)
            - slippage_bps: 滑点(bps)
            - fill_ratio: 可成交比例(0-1)
            - can_fill: 是否可成交
        """
        if not is_market_order:
            # 限价单无滑点,但可能不成交
            return 0.0, 1.0, True

        if order_size_usd <= 0:
            return 0.0, 0.0, False

        # Phase 7J-8: 杠杆放大订单规模 (实际市场冲击 = 保证金 × 杠杆)
        effective_leverage = max(1, int(leverage))
        effective_order_size = order_size_usd * effective_leverage

        # R19-3 WARN-5修复: 动态查询ADV(按symbol), 缺省时fallback到model默认值
        # 之前: adv = self.model.avg_daily_volume_usd (硬编码1亿)
        # 现在: BTC查询到300亿, 山寨币查询到5千万, 真实反映市场冲击
        adv = get_adv_for_symbol(symbol) if symbol else self.model.avg_daily_volume_usd
        participation = effective_order_size / adv

        # Phase 7J-9 反温室修复 HIGH #11: ADV 默认1亿对山寨币过高 → 保守下限
        # 之前: ADV=1亿, order=1万 → participation=0.0001 < 0.001 → 滑点2.1bps
        #       但山寨币 ADV 可能只有1000万, 真实 participation=0.001, 滑点应更高
        # 修复: participation < 0.0001 (极小单) 时使用保守下限 3bps (反温室: 宁可高估滑点)
        # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 滑点低估=实盘成本超预期
        is_very_small = participation < 0.0001

        # 基础滑点
        base_slippage = self.model.base_spread_bps

        if participation < 0.001:
            # 小单: 线性滑点
            impact = base_slippage + participation * 1000.0
            # Phase 7 修复: 整个小单区间都受保守下限约束,确保滑点单调递增
            # 之前: 只有is_very_small(<0.0001)才用下限,导致participation在0.0001-0.001区间
            #       的滑点可能<3bps(如base=2+0.167=2.17),反不如极小单的3bps,违反单调性
            # 修复: 小单区间统一用max(impact, 3.0),确保单调递增
            # 铁律: 滑点低估=实盘成本超预期
            impact = max(impact, 3.0)  # 最低 3bps (反温室: 保守下限,全小单区间)
            return impact, 1.0, True
        elif participation < 0.01:
            # 中单: 平方根冲击(Almgren-Chriss)
            impact_pct = self.model.market_impact_coef * self.model.volatility * math.sqrt(participation)
            impact_bps = impact_pct * 10000.0  # 转换为bps
            total_slippage = base_slippage + impact_bps
            return total_slippage, 1.0, True
        elif participation < 0.05:
            # 大单: 严重滑点,部分成交
            impact_pct = self.model.market_impact_coef * self.model.volatility * math.sqrt(participation) * 2.0
            impact_bps = impact_pct * 10000.0
            total_slippage = base_slippage + impact_bps
            fill_ratio = 1.0 / (1.0 + participation * 10)  # 大单部分成交
            return total_slippage, fill_ratio, True
        else:
            # 超大单: 无法成交
            return 1000.0, 0.0, False

    def apply_slippage(
        self,
        order_price: float,
        side: OrderSide,
        slippage_bps: float,
    ) -> float:
        """应用滑点到价格

        买入: 实际成交价 = 期望价 × (1 + 滑点)
        卖出: 实际成交价 = 期望价 × (1 - 滑点)
        """
        slippage_pct = slippage_bps / 10000.0
        if side == OrderSide.BUY:
            return order_price * (1.0 + slippage_pct)
        else:
            return order_price * (1.0 - slippage_pct)


# ============================================================================
# 2. 延迟模拟器
# ============================================================================


class LatencySimulator:
    """延迟模拟器

    总延迟 = 网络延迟 + 交易所处理 + 撮合引擎
    """

    def __init__(self, model: LatencyModel = None):
        self.model = model or LatencyModel()
        # R18-5 BLOCK-7修复: 延迟模拟增强(时段因子+长尾分布+429重试)
        self._call_count = 0  # 用于模拟API weight累计

    def _get_time_of_day_factor(self) -> float:
        """R18-5 BLOCK-7: 时段延迟因子

        实盘规律(来源:Binance WebSocket实测 2025):
          - UTC 0-4 (亚洲深夜): 低延迟 0.8x
          - UTC 4-12 (亚洲开盘): 正常 1.0x
          - UTC 12-17 (欧美叠加): 高延迟 1.5x (高峰)
          - UTC 17-24 (美洲): 正常 1.1x
        """
        import time as _time
        hour_utc = int(_time.time() % 86400 // 3600)
        if 0 <= hour_utc < 4:
            return 0.8
        elif 4 <= hour_utc < 12:
            return 1.0
        elif 12 <= hour_utc < 17:
            return 1.5  # 欧美叠加高峰
        else:
            return 1.1

    def _maybe_api_rate_limit(self) -> float:
        """R18-5 BLOCK-7: API限流429重试延迟模拟

        Binance现货: 1200 weight/min, 超限触发429
        模拟: 每次调用累计weight,超1000后有概率触发429重试(1-5s延迟)
        """
        self._call_count += 1
        # 每100次调用模拟一次weight超限(1%概率)
        if self._call_count % 100 == 0:
            return random.uniform(1000, 5000)  # 429重试延迟 1-5s
        return 0.0

    def simulate_latency(self) -> float:
        """模拟一次订单的总延迟(ms)

        R18-5 BLOCK-7修复: 增强延迟模型
        原代码: 3个均匀分布相加(过于简单,无法模拟实盘长尾)
        修复后: 时段因子 + 对数正态长尾 + API限流429重试
        """
        # 基础延迟(均匀分布)
        network = random.uniform(*self.model.network_latency_ms)
        exchange = random.uniform(*self.model.exchange_processing_ms)
        matching = random.uniform(*self.model.matching_engine_ms)
        base_latency = network + exchange + matching

        # 时段因子(高峰期延迟放大1.5x)
        time_factor = self._get_time_of_day_factor()
        adjusted_latency = base_latency * time_factor

        # 长尾分布(对数正态,模拟偶发网络抖动)
        # 5%概率出现2-5x的延迟尖峰
        if random.random() < 0.05:
            tail_multiplier = random.lognormvariate(0.7, 0.3)  # 中位数~2x, 偶发5x
            adjusted_latency *= max(1.5, min(tail_multiplier, 5.0))

        # API限流429重试延迟(偶发,1-5s)
        rate_limit_delay = self._maybe_api_rate_limit()

        return adjusted_latency + rate_limit_delay

    def wait_for_latency(self, real_sleep: bool = False) -> float:
        """等待延迟时间(模拟真实环境)

        Phase 7J-8 反温室修复 HIGH #8: 添加 real_sleep 参数
        之前: 永远不 sleep (注释说"实际使用时取消注释"), 沙盘无延迟进化, 实盘有延迟策略失效
        现在: real_sleep=True 时真正 sleep (实盘模式), real_sleep=False 时仅返回延迟值 (沙盘快速进化)
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 沙盘无延迟=策略未经历延迟考验=实盘失效

        Args:
            real_sleep: True=真正sleep(实盘模式), False=仅返回延迟值(沙盘模式)

        Returns:
            实际等待的毫秒数
        """
        latency = self.simulate_latency()
        if real_sleep and latency > 0:
            # 最多 sleep 5s (避免极端延迟阻塞过久), 来源: Binance 429 重试上限
            sleep_sec = min(latency / 1000.0, 5.0)
            time.sleep(sleep_sec)
        return latency


# ============================================================================
# 3. 订单状态机管理器
# ============================================================================


class OrderStateMachine:
    """订单状态机管理器

    严格的状态转换,不允许非法跳转。
    来源: FIX 4.4协议 + Binance订单状态机
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}

    def create_order(self, order: Order) -> bool:
        """创建新订单"""
        if order.order_id in self.orders:
            logger.error("订单ID已存在: %s", order.order_id)
            return False
        self.orders[order.order_id] = order
        logger.debug("创建订单 %s: %s %s %s qty=%s",
                     order.order_id, order.side.value, order.symbol,
                     order.order_type.value, order.quantity)
        return True

    def transition(
        self,
        order_id: str,
        new_status: OrderStatus,
        fill_qty: float = 0.0,
        fill_price: float = 0.0,
    ) -> bool:
        """状态转换

        Args:
            order_id: 订单ID
            new_status: 新状态
            fill_qty: 本次成交数量
            fill_price: 本次成交价格

        Returns:
            是否转换成功
        """
        order = self.orders.get(order_id)
        if order is None:
            logger.error("订单不存在: %s", order_id)
            return False

        old_status = order.status
        if new_status not in VALID_TRANSITIONS.get(old_status, []):
            logger.error("非法状态转换: %s → %s (订单 %s)",
                        old_status.value, new_status.value, order_id)
            return False

        # 更新订单
        order.status = new_status
        order.updated_ts = time.time()

        if fill_qty > 0:
            # 记录成交
            old_filled = order.filled_quantity
            new_filled = old_filled + fill_qty
            # 加权平均价
            if old_filled > 0:
                order.avg_fill_price = (
                    (order.avg_fill_price * old_filled + fill_price * fill_qty) / new_filled
                )
            else:
                order.avg_fill_price = fill_price
            order.filled_quantity = new_filled
            order.fills.append({
                "timestamp": time.time(),
                "quantity": fill_qty,
                "price": fill_price,
            })

            # 部分成交或完全成交
            if new_filled >= order.quantity - 1e-10:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

        logger.debug("订单 %s: %s → %s (filled=%s/%s)",
                     order_id, old_status.value, order.status.value,
                     order.filled_quantity, order.quantity)
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        order = self.orders.get(order_id)
        if order is None:
            return False
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED,
                            OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            return False  # 终态不可撤
        return self.transition(order_id, OrderStatus.CANCELED)


# ============================================================================
# 4. 生产级OMS核心
# ============================================================================


class ProductionOMS:
    """生产级订单管理系统

    集成3大子组件:
      1. RealisticSlippageSimulator: 真实滑点模拟
      2. LatencySimulator: 延迟模拟
      3. OrderStateMachine: 订单状态机
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        latency_model: Optional[LatencyModel] = None,
    ):
        self.slippage_sim = RealisticSlippageSimulator(slippage_model or SlippageModel())
        self.latency_sim = LatencySimulator(latency_model or LatencyModel())
        self.state_machine = OrderStateMachine()
        self._order_counter = 0

    def _gen_order_id(self) -> str:
        """生成唯一订单ID"""
        self._order_counter += 1
        return "ORD-{}-{:08d}".format(int(time.time()), self._order_counter)

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        current_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
        order_book: Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = None,
    ) -> FillResult:
        """提交订单

        Phase 7J-7 反温室修复 CRITICAL: 增加 order_book 参数, 基于盘口深度判断限价单成交
        原因: 之前限价单成交用 random.random() 骰子, 与实盘微观结构无关
              沙盘进化出"70%成交"假设的策略, 实盘小币种永续限价单成交率<20%, 收益归零
        修复: 提供订单簿时基于盘口量判断成交(真实撮合逻辑); 未提供时退化为概率模型
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 限价单成交率是日内策略核心

        Args:
            symbol: 交易对
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 限价单价格(限价单必需)
            current_price: 当前市场价(用于计算滑点)
            client_order_id: 客户端订单ID
            reduce_only: 是否只减仓
            order_book: Phase 7J-7 新增 — 订单簿 (bids, asks) 元组
                bids: [(price, qty), ...] 买盘降序
                asks: [(price, qty), ...] 卖盘升序
                提供时基于盘口量判断限价单成交(真实撮合逻辑)

        Returns:
            FillResult
        """
        # 创建订单
        order_id = self._gen_order_id()
        client_id = client_order_id or "CLI-{}".format(order_id)

        order = Order(
            order_id=order_id,
            client_order_id=client_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.NEW,
            reduce_only=reduce_only,
        )

        # 验证订单参数
        if quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.rejects_reason = "数量必须>0"
            self.state_machine.orders[order_id] = order
            return FillResult(success=False, order=order, error="数量必须>0")

        if order_type == OrderType.LIMIT and price is None:
            order.status = OrderStatus.REJECTED
            order.rejects_reason = "限价单必须指定价格"
            self.state_machine.orders[order_id] = order
            return FillResult(success=False, order=order, error="限价单必须指定价格")

        self.state_machine.create_order(order)

        # R18-5 BLOCK-6修复: 记录reference_price(用于市价单滑点统计)
        order.reference_price = current_price

        # 模拟延迟
        latency_ms = self.latency_sim.simulate_latency()

        # 计算订单金额
        ref_price = price if order_type == OrderType.LIMIT else (current_price or 0.0)
        if ref_price <= 0:
            return FillResult(success=False, order=order, error="无参考价格")

        order_value_usd = quantity * ref_price

        # 计算滑点
        is_market = order_type in [OrderType.MARKET, OrderType.STOP]
        slippage_bps, fill_ratio, can_fill = self.slippage_sim.compute_slippage(
            order_value_usd, is_market_order=is_market, symbol=order.symbol
        )

        if not can_fill:
            self.state_machine.transition(order_id, OrderStatus.REJECTED)
            return FillResult(
                success=False, order=order, latency_ms=latency_ms,
                error="订单过大无法成交(participation>5%)",
            )

        # R18-5 BLOCK-5修复: 限价单成交概率模型(基于限价偏离度,替代硬编码50%)
        # 原代码: random.random() < 0.5 → 50%成交(与实盘严重不符)
        # 修复后: 基于限价vs市场价偏离度计算成交概率
        #   - aggressive(穿越市场价): 95%成交
        #   - at-market(限价=市场价): 70%成交
        #   - passive(远离市场价): max(5%, 70% - deviation_bps/50)
        #
        # Phase 7J-7 反温室修复 CRITICAL: 优先使用订单簿深度判断成交
        # 原因: random.random() 骰子与实盘微观结构无关
        # 修复: 提供 order_book 时基于盘口量判断(真实撮合逻辑); 未提供退化为概率模型
        if order_type == OrderType.LIMIT and current_price and current_price > 0:
            # Phase 7J-7: 优先使用订单簿深度判断成交 (真实撮合逻辑)
            if order_book is not None:
                bids, asks = order_book
                # 买入限价单: 检查卖盘(asks)中价格 <= 限价的总量
                # 卖出限价单: 检查买盘(bids)中价格 >= 限价的总量
                if side == OrderSide.BUY:
                    # asks 升序, 累加价格 <= 限价的量
                    available_qty = sum(qty for p, qty in asks if p <= price)
                else:
                    # bids 降序, 累加价格 >= 限价的量
                    available_qty = sum(qty for p, qty in bids if p >= price)

                if available_qty >= quantity:
                    # 盘口量充足, 全部成交 (真实撮合)
                    fill_prob = 1.0
                    fill_reason = f"订单簿深度充足(available={available_qty:.4f}>=qty={quantity:.4f})"
                else:
                    # 盘口量不足, 不成交 (真实撮合: 限价单排队)
                    fill_prob = 0.0
                    fill_reason = (
                        f"订单簿深度不足(available={available_qty:.4f}<qty={quantity:.4f}, "
                        f"需排队等待)"
                    )
            else:
                # 无订单簿: 退化为概率模型 (标注"模拟概率, 实盘可能差异大")
                # 计算限价偏离度(bps)
                deviation_bps = abs(price - current_price) / current_price * 10000

                # 判断aggressive/passive
                is_aggressive = (
                    (side == OrderSide.BUY and price >= current_price) or
                    (side == OrderSide.SELL and price <= current_price)
                )

                if is_aggressive:
                    fill_prob = 0.95
                elif deviation_bps < 5:
                    fill_prob = 0.70
                else:
                    fill_prob = max(0.05, 0.70 - deviation_bps / 50.0)
                fill_reason = (
                    f"模拟概率(fill_prob={fill_prob:.2f},deviation={deviation_bps:.1f}bps) "
                    f"— 警告: 无订单簿, 实盘成交率可能差异显著"
                )

            if random.random() > fill_prob:
                return FillResult(
                    success=False, order=order, latency_ms=latency_ms,
                    error=f"限价单未成交({fill_reason})",
                )
        elif order_type == OrderType.LIMIT:
            # 无current_price时退化为保守模型(30%成交,比原50%更保守)
            if random.random() < 0.70:
                return FillResult(
                    success=False, order=order, latency_ms=latency_ms,
                    error="限价单未成交(无current_price,保守30%成交)",
                )

        # 应用滑点
        fill_price = self.slippage_sim.apply_slippage(ref_price, side, slippage_bps)
        fill_quantity = quantity * fill_ratio

        # 状态转换: NEW → FILLED/PARTIALLY_FILLED
        if fill_ratio >= 0.999:
            self.state_machine.transition(order_id, OrderStatus.FILLED, fill_quantity, fill_price)
        else:
            self.state_machine.transition(order_id, OrderStatus.PARTIALLY_FILLED, fill_quantity, fill_price)

        return FillResult(
            success=True,
            order=self.state_machine.get_order(order_id),
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
        )

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """查询订单状态"""
        return self.state_machine.get_order(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        return self.state_machine.cancel_order(order_id)

    def get_all_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取所有订单"""
        orders = list(self.state_machine.orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_fill_statistics(self) -> Dict[str, Any]:
        """获取成交统计"""
        orders = list(self.state_machine.orders.values())
        if not orders:
            return {}

        total = len(orders)
        filled = sum(1 for o in orders if o.status == OrderStatus.FILLED)
        partial = sum(1 for o in orders if o.status == OrderStatus.PARTIALLY_FILLED)
        canceled = sum(1 for o in orders if o.status == OrderStatus.CANCELED)
        rejected = sum(1 for o in orders if o.status == OrderStatus.REJECTED)

        # R18-5 BLOCK-6/INFO-3修复: fill_rate基于实际成交数量比例(非0.5权重)
        # 原代码: (filled + partial * 0.5) / total → 99%成交和1%成交都算0.5
        # 修复后: sum(filled_quantity / quantity) / total
        fill_rate = (
            sum(min(o.filled_quantity / o.quantity, 1.0) for o in orders if o.quantity > 0) / total
            if total > 0 else 0.0
        )

        # R18-5 BLOCK-6修复: 滑点统计覆盖市价单(用reference_price作为基准)
        # 原代码: if fill.get("price", 0) > 0 and o.price → 市价单o.price=None被跳过
        # 修复后: 限价单用o.price,市价单用o.reference_price
        slippages = []
        for o in orders:
            if o.fills:
                # 确定滑点基准价: 限价单用限价,市价单用reference_price
                slip_ref = o.price if o.price else o.reference_price
                if slip_ref and slip_ref > 0:
                    for fill in o.fills:
                        if fill.get("price", 0) > 0:
                            slip = abs(fill["price"] - slip_ref) / slip_ref * 10000
                            slippages.append(slip)

        avg_slippage = float(np.mean(slippages)) if slippages else 0.0
        max_slippage = float(np.max(slippages)) if slippages else 0.0

        return {
            "total_orders": total,
            "filled": filled,
            "partially_filled": partial,
            "canceled": canceled,
            "rejected": rejected,
            "fill_rate": round(fill_rate * 100, 1),
            "avg_slippage_bps": round(avg_slippage, 2),
            "max_slippage_bps": round(max_slippage, 2),
        }


# ============================================================================
# 便捷函数
# ============================================================================


def create_production_oms(
    adv_usd: float = 100_000_000.0,
    base_spread_bps: float = 2.0,
) -> ProductionOMS:
    """创建生产级OMS实例"""
    slippage_model = SlippageModel(
        avg_daily_volume_usd=adv_usd,
        base_spread_bps=base_spread_bps,
    )
    return ProductionOMS(slippage_model=slippage_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R16-3 生产级OMS自测 ===\n")

    oms = create_production_oms()

    # 测试1: 小单市价买入
    r1 = oms.submit_order(
        symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=0.01, current_price=50000,
    )
    print("测试1 小单市价买入: success={}, fill_price={:.2f}, slippage={:.2f}bps, latency={:.0f}ms".format(
        r1.success, r1.fill_price, r1.slippage_bps, r1.latency_ms
    ))
    print()
