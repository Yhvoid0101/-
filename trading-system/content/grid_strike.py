# -*- coding: utf-8 -*-
# v447_optimize_candidate: 此模块已标记为优化候选
# 原因: 零影响+中等重叠 30.8% > 30% (可合并)
# 建议操作: merge_or_lazy_load
# 标记时间: 2026-06-28T20:18:41
# 详情: _archive_v447/MANIFEST_v447.json

"""
网格打击模块 (Grid Strike) — Triple Barrier风控的网格交易

全网顶级成熟合约交易技术 #3：在价格区间内分层挂单，每层独立风控

核心原理（来源：Hummingbot Grid Strike 2025、Binance Grid Trading）：
  1. 在 [start_price, end_price] 区间内生成N个等距网格层级
  2. 每层独立挂买卖单 + Triple Barrier风控（止盈+止损+时间）
  3. 价格上涨触发卖单成交，下跌触发买单成交
  4. 每层独立结算，互不干扰

Triple Barrier系统（来源：Marcos López de Prado "Advances in Financial Machine Learning"）：
  1. 止盈屏障（Take Profit）：达到目标收益平仓
  2. 止损屏障（Stop Loss）：达到最大亏损平仓
  3. 时间屏障（Time Limit）：超过持仓时间平仓

策略变体：
  - 多头网格（Long Grid）：底部买入，顶部卖出（适合震荡市）
  - 空头网格（Short Grid）：顶部卖出，底部买入（适合下跌市）
  - 中性网格（Neutral Grid）：双向挂单（适合盘整市）

参考：
  - Hummingbot Grid Strike (hummingbot.org/blog/strategy-guide-grid-strike/)
  - Hummingbot Grid Executor (hummingbot.org/strategies/v2-strategies/executors/gridexecutor/)
  - López de Prado "Advances in Financial Machine Learning" 2018
  - Binance Grid Trading (binance.com/en/square/post/24240367765730)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger("hermes.grid_strike")


# ============================================================================
# 数据结构
# ============================================================================


class GridDirection(Enum):
    """网格方向"""
    LONG = "long"        # 多头网格（低买高卖）
    SHORT = "short"      # 空头网格（高卖低买）
    NEUTRAL = "neutral"  # 中性网格（双向）


class GridLevelStatus(Enum):
    """网格层级状态"""
    NOT_ACTIVE = "not_active"
    OPEN_ORDER_PLACED = "open_order_placed"     # 入场单已挂
    OPEN_ORDER_FILLED = "open_order_filled"     # 入场单已成交
    CLOSE_ORDER_PLACED = "close_order_placed"    # 出场单已挂
    COMPLETE = "complete"                        # 入场+出场都完成
    STOPPED = "stopped"                          # 风控触发停止


@dataclass(slots=True)
class TripleBarrier:
    """三重屏障风控（López de Prado）"""
    take_profit: float = 0.02          # 止盈比例（2%）
    stop_loss: float = 0.01            # 止损比例（1%）
    time_limit: float = 86400.0        # 时间限制（秒，默认24小时）
    trailing_stop: bool = False        # 是否启用追踪止损
    trailing_distance: float = 0.01    # 追踪止损距离（1%）


@dataclass(slots=True)
class GridLevel:
    """网格层级"""
    level_id: int
    price: float                              # 该层级的入场价
    order_amount: float                      # 订单数量
    side: str                                 # "buy" / "sell"
    status: GridLevelStatus = GridLevelStatus.NOT_ACTIVE
    entry_order_id: Optional[str] = None
    close_order_id: Optional[str] = None
    entry_price: float = 0.0
    entry_time: float = 0.0
    realized_pnl: float = 0.0
    high_water_mark: float = 0.0             # 最高盈利（用于追踪止损）
    barrier: TripleBarrier = field(default_factory=TripleBarrier)


@dataclass(slots=True)
class GridConfig:
    """网格配置"""
    symbol: str
    direction: GridDirection = GridDirection.NEUTRAL
    start_price: float = 0.0                  # 网格起始价
    end_price: float = 0.0                    # 网格结束价
    levels: int = 10                           # 网格层数
    order_amount: float = 0.01                 # 每层订单数量
    leverage: int = 3                          # 杠杆
    barrier: TripleBarrier = field(default_factory=TripleBarrier)
    refresh_interval: float = 10.0            # 刷新间隔（秒）


@dataclass(slots=True)
class GridStats:
    """网格统计"""
    levels_activated: int = 0
    levels_completed: int = 0
    levels_stopped: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_volume: float = 0.0
    win_rate: float = 0.0
    started_at: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0


# ============================================================================
# 网格打击引擎
# ============================================================================


class GridStrike:
    """网格打击策略引擎

    核心流程：
      1. generate_grid() — 生成网格层级
      2. activate_levels() — 激活层级（挂入场单）
      3. monitor_levels() — 监控层级状态
      4. check_triple_barrier() — 检查三重屏障
      5. close_level() — 平仓层级

    使用：
        grid = GridStrike(
            engine=engine,
            config=GridConfig(
                symbol="BTC/USDT:USDT",
                direction=GridDirection.NEUTRAL,
                start_price=49000,
                end_price=51000,
                levels=10,
                order_amount=0.01,
                leverage=3,
                barrier=TripleBarrier(
                    take_profit=0.02,
                    stop_loss=0.01,
                    time_limit=86400,
                ),
            )
        )
        grid.start()
        while grid.is_running():
            grid.monitor_levels()
            time.sleep(10)
    """

    def __init__(self, engine=None, config: GridConfig = None):
        self.engine = engine
        self.config = config or GridConfig(symbol="BTC/USDT:USDT")
        self.is_running = False
        self._levels: List[GridLevel] = []
        self._stats = GridStats()
        self._last_refresh = 0.0

    # ------------------------------------------------------------------
    # Phase 1: 生成网格
    # ------------------------------------------------------------------

    # ==================================================================
    # Phase 14.25: 信号方法适配器 (根治"未找到任何信号方法" + Invalid price range)
    # ==================================================================

    def analyze(self, ctx=None):
        """Phase 14.25: 信号方法适配器 — 让GridStrike兼容strategy_registry

        根因修复:
        1. GridStrike无标准信号方法(analyze/scan等),导致strategy_registry
           遍历_METHOD_PRIORITY全部失败,触发"未找到任何信号方法"warning
        2. GridConfig.start_price/end_price默认0,generate_grid()检查
           start_price<=0时报"Invalid price range: 0.00-0.00"

        适配逻辑:
        - 从ctx.close自动发现当前价格(若start_price/end_price为0)
        - 以当前价格为中心,±5%作为网格区间
        - 惰性生成网格(首次调用时)
        - 返回中性信号(网格策略是持续挂单,非信号驱动)

        Args:
            ctx: MarketContext (strategy_registry传入), 含close/symbol/timestamp

        Returns:
            dict: 信号字典, _normalize_result转换为Signal
        """
        try:
            # 从ctx获取当前价格
            current_price = 0.0
            symbol = self.config.symbol if self.config else "BTC-USDT"
            if ctx is not None:
                current_price = float(getattr(ctx, "close", 0.0) or 0.0)
                if hasattr(ctx, "symbol") and ctx.symbol:
                    symbol = ctx.symbol

            # Phase 14.25 根因修复: 自动设置start_price/end_price
            # 当GridConfig未配置价格区间时(默认0),从当前价格推导
            if self.config and self.config.start_price <= 0 and current_price > 0:
                # 以当前价格为中心,±5%作为网格区间
                self.config.start_price = current_price * 0.95
                self.config.end_price = current_price * 1.05
                logger.info(
                    "Phase 14.25 GridStrike auto-set price range: %.2f-%.2f (current=%.2f)",
                    self.config.start_price, self.config.end_price, current_price,
                )

            # 惰性生成网格(首次调用或网格为空时)
            if self.config and not self._levels and self.config.start_price > 0:
                self.generate_grid()

            # 网格策略是持续挂单模式,返回中性信号
            # confidence低=不强制开仓,让进化系统评估网格本身的PnL
            return {
                "signal": "neutral",
                "confidence": 0.3,
                "reason": "grid_active",
                "symbol": symbol,
                "grid_levels": len(self._levels),
            }
        except Exception as e:
            logger.debug("GridStrike.analyze异常: %s", e, exc_info=False)
            return {"signal": "neutral", "confidence": 0.0, "reason": f"error: {e}"}

    def update(self, ctx=None):
        """Phase 14.25: 通用update适配器(兼容_METHOD_PRIORITY中的update方法名)"""
        return self.analyze(ctx)
    def generate_grid(self) -> List[GridLevel]:
        """生成网格层级"""
        if self.config.start_price <= 0 or self.config.end_price <= 0:
            logger.error("Invalid price range: %.2f - %.2f",
                         self.config.start_price, self.config.end_price)
            return []

        if self.config.start_price >= self.config.end_price:
            logger.error("start_price must be less than end_price")
            return []

        price_step = (self.config.end_price - self.config.start_price) / (self.config.levels - 1)
        self._levels = []

        for i in range(self.config.levels):
            price = self.config.start_price + i * price_step

            # 根据方向决定订单方向
            if self.config.direction == GridDirection.LONG:
                side = "buy"
            elif self.config.direction == GridDirection.SHORT:
                side = "sell"
            else:  # NEUTRAL
                # 中性网格：交替买卖
                side = "buy" if i % 2 == 0 else "sell"

            level = GridLevel(
                level_id=i,
                price=price,
                order_amount=self.config.order_amount,
                side=side,
                barrier=TripleBarrier(
                    take_profit=self.config.barrier.take_profit,
                    stop_loss=self.config.barrier.stop_loss,
                    time_limit=self.config.barrier.time_limit,
                    trailing_stop=self.config.barrier.trailing_stop,
                    trailing_distance=self.config.barrier.trailing_distance,
                ),
            )
            self._levels.append(level)

        logger.info(
            "Generated grid: %s %d levels %.2f-%.2f step=%.2f",
            self.config.direction.value, len(self._levels),
            self.config.start_price, self.config.end_price, price_step,
        )
        return self._levels

    # ------------------------------------------------------------------
    # Phase 2: 激活层级（挂入场单）
    # ------------------------------------------------------------------

    def activate_levels(self):
        """激活所有未激活的层级"""
        if not self._levels:
            if self.config.start_price <= 0 or self.config.end_price <= 0:
                return
            self.generate_grid()

        for level in self._levels:
            if level.status == GridLevelStatus.NOT_ACTIVE:
                self._place_entry_order(level)

    def _place_entry_order(self, level: GridLevel) -> bool:
        """挂入场单"""
        try:
            # 这里应该调用engine.submit_order
            # 简化实现：直接更新状态
            level.status = GridLevelStatus.OPEN_ORDER_PLACED
            level.entry_order_id = f"grid_entry_{level.level_id}_{int(time.time())}"
            self._stats.levels_activated += 1
            return True
        except Exception as e:
            logger.error("place entry order failed: %s", str(e)[:100])
            return False

    # ------------------------------------------------------------------
    # Phase 3: 监控层级
    # ------------------------------------------------------------------

    def monitor_levels(self) -> Dict[str, Any]:
        """监控所有层级状态"""
        if not self.is_running:
            return {"error": "not running"}

        if time.time() - self._last_refresh < self.config.refresh_interval:
            return {"skipped": "refresh interval not reached"}

        self._last_refresh = time.time()

        try:
            # 获取当前价格
            ticker = self.engine.get_ticker(self.config.symbol) if self.engine else None
            if not ticker:
                return {"error": "no ticker"}

            current_price = ticker.last

            # 检查每个层级
            for level in self._levels:
                if level.status == GridLevelStatus.OPEN_ORDER_PLACED:
                    self._check_entry_fill(level, current_price)
                elif level.status == GridLevelStatus.OPEN_ORDER_FILLED:
                    self._check_triple_barrier(level, current_price)

            return {
                "current_price": current_price,
                "active_levels": sum(1 for l in self._levels if l.status == GridLevelStatus.OPEN_ORDER_FILLED),
                "pending_levels": sum(1 for l in self._levels if l.status == GridLevelStatus.OPEN_ORDER_PLACED),
                "completed_levels": sum(1 for l in self._levels if l.status == GridLevelStatus.COMPLETE),
                "stopped_levels": sum(1 for l in self._levels if l.status == GridLevelStatus.STOPPED),
            }
        except Exception as e:
            logger.error("monitor_levels failed: %s", str(e)[:200])
            return {"error": str(e)[:100]}

    def _check_entry_fill(self, level: GridLevel, current_price: float):
        """检查入场单是否成交"""
        if level.side == "buy" and current_price <= level.price:
            # 买单成交
            level.status = GridLevelStatus.OPEN_ORDER_FILLED
            level.entry_price = current_price
            level.entry_time = time.time()
            level.high_water_mark = current_price
            self._place_close_order(level)
        elif level.side == "sell" and current_price >= level.price:
            # 卖单成交
            level.status = GridLevelStatus.OPEN_ORDER_FILLED
            level.entry_price = current_price
            level.entry_time = time.time()
            level.high_water_mark = current_price
            self._place_close_order(level)

    def _place_close_order(self, level: GridLevel) -> bool:
        """挂出场单"""
        try:
            level.close_order_id = f"grid_close_{level.level_id}_{int(time.time())}"
            level.status = GridLevelStatus.CLOSE_ORDER_PLACED
            return True
        except Exception as e:
            logger.error("place close order failed: %s", str(e)[:100])
            return False

    # ------------------------------------------------------------------
    # Phase 4: 三重屏障检查（López de Prado）
    # ------------------------------------------------------------------

    def _check_triple_barrier(self, level: GridLevel, current_price: float):
        """检查三重屏障"""
        if level.status != GridLevelStatus.CLOSE_ORDER_PLACED and \
           level.status != GridLevelStatus.OPEN_ORDER_FILLED:
            return

        # 计算盈亏比例
        if level.side == "buy":
            pnl_ratio = (current_price - level.entry_price) / level.entry_price
        else:  # sell
            pnl_ratio = (level.entry_price - current_price) / level.entry_price

        # 更新最高盈利
        if pnl_ratio > level.high_water_mark:
            level.high_water_mark = pnl_ratio

        # 屏障1：止盈
        if pnl_ratio >= level.barrier.take_profit:
            self._close_level(level, "take_profit", current_price, pnl_ratio)
            return

        # 屏障2：止损
        if pnl_ratio <= -level.barrier.stop_loss:
            self._close_level(level, "stop_loss", current_price, pnl_ratio)
            return

        # 屏障3：追踪止损
        if level.barrier.trailing_stop:
            drawdown_from_peak = level.high_water_mark - pnl_ratio
            if drawdown_from_peak >= level.barrier.trailing_distance and level.high_water_mark > 0:
                self._close_level(level, "trailing_stop", current_price, pnl_ratio)
                return

        # 屏障4：时间限制
        elapsed = time.time() - level.entry_time
        if elapsed >= level.barrier.time_limit:
            self._close_level(level, "time_limit", current_price, pnl_ratio)
            return

    def _close_level(self, level: GridLevel, reason: str,
                     close_price: float, pnl_ratio: float):
        """平仓层级"""
        level.status = GridLevelStatus.COMPLETE if reason == "take_profit" else GridLevelStatus.STOPPED
        level.realized_pnl = level.order_amount * close_price * pnl_ratio
        self._stats.levels_completed += 1 if reason == "take_profit" else 0
        self._stats.levels_stopped += 1 if reason != "take_profit" else 0
        self._stats.total_pnl += level.realized_pnl

        logger.debug(
            "Grid level %d closed: reason=%s pnl=%.4f (%.2f%%)",
            level.level_id, reason, level.realized_pnl, pnl_ratio * 100,
        )

    # ------------------------------------------------------------------
    # 启动/停止
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """启动网格"""
        if self.is_running:
            return True
        if self.config.start_price <= 0 or self.config.end_price <= 0:
            logger.info(
                "Grid startup deferred: waiting for real price context for %s",
                self.config.symbol,
            )
            return False
        if not self._levels:
            self.generate_grid()
        if not self._levels:
            logger.warning("Grid startup skipped: no valid levels for %s", self.config.symbol)
            return False
        self.is_running = True
        self.activate_levels()
        self._stats = GridStats()
        logger.info(
            "Grid started: %s %s levels=%d range=%.2f-%.2f",
            self.config.symbol, self.config.direction.value,
            len(self._levels), self.config.start_price, self.config.end_price,
        )
        return True

    def stop(self, reason: str = "manual"):
        """停止网格"""
        self.is_running = False
        # 撤销所有未成交订单
        for level in self._levels:
            if level.status in (GridLevelStatus.OPEN_ORDER_PLACED, GridLevelStatus.CLOSE_ORDER_PLACED):
                level.status = GridLevelStatus.STOPPED
        logger.info("Grid stopped: reason=%s", reason)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_levels(self) -> List[Dict[str, Any]]:
        return [
            {
                "level_id": l.level_id,
                "price": l.price,
                "side": l.side,
                "status": l.status.value,
                "entry_price": l.entry_price,
                "realized_pnl": l.realized_pnl,
            }
            for l in self._levels
        ]

    def get_stats(self) -> Dict[str, Any]:
        completed = self._stats.levels_completed + self._stats.levels_stopped
        if completed > 0:
            self._stats.win_rate = self._stats.levels_completed / completed
        self._stats.uptime_seconds = time.time() - self._stats.started_at
        return {
            "symbol": self.config.symbol,
            "direction": self.config.direction.value,
            "is_running": self.is_running,
            "stats": {
                "levels_activated": self._stats.levels_activated,
                "levels_completed": self._stats.levels_completed,
                "levels_stopped": self._stats.levels_stopped,
                "total_pnl": self._stats.total_pnl,
                "win_rate": self._stats.win_rate,
                "uptime_seconds": self._stats.uptime_seconds,
            },
        }


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    "GridDirection",
    "GridLevelStatus",
    "TripleBarrier",
    "GridLevel",
    "GridConfig",
    "GridStats",
    "GridStrike",
]
