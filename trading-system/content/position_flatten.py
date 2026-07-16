"""
PositionFlattenMixin — Phase 7.6 架构解耦
从 evolution_loop.py 提取的持仓平仓逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖:
      _kill_switch_flatten_all: self.population.population, self.engine, self._kill_switch_active, self.production.audit_logger
      _force_close_all_positions: self.population.population, self.engine, self.risk_control, self.frontier
  - 属性分散保留: 所有 self 属性在 Host __init__ 中初始化, Mixin 仅访问
  - 零循环依赖: 仅依赖 logger + typing

来源:
  - _kill_switch_flatten_all: Kill Switch 触发的物理逃生平仓
  - _force_close_all_positions: 进化周期结束时的强制结算平仓
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PositionFlattenMixin:
    """持仓平仓 Mixin (Phase 7.6 提取)

    Host 鸭子类型依赖 (属性分散在 Host __init__):
      - self.population.population: List[agent] (需有 agent_id 属性)
      - self.engine: 需有 get_positions/close_position/get_market/get_equity 方法
      - self._kill_switch_active: bool (写)
      - self.production.audit_logger: 需有 log_risk 方法
      - self.risk_control: 需有 post_trade_update 方法
      - self.frontier: 需有 update_trade_result 方法
    """

    def _kill_switch_flatten_all(self, reason: str) -> int:
        """Kill Switch触发的平仓回调 — 强制平仓所有Agent的所有持仓

        生产级物理逃生：当全局回撤≥12%或日内亏损≥5%时，
        立即平仓所有持仓并阻止新开仓，必须人工reset()才能恢复。
        """
        flattened = 0
        for agent in self.population.population:
            aid = agent.agent_id
            positions = self.engine.get_positions(aid)
            for symbol, trade in list(positions.items()):
                if trade.status != "open":
                    continue
                order = self.engine.close_position(aid, symbol)
                if order and order.status.value in ("filled", "partial"):
                    flattened += 1

        self._kill_switch_active = True
        self.production.audit_logger.log_risk(
            agent_id="system",
            event="KILL_SWITCH_FLATTEN",
            details={"reason": reason, "positions_flattened": flattened, "action": "halt_trading"},
            severity="critical",
        )
        logger.critical("KILL SWITCH TRIGGERED: %s | flattened %d positions", reason, flattened)
        return flattened

    def _force_close_all_positions(
        self, agent_trades: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """强制平仓所有持仓（在进化周期结束时结算）

        确保每个Agent都有交易记录，使GT-Score能正确计算。
        """
        for agent in self.population.population:
            aid = agent.agent_id
            positions = self.engine.get_positions(aid)
            for symbol, trade in list(positions.items()):
                if trade.status != "open":
                    continue
                market = self.engine.get_market(symbol)
                if not market:
                    continue
                order = self.engine.close_position(aid, symbol)
                if order and order.status.value in ("filled", "partial"):
                    agent_equity = self.engine.get_equity(aid)
                    self.risk_control.post_trade_update(
                        aid, trade.pnl, trade.quantity * trade.entry_price,
                        equity=agent_equity,
                    )
                    # 前沿增强：更新Dynamic Kelly贝叶斯（强制平仓也反馈交易结果）
                    self.frontier.update_trade_result(trade.pnl)
                    # 计算持仓K线数和时段
                    holding_bars = 0
                    if trade.entry_time > 0 and trade.exit_time > 0:
                        holding_bars = max(1, int((trade.exit_time - trade.entry_time) / 3600))
                    entry_hour = int(trade.entry_time % 86400 // 3600) if trade.entry_time > 0 else 0
                    if 0 <= entry_hour < 8:
                        session = "asian"
                    elif 8 <= entry_hour < 13:
                        session = "european"
                    elif 13 <= entry_hour < 17:
                        session = "overlap"
                    elif 17 <= entry_hour < 21:
                        session = "us"
                    else:
                        session = "late"

                    agent_trades[aid].append({
                        "pnl": trade.pnl,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "side": trade.side.value,
                        "status": trade.status,
                        "entry_timestamp": trade.entry_time,
                        "exit_timestamp": trade.exit_time,
                        "holding_bars": holding_bars,
                        "session": session,
                        # BLOCK-3/4/5修复：从trade对象获取字段，而非硬编码空值
                        "signal_source": getattr(trade, 'signal_source', ''),
                        "cascade_alerted": getattr(trade, 'cascade_alerted', False),
                        "funding_payment": getattr(trade, 'funding_payment', 0.0),
                    })
