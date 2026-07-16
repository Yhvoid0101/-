# -*- coding: utf-8 -*-
"""
基差套利模块 (Basis Arbitrage / Cash & Carry) — 现货+永续合约基差套利

全网顶级成熟合约交易技术 #4：捕捉现货与永续合约的基差+资金费率双重收益

核心原理（来源：tradinghack.net 2025、Binance Cash & Carry、CME Basis Trading）：
  1. 当永续合约价格 > 现货价格时（升水/Contango），基差为正
  2. 买入现货（做多）+ 做空永续合约 = 锁定基差收益 + 收取资金费率
  3. 当基差收敛时（期货向现货回归），平仓获利
  4. 同时收取资金费率（永续合约空头在正费率时收取）

与资金费率套利的区别：
  - 资金费率套利：仅赚取资金费率，忽略基差
  - 基差套利：同时赚取基差收敛收益 + 资金费率（双重收益）
  - 基差套利更主动：在基差扩大时开仓，收敛时平仓

收益公式：
  总收益 = 基差收益 + 资金费率收益 - 交易成本
  基差收益 = (开仓基差 - 平仓基差) × 数量
  资金费率收益 = 持仓时间 × 资金费率 × 数量 × 价格

风险：
  - 基差扩大风险：开仓后基差继续扩大（需要追加保证金）
  - 资金费率反转风险
  - 清算风险（永续合约空头）
  - 交易所风险

参考：
  - tradinghack.net 資金調達率アービトラージ完全実務ガイド (2025.08.29)
  - tradinghack.net 永続先物の資金調達率でデルタニュートラル完全ガイド (2025.11.10)
  - CME Group Basis Trading Guide
  - Binance Futures Basis Arbitrage
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger("hermes.basis_arbitrage")


# ============================================================================
# 数据结构
# ============================================================================


class BasisStatus(Enum):
    """基差套利状态"""
    IDLE = "idle"
    SCANNING = "scanning"
    OPEN = "open"                       # 持仓中
    CLOSING = "closing"
    CLOSED = "closed"
    STOPPED = "stopped"


@dataclass(slots=True)
class BasisSnapshot:
    """基差快照"""
    symbol: str
    spot_price: float
    perp_price: float
    basis: float                          # 绝对基差 = perp - spot
    basis_ratio: float                    # 基差比率 = (perp - spot) / spot
    funding_rate: float                   # 当前资金费率
    annualized_basis: float = 0.0         # 年化基差收益
    annualized_funding: float = 0.0       # 年化资金费率收益
    total_annualized: float = 0.0        # 总年化收益
    timestamp: float = field(default_factory=time.time)

    def is_contango(self) -> bool:
        """是否为升水（期货 > 為貨）"""
        return self.perp_price > self.spot_price

    def is_backwardation(self) -> bool:
        """是否为贴水（期货 < 现货）"""
        return self.perp_price < self.spot_price


@dataclass(slots=True)
class BasisPosition:
    """基差套利持仓"""
    position_id: str
    symbol: str
    spot_exchange: str
    perp_exchange: str
    # 现货多头
    spot_qty: float
    spot_entry_price: float
    # 永续合约空头
    perp_qty: float
    perp_entry_price: float
    # 基差信息（必填）
    basis_at_open: float                          # 开仓时基差
    basis_ratio_at_open: float                    # 开仓时基差比率
    funding_rate_at_open: float                   # 开仓时资金费率
    # 可选字段
    perp_leverage: int = 1
    # 收益跟踪
    funding_collected: float = 0.0                # 累计资金费率收益
    basis_pnl: float = 0.0                        # 基差收益
    realized_pnl: float = 0.0
    # 状态
    status: BasisStatus = BasisStatus.OPEN
    opened_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    close_reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BasisConfig:
    """基差套利配置"""
    min_basis_ratio: float = 0.003         # 最低基差比率（0.3%）
    max_basis_ratio: float = 0.05          # 最高基差比率（5%，超过则风险高）
    min_funding_rate: float = 0.00005      # 最低资金费率
    target_basis_convergence: float = 0.001  # 目标基差收敛（0.1%）
    max_leverage: int = 3
    max_holding_time: float = 86400 * 7   # 最大持仓时间（7天）
    stop_loss_ratio: float = 0.02          # 止损比例（2%）


# ============================================================================
# 基差套利引擎
# ============================================================================


class BasisArbitrageur:
    """基差套利引擎

    核心流程：
      1. scan_basis() — 扫描所有交易对的基差
      2. evaluate_opportunity() — 评估套利机会
      3. open_position() — 开仓（现货买入 + 永续做空）
      4. monitor_position() — 监控基差收敛 + 资金费率
      5. close_position() — 平仓（基差收敛或达到目标）

    使用：
        arb = BasisArbitrageur(
            spot_engine=spot_engine,
            perp_engine=perp_engine,
            config=BasisConfig(
                min_basis_ratio=0.003,
                max_basis_ratio=0.05,
                min_funding_rate=0.00005,
            )
        )
        opportunities = arb.scan_basis(["BTC/USDT", "ETH/USDT"])
        if opportunities:
            arb.open_position(opportunities[0])
    """

    def __init__(self, spot_engine=None, perp_engine=None, config: BasisConfig = None):
        self.spot_engine = spot_engine
        self.perp_engine = perp_engine
        self.config = config or BasisConfig()
        self._positions: Dict[str, BasisPosition] = {}
        self._stats = {
            "opportunities_found": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "positions_stopped": 0,
            "total_basis_pnl": 0.0,
            "total_funding_collected": 0.0,
            "total_realized_pnl": 0.0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Phase 1: 扫描基差
    # ------------------------------------------------------------------

    def scan_basis(self, symbols: List[str]) -> List[BasisSnapshot]:
        """扫描所有交易对的基差"""
        snapshots = []
        for symbol in symbols:
            try:
                snap = self._get_basis_snapshot(symbol)
                if snap and self._is_opportunity(snap):
                    snapshots.append(snap)
            except Exception as e:
                logger.debug("scan %s failed: %s", symbol, str(e)[:100])

        # 按总年化收益排序
        snapshots.sort(key=lambda x: x.total_annualized, reverse=True)
        self._stats["opportunities_found"] += len(snapshots)
        return snapshots

    def _get_basis_snapshot(self, symbol: str) -> Optional[BasisSnapshot]:
        """获取单个交易对的基差快照"""
        if not self.spot_engine or not self.perp_engine:
            return None

        spot_symbol = symbol.split(":")[0] if ":" in symbol else symbol
        perp_symbol = f"{symbol}:USDT" if ":" not in symbol else symbol

        try:
            spot_ticker = self.spot_engine.get_ticker(spot_symbol)
            perp_ticker = self.perp_engine.get_ticker(perp_symbol)

            spot_price = spot_ticker.last
            perp_price = perp_ticker.last
            funding_rate = perp_ticker.funding_rate

            if spot_price <= 0:
                return None

            basis = perp_price - spot_price
            basis_ratio = basis / spot_price

            # 年化收益计算
            # 假设基差在7天内收敛
            annualized_basis = (basis_ratio / 7) * 365 if basis > 0 else 0
            # 资金费率年化（每日3次结算）
            annualized_funding = funding_rate * 3 * 365
            total_annualized = annualized_basis + annualized_funding

            return BasisSnapshot(
                symbol=symbol,
                spot_price=spot_price,
                perp_price=perp_price,
                basis=basis,
                basis_ratio=basis_ratio,
                funding_rate=funding_rate,
                annualized_basis=annualized_basis,
                annualized_funding=annualized_funding,
                total_annualized=total_annualized,
            )
        except Exception as e:
            logger.debug("get basis %s failed: %s", symbol, str(e)[:100])
            return None

    def _is_opportunity(self, snap: BasisSnapshot) -> bool:
        """判断是否为套利机会"""
        # 升水（contango）+ 正资金费率 = 双重收益
        if snap.is_contango() and snap.funding_rate >= self.config.min_funding_rate:
            if self.config.min_basis_ratio <= snap.basis_ratio <= self.config.max_basis_ratio:
                return True
        return False

    # ------------------------------------------------------------------
    # Phase 2: 开仓
    # ------------------------------------------------------------------

    def open_position(self, snapshot: BasisSnapshot,
                      capital: float = 10000.0,
                      leverage: int = 1) -> Optional[BasisPosition]:
        """开仓：现货买入 + 永续做空"""
        if leverage > self.config.max_leverage:
            leverage = self.config.max_leverage

        try:
            position_value = capital * leverage
            spot_qty = position_value / snapshot.spot_price
            perp_qty = position_value / snapshot.perp_price

            position_id = f"basis_{int(time.time())}_{snapshot.symbol.replace('/', '_')}"

            position = BasisPosition(
                position_id=position_id,
                symbol=snapshot.symbol,
                spot_exchange=self.spot_engine.name if self.spot_engine else "default",
                perp_exchange=self.perp_engine.name if self.perp_engine else "default",
                spot_qty=spot_qty,
                spot_entry_price=snapshot.spot_price,
                perp_qty=perp_qty,
                perp_entry_price=snapshot.perp_price,
                perp_leverage=leverage,
                basis_at_open=snapshot.basis,
                basis_ratio_at_open=snapshot.basis_ratio,
                funding_rate_at_open=snapshot.funding_rate,
                status=BasisStatus.OPEN,
                meta={
                    "annualized_basis": snapshot.annualized_basis,
                    "annualized_funding": snapshot.annualized_funding,
                    "total_annualized": snapshot.total_annualized,
                },
            )

            self._positions[position_id] = position
            self._stats["positions_opened"] += 1
            logger.info(
                "Opened basis position %s: %s basis=%.4f%% funding=%.6f annual=%.2f%%",
                position_id, snapshot.symbol,
                snapshot.basis_ratio * 100, snapshot.funding_rate,
                snapshot.total_annualized * 100,
            )
            return position

        except Exception as e:
            self._stats["errors"] += 1
            logger.error("open_position failed: %s", str(e)[:200])
            return None

    # ------------------------------------------------------------------
    # Phase 3: 监控持仓
    # ------------------------------------------------------------------

    def monitor_position(self, position_id: str) -> Dict[str, Any]:
        """监控基差收敛情况"""
        position = self._positions.get(position_id)
        if not position or position.status != BasisStatus.OPEN:
            return {"error": "position not found or not open"}

        try:
            # 获取当前价格
            spot_symbol = position.symbol.split(":")[0] if ":" in position.symbol else position.symbol
            perp_symbol = f"{position.symbol}:USDT" if ":" not in position.symbol else position.symbol

            spot_ticker = self.spot_engine.get_ticker(spot_symbol)
            perp_ticker = self.perp_engine.get_ticker(perp_symbol)

            current_basis = perp_ticker.last - spot_ticker.last
            current_basis_ratio = current_basis / spot_ticker.last if spot_ticker.last > 0 else 0.0

            # 计算基差收益（开仓基差 - 当前基差）
            basis_pnl = (position.basis_at_open - current_basis) * position.spot_qty
            position.basis_pnl = basis_pnl

            # 检查是否需要平仓
            should_close = False
            close_reason = ""

            # 1. 基差收敛到目标
            if abs(current_basis_ratio) <= self.config.target_basis_convergence:
                should_close = True
                close_reason = "basis_converged"

            # 2. 基差反转（变成贴水）
            elif current_basis_ratio < 0:
                should_close = True
                close_reason = "basis_reversed"

            # 3. 资金费率反转
            elif perp_ticker.funding_rate < 0:
                should_close = True
                close_reason = "funding_reversed"

            # 4. 止损
            elif basis_pnl < -position.spot_qty * position.spot_entry_price * self.config.stop_loss_ratio:
                should_close = True
                close_reason = "stop_loss"

            # 5. 持仓时间超限
            elif time.time() - position.opened_at > self.config.max_holding_time:
                should_close = True
                close_reason = "time_limit"

            return {
                "position_id": position_id,
                "current_basis": current_basis,
                "current_basis_ratio": current_basis_ratio,
                "basis_pnl": basis_pnl,
                "funding_collected": position.funding_collected,
                "should_close": should_close,
                "close_reason": close_reason,
            }
        except Exception as e:
            logger.error("monitor %s failed: %s", position_id, str(e)[:100])
            return {"error": str(e)[:100]}

    # ------------------------------------------------------------------
    # Phase 4: 平仓
    # ------------------------------------------------------------------

    def close_position(self, position_id: str, reason: str = "manual") -> Optional[BasisPosition]:
        """平仓"""
        position = self._positions.get(position_id)
        if not position:
            return None

        try:
            position.status = BasisStatus.CLOSED
            position.closed_at = time.time()
            position.close_reason = reason
            position.realized_pnl = position.basis_pnl + position.funding_collected

            self._stats["positions_closed"] += 1
            self._stats["total_basis_pnl"] += position.basis_pnl
            self._stats["total_funding_collected"] += position.funding_collected
            self._stats["total_realized_pnl"] += position.realized_pnl

            logger.info(
                "Closed basis position %s: reason=%s basis_pnl=%.4f funding=%.4f total=%.4f",
                position_id, reason, position.basis_pnl,
                position.funding_collected, position.realized_pnl,
            )
            return position
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("close_position failed: %s", str(e)[:200])
            return None

    # ------------------------------------------------------------------
    # 资金费率结算
    # ------------------------------------------------------------------

    def settle_funding(self, position_id: str) -> float:
        """结算资金费率"""
        position = self._positions.get(position_id)
        if not position or position.status != BasisStatus.OPEN:
            return 0.0

        try:
            perp_symbol = f"{position.symbol}:USDT" if ":" not in position.symbol else position.symbol
            perp_ticker = self.perp_engine.get_ticker(perp_symbol)
            funding_payment = position.perp_qty * perp_ticker.last * perp_ticker.funding_rate
            position.funding_collected += funding_payment
            return funding_payment
        except Exception as e:
            logger.error("settle_funding failed: %s", str(e)[:100])
            return 0.0

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_position(self, position_id: str) -> Optional[BasisPosition]:
        return self._positions.get(position_id)

    def get_all_positions(self) -> List[BasisPosition]:
        return list(self._positions.values())

    def get_active_positions(self) -> List[BasisPosition]:
        return [p for p in self._positions.values() if p.status == BasisStatus.OPEN]

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    "BasisStatus",
    "BasisSnapshot",
    "BasisPosition",
    "BasisConfig",
    "BasisArbitrageur",
]
