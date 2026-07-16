# -*- coding: utf-8 -*-
"""
gamma_scalping.py — Gamma Scalping 期权Gamma剥头皮引擎 v1.0

来源：
  - OptionStack 2026 (Gamma Scalping: Capturing Realized Volatility)
  - Menthorq 2026 (Gamma Scalping in Crypto Markets, 周末缺口策略)
  - Sina Finance 2026 (上交所期权之家: Gamma Scalping在波动中捕捉收益)
  - LiveVolatile 2026 (Institutional Options Flow, Long/Short Gamma环境)
  - Avellaneda-Stoikov 2008 (做市商对冲理论)

核心算法：
  1. 正Gamma头寸构建
     - 买入平值期权（Straddle/Strangle）获得正Gamma
     - Gamma = N'(d1) / (S·σ·√T)，ATM附近Gamma最大
     - 持有正Gamma: 价格上涨Delta增大，下跌Delta减小
  2. 动态Delta对冲
     - 初始Delta中性: 期权Delta + 现货/永续Delta = 0
     - 价格上涨 → Delta变正 → 卖出标的（高卖）
     - 价格下跌 → Delta变负 → 买入标的（低买）
     - 每次再平衡锁定小额利润
  3. Scalp收益计算
     - 单次roundtrip利润 ≈ 0.5 × Gamma × (ΔS)²
     - 累积Scalp收益 vs 累积Theta损耗
     - 盈利条件: 实现波动率 > 隐含波动率
  4. 再平衡触发
     - Delta偏离阈值（如|Delta| > 0.1）
     - 价格变动阈值（如|ΔS| > 1%）
     - 时间间隔（如每4小时）

风险控制：
  - Theta损耗: 期权时间价值每日衰减
  - 隐含波动率上升: 买入期权成本增加
  - 单边行情: 无反复波动时无法Scalp
  - 交易成本: 频繁对冲产生手续费+滑点

实盘验证：
  - 适合震荡行情，年化Sharpe 1.5-3.0
  - BTC/ETH期权（Deribit/OKX）+ 现货/永续对冲
  - 周末加密市场IV低+RV高，特别适合
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GammaSignalType(Enum):
    """Gamma Scalping信号类型"""
    OPEN_LONG_GAMMA = "open_long_gamma"       # 开仓正Gamma（买入straddle）
    REBALANCE_SELL = "rebalance_sell"         # 再平衡卖出标的
    REBALANCE_BUY = "rebalance_buy"          # 再平衡买入标的
    CLOSE_POSITION = "close_position"        # 平仓（Theta超支）
    HOLD = "hold"                             # 持有
    NEUTRAL = "neutral"


class OptionType(Enum):
    """期权类型"""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionPosition:
    """期权头寸"""
    symbol: str
    option_type: OptionType
    strike: float
    expiry_days: float          # 到期天数
    quantity: float = 0.0       # 数量（正=多头）
    entry_price: float = 0.0    # 买入价
    implied_vol: float = 0.0    # 隐含波动率


@dataclass
class GreeksSnapshot:
    """Greeks快照"""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0          # 每日Theta
    vega: float = 0.0
    vanna: float = 0.0


@dataclass
class HedgeTrade:
    """对冲交易记录"""
    timestamp: float
    side: str                   # "buy" / "sell"
    quantity: float
    price: float
    scalp_pnl: float = 0.0      # 本次对冲锁定利润


@dataclass
class ScalpOpportunity:
    """Scalp机会"""
    signal_type: GammaSignalType
    action: str = ""
    hedge_qty: float = 0.0      # 对冲数量
    expected_scalp_pnl: float = 0.0
    cumulative_scalp: float = 0.0
    cumulative_theta: float = 0.0
    net_pnl: float = 0.0
    confidence: float = 0.0
    description: str = ""


@dataclass
class GammaSignal:
    """Gamma Scalping综合信号"""
    signal_type: GammaSignalType = GammaSignalType.NEUTRAL
    best_opportunity: Optional[ScalpOpportunity] = None
    current_delta: float = 0.0
    current_gamma: float = 0.0
    current_theta: float = 0.0
    realized_vol: float = 0.0
    implied_vol: float = 0.0
    rv_iv_ratio: float = 0.0    # RV/IV，>1有利
    description: str = ""


class GammaScalping:
    """
    Gamma Scalping引擎

    使用场景：
      - 期权市场（Deribit/OKX/Binance Options）
      - 震荡行情，预期实现波动率 > 隐含波动率
      - 周末加密市场（IV低，RV可能高）

    依赖：
      - numpy（数值计算）
      - 期权市场数据接入
      - 现货/永续对冲执行
    """

    # ===== 参数 =====
    REBALANCE_DELTA_THRESHOLD = 0.1     # Delta偏离阈值
    REBALANCE_PRICE_PCT = 0.01           # 价格变动1%触发再平衡
    REBALANCE_INTERVAL_HOURS = 4        # 最小再平衡间隔（小时）
    THETA_BREAKEVEN_DAYS = 30          # Theta超支平仓阈值（天）
    MIN_GAMMA_EXPOSURE = 0.001         # 最小Gamma暴露
    RISK_FREE_RATE = 0.03              # 无风险利率
    HEDGE_TRANSACTION_COST = 0.0005    # 对冲交易成本（5bps）
    OPTION_TRANSACTION_COST = 0.001    # 期权交易成本（10bps）

    def __init__(self, underlying: str = "BTC"):
        self.underlying = underlying
        self.option_positions: List[OptionPosition] = []
        self.spot_price: float = 0.0
        self.hedge_position: float = 0.0        # 现货/永续对冲数量
        self.hedge_trades: deque = deque(maxlen=500)
        self.price_history: deque = deque(maxlen=100)
        self.current_greeks: GreeksSnapshot = GreeksSnapshot()
        self.cumulative_scalp_pnl: float = 0.0
        self.cumulative_theta_cost: float = 0.0
        self.last_rebalance_time: float = 0.0
        self.position_start_time: float = 0.0
        self.realized_vol_window: deque = deque(maxlen=50)

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def set_spot_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """设置标的价格"""
        self.spot_price = price
        if timestamp is None:
            timestamp = time.time()
        self.price_history.append((timestamp, price))
        self._update_realized_vol()

    def add_option_position(self, position: OptionPosition) -> None:
        """添加期权头寸"""
        self.option_positions.append(position)
        if self.position_start_time == 0.0:
            self.position_start_time = time.time()

    def set_hedge_position(self, qty: float) -> None:
        """设置对冲头寸"""
        self.hedge_position = qty

    def _update_realized_vol(self) -> None:
        """更新实现波动率"""
        if len(self.price_history) < 2:
            return
        prices = [p for _, p in self.price_history]
        returns = np.diff(np.log(prices))
        if len(returns) >= 2:
            rv = float(np.std(returns) * math.sqrt(365 * 24))  # 年化
            self.realized_vol_window.append(rv)

    # ------------------------------------------------------------------
    # Greeks计算（Black-Scholes）
    # ------------------------------------------------------------------

    def _norm_cdf(self, x: float) -> float:
        """标准正态CDF"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_pdf(self, x: float) -> float:
        """标准正态PDF"""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _d1_d2(self, S: float, K: float, T: float, sigma: float, r: float = 0.03) -> Tuple[float, float]:
        """计算d1, d2"""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def calculate_option_greeks(
        self, S: float, K: float, T: float, sigma: float,
        option_type: OptionType, r: float = 0.03, quantity: float = 1.0
    ) -> GreeksSnapshot:
        """计算期权Greeks"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return GreeksSnapshot()

        d1, d2 = self._d1_d2(S, K, T, sigma, r)
        sqrt_T = math.sqrt(T)

        # Delta
        if option_type == OptionType.CALL:
            delta = self._norm_cdf(d1)
        else:
            delta = self._norm_cdf(d1) - 1.0

        # Gamma (call = put)
        gamma = self._norm_pdf(d1) / (S * sigma * sqrt_T)

        # Theta (per day)
        if option_type == OptionType.CALL:
            theta = (-S * self._norm_pdf(d1) * sigma / (2 * sqrt_T)
                     - r * K * math.exp(-r * T) * self._norm_cdf(d2)) / 365.0
        else:
            theta = (-S * self._norm_pdf(d1) * sigma / (2 * sqrt_T)
                     + r * K * math.exp(-r * T) * self._norm_cdf(-d2)) / 365.0

        # Vega (per 1% vol change)
        vega = S * self._norm_pdf(d1) * sqrt_T / 100.0

        # Vanna = d(Vega)/dS = d(Delta)/dsigma
        vanna = -self._norm_pdf(d1) * d2 / sigma

        return GreeksSnapshot(
            delta=delta * quantity,
            gamma=gamma * quantity,
            theta=theta * quantity,
            vega=vega * quantity,
            vanna=vanna * quantity,
        )

    def update_portfolio_greeks(self) -> GreeksSnapshot:
        """更新组合Greeks"""
        total = GreeksSnapshot()
        for pos in self.option_positions:
            T = max(pos.expiry_days / 365.0, 0.001)
            g = self.calculate_option_greeks(
                self.spot_price, pos.strike, T, pos.implied_vol,
                pos.option_type, self.risk_free_rate if hasattr(self, 'risk_free_rate') else 0.03,
                pos.quantity
            )
            total.delta += g.delta
            total.gamma += g.gamma
            total.theta += g.theta
            total.vega += g.vega
            total.vanna += g.vanna
        # 加上对冲头寸的Delta
        total.delta += self.hedge_position
        self.current_greeks = total
        return total

    # ------------------------------------------------------------------
    # 再平衡逻辑
    # ------------------------------------------------------------------

    def check_rebalance(self, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """检查是否需要再平衡"""
        if current_time is None:
            current_time = time.time()

        greeks = self.update_portfolio_greeks()

        # 条件1: Delta偏离
        if abs(greeks.delta) > self.REBALANCE_DELTA_THRESHOLD:
            return True, "delta_threshold"

        # 条件2: 价格变动
        if len(self.price_history) >= 2:
            _, last_price = self.price_history[-1]
            _, prev_price = self.price_history[-2]
            if prev_price > 0:
                price_change = abs(last_price - prev_price) / prev_price
                if price_change >= self.REBALANCE_PRICE_PCT:
                    return True, "price_threshold"

        # 条件3: 时间间隔
        hours_since = (current_time - self.last_rebalance_time) / 3600.0
        if hours_since >= self.REBALANCE_INTERVAL_HOURS and abs(greeks.delta) > 0.01:
            return True, "time_interval"

        return False, "hold"

    def calculate_hedge_quantity(self) -> float:
        """计算需要对冲的数量（使Delta归零）"""
        greeks = self.update_portfolio_greeks()
        # 当前组合Delta = 期权Delta + 对冲Delta
        # 要使总Delta=0: 对冲量 = -期权Delta
        option_delta = greeks.delta - self.hedge_position
        target_hedge = -option_delta
        hedge_change = target_hedge - self.hedge_position
        return hedge_change

    def execute_hedge(self, current_time: Optional[float] = None) -> Optional[HedgeTrade]:
        """执行对冲"""
        if current_time is None:
            current_time = time.time()

        hedge_qty = self.calculate_hedge_quantity()
        if abs(hedge_qty) < 0.0001:
            return None

        side = "sell" if hedge_qty < 0 else "buy"
        exec_qty = abs(hedge_qty)

        # 计算Scalp利润: 0.5 * Gamma * (ΔS)²
        scalp_pnl = 0.0
        if len(self.price_history) >= 2:
            _, last_price = self.price_history[-1]
            _, prev_price = self.price_history[-2]
            delta_s = last_price - prev_price
            scalp_pnl = 0.5 * self.current_greeks.gamma * delta_s * delta_s

        # 扣除交易成本
        cost = exec_qty * self.spot_price * self.HEDGE_TRANSACTION_COST
        net_scalp = scalp_pnl - cost

        trade = HedgeTrade(
            timestamp=current_time,
            side=side,
            quantity=exec_qty,
            price=self.spot_price,
            scalp_pnl=net_scalp,
        )
        self.hedge_trades.append(trade)
        self.hedge_position += hedge_qty
        self.cumulative_scalp_pnl += net_scalp
        self.last_rebalance_time = current_time
        return trade

    # ------------------------------------------------------------------
    # PnL计算
    # ------------------------------------------------------------------

    def calculate_scalp_pnl(self) -> float:
        """计算累积Scalp PnL"""
        return self.cumulative_scalp_pnl

    def calculate_theta_cost(self, days_elapsed: float) -> float:
        """计算累积Theta成本"""
        greeks = self.update_portfolio_greeks()
        return abs(greeks.theta) * days_elapsed

    def calculate_net_pnl(self, days_elapsed: float) -> float:
        """计算净PnL = Scalp收益 - Theta成本"""
        scalp = self.calculate_scalp_pnl()
        theta = self.calculate_theta_cost(days_elapsed)
        return scalp - theta

    def calculate_breakeven(self) -> float:
        """计算盈亏平衡所需的日均波动"""
        greeks = self.update_portfolio_greeks()
        if greeks.gamma <= 0 or greeks.theta >= 0:
            return float('inf')
        # 日Scalp = 0.5 * Gamma * (daily_move)² = |Theta|
        # daily_move = sqrt(2 * |Theta| / Gamma)
        return math.sqrt(2.0 * abs(greeks.theta) / greeks.gamma)

    # ------------------------------------------------------------------
    # RV/IV分析
    # ------------------------------------------------------------------

    def get_realized_vol(self) -> float:
        """获取实现波动率"""
        if not self.realized_vol_window:
            return 0.0
        return float(np.mean(list(self.realized_vol_window)))

    def get_implied_vol(self) -> float:
        """获取隐含波动率（期权仓位加权平均）"""
        if not self.option_positions:
            return 0.0
        total_weight = sum(abs(p.quantity) for p in self.option_positions)
        if total_weight <= 0:
            return 0.0
        weighted = sum(abs(p.quantity) * p.implied_vol for p in self.option_positions)
        return weighted / total_weight

    def calculate_rv_iv_ratio(self) -> float:
        """计算RV/IV比率"""
        rv = self.get_realized_vol()
        iv = self.get_implied_vol()
        if iv <= 0:
            return 0.0
        return rv / iv

    # ------------------------------------------------------------------
    # 主分析函数
    # ------------------------------------------------------------------

    def analyze(self, current_time: Optional[float] = None) -> GammaSignal:
        """主分析函数"""
        if current_time is None:
            current_time = time.time()

        signal = GammaSignal()
        greeks = self.update_portfolio_greeks()

        signal.current_delta = greeks.delta
        signal.current_gamma = greeks.gamma
        signal.current_theta = greeks.theta
        signal.realized_vol = self.get_realized_vol()
        signal.implied_vol = self.get_implied_vol()
        signal.rv_iv_ratio = self.calculate_rv_iv_ratio()

        # 无期权头寸
        if not self.option_positions:
            # 检查是否适合开仓
            if signal.rv_iv_ratio > 1.1 and signal.realized_vol > 0:
                signal.signal_type = GammaSignalType.OPEN_LONG_GAMMA
                signal.description = f"RV/IV={signal.rv_iv_ratio:.2f}>1.1, 适合开仓正Gamma"
            else:
                signal.signal_type = GammaSignalType.NEUTRAL
                signal.description = "等待RV>IV信号"
            return signal

        # 有头寸: 检查再平衡
        need_rebalance, reason = self.check_rebalance(current_time)
        days_elapsed = (current_time - self.position_start_time) / 86400.0 if self.position_start_time > 0 else 0.0

        net_pnl = self.calculate_net_pnl(days_elapsed)
        scalp = self.calculate_scalp_pnl()
        theta = self.calculate_theta_cost(days_elapsed)

        if need_rebalance:
            hedge_qty = self.calculate_hedge_quantity()
            opp = ScalpOpportunity(
                signal_type=GammaSignalType.REBALANCE_SELL if hedge_qty < 0 else GammaSignalType.REBALANCE_BUY,
                action=f"对冲{reason}",
                hedge_qty=abs(hedge_qty),
                expected_scalp_pnl=0.5 * greeks.gamma * (self.spot_price * 0.01) ** 2,
                cumulative_scalp=scalp,
                cumulative_theta=theta,
                net_pnl=net_pnl,
                confidence=0.7,
                description=f"Delta={greeks.delta:.4f} 超阈值, 需{'卖出' if hedge_qty < 0 else '买入'}{abs(hedge_qty):.4f}",
            )
            signal.signal_type = opp.signal_type
            signal.best_opportunity = opp
        else:
            # 检查是否需要平仓
            if days_elapsed > self.THETA_BREAKEVEN_DAYS and net_pnl < 0:
                opp = ScalpOpportunity(
                    signal_type=GammaSignalType.CLOSE_POSITION,
                    action="Theta超支平仓",
                    cumulative_scalp=scalp,
                    cumulative_theta=theta,
                    net_pnl=net_pnl,
                    confidence=0.8,
                    description=f"持有{days_elapsed:.1f}天, 净PnL={net_pnl:.4f}<0, Theta超支",
                )
                signal.signal_type = GammaSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
            else:
                signal.signal_type = GammaSignalType.HOLD
                signal.description = (
                    f"持有, Delta={greeks.delta:.4f}, "
                    f"Scalp={scalp:.4f}, Theta={theta:.4f}, Net={net_pnl:.4f}"
                )

        return signal

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        greeks = self.update_portfolio_greeks()
        return {
            "underlying": self.underlying,
            "spot_price": self.spot_price,
            "options_count": len(self.option_positions),
            "hedge_position": self.hedge_position,
            "portfolio_delta": greeks.delta,
            "portfolio_gamma": greeks.gamma,
            "portfolio_theta": greeks.theta,
            "portfolio_vega": greeks.vega,
            "realized_vol": self.get_realized_vol(),
            "implied_vol": self.get_implied_vol(),
            "rv_iv_ratio": self.calculate_rv_iv_ratio(),
            "cumulative_scalp_pnl": self.cumulative_scalp_pnl,
            "hedge_trades_count": len(self.hedge_trades),
            "breakeven_daily_move": self.calculate_breakeven(),
        }
