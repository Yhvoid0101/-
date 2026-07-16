# -*- coding: utf-8 -*-
"""
海龟交易法则深度集成模块 (Turtle Trading System) — v2.3 实盘验证体系

借鉴经过实盘验证的成熟交易体系，防止"沙盘赚钱、实盘亏钱"。

海龟交易法则核心要素（Richard Dennis 1980s实盘验证）：
  1. Donchian通道突破（20日高点入场/10日低点出场）
     - 来源：Richard Donchian 1960 + Richard Dennis 1980s实盘
     - 实盘验证：11年PF=1.18, 夏普=0.92（原版海龟）
     - 改进版AdTurtle：62.71%年化收益，最大回撤<15%
  2. ATR动态止损（-2×ATR）
     - 来源：Wilder 1978 ATR
     - 实盘验证：截断亏损，让利润奔跑
  3. 趋势跟踪仓位管理（每0.5%权益波动=1单位）
     - 来源：Richard Dennis仓位规则
     - 实盘验证：波动率归一化仓位，避免过度杠杆
  4. 金字塔加仓（每0.5×ATR盈利加仓1单位，最大4-6单位）
     - 来源：海龟加仓规则
     - 实盘验证：趋势中加仓，震荡中不加仓
  5. 逆向思考机制（胜率30-35%正常，不预测市场）
     - 来源：趋势跟踪的本质特征
     - 实盘验证：低胜率高盈亏比，让利润奔跑

设计原则：
  - 不替换现有系统，作为"实盘验证基准"集成到决策引擎
  - 基因可以编码"海龟倾向"，进化出海龟式策略
  - 作为反过拟合的"基准策略"——如果进化出的策略不如海龟，说明过拟合
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("hermes.turtle_trading")


# ============================================================================
# 1. Donchian 通道
# ============================================================================


@dataclass
class DonchianChannel:
    """Donchian通道计算结果"""
    upper: float        # 上轨（N日最高价）
    lower: float        # 下轨（N日最低价）
    middle: float       # 中轨
    breakout_up: bool   # 突破上轨
    breakout_down: bool  # 突破下轨
    period: int         # 通道周期


class DonchianChannelCalculator:
    """Donchian通道计算器

    来源：Richard Donchian 1960 "通道趋势跟随系统"
    实盘验证：Richard Dennis 1980s海龟实验，11年PF=1.18

    系统1（短期）：20日高点入场，10日低点出场
    系统2（长期）：55日高点入场，20日低点出场
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        """
        Args:
            entry_period: 入场通道周期（默认20日）
            exit_period: 出场通道周期（默认10日）
        """
        self.entry_period = entry_period
        self.exit_period = exit_period

    def calculate(
        self,
        highs: List[float],
        lows: List[float],
        current_close: float,
    ) -> Tuple[DonchianChannel, DonchianChannel]:
        """计算入场通道和出场通道

        Args:
            highs: 历史最高价序列
            lows: 历史最低价序列
            current_close: 当前收盘价

        Returns:
            (entry_channel, exit_channel)
        """
        # 入场通道（不含当日，避免前视偏差）
        if len(highs) >= self.entry_period + 1:
            entry_highs = highs[-(self.entry_period + 1):-1]
            entry_lows = lows[-(self.entry_period + 1):-1]
        else:
            entry_highs = highs[:-1] if len(highs) > 1 else highs
            entry_lows = lows[:-1] if len(lows) > 1 else lows

        entry_upper = max(entry_highs) if entry_highs else current_close
        entry_lower = min(entry_lows) if entry_lows else current_close

        # 出场通道
        if len(highs) >= self.exit_period + 1:
            exit_highs = highs[-(self.exit_period + 1):-1]
            exit_lows = lows[-(self.exit_period + 1):-1]
        else:
            exit_highs = highs[:-1] if len(highs) > 1 else highs
            exit_lows = lows[:-1] if len(lows) > 1 else lows

        exit_upper = max(exit_highs) if exit_highs else current_close
        exit_lower = min(exit_lows) if exit_lows else current_close

        # 突破判断
        entry_breakout_up = current_close > entry_upper
        entry_breakout_down = current_close < entry_lower
        exit_breakout_up = current_close > exit_upper
        exit_breakout_down = current_close < exit_lower

        entry_channel = DonchianChannel(
            upper=entry_upper,
            lower=entry_lower,
            middle=(entry_upper + entry_lower) / 2,
            breakout_up=entry_breakout_up,
            breakout_down=entry_breakout_down,
            period=self.entry_period,
        )

        exit_channel = DonchianChannel(
            upper=exit_upper,
            lower=exit_lower,
            middle=(exit_upper + exit_lower) / 2,
            breakout_up=exit_breakout_up,
            breakout_down=exit_breakout_down,
            period=self.exit_period,
        )

        return entry_channel, exit_channel


# ============================================================================
# 2. ATR 动态止损
# ============================================================================


class ATRStopLoss:
    """ATR动态止损

    来源：Wilder 1978 "New Concepts in Technical Trading Systems"
    实盘验证：海龟交易法则核心风控，截断亏损

    止损规则：
      多头：止损 = 入场价 - N × ATR
      空头：止损 = 入场价 + N × ATR
      N通常为2（海龟标准）

    移动止损：
      价格上涨0.5×ATR → 止损上移0.5×ATR（追踪止损）
    """

    def __init__(self, atr_multiplier: float = 2.0, trailing_step: float = 0.5):
        """
        Args:
            atr_multiplier: ATR倍数（默认2.0，海龟标准）
            trailing_step: 追踪止损步长（默认0.5×ATR）
        """
        self.atr_multiplier = atr_multiplier
        self.trailing_step = trailing_step

    def compute_stop_loss(
        self,
        entry_price: float,
        atr: float,
        side: str = "long",
    ) -> float:
        """计算止损价

        Args:
            entry_price: 入场价
            atr: 当前ATR值
            side: "long" 或 "short"

        Returns:
            止损价
        """
        if side == "long":
            return entry_price - self.atr_multiplier * atr
        else:
            return entry_price + self.atr_multiplier * atr

    def compute_take_profit(
        self,
        entry_price: float,
        atr: float,
        side: str = "long",
        rr_ratio: float = 2.5,
    ) -> float:
        """计算止盈价（基于风险回报比）

        Args:
            entry_price: 入场价
            atr: 当前ATR值
            side: "long" 或 "short"
            rr_ratio: 风险回报比（默认1:2.5）

        Returns:
            止盈价
        """
        risk = self.atr_multiplier * atr
        if side == "long":
            return entry_price + rr_ratio * risk
        else:
            return entry_price - rr_ratio * risk

    def update_trailing_stop(
        self,
        current_stop: float,
        current_price: float,
        entry_price: float,
        atr: float,
        side: str = "long",
    ) -> float:
        """更新追踪止损

        Args:
            current_stop: 当前止损价
            current_price: 当前价格
            entry_price: 入场价
            atr: 当前ATR值
            side: "long" 或 "short"

        Returns:
            新的止损价（只向有利方向移动）
        """
        if side == "long":
            # 多头：价格上涨0.5×ATR，止损上移0.5×ATR
            price_gain = current_price - entry_price
            if price_gain > self.trailing_step * atr:
                new_stop = current_stop + self.trailing_step * atr
                return max(new_stop, current_stop)  # 只上移不下移
            return current_stop
        else:
            # 空头：价格下跌0.5×ATR，止损下移0.5×ATR
            price_gain = entry_price - current_price
            if price_gain > self.trailing_step * atr:
                new_stop = current_stop - self.trailing_step * atr
                return min(new_stop, current_stop)  # 只下移不上移
            return current_stop


# ============================================================================
# 3. 趋势跟踪仓位管理
# ============================================================================


class TurtlePositionSizer:
    """海龟仓位管理器

    来源：Richard Dennis仓位规则
    实盘验证：波动率归一化仓位，避免过度杠杆

    核心规则：
      1单位 = (权益 × 风险比例) / (ATR × 合约乘数)
      风险比例通常为1%（每单位最大亏损1%权益）
      最大4-6单位加仓

    公式：
      unit_size = (equity × risk_pct) / (atr_multiplier × atr)
      max_units = 4 (海龟标准) 或 6 (激进)
    """

    def __init__(
        self,
        risk_pct: float = 0.01,       # 每单位风险1%权益
        max_units: int = 4,            # 最大加仓单位数
        atr_multiplier: float = 2.0,   # ATR止损倍数
    ):
        self.risk_pct = risk_pct
        self.max_units = max_units
        self.atr_multiplier = atr_multiplier

    def compute_unit_size(
        self,
        equity: float,
        atr: float,
        price: float,
    ) -> float:
        """计算1单位的仓位大小

        Args:
            equity: 当前权益
            atr: 当前ATR值
            price: 当前价格

        Returns:
            1单位的仓位大小（数量）
        """
        if atr <= 0 or price <= 0:
            return 0.0

        # 1单位 = (权益 × 风险比例) / (ATR止损倍数 × ATR)
        risk_amount = equity * self.risk_pct
        stop_distance = self.atr_multiplier * atr
        unit_size = risk_amount / stop_distance

        # 转换为合约数量
        contracts = unit_size / price

        return max(contracts, 0.0)

    def compute_max_position(
        self,
        equity: float,
        atr: float,
        price: float,
    ) -> float:
        """计算最大仓位（所有单位加总）

        Args:
            equity: 当前权益
            atr: 当前ATR值
            price: 当前价格

        Returns:
            最大仓位大小
        """
        unit = self.compute_unit_size(equity, atr, price)
        return unit * self.max_units

    def compute_pyramid_entry(
        self,
        entry_price: float,
        atr: float,
        current_units: int,
        side: str = "long",
    ) -> Optional[float]:
        """计算金字塔加仓的触发价

        海龟规则：每0.5×ATR盈利加仓1单位

        Args:
            entry_price: 原始入场价
            atr: 当前ATR值
            current_units: 当前已加仓单位数
            side: "long" 或 "short"

        Returns:
            加仓触发价（None=不加仓）
        """
        if current_units >= self.max_units:
            return None

        # 每0.5×ATR盈利加仓
        pyramid_step = 0.5 * atr
        if side == "long":
            return entry_price + pyramid_step * current_units
        else:
            return entry_price - pyramid_step * current_units


# ============================================================================
# 4. 海龟信号系统
# ============================================================================


@dataclass
class TurtleSignal:
    """海龟交易信号"""
    action: str              # "long" / "short" / "exit_long" / "exit_short" / "hold"
    signal_type: str         # "system1" / "system2" / "exit"
    breakout_price: float    # 突破价
    stop_loss: float         # 止损价
    take_profit: float       # 止盈价
    unit_size: float         # 建议仓位单位
    confidence: float        # 信号置信度
    reasoning: str           # 信号理由


class TurtleSignalSystem:
    """海龟信号系统

    双系统架构：
      系统1（短期）：20日突破入场，10日突破出场
        - 更敏感，更多交易，更多假突破
        - 适合高波动市场
      系统2（长期）：55日突破入场，20日突破出场
        - 更稳定，更少交易，更大趋势
        - 适合趋势明确市场

    实盘验证：
      - 原版海龟11年PF=1.18, 夏普=0.92
      - AdTurtle改进版62.71%年化收益，最大回撤<15%
      - 胜率30-35%正常（趋势跟踪特征），盈亏比>2:1

    逆向思考机制：
      - 不预测市场，跟随趋势
      - 接受低胜率，追求高盈亏比
      - 截断亏损，让利润奔跑
    """

    def __init__(
        self,
        enable_system1: bool = True,
        enable_system2: bool = True,
        atr_stop: ATRStopLoss = None,
        position_sizer: TurtlePositionSizer = None,
    ):
        self.enable_system1 = enable_system1
        self.enable_system2 = enable_system2
        self.atr_stop = atr_stop or ATRStopLoss()
        self.position_sizer = position_sizer or TurtlePositionSizer()

        # 系统1：20日入场，10日出场
        self.system1 = DonchianChannelCalculator(
            entry_period=20, exit_period=10,
        ) if enable_system1 else None

        # 系统2：55日入场，20日出场
        self.system2 = DonchianChannelCalculator(
            entry_period=55, exit_period=20,
        ) if enable_system2 else None

    def generate_signal(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        current_atr: float,
        equity: float,
        current_position: str = "none",  # "none" / "long" / "short"
    ) -> Optional[TurtleSignal]:
        """生成海龟交易信号

        Args:
            highs: 历史最高价序列
            lows: 历史最低价序列
            closes: 历史收盘价序列
            current_atr: 当前ATR值
            equity: 当前权益
            current_position: 当前持仓方向

        Returns:
            TurtleSignal 或 None
        """
        if len(closes) < 60 or current_atr <= 0:
            return None

        current_close = closes[-1]
        signals: List[TurtleSignal] = []

        # 系统1信号
        if self.system1:
            sig1 = self._check_system(
                self.system1, highs, lows, current_close,
                current_atr, equity, current_position, "system1",
            )
            if sig1:
                signals.append(sig1)

        # 系统2信号
        if self.system2:
            sig2 = self._check_system(
                self.system2, highs, lows, current_close,
                current_atr, equity, current_position, "system2",
            )
            if sig2:
                signals.append(sig2)

        if not signals:
            return None

        # 优先级：出场信号 > 系统2 > 系统1
        exit_signals = [s for s in signals if s.action.startswith("exit")]
        if exit_signals:
            return exit_signals[0]

        system2_signals = [s for s in signals if s.signal_type == "system2"]
        if system2_signals:
            return system2_signals[0]

        return signals[0]

    def _check_system(
        self,
        channel_calc: DonchianChannelCalculator,
        highs: List[float],
        lows: List[float],
        current_close: float,
        atr: float,
        equity: float,
        current_position: str,
        system_name: str,
    ) -> Optional[TurtleSignal]:
        """检查单个系统的信号"""
        entry_channel, exit_channel = channel_calc.calculate(
            highs, lows, current_close,
        )

        # 出场信号优先
        if current_position == "long" and exit_channel.breakout_down:
            return TurtleSignal(
                action="exit_long",
                signal_type="exit",
                breakout_price=exit_channel.lower,
                stop_loss=0.0,
                take_profit=0.0,
                unit_size=0.0,
                confidence=0.8,
                reasoning=f"{system_name} exit: close<{exit_channel.period}d low",
            )

        if current_position == "short" and exit_channel.breakout_up:
            return TurtleSignal(
                action="exit_short",
                signal_type="exit",
                breakout_price=exit_channel.upper,
                stop_loss=0.0,
                take_profit=0.0,
                unit_size=0.0,
                confidence=0.8,
                reasoning=f"{system_name} exit: close>{exit_channel.period}d high",
            )

        # 入场信号（只在无持仓时）
        if current_position == "none":
            if entry_channel.breakout_up:
                stop_loss = self.atr_stop.compute_stop_loss(
                    current_close, atr, "long",
                )
                take_profit = self.atr_stop.compute_take_profit(
                    current_close, atr, "long",
                )
                unit_size = self.position_sizer.compute_unit_size(
                    equity, atr, current_close,
                )
                return TurtleSignal(
                    action="long",
                    signal_type=system_name,
                    breakout_price=entry_channel.upper,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    unit_size=unit_size,
                    confidence=0.6 if system_name == "system1" else 0.7,
                    reasoning=f"{system_name} entry: close>{entry_channel.period}d high",
                )

            if entry_channel.breakout_down:
                stop_loss = self.atr_stop.compute_stop_loss(
                    current_close, atr, "short",
                )
                take_profit = self.atr_stop.compute_take_profit(
                    current_close, atr, "short",
                )
                unit_size = self.position_sizer.compute_unit_size(
                    equity, atr, current_close,
                )
                return TurtleSignal(
                    action="short",
                    signal_type=system_name,
                    breakout_price=entry_channel.lower,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    unit_size=unit_size,
                    confidence=0.6 if system_name == "system1" else 0.7,
                    reasoning=f"{system_name} entry: close<{entry_channel.period}d low",
                )

        return None


# ============================================================================
# 5. 海龟交易系统综合
# ============================================================================


class TurtleTradingSystem:
    """海龟交易系统综合 — v2.3 实盘验证基准

    作为"实盘验证基准"集成到进化系统：
      1. 提供经过实盘验证的信号（Donchian突破+ATR止损）
      2. 提供波动率归一化仓位管理
      3. 作为反过拟合的基准——如果进化策略不如海龟，说明过拟合

    实盘验证数据：
      - 原版海龟11年：PF=1.18, 夏普=0.92, 胜率30-35%
      - AdTurtle改进：62.71%年化收益，最大回撤<15%
      - Barclay CTA Index 2022-2025：年化3-7%

    逆向思考机制：
      - 胜率30-35%是趋势跟踪的正常水平
      - 不追求高胜率，追求高盈亏比（>2:1）
      - 截断亏损（-2×ATR止损），让利润奔跑（10日/20日出场）
    """

    def __init__(
        self,
        enable_system1: bool = True,
        enable_system2: bool = True,
        atr_multiplier: float = 2.0,
        risk_pct: float = 0.01,
        max_units: int = 4,
    ):
        self.signal_system = TurtleSignalSystem(
            enable_system1=enable_system1,
            enable_system2=enable_system2,
            atr_stop=ATRStopLoss(atr_multiplier=atr_multiplier),
            position_sizer=TurtlePositionSizer(
                risk_pct=risk_pct,
                max_units=max_units,
                atr_multiplier=atr_multiplier,
            ),
        )
        self._price_history: deque = deque(maxlen=60)
        self._high_history: deque = deque(maxlen=60)
        self._low_history: deque = deque(maxlen=60)
        self._last_signal: Optional[TurtleSignal] = None

    def update(
        self,
        high: float,
        low: float,
        close: float,
        atr: float,
        equity: float,
        current_position: str = "none",
    ) -> Dict[str, Any]:
        """更新海龟系统并生成信号

        Args:
            high: 当前K线最高价
            low: 当前K线最低价
            close: 当前K线收盘价
            atr: 当前ATR值
            equity: 当前权益
            current_position: 当前持仓方向

        Returns:
            信号字典，包含action/confidence/stop_loss/take_profit等
        """
        self._price_history.append(close)
        self._high_history.append(high)
        self._low_history.append(low)

        # 需要足够历史数据
        if len(self._price_history) < 60:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reasoning": "insufficient_history",
            }

        highs = list(self._high_history)
        lows = list(self._low_history)
        closes = list(self._price_history)

        signal = self.signal_system.generate_signal(
            highs=highs,
            lows=lows,
            closes=closes,
            current_atr=atr,
            equity=equity,
            current_position=current_position,
        )

        if signal is None:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reasoning": "no_signal",
            }

        self._last_signal = signal

        return {
            "action": signal.action,
            "signal_type": signal.signal_type,
            "breakout_price": signal.breakout_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "unit_size": signal.unit_size,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            # 海龟特征：低胜率高盈亏比
            "expected_win_rate": 0.33,  # 趋势跟踪典型胜率
            "expected_rr_ratio": 2.5,   # 风险回报比
        }

    def get_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": "active",
            "history_length": len(self._price_history),
            "last_signal": self._last_signal.action if self._last_signal else "none",
            "last_signal_type": self._last_signal.signal_type if self._last_signal else "none",
        }


# ============================================================================
# 6. 便捷函数
# ============================================================================


def create_turtle_system(
    style: str = "balanced",
) -> TurtleTradingSystem:
    """创建预配置的海龟交易系统

    Args:
        style: "conservative" / "balanced" / "aggressive"

    Returns:
        配置好的TurtleTradingSystem
    """
    if style == "conservative":
        # 保守：只系统2，2×ATR止损，1%风险，4单位
        return TurtleTradingSystem(
            enable_system1=False,
            enable_system2=True,
            atr_multiplier=2.0,
            risk_pct=0.01,
            max_units=4,
        )
    elif style == "aggressive":
        # 激进：双系统，1.5×ATR止损，1.5%风险，6单位
        return TurtleTradingSystem(
            enable_system1=True,
            enable_system2=True,
            atr_multiplier=1.5,
            risk_pct=0.015,
            max_units=6,
        )
    else:
        # 平衡：双系统，2×ATR止损，1%风险，4单位
        return TurtleTradingSystem(
            enable_system1=True,
            enable_system2=True,
            atr_multiplier=2.0,
            risk_pct=0.01,
            max_units=4,
        )
