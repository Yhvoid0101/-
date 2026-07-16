# -*- coding: utf-8 -*-
# v447_optimize_candidate: 此模块已标记为优化候选
# 原因: 零影响+中等重叠 34.6% > 30% (可合并)
# 建议操作: merge_or_lazy_load
# 标记时间: 2026-06-28T20:18:41
# 详情: _archive_v447/MANIFEST_v447.json

"""
term_structure_trading.py — 波动率期限结构交易引擎 v1.0

来源：
  - Gate 2026 (时间结构策略: 日历价差/对角价差)
  - Hypercall 2026 (Calendar Spread: term structure trade, not direction bet)
  - Thrive.fi 2026 (Crypto Volatility Trading: DVOL, IV/RV, Vol Crush)
  - Biturai 2026 (Kalender-Spread: Theta差异化套利)
  - tradinghack 2026 (IV歪み: タームストラクチャー段差)

核心算法：
  1. 期限结构形态识别
     - Contango: 远月IV > 近月IV (正常状态, 远月风险溢价)
     - Backwardation: 近月IV > 远月IV (恐慌状态, 事件驱动)
     - Flat: 近远月IV接近 (均衡状态)
  2. 日历价差策略 (Calendar Spread)
     - 卖近月ATM + 买远月ATM (同执行价)
     - 盈利: 近月Theta衰减快于远月 + 远月IV不崩
     - 最佳: 标的价格在近月到期时接近执行价
  3. 对角价差策略 (Diagonal Spread)
     - 买远月OTM + 卖近月OTM (不同执行价+到期日)
     - 方向性+时间性结合
  4. 事件驱动期限结构
     - FOMC/CPI/ETF决策前: 近月IV膨胀
     - 事件后: Vol Crush, 近月IV崩塌
     - 策略: 事件前卖近月买远月, 事件后平仓

风险控制：
  - 远月IV风险: 远月IV也下跌则亏损
  - Greeks变化: Delta/Gamma/Vega动态
  - 执行价选择: ATM最优, OTM次之

实盘验证：
  - SPX日历价差年化Sharpe 1.0-1.8
  - BTC事件驱动(减半/FOMC): 单次收益3-8%
  - Vol Crush策略: 事件后IV下降15-30 vol点
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional



class TermStructureSignalType(Enum):
    """期限结构信号类型"""
    CALENDAR_SPREAD = "calendar_spread"           # 日历价差(卖近买远)
    DIAGONAL_SPREAD = "diagonal_spread"           # 对角价差
    VOL_CRUSH_SELL = "vol_crush_sell"             # 事件前卖波动率
    VOL_EXPANSION_BUY = "vol_expansion_buy"       # 买波动率(IV<RV)
    CLOSE_POSITION = "close_position"
    HOLD = "hold"
    NEUTRAL = "neutral"


class TermStructureShape(Enum):
    """期限结构形态"""
    CONTANGO = "contango"        # 远月 > 近月 (正常)
    BACKWARDATION = "backwardation"  # 近月 > 远月 (恐慌)
    FLAT = "flat"                # 接近


@dataclass
class OptionExpiryData:
    """期权到期数据"""
    expiry_days: float            # 到期天数
    implied_vol: float            # 隐含波动率
    realized_vol: float = 0.0     # 实现波动率
    atm_price: float = 0.0        # ATM期权价格


@dataclass
class TermStructureOpportunity:
    """期限结构机会"""
    signal_type: TermStructureSignalType
    shape: TermStructureShape = TermStructureShape.FLAT
    near_term_iv: float = 0.0
    far_term_iv: float = 0.0
    iv_spread: float = 0.0          # 远月-近月 IV价差
    near_term_theta: float = 0.0
    far_term_theta: float = 0.0
    theta_advantage: float = 0.0    # Theta优势 = 近月Theta - 远月Theta
    expected_edge_usd: float = 0.0
    confidence: float = 0.0
    description: str = ""


@dataclass
class TermStructureSignal:
    """期限结构综合信号"""
    signal_type: TermStructureSignalType = TermStructureSignalType.NEUTRAL
    best_opportunity: Optional[TermStructureOpportunity] = None
    current_shape: TermStructureShape = TermStructureShape.FLAT
    current_iv_spread: float = 0.0
    description: str = ""


class TermStructureTrading:
    """
    波动率期限结构交易引擎

    使用场景：
      - 期权市场（Deribit/OKX/Binance Options）
      - 事件驱动交易（FOMC/CPI/ETF决策/减半）
      - 期限结构套利

    依赖：
      - numpy（数值计算）
      - 近月+远月期权IV数据
    """

    # ===== 阈值参数 =====
    CONTANGO_THRESHOLD = 0.02       # 远月比近月高2vol点 = Contango
    BACKWARDATION_THRESHOLD = -0.02  # 近月比远月高2vol点 = Backwardation
    VOL_CRUSH_THRESHOLD = 0.05       # 5vol点 = Vol Crush机会
    VOL_EXPANSION_THRESHOLD = -0.03  # IV<RV 3vol点 = 买vol
    EVENT_DAYS_THRESHOLD = 7         # 7天内事件 = 事件驱动
    THETA_ADVANTAGE_MIN = 0.5        # Theta优势最小值

    # ===== 策略参数 =====
    CALENDAR_STRIKE_OFFSET = 0.0     # 日历价差执行价偏移(0=ATM)
    DIAGONAL_STRIKE_OFFSET = 0.05    # 对角价差执行价偏移(5% OTM)
    MAX_POSITION_VEGA = 5000         # 最大Vega暴露

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        self.near_term: Optional[OptionExpiryData] = None
        self.far_term: Optional[OptionExpiryData] = None
        self.spot_price: float = 0.0
        self.term_structure_history: deque = deque(maxlen=100)
        self.position_side: Optional[str] = None
        self.position_start_time: float = 0.0
        self.event_days_ahead: Optional[int] = None

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def set_spot_price(self, spot: float) -> None:
        """设置现货价格"""
        self.spot_price = spot

    def set_near_term(self, expiry_days: float, iv: float, rv: float = 0.0, atm_price: float = 0.0) -> None:
        """设置近月期权数据"""
        self.near_term = OptionExpiryData(
            expiry_days=expiry_days, implied_vol=iv,
            realized_vol=rv, atm_price=atm_price,
        )

    def set_far_term(self, expiry_days: float, iv: float, rv: float = 0.0, atm_price: float = 0.0) -> None:
        """设置远月期权数据"""
        self.far_term = OptionExpiryData(
            expiry_days=expiry_days, implied_vol=iv,
            realized_vol=rv, atm_price=atm_price,
        )

    def set_event_days_ahead(self, days: int) -> None:
        """设置事件天数"""
        self.event_days_ahead = days

    def update_term_structure_history(self, iv_spread: float) -> None:
        """更新期限结构历史"""
        self.term_structure_history.append(iv_spread)

    # ------------------------------------------------------------------
    # 期限结构分析
    # ------------------------------------------------------------------

    def calculate_iv_spread(self) -> float:
        """计算IV价差 = 远月IV - 近月IV"""
        if not self.near_term or not self.far_term:
            return 0.0
        return self.far_term.implied_vol - self.near_term.implied_vol

    def identify_shape(self) -> TermStructureShape:
        """识别期限结构形态"""
        spread = self.calculate_iv_spread()
        if spread > self.CONTANGO_THRESHOLD:
            return TermStructureShape.CONTANGO
        elif spread < self.BACKWARDATION_THRESHOLD:
            return TermStructureShape.BACKWARDATION
        else:
            return TermStructureShape.FLAT

    def calculate_theta(self, expiry_data: OptionExpiryData) -> float:
        """
        计算ATM期权Theta（简化版）
        Theta ≈ -S × IV / (2 × √T) × φ(0) / 365
        φ(0) = 1/√(2π) ≈ 0.399
        """
        if expiry_data.expiry_days <= 0 or expiry_data.implied_vol <= 0:
            return 0.0
        T = expiry_data.expiry_days / 365.0
        # 简化: Theta per day ≈ -0.4 × S × IV × √(1/T) / 365
        theta = -0.4 * self.spot_price * expiry_data.implied_vol / math.sqrt(T) / 365.0
        return theta

    def calculate_theta_advantage(self) -> float:
        """计算Theta优势 = |近月Theta| - |远月Theta|"""
        if not self.near_term or not self.far_term:
            return 0.0
        near_theta = self.calculate_theta(self.near_term)
        far_theta = self.calculate_theta(self.far_term)
        # 近月Theta衰减更快(更负), 优势 = |近月| - |远月|
        return abs(near_theta) - abs(far_theta)

    def calculate_calendar_pnl(self) -> float:
        """计算日历价差预期收益"""
        if not self.near_term or not self.far_term:
            return 0.0
        theta_adv = self.calculate_theta_advantage()
        # 预期收益 ≈ Theta优势 × 持有天数
        hold_days = min(self.near_term.expiry_days, 30)
        return theta_adv * hold_days

    def calculate_vol_crush_potential(self) -> float:
        """计算Vol Crush潜力"""
        if not self.near_term or not self.far_term:
            return 0.0
        # 事件后近月IV预期下降量
        # 简化: 如果近月IV远高于远月IV, Vol Crush潜力大
        spread = self.calculate_iv_spread()
        if spread < 0:  # Backwardation
            return abs(spread)
        return 0.0

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def analyze(self) -> TermStructureSignal:
        """主分析函数"""
        signal = TermStructureSignal()

        if not self.near_term or not self.far_term:
            signal.description = "缺少期权数据"
            return signal

        iv_spread = self.calculate_iv_spread()
        shape = self.identify_shape()
        theta_adv = self.calculate_theta_advantage()

        signal.current_shape = shape
        signal.current_iv_spread = iv_spread

        # 有持仓: 检查平仓
        if self.position_side == "calendar":
            # 日历价差平仓: 近月到期或IV价差反转
            if self.near_term.expiry_days <= 1 or iv_spread > 0:
                opp = TermStructureOpportunity(
                    signal_type=TermStructureSignalType.CLOSE_POSITION,
                    shape=shape, near_term_iv=self.near_term.implied_vol,
                    far_term_iv=self.far_term.implied_vol, iv_spread=iv_spread,
                    theta_advantage=theta_adv, confidence=0.8,
                    description=f"日历价差平仓: 近月到期或IV反转",
                )
                signal.signal_type = TermStructureSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            signal.signal_type = TermStructureSignalType.HOLD
            signal.description = f"持有日历价差, IV价差={iv_spread:.4f}"
            return signal

        # 无持仓: 检查入场

        # 1. 事件驱动 Vol Crush
        if self.event_days_ahead and self.event_days_ahead <= self.EVENT_DAYS_THRESHOLD:
            if shape == TermStructureShape.BACKWARDATION:
                crush_potential = self.calculate_vol_crush_potential()
                if crush_potential > self.VOL_CRUSH_THRESHOLD:
                    opp = TermStructureOpportunity(
                        signal_type=TermStructureSignalType.VOL_CRUSH_SELL,
                        shape=shape, near_term_iv=self.near_term.implied_vol,
                        far_term_iv=self.far_term.implied_vol, iv_spread=iv_spread,
                        theta_advantage=theta_adv,
                        expected_edge_usd=crush_potential * 1000,
                        confidence=0.8,
                        description=(
                            f"事件{self.event_days_ahead}天内, Backwardation, "
                            f"Vol Crush潜力={crush_potential:.4f}, "
                            f"卖近月IV买远月IV"
                        ),
                    )
                    signal.signal_type = TermStructureSignalType.VOL_CRUSH_SELL
                    signal.best_opportunity = opp
                    return signal

        # 2. 日历价差 (Contango + Theta优势)
        if shape == TermStructureShape.CONTANGO and theta_adv > self.THETA_ADVANTAGE_MIN:
            pnl = self.calculate_calendar_pnl()
            opp = TermStructureOpportunity(
                signal_type=TermStructureSignalType.CALENDAR_SPREAD,
                shape=shape, near_term_iv=self.near_term.implied_vol,
                far_term_iv=self.far_term.implied_vol, iv_spread=iv_spread,
                theta_advantage=theta_adv,
                expected_edge_usd=pnl,
                confidence=0.7,
                description=(
                    f"Contango, Theta优势={theta_adv:.2f}, "
                    f"卖近月ATM买远月ATM, 预期收益={pnl:.2f}"
                ),
            )
            signal.signal_type = TermStructureSignalType.CALENDAR_SPREAD
            signal.best_opportunity = opp
            return signal

        # 3. Vol Expansion (IV < RV, 买波动率)
        if self.near_term.realized_vol > 0:
            iv_rv_spread = self.near_term.implied_vol - self.near_term.realized_vol
            if iv_rv_spread < self.VOL_EXPANSION_THRESHOLD:
                opp = TermStructureOpportunity(
                    signal_type=TermStructureSignalType.VOL_EXPANSION_BUY,
                    shape=shape, near_term_iv=self.near_term.implied_vol,
                    far_term_iv=self.far_term.implied_vol, iv_spread=iv_spread,
                    theta_advantage=theta_adv,
                    expected_edge_usd=abs(iv_rv_spread) * 1000,
                    confidence=0.65,
                    description=(
                        f"IV<RV, IV-RV={iv_rv_spread:.4f}, "
                        f"买Straddle赌波动率扩张"
                    ),
                )
                signal.signal_type = TermStructureSignalType.VOL_EXPANSION_BUY
                signal.best_opportunity = opp
                return signal

        # 4. 对角价差 (Backwardation但无事件)
        if shape == TermStructureShape.BACKWARDATION and theta_adv > self.THETA_ADVANTAGE_MIN:
            opp = TermStructureOpportunity(
                signal_type=TermStructureSignalType.DIAGONAL_SPREAD,
                shape=shape, near_term_iv=self.near_term.implied_vol,
                far_term_iv=self.far_term.implied_vol, iv_spread=iv_spread,
                theta_advantage=theta_adv,
                expected_edge_usd=theta_adv * 20,
                confidence=0.6,
                description=(
                    f"Backwardation, 对角价差: "
                    f"买远月OTM卖近月OTM, Theta优势={theta_adv:.2f}"
                ),
            )
            signal.signal_type = TermStructureSignalType.DIAGONAL_SPREAD
            signal.best_opportunity = opp
            return signal

        signal.description = (
            f"观望: shape={shape.value}, IV价差={iv_spread:.4f}, "
            f"Theta优势={theta_adv:.2f}"
        )
        return signal

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "near_term_iv": self.near_term.implied_vol if self.near_term else 0,
            "near_term_days": self.near_term.expiry_days if self.near_term else 0,
            "far_term_iv": self.far_term.implied_vol if self.far_term else 0,
            "far_term_days": self.far_term.expiry_days if self.far_term else 0,
            "iv_spread": self.calculate_iv_spread(),
            "term_structure_shape": self.identify_shape().value,
            "theta_advantage": self.calculate_theta_advantage(),
            "event_days_ahead": self.event_days_ahead,
            "position_side": self.position_side,
            "history_count": len(self.term_structure_history),
        }
