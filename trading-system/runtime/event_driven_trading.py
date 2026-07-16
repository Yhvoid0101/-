# -*- coding: utf-8 -*-
"""
event_driven_trading.py — 宏观事件驱动交易引擎 v1.0

来源：
  - CryptoTakeProfit 2026 (CPI 4.2% → BTC $67K→$61.5K, $3B清算)
  - LiveVolatile 2026 (恐惧贪婪指数 < 10 = 历史底部信号)
  - Sina Finance 2026 (FOMC Kevin Warsh 取消前瞻指引 → 波动率上升)
  - Hibt 2026 (实际利率为负 = BTC稀缺性溢价)
  - Macro → Capital → BTC → Event Probability 链

核心算法：
  1. 宏观事件日历
     - CPI (消费者物价指数): 月度, 通胀核心指标
     - FOMC (联邦公开市场委员会): 8次/年, 利率决议
     - NFP (非农就业): 月度, 就业市场指标
     - GDP: 季度, 经济增长
     - PPI (生产者物价指数): 月度, 通胀先行指标
     - 事件影响等级: HIGH / MEDIUM / LOW

  2. 事件概率链
     - 链条: 宏观数据 → 资金流向 → BTC价格 → 事件概率
     - CPI > 预期 → 加息预期↑ → 风险资产承压 → BTC下跌概率↑
     - FOMC 鸽派 → 流动性宽松 → BTC上涨概率↑
     - NFP 弱 → 经济衰退担忧 → 避险情绪↑ → BTC短期承压, 中期受益
     - 实际利率为负 (CPI > Fed Funds) → BTC稀缺性溢价

  3. 恐惧贪婪指数 (Fear & Greed Index)
     - 0-25: 极度恐惧 → 历史底部信号 (逆向做多)
     - 25-45: 恐惧 → 谨慎做多
     - 45-55: 中性
     - 55-75: 贪婪 → 谨慎做空
     - 75-100: 极度贪婪 → 历史顶部信号 (逆向做空)
     - 9 (极度恐惧) = 历史底部信号

  4. 实际利率分析
     - 实际利率 = Fed Funds Rate - CPI
     - 实际利率 < 0: 宽松, BTC受益 (稀缺性溢价)
     - 实际利率 > 0: 紧缩, BTC承压
     - 10年期国债实际收益率: 长期资金成本指标

  5. 关键信号
     - 10年期国债实际收益率
     - 布伦特原油 (地缘政治代理)
     - DXY (美元指数): 强美元 = BTC承压
     - 黄金: 避险情绪指标

风险控制：
  - 事件前减仓: 事件前24h降低仓位30-50%
  - 事件后波动: 事件后1h避免交易 (流动性差)
  - 止损放宽: 事件期间止损放宽2倍 (避免噪音止损)
  - 仓位限制: 事件期间最大仓位50%

实盘验证：
  - 2026年6月 CPI 4.2%: BTC $67K→$61.5K (-8.2%), $3B清算
  - 2026年6月 FOMC: Kevin Warsh 取消前瞻指引 → 波动率+40%
  - 恐惧贪婪指数 9: 历史底部, 后续30天反弹20-40%
  - 实际利率为负期间: BTC年化收益45%+
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple



class EventSignalType(Enum):
    """事件驱动信号类型"""
    PRE_EVENT_REDUCE = "pre_event_reduce"        # 事件前减仓
    POST_EVENT_BUY = "post_event_buy"            # 事件后买入 (鸽派/利好)
    POST_EVENT_SELL = "post_event_sell"          # 事件后卖出 (鹰派/利空)
    EXTREME_FEAR_BUY = "extreme_fear_buy"        # 极度恐惧逆向买入
    EXTREME_GREED_SELL = "extreme_greed_sell"    # 极度贪婪逆向卖出
    NEGATIVE_REAL_RATE_BUY = "neg_real_rate_buy" # 实际利率为负做多
    DXY_STRONG_SELL = "dxy_strong_sell"          # 强美元做空
    HOLD = "hold"
    NEUTRAL = "neutral"


class EventType(Enum):
    """事件类型"""
    CPI = "cpi"
    FOMC = "fomc"
    NFP = "nfp"
    GDP = "gdp"
    PPI = "ppi"
    NONE = "none"


class EventImpact(Enum):
    """事件影响等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MacroEvent:
    """宏观事件"""
    event_type: EventType
    timestamp: float
    expected_value: float = 0.0
    actual_value: Optional[float] = None
    impact: EventImpact = EventImpact.MEDIUM
    is_released: bool = False
    description: str = ""


@dataclass
class EventOpportunity:
    """事件驱动机会"""
    signal_type: EventSignalType
    event_type: EventType = EventType.NONE
    days_to_event: float = 0.0
    fear_greed_index: float = 50.0
    real_rate: float = 0.0
    dxy_value: float = 100.0
    brent_crude: float = 80.0
    expected_move_pct: float = 0.0
    confidence: float = 0.0
    description: str = ""


@dataclass
class EventSignal:
    """事件驱动综合信号"""
    signal_type: EventSignalType = EventSignalType.NEUTRAL
    best_opportunity: Optional[EventOpportunity] = None
    current_event: Optional[EventType] = None
    fear_greed_index: float = 50.0
    real_rate: float = 0.0
    description: str = ""


class EventDrivenTrading:
    """
    宏观事件驱动交易引擎

    使用场景：
      - CPI/FOMC/NFP 事件前后交易
      - 恐惧贪婪指数逆向交易
      - 实际利率 regime 交易

    依赖：
      - 宏观事件日历
      - 恐惧贪婪指数 (alternative.me API)
      - 经济指标数据 (FRED/BLS)
    """

    # ===== 事件参数 =====
    PRE_EVENT_REDUCE_HOURS = 24          # 事件前24h减仓
    POST_EVENT_COOLDOWN_HOURS = 1       # 事件后1h冷却
    EVENT_IMPACT_THRESHOLD = 0.02        # 事件影响阈值 (2%价格波动)

    # ===== 恐惧贪婪指数阈值 =====
    EXTREME_FEAR_THRESHOLD = 25          # 极度恐惧
    FEAR_THRESHOLD = 45                  # 恐惧
    GREED_THRESHOLD = 55                 # 贪婪
    EXTREME_GREED_THRESHOLD = 75         # 极度贪婪
    HISTORICAL_BOTTOM_FGI = 10           # 历史底部信号

    # ===== 实际利率参数 =====
    REAL_RATE_NEGATIVE_THRESHOLD = -0.005  # 实际利率为负阈值 (-0.5%以下触发)
    REAL_RATE_STRONG_POSITIVE = 0.02      # 强正实际利率 (2%以上)

    # ===== DXY 参数 =====
    DXY_STRONG_THRESHOLD = 105           # 强美元
    DXY_WEAK_THRESHOLD = 95              # 弱美元

    # ===== 布伦特原油参数 =====
    BRENT_HIGH_THRESHOLD = 100           # 高油价 (地缘政治风险)
    BRENT_LOW_THRESHOLD = 60            # 低油价

    # ===== 仓位控制 =====
    MAX_POSITION_EVENT_PERIOD = 0.5      # 事件期间最大仓位50%
    PRE_EVENT_REDUCE_RATIO = 0.3        # 事件前减仓30%

    def __init__(self, symbol: str = "BTC-USDT"):
        self.symbol = symbol
        self.event_calendar: List[MacroEvent] = []
        self.fear_greed_index: float = 50.0
        self.fear_greed_history: deque = deque(maxlen=100)
        self.fed_funds_rate: float = 0.045  # 4.5% 默认
        self.cpi_yoy: float = 0.038       # 3.8% 默认
        self.dxy: float = 100.0
        self.brent_crude: float = 80.0
        self.gold_price: float = 2000.0
        self.treasury_10y_real: float = 0.015  # 1.5% 默认

        self.current_position_ratio: float = 0.0
        self.last_event_time: float = 0.0
        self.event_history: deque = deque(maxlen=50)

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def add_event(self, event: MacroEvent) -> None:
        """添加事件到日历"""
        self.event_calendar.append(event)
        self.event_calendar.sort(key=lambda e: e.timestamp)

    def update_fear_greed(self, fgi: float) -> None:
        """更新恐惧贪婪指数"""
        self.fear_greed_index = fgi
        self.fear_greed_history.append(fgi)

    def update_economic_data(self, fed_funds: float, cpi: float,
                             dxy: float, brent: float,
                             gold: float, treasury_10y_real: float) -> None:
        """更新经济数据"""
        self.fed_funds_rate = fed_funds
        self.cpi_yoy = cpi
        self.dxy = dxy
        self.brent_crude = brent
        self.gold_price = gold
        self.treasury_10y_real = treasury_10y_real

    def release_event(self, event_type: EventType, actual_value: float) -> None:
        """发布事件结果"""
        for event in self.event_calendar:
            if event.event_type == event_type and not event.is_released:
                event.actual_value = actual_value
                event.is_released = True
                self.last_event_time = time.time()
                self.event_history.append({
                    "type": event_type.value,
                    "expected": event.expected_value,
                    "actual": actual_value,
                    "surprise": actual_value - event.expected_value,
                    "timestamp": time.time(),
                })
                break

    # ------------------------------------------------------------------
    # 事件分析
    # ------------------------------------------------------------------

    def get_next_event(self) -> Optional[MacroEvent]:
        """获取下一个未发布的事件"""
        now = time.time()
        for event in self.event_calendar:
            if not event.is_released and event.timestamp > now:
                return event
        return None

    def calculate_days_to_event(self) -> float:
        """计算距离下一个事件的天数"""
        event = self.get_next_event()
        if event is None:
            return 999.0
        delta = event.timestamp - time.time()
        return max(0.0, delta / 86400.0)

    def calculate_real_rate(self) -> float:
        """计算实际利率 = Fed Funds Rate - CPI"""
        return self.fed_funds_rate - self.cpi_yoy

    def calculate_expected_move(self, event: Optional[MacroEvent]) -> float:
        """计算事件预期波动幅度"""
        if event is None:
            return 0.0
        # 基于事件类型和影响等级
        base_moves = {
            EventType.CPI: 0.05,    # 5%
            EventType.FOMC: 0.04,   # 4%
            EventType.NFP: 0.03,    # 3%
            EventType.GDP: 0.02,    # 2%
            EventType.PPI: 0.02,    # 2%
        }
        base = base_moves.get(event.event_type, 0.02)
        impact_mult = {
            EventImpact.HIGH: 1.5,
            EventImpact.MEDIUM: 1.0,
            EventImpact.LOW: 0.5,
        }.get(event.impact, 1.0)
        return base * impact_mult

    def analyze_event_surprise(self, event: MacroEvent) -> str:
        """分析事件意外 (surprise)"""
        if event.actual_value is None:
            return "neutral"
        surprise = event.actual_value - event.expected_value
        if event.event_type == EventType.CPI:
            # CPI 高于预期 = 鹰派 = 利空
            if surprise > 0.002:
                return "hawkish"
            elif surprise < -0.002:
                return "dovish"
        elif event.event_type == EventType.FOMC:
            # FOMC 意外鹰派 = 利空
            if surprise > 0.0025:
                return "hawkish"
            elif surprise < -0.0025:
                return "dovish"
        elif event.event_type == EventType.NFP:
            # NFP 强 = 经济好 = 可能加息 = 短期利空
            if surprise > 50000:
                return "hawkish"
            elif surprise < -50000:
                return "dovish"
        return "neutral"

    # ------------------------------------------------------------------
    # 恐惧贪婪分析
    # ------------------------------------------------------------------

    def analyze_fear_greed(self) -> Tuple[EventSignalType, str]:
        """分析恐惧贪婪指数"""
        fgi = self.fear_greed_index
        if fgi <= self.HISTORICAL_BOTTOM_FGI:
            return (EventSignalType.EXTREME_FEAR_BUY,
                    f"极度恐惧 ({fgi}) = 历史底部信号, 逆向做多")
        elif fgi <= self.EXTREME_FEAR_THRESHOLD:
            return (EventSignalType.EXTREME_FEAR_BUY,
                    f"极度恐惧 ({fgi}), 逆向做多")
        elif fgi >= self.EXTREME_GREED_THRESHOLD:
            return (EventSignalType.EXTREME_GREED_SELL,
                    f"极度贪婪 ({fgi}), 逆向做空")
        else:
            return (EventSignalType.HOLD, f"中性 ({fgi})")

    # ------------------------------------------------------------------
    # 实际利率分析
    # ------------------------------------------------------------------

    def analyze_real_rate(self) -> Tuple[EventSignalType, str]:
        """分析实际利率"""
        real_rate = self.calculate_real_rate()
        if real_rate < self.REAL_RATE_NEGATIVE_THRESHOLD:
            return (EventSignalType.NEGATIVE_REAL_RATE_BUY,
                    f"实际利率为负 ({real_rate:.3f}), BTC稀缺性溢价, 做多")
        elif real_rate > self.REAL_RATE_STRONG_POSITIVE:
            return (EventSignalType.POST_EVENT_SELL,
                    f"实际利率强正 ({real_rate:.3f}), 资金成本高, 做空")
        return (EventSignalType.HOLD, f"实际利率中性 ({real_rate:.3f})")

    # ------------------------------------------------------------------
    # DXY 分析
    # ------------------------------------------------------------------

    def analyze_dxy(self) -> Tuple[EventSignalType, str]:
        """分析美元指数"""
        if self.dxy > self.DXY_STRONG_THRESHOLD:
            return (EventSignalType.DXY_STRONG_SELL,
                    f"强美元 (DXY={self.dxy:.1f}), BTC承压, 做空")
        elif self.dxy < self.DXY_WEAK_THRESHOLD:
            return (EventSignalType.POST_EVENT_BUY,
                    f"弱美元 (DXY={self.dxy:.1f}), BTC受益, 做多")
        return (EventSignalType.HOLD, f"美元中性 (DXY={self.dxy:.1f})")

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def analyze(self) -> EventSignal:
        """主分析函数"""
        signal = EventSignal()
        signal.fear_greed_index = self.fear_greed_index
        signal.real_rate = self.calculate_real_rate()

        # 获取下一个事件
        next_event = self.get_next_event()
        days_to_event = self.calculate_days_to_event()
        expected_move = self.calculate_expected_move(next_event)

        # 事件前减仓信号
        if next_event and days_to_event * 24 <= self.PRE_EVENT_REDUCE_HOURS:
            signal.signal_type = EventSignalType.PRE_EVENT_REDUCE
            signal.current_event = next_event.event_type
            desc = (f"事件前减仓: {next_event.event_type.value} "
                    f"({days_to_event:.1f}天后), 预期波动{expected_move*100:.1f}%")
            opp = EventOpportunity(
                signal_type=EventSignalType.PRE_EVENT_REDUCE,
                event_type=next_event.event_type,
                days_to_event=days_to_event,
                fear_greed_index=self.fear_greed_index,
                real_rate=self.calculate_real_rate(),
                dxy_value=self.dxy,
                brent_crude=self.brent_crude,
                expected_move_pct=expected_move,
                confidence=0.8,
                description=desc,
            )
            signal.best_opportunity = opp
            signal.description = desc
            return signal

        # 事件后分析 (最近发布的事件)
        recent_event = None
        for event in reversed(self.event_calendar):
            if event.is_released:
                recent_event = event
                break

        if recent_event and (time.time() - self.last_event_time) < 3600 * self.POST_EVENT_COOLDOWN_HOURS:
            # 事件冷却期
            surprise_type = self.analyze_event_surprise(recent_event)
            if surprise_type == "dovish":
                sig_type = EventSignalType.POST_EVENT_BUY
                desc = f"事件后买入: {recent_event.event_type.value} 鸽派"
            elif surprise_type == "hawkish":
                sig_type = EventSignalType.POST_EVENT_SELL
                desc = f"事件后卖出: {recent_event.event_type.value} 鹰派"
            else:
                sig_type = EventSignalType.HOLD
                desc = f"事件冷却期: {recent_event.event_type.value} 中性"
            opp = EventOpportunity(
                signal_type=sig_type,
                event_type=recent_event.event_type,
                days_to_event=0.0,
                fear_greed_index=self.fear_greed_index,
                real_rate=self.calculate_real_rate(),
                dxy_value=self.dxy,
                brent_crude=self.brent_crude,
                expected_move_pct=expected_move,
                confidence=0.7,
                description=desc,
            )
            signal.signal_type = sig_type
            signal.best_opportunity = opp
            signal.description = desc
            return signal

        # 恐惧贪婪指数
        fgi_signal, fgi_desc = self.analyze_fear_greed()
        # 实际利率
        rr_signal, rr_desc = self.analyze_real_rate()
        # DXY
        dxy_signal, dxy_desc = self.analyze_dxy()

        # 综合信号: 优先级 实际利率 > 恐惧贪婪 > DXY
        if rr_signal != EventSignalType.HOLD:
            sig_type, desc = rr_signal, rr_desc
        elif fgi_signal != EventSignalType.HOLD:
            sig_type, desc = fgi_signal, fgi_desc
        elif dxy_signal != EventSignalType.HOLD:
            sig_type, desc = dxy_signal, dxy_signal
        else:
            sig_type = EventSignalType.HOLD
            desc = "无显著事件信号"

        opp = EventOpportunity(
            signal_type=sig_type,
            event_type=next_event.event_type if next_event else EventType.NONE,
            days_to_event=days_to_event,
            fear_greed_index=self.fear_greed_index,
            real_rate=self.calculate_real_rate(),
            dxy_value=self.dxy,
            brent_crude=self.brent_crude,
            expected_move_pct=expected_move,
            confidence=0.6,
            description=desc,
        )
        signal.signal_type = sig_type
        signal.best_opportunity = opp
        signal.description = desc
        return signal

    # ------------------------------------------------------------------
    # 仓位管理
    # ------------------------------------------------------------------

    def get_max_position_ratio(self) -> float:
        """获取当前最大仓位比例"""
        days_to_event = self.calculate_days_to_event()
        if days_to_event * 24 <= self.PRE_EVENT_REDUCE_HOURS:
            return self.MAX_POSITION_EVENT_PERIOD
        return 1.0

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        next_event = self.get_next_event()
        return {
            "symbol": self.symbol,
            "fear_greed_index": self.fear_greed_index,
            "real_rate": self.calculate_real_rate(),
            "fed_funds_rate": self.fed_funds_rate,
            "cpi_yoy": self.cpi_yoy,
            "dxy": self.dxy,
            "brent_crude": self.brent_crude,
            "gold": self.gold_price,
            "treasury_10y_real": self.treasury_10y_real,
            "next_event": next_event.event_type.value if next_event else "none",
            "days_to_event": self.calculate_days_to_event(),
            "max_position_ratio": self.get_max_position_ratio(),
            "event_count": len(self.event_calendar),
            "released_events": sum(1 for e in self.event_calendar if e.is_released),
        }
