# -*- coding: utf-8 -*-
"""
seasonality_cycle.py — 季节性周期策略引擎 v1.0

来源：
  - theledgermind 2026 (Bitcoin Market Cycle 2026: Data-Driven Analysis)
  - finance.sina.com.cn 2026 (比特币500天减半规则将于2026年11月发出下一个买入信号)
  - deficryptonews 2026 (Bitcoin Post-Halving Cycle Month 26 On-Chain)
  - moneypartners 2026 (統計データから読み解くビットコインのサイクル)
  - protraderdaily 2026 (Why Bitcoin 2026 Bull Run Could Hit $250K)
  - 传统季节性效应文献 (Month-end, Weekend, Turn-of-month, Quarter-end)

核心算法：
  1. 比特币减半4年周期 (Halving Cycle)
     - 历史规律: 2012/2016/2020/2024四次减半,每4年一次
     - 减半后480-550天见顶 (历史平均535天)
     - 顶峰后~365天见底
     - 收益递减: 9592% → 2943% → 685% → 2.3-4.7x

  2. 500天Pre-Halving买入规则 (2026新发现)
     - 减半前500天买入,减半后500天卖出
     - 2018年12月($3200)/2015年初均捕捉到底部
     - 2026年11月30日 = 下一个买入窗口 (对应2028年4月减半)

  3. 链上周期指标
     - MVRV比率: >3.5见顶, 2.7=当前(2026), >3.0谨慎
     - Reserve Risk: <0.002 = 顶部预警, 0.0045=当前
     - Exchange Netflow: 流出=积累, 流入=分发

  4. 月末效应 (Month-End Effect)
     - 月末5天: 机构再平衡, 流动性变化
     - 季末效应更强 (Q1/Q2/Q3/Q4 end)
     - 12月效应: 税收损失收割 + 年末窗口装饰

  5. 周末效应 (Weekend Effect)
     - 加密周末交易量低30-50%
     - 周六/周日波动率高
     - 周一开盘gap风险

  6. 月初效应 (Turn-of-Month)
     - 每月最后1天+前3天 = 强势期
     - 养老金/工资定投流入

  7. 周内效应 (Day-of-Week)
     - 周二/周三: 趋势最强
     - 周六: 平均最弱
     - 加密24/7但法币入金工作日

性能（来源：moneypartners 2026 + theledgermind 2026）：
  - 4年周期: 减半后480-550天见顶, 顶峰后365天见底
  - 500天规则: 三轮周期均高度吻合
  - MVRV>3.5历史标记所有周期顶部
  - Reserve Risk<0.002 → 顶部预警
  - 月末/季末效应: 超额收益2-5% (传统+加密)

依赖：
  - numpy (数值计算)
  - datetime (时间处理)
  - 比特币减半历史数据
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SeasonalitySignalType(Enum):
    """季节性信号类型"""
    HALVING_ACCUMULATION = "halving_accumulation"      # 减半前积累期买入
    HALVING_MARKUP = "halving_markup"                  # 减半后上涨期持有
    HALVING_DISTRIBUTION = "halving_distribution"      # 周期顶部分发
    HALVING_BEAR = "halving_bear"                      # 周期熊市
    MONTH_END_BULLISH = "month_end_bull"               # 月末看涨
    MONTH_END_BEARISH = "month_end_bear"               # 月末看跌(税损)
    WEEKEND_RISK = "weekend_risk"                      # 周末风险
    TURN_OF_MONTH_BULLISH = "tom_bull"                 # 月初看涨
    QUARTER_END_REBALANCE = "quarter_end"              # 季末再平衡
    CYCLE_TOP_WARNING = "cycle_top_warning"            # 周期顶部预警
    CYCLE_BOTTOM_OPPORTUNITY = "cycle_bottom"          # 周期底部机会
    HOLD = "hold"
    NEUTRAL = "neutral"


class CyclePhase(Enum):
    """比特币周期阶段"""
    ACCUMULATION = "accumulation"      # 减半前积累期 (~500天)
    MARKUP = "markup"                  # 减半后上涨期 (~535天)
    DISTRIBUTION = "distribution"      # 顶部分发期 (~90天)
    BEAR_MARKET = "bear_market"        # 熊市 (~365天)
    UNKNOWN = "unknown"


@dataclass
class HalvingCycle:
    """减半周期数据"""
    date: datetime
    block_height: int
    pre_price: float
    peak_date: Optional[datetime] = None
    peak_price: float = 0.0
    peak_days_after: int = 0
    bottom_date: Optional[datetime] = None
    bottom_price: float = 0.0
    bottom_days_after_peak: int = 0
    roi_multiple: float = 0.0


@dataclass
class SeasonalityOpportunity:
    """季节性机会"""
    signal_type: SeasonalitySignalType
    phase: CyclePhase = CyclePhase.UNKNOWN
    days_to_halving: int = 0
    days_after_halving: int = 0
    days_to_peak: int = 0                    # 距预计顶峰天数 (负=已过)
    mvrv_ratio: float = 0.0
    reserve_risk: float = 0.0
    cycle_score: float = 0.0                 # 周期分数 [-1, 1]
    seasonality_score: float = 0.0           # 季节性分数 [-1, 1]
    confidence: float = 0.0
    description: str = ""


@dataclass
class SeasonalitySignal:
    """季节性综合信号"""
    signal_type: SeasonalitySignalType = SeasonalitySignalType.NEUTRAL
    best_opportunity: Optional[SeasonalityOpportunity] = None
    current_phase: CyclePhase = CyclePhase.UNKNOWN
    cycle_score: float = 0.0
    seasonality_score: float = 0.0
    days_to_next_halving: int = 0
    days_after_last_halving: int = 0
    description: str = ""


class SeasonalityCycle:
    """
    季节性周期策略引擎

    使用场景：
      - 比特币4年减半周期定位
      - 月末/季末再平衡信号
      - 周末风险预警
      - 链上周期指标(MVRV/Reserve Risk)
      - 中长期择时

    依赖：
      - numpy
      - datetime
    """

    # ===== 历史减半数据 (来源: moneypartners 2026 + theledgermind 2026) =====
    HISTORICAL_HALVINGS = [
        HalvingCycle(
            date=datetime(2012, 11, 28), block_height=210000, pre_price=12.35,
            peak_date=datetime(2013, 11, 29), peak_price=1177.0, peak_days_after=366,
            bottom_date=datetime(2015, 8, 18), bottom_price=162.0,
            bottom_days_after_peak=627, roi_multiple=95.3,
        ),
        HalvingCycle(
            date=datetime(2016, 7, 9), block_height=420000, pre_price=650.0,
            peak_date=datetime(2017, 12, 17), peak_price=19804.0, peak_days_after=526,
            bottom_date=datetime(2018, 12, 15), bottom_price=3124.0,
            bottom_days_after_peak=363, roi_multiple=30.4,
        ),
        HalvingCycle(
            date=datetime(2020, 5, 11), block_height=630000, pre_price=8821.0,
            peak_date=datetime(2021, 11, 10), peak_price=68991.0, peak_days_after=546,
            bottom_date=datetime(2022, 11, 21), bottom_price=15473.0,
            bottom_days_after_peak=376, roi_multiple=7.8,
        ),
        HalvingCycle(
            date=datetime(2024, 4, 19), block_height=840000, pre_price=64250.0,
            peak_date=datetime(2025, 10, 6), peak_price=126296.0, peak_days_after=535,
            bottom_date=None, bottom_price=0.0,
            bottom_days_after_peak=0, roi_multiple=1.97,
        ),
    ]

    # 下一次减半预测 (来源: finance.sina 2026)
    NEXT_HALVING_DATE = datetime(2028, 4, 13)
    NEXT_HALVING_BLOCK = 1050000

    # ===== 周期时间窗口 (来源: moneypartners 2026) =====
    PRE_HALVING_BUY_DAYS = 500          # 减半前500天买入
    POST_HALVING_PEAK_MIN = 480         # 减半后最少480天见顶
    POST_HALVING_PEAK_MAX = 550         # 最多550天见顶
    POST_HALVING_PEAK_AVG = 535         # 平均535天
    PEAK_TO_BOTTOM_DAYS = 365           # 顶峰后365天见底

    # ===== 链上指标阈值 (来源: theledgermind 2026) =====
    MVRV_CYCLE_TOP = 3.5                # MVRV>3.5历史标记顶部
    MVRV_CAUTION = 3.0                  # MVRV>3.0谨慎
    MVRV_FAIR_VALUE = 1.0               # 公允价值
    MVRV_UNDervalued = 1.5              # 低估
    RESERVE_RISK_TOP_WARNING = 0.002    # <0.002顶部预警
    RESERVE_RISK_BULLISH = 0.008        # 看涨
    RESERVE_RISK_NEUTRAL = 0.005        # 中性

    # ===== 月末效应 =====
    MONTH_END_WINDOW_DAYS = 5           # 月末5天
    QUARTER_END_BONUS = 0.30            # 季末额外权重
    DECEMBER_TAX_LOSS_DAYS = 10         # 12月税损收割窗口

    # ===== 周末效应 =====
    WEEKEND_VOL_PREMIUM = 0.30          # 周末波动率溢价30%
    WEEKEND_VOLUME_DROP = 0.40          # 周末交易量下降40%
    MONDAY_GAP_RISK = 0.20              # 周一开盘gap风险

    # ===== 月初效应 =====
    TURN_OF_MONTH_DAYS = 4              # 月初4天 (前月最后1天+本月前3天)
    TURN_OF_MONTH_BONUS = 0.25          # 月初效应收益加成

    # ===== 周内效应 =====
    # 历史统计: 周二/周三最强, 周六最弱
    DOW_WEIGHTS = {
        0: -0.05,   # Monday (gap风险)
        1: 0.15,    # Tuesday (最强)
        2: 0.12,    # Wednesday
        3: 0.05,    # Thursday
        4: -0.05,   # Friday
        5: -0.20,   # Saturday (最弱)
        6: -0.10,   # Sunday
    }

    # 信号阈值
    CYCLE_STRONG_BUY = 0.60
    CYCLE_BUY = 0.30
    CYCLE_SELL = -0.30
    CYCLE_STRONG_SELL = -0.60

    def __init__(self, current_date: Optional[datetime] = None):
        """
        初始化季节性周期引擎

        Args:
            current_date: 当前日期, 默认使用datetime.now()
        """
        self.current_date = current_date or datetime.now()
        self.last_halving = self.HISTORICAL_HALVINGS[-1]
        self.current_phase: CyclePhase = CyclePhase.UNKNOWN
        self.days_after_halving: int = 0
        self.days_to_next_halving: int = 0
        self.days_after_peak: int = 0
        self.estimated_peak_date: Optional[datetime] = None
        self.estimated_bottom_date: Optional[datetime] = None
        self.mvrv_ratio: float = 0.0
        self.reserve_risk: float = 0.005
        self.cycle_score: float = 0.0
        self.seasonality_score: float = 0.0
        self.signal_count: int = 0

        # 价格历史 (用于波动率分析)
        self.price_history: deque = deque(maxlen=100)
        self.returns_history: deque = deque(maxlen=100)

    def set_date(self, dt: datetime) -> None:
        """设置当前日期"""
        self.current_date = dt

    def update_onchain_metrics(self, mvrv: float, reserve_risk: float) -> None:
        """
        更新链上指标

        Args:
            mvrv: MVRV比率
            reserve_risk: Reserve Risk值
        """
        self.mvrv_ratio = float(mvrv)
        self.reserve_risk = float(reserve_risk)

    def update_price(self, price: float) -> None:
        """更新价格 (用于波动率分析)"""
        if self.price_history and self.price_history[-1] > 0:
            ret = (price - self.price_history[-1]) / self.price_history[-1]
            self.returns_history.append(ret)
        self.price_history.append(float(price))

    def _compute_cycle_phase(self) -> Tuple[CyclePhase, int, int]:
        """
        计算当前周期阶段
        返回: (phase, days_after_halving, days_to_next_halving)
        """
        days_after_halving = (self.current_date - self.last_halving.date).days
        days_to_next_halving = (self.NEXT_HALVING_DATE - self.current_date).days

        # 确定阶段
        if days_after_halving < 0:
            # 减半前: 检查是否在500天买入窗口
            if -days_after_halving <= self.PRE_HALVING_BUY_DAYS:
                return CyclePhase.ACCUMULATION, days_after_halving, days_to_next_halving
            else:
                return CyclePhase.UNKNOWN, days_after_halving, days_to_next_halving

        # 减半后: 检查是否在上涨期/分发期/熊市
        # 估计峰值日期 (使用上一个周期的实际峰值,或预测)
        if self.last_halving.peak_date is not None:
            peak_date = self.last_halving.peak_date
        else:
            # 预测峰值 = 减半后535天
            peak_date = self.last_halving.date + timedelta(days=self.POST_HALVING_PEAK_AVG)

        self.estimated_peak_date = peak_date
        days_after_peak = (self.current_date - peak_date).days
        self.days_after_peak = days_after_peak

        # 估计底部 = 峰值后365天
        bottom_date = peak_date + timedelta(days=self.PEAK_TO_BOTTOM_DAYS)
        self.estimated_bottom_date = bottom_date

        if days_after_halving < self.POST_HALVING_PEAK_MIN:
            # 减半后<480天: 上涨期
            return CyclePhase.MARKUP, days_after_halving, days_to_next_halving
        elif days_after_peak < 0:
            # 减半后480-550天但还未达峰值: 仍属上涨期
            return CyclePhase.MARKUP, days_after_halving, days_to_next_halving
        elif days_after_peak < 90:
            # 峰值后<90天: 分发期
            return CyclePhase.DISTRIBUTION, days_after_halving, days_to_next_halving
        elif days_after_peak < self.PEAK_TO_BOTTOM_DAYS:
            # 峰值后90-365天: 熊市
            return CyclePhase.BEAR_MARKET, days_after_halving, days_to_next_halving
        else:
            # 峰值后>365天: 进入下一轮积累期
            return CyclePhase.ACCUMULATION, days_after_halving, days_to_next_halving

    def _compute_cycle_score(self) -> float:
        """
        计算周期分数 [-1, 1]
        来源: 4年减半周期历史规律
        """
        score = 0.0
        phase = self.current_phase

        if phase == CyclePhase.ACCUMULATION:
            # 积累期: 看涨
            # 越接近减半越看涨
            if self.days_to_next_halving > 0:
                # 距减半越近,分数越高
                progress = 1.0 - min(self.days_to_next_halving / self.PRE_HALVING_BUY_DAYS, 1.0)
                score = 0.40 + 0.40 * progress  # 0.4 → 0.8
            else:
                score = 0.50

        elif phase == CyclePhase.MARKUP:
            # 上涨期: 强看涨
            # 早期最看涨,后期分数降低
            if self.days_after_halving < self.POST_HALVING_PEAK_MIN:
                progress = self.days_after_halving / self.POST_HALVING_PEAK_MIN
                score = 0.80 - 0.20 * progress  # 0.8 → 0.6
            else:
                # 接近峰值,分数降低
                score = 0.40

        elif phase == CyclePhase.DISTRIBUTION:
            # 分发期: 看跌
            score = -0.50

        elif phase == CyclePhase.BEAR_MARKET:
            # 熊市: 看跌,但接近底部时转中性
            if self.days_after_peak > 0:
                progress = min(self.days_after_peak / self.PEAK_TO_BOTTOM_DAYS, 1.0)
                score = -0.70 + 0.50 * progress  # -0.7 → -0.2
            else:
                score = -0.50

        # 链上指标调整
        if self.mvrv_ratio > 0:
            if self.mvrv_ratio > self.MVRV_CYCLE_TOP:
                # MVRV>3.5: 顶部信号,大幅降低分数
                score -= 0.40
            elif self.mvrv_ratio > self.MVRV_CAUTION:
                # MVRV>3.0: 谨慎
                score -= 0.20
            elif self.mvrv_ratio < self.MVRV_UNDervalued:
                # MVRV<1.5: 低估,提升分数
                score += 0.20

        if self.reserve_risk > 0:
            if self.reserve_risk < self.RESERVE_RISK_TOP_WARNING:
                # Reserve Risk<0.002: 顶部预警
                score -= 0.30
            elif self.reserve_risk > self.RESERVE_RISK_BULLISH:
                # Reserve Risk>0.008: 看涨
                score += 0.15

        return max(-1.0, min(1.0, score))

    def _compute_seasonality_score(self) -> float:
        """
        计算季节性分数 [-1, 1]
        综合: 月末/季末 + 周末 + 月初 + 周内
        """
        score = 0.0
        day_of_month = self.current_date.day
        day_of_week = self.current_date.weekday()  # 0=Monday
        month = self.current_date.month
        days_in_month = (self.current_date.replace(day=28) +
                         timedelta(days=4)).replace(day=1) - timedelta(days=1)
        days_in_month = days_in_month.day

        # 1. 月末效应 (最后5天)
        days_to_month_end = days_in_month - day_of_month
        if days_to_month_end < self.MONTH_END_WINDOW_DAYS:
            # 月末效应
            month_end_score = 0.10
            # 季末效应 (3/6/9/12月末)
            if month in [3, 6, 9, 12]:
                month_end_score += self.QUARTER_END_BONUS * 0.3
            # 12月税损收割
            if month == 12 and days_to_month_end < self.DECEMBER_TAX_LOSS_DAYS:
                month_end_score -= 0.20  # 税损卖出压力
            score += month_end_score

        # 2. 月初效应 (前4天)
        if day_of_month <= self.TURN_OF_MONTH_DAYS:
            score += self.TURN_OF_MONTH_BONUS * 0.5

        # 3. 周内效应
        dow_score = self.DOW_WEIGHTS.get(day_of_week, 0.0)
        score += dow_score * 0.3

        # 4. 周末风险调整 (周六/周日)
        if day_of_week == 5:  # Saturday
            score -= 0.15  # 周末流动性低
        elif day_of_week == 6:  # Sunday
            score -= 0.10  # 周日略有恢复

        return max(-1.0, min(1.0, score))

    def _select_signal(self) -> SeasonalitySignal:
        """选择最佳信号"""
        signal = SeasonalitySignal(
            current_phase=self.current_phase,
            cycle_score=self.cycle_score,
            seasonality_score=self.seasonality_score,
            days_to_next_halving=self.days_to_next_halving,
            days_after_last_halving=self.days_after_halving,
        )

        # 优先级1: 周期顶部预警 (MVRV>3.5 或 Reserve Risk<0.002)
        if self.mvrv_ratio > self.MVRV_CYCLE_TOP or \
           self.reserve_risk < self.RESERVE_RISK_TOP_WARNING:
            signal.signal_type = SeasonalitySignalType.CYCLE_TOP_WARNING
            signal.best_opportunity = SeasonalityOpportunity(
                signal_type=SeasonalitySignalType.CYCLE_TOP_WARNING,
                phase=self.current_phase,
                days_to_halving=self.days_to_next_halving,
                days_after_halving=self.days_after_halving,
                days_to_peak=-self.days_after_peak if self.days_after_peak > 0 else 0,
                mvrv_ratio=self.mvrv_ratio,
                reserve_risk=self.reserve_risk,
                cycle_score=self.cycle_score,
                seasonality_score=self.seasonality_score,
                confidence=0.85,
                description=f"周期顶部预警: MVRV={self.mvrv_ratio:.2f}, ReserveRisk={self.reserve_risk:.4f}"
            )
            signal.description = "链上指标显示周期顶部,建议减仓"
            return signal

        # 优先级2: 周期底部机会 (积累期 + MVRV低估)
        if (self.current_phase == CyclePhase.ACCUMULATION and
            0 < self.mvrv_ratio < self.MVRV_UNDervalued):
            signal.signal_type = SeasonalitySignalType.CYCLE_BOTTOM_OPPORTUNITY
            signal.best_opportunity = SeasonalityOpportunity(
                signal_type=SeasonalitySignalType.CYCLE_BOTTOM_OPPORTUNITY,
                phase=self.current_phase,
                days_to_halving=self.days_to_next_halving,
                days_after_halving=self.days_after_halving,
                mvrv_ratio=self.mvrv_ratio,
                reserve_risk=self.reserve_risk,
                cycle_score=self.cycle_score,
                confidence=0.80,
                description=f"周期底部机会: MVRV={self.mvrv_ratio:.2f}, 减半前{self.days_to_next_halving}天"
            )
            signal.description = "周期底部区域,建议逢低买入"
            return signal

        # 优先级3: 减半周期信号
        if self.current_phase == CyclePhase.ACCUMULATION and \
           self.days_to_next_halving <= self.PRE_HALVING_BUY_DAYS and \
           self.days_to_next_halving > 0:
            signal.signal_type = SeasonalitySignalType.HALVING_ACCUMULATION
            signal.best_opportunity = SeasonalityOpportunity(
                signal_type=SeasonalitySignalType.HALVING_ACCUMULATION,
                phase=self.current_phase,
                days_to_halving=self.days_to_next_halving,
                days_after_halving=self.days_after_halving,
                cycle_score=self.cycle_score,
                confidence=0.75,
                description=f"减半前积累期: 距下次减半{self.days_to_next_halving}天"
            )
            signal.description = "500天Pre-Halving买入窗口"
            return signal

        if self.current_phase == CyclePhase.MARKUP:
            signal.signal_type = SeasonalitySignalType.HALVING_MARKUP
            signal.best_opportunity = SeasonalityOpportunity(
                signal_type=SeasonalitySignalType.HALVING_MARKUP,
                phase=self.current_phase,
                days_after_halving=self.days_after_halving,
                days_to_halving=self.days_to_next_halving,
                cycle_score=self.cycle_score,
                confidence=0.70,
                description=f"减半后上涨期: 已过{self.days_after_halving}天, 峰值预计{self.POST_HALVING_PEAK_AVG}天"
            )
            signal.description = "减半后上涨期,持有"
            return signal

        if self.current_phase == CyclePhase.DISTRIBUTION:
            signal.signal_type = SeasonalitySignalType.HALVING_DISTRIBUTION
            signal.best_opportunity = SeasonalityOpportunity(
                signal_type=SeasonalitySignalType.HALVING_DISTRIBUTION,
                phase=self.current_phase,
                days_after_halving=self.days_after_halving,
                cycle_score=self.cycle_score,
                confidence=0.65,
                description=f"周期顶部分发期: 峰值后{self.days_after_peak}天"
            )
            signal.description = "周期顶部分发,减仓"
            return signal

        if self.current_phase == CyclePhase.BEAR_MARKET:
            signal.signal_type = SeasonalitySignalType.HALVING_BEAR
            signal.best_opportunity = SeasonalityOpportunity(
                signal_type=SeasonalitySignalType.HALVING_BEAR,
                phase=self.current_phase,
                days_after_halving=self.days_after_halving,
                cycle_score=self.cycle_score,
                confidence=0.60,
                description=f"熊市期: 峰值后{self.days_after_peak}天, 预计{self.PEAK_TO_BOTTOM_DAYS}天见底"
            )
            signal.description = "熊市期,空仓等待"
            return signal

        # 优先级4: 季节性信号 (短期)
        if self.seasonality_score > 0.20:
            day_of_month = self.current_date.day
            days_in_month = (self.current_date.replace(day=28) +
                             timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_in_month = days_in_month.day
            days_to_month_end = days_in_month - day_of_month

            if day_of_month <= self.TURN_OF_MONTH_DAYS:
                signal.signal_type = SeasonalitySignalType.TURN_OF_MONTH_BULLISH
                signal.best_opportunity = SeasonalityOpportunity(
                    signal_type=SeasonalitySignalType.TURN_OF_MONTH_BULLISH,
                    seasonality_score=self.seasonality_score,
                    confidence=0.55,
                    description=f"月初效应: 月第{day_of_month}天, 历史强势期"
                )
                signal.description = "月初效应,短期看涨"
                return signal

            if days_to_month_end < self.MONTH_END_WINDOW_DAYS:
                if self.current_date.month in [3, 6, 9, 12]:
                    signal.signal_type = SeasonalitySignalType.QUARTER_END_REBALANCE
                    signal.best_opportunity = SeasonalityOpportunity(
                        signal_type=SeasonalitySignalType.QUARTER_END_REBALANCE,
                        seasonality_score=self.seasonality_score,
                        confidence=0.60,
                        description=f"季末再平衡: {self.current_date.month}月最后{days_to_month_end}天"
                    )
                    signal.description = "季末再平衡,关注流动性变化"
                    return signal

                signal.signal_type = SeasonalitySignalType.MONTH_END_BULLISH
                signal.best_opportunity = SeasonalityOpportunity(
                    signal_type=SeasonalitySignalType.MONTH_END_BULLISH,
                    seasonality_score=self.seasonality_score,
                    confidence=0.50,
                    description=f"月末效应: 距月末{days_to_month_end}天"
                )
                signal.description = "月末效应,温和看涨"
                return signal

        elif self.seasonality_score < -0.20:
            day_of_week = self.current_date.weekday()
            if day_of_week == 5:  # Saturday
                signal.signal_type = SeasonalitySignalType.WEEKEND_RISK
                signal.best_opportunity = SeasonalityOpportunity(
                    signal_type=SeasonalitySignalType.WEEKEND_RISK,
                    seasonality_score=self.seasonality_score,
                    confidence=0.55,
                    description="周末风险: 流动性低,波动率高"
                )
                signal.description = "周末风险,减仓或对冲"
                return signal

            if self.current_date.month == 12:
                signal.signal_type = SeasonalitySignalType.MONTH_END_BEARISH
                signal.best_opportunity = SeasonalityOpportunity(
                    signal_type=SeasonalitySignalType.MONTH_END_BEARISH,
                    seasonality_score=self.seasonality_score,
                    confidence=0.50,
                    description="12月税损收割: 卖出压力"
                )
                signal.description = "12月税损卖出压力"
                return signal

        # 默认: 基于综合分数
        if self.cycle_score > self.CYCLE_BUY:
            signal.signal_type = SeasonalitySignalType.HALVING_MARKUP
            signal.description = f"周期分数{self.cycle_score:.2f},温和看涨"
        elif self.cycle_score < self.CYCLE_SELL:
            signal.signal_type = SeasonalitySignalType.HOLD
            signal.description = f"周期分数{self.cycle_score:.2f},持有观望"
        else:
            signal.signal_type = SeasonalitySignalType.NEUTRAL
            signal.description = "季节性信号中性"

        return signal

    def analyze(self) -> SeasonalitySignal:
        """
        分析季节性周期信号

        Returns:
            SeasonalitySignal
        """
        # 1. 计算周期阶段
        self.current_phase, self.days_after_halving, self.days_to_next_halving = \
            self._compute_cycle_phase()

        # 2. 计算周期分数
        self.cycle_score = self._compute_cycle_score()

        # 3. 计算季节性分数
        self.seasonality_score = self._compute_seasonality_score()

        # 4. 选择信号
        signal = self._select_signal()
        self.signal_count += 1
        return signal

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "current_date": self.current_date.isoformat(),
            "current_phase": self.current_phase.value,
            "last_halving_date": self.last_halving.date.isoformat(),
            "days_after_halving": self.days_after_halving,
            "next_halving_date": self.NEXT_HALVING_DATE.isoformat(),
            "days_to_next_halving": self.days_to_next_halving,
            "estimated_peak_date": self.estimated_peak_date.isoformat() if self.estimated_peak_date else None,
            "estimated_bottom_date": self.estimated_bottom_date.isoformat() if self.estimated_bottom_date else None,
            "days_after_peak": self.days_after_peak,
            "mvrv_ratio": round(self.mvrv_ratio, 4),
            "reserve_risk": round(self.reserve_risk, 6),
            "cycle_score": round(self.cycle_score, 4),
            "seasonality_score": round(self.seasonality_score, 4),
            "signal_count": self.signal_count,
        }


# ============================================================================
# 自检 (Self-test)
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("seasonality_cycle.py 自检")
    print("=" * 70)

    # 测试1: 当前日期 (2026年6月)
    print("\n[1] 测试当前日期 (2026-06-15)...")
    engine = SeasonalityCycle(current_date=datetime(2026, 6, 15))
    engine.update_onchain_metrics(mvrv=2.7, reserve_risk=0.0045)
    signal = engine.analyze()
    status = engine.get_status()
    print(f"  阶段: {status['current_phase']}")
    print(f"  减半后天数: {status['days_after_halving']}")
    print(f"  距下次减半: {status['days_to_next_halving']}天")
    print(f"  周期分数: {status['cycle_score']}")
    print(f"  季节性分数: {status['seasonality_score']}")
    print(f"  信号: {signal.signal_type.value}")
    print(f"  描述: {signal.description}")

    # 验证: 2026年6月应该在Markup阶段 (2024年4月减半后约790天,已过峰值535天)
    # 实际应该是Distribution或Bear_Market阶段
    assert status['days_after_halving'] > 500, f"减半后应>500天,实际:{status['days_after_halving']}"
    print("  ✓ 减半后天数正确")

    # 测试2: 2027年11月 (积累期, 下次减半前约500天)
    print("\n[2] 测试2027年11月 (Pre-Halving买入窗口)...")
    engine2 = SeasonalityCycle(current_date=datetime(2027, 11, 30))
    engine2.update_onchain_metrics(mvrv=1.2, reserve_risk=0.015)
    signal2 = engine2.analyze()
    status2 = engine2.get_status()
    print(f"  阶段: {status2['current_phase']}")
    print(f"  距下次减半: {status2['days_to_next_halving']}天")
    print(f"  周期分数: {status2['cycle_score']}")
    print(f"  信号: {signal2.signal_type.value}")
    assert status2['current_phase'] == CyclePhase.ACCUMULATION.value, \
        f"应为积累期,实际:{status2['current_phase']}"
    assert status2['days_to_next_halving'] <= 500, \
        f"应在500天买入窗口内,实际:{status2['days_to_next_halving']}"
    print("  ✓ Pre-Halving积累期正确识别")

    # 测试3: 2025年10月 (峰值期)
    print("\n[3] 测试2025年10月 (峰值期)...")
    engine3 = SeasonalityCycle(current_date=datetime(2025, 10, 6))
    engine3.update_onchain_metrics(mvrv=3.8, reserve_risk=0.0015)
    signal3 = engine3.analyze()
    status3 = engine3.get_status()
    print(f"  阶段: {status3['current_phase']}")
    print(f"  MVRV: {status3['mvrv_ratio']}")
    print(f"  信号: {signal3.signal_type.value}")
    assert status3['mvrv_ratio'] > 3.5, "MVRV>3.5应触发顶部预警"
    assert signal3.signal_type == SeasonalitySignalType.CYCLE_TOP_WARNING, \
        f"应为顶部预警,实际:{signal3.signal_type.value}"
    print("  ✓ MVRV>3.5正确触发顶部预警")

    # 测试4: 周末效应
    print("\n[4] 测试周末效应 (2026-06-13 周六)...")
    engine4 = SeasonalityCycle(current_date=datetime(2026, 6, 13))
    engine4.update_onchain_metrics(mvrv=2.5, reserve_risk=0.005)
    signal4 = engine4.analyze()
    status4 = engine4.get_status()
    print(f"  季节性分数: {status4['seasonality_score']}")
    print(f"  信号: {signal4.signal_type.value}")
    assert status4['seasonality_score'] < 0, "周六季节性分数应为负"
    print("  ✓ 周末效应正确识别")

    # 测试5: 月初效应
    print("\n[5] 测试月初效应 (2026-06-01)...")
    engine5 = SeasonalityCycle(current_date=datetime(2026, 6, 1))
    engine5.update_onchain_metrics(mvrv=2.5, reserve_risk=0.005)
    signal5 = engine5.analyze()
    status5 = engine5.get_status()
    print(f"  季节性分数: {status5['seasonality_score']}")
    print(f"  信号: {signal5.signal_type.value}")
    assert status5['seasonality_score'] > 0, "月初季节性分数应为正"
    print("  ✓ 月初效应正确识别")

    print("\n" + "=" * 70)
    print("✓ seasonality_cycle.py 自检通过")
    print("=" * 70)
