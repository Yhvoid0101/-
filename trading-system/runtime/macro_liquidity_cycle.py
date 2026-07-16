# -*- coding: utf-8 -*-
"""
macro_liquidity_cycle.py — 宏观流动性周期建模引擎 v1.0

来源：
  - Onramp Institutional 2026 (Bitcoin's Macro Liquidity Cycle: M2/DXY/Real Rates)
  - afsheenjafry.co.uk 2025 (BTC vs Global M2 r²=0.89, 60-70天滞后)
  - asapdrew.com 2026 (G4 broad money, 滞后1-3月)
  - bitcoinfoundation.org 2026 (Fed Rates/Inflation/Liquidity Impact)
  - Lyn Alden 2024 (Bitcoin: A Global Liquidity Barometer)
  - hotcoin.com 2026 (2026宏观变量前瞻分析)

核心算法：
  1. Global M2 流动性周期
     - BTC vs Global M2 相关性: 0.94 (2013-2024)
     - 滞后期: 60-70天 (M2扩张 → BTC上涨)
     - G4 broad money: US M2 + EU M3 + China M2 + Japan M3
     - 65月周期: 全球流动性节奏

  2. 美元指数 (DXY) 反向关系
     - DXY上升 → 全球金融条件收紧 → BTC承压
     - DXY下降 → 美元走弱 → BTC利好
     - 阈值: DXY>105 强美元, DXY<95 弱美元

  3. 实际利率 (Real Yields)
     - 10Y TIPS (通胀保值债券) 实际收益率
     - 实际利率上升 → 非收益资产(BTC)承压
     - 实际利率下降 → BTC利好
     - 阈值: real_rate > 2% 强阻力, < 0% 强利好

  4. 流动性Regime检测
     - Expanding: M2 YoY增长 > 中位数 (BTC利好)
     - Contracting: M2 YoY增长 < 中位数 (BTC利空)
     - 概率分布偏移: Expanding时负收益概率下降
     - 12月回撤 >20% 概率显著降低

  5. 跨市场流动性传导
     - 股票市场 → BTC: 风险偏好传导
     - ETF资金流: 机构资本部署
     - 稳定币mint/burn: 链上流动性
     - 期货未平仓合约 (OI): 杠杆流动性

  6. 65月周期拐点检测
     - 全球流动性65月周期
     - 拐点窗口: ±3个月
     - 历史规律: 拐点后6-12月BTC显著波动

性能（来源：afsheenjafry 2025 + Onramp 2026）：
  - BTC vs M2 相关性: 0.94 (r²=0.89)
  - 滞后期: 60-70天 (1-3月)
  - Expanding regime: 12月负收益概率显著降低
  - Contracting regime: 回撤 >20% 概率上升

依赖：
  - numpy
  - 宏观数据 (M2/DXY/Real Rates/OI/Stablecoin)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple



class MacroSignalType(Enum):
    """宏观流动性信号类型"""
    LIQUIDITY_EXPANSION_BUY = "liquidity_expansion_buy"      # 流动性扩张买入
    LIQUIDITY_CONTRACTION_SELL = "liquidity_contraction_sell"  # 流动性收缩卖出
    DOLLAR_STRENGTH_WARNING = "dollar_strength_warning"      # 美元走强预警
    DOLLAR_WEAKNESS_BUY = "dollar_weakness_buy"              # 美元走弱买入
    REAL_RATE_WARNING = "real_rate_warning"                  # 实际利率上升预警
    REAL_RATE_BENEFIT = "real_rate_benefit"                  # 实际利率下降利好
    CYCLE_INFLECTION = "cycle_inflection"                    # 65月周期拐点
    REGIME_SHIFT = "regime_shift"                            # 流动性状态切换
    ETF_INFLOW_BULLISH = "etf_inflow_bullish"                # ETF资金流入看涨
    RISK_OFF = "risk_off"                                    # 避险模式
    HOLD = "hold"
    NEUTRAL = "neutral"


class LiquidityRegime(Enum):
    """流动性状态"""
    EXPANDING = "expanding"        # M2扩张 (BTC利好)
    NEUTRAL = "neutral"            # 中性
    CONTRACTING = "contracting"    # M2收缩 (BTC利空)
    INFLECTION = "inflection"      # 周期拐点


@dataclass
class MacroData:
    """宏观数据"""
    global_m2: float = 0.0                # 全球M2 (万亿美元)
    global_m2_yoy: float = 0.0            # 全球M2同比增速 (%)
    dxy: float = 100.0                    # 美元指数
    real_rate_10y: float = 0.0            # 10年实际利率 (%)
    fed_funds_rate: float = 0.0           # 联邦基金利率 (%)
    btc_dominance: float = 50.0           # BTC市占率 (%)
    stablecoin_mcap: float = 0.0          # 稳定币市值 (亿)
    futures_oi: float = 0.0               # 期货未平仓合约 (亿)
    etf_flow: float = 0.0                 # ETF资金流 (亿)


@dataclass
class MacroOpportunity:
    """宏观交易机会"""
    signal_type: MacroSignalType
    liquidity_regime: LiquidityRegime = LiquidityRegime.NEUTRAL
    m2_momentum: float = 0.0              # M2动量
    dxy_momentum: float = 0.0             # DXY动量
    real_rate_momentum: float = 0.0       # 实际利率动量
    liquidity_score: float = 0.0          # 综合流动性分数 [-1, 1]
    cycle_position: float = 0.0           # 周期位置 [0, 1] (0=底部, 1=顶部)
    cycle_days_to_inflection: int = 0     # 距离拐点天数
    risk_level: float = 0.0               # 风险等级 [0, 1]
    confidence: float = 0.0
    description: str = ""


@dataclass
class MacroSignal:
    """宏观综合信号"""
    signal_type: MacroSignalType = MacroSignalType.NEUTRAL
    best_opportunity: Optional[MacroOpportunity] = None
    current_regime: LiquidityRegime = LiquidityRegime.NEUTRAL
    liquidity_score: float = 0.0
    cycle_position: float = 0.0
    description: str = ""


class MacroLiquidityCycle:
    """
    宏观流动性周期建模引擎

    使用场景：
      - 宏观流动性方向判断
      - 周期拐点预警
      - 跨市场流动性传导分析
      - 长期仓位调整

    依赖：
      - 宏观经济数据 (M2/DXY/Real Rates)
      - 链上数据 (Stablecoin/OI)
      - ETF资金流数据
    """

    # ===== M2 阈值 =====
    M2_EXPANSION_THRESHOLD = 5.0           # M2 YoY > 5% = 扩张
    M2_CONTRACTION_THRESHOLD = 0.0         # M2 YoY < 0% = 收缩
    M2_LAG_DAYS = 65                       # M2 → BTC 滞后天数
    M2_HISTORY_WINDOW = 100                # M2历史窗口

    # ===== DXY 阈值 =====
    DXY_STRONG = 105.0                     # 强美元
    DXY_WEAK = 95.0                        # 弱美元
    DXY_NEUTRAL_LOW = 98.0
    DXY_NEUTRAL_HIGH = 102.0

    # ===== 实际利率阈值 =====
    REAL_RATE_HIGH = 2.0                   # 实际利率 > 2% = 强阻力
    REAL_RATE_LOW = 0.0                    # 实际利率 < 0% = 强利好
    REAL_RATE_VERY_LOW = -1.0              # 极低实际利率

    # ===== 65月周期 =====
    CYCLE_LENGTH_MONTHS = 65               # 65月周期
    CYCLE_INFLECTION_WINDOW = 3            # 拐点窗口 (±3月)

    # ===== ETF 资金流 =====
    ETF_INFLOW_BULLISH = 1.0               # ETF日流入 > 1亿 = 看涨
    ETF_OUTFLOW_BEARISH = -1.0             # ETF日流出 < -1亿 = 看跌

    # ===== 稳定币 =====
    STABLECOIN_MINT_BULLISH = 0.02         # 稳定币市值日增 > 2% = 看涨
    STABLECOIN_BURN_BEARISH = -0.02        # 稳定币市值日减 > 2% = 看跌

    # ===== 综合评分权重 =====
    WEIGHT_M2 = 0.35                       # M2权重最高
    WEIGHT_DXY = 0.25
    WEIGHT_REAL_RATE = 0.20
    WEIGHT_STABLECOIN = 0.10
    WEIGHT_ETF = 0.10

    # ===== 历史窗口 =====
    HISTORY_WINDOW = 200

    def __init__(self, symbol: str = "BTC-USDT"):
        """
        初始化宏观流动性周期引擎

        Args:
            symbol: 交易对
        """
        self.symbol = symbol

        # 当前宏观数据
        self.current_macro: MacroData = MacroData()

        # 历史数据
        self.m2_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self.dxy_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self.real_rate_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self.stablecoin_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self.etf_flow_history: deque = deque(maxlen=self.HISTORY_WINDOW)
        self.liquidity_score_history: deque = deque(maxlen=self.HISTORY_WINDOW)

        # 周期位置
        self.cycle_start_month: int = 0
        self.current_month_in_cycle: int = 0

        # 状态
        self.current_regime: LiquidityRegime = LiquidityRegime.NEUTRAL
        self.liquidity_score: float = 0.0
        self.cycle_position: float = 0.5
        self.cycle_days_to_inflection: int = 0

        # 滞后M2预测
        self.m2_lagged_history: deque = deque(maxlen=self.HISTORY_WINDOW)

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_macro_data(self, macro: MacroData) -> None:
        """更新宏观数据"""
        self.current_macro = macro
        self.m2_history.append(macro.global_m2_yoy)
        self.dxy_history.append(macro.dxy)
        self.real_rate_history.append(macro.real_rate_10y)
        self.stablecoin_history.append(macro.stablecoin_mcap)
        self.etf_flow_history.append(macro.etf_flow)

    def update_m2(self, m2_yoy: float) -> None:
        """更新M2同比增速"""
        self.current_macro.global_m2_yoy = m2_yoy
        self.m2_history.append(m2_yoy)

    def update_dxy(self, dxy: float) -> None:
        """更新美元指数"""
        self.current_macro.dxy = dxy
        self.dxy_history.append(dxy)

    def update_real_rate(self, real_rate: float) -> None:
        """更新实际利率"""
        self.current_macro.real_rate_10y = real_rate
        self.real_rate_history.append(real_rate)

    def update_stablecoin(self, mcap: float) -> None:
        """更新稳定币市值"""
        self.current_macro.stablecoin_mcap = mcap
        self.stablecoin_history.append(mcap)

    def update_etf_flow(self, flow: float) -> None:
        """更新ETF资金流"""
        self.current_macro.etf_flow = flow
        self.etf_flow_history.append(flow)

    def update_cycle_position(self, month_in_cycle: int) -> None:
        """更新周期位置"""
        self.current_month_in_cycle = month_in_cycle
        self.cycle_position = month_in_cycle / self.CYCLE_LENGTH_MONTHS
        # 距离拐点天数 (假设1月=30天)
        months_to_inflection = self.CYCLE_LENGTH_MONTHS - month_in_cycle
        self.cycle_days_to_inflection = months_to_inflection * 30

    # ------------------------------------------------------------------
    # M2 流动性分析
    # ------------------------------------------------------------------

    def calculate_m2_momentum(self) -> float:
        """计算M2动量 (当前 vs 20期前)"""
        if len(self.m2_history) < 20:
            return 0.0
        current = self.m2_history[-1]
        past = self.m2_history[-20]
        return float(current - past)

    def calculate_m2_lagged_signal(self) -> float:
        """
        计算M2滞后信号 (60-70天前的M2 → 当前BTC信号)
        假设每天1个数据点,滞后约65天
        """
        if len(self.m2_history) < self.M2_LAG_DAYS:
            return 0.0
        lagged_m2 = self.m2_history[-self.M2_LAG_DAYS]
        current_m2 = self.m2_history[-1]

        # 滞后M2上升 → 当前BTC利好
        if lagged_m2 > self.M2_EXPANSION_THRESHOLD:
            return 1.0
        elif lagged_m2 < self.M2_CONTRACTION_THRESHOLD:
            return -1.0
        else:
            return 0.0

    def detect_liquidity_regime(self) -> LiquidityRegime:
        """检测流动性状态"""
        if not self.m2_history:
            return LiquidityRegime.NEUTRAL

        current_m2 = self.m2_history[-1]

        # 检查周期拐点
        months_to_inflection = self.CYCLE_LENGTH_MONTHS - self.current_month_in_cycle
        if months_to_inflection <= self.CYCLE_INFLECTION_WINDOW:
            return LiquidityRegime.INFLECTION

        # M2 YoY 判断
        if current_m2 >= self.M2_EXPANSION_THRESHOLD:
            regime = LiquidityRegime.EXPANDING
        elif current_m2 <= self.M2_CONTRACTION_THRESHOLD:
            regime = LiquidityRegime.CONTRACTING
        else:
            regime = LiquidityRegime.NEUTRAL

        self.current_regime = regime
        return regime

    # ------------------------------------------------------------------
    # DXY 美元指数分析
    # ------------------------------------------------------------------

    def calculate_dxy_momentum(self) -> float:
        """计算DXY动量"""
        if len(self.dxy_history) < 20:
            return 0.0
        current = self.dxy_history[-1]
        past = self.dxy_history[-20]
        return float(current - past)

    def assess_dxy_impact(self) -> float:
        """
        评估DXY对BTC的影响 [-1, 1]
        正值 = 利好BTC, 负值 = 利空BTC
        """
        if not self.dxy_history:
            return 0.0

        dxy = self.dxy_history[-1]

        if dxy >= self.DXY_STRONG:
            return -1.0  # 强美元 = BTC利空
        elif dxy <= self.DXY_WEAK:
            return 1.0   # 弱美元 = BTC利好
        elif dxy < self.DXY_NEUTRAL_LOW:
            return 0.5
        elif dxy > self.DXY_NEUTRAL_HIGH:
            return -0.5
        else:
            return 0.0

    # ------------------------------------------------------------------
    # 实际利率分析
    # ------------------------------------------------------------------

    def calculate_real_rate_momentum(self) -> float:
        """计算实际利率动量"""
        if len(self.real_rate_history) < 20:
            return 0.0
        current = self.real_rate_history[-1]
        past = self.real_rate_history[-20]
        return float(current - past)

    def assess_real_rate_impact(self) -> float:
        """
        评估实际利率对BTC的影响 [-1, 1]
        正值 = 利好BTC, 负值 = 利空BTC
        """
        if not self.real_rate_history:
            return 0.0

        rate = self.real_rate_history[-1]

        if rate <= self.REAL_RATE_VERY_LOW:
            return 1.0   # 极低实际利率 = 强利好
        elif rate <= self.REAL_RATE_LOW:
            return 0.5   # 低实际利率 = 利好
        elif rate >= self.REAL_RATE_HIGH:
            return -1.0  # 高实际利率 = 强利空
        else:
            # 线性插值
            return float((self.REAL_RATE_HIGH - rate) / (self.REAL_RATE_HIGH - self.REAL_RATE_LOW) * 2 - 1)

    # ------------------------------------------------------------------
    # 稳定币流动性分析
    # ------------------------------------------------------------------

    def assess_stablecoin_impact(self) -> float:
        """
        评估稳定币流动性对BTC的影响 [-1, 1]
        mint (市值增加) = 资金入场 = 利好
        burn (市值减少) = 资金离场 = 利空
        """
        if len(self.stablecoin_history) < 2:
            return 0.0

        current = self.stablecoin_history[-1]
        past = self.stablecoin_history[-2]

        if past <= 0:
            return 0.0

        change_pct = (current - past) / past

        if change_pct >= self.STABLECOIN_MINT_BULLISH:
            return 1.0
        elif change_pct <= self.STABLECOIN_BURN_BEARISH:
            return -1.0
        else:
            return change_pct / self.STABLECOIN_MINT_BULLISH

    # ------------------------------------------------------------------
    # ETF 资金流分析
    # ------------------------------------------------------------------

    def assess_etf_impact(self) -> float:
        """
        评估ETF资金流对BTC的影响 [-1, 1]
        流入 = 机构买入 = 利好
        流出 = 机构卖出 = 利空
        """
        if not self.etf_flow_history:
            return 0.0

        flow = self.etf_flow_history[-1]

        if flow >= self.ETF_INFLOW_BULLISH:
            return 1.0
        elif flow <= self.ETF_OUTFLOW_BEARISH:
            return -1.0
        else:
            return float(flow / self.ETF_INFLOW_BULLISH)

    # ------------------------------------------------------------------
    # 综合流动性评分
    # ------------------------------------------------------------------

    def calculate_liquidity_score(self) -> float:
        """
        计算综合流动性分数 [-1, 1]

        权重:
          - M2 (35%): 主要驱动
          - DXY (25%): 反向关系
          - 实际利率 (20%): 折现率
          - 稳定币 (10%): 链上流动性
          - ETF (10%): 机构资金
        """
        m2_score = self.calculate_m2_lagged_signal()
        dxy_score = self.assess_dxy_impact()
        real_rate_score = self.assess_real_rate_impact()
        stablecoin_score = self.assess_stablecoin_impact()
        etf_score = self.assess_etf_impact()

        score = (
            self.WEIGHT_M2 * m2_score
            + self.WEIGHT_DXY * dxy_score
            + self.WEIGHT_REAL_RATE * real_rate_score
            + self.WEIGHT_STABLECOIN * stablecoin_score
            + self.WEIGHT_ETF * etf_score
        )

        score = max(-1.0, min(1.0, score))
        self.liquidity_score = score
        self.liquidity_score_history.append(score)
        return score

    # ------------------------------------------------------------------
    # 周期分析
    # ------------------------------------------------------------------

    def assess_cycle_position(self) -> Tuple[float, bool]:
        """
        评估周期位置

        Returns:
            (cycle_position [0,1], is_inflection: bool)
        """
        pos = self.cycle_position
        months_to_inflection = self.CYCLE_LENGTH_MONTHS - self.current_month_in_cycle
        is_inflection = months_to_inflection <= self.CYCLE_INFLECTION_WINDOW
        return pos, is_inflection

    # ------------------------------------------------------------------
    # 综合分析
    # ------------------------------------------------------------------

    def analyze(self) -> MacroSignal:
        """综合分析,返回宏观信号"""
        # 1. 流动性状态
        regime = self.detect_liquidity_regime()

        # 2. 综合流动性分数
        score = self.calculate_liquidity_score()

        # 3. 各维度动量
        m2_mom = self.calculate_m2_momentum()
        dxy_mom = self.calculate_dxy_momentum()
        rr_mom = self.calculate_real_rate_momentum()

        # 4. 周期位置
        cycle_pos, is_inflection = self.assess_cycle_position()

        # 5. 风险等级
        risk_level = 0.0
        if regime == LiquidityRegime.CONTRACTING:
            risk_level = 0.7
        elif regime == LiquidityRegime.INFLECTION:
            risk_level = 0.5
        elif regime == LiquidityRegime.EXPANDING:
            risk_level = 0.2
        else:
            risk_level = 0.4

        # === 决策逻辑 (按优先级) ===

        # 优先级1: 周期拐点
        if is_inflection:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.CYCLE_INFLECTION,
                liquidity_regime=regime,
                m2_momentum=m2_mom,
                dxy_momentum=dxy_mom,
                real_rate_momentum=rr_mom,
                liquidity_score=score,
                cycle_position=cycle_pos,
                cycle_days_to_inflection=self.cycle_days_to_inflection,
                risk_level=risk_level,
                confidence=0.7,
                description=f"65月周期拐点 距离{self.cycle_days_to_inflection}天",
            )
            return MacroSignal(
                signal_type=MacroSignalType.CYCLE_INFLECTION,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级2: 流动性扩张买入
        if regime == LiquidityRegime.EXPANDING and score > 0.3:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.LIQUIDITY_EXPANSION_BUY,
                liquidity_regime=regime,
                m2_momentum=m2_mom,
                dxy_momentum=dxy_mom,
                real_rate_momentum=rr_mom,
                liquidity_score=score,
                cycle_position=cycle_pos,
                cycle_days_to_inflection=self.cycle_days_to_inflection,
                risk_level=risk_level,
                confidence=min(1.0, score + 0.3),
                description=f"流动性扩张 M2YoY={self.current_macro.global_m2_yoy:.1f}% 买入",
            )
            return MacroSignal(
                signal_type=MacroSignalType.LIQUIDITY_EXPANSION_BUY,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级3: 流动性收缩卖出
        if regime == LiquidityRegime.CONTRACTING and score < -0.3:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.LIQUIDITY_CONTRACTION_SELL,
                liquidity_regime=regime,
                m2_momentum=m2_mom,
                dxy_momentum=dxy_mom,
                real_rate_momentum=rr_mom,
                liquidity_score=score,
                cycle_position=cycle_pos,
                cycle_days_to_inflection=self.cycle_days_to_inflection,
                risk_level=risk_level,
                confidence=min(1.0, abs(score) + 0.3),
                description=f"流动性收缩 M2YoY={self.current_macro.global_m2_yoy:.1f}% 卖出",
            )
            return MacroSignal(
                signal_type=MacroSignalType.LIQUIDITY_CONTRACTION_SELL,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级4: 美元走强预警
        dxy_impact = self.assess_dxy_impact()
        if dxy_impact <= -0.5:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.DOLLAR_STRENGTH_WARNING,
                liquidity_regime=regime,
                dxy_momentum=dxy_mom,
                liquidity_score=score,
                cycle_position=cycle_pos,
                risk_level=max(risk_level, 0.6),
                confidence=0.6,
                description=f"美元走强 DXY={self.current_macro.dxy:.1f} BTC承压",
            )
            return MacroSignal(
                signal_type=MacroSignalType.DOLLAR_STRENGTH_WARNING,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级5: 实际利率预警
        rr_impact = self.assess_real_rate_impact()
        if rr_impact <= -0.5:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.REAL_RATE_WARNING,
                liquidity_regime=regime,
                real_rate_momentum=rr_mom,
                liquidity_score=score,
                cycle_position=cycle_pos,
                risk_level=max(risk_level, 0.6),
                confidence=0.6,
                description=f"实际利率上升 RR={self.current_macro.real_rate_10y:.2f}% BTC承压",
            )
            return MacroSignal(
                signal_type=MacroSignalType.REAL_RATE_WARNING,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级6: ETF资金流入
        etf_impact = self.assess_etf_impact()
        if etf_impact >= 0.8:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.ETF_INFLOW_BULLISH,
                liquidity_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                risk_level=risk_level,
                confidence=0.6,
                description=f"ETF资金流入 ETF={self.current_macro.etf_flow:.1f}亿 看涨",
            )
            return MacroSignal(
                signal_type=MacroSignalType.ETF_INFLOW_BULLISH,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级7: 风险避险模式 (多维度利空)
        if score < -0.5:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.RISK_OFF,
                liquidity_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                risk_level=0.8,
                confidence=0.7,
                description=f"风险避险模式 综合分数={score:.2f}",
            )
            return MacroSignal(
                signal_type=MacroSignalType.RISK_OFF,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 优先级8: 美元走弱买入
        if dxy_impact >= 0.5:
            opp = MacroOpportunity(
                signal_type=MacroSignalType.DOLLAR_WEAKNESS_BUY,
                liquidity_regime=regime,
                dxy_momentum=dxy_mom,
                liquidity_score=score,
                cycle_position=cycle_pos,
                risk_level=risk_level,
                confidence=0.5,
                description=f"美元走弱 DXY={self.current_macro.dxy:.1f} BTC利好",
            )
            return MacroSignal(
                signal_type=MacroSignalType.DOLLAR_WEAKNESS_BUY,
                best_opportunity=opp,
                current_regime=regime,
                liquidity_score=score,
                cycle_position=cycle_pos,
                description=opp.description,
            )

        # 默认: 持有
        return MacroSignal(
            signal_type=MacroSignalType.HOLD,
            current_regime=regime,
            liquidity_score=score,
            cycle_position=cycle_pos,
            description=f"无显著宏观信号 综合分数={score:.2f}",
        )

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "symbol": self.symbol,
            "current_regime": self.current_regime.value,
            "liquidity_score": self.liquidity_score,
            "cycle_position": self.cycle_position,
            "cycle_days_to_inflection": self.cycle_days_to_inflection,
            "current_m2_yoy": self.current_macro.global_m2_yoy,
            "current_dxy": self.current_macro.dxy,
            "current_real_rate": self.current_macro.real_rate_10y,
            "current_etf_flow": self.current_macro.etf_flow,
            "m2_history_len": len(self.m2_history),
            "liquidity_score_history_len": len(self.liquidity_score_history),
        }
