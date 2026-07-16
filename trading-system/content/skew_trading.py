# -*- coding: utf-8 -*-
"""
skew_trading.py — 期权偏度交易引擎 v1.0

来源：
  - Hypercall 2026 (Risk Reversal: 25-delta最受关注的偏度指标)
  - CBOE 2026 (The Power of the Risk-Reversal: 系统性卖过贵put买便宜call)
  - KenMacro 2026 (Butterfly Spread: 1-2-1 ratio, pinning行为)
  - tradinghack 2026 (IV歪み: スキュー歪みのリスクリバーサル)
  - xueqiu 2026 (期权套利: IV偏离套利+Delta中性)

核心算法：
  1. 偏度计算
     - 25Δ Risk Reversal (RR) = IV_25ΔCall - IV_25ΔPut
     - RR > 0: Call比Put贵 (看涨偏度, rally risk)
     - RR < 0: Put比Call贵 (看跌偏度, crash risk, 常见)
     - 25Δ Butterfly (BF) = (IV_25ΔCall + IV_25ΔPut)/2 - IV_ATM
  2. 偏度交易策略
     - 策略A: 风险逆转 (Risk Reversal)
       * 偏度极端时: 卖贵的腿, 买便宜的腿
       * RR极端负: 卖Put买Call (Put太贵)
       * RR极端正: 卖Call买Put (Call太贵)
     - 策略B: 蝶式价差 (Butterfly Spread)
       * 1-2-1 ratio, 三等距执行价
       * 赌价格pinning到中间执行价
       * 适合事件后低波动期
     - 策略C: 偏度均值回归
       * RR历史均值+偏离阈值
       * 偏度极端 → 均值回归
  3. CBOE RXM策略
     - 每月卖25Δ SPX Put + 买25Δ SPX Call
     - Delta对冲后: CAGR 2.5%, Vol 5.4%, IR 0.47
     - 系统性收割波动率风险溢价

风险控制：
  - 裸卖Put风险: 无限下行风险, 需止损或对冲
  - Dead Zone: RR两执行价之间无收益
  - 偏度持续: 偏度可能长期不回归

实盘验证：
  - CBOE RXM Index: IR 0.47 (高于S&P 500的0.31)
  - 25Δ RR是每个vol desk最关注的偏度指标
  - ETH RR在Merge前转正, 预示rally risk
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class SkewSignalType(Enum):
    """偏度信号类型"""
    SELL_PUT_BUY_CALL = "sell_put_buy_call"     # 卖Put买Call (RR负极端)
    SELL_CALL_BUY_PUT = "sell_call_buy_put"     # 卖Call买Put (RR正极端)
    BUTTERFLY_SPREAD = "butterfly_spread"       # 蝶式价差
    SKEW_MEAN_REVERSION = "skew_mean_reversion" # 偏度均值回归
    CLOSE_POSITION = "close_position"
    HOLD = "hold"
    NEUTRAL = "neutral"


@dataclass
class SkewData:
    """偏度数据快照"""
    timestamp: float
    iv_atm: float = 0.0
    iv_25delta_call: float = 0.0
    iv_25delta_put: float = 0.0
    spot_price: float = 0.0


@dataclass
class SkewOpportunity:
    """偏度机会"""
    signal_type: SkewSignalType
    risk_reversal: float = 0.0       # 25Δ RR
    butterfly: float = 0.0           # 25Δ BF
    rr_percentile: float = 0.5       # RR历史百分位
    rr_deviation: float = 0.0        # RR偏离均值
    expected_edge_usd: float = 0.0
    confidence: float = 0.0
    description: str = ""


@dataclass
class SkewSignal:
    """偏度综合信号"""
    signal_type: SkewSignalType = SkewSignalType.NEUTRAL
    best_opportunity: Optional[SkewOpportunity] = None
    current_rr: float = 0.0
    current_bf: float = 0.0
    rr_zscore: float = 0.0
    description: str = ""


class SkewTrading:
    """
    期权偏度交易引擎

    使用场景：
      - 期权市场（Deribit/OKX/Binance Options）
      - 25Δ RR偏度交易
      - 事件后pinning交易

    依赖：
      - numpy（数值计算）
      - ATM/25Δ Call/Put IV数据
    """

    # ===== 偏度阈值 =====
    RR_EXTREME_NEGATIVE = -0.05    # -5 vol点 = Put极端贵
    RR_EXTREME_POSITIVE = 0.05     # +5 vol点 = Call极端贵
    RR_MEAN_REVERSION_Z = 2.0      # Z-score 2σ = 均值回归
    BF_HIGH_THRESHOLD = 0.03       # BF 3 vol点 = 翼部贵

    # ===== 蝶式参数 =====
    BUTTERFLY_WING_WIDTH = 0.05    # 翼宽5%
    BUTTERFLY_MAX_COST = 0.02      # 最大成本2% of spot

    # ===== 历史窗口 =====
    RR_HISTORY_WINDOW = 100

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        self.current_data: Optional[SkewData] = None
        self.rr_history: deque = deque(maxlen=self.RR_HISTORY_WINDOW)
        self.bf_history: deque = deque(maxlen=self.RR_HISTORY_WINDOW)
        self.position_side: Optional[str] = None
        self.position_start_time: float = 0.0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_skew_data(self, data: SkewData) -> None:
        """更新偏度数据"""
        self.current_data = data
        rr = self.calculate_risk_reversal()
        bf = self.calculate_butterfly()
        self.rr_history.append(rr)
        self.bf_history.append(bf)

    def calculate_risk_reversal(self) -> float:
        """计算25Δ Risk Reversal = IV_25ΔCall - IV_25ΔPut"""
        if not self.current_data:
            return 0.0
        return self.current_data.iv_25delta_call - self.current_data.iv_25delta_put

    def calculate_butterfly(self) -> float:
        """计算25Δ Butterfly = (IV_25ΔCall + IV_25ΔPut)/2 - IV_ATM"""
        if not self.current_data:
            return 0.0
        return (self.current_data.iv_25delta_call + self.current_data.iv_25delta_put) / 2 - self.current_data.iv_atm

    # ------------------------------------------------------------------
    # 偏度分析
    # ------------------------------------------------------------------

    def get_rr_mean(self) -> float:
        """获取RR历史均值"""
        if not self.rr_history:
            return 0.0
        return float(np.mean(list(self.rr_history)))

    def get_rr_std(self) -> float:
        """获取RR历史标准差"""
        if len(self.rr_history) < 2:
            return 0.01  # 默认1vol点
        return float(np.std(list(self.rr_history)))

    def calculate_rr_zscore(self) -> float:
        """计算RR Z-score"""
        mean = self.get_rr_mean()
        std = self.get_rr_std()
        if std <= 0:
            return 0.0
        return (self.calculate_risk_reversal() - mean) / std

    def get_rr_percentile(self) -> float:
        """获取RR历史百分位"""
        if len(self.rr_history) < 10:
            return 0.5
        current = self.calculate_risk_reversal()
        rank = sum(1 for r in self.rr_history if r <= current)
        return rank / len(self.rr_history)

    # ------------------------------------------------------------------
    # 策略计算
    # ------------------------------------------------------------------

    def calculate_rr_edge(self) -> float:
        """计算RR交易预期收益"""
        if not self.current_data:
            return 0.0
        rr = self.calculate_risk_reversal()
        # 简化: 偏差越大, 预期收益越大
        return abs(rr) * 1000  # $1000 per vol point

    def calculate_butterfly_cost(self, spot: float) -> float:
        """计算蝶式价差成本"""
        # 简化: 蝶式成本 ≈ 翼宽 × IV × √T
        if not self.current_data:
            return 0.0
        iv = self.current_data.iv_atm
        T = 30 / 365  # 30天
        return self.BUTTERFLY_WING_WIDTH * spot * iv * math.sqrt(T)

    def calculate_butterfly_max_profit(self, spot: float) -> float:
        """计算蝶式最大利润"""
        cost = self.calculate_butterfly_cost(spot)
        wing = self.BUTTERFLY_WING_WIDTH * spot
        return wing - cost

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def analyze(self) -> SkewSignal:
        """主分析函数"""
        signal = SkewSignal()

        if not self.current_data:
            signal.description = "无偏度数据"
            return signal

        rr = self.calculate_risk_reversal()
        bf = self.calculate_butterfly()
        z_score = self.calculate_rr_zscore()
        rr_percentile = self.get_rr_percentile()
        rr_mean = self.get_rr_mean()
        rr_deviation = rr - rr_mean

        signal.current_rr = rr
        signal.current_bf = bf
        signal.rr_zscore = z_score

        # 有持仓: 检查平仓
        if self.position_side == "rr_short_put":
            # 卖Put买Call平仓: RR回归到均值
            if abs(z_score) < 0.5:
                opp = SkewOpportunity(
                    signal_type=SkewSignalType.CLOSE_POSITION,
                    risk_reversal=rr, butterfly=bf,
                    rr_percentile=rr_percentile, rr_deviation=rr_deviation,
                    confidence=0.8,
                    description=f"RR平仓: Z={z_score:.2f}回归均值",
                )
                signal.signal_type = SkewSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            signal.signal_type = SkewSignalType.HOLD
            signal.description = f"持有卖Put买Call, RR={rr:.4f}, Z={z_score:.2f}"
            return signal

        if self.position_side == "rr_short_call":
            if abs(z_score) < 0.5:
                opp = SkewOpportunity(
                    signal_type=SkewSignalType.CLOSE_POSITION,
                    risk_reversal=rr, butterfly=bf,
                    rr_percentile=rr_percentile, rr_deviation=rr_deviation,
                    confidence=0.8,
                    description=f"RR平仓: Z={z_score:.2f}回归均值",
                )
                signal.signal_type = SkewSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            signal.signal_type = SkewSignalType.HOLD
            signal.description = f"持有卖Call买Put, RR={rr:.4f}, Z={z_score:.2f}"
            return signal

        # 无持仓: 检查入场

        # 1. RR极端负: 卖Put买Call (Put太贵)
        if rr < self.RR_EXTREME_NEGATIVE or z_score < -self.RR_MEAN_REVERSION_Z:
            edge = self.calculate_rr_edge()
            opp = SkewOpportunity(
                signal_type=SkewSignalType.SELL_PUT_BUY_CALL,
                risk_reversal=rr, butterfly=bf,
                rr_percentile=rr_percentile, rr_deviation=rr_deviation,
                expected_edge_usd=edge,
                confidence=0.75,
                description=(
                    f"RR={rr:.4f}极端负, Z={z_score:.2f}, "
                    f"Put过贵, 卖25ΔPut买25ΔCall, "
                    f"预期收益=${edge:.0f}"
                ),
            )
            signal.signal_type = SkewSignalType.SELL_PUT_BUY_CALL
            signal.best_opportunity = opp
            return signal

        # 2. RR极端正: 卖Call买Put (Call太贵)
        if rr > self.RR_EXTREME_POSITIVE or z_score > self.RR_MEAN_REVERSION_Z:
            edge = self.calculate_rr_edge()
            opp = SkewOpportunity(
                signal_type=SkewSignalType.SELL_CALL_BUY_PUT,
                risk_reversal=rr, butterfly=bf,
                rr_percentile=rr_percentile, rr_deviation=rr_deviation,
                expected_edge_usd=edge,
                confidence=0.7,
                description=(
                    f"RR={rr:.4f}极端正, Z={z_score:.2f}, "
                    f"Call过贵, 卖25ΔCall买25ΔPut, "
                    f"预期收益=${edge:.0f}"
                ),
            )
            signal.signal_type = SkewSignalType.SELL_CALL_BUY_PUT
            signal.best_opportunity = opp
            return signal

        # 3. 蝶式价差 (BF高, 赌pinning)
        if bf > self.BF_HIGH_THRESHOLD:
            spot = self.current_data.spot_price
            cost = self.calculate_butterfly_cost(spot)
            max_profit = self.calculate_butterfly_max_profit(spot)
            if cost < self.BUTTERFLY_MAX_COST * spot and max_profit > 0:
                opp = SkewOpportunity(
                    signal_type=SkewSignalType.BUTTERFLY_SPREAD,
                    risk_reversal=rr, butterfly=bf,
                    rr_percentile=rr_percentile, rr_deviation=rr_deviation,
                    expected_edge_usd=max_profit,
                    confidence=0.65,
                    description=(
                        f"BF={bf:.4f}高, 翼部贵, "
                        f"蝶式价差成本={cost:.2f}, "
                        f"最大利润={max_profit:.2f}, "
                        f"赌价格pinning到中间执行价"
                    ),
                )
                signal.signal_type = SkewSignalType.BUTTERFLY_SPREAD
                signal.best_opportunity = opp
                return signal

        # 4. 偏度均值回归
        if abs(z_score) > 1.5 and abs(rr) < max(abs(self.RR_EXTREME_NEGATIVE), self.RR_EXTREME_POSITIVE):
            opp = SkewOpportunity(
                signal_type=SkewSignalType.SKEW_MEAN_REVERSION,
                risk_reversal=rr, butterfly=bf,
                rr_percentile=rr_percentile, rr_deviation=rr_deviation,
                expected_edge_usd=abs(rr_deviation) * 500,
                confidence=0.6,
                description=(
                    f"偏度均值回归: Z={z_score:.2f}, "
                    f"RR={rr:.4f}偏离均值{rr_mean:.4f}, "
                    f"赌RR回归"
                ),
            )
            signal.signal_type = SkewSignalType.SKEW_MEAN_REVERSION
            signal.best_opportunity = opp
            return signal

        signal.description = (
            f"观望: RR={rr:.4f}, BF={bf:.4f}, Z={z_score:.2f}, "
            f"百分位={rr_percentile:.2f}"
        )
        return signal

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "symbol": self.symbol,
            "current_rr": self.calculate_risk_reversal(),
            "current_bf": self.calculate_butterfly(),
            "rr_mean": self.get_rr_mean(),
            "rr_std": self.get_rr_std(),
            "rr_zscore": self.calculate_rr_zscore(),
            "rr_percentile": self.get_rr_percentile(),
            "position_side": self.position_side,
            "history_count": len(self.rr_history),
        }
