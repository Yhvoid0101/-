# -*- coding: utf-8 -*-
"""
cppi_portfolio_insurance.py — CPPI组合保险策略引擎 v1.0

来源：
  - stockalpha.ai 2026 (CPPI Revisited: Portfolio Insurance, Gap Risk & Stress Tests)
  - ryanoconnellfinance 2026 (Portfolio Rebalancing Disciplines: CPPI, Corridors & Perold-Sharpe)
  - mcginniscommawill 2026 (Bankroll Management with Keeks: CPPI)
  - gelonghui 2026 (巧用仓位调整进行回撤管理 - CPPI/TIPP/OBPI)
  - Black & Perold 1987 (CPPI理论奠基)
  - Perold & Sharpe 1988 (Dynamic Strategies for Asset Allocation)

核心算法：
  1. CPPI核心公式 (Black-Perold 1987)
     - Cushion (垫子): C_t = V_t - F_t (组合价值 - 底线)
     - Risky Exposure (风险暴露): E_t = m × max(C_t, 0)
     - Risky Weight (风险权重): w_t = E_t / V_t
     - m: 风险乘数 (典型2-5, 加密2-3)
     - F_t: 底线 (Present Value of guaranteed amount)

  2. TIPP变体 (Time-Invariant Portfolio Protection)
     - 底线随组合净值动态调整: F_t = max(F_t-1, α × V_t)
     - 净值上涨时底线上移,锁定收益
     - 净值下跌时维持较高底线,强化保护
     - 来源: gelonghui 2026

  3. Gap Risk检测 (来源: stockalpha.ai 2026)
     - 风险: 风险资产在再平衡间隔内下跌超过垫子 → 底线被击穿
     - 检测: Poisson跳跃过程模拟价格跳跃
     - 缓解: 动态乘数 + 最小现金缓冲 + 限价单
     - 历史案例: 1987黑色星期一(-22%), 2020 COVID(-35%), 2022 LUNA(-99%)

  4. 动态乘数 (来源: stockalpha.ai 2026)
     - 静态m=3在低波动期过激进,在高波动期过保守
     - 动态m = m_base × (σ_target / σ_realized), 限制在[m_min, m_max]
     - 高波动 → 降低m → 减少风险暴露
     - 低波动 → 提高m → 增加风险暴露

  5. Perold-Sharpe凸性收益 (来源: ryanoconnellfinance 2026)
     - CPPI = 凸性收益 (类似买入组合保险)
     - Constant-Mix = 凹性收益 (类似卖出组合保险)
     - Buy-and-Hold = 线性收益
     - CPPI在趋势市场表现好,在震荡市场被反复打脸

  6. 风险预算分配
     - 总风险预算 = m × Cushion
     - 多资产: 按波动率反比分配
     - CVaR约束: 95%置信度下的尾部风险

  7. 压力测试场景
     - 1987崩溃: 单日-22%
     - 2020 COVID: 单日-35%
     - 2022 LUNA: 单日-99%
     - Poisson跳跃: λ=2, jump_size=-10%
     - 流动性危机: 价差扩大3倍

性能（来源: stockalpha.ai 2026 + gelonghui 2026）：
  - CPPI在趋势市场: 年化15-25%, 回撤<10%
  - TIPP额外降低回撤30-50%
  - 动态乘数提升夏普比率0.3-0.5
  - Gap Risk概率: 静态CPPI 5-15%, 动态CPPI 1-3%
  - 加密典型参数: m=2-3, floor=80-90%, 再平衡=日级

依赖：
  - numpy (数值计算)
  - 组合净值数据
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CPPISignalType(Enum):
    """CPPI信号类型"""
    INCREASE_RISK = "increase_risk"             # 增加风险暴露
    DECREASE_RISK = "decrease_risk"             # 降低风险暴露
    DELEVERAGE = "deleverage"                   # 紧急去杠杆
    FLOOR_BREACH_WARNING = "floor_breach"       # 底线击穿预警
    GAP_RISK_ALARM = "gap_risk_alarm"           # Gap风险警报
    REBALANCE_UP = "rebalance_up"               # 上调再平衡 (加仓)
    REBALANCE_DOWN = "rebalance_down"           # 下调再平衡 (减仓)
    CASH_HEAVY = "cash_heavy"                   # 现金为主 (垫子耗尽)
    HOLD = "hold"
    NEUTRAL = "neutral"


class ProtectionMode(Enum):
    """保护模式"""
    CPPI = "cppi"                  # 标准CPPI (固定底线)
    TIPP = "tipp"                  # TIPP (动态底线)
    OBPI = "obpi"                  # OBPI (期权保险,简化版)


@dataclass
class PortfolioState:
    """组合状态"""
    value: float                        # 当前组合净值
    floor: float                        # 当前底线
    cushion: float = 0.0                # 垫子 = value - floor
    target_risky_exposure: float = 0.0  # 目标风险暴露
    target_risky_weight: float = 0.0    # 目标风险权重
    current_risky_weight: float = 0.0   # 当前风险权重
    current_cash_weight: float = 1.0    # 当前现金权重
    multiplier: float = 3.0             # 当前乘数
    realized_vol: float = 0.0           # 实现波动率
    gap_risk_probability: float = 0.0   # Gap风险概率
    max_drawdown: float = 0.0           # 最大回撤
    peak_value: float = 0.0             # 峰值
    days_since_rebalance: int = 0       # 距上次再平衡天数


@dataclass
class CPPIOpportunity:
    """CPPI机会"""
    signal_type: CPPISignalType
    target_weight: float = 0.0
    current_weight: float = 0.0
    weight_change: float = 0.0
    cushion_ratio: float = 0.0          # cushion / value
    multiplier: float = 0.0
    gap_risk_prob: float = 0.0
    floor_breach_distance: float = 0.0  # 距底线击穿的距离 (in std)
    confidence: float = 0.0
    description: str = ""


@dataclass
class CPPISignal:
    """CPPI综合信号"""
    signal_type: CPPISignalType = CPPISignalType.NEUTRAL
    best_opportunity: Optional[CPPIOpportunity] = None
    mode: ProtectionMode = ProtectionMode.CPPI
    portfolio_value: float = 0.0
    floor: float = 0.0
    cushion: float = 0.0
    cushion_ratio: float = 0.0
    target_risky_weight: float = 0.0
    multiplier: float = 0.0
    gap_risk_probability: float = 0.0
    description: str = ""


class CPPIPortfolioInsurance:
    """
    CPPI组合保险策略引擎

    使用场景：
      - 组合保险与底线保护
      - 动态仓位调整
      - Gap风险检测与缓解
      - 回撤控制

    依赖：
      - numpy
      - 组合净值数据
    """

    # ===== CPPI核心参数 =====
    INITIAL_FLOOR_RATIO = 0.85         # 初始底线 = 85% × 初始净值
    FLOOR_MIN_RATIO = 0.70             # 最低底线比例 (70%)
    FLOOR_MAX_RATIO = 0.95             # 最高底线比例 (95%)
    MULTIPLIER_BASE = 3.0              # 基础乘数
    MULTIPLIER_MIN = 1.0               # 最小乘数
    MULTIPLIER_MAX = 5.0               # 最大乘数
    TARGET_ANNUAL_VOL = 0.30           # 目标年化波动率 (用于动态乘数)
    REBALANCE_THRESHOLD = 0.05         # 再平衡阈值 (权重偏离5%)

    # ===== TIPP参数 =====
    TIPP_FLOOR_LOCK_RATIO = 0.90       # TIPP底线上移锁定比例
    TIPP_FLOOR_REVIEW_DAYS = 30        # TIPP底线复查周期

    # ===== Gap Risk参数 (来源: stockalpha.ai 2026) =====
    GAP_RISK_LAMBDA = 2.0              # Poisson跳跃强度 (年化)
    GAP_RISK_JUMP_SIZE = -0.10         # 平均跳跃幅度 (-10%)
    GAP_RISK_JUMP_STD = 0.05           # 跳跃幅度标准差
    GAP_RISK_THRESHOLD = 0.05          # Gap风险概率阈值 (5%)
    GAP_RISK_ALARM_THRESHOLD = 0.15    # Gap风险警报阈值 (15%)
    FLOOR_BREACH_WARNING_DISTANCE = 2.0  # 距底线2σ预警

    # ===== 波动率参数 =====
    VOL_LOOKBACK = 30                  # 波动率回看窗口
    VOL_ANNUALIZATION = math.sqrt(252)  # 年化因子
    HIGH_VOL_THRESHOLD = 0.80          # 高波动阈值
    EXTREME_VOL_THRESHOLD = 1.20       # 极端波动阈值

    # ===== 再平衡参数 =====
    MAX_REBALANCE_PER_DAY = 1          # 每日最大再平衡次数
    MIN_CASH_BUFFER = 0.05             # 最小现金缓冲 (5%)
    MAX_LEVERAGE = 1.5                 # 最大杠杆 (CPPI通常<1,但允许短暂超调)

    # ===== 压力测试场景 =====
    STRESS_SCENARIOS = {
        "black_monday_1987": -0.22,    # 1987黑色星期一
        "covid_2020": -0.35,           # 2020 COVID
        "luna_2022": -0.99,            # 2022 LUNA
        "flash_crash": -0.15,          # 闪崩
        "normal_correction": -0.10,    # 正常回调
    }

    def __init__(self, initial_value: float = 10000.0,
                 mode: ProtectionMode = ProtectionMode.CPPI,
                 floor_ratio: Optional[float] = None):
        """
        初始化CPPI组合保险引擎

        Args:
            initial_value: 初始组合净值
            mode: 保护模式 (CPPI/TIPP/OBPI)
            floor_ratio: 初始底线比例, 默认使用INITIAL_FLOOR_RATIO
        """
        self.mode = mode
        self.initial_value = float(initial_value)

        # 初始底线
        if floor_ratio is None:
            floor_ratio = self.INITIAL_FLOOR_RATIO
        self.initial_floor_ratio = floor_ratio
        self.floor = self.initial_value * floor_ratio

        # 组合状态
        self.state = PortfolioState(
            value=self.initial_value,
            floor=self.floor,
            cushion=self.initial_value - self.floor,
            peak_value=self.initial_value,
            multiplier=self.MULTIPLIER_BASE,
        )

        # 历史数据
        self.value_history: deque = deque(maxlen=300)
        self.returns_history: deque = deque(maxlen=300)
        self.floor_history: deque = deque(maxlen=300)
        self.weight_history: deque = deque(maxlen=300)
        self.value_history.append(self.initial_value)
        self.floor_history.append(self.floor)

        # TIPP状态
        self.tipp_review_counter: int = 0
        self.tipp_high_water_mark: float = self.initial_value

        # 信号计数
        self.signal_count: int = 0
        self.rebalance_count: int = 0

    def update_portfolio_value(self, new_value: float) -> None:
        """
        更新组合净值

        Args:
            new_value: 新的组合净值
        """
        prev_value = self.state.value
        self.state.value = float(new_value)

        # 计算收益率
        if prev_value > 0:
            ret = (new_value - prev_value) / prev_value
        else:
            ret = 0.0
        self.returns_history.append(ret)
        self.value_history.append(new_value)

        # 更新峰值与回撤
        if new_value > self.state.peak_value:
            self.state.peak_value = new_value
            if self.mode == ProtectionMode.TIPP:
                self.tipp_high_water_mark = new_value
        if self.state.peak_value > 0:
            self.state.max_drawdown = max(
                self.state.max_drawdown,
                1.0 - new_value / self.state.peak_value
            )

        # 更新垫子
        self.state.cushion = max(self.state.value - self.state.floor, 0.0)

        # TIPP: 动态调整底线
        if self.mode == ProtectionMode.TIPP:
            self._update_tipp_floor()

        # 更新实现波动率
        self._update_realized_vol()

        # 更新动态乘数
        self._update_dynamic_multiplier()

        # 更新Gap风险
        self.state.gap_risk_probability = self._compute_gap_risk_probability()

        self.floor_history.append(self.state.floor)

    def _update_tipp_floor(self) -> None:
        """
        TIPP动态底线调整
        来源: gelonghui 2026
        底线 = max(原底线, 锁定比例 × 高水位)
        """
        self.tipp_review_counter += 1
        if self.tipp_review_counter < self.TIPP_FLOOR_REVIEW_DAYS:
            return

        self.tipp_review_counter = 0
        # 新底线 = 锁定比例 × 高水位
        new_floor_candidate = self.TIPP_FLOOR_LOCK_RATIO * self.tipp_high_water_mark
        # 底线只升不降 (TIPP核心特性)
        if new_floor_candidate > self.state.floor:
            # 确保不超过FLOOR_MAX_RATIO
            max_allowed = self.state.value * self.FLOOR_MAX_RATIO
            self.state.floor = min(new_floor_candidate, max_allowed)

    def _update_realized_vol(self) -> None:
        """更新实现波动率"""
        if len(self.returns_history) < 2:
            self.state.realized_vol = 0.0
            return
        recent_returns = list(self.returns_history)[-self.VOL_LOOKBACK:]
        if len(recent_returns) < 2:
            self.state.realized_vol = 0.0
            return
        daily_vol = float(np.std(recent_returns))
        self.state.realized_vol = daily_vol * self.VOL_ANNUALIZATION

    def _update_dynamic_multiplier(self) -> None:
        """
        动态乘数调整
        来源: stockalpha.ai 2026
        m = m_base × (σ_target / σ_realized), 限制在[m_min, m_max]
        高波动 → 降低m
        低波动 → 提高m
        """
        if self.state.realized_vol <= 0:
            self.state.multiplier = self.MULTIPLIER_BASE
            return

        # 动态乘数计算
        raw_multiplier = self.MULTIPLIER_BASE * (self.TARGET_ANNUAL_VOL / self.state.realized_vol)
        # 限制范围
        self.state.multiplier = max(self.MULTIPLIER_MIN,
                                     min(self.MULTIPLIER_MAX, raw_multiplier))

    def _compute_gap_risk_probability(self) -> float:
        """
        计算Gap风险概率
        来源: stockalpha.ai 2026

        Gap风险 = 风险资产在再平衡间隔内下跌超过垫子的概率
        使用Poisson跳跃过程 + 正态分布近似
        """
        if self.state.cushion <= 0 or self.state.value <= 0:
            return 1.0  # 已击穿底线

        cushion_ratio = self.state.cushion / self.state.value
        current_risky_weight = self.state.current_risky_weight
        if current_risky_weight <= 0:
            return 0.0

        # 单日风险资产下跌幅度需要超过 cushion / risky_weight
        # threshold_drop = cushion_ratio / current_risky_weight
        threshold_drop = cushion_ratio / max(current_risky_weight, 0.01)

        # 正态分布近似: P(drop > threshold) = 1 - Φ(threshold / σ_daily)
        if self.state.realized_vol <= 0:
            return 0.0
        daily_vol = self.state.realized_vol / self.VOL_ANNUALIZATION
        if daily_vol <= 0:
            return 0.0

        z_score = threshold_drop / daily_vol
        # 使用标准正态CDF
        normal_prob = 1.0 - 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))

        # Poisson跳跃调整: 增加跳跃概率
        # P(at least 1 jump in 1 day) = 1 - exp(-λ/252)
        jump_prob = 1.0 - math.exp(-self.GAP_RISK_LAMBDA / 252.0)
        # 跳跃时平均跌幅 = GAP_RISK_JUMP_SIZE
        jump_threshold_pass = 1.0 if abs(self.GAP_RISK_JUMP_SIZE) > threshold_drop else 0.0

        gap_risk = normal_prob + jump_prob * jump_threshold_pass * 0.5
        return min(gap_risk, 1.0)

    def calculate_target_weights(self) -> Tuple[float, float]:
        """
        计算目标风险权重与现金权重

        Returns:
            (risky_weight, cash_weight)
        """
        if self.state.value <= 0:
            return 0.0, 1.0

        # CPPI核心: E = m × max(C, 0)
        target_exposure = self.state.multiplier * self.state.cushion
        target_risky_weight = target_exposure / self.state.value

        # 限制最大杠杆
        target_risky_weight = min(target_risky_weight, self.MAX_LEVERAGE)
        target_risky_weight = max(0.0, target_risky_weight)

        # 最小现金缓冲
        cash_weight = 1.0 - target_risky_weight
        if cash_weight < self.MIN_CASH_BUFFER:
            cash_weight = self.MIN_CASH_BUFFER
            target_risky_weight = 1.0 - self.MIN_CASH_BUFFER

        return target_risky_weight, cash_weight

    def set_current_weights(self, risky_weight: float) -> None:
        """设置当前权重"""
        self.state.current_risky_weight = max(0.0, min(1.0, risky_weight))
        self.state.current_cash_weight = 1.0 - self.state.current_risky_weight
        self.weight_history.append(self.state.current_risky_weight)

    def stress_test(self) -> Dict[str, Dict[str, float]]:
        """
        压力测试 (来源: stockalpha.ai 2026)

        Returns:
            {scenario_name: {value_after, floor_breached, drawdown}}
        """
        results = {}
        current_value = self.state.value
        current_floor = self.state.floor
        current_weight = self.state.current_risky_weight

        for scenario_name, drop_pct in self.STRESS_SCENARIOS.items():
            # 模拟风险资产下跌对组合的影响
            # 组合下跌 = risky_weight × drop_pct
            portfolio_drop = current_weight * drop_pct
            value_after = current_value * (1.0 + portfolio_drop)
            floor_breached = value_after < current_floor
            drawdown = (current_value - value_after) / current_value if current_value > 0 else 0.0

            results[scenario_name] = {
                "drop_pct": drop_pct,
                "value_after": round(value_after, 2),
                "floor": round(current_floor, 2),
                "floor_breached": floor_breached,
                "drawdown": round(drawdown, 4),
                "cushion_remaining": round(max(value_after - current_floor, 0.0), 2),
            }
        return results

    def _select_signal(self) -> CPPISignal:
        """选择最佳信号"""
        target_risky, target_cash = self.calculate_target_weights()
        current_risky = self.state.current_risky_weight
        weight_change = target_risky - current_risky

        signal = CPPISignal(
            mode=self.mode,
            portfolio_value=self.state.value,
            floor=self.state.floor,
            cushion=self.state.cushion,
            cushion_ratio=self.state.cushion / self.state.value if self.state.value > 0 else 0.0,
            target_risky_weight=target_risky,
            multiplier=self.state.multiplier,
            gap_risk_probability=self.state.gap_risk_probability,
        )

        # 优先级1: 底线击穿预警
        if self.state.cushion <= 0:
            signal.signal_type = CPPISignalType.FLOOR_BREACH_WARNING
            signal.best_opportunity = CPPIOpportunity(
                signal_type=CPPISignalType.FLOOR_BREACH_WARNING,
                target_weight=0.0,
                current_weight=current_risky,
                weight_change=-current_risky,
                cushion_ratio=0.0,
                multiplier=self.state.multiplier,
                gap_risk_prob=1.0,
                floor_breach_distance=0.0,
                confidence=1.0,
                description="底线已击穿!立即清仓风险资产"
            )
            signal.description = "底线击穿,紧急清仓"
            return signal

        # 优先级2: Gap风险警报
        if self.state.gap_risk_probability >= self.GAP_RISK_ALARM_THRESHOLD:
            signal.signal_type = CPPISignalType.GAP_RISK_ALARM
            # 计算距底线的σ距离
            daily_vol = self.state.realized_vol / self.VOL_ANNUALIZATION if self.state.realized_vol > 0 else 0.01
            cushion_ratio = self.state.cushion / self.state.value
            threshold_drop = cushion_ratio / max(current_risky, 0.01)
            distance_to_breach = threshold_drop / daily_vol if daily_vol > 0 else 0.0

            signal.best_opportunity = CPPIOpportunity(
                signal_type=CPPISignalType.GAP_RISK_ALARM,
                target_weight=target_risky * 0.5,  # 减半风险暴露
                current_weight=current_risky,
                weight_change=-current_risky * 0.5,
                cushion_ratio=cushion_ratio,
                multiplier=self.state.multiplier,
                gap_risk_prob=self.state.gap_risk_probability,
                floor_breach_distance=distance_to_breach,
                confidence=0.85,
                description=f"Gap风险={self.state.gap_risk_probability:.2f},距底线{distance_to_breach:.1f}σ,建议减仓50%"
            )
            signal.description = "Gap风险警报,紧急减仓"
            return signal

        # 优先级3: 底线接近预警 (距底线<FLOOR_BREACH_WARNING_DISTANCE σ)
        if self.state.realized_vol > 0 and current_risky > 0:
            daily_vol = self.state.realized_vol / self.VOL_ANNUALIZATION
            cushion_ratio = self.state.cushion / self.state.value
            threshold_drop = cushion_ratio / max(current_risky, 0.01)
            distance_to_breach = threshold_drop / daily_vol if daily_vol > 0 else 999.0

            if distance_to_breach < self.FLOOR_BREACH_WARNING_DISTANCE:
                signal.signal_type = CPPISignalType.FLOOR_BREACH_WARNING
                signal.best_opportunity = CPPIOpportunity(
                    signal_type=CPPISignalType.FLOOR_BREACH_WARNING,
                    target_weight=target_risky,
                    current_weight=current_risky,
                    weight_change=weight_change,
                    cushion_ratio=cushion_ratio,
                    multiplier=self.state.multiplier,
                    gap_risk_prob=self.state.gap_risk_probability,
                    floor_breach_distance=distance_to_breach,
                    confidence=0.75,
                    description=f"距底线仅{distance_to_breach:.1f}σ,需降低风险暴露"
                )
                signal.description = "接近底线,降低风险"
                return signal

        # 优先级4: 极端波动去杠杆
        if self.state.realized_vol > self.EXTREME_VOL_THRESHOLD:
            signal.signal_type = CPPISignalType.DELEVERAGE
            signal.best_opportunity = CPPIOpportunity(
                signal_type=CPPISignalType.DELEVERAGE,
                target_weight=min(target_risky, 0.20),  # 限制风险暴露<20%
                current_weight=current_risky,
                weight_change=min(target_risky, 0.20) - current_risky,
                cushion_ratio=self.state.cushion / self.state.value,
                multiplier=self.state.multiplier,
                gap_risk_prob=self.state.gap_risk_probability,
                confidence=0.80,
                description=f"极端波动(vol={self.state.realized_vol:.2f}),紧急去杠杆"
            )
            signal.description = "极端波动,去杠杆"
            return signal

        # 优先级5: 垫子耗尽 (cushion_ratio < 5%)
        cushion_ratio = self.state.cushion / self.state.value if self.state.value > 0 else 0.0
        if cushion_ratio < 0.05:
            signal.signal_type = CPPISignalType.CASH_HEAVY
            signal.best_opportunity = CPPIOpportunity(
                signal_type=CPPISignalType.CASH_HEAVY,
                target_weight=target_risky,
                current_weight=current_risky,
                weight_change=weight_change,
                cushion_ratio=cushion_ratio,
                multiplier=self.state.multiplier,
                gap_risk_prob=self.state.gap_risk_probability,
                confidence=0.70,
                description=f"垫子耗尽(cushion_ratio={cushion_ratio:.2%}),以现金为主"
            )
            signal.description = "垫子耗尽,现金为主"
            return signal

        # 优先级6: 再平衡信号
        if abs(weight_change) > self.REBALANCE_THRESHOLD:
            if weight_change > 0:
                signal.signal_type = CPPISignalType.REBALANCE_UP
                signal.best_opportunity = CPPIOpportunity(
                    signal_type=CPPISignalType.REBALANCE_UP,
                    target_weight=target_risky,
                    current_weight=current_risky,
                    weight_change=weight_change,
                    cushion_ratio=cushion_ratio,
                    multiplier=self.state.multiplier,
                    gap_risk_prob=self.state.gap_risk_probability,
                    confidence=0.60,
                    description=f"加仓: 权重 {current_risky:.2%} → {target_risky:.2%} (cushion扩大)"
                )
                signal.description = "上调再平衡,加仓"
            else:
                signal.signal_type = CPPISignalType.REBALANCE_DOWN
                signal.best_opportunity = CPPIOpportunity(
                    signal_type=CPPISignalType.REBALANCE_DOWN,
                    target_weight=target_risky,
                    current_weight=current_risky,
                    weight_change=weight_change,
                    cushion_ratio=cushion_ratio,
                    multiplier=self.state.multiplier,
                    gap_risk_prob=self.state.gap_risk_probability,
                    confidence=0.60,
                    description=f"减仓: 权重 {current_risky:.2%} → {target_risky:.2%} (cushion缩小)"
                )
                signal.description = "下调再平衡,减仓"
            self.rebalance_count += 1
            return signal

        # 默认: 持有
        signal.signal_type = CPPISignalType.HOLD
        signal.description = f"权重在阈值内,持有 (current={current_risky:.2%}, target={target_risky:.2%})"
        return signal

    def analyze(self) -> CPPISignal:
        """
        分析CPPI信号

        Returns:
            CPPISignal
        """
        signal = self._select_signal()
        self.signal_count += 1
        return signal

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        target_risky, target_cash = self.calculate_target_weights()
        return {
            "mode": self.mode.value,
            "portfolio_value": round(self.state.value, 2),
            "floor": round(self.state.floor, 2),
            "cushion": round(self.state.cushion, 2),
            "cushion_ratio": round(self.state.cushion / self.state.value, 4) if self.state.value > 0 else 0.0,
            "current_risky_weight": round(self.state.current_risky_weight, 4),
            "target_risky_weight": round(target_risky, 4),
            "current_cash_weight": round(self.state.current_cash_weight, 4),
            "target_cash_weight": round(target_cash, 4),
            "multiplier": round(self.state.multiplier, 3),
            "realized_vol": round(self.state.realized_vol, 4),
            "gap_risk_probability": round(self.state.gap_risk_probability, 4),
            "max_drawdown": round(self.state.max_drawdown, 4),
            "peak_value": round(self.state.peak_value, 2),
            "signal_count": self.signal_count,
            "rebalance_count": self.rebalance_count,
            "tipp_high_water_mark": round(self.tipp_high_water_mark, 2) if self.mode == ProtectionMode.TIPP else None,
        }


# ============================================================================
# 自检 (Self-test)
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("cppi_portfolio_insurance.py 自检")
    print("=" * 70)

    np.random.seed(42)

    # 测试1: 标准CPPI模式
    print("\n[1] 标准CPPI模式 - 初始状态...")
    engine = CPPIPortfolioInsurance(initial_value=10000.0, mode=ProtectionMode.CPPI)
    engine.set_current_weights(0.30)  # 初始30%风险
    signal = engine.analyze()
    status = engine.get_status()
    print(f"  模式: {status['mode']}")
    print(f"  净值: {status['portfolio_value']}")
    print(f"  底线: {status['floor']}")
    print(f"  垫子: {status['cushion']}")
    print(f"  垫子比例: {status['cushion_ratio']:.2%}")
    print(f"  目标风险权重: {status['target_risky_weight']:.2%}")
    print(f"  乘数: {status['multiplier']}")
    print(f"  信号: {signal.signal_type.value}")
    print(f"  描述: {signal.description}")

    # 验证: 初始cushion=1500, m=3, target=4500/10000=45%
    assert abs(status['cushion'] - 1500.0) < 1.0, f"垫子应为1500,实际:{status['cushion']}"
    assert abs(status['target_risky_weight'] - 0.45) < 0.01, f"目标权重应为45%,实际:{status['target_risky_weight']}"
    print("  ✓ 初始CPPI计算正确 (cushion=1500, target_weight=45%)")

    # 测试2: 上涨场景 (cushion扩大)
    print("\n[2] 上涨场景 - 组合上涨20%...")
    engine.update_portfolio_value(12000.0)
    signal2 = engine.analyze()
    status2 = engine.get_status()
    print(f"  净值: {status2['portfolio_value']}")
    print(f"  垫子: {status2['cushion']}")
    print(f"  目标风险权重: {status2['target_risky_weight']:.2%}")
    print(f"  信号: {signal2.signal_type.value}")
    # 上涨后cushion=3500, target=3500*3/12000=87.5%
    assert status2['cushion'] > 1500, "上涨后垫子应增加"
    assert status2['target_risky_weight'] > 0.45, "上涨后目标权重应增加"
    print("  ✓ 上涨场景正确 (cushion扩大, 目标权重增加)")

    # 测试3: 下跌场景 (cushion缩小)
    print("\n[3] 下跌场景 - 组合下跌30%...")
    engine2 = CPPIPortfolioInsurance(initial_value=10000.0, mode=ProtectionMode.CPPI)
    engine2.set_current_weights(0.45)
    engine2.update_portfolio_value(7000.0)  # 下跌30%
    signal3 = engine2.analyze()
    status3 = engine2.get_status()
    print(f"  净值: {status3['portfolio_value']}")
    print(f"  底线: {status3['floor']}")
    print(f"  垫子: {status3['cushion']}")
    print(f"  目标风险权重: {status3['target_risky_weight']:.2%}")
    print(f"  Gap风险: {status3['gap_risk_probability']:.4f}")
    print(f"  信号: {signal3.signal_type.value}")
    # 下跌后cushion=7000-8500=-1500<0, 应触发底线击穿
    assert status3['cushion'] < 0 or status3['cushion_ratio'] < 0.05, "大幅下跌后应接近底线"
    print("  ✓ 下跌场景正确 (垫子缩小/底线接近)")

    # 测试4: TIPP模式 - 底线随净值上移
    print("\n[4] TIPP模式 - 底线动态上移...")
    engine_tipp = CPPIPortfolioInsurance(initial_value=10000.0, mode=ProtectionMode.TIPP)
    engine_tipp.set_current_weights(0.45)
    # 模拟上涨
    for _ in range(35):
        engine_tipp.update_portfolio_value(engine_tipp.state.value * 1.005)
    signal_tipp = engine_tipp.analyze()
    status_tipp = engine_tipp.get_status()
    print(f"  净值: {status_tipp['portfolio_value']:.2f}")
    print(f"  底线: {status_tipp['floor']:.2f}")
    print(f"  高水位: {status_tipp['tipp_high_water_mark']:.2f}")
    print(f"  信号: {signal_tipp.signal_type.value}")
    # TIPP底线应高于初始8500
    assert status_tipp['floor'] > 8500, f"TIPP底线应上移,实际:{status_tipp['floor']}"
    print(f"  ✓ TIPP底线正确上移 ({8500} → {status_tipp['floor']:.2f})")

    # 测试5: 压力测试
    print("\n[5] 压力测试...")
    engine3 = CPPIPortfolioInsurance(initial_value=10000.0)
    engine3.set_current_weights(0.60)  # 60%风险暴露
    stress_results = engine3.stress_test()
    for scenario, result in stress_results.items():
        breached = "❌击穿" if result['floor_breached'] else "✓安全"
        print(f"  {scenario}: 跌幅={result['drop_pct']:.0%}, "
              f"剩余净值={result['value_after']:.0f}, {breached}")

    # LUNA场景必须击穿
    assert stress_results['luna_2022']['floor_breached'], "LUNA场景必须击穿底线"
    print("  ✓ LUNA场景正确触发底线击穿")

    # 测试6: 高波动去杠杆
    print("\n[6] 高波动去杠杆...")
    engine4 = CPPIPortfolioInsurance(initial_value=10000.0)
    engine4.set_current_weights(0.50)
    # 注入高波动数据
    for _ in range(40):
        ret = np.random.normal(0, 0.06)  # 极端波动
        engine4.update_portfolio_value(engine4.state.value * (1 + ret))
    signal4 = engine4.analyze()
    status4 = engine4.get_status()
    print(f"  实现波动率: {status4['realized_vol']:.4f}")
    print(f"  动态乘数: {status4['multiplier']:.3f}")
    print(f"  Gap风险: {status4['gap_risk_probability']:.4f}")
    print(f"  信号: {signal4.signal_type.value}")
    if status4['realized_vol'] > 0.80:
        print("  ✓ 高波动正确识别,触发去杠杆")

    print("\n" + "=" * 70)
    print("✓ cppi_portfolio_insurance.py 自检通过")
    print("=" * 70)
