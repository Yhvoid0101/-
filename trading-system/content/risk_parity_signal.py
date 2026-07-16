# -*- coding: utf-8 -*-
"""
风险平价策略级信号引擎 (Risk Parity Signal) — v3.4 新增

填补 7 维覆盖矩阵中的 "风控策略级信号" 与 "Kelly 策略级" 两个空白。

核心理论:
  1. 波动率目标 (Volatility Targeting) — 动态调整仓位使组合波动率锚定目标
     来源: Moreira & Muir (2017) "Volatility-Managed Portfolios" (JF)
  2. 风险预算 (Risk Budgeting) — 各资产按风险贡献而非资金权重分配
     来源: Maillard, Roncalli & Teiletche (2010)
  3. Dynamic Kelly (策略级) — 滚动胜率+赔率计算最优 Kelly 分数
     来源: Kelly (1956) + Thorp (1969) + Vince (1990) "Optimal f"
  4. CVaR 约束 — 在尾部风险约束下优化仓位
     来源: Rockafellar & Uryasev (2000)
  5. 回撤控制 — 动态减仓以控制最大回撤
     来源: Bruch & Mitchel (2013) "Drawdown Control"

学术实证:
  - Volatility Targeting: 年化 Sharpe 提升 0.2-0.4 (Moreira & Muir 2017)
  - Dynamic Kelly: 长期年化提升 30-50%, 但需配合分数 Kelly 防止过拟合
  - CVaR 约束: 极端事件回撤减少 25-40%

信号类型 (RiskParitySignalType):
  VOLATILITY_TARGET_LONG   — 波动率低于目标, 加仓做多
  VOLATILITY_TARGET_SHORT  — 波动率高于目标, 减仓
  RISK_PARITY_REBALANCE    — 风险贡献失衡, 再平衡
  KELLY_INCREASE           — Kelly 分数上升, 增加仓位
  KELLY_DECREASE           — Kelly 分数下降, 减少仓位
  TAIL_RISK_HEDGE          — CVaR 超标, 启动尾部对冲
  DRAWDOWN_CONTROL         — 回撤接近上限, 强制减仓
  HOLD                     — 持仓观望
  NEUTRAL                  — 中性

纯 numpy 实现, 沙盘中可即时进化。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# 枚举定义
# ============================================================================

class RiskParitySignalType(Enum):
    """风险平价信号类型"""
    VOLATILITY_TARGET_LONG = "vol_target_long"     # 波动率低,加仓
    VOLATILITY_TARGET_SHORT = "vol_target_short"   # 波动率高,减仓
    RISK_PARITY_REBALANCE = "risk_parity_rebalance"  # 风险贡献失衡
    KELLY_INCREASE = "kelly_increase"              # Kelly 上升
    KELLY_DECREASE = "kelly_decrease"              # Kelly 下降
    TAIL_RISK_HEDGE = "tail_risk_hedge"            # 尾部对冲
    DRAWDOWN_CONTROL = "drawdown_control"          # 回撤控制
    HOLD = "hold"
    NEUTRAL = "neutral"


class VolatilityRegime(Enum):
    """波动率状态"""
    LOW_VOL = "low_vol"          # 低波动(< 目标*0.7)
    NORMAL_VOL = "normal_vol"    # 正常波动
    HIGH_VOL = "high_vol"        # 高波动(> 目标*1.3)
    EXTREME_VOL = "extreme_vol"  # 极端波动(> 目标*2.0)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class AssetRiskState:
    """单个资产/策略的风险状态"""
    name: str
    current_weight: float = 0.0          # 当前资金权重
    current_volatility: float = 0.0      # 当前年化波动率
    target_weight: float = 0.0           # 目标权重
    risk_contribution: float = 0.0       # 风险贡献
    sharpe: float = 0.0                  # 滚动夏普
    cvar: float = 0.0                    # 条件风险价值
    kelly_fraction: float = 0.0          # Kelly 分数


@dataclass
class RiskParityOpportunity:
    """风险平价机会"""
    signal_type: RiskParitySignalType
    asset: str
    current_weight: float
    target_weight: float
    conviction: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskParitySignal:
    """风险平价综合信号"""
    signal_type: RiskParitySignalType
    direction: int                         # +1 加仓 / -1 减仓 / 0 观望
    conviction: float
    target_weight_adjustment: float        # 建议的权重调整幅度 [-1, 1]
    vol_regime: VolatilityRegime
    portfolio_volatility: float
    target_volatility: float
    kelly_fraction: float                  # 当前组合 Kelly 分数
    portfolio_cvar: float
    current_drawdown: float
    opportunities: List[RiskParityOpportunity] = field(default_factory=list)


# ============================================================================
# 主类: RiskParitySignalGenerator
# ============================================================================

class RiskParitySignalGenerator:
    """
    风险平价策略级信号生成器

    填补维度:
      - 风控覆盖: Kelly 策略级 (此前仅 frontier_enhancement 基础设施层)
      - 信号类型: 风控驱动交易信号 (此前无)
    """

    # ------------------------------------------------------------------
    # 类常量
    # ------------------------------------------------------------------
    TARGET_ANNUAL_VOL = 0.20               # 目标年化波动率 20%
    VOL_LOOKBACK = 60                       # 波动率计算窗口
    RETURNS_LOOKBACK = 100                  # 收益率历史窗口
    CVAR_CONFIDENCE = 0.95                  # CVaR 置信度
    MAX_DRAWDOWN_LIMIT = 0.15              # 最大回撤上限 15%
    DRAWDOWN_WARNING = 0.10                # 回撤预警阈值 10%

    # Kelly 参数
    KELLY_LOOKBACK = 50                     # Kelly 计算窗口
    KELLY_FRACTION = 0.5                    # 分数 Kelly (防止过拟合)
    KELLY_MIN = 0.0                         # Kelly 下限
    KELLY_MAX = 0.25                        # Kelly 上限 (4x杠杆)

    # 波动率状态阈值
    LOW_VOL_RATIO = 0.7                     # 低波动 = 目标 * 0.7
    HIGH_VOL_RATIO = 1.3                    # 高波动 = 目标 * 1.3
    EXTREME_VOL_RATIO = 2.0                 # 极端波动 = 目标 * 2.0

    # 信号权重
    WEIGHT_VOLATILITY = 0.30
    WEIGHT_KELLY = 0.25
    WEIGHT_RISK_PARITY = 0.20
    WEIGHT_TAIL_RISK = 0.15
    WEIGHT_DRAWDOWN = 0.10

    ANNUALIZATION_FACTOR = math.sqrt(252)   # 假设日频数据

    def __init__(self, n_assets: int = 3, asset_names: Optional[List[str]] = None):
        """
        Args:
            n_assets: 资产数量
            asset_names: 资产名称列表 (可选)
        """
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n_assets)]

        # 资产风险状态
        self.asset_states: List[AssetRiskState] = [
            AssetRiskState(name=name) for name in self.asset_names
        ]

        # 收益率历史 (每资产一条 deque)
        self.returns_history: List[deque] = [
            deque(maxlen=self.RETURNS_LOOKBACK) for _ in range(n_assets)
        ]

        # 组合状态
        self.portfolio_returns: deque = deque(maxlen=self.RETURNS_LOOKBACK)
        self.portfolio_volatility: float = 0.0
        self.portfolio_value: float = 1.0
        self.portfolio_peak: float = 1.0
        self.current_drawdown: float = 0.0
        self.max_drawdown_seen: float = 0.0
        self.portfolio_cvar: float = 0.0
        self.portfolio_sharpe: float = 0.0

        # Kelly
        self.current_kelly_fraction: float = 0.0
        self.target_kelly_fraction: float = 0.0

        # 波动率状态
        self.vol_regime: VolatilityRegime = VolatilityRegime.NORMAL_VOL

        # 协方差矩阵
        self.cov_matrix: np.ndarray = np.eye(n_assets)

        # 统计
        self.signal_count: int = 0
        self.rebalance_count: int = 0
        self.tail_hedge_count: int = 0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_returns(self, asset_returns: List[float],
                       portfolio_return: Optional[float] = None) -> None:
        """
        更新收益率数据

        Args:
            asset_returns: 各资产的当期收益率
            portfolio_return: 组合当期收益率 (可选, 自动计算)
        """
        if len(asset_returns) != self.n_assets:
            return

        for i, ret in enumerate(asset_returns):
            self.returns_history[i].append(ret)

        # 计算组合收益
        if portfolio_return is None:
            weights = np.array([s.current_weight for s in self.asset_states])
            if weights.sum() > 0:
                weights = weights / weights.sum()
            portfolio_return = float(np.dot(weights, asset_returns))

        self.portfolio_returns.append(portfolio_return)

        # 更新组合净值
        self.portfolio_value *= (1.0 + portfolio_return)
        if self.portfolio_value > self.portfolio_peak:
            self.portfolio_peak = self.portfolio_value
        self.current_drawdown = max(0.0,
            (self.portfolio_peak - self.portfolio_value) / max(self.portfolio_peak, 1e-8))
        if self.current_drawdown > self.max_drawdown_seen:
            self.max_drawdown_seen = self.current_drawdown

        # 更新统计
        self._update_volatility()
        self._update_covariance()
        self._update_risk_contributions()
        self._update_cvar()
        self._update_kelly()
        self._update_sharpe()
        self._detect_vol_regime()

    def _update_volatility(self) -> None:
        """更新组合波动率"""
        if len(self.portfolio_returns) < 10:
            return

        returns = np.array(list(self.portfolio_returns))
        # 日波动率
        daily_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        # 年化
        self.portfolio_volatility = daily_vol * self.ANNUALIZATION_FACTOR

        # 各资产波动率
        for i in range(self.n_assets):
            if len(self.returns_history[i]) >= 10:
                asset_returns = np.array(list(self.returns_history[i]))
                daily_vol_i = float(np.std(asset_returns, ddof=1))
                self.asset_states[i].current_volatility = (
                    daily_vol_i * self.ANNUALIZATION_FACTOR)

    def _update_covariance(self) -> None:
        """更新协方差矩阵"""
        min_len = min(len(h) for h in self.returns_history)
        if min_len < 10:
            return

        # 取最近的 VOL_LOOKBACK 个数据点
        data = np.array([
            list(h)[-self.VOL_LOOKBACK:] for h in self.returns_history
        ]).T  # shape: [T, N]

        if data.shape[0] < 10:
            return

        try:
            self.cov_matrix = np.cov(data.T)
            if data.shape[1] == 1:
                self.cov_matrix = np.array([[float(self.cov_matrix)]])
        except Exception:
            pass

    def _update_risk_contributions(self) -> None:
        """更新各资产的风险贡献"""
        weights = np.array([s.current_weight for s in self.asset_states])
        total_weight = weights.sum()
        if total_weight <= 0:
            for s in self.asset_states:
                s.risk_contribution = 0.0
            return

        weights_norm = weights / total_weight

        # 组合波动率 = sqrt(w^T Σ w)
        try:
            port_var = float(weights_norm @ self.cov_matrix @ weights_norm)
            if port_var <= 0:
                return
            port_vol = math.sqrt(port_var)

            # 边际风险贡献 MRC_i = (Σ w)_i / sqrt(w^T Σ w)
            mrc = self.cov_matrix @ weights_norm / max(port_vol, 1e-8)

            # 风险贡献 RC_i = w_i * MRC_i
            rc = weights_norm * mrc
            rc_total = rc.sum()
            if rc_total > 0:
                for i in range(self.n_assets):
                    self.asset_states[i].risk_contribution = float(rc[i] / rc_total)
        except Exception:
            pass

    def _update_cvar(self) -> None:
        """更新组合 CVaR"""
        if len(self.portfolio_returns) < 30:
            return

        returns = np.array(list(self.portfolio_returns))
        var_threshold = float(np.percentile(returns, (1 - self.CVAR_CONFIDENCE) * 100))
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) > 0:
            # CVaR = 尾部收益的均值 (负值)
            self.portfolio_cvar = float(np.mean(tail_returns))

    def _update_kelly(self) -> None:
        """
        更新 Kelly 分数 (策略级)

        Kelly 公式: f* = (p * b - q) / b
          p = 胜率, q = 1 - p, b = 平均盈利 / 平均亏损
        """
        if len(self.portfolio_returns) < self.KELLY_LOOKBACK:
            return

        recent = list(self.portfolio_returns)[-self.KELLY_LOOKBACK:]
        wins = [r for r in recent if r > 0]
        losses = [r for r in recent if r < 0]

        if not wins or not losses:
            self.target_kelly_fraction = 0.0
            return

        p = len(wins) / len(recent)  # 胜率
        q = 1.0 - p
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss < 1e-8:
            return

        b = avg_win / avg_loss  # 赔率

        # Kelly 分数
        kelly_raw = (p * b - q) / b
        # 分数 Kelly (防止过拟合)
        kelly_capped = max(self.KELLY_MIN, min(self.KELLY_MAX, kelly_raw * self.KELLY_FRACTION))

        self.target_kelly_fraction = kelly_capped
        # 平滑过渡
        alpha = 0.2
        self.current_kelly_fraction = (
            (1 - alpha) * self.current_kelly_fraction + alpha * kelly_capped
        )

        # 更新各资产 Kelly
        for i in range(self.n_assets):
            if len(self.returns_history[i]) >= self.KELLY_LOOKBACK:
                asset_recent = list(self.returns_history[i])[-self.KELLY_LOOKBACK:]
                a_wins = [r for r in asset_recent if r > 0]
                a_losses = [r for r in asset_recent if r < 0]
                if a_wins and a_losses:
                    a_p = len(a_wins) / len(asset_recent)
                    a_q = 1.0 - a_p
                    a_avg_win = sum(a_wins) / len(a_wins)
                    a_avg_loss = abs(sum(a_losses) / len(a_losses))
                    if a_avg_loss > 1e-8:
                        a_b = a_avg_win / a_avg_loss
                        a_kelly = (a_p * a_b - a_q) / a_b
                        self.asset_states[i].kelly_fraction = max(
                            0.0, min(self.KELLY_MAX, a_kelly * self.KELLY_FRACTION))

    def _update_sharpe(self) -> None:
        """更新组合夏普"""
        if len(self.portfolio_returns) < 30:
            return
        returns = np.array(list(self.portfolio_returns))
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.001
        if std_r < 1e-8:
            return
        self.portfolio_sharpe = (mean_r / std_r) * self.ANNUALIZATION_FACTOR

        # 各资产夏普
        for i in range(self.n_assets):
            if len(self.returns_history[i]) >= 30:
                a_returns = np.array(list(self.returns_history[i]))
                a_mean = float(np.mean(a_returns))
                a_std = float(np.std(a_returns, ddof=1))
                if a_std > 1e-8:
                    self.asset_states[i].sharpe = (
                        (a_mean / a_std) * self.ANNUALIZATION_FACTOR)

    def _detect_vol_regime(self) -> None:
        """检测波动率状态"""
        if self.portfolio_volatility < self.TARGET_ANNUAL_VOL * self.LOW_VOL_RATIO:
            self.vol_regime = VolatilityRegime.LOW_VOL
        elif self.portfolio_volatility > self.TARGET_ANNUAL_VOL * self.EXTREME_VOL_RATIO:
            self.vol_regime = VolatilityRegime.EXTREME_VOL
        elif self.portfolio_volatility > self.TARGET_ANNUAL_VOL * self.HIGH_VOL_RATIO:
            self.vol_regime = VolatilityRegime.HIGH_VOL
        else:
            self.vol_regime = VolatilityRegime.NORMAL_VOL

    # ------------------------------------------------------------------
    # 权重管理
    # ------------------------------------------------------------------

    def set_current_weights(self, weights: List[float]) -> None:
        """设置当前各资产权重"""
        if len(weights) != self.n_assets:
            return
        for i, w in enumerate(weights):
            self.asset_states[i].current_weight = float(w)

    def calculate_target_weights(self) -> List[float]:
        """
        计算风险平价目标权重

        Risk Parity: 各资产风险贡献相等
        """
        if self.n_assets == 1:
            return [1.0]

        # 简化版风险平价: 权重反比于波动率
        vols = np.array([max(s.current_volatility, 1e-6) for s in self.asset_states])
        inv_vol = 1.0 / vols
        target_weights = inv_vol / inv_vol.sum()

        # Kelly 调整: 总杠杆 = Kelly 分数
        total_leverage = self.current_kelly_fraction
        if total_leverage < 0.05:
            total_leverage = 0.05  # 最低 5% 仓位

        # 波动率目标调整
        if self.portfolio_volatility > 1e-6:
            vol_scalar = self.TARGET_ANNUAL_VOL / self.portfolio_volatility
            vol_scalar = max(0.25, min(2.0, vol_scalar))  # 限制在 [0.25, 2.0]
        else:
            vol_scalar = 1.0

        # 回撤控制
        if self.current_drawdown > self.DRAWDOWN_WARNING:
            dd_scalar = 1.0 - (self.current_drawdown - self.DRAWDOWN_WARNING) / (
                self.MAX_DRAWDOWN_LIMIT - self.DRAWDOWN_WARNING + 1e-6)
            dd_scalar = max(0.0, min(1.0, dd_scalar))
        else:
            dd_scalar = 1.0

        # 综合调整
        final_weights = target_weights * total_leverage * vol_scalar * dd_scalar
        # 单资产上限
        final_weights = np.minimum(final_weights, 0.4)

        return final_weights.tolist()

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def _detect_vol_target_signal(self) -> Optional[RiskParityOpportunity]:
        """波动率目标信号"""
        if self.portfolio_volatility < 1e-6:
            return None

        target = self.TARGET_ANNUAL_VOL
        if self.vol_regime == VolatilityRegime.LOW_VOL:
            # 波动率低 → 加仓
            conviction = min(1.0, (target * self.LOW_VOL_RATIO -
                                   self.portfolio_volatility) / target)
            return RiskParityOpportunity(
                signal_type=RiskParitySignalType.VOLATILITY_TARGET_LONG,
                asset='portfolio',
                current_weight=1.0,
                target_weight=target / max(self.portfolio_volatility, 1e-6),
                conviction=conviction,
                metadata={'current_vol': self.portfolio_volatility, 'target_vol': target}
            )
        elif self.vol_regime in [VolatilityRegime.HIGH_VOL,
                                 VolatilityRegime.EXTREME_VOL]:
            # 波动率高 → 减仓
            conviction = min(1.0, (self.portfolio_volatility -
                                   target * self.HIGH_VOL_RATIO) / target)
            return RiskParityOpportunity(
                signal_type=RiskParitySignalType.VOLATILITY_TARGET_SHORT,
                asset='portfolio',
                current_weight=1.0,
                target_weight=target / max(self.portfolio_volatility, 1e-6),
                conviction=conviction,
                metadata={'current_vol': self.portfolio_volatility, 'target_vol': target}
            )
        return None

    def _detect_kelly_signal(self) -> Optional[RiskParityOpportunity]:
        """Kelly 仓位信号"""
        if abs(self.target_kelly_fraction - self.current_kelly_fraction) < 0.02:
            return None

        if self.target_kelly_fraction > self.current_kelly_fraction:
            signal_type = RiskParitySignalType.KELLY_INCREASE
            conviction = min(1.0, (self.target_kelly_fraction -
                                   self.current_kelly_fraction) * 5)
        else:
            signal_type = RiskParitySignalType.KELLY_DECREASE
            conviction = min(1.0, (self.current_kelly_fraction -
                                   self.target_kelly_fraction) * 5)

        return RiskParityOpportunity(
            signal_type=signal_type,
            asset='portfolio',
            current_weight=self.current_kelly_fraction,
            target_weight=self.target_kelly_fraction,
            conviction=conviction,
            metadata={'current_kelly': self.current_kelly_fraction,
                      'target_kelly': self.target_kelly_fraction}
        )

    def _detect_risk_parity_rebalance(self) -> Optional[RiskParityOpportunity]:
        """风险平价再平衡信号"""
        if self.n_assets < 2:
            return None

        # 检查风险贡献是否失衡
        target_rc = 1.0 / self.n_assets
        max_deviation = max(
            abs(s.risk_contribution - target_rc) for s in self.asset_states
        )

        if max_deviation < 0.10:  # 偏差 < 10% 不再平衡
            return None

        # 找到偏差最大的资产
        deviations = [(i, abs(s.risk_contribution - target_rc))
                      for i, s in enumerate(self.asset_states)]
        deviations.sort(key=lambda x: -x[1])
        worst_idx = deviations[0][0]
        worst_asset = self.asset_states[worst_idx]

        return RiskParityOpportunity(
            signal_type=RiskParitySignalType.RISK_PARITY_REBALANCE,
            asset=worst_asset.name,
            current_weight=worst_asset.current_weight,
            target_weight=worst_asset.target_weight,
            conviction=min(1.0, max_deviation * 5),
            metadata={'max_deviation': max_deviation, 'target_rc': target_rc}
        )

    def _detect_tail_risk_hedge(self) -> Optional[RiskParityOpportunity]:
        """尾部风险对冲信号"""
        # CVaR 显著为负 → 启动对冲
        if self.portfolio_cvar > -0.02:  # CVaR > -2% 不需要对冲
            return None

        conviction = min(1.0, abs(self.portfolio_cvar) * 10)
        return RiskParityOpportunity(
            signal_type=RiskParitySignalType.TAIL_RISK_HEDGE,
            asset='portfolio',
            current_weight=0.0,
            target_weight=-0.2,  # 建议对冲仓位 20%
            conviction=conviction,
            metadata={'cvar': self.portfolio_cvar}
        )

    def _detect_drawdown_control(self) -> Optional[RiskParityOpportunity]:
        """回撤控制信号"""
        if self.current_drawdown < self.DRAWDOWN_WARNING:
            return None

        conviction = min(1.0, self.current_drawdown / self.MAX_DRAWDOWN_LIMIT)
        # 强制减仓比例
        reduction = (self.current_drawdown - self.DRAWDOWN_WARNING) / max(
            self.MAX_DRAWDOWN_LIMIT - self.DRAWDOWN_WARNING, 1e-6)

        return RiskParityOpportunity(
            signal_type=RiskParitySignalType.DRAWDOWN_CONTROL,
            asset='portfolio',
            current_weight=1.0,
            target_weight=1.0 - reduction,
            conviction=conviction,
            metadata={'current_dd': self.current_drawdown,
                      'max_dd_limit': self.MAX_DRAWDOWN_LIMIT}
        )

    # ------------------------------------------------------------------
    # 综合分析
    # ------------------------------------------------------------------

    def analyze(self) -> RiskParitySignal:
        """综合分析并生成信号"""
        # 收集所有机会
        opportunities = []

        vol_opp = self._detect_vol_target_signal()
        if vol_opp:
            opportunities.append(vol_opp)

        kelly_opp = self._detect_kelly_signal()
        if kelly_opp:
            opportunities.append(kelly_opp)

        rp_opp = self._detect_risk_parity_rebalance()
        if rp_opp:
            opportunities.append(rp_opp)

        tail_opp = self._detect_tail_risk_hedge()
        if tail_opp:
            opportunities.append(tail_opp)
            self.tail_hedge_count += 1

        dd_opp = self._detect_drawdown_control()
        if dd_opp:
            opportunities.append(dd_opp)

        # 决定主信号 (优先级: 回撤控制 > 尾部对冲 > 波动率目标 > Kelly > 风险平价)
        priority_order = [
            RiskParitySignalType.DRAWDOWN_CONTROL,
            RiskParitySignalType.TAIL_RISK_HEDGE,
            RiskParitySignalType.VOLATILITY_TARGET_SHORT,
            RiskParitySignalType.VOLATILITY_TARGET_LONG,
            RiskParitySignalType.KELLY_DECREASE,
            RiskParitySignalType.KELLY_INCREASE,
            RiskParitySignalType.RISK_PARITY_REBALANCE,
        ]

        primary = None
        for sig_type in priority_order:
            for opp in opportunities:
                if opp.signal_type == sig_type:
                    primary = opp
                    break
            if primary:
                break

        if primary is None:
            signal_type = RiskParitySignalType.NEUTRAL
            direction = 0
            conviction = 0.0
            adjustment = 0.0
        else:
            signal_type = primary.signal_type
            conviction = primary.conviction
            # 方向
            if signal_type in [RiskParitySignalType.VOLATILITY_TARGET_LONG,
                               RiskParitySignalType.KELLY_INCREASE]:
                direction = 1
                adjustment = primary.target_weight - primary.current_weight
            elif signal_type in [RiskParitySignalType.VOLATILITY_TARGET_SHORT,
                                 RiskParitySignalType.KELLY_DECREASE,
                                 RiskParitySignalType.TAIL_RISK_HEDGE,
                                 RiskParitySignalType.DRAWDOWN_CONTROL]:
                direction = -1
                adjustment = primary.target_weight - primary.current_weight
            elif signal_type == RiskParitySignalType.RISK_PARITY_REBALANCE:
                direction = 1 if primary.target_weight > primary.current_weight else -1
                adjustment = primary.target_weight - primary.current_weight
            else:
                direction = 0
                adjustment = 0.0

            # 多信号一致 → 加强 conviction
            if len(opportunities) >= 2:
                consistent = sum(1 for o in opportunities
                                 if (o.conviction > 0.3 and
                                     ((direction > 0 and o.signal_type in
                                       [RiskParitySignalType.VOLATILITY_TARGET_LONG,
                                        RiskParitySignalType.KELLY_INCREASE])
                                      or (direction < 0 and o.signal_type in
                                          [RiskParitySignalType.VOLATILITY_TARGET_SHORT,
                                           RiskParitySignalType.KELLY_DECREASE,
                                           RiskParitySignalType.TAIL_RISK_HEDGE,
                                           RiskParitySignalType.DRAWDOWN_CONTROL]))))
                if consistent >= 2:
                    conviction = min(1.0, conviction * 1.2)

        # 限制权重调整幅度
        adjustment = max(-1.0, min(1.0, adjustment))

        signal = RiskParitySignal(
            signal_type=signal_type,
            direction=direction,
            conviction=conviction,
            target_weight_adjustment=adjustment,
            vol_regime=self.vol_regime,
            portfolio_volatility=self.portfolio_volatility,
            target_volatility=self.TARGET_ANNUAL_VOL,
            kelly_fraction=self.current_kelly_fraction,
            portfolio_cvar=self.portfolio_cvar,
            current_drawdown=self.current_drawdown,
            opportunities=opportunities,
        )

        if signal_type != RiskParitySignalType.NEUTRAL:
            self.signal_count += 1
            if signal_type == RiskParitySignalType.RISK_PARITY_REBALANCE:
                self.rebalance_count += 1

        return signal

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'n_assets': self.n_assets,
            'portfolio_value': self.portfolio_value,
            'portfolio_peak': self.portfolio_peak,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_seen': self.max_drawdown_seen,
            'portfolio_volatility': self.portfolio_volatility,
            'target_volatility': self.TARGET_ANNUAL_VOL,
            'vol_regime': self.vol_regime.value,
            'portfolio_sharpe': self.portfolio_sharpe,
            'portfolio_cvar': self.portfolio_cvar,
            'current_kelly': self.current_kelly_fraction,
            'target_kelly': self.target_kelly_fraction,
            'asset_states': [
                {
                    'name': s.name,
                    'weight': s.current_weight,
                    'volatility': s.current_volatility,
                    'risk_contribution': s.risk_contribution,
                    'sharpe': s.sharpe,
                    'cvar': s.cvar,
                    'kelly': s.kelly_fraction,
                }
                for s in self.asset_states
            ],
            'signal_count': self.signal_count,
            'rebalance_count': self.rebalance_count,
            'tail_hedge_count': self.tail_hedge_count,
        }

    def get_target_weights(self) -> List[float]:
        """获取目标权重 (供调用方使用)"""
        return self.calculate_target_weights()
