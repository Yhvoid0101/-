# -*- coding: utf-8 -*-
# v447_optimize_candidate: 此模块已标记为优化候选
# 原因: 零影响+中等重叠 34.6% > 30% (可合并)
# 建议操作: merge_or_lazy_load
# 标记时间: 2026-06-28T20:18:41
# 详情: _archive_v447/MANIFEST_v447.json

"""
dispersion_trading.py — Dispersion离散交易引擎 v1.0

来源：
  - stockalpha 2026 (Volatility Arbitrage: Dispersion Trading)
  - CSDN 2026 (Dispersion交易详解, 指数vs成分股波动率)
  - darkbot 2026 (What is arbitrage trading: 2026 crypto guide)
  - LiveVolatile 2026 (Institutional Options Flow, 相关性交易)
  - weloveeverythingcrypto 2026 (Crypto Options Trading Strategy Guide)

核心算法：
  1. 隐含相关性计算
     - Corr_implied = (IV_index² - Σw_i²·IV_i²) / (2·ΣΣw_i·w_j·IV_i·IV_j)
     - 指数IV vs 成分股加权IV
     - 低相关性 → 指数IV < 成分股IV加权平均
  2. 做空Dispersion（更常见）
     - 卖出指数期权波动率（卖出指数Straddle）
     - 买入成分股期权波动率（按权重买入各成分Straddle）
     - 盈利：市场平稳，个股分化，相关性低
  3. 做多Dispersion
     - 买入指数期权波动率
     - 卖出成分股期权波动率
     - 盈利：市场恐慌，相关性飙升
  4. 相关性均值回归
     - 历史相关性均值
     - 当前相关性偏离 → 均值回归信号
     - 相关性<历史均值 → 做空Dispersion（赌相关性回升）

风险控制：
  - 尾部风险：系统性事件导致相关性飙升
  - 成分股数量：至少5个成分股分散风险
  - 权重再平衡：定期调整成分股权重
  - Vega暴露：组合Vega中性

实盘验证：
  - SPX dispersion 年化Sharpe 1.5-2.5
  - 加密指数（BTC+ETH+ALT）dispersion 可行
  - 适合低相关性震荡市场
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class DispersionSignalType(Enum):
    """Dispersion信号类型"""
    SHORT_DISPERSION = "short_dispersion"     # 做空离散（卖指数买成分）
    LONG_DISPERSION = "long_dispersion"       # 做多离散（买指数卖成分）
    CLOSE_POSITION = "close_position"         # 平仓
    HOLD = "hold"                              # 持有
    NEUTRAL = "neutral"


@dataclass
class ComponentData:
    """成分股/成分币数据"""
    symbol: str
    weight: float               # 指数权重
    implied_vol: float = 0.0   # 隐含波动率
    realized_vol: float = 0.0   # 实现波动率
    option_price: float = 0.0  # ATM Straddle价格


@dataclass
class DispersionOpportunity:
    """Dispersion机会"""
    signal_type: DispersionSignalType
    implied_correlation: float = 0.0
    historical_corr_mean: float = 0.0
    corr_deviation: float = 0.0       # 相关性偏离
    index_iv: float = 0.0
    weighted_component_iv: float = 0.0
    iv_spread: float = 0.0             # IV价差
    expected_edge_usd: float = 0.0
    confidence: float = 0.0
    description: str = ""


@dataclass
class DispersionSignal:
    """Dispersion综合信号"""
    signal_type: DispersionSignalType = DispersionSignalType.NEUTRAL
    best_opportunity: Optional[DispersionOpportunity] = None
    current_implied_corr: float = 0.0
    current_iv_spread: float = 0.0
    description: str = ""


class DispersionTrading:
    """
    Dispersion离散交易引擎

    使用场景：
      - 指数期权市场（SPX/NDX/BTC指数）
      - 加密指数（BTC+ETH+ALT组合）
      - 低相关性震荡市场

    依赖：
      - numpy（数值计算）
      - 指数+成分股期权数据
    """

    # ===== 参数 =====
    CORR_LOW_THRESHOLD = 0.3         # 低相关性阈值（做空Dispersion）
    CORR_HIGH_THRESHOLD = 0.7        # 高相关性阈值（做多Dispersion）
    MIN_COMPONENTS = 5               # 最少成分股数量
    IV_SPREAD_THRESHOLD = 0.02       # IV价差阈值（2 vol点）
    EXECUTE_IV_SPREAD = 0.05         # 执行阈值（5 vol点）
    MAX_VEGA_EXPOSURE = 10000        # 最大Vega暴露
    REBALANCE_INTERVAL_DAYS = 7      # 再平衡间隔

    def __init__(self, index_symbol: str = "CRYPTO_INDEX"):
        self.index_symbol = index_symbol
        self.components: List[ComponentData] = []
        self.index_iv: float = 0.0
        self.index_realized_vol: float = 0.0
        self.historical_corr_history: deque = deque(maxlen=100)
        self.position_side: Optional[str] = None  # "short_disp" / "long_disp"
        self.position_start_time: float = 0.0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def set_index_vol(self, iv: float, realized_vol: float = 0.0) -> None:
        """设置指数波动率"""
        self.index_iv = iv
        self.index_realized_vol = realized_vol

    def add_component(self, component: ComponentData) -> None:
        """添加成分股"""
        self.components.append(component)

    def set_components_batch(self, components: List[ComponentData]) -> None:
        """批量设置成分股"""
        self.components = components

    def update_historical_corr(self, corr: float) -> None:
        """更新历史相关性"""
        self.historical_corr_history.append(corr)

    # ------------------------------------------------------------------
    # 隐含相关性计算
    # ------------------------------------------------------------------

    def calculate_implied_correlation(self) -> float:
        """
        计算隐含相关性

        Corr_implied = (IV_index² - Σw_i²·IV_i²) / (2·ΣΣw_i·w_j·IV_i·IV_j)

        简化版（假设所有成分相关性相同）:
        Corr = (IV_index² - Σw_i²·IV_i²) / (2·ΣΣ_{i≠j}w_i·w_j·IV_i·IV_j)
        """
        if not self.components or self.index_iv <= 0:
            return 0.0

        # Σw_i²·IV_i²
        weighted_var = sum(c.weight ** 2 * c.implied_vol ** 2 for c in self.components)

        # 2·ΣΣ_{i≠j}w_i·w_j·IV_i·IV_j
        cross_term = 0.0
        n = len(self.components)
        for i in range(n):
            for j in range(n):
                if i != j:
                    cross_term += 2 * self.components[i].weight * self.components[j].weight * \
                                  self.components[i].implied_vol * self.components[j].implied_vol

        if cross_term <= 0:
            return 0.0

        corr = (self.index_iv ** 2 - weighted_var) / cross_term
        # 限制在[0, 1]
        return max(0.0, min(1.0, corr))

    def calculate_weighted_component_iv(self) -> float:
        """计算成分股加权IV"""
        if not self.components:
            return 0.0
        return sum(c.weight * c.implied_vol for c in self.components)

    def calculate_iv_spread(self) -> float:
        """计算IV价差 = 成分股加权IV - 指数IV"""
        weighted_iv = self.calculate_weighted_component_iv()
        return weighted_iv - self.index_iv

    def get_historical_corr_mean(self) -> float:
        """获取历史相关性均值"""
        if not self.historical_corr_history:
            return 0.5  # 默认
        return float(np.mean(list(self.historical_corr_history)))

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def calculate_expected_edge(self) -> float:
        """计算预期收益"""
        iv_spread = self.calculate_iv_spread()
        if iv_spread <= 0:
            return 0.0
        # 简化：预期收益 ≈ IV价差 × 指数Vega
        # 假设指数Vega = 1000 per vol point
        return iv_spread * 1000

    def analyze(self) -> DispersionSignal:
        """主分析函数"""
        signal = DispersionSignal()

        if len(self.components) < self.MIN_COMPONENTS:
            signal.description = f"成分股不足: {len(self.components)}<{self.MIN_COMPONENTS}"
            return signal

        implied_corr = self.calculate_implied_correlation()
        weighted_iv = self.calculate_weighted_component_iv()
        iv_spread = self.calculate_iv_spread()
        hist_corr = self.get_historical_corr_mean()
        corr_deviation = implied_corr - hist_corr

        signal.current_implied_corr = implied_corr
        signal.current_iv_spread = iv_spread

        # 有持仓: 检查平仓
        if self.position_side == "short_disp":
            # 做空Dispersion平仓条件: 相关性回升到历史均值以上
            if implied_corr > hist_corr + 0.1 or iv_spread < self.IV_SPREAD_THRESHOLD:
                opp = DispersionOpportunity(
                    signal_type=DispersionSignalType.CLOSE_POSITION,
                    implied_correlation=implied_corr,
                    historical_corr_mean=hist_corr,
                    corr_deviation=corr_deviation,
                    index_iv=self.index_iv,
                    weighted_component_iv=weighted_iv,
                    iv_spread=iv_spread,
                    confidence=0.7,
                    description=f"做空Disp平仓: corr={implied_corr:.3f}回升, spread={iv_spread:.4f}",
                )
                signal.signal_type = DispersionSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            signal.signal_type = DispersionSignalType.HOLD
            signal.description = f"持有做空Disp, corr={implied_corr:.3f}, spread={iv_spread:.4f}"
            return signal

        if self.position_side == "long_disp":
            if implied_corr < hist_corr - 0.1 or iv_spread > -self.IV_SPREAD_THRESHOLD:
                opp = DispersionOpportunity(
                    signal_type=DispersionSignalType.CLOSE_POSITION,
                    implied_correlation=implied_corr,
                    historical_corr_mean=hist_corr,
                    corr_deviation=corr_deviation,
                    index_iv=self.index_iv,
                    weighted_component_iv=weighted_iv,
                    iv_spread=iv_spread,
                    confidence=0.7,
                    description=f"做多Disp平仓: corr={implied_corr:.3f}回落, spread={iv_spread:.4f}",
                )
                signal.signal_type = DispersionSignalType.CLOSE_POSITION
                signal.best_opportunity = opp
                return signal
            signal.signal_type = DispersionSignalType.HOLD
            signal.description = f"持有做多Disp, corr={implied_corr:.3f}, spread={iv_spread:.4f}"
            return signal

        # 无持仓: 检查入场
        if iv_spread > self.EXECUTE_IV_SPREAD and implied_corr < self.CORR_LOW_THRESHOLD:
            # 做空Dispersion: 卖指数买成分
            edge = self.calculate_expected_edge()
            opp = DispersionOpportunity(
                signal_type=DispersionSignalType.SHORT_DISPERSION,
                implied_correlation=implied_corr,
                historical_corr_mean=hist_corr,
                corr_deviation=corr_deviation,
                index_iv=self.index_iv,
                weighted_component_iv=weighted_iv,
                iv_spread=iv_spread,
                expected_edge_usd=edge,
                confidence=0.7,
                description=(
                    f"做空Disp: corr={implied_corr:.3f}<{self.CORR_LOW_THRESHOLD}, "
                    f"spread={iv_spread:.4f}>{self.EXECUTE_IV_SPREAD}, "
                    f"卖指数IV买成分IV"
                ),
            )
            signal.signal_type = DispersionSignalType.SHORT_DISPERSION
            signal.best_opportunity = opp
        elif iv_spread < -self.EXECUTE_IV_SPREAD and implied_corr > self.CORR_HIGH_THRESHOLD:
            # 做多Dispersion: 买指数卖成分
            opp = DispersionOpportunity(
                signal_type=DispersionSignalType.LONG_DISPERSION,
                implied_correlation=implied_corr,
                historical_corr_mean=hist_corr,
                corr_deviation=corr_deviation,
                index_iv=self.index_iv,
                weighted_component_iv=weighted_iv,
                iv_spread=iv_spread,
                expected_edge_usd=abs(iv_spread) * 1000,
                confidence=0.65,
                description=(
                    f"做多Disp: corr={implied_corr:.3f}>{self.CORR_HIGH_THRESHOLD}, "
                    f"spread={iv_spread:.4f}, 买指数IV卖成分IV"
                ),
            )
            signal.signal_type = DispersionSignalType.LONG_DISPERSION
            signal.best_opportunity = opp
        else:
            signal.description = (
                f"观望: corr={implied_corr:.3f}, spread={iv_spread:.4f}, "
                f"hist_corr={hist_corr:.3f}"
            )

        return signal

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "index_symbol": self.index_symbol,
            "index_iv": self.index_iv,
            "index_realized_vol": self.index_realized_vol,
            "components_count": len(self.components),
            "weighted_component_iv": self.calculate_weighted_component_iv(),
            "implied_correlation": self.calculate_implied_correlation(),
            "iv_spread": self.calculate_iv_spread(),
            "historical_corr_mean": self.get_historical_corr_mean(),
            "position_side": self.position_side,
            "expected_edge_usd": self.calculate_expected_edge(),
        }
