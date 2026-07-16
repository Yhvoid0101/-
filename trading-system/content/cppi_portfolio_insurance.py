# -*- coding: utf-8 -*-
"""
cppi_portfolio_insurance.py — CPPI组合保险策略 v1.0

Phase 8.6: 从抽象世界观升级为具体策略实现

算法 (Black-Perold 1987 + TIPP):
  1. CPPI: Exposure = m × max(V - F, 0)
     V=组合价值, F=底线, m=乘数
  2. TIPP: 底线上移 F = max(F, V × lock_ratio)
  3. Gap风险检测: 跳空>阈值时降低乘数
  4. 动态乘数: 高波动降乘数, 低波动升乘数
  5. 压力测试: 模拟-10%跳空检查底线是否击穿

信号类型：
  - CPPI_LONG: 有cushion → 按比例做多
  - CPPI_REDUCE: gap风险高 → 降低仓位
  - CPPI_PROTECT: cushion接近0 → 全部转入安全资产
  - CPPI_FLAT: 无cushion
"""

from __future__ import annotations
import math
from collections import deque
from typing import Any, Dict, Optional

import numpy as np


class CPPIPortfolioInsuranceStrategy:
    """CPPI+TIPP组合保险策略

    通过动态调整风险资产(做多)和安全资产(现金)比例来保底。
    """

    def __init__(
        self,
        initial_floor_ratio: float = 0.85,
        floor_min_ratio: float = 0.70,
        floor_max_ratio: float = 0.95,
        multiplier_base: float = 3.0,
        multiplier_min: float = 1.0,
        multiplier_max: float = 5.0,
        target_annual_vol: float = 0.30,
        rebalance_threshold: float = 0.05,
        tipp_floor_lock_ratio: float = 0.90,
        gap_risk_lambda: float = 2.0,
        gap_risk_jump_size: float = -0.10,
        gap_risk_threshold: float = 0.05,
        gap_risk_alarm_threshold: float = 0.15,
        floor_breach_warning_distance: float = 2.0,
        max_leverage: float = 1.5,
        min_cash_buffer: float = 0.05,
        **kwargs: Any,
    ):
        self.floor_ratio = initial_floor_ratio
        self.floor_min_ratio = floor_min_ratio
        self.floor_max_ratio = floor_max_ratio
        self.multiplier_base = multiplier_base
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        self.target_annual_vol = target_annual_vol
        self.rebalance_threshold = rebalance_threshold
        self.tipp_floor_lock_ratio = tipp_floor_lock_ratio
        self.gap_risk_lambda = gap_risk_lambda
        self.gap_risk_jump_size = gap_risk_jump_size
        self.gap_risk_threshold = gap_risk_threshold
        self.gap_risk_alarm_threshold = gap_risk_alarm_threshold
        self.floor_breach_warning_distance = floor_breach_warning_distance
        self.max_leverage = max_leverage
        self.min_cash_buffer = min_cash_buffer

        self._portfolio_value: float = 1.0  # normalized
        self._peak_value: float = 1.0
        self._floor: float = initial_floor_ratio
        self._current_weights: Dict[str, float] = {}
        self._prices: deque = deque(maxlen=100)
        self._prev_value: float = 1.0
        self._gap_risk_score: float = 0.0

    def update_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收价格 (用于波动率估计)"""
        self._prices.append(price)
        self._update_gap_risk()

    def update_portfolio_value(self, value: float) -> None:
        """Feeder: 接收组合价值"""
        self._prev_value = self._portfolio_value
        self._portfolio_value = value
        if value > self._peak_value:
            self._peak_value = value

        # TIPP: 底线上移
        tipp_floor = value * self.tipp_floor_lock_ratio
        self._floor = max(self._floor, tipp_floor)

        # 确保底线在合理范围
        self._floor = max(
            value * self.floor_min_ratio,
            min(value * self.floor_max_ratio, self._floor)
        )

    def set_current_weights(self, weights: Dict[str, float]) -> None:
        """Feeder: 设置当前持仓权重"""
        self._current_weights = dict(weights)

    def _compute_volatility(self) -> float:
        """计算近期波动率"""
        if len(self._prices) < 10:
            return self.target_annual_vol
        rets = np.diff(list(self._prices)[-20:])
        daily_vol = float(np.std(rets) / np.mean(np.abs(rets))) if np.mean(np.abs(rets)) > 0 else 0.02
        return daily_vol * math.sqrt(252)

    def _update_gap_risk(self) -> None:
        """更新gap风险评分"""
        if len(self._prices) < 5:
            self._gap_risk_score = 0.0
            return
        rets = np.diff(list(self._prices)[-5:])
        max_drop = float(min(rets)) if len(rets) > 0 else 0.0
        # gap风险 = 跳空幅度 / 阈值
        if abs(max_drop) > self.gap_risk_threshold:
            self._gap_risk_score = min(1.0, abs(max_drop) / self.gap_risk_alarm_threshold)
        else:
            self._gap_risk_score *= 0.9  # 衰减

    def _get_dynamic_multiplier(self) -> float:
        """动态乘数: 高波动降乘数"""
        vol = self._compute_volatility()
        if vol <= 0:
            return self.multiplier_base
        # 波动率调整因子
        vol_ratio = self.target_annual_vol / max(vol, 0.01)
        adjusted = self.multiplier_base * vol_ratio
        return max(self.multiplier_min, min(self.multiplier_max, adjusted))

    def analyze(self) -> Dict[str, Any]:
        """主分析函数"""
        result: Dict[str, Any] = {
            "signal_type": "CPPI_FLAT",
            "direction": "flat",
            "confidence": 0.0,
            "strength": 0.0,
            "description": "",
            "target_exposure": 0.0,
            "floor": self._floor,
            "cushion": 0.0,
        }

        V = self._portfolio_value
        F = self._floor
        cushion = max(V - F, 0.0)
        result["cushion"] = cushion

        # cushion为0 → 保护模式
        if cushion <= 0:
            result.update({
                "signal_type": "CPPI_PROTECT",
                "direction": "flat",
                "confidence": 1.0,
                "strength": 1.0,
                "description": f"底线击穿! V={V:.4f} F={F:.4f}, 全部转入安全资产",
                "target_exposure": 0.0,
            })
            return result

        # gap风险高 → 降低风险敞口
        if self._gap_risk_score > 0.5:
            m = self._get_dynamic_multiplier() * (1.0 - self._gap_risk_score * 0.5)
            result.update({
                "signal_type": "CPPI_REDUCE",
                "direction": "long",
                "confidence": 0.6,
                "strength": 0.5,
                "description": f"Gap风险={self._gap_risk_score:.2f}, 降低乘数至{m:.2f}",
                "target_exposure": min(self.max_leverage, m * cushion / V),
            })
            return result

        # 底线接近警告
        cushion_ratio = cushion / V if V > 0 else 0
        if cushion_ratio < 1.0 / self.floor_breach_warning_distance:
            result.update({
                "signal_type": "CPPI_PROTECT",
                "direction": "long",
                "confidence": 0.4,
                "strength": 0.3,
                "description": f"底线接近警告 cushion/V={cushion_ratio:.2%}",
                "target_exposure": min(self.max_leverage, cushion / V * 0.5),
            })
            return result

        # 正常CPPI: 计算风险敞口
        m = self._get_dynamic_multiplier()
        target_exposure = m * cushion / V
        target_exposure = min(self.max_leverage, max(0.0, target_exposure))

        # 保留最低现金缓冲
        target_exposure = min(target_exposure, 1.0 - self.min_cash_buffer)

        result.update({
            "signal_type": "CPPI_LONG",
            "direction": "long",
            "confidence": min(1.0, target_exposure),
            "strength": target_exposure,
            "description": f"CPPI做多 m={m:.2f} cushion={cushion:.4f} exposure={target_exposure:.2%}",
            "target_exposure": target_exposure,
        })

        return result

    def stress_test(self) -> Dict[str, Any]:
        """压力测试: 模拟gap_risk_jump_size跳空"""
        V = self._portfolio_value
        F = self._floor
        cushion = max(V - F, 0.0)
        m = self._get_dynamic_multiplier()
        exposure = m * cushion / V
        # 模拟跳空
        jump = self.gap_risk_jump_size
        new_V = V * (1 + jump * exposure)
        new_cushion = max(new_V - F, 0.0)
        breach = new_cushion <= 0
        return {
            "breach": breach,
            "pre_value": V,
            "post_value": new_V,
            "pre_cushion": cushion,
            "post_cushion": new_cushion,
            "exposure": exposure,
            "multiplier": m,
        }
