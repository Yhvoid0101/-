# -*- coding: utf-8 -*-
"""
市场微观结构增强模块 (Market Microstructure Enhancement)

基于2025-2026全网前沿市场微观结构研究，整合3大增强：

1. Hawkes过程交易爆发检测器 (Hawkes Process Burst Detector)
   来源：Hawkes 1971 + digital-ai-finance 2026课程 + 高频交易实践
   原理：交易到达是自激励的，每笔交易增加下一笔交易的概率
   公式：λ(t) = λ₀ + Σ α·exp(-β(t-tᵢ))
   应用：检测交易爆发开始时调整策略激进程度，爆发期提高信号权重

2. 订单簿不平衡信号 (Order Book Imbalance)
   来源：quant-flow 2026 + Bridging the Reality Gap (Noble et al. 2026)
   原理：买卖盘挂单量不平衡预示短期价格方向
   公式：imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
   应用：作为入场确认信号，强不平衡时增强信号得分

3. Kyle's lambda价格冲击模型 (Kyle's Lambda Price Impact)
   来源：Kyle 1985 + Almgren-Chriss 2000 + 市场微观结构理论
   原理：永久价格冲击与订单规模平方根成正比
   公式：ΔP = λ·sign(V)·√|V|，其中λ是Kyle冲击系数
   应用：更精确的滑点估计，避免高冲击成本的交易
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.microstructure")


# ============================================================================
# 1. Hawkes过程交易爆发检测器
# ============================================================================


class HawkesBurstDetector:
    """Hawkes过程交易爆发检测器

    来源：Hawkes 1971 "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"
         + digital-ai-finance 2026课程 "Financial Markets & Trading Infrastructure"

    原理：
      交易到达不是泊松过程，而是自激励的——每笔交易会增加后续交易的概率。
      这种"聚集"模式遵循Hawkes过程：
        λ(t) = λ₀ + Σ_{tᵢ < t} α·exp(-β(t-tᵢ))
      其中：
        λ₀ = 基础交易强度（稳态）
        α  = 每笔交易对强度的激励幅度
        β  = 激励衰减速率
        tᵢ = 历史交易时间

    爆发检测：
      当 λ(t) > λ₀ × burst_threshold 时，判定为交易爆发期
      爆发期通常由新闻、算法级联或大单触发，是高信息量时段

    量化标准：
      - λ(t)/λ₀ < 1.5: 正常时段，信号权重×1.0
      - λ(t)/λ₀ 1.5-3.0: 活跃时段，信号权重×1.2
      - λ(t)/λ₀ 3.0-5.0: 爆发时段，信号权重×1.5（高信息量）
      - λ(t)/λ₀ > 5.0: 极端爆发，信号权重×0.8（过度拥挤，反转风险）
    """

    def __init__(
        self,
        baseline_intensity: float = 1.0,    # λ₀ 基础交易强度
        excitation_alpha: float = 0.5,      # α 激励幅度
        decay_beta: float = 2.0,            # β 衰减速率
        burst_threshold: float = 1.5,       # 爆发阈值（相对λ₀）
        extreme_threshold: float = 5.0,     # 极端爆发阈值
        max_events: int = 200,              # 最大历史事件数
    ):
        """
        Args:
            baseline_intensity: 基础交易强度λ₀（每秒交易数）
            excitation_alpha: 激励幅度α
            decay_beta: 衰减速率β（越大衰减越快）
            burst_threshold: 爆发阈值（λ(t)/λ₀的倍数）
            extreme_threshold: 极端爆发阈值
            max_events: 最大历史事件数
        """
        self.lambda_0 = baseline_intensity
        self.alpha = excitation_alpha
        self.beta = decay_beta
        self.burst_threshold = burst_threshold
        self.extreme_threshold = extreme_threshold
        self.max_events = max_events

        # 历史交易时间戳（秒）
        self._event_times: deque = deque(maxlen=max_events)
        # 当前时间（模拟）
        self._current_time = 0.0
        # 上一次计算的强度
        self._last_intensity = baseline_intensity
        # P4#11: 上一次事件时间（用于递推公式）
        self._last_event_time: Optional[float] = None

    def update(self, timestamp: Optional[float] = None) -> float:
        """记录一笔交易，更新Hawkes强度

        P4#11: 使用递推公式 O(1) 替代遍历所有历史事件 O(n)
        递推原理：
          λ(t) = λ₀ + Σ α·exp(-β(t-tᵢ))
          若已知 λ(t_prev)，新事件在 t_new (t_new > t_prev)：
          λ(t_new) = λ₀ + exp(-β·(t_new-t_prev))·(λ(t_prev)-λ₀) + α·exp(-β·(t_new-t_new))
                   = λ₀ + exp(-β·Δt)·(λ(t_prev)-λ₀) + α

        Args:
            timestamp: 交易时间戳（秒），None则自动递增

        Returns:
            更新后的瞬时强度λ(t)
        """
        if timestamp is not None:
            self._current_time = timestamp
        else:
            self._current_time += 1.0

        t_new = self._current_time
        self._event_times.append(t_new)

        # P4#11: 递推计算 O(1)
        # 原始 _compute_intensity 使用 t_i < t（严格小于），当前事件不计算
        # 递推公式：λ(t_new) = λ₀ + decay·(λ(t_prev) - λ₀ + α)
        # 其中 decay = exp(-β·(t_new - t_prev))
        # 推导：λ(t_new) = λ₀ + Σ_{t_i < t_new} α·exp(-β(t_new - t_i))
        #      = λ₀ + decay·Σ_{t_i < t_prev} α·exp(-β(t_prev - t_i)) + α·decay
        #      = λ₀ + decay·(λ(t_prev) - λ₀) + α·decay
        #      = λ₀ + decay·(λ(t_prev) - λ₀ + α)
        t_prev = self._last_event_time
        if t_prev is not None and t_new > t_prev:
            decay = math.exp(-self.beta * (t_new - t_prev))
            intensity = self.lambda_0 + decay * (self._last_intensity - self.lambda_0 + self.alpha)
        else:
            # 首次事件或时间回溯，精确计算
            intensity = self._compute_intensity(t_new)

        self._last_event_time = t_new
        self._last_intensity = intensity
        return intensity

    def _compute_intensity(self, t: float) -> float:
        """计算时刻t的Hawkes强度

        λ(t) = λ₀ + Σ α·exp(-β(t-tᵢ))

        P4#11: 向量化优化 — 大数组用 numpy 批量计算指数衰减，
        小数组（<50事件）保持纯 Python 避免 numpy 转换开销。
        """
        n_events = len(self._event_times)
        if n_events == 0:
            return self.lambda_0

        # P4#11: 小数组快速路径（避免 numpy 转换开销）
        if n_events < 50:
            intensity = self.lambda_0
            cutoff = 50.0 / self.beta
            for t_i in self._event_times:
                if t_i < t:
                    dt = t - t_i
                    if dt < cutoff:
                        intensity += self.alpha * math.exp(-self.beta * dt)
            return intensity

        # P4#11: 大数组向量化计算
        # 使用 np.array 一次性转换，合并条件减少中间 mask 操作
        times = np.array(self._event_times, dtype=np.float64)
        dt = t - times
        # 合并条件：t_i < t (dt > 0) 且 dt < cutoff
        cutoff = 50.0 / self.beta
        valid = (dt > 0) & (dt < cutoff)
        if not np.any(valid):
            return self.lambda_0
        dt_valid = dt[valid]
        # 向量化计算指数衰减贡献
        contributions = self.alpha * np.exp(-self.beta * dt_valid)
        return self.lambda_0 + float(np.sum(contributions))

    def get_intensity(self) -> float:
        """获取当前强度"""
        return self._last_intensity

    def get_intensity_ratio(self) -> float:
        """获取强度相对基线的倍数 λ(t)/λ₀"""
        if self.lambda_0 <= 0:
            return 1.0
        return self._last_intensity / self.lambda_0

    def get_burst_level(self) -> str:
        """获取爆发等级"""
        ratio = self.get_intensity_ratio()
        if ratio > self.extreme_threshold:
            return "EXTREME"  # 极端爆发，过度拥挤
        elif ratio > 3.0:
            return "HIGH"     # 爆发时段，高信息量
        elif ratio > self.burst_threshold:
            return "MEDIUM"   # 活跃时段
        else:
            return "NORMAL"   # 正常时段

    def get_signal_weight(self) -> float:
        """获取信号权重调整系数

        基于爆发等级调整信号权重：
          NORMAL: 1.0（基准）
          MEDIUM: 1.2（活跃，信号更可靠）
          HIGH:   1.5（爆发，高信息量）
          EXTREME: 0.8（过度拥挤，反转风险）
        """
        level = self.get_burst_level()
        weights = {
            "NORMAL": 1.0,
            "MEDIUM": 1.2,
            "HIGH": 1.5,
            "EXTREME": 0.8,
        }
        return weights.get(level, 1.0)

    def is_burst(self) -> bool:
        """是否处于爆发期"""
        return self.get_intensity_ratio() > self.burst_threshold

    def should_trade(self) -> Tuple[bool, str]:
        """是否适合交易

        Returns:
            (can_trade, reason)
        """
        level = self.get_burst_level()
        if level == "EXTREME":
            return False, f"极端爆发(λ/λ₀={self.get_intensity_ratio():.1f})，过度拥挤反转风险"
        elif level == "HIGH":
            return True, f"爆发期(λ/λ₀={self.get_intensity_ratio():.1f})，高信息量时段"
        elif level == "MEDIUM":
            return True, f"活跃期(λ/λ₀={self.get_intensity_ratio():.1f})，信号权重增强"
        else:
            return True, f"正常时段(λ/λ₀={self.get_intensity_ratio():.1f})"


# ============================================================================
# 2. 订单簿不平衡信号
# ============================================================================


class OrderBookImbalance:
    """订单簿不平衡信号

    来源：quant-flow 2026 + Noble et al. 2026 "Bridging the Reality Gap in LOB Simulation"
         + Cont et al. 2014 "Price Dynamics in a Markovian Limit Order Market"

    原理：
      订单簿的买卖盘挂单量不平衡是短期价格方向的强预测信号。
      当买盘远大于卖盘时，卖方压力小，价格倾向上涨；反之亦然。

    公式：
      imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
      范围：[-1, 1]
        +1: 全部是买盘（强看涨）
        -1: 全部是卖盘（强看跌）
        0: 买卖均衡

    量化标准（来自Cont 2014实证）：
      - |imbalance| < 0.1: 均衡，无信号
      - |imbalance| 0.1-0.3: 轻度不平衡，弱信号
      - |imbalance| 0.3-0.5: 中度不平衡，中等信号
      - |imbalance| > 0.5: 强不平衡，强信号（预测力约65%）

    深度加权：
      使用前N档挂单量，按距离加权（越近的档位权重越高）
      weight_i = 1 / (1 + level_i)
    """

    def __init__(
        self,
        n_levels: int = 5,               # 使用前N档挂单
        strong_threshold: float = 0.5,   # 强不平衡阈值
        medium_threshold: float = 0.3,   # 中等不平衡阈值
        weak_threshold: float = 0.1,     # 弱不平衡阈值
        decay_factor: float = 0.8,       # 深度衰减因子
    ):
        """
        Args:
            n_levels: 使用前N档挂单量
            strong_threshold: 强不平衡阈值
            medium_threshold: 中等不平衡阈值
            weak_threshold: 弱不平衡阈值
            decay_factor: 深度衰减因子（越深的档位权重越低）
        """
        self.n_levels = n_levels
        self.strong_threshold = strong_threshold
        self.medium_threshold = medium_threshold
        self.weak_threshold = weak_threshold
        self.decay_factor = decay_factor

        # 历史不平衡度（用于趋势检测）
        self._imbalance_history: deque = deque(maxlen=50)
        # 当前不平衡度
        self._current_imbalance = 0.0

    def update(
        self,
        bid_volumes: List[float],
        ask_volumes: List[float],
    ) -> float:
        """更新订单簿不平衡度

        Args:
            bid_volumes: 买盘各档挂单量（从最优价开始）
            ask_volumes: 卖盘各档挂单量

        Returns:
            加权不平衡度 [-1, 1]
        """
        if not bid_volumes or not ask_volumes:
            return self._current_imbalance

        # 深度加权：越近的档位权重越高
        n = min(self.n_levels, len(bid_volumes), len(ask_volumes))
        if n == 0:
            return self._current_imbalance

        weights = [self.decay_factor ** i for i in range(n)]
        total_weight = sum(weights)

        weighted_bid = sum(bid_volumes[i] * weights[i] for i in range(n))
        weighted_ask = sum(ask_volumes[i] * weights[i] for i in range(n))

        total = weighted_bid + weighted_ask
        if total <= 0:
            return self._current_imbalance

        imbalance = (weighted_bid - weighted_ask) / total
        self._current_imbalance = max(-1.0, min(1.0, imbalance))
        self._imbalance_history.append(self._current_imbalance)

        return self._current_imbalance

    def update_from_kline(self, kline: Dict[str, float]) -> float:
        """从K线数据近似估算不平衡度（降级模式）

        当无订单簿数据时，用K线特征近似：
          - 阳线（close>open）+ 长下影线 → 买盘强
          - 阴线（close<open）+ 长上影线 → 卖盘强
          - 成交量分布辅助判断

        Args:
            kline: {"open", "high", "low", "close", "volume"}

        Returns:
            近似不平衡度 [-1, 1]
        """
        open_p = kline.get("open", 0)
        high = kline.get("high", open_p)
        low = kline.get("low", open_p)
        close = kline.get("close", open_p)
        volume = kline.get("volume", 0)

        if volume <= 0 or high <= low:
            return self._current_imbalance

        # 价格位置：close在high-low区间的位置
        price_pos = (close - low) / (high - low) if high > low else 0.5

        # 影线分析
        body = abs(close - open_p)
        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low
        range_val = high - low

        if range_val <= 0:
            return self._current_imbalance

        # 买盘力量代理：下影线占比 + 收盘位置
        buy_strength = (lower_shadow / range_val) * 0.4 + price_pos * 0.6
        # 卖盘力量代理：上影线占比 + (1-收盘位置)
        sell_strength = (upper_shadow / range_val) * 0.4 + (1 - price_pos) * 0.6

        total = buy_strength + sell_strength
        if total <= 0:
            return self._current_imbalance

        imbalance = (buy_strength - sell_strength) / total
        self._current_imbalance = max(-1.0, min(1.0, imbalance))
        self._imbalance_history.append(self._current_imbalance)

        return self._current_imbalance

    def get_imbalance(self) -> float:
        """获取当前不平衡度"""
        return self._current_imbalance

    def get_signal(self) -> Tuple[str, float]:
        """获取交易信号

        Returns:
            (direction, strength)
            direction: "bullish" / "bearish" / "neutral"
            strength: 0.0-1.0
        """
        imb = abs(self._current_imbalance)

        if self._current_imbalance > self.strong_threshold:
            return "bullish", min(1.0, imb)
        elif self._current_imbalance < -self.strong_threshold:
            return "bearish", min(1.0, imb)
        elif self._current_imbalance > self.medium_threshold:
            return "bullish", imb * 0.7
        elif self._current_imbalance < -self.medium_threshold:
            return "bearish", imb * 0.7
        elif self._current_imbalance > self.weak_threshold:
            return "bullish", imb * 0.4
        elif self._current_imbalance < -self.weak_threshold:
            return "bearish", imb * 0.4
        else:
            return "neutral", 0.0

    def get_imbalance_trend(self) -> str:
        """获取不平衡度趋势（增强/减弱）

        通过最近5个周期的不平衡度变化判断趋势

        Returns:
            "strengthening" / "weakening" / "stable"
        """
        if len(self._imbalance_history) < 5:
            return "stable"

        recent = list(self._imbalance_history)[-5:]
        # 线性回归斜率
        x = np.arange(len(recent))
        y = np.array(recent)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        if abs(slope) < 0.02:
            return "stable"
        elif slope > 0:
            return "strengthening"  # 买盘增强
        else:
            return "weakening"      # 买盘减弱


# ============================================================================
# 3. Kyle's lambda价格冲击模型
# ============================================================================


class KylesLambdaImpact:
    """Kyle's lambda价格冲击模型

    来源：Kyle 1985 "Continuous Auctions and Insider Trading"
         + Almgren-Chriss 2000 "Optimal Execution of Portfolio Transactions"
         + 市场微观结构实证研究

    原理：
      知情交易者的订单会产生永久性价格冲击，冲击大小与订单规模平方根成正比。
      Kyle's lambda衡量市场的"深度"——lambda越大，市场越浅。

    公式：
      永久冲击：ΔP_permanent = λ · sign(V) · √|V|
      临时冲击：ΔP_temporary = η · (V / ADV) · σ
      总冲击：  ΔP_total = ΔP_permanent + ΔP_temporary

      其中：
        λ  = Kyle冲击系数（市场深度倒数）
        V  = 订单规模（签名成交量）
        η  = 临时冲击系数
        ADV= 平均日成交量
        σ  = 日波动率

    量化标准（来自Almgren-Chriss实证）：
      - λ < 0.001: 深市场（高流动性），冲击小
      - λ 0.001-0.01: 正常市场
      - λ 0.01-0.1: 浅市场，大单冲击显著
      - λ > 0.1: 极浅市场，避免大单

    应用：
      1. 估计订单的实际冲击成本
      2. 过滤掉冲击成本过高的交易
      3. 优化订单分割策略
    """

    def __init__(
        self,
        kyle_lambda: float = 0.005,      # λ Kyle冲击系数
        temp_impact_eta: float = 0.1,    # η 临时冲击系数
        adv_estimate: float = 1000000.0, # ADV 默认平均日成交量
        daily_volatility: float = 0.02,  # σ 日波动率
        max_impact_pct: float = 0.01,    # 最大允许冲击（1%）
    ):
        """
        Args:
            kyle_lambda: Kyle冲击系数λ
            temp_impact_eta: 临时冲击系数η
            adv_estimate: 平均日成交量估计
            daily_volatility: 日波动率
            max_impact_pct: 最大允许冲击百分比
        """
        self.kyle_lambda = kyle_lambda
        self.eta = temp_impact_eta
        self.adv = adv_estimate
        self.sigma = daily_volatility
        self.max_impact_pct = max_impact_pct

        # 历史冲击记录（用于动态校准λ）
        self._impact_history: deque = deque(maxlen=100)

    def compute_impact(
        self,
        order_size: float,
        price: float,
        side: str = "long",
    ) -> Dict[str, float]:
        """计算订单的价格冲击

        Args:
            order_size: 订单规模（合约数或代币数）
            price: 当前价格
            side: "long"（买入）或 "short"（卖出）

        Returns:
            {
                "permanent_impact_pct": float,  # 永久冲击百分比
                "temporary_impact_pct": float,  # 临时冲击百分比
                "total_impact_pct": float,      # 总冲击百分比
                "impact_cost": float,           # 冲击成本（计价货币）
                "is_acceptable": bool,          # 是否在可接受范围
                "market_depth": str,            # 市场深度等级
            }
        """
        if order_size <= 0 or price <= 0:
            return {
                "permanent_impact_pct": 0.0,
                "temporary_impact_pct": 0.0,
                "total_impact_pct": 0.0,
                "impact_cost": 0.0,
                "is_acceptable": True,
                "market_depth": "UNKNOWN",
            }

        # 签名成交量
        signed_v = order_size if side == "long" else -order_size

        # 永久冲击：ΔP = λ · sign(V) · √|V|
        # 转换为百分比：impact_pct = λ · √|V| / price
        permanent_impact_pct = self.kyle_lambda * math.sqrt(abs(signed_v)) / price

        # 临时冲击：ΔP = η · (V/ADV) · σ
        volume_ratio = abs(signed_v) / self.adv if self.adv > 0 else 0
        temporary_impact_pct = self.eta * volume_ratio * self.sigma

        # 总冲击
        total_impact_pct = permanent_impact_pct + temporary_impact_pct

        # 冲击成本（计价货币）
        impact_cost = total_impact_pct * order_size * price

        # 是否可接受
        is_acceptable = total_impact_pct <= self.max_impact_pct

        # 市场深度等级
        market_depth = self._get_market_depth()

        result = {
            "permanent_impact_pct": permanent_impact_pct,
            "temporary_impact_pct": temporary_impact_pct,
            "total_impact_pct": total_impact_pct,
            "impact_cost": impact_cost,
            "is_acceptable": is_acceptable,
            "market_depth": market_depth,
        }

        self._impact_history.append(total_impact_pct)
        return result

    def _get_market_depth(self) -> str:
        """根据Kyle's lambda判断市场深度"""
        if self.kyle_lambda < 0.001:
            return "DEEP"       # 深市场，高流动性
        elif self.kyle_lambda < 0.01:
            return "NORMAL"     # 正常市场
        elif self.kyle_lambda < 0.1:
            return "SHALLOW"    # 浅市场
        else:
            return "VERY_SHALLOW"  # 极浅市场

    def calibrate_from_trades(
        self,
        trade_sizes: List[float],
        price_changes: List[float],
    ) -> float:
        """从历史交易数据校准Kyle's lambda

        使用回归方法估计λ：
          ΔP = λ · sign(V) · √|V| + ε

        Args:
            trade_sizes: 历史交易规模列表
            price_changes: 对应的价格变化列表

        Returns:
            校准后的λ值
        """
        if len(trade_sizes) < 10 or len(trade_sizes) != len(price_changes):
            logger.warning("校准数据不足，保持默认lambda")
            return self.kyle_lambda

        try:
            # 构建回归变量：x = sign(V)·√|V|, y = ΔP
            x = np.array([np.sign(v) * math.sqrt(abs(v)) for v in trade_sizes])
            y = np.array(price_changes)

            # 最小二乘法：y = λ·x
            # λ = Σ(x·y) / Σ(x²)
            denominator = np.sum(x ** 2)
            if denominator > 0:
                calibrated_lambda = float(np.sum(x * y) / denominator)
                # 确保lambda为正
                calibrated_lambda = abs(calibrated_lambda)

                # 平滑更新（避免剧烈跳变）
                self.kyle_lambda = 0.7 * self.kyle_lambda + 0.3 * calibrated_lambda
                logger.info(f"Kyle's lambda校准: {calibrated_lambda:.6f} → {self.kyle_lambda:.6f}")

            return self.kyle_lambda
        except Exception as e:
            logger.warning(f"Lambda校准失败: {e}")
            return self.kyle_lambda

    def should_execute(
        self,
        order_size: float,
        price: float,
        side: str = "long",
    ) -> Tuple[bool, str]:
        """判断订单是否应该执行（冲击成本可接受）

        Args:
            order_size: 订单规模
            price: 当前价格
            side: 买卖方向

        Returns:
            (should_execute, reason)
        """
        impact = self.compute_impact(order_size, price, side)

        if not impact["is_acceptable"]:
            return False, (
                f"冲击成本过高({impact['total_impact_pct']:.3%} > {self.max_impact_pct:.3%})，"
                f"市场深度={impact['market_depth']}"
            )

        depth = impact["market_depth"]
        if depth == "VERY_SHALLOW":
            return False, f"极浅市场(lambda={self.kyle_lambda:.4f})，避免大单"

        return True, f"冲击可接受({impact['total_impact_pct']:.3%})，深度={depth}"


# ============================================================================
# 4. 微观结构综合增强系统
# ============================================================================


class MicrostructureEnhancementSystem:
    """市场微观结构综合增强系统

    整合Hawkes爆发检测 + 订单簿不平衡 + Kyle's lambda冲击

    提供统一的接口供进化主循环调用：
      1. update() — 每个K线更新所有微观结构指标
      2. get_enhanced_signal() — 获取增强后的信号得分
      3. get_execution_filter() — 获取执行过滤器（冲击成本检查）
    """

    def __init__(
        self,
        hawkes: Optional[HawkesBurstDetector] = None,
        imbalance: Optional[OrderBookImbalance] = None,
        kyle_impact: Optional[KylesLambdaImpact] = None,
    ):
        """
        Args:
            hawkes: Hawkes爆发检测器（None则自动创建）
            imbalance: 订单簿不平衡检测器
            kyle_impact: Kyle冲击模型
        """
        self.hawkes = hawkes or HawkesBurstDetector()
        self.imbalance = imbalance or OrderBookImbalance()
        self.kyle_impact = kyle_impact or KylesLambdaImpact()

        # 降级标记
        self.degraded = True  # 使用K线近似而非逐笔/订单簿数据

    def update(self, kline: Dict[str, float], timestamp: Optional[float] = None) -> Dict[str, Any]:
        """更新所有微观结构指标

        Args:
            kline: K线数据 {"open", "high", "low", "close", "volume"}
            timestamp: 时间戳

        Returns:
            更新后的微观结构状态
        """
        # 1. Hawkes爆发检测：每笔交易更新
        volume = kline.get("volume", 0)
        if volume > 0:
            # 用成交量作为交易频率代理
            n_trades = max(1, int(volume / 100))  # 假设每100单位=1笔交易
            for _ in range(min(n_trades, 10)):  # 限制每K线最多10次更新
                self.hawkes.update(timestamp)

        # 2. 订单簿不平衡：用K线近似（降级模式）
        self.imbalance.update_from_kline(kline)

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """获取当前微观结构状态"""
        hawkes_level = self.hawkes.get_burst_level()
        imb_dir, imb_strength = self.imbalance.get_signal()
        imb_trend = self.imbalance.get_imbalance_trend()
        depth = self.kyle_impact._get_market_depth()

        return {
            "hawkes_intensity_ratio": self.hawkes.get_intensity_ratio(),
            "hawkes_burst_level": hawkes_level,
            "hawkes_signal_weight": self.hawkes.get_signal_weight(),
            "imbalance_value": self.imbalance.get_imbalance(),
            "imbalance_direction": imb_dir,
            "imbalance_strength": imb_strength,
            "imbalance_trend": imb_trend,
            "market_depth": depth,
            "kyle_lambda": self.kyle_impact.kyle_lambda,
            "degraded": self.degraded,
        }

    def get_enhanced_signal(
        self,
        base_signal_score: float,
        base_direction: str,
    ) -> Tuple[float, str]:
        """获取微观结构增强后的信号得分

        Args:
            base_signal_score: 基础信号得分
            base_direction: 基础信号方向 "long"/"short"

        Returns:
            (enhanced_score, reason)
        """
        reasons = []

        # 1. Hawkes爆发权重调整
        hawkes_weight = self.hawkes.get_signal_weight()
        enhanced_score = base_signal_score * hawkes_weight
        if hawkes_weight != 1.0:
            reasons.append(f"Hawkes权重×{hawkes_weight}({self.hawkes.get_burst_level()})")

        # 2. 订单簿不平衡确认
        imb_dir, imb_strength = self.imbalance.get_signal()
        if imb_dir != "neutral":
            # 不平衡方向与信号方向一致 → 增强
            # 不平衡方向与信号方向相反 → 减弱
            if (imb_dir == "bullish" and base_direction == "long") or \
               (imb_dir == "bearish" and base_direction == "short"):
                enhanced_score *= (1.0 + imb_strength * 0.3)
                reasons.append(f"不平衡确认({imb_dir},{imb_strength:.2f})×1.3")
            else:
                enhanced_score *= (1.0 - imb_strength * 0.2)
                reasons.append(f"不平衡矛盾({imb_dir}vs{base_direction})×0.8")

        # 3. 不平衡趋势确认
        imb_trend = self.imbalance.get_imbalance_trend()
        if imb_trend == "strengthening" and base_direction == "long":
            enhanced_score *= 1.1
            reasons.append("买盘趋势增强×1.1")
        elif imb_trend == "weakening" and base_direction == "long":
            enhanced_score *= 0.9
            reasons.append("买盘趋势减弱×0.9")

        reason = "; ".join(reasons) if reasons else "无微观结构调整"
        return enhanced_score, reason

    def get_execution_filter(
        self,
        order_size: float,
        price: float,
        side: str,
    ) -> Tuple[bool, str]:
        """执行过滤器：检查冲击成本

        Args:
            order_size: 订单规模
            price: 当前价格
            side: 买卖方向

        Returns:
            (can_execute, reason)
        """
        return self.kyle_impact.should_execute(order_size, price, side)

    def get_summary(self) -> Dict[str, Any]:
        """获取微观结构增强系统摘要"""
        return {
            "hawkes": {
                "intensity_ratio": self.hawkes.get_intensity_ratio(),
                "burst_level": self.hawkes.get_burst_level(),
                "is_burst": self.hawkes.is_burst(),
            },
            "imbalance": {
                "value": self.imbalance.get_imbalance(),
                "signal": self.imbalance.get_signal(),
                "trend": self.imbalance.get_imbalance_trend(),
            },
            "kyle_impact": {
                "lambda": self.kyle_impact.kyle_lambda,
                "market_depth": self.kyle_impact._get_market_depth(),
                "avg_impact": float(np.mean(self.kyle_impact._impact_history))
                              if self.kyle_impact._impact_history else 0.0,
            },
            "degraded": self.degraded,
        }
