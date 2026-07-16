#!/usr/bin/env python3
"""清算热力图计算器 — Layer 8 质变数据模块2

清算热力图（Liquidation Heatmap）显示不同价格水平的清算密度。
当价格接近清算密集区时，会发生"清算级联"——大量杠杆持仓被强制平仓，
创造巨大流动性，推动价格快速移动。

核心概念:
  - 多头清算密集区: 当前价格下方，大量多头杠杆被清算的价格水平
    → 价格下跌到该水平时会触发级联清算，创造卖压
  - 空头清算密集区: 当前价格上方，大量空头杠杆被清算的价格水平
    → 价格上涨到该水平时会触发级联清算，创造买压
  - 清算级联风险: 相邻价格水平清算量梯度大时，级联风险高

门控逻辑:
  - 做多 + 上方有空头清算密集区 → BOOST (空头清算推动价格上涨)
  - 做多 + 下方有多头清算密集区 → REDUCE (多头清算可能推动价格下跌)
  - 做空 + 下方有多头清算密集区 → BOOST (多头清算推动价格下跌)
  - 做空 + 上方有空头清算密集区 → REDUCE (空头清算可能推动价格上涨)
  - 清算级联风险高 → REDUCE (避免被级联波及)

用法:
    from sandbox_trading.liquidation_calculator import LiquidationCalculator
    calc = LiquidationCalculator()
    result = calc.calculate(liquidations, spot_price, symbol="BTC")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("hermes.liquidation")


# ============================================================================
# 清算热力图结果数据结构
# ============================================================================

@dataclass(slots=True)
class LiquidationResult:
    """清算热力图计算结果

    所有金额单位均为 USD
    """
    symbol: str = ""
    timestamp: float = 0.0
    spot_price: float = 0.0

    # 总清算量
    total_long_liq: float = 0.0    # 多头清算总量(USD)
    total_short_liq: float = 0.0   # 空头清算总量(USD)
    total_liq: float = 0.0         # 总清算量

    # 清算密集区
    long_liq_cluster: float = 0.0     # 多头清算密集区价格(下方)
    short_liq_cluster: float = 0.0    # 空头清算密集区价格(上方)
    long_cluster_amount: float = 0.0  # 多头密集区清算量
    short_cluster_amount: float = 0.0 # 空头密集区清算量

    # 清算级联风险 (0-1)
    cascade_risk: float = 0.0

    # 每个价格水平的清算量 (用于热力图)
    per_price_long_liq: Dict[str, float] = field(default_factory=dict)
    per_price_short_liq: Dict[str, float] = field(default_factory=dict)

    # 统计
    n_liquidations: int = 0
    n_price_levels: int = 0

    # 计算时间
    calc_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "spot_price": self.spot_price,
            "total_long_liq": self.total_long_liq,
            "total_short_liq": self.total_short_liq,
            "total_liq": self.total_liq,
            "long_liq_cluster": self.long_liq_cluster,
            "short_liq_cluster": self.short_liq_cluster,
            "long_cluster_amount": self.long_cluster_amount,
            "short_cluster_amount": self.short_cluster_amount,
            "cascade_risk": self.cascade_risk,
            "per_price_long_liq": dict(self.per_price_long_liq),
            "per_price_short_liq": dict(self.per_price_short_liq),
            "n_liquidations": self.n_liquidations,
            "n_price_levels": self.n_price_levels,
            "calc_time_ms": self.calc_time_ms,
        }


# ============================================================================
# 清算热力图计算器
# ============================================================================

class LiquidationCalculator:
    """清算热力图计算器

    从清算数据计算各价格水平的清算密度，识别清算密集区和级联风险。
    """

    # 价格水平聚合精度 (百分比)
    PRICE_LEVEL_PCT = 0.005  # 0.5% 间隔

    # 清算密集区阈值 (USD)
    CLUSTER_THRESHOLD_USD = 1e5  # 10万美元以上视为密集区

    # 级联风险梯度阈值
    CASCADE_GRADIENT_THRESHOLD = 5.0  # 相邻水平清算量比值>5视为级联风险

    def __init__(self, price_level_pct: float = None):
        """
        Args:
            price_level_pct: 价格水平聚合精度 (默认0.5%)
        """
        self.price_level_pct = price_level_pct or self.PRICE_LEVEL_PCT

    def calculate(
        self,
        liquidations: List[Dict[str, Any]],
        spot_price: float,
        timestamp: Optional[float] = None,
        symbol: str = "",
    ) -> LiquidationResult:
        """计算清算热力图

        Args:
            liquidations: 清算数据列表，每条需包含:
                - price: 清算价格
                - side: "long" 或 "short"
                - amount_usd: 清算金额(USD)
            spot_price: 当前标的价格
            timestamp: 时间戳
            symbol: 交易对

        Returns:
            LiquidationResult
        """
        import time as _time
        t0 = _time.time()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        result = LiquidationResult(
            symbol=symbol,
            timestamp=timestamp,
            spot_price=spot_price,
        )

        if not liquidations or spot_price <= 0:
            return result

        # 按价格水平聚合
        long_liq: Dict[float, float] = {}   # price_level -> amount
        short_liq: Dict[float, float] = {}
        total_long = 0.0
        total_short = 0.0

        for liq in liquidations:
            try:
                price = float(liq.get("price", 0))
                side = str(liq.get("side", "")).lower()
                amount = float(liq.get("amount_usd", liq.get("amount", 0)))

                if price <= 0 or amount <= 0:
                    continue

                # 聚合到价格水平
                level = self._price_to_level(price, spot_price)

                if side in ("long", "buy"):
                    long_liq[level] = long_liq.get(level, 0.0) + amount
                    total_long += amount
                elif side in ("short", "sell"):
                    short_liq[level] = short_liq.get(level, 0.0) + amount
                    total_short += amount

            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Liquidation skipped: %s", e)
                continue

        # 填充结果
        result.total_long_liq = total_long
        result.total_short_liq = total_short
        result.total_liq = total_long + total_short
        result.n_liquidations = len(liquidations)
        result.n_price_levels = len(set(list(long_liq.keys()) + list(short_liq.keys())))

        # 识别清算密集区
        result.long_liq_cluster, result.long_cluster_amount = self._find_cluster(
            long_liq, spot_price, below=True
        )
        result.short_liq_cluster, result.short_cluster_amount = self._find_cluster(
            short_liq, spot_price, below=False
        )

        # 计算级联风险
        result.cascade_risk = self._calc_cascade_risk(
            long_liq, short_liq, spot_price
        )

        # 转换为字符串key
        result.per_price_long_liq = {f"{k:.2f}": v for k, v in long_liq.items()}
        result.per_price_short_liq = {f"{k:.2f}": v for k, v in short_liq.items()}

        result.calc_time_ms = (_time.time() - t0) * 1000

        logger.info(
            "Liquidation calc: symbol=%s spot=%.2f total_liq=%.2f long=%.2f short=%.2f "
            "long_cluster=%.2f(%.2f) short_cluster=%.2f(%.2f) cascade=%.2f n=%d calc=%.1fms",
            symbol, spot_price, result.total_liq, total_long, total_short,
            result.long_liq_cluster, result.long_cluster_amount,
            result.short_liq_cluster, result.short_cluster_amount,
            result.cascade_risk, result.n_liquidations, result.calc_time_ms,
        )

        return result

    def _price_to_level(self, price: float, spot_price: float) -> float:
        """将价格聚合到价格水平

        Args:
            price: 实际价格
            spot_price: 当前价格

        Returns:
            float: 聚合后的价格水平
        """
        # 按百分比聚合
        pct = price / spot_price
        level_pct = round(pct / self.price_level_pct) * self.price_level_pct
        return level_pct * spot_price

    def _find_cluster(
        self,
        liq_dict: Dict[float, float],
        spot_price: float,
        below: bool = True,
    ) -> Tuple[float, float]:
        """查找清算密集区

        Args:
            liq_dict: 价格水平 -> 清算量
            spot_price: 当前价格
            below: True=找下方的密集区, False=找上方

        Returns:
            (cluster_price, cluster_amount)
        """
        if not liq_dict:
            return 0.0, 0.0

        # 筛选上方/下方
        if below:
            filtered = {p: a for p, a in liq_dict.items() if p < spot_price}
        else:
            filtered = {p: a for p, a in liq_dict.items() if p > spot_price}

        if not filtered:
            return 0.0, 0.0

        # 找清算量最大的价格水平
        best_price = max(filtered.items(), key=lambda x: x[1])[0]
        best_amount = filtered[best_price]

        # 检查是否超过阈值
        if best_amount < self.CLUSTER_THRESHOLD_USD:
            return 0.0, 0.0

        return best_price, best_amount

    def _calc_cascade_risk(
        self,
        long_liq: Dict[float, float],
        short_liq: Dict[float, float],
        spot_price: float,
    ) -> float:
        """计算清算级联风险 (0-1)

        级联风险 = max(相邻价格水平清算量梯度)

        Args:
            long_liq: 多头清算字典
            short_liq: 空头清算字典
            spot_price: 当前价格

        Returns:
            float: 级联风险评分 (0-1)
        """
        if not long_liq and not short_liq:
            return 0.0

        # 合并所有价格水平
        all_levels = sorted(set(list(long_liq.keys()) + list(short_liq.keys())))
        if len(all_levels) < 2:
            return 0.0

        # 计算相邻水平的清算量比值
        max_gradient = 0.0
        for i in range(1, len(all_levels)):
            prev_level = all_levels[i - 1]
            curr_level = all_levels[i]

            prev_amount = long_liq.get(prev_level, 0) + short_liq.get(prev_level, 0)
            curr_amount = long_liq.get(curr_level, 0) + short_liq.get(curr_level, 0)

            if prev_amount > 0 and curr_amount > 0:
                gradient = max(prev_amount / curr_amount, curr_amount / prev_amount)
                if gradient > max_gradient:
                    max_gradient = gradient

        # 归一化到0-1
        if max_gradient <= 1.0:
            return 0.0

        # gradient > CASCADE_GRADIENT_THRESHOLD 时风险=1
        risk = min(1.0, (max_gradient - 1.0) / (self.CASCADE_GRADIENT_THRESHOLD - 1.0))
        return risk


# ============================================================================
# 便捷函数
# ============================================================================

def calculate_liquidation_heatmap(
    liquidations: List[Dict[str, Any]],
    spot_price: float,
    symbol: str = "",
    timestamp: Optional[float] = None,
) -> LiquidationResult:
    """便捷函数: 计算清算热力图"""
    calc = LiquidationCalculator()
    return calc.calculate(liquidations, spot_price, timestamp, symbol)


def create_synthetic_liquidations(
    spot_price: float,
    n_levels: int = 20,
    price_range_pct: float = 0.10,
    base_amount: float = 50000.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """创建合成清算数据 (用于测试和降级模式)

    生成合理的清算分布:
      - 当前价格附近清算量最大 (杠杆最密集)
      - 多头清算在下方, 空头清算在上方
      - 随机扰动模拟真实分布

    Args:
        spot_price: 当前价格
        n_levels: 价格水平数量
        price_range_pct: 价格范围百分比 (±10%)
        base_amount: 基础清算金额(USD)
        seed: 随机种子

    Returns:
        list: 合成清算数据
    """
    import random
    random.seed(seed)

    liquidations = []
    step = spot_price * price_range_pct / n_levels

    for i in range(-n_levels, n_levels + 1):
        price = spot_price + i * step
        if price <= 0:
            continue

        # 距离当前价格越近, 清算量越大
        distance = abs(i) / n_levels
        amount_factor = 1.0 - distance * 0.5  # 近处1.0, 远处0.5

        # 随机扰动
        amount_factor *= (0.7 + random.random() * 0.6)

        # 多头清算 (价格低于当前)
        if i < 0:
            amount = base_amount * amount_factor * (1 + random.random())
            liquidations.append({
                "price": price,
                "side": "long",
                "amount_usd": amount,
            })
        # 空头清算 (价格高于当前)
        elif i > 0:
            amount = base_amount * amount_factor * (1 + random.random())
            liquidations.append({
                "price": price,
                "side": "short",
                "amount_usd": amount,
            })

    # 添加一些大额清算 (清算密集区)
    # 多头密集区 (下方5%)
    long_cluster_price = spot_price * 0.95
    liquidations.append({
        "price": long_cluster_price,
        "side": "long",
        "amount_usd": base_amount * 5,
    })
    # 空头密集区 (上方5%)
    short_cluster_price = spot_price * 1.05
    liquidations.append({
        "price": short_cluster_price,
        "side": "short",
        "amount_usd": base_amount * 4,
    })

    return liquidations
