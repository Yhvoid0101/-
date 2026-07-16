#!/usr/bin/env python3
"""SOPR (Spent Output Profit Ratio) 计算器 — Layer 8 质变数据模块3

SOPR = 已花费输出的价值 / 这些输出创建时的价值

核心概念:
  - SOPR > 1: 持有者盈利卖出 (市场情绪乐观, 趋势可能持续)
  - SOPR < 1: 持有者亏损卖出 (市场恐慌抛售, 下跌趋势可能加速)
  - SOPR = 1: 持有者盈亏平衡 (临界点, 常作为支撑/阻力)

SOPR 变体:
  - aSOPR (Adjusted SOPR): 排除矿工和交易所内部转账, 更纯粹反映投机者行为
  - LTH-SOPR (Long-Term Holder): 长期持有者 (>155天), 趋势锚定
  - STH-SOPR (Short-Term Holder): 短期持有者 (<155天), 情绪敏感

门控逻辑:
  - SOPR > 1.1 + 做多 → BOOST (盈利情绪持续, 趋势向上)
  - SOPR < 0.9 + 做空 → BOOST (恐慌抛售, 趋势向下)
  - SOPR > 1.1 + 做空 → REDUCE (盈利情绪对抗做空)
  - SOPR < 0.9 + 做多 → REDUCE (恐慌情绪对抗做多)
  - SOPR 极端值 (>1.3 或 <0.7) → FORCE (调整止盈止损)

用法:
    from sandbox_trading.sopr_calculator import SOPRCalculator
    calc = SOPRCalculator()
    result = calc.calculate(spent_outputs, symbol="BTC")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.sopr")


# ============================================================================
# SOPR 结果数据结构
# ============================================================================

@dataclass(slots=True)
class SOPRResult:
    """SOPR 计算结果

    SOPR = sum(spent_value_created) / sum(spent_value_spent)
    所有金额单位均为 USD
    """
    symbol: str = ""
    timestamp: float = 0.0

    # 基本 SOPR
    sopr: float = 0.0           # 整体 SOPR
    a_sopr: float = 0.0         # 调整后 SOPR (排除矿工/交易所)
    lth_sopr: float = 0.0       # 长期持有者 SOPR
    sth_sopr: float = 0.0       # 短期持有者 SOPR

    # 分量
    total_spent_value: float = 0.0      # 已花费输出总价值 (USD, 卖出时)
    total_created_value: float = 0.0    # 创建时总价值 (USD, 买入时)
    n_outputs: int = 0                   # 输出数量
    n_profit: int = 0                    # 盈利输出数量
    n_loss: int = 0                      # 亏损输出数量

    # 比率
    profit_ratio: float = 0.0    # 盈利输出占比 (0-1)
    loss_ratio: float = 0.0      # 亏损输出占比 (0-1)

    # 趋势 (与前一周期对比)
    sopr_trend: float = 0.0      # SOPR 变化方向 (+1 上升 / -1 下降 / 0 持平)
    sopr_change: float = 0.0     # SOPR 变化值

    # 极端标记
    is_extreme_profit: bool = False   # SOPR > 1.3
    is_extreme_loss: bool = False     # SOPR < 0.7

    # 计算时间
    calc_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "sopr": self.sopr,
            "a_sopr": self.a_sopr,
            "lth_sopr": self.lth_sopr,
            "sth_sopr": self.sth_sopr,
            "total_spent_value": self.total_spent_value,
            "total_created_value": self.total_created_value,
            "n_outputs": self.n_outputs,
            "n_profit": self.n_profit,
            "n_loss": self.n_loss,
            "profit_ratio": self.profit_ratio,
            "loss_ratio": self.loss_ratio,
            "sopr_trend": self.sopr_trend,
            "sopr_change": self.sopr_change,
            "is_extreme_profit": self.is_extreme_profit,
            "is_extreme_loss": self.is_extreme_loss,
            "calc_time_ms": self.calc_time_ms,
        }


# ============================================================================
# SOPR 计算器
# ============================================================================

class SOPRCalculator:
    """SOPR (Spent Output Profit Ratio) 计算器

    从链上已花费输出数据计算 SOPR 及其变体。
    """

    # 长期持有者阈值 (天数) — 持有 > 155 天视为长期持有者
    LTH_THRESHOLD_DAYS = 155

    # 极端阈值
    EXTREME_PROFIT_THRESHOLD = 1.3   # SOPR > 1.3 视为极端盈利
    EXTREME_LOSS_THRESHOLD = 0.7     # SOPR < 0.7 视为极端亏损

    # 趋势变化阈值
    TREND_CHANGE_THRESHOLD = 0.01    # SOPR 变化 > 0.01 视为趋势变化

    # 矿工地址过滤 (简化: 排除金额 > 50 BTC 的大额转账视为矿工/交易所)
    MINER_FILTER_BTC_THRESHOLD = 50.0

    def __init__(self):
        """初始化 SOPR 计算器"""

    def calculate(
        self,
        spent_outputs: List[Dict[str, Any]],
        symbol: str = "",
        timestamp: Optional[float] = None,
        prev_sopr: Optional[float] = None,
    ) -> SOPRResult:
        """计算 SOPR

        Args:
            spent_outputs: 已花费输出列表, 每条需包含:
                - created_value: 创建时价值 (USD)
                - spent_value: 花费时价值 (USD)
                - created_ts: 创建时间戳
                - spent_ts: 花费时间戳
                - amount_btc: BTC 数量 (可选, 用于矿工过滤)
            symbol: 交易对
            timestamp: 时间戳
            prev_sopr: 前一周期 SOPR (用于趋势计算)

        Returns:
            SOPRResult
        """
        import time as _time
        t0 = _time.time()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        result = SOPRResult(
            symbol=symbol,
            timestamp=timestamp,
        )

        if not spent_outputs:
            result.sopr = 1.0  # 默认盈亏平衡
            result.a_sopr = 1.0
            result.lth_sopr = 1.0
            result.sth_sopr = 1.0
            return result

        # 基本统计
        total_spent = 0.0
        total_created = 0.0
        n_profit = 0
        n_loss = 0
        n_total = 0

        # 分类统计 (长期/短期持有者)
        lth_spent = 0.0
        lth_created = 0.0
        sth_spent = 0.0
        sth_created = 0.0

        # 调整后统计 (排除矿工)
        adj_spent = 0.0
        adj_created = 0.0

        lth_threshold_sec = self.LTH_THRESHOLD_DAYS * 86400

        for out in spent_outputs:
            try:
                created_val = float(out.get("created_value", 0))
                spent_val = float(out.get("spent_value", 0))
                created_ts = float(out.get("created_ts", 0))
                spent_ts = float(out.get("spent_ts", timestamp))
                amount_btc = float(out.get("amount_btc", 0))

                if created_val <= 0 or spent_val <= 0:
                    continue

                # 基本累计
                total_spent += spent_val
                total_created += created_val
                n_total += 1

                # 盈亏统计
                if spent_val > created_val:
                    n_profit += 1
                elif spent_val < created_val:
                    n_loss += 1

                # 长期/短期持有者分类
                holding_period = spent_ts - created_ts
                if holding_period > lth_threshold_sec:
                    lth_spent += spent_val
                    lth_created += created_val
                else:
                    sth_spent += spent_val
                    sth_created += created_val

                # 调整后 SOPR (排除矿工: amount_btc > 阈值)
                if symbol.upper() == "BTC" and amount_btc > self.MINER_FILTER_BTC_THRESHOLD:
                    continue  # 跳过矿工/交易所大额转账

                adj_spent += spent_val
                adj_created += created_val

            except (ValueError, TypeError, KeyError) as e:
                logger.debug("Output skipped: %s", e)
                continue

        if total_created <= 0 or n_total == 0:
            result.sopr = 1.0
            result.a_sopr = 1.0
            result.lth_sopr = 1.0
            result.sth_sopr = 1.0
            return result

        # 计算 SOPR
        result.sopr = total_spent / total_created
        result.a_sopr = (adj_spent / adj_created) if adj_created > 0 else result.sopr
        result.lth_sopr = (lth_spent / lth_created) if lth_created > 0 else result.sopr
        result.sth_sopr = (sth_spent / sth_created) if sth_created > 0 else result.sopr

        # 填充统计
        result.total_spent_value = total_spent
        result.total_created_value = total_created
        result.n_outputs = n_total
        result.n_profit = n_profit
        result.n_loss = n_loss
        result.profit_ratio = n_profit / n_total if n_total > 0 else 0.0
        result.loss_ratio = n_loss / n_total if n_total > 0 else 0.0

        # 极端标记
        result.is_extreme_profit = result.sopr > self.EXTREME_PROFIT_THRESHOLD
        result.is_extreme_loss = result.sopr < self.EXTREME_LOSS_THRESHOLD

        # 趋势计算
        if prev_sopr is not None and prev_sopr > 0:
            change = result.sopr - prev_sopr
            result.sopr_change = change
            if change > self.TREND_CHANGE_THRESHOLD:
                result.sopr_trend = 1.0
            elif change < -self.TREND_CHANGE_THRESHOLD:
                result.sopr_trend = -1.0
            else:
                result.sopr_trend = 0.0

        result.calc_time_ms = (_time.time() - t0) * 1000

        logger.info(
            "SOPR calc: symbol=%s sopr=%.4f a_sopr=%.4f lth=%.4f sth=%.4f "
            "n=%d profit=%d loss=%d profit_ratio=%.2f trend=%+.1f calc=%.1fms",
            symbol, result.sopr, result.a_sopr, result.lth_sopr, result.sth_sopr,
            n_total, n_profit, n_loss, result.profit_ratio,
            result.sopr_trend, result.calc_time_ms,
        )

        return result


# ============================================================================
# 便捷函数
# ============================================================================

def calculate_sopr(
    spent_outputs: List[Dict[str, Any]],
    symbol: str = "",
    timestamp: Optional[float] = None,
    prev_sopr: Optional[float] = None,
) -> SOPRResult:
    """便捷函数: 计算 SOPR"""
    calc = SOPRCalculator()
    return calc.calculate(spent_outputs, symbol, timestamp, prev_sopr)


def create_synthetic_sopr_data(
    spot_price: float,
    symbol: str = "BTC",
    n_outputs: int = 1000,
    profit_ratio: float = 0.55,
    avg_holding_days: int = 30,
    sopr_target: float = 1.05,
    seed: int = 42,
    timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """创建合成 SOPR 数据 (用于测试和降级模式)

    生成合理的已花费输出数据:
      - profit_ratio 比例的输出盈利 (spent > created)
      - (1-profit_ratio) 比例的输出亏损 (spent < created)
      - 平均 SOPR 接近 sopr_target
      - 包含长期/短期持有者分布
      - 包含少量矿工大额转账

    Args:
        spot_price: 当前现货价格 (用于生成金额)
        symbol: 币种
        n_outputs: 输出数量
        profit_ratio: 盈利输出占比 (0-1)
        avg_holding_days: 平均持有天数
        sopr_target: 目标 SOPR 值
        seed: 随机种子
        timestamp: 时间戳

    Returns:
        list: 合成已花费输出数据
    """
    import random
    random.seed(seed)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).timestamp()

    outputs = []
    n_profit = int(n_outputs * profit_ratio)
    n_loss = n_outputs - n_profit

    # 盈利输出: spent > created
    for i in range(n_profit):
        # 持有期分布 (短期多, 长期少)
        holding_days = max(1, int(np.random.exponential(avg_holding_days)))
        holding_sec = holding_days * 86400
        created_ts = timestamp - holding_sec

        # 创建价值 (基于现货价格, 假设价格当时较低)
        price_appreciation = sopr_target * (0.8 + random.random() * 0.4)
        amount_btc = np.random.exponential(0.5)  # 大多数小额
        created_value = amount_btc * spot_price / price_appreciation
        spent_value = created_value * price_appreciation

        outputs.append({
            "created_value": created_value,
            "spent_value": spent_value,
            "created_ts": created_ts,
            "spent_ts": timestamp,
            "amount_btc": amount_btc,
        })

    # 亏损输出: spent < created
    for i in range(n_loss):
        holding_days = max(1, int(np.random.exponential(avg_holding_days)))
        holding_sec = holding_days * 86400
        created_ts = timestamp - holding_sec

        # 创建价值 (假设价格当时较高)
        price_depreciation = (2.0 - sopr_target) * (0.8 + random.random() * 0.4)
        price_depreciation = max(0.5, price_depreciation)  # 至少0.5
        amount_btc = np.random.exponential(0.5)
        created_value = amount_btc * spot_price * price_depreciation
        spent_value = created_value / price_depreciation

        outputs.append({
            "created_value": created_value,
            "spent_value": spent_value,
            "created_ts": created_ts,
            "spent_ts": timestamp,
            "amount_btc": amount_btc,
        })

    # 添加少量矿工大额转账 (amount_btc > 50)
    for i in range(5):
        holding_days = random.randint(1, 7)  # 矿工通常持有时间短
        holding_sec = holding_days * 86400
        created_ts = timestamp - holding_sec
        amount_btc = 60.0 + random.random() * 40.0  # 60-100 BTC
        created_value = amount_btc * spot_price * 0.98
        spent_value = amount_btc * spot_price

        outputs.append({
            "created_value": created_value,
            "spent_value": spent_value,
            "created_ts": created_ts,
            "spent_ts": timestamp,
            "amount_btc": amount_btc,
        })

    random.shuffle(outputs)
    return outputs
