#!/usr/bin/env python3
"""稳定币溢价门控模块 — Layer 1 市场数据层 P2

稳定币溢价（Stablecoin Premium）指 USDT/USDC 等稳定币相对于美元的溢价/折价：
  - 溢价 > 0：市场恐慌，资金流入稳定币避险 → 看跌信号
  - 折价 < 0：市场乐观，资金流出稳定币 → 看涨信号

门控逻辑:
  - 极端溢价 (>= 1%) → 做空 BOOST / 做多 REDUCE
  - 高溢价 (>= 0.5%) → 做空 BOOST / 做多 REDUCE
  - 极端折价 (<= -1%) → 做多 BOOST / 做空 REDUCE
  - 高折价 (<= -0.5%) → 做多 BOOST / 做空 REDUCE
  - 正常 → ALLOW

用法:
    from sandbox_trading.stablecoin_premium_gate import StablecoinPremiumGate
    gate = StablecoinPremiumGate(enabled=True)
    decision = gate.apply_to_decision(decision, symbol="BTC-USDT")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# 预取数据读取器 (可选, 失败时降级到原有逻辑)
try:
    from .prefetched_data_reader import PrefetchedDataReader
except ImportError:
    try:
        from prefetched_data_reader import PrefetchedDataReader
    except ImportError:
        PrefetchedDataReader = None

logger = logging.getLogger("hermes.stablecoin_premium_gate")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class SPPremiumLevel(Enum):
    """稳定币溢价等级"""
    NORMAL = "normal"                # 正常
    HIGH_DISCOUNT = "high_discount"  # 高折价 (乐观)
    EXTREME_DISCOUNT = "extreme_discount"  # 极端折价
    HIGH_PREMIUM = "high_premium"    # 高溢价 (恐慌)
    EXTREME_PREMIUM = "extreme_premium"    # 极端溢价


class SPGateAction(Enum):
    """稳定币溢价门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class SPAnalysisResult:
    """稳定币溢价分析结果"""
    usdt_price: float = 1.0           # USDT/USD 价格
    premium_pct: float = 0.0          # 溢价百分比 (正=溢价, 负=折价)
    level: SPPremiumLevel = SPPremiumLevel.NORMAL
    timestamp: float = field(default_factory=time.time)
    has_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "usdt_price": self.usdt_price,
            "premium_pct": self.premium_pct,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "has_data": self.has_data,
        }


# ============================================================================
# 稳定币溢价门控主类
# ============================================================================

class StablecoinPremiumGate:
    """稳定币溢价门控

    工作流程:
      1. 获取USDT/USD价格
      2. 计算溢价率
      3. 基于溢价率调整策略决策

    门控规则:
      1. 极端溢价 (>= PREMIUM_EXTREME) → 做空 BOOST / 做多 REDUCE
      2. 高溢价 (>= PREMIUM_HIGH) → 做空 BOOST / 做多 REDUCE
      3. 极端折价 (<= DISCOUNT_EXTREME) → 做多 BOOST / 做空 REDUCE
      4. 高折价 (<= DISCOUNT_HIGH) → 做多 BOOST / 做空 REDUCE
      5. 正常 → ALLOW
    """

    # 溢价阈值 (百分比)
    PREMIUM_HIGH = 0.005         # 0.5% 溢价 → 恐慌
    PREMIUM_EXTREME = 0.01       # 1% 溢价 → 极度恐慌
    DISCOUNT_HIGH = -0.005       # 0.5% 折价 → 乐观
    DISCOUNT_EXTREME = -0.01     # 1% 折价 → 极度乐观

    # 调整系数
    BOOST_MULTIPLIER = 1.15
    REDUCE_MULTIPLIER = 0.7
    EXTREME_REDUCE_MULTIPLIER = 0.5

    # 基准价格 (USDT/USD 理论1:1)
    BASE_PRICE = 1.0

    def __init__(self, enabled: bool = True):
        """初始化稳定币溢价门控

        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis: Optional[SPAnalysisResult] = None
        self._price_history: List[float] = []

        logger.info(
            "StablecoinPremiumGate initialized: enabled=%s "
            "premium_high=%.4f premium_extreme=%.4f "
            "discount_high=%.4f discount_extreme=%.4f",
            self.enabled,
            self.PREMIUM_HIGH, self.PREMIUM_EXTREME,
            self.DISCOUNT_HIGH, self.DISCOUNT_EXTREME,
        )

    # ----------------------------------------------------------------
    # 溢价分析
    # ----------------------------------------------------------------

    def analyze_premium(self, usdt_price: float) -> SPAnalysisResult:
        """分析稳定币溢价

        Args:
            usdt_price: USDT/USD 当前价格

        Returns:
            SPAnalysisResult
        """
        result = SPAnalysisResult()

        if usdt_price <= 0:
            return result

        result.usdt_price = usdt_price
        result.premium_pct = (usdt_price - self.BASE_PRICE) / self.BASE_PRICE

        # 判断等级
        if result.premium_pct >= self.PREMIUM_EXTREME:
            result.level = SPPremiumLevel.EXTREME_PREMIUM
        elif result.premium_pct >= self.PREMIUM_HIGH:
            result.level = SPPremiumLevel.HIGH_PREMIUM
        elif result.premium_pct <= self.DISCOUNT_EXTREME:
            result.level = SPPremiumLevel.EXTREME_DISCOUNT
        elif result.premium_pct <= self.DISCOUNT_HIGH:
            result.level = SPPremiumLevel.HIGH_DISCOUNT
        else:
            result.level = SPPremiumLevel.NORMAL

        result.has_data = True

        # 更新历史
        self._price_history.append(usdt_price)
        if len(self._price_history) > 100:
            self._price_history.pop(0)

        return result

    # ----------------------------------------------------------------
    # 决策门控
    # ----------------------------------------------------------------

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "",
        usdt_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """对策略决策应用门控

        Args:
            decision: 策略决策字典
            symbol: 交易对
            usdt_price: USDT/USD价格 (可选, 不传则用合成数据)

        Returns:
            调整后的决策字典
        """
        if not self.enabled:
            return decision

        self._total_calls += 1

        # 获取USDT价格
        # 优先读取预取数据 (ticker 中的 change_percentage 推算 USDT 溢价)
        if usdt_price is None and PrefetchedDataReader is not None:
            try:
                ticker = PrefetchedDataReader.read_ticker(symbol)
                if ticker is not None:
                    # 从 BTC 涨跌幅推算 USDT 溢价:
                    #   BTC 上涨 → USDT 需求增加 → 溢价
                    #   BTC 下跌 → USDT 被抛售 → 折价
                    change_pct = float(ticker.get("change_percentage", 0))
                    # 1% BTC 涨幅 → 约 0.005% USDT 溢价
                    usdt_price = 1.0 + (change_pct / 100.0) * 0.005
                    usdt_price = max(0.95, min(1.05, usdt_price))
                    logger.debug(
                        "Using prefetched ticker for SP: change=%.2f%% → usdt_price=%.5f",
                        change_pct, usdt_price,
                    )
            except Exception as e:
                logger.debug("PrefetchedDataReader ticker failed for SP: %s", e)

        if usdt_price is None:
            usdt_price = self._generate_synthetic_price()

        if usdt_price <= 0:
            self._no_data_count += 1
            return decision

        # 分析溢价
        analysis = self.analyze_premium(usdt_price)
        self._last_analysis = analysis

        if not analysis.has_data:
            self._no_data_count += 1
            return decision

        # 应用门控规则
        action, multiplier, reason = self._evaluate_rules(analysis, decision)

        if action == SPGateAction.ALLOW or action == SPGateAction.NEUTRAL:
            return decision

        # 调整仓位
        new_decision = dict(decision)
        orig_qty = new_decision.get("quantity", 0.0)
        new_qty = orig_qty * multiplier
        if new_qty < 0:
            new_qty = 0.0
        new_decision["quantity"] = new_qty
        new_decision["sp_gate_action"] = action.value
        new_decision["sp_gate_multiplier"] = multiplier
        new_decision["sp_gate_reason"] = reason
        new_decision["sp_premium_pct"] = analysis.premium_pct
        new_decision["sp_premium_level"] = analysis.level.value

        self._total_adjusted += 1
        if action == SPGateAction.BOOST:
            self._boost_count += 1
        else:
            self._reduce_count += 1

        logger.debug(
            "SPGate %s symbol=%s qty=%.6f→%.6f premium_pct=%.5f level=%s reason=%s",
            action.value.upper(), symbol, orig_qty, new_qty,
            analysis.premium_pct, analysis.level.value, reason,
        )

        return new_decision

    def _evaluate_rules(
        self,
        analysis: SPAnalysisResult,
        decision: Dict[str, Any],
    ) -> tuple:
        """根据分析结果和决策方向，应用门控规则

        Returns:
            (action, multiplier, reason)
        """
        action_str = str(decision.get("action", "")).lower()
        is_long = "buy" in action_str or "long" in action_str
        is_short = "sell" in action_str or "short" in action_str

        level = analysis.level

        # 规则1: 极端溢价 → 做空 BOOST / 做多 REDUCE
        if level == SPPremiumLevel.EXTREME_PREMIUM:
            if is_short:
                return (
                    SPGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"extreme_premium_short_boost={analysis.premium_pct:.5f}",
                )
            if is_long:
                return (
                    SPGateAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_premium_long_reduce={analysis.premium_pct:.5f}",
                )

        # 规则2: 高溢价 → 做空 BOOST / 做多 REDUCE
        if level == SPPremiumLevel.HIGH_PREMIUM:
            if is_short:
                return (
                    SPGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"high_premium_short_boost={analysis.premium_pct:.5f}",
                )
            if is_long:
                return (
                    SPGateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"high_premium_long_reduce={analysis.premium_pct:.5f}",
                )

        # 规则3: 极端折价 → 做多 BOOST / 做空 REDUCE
        if level == SPPremiumLevel.EXTREME_DISCOUNT:
            if is_long:
                return (
                    SPGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"extreme_discount_long_boost={analysis.premium_pct:.5f}",
                )
            if is_short:
                return (
                    SPGateAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_discount_short_reduce={analysis.premium_pct:.5f}",
                )

        # 规则4: 高折价 → 做多 BOOST / 做空 REDUCE
        if level == SPPremiumLevel.HIGH_DISCOUNT:
            if is_long:
                return (
                    SPGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"high_discount_long_boost={analysis.premium_pct:.5f}",
                )
            if is_short:
                return (
                    SPGateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"high_discount_short_reduce={analysis.premium_pct:.5f}",
                )

        return SPGateAction.ALLOW, 1.0, "normal_premium"

    # ----------------------------------------------------------------
    # 合成数据生成
    # ----------------------------------------------------------------

    def _generate_synthetic_price(self) -> float:
        """生成合成USDT价格 (用于沙盘测试)

        基于时间戳和随机因子生成有波动的USDT价格
        """
        # 基于时间戳生成伪随机波动
        ts = time.time()
        # 用sin函数生成周期性波动 + 随机扰动
        import math
        wave = math.sin(ts / 3600) * 0.003  # 1小时周期，振幅0.3%
        noise = ((ts * 1000) % 1000) / 100000.0 - 0.005  # ±0.5%噪声
        price = self.BASE_PRICE + wave + noise
        return max(0.95, min(1.05, price))  # 限制在[0.95, 1.05]

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "total_calls": self._total_calls,
            "total_adjusted": self._total_adjusted,
            "boost_count": self._boost_count,
            "reduce_count": self._reduce_count,
            "no_data_count": self._no_data_count,
            "adjust_rate": (
                self._total_adjusted / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "last_analysis": self._last_analysis.to_dict() if self._last_analysis else None,
            "price_history_len": len(self._price_history),
        }

    def reset(self):
        """重置统计"""
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis = None
        self._price_history.clear()
        logger.info("StablecoinPremiumGate stats reset")
