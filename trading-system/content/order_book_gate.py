#!/usr/bin/env python3
"""订单簿深度门控模块 — Layer 1 市场数据层

基于 UnifiedOrderBook (matching_engine_interface.py:123) 进行深度分析，
对策略决策进行门控调整：
  - 大单墙检测：某价位挂单量远超对侧 → 反向 BOOST / 顺势 REDUCE
  - 不平衡检测：买卖盘总量显著失衡 → 顺势 BOOST
  - 价差检测：spread过大 → REDUCE (流动性差)

门控动作:
  - ALLOW: 正常通过
  - BOOST: 放大仓位 (×1.15)
  - REDUCE: 缩小仓位 (×0.7)
  - NEUTRAL: 中性 (不调整)

用法:
    from sandbox_trading.order_book_gate import OrderBookGate
    gate = OrderBookGate(enabled=True)
    result = gate.apply_to_decision(decision, symbol="BTC-USDT", order_book=ob)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# 预取数据读取器 (可选, 失败时降级到原有逻辑)
try:
    from .prefetched_data_reader import PrefetchedDataReader
except ImportError:
    try:
        from prefetched_data_reader import PrefetchedDataReader
    except ImportError:
        PrefetchedDataReader = None

logger = logging.getLogger("hermes.order_book_gate")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class OBGateAction(Enum):
    """订单簿门控动作"""
    ALLOW = "allow"          # 正常通过
    BOOST = "boost"          # 放大仓位
    REDUCE = "reduce"        # 缩小仓位
    NEUTRAL = "neutral"      # 中性 (不调整)


@dataclass(slots=True)
class OBAnalysisResult:
    """订单簿分析结果"""
    mid_price: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0              # spread / mid_price
    bid_depth: float = 0.0               # 买盘总深度
    ask_depth: float = 0.0               # 卖盘总深度
    imbalance: float = 0.0               # (bid-ask)/(bid+ask), [-1, 1]
    bid_wall: Optional[Tuple[float, float]] = None  # (price, qty)
    ask_wall: Optional[Tuple[float, float]] = None
    bid_wall_ratio: float = 0.0          # bid_wall.qty / avg_other_bid_qty
    ask_wall_ratio: float = 0.0
    has_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "imbalance": self.imbalance,
            "bid_wall": self.bid_wall,
            "ask_wall": self.ask_wall,
            "bid_wall_ratio": self.bid_wall_ratio,
            "ask_wall_ratio": self.ask_wall_ratio,
            "has_data": self.has_data,
        }


@dataclass(slots=True)
class OBGateResult:
    """门控结果"""
    action: OBGateAction = OBGateAction.ALLOW
    multiplier: float = 1.0
    reason: str = ""
    analysis: Optional[OBAnalysisResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "multiplier": self.multiplier,
            "reason": self.reason,
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }


# ============================================================================
# 订单簿门控主类
# ============================================================================

class OrderBookGate:
    """订单簿深度门控

    工作流程:
      1. analyze_order_book(): 分析订单簿，返回 OBAnalysisResult
      2. apply_to_decision(): 根据分析结果调整策略决策

    门控规则 (10条):
      1. 极端价差 (spread_pct >= SPREAD_PCT_EXTREME) → EXTREME_REDUCE ×0.5
      2. 高价差 (spread_pct >= SPREAD_PCT_HIGH) → REDUCE ×0.7
      3. 买盘墙 + 做多 → BOOST ×1.15
      4. 买盘墙 + 做空 → REDUCE ×0.7
      5. 卖盘墙 + 做空 → BOOST ×1.15
      6. 卖盘墙 + 做多 → REDUCE ×0.7
      7. 强买盘不平衡 + 做多 → BOOST
      8. 强买盘不平衡 + 做空 → REDUCE
      9. 强卖盘不平衡 + 做空 → BOOST
     10. 强卖盘不平衡 + 做多 → REDUCE
    """

    # 不平衡阈值
    IMBALANCE_STRONG = 0.3
    IMBALANCE_EXTREME = 0.5

    # 大单墙阈值 (wall_qty / avg_other_qty)
    WALL_RATIO_THRESHOLD = 3.0

    # 价差阈值 (spread / mid_price)
    SPREAD_PCT_HIGH = 0.002
    SPREAD_PCT_EXTREME = 0.005

    # 调整系数
    BOOST_MULTIPLIER = 1.15
    REDUCE_MULTIPLIER = 0.7
    EXTREME_REDUCE_MULTIPLIER = 0.5

    # 分析深度层数
    ANALYSIS_LEVELS = 10

    def __init__(self, enabled: bool = True):
        """初始化订单簿门控

        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis: Optional[OBAnalysisResult] = None

        logger.info(
            "OrderBookGate initialized: enabled=%s levels=%d "
            "imbalance_strong=%.2f wall_ratio=%.1f spread_high=%.4f spread_extreme=%.4f",
            self.enabled, self.ANALYSIS_LEVELS,
            self.IMBALANCE_STRONG, self.WALL_RATIO_THRESHOLD,
            self.SPREAD_PCT_HIGH, self.SPREAD_PCT_EXTREME,
        )

    # ----------------------------------------------------------------
    # 订单簿分析
    # ----------------------------------------------------------------

    def analyze_order_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> OBAnalysisResult:
        """分析订单簿

        Args:
            bids: [(price, qty), ...] 降序
            asks: [(price, qty), ...] 升序

        Returns:
            OBAnalysisResult
        """
        result = OBAnalysisResult()

        if not bids or not asks:
            return result

        # 截取分析层数
        bids_n = bids[: self.ANALYSIS_LEVELS]
        asks_n = asks[: self.ANALYSIS_LEVELS]

        # 中间价与价差
        best_bid = bids_n[0][0]
        best_ask = asks_n[0][0]
        result.mid_price = (best_bid + best_ask) / 2.0
        result.spread = best_ask - best_bid
        if result.mid_price > 0:
            result.spread_pct = result.spread / result.mid_price

        # 深度
        result.bid_depth = sum(q for _, q in bids_n)
        result.ask_depth = sum(q for _, q in asks_n)
        total = result.bid_depth + result.ask_depth
        if total > 0:
            result.imbalance = (result.bid_depth - result.ask_depth) / total

        # 大单墙检测
        result.bid_wall, result.bid_wall_ratio = self._detect_wall(bids_n)
        result.ask_wall, result.ask_wall_ratio = self._detect_wall(asks_n)

        result.has_data = True
        return result

    def _detect_wall(
        self,
        levels: List[Tuple[float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """检测大单墙

        Returns:
            (wall_price_qty, wall_ratio)
        """
        if len(levels) < 2:
            return None, 0.0

        # 找出最大挂单
        max_idx = 0
        max_qty = levels[0][1]
        for i in range(1, len(levels)):
            if levels[i][1] > max_qty:
                max_qty = levels[i][1]
                max_idx = i

        # 计算其他层位的平均挂单量
        other_levels = [q for i, (_, q) in enumerate(levels) if i != max_idx]
        if not other_levels:
            return None, 0.0
        avg_other = sum(other_levels) / len(other_levels)
        if avg_other <= 0:
            return None, 0.0

        ratio = max_qty / avg_other
        if ratio >= self.WALL_RATIO_THRESHOLD:
            return levels[max_idx], ratio
        return None, 0.0

    # ----------------------------------------------------------------
    # 决策门控
    # ----------------------------------------------------------------

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "",
        order_book: Any = None,
        bids: Optional[List[Tuple[float, float]]] = None,
        asks: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """对策略决策应用门控

        Args:
            decision: 策略决策字典 (action, quantity, ...)
            symbol: 交易对
            order_book: UnifiedOrderBook 对象 (可选, 优先使用)
            bids: 直接传入的bids (可选)
            asks: 直接传入的asks (可选)

        Returns:
            调整后的决策字典, 或 None (不调整)
        """
        if not self.enabled:
            return decision

        self._total_calls += 1

        # 提取 bids/asks
        ob_bids, ob_asks = self._extract_bids_asks(order_book, bids, asks)

        # 如果没有提供 bids/asks, 优先读取预取数据
        if not ob_bids or not ob_asks:
            if PrefetchedDataReader is not None:
                try:
                    cached_ob = PrefetchedDataReader.read_order_book(symbol)
                    if cached_ob is not None:
                        cached_bids = cached_ob.get("bids", [])
                        cached_asks = cached_ob.get("asks", [])
                        if cached_bids and cached_asks:
                            ob_bids = [tuple(b) for b in cached_bids]
                            ob_asks = [tuple(a) for a in cached_asks]
                            logger.debug("Using prefetched order book for %s", symbol)
                except Exception as e:
                    logger.debug("PrefetchedDataReader OB failed for %s: %s", symbol, e)
        if not ob_bids or not ob_asks:
            self._no_data_count += 1
            return decision

        # 分析订单簿
        analysis = self.analyze_order_book(ob_bids, ob_asks)
        self._last_analysis = analysis

        if not analysis.has_data:
            self._no_data_count += 1
            return decision

        # 应用门控规则
        action, multiplier, reason = self._evaluate_rules(analysis, decision)

        if action == OBGateAction.ALLOW or action == OBGateAction.NEUTRAL:
            return decision

        # 调整仓位
        new_decision = dict(decision)
        orig_qty = new_decision.get("quantity", 0.0)
        new_qty = orig_qty * multiplier
        if new_qty < 0:
            new_qty = 0.0
        new_decision["quantity"] = new_qty
        new_decision["ob_gate_action"] = action.value
        new_decision["ob_gate_multiplier"] = multiplier
        new_decision["ob_gate_reason"] = reason
        new_decision["ob_spread_pct"] = analysis.spread_pct
        new_decision["ob_imbalance"] = analysis.imbalance

        self._total_adjusted += 1
        if action == OBGateAction.BOOST:
            self._boost_count += 1
        else:
            self._reduce_count += 1

        logger.debug(
            "OBGate %s symbol=%s qty=%.6f→%.6f reason=%s "
            "spread_pct=%.5f imbalance=%.3f",
            action.value.upper(), symbol, orig_qty, new_qty, reason,
            analysis.spread_pct, analysis.imbalance,
        )

        return new_decision

    def _extract_bids_asks(
        self,
        order_book: Any,
        bids: Optional[List[Tuple[float, float]]],
        asks: Optional[List[Tuple[float, float]]],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """从order_book对象或直接参数中提取bids/asks"""
        # 优先使用直接传入的bids/asks
        if bids and asks:
            return bids, asks

        # 从order_book对象提取
        if order_book is not None:
            try:
                ob_bids = getattr(order_book, "bids", None)
                ob_asks = getattr(order_book, "asks", None)
                if ob_bids and ob_asks:
                    return list(ob_bids), list(ob_asks)
            except Exception:
                pass

            # 尝试字典形式
            if isinstance(order_book, dict):
                ob_bids = order_book.get("bids")
                ob_asks = order_book.get("asks")
                if ob_bids and ob_asks:
                    return list(ob_bids), list(ob_asks)

        return [], []

    def _evaluate_rules(
        self,
        analysis: OBAnalysisResult,
        decision: Dict[str, Any],
    ) -> Tuple[OBGateAction, float, str]:
        """根据分析结果和决策方向，应用门控规则

        Returns:
            (action, multiplier, reason)
        """
        action_str = str(decision.get("action", "")).lower()
        is_long = "buy" in action_str or "long" in action_str
        is_short = "sell" in action_str or "short" in action_str

        # 规则1: 极端价差 → 极度缩减
        if analysis.spread_pct >= self.SPREAD_PCT_EXTREME:
            return (
                OBGateAction.REDUCE,
                self.EXTREME_REDUCE_MULTIPLIER,
                f"extreme_spread_pct={analysis.spread_pct:.5f}",
            )

        # 规则2: 高价差 → 缩减
        if analysis.spread_pct >= self.SPREAD_PCT_HIGH:
            return (
                OBGateAction.REDUCE,
                self.REDUCE_MULTIPLIER,
                f"high_spread_pct={analysis.spread_pct:.5f}",
            )

        # 规则3-6: 大单墙检测
        if analysis.bid_wall is not None and is_long:
            # 买盘墙 + 做多 → BOOST
            return (
                OBGateAction.BOOST,
                self.BOOST_MULTIPLIER,
                f"bid_wall_ratio={analysis.bid_wall_ratio:.2f}",
            )

        if analysis.bid_wall is not None and is_short:
            # 买盘墙 + 做空 → REDUCE
            return (
                OBGateAction.REDUCE,
                self.REDUCE_MULTIPLIER,
                f"bid_wall_blocks_short_ratio={analysis.bid_wall_ratio:.2f}",
            )

        if analysis.ask_wall is not None and is_short:
            # 卖盘墙 + 做空 → BOOST
            return (
                OBGateAction.BOOST,
                self.BOOST_MULTIPLIER,
                f"ask_wall_ratio={analysis.ask_wall_ratio:.2f}",
            )

        if analysis.ask_wall is not None and is_long:
            # 卖盘墙 + 做多 → REDUCE
            return (
                OBGateAction.REDUCE,
                self.REDUCE_MULTIPLIER,
                f"ask_wall_blocks_long_ratio={analysis.ask_wall_ratio:.2f}",
            )

        # 规则7-10: 强不平衡检测
        if analysis.imbalance >= self.IMBALANCE_STRONG:
            if is_long:
                return (
                    OBGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"bid_imbalance={analysis.imbalance:.3f}",
                )
            if is_short:
                return (
                    OBGateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"bid_imbalance_blocks_short={analysis.imbalance:.3f}",
                )

        if analysis.imbalance <= -self.IMBALANCE_STRONG:
            if is_short:
                return (
                    OBGateAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"ask_imbalance={analysis.imbalance:.3f}",
                )
            if is_long:
                return (
                    OBGateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"ask_imbalance_blocks_long={analysis.imbalance:.3f}",
                )

        return OBGateAction.ALLOW, 1.0, "no_trigger"

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------

    def generate_synthetic_order_book(
        self,
        mid_price: float,
        spread_pct: float = 0.0005,
        depth_levels: int = 10,
        base_qty: float = 1.0,
        imbalance: float = 0.0,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """生成模拟订单簿 (用于沙盘测试)

        Args:
            mid_price: 中间价
            spread_pct: 价差百分比
            depth_levels: 深度层数
            base_qty: 基础挂单量
            imbalance: 不平衡度 [-1, 1], 正=买盘多, 负=卖盘多

        Returns:
            (bids, asks)
        """
        if mid_price <= 0:
            return [], []

        half_spread = mid_price * spread_pct / 2.0
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread

        # 根据不平衡度调整买卖盘基础量
        bid_factor = 1.0 + imbalance
        ask_factor = 1.0 - imbalance

        bids: List[Tuple[float, float]] = []
        asks: List[Tuple[float, float]] = []

        tick = mid_price * 0.0001  # 1 bps

        for i in range(depth_levels):
            bid_price = best_bid - i * tick
            ask_price = best_ask + i * tick
            bid_qty = base_qty * bid_factor * (1.0 + 0.1 * i)
            ask_qty = base_qty * ask_factor * (1.0 + 0.1 * i)
            bids.append((bid_price, bid_qty))
            asks.append((ask_price, ask_qty))

        return bids, asks

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
        }

    def reset(self):
        """重置统计"""
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis = None
        logger.info("OrderBookGate stats reset")
