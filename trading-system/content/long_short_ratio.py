#!/usr/bin/env python3
"""多空持仓比模块 — Layer 8 行为数据 P0

Long/Short Ratio (多空持仓比) 揭示市场参与者的持仓意图。
与OI(规模)、资金费率(成本)组成"情绪三件套"，判断市场拥挤度。

数据来源:
  - Gate.io futures tickers: total_size (持仓量), funding_rate (资金费率)
  - 基于价格变化+OI变化+资金费率推算多空比 (无公开API时的降级方案)

推算逻辑:
  - 资金费率 > 0 → 多头支付 → 多头拥挤 → long_ratio > 0.5
  - 资金费率 < 0 → 空头支付 → 空头拥挤 → long_ratio < 0.5
  - |资金费率| 越大 → 拥挤程度越高 → long_ratio 偏离0.5越多
  - 价格上涨 + OI增加 → 多头加仓 → long_ratio 上升
  - 价格下跌 + OI增加 → 空头加仓 → long_ratio 下降

门控逻辑:
  - long_ratio > 0.7 (多头拥挤) + 做多 → REDUCE (拥挤反转风险)
  - long_ratio > 0.7 + 做空 → BOOST (拥挤反转做空)
  - long_ratio < 0.3 (空头拥挤) + 做空 → REDUCE
  - long_ratio < 0.3 + 做多 → BOOST
  - long_ratio 在 [0.4, 0.6] → ALLOW (均衡)

用法:
    from sandbox_trading.long_short_ratio import LongShortRatioGate
    gate = LongShortRatioGate(enabled=True)
    result = gate.apply_to_decision(decision, symbol="BTC")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# 预取数据读取器 (可选, 失败时降级到原有逻辑)
try:
    from .prefetched_data_reader import PrefetchedDataReader
except ImportError:
    try:
        from prefetched_data_reader import PrefetchedDataReader
    except ImportError:
        PrefetchedDataReader = None

logger = logging.getLogger("hermes.long_short_ratio")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class LSRGateAction(Enum):
    """多空比门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class LongShortRatioResult:
    """多空比计算结果"""
    symbol: str = ""
    timestamp: float = 0.0
    long_ratio: float = 0.5        # 多头占比 (0-1)
    short_ratio: float = 0.5       # 空头占比 (0-1)
    long_short_ratio: float = 1.0  # 多空比 (long/short)
    is_long_crowded: bool = False  # 多头拥挤
    is_short_crowded: bool = False # 空头拥挤
    funding_rate: float = 0.0
    oi_total: float = 0.0
    price: float = 0.0
    crowding_score: float = 0.0    # 拥挤度评分 (-1到1, 正=多头拥挤, 负=空头拥挤)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "long_ratio": self.long_ratio,
            "short_ratio": self.short_ratio,
            "long_short_ratio": self.long_short_ratio,
            "is_long_crowded": self.is_long_crowded,
            "is_short_crowded": self.is_short_crowded,
            "funding_rate": self.funding_rate,
            "oi_total": self.oi_total,
            "price": self.price,
            "crowding_score": self.crowding_score,
        }


@dataclass(slots=True)
class LSRGateResult:
    """多空比门控结果"""
    action: LSRGateAction = LSRGateAction.NEUTRAL
    reason: str = ""
    original_qty: float = 0.0
    adjusted_qty: float = 0.0
    long_ratio: float = 0.5
    crowding_score: float = 0.0
    applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "original_qty": self.original_qty,
            "adjusted_qty": self.adjusted_qty,
            "long_ratio": self.long_ratio,
            "crowding_score": self.crowding_score,
            "applied": self.applied,
        }


# ============================================================================
# 多空比计算器
# ============================================================================

class LongShortRatioCalculator:
    """多空比计算器 — 基于资金费率+OI+价格推算

    当无公开多空比API时, 使用以下公式推算:
      long_ratio = 0.5 + funding_rate * SENSITIVITY + price_momentum * MOMENTUM_WEIGHT
      long_ratio = clamp(long_ratio, 0.05, 0.95)
    """

    # 资金费率敏感度 (1%费率 → 偏离0.5约0.2)
    FUNDING_SENSITIVITY = 20.0

    # 价格动量权重 (1%涨幅 → 偏离0.5约0.05)
    MOMENTUM_WEIGHT = 5.0

    # 拥挤阈值
    LONG_CROWDED_THRESHOLD = 0.7    # long_ratio > 0.7 → 多头拥挤
    SHORT_CROWDED_THRESHOLD = 0.3   # long_ratio < 0.3 → 空头拥挤

    def __init__(self):
        self._price_history: Dict[str, list] = {}
        self._oi_history: Dict[str, list] = {}

    def calculate(
        self,
        symbol: str,
        funding_rate: float,
        oi_total: float,
        current_price: float,
        timestamp: Optional[float] = None,
    ) -> LongShortRatioResult:
        """计算多空比

        Args:
            symbol: 交易对
            funding_rate: 当前资金费率
            oi_total: 当前持仓总量(USD)
            current_price: 当前价格
            timestamp: 时间戳

        Returns:
            LongShortRatioResult
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        result = LongShortRatioResult(
            symbol=symbol,
            timestamp=timestamp,
            funding_rate=funding_rate,
            oi_total=oi_total,
            price=current_price,
        )

        # 基础多空比: 0.5 + 资金费率影响
        long_ratio = 0.5 + funding_rate * self.FUNDING_SENSITIVITY

        # 价格动量影响
        price_momentum = self._calc_price_momentum(symbol, current_price)
        long_ratio += price_momentum * self.MOMENTUM_WEIGHT

        # 限制范围
        long_ratio = max(0.05, min(0.95, long_ratio))

        result.long_ratio = long_ratio
        result.short_ratio = 1.0 - long_ratio
        result.long_short_ratio = long_ratio / max(0.001, result.short_ratio)
        result.is_long_crowded = long_ratio > self.LONG_CROWDED_THRESHOLD
        result.is_short_crowded = long_ratio < self.SHORT_CROWDED_THRESHOLD
        result.crowding_score = (long_ratio - 0.5) * 2.0  # -1 到 1

        # 更新历史
        self._price_history.setdefault(symbol, []).append(current_price)
        if len(self._price_history[symbol]) > 100:
            self._price_history[symbol].pop(0)
        self._oi_history.setdefault(symbol, []).append(oi_total)
        if len(self._oi_history[symbol]) > 100:
            self._oi_history[symbol].pop(0)

        logger.info(
            "LSR calc: %s long_ratio=%.3f short_ratio=%.3f L/S=%.2f funding=%.6f crowded=%s/%s score=%.3f",
            symbol, result.long_ratio, result.short_ratio,
            result.long_short_ratio, funding_rate,
            result.is_long_crowded, result.is_short_crowded,
            result.crowding_score,
        )

        return result

    def _calc_price_momentum(self, symbol: str, current_price: float) -> float:
        """计算价格动量 (最近N周期收益率)"""
        history = self._price_history.get(symbol, [])
        if len(history) < 2:
            return 0.0
        # 取最近5个价格的平均作为基准
        lookback = min(5, len(history))
        ref_price = sum(history[-lookback:]) / lookback
        if ref_price <= 0:
            return 0.0
        return (current_price - ref_price) / ref_price


# ============================================================================
# 多空比门控
# ============================================================================

class LongShortRatioGate:
    """多空比决策门控

    基于多空比拥挤度对策略决策进行门控。
    """

    # 门控阈值
    LONG_CROWDED_THRESHOLD = 0.7    # 多头拥挤
    SHORT_CROWDED_THRESHOLD = 0.3   # 空头拥挤
    EXTREME_CROWDED_THRESHOLD = 0.85  # 极端拥挤

    # 数量调整倍数
    BOOST_MULTIPLIER = 1.1
    REDUCE_MULTIPLIER = 0.7
    EXTREME_REDUCE_MULTIPLIER = 0.5

    # 数据目录
    DATA_DIR = "/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/long_short_ratio"

    def __init__(self, enabled: bool = True):
        """初始化多空比门控"""
        self.enabled = enabled
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._calculator = LongShortRatioCalculator()
        self._stats = {
            "total_calls": 0,
            "boost": 0,
            "reduce": 0,
            "allow": 0,
            "neutral": 0,
        }

        if enabled:
            self._load_data()

    def _load_data(self, symbol: Optional[str] = None):
        """加载多空比数据"""
        symbols = [symbol] if symbol else ["BTC", "ETH"]

        for sym in symbols:
            try:
                data_dir = Path(self.DATA_DIR)
                if not data_dir.exists():
                    logger.warning("LSR data dir not found: %s", self.DATA_DIR)
                    continue

                pattern = f"{sym}_*.jsonl"
                files = sorted(data_dir.glob(pattern))
                if not files:
                    logger.warning("No LSR data file for %s", sym)
                    continue

                latest_file = files[-1]
                with open(latest_file, "r") as f:
                    lines = f.readlines()

                if not lines:
                    logger.warning("Empty LSR data file: %s", latest_file)
                    continue

                latest_data = json.loads(lines[-1])
                self._cache[sym] = latest_data
                logger.info(
                    "LSR data loaded: %s long_ratio=%.3f short_ratio=%.3f crowded=%s/%s",
                    sym, latest_data.get("long_ratio", 0.5),
                    latest_data.get("short_ratio", 0.5),
                    latest_data.get("is_long_crowded", False),
                    latest_data.get("is_short_crowded", False),
                )

            except Exception as e:
                logger.warning("Failed to load LSR data for %s: %s", sym, e)

    def get_ratio(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取多空比数据 (支持多种symbol格式)

        优先级:
          1. 预取数据 (auto_data_fetcher_service 抓取的 Gate.io 数据)
          2. 本地缓存 (历史数据文件)
        """
        # 1. 优先读取预取数据
        if PrefetchedDataReader is not None:
            try:
                cached = PrefetchedDataReader.read_lsr(symbol)
                if cached is not None:
                    # 预取数据可能是列表 (兼容 Binance 格式) 或字典
                    if isinstance(cached, list) and len(cached) > 0:
                        cached = cached[0]
                    if isinstance(cached, dict) and "long_ratio" in cached:
                        logger.debug("Using prefetched LSR for %s", symbol)
                        return cached
            except Exception as e:
                logger.debug("PrefetchedDataReader LSR failed for %s: %s", symbol, e)

        # 2. 本地缓存
        sym = symbol.upper()
        if sym in self._cache:
            return self._cache[sym]
        base = sym.split("-")[0].split("/")[0].split("_")[0]
        return self._cache.get(base)

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "BTC",
        **kwargs,
    ) -> Dict[str, Any]:
        """对决策应用多空比门控 (返回decision字典)"""
        result = self._apply_gate(decision, symbol)

        decision["lsr_gate_action"] = result.action.value
        decision["lsr_long_ratio"] = result.long_ratio
        decision["lsr_crowding_score"] = result.crowding_score
        decision["lsr_reason"] = result.reason
        if result.adjusted_qty > 0:
            decision["quantity"] = result.adjusted_qty

        return decision

    def _apply_gate(
        self,
        decision: Dict[str, Any],
        symbol: str = "BTC",
    ) -> LSRGateResult:
        """内部门控逻辑"""
        self._stats["total_calls"] += 1
        result = LSRGateResult()

        if not self.enabled:
            result.action = LSRGateAction.NEUTRAL
            result.reason = "gate disabled"
            self._stats["neutral"] += 1
            return result

        lsr_data = self.get_ratio(symbol)
        if not lsr_data:
            result.action = LSRGateAction.NEUTRAL
            result.reason = f"no LSR data for {symbol}"
            self._stats["neutral"] += 1
            return result

        long_ratio = float(lsr_data.get("long_ratio", 0.5))
        is_long_crowded = bool(lsr_data.get("is_long_crowded", False))
        is_short_crowded = bool(lsr_data.get("is_short_crowded", False))
        crowding_score = float(lsr_data.get("crowding_score", 0.0))

        result.long_ratio = long_ratio
        result.crowding_score = crowding_score

        action_type = str(decision.get("action", "hold")).lower()
        original_qty = float(decision.get("quantity", 0))
        result.original_qty = original_qty

        if action_type == "hold" or original_qty <= 0:
            result.action = LSRGateAction.ALLOW
            result.reason = "hold or zero qty"
            result.adjusted_qty = original_qty
            result.applied = True
            self._stats["allow"] += 1
            return result

        is_long = (action_type == "long")
        is_short = (action_type == "short")

        # 极端拥挤检查
        is_extreme_long = long_ratio > self.EXTREME_CROWDED_THRESHOLD
        is_extreme_short = long_ratio < (1.0 - self.EXTREME_CROWDED_THRESHOLD)

        if is_extreme_long and is_long:
            # 极端多头拥挤 + 做多 → 大幅缩仓 (反转风险极高)
            result.action = LSRGateAction.REDUCE
            result.reason = f"extreme long crowded L={long_ratio:.3f}>{self.EXTREME_CROWDED_THRESHOLD} vs long"
            result.adjusted_qty = original_qty * self.EXTREME_REDUCE_MULTIPLIER
            result.applied = True
            self._stats["reduce"] += 1
        elif is_extreme_short and is_short:
            # 极端空头拥挤 + 做空 → 大幅缩仓
            result.action = LSRGateAction.REDUCE
            result.reason = f"extreme short crowded L={long_ratio:.3f}<{1.0-self.EXTREME_CROWDED_THRESHOLD:.3f} vs short"
            result.adjusted_qty = original_qty * self.EXTREME_REDUCE_MULTIPLIER
            result.applied = True
            self._stats["reduce"] += 1
        elif is_long_crowded:
            # 多头拥挤
            if is_long:
                # 多头拥挤 + 做多 → 缩仓
                result.action = LSRGateAction.REDUCE
                result.reason = f"long crowded L={long_ratio:.3f}>{self.LONG_CROWDED_THRESHOLD} vs long"
                result.adjusted_qty = original_qty * self.REDUCE_MULTIPLIER
                result.applied = True
                self._stats["reduce"] += 1
            elif is_short:
                # 多头拥挤 + 做空 → 加仓 (拥挤反转)
                result.action = LSRGateAction.BOOST
                result.reason = f"long crowded L={long_ratio:.3f}>{self.LONG_CROWDED_THRESHOLD} + short"
                result.adjusted_qty = original_qty * self.BOOST_MULTIPLIER
                result.applied = True
                self._stats["boost"] += 1
        elif is_short_crowded:
            # 空头拥挤
            if is_short:
                # 空头拥挤 + 做空 → 缩仓
                result.action = LSRGateAction.REDUCE
                result.reason = f"short crowded L={long_ratio:.3f}<{self.SHORT_CROWDED_THRESHOLD} vs short"
                result.adjusted_qty = original_qty * self.REDUCE_MULTIPLIER
                result.applied = True
                self._stats["reduce"] += 1
            elif is_long:
                # 空头拥挤 + 做多 → 加仓 (拥挤反转)
                result.action = LSRGateAction.BOOST
                result.reason = f"short crowded L={long_ratio:.3f}<{self.SHORT_CROWDED_THRESHOLD} + long"
                result.adjusted_qty = original_qty * self.BOOST_MULTIPLIER
                result.applied = True
                self._stats["boost"] += 1
        else:
            # 均衡区间
            result.action = LSRGateAction.ALLOW
            result.reason = f"balanced L={long_ratio:.3f}"
            result.adjusted_qty = original_qty
            result.applied = True
            self._stats["allow"] += 1

        if result.applied and result.adjusted_qty != original_qty:
            logger.info(
                "LSRGate %s: action=%s L=%.3f qty=%.6f→%.6f reason=%s",
                result.action.value.upper(), action_type, long_ratio,
                original_qty, result.adjusted_qty, result.reason,
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取门控统计"""
        return {
            **self._stats,
            "cached_symbols": list(self._cache.keys()),
        }

    def reload(self):
        """重新加载数据"""
        self._cache.clear()
        if self.enabled:
            self._load_data()


# ============================================================================
# 便捷函数
# ============================================================================

def create_synthetic_lsr_data(
    symbol: str,
    long_ratio: float,
    funding_rate: float = 0.0,
    oi_total: float = 0.0,
    price: float = 0.0,
) -> Dict[str, Any]:
    """创建合成多空比数据 (用于测试和降级模式)"""
    short_ratio = 1.0 - long_ratio
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "long_ratio": long_ratio,
        "short_ratio": short_ratio,
        "long_short_ratio": long_ratio / max(0.001, short_ratio),
        "is_long_crowded": long_ratio > 0.7,
        "is_short_crowded": long_ratio < 0.3,
        "funding_rate": funding_rate,
        "oi_total": oi_total,
        "price": price,
        "crowding_score": (long_ratio - 0.5) * 2.0,
    }
