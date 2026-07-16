#!/usr/bin/env python3
"""SOPR 决策门控 — Layer 8 质变数据模块3

基于 SOPR (Spent Output Profit Ratio) 对策略决策进行门控。

门控逻辑:
  - SOPR > 1.1 + 做多 → BOOST (盈利情绪持续, 趋势向上)
  - SOPR < 0.9 + 做空 → BOOST (恐慌抛售, 趋势向下)
  - SOPR > 1.1 + 做空 → REDUCE (盈利情绪对抗做空)
  - SOPR < 0.9 + 做多 → REDUCE (恐慌情绪对抗做多)
  - SOPR 极端值 (>1.3 或 <0.7) → FORCE (调整止盈止损)
  - SOPR 趋势上升 + 做多 → 额外 BOOST
  - SOPR 趋势下降 + 做空 → 额外 BOOST

用法:
    from sandbox_trading.sopr_gate import SOPRGate, SOPRGateAction
    gate = SOPRGate(enabled=True)
    result = gate.apply_to_decision(decision, symbol="BTC")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.sopr_gate")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class SOPRGateAction(Enum):
    """SOPR 门控动作"""
    ALLOW = "allow"        # 放行, 不调整
    BOOST = "boost"        # 加仓 (×1.1)
    REDUCE = "reduce"      # 缩仓 (×0.7)
    FORCE = "force"        # 强制调整止盈止损
    NEUTRAL = "neutral"    # 中性 (无数据或门控关闭)


@dataclass(slots=True)
class SOPRGateResult:
    """SOPR 门控结果"""
    action: SOPRGateAction = SOPRGateAction.NEUTRAL
    reason: str = ""
    original_qty: float = 0.0
    adjusted_qty: float = 0.0
    sopr_value: float = 0.0
    a_sopr_value: float = 0.0
    lth_sopr: float = 0.0
    sth_sopr: float = 0.0
    sopr_trend: float = 0.0
    is_extreme: bool = False
    applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "original_qty": self.original_qty,
            "adjusted_qty": self.adjusted_qty,
            "sopr_value": self.sopr_value,
            "a_sopr_value": self.a_sopr_value,
            "lth_sopr": self.lth_sopr,
            "sth_sopr": self.sth_sopr,
            "sopr_trend": self.sopr_trend,
            "is_extreme": self.is_extreme,
            "applied": self.applied,
        }


# ============================================================================
# SOPR 门控
# ============================================================================

class SOPRGate:
    """SOPR 决策门控

    基于 SOPR 值及其变体 (aSOPR, LTH-SOPR, STH-SOPR) 对策略决策进行门控。

    门控层级:
      1. 极端值检查: SOPR > 1.3 或 < 0.7 → FORCE
      2. 趋势检查: SOPR 上升/下降 + 策略方向一致 → BOOST
      3. 基本检查: SOPR > 1.1 / < 0.9 + 策略方向 → BOOST/REDUCE
    """

    # SOPR 阈值
    PROFIT_THRESHOLD = 1.1        # SOPR > 1.1 视为盈利情绪
    LOSS_THRESHOLD = 0.9          # SOPR < 0.9 视为亏损情绪
    EXTREME_PROFIT = 1.3          # SOPR > 1.3 视为极端盈利
    EXTREME_LOSS = 0.7            # SOPR < 0.7 视为极端亏损

    # 数量调整倍数
    BOOST_MULTIPLIER = 1.1
    REDUCE_MULTIPLIER = 0.7

    # 趋势增强倍数 (趋势一致时额外加仓)
    TREND_BOOST_BONUS = 0.05      # 额外 +5%

    # 数据目录 (WSL 路径)
    DATA_DIR = "/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/sopr"

    def __init__(self, enabled: bool = True):
        """初始化 SOPR 门控

        Args:
            enabled: 是否启用门控
        """
        self.enabled = enabled
        self._cache: Dict[str, Dict[str, Any]] = {}  # symbol -> latest sopr data
        self._stats = {
            "total_calls": 0,
            "boost": 0,
            "reduce": 0,
            "force": 0,
            "allow": 0,
            "neutral": 0,
        }

        if enabled:
            self._load_data()

    def _load_data(self, symbol: Optional[str] = None):
        """加载 SOPR 数据

        Args:
            symbol: 指定币种 (None=加载全部)
        """
        symbols = [symbol] if symbol else ["BTC", "ETH"]

        for sym in symbols:
            try:
                # 查找最新的数据文件
                data_dir = Path(self.DATA_DIR)
                if not data_dir.exists():
                    logger.warning("SOPR data dir not found: %s", self.DATA_DIR)
                    continue

                # 查找 {SYMBOL}_*.jsonl 文件
                pattern = f"{sym}_*.jsonl"
                files = sorted(data_dir.glob(pattern))
                if not files:
                    logger.warning("No SOPR data file for %s", sym)
                    continue

                # 读取最新文件
                latest_file = files[-1]
                with open(latest_file, "r") as f:
                    lines = f.readlines()

                if not lines:
                    logger.warning("Empty SOPR data file: %s", latest_file)
                    continue

                # 取最后一行 (最新数据)
                latest_data = json.loads(lines[-1])
                self._cache[sym] = latest_data
                logger.info(
                    "SOPR data loaded: %s sopr=%.4f a_sopr=%.4f lth=%.4f sth=%.4f",
                    sym, latest_data.get("sopr", 0),
                    latest_data.get("a_sopr", 0),
                    latest_data.get("lth_sopr", 0),
                    latest_data.get("sth_sopr", 0),
                )

            except Exception as e:
                logger.warning("Failed to load SOPR data for %s: %s", sym, e)

    def get_sopr(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取缓存的 SOPR 数据

        支持多种symbol格式: "BTC", "btc", "BTC-USDT", "BTC/USD"

        Args:
            symbol: 币种 (可能带后缀如 -USDT)

        Returns:
            dict or None: SOPR 数据
        """
        sym = symbol.upper()
        # 直接查找
        if sym in self._cache:
            return self._cache[sym]
        # 去掉后缀查找 (BTC-USDT → BTC)
        base = sym.split("-")[0].split("/")[0]
        return self._cache.get(base)

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "BTC",
        **kwargs,
    ) -> Dict[str, Any]:
        """对决策应用 SOPR 门控 (包装方法, 返回 decision 字典)

        修改 decision 字典并返回, 设置以下字段:
          - sopr_gate_action: "boost"/"reduce"/"force"/"allow"/"neutral"
          - sopr_value: SOPR 值
          - sopr_reason: 调整原因
          - quantity: 调整后的数量

        Args:
            decision: 策略决策, 需包含:
                - action: "long" / "short" / "hold"
                - quantity: 数量
            symbol: 交易对
            **kwargs: 兼容额外参数 (agent_id, current_price 等)

        Returns:
            Dict: 修改后的 decision 字典
        """
        result = self._apply_sopr_gate(decision, symbol)

        # 修改 decision 字典
        decision["sopr_gate_action"] = result.action.value
        decision["sopr_value"] = result.sopr_value
        decision["sopr_reason"] = result.reason
        if result.adjusted_qty > 0:
            decision["quantity"] = result.adjusted_qty

        return decision

    def _apply_sopr_gate(
        self,
        decision: Dict[str, Any],
        symbol: str = "BTC",
    ) -> SOPRGateResult:
        """对决策应用 SOPR 门控 (内部方法, 返回 SOPRGateResult)

        Args:
            decision: 策略决策
            symbol: 交易对

        Returns:
            SOPRGateResult
        """
        self._stats["total_calls"] += 1

        result = SOPRGateResult()

        if not self.enabled:
            result.action = SOPRGateAction.NEUTRAL
            result.reason = "gate disabled"
            self._stats["neutral"] += 1
            return result

        # 获取 SOPR 数据
        sopr_data = self.get_sopr(symbol)
        if not sopr_data:
            result.action = SOPRGateAction.NEUTRAL
            result.reason = f"no SOPR data for {symbol}"
            self._stats["neutral"] += 1
            return result

        sopr = float(sopr_data.get("sopr", 1.0))
        a_sopr = float(sopr_data.get("a_sopr", sopr))
        lth_sopr = float(sopr_data.get("lth_sopr", sopr))
        sth_sopr = float(sopr_data.get("sth_sopr", sopr))
        sopr_trend = float(sopr_data.get("sopr_trend", 0.0))
        is_extreme_profit = bool(sopr_data.get("is_extreme_profit", False))
        is_extreme_loss = bool(sopr_data.get("is_extreme_loss", False))

        result.sopr_value = sopr
        result.a_sopr_value = a_sopr
        result.lth_sopr = lth_sopr
        result.sth_sopr = sth_sopr
        result.sopr_trend = sopr_trend
        result.is_extreme = is_extreme_profit or is_extreme_loss

        action_type = str(decision.get("action", "hold")).lower()
        original_qty = float(decision.get("quantity", 0))
        result.original_qty = original_qty

        if action_type == "hold" or original_qty <= 0:
            result.action = SOPRGateAction.ALLOW
            result.reason = "hold or zero qty"
            result.adjusted_qty = original_qty
            result.applied = True
            self._stats["allow"] += 1
            return result

        # 层级1: 极端值检查 → FORCE
        if is_extreme_profit or is_extreme_loss:
            result.action = SOPRGateAction.FORCE
            if is_extreme_profit:
                result.reason = f"extreme profit SOPR={sopr:.3f}>{self.EXTREME_PROFIT}"
            else:
                result.reason = f"extreme loss SOPR={sopr:.3f}<{self.EXTREME_LOSS}"

            # 极端情绪: 调整止盈止损 (通过调整数量体现)
            # 极端盈利 + 做多 → 获利了结 (缩仓)
            # 极端盈利 + 做空 → 风险增加 (缩仓)
            # 极端亏损 + 做多 → 抄底机会 (加仓)
            # 极端亏损 + 做空 → 恐慌持续 (加仓)
            if is_extreme_profit:
                result.adjusted_qty = original_qty * self.REDUCE_MULTIPLIER
            else:  # is_extreme_loss
                result.adjusted_qty = original_qty * self.BOOST_MULTIPLIER

            result.applied = True
            self._stats["force"] += 1
            logger.info(
                "SOPRGate FORCE: action=%s sopr=%.3f qty=%.6f→%.6f reason=%s",
                action_type, sopr, original_qty, result.adjusted_qty, result.reason,
            )
            return result

        # 层级2: 趋势增强检查
        trend_bonus = 0.0
        if sopr_trend > 0 and action_type == "long":
            # SOPR 上升 + 做多 → 趋势一致, 额外加仓
            trend_bonus = self.TREND_BOOST_BONUS
        elif sopr_trend < 0 and action_type == "short":
            # SOPR 下降 + 做空 → 趋势一致, 额外加仓
            trend_bonus = self.TREND_BOOST_BONUS

        # 层级3: 基本 SOPR 门控
        is_long = (action_type == "long")
        is_short = (action_type == "short")

        if sopr > self.PROFIT_THRESHOLD:
            # 盈利情绪
            if is_long:
                # 做多 + 盈利情绪 → BOOST
                multiplier = self.BOOST_MULTIPLIER + trend_bonus
                result.action = SOPRGateAction.BOOST
                result.reason = f"profit SOPR={sopr:.3f}>{self.PROFIT_THRESHOLD} + long"
                result.adjusted_qty = original_qty * multiplier
                result.applied = True
                self._stats["boost"] += 1
            elif is_short:
                # 做空 + 盈利情绪 → REDUCE
                result.action = SOPRGateAction.REDUCE
                result.reason = f"profit SOPR={sopr:.3f}>{self.PROFIT_THRESHOLD} vs short"
                result.adjusted_qty = original_qty * self.REDUCE_MULTIPLIER
                result.applied = True
                self._stats["reduce"] += 1
            else:
                result.action = SOPRGateAction.ALLOW
                result.reason = "unknown action"
                result.adjusted_qty = original_qty
                self._stats["allow"] += 1

        elif sopr < self.LOSS_THRESHOLD:
            # 亏损情绪
            if is_short:
                # 做空 + 亏损情绪 → BOOST
                multiplier = self.BOOST_MULTIPLIER + trend_bonus
                result.action = SOPRGateAction.BOOST
                result.reason = f"loss SOPR={sopr:.3f}<{self.LOSS_THRESHOLD} + short"
                result.adjusted_qty = original_qty * multiplier
                result.applied = True
                self._stats["boost"] += 1
            elif is_long:
                # 做多 + 亏损情绪 → REDUCE
                result.action = SOPRGateAction.REDUCE
                result.reason = f"loss SOPR={sopr:.3f}<{self.LOSS_THRESHOLD} vs long"
                result.adjusted_qty = original_qty * self.REDUCE_MULTIPLIER
                result.applied = True
                self._stats["reduce"] += 1
            else:
                result.action = SOPRGateAction.ALLOW
                result.reason = "unknown action"
                result.adjusted_qty = original_qty
                self._stats["allow"] += 1

        else:
            # SOPR 在 [0.9, 1.1] 之间, 中性区间
            result.action = SOPRGateAction.ALLOW
            result.reason = f"neutral SOPR={sopr:.3f} in [{self.LOSS_THRESHOLD}, {self.PROFIT_THRESHOLD}]"
            result.adjusted_qty = original_qty
            result.applied = True
            self._stats["allow"] += 1

        if result.applied and result.adjusted_qty != original_qty:
            logger.info(
                "SOPRGate %s: action=%s sopr=%.3f qty=%.6f→%.6f reason=%s",
                result.action.value.upper(), action_type, sopr,
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
