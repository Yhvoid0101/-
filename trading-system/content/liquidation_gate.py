#!/usr/bin/env python3
"""清算热力图决策门控 — Layer 8 质变数据模块2

将清算热力图接入交易决策，根据清算密集区和级联风险调整策略。

门控逻辑:
  1. 做多 + 上方有空头清算密集区 → BOOST (空头清算推动价格上涨)
  2. 做多 + 下方有多头清算密集区 → REDUCE (多头清算可能推动价格下跌)
  3. 做空 + 下方有多头清算密集区 → BOOST (多头清算推动价格下跌)
  4. 做空 + 上方有空头清算密集区 → REDUCE (空头清算可能推动价格上涨)
  5. 清算级联风险高 → REDUCE (避免被级联波及)
  6. 无清算数据 → 透传 (ALLOW)

用法:
    from sandbox_trading.liquidation_gate import LiquidationGate
    gate = LiquidationGate(enabled=True)
    gate.load_data(symbol)
    decision = gate.apply_to_decision(decision, current_price, agent_id, symbol)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("hermes.liq_gate")


class LiqGateAction(Enum):
    """清算门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    FORCE = "force"      # 强制调整止盈到清算密集区
    NEUTRAL = "neutral"


@dataclass(slots=True)
class LiqGateResult:
    """清算门控结果"""
    action: LiqGateAction = LiqGateAction.NEUTRAL
    quantity_multiplier: float = 1.0
    target_price: float = 0.0
    cascade_risk: float = 0.0
    reason: str = ""


class LiquidationGate:
    """清算热力图决策门控"""

    # 数据目录
    LIQ_DATA_DIR_WSL = "/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/liquidations"
    LIQ_DATA_DIR_WIN = r"D:\hermes\hermes_obsidian_vault\Hermes\MarketData\liquidations"

    # 距离阈值
    CLOSE_DISTANCE_PCT = 0.03    # 3%以内视为近距离
    FAR_DISTANCE_PCT = 0.15      # 15%以外视为无效

    # 数量调整因子
    BOOST_MULTIPLIER = 1.1
    REDUCE_MULTIPLIER = 0.7

    # 级联风险阈值
    HIGH_CASCADE_RISK = 0.7      # 级联风险>0.7时强制缩仓

    # 清算密集区阈值 (USD)
    SIGNIFICANT_LIQ_AMOUNT = 1e5  # 10万美元以上视为显著

    def __init__(self, enabled: bool = True, data_dir: Optional[str] = None):
        self.enabled = enabled
        self.data_dir = data_dir or self.LIQ_DATA_DIR_WSL
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_count = 0
        self._apply_count = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._force_count = 0

    def load_data(self, symbol: str, date_range: Optional[Tuple[str, str]] = None) -> int:
        """加载清算数据"""
        if not self.enabled:
            return 0

        try:
            sym = symbol.split("-")[0].upper()
            data_path = Path(self.data_dir)
            if not data_path.exists():
                logger.debug("Liq data dir not found: %s", self.data_dir)
                return 0

            pattern = f"{sym}_*.jsonl"
            files = sorted(data_path.glob(pattern))
            if not files:
                logger.debug("No liq data files for %s", sym)
                return 0

            records = []
            for f in files:
                try:
                    fname = f.stem
                    parts = fname.split("_")
                    if len(parts) >= 2:
                        file_date = parts[-1]
                        if date_range:
                            if not (date_range[0] <= file_date <= date_range[1]):
                                continue

                    with open(f, 'r') as fp:
                        for line in fp:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                                records.append(rec)
                            except json.JSONDecodeError:
                                continue
                except (IOError, OSError) as e:
                    logger.warning("Failed to read liq file %s: %s", f, e)
                    continue

            if not records:
                return 0

            records.sort(key=lambda r: r.get("timestamp", 0))
            latest = records[-1]

            self._cache[sym] = {
                "latest": latest,
                "all_records": records,
                "n_records": len(records),
            }
            self._load_count += 1

            logger.info(
                "Liq data loaded: symbol=%s records=%d total_liq=%.2f "
                "long_cluster=%.2f short_cluster=%.2f cascade=%.2f",
                sym, len(records),
                latest.get("total_liq", 0),
                latest.get("long_liq_cluster", 0),
                latest.get("short_liq_cluster", 0),
                latest.get("cascade_risk", 0),
            )
            return len(records)

        except Exception as e:
            logger.warning("Liq data load failed for %s: %s", symbol, e)
            return 0

    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新清算数据"""
        sym = symbol.split("-")[0].upper()
        cached = self._cache.get(sym)
        if cached:
            return cached.get("latest")
        return None

    def apply_to_decision(
        self,
        decision: Optional[Dict[str, Any]],
        current_price: float,
        agent_id: str = "",
        symbol: str = "",
    ) -> Optional[Dict[str, Any]]:
        """对决策应用清算门控"""
        if not self.enabled or decision is None:
            return decision

        action = decision.get("action", "neutral")
        if action == "neutral":
            return decision

        liq_data = self.get_latest(symbol)
        if not liq_data:
            return decision

        self._apply_count += 1

        # 提取清算数据
        long_cluster = float(liq_data.get("long_liq_cluster", 0))
        short_cluster = float(liq_data.get("short_liq_cluster", 0))
        long_amount = float(liq_data.get("long_cluster_amount", 0))
        short_amount = float(liq_data.get("short_cluster_amount", 0))
        cascade_risk = float(liq_data.get("cascade_risk", 0))

        # 级联风险高 → 强制缩仓
        if cascade_risk >= self.HIGH_CASCADE_RISK:
            orig_qty = decision.get("quantity", 0)
            decision["quantity"] = orig_qty * self.REDUCE_MULTIPLIER
            decision["liq_gate_action"] = "reduce"
            decision["liq_cascade_risk"] = cascade_risk
            decision["liq_reason"] = f"high cascade risk={cascade_risk:.2f}"

            self._reduce_count += 1
            if self._reduce_count <= 5 or self._reduce_count % 100 == 0:
                logger.info(
                    "LiqGate REDUCE(cascade) #%d: agent=%s symbol=%s cascade=%.2f qty=%.6f→%.6f",
                    self._reduce_count, agent_id[:12], symbol,
                    cascade_risk, orig_qty, decision["quantity"],
                )
            return decision

        # 分析清算密集区
        strategy_direction = "up" if action == "long" else "down"
        orig_qty = decision.get("quantity", 0)

        # 计算距离
        long_dist = abs(long_cluster - current_price) / current_price if long_cluster > 0 else 999
        short_dist = abs(short_cluster - current_price) / current_price if short_cluster > 0 else 999

        # 判断哪个清算密集区更近且显著
        long_significant = long_cluster > 0 and long_amount >= self.SIGNIFICANT_LIQ_AMOUNT and long_dist <= self.FAR_DISTANCE_PCT
        short_significant = short_cluster > 0 and short_amount >= self.SIGNIFICANT_LIQ_AMOUNT and short_dist <= self.FAR_DISTANCE_PCT

        if not long_significant and not short_significant:
            return decision

        # 确定主要清算磁吸方向
        # 多头清算在下方 → 价格被吸引向下
        # 空头清算在上方 → 价格被吸引向上

        if action == "long":
            # 做多
            if short_significant and (not long_significant or short_dist < long_dist):
                # 上方有空头清算密集区 → 空头清算推动价格上涨 → BOOST
                decision["quantity"] = orig_qty * self.BOOST_MULTIPLIER
                decision["liq_gate_action"] = "boost"
                decision["liq_target_price"] = short_cluster
                decision["liq_reason"] = f"short liq cluster above at {short_cluster:.2f}"

                self._boost_count += 1
                if self._boost_count <= 5 or self._boost_count % 100 == 0:
                    logger.info(
                        "LiqGate BOOST #%d: agent=%s symbol=%s action=long target=%.2f dist=%.2f%% qty=%.6f→%.6f",
                        self._boost_count, agent_id[:12], symbol,
                        short_cluster, short_dist * 100, orig_qty, decision["quantity"],
                    )

            elif long_significant:
                # 下方有多头清算密集区 → 多头清算可能推动价格下跌 → REDUCE
                decision["quantity"] = orig_qty * self.REDUCE_MULTIPLIER
                decision["liq_gate_action"] = "reduce"
                decision["liq_target_price"] = long_cluster
                decision["liq_reason"] = f"long liq cluster below at {long_cluster:.2f}"

                self._reduce_count += 1
                if self._reduce_count <= 5 or self._reduce_count % 100 == 0:
                    logger.info(
                        "LiqGate REDUCE #%d: agent=%s symbol=%s action=long target=%.2f dist=%.2f%% qty=%.6f→%.6f",
                        self._reduce_count, agent_id[:12], symbol,
                        long_cluster, long_dist * 100, orig_qty, decision["quantity"],
                    )

        elif action == "short":
            # 做空
            if long_significant and (not short_significant or long_dist < short_dist):
                # 下方有多头清算密集区 → 多头清算推动价格下跌 → BOOST
                decision["quantity"] = orig_qty * self.BOOST_MULTIPLIER
                decision["liq_gate_action"] = "boost"
                decision["liq_target_price"] = long_cluster
                decision["liq_reason"] = f"long liq cluster below at {long_cluster:.2f}"

                self._boost_count += 1
                if self._boost_count <= 5 or self._boost_count % 100 == 0:
                    logger.info(
                        "LiqGate BOOST #%d: agent=%s symbol=%s action=short target=%.2f dist=%.2f%% qty=%.6f→%.6f",
                        self._boost_count, agent_id[:12], symbol,
                        long_cluster, long_dist * 100, orig_qty, decision["quantity"],
                    )

            elif short_significant:
                # 上方有空头清算密集区 → 空头清算可能推动价格上涨 → REDUCE
                decision["quantity"] = orig_qty * self.REDUCE_MULTIPLIER
                decision["liq_gate_action"] = "reduce"
                decision["liq_target_price"] = short_cluster
                decision["liq_reason"] = f"short liq cluster above at {short_cluster:.2f}"

                self._reduce_count += 1
                if self._reduce_count <= 5 or self._reduce_count % 100 == 0:
                    logger.info(
                        "LiqGate REDUCE #%d: agent=%s symbol=%s action=short target=%.2f dist=%.2f%% qty=%.6f→%.6f",
                        self._reduce_count, agent_id[:12], symbol,
                        short_cluster, short_dist * 100, orig_qty, decision["quantity"],
                    )

        # 近距离清算密集区 → 调整止盈
        if decision.get("liq_gate_action") in ("boost", "reduce"):
            target = decision.get("liq_target_price", 0)
            if target > 0 and target != current_price:
                if action == "long" and target > current_price:
                    current_tp = decision.get("take_profit", 0)
                    if current_tp <= 0 or target > current_tp:
                        decision["take_profit"] = target
                elif action == "short" and target < current_price:
                    current_tp = decision.get("take_profit", 0)
                    if current_tp <= 0 or target < current_tp:
                        decision["take_profit"] = target

        return decision

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "load_count": self._load_count,
            "apply_count": self._apply_count,
            "boost_count": self._boost_count,
            "reduce_count": self._reduce_count,
            "force_count": self._force_count,
            "cached_symbols": len(self._cache),
        }
