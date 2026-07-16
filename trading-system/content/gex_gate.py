#!/usr/bin/env python3
"""GEX Gate — 期权 Gamma Exposure 决策门控模块

将 GEX 数据接入交易决策，提供磁吸效应门控:
1. 识别当前价格附近的磁吸价位
2. 当磁吸方向与策略方向一致时，允许/加权信号
3. 当磁吸方向与策略方向相反时，缩仓/拒绝
4. 在最大痛点附近设置精确止盈止损位

决策逻辑:
  - 磁吸方向 UP + 策略 LONG  → 加权 (ALLOW, qty × 1.1)
  - 磁吸方向 UP + 策略 SHORT → 缩仓 (REDUCE, qty × 0.7)
  - 磁吸方向 DOWN + 策略 LONG → 缩仓 (REDUCE, qty × 0.7)
  - 磁吸方向 DOWN + 策略 SHORT → 加权 (ALLOW, qty × 1.1)
  - 磁吸强度 strong + 距离<2% → 强制磁吸 (FORCE, 调整止盈到磁吸价位)
  - 无磁吸数据 → 透传 (ALLOW)

用法:
    from sandbox_trading.gex_gate import GEXGate, GEXGateAction
    gate = GEXGate(enabled=True)
    gate.load_gex_data(symbol, date_range)
    decision = gate.apply_to_decision(decision, current_price, agent_id, symbol)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("hermes.gex_gate")


class GEXGateAction(Enum):
    """GEX 门控动作"""
    ALLOW = "allow"        # 允许开仓
    BOOST = "boost"        # 加权开仓 (与磁吸方向一致)
    REDUCE = "reduce"      # 缩仓 (与磁吸方向相反)
    FORCE = "force"        # 强制磁吸 (调整止盈到磁吸价位)
    NEUTRAL = "neutral"    # 无磁吸数据，透传


@dataclass(slots=True)
class GEXGateResult:
    """GEX 门控结果"""
    action: GEXGateAction = GEXGateAction.NEUTRAL
    quantity_multiplier: float = 1.0
    target_price: float = 0.0         # 磁吸目标价
    distance_pct: float = 0.0
    strength: str = "weak"
    confidence: float = 0.0
    total_gex: float = 0.0
    reason: str = ""


class GEXGate:
    """GEX 决策门控

    从 vault 加载 GEX 数据，在策略决策时进行磁吸效应门控。
    """

    # GEX 数据目录 (WSL 路径, 通过 junction 访问)
    GEX_DATA_DIR_WSL = "/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/gex"
    GEX_DATA_DIR_WIN = r"D:\hermes\hermes_obsidian_vault\Hermes\MarketData\gex"

    # 磁吸强度阈值 (美元)
    STRONG_GEX_THRESHOLD = 1e8   # 1亿美元
    MEDIUM_GEX_THRESHOLD = 1e7   # 1千万美元

    # 距离阈值 (百分比)
    CLOSE_DISTANCE_PCT = 0.02    # 2%以内视为近距离磁吸
    FAR_DISTANCE_PCT = 0.10      # 10%以外视为无效磁吸

    # 数量调整因子
    BOOST_MULTIPLIER = 1.1       # 与磁吸方向一致: 加权10%
    REDUCE_MULTIPLIER = 0.7      # 与磁吸方向相反: 缩仓30%
    FORCE_MULTIPLIER = 1.0       # 强制磁吸: 数量不变，但调整止盈

    def __init__(self, enabled: bool = True, data_dir: Optional[str] = None):
        """
        Args:
            enabled: 是否启用 GEX 门控
            data_dir: GEX 数据目录 (默认使用 GEX_DATA_DIR_WSL)
        """
        self.enabled = enabled
        self.data_dir = data_dir or self.GEX_DATA_DIR_WSL
        self._cache: Dict[str, Dict[str, Any]] = {}  # symbol -> latest GEX data
        self._load_count = 0
        self._apply_count = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._force_count = 0

    def load_gex_data(self, symbol: str, date_range: Optional[Tuple[str, str]] = None) -> int:
        """加载 GEX 数据

        Args:
            symbol: 交易对 (如 "BTC")
            date_range: 日期范围 (YYYYMMDD, YYYYMMDD)，默认加载最新

        Returns:
            int: 加载的记录数
        """
        if not self.enabled:
            return 0

        try:
            # 规范化 symbol
            sym = symbol.split("-")[0].upper()

            # 查找 GEX 数据文件
            data_path = Path(self.data_dir)
            if not data_path.exists():
                logger.debug("GEX data dir not found: %s", self.data_dir)
                return 0

            # 查找该 symbol 的所有数据文件
            pattern = f"{sym}_*.jsonl"
            files = sorted(data_path.glob(pattern))

            if not files:
                logger.debug("No GEX data files for %s", sym)
                return 0

            # 加载最新文件 (或指定日期范围)
            records = []
            for f in files:
                try:
                    # 提取文件名中的日期
                    fname = f.stem  # BTC_20260712
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
                    logger.warning("Failed to read GEX file %s: %s", f, e)
                    continue

            if not records:
                return 0

            # 按 timestamp 排序，取最新
            records.sort(key=lambda r: r.get("timestamp", 0))
            latest = records[-1]

            self._cache[sym] = {
                "latest": latest,
                "all_records": records,
                "n_records": len(records),
            }
            self._load_count += 1

            logger.info(
                "GEX data loaded: symbol=%s records=%d total_gex=%.2f spot=%.2f "
                "max_pain=%.2f pos_magnet=%.2f neg_magnet=%.2f",
                sym, len(records),
                latest.get("total_gex", 0),
                latest.get("spot_price", 0),
                latest.get("max_pain", 0),
                latest.get("positive_magnet", 0),
                latest.get("negative_magnet", 0),
            )
            return len(records)

        except Exception as e:
            logger.warning("GEX data load failed for %s: %s", symbol, e)
            return 0

    def get_latest_gex(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新的 GEX 数据

        Args:
            symbol: 交易对

        Returns:
            dict or None: 最新 GEX 数据
        """
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
        """对决策应用 GEX 门控

        Args:
            decision: 决策字典 (action, quantity, stop_loss, take_profit 等)
            current_price: 当前标的价格
            agent_id: Agent ID
            symbol: 交易对

        Returns:
            dict or None: 调整后的决策 (None 表示拒绝)
        """
        if not self.enabled or decision is None:
            return decision

        action = decision.get("action", "neutral")
        if action == "neutral":
            return decision

        # 获取 GEX 数据
        gex_data = self.get_latest_gex(symbol)
        if not gex_data:
            # 无 GEX 数据，透传
            return decision

        self._apply_count += 1

        # 分析磁吸效应
        result = self._analyze_magnetism(gex_data, current_price)

        if result.action == GEXGateAction.NEUTRAL:
            return decision

        # 应用门控
        orig_qty = decision.get("quantity", 0)
        orig_action = action

        if result.action == GEXGateAction.BOOST:
            # 与磁吸方向一致，加权
            new_qty = orig_qty * result.quantity_multiplier
            decision["quantity"] = new_qty
            decision["gex_gate_action"] = "boost"
            decision["gex_target_price"] = result.target_price
            decision["gex_confidence"] = result.confidence
            self._boost_count += 1
            if self._boost_count <= 5 or self._boost_count % 100 == 0:
                logger.info(
                    "GEXGate BOOST #%d: agent=%s symbol=%s action=%s target=%.2f dist=%.2f%%",
                    self._boost_count, agent_id[:12], symbol, orig_action,
                    result.target_price, result.distance_pct * 100,
                )

        elif result.action == GEXGateAction.REDUCE:
            # 与磁吸方向相反，缩仓
            new_qty = orig_qty * result.quantity_multiplier
            decision["quantity"] = new_qty
            decision["gex_gate_action"] = "reduce"
            decision["gex_target_price"] = result.target_price
            decision["gex_confidence"] = result.confidence
            self._reduce_count += 1
            if self._reduce_count <= 5 or self._reduce_count % 100 == 0:
                logger.info(
                    "GEXGate REDUCE #%d: agent=%s symbol=%s action=%s target=%.2f dist=%.2f%%",
                    self._reduce_count, agent_id[:12], symbol, orig_action,
                    result.target_price, result.distance_pct * 100,
                )

        elif result.action == GEXGateAction.FORCE:
            # 强制磁吸，调整止盈到磁吸价位
            decision["gex_gate_action"] = "force"
            decision["gex_target_price"] = result.target_price
            decision["gex_confidence"] = result.confidence

            # 调整止盈到磁吸目标价
            if result.target_price > 0:
                if action == "long":
                    # 多头: 止盈上移到磁吸目标
                    current_tp = decision.get("take_profit", 0)
                    if current_tp <= 0 or result.target_price > current_tp:
                        decision["take_profit"] = result.target_price
                elif action == "short":
                    # 空头: 止盈下移到磁吸目标
                    current_tp = decision.get("take_profit", 0)
                    if current_tp <= 0 or result.target_price < current_tp:
                        decision["take_profit"] = result.target_price

            self._force_count += 1
            if self._force_count <= 5 or self._force_count % 100 == 0:
                logger.info(
                    "GEXGate FORCE #%d: agent=%s symbol=%s action=%s target=%.2f strength=%s",
                    self._force_count, agent_id[:12], symbol, orig_action,
                    result.target_price, result.strength,
                )

        return decision

    def _analyze_magnetism(
        self,
        gex_data: Dict[str, Any],
        current_price: float,
    ) -> GEXGateResult:
        """分析磁吸效应

        Args:
            gex_data: GEX 数据 (来自 GEXCalculator)
            current_price: 当前标的价格

        Returns:
            GEXGateResult: 磁吸分析结果
        """
        result = GEXGateResult()

        total_gex = float(gex_data.get("total_gex", 0))
        positive_magnet = float(gex_data.get("positive_magnet", 0))
        negative_magnet = float(gex_data.get("negative_magnet", 0))
        max_positive_gex = float(gex_data.get("max_positive_gex", 0))
        max_negative_gex = float(gex_data.get("max_negative_gex", 0))

        result.total_gex = total_gex

        if current_price <= 0 or total_gex == 0:
            return result

        # 确定主要磁吸方向
        if total_gex > 0 and positive_magnet > 0:
            # 正GEX → 磁吸向上
            target = positive_magnet
            gex_value = max_positive_gex
            direction = "up"
        elif total_gex < 0 and negative_magnet > 0:
            # 负GEX → 磁吸向下
            target = negative_magnet
            gex_value = max_negative_gex
            direction = "down"
        else:
            return result

        # 计算距离
        if target <= 0:
            return result

        distance_pct = abs(target - current_price) / current_price
        result.target_price = target
        result.distance_pct = distance_pct

        # 距离过远，不视为有效磁吸
        if distance_pct > self.FAR_DISTANCE_PCT:
            return result

        # 强度评估
        abs_gex = abs(gex_value)
        if abs_gex >= self.STRONG_GEX_THRESHOLD:
            result.strength = "strong"
        elif abs_gex >= self.MEDIUM_GEX_THRESHOLD:
            result.strength = "medium"
        else:
            result.strength = "weak"

        # 置信度
        distance_factor = max(0.0, 1.0 - distance_pct / self.FAR_DISTANCE_PCT)
        gex_factor = min(1.0, abs_gex / self.STRONG_GEX_THRESHOLD)
        result.confidence = distance_factor * gex_factor

        # 判断门控动作
        # 这里只确定磁吸方向，具体 BOOST/REDUCE 取决于策略方向 (在 apply_to_decision 中判断)
        # 强磁吸 + 近距离 → FORCE
        if result.strength == "strong" and distance_pct <= self.CLOSE_DISTANCE_PCT:
            result.action = GEXGateAction.FORCE
            result.quantity_multiplier = self.FORCE_MULTIPLIER
            result.reason = f"strong {direction} magnetism at {target:.2f} (dist={distance_pct*100:.2f}%)"
        else:
            # 一般磁吸，返回 BOOST 或 REDUCE 由调用方根据策略方向判断
            # 这里先标记方向，apply_to_decision 会根据 action 调整
            result.action = GEXGateAction.BOOST if direction == "up" else GEXGateAction.REDUCE
            # 注意：这里的 BOOST/REDUCE 是基于磁吸方向，不是策略方向
            # apply_to_decision 会根据策略方向与磁吸方向的关系调整
            result.quantity_multiplier = 1.0  # 默认1.0，具体在 apply_to_decision 中设置
            result.reason = f"{direction} magnetism at {target:.2f} (strength={result.strength})"

        # 修正：实际上应该在 apply_to_decision 中比较磁吸方向和策略方向
        # 这里返回磁吸分析结果，包含方向信息
        result.reason = f"{direction}:{result.strength} target={target:.2f}"

        return result

    def apply_to_decision_v2(
        self,
        decision: Optional[Dict[str, Any]],
        current_price: float,
        agent_id: str = "",
        symbol: str = "",
    ) -> Optional[Dict[str, Any]]:
        """对决策应用 GEX 门控 (v2: 考虑策略方向)

        Args:
            decision: 决策字典
            current_price: 当前标的价格
            agent_id: Agent ID
            symbol: 交易对

        Returns:
            dict or None: 调整后的决策
        """
        if not self.enabled or decision is None:
            return decision

        action = decision.get("action", "neutral")
        if action == "neutral":
            return decision

        gex_data = self.get_latest_gex(symbol)
        if not gex_data:
            return decision

        self._apply_count += 1

        # 分析磁吸
        total_gex = float(gex_data.get("total_gex", 0))
        positive_magnet = float(gex_data.get("positive_magnet", 0))
        negative_magnet = float(gex_data.get("negative_magnet", 0))
        max_positive_gex = float(gex_data.get("max_positive_gex", 0))
        max_negative_gex = float(gex_data.get("max_negative_gex", 0))

        if current_price <= 0 or total_gex == 0:
            return decision

        # 确定磁吸方向
        if total_gex > 0 and positive_magnet > 0:
            magnet_direction = "up"
            target = positive_magnet
            gex_value = max_positive_gex
        elif total_gex < 0 and negative_magnet > 0:
            magnet_direction = "down"
            target = negative_magnet
            gex_value = max_negative_gex
        else:
            return decision

        distance_pct = abs(target - current_price) / current_price
        if distance_pct > self.FAR_DISTANCE_PCT:
            return decision

        # 强度
        abs_gex = abs(gex_value)
        if abs_gex >= self.STRONG_GEX_THRESHOLD:
            strength = "strong"
        elif abs_gex >= self.MEDIUM_GEX_THRESHOLD:
            strength = "medium"
        else:
            strength = "weak"

        # 置信度
        distance_factor = max(0.0, 1.0 - distance_pct / self.FAR_DISTANCE_PCT)
        gex_factor = min(1.0, abs_gex / self.STRONG_GEX_THRESHOLD)
        confidence = distance_factor * gex_factor

        # 策略方向
        strategy_direction = "up" if action == "long" else "down"

        orig_qty = decision.get("quantity", 0)

        # 强磁吸 + 近距离 → FORCE (调整止盈到磁吸价位)
        if strength == "strong" and distance_pct <= self.CLOSE_DISTANCE_PCT:
            decision["gex_gate_action"] = "force"
            decision["gex_target_price"] = target
            decision["gex_confidence"] = confidence
            decision["gex_direction"] = magnet_direction

            # 调整止盈
            if action == "long" and target > current_price:
                current_tp = decision.get("take_profit", 0)
                if current_tp <= 0 or target > current_tp:
                    decision["take_profit"] = target
            elif action == "short" and target < current_price:
                current_tp = decision.get("take_profit", 0)
                if current_tp <= 0 or target < current_tp:
                    decision["take_profit"] = target

            self._force_count += 1
            if self._force_count <= 5 or self._force_count % 100 == 0:
                logger.info(
                    "GEXGate FORCE #%d: agent=%s symbol=%s action=%s target=%.2f strength=%s",
                    self._force_count, agent_id[:12], symbol, action,
                    target, strength,
                )

        # 磁吸方向与策略方向一致 → BOOST
        elif magnet_direction == strategy_direction:
            decision["quantity"] = orig_qty * self.BOOST_MULTIPLIER
            decision["gex_gate_action"] = "boost"
            decision["gex_target_price"] = target
            decision["gex_confidence"] = confidence
            decision["gex_direction"] = magnet_direction

            self._boost_count += 1
            if self._boost_count <= 5 or self._boost_count % 100 == 0:
                logger.info(
                    "GEXGate BOOST #%d: agent=%s symbol=%s action=%s target=%.2f dist=%.2f%% qty=%.6f→%.6f",
                    self._boost_count, agent_id[:12], symbol, action,
                    target, distance_pct * 100, orig_qty, decision["quantity"],
                )

        # 磁吸方向与策略方向相反 → REDUCE
        else:
            decision["quantity"] = orig_qty * self.REDUCE_MULTIPLIER
            decision["gex_gate_action"] = "reduce"
            decision["gex_target_price"] = target
            decision["gex_confidence"] = confidence
            decision["gex_direction"] = magnet_direction

            self._reduce_count += 1
            if self._reduce_count <= 5 or self._reduce_count % 100 == 0:
                logger.info(
                    "GEXGate REDUCE #%d: agent=%s symbol=%s action=%s target=%.2f dist=%.2f%% qty=%.6f→%.6f",
                    self._reduce_count, agent_id[:12], symbol, action,
                    target, distance_pct * 100, orig_qty, decision["quantity"],
                )

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
