"""Macro Event Gate — Layer 6: Risk Control Layer

Aegis-X 8层架构补全: 宏观事件门控 (50% → 100%)

Purpose:
    在CPI/利率决议等重大宏观事件前后自动缩仓或禁止开仓.
    一次数据发布就能打穿所有策略的止损, 必须在事件前主动降风险.

Design:
    1. Load macro events from JSONL files (macro_calendar directory)
    2. Build event timeline indexed by Unix timestamp
    3. At each tick, check if current time is near a high-importance event
    4. Apply position reduction / trade blocking based on event proximity

Event Windows (configurable):
    - Pre-event window: [T-6h, T-1h] → reduce position by 50%
    - Event window: [T-1h, T+1h] → block new trades (multiplier=0.0)
    - Post-event window: [T+1h, T+2h] → reduce position by 50%
    - Normal: allow full position (multiplier=1.0)

Data Format (JSONL):
    {"timestamp": "2026-07-15T13:30:00Z", "event": "CPI", "country": "US",
     "importance": "high", "actual": null, "forecast": null, "previous": null,
     "source": "hardcoded"}

Usage:
    gate = MacroEventGate(enabled=True)
    gate.load_events()
    # At each tick:
    action, multiplier, reason = gate.check(market_timestamp)
    if action == "block":
        decision = None  # reject trade
    elif action == "reduce":
        decision["quantity"] *= multiplier  # reduce position
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GateAction(Enum):
    """宏观事件门控动作"""

    ALLOW = "allow"  # 允许正常交易
    REDUCE = "reduce"  # 缩仓50%
    BLOCK = "block"  # 禁止新开仓


class MacroEventGate:
    """宏观事件门控器

    在重大宏观事件(CPI/FOMC/NFP)前后自动降风险.
    """

    # 数据目录 (D盘junction路径, WSL通过/mnt/d访问)
    MACRO_DATA_DIR_WSL = "/mnt/d/hermes/hermes_obsidian_vault/Hermes/MarketData/macro_calendar"
    MACRO_DATA_DIR_WIN = r"D:\hermes\hermes_obsidian_vault\Hermes\MarketData\macro_calendar"

    # 事件窗口 (秒)
    PRE_EVENT_START = 6 * 3600  # 事件前6小时开始缩仓
    PRE_EVENT_END = 1 * 3600  # 事件前1小时进入阻塞
    EVENT_BLOCK_END = 1 * 3600  # 事件后1小时解除阻塞
    POST_EVENT_END = 2 * 3600  # 事件后2小时恢复正常

    # 仓位调整
    REDUCE_MULTIPLIER = 0.5  # 缩仓50%
    BLOCK_MULTIPLIER = 0.0  # 禁止开仓
    ALLOW_MULTIPLIER = 1.0  # 正常仓位

    # 只关注高重要性事件
    HIGH_IMPORTANCE = {"high", "critical", "3"}

    # 类级缓存: 所有实例共享同一份事件数据, 避免重复文件I/O
    # 根因: 多个 MacroEventGate 实例各自独立加载事件文件, 导致27次文件I/O(15.5秒)
    # 修复: 首次加载后缓存到类级, 后续实例直接复用
    _class_events_cache: List[Tuple[float, str, str]] = []
    _class_events_loaded: bool = False
    _class_events_count: int = 0

    def __init__(self, enabled: bool = True, data_dir: Optional[str] = None):
        self.enabled = enabled
        self.data_dir = data_dir or self.MACRO_DATA_DIR_WSL

        # 事件列表: [(unix_ts, event_name, importance), ...]
        self._events: List[Tuple[float, str, str]] = []
        self._events_loaded: bool = False

        # 统计
        self._check_count: int = 0
        self._reduce_count: int = 0
        self._block_count: int = 0

        logger.info(
            "Layer6 MacroEventGate initialized: enabled=%s data_dir=%s",
            self.enabled,
            self.data_dir,
        )

    def _parse_timestamp(self, ts_str: str) -> Optional[float]:
        """解析ISO 8601时间戳为Unix timestamp"""
        try:
            # Handle "2026-07-15T13:30:00Z" format
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse timestamp '%s': %s", ts_str, e)
            return None

    def load_events(self) -> int:
        """从JSONL文件加载宏观事件

        优先使用类级缓存, 避免多实例重复加载同一份文件.

        Returns:
            加载的事件数量
        """
        # 类级缓存命中: 所有实例共享同一份事件数据, 文件I/O只发生一次
        if MacroEventGate._class_events_loaded:
            self._events = MacroEventGate._class_events_cache
            self._events_loaded = True
            logger.debug(
                "Layer6 MacroEventGate: using class cache (%d events)",
                MacroEventGate._class_events_count,
            )
            return MacroEventGate._class_events_count

        self._events = []
        self._events_loaded = True

        if not os.path.exists(self.data_dir):
            logger.warning(
                "Layer6 MacroEventGate: data dir not found: %s",
                self.data_dir,
            )
            # 写入类级缓存(空列表), 避免后续实例重复检查目录
            MacroEventGate._class_events_cache = []
            MacroEventGate._class_events_loaded = True
            MacroEventGate._class_events_count = 0
            return 0

        count = 0
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(self.data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        ts_str = rec.get("timestamp", "")
                        unix_ts = self._parse_timestamp(ts_str)
                        if unix_ts is None:
                            continue

                        event_name = rec.get("event", "UNKNOWN")
                        importance = rec.get("importance", "low").lower()

                        # 只保留高重要性事件
                        if importance not in self.HIGH_IMPORTANCE:
                            continue

                        self._events.append((unix_ts, event_name, importance))
                        count += 1
            except OSError as e:
                logger.warning("Failed to read %s: %s", fpath, e)

        # 按时间排序
        self._events.sort(key=lambda x: x[0])

        # 写入类级缓存: 后续实例直接复用, 避免重复文件I/O
        MacroEventGate._class_events_cache = list(self._events)
        MacroEventGate._class_events_loaded = True
        MacroEventGate._class_events_count = len(self._events)

        logger.info(
            "Layer6 MacroEventGate: loaded %d high-importance events from %s (cached at class level)",
            len(self._events),
            self.data_dir,
        )

        # 打印前5个事件用于调试
        for i, (ts, name, imp) in enumerate(self._events[:5]):
            dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            logger.info("  Event %d: %s %s (%s)", i + 1, dt_str, name, imp)

        return len(self._events)

    def check(self, current_ts: float) -> Tuple[GateAction, float, str]:
        """检查当前时间是否在宏观事件窗口内

        Args:
            current_ts: 当前Unix时间戳 (秒)

        Returns:
            (action, multiplier, reason)
        """
        if not self.enabled:
            return GateAction.ALLOW, self.ALLOW_MULTIPLIER, "gate disabled"

        if not self._events_loaded:
            self.load_events()

        if not self._events or current_ts <= 0:
            return GateAction.ALLOW, self.ALLOW_MULTIPLIER, "no events"

        self._check_count += 1

        # 检查是否在任何事件的窗口内
        for event_ts, event_name, importance in self._events:
            time_diff = current_ts - event_ts

            # 事件后窗口: [T+1h, T+2h] → reduce
            if self.EVENT_BLOCK_END <= time_diff <= self.POST_EVENT_END:
                self._reduce_count += 1
                return (
                    GateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"post-event {event_name} ({time_diff/3600:.1f}h after)",
                )

            # 事件窗口: [T-1h, T+1h] → block
            if -self.PRE_EVENT_END <= time_diff <= self.EVENT_BLOCK_END:
                self._block_count += 1
                return (
                    GateAction.BLOCK,
                    self.BLOCK_MULTIPLIER,
                    f"during {event_name} ({time_diff/3600:+.1f}h)",
                )

            # 事件前窗口: [T-6h, T-1h] → reduce
            if -self.PRE_EVENT_START <= time_diff <= -self.PRE_EVENT_END:
                self._reduce_count += 1
                hours_before = -time_diff / 3600
                return (
                    GateAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"pre-event {event_name} ({hours_before:.1f}h before)",
                )

        return GateAction.ALLOW, self.ALLOW_MULTIPLIER, "no nearby event"

    def apply_to_decision(
        self,
        decision: Optional[Dict[str, Any]],
        current_ts: float,
        agent_id: str = "",
        symbol: str = "",
    ) -> Optional[Dict[str, Any]]:
        """将宏观事件门控应用到交易决策

        Args:
            decision: 交易决策字典
            current_ts: 当前Unix时间戳
            agent_id: Agent ID
            symbol: 交易对

        Returns:
            修改后的decision, 或None(拒绝交易)
        """
        if not self.enabled or decision is None:
            return decision

        # 只对开仓决策进行门控
        action_type = decision.get("action", "neutral")
        if action_type not in ("long", "short"):
            return decision

        gate_action, multiplier, reason = self.check(current_ts)

        # 注入门控信息到decision
        decision["macro_gate_action"] = gate_action.value
        decision["macro_gate_reason"] = reason

        if gate_action == GateAction.BLOCK:
            # 事件期间: 禁止新开仓
            if self._block_count <= 5 or self._block_count % 100 == 0:
                logger.info(
                    "Layer6 MacroGate BLOCK #%d: agent=%s symbol=%s reason=%s",
                    self._block_count, agent_id[:12], symbol, reason,
                )
            return None

        if gate_action == GateAction.REDUCE:
            # 事件前后: 缩仓50%
            orig_qty = decision.get("quantity", 0)
            if orig_qty and orig_qty > 0:
                decision["quantity"] = orig_qty * multiplier
                if self._reduce_count <= 5 or self._reduce_count % 100 == 0:
                    logger.info(
                        "Layer6 MacroGate REDUCE #%d: agent=%s symbol=%s qty=%.6f→%.6f reason=%s",
                        self._reduce_count, agent_id[:12], symbol,
                        orig_qty, decision["quantity"], reason,
                    )

        return decision

    def get_stats(self) -> Dict[str, Any]:
        """获取门控器统计信息"""
        return {
            "enabled": self.enabled,
            "events_loaded": len(self._events),
            "is_loaded": self._events_loaded,
            "check_count": self._check_count,
            "reduce_count": self._reduce_count,
            "block_count": self._block_count,
            "reduce_rate": self._reduce_count / max(1, self._check_count),
            "block_rate": self._block_count / max(1, self._check_count),
        }
