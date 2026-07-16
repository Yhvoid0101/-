"""BLOCKED 传播 — 确保层故障不被吞掉。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .event_bus import EventBus
from .event_record import EventRecord
from .dead_letter import DeadLetterQueue


@dataclass
class BlockedException(Exception):
    """BLOCKED 异常 — 层故障传播。"""
    layer: str = ""
    error_code: str = ""
    error_message: str = ""
    correlation_id: str = ""
    event_id: str = ""

    def __str__(self) -> str:
        return f"[BLOCKED] L{self.layer}: {self.error_code} - {self.error_message}"


class BlockedPropagator:
    """BLOCKED 传播器。

    铁律：
    - 任意层输出 BLOCKED → 下游层停止处理
    - 记录到审计链
    - 通知变更控制面
    - 不被吞掉
    """

    def __init__(self, bus: EventBus, dead_letter: DeadLetterQueue):
        self._bus = bus
        self._dead_letter = dead_letter
        self._blocked_history: List[Dict[str, Any]] = []

    def propagate(self, layer: str, error_code: str, error_message: str,
                  correlation_id: str = "", source_event: Optional[EventRecord] = None) -> BlockedException:
        """传播 BLOCKED 状态。

        1. 记录到审计链（事件总线）
        2. 添加到死信队列
        3. 返回 BlockedException 供上层处理
        """
        # 记录到审计链
        self._bus.publish(
            event_type="BLOCKED",
            payload={
                "layer": layer,
                "error_code": error_code,
                "error_message": error_message,
                "source_event_id": source_event.event_id if source_event else "",
            },
            correlation_id=correlation_id,
            source_layer=layer,
        )

        # 添加到死信队列
        if source_event:
            self._dead_letter.add(source_event, error_message, error_code)

        # 记录历史
        self._blocked_history.append({
            "layer": layer,
            "error_code": error_code,
            "error_message": error_message,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        })

        return BlockedException(
            layer=layer,
            error_code=error_code,
            error_message=error_message,
            correlation_id=correlation_id,
            event_id=source_event.event_id if source_event else "",
        )

    def get_blocked_history(self) -> List[Dict[str, Any]]:
        """获取 BLOCKED 历史。"""
        return list(self._blocked_history)

    def get_blocked_count(self) -> int:
        """获取 BLOCKED 总数。"""
        return len(self._blocked_history)

    def get_blocked_by_layer(self, layer: str) -> List[Dict[str, Any]]:
        """获取指定层的 BLOCKED 记录。"""
        return [b for b in self._blocked_history if b["layer"] == layer]
