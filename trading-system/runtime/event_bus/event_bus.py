"""事件总线 — 单进程，append-only 事件日志。

支持发布/订阅/重放。事件不可篡改。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from .event_logger import EventLogger
from .event_serializer import EventSerializer
from .event_record import EventRecord


# EventRecord 已移到 event_record.py 避免循环导入
class _Placeholder:
    """事件记录 — append-only 日志中的一条。"""
    event_id: str = ""
    event_type: str = ""             # 事件类型（如 "MarketDataEnvelope", "AuditVerdict"）
    timestamp: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)  # 序列化后的契约
    correlation_id: str = ""
    source_layer: str = ""           # 来源层（L1-L8）
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventRecord":
        return cls(**d)


class EventBus:
    """单进程事件总线。

    特性：
    - append-only 事件日志，不可篡改
    - 发布/订阅模式
    - 事件持久化到文件
    - 支持重放
    - 幂等去重（基于 event_id）
    """

    def __init__(self, logger: Optional[EventLogger] = None, enable_persistence: bool = True):
        self._records: List[EventRecord] = []
        self._subscribers: Dict[str, List[Callable[[EventRecord], None]]] = {}
        self._event_ids: set = set()  # 幂等去重
        self._logger = logger
        self._serializer = EventSerializer()
        self._enable_persistence = enable_persistence

    def publish(self, event_type: str, payload: Dict[str, Any],
                correlation_id: str = "", source_layer: str = "",
                event_id: str = "") -> EventRecord:
        """发布事件到总线。幂等：重复 event_id 不会重复发布。"""
        eid = event_id or str(uuid.uuid4())
        # 幂等去重
        if eid in self._event_ids:
            return next(r for r in self._records if r.event_id == eid)
        self._event_ids.add(eid)

        record = EventRecord(
            event_id=eid,
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
            correlation_id=correlation_id,
            source_layer=source_layer,
            signature=self._serializer.compute_signature(payload, eid),
        )
        self._records.append(record)

        # 持久化
        if self._enable_persistence and self._logger:
            self._logger.append(record)

        # 通知订阅者
        for handler in self._subscribers.get(event_type, []):
            handler(record)
        for handler in self._subscribers.get("*", []):  # 通配订阅
            handler(record)

        return record

    def subscribe(self, event_type: str, handler: Callable[[EventRecord], None]) -> None:
        """订阅事件。event_type="*" 订阅所有事件。"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def replay(self, from_timestamp: float = 0.0) -> List[EventRecord]:
        """重放事件（从指定时间戳开始）。"""
        return [r for r in self._records if r.timestamp >= from_timestamp]

    def replay_and_notify(self, from_timestamp: float = 0.0) -> int:
        """重放事件并通知订阅者。返回重放事件数。"""
        count = 0
        for record in self.replay(from_timestamp):
            for handler in self._subscribers.get(record.event_type, []):
                handler(record)
            count += 1
        return count

    def get_records(self, event_type: str = "", correlation_id: str = "") -> List[EventRecord]:
        """查询事件记录。"""
        result = self._records
        if event_type:
            result = [r for r in result if r.event_type == event_type]
        if correlation_id:
            result = [r for r in result if r.correlation_id == correlation_id]
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        type_counts: Dict[str, int] = {}
        for r in self._records:
            type_counts[r.event_type] = type_counts.get(r.event_type, 0) + 1
        return {
            "total_events": len(self._records),
            "unique_ids": len(self._event_ids),
            "type_counts": type_counts,
        }

    def clear(self) -> None:
        """清空事件日志（仅用于测试）。"""
        self._records.clear()
        self._event_ids.clear()
