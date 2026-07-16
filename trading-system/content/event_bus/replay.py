"""事件重放引擎 — 从日志重放事件，恢复状态。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .event_bus import EventBus
from .event_record import EventRecord
from .event_logger import EventLogger
from .idempotency import IdempotencyChecker
from .state_store import StateStore


class ReplayEngine:
    """事件重放引擎。

    从事件日志重放事件，恢复状态。
    铁律：同一事件序列重放，状态完全一致。
    """

    def __init__(self, state_store: StateStore, idempotency: IdempotencyChecker):
        self._state = state_store
        self._idempotency = idempotency
        self._handlers: Dict[str, Callable[[EventRecord, StateStore], None]] = {}

    def register_handler(self, event_type: str,
                         handler: Callable[[EventRecord, StateStore], None]) -> None:
        """注册事件处理器。处理器接收事件记录和状态存储。"""
        self._handlers[event_type] = handler

    def replay_from_log(self, logger: EventLogger, from_timestamp: float = 0.0) -> Dict[str, Any]:
        """从日志文件重放事件。

        重放前重置幂等检查器和状态存储，确保确定性。
        """
        self._idempotency.reset()
        self._state.clear()

        records = logger.read_all()
        processed = 0
        skipped = 0
        errors = 0

        for record in records:
            if record.timestamp < from_timestamp:
                continue

            # 幂等检查
            if not self._idempotency.check_and_mark(record.event_id):
                skipped += 1
                continue

            # 调用处理器
            handler = self._handlers.get(record.event_type)
            if handler:
                try:
                    handler(record, self._state)
                    processed += 1
                except Exception as e:
                    errors += 1
            else:
                # 无处理器的事件，仅记录
                processed += 1

        return {
            "total": len(records),
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }

    def replay_from_bus(self, bus: EventBus, from_timestamp: float = 0.0) -> Dict[str, Any]:
        """从事件总线重放事件。"""
        self._idempotency.reset()
        self._state.clear()

        records = bus.replay(from_timestamp)
        processed = 0
        skipped = 0

        for record in records:
            if not self._idempotency.check_and_mark(record.event_id):
                skipped += 1
                continue
            handler = self._handlers.get(record.event_type)
            if handler:
                handler(record, self._state)
            processed += 1

        return {
            "total": len(records),
            "processed": processed,
            "skipped": skipped,
            "errors": 0,
        }

    def verify_state_consistency(self, other_state: StateStore) -> bool:
        """验证两个状态存储是否一致。"""
        return self._state.to_dict() == other_state.to_dict()
