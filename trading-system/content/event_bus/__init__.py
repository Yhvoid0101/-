"""Phase 1 沙盘事件总线包 — 单进程，append-only 事件日志。

铁律合规：
- 幂等可重放：同一事件序列重放，结果完全一致
- BLOCKED 不被吞掉：层故障传播到审计链
"""
from .event_record import EventRecord
from .event_bus import EventBus
from .event_logger import EventLogger
from .event_serializer import EventSerializer

__all__ = ["EventBus", "EventRecord", "EventLogger", "EventSerializer"]
