"""事件记录 — 独立模块，避免循环导入。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class EventRecord:
    """事件记录 — append-only 日志中的一条。"""
    event_id: str = ""
    event_type: str = ""
    timestamp: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    source_layer: str = ""
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventRecord":
        return cls(**d)
