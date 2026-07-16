"""死信队列 — 处理失败事件。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .event_record import EventRecord


@dataclass
class DeadLetterEntry:
    """死信条目 — 处理失败的事件。"""
    original_event: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_type: str = ""
    failed_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    is_terminal: bool = False  # 是否终态（不可重试）


class DeadLetterQueue:
    """死信队列 — 存储处理失败的事件。

    铁律：层故障不被吞掉，失败事件进入死信队列。
    """

    def __init__(self, max_retries: int = 3):
        self._entries: List[DeadLetterEntry] = []
        self._max_retries = max_retries

    def add(self, event: EventRecord, error: str, error_type: str = "") -> DeadLetterEntry:
        """添加失败事件到死信队列。"""
        entry = DeadLetterEntry(
            original_event=event.to_dict(),
            error=error,
            error_type=error_type,
            failed_at=time.time(),
            max_retries=self._max_retries,
        )
        self._entries.append(entry)
        return entry

    def retry(self, index: int) -> bool:
        """重试死信条目。返回 True 如果还可以继续重试。"""
        if index >= len(self._entries):
            return False
        entry = self._entries[index]
        entry.retry_count += 1
        if entry.retry_count >= entry.max_retries:
            entry.is_terminal = True
            return False
        return True

    def get_entries(self, terminal_only: bool = False) -> List[DeadLetterEntry]:
        """获取死信条目。"""
        if terminal_only:
            return [e for e in self._entries if e.is_terminal]
        return list(self._entries)

    def get_count(self) -> int:
        """获取死信条目数。"""
        return len(self._entries)

    def get_terminal_count(self) -> int:
        """获取终态条目数。"""
        return sum(1 for e in self._entries if e.is_terminal)

    def clear(self) -> None:
        """清空死信队列（仅用于测试）。"""
        self._entries.clear()
