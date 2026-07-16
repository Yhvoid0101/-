"""幂等处理 — 基于 event_id 去重。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set


class IdempotencyChecker:
    """幂等检查器 — 基于 event_id 去重。

    确保同一事件序列重放时，不产生重复副作用。
    """

    def __init__(self, max_size: int = 100000):
        self._seen_ids: Set[str] = set()
        self._max_size = max_size
        self._eviction_order: list = []  # FIFO 淘汰

    def is_duplicate(self, event_id: str) -> bool:
        """检查是否为重复事件。"""
        return event_id in self._seen_ids

    def mark_seen(self, event_id: str) -> bool:
        """标记 event_id 为已处理。返回 True 如果是新事件，False 如果是重复。"""
        if event_id in self._seen_ids:
            return False  # 重复事件
        self._seen_ids.add(event_id)
        self._eviction_order.append(event_id)
        # 容量淘汰
        if len(self._seen_ids) > self._max_size:
            old_id = self._eviction_order.pop(0)
            self._seen_ids.discard(old_id)
        return True

    def check_and_mark(self, event_id: str) -> bool:
        """检查并标记。返回 True 如果是新事件（应处理），False 如果是重复（应跳过）。"""
        return self.mark_seen(event_id)

    def reset(self) -> None:
        """重置（仅用于测试）。"""
        self._seen_ids.clear()
        self._eviction_order.clear()

    def get_count(self) -> int:
        """获取已处理事件数。"""
        return len(self._seen_ids)
