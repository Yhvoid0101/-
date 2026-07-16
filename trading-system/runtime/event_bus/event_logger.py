"""事件日志 — append-only 持久化到文件。"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from .event_record import EventRecord


class EventLogger:
    """append-only 事件日志，持久化到文件。不可篡改。"""

    def __init__(self, log_path: str, max_size_mb: float = 100.0):
        self._log_path = log_path
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._lock = threading.Lock()
        self._line_count = 0

        # 确保目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # 如果文件不存在，创建空文件
        if not os.path.exists(log_path):
            open(log_path, "w").close()

    def append(self, record: EventRecord) -> None:
        """追加事件到日志文件。append-only，不可篡改。"""
        with self._lock:
            line = json.dumps(record.to_dict(), default=str, ensure_ascii=False)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._line_count += 1

            # 大小检查，自动轮转
            if os.path.getsize(self._log_path) > self._max_size_bytes:
                self._rotate()

    def _rotate(self) -> None:
        """日志轮转。"""
        rotated = self._log_path + ".1"
        if os.path.exists(rotated):
            os.remove(rotated)
        os.rename(self._log_path, rotated)
        open(self._log_path, "w").close()

    def read_all(self) -> list:
        """读取所有事件记录。"""
        records = []
        if not os.path.exists(self._log_path):
            return records
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        records.append(EventRecord.from_dict(d))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_line_count(self) -> int:
        """获取日志行数。"""
        return self._line_count

    def get_size_bytes(self) -> int:
        """获取日志文件大小（字节）。"""
        if not os.path.exists(self._log_path):
            return 0
        return os.path.getsize(self._log_path)
