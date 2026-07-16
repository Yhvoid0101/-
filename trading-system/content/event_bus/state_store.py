"""状态存储 — key-value 状态存储，支持事件重放时恢复状态。"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional


class StateStore:
    """key-value 状态存储。

    支持快照和恢复，用于事件重放时恢复状态。
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值。"""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置状态值。"""
        self._data[key] = value

    def delete(self, key: str) -> None:
        """删除状态值。"""
        self._data.pop(key, None)

    def snapshot(self, name: str) -> None:
        """创建状态快照（深拷贝）。"""
        self._snapshots[name] = copy.deepcopy(self._data)

    def restore(self, name: str) -> bool:
        """从快照恢复状态。"""
        if name not in self._snapshots:
            return False
        self._data = copy.deepcopy(self._snapshots[name])
        return True

    def list_keys(self) -> list:
        """列出所有 key。"""
        return list(self._data.keys())

    def list_snapshots(self) -> list:
        """列出所有快照。"""
        return list(self._snapshots.keys())

    def clear(self) -> None:
        """清空状态（不包括快照）。"""
        self._data.clear()

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典。"""
        return copy.deepcopy(self._data)
