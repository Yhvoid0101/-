"""事件序列化 — JSON 序列化，支持 dataclass。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


class EventSerializer:
    """事件序列化工具。"""

    @staticmethod
    def serialize(obj: Any) -> str:
        """序列化对象为 JSON 字符串。"""
        if is_dataclass(obj):
            d = asdict(obj)
        elif isinstance(obj, dict):
            d = obj
        else:
            d = {"value": str(obj)}
        return json.dumps(d, default=str, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def deserialize(json_str: str) -> Dict[str, Any]:
        """从 JSON 字符串反序列化。"""
        return json.loads(json_str)

    @staticmethod
    def compute_signature(payload: Dict[str, Any], event_id: str) -> str:
        """计算事件签名（SHA-256，前32位）。防篡改。"""
        content = f"{event_id}|{json.dumps(payload, sort_keys=True, default=str)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def verify_signature(payload: Dict[str, Any], event_id: str, signature: str) -> bool:
        """验证签名。"""
        expected = EventSerializer.compute_signature(payload, event_id)
        return expected == signature
