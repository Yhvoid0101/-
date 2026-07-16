"""契约基础模块 — 所有版本化契约的公共字段和工具。

铁律合规：
- 零模拟：所有契约必须携带真实数据谱系，禁止合成数据通过契约传播
- 零降级：数据不可用时输出 BLOCKED，不允许 silent fallback
- 幂等可重放：每个契约有唯一 event_id，支持重放
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class ContractStatus(str, Enum):
    """契约状态枚举。BLOCKED 语义不可被吞掉。"""
    OK = "OK"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"


class Severity(str, Enum):
    """问题严重程度。"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DataLineage:
    """数据谱系 — 追溯数据真实来源。"""
    source: str
    event_time: float
    ingest_time: float
    ttl_seconds: float
    checksum: str
    origin: str = ""
    is_cached: bool = False
    cache_generated_at: Optional[float] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        current = now if now is not None else time.time()
        return current > (self.ingest_time + self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataLineage":
        return cls(**d)


@dataclass
class ContractBase:
    """所有契约的基类 — 版本化、幂等、谱系、签名。"""
    schema_version: str = "1.0.0"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    timestamp: float = field(default_factory=time.time)
    lineage: DataLineage = field(default_factory=lambda: DataLineage(
        source="unknown", event_time=0.0, ingest_time=0.0, ttl_seconds=3600.0, checksum="0" * 16))
    signature: str = ""
    status: ContractStatus = ContractStatus.OK
    error_code: str = ""
    error_message: str = ""

    def compute_signature(self, payload: Dict[str, Any]) -> str:
        sign_content = f"{self.event_id}|{self.correlation_id}|{self.timestamp}|{json.dumps(payload, sort_keys=True, default=str)}"
        return hashlib.sha256(sign_content.encode("utf-8")).hexdigest()[:32]

    def is_blocked(self) -> bool:
        return self.status == ContractStatus.BLOCKED

    def is_valid(self) -> bool:
        if self.is_blocked():
            return False
        if self.lineage.is_expired():
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContractBase":
        if "status" in d and isinstance(d["status"], str):
            d["status"] = ContractStatus(d["status"])
        if "lineage" in d and isinstance(d["lineage"], dict):
            d["lineage"] = DataLineage.from_dict(d["lineage"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_blocked(base: ContractBase, error_code: str, error_message: str, correlation_id: str = "") -> ContractBase:
    """生成 BLOCKED 契约。铁律：数据不可用必须 BLOCKED，不允许估算替代。"""
    base.status = ContractStatus.BLOCKED
    base.error_code = error_code
    base.error_message = error_message
    if correlation_id:
        base.correlation_id = correlation_id
    return base
