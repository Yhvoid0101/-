"""Model Registry — 模型版本管理。

铁律：版本、训练数据哈希、指标、审批状态。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .policy_registry import ApprovalStatus


@dataclass
class ModelEntry:
    """模型版本条目。"""
    model_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    model_type: str = ""              # 模型类型（如 "regime_classifier", "ood_detector"）
    train_data_hash: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    approved_by: str = ""
    approved_at: float = 0.0
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["approval_status"] = self.approval_status.value
        return d


class ModelRegistry:
    """模型注册表 — 版本管理。"""

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}

    def register(self, name: str, model_type: str, train_data_hash: str,
                 metrics: Dict[str, float] = None,
                 hyperparams: Dict[str, Any] = None) -> ModelEntry:
        """注册新模型版本。"""
        model_id = hashlib.sha256(f"{name}|{model_type}|{time.time()}".encode()).hexdigest()[:16]
        entry = ModelEntry(
            model_id=model_id,
            name=name,
            model_type=model_type,
            train_data_hash=train_data_hash,
            metrics=metrics or {},
            hyperparams=hyperparams or {},
        )
        self._models[model_id] = entry
        return entry

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def list_by_type(self, model_type: str) -> List[ModelEntry]:
        return [m for m in self._models.values() if m.model_type == model_type]

    def update_status(self, model_id: str, status: ApprovalStatus,
                      approved_by: str = "") -> bool:
        if model_id not in self._models:
            return False
        entry = self._models[model_id]
        entry.approval_status = status
        if approved_by:
            entry.approved_by = approved_by
            entry.approved_at = time.time()
        return True

    def list_all(self) -> List[ModelEntry]:
        return list(self._models.values())

    def count(self) -> int:
        return len(self._models)
