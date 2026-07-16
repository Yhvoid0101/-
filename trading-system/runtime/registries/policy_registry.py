"""Policy Registry — 策略版本管理。

铁律：版本、训练数据哈希、指标、审批状态。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ApprovalStatus(str, Enum):
    """审批状态流转。"""
    DRAFT = "draft"           # 草稿
    SUBMITTED = "submitted"   # 已提交
    REVIEWED = "reviewed"     # 已审核
    APPROVED = "approved"     # 已批准
    PROMOTED = "promoted"     # 已晋升生产
    RETIRED = "retired"       # 已退役
    REJECTED = "rejected"     # 已拒绝


@dataclass
class PolicyEntry:
    """策略版本条目。"""
    policy_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    gene_hash: str = ""               # 基因哈希
    train_data_hash: str = ""         # 训练数据哈希
    metrics: Dict[str, float] = field(default_factory=dict)
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    approved_by: str = ""
    approved_at: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["approval_status"] = self.approval_status.value
        return d


class PolicyRegistry:
    """策略注册表 — 版本管理。"""

    def __init__(self):
        self._policies: Dict[str, PolicyEntry] = {}  # policy_id -> entry

    def register(self, name: str, gene_hash: str, train_data_hash: str,
                 metrics: Dict[str, float] = None) -> PolicyEntry:
        """注册新策略版本。"""
        policy_id = hashlib.sha256(f"{name}|{gene_hash}|{time.time()}".encode()).hexdigest()[:16]
        entry = PolicyEntry(
            policy_id=policy_id,
            name=name,
            gene_hash=gene_hash,
            train_data_hash=train_data_hash,
            metrics=metrics or {},
        )
        self._policies[policy_id] = entry
        return entry

    def get(self, policy_id: str) -> Optional[PolicyEntry]:
        """查询策略。"""
        return self._policies.get(policy_id)

    def list_by_name(self, name: str) -> List[PolicyEntry]:
        """按名称列出所有版本。"""
        return [p for p in self._policies.values() if p.name == name]

    def update_status(self, policy_id: str, status: ApprovalStatus,
                      approved_by: str = "") -> bool:
        """更新审批状态。"""
        if policy_id not in self._policies:
            return False
        entry = self._policies[policy_id]
        entry.approval_status = status
        if approved_by:
            entry.approved_by = approved_by
            entry.approved_at = time.time()
        return True

    def compare_versions(self, id_a: str, id_b: str) -> Dict[str, Any]:
        """对比两个版本的指标。"""
        a = self._policies.get(id_a)
        b = self._policies.get(id_b)
        if not a or not b:
            return {}
        diff = {}
        for key in set(list(a.metrics.keys()) + list(b.metrics.keys())):
            va = a.metrics.get(key, 0.0)
            vb = b.metrics.get(key, 0.0)
            diff[key] = {"a": va, "b": vb, "delta": vb - va}
        return diff

    def list_all(self) -> List[PolicyEntry]:
        """列出所有策略。"""
        return list(self._policies.values())

    def count(self) -> int:
        """策略总数。"""
        return len(self._policies)
