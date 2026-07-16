"""AuditVerdict — 第5层智能审核层输出契约。

铁律：审核只能 ALLOW/BLOCK，不使用"降级后继续"掩盖未知风险。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .base import ContractBase, make_blocked


class VerdictAction(str, Enum):
    """审核裁决动作。只有 ALLOW 和 BLOCK。"""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class BlockReason(str, Enum):
    """阻断原因码。"""
    OVERFITTING = "OVERFITTING"               # 过拟合
    OOD_DETECTED = "OOD_DETECTED"             # OOD 检测
    DATA_LEAKAGE = "DATA_LEAKAGE"             # 数据泄漏
    REGIME_UNKNOWN = "REGIME_UNKNOWN"         # 状态未知
    STRATEGY_DECAY = "STRATEGY_DECAY"         # 策略衰减
    CORRELATION_ANOMALY = "CORRELATION_ANOMALY"  # 异常相关性
    GATE_CONFLICT = "GATE_CONFLICT"           # Gate 冲突
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"  # 样本不足
    CHALLENGE_FAILED = "CHALLENGE_FAILED"     # 挑战集未通过


@dataclass
class ChallengeResult:
    """挑战集结果。"""
    challenge_name: str = ""
    passed: bool = False
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditVerdict(ContractBase):
    """第5层输出：审核裁决。"""
    action: VerdictAction = VerdictAction.BLOCK
    block_reasons: List[BlockReason] = field(default_factory=list)
    overfitting_evidence: Dict[str, float] = field(default_factory=dict)
    ood_evidence: Dict[str, float] = field(default_factory=dict)
    challenge_results: List[ChallengeResult] = field(default_factory=list)
    rule_version: str = ""             # 审核规则版本
    rule_ttl: float = 86400.0          # 规则TTL（秒）

    def __post_init__(self):
        if self.action == VerdictAction.BLOCK and not self.block_reasons:
            self.block_reasons = [BlockReason.CHALLENGE_FAILED]

    def is_allowed(self) -> bool:
        return self.action == VerdictAction.ALLOW and not self.is_blocked()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["action"] = self.action.value
        d["block_reasons"] = [r.value for r in self.block_reasons]
        return d


def block_verdict(reasons: List[BlockReason], evidence: str, correlation_id: str = "") -> AuditVerdict:
    """生成 BLOCK 裁决。"""
    v = AuditVerdict(action=VerdictAction.BLOCK, block_reasons=reasons)
    v.error_message = evidence
    if correlation_id:
        v.correlation_id = correlation_id
    return v


def allow_verdict(rule_version: str, correlation_id: str = "") -> AuditVerdict:
    """生成 ALLOW 裁决。"""
    v = AuditVerdict(action=VerdictAction.ALLOW, rule_version=rule_version)
    if correlation_id:
        v.correlation_id = correlation_id
    return v
