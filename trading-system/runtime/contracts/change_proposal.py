"""ChangeProposal — 结构变更提案契约。

铁律：结构提案只能在隔离分支/工作树中实验，通过全部门禁后人工批准晋升。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .base import ContractBase, make_blocked


class ProposalStage(str, Enum):
    """提案阶段流程：Proposal -> Shadow -> Challenge -> Gate -> Promote -> Monitor -> Rollback。"""
    PROPOSED = "proposed"       # 刚提交
    SHADOW = "shadow"           # 影子运行中
    CHALLENGE = "challenge"     # 挑战集测试中
    GATE = "gate"               # 门禁检查中
    PROMOTED = "promoted"       # 已晋升生产
    MONITORING = "monitoring"   # 监控中
    ROLLED_BACK = "rolled_back" # 已回滚
    REJECTED = "rejected"       # 已拒绝


class ChangeType(str, Enum):
    """变更类型。"""
    FEATURE_ADDITION = "feature_addition"        # 新增特征
    REGIME_DEFINITION = "regime_definition"       # 新 regime 定义
    STRATEGY_FAMILY = "strategy_family"           # 新策略族
    GATE_ADDITION = "gate_addition"               # 新 Gate
    INTERFACE_SPLIT = "interface_split"           # 接口拆分
    PARAMETER_TUNING = "parameter_tuning"         # 参数调优
    SCHEMA_CHANGE = "schema_change"               # 数据 schema 变更


@dataclass
class ChangeProposal(ContractBase):
    """结构变更提案。

    智能体只能提交提案，不能直接写生产配置或代码。
    """
    change_type: ChangeType = ChangeType.PARAMETER_TUNING
    target_component: str = ""          # 目标组件
    hypothesis: str = ""                # 假设
    patch_scope: str = ""               # 补丁范围
    expected_benefit: float = 0.0       # 预期收益
    risk_assessment: str = ""           # 风险评估
    rollback_point: str = ""            # 回滚点
    validation_matrix: List[str] = field(default_factory=list)  # 验证矩阵

    # 流程控制
    stage: ProposalStage = ProposalStage.PROPOSED
    submitted_by: str = ""              # 提交者（智能体ID）
    submitted_at: float = 0.0           # 提交时间
    approved_by: str = ""               # 审批者（人工）
    approved_at: float = 0.0            # 审批时间

    # 验证证据
    shadow_results: Dict[str, Any] = field(default_factory=dict)
    challenge_results: Dict[str, Any] = field(default_factory=dict)
    gate_results: Dict[str, Any] = field(default_factory=dict)

    # 工作树隔离
    worktree_branch: str = ""           # 隔离工作树分支
    is_isolated: bool = False           # 是否在隔离环境中

    def can_promote(self) -> bool:
        """是否可以晋升到生产。需要人工批准 + 全门禁通过。"""
        return (
            self.stage == ProposalStage.GATE
            and self.approved_by != ""
            and self.is_isolated
            and all(self.gate_results.values())
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["change_type"] = self.change_type.value
        d["stage"] = self.stage.value
        return d


def rejected_proposal(reason: str, correlation_id: str = "") -> ChangeProposal:
    """生成被拒绝的提案。"""
    p = ChangeProposal(stage=ProposalStage.REJECTED)
    return make_blocked(p, "PROPOSAL_REJECTED", reason, correlation_id)
