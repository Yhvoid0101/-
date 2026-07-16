"""Change Control Plane — 变更控制面。

流程：Proposal -> Shadow -> Challenge -> Gate -> Promote -> Monitor -> Rollback

铁律：
- 智能体只能提交 ChangeProposal，不能直接写生产配置或代码
- 审批通过后自动晋升到 Shadow
- Shadow 通过后晋升到生产
- 任何阶段可回滚
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..contracts.change_proposal import ChangeProposal, ProposalStage


@dataclass
class ChangeRequest:
    """变更请求 — 跟踪一个 ChangeProposal 的全流程。"""
    request_id: str = ""
    proposal: Dict[str, Any] = field(default_factory=dict)  # 序列化的 ChangeProposal
    current_stage: ProposalStage = ProposalStage.PROPOSED
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    shadow_results: Dict[str, Any] = field(default_factory=dict)
    challenge_results: Dict[str, Any] = field(default_factory=dict)
    gate_results: Dict[str, Any] = field(default_factory=dict)
    monitor_metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_reason: str = ""

    def advance_stage(self, new_stage: ProposalStage, notes: str = "") -> None:
        """推进阶段。"""
        self.stage_history.append({
            "from": self.current_stage.value,
            "to": new_stage.value,
            "timestamp": time.time(),
            "notes": notes,
        })
        self.current_stage = new_stage

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["current_stage"] = self.current_stage.value
        return d


class ChangeControlPlane:
    """变更控制面。

    管理所有 ChangeProposal 的生命周期。
    """

    def __init__(self):
        self._requests: Dict[str, ChangeRequest] = {}
        self._production_active: Dict[str, str] = {}  # target_component -> request_id

    def submit(self, proposal: ChangeProposal) -> ChangeRequest:
        """提交变更提案。

        铁律：智能体只能提交提案，不能直接写生产。
        """
        request = ChangeRequest(
            request_id=proposal.event_id,
            proposal=proposal.to_dict(),
            current_stage=ProposalStage.PROPOSED,
        )
        self._requests[proposal.event_id] = request
        return request

    def advance_to_shadow(self, request_id: str) -> bool:
        """推进到 Shadow 阶段。"""
        req = self._requests.get(request_id)
        if not req or req.current_stage != ProposalStage.PROPOSED:
            return False
        req.advance_stage(ProposalStage.SHADOW, "进入影子运行")
        return True

    def advance_to_challenge(self, request_id: str, shadow_results: Dict = None) -> bool:
        """推进到 Challenge 阶段。"""
        req = self._requests.get(request_id)
        if not req or req.current_stage != ProposalStage.SHADOW:
            return False
        if shadow_results:
            req.shadow_results = shadow_results
        req.advance_stage(ProposalStage.CHALLENGE, "进入挑战集测试")
        return True

    def advance_to_gate(self, request_id: str, challenge_results: Dict = None) -> bool:
        """推进到 Gate 阶段。"""
        req = self._requests.get(request_id)
        if not req or req.current_stage != ProposalStage.CHALLENGE:
            return False
        if challenge_results:
            req.challenge_results = challenge_results
        req.advance_stage(ProposalStage.GATE, "进入门禁检查")
        return True

    def promote(self, request_id: str, approved_by: str,
                target_component: str, gate_results: Dict = None) -> bool:
        """晋升到生产。

        铁律：需要人工批准 + 全门禁通过。
        """
        req = self._requests.get(request_id)
        if not req or req.current_stage != ProposalStage.GATE:
            return False
        if not approved_by:
            return False
        if gate_results:
            req.gate_results = gate_results
            if not all(gate_results.values()):
                return False  # 门禁未全部通过
        req.advance_stage(ProposalStage.PROMOTED, f"人工批准: {approved_by}")
        self._production_active[target_component] = request_id
        return True

    def monitor(self, request_id: str, metrics: Dict[str, Any]) -> bool:
        """监控晋升后的变更。"""
        req = self._requests.get(request_id)
        if not req or req.current_stage != ProposalStage.PROMOTED:
            return False
        req.monitor_metrics = metrics
        req.advance_stage(ProposalStage.MONITORING, "进入监控")
        return True

    def rollback(self, request_id: str, reason: str, target_component: str = "") -> bool:
        """回滚变更。"""
        req = self._requests.get(request_id)
        if not req:
            return False
        req.rollback_reason = reason
        req.advance_stage(ProposalStage.ROLLED_BACK, f"回滚: {reason}")
        if target_component and self._production_active.get(target_component) == request_id:
            del self._production_active[target_component]
        return True

    def reject(self, request_id: str, reason: str) -> bool:
        """拒绝提案。"""
        req = self._requests.get(request_id)
        if not req:
            return False
        req.advance_stage(ProposalStage.REJECTED, f"拒绝: {reason}")
        return True

    def get_request(self, request_id: str) -> Optional[ChangeRequest]:
        """获取变更请求。"""
        return self._requests.get(request_id)

    def list_requests(self, stage: ProposalStage = None) -> List[ChangeRequest]:
        """列出变更请求。"""
        if stage:
            return [r for r in self._requests.values() if r.current_stage == stage]
        return list(self._requests.values())

    def get_production_active(self) -> Dict[str, str]:
        """获取当前生产中的变更。"""
        return dict(self._production_active)

    def count(self) -> int:
        """变更请求总数。"""
        return len(self._requests)
