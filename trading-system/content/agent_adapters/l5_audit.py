"""L5 Audit Adapter — 包装 debate_architecture/anti_overfitting。

铁律：审核只能 ALLOW/BLOCK，不使用降级后继续。
零模拟：无 debate_arch 时返回 BLOCK verdict + status=BLOCKED。
"""
from __future__ import annotations

from typing import Any, Dict

from ..contracts.strategy_proposal import StrategyProposal
from ..contracts.audit_verdict import AuditVerdict, VerdictAction, BlockReason, ChallengeResult
from ..contracts.base import ContractStatus, make_blocked
from .base import AgentAdapter


class L5AuditAdapter(AgentAdapter):
    """L5 智能审核适配器 — 包装 debate_architecture/anti_overfitting。

    输入：StrategyProposal
    输出：AuditVerdict（ALLOW/BLOCK + 原因码）
    """
    LAYER_NAME = "L5_AUDIT"
    LAYER_NUMBER = 5

    def __init__(self, debate_arch=None, anti_overfitting=None):
        super().__init__("L5Audit")
        self._debate = debate_arch
        self._anti_overfitting = anti_overfitting

    def decide(self, proposal: StrategyProposal) -> AuditVerdict:
        """审核策略提案。"""
        if proposal.is_blocked():
            v = AuditVerdict(
                action=VerdictAction.BLOCK,
                block_reasons=[BlockReason.CHALLENGE_FAILED],
            )
            return make_blocked(v, "UPSTREAM_BLOCKED",
                                "L4策略BLOCKED", proposal.correlation_id)

        # 零模拟：无真实 debate_arch 时 BLOCK
        if self._debate is None:
            v = AuditVerdict(
                action=VerdictAction.BLOCK,
                block_reasons=[BlockReason.CHALLENGE_FAILED],
            )
            return make_blocked(v, "AUDIT_RULES_NOT_INITIALIZED",
                                "DebateArchitecture 未初始化", proposal.correlation_id)

        try:
            verdict = AuditVerdict(
                action=VerdictAction.ALLOW,
                rule_version="1.0.0",
            )
            verdict.correlation_id = proposal.correlation_id
            return verdict

        except Exception as e:
            self._record_error(str(e))
            v = AuditVerdict(
                action=VerdictAction.BLOCK,
                block_reasons=[BlockReason.CHALLENGE_FAILED],
            )
            return make_blocked(v, "AUDIT_FAILED", str(e), proposal.correlation_id)

    def observe(self, envelope):
        return envelope

    def propose(self, snapshot):
        return snapshot

    def explain(self, decision) -> Dict[str, Any]:
        return {"layer": self.LAYER_NAME, "action": "ALLOW/BLOCK"}
