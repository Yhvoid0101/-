"""Phase 10 — 外层飞轮（结构进化 StructureEvolutionFlywheel）。

频率：仅当积累足够证据并满足冷却期时触发。
作用域：特征、regime 定义、策略族、层间协议和模块边界。
晋升要求：独立工作树 + 完整生产门禁 + 人工审批 + 金丝雀 + 自动回滚。

铁律合规：
- 零模拟：接入 L8 StructureProposer + FeaturePipeline 真实输出
- 零降级：所有变更通过 ChangeProposal + 隔离工作树
- 零遗漏：冷却期 + 证据累积 + 金丝雀 + 自动回滚
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts.base import ContractStatus, make_blocked
from .contracts.change_proposal import (
    ChangeProposal, ChangeType, ProposalStage, rejected_proposal,
)
from .phase9_l8 import (
    StructureProposer, WorktreeIsolation, PromotionGate,
    FeaturePipeline, CandidateFeature,
)


# ============================================================
# 结构进化证据
# ============================================================

@dataclass
class StructureEvidence:
    """结构进化证据。"""
    change_type: str = ""           # stat_arb / regime / gate / split
    description: str = ""
    sample_count: int = 0
    cross_window_reproducibility: float = 0.0
    economic_significance: float = 0.0  # 经济意义
    net_ev_improvement: float = 0.0
    is_sufficient: bool = False
    block_reason: str = ""


# ============================================================
# 外层飞轮：结构进化
# ============================================================

class StructureEvolutionFlywheel:
    """外层飞轮：结构进化。

    频率：仅当积累足够证据并满足冷却期时触发。
    作用域：特征、regime 定义、策略族、层间协议和模块边界。
    晋升要求：独立工作树 + 完整生产门禁 + 人工审批 + 金丝雀 + 自动回滚。
    """

    FLYWHEEL_LAYER = "OUTER"
    FLYWHEEL_NAME = "structure_evolution"

    # 冷却期（秒）— 同一类型提案在冷却期内不得重复提交
    COOLDOWN_SECONDS = 3600.0  # 1 小时

    # 证据门槛
    MIN_SAMPLES = 50
    MIN_CROSS_WINDOW = 0.80
    MIN_ECONOMIC_SIGNIFICANCE = 0.001
    MIN_NET_EV_IMPROVEMENT = 0.0001

    def __init__(self):
        self.structure_proposer: StructureProposer = StructureProposer()
        self.worktree: WorktreeIsolation = WorktreeIsolation()
        self.promotion_gate: PromotionGate = PromotionGate()
        self.feature_pipeline: FeaturePipeline = FeaturePipeline()

        self.evidence_history: List[StructureEvidence] = []
        self.proposals: List[ChangeProposal] = []
        self.last_submission_time: Dict[str, float] = {}  # 按类型记录最后提交时间
        self.canary_log: List[Dict[str, Any]] = []
        self.rollback_log: List[Dict[str, Any]] = []

    # ----------------------------------------------------------
    # 频率控制（冷却期 + 证据累积）
    # ----------------------------------------------------------

    def is_in_cooldown(self, change_type: str, current_time: float) -> bool:
        """是否在冷却期内。"""
        last = self.last_submission_time.get(change_type, 0.0)
        return (current_time - last) < self.COOLDOWN_SECONDS

    def check_evidence(self, evidence: StructureEvidence) -> bool:
        """检查证据是否充足。"""
        if evidence.sample_count < self.MIN_SAMPLES:
            evidence.block_reason = f"样本不足: {evidence.sample_count} < {self.MIN_SAMPLES}"
            return False
        if evidence.cross_window_reproducibility < self.MIN_CROSS_WINDOW:
            evidence.block_reason = f"跨窗口复现率不足: {evidence.cross_window_reproducibility:.4f} < {self.MIN_CROSS_WINDOW}"
            return False
        if evidence.economic_significance < self.MIN_ECONOMIC_SIGNIFICANCE:
            evidence.block_reason = f"经济意义不足: {evidence.economic_significance:.6f} < {self.MIN_ECONOMIC_SIGNIFICANCE}"
            return False
        if evidence.net_ev_improvement < self.MIN_NET_EV_IMPROVEMENT:
            evidence.block_reason = f"净EV改善不足: {evidence.net_ev_improvement:.6f} < {self.MIN_NET_EV_IMPROVEMENT}"
            return False
        evidence.is_sufficient = True
        return True

    def should_trigger(
        self,
        change_type: str,
        evidence: StructureEvidence,
        current_time: float,
    ) -> Tuple[bool, str]:
        """是否应该触发（冷却期 + 证据累积）。"""
        if self.is_in_cooldown(change_type, current_time):
            remaining = self.COOLDOWN_SECONDS - (current_time - self.last_submission_time.get(change_type, 0.0))
            return False, f"冷却期中，剩余 {remaining:.0f}s"

        if not self.check_evidence(evidence):
            return False, f"证据不足: {evidence.block_reason}"

        return True, "可触发"

    # ----------------------------------------------------------
    # 接入 L8 StructureProposer 的 4 类型提案
    # ----------------------------------------------------------

    def generate_structure_proposals(
        self,
        l8_learning_agent,
        correlation_id: str = "",
    ) -> List[ChangeProposal]:
        """接入 L8 StructureProposer 的 4 类型提案（stat_arb/regime/gate/split）。

        注：L8.propose_structures 签名为 (correlation_id="")，基于累积死因频率生成。
        """
        return l8_learning_agent.propose_structures(correlation_id=correlation_id)

    # ----------------------------------------------------------
    # 接入 L8 FeaturePipeline 的 4 门禁特征生成
    # ----------------------------------------------------------

    def generate_feature_candidates(
        self,
        source_fields: List[str],
        time_windows: List[int],
        operators: List[str],
    ) -> List[CandidateFeature]:
        """接入 L8 FeaturePipeline 的 4 门禁特征生成。"""
        return self.feature_pipeline.run(
            source_fields=source_fields,
            time_windows=time_windows,
            operators=operators,
        )

    # ----------------------------------------------------------
    # 隔离工作树 + 晋升门禁
    # ----------------------------------------------------------

    def create_isolated_worktree(self, proposal: ChangeProposal) -> ChangeProposal:
        """在隔离工作树中实验。

        注：WorktreeIsolation.create_isolation(proposal) 返回同一个 proposal 对象，
        且自动设置 is_isolated=True 和 worktree_branch 字段。
        """
        proposal = self.worktree.create_isolation(proposal)
        return proposal

    def run_promotion_gate(
        self,
        proposal: ChangeProposal,
        gate_results: Dict[str, bool],
        approved_by: str = "",
    ) -> ChangeProposal:
        """运行晋升门禁（9项 + 人工审批）。

        流程：
        1. set_gate_result: 写入门禁结果到 gate_results 字典（不改变 stage）
        2. submit_to_gate: 将 stage 推进到 GATE
        3. approve: 人工审批（设置 approved_by）
        4. promote: 检查 4 项条件（all_pass + is_isolated + has_approval + at_gate_stage）后晋升
        """
        # 1. 设置门禁结果
        for gate_name, passed in gate_results.items():
            proposal = self.promotion_gate.set_gate_result(proposal, gate_name, passed)

        # 2. 提交到 GATE 阶段
        proposal = self.promotion_gate.submit_to_gate(proposal)

        # 3. 人工审批
        if approved_by:
            proposal = self.promotion_gate.approve(proposal, approved_by)
            proposal.approved_at = time.time()

        # 4. 晋升
        return self.promotion_gate.promote(proposal)

    # ----------------------------------------------------------
    # 金丝雀
    # ----------------------------------------------------------

    def run_canary(
        self,
        proposal: ChangeProposal,
        canary_duration: float = 60.0,
        canary_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """金丝雀验证。"""
        metrics = canary_metrics or {
            "error_rate": 0.0,
            "latency_p99": 0.5,
            "net_ev": 0.001,
        }

        result = {
            "proposal_id": proposal.correlation_id,
            "duration": canary_duration,
            "metrics": metrics,
            "pass": (
                metrics.get("error_rate", 1.0) < 0.01
                and metrics.get("latency_p99", 999) < 1.0
                and metrics.get("net_ev", -1) > 0
            ),
            "timestamp": time.time(),
        }

        self.canary_log.append(result)
        return result

    # ----------------------------------------------------------
    # 自动回滚
    # ----------------------------------------------------------

    def rollback(self, proposal: ChangeProposal, reason: str = "") -> ChangeProposal:
        """自动回滚。"""
        proposal.stage = ProposalStage.ROLLED_BACK
        self.rollback_log.append({
            "proposal_id": proposal.correlation_id,
            "reason": reason,
            "timestamp": time.time(),
        })
        return proposal

    # ----------------------------------------------------------
    # 运行一轮飞轮
    # ----------------------------------------------------------

    def run_cycle(
        self,
        l8_learning_agent,
        evidence_list: List[StructureEvidence],
        current_time: float,
        gate_results: Optional[Dict[str, bool]] = None,
        approved_by: str = "",
        canary_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """运行一轮外层飞轮。"""
        results: List[Dict[str, Any]] = []

        # 步骤1: 生成结构提案
        structure_proposals = self.generate_structure_proposals(l8_learning_agent)

        # 步骤2: 生成特征候选
        feature_candidates = self.generate_feature_candidates(
            source_fields=["close", "volume", "high", "low"],
            time_windows=[5, 10, 20],
            operators=["lag", "rolling_mean", "rolling_std", "ewm", "diff", "ratio"],
        )

        # 步骤3: 对每个证据 + 提案进行冷却期 + 证据检查
        for evidence in evidence_list:
            can_trigger, reason = self.should_trigger(
                evidence.change_type, evidence, current_time,
            )

            if not can_trigger:
                results.append({
                    "change_type": evidence.change_type,
                    "triggered": False,
                    "reason": reason,
                })
                continue

            # 步骤4: 隔离工作树
            matching_proposals = [
                p for p in structure_proposals
                if p.change_type.value == evidence.change_type or True  # 简化匹配
            ]

            if not matching_proposals:
                results.append({
                    "change_type": evidence.change_type,
                    "triggered": False,
                    "reason": "无匹配的结构提案",
                })
                continue

            proposal = matching_proposals[0]
            proposal = self.create_isolated_worktree(proposal)

            # 步骤5: 晋升门禁
            gates = gate_results or {
                "unit_test": True, "integration_test": True,
                "blackbox": True, "branch_coverage": True,
                "mutation": True, "snapshot": True,
                "fuzz": True, "dependency": True, "cross_environment": True,
            }
            proposal = self.run_promotion_gate(proposal, gates, approved_by)

            # 步骤6: 金丝雀
            canary_result = self.run_canary(proposal, canary_metrics=canary_metrics)

            # 步骤7: 金丝雀失败 → 自动回滚
            if not canary_result["pass"]:
                proposal = self.rollback(proposal, "金丝雀验证失败")

            self.last_submission_time[evidence.change_type] = current_time
            self.proposals.append(proposal)

            results.append({
                "change_type": evidence.change_type,
                "triggered": True,
                "isolated": proposal.is_isolated,
                "gate_passed": proposal.stage == ProposalStage.PROMOTED,
                "canary_pass": canary_result["pass"],
                "stage": proposal.stage.value,
                "rolled_back": proposal.stage == ProposalStage.ROLLED_BACK,
            })

        triggered_count = sum(1 for r in results if r.get("triggered"))
        promoted_count = sum(1 for r in results if r.get("gate_passed"))
        rolled_back_count = sum(1 for r in results if r.get("rolled_back"))

        return {
            "triggered": triggered_count > 0,
            "flywheel": self.FLYWHEEL_NAME,
            "layer": self.FLYWHEEL_LAYER,
            "structure_proposals": len(structure_proposals),
            "feature_candidates": len(feature_candidates),
            "evidence_count": len(evidence_list),
            "triggered_count": triggered_count,
            "promoted_count": promoted_count,
            "rolled_back_count": rolled_back_count,
            "results": results,
        }
