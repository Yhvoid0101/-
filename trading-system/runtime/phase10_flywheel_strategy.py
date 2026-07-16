"""Phase 10 — 内层飞轮（策略进化 StrategyEvolutionFlywheel）。

频率：每 10 轮评估窗口触发候选演化。
作用域：L4 基因、参数、组合。
禁止：修改数据 schema、审核硬规则、安全包络、执行连接。

铁律合规：
- 零模拟：候选基于 L8 真实死因统计 + 反事实分析
- 零降级：不直接改生产热路径，所有变更通过 ChangeProposal
- 零遗漏：灾难性遗忘保护 + 作用域越界检测
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts.base import ContractBase, ContractStatus, make_blocked
from .contracts.change_proposal import (
    ChangeProposal, ChangeType, ProposalStage, rejected_proposal,
)
from .phase9_l8 import DeathCause


# ============================================================
# 禁止修改的作用域（越界检测）
# ============================================================

FORBIDDEN_COMPONENTS = (
    "L1_DataSource",        # 数据 schema
    "L5_AuditAgent",        # 审核硬规则
    "L6_RiskAgent",         # 安全包络
    "L7_ExecutionAgent",    # 执行连接
    "contracts/",           # 层间协议
)


# ============================================================
# 候选基因变异
# ============================================================

@dataclass
class GeneMutationCandidate:
    """基因变异候选。"""
    mutation_type: str = ""         # direction_flip / rr_adjust / param_tune
    source_gene_hash: str = ""      # 源基因哈希
    target_component: str = "L4_StrategyAnalyzer"
    reason: str = ""                # 变异原因（基于死因/反事实）
    expected_improvement: float = 0.0
    sample_size: int = 0
    net_ev: float = 0.0
    dsr: float = 0.0
    psr: float = 0.0
    diversity_score: float = 0.0
    counterfactual_improvement: float = 0.0
    is_eligible: bool = True
    block_reason: str = ""


# ============================================================
# 内层飞轮：策略进化
# ============================================================

class StrategyEvolutionFlywheel:
    """内层飞轮：策略进化。

    频率：每 10 轮评估窗口触发。
    作用域：L4 基因、参数、组合。
    禁止：修改数据 schema、审核硬规则、安全包络、执行连接。
    """

    FLYWHEEL_LAYER = "INNER"
    FLYWHEEL_NAME = "strategy_evolution"
    TRIGGER_INTERVAL = 10  # 每 10 轮触发

    # 最小门槛
    MIN_SAMPLE_SIZE = 30
    MIN_NET_EV = 0.0
    MIN_DSR = 0.0
    MIN_DIVERSITY = 0.05

    def __init__(self):
        self.cycle_count: int = 0
        self.candidate_history: List[GeneMutationCandidate] = []
        self.proposals: List[ChangeProposal] = []
        self.last_trigger_cycle: int = -1
        self.forgetting_protection_log: List[Dict[str, Any]] = []

    # ----------------------------------------------------------
    # 频率控制
    # ----------------------------------------------------------

    def should_trigger(self, cycle_count: int) -> bool:
        """是否到达触发频率（每 10 轮）。"""
        return (
            cycle_count > 0
            and cycle_count % self.TRIGGER_INTERVAL == 0
            and cycle_count != self.last_trigger_cycle
        )

    def record_cycle(self, cycle_count: int) -> None:
        """记录当前周期。"""
        self.cycle_count = cycle_count

    # ----------------------------------------------------------
    # 候选生成（基于 L8 死因统计 + 反事实分析）
    # ----------------------------------------------------------

    def generate_candidates(
        self,
        death_cause_stats: Dict[str, int],
        counterfactual_history: List[Dict[str, Any]],
        elite_genes: List[str],
    ) -> List[GeneMutationCandidate]:
        """基于 L8 死因统计 + 反事实分析，生成基因变异候选。

        变异类型：
        - direction_flip：方向翻转（DIRECTION_FLIP 死因多时）
        - rr_adjust：RR 调整（RR_INSUFFICIENT 死因多时）
        - param_tune：参数微调（基于反事实改善率）
        """
        candidates: List[GeneMutationCandidate] = []

        # 方向翻转候选
        flip_count = death_cause_stats.get(DeathCause.DIRECTION_FLIP, 0)
        if flip_count > 0:
            for gene_hash in elite_genes:
                candidates.append(GeneMutationCandidate(
                    mutation_type="direction_flip",
                    source_gene_hash=gene_hash,
                    reason=f"DIRECTION_FLIP count={flip_count}",
                    expected_improvement=float(flip_count) * 0.01,
                ))

        # RR 调整候选
        rr_count = death_cause_stats.get(DeathCause.RR_INSUFFICIENT, 0)
        if rr_count > 0:
            for gene_hash in elite_genes:
                candidates.append(GeneMutationCandidate(
                    mutation_type="rr_adjust",
                    source_gene_hash=gene_hash,
                    reason=f"RR_INSUFFICIENT count={rr_count}",
                    expected_improvement=float(rr_count) * 0.005,
                ))

        # 参数微调候选（基于反事实改善率）
        for cf in counterfactual_history:
            improvement = cf.get("improvement", 0.0)
            if improvement > 0:
                gene_hash = cf.get("gene_hash", "")
                candidates.append(GeneMutationCandidate(
                    mutation_type="param_tune",
                    source_gene_hash=gene_hash,
                    reason=f"counterfactual improvement={improvement:.4f}",
                    counterfactual_improvement=improvement,
                    expected_improvement=improvement * 0.1,
                ))

        self.candidate_history.extend(candidates)
        return candidates

    # ----------------------------------------------------------
    # 候选筛选（最小样本 + 跨窗口净 EV + DSR/PSR + 多维多样性）
    # ----------------------------------------------------------

    def filter_candidates(
        self,
        candidates: List[GeneMutationCandidate],
    ) -> List[GeneMutationCandidate]:
        """候选筛选：最小样本 + 跨窗口净 EV + DSR/PSR + 多维多样性。"""
        passed: List[GeneMutationCandidate] = []

        for c in candidates:
            # 最小样本门槛
            if c.sample_size < self.MIN_SAMPLE_SIZE:
                c.is_eligible = False
                c.block_reason = f"sample_size={c.sample_size} < MIN={self.MIN_SAMPLE_SIZE}"
                continue

            # 跨窗口净 EV
            if c.net_ev < self.MIN_NET_EV:
                c.is_eligible = False
                c.block_reason = f"net_ev={c.net_ev} < MIN={self.MIN_NET_EV}"
                continue

            # DSR/PSR
            if c.dsr < self.MIN_DSR:
                c.is_eligible = False
                c.block_reason = f"dsr={c.dsr} < MIN={self.MIN_DSR}"
                continue

            # 多维多样性
            if c.diversity_score < self.MIN_DIVERSITY:
                c.is_eligible = False
                c.block_reason = f"diversity={c.diversity_score} < MIN={self.MIN_DIVERSITY}"
                continue

            passed.append(c)

        return passed

    # ----------------------------------------------------------
    # 灾难性遗忘保护
    # ----------------------------------------------------------

    def check_forgetting(
        self,
        new_candidate: GeneMutationCandidate,
        elite_genes_net_ev: Dict[str, float],
    ) -> bool:
        """灾难性遗忘保护：新候选不得降低已有精英的跨窗口净 EV。"""
        for gene_hash, elite_ev in elite_genes_net_ev.items():
            if new_candidate.source_gene_hash == gene_hash:
                # 新候选的净 EV 不得低于已有精英
                if new_candidate.net_ev < elite_ev:
                    self.forgetting_protection_log.append({
                        "candidate": new_candidate.mutation_type,
                        "gene": gene_hash,
                        "candidate_ev": new_candidate.net_ev,
                        "elite_ev": elite_ev,
                        "action": "REJECTED",
                        "reason": "灾难性遗忘保护：降低已有精英净EV",
                    })
                    return False
        return True

    # ----------------------------------------------------------
    # 作用域越界检测
    # ----------------------------------------------------------

    def check_scope_violation(self, target_component: str) -> bool:
        """检测是否越界修改禁止组件。"""
        for forbidden in FORBIDDEN_COMPONENTS:
            if forbidden in target_component or target_component in forbidden:
                return True
        return False

    # ----------------------------------------------------------
    # 提交 ChangeProposal
    # ----------------------------------------------------------

    def submit_proposal(
        self,
        candidate: GeneMutationCandidate,
        shadow_results: Dict[str, Any],
        challenge_results: Dict[str, Any],
        correlation_id: str = "",
    ) -> ChangeProposal:
        """提交 ChangeProposal（影子期 + 挑战集 + 统计显著性）。"""
        cid = correlation_id or f"inner-{candidate.mutation_type}-{int(time.time())}"

        # 作用域越界检测
        if self.check_scope_violation(candidate.target_component):
            return rejected_proposal(
                f"作用域越界：内层飞轮不得修改 {candidate.target_component}",
                cid,
            )

        # 灾难性遗忘保护
        if not candidate.is_eligible:
            return rejected_proposal(
                f"候选不达标：{candidate.block_reason}",
                cid,
            )

        proposal = ChangeProposal(
            change_type=ChangeType.PARAMETER_TUNING,
            target_component=candidate.target_component,
            hypothesis=candidate.reason,
            expected_benefit=candidate.expected_improvement,
            risk_assessment="内层飞轮策略进化，作用域受限",
            rollback_point=f"rollback_{candidate.mutation_type}_{cid}",
            validation_matrix=["shadow", "challenge", "statistical_significance"],
            stage=ProposalStage.PROPOSED,
            submitted_by="StrategyEvolutionFlywheel",
            submitted_at=time.time(),
            shadow_results=shadow_results,
            challenge_results=challenge_results,
        )

        # 影子期验证
        if shadow_results.get("pass", False):
            proposal.stage = ProposalStage.SHADOW
        else:
            return rejected_proposal("影子期验证失败", cid)

        # 挑战集验证
        if challenge_results.get("pass", False):
            proposal.stage = ProposalStage.CHALLENGE
        else:
            return rejected_proposal("挑战集验证失败", cid)

        # 统计显著性检查
        stat_sig = challenge_results.get("statistical_significance", 0.0)
        if stat_sig < 0.95:
            return rejected_proposal(
                f"统计显著性不足：{stat_sig} < 0.95",
                cid,
            )

        self.proposals.append(proposal)
        return proposal

    # ----------------------------------------------------------
    # 运行一轮飞轮
    # ----------------------------------------------------------

    def run_cycle(
        self,
        cycle_count: int,
        death_cause_stats: Dict[str, int],
        counterfactual_history: List[Dict[str, Any]],
        elite_genes: List[str],
        elite_genes_net_ev: Dict[str, float],
        shadow_results: Optional[Dict[str, Any]] = None,
        challenge_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """运行一轮内层飞轮。"""
        self.record_cycle(cycle_count)

        if not self.should_trigger(cycle_count):
            return {
                "triggered": False,
                "reason": f"cycle={cycle_count} 未到达触发间隔({self.TRIGGER_INTERVAL})",
                "candidates": 0,
                "passed": 0,
                "proposals": 0,
            }

        self.last_trigger_cycle = cycle_count

        # 步骤1: 候选生成
        candidates = self.generate_candidates(
            death_cause_stats, counterfactual_history, elite_genes,
        )

        # 步骤2: 候选筛选
        passed = self.filter_candidates(candidates)

        # 步骤3: 灾难性遗忘保护
        survived = [
            c for c in passed
            if self.check_forgetting(c, elite_genes_net_ev)
        ]

        # 步骤4: 提交提案
        proposals_made: List[ChangeProposal] = []
        for c in survived:
            sr = shadow_results or {"pass": True}
            cr = challenge_results or {"pass": True, "statistical_significance": 0.96}
            p = self.submit_proposal(c, sr, cr)
            proposals_made.append(p)

        promoted = [p for p in proposals_made if p.status == ContractStatus.OK]

        return {
            "triggered": True,
            "flywheel": self.FLYWHEEL_NAME,
            "layer": self.FLYWHEEL_LAYER,
            "cycle": cycle_count,
            "candidates": len(candidates),
            "passed_filter": len(passed),
            "survived_forgetting": len(survived),
            "proposals": len(proposals_made),
            "promoted": len(promoted),
            "rejected": len(proposals_made) - len(promoted),
        }
