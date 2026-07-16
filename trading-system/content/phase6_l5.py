"""Phase 6 - 第5层智能审核智能体 (L5 AuditAgent)。

铁律合规：
- 零模拟：所有审核基于真实证据，不伪造挑战集结果
- 零降级：审核只能 ALLOW/BLOCK 二元决策，禁止"降级后继续"
- 零死角：9种阻断原因码全覆盖
- 零遗漏：上游 BLOCKED 必须传播，不允许 L5 改成 ALLOW
- 零纸面通过：所有规则必须有反事实测试和真实证据
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .contracts.audit_verdict import (
    AuditVerdict,
    BlockReason,
    ChallengeResult,
    VerdictAction,
    allow_verdict,
    block_verdict,
)
from .contracts.base import ContractBase, ContractStatus, make_blocked
from .contracts.market_state_snapshot import MarketRegime, MarketStateSnapshot
from .contracts.opportunity_candidate import OpportunityCandidate, OpportunityDirection
from .contracts.strategy_proposal import StrategyProposal, ValidationEvidence


# ============================================================================
# P6-T1: L5AuditAgent 二元决策 + 9种阻断原因码
# ============================================================================


class PropagationReason(str, Enum):
    """L5 传播专用原因码（不属于9种，仅用于上游传播场景）。"""
    UPSTREAM_PROPAGATED = "UPSTREAM_PROPAGATED"
    NO_PROPOSAL = "NO_PROPOSAL"


@dataclass
class AuditContext:
    """审核上下文 — 集中收集所有上游契约和辅助证据。"""
    proposal: Optional[StrategyProposal] = None
    candidate: Optional[OpportunityCandidate] = None
    market_state: Optional[MarketStateSnapshot] = None
    market_data: Optional[ContractBase] = None
    correlation_id: str = ""
    # 真实证据字段（由调用方填充，禁止 mock）
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    training_score: float = 0.0
    validation_score: float = 0.0
    correlated_strategy_pnl: List[float] = field(default_factory=list)
    direction_exposure: Dict[str, float] = field(default_factory=dict)
    gate_results: Dict[str, bool] = field(default_factory=dict)  # gate_name -> passed


class L5AuditAgent:
    """L5 智能审核智能体。

    二元决策：仅 ALLOW / BLOCK。
    9种阻断原因码（来自 BlockReason 枚举）：
        OOD_DETECTED, OVERFITTING, DATA_LEAKAGE, REGIME_UNKNOWN,
        STRATEGY_DECAY, CORRELATION_ANOMALY, GATE_CONFLICT,
        INSUFFICIENT_SAMPLE, CHALLENGE_FAILED
    """

    RULE_VERSION = "L5.v1.0.0"
    DEFAULT_RULE_TTL = 86400.0  # 24h

    def __init__(
        self,
        antigen_rules: Optional[List["AntigenRule"]] = None,
        auditors: Optional[List["BaseAuditor"]] = None,
        min_wilson_lower: float = 0.0,
        min_recent_trades: int = 10,
        max_pbo: float = 0.5,
        max_correlation: float = 0.85,
        max_direction_concentration: float = 0.8,
    ) -> None:
        self.antigen_rules: List["AntigenRule"] = antigen_rules or []
        self.auditors: List["BaseAuditor"] = auditors or self._default_auditors()
        self.min_wilson_lower = min_wilson_lower
        self.min_recent_trades = min_recent_trades
        self.max_pbo = max_pbo
        self.max_correlation = max_correlation
        self.max_direction_concentration = max_direction_concentration

    # -- public API ---------------------------------------------------------

    def audit(self, ctx: AuditContext) -> AuditVerdict:
        """对 L4 StrategyProposal 执行审核，返回 AuditVerdict。"""
        if ctx.correlation_id:
            cid = ctx.correlation_id
        else:
            cid = str(uuid.uuid4())

        # 1) 上游传播：proposal 缺失或 BLOCKED → 直接 BLOCK
        if ctx.proposal is None:
            v = block_verdict(
                [BlockReason.CHALLENGE_FAILED],
                "L5: 无 L4 StrategyProposal 输入",
                cid,
            )
            v.error_code = PropagationReason.NO_PROPOSAL.value
            v.rule_version = self.RULE_VERSION
            return v
        if ctx.proposal.is_blocked():
            v = block_verdict(
                [BlockReason.CHALLENGE_FAILED],
                f"L5: 上游 L4 BLOCKED 传播 - {ctx.proposal.error_code}: {ctx.proposal.error_message}",
                cid,
            )
            v.error_code = PropagationReason.UPSTREAM_PROPAGATED.value
            v.rule_version = self.RULE_VERSION
            return v
        # 2) 上游 L3 / L2 BLOCKED 传播
        if ctx.candidate is not None and ctx.candidate.is_blocked():
            v = block_verdict(
                [BlockReason.CHALLENGE_FAILED],
                f"L5: 上游 L3 BLOCKED 传播 - {ctx.candidate.error_code}",
                cid,
            )
            v.error_code = PropagationReason.UPSTREAM_PROPAGATED.value
            v.rule_version = self.RULE_VERSION
            return v
        if ctx.market_state is not None and ctx.market_state.is_blocked():
            v = block_verdict(
                [BlockReason.CHALLENGE_FAILED],
                f"L5: 上游 L2 BLOCKED 传播 - {ctx.market_state.error_code}",
                cid,
            )
            v.error_code = PropagationReason.UPSTREAM_PROPAGATED.value
            v.rule_version = self.RULE_VERSION
            return v

        # 3) 7类统一审核 + 抗原规则审核
        sub_verdicts: List["SubVerdict"] = []
        for auditor in self.auditors:
            sv = auditor.audit(ctx, self)
            sub_verdicts.append(sv)

        # 抗原规则触发
        rule_violated = self._check_antigen_rules(ctx)
        if rule_violated is not None:
            sub_verdicts.append(SubVerdict(
                auditor_name="AntigenRuleAuditor",
                action=VerdictAction.BLOCK,
                reason=BlockReason.CHALLENGE_FAILED,
                evidence=rule_violated,
            ))

        # 4) 最严格合并：任一 BLOCK → 最终 BLOCK
        any_block = any(sv.action == VerdictAction.BLOCK for sv in sub_verdicts)
        if any_block:
            reasons = [sv.reason for sv in sub_verdicts if sv.action == VerdictAction.BLOCK]
            # 去重
            seen = set()
            unique_reasons: List[BlockReason] = []
            for r in reasons:
                if r not in seen:
                    seen.add(r)
                    unique_reasons.append(r)
            evidences = "; ".join(
                f"[{sv.auditor_name}] {sv.evidence}"
                for sv in sub_verdicts if sv.action == VerdictAction.BLOCK
            )
            v = block_verdict(unique_reasons, evidences, cid)
            v.rule_version = self.RULE_VERSION
            v.rule_ttl = self.DEFAULT_RULE_TTL
            v.challenge_results = [
                ChallengeResult(
                    challenge_name=sv.auditor_name,
                    passed=False,
                    score=0.0,
                    details={"reason": sv.reason.value, "evidence": sv.evidence},
                )
                for sv in sub_verdicts if sv.action == VerdictAction.BLOCK
            ]
            return v

        # 5) 全部 ALLOW
        v = allow_verdict(self.RULE_VERSION, cid)
        v.rule_ttl = self.DEFAULT_RULE_TTL
        v.challenge_results = [
            ChallengeResult(
                challenge_name=sv.auditor_name,
                passed=True,
                score=1.0,
                details={"evidence": sv.evidence},
            )
            for sv in sub_verdicts
        ]
        return v

    # -- internal -----------------------------------------------------------

    def _default_auditors(self) -> List["BaseAuditor"]:
        return [
            OODAuditor(),
            OverfittingAuditor(),
            DataLeakageAuditor(),
            StateUnknownAuditor(),
            StrategyDecayAuditor(),
            CorrelationAuditor(),
            GateConflictAuditor(),
        ]

    def _check_antigen_rules(self, ctx: AuditContext) -> Optional[str]:
        """检查抗原规则是否触发。返回触发的规则描述，None 表示未触发。"""
        now = time.time()
        for rule in self.antigen_rules:
            # 过期或退役规则不触发
            if rule.status != AntigenRuleStatus.ACTIVE:
                continue
            if rule.is_expired(now):
                rule.status = AntigenRuleStatus.EXPIRED
                continue
            try:
                if rule.predicate(ctx):
                    return f"rule={rule.id} v{rule.version} - {rule.description}"
            except Exception as exc:  # 规则评估异常 → 视为阻断（保守）
                return f"rule={rule.id} error={exc}"
        return None


# ============================================================================
# P6-T3: 7类统一审核器
# ============================================================================


@dataclass
class SubVerdict:
    """子审核结果。"""
    auditor_name: str
    action: VerdictAction
    reason: BlockReason = BlockReason.CHALLENGE_FAILED
    evidence: str = ""


class BaseAuditor:
    """审核器基类。"""

    name: str = "BaseAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        raise NotImplementedError


class OODAuditor(BaseAuditor):
    name = "OODAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        if ctx.market_state is None:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OOD_DETECTED,
                              "L2 MarketStateSnapshot 缺失，无法评估 OOD")
        if ctx.market_state.is_ood:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OOD_DETECTED,
                              f"is_ood=True, ood_distance={ctx.market_state.evidence.ood_distance:.4f}")
        # OOD 距离阈值检查
        if ctx.market_state.evidence.ood_distance > 3.0:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OOD_DETECTED,
                              f"ood_distance={ctx.market_state.evidence.ood_distance:.4f} > 3.0")
        return SubVerdict(self.name, VerdictAction.ALLOW, evidence="OOD 距离在阈值内")


class OverfittingAuditor(BaseAuditor):
    name = "OverfittingAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        prop = ctx.proposal
        if prop is None:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OVERFITTING,
                              "无 StrategyProposal，无法评估过拟合")
        v = prop.validation
        # PBO 超阈值
        if v.pbo_score >= agent.max_pbo:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OVERFITTING,
                              f"pbo_score={v.pbo_score:.4f} >= {agent.max_pbo}")
        # walk-forward 未通过
        if not v.walk_forward_pass:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OVERFITTING,
                              "walk_forward_pass=False")
        # purged CV 未通过
        if not v.purged_cv_pass:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OVERFITTING,
                              "purged_cv_pass=False")
        # 训练-验证差异过大
        if ctx.training_score > 0 and ctx.validation_score > 0:
            ratio = ctx.training_score / max(ctx.validation_score, 1e-9)
            if ratio > 2.0:
                return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.OVERFITTING,
                                  f"train/val ratio={ratio:.2f} > 2.0")
        return SubVerdict(self.name, VerdictAction.ALLOW, evidence="过拟合检查通过")


class DataLeakageAuditor(BaseAuditor):
    name = "DataLeakageAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        prop = ctx.proposal
        if prop is None:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.DATA_LEAKAGE,
                              "无 StrategyProposal，无法评估数据泄漏")
        v = prop.validation
        # purged CV 未通过 = 数据泄漏嫌疑
        if not v.purged_cv_pass:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.DATA_LEAKAGE,
                              "purged_cv_pass=False → 数据泄漏嫌疑")
        # event-time 违反（lineage 过期 = 数据可能含未来信息）
        if prop.lineage.is_expired():
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.DATA_LEAKAGE,
                              "proposal lineage 已过期，event-time 违反")
        # 检查 candidate lineage
        if ctx.candidate is not None and ctx.candidate.lineage.is_expired():
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.DATA_LEAKAGE,
                              "candidate lineage 已过期，event-time 违反")
        # 检查 market_state lineage
        if ctx.market_state is not None and ctx.market_state.lineage.is_expired():
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.DATA_LEAKAGE,
                              "market_state lineage 已过期，event-time 违反")
        return SubVerdict(self.name, VerdictAction.ALLOW, evidence="数据泄漏检查通过")


class StateUnknownAuditor(BaseAuditor):
    name = "StateUnknownAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        if ctx.market_state is None:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.REGIME_UNKNOWN,
                              "L2 MarketStateSnapshot 缺失")
        # regime = UNKNOWN
        if ctx.market_state.regime == MarketRegime.UNKNOWN:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.REGIME_UNKNOWN,
                              "regime=UNKNOWN，状态未知不强行归类")
        # 高不确定性
        if ctx.market_state.uncertainty > 0.7:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.REGIME_UNKNOWN,
                              f"uncertainty={ctx.market_state.uncertainty:.4f} > 0.7")
        # 不稳定
        if not ctx.market_state.is_stable:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.REGIME_UNKNOWN,
                              "is_stable=False，regime 不稳定")
        return SubVerdict(self.name, VerdictAction.ALLOW, evidence="状态已知且稳定")


class StrategyDecayAuditor(BaseAuditor):
    name = "StrategyDecayAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        # 近期交易数不足 → 无法评估衰减，但属于样本问题，归到 INSUFFICIENT_SAMPLE
        if len(ctx.recent_trades) < agent.min_recent_trades:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.INSUFFICIENT_SAMPLE,
                              f"recent_trades={len(ctx.recent_trades)} < {agent.min_recent_trades}")
        # 近期胜率衰减：近 window 与全量比较
        recent = ctx.recent_trades[-agent.min_recent_trades:]
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        recent_winrate = wins / max(len(recent), 1)
        all_wins = sum(1 for t in ctx.recent_trades if t.get("pnl", 0) > 0)
        overall_winrate = all_wins / max(len(ctx.recent_trades), 1)
        # 近期表现显著衰减（差距 > 0.2）
        if recent_winrate < overall_winrate - 0.2:
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.STRATEGY_DECAY,
                              f"recent_winrate={recent_winrate:.4f} << overall={overall_winrate:.4f}")
        # 近期 DSR 低于阈值（用 pnl mean/std 近似）
        pnls = [t.get("pnl", 0.0) for t in recent]
        if pnls:
            mean_pnl = sum(pnls) / len(pnls)
            var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
            std_pnl = var_pnl ** 0.5
            dsr = mean_pnl / max(std_pnl, 1e-9)
            if dsr < 0.0:
                return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.STRATEGY_DECAY,
                                  f"近似 DSR={dsr:.4f} < 0")
        return SubVerdict(self.name, VerdictAction.ALLOW, evidence="策略衰减检查通过")


class CorrelationAuditor(BaseAuditor):
    name = "CorrelationAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        # 与已上线策略相关性
        if ctx.correlated_strategy_pnl:
            # 简单相关性：取最大绝对相关系数
            max_corr = max(abs(c) for c in ctx.correlated_strategy_pnl)
            if max_corr > agent.max_correlation:
                return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.CORRELATION_ANOMALY,
                                  f"max_corr={max_corr:.4f} > {agent.max_correlation}")
        # 方向暴露集中
        if ctx.direction_exposure:
            total = sum(ctx.direction_exposure.values()) or 1.0
            for direction, exposure in ctx.direction_exposure.items():
                concentration = exposure / total
                if concentration > agent.max_direction_concentration:
                    return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.CORRELATION_ANOMALY,
                                      f"direction={direction} concentration={concentration:.4f} > {agent.max_direction_concentration}")
        return SubVerdict(self.name, VerdictAction.ALLOW, evidence="相关性检查通过")


class GateConflictAuditor(BaseAuditor):
    name = "GateConflictAuditor"

    def audit(self, ctx: AuditContext, agent: L5AuditAgent) -> SubVerdict:
        if not ctx.gate_results:
            return SubVerdict(self.name, VerdictAction.ALLOW, evidence="无多 Gate 结果，无冲突")
        # 所有 Gate 都通过 → 无冲突
        if all(ctx.gate_results.values()):
            return SubVerdict(self.name, VerdictAction.ALLOW, evidence="所有 Gate 一致通过")
        # 所有 Gate 都未通过 → 一致 BLOCK（但归到 CHALLENGE_FAILED，不算冲突）
        if not any(ctx.gate_results.values()):
            return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.CHALLENGE_FAILED,
                              "所有 Gate 一致拒绝")
        # 部分 Pass 部分 Fail = 冲突 → BLOCK(GATE_CONFLICT)
        return SubVerdict(self.name, VerdictAction.BLOCK, BlockReason.GATE_CONFLICT,
                          f"Gate 结果冲突: {ctx.gate_results}")


# ============================================================================
# P6-T2: AntigenRule + AntigenRuleGenerator
# ============================================================================


class AntigenRuleStatus(str, Enum):
    """抗原规则状态。"""
    PROPOSED = "PROPOSED"        # 候选
    ACTIVE = "ACTIVE"            # 生效
    EXPIRED = "EXPIRED"          # TTL 过期
    DEPRECATED = "DEPRECATED"    # 误杀率超标退役
    REJECTED = "REJECTED"        # 不满足候选门禁
    RETIRED = "RETIRED"          # 反事实测试无效退役


@dataclass
class CounterfactualTest:
    """反事实测试结果。"""
    tested: bool = False
    passed: bool = False
    with_rule_loss: float = 0.0   # 触发规则时的损失
    without_rule_loss: float = 0.0  # 不触发规则时的损失
    details: str = ""


@dataclass
class AntigenRule:
    """抗原规则。

    每条规则携带：id, 版本, TTL, 来源证据, 支持度, 置信度, 误杀率, 反事实测试, 生效窗口。
    """
    id: str
    version: str = "1.0.0"
    description: str = ""
    # 来源证据
    source_evidence: str = ""
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 86400.0  # 默认 24h
    effective_window: Tuple[float, float] = (0.0, 0.0)
    # 候选门禁指标
    support: int = 0                # 支持度（失败笔数）
    confidence: float = 0.0         # 置信度
    false_positive_rate: float = 0.0  # 误杀率
    cross_window_count: int = 0     # 跨窗口复现次数
    # 反事实测试
    counterfactual: CounterfactualTest = field(default_factory=CounterfactualTest)
    # 状态
    status: AntigenRuleStatus = AntigenRuleStatus.PROPOSED
    # 规则谓词（运行时评估）
    predicate: Callable[[AuditContext], bool] = field(default=lambda ctx: False, repr=False)

    def is_expired(self, now: Optional[float] = None) -> bool:
        current = now if now is not None else time.time()
        return current > (self.created_at + self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "source_evidence": self.source_evidence,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "effective_window": self.effective_window,
            "support": self.support,
            "confidence": self.confidence,
            "false_positive_rate": self.false_positive_rate,
            "cross_window_count": self.cross_window_count,
            "counterfactual": {
                "tested": self.counterfactual.tested,
                "passed": self.counterfactual.passed,
                "with_rule_loss": self.counterfactual.with_rule_loss,
                "without_rule_loss": self.counterfactual.without_rule_loss,
                "details": self.counterfactual.details,
            },
            "status": self.status.value,
        }


@dataclass
class CandidateRuleSpec:
    """候选规则规格（来自失败簇）。"""
    description: str
    source_evidence: str
    support: int
    confidence: float
    false_positive_rate: float
    cross_window_count: int
    with_rule_loss: float
    without_rule_loss: float
    predicate: Callable[[AuditContext], bool]


class AntigenRuleGenerator:
    """抗原规则生成器。

    从失败簇生成候选抗原规则；候选必须满足：
    - support >= min_support (默认 10)
    - confidence >= min_confidence (默认 0.7)
    - false_positive_rate <= max_false_positive (默认 0.05)
    - cross_window_count >= min_cross_window (默认 2)
    - 反事实测试通过（with_rule_loss < without_rule_loss）
    """

    def __init__(
        self,
        min_support: int = 10,
        min_confidence: float = 0.7,
        max_false_positive: float = 0.05,
        min_cross_window: int = 2,
        default_ttl: float = 86400.0,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_false_positive = max_false_positive
        self.min_cross_window = min_cross_window
        self.default_ttl = default_ttl

    def generate(self, spec: CandidateRuleSpec) -> AntigenRule:
        """从候选规格生成抗原规则。不满足门禁 → REJECTED。"""
        rule_id = f"rule_{hashlib.sha256(spec.description.encode()).hexdigest()[:12]}"
        now = time.time()

        # 反事实测试结果
        cf_passed = spec.with_rule_loss < spec.without_rule_loss
        cf = CounterfactualTest(
            tested=True,
            passed=cf_passed,
            with_rule_loss=spec.with_rule_loss,
            without_rule_loss=spec.without_rule_loss,
            details=(
                f"with_rule_loss={spec.with_rule_loss:.4f} "
                f"without_rule_loss={spec.without_rule_loss:.4f}"
            ),
        )

        rule = AntigenRule(
            id=rule_id,
            version="1.0.0",
            description=spec.description,
            source_evidence=spec.source_evidence,
            created_at=now,
            ttl_seconds=self.default_ttl,
            effective_window=(now, now + self.default_ttl),
            support=spec.support,
            confidence=spec.confidence,
            false_positive_rate=spec.false_positive_rate,
            cross_window_count=spec.cross_window_count,
            counterfactual=cf,
            status=AntigenRuleStatus.PROPOSED,
            predicate=spec.predicate,
        )

        # 候选门禁检查
        if spec.support < self.min_support:
            rule.status = AntigenRuleStatus.REJECTED
            rule.source_evidence += f" | REJECTED: support={spec.support} < {self.min_support}"
            return rule
        if spec.confidence < self.min_confidence:
            rule.status = AntigenRuleStatus.REJECTED
            rule.source_evidence += f" | REJECTED: confidence={spec.confidence:.4f} < {self.min_confidence}"
            return rule
        if spec.false_positive_rate > self.max_false_positive:
            rule.status = AntigenRuleStatus.REJECTED
            rule.source_evidence += f" | REJECTED: fp_rate={spec.false_positive_rate:.4f} > {self.max_false_positive}"
            return rule
        if spec.cross_window_count < self.min_cross_window:
            rule.status = AntigenRuleStatus.REJECTED
            rule.source_evidence += f" | REJECTED: cross_window={spec.cross_window_count} < {self.min_cross_window}"
            return rule
        if not cf_passed:
            rule.status = AntigenRuleStatus.RETIRED
            rule.source_evidence += " | RETIRED: 反事实测试无效（with_rule_loss >= without_rule_loss）"
            return rule

        # 通过门禁 → ACTIVE
        rule.status = AntigenRuleStatus.ACTIVE
        return rule

    def check_and_retire(self, rule: AntigenRule, now: Optional[float] = None) -> AntigenRule:
        """检查并自动退役过期/失效规则。"""
        current = now if now is not None else time.time()
        if rule.status != AntigenRuleStatus.ACTIVE:
            return rule
        # TTL 过期 → EXPIRED
        if rule.is_expired(current):
            rule.status = AntigenRuleStatus.EXPIRED
            return rule
        # 误杀率超标 → DEPRECATED
        if rule.false_positive_rate > self.max_false_positive:
            rule.status = AntigenRuleStatus.DEPRECATED
            return rule
        return rule


# ============================================================================
# 验证工具
# ============================================================================


def make_audit_context(
    proposal: Optional[StrategyProposal] = None,
    candidate: Optional[OpportunityCandidate] = None,
    market_state: Optional[MarketStateSnapshot] = None,
    **kwargs: Any,
) -> AuditContext:
    """构造 AuditContext 的便捷工厂。"""
    return AuditContext(
        proposal=proposal,
        candidate=candidate,
        market_state=market_state,
        **kwargs,
    )
