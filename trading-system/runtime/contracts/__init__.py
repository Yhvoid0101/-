"""Phase 1 版本化契约包 — 9类契约 + Schema Registry。

铁律合规：
- 零模拟：所有契约携带数据谱系，SYNTHETIC 数据输出 BLOCKED
- 零降级：数据不可用输出 BLOCKED，不允许 silent fallback
- 幂等可重放：每个契约有唯一 event_id，支持重放
- BLOCKED 不被吞掉：层故障传播到审计链
"""
from .base import (
    ContractBase,
    ContractStatus,
    DataLineage,
    Severity,
    make_blocked,
)
from .market_data_envelope import (
    MarketDataEnvelope,
    DataQuality,
    blocked_synthetic,
    blocked_expired,
)
from .market_state_snapshot import (
    MarketStateSnapshot,
    MarketRegime,
    RegimeEvidence,
)
from .opportunity_candidate import (
    OpportunityCandidate,
    OpportunityDirection,
    EVBreakdown,
    blocked_no_opportunity,
)
from .strategy_proposal import (
    StrategyProposal,
    GeneSpec,
    ValidationEvidence,
    blocked_invalid_strategy,
)
from .audit_verdict import (
    AuditVerdict,
    VerdictAction,
    BlockReason,
    ChallengeResult,
    block_verdict,
    allow_verdict,
)
from .risk_budget import (
    RiskBudget,
    SafetyEnvelope,
    blocked_kill_switch,
)
from .execution_contracts import (
    ExecutionIntent,
    ExecutionReport,
    OrderAlgorithm,
    OrderStatus,
    blocked_execution_failed,
)
from .learning_event import (
    LearningEvent,
    CounterfactualAnalysis,
    LayerResponsibility,
)
from .change_proposal import (
    ChangeProposal,
    ProposalStage,
    ChangeType,
    rejected_proposal,
)
from .schema_registry import (
    register,
    get_schema,
    check_compatibility,
    serialize,
    deserialize,
    list_registered,
    register_all,
)

__all__ = [
    # 基础
    "ContractBase", "ContractStatus", "DataLineage", "Severity", "make_blocked",
    # L1
    "MarketDataEnvelope", "DataQuality", "blocked_synthetic", "blocked_expired",
    # L2
    "MarketStateSnapshot", "MarketRegime", "RegimeEvidence",
    # L3
    "OpportunityCandidate", "OpportunityDirection", "EVBreakdown", "blocked_no_opportunity",
    # L4
    "StrategyProposal", "GeneSpec", "ValidationEvidence", "blocked_invalid_strategy",
    # L5
    "AuditVerdict", "VerdictAction", "BlockReason", "ChallengeResult", "block_verdict", "allow_verdict",
    # L6
    "RiskBudget", "SafetyEnvelope", "blocked_kill_switch",
    # L7
    "ExecutionIntent", "ExecutionReport", "OrderAlgorithm", "OrderStatus", "blocked_execution_failed",
    # L8
    "LearningEvent", "CounterfactualAnalysis", "LayerResponsibility",
    # 变更
    "ChangeProposal", "ProposalStage", "ChangeType", "rejected_proposal",
    # Registry
    "register", "get_schema", "check_compatibility", "serialize", "deserialize", "list_registered", "register_all",
]
