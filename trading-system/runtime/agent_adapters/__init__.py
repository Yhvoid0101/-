"""Phase 1 Agent Adapter 包 — 8层统一接口。

铁律合规：
- 不改变交易逻辑：Adapter 包装现有模块
- SYNTHETIC/DEGRADED 路径输出 BLOCKED
"""
from .base import AgentAdapter, HealthStatus, HealthCheck
from .l1_market_data import L1MarketDataAdapter
from .l2_market_understanding import L2MarketUnderstandingAdapter
from .l3_opportunity import L3OpportunityAdapter
from .l4_strategy import L4StrategyAdapter
from .l5_audit import L5AuditAdapter
from .l6_risk import L6RiskAdapter
from .l7_execution import L7ExecutionAdapter
from .l8_learning import L8LearningAdapter

__all__ = [
    "AgentAdapter", "HealthStatus", "HealthCheck",
    "L1MarketDataAdapter", "L2MarketUnderstandingAdapter",
    "L3OpportunityAdapter", "L4StrategyAdapter",
    "L5AuditAdapter", "L6RiskAdapter",
    "L7ExecutionAdapter", "L8LearningAdapter",
]
