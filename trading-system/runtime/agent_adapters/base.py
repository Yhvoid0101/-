"""Agent Adapter 基础接口 — 8层统一接口。

6个方法：observe, decide, propose, explain, health, rollback
4种健康状态：HEALTHY, DEGRADED, BLOCKED, FAILED
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional

from ..contracts.base import ContractBase, ContractStatus


class HealthStatus(str, Enum):
    """健康状态。"""
    HEALTHY = "HEALTHY"       # 正常运行
    DEGRADED = "DEGRADED"     # 降级运行（有警告但不阻断）
    BLOCKED = "BLOCKED"       # 阻断（数据不可用/无真实来源）
    FAILED = "FAILED"         # 失败（模块异常）


@dataclass
class HealthCheck:
    """健康检查结果。"""
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    error_count: int = 0
    last_error: str = ""


class AgentAdapter:
    """Agent Adapter 抽象基类 — 8层统一接口。

    每层实现这个接口，包装现有模块。
    6个方法：
    - observe: 观察环境（输入契约 → 输出快照）
    - decide: 做出决策（输入提案 → 输出裁决）
    - propose: 提出候选（输入快照 → 输出候选）
    - explain: 解释决策（输入决策 → 输出解释）
    - health: 健康状态（输出健康检查）
    - rollback: 回滚到检查点（输入检查点 → 无输出）
    """

    LAYER_NAME: str = ""        # 层名（如 "L1_MARKET_DATA"）
    LAYER_NUMBER: int = 0       # 层号（1-8）

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._health: HealthCheck = HealthCheck()
        self._checkpoints: Dict[str, Any] = {}
        self._error_count: int = 0
        self._start_time: float = time.time()

    def observe(self, envelope: ContractBase) -> ContractBase:
        """观察环境。子类实现。"""
        raise NotImplementedError(f"{self.name}.observe() 未实现")

    def decide(self, proposal: ContractBase) -> ContractBase:
        """做出决策。子类实现。"""
        raise NotImplementedError(f"{self.name}.decide() 未实现")

    def propose(self, snapshot: ContractBase) -> ContractBase:
        """提出候选。子类实现。"""
        raise NotImplementedError(f"{self.name}.propose() 未实现")

    def explain(self, decision: ContractBase) -> Dict[str, Any]:
        """解释决策。子类实现。"""
        raise NotImplementedError(f"{self.name}.explain() 未实现")

    def health(self) -> HealthCheck:
        """健康状态。"""
        self._health.uptime_seconds = time.time() - self._start_time
        self._health.error_count = self._error_count
        return self._health

    def rollback(self, checkpoint_id: str) -> bool:
        """回滚到检查点。"""
        if checkpoint_id not in self._checkpoints:
            return False
        # 子类应重写此方法实现具体回滚逻辑
        return True

    def save_checkpoint(self, checkpoint_id: str, state: Any) -> None:
        """保存检查点。"""
        self._checkpoints[checkpoint_id] = state

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """获取检查点。"""
        return self._checkpoints.get(checkpoint_id)

    def _record_error(self, error: str) -> None:
        """记录错误。"""
        self._error_count += 1
        self._health.last_error = error
        self._health.error_count = self._error_count
        if self._health.status == HealthStatus.HEALTHY:
            self._health.status = HealthStatus.DEGRADED
        self._health.message = error

    def _set_blocked(self, reason: str) -> None:
        """设置 BLOCKED 状态。"""
        self._health.status = HealthStatus.BLOCKED
        self._health.message = reason

    def _set_healthy(self) -> None:
        """设置健康状态。"""
        self._health.status = HealthStatus.HEALTHY
        self._health.message = ""
