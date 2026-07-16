"""ExecutionIntent / ExecutionReport — 第7层执行管理层契约。

铁律：ExecutionReport完整回灌第4、6、8层，执行失败不可伪造成成交。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import ContractBase, make_blocked


class OrderAlgorithm(str, Enum):
    """执行算法。只使用交易所真实支持的订单类型。"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ICEBERG = "iceberg"


class OrderStatus(str, Enum):
    """订单状态。"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ExecutionIntent(ContractBase):
    """第7层输入：执行意图。"""
    symbol: str = ""
    side: str = ""                      # "buy" / "sell"
    order_type: OrderAlgorithm = OrderAlgorithm.MARKET
    quantity: float = 0.0
    price: Optional[float] = None       # 限价单价格
    time_in_force: str = "GTC"          # GTC/IOC/FOK
    client_order_id: str = ""           # 客户端订单ID（幂等）

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["order_type"] = self.order_type.value
        return d


@dataclass
class ExecutionReport(ContractBase):
    """第7层输出：执行报告。

    铁律：执行失败不可伪造成成交；真实成交、滑点、冲击、拒单必须如实记录。
    """
    symbol: str = ""
    order_id: str = ""                  # 交易所订单ID
    client_order_id: str = ""           # 客户端订单ID
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0        # 实际成交数量
    avg_fill_price: float = 0.0         # 平均成交价
    slippage: float = 0.0              # 滑点（基点）
    market_impact: float = 0.0         # 市场冲击（基点）
    fee_paid: float = 0.0              # 手续费
    rejection_reason: str = ""          # 拒单原因
    execution_latency_ms: float = 0.0  # 执行延迟（毫秒）

    # 真实性标记
    is_real_execution: bool = True      # 是否真实执行（False = 沙盘撮合）
    exchange: str = "gateio"

    def __post_init__(self):
        """拒单和失败不可伪造成成交。"""
        if self.status in (OrderStatus.REJECTED, OrderStatus.FAILED):
            if self.filled_quantity > 0:
                self.status = OrderStatus.FAILED
                self.error_code = "FILLED_ON_FAILED"
                self.error_message = "失败订单不应有成交量"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["status"] = self.status.value
        return d


def blocked_execution_failed(reason: str, correlation_id: str = "") -> ExecutionReport:
    """执行失败 BLOCKED 报告。"""
    rep = ExecutionReport(status=OrderStatus.FAILED, rejection_reason=reason)
    return make_blocked(rep, "EXECUTION_FAILED", reason, correlation_id)
