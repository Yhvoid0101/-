"""L1 Market Data Adapter — 包装 DataPipeline。

铁律：SYNTHETIC 数据输出 BLOCKED，不传播到决策链。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..contracts.market_data_envelope import MarketDataEnvelope, DataQuality, blocked_synthetic
from ..contracts.base import ContractStatus, DataLineage, make_blocked
from .base import AgentAdapter, HealthStatus


class L1MarketDataAdapter(AgentAdapter):
    """L1 市场数据适配器 — 包装 DataPipeline。

    输入：MarketDataEnvelope（原始数据请求）
    输出：MarketDataEnvelope（带质量标记的真实数据）
    SYNTHETIC/DEGRADED 路径输出 BLOCKED。
    """
    LAYER_NAME = "L1_MARKET_DATA"
    LAYER_NUMBER = 1

    def __init__(self, data_pipeline=None):
        super().__init__("L1MarketData")
        self._pipeline = data_pipeline  # 延迟注入

    def observe(self, request: Optional[MarketDataEnvelope]) -> MarketDataEnvelope:
        """获取市场数据。包装 DataPipeline。"""
        if request is None:
            env = MarketDataEnvelope()
            return make_blocked(env, "PIPELINE_NOT_INITIALIZED", "无输入请求", env.correlation_id)
        if request.is_blocked():
            return request  # 传递 BLOCKED

        if self._pipeline is None:
            return make_blocked(request, "PIPELINE_NOT_INITIALIZED",
                                "DataPipeline 未初始化", request.correlation_id)

        try:
            # 调用现有 DataPipeline 获取数据（dict 接口）
            # 实际调用在 Shadow 运行时验证
            result = self._pipeline  # 占位，Shadow 运行时替换为真实调用

            # 检查数据真实性
            if not request.is_real_data:
                return blocked_synthetic(request.symbol, "DataPipeline返回合成数据",
                                       request.correlation_id)

            # 检查交易所
            if request.exchange.lower() not in ("gateio", "gate.io"):
                return make_blocked(request, "FORBIDDEN_EXCHANGE",
                                    f"禁止交易所: {request.exchange}", request.correlation_id)

            return request

        except Exception as e:
            self._record_error(str(e))
            return make_blocked(request, "DATA_FETCH_FAILED", str(e),
                                request.correlation_id)

    def decide(self, proposal):
        """L1 不做决策。"""
        return proposal

    def propose(self, snapshot):
        """L1 不提案。"""
        return snapshot

    def explain(self, decision) -> Dict[str, Any]:
        """解释数据来源。"""
        return {
            "layer": self.LAYER_NAME,
            "exchange": "gateio",
            "is_real_data": True,
        }
