"""Schema Registry — 契约版本注册和兼容性检查。

铁律：所有层输出都可重放、可审计、可复现。
"""
from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, List, Type, Union

from .base import ContractBase, ContractStatus, DataLineage


# 已注册契约类型
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register(contract_name: str, version: str, schema_class: Type) -> None:
    """注册契约版本。"""
    if contract_name not in _REGISTRY:
        _REGISTRY[contract_name] = {}
    _REGISTRY[contract_name][version] = {
        "class": schema_class,
        "version": version,
        "fields": list(schema_class.__dataclass_fields__.keys()) if is_dataclass(schema_class) else [],
    }


def get_schema(contract_name: str, version: str) -> Type:
    """获取契约版本对应的类。"""
    if contract_name not in _REGISTRY:
        raise KeyError(f"未注册契约: {contract_name}")
    if version not in _REGISTRY[contract_name]:
        raise KeyError(f"未注册版本: {contract_name}@{version}")
    return _REGISTRY[contract_name][version]["class"]


def check_compatibility(contract_name: str, version_a: str, version_b: str) -> bool:
    """检查两个版本的兼容性（字段包含关系）。"""
    if contract_name not in _REGISTRY:
        return False
    va = _REGISTRY[contract_name].get(version_a, {}).get("fields", [])
    vb = _REGISTRY[contract_name].get(version_b, {}).get("fields", [])
    # 后向兼容：新版本字段必须包含旧版本所有字段
    return set(va).issubset(set(vb))


def serialize(contract: ContractBase) -> str:
    """序列化契约为 JSON 字符串。"""
    d = contract.to_dict() if hasattr(contract, "to_dict") else asdict(contract)
    return json.dumps(d, default=str, sort_keys=True, ensure_ascii=False)


def deserialize(contract_name: str, version: str, json_str: str) -> ContractBase:
    """从 JSON 字符串反序列化契约。"""
    schema_class = get_schema(contract_name, version)
    d = json.loads(json_str)
    # 处理枚举字段
    if "status" in d and isinstance(d["status"], str):
        d["status"] = ContractStatus(d["status"])
    if "lineage" in d and isinstance(d["lineage"], dict):
        d["lineage"] = DataLineage.from_dict(d["lineage"])
    # 过滤未知字段
    known_fields = set(schema_class.__dataclass_fields__.keys())
    filtered = {k: v for k, v in d.items() if k in known_fields}
    return schema_class(**filtered)


def list_registered() -> Dict[str, List[str]]:
    """列出所有已注册契约和版本。"""
    return {name: list(versions.keys()) for name, versions in _REGISTRY.items()}


def register_all() -> None:
    """注册所有内置契约。"""
    from .market_data_envelope import MarketDataEnvelope
    from .market_state_snapshot import MarketStateSnapshot
    from .opportunity_candidate import OpportunityCandidate
    from .strategy_proposal import StrategyProposal
    from .audit_verdict import AuditVerdict
    from .risk_budget import RiskBudget
    from .execution_contracts import ExecutionIntent, ExecutionReport
    from .learning_event import LearningEvent
    from .change_proposal import ChangeProposal

    register("MarketDataEnvelope", "1.0.0", MarketDataEnvelope)
    register("MarketStateSnapshot", "1.0.0", MarketStateSnapshot)
    register("OpportunityCandidate", "1.0.0", OpportunityCandidate)
    register("StrategyProposal", "1.0.0", StrategyProposal)
    register("AuditVerdict", "1.0.0", AuditVerdict)
    register("RiskBudget", "1.0.0", RiskBudget)
    register("ExecutionIntent", "1.0.0", ExecutionIntent)
    register("ExecutionReport", "1.0.0", ExecutionReport)
    register("LearningEvent", "1.0.0", LearningEvent)
    register("ChangeProposal", "1.0.0", ChangeProposal)


# 自动注册
register_all()
