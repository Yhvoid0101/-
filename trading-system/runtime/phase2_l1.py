from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .contracts.base import ContractStatus, DataLineage, make_blocked
from .contracts.market_data_envelope import DataQuality, MarketDataEnvelope


class DataQualityStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass
class L1FetchResult:
    status: ContractStatus
    error_code: str = ""
    error_message: str = ""
    lineage: Optional[DataLineage] = None
    quality: DataQuality = None
    quality_status: DataQualityStatus = DataQualityStatus.FAILED
    envelope: Optional[MarketDataEnvelope] = None

    def is_blocked(self) -> bool:
        return self.status == ContractStatus.BLOCKED


class GateIoL1DataSource:
    def __init__(self, provider=None, source: str = "gateio", ttl_seconds: float = 3600.0):
        self.provider = provider
        self.source = source.lower()
        self.ttl_seconds = ttl_seconds

    def fetch(self, symbol: str, interval: str, limit: int = 100) -> L1FetchResult:
        if self.source != "gateio":
            return self._blocked(symbol, "FORBIDDEN_SOURCE", f"禁止数据源: {self.source}")
        if self.provider is None:
            return self._blocked(symbol, "PROVIDER_NOT_INITIALIZED", "Gate.io provider 未初始化")
        ingest_time = time.time()
        try:
            rows = self.provider.get_perp_klines(symbol, interval, limit=limit)
        except Exception as exc:
            return self._blocked(symbol, "PROVIDER_FAILURE", str(exc))
        if not rows:
            return self._blocked(symbol, "EMPTY_DATA", "Gate.io 返回空数据")
        normalized = self._normalize(rows)
        event_time = float(normalized[-1]["timestamp"])
        # 动态 TTL：按 interval 计算（1h K线允许 1h 延迟，1m K线允许 60s 延迟）
        effective_ttl = self._effective_ttl(interval)
        quality = self._quality(normalized, interval, ingest_time, event_time, effective_ttl)
        quality_status = self._classify_quality(quality, ingest_time, event_time, effective_ttl)
        checksum = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        lineage = DataLineage(
            source="gateio",
            event_time=event_time,
            ingest_time=ingest_time,
            ttl_seconds=effective_ttl,
            checksum=checksum,
        )
        envelope = MarketDataEnvelope(
            symbol=symbol,
            interval=interval,
            klines=normalized,
            quality=quality,
            exchange="gateio",
            is_real_data=True,
            lineage=lineage,
        )
        if lineage.is_expired(ingest_time) or quality.is_stale:
            return self._blocked_result(envelope, lineage, quality, quality_status, "STALE_DATA", f"Gate.io 数据已过期 (ttl={effective_ttl}s, age={ingest_time - event_time:.0f}s)")
        if quality.anomaly_count or quality.gap_count or quality.completeness < 1.0:
            return self._blocked_result(envelope, lineage, quality, quality_status, "DATA_QUALITY_INVALID", "Gate.io 数据质量校验失败")
        return L1FetchResult(status=ContractStatus.OK, lineage=lineage, quality=quality, quality_status=DataQualityStatus.HEALTHY, envelope=envelope)

    def _classify_quality(self, quality: DataQuality, ingest_time: float, event_time: float, effective_ttl: float) -> DataQualityStatus:
        if quality.anomaly_count or quality.gap_count or quality.completeness < 1.0:
            return DataQualityStatus.BLOCKED
        if quality.is_stale or (ingest_time - event_time) > effective_ttl:
            return DataQualityStatus.BLOCKED
        if quality.latency_ms > 5000:
            return DataQualityStatus.DEGRADED
        return DataQualityStatus.HEALTHY

    # Interval → 秒数映射（用于动态 TTL）
    _INTERVAL_SECONDS = {
        "1m": 60.0, "5m": 300.0, "15m": 900.0, "30m": 1800.0,
        "1h": 3600.0, "4h": 14400.0, "1d": 86400.0, "1w": 604800.0,
    }

    def _effective_ttl(self, interval: str) -> float:
        """按 interval 动态计算 TTL。

        规则：TTL = max(self.ttl_seconds, interval_seconds * 1.5)
        1h K线允许 1.5h 延迟（最后一根天然延迟最多 1h + 缓冲）
        1m K线允许 60s 延迟（如果 ttl_seconds >= 60）
        """
        interval_sec = self._INTERVAL_SECONDS.get(interval, 3600.0)
        # TTL 至少是 interval 的 1.5 倍（覆盖最后一根 K线天然延迟）
        return max(self.ttl_seconds, interval_sec * 1.5)

    def _normalize(self, rows):
        normalized = []
        for row in rows:
            if isinstance(row, dict):
                item = dict(row)
            else:
                item = {
                    "timestamp": row.timestamp,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "quote_volume": getattr(row, "quote_volume", 0.0),
                }
            normalized.append({
                "timestamp": float(item["timestamp"]),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "volume": float(item.get("volume", 0.0)),
                "quote_volume": float(item.get("quote_volume", 0.0)),
            })
        return sorted(normalized, key=lambda value: value["timestamp"])

    def _quality(self, rows, interval: str, ingest_time: float, event_time: float, effective_ttl: float) -> DataQuality:
        expected = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}.get(interval)
        anomaly_count = 0
        gap_count = 0
        for row in rows:
            if min(row["open"], row["high"], row["low"], row["close"]) <= 0:
                anomaly_count += 1
            if not row["low"] <= row["open"] <= row["high"] or not row["low"] <= row["close"] <= row["high"]:
                anomaly_count += 1
        if expected:
            for previous, current in zip(rows, rows[1:]):
                delta = current["timestamp"] - previous["timestamp"]
                if abs(delta - expected) > 1e-6:
                    gap_count += 1
        latency_ms = max(0.0, (ingest_time - event_time) * 1000)
        return DataQuality(
            completeness=1.0 if not gap_count else max(0.0, 1.0 - gap_count / max(1, len(rows) - 1)),
            latency_ms=latency_ms,
            gap_count=gap_count,
            anomaly_count=anomaly_count,
            is_stale=ingest_time - event_time > effective_ttl,
        )

    def _blocked(self, symbol, code, message):
        envelope = MarketDataEnvelope(symbol=symbol, exchange="gateio", is_real_data=True)
        return self._blocked_result(envelope, envelope.lineage, envelope.quality, DataQualityStatus.FAILED, code, message)

    def _blocked_result(self, envelope, lineage, quality, quality_status, code, message):
        blocked = make_blocked(envelope, code, message, envelope.correlation_id)
        return L1FetchResult(status=ContractStatus.BLOCKED, error_code=code, error_message=message, lineage=lineage, quality=quality, quality_status=quality_status, envelope=blocked)



# ============================================================
# DataSourceProposal — 数据源提案与硬阻断
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List


class ProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    SHADOW = "SHADOW"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class DataSourceProposal:
    source: str
    symbol: str
    endpoint: str
    history_coverage_days: float
    latency_ms: float
    cost: float
    expected_information_gain: float
    validation_plan: str
    is_synthetic: bool = False
    proposal_id: str = ""
    status: ProposalStatus = ProposalStatus.PROPOSED
    rejection_reason: str = ""
    approved_by: str = ""


@dataclass
class _ProposalRecord:
    proposal: DataSourceProposal
    status: ProposalStatus = ProposalStatus.PROPOSED
    rejection_reason: str = ""
    approved_by: str = ""


class DataSourceProposalManager:
    """数据源提案管理器 — 硬阻断非 Gate.io/合成/无证据提案。"""

    def __init__(self):
        self._records: Dict[str, _ProposalRecord] = {}
        self._counter = 0

    def submit(self, proposal: DataSourceProposal) -> DataSourceProposal:
        """提交提案。非 Gate.io/合成/无证据直接 REJECTED。"""
        self._counter += 1
        proposal.proposal_id = f"dsp-{self._counter:04d}"

        if proposal.source.lower() != "gateio":
            return self._reject(proposal, "FORBIDDEN_SOURCE", f"禁止数据源: {proposal.source}")
        if proposal.is_synthetic:
            return self._reject(proposal, "SYNTHETIC_DATA", "合成数据提案被拒绝")
        if not proposal.endpoint or not proposal.validation_plan:
            return self._reject(proposal, "NO_EVIDENCE", "缺少 endpoint 或验证计划")
        if proposal.history_coverage_days <= 0 or proposal.expected_information_gain <= 0:
            return self._reject(proposal, "NO_EVIDENCE", "历史覆盖或信息增益为零")

        proposal.status = ProposalStatus.SHADOW
        self._records[proposal.proposal_id] = _ProposalRecord(
            proposal=proposal,
            status=ProposalStatus.SHADOW,
        )
        return proposal

    def promote(self, proposal_id: str, approved_by: str) -> DataSourceProposal:
        """晋升提案到 PROMOTED。必须在 SHADOW 状态且有审批人。"""
        record = self._records.get(proposal_id)
        if record is None or record.status != ProposalStatus.SHADOW:
            dummy = DataSourceProposal(
                source="gateio", symbol="", endpoint="",
                history_coverage_days=0, latency_ms=0, cost=0,
                expected_information_gain=0, validation_plan="",
                proposal_id=proposal_id,
            )
            return self._reject(dummy, "NOT_IN_SHADOW", f"提案 {proposal_id} 不在 SHADOW 状态")
        if not approved_by:
            return self._reject(record.proposal, "NO_APPROVER", "缺少审批人")

        record.status = ProposalStatus.PROMOTED
        record.approved_by = approved_by
        record.proposal.status = ProposalStatus.PROMOTED
        record.proposal.approved_by = approved_by
        return record.proposal

    def rollback(self, proposal_id: str, reason: str) -> DataSourceProposal:
        """回滚已晋升的提案。"""
        record = self._records.get(proposal_id)
        if record is None:
            dummy = DataSourceProposal(
                source="gateio", symbol="", endpoint="",
                history_coverage_days=0, latency_ms=0, cost=0,
                expected_information_gain=0, validation_plan="",
                proposal_id=proposal_id,
            )
            return self._reject(dummy, "NOT_FOUND", f"提案 {proposal_id} 不存在")

        record.status = ProposalStatus.ROLLED_BACK
        record.proposal.status = ProposalStatus.ROLLED_BACK
        record.proposal.rejection_reason = reason
        return record.proposal

    def list_proposals(self) -> List[DataSourceProposal]:
        return [r.proposal for r in self._records.values()]

    def _reject(self, proposal: DataSourceProposal, code: str, message: str) -> DataSourceProposal:
        proposal.status = ProposalStatus.REJECTED
        proposal.rejection_reason = f"{code}: {message}"
        self._records[proposal.proposal_id] = _ProposalRecord(
            proposal=proposal,
            status=ProposalStatus.REJECTED,
            rejection_reason=proposal.rejection_reason,
        )
        return proposal
