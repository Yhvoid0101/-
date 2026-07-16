#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 11 — 关键指标监控与告警系统。

实现 StabilityMonitor 实时监控 100/1000 轮运行。

6 类监控指标：
1. audit_chain_monotonicity：审计链单调性
2. state_machine_progress：状态机停滞
3. death_cause_distribution：死因集中度
4. cycle_count_progress：cycle_count 递增
5. duration_anomaly：延迟异常
6. flywheel_trigger_frequency：飞轮触发频率偏差

4 级告警：INFO / WARN / ERROR / CRITICAL（CRITICAL 自动触发回滚）

铁律合规：
- 零模拟：真实指标采集
- 零死角：6 类指标全覆盖
- 零跳过：无 skip
- 零限制：所有告警可追溯
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 告警级别
# ============================================================

class AlertLevel(str, Enum):
    """告警级别。"""
    INFO = "info"          # 信息
    WARN = "warn"          # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重（自动触发回滚）


# ============================================================
# 告警数据类
# ============================================================

@dataclass
class Alert:
    """告警条目。"""
    timestamp: float = 0.0
    level: str = ""
    metric: str = ""          # 指标名称
    threshold: str = ""       # 阈值描述
    actual_value: str = ""    # 实际值
    round_num: int = 0        # 轮次
    symbol: str = ""          # 币种
    interval: str = ""        # 周期
    message: str = ""         # 告警消息
    trigger_rollback: bool = False  # 是否触发回滚


# ============================================================
# 指标快照
# ============================================================

@dataclass
class MetricSnapshot:
    """指标快照（每 10 轮采集一次）。"""
    round_num: int = 0
    timestamp: float = 0.0
    cycle_count: int = 0
    sample_count: int = 0
    audit_chain_length: int = 0
    state: str = ""
    duration_ms: float = 0.0
    inner_triggered: bool = False
    middle_triggered: bool = False
    outer_triggered: bool = False
    death_cause_counts: Dict[str, int] = field(default_factory=dict)
    alerts_count: int = 0


# ============================================================
# 稳定性监控器
# ============================================================

class StabilityMonitor:
    """稳定性监控器。

    6 类指标监控 + 4 级告警 + 快照记录。
    """

    # 告警阈值
    STATE_STAGNATION_ROUNDS = 50       # 状态机停滞阈值（轮）
    DEATH_CAUSE_WARN_THRESHOLD = 0.80  # 死因集中度 WARN 阈值
    DEATH_CAUSE_CRITICAL_THRESHOLD = 0.95  # 死因集中度 CRITICAL 阈值
    DURATION_ANOMALY_MULTIPLIER = 3.0  # 延迟异常倍数（>avg*3）
    FLYWHEEL_FREQ_DEVIATION = 0.20     # 飞轮触发频率偏差阈值（20%）
    AUDIT_GROWTH_MIN = 1               # 审计链每轮最少增长条目数

    def __init__(
        self,
        artifacts_dir: str = "/home/lmy/hermes_v6/sandbox_trading/phase11_artifacts",
        symbol: str = "",
        interval: str = "",
    ):
        self.artifacts_dir = artifacts_dir
        self.symbol = symbol
        self.interval = interval
        self.alerts: List[Alert] = []
        self.snapshots: List[MetricSnapshot] = []
        self._round_history: List[Dict[str, Any]] = []
        self._durations: List[float] = []
        self._state_stagnation_count: int = 0
        self._last_state: str = ""
        self._inner_trigger_count: int = 0
        self._middle_trigger_count: int = 0
        self._outer_trigger_count: int = 0

        os.makedirs(artifacts_dir, exist_ok=True)

        self._alerts_path = os.path.join(artifacts_dir, "alerts.jsonl")
        # 清空旧告警
        open(self._alerts_path, "w").close()

    # ----------------------------------------------------------
    # 告警生成
    # ----------------------------------------------------------

    def _emit_alert(
        self,
        level: AlertLevel,
        metric: str,
        threshold: str,
        actual_value: str,
        round_num: int,
        message: str,
    ) -> Alert:
        """生成告警。"""
        alert = Alert(
            timestamp=time.time(),
            level=level.value,
            metric=metric,
            threshold=threshold,
            actual_value=actual_value,
            round_num=round_num,
            symbol=self.symbol,
            interval=self.interval,
            message=message,
            trigger_rollback=(level == AlertLevel.CRITICAL),
        )
        self.alerts.append(alert)

        # 写入告警日志
        with open(self._alerts_path, "a") as f:
            f.write(json.dumps(asdict(alert), default=str) + "\n")

        # 控制台输出
        level_icon = {"info": "ℹ", "warn": "⚠", "error": "✗", "critical": "⛔"}
        print(f"  [{level.value.upper()}] Round {round_num} {metric}: {message}")

        return alert

    # ----------------------------------------------------------
    # 6 类指标检查
    # ----------------------------------------------------------

    def check_audit_chain_monotonicity(
        self,
        current_length: int,
        prev_length: int,
        round_num: int,
    ) -> Optional[Alert]:
        """指标1: 审计链单调性（必须增长）。"""
        if round_num > 1 and current_length <= prev_length:
            return self._emit_alert(
                AlertLevel.ERROR,
                "audit_chain_monotonicity",
                f"current > prev ({prev_length})",
                str(current_length),
                round_num,
                f"审计链未增长: {current_length} <= {prev_length}",
            )
        return None

    def check_state_machine_progress(
        self,
        state: str,
        round_num: int,
    ) -> Optional[Alert]:
        """指标2: 状态机停滞（同一非idle状态>50轮）。"""
        if state != "idle":
            if state == self._last_state:
                self._state_stagnation_count += 1
            else:
                self._state_stagnation_count = 1
                self._last_state = state

            if self._state_stagnation_count > self.STATE_STAGNATION_ROUNDS:
                return self._emit_alert(
                    AlertLevel.CRITICAL,
                    "state_machine_progress",
                    f"stagnation <= {self.STATE_STAGNATION_ROUNDS} rounds",
                    f"stagnation={self._state_stagnation_count} state={state}",
                    round_num,
                    f"状态机停滞: state={state} 持续 {self._state_stagnation_count} 轮",
                )
        else:
            self._state_stagnation_count = 0
            self._last_state = ""
        return None

    def check_death_cause_distribution(
        self,
        death_cause_counts: Dict[str, int],
        round_num: int,
    ) -> Optional[Alert]:
        """指标3: 死因分布集中度。"""
        total = sum(death_cause_counts.values())
        if total == 0:
            return None

        max_single = max(death_cause_counts.values())
        concentration = max_single / total
        max_cause = max(death_cause_counts, key=death_cause_counts.get)

        if concentration >= self.DEATH_CAUSE_CRITICAL_THRESHOLD:
            return self._emit_alert(
                AlertLevel.CRITICAL,
                "death_cause_distribution",
                f"concentration < {self.DEATH_CAUSE_CRITICAL_THRESHOLD:.0%}",
                f"concentration={concentration:.1%} cause={max_cause}",
                round_num,
                f"死因垄断: {max_cause} 占 {concentration:.1%}",
            )
        elif concentration >= self.DEATH_CAUSE_WARN_THRESHOLD:
            return self._emit_alert(
                AlertLevel.WARN,
                "death_cause_distribution",
                f"concentration < {self.DEATH_CAUSE_WARN_THRESHOLD:.0%}",
                f"concentration={concentration:.1%} cause={max_cause}",
                round_num,
                f"死因集中: {max_cause} 占 {concentration:.1%}",
            )
        return None

    def check_cycle_count_progress(
        self,
        current_cycle: int,
        prev_cycle: int,
        round_num: int,
    ) -> Optional[Alert]:
        """指标4: cycle_count 递增。"""
        if round_num > 1 and current_cycle <= prev_cycle:
            return self._emit_alert(
                AlertLevel.ERROR,
                "cycle_count_progress",
                f"current > prev ({prev_cycle})",
                str(current_cycle),
                round_num,
                f"cycle_count 未递增: {current_cycle} <= {prev_cycle}",
            )
        return None

    def check_duration_anomaly(
        self,
        duration_ms: float,
        round_num: int,
    ) -> Optional[Alert]:
        """指标5: 延迟异常（>avg*3）。"""
        self._durations.append(duration_ms)

        if len(self._durations) < 10:
            return None  # 前 10 轮建立基线

        avg_duration = sum(self._durations) / len(self._durations)
        threshold = avg_duration * self.DURATION_ANOMALY_MULTIPLIER

        if duration_ms > threshold:
            return self._emit_alert(
                AlertLevel.WARN,
                "duration_anomaly",
                f"duration <= avg*{self.DURATION_ANOMALY_MULTIPLIER} ({threshold:.1f}ms)",
                f"duration={duration_ms:.1f}ms",
                round_num,
                f"延迟异常: {duration_ms:.1f}ms > {threshold:.1f}ms (avg={avg_duration:.1f}ms)",
            )
        return None

    def check_flywheel_trigger_frequency(
        self,
        inner_triggered: bool,
        middle_triggered: bool,
        outer_triggered: bool,
        round_num: int,
    ) -> Optional[Alert]:
        """指标6: 飞轮触发频率偏差。"""
        if inner_triggered:
            self._inner_trigger_count += 1
        if middle_triggered:
            self._middle_trigger_count += 1
        if outer_triggered:
            self._outer_trigger_count += 1

        # 内层飞轮预期：每 10 轮触发 1 次
        if round_num >= 10 and round_num % 10 == 0:
            expected_inner = round_num // 10
            actual_inner = self._inner_trigger_count
            if expected_inner > 0:
                deviation = abs(actual_inner - expected_inner) / expected_inner
                if deviation > self.FLYWHEEL_FREQ_DEVIATION:
                    return self._emit_alert(
                        AlertLevel.WARN,
                        "flywheel_trigger_frequency",
                        f"deviation <= {self.FLYWHEEL_FREQ_DEVIATION:.0%}",
                        f"inner expected={expected_inner} actual={actual_inner} deviation={deviation:.1%}",
                        round_num,
                        f"内层飞轮频率偏差: expected={expected_inner} actual={actual_inner}",
                    )
        return None

    # ----------------------------------------------------------
    # 综合检查（每轮调用）
    # ----------------------------------------------------------

    def check_round(
        self,
        round_num: int,
        cycle_count: int,
        sample_count: int,
        audit_chain_length: int,
        state: str,
        duration_ms: float,
        inner_triggered: bool,
        middle_triggered: bool,
        outer_triggered: bool,
        death_cause_counts: Dict[str, int],
    ) -> List[Alert]:
        """每轮综合检查（6 类指标全覆盖）。"""
        alerts: List[Alert] = []

        # 获取上一轮指标
        prev = self._round_history[-1] if self._round_history else {}

        # 6 类指标检查
        a1 = self.check_audit_chain_monotonicity(
            audit_chain_length, prev.get("audit_chain_length", 0), round_num,
        )
        if a1:
            alerts.append(a1)

        a2 = self.check_state_machine_progress(state, round_num)
        if a2:
            alerts.append(a2)

        a3 = self.check_death_cause_distribution(death_cause_counts, round_num)
        if a3:
            alerts.append(a3)

        a4 = self.check_cycle_count_progress(
            cycle_count, prev.get("cycle_count", 0), round_num,
        )
        if a4:
            alerts.append(a4)

        a5 = self.check_duration_anomaly(duration_ms, round_num)
        if a5:
            alerts.append(a5)

        a6 = self.check_flywheel_trigger_frequency(
            inner_triggered, middle_triggered, outer_triggered, round_num,
        )
        if a6:
            alerts.append(a6)

        # 记录历史
        self._round_history.append({
            "round": round_num,
            "cycle_count": cycle_count,
            "sample_count": sample_count,
            "audit_chain_length": audit_chain_length,
            "state": state,
            "duration_ms": duration_ms,
            "inner_triggered": inner_triggered,
            "middle_triggered": middle_triggered,
            "outer_triggered": outer_triggered,
            "death_cause_counts": dict(death_cause_counts),
        })

        # 每 10 轮保存快照
        if round_num % 10 == 0:
            self._save_snapshot(round_num, cycle_count, sample_count,
                                audit_chain_length, state, duration_ms,
                                inner_triggered, middle_triggered, outer_triggered,
                                death_cause_counts)

        return alerts

    # ----------------------------------------------------------
    # 快照保存
    # ----------------------------------------------------------

    def _save_snapshot(
        self,
        round_num: int,
        cycle_count: int,
        sample_count: int,
        audit_chain_length: int,
        state: str,
        duration_ms: float,
        inner_triggered: bool,
        middle_triggered: bool,
        outer_triggered: bool,
        death_cause_counts: Dict[str, int],
    ) -> None:
        """保存指标快照。"""
        snapshot = MetricSnapshot(
            round_num=round_num,
            timestamp=time.time(),
            cycle_count=cycle_count,
            sample_count=sample_count,
            audit_chain_length=audit_chain_length,
            state=state,
            duration_ms=duration_ms,
            inner_triggered=inner_triggered,
            middle_triggered=middle_triggered,
            outer_triggered=outer_triggered,
            death_cause_counts=dict(death_cause_counts),
            alerts_count=len(self.alerts),
        )
        self.snapshots.append(snapshot)

        snapshot_path = os.path.join(
            self.artifacts_dir, f"snapshot_{self.symbol}_{self.interval}_{round_num}.json"
        )
        with open(snapshot_path, "w") as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)

    # ----------------------------------------------------------
    # 检查是否需要回滚
    # ----------------------------------------------------------

    def should_rollback(self) -> bool:
        """检查是否有 CRITICAL 告警需要回滚。"""
        return any(a.trigger_rollback for a in self.alerts)

    # ----------------------------------------------------------
    # 获取告警统计
    # ----------------------------------------------------------

    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警统计。"""
        level_counts = {level.value: 0 for level in AlertLevel}
        for a in self.alerts:
            level_counts[a.level] = level_counts.get(a.level, 0) + 1

        return {
            "total_alerts": len(self.alerts),
            "info": level_counts.get("info", 0),
            "warn": level_counts.get("warn", 0),
            "error": level_counts.get("error", 0),
            "critical": level_counts.get("critical", 0),
            "snapshots_count": len(self.snapshots),
            "should_rollback": self.should_rollback(),
            "inner_trigger_count": self._inner_trigger_count,
            "middle_trigger_count": self._middle_trigger_count,
            "outer_trigger_count": self._outer_trigger_count,
        }
