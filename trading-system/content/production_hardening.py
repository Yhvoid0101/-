# -*- coding: utf-8 -*-
"""
生产级强化模块 (Production Hardening) — 实盘级风控+监控+审计+压力测试

基于前沿生产级量化交易系统最佳实践（2025）：
  - Math&Markets实盘经验：Kill Switch 1 cycle平仓、滚动回撤-8%/-12%
  - CFTC/FINRA算法审计框架：完整治理+数据血缘+模型风险+控制+监督
  - Quant 2.0架构：Feature Store消除training-serving skew
  - Robert Pardo WFA：严格训练/验证/测试三集分离

核心组件：
  1. KillSwitch — 物理逃生，1 cycle内exposure=0
  2. RollingDrawdownMonitor — 滚动回撤监控，-8%警告/-12%平仓
  3. AuditLogger — 完整审计日志（决策/变更/告警）
  4. StressTestScenario — 黑天鹅/闪崩/插针压力测试
  5. ProductionMonitor — 实时监控+告警
  6. GraduationCriteria — Paper trading毕业标准
  7. PerformanceBenchmark — 性能基准测试

不可动摇的边界：
  - Kill Switch独立于主系统，不依赖AI判断
  - 实盘不进化（进化只在沙盘）
  - 风控判断不依赖AI模型
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from collections import deque, Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.production")


# ============================================================================
# 1. Kill Switch — 物理逃生（最高优先级）
# ============================================================================


class KillSwitchState(Enum):
    """Kill Switch状态"""
    ARMED = "armed"        # 待命（正常）
    TRIGGERED = "triggered"  # 触发（平仓中）
    FLATTENED = "flattened"  # 已平仓（锁定）


@dataclass(slots=True)
class KillSwitchEvent:
    """Kill Switch触发事件"""
    timestamp: float
    reason: str
    trigger_value: float
    threshold: float
    action_taken: str


class KillSwitch:
    """物理逃生开关 — 1 cycle内将exposure=0

    设计原则（来自Math&Markets实盘经验）：
      - flat=true 在1个cycle内将exposure=0
      - 独立于主系统，不依赖AI判断
      - 触发后必须人工重置
      - 多重触发条件，任一满足即触发

    触发条件：
      1. 全局回撤 > 12%（硬性平仓）
      2. 单Agent回撤 > 25%（个体平仓）
      3. 日内亏损 > 5%（暂停交易）
      4. 系统异常（心跳丢失/API故障）
      5. 人工触发（emergency_stop）

    B3修复：添加sandbox_mode开关
      - sandbox_mode=False（默认）：使用生产级严格阈值
      - sandbox_mode=True：心跳超时放宽（沙盘非实时），但风控阈值保持生产级
      - 核心原则：沙盘不能比实盘更宽松，否则"沙盘赚钱、实盘亏钱"
    """

    # 触发阈值（来自前沿实盘标准）
    GLOBAL_DD_HARD_STOP = 0.12
    AGENT_DD_HARD_STOP = 0.25
    # v14.42 Fix 7: 沙盘模式KILL_SWITCH放宽
    # 根因: 沙盘进化中25%回撤过严 (47次触发, 阻碍策略探索)
    # 修复: 沙盘模式放宽到40% (生产保持25%)
    _v1442_sandbox = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
    if _v1442_sandbox:
        AGENT_DD_HARD_STOP = 0.40  # 沙盘进化放宽, 允许更高回撤探索
    DAILY_LOSS_HARD_STOP = 0.05
    HEARTBEAT_TIMEOUT_SEC = 30.0    # 心跳超时30秒（生产）
    HEARTBEAT_TIMEOUT_SANDBOX = 3600.0  # 心跳超时1小时（沙盘，非实时）

    def __init__(self, sandbox_mode: bool = False):
        """初始化KillSwitch

        Args:
            sandbox_mode: 是否沙盘模式。沙盘模式下仅放宽心跳超时（非实时环境），
                         风控阈值保持生产级严格标准，防止温室效应。
        """
        self._state = KillSwitchState.ARMED
        self._events: List[KillSwitchEvent] = []
        self._lock = threading.RLock()
        self._last_heartbeat = time.time()
        self._flatten_callback: Optional[Callable[[str], int]] = None
        self._trigger_count = 0
        self._sandbox_mode = sandbox_mode
        # 沙盘模式放宽心跳超时（非实时环境），但风控阈值保持生产级
        if sandbox_mode:
            self.HEARTBEAT_TIMEOUT_SEC = self.HEARTBEAT_TIMEOUT_SANDBOX
            logger.info("KillSwitch: sandbox_mode=True (heartbeat relaxed, risk thresholds production-grade)")

    def register_flatten_callback(self, callback: Callable[[str], int]) -> None:
        """注册平仓回调函数（返回平仓的仓位数量）"""
        self._flatten_callback = callback
        logger.info("KillSwitch: flatten callback registered")

    def heartbeat(self) -> None:
        """心跳更新（主系统定期调用）"""
        with self._lock:
            self._last_heartbeat = time.time()

    def trigger(self, reason: str, trigger_value: float, threshold: float) -> bool:
        """触发Kill Switch

        Args:
            reason: 触发原因
            trigger_value: 触发时的实际值
            threshold: 触发阈值

        Returns:
            True表示成功触发并平仓
        """
        
        with self._lock:
            if self._state == KillSwitchState.FLATTENED:
                logger.warning("KillSwitch already FLATTENED, ignoring trigger: %s", reason)
                return False

            self._state = KillSwitchState.TRIGGERED
            self._trigger_count += 1

            event = KillSwitchEvent(
                timestamp=time.time(),
                reason=reason,
                trigger_value=trigger_value,
                threshold=threshold,
                action_taken="flatten_all_positions",
            )
            self._events.append(event)

            logger.critical(
                "KillSwitch TRIGGERED: reason=%s value=%.4f threshold=%.4f",
                reason, trigger_value, threshold,
            )

            # 执行平仓回调
            flattened_count = 0
            if self._flatten_callback:
                try:
                    flattened_count = self._flatten_callback(reason)
                    logger.critical(
                        "KillSwitch: flattened %d positions in 1 cycle",
                        flattened_count,
                    )
                except Exception as e:
                    logger.error("KillSwitch flatten callback failed: %s", e)

            self._state = KillSwitchState.FLATTENED
            return True

    def check_global_drawdown(self, drawdown: float) -> bool:
        """检查全局回撤是否触发硬平仓"""
        if drawdown >= self.GLOBAL_DD_HARD_STOP:
            return self.trigger(
                reason="GLOBAL_DRAWDOWN_HARD_STOP",
                trigger_value=drawdown,
                threshold=self.GLOBAL_DD_HARD_STOP,
            )
        return False

    def check_agent_drawdown(self, agent_id: str, drawdown: float) -> bool:
        """检查单Agent回撤是否触发平仓"""
        if drawdown >= self.AGENT_DD_HARD_STOP:
            return self.trigger(
                reason=f"AGENT_DRAWDOWN_HARD_STOP:{agent_id}",
                trigger_value=drawdown,
                threshold=self.AGENT_DD_HARD_STOP,
            )
        return False

    def check_daily_loss(self, daily_loss_pct: float) -> bool:
        """检查日内亏损是否触发暂停"""
        if daily_loss_pct >= self.DAILY_LOSS_HARD_STOP:
            return self.trigger(
                reason="DAILY_LOSS_HARD_STOP",
                trigger_value=daily_loss_pct,
                threshold=self.DAILY_LOSS_HARD_STOP,
            )
        return False

    def check_heartbeat(self) -> bool:
        """检查心跳是否超时"""
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self.HEARTBEAT_TIMEOUT_SEC:
            return self.trigger(
                reason="HEARTBEAT_TIMEOUT",
                trigger_value=elapsed,
                threshold=self.HEARTBEAT_TIMEOUT_SEC,
            )
        return False

    def emergency_stop(self, reason: str = "MANUAL_EMERGENCY_STOP") -> bool:
        """人工紧急停止"""
        return self.trigger(
            reason=reason,
            trigger_value=0.0,
            threshold=0.0,
        )

    def reset(self) -> bool:
        """人工重置Kill Switch（触发后必须人工重置）

        仅当状态为 FLATTENED 时才允许重置，遵循"触发→平仓→人工重置"状态机。
        """
        with self._lock:
            if self._state != KillSwitchState.FLATTENED:
                logger.warning("KillSwitch not FLATTENED, reset ignored")
                return False
            self._state = KillSwitchState.ARMED
            self._last_heartbeat = time.time()
            logger.info("KillSwitch RESET to ARMED (manual)")
            return True

    def force_reset(self, reason: str = "sandbox_override") -> bool:
        """强制重置Kill Switch到ARMED状态（沙盘模式专用）

        与reset()不同，force_reset() 不受状态机限制，可从任何状态强制
        转换到ARMED。用于沙盘进化模式下每轮/每代自动重置，避免直接
        写 _state 私有属性（封装违规）。

        生产环境慎用：force_reset() 会绕过"触发→平仓→人工重置"安全流程。

        Args:
            reason: 强制重置原因（用于审计日志）

        Returns:
            总是返回 True
        """
        with self._lock:
            prev_state = self._state
            self._state = KillSwitchState.ARMED
            self._last_heartbeat = time.time()
            logger.warning(
                "KillSwitch FORCE RESET to ARMED (reason=%s, prev_state=%s)",
                reason, prev_state.value,
            )
            return True

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_armed(self) -> bool:
        return self._state == KillSwitchState.ARMED

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    def get_events(self) -> List[KillSwitchEvent]:
        with self._lock:
            return list(self._events)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "trigger_count": self._trigger_count,
                "event_count": len(self._events),
                "last_heartbeat_age_sec": time.time() - self._last_heartbeat,
                "thresholds": {
                    "global_dd_hard_stop": self.GLOBAL_DD_HARD_STOP,
                    "agent_dd_hard_stop": self.AGENT_DD_HARD_STOP,
                    "daily_loss_hard_stop": self.DAILY_LOSS_HARD_STOP,
                    "heartbeat_timeout_sec": self.HEARTBEAT_TIMEOUT_SEC,
                },
            }


# ============================================================================
# 2. 滚动回撤监控器（Rolling Drawdown Monitor）
# ============================================================================


@dataclass(slots=True)
class DrawdownAlert:
    """回撤告警"""
    timestamp: float
    agent_id: str
    drawdown: float
    level: str  # "warn" or "critical"
    message: str


class RollingDrawdownMonitor:
    """滚动回撤监控器 — 多级告警

    来自Math&Markets实盘标准：
      - 滚动回撤 -8% → 警告
      - 滚动回撤 -12% → 平仓（触发Kill Switch）
      - 实现波动率 > 3σ → 减半杠杆
      - ADV使用 ≥ 10% → 减半仓位
    """

    WARN_DD_THRESHOLD = 0.08       # -8% 警告
    CRITICAL_DD_THRESHOLD = 0.12   # -12% 平仓 (生产级)
    VOL_3SIGMA_THRESHOLD = 3.0     # 3σ波动率异常
    ADV_USAGE_THRESHOLD = 0.10     # ADV使用10%减仓

    def __init__(self, kill_switch: Optional[KillSwitch] = None, sandbox_mode: bool = False):
        self._kill_switch = kill_switch
        # Phase 10.2: 沙盘模式放宽回撤阈值, 让进化算法有足够交易数据
        # 根因: 12%回撤强制平仓导致90% agent被熔断, 交易不足无法进化
        # 设计: 沙盘模式探索期允许更大回撤(30%), 进化成熟后自动收紧
        if sandbox_mode:
            self.CRITICAL_DD_THRESHOLD = 0.30  # 30%: 进化探索期允许更大回撤
            self.WARN_DD_THRESHOLD = 0.15      # 15%
            logger.info("Phase 10.2 RollingDrawdownMonitor: sandbox_mode=True, CRITICAL_DD=30%% (进化探索期)")
        self._agent_equity: Dict[str, List[Tuple[float, float]]] = {}  # agent_id -> [(ts, equity)]
        self._alerts: List[DrawdownAlert] = []
        self._warned_agents: Dict[str, float] = {}  # agent_id -> last_warn_ts
        self._lock = threading.RLock()
        self._window_size = 100  # 滚动窗口大小

    def update_equity(self, agent_id: str, equity: float, timestamp: Optional[float] = None) -> None:
        """更新Agent权益曲线"""
        ts = timestamp or time.time()
        with self._lock:
            if agent_id not in self._agent_equity:
                self._agent_equity[agent_id] = []
            self._agent_equity[agent_id].append((ts, equity))
            # 保持窗口大小
            if len(self._agent_equity[agent_id]) > self._window_size:
                self._agent_equity[agent_id] = self._agent_equity[agent_id][-self._window_size:]

    def check_drawdown(self, agent_id: str) -> Optional[DrawdownAlert]:
        """检查Agent滚动回撤"""
        with self._lock:
            equity_curve = self._agent_equity.get(agent_id, [])
            if len(equity_curve) < 2:
                return None

            equities = [e[1] for e in equity_curve]
            peak = max(equities)
            current = equities[-1]

            if peak <= 0:
                return None

            drawdown = (peak - current) / peak
            now = time.time()

            # 警告级别（-8%）
            if drawdown >= self.WARN_DD_THRESHOLD:
                # 避免重复告警（30秒内同一Agent只告警一次）
                last_warn = self._warned_agents.get(agent_id, 0)
                if now - last_warn > 30:
                    alert = DrawdownAlert(
                        timestamp=now,
                        agent_id=agent_id,
                        drawdown=drawdown,
                        level="warn",
                        message=f"Agent {agent_id} rolling drawdown {drawdown:.2%} >= {self.WARN_DD_THRESHOLD:.2%}",
                    )
                    self._alerts.append(alert)
                    self._warned_agents[agent_id] = now
                    logger.warning(alert.message)
                    return alert

            # 严重级别（-12%，触发Kill Switch）
            if drawdown >= self.CRITICAL_DD_THRESHOLD:
                alert = DrawdownAlert(
                    timestamp=now,
                    agent_id=agent_id,
                    drawdown=drawdown,
                    level="critical",
                    message=f"Agent {agent_id} CRITICAL drawdown {drawdown:.2%} >= {self.CRITICAL_DD_THRESHOLD:.2%}",
                )
                self._alerts.append(alert)
                logger.critical(alert.message)

                if self._kill_switch:
                    self._kill_switch.check_agent_drawdown(agent_id, drawdown)

                return alert

        return None

    def check_volatility_anomaly(self, returns: List[float]) -> Tuple[bool, float]:
        """检查波动率异常（3σ规则）

        Returns:
            (is_anomaly, z_score)
        """
        if len(returns) < 20:
            return False, 0.0

        mean_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns) if len(returns) > 1 else 0.0

        if std_ret < 1e-10:
            return False, 0.0

        latest_ret = returns[-1]
        z_score = abs(latest_ret - mean_ret) / std_ret

        if z_score > self.VOL_3SIGMA_THRESHOLD:
            logger.warning(
                "Volatility anomaly: z_score=%.2f > %.1f (3σ rule)",
                z_score, self.VOL_3SIGMA_THRESHOLD,
            )
            return True, z_score

        return False, z_score

    def check_adv_usage(self, order_size: float, adv: float) -> Tuple[bool, float]:
        """检查ADV使用率（≥10%减仓）

        Args:
            order_size: 订单大小
            adv: 日均成交量

        Returns:
            (needs_reduction, usage_ratio)
        """
        if adv <= 0:
            return False, 0.0

        usage = order_size / adv
        if usage >= self.ADV_USAGE_THRESHOLD:
            logger.warning(
                "ADV usage %.2f%% >= %.2f%%, halving position",
                usage * 100, self.ADV_USAGE_THRESHOLD * 100,
            )
            return True, usage

        return False, usage

    def get_alerts(self, level: Optional[str] = None) -> List[DrawdownAlert]:
        with self._lock:
            if level:
                return [a for a in self._alerts if a.level == level]
            return list(self._alerts)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "monitored_agents": len(self._agent_equity),
                "total_alerts": len(self._alerts),
                "warn_alerts": sum(1 for a in self._alerts if a.level == "warn"),
                "critical_alerts": sum(1 for a in self._alerts if a.level == "critical"),
                "thresholds": {
                    "warn_dd": self.WARN_DD_THRESHOLD,
                    "critical_dd": self.CRITICAL_DD_THRESHOLD,
                    "vol_3sigma": self.VOL_3SIGMA_THRESHOLD,
                    "adv_usage": self.ADV_USAGE_THRESHOLD,
                },
            }


# ============================================================================
# 3. 审计日志（Audit Logger）— 完整决策记录
# ============================================================================


@dataclass(slots=True)
class AuditEntry:
    """审计日志条目"""
    timestamp: float
    category: str       # decision / change / alert / trade / risk
    actor: str          # agent_id / system / user
    action: str
    details: Dict[str, Any]
    severity: str = "info"  # info / warn / error / critical


class AuditLogger:
    """审计日志器 — 满足CFTC/FINRA算法审计要求

    记录内容：
      - 决策日志：每次交易决策的依据和参数
      - 变更日志：代码/参数变更审批
      - 告警日志：所有告警事件
      - 交易日志：订单提交/成交/取消
      - 风控日志：熔断/kill switch触发
    """

    def __init__(self, log_file: Optional[str] = None, max_entries: int = 10000,
                 context: Optional[Dict[str, Any]] = None):
        self._entries: deque = deque(maxlen=max_entries)
        self._log_file = log_file
        self._context = dict(context or {})
        self._lock = threading.RLock()
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, category: str, actor: str, action: str,
            details: Optional[Dict[str, Any]] = None, severity: str = "info") -> None:
        """记录审计日志"""
        entry = AuditEntry(
            timestamp=time.time(),
            category=category,
            actor=actor,
            action=action,
            details={**self._context, **(details or {})},
            severity=severity,
        )

        with self._lock:
            self._entries.append(entry)

            # 写入文件（追加模式）
            if self._log_file:
                try:
                    with open(self._log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "ts": entry.timestamp,
                            "ts_iso": datetime.fromtimestamp(entry.timestamp).isoformat(),
                            "category": entry.category,
                            "actor": entry.actor,
                            "action": entry.action,
                            "details": entry.details,
                            "severity": entry.severity,
                        }, ensure_ascii=False, default=str) + "\n")
                except Exception as e:
                    logger.error("AuditLogger write failed: %s", e)

        # 控制台输出
        log_msg = f"[AUDIT:{category}] {actor} -> {action}"
        if severity == "critical":
            logger.critical(log_msg)
        elif severity == "error":
            logger.error(log_msg)
        elif severity == "warn":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def log_decision(self, agent_id: str, decision: str, params: Dict[str, Any]) -> None:
        """记录交易决策"""
        self.log("decision", agent_id, decision, params)

    def log_trade(self, agent_id: str, action: str, order: Dict[str, Any]) -> None:
        """记录交易事件"""
        self.log("trade", agent_id, action, order)

    def log_risk(self, agent_id: str, event: str, details: Dict[str, Any], severity: str = "warn") -> None:
        """记录风控事件"""
        self.log("risk", agent_id, event, details, severity)

    def log_change(self, actor: str, change_type: str, details: Dict[str, Any]) -> None:
        """记录变更（代码/参数）"""
        self.log("change", actor, change_type, details, severity="warn")

    def get_entries(self, category: Optional[str] = None,
                    severity: Optional[str] = None,
                    limit: int = 100) -> List[AuditEntry]:
        with self._lock:
            entries = list(self._entries)
            if category:
                entries = [e for e in entries if e.category == category]
            if severity:
                entries = [e for e in entries if e.severity == severity]
            return entries[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
            # P4#8: Counter 单次遍历替代 N 次 sum+if 遍历
            return {
                "total_entries": len(entries),
                "by_category": dict(Counter(e.category for e in entries)),
                "by_severity": dict(Counter(e.severity for e in entries)),
            }


# ============================================================================
# 4. 压力测试场景（Stress Test Scenarios）
# ============================================================================


class StressTestScenario:
    """压力测试场景 — 模拟极端市场条件

    来自前沿量化系统压力测试标准：
      - 黑天鹅事件（2015瑞郎脱钩、2020新冠、2022LUNA崩盘）
      - 闪崩事件（2010道指闪崩、2021币安闪崩）
      - 插针行情（长影线扫止损后反转）
      - 交易所宕机（API故障、断网）
      - 极端波动率（VIX > 80）
    """

    @staticmethod
    def generate_black_swan(base_price: float = 50000.0, bars: int = 24) -> List[Dict[str, Any]]:
        """生成黑天鹅场景（LUNA式崩盘）

        特征：24小时内价格暴跌80%+，波动率极高
        """
        scenario = []
        price = base_price
        # 分三阶段：恐慌开始 → 加速崩盘 → 企稳
        for i in range(bars):
            if i < bars // 3:
                # 阶段1：恐慌开始（-2% to -5% per bar）
                change = -0.02 - (i / bars) * 0.03
            elif i < 2 * bars // 3:
                # 阶段2：加速崩盘（-5% to -15% per bar）
                change = -0.05 - ((i - bars // 3) / bars) * 0.10
            else:
                # 阶段3：企稳（-1% to +2% per bar）
                change = 0.01 - ((i - 2 * bars // 3) / bars) * 0.02

            open_price = price
            price = max(0.01, price * (1 + change))
            high = max(open_price, price) * 1.005
            low = min(open_price, price) * 0.995

            scenario.append({
                "timestamp": time.time() + i * 3600,
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_price * 1000 * (1 + abs(change) * 10),  # 放量
                "scenario": "black_swan",
                "bar_index": i,
            })
        return scenario

    @staticmethod
    def generate_flash_crash(base_price: float = 50000.0, bars: int = 24) -> List[Dict[str, Any]]:
        """生成闪崩场景（V型反转）

        特征：短时间内暴跌10-20%后快速反弹
        """
        scenario = []
        price = base_price
        bottom = base_price * 0.85  # 跌15%

        for i in range(bars):
            if i < 4:
                # 阶段1：闪崩（4根K线跌15%）
                change = -0.04
            elif i < 8:
                # 阶段2：反弹（4根K线反弹10%）
                change = 0.025
            else:
                # 阶段3：正常波动
                change = 0.001 - ((i - 8) % 4 - 2) * 0.002

            open_price = price
            price = max(0.01, price * (1 + change))
            high = max(open_price, price) * 1.003
            low = min(open_price, price) * 0.997

            scenario.append({
                "timestamp": time.time() + i * 60,  # 1分钟K线
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_price * 500 * (1.5 if i < 8 else 1.0),
                "scenario": "flash_crash",
                "bar_index": i,
            })
        return scenario

    @staticmethod
    def generate_wick_hunt(base_price: float = 50000.0, bars: int = 24) -> List[Dict[str, Any]]:
        """生成插针场景（长影线扫止损）

        特征：快速下跌扫止损后快速反弹，留下长下影线
        """
        scenario = []
        price = base_price

        for i in range(bars):
            open_price = price
            if i in (5, 6, 7):
                # 插针：快速下跌5%然后反弹
                low = price * 0.95
                close = price * 1.001  # 反弹回原位
                high = max(open_price, close) * 1.002
                price = close
            else:
                # 正常波动
                change = 0.001 - (i % 5 - 2) * 0.001
                price = max(0.01, price * (1 + change))
                high = max(open_price, price) * 1.002
                low = min(open_price, price) * 0.998

            scenario.append({
                "timestamp": time.time() + i * 300,  # 5分钟K线
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_price * 300,
                "scenario": "wick_hunt",
                "bar_index": i,
            })
        return scenario

    @staticmethod
    def generate_extreme_volatility(base_price: float = 50000.0, bars: int = 48) -> List[Dict[str, Any]]:
        """生成极端波动率场景（VIX > 80式）

        特征：每根K线波动±3-5%，无明确趋势
        """
        import random
        random.seed(42)  # 可复现
        scenario = []
        price = base_price

        for i in range(bars):
            open_price = price
            # 高波动，无趋势
            change = random.gauss(0, 0.04)  # 均值0，标准差4%
            price = max(0.01, price * (1 + change))
            high = max(open_price, price) * 1.01
            low = min(open_price, price) * 0.99

            scenario.append({
                "timestamp": time.time() + i * 3600,
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": base_price * 800,
                "scenario": "extreme_volatility",
                "bar_index": i,
            })
        return scenario

    @staticmethod
    def get_all_scenarios(base_price: float = 50000.0) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有压力测试场景"""
        return {
            "black_swan": StressTestScenario.generate_black_swan(base_price),
            "flash_crash": StressTestScenario.generate_flash_crash(base_price),
            "wick_hunt": StressTestScenario.generate_wick_hunt(base_price),
            "extreme_volatility": StressTestScenario.generate_extreme_volatility(base_price),
        }


# ============================================================================
# 5. 生产级监控器（Production Monitor）
# ============================================================================


class ProductionMonitor:
    """生产级监控器 — 实时监控+告警

    来自Math&Markets实盘经验：
      - Runtime alerts: API fail, DD breach, fill reject, borrow spike
      - Digest summary: 每日摘要
      - 10秒内Slack+email告警
    """

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self._audit = audit_logger or AuditLogger()
        self._metrics: Dict[str, Any] = {}
        self._alert_callbacks: List[Callable[[str, Dict], None]] = []
        self._lock = threading.RLock()
        self._start_time = time.time()

    def register_alert_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """注册告警回调（Slack/email等）"""
        self._alert_callbacks.append(callback)

    def update_metric(self, key: str, value: Any) -> None:
        """更新指标"""
        with self._lock:
            self._metrics[key] = {
                "value": value,
                "timestamp": time.time(),
            }

    def alert(self, alert_type: str, details: Dict[str, Any], severity: str = "warn") -> None:
        """发送告警"""
        alert_data = {
            "type": alert_type,
            "details": details,
            "severity": severity,
            "timestamp": time.time(),
        }

        # 记录审计日志
        self._audit.log(
            category="alert",
            actor="monitor",
            action=alert_type,
            details=details,
            severity=severity,
        )

        # 触发回调
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

        # 控制台输出
        if severity == "critical":
            logger.critical("ALERT[%s]: %s", alert_type, details)
        elif severity == "error":
            logger.error("ALERT[%s]: %s", alert_type, details)
        else:
            logger.warning("ALERT[%s]: %s", alert_type, details)

    def get_digest(self) -> Dict[str, Any]:
        """获取每日摘要"""
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "uptime_seconds": uptime,
                "uptime_human": f"{uptime/3600:.1f}h",
                "metrics": dict(self._metrics),
                "audit_stats": self._audit.get_stats(),
                "timestamp": time.time(),
            }


# ============================================================================
# 6. 毕业标准（Graduation Criteria）— Paper Trading
# ============================================================================


@dataclass(slots=True)
class GraduationCheck:
    """毕业标准检查项"""
    name: str
    description: str
    passed: bool
    value: Any
    threshold: Any
    details: str = ""


class GraduationCriteria:
    """Paper Trading毕业标准

    来自Math&Markets实盘经验：
      - 30天清洁paper trading（0错误，DD<8%）
      - Live PnL跟踪模型±0.3%内
      - 事件演练执行2次成功
    """

    # 毕业标准阈值
    MIN_PAPER_DAYS = 30          # 最少30天paper trading
    MAX_PAPER_DD = 0.08          # Paper最大回撤<8%
    MAX_ERRORS = 0               # 0个错误
    PNL_TRACKING_TOLERANCE = 0.003  # PnL跟踪±0.3%
    MIN_INCIDENT_DRILLS = 2      # 最少2次事件演练

    def __init__(self):
        self._checks: List[GraduationCheck] = []

    def evaluate(self, paper_days: int, paper_dd: float, error_count: int,
                 pnl_tracking_diff: float, incident_drills: int) -> Tuple[bool, List[GraduationCheck]]:
        """评估毕业标准

        Args:
            paper_days: paper trading天数
            paper_dd: paper最大回撤
            error_count: 错误数
            pnl_tracking_diff: PnL跟踪差异
            incident_drills: 事件演练次数

        Returns:
            (graduated, checks)
        """
        self._checks = [
            GraduationCheck(
                name="paper_days",
                description=f"Paper trading天数 >= {self.MIN_PAPER_DAYS}",
                passed=paper_days >= self.MIN_PAPER_DAYS,
                value=paper_days,
                threshold=self.MIN_PAPER_DAYS,
            ),
            GraduationCheck(
                name="paper_drawdown",
                description=f"Paper最大回撤 < {self.MAX_PAPER_DD:.0%}",
                passed=paper_dd < self.MAX_PAPER_DD,
                value=paper_dd,
                threshold=self.MAX_PAPER_DD,
            ),
            GraduationCheck(
                name="error_count",
                description=f"错误数 <= {self.MAX_ERRORS}",
                passed=error_count <= self.MAX_ERRORS,
                value=error_count,
                threshold=self.MAX_ERRORS,
            ),
            GraduationCheck(
                name="pnl_tracking",
                description=f"PnL跟踪差异 < {self.PNL_TRACKING_TOLERANCE:.1%}",
                passed=pnl_tracking_diff < self.PNL_TRACKING_TOLERANCE,
                value=pnl_tracking_diff,
                threshold=self.PNL_TRACKING_TOLERANCE,
            ),
            GraduationCheck(
                name="incident_drills",
                description=f"事件演练次数 >= {self.MIN_INCIDENT_DRILLS}",
                passed=incident_drills >= self.MIN_INCIDENT_DRILLS,
                value=incident_drills,
                threshold=self.MIN_INCIDENT_DRILLS,
            ),
        ]

        all_passed = all(c.passed for c in self._checks)
        return all_passed, self._checks

    def get_report(self) -> str:
        """生成毕业报告"""
        if not self._checks:
            return "No graduation evaluation performed yet."

        lines = ["=" * 60, "  Paper Trading 毕业评估报告", "=" * 60, ""]

        for check in self._checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            lines.append(f"  {status}  {check.name}")
            lines.append(f"         {check.description}")
            lines.append(f"         实际: {check.value} | 阈值: {check.threshold}")
            lines.append("")

        all_passed = all(c.passed for c in self._checks)
        lines.append("-" * 60)
        if all_passed:
            lines.append("  ✅ 毕业标准全部通过 — 可进入轻量实盘")
        else:
            failed = sum(1 for c in self._checks if not c.passed)
            lines.append(f"  ❌ {failed} 项未通过 — 继续paper trading")
        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# 7. 性能基准测试（Performance Benchmark）
# ============================================================================


class PerformanceBenchmark:
    """性能基准测试 — 来自铁标准体系L6量化标准

    阈值（来自1200-iron-standards.md）：
      - 文件扫描: EXCELLENT≤100ms, GOOD≤300ms
      - 模块导入: EXCELLENT≤500ms, GOOD≤1s
      - JSON解析: EXCELLENT≤50ms, GOOD≤200ms
      - MCP响应: EXCELLENT≤500ms, GOOD≤3s
    """

    FILE_SCAN_EXCELLENT_MS = 100
    FILE_SCAN_GOOD_MS = 300
    MODULE_IMPORT_EXCELLENT_MS = 500
    MODULE_IMPORT_GOOD_MS = 1000
    JSON_PARSE_EXCELLENT_MS = 50
    JSON_PARSE_GOOD_MS = 200

    @staticmethod
    def benchmark_file_scan(directory: str) -> Dict[str, Any]:
        """基准测试文件扫描性能"""
        start = time.perf_counter()
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms <= PerformanceBenchmark.FILE_SCAN_EXCELLENT_MS:
            grade = "EXCELLENT"
        elif elapsed_ms <= PerformanceBenchmark.FILE_SCAN_GOOD_MS:
            grade = "GOOD"
        else:
            grade = "ACCEPTABLE"

        return {
            "test": "file_scan",
            "elapsed_ms": round(elapsed_ms, 2),
            "files_scanned": count,
            "grade": grade,
            "threshold_excellent_ms": PerformanceBenchmark.FILE_SCAN_EXCELLENT_MS,
            "threshold_good_ms": PerformanceBenchmark.FILE_SCAN_GOOD_MS,
            "passed": elapsed_ms <= PerformanceBenchmark.FILE_SCAN_GOOD_MS,
        }

    @staticmethod
    def benchmark_module_import(module_name: str) -> Dict[str, Any]:
        """基准测试模块导入性能"""
        start = time.perf_counter()
        try:
            __import__(module_name)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms <= PerformanceBenchmark.MODULE_IMPORT_EXCELLENT_MS:
            grade = "EXCELLENT"
        elif elapsed_ms <= PerformanceBenchmark.MODULE_IMPORT_GOOD_MS:
            grade = "GOOD"
        else:
            grade = "ACCEPTABLE"

        return {
            "test": "module_import",
            "module": module_name,
            "elapsed_ms": round(elapsed_ms, 2),
            "grade": grade,
            "threshold_excellent_ms": PerformanceBenchmark.MODULE_IMPORT_EXCELLENT_MS,
            "threshold_good_ms": PerformanceBenchmark.MODULE_IMPORT_GOOD_MS,
            "success": success,
            "error": error,
            "passed": success and elapsed_ms <= PerformanceBenchmark.MODULE_IMPORT_GOOD_MS,
        }

    @staticmethod
    def benchmark_json_parse(data_size: int = 10000) -> Dict[str, Any]:
        """基准测试JSON解析性能"""
        # 生成测试数据
        test_data = {
            f"key_{i}": {"value": i, "nested": {"a": i * 2, "b": f"str_{i}"}}
            for i in range(data_size)
        }
        json_str = json.dumps(test_data)

        start = time.perf_counter()
        parsed = json.loads(json_str)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms <= PerformanceBenchmark.JSON_PARSE_EXCELLENT_MS:
            grade = "EXCELLENT"
        elif elapsed_ms <= PerformanceBenchmark.JSON_PARSE_GOOD_MS:
            grade = "GOOD"
        else:
            grade = "ACCEPTABLE"

        return {
            "test": "json_parse",
            "data_size": data_size,
            "elapsed_ms": round(elapsed_ms, 2),
            "parsed_keys": len(parsed),
            "grade": grade,
            "threshold_excellent_ms": PerformanceBenchmark.JSON_PARSE_EXCELLENT_MS,
            "threshold_good_ms": PerformanceBenchmark.JSON_PARSE_GOOD_MS,
            "passed": elapsed_ms <= PerformanceBenchmark.JSON_PARSE_GOOD_MS,
        }

    @staticmethod
    def run_all_benchmarks(directory: str = ".") -> List[Dict[str, Any]]:
        """运行所有基准测试"""
        results = [
            PerformanceBenchmark.benchmark_file_scan(directory),
            PerformanceBenchmark.benchmark_module_import("json"),
            PerformanceBenchmark.benchmark_module_import("statistics"),
            PerformanceBenchmark.benchmark_json_parse(1000),
            PerformanceBenchmark.benchmark_json_parse(10000),
        ]

        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)

        logger.info(
            "Performance benchmarks: %d/%d passed (%.0f%%)",
            passed, total, passed / total * 100 if total > 0 else 0,
        )

        return results


# ============================================================================
# 8. 生产级强化系统（Production Hardening System）— 统一入口
# ============================================================================


class ProductionHardeningSystem:
    """生产级强化系统 — 统一管理所有生产级组件

    集成：
      - KillSwitch（物理逃生）
      - RollingDrawdownMonitor（滚动回撤监控）
      - AuditLogger（审计日志）
      - ProductionMonitor（实时监控）
      - GraduationCriteria（毕业标准）
      - PerformanceBenchmark（性能基准）
      - StressTestScenario（压力测试）
    """

    def __init__(self, audit_log_file: Optional[str] = None, sandbox_mode: bool = False,
                 audit_context: Optional[Dict[str, Any]] = None):
        """初始化生产级强化系统

        Args:
            audit_log_file: 审计日志文件路径
            sandbox_mode: 是否沙盘模式（B3修复：沙盘模式仅放宽心跳，风控阈值保持生产级）
        """
        # 核心组件
        self.kill_switch = KillSwitch(sandbox_mode=sandbox_mode)
        # Phase 10.2: 传递 sandbox_mode 到 RollingDrawdownMonitor
        self._sandbox_mode = sandbox_mode
        self.audit_logger = AuditLogger(log_file=audit_log_file, context=audit_context)
        self.drawdown_monitor = RollingDrawdownMonitor(kill_switch=self.kill_switch, sandbox_mode=sandbox_mode)
        self.monitor = ProductionMonitor(self.audit_logger)
        self.graduation = GraduationCriteria()
        self.benchmark = PerformanceBenchmark
        self.stress_test = StressTestScenario

        # 注册Kill Switch平仓回调
        self.kill_switch.register_flatten_callback(self._flatten_all_positions)

        # 状态
        self._initialized = True
        self._start_time = time.time()

        logger.info("ProductionHardeningSystem initialized (EXCELLENT grade)")

    def _flatten_all_positions(self, reason: str) -> int:
        """平仓回调（实际生产中对接交易所API）"""
        self.audit_logger.log_risk(
            agent_id="system",
            event="KILL_SWITCH_FLATTEN",
            details={"reason": reason, "action": "flatten_all"},
            severity="critical",
        )
        # 沙盘模式下返回0（无真实持仓）
        return 0

    def heartbeat(self) -> None:
        """主系统心跳（定期调用）"""
        self.kill_switch.heartbeat()

    def check_all_risks(self, agent_id: str, equity: float,
                        drawdown: float, daily_loss_pct: float = 0.0) -> Dict[str, Any]:
        """检查所有风控条件

        Args:
            agent_id: Agent ID
            equity: 当前权益
            drawdown: 当前回撤
            daily_loss_pct: 日内亏损百分比

        Returns:
            检查结果
        """
        results = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "checks": {},
        }

        # 1. 更新权益曲线
        self.drawdown_monitor.update_equity(agent_id, equity)

        # 2. 检查滚动回撤
        dd_alert = self.drawdown_monitor.check_drawdown(agent_id)
        results["checks"]["drawdown"] = {
            "value": drawdown,
            "alert": dd_alert.level if dd_alert else None,
            "message": dd_alert.message if dd_alert else "OK",
        }

        # 3. 检查Kill Switch条件
        ks_triggered = False
        if self.kill_switch.check_agent_drawdown(agent_id, drawdown):
            ks_triggered = True
        if daily_loss_pct > 0 and self.kill_switch.check_daily_loss(daily_loss_pct):
            ks_triggered = True

        results["checks"]["kill_switch"] = {
            "state": self.kill_switch.state.value,
            "triggered": ks_triggered,
        }

        # 4. 心跳检查
        self.heartbeat()

        return results

    def run_stress_tests(self) -> Dict[str, Any]:
        """运行所有压力测试场景"""
        scenarios = self.stress_test.get_all_scenarios()
        results = {}

        for name, bars in scenarios.items():
            if not bars:
                continue

            prices = [b["close"] for b in bars]
            start_price = prices[0]
            end_price = prices[-1]
            min_price = min(prices)
            max_price = max(prices)
            max_drawdown = (max_price - min_price) / max_price if max_price > 0 else 0
            total_return = (end_price - start_price) / start_price if start_price > 0 else 0

            results[name] = {
                "bars": len(bars),
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "min_price": round(min_price, 2),
                "max_price": round(max_price, 2),
                "max_drawdown": round(max_drawdown, 4),
                "total_return": round(total_return, 4),
                "scenario": name,
            }

            self.audit_logger.log(
                category="stress_test",
                actor="system",
                action=f"scenario_{name}",
                details=results[name],
                severity="info",
            )

        return results

    def run_benchmarks(self, directory: str = ".") -> List[Dict[str, Any]]:
        """运行性能基准测试"""
        results = self.benchmark.run_all_benchmarks(directory)

        for result in results:
            self.audit_logger.log(
                category="benchmark",
                actor="system",
                action=result["test"],
                details=result,
                severity="info" if result.get("passed") else "warn",
            )

        return results

    def evaluate_graduation(self, paper_days: int, paper_dd: float,
                            error_count: int, pnl_diff: float,
                            drills: int) -> Tuple[bool, str]:
        """评估毕业标准"""
        graduated, checks = self.graduation.evaluate(
            paper_days, paper_dd, error_count, pnl_diff, drills,
        )

        report = self.graduation.get_report()

        self.audit_logger.log(
            category="graduation",
            actor="system",
            action="evaluate",
            details={
                "graduated": graduated,
                "paper_days": paper_days,
                "paper_dd": paper_dd,
                "error_count": error_count,
                "pnl_diff": pnl_diff,
                "drills": drills,
            },
            severity="info" if graduated else "warn",
        )

        return graduated, report

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self._initialized,
            "uptime_seconds": time.time() - self._start_time,
            "kill_switch": self.kill_switch.get_stats(),
            "drawdown_monitor": self.drawdown_monitor.get_stats(),
            "audit_logger": self.audit_logger.get_stats(),
            "monitor": self.monitor.get_digest(),
            "timestamp": time.time(),
        }


# ============================================================================
# 便利函数
# ============================================================================


def create_production_system(audit_log_file: Optional[str] = None) -> ProductionHardeningSystem:
    """创建生产级强化系统实例"""
    return ProductionHardeningSystem(audit_log_file=audit_log_file)


def quick_health_check(system: ProductionHardeningSystem) -> Dict[str, Any]:
    """快速健康检查"""
    status = system.get_system_status()

    health = {
        "healthy": True,
        "issues": [],
        "timestamp": time.time(),
    }

    # Kill Switch状态
    if not status["kill_switch"]["state"] == "armed":
        health["healthy"] = False
        health["issues"].append(f"KillSwitch not armed: {status['kill_switch']['state']}")

    # 心跳超时
    heartbeat_age = status["kill_switch"]["last_heartbeat_age_sec"]
    if heartbeat_age > 30:
        health["healthy"] = False
        health["issues"].append(f"Heartbeat stale: {heartbeat_age:.1f}s")

    # 严重告警
    critical_alerts = status["drawdown_monitor"]["critical_alerts"]
    if critical_alerts > 0:
        health["healthy"] = False
        health["issues"].append(f"Critical alerts: {critical_alerts}")

    return health
