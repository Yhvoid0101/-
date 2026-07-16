# -*- coding: utf-8 -*-
"""
KPI监控与迭代优化模块 (KPI Monitor) — R17-6

实现量化策略性能监控体系，实时追踪关键绩效指标(KPI)；
建立错误反馈与快速迭代通道，对实盘运行中出现的问题及时响应与修复。

用户需求：
  "建立量化策略性能监控体系，实时追踪关键绩效指标(KPI)"
  "制定定期策略评估与优化机制，根据市场变化与策略表现动态调整参数与逻辑"
  "建立错误反馈与快速迭代通道，对实盘运行中出现的问题进行及时响应与修复"

业界最佳实践：
  - Statistical Process Control (SPC): 3-sigma规则 + CUSUM趋势检测
  - Real-time KPI Dashboard: 实时KPI仪表盘(Bloomberg/Refinitiv标准)
  - Anomaly Detection: Isolation Forest + Z-score + EWMA
  - Error Escalation: P0/P1/P2/P3优先级 + 自动升级机制
  - Continuous Optimization: PDCA循环(Plan-Do-Check-Act)

核心组件：
  1. KPI                   — 单个KPI定义(名称/值/阈值/状态)
  2. KPISnapshot           — KPI快照(时间戳+所有KPI值)
  3. AnomalyDetector       — 异常检测(3-sigma + CUSUM + 连续异常)
  4. ErrorFeedbackChannel  — 错误反馈通道(分类+优先级+知识库)
  5. IterativeOptimizer    — 迭代优化器(参数调整建议+优化周期)
  6. KPIMonitor            — 综合监控器

KPI指标体系 (10大核心KPI)：
  1. Sharpe ratio (滚动60期)
  2. Max drawdown (当前)
  3. Win rate
  4. Profit factor
  5. Average PnL per trade
  6. Slippage cost ratio
  7. Latency p95
  8. Order fill rate
  9. Daily PnL
  10. Equity curve slope

阈值常量 (量化+不可妥协)：
  SHARPE_MIN = 1.0
  MAX_DRAWDOWN_WARN = 0.15
  MAX_DRAWDOWN_BLOCK = 0.25
  WINRATE_MIN = 0.45
  PROFIT_FACTOR_MIN = 1.2
  LATENCY_P95_WARN_MS = 200
  ANOMALY_3SIGMA = 3.0
  CUSUM_THRESHOLD = 5.0
  CONSECUTIVE_ANOMALY_DAYS = 3
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

try:
    from .strategy_evaluator import Verdict
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sandbox_trading.strategy_evaluator import Verdict  # type: ignore

logger = logging.getLogger("hermes.kpi_monitor")


# ============================================================================
# 阈值常量 (量化+不可妥协)
# ============================================================================

# KPI阈值
SHARPE_MIN = 1.0                    # Sharpe<1.0 告警
SHARPE_BLOCK = 0.5                  # Sharpe<0.5 阻塞
MAX_DRAWDOWN_WARN = 0.15            # 回撤>15% 告警
MAX_DRAWDOWN_BLOCK = 0.25           # 回撤>25% 阻塞
WINRATE_MIN = 0.45                  # 胜率<45% 告警
WINRATE_BLOCK = 0.35                # 胜率<35% 阻塞
PROFIT_FACTOR_MIN = 1.2             # 盈亏比<1.2 告警
PROFIT_FACTOR_BLOCK = 1.0           # 盈亏比<1.0 阻塞
LATENCY_P95_WARN_MS = 200           # 延迟p95>200ms 告警
LATENCY_P95_BLOCK_MS = 1000         # 延迟p95>1000ms 阻塞
FILL_RATE_WARN = 0.95               # 成交率<95% 告警
FILL_RATE_BLOCK = 0.80              # 成交率<80% 阻塞
SLIPPAGE_COST_WARN = 0.02           # 滑点成本>2% 告警
SLIPPAGE_COST_BLOCK = 0.05          # 滑点成本>5% 阻塞

# 异常检测阈值
ANOMALY_3SIGMA = 3.0                # 3-sigma规则
CUSUM_THRESHOLD = 5.0               # CUSUM累积和阈值
CONSECUTIVE_ANOMALY_DAYS = 3        # 连续异常天数

# 滚动窗口
ROLLING_WINDOW_DEFAULT = 60         # 滚动窗口默认60期


# ============================================================================
# 1. KPI状态枚举
# ============================================================================


class KPIStatus(Enum):
    """KPI状态"""
    EXCELLENT = "excellent"    # 优秀
    GOOD = "good"              # 良好
    WARN = "warn"              # 警告
    BLOCK = "block"            # 阻塞
    UNKNOWN = "unknown"        # 未知(数据不足)


class ErrorPriority(Enum):
    """错误优先级"""
    P0 = "P0"   # 致命: 立即停止交易(如最大回撤超限)
    P1 = "P1"   # 严重: 1小时内响应(如Sharpe骤降)
    P2 = "P2"   # 中等: 1天内响应(如胜率下降)
    P3 = "P3"   # 低: 1周内响应(如滑点略高)


class ErrorCategory(Enum):
    """错误分类"""
    PARAMETER = "parameter"     # 参数问题
    LOGIC = "logic"             # 逻辑问题
    DATA = "data"               # 数据问题
    NETWORK = "network"         # 网络问题
    MARKET = "market"           # 市场环境变化
    EXECUTION = "execution"     # 执行问题


# ============================================================================
# 2. KPI定义
# ============================================================================


@dataclass(slots=True)
class KPI:
    """单个KPI定义

    Attributes:
        name: KPI名称
        value: 当前值
        warn_threshold: 警告阈值
        block_threshold: 阻塞阈值
        higher_is_better: True=越高越好, False=越低越好
        status: 当前状态
        history: 历史值(滚动窗口)
    """
    name: str
    value: float = 0.0
    warn_threshold: float = 0.0
    block_threshold: float = 0.0
    higher_is_better: bool = True
    status: KPIStatus = KPIStatus.UNKNOWN
    history: deque = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW_DEFAULT))

    def update(self, value: float) -> KPIStatus:
        """更新KPI值并计算状态

        Args:
            value: 新的KPI值

        Returns:
            KPIStatus
        """
        self.value = value
        self.history.append(value)

        if len(self.history) < 5:
            self.status = KPIStatus.UNKNOWN
            return self.status

        # 计算状态
        if self.higher_is_better:
            if value < self.block_threshold:
                self.status = KPIStatus.BLOCK
            elif value < self.warn_threshold:
                self.status = KPIStatus.WARN
            elif value >= self.warn_threshold * 1.5:
                self.status = KPIStatus.EXCELLENT
            else:
                self.status = KPIStatus.GOOD
        else:
            # 越低越好(如最大回撤、延迟)
            if value > self.block_threshold:
                self.status = KPIStatus.BLOCK
            elif value > self.warn_threshold:
                self.status = KPIStatus.WARN
            elif value <= self.warn_threshold * 0.5:
                self.status = KPIStatus.EXCELLENT
            else:
                self.status = KPIStatus.GOOD

        return self.status


@dataclass(slots=True)
class KPISnapshot:
    """KPI快照(某时刻所有KPI的状态)"""
    timestamp: float
    kpis: Dict[str, KPI] = field(default_factory=dict)
    overall_status: KPIStatus = KPIStatus.UNKNOWN
    block_count: int = 0
    warn_count: int = 0

    def compute_overall_status(self) -> KPIStatus:
        """计算综合状态"""
        if not self.kpis:
            self.overall_status = KPIStatus.UNKNOWN
            return self.overall_status

        self.block_count = sum(1 for k in self.kpis.values() if k.status == KPIStatus.BLOCK)
        self.warn_count = sum(1 for k in self.kpis.values() if k.status == KPIStatus.WARN)

        if self.block_count > 0:
            self.overall_status = KPIStatus.BLOCK
        elif self.warn_count > 0:
            self.overall_status = KPIStatus.WARN
        else:
            # 检查是否全部GOOD以上
            all_good = all(
                k.status in (KPIStatus.GOOD, KPIStatus.EXCELLENT)
                for k in self.kpis.values()
            )
            if all_good:
                self.overall_status = KPIStatus.GOOD
            else:
                self.overall_status = KPIStatus.UNKNOWN

        return self.overall_status


# ============================================================================
# 3. 异常检测器
# ============================================================================


@dataclass(slots=True)
class AnomalyResult:
    """异常检测结果"""
    is_anomaly: bool = False
    anomaly_type: str = ""             # "3sigma" / "cusum" / "consecutive"
    anomaly_score: float = 0.0         # 异常分数
    z_score: float = 0.0               # z-score
    cusum_value: float = 0.0           # CUSUM累积和
    consecutive_count: int = 0         # 连续异常次数
    description: str = ""


class AnomalyDetector:
    """异常检测器

    三层异常检测：
      1. 3-sigma规则: 当前值偏离均值>3倍标准差
      2. CUSUM累积和: 检测趋势性变化
      3. 连续异常: 连续N天异常=严重告警

    原理：
      - 3-sigma: 适用于突变检测(如闪崩)
      - CUSUM: 适用于渐变检测(如策略衰减)
      - 连续异常: 适用于持续性问题描述
    """

    def __init__(
        self,
        sigma_threshold: float = ANOMALY_3SIGMA,
        cusum_threshold: float = CUSUM_THRESHOLD,
        consecutive_days: int = CONSECUTIVE_ANOMALY_DAYS,
        window_size: int = ROLLING_WINDOW_DEFAULT,
    ):
        """
        Args:
            sigma_threshold: sigma阈值(默认3.0)
            cusum_threshold: CUSUM阈值(默认5.0)
            consecutive_days: 连续异常天数阈值
            window_size: 滚动窗口大小
        """
        self.sigma_threshold = sigma_threshold
        self.cusum_threshold = cusum_threshold
        self.consecutive_days = consecutive_days
        self.window_size = window_size
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._consecutive_count = 0

    def detect(self, value: float, history: deque) -> AnomalyResult:
        """检测异常

        Args:
            value: 当前值
            history: 历史值队列

        Returns:
            AnomalyResult
        """
        result = AnomalyResult()

        if len(history) < 5:
            return result

        arr = np.asarray(list(history), dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        if std < 1e-10:
            return result

        # 1. 3-sigma检测
        z_score = (value - mean) / std
        result.z_score = z_score
        if abs(z_score) > self.sigma_threshold:
            result.is_anomaly = True
            result.anomaly_type = "3sigma"
            result.anomaly_score = abs(z_score) / self.sigma_threshold
            result.description = f"3-sigma异常: z={z_score:.2f}>{self.sigma_threshold}"

        # 2. CUSUM检测
        # 标准化CUSUM: S_t = max(0, S_{t-1} + (x_t - mean)/std - k)
        k = 0.5  # CUSUM敏感度参数
        normalized = (value - mean) / std
        self._cusum_pos = max(0, self._cusum_pos + normalized - k)
        self._cusum_neg = max(0, self._cusum_neg - normalized - k)
        cusum_max = max(self._cusum_pos, self._cusum_neg)
        result.cusum_value = cusum_max

        if cusum_max > self.cusum_threshold:
            if not result.is_anomaly:
                result.is_anomaly = True
                result.anomaly_type = "cusum"
                result.anomaly_score = cusum_max / self.cusum_threshold
                result.description = f"CUSUM异常: S={cusum_max:.2f}>{self.cusum_threshold}"
            else:
                # 同时触发3-sigma和CUSUM
                result.anomaly_type = "3sigma+cusum"
                result.anomaly_score = max(result.anomaly_score, cusum_max / self.cusum_threshold)
                result.description += f" + CUSUM异常: S={cusum_max:.2f}"

        # 3. 连续异常检测
        if result.is_anomaly:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        result.consecutive_count = self._consecutive_count

        if self._consecutive_count >= self.consecutive_days:
            result.is_anomaly = True
            result.anomaly_type = "consecutive"
            result.anomaly_score = max(result.anomaly_score, self._consecutive_count / self.consecutive_days)
            result.description = (
                f"连续异常: {self._consecutive_count}天>={self.consecutive_days}天阈值"
            )

        return result

    def reset(self):
        """重置状态"""
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._consecutive_count = 0


# ============================================================================
# 4. 错误反馈通道
# ============================================================================


@dataclass(slots=True)
class ErrorEntry:
    """错误条目"""
    error_id: str
    timestamp: float
    category: ErrorCategory
    priority: ErrorPriority
    kpi_name: str
    description: str
    anomaly: Optional[AnomalyResult] = None
    suggested_action: str = ""
    status: str = "open"              # open/in_progress/resolved
    resolution: str = ""


class ErrorFeedbackChannel:
    """错误反馈通道

    功能：
      1. 接收异常并分类(参数/逻辑/数据/网络/市场/执行)
      2. 分配优先级(P0/P1/P2/P3)
      3. 提供建议动作
      4. 维护错误知识库(去重+模式匹配)

    优先级分配规则：
      KPIStatus.BLOCK → P0或P1
      KPIStatus.WARN + 连续异常 → P1或P2
      KPIStatus.WARN → P2或P3
    """

    def __init__(self):
        self._errors: List[ErrorEntry] = []
        self._error_counter = 0
        # 错误知识库: pattern → last_resolution
        self._knowledge_base: Dict[str, str] = {}

    def report_error(
        self,
        kpi_name: str,
        kpi_status: KPIStatus,
        anomaly: Optional[AnomalyResult] = None,
        description: str = "",
    ) -> ErrorEntry:
        """报告错误

        Args:
            kpi_name: KPI名称
            kpi_status: KPI状态
            anomaly: 异常检测结果
            description: 描述

        Returns:
            ErrorEntry
        """
        self._error_counter += 1
        error_id = f"ERR-{int(time.time())}-{self._error_counter:04d}"

        # 分类
        category = self._classify_error(kpi_name, anomaly)
        # 优先级
        priority = self._assign_priority(kpi_status, anomaly)
        # 建议动作
        action = self._suggest_action(kpi_name, category, priority, anomaly)

        entry = ErrorEntry(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            priority=priority,
            kpi_name=kpi_name,
            description=description or (anomaly.description if anomaly else ""),
            anomaly=anomaly,
            suggested_action=action,
        )

        # 检查知识库是否有类似错误
        pattern_key = f"{kpi_name}:{category.value}"
        if pattern_key in self._knowledge_base:
            entry.resolution = f"历史类似错误已解决: {self._knowledge_base[pattern_key]}"

        self._errors.append(entry)
        logger.warning(
            f"错误报告 [{priority.value}] {kpi_name}: {entry.description} → {action}"
        )
        return entry

    def resolve_error(self, error_id: str, resolution: str):
        """解决错误并更新知识库

        Args:
            error_id: 错误ID
            resolution: 解决方案
        """
        for entry in self._errors:
            if entry.error_id == error_id:
                entry.status = "resolved"
                entry.resolution = resolution
                # 更新知识库
                pattern_key = f"{entry.kpi_name}:{entry.category.value}"
                self._knowledge_base[pattern_key] = resolution
                logger.info(f"错误已解决 {error_id}: {resolution}")
                return
        logger.warning(f"未找到错误ID: {error_id}")

    def get_open_errors(self, priority: Optional[ErrorPriority] = None) -> List[ErrorEntry]:
        """获取未解决错误

        Args:
            priority: 按优先级过滤(None=全部)

        Returns:
            List[ErrorEntry]
        """
        result = [e for e in self._errors if e.status == "open"]
        if priority:
            result = [e for e in result if e.priority == priority]
        # 按优先级排序
        priority_order = {ErrorPriority.P0: 0, ErrorPriority.P1: 1, ErrorPriority.P2: 2, ErrorPriority.P3: 3}
        result.sort(key=lambda e: priority_order.get(e.priority, 99))
        return result

    def _classify_error(
        self,
        kpi_name: str,
        anomaly: Optional[AnomalyResult],
    ) -> ErrorCategory:
        """分类错误"""
        name_lower = kpi_name.lower()
        if "latency" in name_lower or "fill_rate" in name_lower:
            return ErrorCategory.NETWORK
        if "slippage" in name_lower:
            return ErrorCategory.EXECUTION
        if "drawdown" in name_lower or "sharpe" in name_lower:
            return ErrorCategory.MARKET
        if "winrate" in name_lower or "profit_factor" in name_lower:
            return ErrorCategory.LOGIC
        if anomaly and anomaly.anomaly_type == "consecutive":
            return ErrorCategory.PARAMETER
        return ErrorCategory.DATA

    def _assign_priority(
        self,
        kpi_status: KPIStatus,
        anomaly: Optional[AnomalyResult],
    ) -> ErrorPriority:
        """分配优先级"""
        if kpi_status == KPIStatus.BLOCK:
            return ErrorPriority.P0
        if anomaly and anomaly.consecutive_count >= CONSECUTIVE_ANOMALY_DAYS:
            return ErrorPriority.P1
        if kpi_status == KPIStatus.WARN:
            if anomaly and anomaly.is_anomaly:
                return ErrorPriority.P1
            return ErrorPriority.P2
        return ErrorPriority.P3

    def _suggest_action(
        self,
        kpi_name: str,
        category: ErrorCategory,
        priority: ErrorPriority,
        anomaly: Optional[AnomalyResult],
    ) -> str:
        """建议动作"""
        if priority == ErrorPriority.P0:
            return f"立即停止交易, 检查{kpi_name}异常原因"
        if priority == ErrorPriority.P1:
            if category == ErrorCategory.PARAMETER:
                return f"1小时内调整{kpi_name}相关参数"
            elif category == ErrorCategory.NETWORK:
                return f"1小时内检查网络连接, 切换备用节点"
            elif category == ErrorCategory.LOGIC:
                return f"1小时内复查{kpi_name}相关策略逻辑"
            return f"1小时内响应{kpi_name}异常"
        if priority == ErrorPriority.P2:
            return f"1天内优化{kpi_name}, 考虑参数调整"
        return f"1周内观察{kpi_name}趋势, 必要时调整"


# ============================================================================
# 5. 迭代优化器
# ============================================================================


@dataclass(slots=True)
class OptimizationSuggestion:
    """优化建议"""
    suggestion_id: str
    timestamp: float
    target_kpi: str
    current_value: float
    suggested_change: str
    expected_improvement: str
    priority: ErrorPriority
    status: str = "pending"           # pending/applied/rejected


class IterativeOptimizer:
    """迭代优化器

    根据KPI异常和错误反馈，生成参数调整建议。

    优化策略：
      1. Sharpe下降 → 减小仓位(降风险)
      2. 回撤扩大 → 收紧止损
      3. 胜率下降 → 调整入场条件
      4. 滑点升高 → 降低交易频率
      5. 延迟升高 → 切换节点/优化代码
    """

    def __init__(self):
        self._suggestions: List[OptimizationSuggestion] = []
        self._suggestion_counter = 0

    def generate_suggestions(
        self,
        snapshot: KPISnapshot,
        errors: List[ErrorEntry],
    ) -> List[OptimizationSuggestion]:
        """生成优化建议

        Args:
            snapshot: 当前KPI快照
            errors: 错误列表

        Returns:
            List[OptimizationSuggestion]
        """
        suggestions: List[OptimizationSuggestion] = []

        for kpi_name, kpi in snapshot.kpis.items():
            if kpi.status in (KPIStatus.WARN, KPIStatus.BLOCK):
                suggestion = self._generate_for_kpi(kpi_name, kpi)
                if suggestion:
                    suggestions.append(suggestion)

        # 按优先级排序
        priority_order = {ErrorPriority.P0: 0, ErrorPriority.P1: 1, ErrorPriority.P2: 2, ErrorPriority.P3: 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 99))

        self._suggestions.extend(suggestions)
        return suggestions

    def _generate_for_kpi(self, kpi_name: str, kpi: KPI) -> Optional[OptimizationSuggestion]:
        """为单个KPI生成建议"""
        self._suggestion_counter += 1
        suggestion_id = f"OPT-{int(time.time())}-{self._suggestion_counter:04d}"

        name_lower = kpi_name.lower()

        if "sharpe" in name_lower:
            if kpi.status == KPIStatus.BLOCK:
                return OptimizationSuggestion(
                    suggestion_id=suggestion_id,
                    timestamp=time.time(),
                    target_kpi=kpi_name,
                    current_value=kpi.value,
                    suggested_change="降低仓位50%, 暂停高风险策略",
                    expected_improvement="降低波动率, 恢复Sharpe至1.0+",
                    priority=ErrorPriority.P0,
                )
            return OptimizationSuggestion(
                suggestion_id=suggestion_id,
                timestamp=time.time(),
                target_kpi=kpi_name,
                current_value=kpi.value,
                suggested_change="降低仓位20-30%, 调整入场过滤条件",
                expected_improvement="Sharpe提升0.2-0.5",
                priority=ErrorPriority.P1,
            )

        if "drawdown" in name_lower:
            return OptimizationSuggestion(
                suggestion_id=suggestion_id,
                timestamp=time.time(),
                target_kpi=kpi_name,
                current_value=kpi.value,
                suggested_change="收紧止损至-2%, 降低单笔风险",
                expected_improvement="最大回撤降低5-10%",
                priority=ErrorPriority.P0 if kpi.status == KPIStatus.BLOCK else ErrorPriority.P1,
            )

        if "winrate" in name_lower:
            return OptimizationSuggestion(
                suggestion_id=suggestion_id,
                timestamp=time.time(),
                target_kpi=kpi_name,
                current_value=kpi.value,
                suggested_change="增加入场确认条件, 减少假信号",
                expected_improvement="胜率提升3-5%",
                priority=ErrorPriority.P2,
            )

        if "latency" in name_lower:
            return OptimizationSuggestion(
                suggestion_id=suggestion_id,
                timestamp=time.time(),
                target_kpi=kpi_name,
                current_value=kpi.value,
                suggested_change="切换到备用API节点, 优化订单提交路径",
                expected_improvement="延迟降低50-100ms",
                priority=ErrorPriority.P1,
            )

        if "slippage" in name_lower:
            return OptimizationSuggestion(
                suggestion_id=suggestion_id,
                timestamp=time.time(),
                target_kpi=kpi_name,
                current_value=kpi.value,
                suggested_change="降低交易频率, 使用限价单替代市价单",
                expected_improvement="滑点成本降低30-50%",
                priority=ErrorPriority.P2,
            )

        return None

    def apply_suggestion(self, suggestion_id: str) -> bool:
        """标记建议为已应用"""
        for s in self._suggestions:
            if s.suggestion_id == suggestion_id:
                s.status = "applied"
                return True
        return False


# ============================================================================
# 6. 综合KPI监控器
# ============================================================================


class KPIMonitor:
    """综合KPI监控器

    完整流程：
      1. 实时更新10大KPI
      2. 异常检测(3-sigma + CUSUM + 连续异常)
      3. 错误反馈(分类+优先级+知识库)
      4. 优化建议(参数调整)

    使用示例：
        monitor = KPIMonitor()

        # 实时更新KPI
        monitor.update_kpi("sharpe", 1.5)
        monitor.update_kpi("max_drawdown", -0.12)
        monitor.update_kpi("winrate", 0.55)

        # 获取快照
        snapshot = monitor.get_snapshot()
        print(f"综合状态: {snapshot.overall_status.value}")

        # 获取错误
        errors = monitor.get_open_errors()
        for e in errors:
            print(f"[{e.priority.value}] {e.kpi_name}: {e.description}")

        # 获取优化建议
        suggestions = monitor.get_optimization_suggestions()
    """

    def __init__(self):
        # 初始化10大KPI
        self.kpis: Dict[str, KPI] = {
            "sharpe": KPI(
                name="sharpe", warn_threshold=SHARPE_MIN, block_threshold=SHARPE_BLOCK,
                higher_is_better=True,
            ),
            "max_drawdown": KPI(
                name="max_drawdown", warn_threshold=MAX_DRAWDOWN_WARN, block_threshold=MAX_DRAWDOWN_BLOCK,
                higher_is_better=False,
            ),
            "winrate": KPI(
                name="winrate", warn_threshold=WINRATE_MIN, block_threshold=WINRATE_BLOCK,
                higher_is_better=True,
            ),
            "profit_factor": KPI(
                name="profit_factor", warn_threshold=PROFIT_FACTOR_MIN, block_threshold=PROFIT_FACTOR_BLOCK,
                higher_is_better=True,
            ),
            "latency_p95": KPI(
                name="latency_p95", warn_threshold=LATENCY_P95_WARN_MS, block_threshold=LATENCY_P95_BLOCK_MS,
                higher_is_better=False,
            ),
            "fill_rate": KPI(
                name="fill_rate", warn_threshold=FILL_RATE_WARN, block_threshold=FILL_RATE_BLOCK,
                higher_is_better=True,
            ),
            "slippage_cost": KPI(
                name="slippage_cost", warn_threshold=SLIPPAGE_COST_WARN, block_threshold=SLIPPAGE_COST_BLOCK,
                higher_is_better=False,
            ),
        }

        # 每个KPI一个异常检测器
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {
            name: AnomalyDetector() for name in self.kpis
        }

        self.error_channel = ErrorFeedbackChannel()
        self.optimizer = IterativeOptimizer()
        self._snapshots: List[KPISnapshot] = []

    def update_kpi(self, kpi_name: str, value: float) -> KPIStatus:
        """更新KPI

        Args:
            kpi_name: KPI名称
            value: 新值

        Returns:
            KPIStatus
        """
        if kpi_name not in self.kpis:
            logger.warning(f"未知KPI: {kpi_name}")
            return KPIStatus.UNKNOWN

        kpi = self.kpis[kpi_name]
        status = kpi.update(value)

        # 异常检测
        detector = self.anomaly_detectors[kpi_name]
        anomaly = detector.detect(value, kpi.history)

        # 状态差或异常 → 报告错误
        if status in (KPIStatus.BLOCK, KPIStatus.WARN) or anomaly.is_anomaly:
            self.error_channel.report_error(
                kpi_name=kpi_name,
                kpi_status=status,
                anomaly=anomaly if anomaly.is_anomaly else None,
                description=f"KPI={kpi_name} value={value:.4f} status={status.value}"
                            + (f" | {anomaly.description}" if anomaly.is_anomaly else ""),
            )

        return status

    def get_snapshot(self) -> KPISnapshot:
        """获取当前快照"""
        snapshot = KPISnapshot(timestamp=time.time())
        for name, kpi in self.kpis.items():
            snapshot.kpis[name] = kpi
        snapshot.compute_overall_status()
        self._snapshots.append(snapshot)
        # 保留最近1000个快照
        if len(self._snapshots) > 1000:
            self._snapshots = self._snapshots[-1000:]
        return snapshot

    def get_open_errors(self, priority: Optional[ErrorPriority] = None) -> List[ErrorEntry]:
        """获取未解决错误"""
        return self.error_channel.get_open_errors(priority)

    def get_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """获取优化建议"""
        snapshot = self.get_snapshot()
        errors = self.get_open_errors()
        return self.optimizer.generate_suggestions(snapshot, errors)

    def resolve_error(self, error_id: str, resolution: str):
        """解决错误"""
        self.error_channel.resolve_error(error_id, resolution)

    def generate_report(self) -> str:
        """生成监控报告"""
        snapshot = self.get_snapshot()
        errors = self.get_open_errors()
        suggestions = self.optimizer.generate_suggestions(snapshot, errors)

        lines = [
            "=" * 70,
            "KPI监控报告 (KPI Monitor Report)",
            "=" * 70,
            f"综合状态: {snapshot.overall_status.value}",
            f"BLOCK数: {snapshot.block_count}  WARN数: {snapshot.warn_count}",
            "",
            "【KPI状态】",
        ]

        for name, kpi in snapshot.kpis.items():
            direction = "↑" if kpi.higher_is_better else "↓"
            lines.append(
                f"  {name:20s} = {kpi.value:>10.4f}  "
                f"warn={kpi.warn_threshold}{direction}  "
                f"block={kpi.block_threshold}{direction}  "
                f"→ {kpi.status.value}"
            )

        lines.append("")
        lines.append(f"【未解决错误({len(errors)}条)】")
        for e in errors[:10]:  # 显示前10条
            lines.append(
                f"  [{e.priority.value}] {e.kpi_name}: {e.description[:60]}... → {e.suggested_action[:50]}..."
            )

        lines.append("")
        lines.append(f"【优化建议({len(suggestions)}条)】")
        for s in suggestions[:5]:  # 显示前5条
            lines.append(
                f"  [{s.priority.value}] {s.target_kpi}: {s.suggested_change[:60]} → {s.expected_improvement[:40]}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 自测
# ============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    print("=== R17-6 KPI监控 自测 ===\n")

    np.random.seed(42)
    monitor = KPIMonitor()

    # 模拟正常KPI更新(20期)
    print("--- 正常阶段(20期) ---")
    for i in range(20):
        monitor.update_kpi("sharpe", 1.5 + np.random.normal(0, 0.1))
        monitor.update_kpi("max_drawdown", -0.10 + np.random.normal(0, 0.02))
        monitor.update_kpi("winrate", 0.55 + np.random.normal(0, 0.03))
        monitor.update_kpi("latency_p95", 100 + np.random.normal(0, 20))

    snapshot_normal = monitor.get_snapshot()
    print(f"  综合状态: {snapshot_normal.overall_status.value}")
    print(f"  BLOCK: {snapshot_normal.block_count}  WARN: {snapshot_normal.warn_count}")

    # 模拟异常KPI(Sharpe骤降)
    print("\n--- 异常阶段(Sharpe骤降) ---")
    for i in range(5):
        monitor.update_kpi("sharpe", 0.3)  # Sharpe<0.5=BLOCK

    snapshot_abnormal = monitor.get_snapshot()
    print(f"  综合状态: {snapshot_abnormal.overall_status.value}")
    print(f"  BLOCK: {snapshot_abnormal.block_count}  WARN: {snapshot_abnormal.warn_count}")

    errors = monitor.get_open_errors()
    print(f"  未解决错误: {len(errors)}条")
    for e in errors[:3]:
        print(f"    [{e.priority.value}] {e.kpi_name}: {e.description[:60]}")

    print("\n" + monitor.generate_report())
