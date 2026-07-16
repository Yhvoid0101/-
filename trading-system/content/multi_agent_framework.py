#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体协作框架 (Multi-Agent Collaboration Framework)

核心功能：
  1. 8个智能体角色定义与责任矩阵
  2. 标准化任务信息共享与进度同步
  3. 跨智能体接口规范与数据交换协议
  4. 任务边界划分与冲突检测
  5. 统一进度报告模板与问题跟踪

设计原则（用户铁律）：
  - "八个协同工作智能体" — 每个Agent必须有明确职责和无重叠边界
  - "任务重叠率控制在5%以内" — 硬约束
  - "标准化信息共享机制" — 统一数据格式、传输频率、校验机制

Agent角色定义：
  Agent-1: 核心协调者 (Coordinator) — 任务分配、进度汇总、冲突仲裁
  Agent-2: 策略研究员 (Strategy Researcher) — 策略发现、回测验证、参数优化
  Agent-3: 风险管理者 (Risk Manager) — 风控参数、回撤监控、流动性评估
  Agent-4: 数据工程师 (Data Engineer) — 数据采集、清洗、存储、实时推送
  Agent-5: 执行优化者 (Execution Optimizer) — 订单执行、滑点控制、延迟优化
  Agent-6: 代码审查者 (Code Reviewer) — 代码质量、安全扫描、性能优化
  Agent-7: 测试验证者 (Test Validator) — 回测验证、模拟测试、差异分析
  Agent-8: 知识管理者 (Knowledge Manager) — 经验沉淀、错误库维护、文档更新

使用方式：
  from multi_agent_framework import AgentCoordinator
  coordinator = AgentCoordinator()
  task = coordinator.create_task("回测验证", {"symbols": ["BTC-USDT"], "period": "6M"})
  coordinator.assign_task(task.id, "Agent-2")
  report = coordinator.get_progress_report()
"""

from __future__ import annotations

import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("hermes.multi_agent")

# ============================================================================
# 枚举定义
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class AgentRole(Enum):
    COORDINATOR = "Agent-1"
    STRATEGY_RESEARCHER = "Agent-2"
    RISK_MANAGER = "Agent-3"
    DATA_ENGINEER = "Agent-4"
    EXECUTION_OPTIMIZER = "Agent-5"
    CODE_REVIEWER = "Agent-6"
    TEST_VALIDATOR = "Agent-7"
    KNOWLEDGE_MANAGER = "Agent-8"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class AgentDefinition:
    """Agent定义"""
    role: AgentRole
    name: str
    description: str
    responsibilities: List[str]
    required_inputs: List[str]      # 需要的输入数据格式
    produces_outputs: List[str]     # 产生的输出数据格式
    tools: List[str]                # 可用工具
    skills: List[str]               # 可用技能
    dependencies: List[str]         # 依赖的其他Agent
    max_concurrent_tasks: int = 3


@dataclass
class Task:
    """任务定义"""
    task_id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    progress_pct: float = 0.0
    blockers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ProgressReport:
    """进度报告"""
    report_id: str
    agent_role: AgentRole
    timestamp: str
    tasks_completed: int
    tasks_in_progress: int
    tasks_blocked: int
    key_metrics: Dict[str, Any]
    issues: List[str]
    next_steps: List[str]
    resource_usage: Dict[str, float]  # 资源使用率


@dataclass
class DataExchangeProtocol:
    """数据交换协议 (v598 Phase R4.1: 真实化, 不再是空壳)

    之前问题: 仅 7 个字段无方法, evolution_loop.py 运行时 0 次使用 → 空壳集成
    R4.1 修复: 新增 4 个核心方法 serialize/deserialize/validate/route, 让协议真正生效

    用户铁律: "标准化信息共享机制" — 统一数据格式、传输频率、校验机制
    """
    format: str = "json"          # 数据格式
    encoding: str = "utf-8"       # 编码
    checksum: str = "sha256"      # 校验机制
    frequency_ms: int = 1000      # 传输频率 (ms)
    compression: str = "none"     # 压缩方式 (none/gzip/zlib)
    max_size_bytes: int = 10_485_760  # 最大数据大小 (10MB)
    version: str = "1.0.0"

    def serialize(self, data: Dict[str, Any]) -> bytes:
        """序列化 dict → bytes (含编码、压缩、格式校验)

        v598 Phase R4.1: 真实实现, 不再是空壳

        Args:
            data: 待序列化的 dict

        Returns:
            bytes: 序列化后的字节流 (含格式头: version|format|encoding|compression)

        Raises:
            ValueError: 数据格式不支持 / 大小超限 / JSON 序列化失败
        """
        if not isinstance(data, dict):
            raise ValueError(f"serialize 期望 dict, 实际 {type(data).__name__}")

        # 格式校验
        if self.format != "json":
            raise ValueError(f"不支持的格式: {self.format} (当前仅支持 json)")

        # JSON 序列化
        try:
            json_bytes = json.dumps(data, ensure_ascii=False,
                                    default=str).encode(self.encoding)
        except (UnicodeEncodeError, TypeError, ValueError) as e:
            raise ValueError(f"JSON 序列化失败: {e}")

        # 大小检查 (压缩前)
        if len(json_bytes) > self.max_size_bytes:
            raise ValueError(
                f"数据大小 {len(json_bytes)} 超过上限 {self.max_size_bytes}"
            )

        # 压缩 (当前仅支持 none, 预留 zlib/gzip 接口)
        if self.compression == "none":
            payload = json_bytes
        elif self.compression in ("zlib", "gzip"):
            import zlib
            payload = zlib.compress(json_bytes)
        else:
            raise ValueError(f"不支持的压缩方式: {self.compression}")

        # 构造格式头: version|format|encoding|compression|checksum_hex
        checksum_hex = self._compute_checksum(payload)
        header = (
            f"{self.version}|{self.format}|{self.encoding}|"
            f"{self.compression}|{checksum_hex}\n"
        ).encode("ascii")

        return header + payload

    def deserialize(self, payload: bytes) -> Dict[str, Any]:
        """反序列化 bytes → dict (含格式头解析、校验、解压)

        v598 Phase R4.1: 真实实现

        Args:
            payload: serialize() 产生的字节流 (含格式头)

        Returns:
            dict: 反序列化后的字典

        Raises:
            ValueError: 格式头损坏 / checksum 不匹配 / 大小超限 / JSON 解析失败
        """
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError(f"deserialize 期望 bytes, 实际 {type(payload).__name__}")

        # 解析格式头 (第一行, 以 \n 分隔)
        try:
            header_end = payload.index(b"\n")
            header_bytes = payload[:header_end]
            payload_bytes = payload[header_end + 1:]
        except ValueError:
            raise ValueError("格式头损坏: 未找到换行分隔符")

        try:
            header_str = header_bytes.decode("ascii")
            parts = header_str.split("|")
            if len(parts) != 5:
                raise ValueError(f"格式头字段数错误: 期望 5, 实际 {len(parts)}")
            version, fmt, encoding, compression, expected_checksum = parts
        except UnicodeDecodeError as e:
            raise ValueError(f"格式头解码失败: {e}")

        # 版本兼容性检查
        if version != self.version:
            logger.warning(
                "协议版本不匹配: 本地 %s, 远端 %s (尝试兼容解析)",
                self.version, version,
            )

        # 大小检查 (解压前)
        if len(payload_bytes) > self.max_size_bytes:
            raise ValueError(
                f"数据大小 {len(payload_bytes)} 超过上限 {self.max_size_bytes}"
            )

        # 解压
        if compression == "none":
            json_bytes = payload_bytes
        elif compression in ("zlib", "gzip"):
            import zlib
            try:
                json_bytes = zlib.decompress(payload_bytes)
            except zlib.error as e:
                raise ValueError(f"解压失败: {e}")
        else:
            raise ValueError(f"不支持的压缩方式: {compression}")

        # checksum 校验 (防篡改/防传输错误)
        actual_checksum = self._compute_checksum(payload_bytes)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"checksum 校验失败: 期望 {expected_checksum}, 实际 {actual_checksum}"
            )

        # JSON 反序列化
        try:
            data = json.loads(json_bytes.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"JSON 反序列化失败: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"反序列化结果应为 dict, 实际 {type(data).__name__}")

        return data

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """验证数据是否符合协议规范 (大小、结构、必需字段)

        v598 Phase R4.1: 真实实现

        Args:
            data: 待验证的 dict

        Returns:
            (passed, message): 通过标志 + 详情
        """
        if not isinstance(data, dict):
            return False, f"数据类型应为 dict, 实际 {type(data).__name__}"

        # 大小检查 (序列化后)
        try:
            json_size = len(json.dumps(data, ensure_ascii=False, default=str))
        except (TypeError, ValueError) as e:
            return False, f"JSON 序列化失败: {e}"

        if json_size > self.max_size_bytes:
            return False, (
                f"数据大小 {json_size} 超过上限 {self.max_size_bytes}"
            )

        # 必需字段检查 (按协议版本要求)
        # v1.0.0 协议: 必须含 source / target / payload 三字段
        required_fields = ["source", "target", "payload"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return False, f"缺少必需字段: {missing}"

        # source/target 必须是字符串 (Agent 角色 ID)
        if not isinstance(data["source"], str) or not data["source"]:
            return False, "source 必须是非空字符串 (Agent 角色 ID)"
        if not isinstance(data["target"], str) or not data["target"]:
            return False, "target 必须是非空字符串 (Agent 角色 ID)"

        # payload 必须是 dict
        if not isinstance(data["payload"], dict):
            return False, f"payload 必须是 dict, 实际 {type(data['payload']).__name__}"

        # 可选: timestamp 字段 (建议存在, 用于追踪)
        if "timestamp" in data:
            if not isinstance(data["timestamp"], (str, int, float)):
                return False, "timestamp 必须是 str/int/float"

        return True, f"验证通过 (size={json_size}B, fields={list(data.keys())})"

    def route(self, target_role: AgentRole, payload: Dict[str, Any],
              source_role: Optional[AgentRole] = None) -> bytes:
        """路由消息到指定 Agent 角色 (封装 source/target + 序列化)

        v598 Phase R4.1: 真实实现 — 让 evolution_loop.py 调度块可调用

        Args:
            target_role: 目标 Agent 角色
            payload: 实际数据负载
            source_role: 源 Agent 角色 (None=COORDINATOR)

        Returns:
            bytes: 序列化后的消息 (含协议头, 可由 deserialize() 还原)

        Raises:
            ValueError: validate 失败 / serialize 失败
        """
        if not isinstance(target_role, AgentRole):
            raise ValueError(
                f"target_role 必须是 AgentRole, 实际 {type(target_role).__name__}"
            )

        if source_role is None:
            source_role = AgentRole.COORDINATOR
        elif not isinstance(source_role, AgentRole):
            raise ValueError(
                f"source_role 必须是 AgentRole, 实际 {type(source_role).__name__}"
            )

        # 构造符合协议的消息 dict
        message = {
            "source": source_role.value,    # e.g. "Agent-1"
            "target": target_role.value,    # e.g. "Agent-3"
            "payload": payload,             # 实际数据负载
            "timestamp": datetime.now().isoformat(),
            "protocol_version": self.version,
        }

        # 验证
        passed, msg = self.validate(message)
        if not passed:
            raise ValueError(f"消息验证失败: {msg}")

        # 序列化
        return self.serialize(message)

    def _compute_checksum(self, data: bytes) -> str:
        """计算 checksum (内部辅助方法)

        Args:
            data: 字节流

        Returns:
            checksum 十六进制字符串
        """
        if self.checksum == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.checksum == "md5":
            return hashlib.md5(data).hexdigest()
        elif self.checksum == "none":
            return "0" * 64  # 占位符
        else:
            raise ValueError(f"不支持的 checksum: {self.checksum}")


# ============================================================================
# 责任矩阵
# ============================================================================

RESPONSIBILITY_MATRIX = {
    AgentRole.COORDINATOR: AgentDefinition(
        role=AgentRole.COORDINATOR,
        name="核心协调者",
        description="任务分配、进度汇总、冲突仲裁、资源调度",
        responsibilities=[
            "协调8个Agent的任务分配和优先级排序",
            "汇总各Agent进度报告，生成统一状态视图",
            "检测任务冲突和资源竞争，进行仲裁",
            "确保任务重叠率 ≤ 5%",
            "维护跨Agent通信信道",
        ],
        required_inputs=["task_requests", "progress_reports", "resource_status"],
        produces_outputs=["task_assignments", "status_dashboard", "conflict_resolutions"],
        tools=["TaskCreate", "TaskUpdate", "TaskList", "AskUserQuestion"],
        skills=["plan-execute", "verification-framework"],
        dependencies=[],
    ),
    AgentRole.STRATEGY_RESEARCHER: AgentDefinition(
        role=AgentRole.STRATEGY_RESEARCHER,
        name="策略研究员",
        description="策略发现、回测验证、参数优化、基因进化",
        responsibilities=[
            "发现和评估新交易策略",
            "运行回测验证策略有效性",
            "优化策略参数（GA/贝叶斯优化）",
            "分析策略在不同市场状态下的表现",
            "维护策略库和性能基线",
        ],
        required_inputs=["market_data", "backtest_config", "optimization_params"],
        produces_outputs=["strategy_report", "backtest_results", "optimized_params", "performance_metrics"],
        tools=["Shell", "Grep", "Read"],
        skills=["hermes-sandbox-trading", "hermes-benchmark"],
        dependencies=["Agent-4", "Agent-7"],
    ),
    AgentRole.RISK_MANAGER: AgentDefinition(
        role=AgentRole.RISK_MANAGER,
        name="风险管理者",
        description="风控参数设定、回撤监控、流动性评估、压力测试",
        responsibilities=[
            "设定和维护风控参数（最大回撤、单笔亏损、连续亏损）",
            "实时监控回撤和风险敞口",
            "评估市场流动性风险",
            "执行压力测试和极端场景分析",
            "生成风险报告和预警",
        ],
        required_inputs=["position_data", "market_data", "risk_params"],
        produces_outputs=["risk_report", "risk_alerts", "adjusted_risk_params"],
        tools=["Shell", "Read"],
        skills=["TRAE-security-review"],
        dependencies=["Agent-4", "Agent-7"],
    ),
    AgentRole.DATA_ENGINEER: AgentDefinition(
        role=AgentRole.DATA_ENGINEER,
        name="数据工程师",
        description="数据采集、清洗、存储、实时推送、质量监控",
        responsibilities=[
            "从交易所API采集实时行情数据",
            "数据清洗和异常检测",
            "维护数据缓存和存储系统",
            "确保数据质量（完整性≥99.99%、准确性≤0.01%误差）",
            "提供实时数据推送服务",
        ],
        required_inputs=["exchange_config", "data_schema"],
        produces_outputs=["market_data_stream", "data_quality_report", "kline_cache"],
        tools=["Shell", "WebFetch"],
        skills=["hermes-deploy"],
        dependencies=[],
    ),
    AgentRole.EXECUTION_OPTIMIZER: AgentDefinition(
        role=AgentRole.EXECUTION_OPTIMIZER,
        name="执行优化者",
        description="订单执行、滑点控制、延迟优化、成交质量分析",
        responsibilities=[
            "优化订单执行算法（TWAP/VWAP/冰山）",
            "控制滑点和市场冲击成本",
            "监控订单执行延迟（目标 ≤ 500ms）",
            "分析成交质量（成交率、成交价格偏离）",
            "维护交易所连接池",
        ],
        required_inputs=["order_requests", "market_depth", "latency_stats"],
        produces_outputs=["execution_report", "fill_quality_metrics", "optimized_routing"],
        tools=["Shell"],
        skills=["hermes-deploy"],
        dependencies=["Agent-4"],
    ),
    AgentRole.CODE_REVIEWER: AgentDefinition(
        role=AgentRole.CODE_REVIEWER,
        name="代码审查者",
        description="代码质量审查、安全扫描、性能优化、架构合规检查",
        responsibilities=[
            "审查代码质量和最佳实践遵循度",
            "执行安全漏洞扫描（硬编码密钥、注入风险）",
            "识别性能瓶颈和冗余模块",
            "检查架构合规性（模块间接口、依赖注入）",
            "维护代码质量基线和改进建议",
        ],
        required_inputs=["code_diff", "module_list", "dependency_graph"],
        produces_outputs=["code_review_report", "security_scan_result", "performance_analysis"],
        tools=["Grep", "Glob", "Read"],
        skills=["TRAE-code-review", "TRAE-security-review"],
        dependencies=[],
    ),
    AgentRole.TEST_VALIDATOR: AgentDefinition(
        role=AgentRole.TEST_VALIDATOR,
        name="测试验证者",
        description="回测验证、模拟测试、差异分析、性能基准测试",
        responsibilities=[
            "运行回测验证策略表现",
            "执行模拟-实盘差异分析",
            "维护性能基准和回归检测",
            "验证修复是否生效（独立验证原则）",
            "生成测试报告和验证证据",
        ],
        required_inputs=["strategy_code", "backtest_data", "performance_baseline"],
        produces_outputs=["test_report", "verification_evidence", "regression_alerts"],
        tools=["Shell", "Read"],
        skills=["verification-framework", "hermes-benchmark", "auto-review-system"],
        dependencies=["Agent-2", "Agent-4"],
    ),
    AgentRole.KNOWLEDGE_MANAGER: AgentDefinition(
        role=AgentRole.KNOWLEDGE_MANAGER,
        name="知识管理者",
        description="经验沉淀、错误库维护、文档更新、记忆管理",
        responsibilities=[
            "记录每次会话的关键决策和踩坑",
            "维护错误知识库（1500-error-knowledge-base）",
            "更新项目文档和AGENTS.md",
            "管理Obsidian笔记和索引",
            "确保永不二过原则（错误模式匹配）",
        ],
        required_inputs=["session_logs", "error_reports", "decision_records"],
        produces_outputs=["knowledge_entries", "error_library_updates", "documentation"],
        tools=["Write", "Edit", "Read"],
        skills=["obsidian-log"],
        dependencies=[],
    ),
}


# ============================================================================
# 核心协调器
# ============================================================================

class AgentCoordinator:
    """多智能体协调器 — 核心协调职责"""

    def __init__(self):
        self.agents: Dict[AgentRole, AgentDefinition] = RESPONSIBILITY_MATRIX
        self.tasks: Dict[str, Task] = {}
        self.reports: Dict[str, List[ProgressReport]] = defaultdict(list)
        self._overlap_cache: Dict[str, float] = {}
        self._protocol = DataExchangeProtocol()

    def create_task(self, title: str, description: str,
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    dependencies: List[str] = None,
                    inputs: Dict[str, Any] = None) -> Task:
        """创建新任务"""
        task_id = self._generate_task_id(title)
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            inputs=inputs or {},
        )
        self.tasks[task_id] = task
        logger.info("创建任务 %s: %s (优先级: %s)", task_id, title, priority.name)
        return task

    def assign_task(self, task_id: str, agent_role: str) -> bool:
        """分配任务给Agent"""
        if task_id not in self.tasks:
            logger.error("任务 %s 不存在", task_id)
            return False

        task = self.tasks[task_id]
        role = AgentRole(agent_role)

        # 检查任务重叠
        overlap = self._check_task_overlap(task, role)
        if overlap > 0.05:  # 超过5%重叠
            # v598 Phase B 修复: logger 格式字符串中 % 必须转义为 %%
            logger.warning(
                "任务 %s 与 Agent %s 现有任务重叠 %.1f%% > 5%%, 需要重新分配",
                task_id, role.value, overlap * 100
            )
            return False

        task.assigned_to = role.value
        task.status = TaskStatus.ASSIGNED
        task.updated_at = datetime.now().isoformat()
        logger.info("任务 %s 分配给 %s", task_id, role.value)
        return True

    def update_task_status(self, task_id: str, status: TaskStatus,
                           progress_pct: float = None, notes: str = None) -> bool:
        """更新任务状态"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = status
        task.updated_at = datetime.now().isoformat()
        if progress_pct is not None:
            task.progress_pct = max(0.0, min(100.0, progress_pct))
        if notes:
            task.notes.append(f"[{datetime.now().strftime('%H:%M:%S')}] {notes}")

        logger.info("任务 %s 状态: %s (进度: %.1f%%)", task_id, status.value, task.progress_pct)
        return True

    def add_blocker(self, task_id: str, blocker: str) -> bool:
        """添加阻塞原因"""
        if task_id not in self.tasks:
            return False
        self.tasks[task_id].blockers.append(blocker)
        self.tasks[task_id].status = TaskStatus.BLOCKED
        self.tasks[task_id].updated_at = datetime.now().isoformat()
        return True

    def submit_report(self, report: ProgressReport) -> None:
        """提交进度报告"""
        self.reports[report.agent_role.value].append(report)
        logger.info(
            "收到 %s 进度报告: 完成=%d, 进行中=%d, 阻塞=%d",
            report.agent_role.value,
            report.tasks_completed, report.tasks_in_progress, report.tasks_blocked,
        )

    def get_progress_report(self) -> Dict[str, Any]:
        """获取全局进度报告"""
        total_tasks = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)
        blocked = sum(1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED)
        pending = sum(1 for t in self.tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED))

        agent_load = {}
        for role in AgentRole:
            agent_tasks = [t for t in self.tasks.values() if t.assigned_to == role.value]
            agent_load[role.value] = {
                "total": len(agent_tasks),
                "completed": sum(1 for t in agent_tasks if t.status == TaskStatus.COMPLETED),
                "in_progress": sum(1 for t in agent_tasks if t.status == TaskStatus.IN_PROGRESS),
                "blocked": sum(1 for t in agent_tasks if t.status == TaskStatus.BLOCKED),
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": total_tasks,
                "completed": completed,
                "in_progress": in_progress,
                "blocked": blocked,
                "pending": pending,
                "completion_rate": f"{completed / max(total_tasks, 1) * 100:.1f}%",
            },
            "agent_load": agent_load,
            "overlap_check": self._check_global_overlap(),
            "blocked_tasks": [
                {"task_id": t.task_id, "title": t.title, "blockers": t.blockers}
                for t in self.tasks.values() if t.status == TaskStatus.BLOCKED
            ],
        }

    def get_data_exchange_spec(self) -> Dict[str, Any]:
        """获取数据交换协议规范"""
        return {
            "protocol_version": self._protocol.version,
            "format": self._protocol.format,
            "encoding": self._protocol.encoding,
            "checksum_algorithm": self._protocol.checksum,
            "transmission_frequency_ms": self._protocol.frequency_ms,
            "compression": self._protocol.compression,
            "max_message_size_bytes": self._protocol.max_size_bytes,
            "message_schema": {
                "task": {
                    "required": ["task_id", "title", "status", "assigned_to", "created_at"],
                    "optional": ["description", "deadline", "dependencies", "progress_pct"],
                },
                "report": {
                    "required": ["report_id", "agent_role", "timestamp", "tasks_completed"],
                    "optional": ["key_metrics", "issues", "next_steps"],
                },
            },
            "checksum_example": self._compute_checksum({"test": "data"}),
        }

    def _generate_task_id(self, title: str) -> str:
        """生成唯一任务ID

        v598 Phase B 修复: 加入随机后缀避免同秒同 title 生成相同 task_id
        之前 bug: 同秒创建相同 title 的两个任务会生成相同 task_id,
                 导致 _check_task_overlap 中 `t.task_id != task.task_id`
                 永远为 False → overlap 检测完全失效 (违反铁律永不二过)
        """
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")  # 含微秒
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        # 加入 4 字节随机后缀, 确保同秒同 title 也有不同 task_id
        _rand_suffix = hashlib.md5(
            f"{ts}-{title}-{time.time_ns()}".encode()
        ).hexdigest()[:4]
        return f"TASK-{ts}-{title_hash}-{_rand_suffix}"

    def _check_task_overlap(self, task: Task, role: AgentRole) -> float:
        """检查任务重叠率"""
        agent_tasks = [t for t in self.tasks.values()
                       if t.assigned_to == role.value and t.task_id != task.task_id]
        if not agent_tasks:
            return 0.0

        # 基于任务描述的简单重叠检测
        task_words = set(task.title.lower().split() + task.description.lower().split())
        max_overlap = 0.0
        for existing in agent_tasks:
            existing_words = set(existing.title.lower().split() + existing.description.lower().split())
            if task_words and existing_words:
                overlap = len(task_words & existing_words) / len(task_words | existing_words)
                max_overlap = max(max_overlap, overlap)

        return max_overlap

    def _check_global_overlap(self) -> Dict[str, Any]:
        """检查全局任务重叠"""
        agent_task_words = {}
        for role in AgentRole:
            agent_tasks = [t for t in self.tasks.values() if t.assigned_to == role.value]
            words = set()
            for t in agent_tasks:
                words.update(t.title.lower().split())
                words.update(t.description.lower().split())
            agent_task_words[role.value] = words

        overlaps = {}
        roles = list(AgentRole)
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                w1 = agent_task_words[roles[i].value]
                w2 = agent_task_words[roles[j].value]
                if w1 and w2:
                    overlap = len(w1 & w2) / len(w1 | w2)
                    if overlap > 0:
                        overlaps[f"{roles[i].value}-{roles[j].value}"] = round(overlap * 100, 1)

        return {
            "overlaps": overlaps,
            "max_overlap_pct": max(overlaps.values()) if overlaps else 0,
            "within_threshold": all(v <= 5.0 for v in overlaps.values()) if overlaps else True,
        }

    @staticmethod
    def _compute_checksum(data: Dict[str, Any]) -> str:
        """计算数据校验和"""
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    coordinator = AgentCoordinator()

    # 创建示例任务
    tasks = [
        ("回测验证 BTC/ETH 策略", "对 BTC-USDT 和 ETH-USDT 运行 6 个月回测", TaskPriority.HIGH),
        ("代码审查 agent_decision_engine.py", "审查 Fix 33 修改的代码质量和安全性", TaskPriority.HIGH),
        ("数据管道延迟优化", "将国内行情数据延迟从 1056ms 降至目标 ≤ 500ms", TaskPriority.CRITICAL),
        ("更新错误知识库", "记录 ERR-047 和 ERR-045 教训到 1500-error-knowledge-base", TaskPriority.MEDIUM),
    ]

    for title, desc, priority in tasks:
        task = coordinator.create_task(title, desc, priority)
        print(f"创建任务: {task.task_id} — {task.title}")

    # 分配任务
    assignments = [
        (list(coordinator.tasks.keys())[0], "Agent-2"),
        (list(coordinator.tasks.keys())[1], "Agent-6"),
        (list(coordinator.tasks.keys())[2], "Agent-5"),
        (list(coordinator.tasks.keys())[3], "Agent-8"),
    ]

    for task_id, agent in assignments:
        coordinator.assign_task(task_id, agent)
        coordinator.update_task_status(task_id, TaskStatus.IN_PROGRESS, progress_pct=25.0)

    # 提交进度报告
    for role in [AgentRole.STRATEGY_RESEARCHER, AgentRole.CODE_REVIEWER]:
        report = ProgressReport(
            report_id=f"RPT-{datetime.now().strftime('%H%M%S')}-{role.value}",
            agent_role=role,
            timestamp=datetime.now().isoformat(),
            tasks_completed=0,
            tasks_in_progress=1,
            tasks_blocked=0,
            key_metrics={"progress_pct": 25.0},
            issues=[],
            next_steps=["继续执行当前任务"],
            resource_usage={"cpu": 0.3, "memory": 0.4},
        )
        coordinator.submit_report(report)

    # 打印全局进度
    progress = coordinator.get_progress_report()
    print("\n" + "=" * 60)
    print("全局进度报告")
    print("=" * 60)
    print(json.dumps(progress, ensure_ascii=False, indent=2))

    # 打印数据交换协议
    print("\n" + "=" * 60)
    print("数据交换协议规范")
    print("=" * 60)
    print(json.dumps(coordinator.get_data_exchange_spec(), ensure_ascii=False, indent=2))