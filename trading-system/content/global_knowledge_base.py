# -*- coding: utf-8 -*-
"""全局经验知识库 (Global Knowledge Base) — 6大核心短板 #5 补齐
================================================================================
来源: _v595_global_knowledge_base.py 重构, 提取纯资产为可导入类
用户核心诉求: "八智能体统一全局经验知识库、任务优先级调度、跨模块联动回滚机制"

设计原则:
  - 统一存储: 所有agent共享同一个知识库JSON文件, 避免信息孤岛
  - 任务优先级: 8个agent的任务统一调度, 高优先级任务先执行
  - 跨模块回滚: 故障传播检测, 一个模块故障触发相关模块回滚
  - 标准化接口: 统一的数据交换格式和进度报告模板
  - 可追溯: 每个决策、每次任务、每条教训都有timestamp+agent+version
  - 零副作用: 无顶层print、无sys.stdout.reconfigure、无顶层文件读写

八智能体定义:
  Agent 1: Coordinator (核心协调) — 任务分配、进度同步、冲突仲裁
  Agent 2: DataPipeline (数据管道) — 行情数据接入、清洗、存储
  Agent 3: StrategyEngine (策略引擎) — 策略开发、回测、优化
  Agent 4: RiskManager (风控管理) — 风险评估、仓位控制、止损
  Agent 5: FeatureEngineering (特征工程) — 特征提取、选择、融合
  Agent 6: BacktestValidator (回测验证) — 回测执行、KPI计算、GO/NO-GO
  Agent 7: LiveDeployer (实盘部署) — 实盘执行、监控、回滚
  Agent 8: KnowledgeBase (知识库) — ERR入库、教训提炼、知识共享

ERR知识库关联:
  - ERR-20260701-v93: 双重嵌套bug教训
  - ERR-20260701-v94: HTF+币种扩展+禁用NEAR三重突破
  - ERR-099: v555配对交易实现35倍改善
  - ERR-110: 0/10特征有预测力
  - ERR-109: 策略本质是均值回归非趋势跟踪
================================================================================
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.global_knowledge_base")


# ============================================================================
# 工具函数
# ============================================================================

def _now_iso() -> str:
    """ISO格式时间戳 (避免utcnow deprecation)"""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# 常量定义 (从 _v595_global_knowledge_base.py 提取)
# ============================================================================

# 八智能体定义 (用户: "作为八个协同工作智能体之一")
EIGHT_AGENTS: Dict[str, Dict[str, Any]] = {
    "Coordinator": {
        "id": 1,
        "role": "核心协调",
        "capabilities": ["任务分配", "进度同步", "冲突仲裁", "资源调度"],
        "priority_level": 1,  # 最高优先级
        "current_load": 0,
        "status": "active",
    },
    "DataPipeline": {
        "id": 2,
        "role": "数据管道",
        "capabilities": ["行情数据接入", "数据清洗", "数据存储", "延迟监控"],
        "priority_level": 2,
        "current_load": 0,
        "status": "active",
    },
    "StrategyEngine": {
        "id": 3,
        "role": "策略引擎",
        "capabilities": ["策略开发", "回测执行", "参数优化", "A/B测试"],
        "priority_level": 2,
        "current_load": 0,
        "status": "active",
    },
    "RiskManager": {
        "id": 4,
        "role": "风控管理",
        "capabilities": ["风险评估", "仓位控制", "止损机制", "压力测试"],
        "priority_level": 1,  # 风控高优先级
        "current_load": 0,
        "status": "active",
    },
    "FeatureEngineering": {
        "id": 5,
        "role": "特征工程",
        "capabilities": ["特征提取", "特征选择", "特征融合", "预测力测试"],
        "priority_level": 3,
        "current_load": 0,
        "status": "active",
    },
    "BacktestValidator": {
        "id": 6,
        "role": "回测验证",
        "capabilities": ["回测执行", "KPI计算", "GO/NO-GO判定", "差异率分析"],
        "priority_level": 2,
        "current_load": 0,
        "status": "active",
    },
    "LiveDeployer": {
        "id": 7,
        "role": "实盘部署",
        "capabilities": ["实盘执行", "实时监控", "自动回滚", "分阶准入"],
        "priority_level": 1,  # 实盘高优先级
        "current_load": 0,
        "status": "active",
    },
    "KnowledgeBase": {
        "id": 8,
        "role": "知识库",
        "capabilities": ["ERR入库", "教训提炼", "知识共享", "回滚触发"],
        "priority_level": 2,
        "current_load": 0,
        "status": "active",
    },
}


# 故障传播图: 模块A故障 → 影响模块B/C
FAULT_PROPAGATION_GRAPH: Dict[str, List[str]] = {
    "DataPipeline.fault": ["StrategyEngine.stop", "BacktestValidator.block", "LiveDeployer.pause"],
    "StrategyEngine.fault": ["BacktestValidator.block", "LiveDeployer.rollback"],
    "RiskManager.alert": ["LiveDeployer.pause", "Coordinator.notify"],
    "FeatureEngineering.fault": ["StrategyEngine.degrade", "BacktestValidator.revalidate"],
    "BacktestValidator.NO_GO": ["LiveDeployer.block", "StrategyEngine.optimize"],
    "LiveDeployer.loss_exceed": ["RiskManager.assess", "Coordinator.rollback_decision"],
    "KnowledgeBase.err_match": ["Coordinator.notify", "StrategyEngine.apply_fix"],
    "LiveDeployer.deployment_ban_active": ["Coordinator.notify", "StrategyEngine.optimize"],
    "BacktestValidator.validation_failed": ["LiveDeployer.block", "Coordinator.notify"],
}


# 回滚chain定义 (基于v594分阶准入规则)
ROLLBACK_CHAIN: Dict[str, Dict[str, Any]] = {
    "Tier3_to_Tier2": {
        "trigger": "实盘KPI退化>20% 或 连续2月亏损",
        "chain": [
            {"step": 1, "agent": "LiveDeployer", "action": "停止标准实盘, 资金缩减到$5k"},
            {"step": 2, "agent": "RiskManager", "action": "重新评估风险敞口"},
            {"step": 3, "agent": "BacktestValidator", "action": "重新跑回测验证策略"},
            {"step": 4, "agent": "KnowledgeBase", "action": "记录退化原因+ERR入库"},
            {"step": 5, "agent": "Coordinator", "action": "通知所有agent回滚到Tier 2状态"},
        ],
    },
    "Tier2_to_Tier1": {
        "trigger": "实盘mlp>5% 或 gap>15% 或 单次亏损>2%",
        "chain": [
            {"step": 1, "agent": "LiveDeployer", "action": "停止迷你实盘, 全部回滚到模拟"},
            {"step": 2, "agent": "RiskManager", "action": "调查单次大亏损根因"},
            {"step": 3, "agent": "BacktestValidator", "action": "重校准成本模型+重跑回测"},
            {"step": 4, "agent": "StrategyEngine", "action": "策略优化解决新发现的问题"},
            {"step": 5, "agent": "KnowledgeBase", "action": "ERR入库+教训提炼+跨agent共享"},
        ],
    },
    "Emergency_Stop": {
        "trigger": "系统性亏损信号 或 单次亏损>2%本金",
        "chain": [
            {"step": 1, "agent": "RiskManager", "action": "立即触发紧急停止"},
            {"step": 2, "agent": "LiveDeployer", "action": "关闭所有持仓, 撤销所有挂单"},
            {"step": 3, "agent": "Coordinator", "action": "召集所有agent进行根因分析"},
            {"step": 4, "agent": "KnowledgeBase", "action": "记录紧急事件+入库CRITICAL ERR"},
            {"step": 5, "agent": "StrategyEngine", "action": "策略全面审查+修复后才能重新准入"},
        ],
    },
}


# 标准进度报告模板
PROGRESS_REPORT_TEMPLATE: Dict[str, Any] = {
    "agent": "",
    "timestamp": "",
    "current_task_id": "",
    "current_task_desc": "",
    "progress_pct": 0,
    "kpi_snapshot": {
        "ann_sim_pct": None,
        "ann_live_pct": None,
        "gap_pct": None,
        "max_dd_pct": None,
        "win_rate_pct": None,
        "sharpe": None,
        "mlp_pct": None,
    },
    "blockers": [],
    "next_steps": [],
    "dependencies_waiting": [],
    "err_ids_referenced": [],
}


# 跨agent数据交换协议
DATA_EXCHANGE_PROTOCOL: Dict[str, Any] = {
    "format": "JSON",
    "encoding": "utf-8",
    "version": "1.0",
    "transmission_frequency": {
        "progress_report": "每任务完成时",
        "kpi_snapshot": "每次回测完成后",
        "err_entry": "错误发生后立即",
        "rollback_trigger": "故障检测后立即",
    },
    "verification_mechanism": {
        "schema_validation": "接收方必须验证JSON schema",
        "timestamp_check": "拒绝过期>1小时的数据",
        "agent_signature": "每条消息必须包含agent字段",
        "err_id_traceability": "所有决策必须可追溯到ERR-ID",
    },
    "interface_specs": {
        "task_assignment": {
            "from": "Coordinator",
            "to": "AnyAgent",
            "fields": ["task_id", "agent", "description", "priority", "dependencies"],
        },
        "kpi_report": {
            "from": "BacktestValidator",
            "to": ["Coordinator", "RiskManager", "LiveDeployer"],
            "fields": ["ann_sim_pct", "ann_live_pct", "gap_pct", "max_dd_pct", "win_rate_pct", "sharpe", "mlp_pct", "verdict"],
        },
        "rollback_alert": {
            "from": "RiskManager",
            "to": ["Coordinator", "LiveDeployer", "KnowledgeBase"],
            "fields": ["rollback_type", "trigger", "affected_modules", "severity"],
        },
        "err_entry": {
            "from": "AnyAgent",
            "to": "KnowledgeBase",
            "fields": ["err_id", "version", "agent", "category", "severity", "description", "root_cause", "fix"],
        },
    },
}


# ============================================================================
# 任务优先级调度器 (从 _v595_global_knowledge_base.py 提取)
# ============================================================================

class TaskPriorityScheduler:
    """八智能体任务优先级调度器

    调度规则:
      1. 优先级数字越小, 优先级越高 (L1 > L2 > L3)
      2. 同优先级按创建时间FIFO
      3. 依赖未完成的任务不调度
      4. agent负载未清零不接新任务
    """

    def __init__(self):
        self.task_queue: List[Dict[str, Any]] = []
        self.task_counter: int = 0
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)

    def submit_task(
        self,
        agent: str,
        task_desc: str,
        priority: int = 3,
        dependencies: Optional[List[str]] = None,
        category: str = "general",
    ) -> str:
        """提交任务到队列

        Args:
            agent: 智能体名称 (必须是EIGHT_AGENTS中定义的)
            task_desc: 任务描述
            priority: 优先级 (1=最高, 2=中, 3=低)
            dependencies: 依赖任务ID列表
            category: 任务类别

        Returns:
            任务ID (如 "TASK-0001")
        """
        self.task_counter += 1
        task_id = f"TASK-{self.task_counter:04d}"
        task: Dict[str, Any] = {
            "task_id": task_id,
            "agent": agent,
            "description": task_desc,
            "priority": priority,
            "status": "pending",
            "dependencies": dependencies or [],
            "category": category,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self.task_queue.append(task)
        for dep in (dependencies or []):
            self.dependency_graph[task_id].append(dep)
        return task_id

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """获取下一个可执行任务 (优先级+依赖检查)

        Returns:
            任务字典 或 None (无可用任务)
        """
        ready_tasks = [
            t for t in self.task_queue
            if t["status"] == "pending"
            and all(self._is_dep_completed(d) for d in t["dependencies"])
        ]
        if not ready_tasks:
            return None
        ready_tasks.sort(key=lambda t: (t["priority"], t["created_at"]))
        return ready_tasks[0]

    def _is_dep_completed(self, task_id: str) -> bool:
        """检查依赖任务是否已完成"""
        for t in self.task_queue:
            if t["task_id"] == task_id:
                return t["status"] == "completed"
        return True  # 依赖不存在视为已完成

    def update_status(self, task_id: str, status: str, summary: str = "") -> bool:
        """更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态 (pending/running/completed/failed)
            summary: 可选摘要

        Returns:
            是否成功更新
        """
        for t in self.task_queue:
            if t["task_id"] == task_id:
                t["status"] = status
                t["updated_at"] = _now_iso()
                if summary:
                    t["summary"] = summary
                return True
        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态汇总"""
        status_count: Dict[str, int] = defaultdict(int)
        for t in self.task_queue:
            status_count[t["status"]] += 1
        return {
            "total": len(self.task_queue),
            "by_status": dict(status_count),
            "by_agent": dict(defaultdict(int, {
                a: sum(1 for t in self.task_queue if t["agent"] == a)
                for a in EIGHT_AGENTS
            })),
        }


# ============================================================================
# GlobalKnowledgeBase 主类
# ============================================================================

class GlobalKnowledgeBase:
    """八智能体统一全局经验知识库

    用户铁律: "八智能体统一全局经验知识库、任务优先级调度、跨模块联动回滚机制"

    功能:
      1. 智能体注册表管理
      2. 任务优先级调度
      3. 故障传播检测
      4. 回滚链执行
      5. ERR条目加载与存储
      6. 标准化报告生成

    零副作用设计:
      - __init__ 不读文件, 不写文件
      - 所有文件操作通过显式方法调用
      - 无顶层 print, 无 sys.stdout.reconfigure
    """

    def __init__(self):
        """初始化知识库 (无文件I/O, 无副作用)"""
        # 智能体注册表 (深拷贝避免修改全局常量)
        import copy
        self.agents: Dict[str, Dict[str, Any]] = copy.deepcopy(EIGHT_AGENTS)

        # 任务调度器
        self.scheduler: TaskPriorityScheduler = TaskPriorityScheduler()

        # 知识条目 (ERR + lessons)
        self.knowledge_entries: List[Dict[str, Any]] = []

        # 活动故障列表 (向后兼容: 简单字符串列表)
        self.active_faults: List[str] = []

        # v598 Phase A: 故障详细记录 (含 context, timestamp, impacts)
        # 用户铁律: "跨模块联动回滚机制" — 必须记录故障上下文供回滚决策使用
        # 之前 detect_fault 仅返回 List[str] 不记录 context — 典型纸面集成
        self.active_fault_records: List[Dict[str, Any]] = []

        # v598 Phase A: 回滚执行历史 (记录真实状态变更, 非 "defined")
        self.rollback_execution_history: List[Dict[str, Any]] = []

        # v598 Phase B / Task#89: 持久化路径 (默认 None, 需 enable_persistence() 激活)
        self._persistence_path: Optional[Path] = None

        logger.debug(
            "GlobalKnowledgeBase 初始化: %d agents, %d tasks, %d entries",
            len(self.agents),
            len(self.scheduler.task_queue),
            len(self.knowledge_entries),
        )

    # ========================================================================
    # 智能体管理
    # ========================================================================

    def register_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> bool:
        """注册或更新智能体

        Args:
            agent_id: 智能体名称 (如 "Coordinator")
            agent_config: 配置字典 (role/capabilities/priority_level/status)

        Returns:
            是否成功注册
        """
        if not agent_id or not isinstance(agent_config, dict):
            return False

        self.agents[agent_id] = {
            "id": agent_config.get("id", len(self.agents) + 1),
            "role": agent_config.get("role", "unknown"),
            "capabilities": agent_config.get("capabilities", []),
            "priority_level": agent_config.get("priority_level", 3),
            "current_load": agent_config.get("current_load", 0),
            "status": agent_config.get("status", "active"),
        }
        logger.info("注册智能体: %s (%s)", agent_id, agent_config.get("role", "unknown"))
        return True

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体状态"""
        return self.agents.get(agent_id)

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """列出所有智能体"""
        return dict(self.agents)

    # ========================================================================
    # 任务管理
    # ========================================================================

    def submit_task(
        self,
        agent: str,
        task_desc: str,
        priority: int = 3,
        dependencies: Optional[List[str]] = None,
        category: str = "general",
    ) -> str:
        """提交任务到优先级调度器

        Args:
            agent: 智能体名称
            task_desc: 任务描述
            priority: 优先级 (1=最高)
            dependencies: 依赖任务ID列表
            category: 任务类别

        Returns:
            任务ID
        """
        return self.scheduler.submit_task(
            agent, task_desc, priority, dependencies, category
        )

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """获取下一个可执行任务"""
        return self.scheduler.get_next_task()

    def update_task_status(self, task_id: str, status: str, summary: str = "") -> bool:
        """更新任务状态"""
        return self.scheduler.update_status(task_id, status, summary)

    def get_task_queue_status(self) -> Dict[str, Any]:
        """获取任务队列状态"""
        return self.scheduler.get_queue_status()

    # ========================================================================
    # 故障传播与回滚
    # ========================================================================

    def detect_fault(
        self,
        fault_origin: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """检测故障传播影响 (v598 Phase A: 真实记录故障上下文)

        用户铁律: "跨模块联动回滚机制" — 必须记录故障上下文供回滚决策使用

        Args:
            fault_origin: 故障源 (如 "BacktestValidator.validation_failed")
            context: 故障上下文 (含 reason/metrics/timestamp 等, 供回滚决策)

        Returns:
            受影响模块列表
        """
        impacts = FAULT_PROPAGATION_GRAPH.get(fault_origin, [])
        if impacts:
            # 向后兼容: 简单字符串列表 (旧调用代码不破坏)
            self.active_faults.append(fault_origin)
            # v598 Phase A: 详细记录含 context + timestamp + impacts
            fault_record: Dict[str, Any] = {
                "fault_origin": fault_origin,
                "context": dict(context) if context else {},
                "impacts": list(impacts),
                "timestamp": _now_iso(),
            }
            self.active_fault_records.append(fault_record)
            logger.warning(
                "故障传播检测: %s → 影响 %d 个模块: %s (context=%s)",
                fault_origin, len(impacts), impacts,
                fault_record["context"],
            )
        return impacts

    def get_active_fault_records(self) -> List[Dict[str, Any]]:
        """获取故障详细记录 (v598 Phase A 新增)"""
        return list(self.active_fault_records)

    def execute_rollback(
        self,
        rollback_type: str,
        target_gate: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行回滚链 (v598 Phase A: 真实状态变更, 非仅 "defined")

        用户铁律: "跨模块联动回滚机制" — 回滚必须真实执行状态变更

        Args:
            rollback_type: 回滚类型 ("Tier3_to_Tier2" / "Tier2_to_Tier1" / "Emergency_Stop")
            target_gate: StagedAdmissionGate 实例, 用于真实执行 Tier 回滚状态变更
                (若为 None, 仅记录日志不调用 gate.rollback_to_tier1())
            context: 回滚上下文 (含 reason/source 等)

        Returns:
            回滚执行结果字典 (含 status="executed" 而非 "defined", 真实状态变更)
        """
        chain = ROLLBACK_CHAIN.get(rollback_type, {})
        if not chain:
            return {"error": f"未知回滚类型: {rollback_type}"}

        _timestamp = _now_iso()
        _context = dict(context) if context else {}

        # v598 Phase A: 真实执行回滚动作 (调用 target_gate 的真实方法)
        action_result: Dict[str, Any] = self._execute_rollback_action(
            rollback_type=rollback_type,
            target_gate=target_gate,
            context=_context,
        )

        result: Dict[str, Any] = {
            "rollback_type": rollback_type,
            "trigger": chain["trigger"],
            "steps": chain["chain"],
            "timestamp": _timestamp,
            "status": "executed",  # defined → executed (真实执行)
            "real_state_change": action_result.get("real_state_change", False),
            "gate_action": action_result.get("gate_action"),
            "context": _context,
        }

        # 记录到回滚执行历史 (非仅日志, 含真实状态变更)
        self.rollback_execution_history.append(result)

        logger.warning(
            "回滚链触发并执行: %s, trigger=%s, %d steps, "
            "real_state_change=%s, gate_action=%s",
            rollback_type, chain["trigger"], len(chain["chain"]),
            result["real_state_change"], result["gate_action"],
        )
        return result

    def _execute_rollback_action(
        self,
        rollback_type: str,
        target_gate: Optional[Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """真实执行回滚动作 (v598 Phase A 新增, 内部辅助方法)

        根据 rollback_type 调用 StagedAdmissionGate 的真实回滚方法,
        实现状态变更而非仅日志记录。

        Args:
            rollback_type: 回滚类型
            target_gate: StagedAdmissionGate 实例 (或 None)
            context: 回滚上下文

        Returns:
            {real_state_change: bool, gate_action: str or None}
        """
        if target_gate is None:
            # 无 gate 实例: 仅记录, 不真实执行 (兼容旧调用方)
            return {"real_state_change": False, "gate_action": None}

        _reason = str(context.get("reason", "auto_rollback"))
        gate_action: Optional[str] = None
        real_state_change: bool = False

        try:
            if rollback_type == "Tier2_to_Tier1":
                # Tier 2 → Tier 1: 调用 staged_admission_gate.rollback_to_tier1()
                if hasattr(target_gate, "rollback_to_tier1"):
                    target_gate.rollback_to_tier1(reason=_reason)
                    gate_action = "rollback_to_tier1"
                    real_state_change = True
            elif rollback_type == "Tier3_to_Tier2":
                # Tier 3 → Tier 2: 调用 staged_admission_gate.rollback_to_tier2()
                if hasattr(target_gate, "rollback_to_tier2"):
                    target_gate.rollback_to_tier2(reason=_reason)
                    gate_action = "rollback_to_tier2"
                    real_state_change = True
            elif rollback_type == "Emergency_Stop":
                # 紧急停止: 强制 ban 部署 + 回滚到 Tier 1
                if hasattr(target_gate, "rollback_to_tier1"):
                    target_gate.rollback_to_tier1(reason=f"emergency_stop:{_reason}")
                    gate_action = "rollback_to_tier1(emergency)"
                    real_state_change = True
            else:
                logger.warning("未知 rollback_type: %s, 仅记录日志", rollback_type)
        except Exception as e:
            logger.error(
                "回滚动作执行异常: rollback_type=%s, error=%s",
                rollback_type, e,
            )
            real_state_change = False
            gate_action = f"error:{type(e).__name__}"

        return {
            "real_state_change": real_state_change,
            "gate_action": gate_action,
        }

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """获取回滚执行历史 (v598 Phase A 新增)"""
        return list(self.rollback_execution_history)

    def get_active_faults(self) -> List[str]:
        """获取当前活动故障列表"""
        return list(self.active_faults)

    def clear_fault(self, fault_origin: str) -> bool:
        """清除已解决的故障"""
        if fault_origin in self.active_faults:
            self.active_faults.remove(fault_origin)
            return True
        return False

    # ========================================================================
    # ERR 知识条目管理
    # ========================================================================

    def load_err_entries(self, err_dir: Path) -> List[Dict[str, Any]]:
        """显式加载ERR条目 (从指定目录的JSON文件)

        Args:
            err_dir: ERR条目目录路径

        Returns:
            加载的ERR条目列表
        """
        err_dir = Path(err_dir)
        if not err_dir.exists():
            logger.warning("ERR目录不存在: %s", err_dir)
            return []

        loaded: List[Dict[str, Any]] = []
        for jp in sorted(err_dir.glob("*err*.json")):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if isinstance(entry, dict) and "err_id" in entry:
                    loaded.append(entry)
            except Exception as e:
                logger.warning("加载ERR条目失败 %s: %s", jp.name, e)

        self.knowledge_entries.extend(loaded)
        logger.info("加载 %d 条ERR条目 from %s", len(loaded), err_dir)
        return loaded

    def add_knowledge_entry(self, entry: Dict[str, Any]) -> bool:
        """添加知识条目

        v598 Phase B / Task#89: 增加 fingerprint 去重 + 自动持久化钩子

        Args:
            entry: 知识条目字典 (必须包含 err_id 或 id 字段)

        Returns:
            是否成功添加
        """
        if not isinstance(entry, dict):
            return False
        if "err_id" not in entry and "id" not in entry:
            return False
        # v598 Phase B / Task#89: fingerprint 去重, 避免重复入库
        fp = entry.get("fingerprint", "")
        if fp:
            existing_fps = {e.get("fingerprint", "") for e in self.knowledge_entries}
            if fp in existing_fps:
                logger.debug("[GlobalKB] 跳过重复 fingerprint: %s", fp[:16])
                return False
        self.knowledge_entries.append(entry)
        # v598 Phase B / Task#89: 自动持久化 (如果设置了持久化路径)
        if hasattr(self, "_persistence_path") and self._persistence_path:
            try:
                self.save_to_file(self._persistence_path)
            except Exception as e:
                logger.warning("[GlobalKB] 自动持久化失败 (不影响内存): %s", e)
        return True

    def get_knowledge_entries(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询知识条目 (支持按类别/严重度过滤)"""
        results = self.knowledge_entries
        if category:
            results = [e for e in results if e.get("category") == category]
        if severity:
            results = [e for e in results if e.get("severity") == severity]
        return list(results)

    # ========================================================================
    # v598 Phase B / Task#89: 持久化 (save/load)
    # 用户铁律: "复盘→教训提取→知识库入库→下次决策调用"
    # 之前 knowledge_entries 是内存列表, 重启丢失 — 典型空壳交付
    # ========================================================================

    def enable_persistence(self, path: Path) -> bool:
        """启用自动持久化 (每次 add_knowledge_entry 后自动 save)

        Args:
            path: 持久化文件路径 (JSON)

        Returns:
            是否成功加载已有数据
        """
        self._persistence_path = Path(path)
        # 如果文件已存在, 先加载
        if self._persistence_path.exists():
            return self.load_from_file(self._persistence_path)
        return True

    def save_to_file(self, path: Path) -> bool:
        """持久化知识库到 JSON 文件

        Args:
            path: 目标文件路径

        Returns:
            是否成功
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        data = {
            "version": "1.0",
            "saved_at": _now_iso(),
            "entries_count": len(self.knowledge_entries),
            "entries": self.knowledge_entries,
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("[GlobalKB] 已持久化 %d 条目到 %s", len(self.knowledge_entries), path)
        return True

    def load_from_file(self, path: Path) -> bool:
        """从 JSON 文件加载知识库

        Args:
            path: 源文件路径

        Returns:
            是否成功
        """
        path = Path(path)
        if not path.exists():
            return False
        import json as _json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            entries = data.get("entries", [])
            # 去重加载 (按 err_id)
            existing_ids = {e.get("err_id") or e.get("id") for e in self.knowledge_entries}
            for entry in entries:
                eid = entry.get("err_id") or entry.get("id")
                if eid and eid not in existing_ids:
                    self.knowledge_entries.append(entry)
                    existing_ids.add(eid)
            logger.info("[GlobalKB] 从 %s 加载 %d 条目 (总 %d)",
                        path, len(entries), len(self.knowledge_entries))
            return True
        except Exception as e:
            logger.warning("[GlobalKB] 加载失败: %s", e)
            return False

    # ========================================================================
    # 报告生成
    # ========================================================================

    def generate_progress_report(
        self,
        agent: str,
        task_id: str = "",
        task_desc: str = "",
        progress_pct: float = 0.0,
        kpi_snapshot: Optional[Dict[str, Any]] = None,
        blockers: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """生成标准化进度报告 (基于 PROGRESS_REPORT_TEMPLATE)

        Args:
            agent: 智能体名称
            task_id: 当前任务ID
            task_desc: 当前任务描述
            progress_pct: 进度百分比
            kpi_snapshot: KPI快照
            blockers: 阻塞项列表
            next_steps: 下一步计划

        Returns:
            标准化进度报告字典
        """
        import copy
        report = copy.deepcopy(PROGRESS_REPORT_TEMPLATE)
        report["agent"] = agent
        report["timestamp"] = _now_iso()
        report["current_task_id"] = task_id
        report["current_task_desc"] = task_desc
        report["progress_pct"] = progress_pct
        if kpi_snapshot:
            report["kpi_snapshot"].update(kpi_snapshot)
        report["blockers"] = blockers or []
        report["next_steps"] = next_steps or []
        return report

    def generate_summary(self) -> Dict[str, Any]:
        """生成知识库汇总报告"""
        return {
            "agents_registered": len(self.agents),
            "agents_active": sum(1 for a in self.agents.values() if a.get("status") == "active"),
            "tasks_total": len(self.scheduler.task_queue),
            "tasks_by_status": self.scheduler.get_queue_status()["by_status"],
            "knowledge_entries": len(self.knowledge_entries),
            "active_faults": len(self.active_faults),
            "rollback_chains_defined": len(ROLLBACK_CHAIN),
            "fault_propagation_paths": len(FAULT_PROPAGATION_GRAPH),
            "timestamp": _now_iso(),
        }

    def save_report(self, path: Path) -> Path:
        """显式保存知识库报告到文件

        Args:
            path: 输出文件路径

        Returns:
            保存的文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": self.generate_summary(),
            "agents": self.agents,
            "task_queue": self.scheduler.task_queue,
            "knowledge_entries": self.knowledge_entries,
            "active_faults": self.active_faults,
            "rollback_chain_definitions": ROLLBACK_CHAIN,
            "fault_propagation_graph": FAULT_PROPAGATION_GRAPH,
            "data_exchange_protocol": DATA_EXCHANGE_PROTOCOL,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("知识库报告已保存: %s", path)
        return path


# ============================================================================
# 模块级便捷函数 (向后兼容, 等价于 GlobalKnowledgeBase 实例方法)
# ============================================================================

def detect_fault_propagation(fault_origin: str) -> List[str]:
    """检测故障传播影响 (模块级便捷函数)

    Args:
        fault_origin: 故障源

    Returns:
        受影响模块列表
    """
    return FAULT_PROPAGATION_GRAPH.get(fault_origin, [])


def execute_rollback_chain(rollback_type: str) -> Dict[str, Any]:
    """执行回滚chain (模块级便捷函数)

    Args:
        rollback_type: 回滚类型

    Returns:
        回滚执行结果字典
    """
    chain = ROLLBACK_CHAIN.get(rollback_type, {})
    if not chain:
        return {"error": f"未知回滚类型: {rollback_type}"}
    return {
        "rollback_type": rollback_type,
        "trigger": chain["trigger"],
        "steps": chain["chain"],
        "timestamp": _now_iso(),
        "status": "defined",
    }


# ============================================================================
# v92 Phase 5: 多智能体协作激活 — 新增 3 个函数
# 用户铁律: "八智能体统一全局经验知识库、任务优先级调度、跨模块联动回滚机制"
# 用户铁律: "教训库使用起来，而不是和复盘一样，在那儿摆看"
# ============================================================================


def register_agent_status(agent_name: str, status: Dict[str, Any]) -> bool:
    """注册 agent 运行时状态（供跨 agent 协作共享）

    v92 Phase 5 新增: 激活 EIGHT_AGENTS 注册表的实际使用
    之前 EIGHT_AGENTS 是静态定义从未被运行时更新 — 典型空壳交付

    Args:
        agent_name: 必须是 EIGHT_AGENTS 中的 key
            (Coordinator/DataPipeline/StrategyEngine/RiskManager/
             FeatureEngineering/BacktestValidator/LiveDeployer/KnowledgeBase)
        status: {"status": "idle|running|blocked|failed",
                 "last_task": str, "kpi": Dict, "last_update": str(timestamp)}

    Returns:
        True 若注册成功, False 若 agent_name 不在 EIGHT_AGENTS
    """
    if agent_name not in EIGHT_AGENTS:
        logger.warning("[GlobalKB] register_agent_status: 未知 agent '%s'", agent_name)
        return False
    EIGHT_AGENTS[agent_name]["runtime_status"] = status
    EIGHT_AGENTS[agent_name]["last_heartbeat"] = status.get("last_update", "")
    return True


def propagate_fault(agent_name: str, fault_type: str) -> Dict[str, Any]:
    """执行故障传播：从故障源 agent 出发，标记所有受影响 agent

    v92 Phase 5 新增: 基于 FAULT_PROPAGATION_GRAPH 执行传播动作
    之前 FAULT_PROPAGATION_GRAPH 定义了但从未被实际使用 — 典型空壳交付

    基于 detect_fault_propagation 查询受影响列表，并对每个受影响 agent
    注册 "blocked" 状态，实现跨 agent 故障联动。

    Args:
        agent_name: 故障源 agent 名（必须在 EIGHT_AGENTS）
        fault_type: 故障类型 key
            (如 "fault", "alert", "NO_GO", "loss_exceed", "err_match")

    Returns:
        {
            "fault_origin": str,
            "fault_type": str,
            "affected_agents": List[str],
            "propagated": bool,  # 是否成功传播标记
        }
    """
    _fault_key = f"{agent_name}.{fault_type}"
    _affected = detect_fault_propagation(_fault_key)
    _propagated = False
    for _agent in _affected:
        _ok = register_agent_status(_agent, {
            "status": "blocked",
            "last_task": f"blocked_by:{_fault_key}",
            "kpi": {},
            "last_update": _now_iso(),
        })
        if _ok:
            _propagated = True
    logger.info(
        "[GlobalKB] propagate_fault(%s): affected=%s propagated=%s",
        _fault_key, _affected, _propagated,
    )
    return {
        "fault_origin": agent_name,
        "fault_type": fault_type,
        "affected_agents": _affected,
        "propagated": _propagated,
    }


def evaluate_rollback(agent_name: str, fault_type: str) -> Dict[str, Any]:
    """评估是否应触发回滚 chain（不执行，仅决策建议）

    v92 Phase 5 新增: 基于 ROLLBACK_CHAIN 评估回滚决策
    之前 ROLLBACK_CHAIN 定义了但从未被实际使用 — 典型空壳交付

    决策规则:
      - LiveDeployer.loss_exceed / RiskManager.alert → Tier3_to_Tier2
      - BacktestValidator.NO_GO → Tier2_to_Tier1
      - KnowledgeBase.err_match → 仅评估不回滚 (教训库命中由 ERR-KB block 处理)
      - 任何 agent 的 Emergency_Stop 触发条件 → Emergency_Stop

    Args:
        agent_name: 故障源 agent
        fault_type: 故障类型

    Returns:
        {
            "should_rollback": bool,
            "rollback_type": Optional[str],  # "Tier3_to_Tier2"/"Tier2_to_Tier1"/"Emergency_Stop"
            "reason": str,
            "chain_preview": List[str],
        }
    """
    _rollback_type = None
    _reason = ""

    if agent_name == "LiveDeployer" and fault_type == "loss_exceed":
        _rollback_type = "Tier3_to_Tier2"
        _reason = "实盘亏损超阈值, 触发 Tier3→Tier2 回滚"
    elif agent_name == "RiskManager" and fault_type == "alert":
        _rollback_type = "Tier3_to_Tier2"
        _reason = "风控告警, 触发 Tier3→Tier2 回滚"
    elif agent_name == "BacktestValidator" and fault_type == "NO_GO":
        _rollback_type = "Tier2_to_Tier1"
        _reason = "回测验证 NO_GO, 触发 Tier2→Tier1 回滚"
    elif fault_type in ("Emergency_Stop", "emergency"):
        _rollback_type = "Emergency_Stop"
        _reason = "紧急停止触发"

    if _rollback_type is None:
        return {
            "should_rollback": False,
            "rollback_type": None,
            "reason": f"无匹配回滚规则: {agent_name}.{fault_type}",
            "chain_preview": [],
        }

    _chain = ROLLBACK_CHAIN.get(_rollback_type, {}).get("chain", [])
    logger.warning(
        "[GlobalKB] evaluate_rollback(%s.%s): should_rollback=True, type=%s",
        agent_name, fault_type, _rollback_type,
    )
    return {
        "should_rollback": True,
        "rollback_type": _rollback_type,
        "reason": _reason,
        "chain_preview": _chain,
    }


# ============================================================================
# 单元测试入口
# ============================================================================

class _MockStagedAdmissionGate:
    """_self_test 内部辅助: 模拟 StagedAdmissionGate 用于回滚测试 (轻量 mock)

    避免 import staged_admission_gate 产生重依赖, 仅模拟 rollback_to_tier1/tier2 接口。
    """

    def __init__(self):
        self.tier: int = 3  # 初始 Tier 3
        self.rollback_log: List[Tuple[str, str]] = []

    def rollback_to_tier1(self, reason: str = "") -> None:
        self.tier = 1
        self.rollback_log.append(("to_tier1", reason))

    def rollback_to_tier2(self, reason: str = "") -> None:
        self.tier = 2
        self.rollback_log.append(("to_tier2", reason))


def _self_test() -> bool:
    """自检: 验证 GlobalKnowledgeBase 核心功能可用

    测试覆盖:
      1. 八智能体注册表 + 任务优先级调度
      2. 故障传播检测 (detect_fault, Phase A 真实 context 记录)
      3. 真实回滚执行 (execute_rollback, Phase A 真实状态变更)
    """
    try:
        kb = GlobalKnowledgeBase()

        # 测试1: 八智能体注册表 + 任务优先级调度
        assert len(kb.agents) == 8, f"应有 8 个默认 agent, 实际 {len(kb.agents)}"
        assert "Coordinator" in kb.agents
        assert "RiskManager" in kb.agents
        assert kb.agents["Coordinator"]["priority_level"] == 1

        # 提交任务 + 优先级调度
        tid1 = kb.submit_task("StrategyEngine", "回测策略A", priority=2)
        tid2 = kb.submit_task("RiskManager", "风险评估", priority=1)  # 高优先级
        assert tid1.startswith("TASK-") and tid2.startswith("TASK-")

        # 优先级排序: RiskManager(1) 应先于 StrategyEngine(2)
        next_task = kb.get_next_task()
        assert next_task is not None, "应返回下一个可执行任务"
        assert next_task["agent"] == "RiskManager", "高优先级任务应先调度"

        # 测试2: 故障传播检测 + Phase A context 记录
        impacts = kb.detect_fault(
            "DataPipeline.fault",
            context={"reason": "test_mock", "latency_ms": 5000},
        )
        assert isinstance(impacts, list) and len(impacts) == 3, \
            f"DataPipeline.fault 应影响 3 个模块, 实际 {len(impacts)}"
        assert any("StrategyEngine" in i for i in impacts)

        # 验证故障记录含 context (Phase A 新增)
        records = kb.get_active_fault_records()
        assert len(records) == 1, f"应有 1 条故障记录, 实际 {len(records)}"
        assert records[0]["fault_origin"] == "DataPipeline.fault"
        assert records[0]["context"]["latency_ms"] == 5000
        assert "timestamp" in records[0]
        assert len(records[0]["impacts"]) == 3

        # 测试3: 真实回滚执行 (Phase A 真实状态变更)
        gate = _MockStagedAdmissionGate()
        assert gate.tier == 3, "初始 Tier 应为 3"

        rollback_result = kb.execute_rollback(
            rollback_type="Tier3_to_Tier2",
            target_gate=gate,
            context={"reason": "KPI退化超20%"},
        )

        # 验证返回 dict 含真实状态变更字段
        assert isinstance(rollback_result, dict)
        assert rollback_result["status"] == "executed", \
            f"status 应为 'executed' (Phase A), 实际 '{rollback_result['status']}'"
        assert rollback_result["real_state_change"] is True, \
            "Phase A 真实回滚: real_state_change 应为 True"
        assert rollback_result["gate_action"] == "rollback_to_tier2"
        assert rollback_result["trigger"]  # trigger 文本非空
        assert len(rollback_result["steps"]) == 5  # 5 步回滚chain

        # 验证 gate 真实状态变更
        assert gate.tier == 2, f"gate.tier 应已变为 2, 实际 {gate.tier}"
        assert len(gate.rollback_log) == 1
        assert gate.rollback_log[0][0] == "to_tier2"

        # 验证回滚历史已记录
        history = kb.get_rollback_history()
        assert len(history) == 1
        assert history[0]["rollback_type"] == "Tier3_to_Tier2"

        # 测试4: 无 gate 时降级 (仅日志, 不真实执行)
        result_no_gate = kb.execute_rollback(
            rollback_type="Emergency_Stop",
            target_gate=None,
            context={"reason": "test"},
        )
        assert result_no_gate["status"] == "executed"
        assert result_no_gate["real_state_change"] is False
        assert result_no_gate["gate_action"] is None

        # 测试5: 未知回滚类型应返回 error
        result_unknown = kb.execute_rollback(rollback_type="Unknown_Type")
        assert "error" in result_unknown

        logger.info("GlobalKnowledgeBase 自检全部通过 ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
