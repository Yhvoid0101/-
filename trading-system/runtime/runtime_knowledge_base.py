#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeKnowledgeBase — 八智能体运行时统一经验知识库 (v595 真集成 Phase 1)
================================================================================
用途: 把 v595 静态 JSON 注册表升级为运行时实体, 在进化后钩子点触发真实任务调度

设计原则:
  - 真闭环 = 写入 + 检索 + 应用 + 验证 (Reflexion 范式)
  - 5 库记忆 (TradingAgents-CN 0.7.0): situation/decision/lesson/procedure/risk
  - BM25 检索 + zero-overlap filter (避免电话效应)
  - Retrieval Quality Gate: 每次"应用教训"必须输出 BM25 score + 检索条目ID
  - 文档过期衰减: exp(-λ*Δt), λ=0.01 (约100代半衰期)
  - 模仿 v328_adapter/evolution_monitor 的"可选依赖注入+守卫调用"模式

来源参考:
  - v595_global_knowledge_base.py: 复用 EIGHT_AGENTS/FAULT_PROPAGATION_GRAPH/ROLLBACK_CHAIN 常量
  - Microsoft R&D-Agent(Q) NeurIPS 2025: 多臂老虎机调度探索/开发方向
  - TradingAgents-CN 0.7.0: reflect_and_memory + 5 库 BM25 + JSONL 持久化
  - Reflexion (arXiv:2303.11366): Actor→Evaluator→Self-Reflector→update_memory
  - EvoRAG (arXiv:2604.15676): 反馈驱动反向传播破解"知识库静态不更新"

集成点:
  - evolution_loop.py __init__: self.knowledge_base = RuntimeKnowledgeBase() if available
  - evolution_loop.py L2618 后: self.knowledge_base.on_generation_end(generation, stats, ...)
================================================================================
"""
import json
import math
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 常量定义 — 源自 v595_global_knowledge_base.py (避免脚本副作用, 独立运行时版本)
# ============================================================

# 八智能体注册表 (用户: "作为八个协同工作智能体之一")
EIGHT_AGENTS = {
    "Coordinator":      {"id": 1, "role": "核心协调",   "priority_level": 1, "status": "active"},
    "DataPipeline":     {"id": 2, "role": "数据管道",   "priority_level": 2, "status": "active"},
    "StrategyEngine":   {"id": 3, "role": "策略引擎",   "priority_level": 2, "status": "active"},
    "RiskManager":      {"id": 4, "role": "风控管理",   "priority_level": 1, "status": "active"},
    "FeatureEngineering": {"id": 5, "role": "特征工程", "priority_level": 3, "status": "active"},
    "BacktestValidator": {"id": 6, "role": "回测验证",  "priority_level": 2, "status": "active"},
    "LiveDeployer":     {"id": 7, "role": "实盘部署",   "priority_level": 1, "status": "active"},
    "KnowledgeBase":    {"id": 8, "role": "知识库",     "priority_level": 2, "status": "active"},
}

# 故障传播图: 模块A故障 → 影响模块B/C (源自 v595)
FAULT_PROPAGATION_GRAPH = {
    "DataPipeline.fault":      ["StrategyEngine.stop", "BacktestValidator.block", "LiveDeployer.pause"],
    "StrategyEngine.fault":    ["BacktestValidator.block", "LiveDeployer.rollback"],
    "RiskManager.alert":       ["LiveDeployer.pause", "Coordinator.notify"],
    "FeatureEngineering.fault":["StrategyEngine.degrade", "BacktestValidator.revalidate"],
    "BacktestValidator.NO_GO": ["LiveDeployer.block", "StrategyEngine.optimize"],
    "LiveDeployer.loss_exceed":["RiskManager.assess", "Coordinator.rollback_decision"],
    "KnowledgeBase.err_match": ["Coordinator.notify", "StrategyEngine.apply_fix"],
}

# 回滚chain定义 (基于 v594 分阶准入规则)
ROLLBACK_CHAIN = {
    "Tier3_to_Tier2": {
        "trigger": "实盘KPI退化>20% 或 连续2月亏损",
        "chain": [
            {"step": 1, "agent": "LiveDeployer",      "action": "停止标准实盘, 资金缩减到$5k"},
            {"step": 2, "agent": "RiskManager",       "action": "重新评估风险敞口"},
            {"step": 3, "agent": "BacktestValidator",  "action": "重新跑回测验证策略"},
            {"step": 4, "agent": "KnowledgeBase",      "action": "记录退化原因+ERR入库"},
            {"step": 5, "agent": "Coordinator",        "action": "通知所有agent回滚到Tier 2状态"},
        ],
    },
    "Tier2_to_Tier1": {
        "trigger": "实盘mlp>5% 或 gap>15% 或 单次亏损>2%",
        "chain": [
            {"step": 1, "agent": "LiveDeployer",      "action": "停止迷你实盘, 全部回滚到模拟"},
            {"step": 2, "agent": "RiskManager",       "action": "调查单次大亏损根因"},
            {"step": 3, "agent": "BacktestValidator",  "action": "重校准成本模型+重跑回测"},
            {"step": 4, "agent": "StrategyEngine",    "action": "策略优化解决新发现的问题"},
            {"step": 5, "agent": "KnowledgeBase",      "action": "ERR入库+教训提炼+跨agent共享"},
        ],
    },
    "Emergency_Stop": {
        "trigger": "系统性亏损信号 或 单次亏损>2%本金",
        "chain": [
            {"step": 1, "agent": "RiskManager",       "action": "立即触发紧急停止"},
            {"step": 2, "agent": "LiveDeployer",      "action": "关闭所有持仓, 撤销所有挂单"},
            {"step": 3, "agent": "Coordinator",       "action": "召集所有agent进行根因分析"},
            {"step": 4, "agent": "KnowledgeBase",     "action": "记录紧急事件+入库CRITICAL ERR"},
            {"step": 5, "agent": "StrategyEngine",     "action": "策略全面审查+修复后才能重新准入"},
        ],
    },
}

# 5 库记忆分类 (TradingAgents-CN 0.7.0 标准实现)
MEMORY_BANKS = ["situation", "decision", "lesson", "procedure", "risk"]

# 文档过期衰减参数: exp(-λ*Δgenerations), λ=0.01 → 100代半衰期
DECAY_LAMBDA = 0.01
# BM25 检索默认参数
BM25_K1 = 1.5
BM25_B = 0.75
# Retrieval Quality Gate 阈值: BM25 score 低于此值不应用
RETRIEVAL_QUALITY_THRESHOLD = 0.5


def _now_iso() -> str:
    """ISO 格式时间戳 (避免 utcnow deprecation)"""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# BM25 检索器 — 5 库记忆 + zero-overlap filter
# ============================================================

class BM25Retriever:
    """BM25 检索器 — 单库检索 (5 个库各自独立实例)

    BM25 算法 (Robertson & Zaragoza 2009):
      score(D, Q) = Σ_t IDF(t) * (f(t,D)*(k1+1)) / (f(t,D) + k1*(1-b+b*|D|/avgdl))
      IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)

    设计参考:
      - TradingAgents-CN 0.7.0: 5 库并行检索 + zero-overlap filter
      - EvoRAG (arXiv:2604.15676): 反馈驱动反向传播
    """

    def __init__(self, bank_name: str, storage_path: Optional[Path] = None):
        self.bank_name = bank_name
        self.storage_path = storage_path
        # 每条记忆: {id, text, generation, timestamp, metadata, decay_weight}
        self.documents: List[Dict[str, Any]] = []
        # 倒排索引: token -> set(doc_idx)
        self.inverted_index: Dict[str, set] = defaultdict(set)
        # 文档频率: token -> 出现该 token 的文档数
        self.doc_freq: Dict[str, int] = defaultdict(int)
        # 平均文档长度 (token 数)
        self.avg_doc_len: float = 0.0
        # 加载状态标志 (Bug 8.2 修复: 避免 _load_from_disk 期间触发 O(n²) 写盘)
        self._loading: bool = False
        # 加载历史记忆
        self._load_from_disk()

    def _tokenize(self, text: str) -> List[str]:
        """简易中英文分词 (按空格 + 中文字符单字 + 英文单词)

        生产环境可替换为 jieba, 这里保持零依赖
        """
        # 提取英文单词
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())
        # 提取中文字符 (单字)
        tokens.extend(re.findall(r"[\u4e00-\u9fff]", text))
        return tokens

    def add_document(self, doc_id: str, text: str, generation: int,
                     metadata: Optional[Dict] = None,
                     timestamp: Optional[str] = None) -> None:
        """添加文档到记忆库

        Args:
            doc_id: 唯一标识 (如 ERR-20260701-v561)
            text: 教训文本 (用于 BM25 检索)
            generation: 进化代数 (用于过期衰减)
            metadata: 附加元数据 (regime/version/agent 等)
            timestamp: ISO 时间戳 (Bug 8.3 修复: 若 None 则使用当前时间, 否则保留磁盘原始时间)
        """
        # 去重: 同 doc_id 替换
        existing_idx = None
        for i, d in enumerate(self.documents):
            if d["id"] == doc_id:
                existing_idx = i
                break

        tokens = self._tokenize(text)
        doc = {
            "id": doc_id,
            "text": text,
            "tokens": tokens,
            "generation": generation,
            # Bug 8.3 修复: 保留磁盘原始 timestamp, 新文档使用当前时间
            "timestamp": timestamp if timestamp is not None else _now_iso(),
            "metadata": metadata or {},
            "decay_weight": 1.0,  # 初始衰减权重=1.0
        }

        if existing_idx is not None:
            # 移除旧文档的索引
            old_tokens = set(self.documents[existing_idx]["tokens"])
            for t in old_tokens:
                self.inverted_index[t].discard(existing_idx)
                if not self.inverted_index[t]:
                    self.doc_freq[t] = max(0, self.doc_freq[t] - 1)
            self.documents[existing_idx] = doc
            idx = existing_idx
        else:
            self.documents.append(doc)
            idx = len(self.documents) - 1

        # 更新倒排索引
        for t in set(tokens):
            self.inverted_index[t].add(idx)
            self.doc_freq[t] = len(self.inverted_index[t])

        # 更新平均文档长度
        total_len = sum(len(d["tokens"]) for d in self.documents)
        self.avg_doc_len = total_len / len(self.documents) if self.documents else 0.0

        # Bug 8.2 修复: 加载期间不写盘, 避免 O(n²) 磁盘写入
        if not self._loading:
            self._save_to_disk()

    def search(self, query: str, top_k: int = 5,
               current_generation: int = 0) -> List[Dict[str, Any]]:
        """BM25 检索 + 过期衰减 + zero-overlap filter

        Args:
            query: 查询文本
            top_k: 返回 top-k 结果
            current_generation: 当前代数 (用于计算衰减)

        Returns:
            List of {doc_id, text, score, generation, metadata, decay}
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        N = len(self.documents)
        scores: List[float] = [0.0] * N

        # BM25 计算
        for qt in set(query_tokens):
            if qt not in self.doc_freq or self.doc_freq[qt] == 0:
                continue
            n_t = self.doc_freq[qt]
            idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)
            for doc_idx in self.inverted_index.get(qt, set()):
                doc = self.documents[doc_idx]
                f = doc["tokens"].count(qt)
                dl = len(doc["tokens"])
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * (dl / max(self.avg_doc_len, 1)))
                if denom > 0:
                    scores[doc_idx] += idf * (f * (BM25_K1 + 1)) / denom

        # 应用过期衰减: decay = exp(-λ * Δgenerations)
        results = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            doc = self.documents[i]
            delta_gen = max(0, current_generation - doc["generation"])
            decay = math.exp(-DECAY_LAMBDA * delta_gen)
            decayed_score = score * decay
            if decayed_score < RETRIEVAL_QUALITY_THRESHOLD:
                continue  # 衰减后低于阈值, 不返回
            results.append({
                "doc_id": doc["id"],
                "text": doc["text"],
                "score": score,  # 原始 BM25 分数
                "decayed_score": decayed_score,  # 衰减后分数
                "generation": doc["generation"],
                "metadata": doc["metadata"],
                "decay": decay,
            })

        # 排序 + top-k
        results.sort(key=lambda x: x["decayed_score"], reverse=True)
        return results[:top_k]

    def _load_from_disk(self) -> None:
        """从磁盘加载历史记忆 (JSONL 格式)

        Bug 8.2 修复: 加载期间设置 _loading=True, 避免 add_document 触发 O(n²) 写盘
        Bug 8.3 修复: 从磁盘记录读取原始 timestamp 并传入 add_document, 保留时间戳
        """
        if self.storage_path is None or not self.storage_path.exists():
            return
        # Bug 8.2 修复: 标记加载状态, add_document 期间不写盘
        self._loading = True
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        # Bug 8.3 修复: 保留磁盘原始 timestamp
                        self.add_document(
                            doc_id=doc["id"],
                            text=doc["text"],
                            generation=doc.get("generation", 0),
                            metadata=doc.get("metadata", {}),
                            timestamp=doc.get("timestamp"),
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue
            logger.info("[RuntimeKnowledgeBase] %s 库加载 %d 条历史记忆",
                        self.bank_name, len(self.documents))
        except Exception as e:
            logger.warning("[RuntimeKnowledgeBase] %s 库加载失败: %s", self.bank_name, e)
        finally:
            # Bug 8.2 修复: 加载完成, 恢复非加载状态
            self._loading = False

    def _save_to_disk(self) -> None:
        """持久化到磁盘 (JSONL 格式, 追加模式)"""
        if self.storage_path is None:
            return
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.storage_path.open("w", encoding="utf-8") as f:
                for doc in self.documents:
                    f.write(json.dumps({
                        "id": doc["id"],
                        "text": doc["text"],
                        "generation": doc["generation"],
                        "timestamp": doc["timestamp"],
                        "metadata": doc["metadata"],
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[RuntimeKnowledgeBase] %s 库持久化失败: %s", self.bank_name, e)


# ============================================================
# 任务优先级调度器 — 源自 v595, 升级为运行时版本
# ============================================================

class TaskPriorityScheduler:
    """八智能体任务优先级调度器 (运行时版本)

    调度规则:
      1. 优先级数字越小, 优先级越高 (L1 > L2 > L3)
      2. 同优先级按创建时间 FIFO
      3. 依赖未完成的任务不调度
    """

    def __init__(self):
        self.task_queue: List[Dict] = []
        self.task_counter: int = 0

    def submit_task(self, agent: str, task_desc: str, priority: int = 3,
                    dependencies: Optional[List[str]] = None,
                    category: str = "general") -> str:
        """提交任务到队列"""
        if agent not in EIGHT_AGENTS:
            logger.warning("[Scheduler] 未知 agent: %s", agent)
            return ""
        self.task_counter += 1
        task_id = f"TASK-{self.task_counter:04d}"
        task = {
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
        return task_id

    def get_next_task(self) -> Optional[Dict]:
        """获取下一个可执行任务 (优先级 + 依赖检查)"""
        ready = [
            t for t in self.task_queue
            if t["status"] == "pending"
            and all(self._is_dep_completed(d) for d in t["dependencies"])
        ]
        if not ready:
            return None
        ready.sort(key=lambda t: (t["priority"], t["created_at"]))
        return ready[0]

    def _is_dep_completed(self, task_id: str) -> bool:
        for t in self.task_queue:
            if t["task_id"] == task_id:
                return t["status"] == "completed"
        return True

    def update_status(self, task_id: str, status: str, summary: str = "") -> bool:
        for t in self.task_queue:
            if t["task_id"] == task_id:
                t["status"] = status
                t["updated_at"] = _now_iso()
                if summary:
                    t["summary"] = summary
                return True
        return False

    def get_queue_status(self) -> Dict[str, Any]:
        status_count = defaultdict(int)
        agent_count = defaultdict(int)
        for t in self.task_queue:
            status_count[t["status"]] += 1
            agent_count[t["agent"]] += 1
        return {
            "total": len(self.task_queue),
            "by_status": dict(status_count),
            "by_agent": dict(agent_count),
        }


# ============================================================
# RuntimeKnowledgeBase — 主类 (运行时实体)
# ============================================================

class RuntimeKnowledgeBase:
    """八智能体运行时统一经验知识库

    生命周期:
      1. EvolutionLoop.__init__ 时创建实例
      2. 每代进化后 (evolution_loop.py L2618 后) 调用 on_generation_end
      3. Loop 结束后 (evolution_loop.py L2774 后) 调用 on_loop_end

    核心接口:
      - on_generation_end(generation, stats, overfitted_agents, hall_of_fame)
      - on_loop_end(round_results) → 生成总结报告
      - retrieve_lessons(query) → BM25 检索 5 库
      - record_err(err_id, agent, description, root_cause, fix) → 入库新教训
    """

    def __init__(self, storage_dir: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.storage_dir = storage_dir

        # 5 库记忆 (TradingAgents-CN 0.7.0 标准)
        bank_dir = storage_dir / "memory_banks" if storage_dir else None
        self.banks: Dict[str, BM25Retriever] = {}
        for bank_name in MEMORY_BANKS:
            bank_path = bank_dir / f"{bank_name}.jsonl" if bank_dir else None
            self.banks[bank_name] = BM25Retriever(bank_name, bank_path)

        # 任务调度器
        self.scheduler = TaskPriorityScheduler()

        # 检索质量证据 (Retrieval Quality Gate)
        # 每次"应用教训"必须记录 BM25 score + 检索条目ID
        self.retrieval_evidence: List[Dict[str, Any]] = []

        # 故障传播图 + 回滚链 (源自 v595)
        self.fault_graph = FAULT_PROPAGATION_GRAPH
        self.rollback_chain = ROLLBACK_CHAIN

        # 统计
        self.total_generations_processed: int = 0
        self.total_lessons_recorded: int = 0
        self.total_lessons_applied: int = 0

        logger.info("[RuntimeKnowledgeBase] 初始化完成, enabled=%s, 5库记忆已就绪", self.enabled)

    def on_generation_end(self, generation: int,
                          stats: Any,
                          overfitted_agents: set,
                          hall_of_fame: List[Any]) -> Dict[str, Any]:
        """进化后自动触发 (集成点: evolution_loop.py L2618 后)

        流程 (Reflexion 范式):
          1. 提取本轮关键事件 (DD退化/Sharpe退化/策略失效段)
          2. BM25 检索历史教训 (5库)
          3. 判断类比适用性 + 应用到下一代变异偏好
          4. 入库新教训到 JSONL
          5. 触发任务调度器

        Args:
            generation: 当前进化代数
            stats: PopulationStats 对象 (含 gt_score/diversity/pbo_risk 等)
            overfitted_agents: 本轮过拟合 Agent ID 集合
            hall_of_fame: Hall of Fame 候选列表

        Returns:
            dict: {lessons_applied, new_errs_recorded, retrieval_evidence, next_tasks}
        """
        if not self.enabled:
            return {"skipped": "disabled"}

        self.total_generations_processed += 1
        result: Dict[str, Any] = {
            "generation": generation,
            "lessons_applied": [],
            "new_errs_recorded": [],
            "retrieval_evidence": [],
            "next_tasks": [],
        }

        try:
            # 1. 提取本轮关键事件
            events = self._extract_key_events(generation, stats, overfitted_agents, hall_of_fame)

            # 2. BM25 检索历史教训
            for event in events:
                query = event["query"]
                retrieved = self.retrieve_lessons(query, top_k=3, current_generation=generation)

                # Retrieval Quality Gate: 必须输出检索证据
                for r in retrieved:
                    result["retrieval_evidence"].append({
                        "event_type": event["type"],
                        "query": query,
                        "doc_id": r["doc_id"],
                        "bm25_score": round(r["score"], 4),
                        "decayed_score": round(r["decayed_score"], 4),
                        "decay": round(r["decay"], 4),
                        "bank": r["bank"],
                        "applied": r["decayed_score"] >= RETRIEVAL_QUALITY_THRESHOLD,
                    })

                # 3. 应用教训到下一代变异偏好
                if retrieved:
                    applied = self._apply_lessons(generation, event, retrieved)
                    if applied:
                        result["lessons_applied"].extend(applied)
                        self.total_lessons_applied += len(applied)

                # 4. 入库新教训 (如有新错误)
                if event.get("is_new_error"):
                    err_id = self.record_err(
                        err_id=f"ERR-GEN{generation:04d}-{event['type']}",
                        agent=event.get("agent", "BacktestValidator"),
                        description=event["description"],
                        root_cause=event.get("root_cause", "N/A"),
                        fix=event.get("fix", "N/A"),
                        bank=event.get("bank", "lesson"),
                        generation=generation,
                        metadata=event.get("metadata", {}),
                    )
                    result["new_errs_recorded"].append(err_id)
                    self.total_lessons_recorded += 1

            # 5. 触发任务调度器 — 提交下一步任务
            next_tasks = self._schedule_next_tasks(generation, events, result)
            result["next_tasks"] = next_tasks

            # 持久化检索证据 (Retrieval Quality Gate 反模式防护)
            self.retrieval_evidence.extend(result["retrieval_evidence"])

            logger.info(
                "[RuntimeKnowledgeBase] gen=%d: events=%d, lessons_applied=%d, "
                "new_errs=%d, retrieval_evidence=%d, next_tasks=%d",
                generation, len(events), len(result["lessons_applied"]),
                len(result["new_errs_recorded"]),
                len(result["retrieval_evidence"]), len(next_tasks),
            )

        except Exception as e:
            logger.warning("[RuntimeKnowledgeBase] on_generation_end failed: %s", e)
            result["error"] = str(e)

        return result

    def on_loop_end(self, round_results: List[Any]) -> Dict[str, Any]:
        """Loop 结束后触发 (集成点: evolution_loop.py L2774 后)

        生成总结报告 + 触发跨模块联动回滚检查
        """
        if not self.enabled:
            return {"skipped": "disabled"}

        report = {
            "timestamp": _now_iso(),
            "total_generations_processed": self.total_generations_processed,
            "total_lessons_recorded": self.total_lessons_recorded,
            "total_lessons_applied": self.total_lessons_applied,
            "queue_status": self.scheduler.get_queue_status(),
            "retrieval_evidence_count": len(self.retrieval_evidence),
            "bank_sizes": {name: len(b.documents) for name, b in self.banks.items()},
        }

        # 检查故障传播 — 若有 CRITICAL 事件触发回滚chain
        critical_events = [r for r in self.retrieval_evidence if r.get("applied")]
        if critical_events:
            report["critical_events"] = len(critical_events)
            report["rollback_check"] = self._check_rollback_triggers(round_results)

        logger.info("[RuntimeKnowledgeBase] loop_end: %s", report)
        return report

    def retrieve_lessons(self, query: str, top_k: int = 3,
                         current_generation: int = 0) -> List[Dict[str, Any]]:
        """BM25 检索 5 库 (并行检索 + 合并 + 排序)

        Args:
            query: 查询文本
            top_k: 每库返回 top-k, 合并后取整体 top-k
            current_generation: 当前代数 (用于过期衰减)

        Returns:
            List of {doc_id, text, score, decayed_score, decay, bank, generation, metadata}
        """
        all_results: List[Dict[str, Any]] = []
        for bank_name, retriever in self.banks.items():
            hits = retriever.search(query, top_k=top_k,
                                    current_generation=current_generation)
            for h in hits:
                h["bank"] = bank_name
                all_results.append(h)

        # 合并后按 decayed_score 排序, 取整体 top-k
        all_results.sort(key=lambda x: x["decayed_score"], reverse=True)
        return all_results[:top_k]

    def record_err(self, err_id: str, agent: str, description: str,
                   root_cause: str, fix: str,
                   bank: str = "lesson",
                   generation: int = 0,
                   metadata: Optional[Dict] = None) -> str:
        """入库新教训到指定记忆库

        Args:
            err_id: 唯一标识 (如 ERR-20260701-v561)
            agent: 来源 agent
            description: 错误描述
            root_cause: 根因
            fix: 修复方案
            bank: 入库到哪个记忆库 (situation/decision/lesson/procedure/risk)
            generation: 进化代数
            metadata: 附加元数据
        """
        if bank not in self.banks:
            bank = "lesson"

        text = f"{description} | 根因: {root_cause} | 修复: {fix} | agent: {agent}"
        meta = {
            "err_id": err_id,
            "agent": agent,
            "root_cause": root_cause,
            "fix": fix,
            **(metadata or {}),
        }
        self.banks[bank].add_document(err_id, text, generation, meta)
        logger.info("[RuntimeKnowledgeBase] ERR 入库: %s → %s 库 (gen=%d)",
                    err_id, bank, generation)
        return err_id

    def _extract_key_events(self, generation: int, stats: Any,
                           overfitted_agents: set,
                           hall_of_fame: List[Any]) -> List[Dict[str, Any]]:
        """从本轮进化结果中提取关键事件

        事件类型:
          - dd_degradation: 回撤退化
          - sharpe_degradation: Sharpe 退化
          - strategy_failure: 策略失效段
          - hall_of_fame_update: Hall of Fame 更新
          - overfitting_detected: 过拟合检测
        """
        events: List[Dict[str, Any]] = []

        # 1. 过拟合事件
        if overfitted_agents:
            events.append({
                "type": "overfitting_detected",
                "query": f"过拟合 DSR PSR 检测 失败 agent",
                "description": f"代{generation}检测到{len(overfitted_agents)}个过拟合Agent",
                "root_cause": "DSR<0.70 或 PSR<0.85, 策略可能记忆了噪声",
                "fix": "标记过拟合Agent不参与交叉变异, 直接淘汰",
                "agent": "BacktestValidator",
                "bank": "risk",
                "is_new_error": len(overfitted_agents) > 3,  # 大量过拟合才入库
                "metadata": {"overfitted_count": len(overfitted_agents)},
            })

        # 2. Hall of Fame 更新事件
        if hall_of_fame:
            events.append({
                "type": "hall_of_fame_update",
                "query": f"Hall of Fame 入选 精英 策略 稳定",
                "description": f"代{generation}Hall of Fame候选{len(hall_of_fame)}个",
                "root_cause": "N/A",
                "fix": "N/A",
                "agent": "Coordinator",
                "bank": "decision",
                "is_new_error": False,
                "metadata": {"hall_size": len(hall_of_fame)},
            })

        # 3. 统计指标事件 (从 stats 提取)
        if stats is not None:
            diversity = getattr(stats, "diversity_score", None) or getattr(stats, "diversity", None)
            if diversity is not None and diversity < 0.15:
                events.append({
                    "type": "diversity_low",
                    "query": f"多样性低 种群收敛 漂变 停滞",
                    "description": f"代{generation}多样性={diversity:.4f}<0.15, 种群收敛风险",
                    "root_cause": "精英选择压力过大, 普通Agent被淘汰",
                    "fix": "注入随机个体40%, 提高变异率, 防止早熟收敛",
                    "agent": "StrategyEngine",
                    "bank": "lesson",
                    "is_new_error": diversity < 0.10,
                    "metadata": {"diversity": diversity},
                })

            pbo = getattr(stats, "pbo_risk", None)
            if pbo is not None and pbo > 0.30:
                events.append({
                    "type": "pbo_high",
                    "query": f"PBO过拟合概率高 回测overfitting 概率",
                    "description": f"代{generation}PBO={pbo:.4f}>0.30, 过拟合风险",
                    "root_cause": "参数空间过大, 多重试验偏差",
                    "fix": "收紧参数搜索范围, 增加Walk-Forward验证",
                    "agent": "BacktestValidator",
                    "bank": "risk",
                    "is_new_error": pbo > 0.50,
                    "metadata": {"pbo": pbo},
                })

        # 4. 默认事件: 进化代际总结
        events.append({
            "type": "generation_summary",
            "query": f"代{generation} 进化总结 策略选择 淘汰",
            "description": f"代{generation}进化完成, stats可用",
            "root_cause": "N/A",
            "fix": "N/A",
            "agent": "KnowledgeBase",
            "bank": "situation",
            "is_new_error": False,
            "metadata": {"generation": generation},
        })

        return events

    def _apply_lessons(self, generation: int, event: Dict[str, Any],
                       retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用检索到的教训到下一代变异偏好

        作用: 修改变异偏好 (供 ReflectorAgent 在 Phase 4 进一步使用)
        """
        applied = []
        for r in retrieved:
            if r["decayed_score"] < RETRIEVAL_QUALITY_THRESHOLD:
                continue
            applied.append({
                "event_type": event["type"],
                "applied_lesson_id": r["doc_id"],
                "applied_score": round(r["decayed_score"], 4),
                "bank": r["bank"],
                "note": f"gen{generation}应用 gen{r['generation']}的教训",
            })
        return applied

    def _schedule_next_tasks(self, generation: int,
                             events: List[Dict[str, Any]],
                             result: Dict[str, Any]) -> List[str]:
        """任务调度器: 提交下一步任务"""
        task_ids = []

        # 根据事件类型提交任务
        for event in events:
            if event["type"] == "overfitting_detected" and event.get("is_new_error"):
                tid = self.scheduler.submit_task(
                    "StrategyEngine",
                    f"代{generation}过拟合修复: 收紧参数搜索+增加Walk-Forward",
                    priority=1, category="strategy_fix",
                )
                task_ids.append(tid)

            elif event["type"] == "diversity_low":
                tid = self.scheduler.submit_task(
                    "StrategyEngine",
                    f"代{generation}多样性低: 注入随机个体+提高变异率",
                    priority=1, category="diversity_fix",
                )
                task_ids.append(tid)

            elif event["type"] == "pbo_high" and event.get("is_new_error"):
                tid = self.scheduler.submit_task(
                    "BacktestValidator",
                    f"代{generation}PBO过高: 重新校准参数空间+多重试验校正",
                    priority=1, category="overfitting_check",
                )
                task_ids.append(tid)

        return task_ids

    def _check_rollback_triggers(self, round_results: List[Any]) -> Dict[str, Any]:
        """检查是否触发回滚chain"""
        rollback_status = {
            "Tier3_to_Tier2": {"triggered": False, "reason": ""},
            "Tier2_to_Tier1": {"triggered": False, "reason": ""},
            "Emergency_Stop": {"triggered": False, "reason": ""},
        }

        # 简单检查: 若有大量过拟合, 触发 Tier2_to_Tier1 警告
        overfit_count = sum(
            1 for r in self.retrieval_evidence[-100:]
            if r.get("event_type") == "overfitting_detected"
        )
        if overfit_count > 10:
            rollback_status["Tier2_to_Tier1"] = {
                "triggered": True,
                "reason": f"最近100代过拟合事件{overfit_count}次, 建议回滚到Tier 1",
            }

        return rollback_status

    def detect_fault_propagation(self, fault_origin: str) -> List[str]:
        """检测故障传播影响 (源自 v595)

        Args:
            fault_origin: 故障源 (如 "DataPipeline.fault")

        Returns:
            受影响的模块列表
        """
        return self.fault_graph.get(fault_origin, [])

    def execute_rollback_chain(self, rollback_type: str) -> Dict[str, Any]:
        """执行回滚chain (源自 v595)

        Args:
            rollback_type: "Tier3_to_Tier2" / "Tier2_to_Tier1" / "Emergency_Stop"

        Returns:
            回滚执行计划
        """
        chain = self.rollback_chain.get(rollback_type)
        if chain is None:
            return {"error": f"未知回滚类型: {rollback_type}"}

        # 入库回滚事件
        self.record_err(
            err_id=f"ERR-ROLLBACK-{rollback_type}-{int(time.time())}",
            agent="Coordinator",
            description=f"触发回滚chain: {rollback_type}",
            root_cause=chain["trigger"],
            fix="执行5步回滚chain",
            bank="risk",
        )

        return {
            "rollback_type": rollback_type,
            "trigger": chain["trigger"],
            "chain": chain["chain"],
            "status": "planned",
            "timestamp": _now_iso(),
        }


# ============================================================
# 单元测试入口
# ============================================================

class _MockStats:
    """_self_test 内部辅助: 模拟 PopulationStats 用于 on_generation_end 测试"""

    diversity_score = 0.08  # 低多样性触发 diversity_low 事件
    pbo_risk = 0.45        # 高 PBO 触发 pbo_high 事件


def _self_test() -> bool:
    """自检: 验证 RuntimeKnowledgeBase 核心功能可用

    测试覆盖:
      1. 5 库记忆 + BM25 检索 (record_err + retrieve_lessons)
      2. on_generation_end 钩子 (Reflexion 范式: 提取事件→检索→应用→入库→调度)
      3. 故障传播图 + 回滚链
    """
    import tempfile

    try:
        # 使用临时目录避免污染磁盘
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = RuntimeKnowledgeBase(storage_dir=Path(tmpdir), enabled=True)

            # 测试1: 5 库记忆 + BM25 检索
            assert len(kb.banks) == 5, f"应有 5 个记忆库, 实际 {len(kb.banks)}"
            for bank_name in MEMORY_BANKS:
                assert bank_name in kb.banks, f"缺少记忆库: {bank_name}"

            # 录入教训到 risk 库
            err_id = kb.record_err(
                err_id="ERR-TEST-001",
                agent="BacktestValidator",
                description="PBO过拟合风险高 参数空间过大",
                root_cause="参数空间过大 多重试验偏差",
                fix="收紧参数搜索范围 增加 Walk-Forward 验证",
                bank="risk",
                generation=1,
            )
            assert err_id == "ERR-TEST-001"
            # 验证文档已入库到 risk 库
            assert len(kb.banks["risk"].documents) == 1, \
                "risk 库应含 1 个文档"

            # BM25 检索 (查询命中关键词)
            results = kb.retrieve_lessons("PBO 过拟合 参数", top_k=3, current_generation=2)
            assert isinstance(results, list)
            assert len(results) >= 1, "应检索到至少 1 条相关教训"
            r = results[0]
            assert r["doc_id"] == "ERR-TEST-001"
            assert r["bank"] == "risk"
            assert r["score"] > 0, "BM25 原始分数应 > 0"
            assert r["decayed_score"] > 0, "衰减后分数应 > 0"
            assert 0.0 < r["decay"] <= 1.0

            # 测试2: on_generation_end 钩子 (Reflexion 范式)
            result = kb.on_generation_end(
                generation=2,
                stats=_MockStats(),
                overfitted_agents={"agent_001", "agent_002", "agent_003", "agent_004"},
                hall_of_fame=["champion_001"],
            )

            assert isinstance(result, dict)
            assert result["generation"] == 2
            # 必需字段存在
            for key in ("lessons_applied", "new_errs_recorded",
                       "retrieval_evidence", "next_tasks"):
                assert key in result, f"结果缺少字段: {key}"
            assert kb.total_generations_processed == 1
            # 低多样性 + 高 PBO + 4 个过拟合 agent 应触发多个事件
            assert len(result["retrieval_evidence"]) > 0, \
                "应输出检索证据 (Retrieval Quality Gate)"

            # 测试3: 故障传播图 + 回滚链
            affected = kb.detect_fault_propagation("StrategyEngine.fault")
            assert len(affected) == 2, \
                f"StrategyEngine.fault 应影响 2 模块, 实际 {len(affected)}"

            rollback_plan = kb.execute_rollback_chain("Tier2_to_Tier1")
            assert rollback_plan["rollback_type"] == "Tier2_to_Tier1"
            assert rollback_plan["status"] == "planned"
            assert "chain" in rollback_plan
            assert len(rollback_plan["chain"]) == 5  # 5 步回滚chain
            assert "trigger" in rollback_plan

            logger.info("RuntimeKnowledgeBase 自检全部通过 ✓")
            return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
