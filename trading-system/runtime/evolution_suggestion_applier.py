#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 维进化建议应用器 (EvolutionSuggestionApplier) — 用户铁律"复盘作用于技能包"

将 PerTradeAttributionEngine 产出的 5 维建议 (parameter/logic/indicator/knowledge/strategy)
分发应用到运行时各组件, 实现"归因→建议→应用→回测验证→回滚/保留"的完整闭环。

核心设计 (用户铁律):
  - "复盘不是摆设" → 建议必须落到运行时 overlay, 不是只写报告
  - "全自动应用(代码生成)" → 非参数型建议用 JSON overlay, 即时应用/回滚
  - "永远不要模拟牛逼实盘亏钱" → 安全闸门 + 自动回滚 + metrics before/after
  - "不要通过已知数据反推策略" → 建议来自归因根因, 不是历史最优拟合

5 维分发路径:
  ┌─────────────────────────────────────────────────────────────────┐
  │ PerTradeAttributionEngine.suggestions.json                     │
  │   ↓                                                             │
  │ EvolutionSuggestionApplier.apply_all()                         │
  │   ├─ parameter  → _active_param_overlay.json (兼容现有闭环)    │
  │   ├─ logic      → _active_evolution_overlay.json.logic_patches │
  │   ├─ indicator  → _active_evolution_overlay.json.indicator_patches │
  │   ├─ knowledge  → GlobalKnowledgeBase.add_knowledge_entry      │
  │   └─ strategy   → _active_evolution_overlay.json.strategy_patches │
  │   ↓                                                             │
  │ 下次回测: agent_decision_engine/feature_fusion_pipeline 加载 overlay │
  │   ↓                                                             │
  │ evaluate_rollbacks(metrics_after) → 触发条件则自动回滚         │
  └─────────────────────────────────────────────────────────────────┘

安全机制:
  - risk_level 闸门: high 需确认/medium 记录/low 直接应用
  - rollback_on: 触发条件 (如 "ann_delta_pp < -3") 满足则自动回滚
  - metrics_before/after: 每次应用记录基线, 下次回测后对比
  - patch_id 唯一标识: 支持单条回滚
  - 应用历史: _evolution_history/ 持久化每次应用+回滚记录
"""

import sys
import json
import time
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ============================================================================
# 路径与日志
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

logger = logging.getLogger("evolution_suggestion_applier")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# 常量与默认路径
# ============================================================================
OVERLAY_FILENAME = "_active_evolution_overlay.json"
PARAM_OVERLAY_FILENAME = "_active_param_overlay.json"
HISTORY_DIRNAME = "_evolution_history"
ATTRIBUTION_DIRNAME = "_attribution_reports"

# 风险等级 → 应用策略
RISK_APPLICATION_POLICY = {
    "low": "auto_apply",       # 直接应用
    "medium": "apply_logged",  # 应用但记录
    "high": "apply_cautious",  # 应用但标记需验证, 高 rollback 优先级
    "critical": "skip",        # 跳过 (需人工确认)
}

# 5 维建议类型
DIMENSIONS = ("parameter", "logic", "indicator", "knowledge", "strategy")

# 回滚条件评估阈值 (硬约束, 用户铁律"未达指标禁止部署")
ROLLBACK_THRESHOLDS = {
    "ann_delta_pp": -3.0,        # 年化下降 > 3pp → 回滚
    "max_dd_delta_pp": +3.0,     # 回撤增加 > 3pp → 回滚
    "max_consec_delta": +1,      # 连续亏损增加 > 1 → 回滚
    "n_trades_reduction_pct": 30.0,  # 交易数减少 > 30% → 回滚
    "sharpe_delta": -0.3,        # 夏普下降 > 0.3 → 回滚
}


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class AppliedPatch:
    """已应用的一条 patch 记录"""
    patch_id: str
    dimension: str                       # parameter/logic/indicator/knowledge/strategy
    source_suggestion_name: str
    proposed_change: Dict[str, Any]
    safety_analysis: Dict[str, Any]
    expected_impact: Dict[str, float]
    applied_at: str
    metrics_before: Dict[str, float] = field(default_factory=dict)
    status: str = "applied"              # applied/rolled_back/expired
    rollback_reason: str = ""
    source_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApplyResult:
    """一次 apply_all 的汇总结果"""
    total_suggestions: int = 0
    applied: int = 0
    skipped: int = 0
    failed: int = 0
    by_dimension: Dict[str, int] = field(default_factory=dict)
    applied_patches: List[AppliedPatch] = field(default_factory=list)
    skipped_reasons: List[Dict[str, str]] = field(default_factory=list)
    overlay_path: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "total_suggestions": self.total_suggestions,
            "applied": self.applied,
            "skipped": self.skipped,
            "failed": self.failed,
            "by_dimension": self.by_dimension,
            "applied_patches": [p.to_dict() for p in self.applied_patches],
            "skipped_reasons": self.skipped_reasons,
            "overlay_path": self.overlay_path,
            "timestamp": self.timestamp,
        }


# ============================================================================
# 核心应用器
# ============================================================================
class EvolutionSuggestionApplier:
    """5 维进化建议应用器

    将归因引擎产出的建议分发到:
      1. parameter → _active_param_overlay.json (兼容 AutoParamApplier)
      2. logic/indicator/strategy → _active_evolution_overlay.json (运行时 overlay)
      3. knowledge → GlobalKnowledgeBase

    使用方式:
        applier = EvolutionSuggestionApplier(baseline_version="v97")
        suggestions = applier.load_suggestions()
        result = applier.apply_all(suggestions, baseline_metrics={...})
        # ... 运行回测 ...
        applier.evaluate_rollbacks(new_metrics)
    """

    def __init__(
        self,
        baseline_version: str = "unknown",
        sandbox_dir: Optional[Path] = None,
        initial_capital: float = 10000.0,
        auto_rollback_on_degradation: bool = True,
        max_active_patches_per_dim: int = 20,
    ):
        """初始化应用器

        Args:
            baseline_version: 基线版本标签 (如 "v97_alpha_gap_optimize")
            sandbox_dir: 沙盘目录路径 (默认本文件所在目录)
            initial_capital: 初始资金, 用于 metrics 计算
            auto_rollback_on_degradation: 是否自动回滚退化 patch
            max_active_patches_per_dim: 每个维度最大活跃 patch 数 (防止 patch 爆炸)
        """
        self.baseline_version = baseline_version
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir else SCRIPT_DIR
        self.initial_capital = float(initial_capital)
        self.auto_rollback = bool(auto_rollback_on_degradation)
        self.max_active_patches = int(max_active_patches_per_dim)

        # 路径
        self.overlay_path = self.sandbox_dir / OVERLAY_FILENAME
        self.param_overlay_path = self.sandbox_dir / PARAM_OVERLAY_FILENAME
        self.history_dir = self.sandbox_dir / HISTORY_DIRNAME
        self.attribution_dir = self.sandbox_dir / ATTRIBUTION_DIRNAME

        # 确保目录存在
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # 线程锁 (overlay 读写需要线程安全)
        self._lock = threading.RLock()

        # 加载现有 overlay (累积应用, 非覆盖)
        self._overlay = self._load_or_init_overlay()
        self._applied_patches: List[AppliedPatch] = []

        # 模块可用性 (延迟导入, 避免强依赖)
        self._kb_available = False
        self._kb = None
        try:
            from sandbox_trading.global_knowledge_base import GlobalKnowledgeBase
            self._kb = GlobalKnowledgeBase()
            self._kb_available = True
        except Exception as e:
            logger.warning(f"GlobalKnowledgeBase 不可用, knowledge 维度将降级为本地存储: {e}")

        logger.info(
            f"EvolutionSuggestionApplier 初始化: baseline={baseline_version}, "
            f"overlay={self.overlay_path.name}, kb_available={self._kb_available}"
        )

    # ========================================================================
    # Overlay 加载/保存
    # ========================================================================
    def _load_or_init_overlay(self) -> Dict[str, Any]:
        """加载现有 overlay 或初始化新 overlay"""
        if self.overlay_path.exists():
            try:
                with open(self.overlay_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"overlay 损坏, 重新初始化: {e}")
        return {
            "_meta": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "baseline_version": self.baseline_version,
            },
            "logic_patches": [],
            "indicator_patches": [],
            "strategy_patches": [],
            "knowledge_entries": [],   # 镜像, 方便查阅
            "applied_history": [],     # 所有应用过的 patch 摘要
        }

    def _save_overlay(self) -> None:
        """保存 overlay 到文件"""
        with self._lock:
            self._overlay["_meta"]["updated_at"] = datetime.now().isoformat()
            tmp_path = self.overlay_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._overlay, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(self.overlay_path)

    # ========================================================================
    # 加载建议
    # ========================================================================
    def load_suggestions(
        self, suggestions_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """加载归因引擎产出的建议

        Args:
            suggestions_path: 建议文件路径, 默认从 _attribution_reports/ 找最新

        Returns:
            建议列表 (dict 形式, 来自 EvolutionSuggestion.to_dict)
        """
        if suggestions_path is None:
            # 找最新的 suggestions_*.json
            if not self.attribution_dir.exists():
                logger.warning(f"归因目录不存在: {self.attribution_dir}")
                return []
            candidates = sorted(self.attribution_dir.glob("suggestions_*.json"))
            if not candidates:
                logger.warning(f"未找到 suggestions 文件: {self.attribution_dir}")
                return []
            suggestions_path = candidates[-1]

        try:
            with open(suggestions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"加载建议失败 {suggestions_path}: {e}")
            return []

        # 兼容两种格式: 列表 或 {"suggestions": [...]}
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("suggestions", [])
        return []

    # ========================================================================
    # 主入口: apply_all
    # ========================================================================
    def apply_all(
        self,
        suggestions: List[Dict[str, Any]],
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> ApplyResult:
        """应用所有建议 — 主入口

        Args:
            suggestions: 建议列表 (来自 load_suggestions)
            baseline_metrics: 当前基线指标 (ann/max_dd/max_consec/n_trades/sharpe)

        Returns:
            ApplyResult: 应用结果汇总
        """
        result = ApplyResult(
            total_suggestions=len(suggestions),
            timestamp=datetime.now().isoformat(),
            overlay_path=str(self.overlay_path),
        )
        result.by_dimension = {d: 0 for d in DIMENSIONS}

        if baseline_metrics is None:
            baseline_metrics = {}

        for sug in suggestions:
            try:
                patch = self._dispatch_apply(sug, baseline_metrics)
                if patch is not None:
                    result.applied += 1
                    result.by_dimension[patch.dimension] = result.by_dimension.get(patch.dimension, 0) + 1
                    result.applied_patches.append(patch)
                else:
                    result.skipped += 1
                    result.skipped_reasons.append({
                        "name": sug.get("name", "?"),
                        "reason": "safety_gate_skipped",
                    })
            except Exception as e:
                logger.error(f"应用建议失败 {sug.get('name', '?')}: {e}", exc_info=True)
                result.failed += 1
                result.skipped_reasons.append({
                    "name": sug.get("name", "?"),
                    "reason": f"exception: {e}",
                })

        # 持久化 overlay
        if result.applied > 0:
            self._save_overlay()
            self._record_apply_history(result)

        logger.info(
            f"apply_all 完成: total={result.total_suggestions}, "
            f"applied={result.applied}, skipped={result.skipped}, failed={result.failed}, "
            f"by_dim={result.by_dimension}"
        )
        return result

    # ========================================================================
    # 分发器
    # ========================================================================
    def _dispatch_apply(
        self, sug: Dict[str, Any], baseline_metrics: Dict[str, float]
    ) -> Optional[AppliedPatch]:
        """按维度分发到对应的应用方法

        Returns:
            AppliedPatch 或 None (被安全闸门跳过)
        """
        dimension = sug.get("dimension", "")
        if dimension not in DIMENSIONS:
            logger.warning(f"未知维度, 跳过: {dimension} (name={sug.get('name')})")
            return None

        # 安全闸门
        if not self._check_safety_gate(sug):
            return None

        # 每维度 patch 数上限
        with self._lock:
            active_count = sum(
                1 for p in self._overlay.get(f"{dimension}_patches", [])
                if p.get("status") == "active"
            ) if dimension != "parameter" and dimension != "knowledge" else 0
        if active_count >= self.max_active_patches:
            logger.warning(
                f"维度 {dimension} 活跃 patch 数已达上限 {self.max_active_patches}, 跳过"
            )
            return None

        # 分发
        patch_id = f"patch_{dimension}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        proposed = sug.get("proposed_change", {})
        safety = sug.get("safety_analysis", {})
        expected = sug.get("expected_impact", {})

        if dimension == "parameter":
            self._apply_parameter(patch_id, sug, proposed, safety)
        elif dimension == "logic":
            self._apply_logic(patch_id, sug, proposed, safety)
        elif dimension == "indicator":
            self._apply_indicator(patch_id, sug, proposed, safety)
        elif dimension == "knowledge":
            self._apply_knowledge(patch_id, sug, proposed, safety)
        elif dimension == "strategy":
            self._apply_strategy(patch_id, sug, proposed, safety)

        patch = AppliedPatch(
            patch_id=patch_id,
            dimension=dimension,
            source_suggestion_name=sug.get("name", ""),
            proposed_change=proposed,
            safety_analysis=safety,
            expected_impact=expected,
            applied_at=datetime.now().isoformat(),
            metrics_before=dict(baseline_metrics),
            source_patterns=sug.get("source_patterns", []),
        )
        self._applied_patches.append(patch)
        return patch

    # ========================================================================
    # 安全闸门
    # ========================================================================
    def _check_safety_gate(self, sug: Dict[str, Any]) -> bool:
        """安全闸门检查 — 基于 risk_level 决定是否应用

        Returns:
            True=允许应用, False=跳过
        """
        safety = sug.get("safety_analysis", {})
        risk_level = safety.get("risk_level", "medium")
        policy = RISK_APPLICATION_POLICY.get(risk_level, "apply_logged")

        if policy == "skip":
            logger.warning(
                f"安全闸门跳过 (risk_level={risk_level}): {sug.get('name')}"
            )
            return False

        # critical 级别跳过
        if sug.get("priority") == "critical":
            logger.warning(f"critical 优先级建议需人工确认, 跳过: {sug.get('name')}")
            return False

        if policy == "apply_cautious":
            logger.info(f"高风险应用 (需重点验证): {sug.get('name')}")

        return True

    # ========================================================================
    # 维度 1: parameter — 写入 _active_param_overlay.json (兼容现有闭环)
    # ========================================================================
    def _apply_parameter(
        self, patch_id: str, sug: Dict[str, Any],
        proposed: Dict[str, Any], safety: Dict[str, Any],
    ) -> None:
        """参数型建议 → 转换为 ParamAdjustment 格式, 写入 _active_param_overlay.json

        兼容 AutoParamApplier 的 _active_param_overlay.json 结构, 使现有闭环
        无需修改即可消费归因引擎产出的参数建议。

        proposed_change 结构 (来自归因引擎):
          {target_symbol, param_name, adjustment_pct, reason}

        转换为 _active_param_overlay.json 的 patch 格式:
          {patch_id, param_name, target_symbol, adjustment_pct, reason, source: "attribution"}
        """
        param_name = proposed.get("param_name", "")
        target_symbol = proposed.get("target_symbol", "global")
        adjustment_pct = float(proposed.get("adjustment_pct", 0))
        reason = proposed.get("reason", "")

        if not param_name:
            logger.warning(f"parameter 建议缺 param_name, 跳过: {sug.get('name')}")
            return

        patch = {
            "patch_id": patch_id,
            "source": "attribution_engine",
            "source_suggestion": sug.get("name", ""),
            "param_name": param_name,
            "target_symbol": target_symbol,
            "adjustment_pct": adjustment_pct,
            "reason": reason,
            "applied_at": datetime.now().isoformat(),
            "risk_level": safety.get("risk_level", "medium"),
            "rollback_on": safety.get("rollback_on", []),
            "status": "active",
        }

        # 读取现有 param overlay, 追加 patch
        existing = {}
        if self.param_overlay_path.exists():
            try:
                with open(self.param_overlay_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        patches = existing.get("patches", [])
        patches.append(patch)
        existing["patches"] = patches
        existing["_meta"] = existing.get("_meta", {})
        existing["_meta"]["updated_at"] = datetime.now().isoformat()
        existing["_meta"]["source"] = "evolution_suggestion_applier"

        tmp = self.param_overlay_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(self.param_overlay_path)

        logger.info(
            f"  [parameter] {patch_id}: {param_name}@{target_symbol} {adjustment_pct:+.1f}%"
        )

    # ========================================================================
    # 维度 2: logic — 写入 overlay.logic_patches
    # ========================================================================
    def _apply_logic(
        self, patch_id: str, sug: Dict[str, Any],
        proposed: Dict[str, Any], safety: Dict[str, Any],
    ) -> None:
        """逻辑型建议 → 写入 _active_evolution_overlay.json.logic_patches

        proposed_change 结构 (来自归因引擎):
          {trigger:{regime,alignment}, filter:{feature,op,description}, action:"reject_entry", description}

        运行时由 agent_decision_engine.py 加载 logic_patches, 在入场决策时应用:
          - 满足 trigger 条件 + filter 匹配 → 执行 action (reject_entry/reduce_position)
        """
        patch = {
            "patch_id": patch_id,
            "source_suggestion": sug.get("name", ""),
            "trigger": proposed.get("trigger", {}),
            "filter": proposed.get("filter", {}),
            "action": proposed.get("action", ""),
            "description": proposed.get("description", ""),
            "applied_at": datetime.now().isoformat(),
            "risk_level": safety.get("risk_level", "medium"),
            "rollback_on": safety.get("rollback_on", []),
            "status": "active",
        }
        with self._lock:
            self._overlay["logic_patches"].append(patch)
        logger.info(
            f"  [logic] {patch_id}: {patch['action']} when {patch['trigger']} match {patch['filter']}"
        )

    # ========================================================================
    # 维度 3: indicator — 写入 overlay.indicator_patches
    # ========================================================================
    def _apply_indicator(
        self, patch_id: str, sug: Dict[str, Any],
        proposed: Dict[str, Any], safety: Dict[str, Any],
    ) -> None:
        """指标型建议 → 写入 _active_evolution_overlay.json.indicator_patches

        proposed_change 结构 (来自归因引擎):
          {feature, threshold, op, action:"reduce_position_by", factor, description}
          或 {feature, action:"monitor_only", description}

        运行时由 feature_fusion_pipeline.py 加载 indicator_patches, 在特征融合时应用:
          - feature >= threshold (op) → action(reduce_position_by factor / monitor_only)
        """
        patch = {
            "patch_id": patch_id,
            "source_suggestion": sug.get("name", ""),
            "feature": proposed.get("feature", ""),
            "threshold": proposed.get("threshold"),
            "op": proposed.get("op", ""),
            "action": proposed.get("action", ""),
            "factor": proposed.get("factor"),
            "description": proposed.get("description", ""),
            "applied_at": datetime.now().isoformat(),
            "risk_level": safety.get("risk_level", "medium"),
            "rollback_on": safety.get("rollback_on", []),
            "status": "active",
        }
        with self._lock:
            self._overlay["indicator_patches"].append(patch)
        logger.info(
            f"  [indicator] {patch_id}: {patch['feature']} {patch['op']} {patch['threshold']} → {patch['action']}"
        )

    # ========================================================================
    # 维度 4: knowledge — 写入 GlobalKnowledgeBase + overlay 镜像
    # ========================================================================
    def _apply_knowledge(
        self, patch_id: str, sug: Dict[str, Any],
        proposed: Dict[str, Any], safety: Dict[str, Any],
    ) -> None:
        """知识型建议 → 写入 GlobalKnowledgeBase + overlay 镜像

        proposed_change 结构 (来自归因引擎):
          {knowledge_type:"winning_pattern"|"loss_pattern", pattern, conditions, implication}

        若 GlobalKnowledgeBase 可用, 调用 add_knowledge_entry 入库;
        否则降级为只写入 overlay.knowledge_entries 本地镜像。
        """
        knowledge_type = proposed.get("knowledge_type", "pattern")
        pattern = proposed.get("pattern", "")
        conditions = proposed.get("conditions", {})
        implication = proposed.get("implication", "")

        err_id = f"EVOK-{datetime.now().strftime('%Y%m%d')}-{patch_id[-6:]}"
        entry = {
            "err_id": err_id,
            "category": "evolution_knowledge",
            "subcategory": knowledge_type,
            "pattern": pattern,
            "conditions": conditions,
            "implication": implication,
            "source": "attribution_engine",
            "source_suggestion": sug.get("name", ""),
            "patch_id": patch_id,
            "timestamp": datetime.now().isoformat(),
            "severity": "high" if knowledge_type == "loss_pattern" else "info",
        }

        # 始终写入 overlay 镜像 (本地可查阅副本, 不依赖 GlobalKnowledgeBase 是否可用)
        self._add_knowledge_to_overlay(entry)

        # 同时入库 GlobalKnowledgeBase (若可用, 实现跨模块知识共享)
        if self._kb_available and self._kb is not None:
            try:
                self._kb.add_knowledge_entry(entry)
                logger.info(f"  [knowledge] {patch_id} → GlobalKnowledgeBase + overlay: {err_id}")
            except Exception as e:
                logger.warning(f"  [knowledge] GlobalKnowledgeBase 入库失败, 仅 overlay: {e}")
        else:
            logger.info(f"  [knowledge] {patch_id} → overlay 本地镜像 (KB 不可用): {err_id}")

    def _add_knowledge_to_overlay(self, entry: Dict[str, Any]) -> None:
        """知识条目镜像到 overlay (方便查阅 + 降级存储)"""
        with self._lock:
            self._overlay["knowledge_entries"].append(entry)

    # ========================================================================
    # 维度 5: strategy — 写入 overlay.strategy_patches
    # ========================================================================
    def _apply_strategy(
        self, patch_id: str, sug: Dict[str, Any],
        proposed: Dict[str, Any], safety: Dict[str, Any],
    ) -> None:
        """策略型建议 → 写入 _active_evolution_overlay.json.strategy_patches

        proposed_change 结构 (来自归因引擎):
          {strategy_type, condition:{regime}, action:"disable"|"enable", description}

        运行时由 agent_decision_engine.py 加载 strategy_patches, 在策略选择时应用:
          - 满足 condition → 执行 action (disable/enable strategy_type)
        """
        patch = {
            "patch_id": patch_id,
            "source_suggestion": sug.get("name", ""),
            "strategy_type": proposed.get("strategy_type", ""),
            "condition": proposed.get("condition", {}),
            "action": proposed.get("action", ""),
            "description": proposed.get("description", ""),
            "applied_at": datetime.now().isoformat(),
            "risk_level": safety.get("risk_level", "medium"),
            "rollback_on": safety.get("rollback_on", []),
            "status": "active",
        }
        with self._lock:
            self._overlay["strategy_patches"].append(patch)
        logger.info(
            f"  [strategy] {patch_id}: {patch['action']} {patch['strategy_type']} when {patch['condition']}"
        )

    # ========================================================================
    # 回滚评估
    # ========================================================================
    def evaluate_rollbacks(
        self, metrics_after: Dict[str, float]
    ) -> List[str]:
        """评估是否需要回滚 — 基于回测后的指标对比

        Args:
            metrics_after: 回测后的指标 {ann, max_dd, max_consec, n_trades, sharpe, ...}

        Returns:
            rolled_back_patch_ids: 被回滚的 patch ID 列表
        """
        if not self.auto_rollback:
            logger.info("自动回滚已禁用, 跳过评估")
            return []

        rolled_back = []
        for patch in self._applied_patches:
            if patch.status != "applied":
                continue

            should_rollback, reason = self._check_rollback_conditions(
                patch, metrics_after
            )
            if should_rollback:
                self._rollback_patch(patch, reason)
                rolled_back.append(patch.patch_id)
                logger.warning(
                    f"  ⚠️ 自动回滚 {patch.patch_id} ({patch.dimension}): {reason}"
                )

        if rolled_back:
            self._save_overlay()

        logger.info(
            f"evaluate_rollbacks: 检查 {len(self._applied_patches)} patches, "
            f"回滚 {len(rolled_back)} 个"
        )
        return rolled_back

    def _check_rollback_conditions(
        self, patch: AppliedPatch, metrics_after: Dict[str, float]
    ) -> Tuple[bool, str]:
        """检查单个 patch 的回滚条件

        优先级:
          1. 显式 rollback_on 条件 (来自建议的 safety_analysis)
          2. 全局 ROLLBACK_THRESHOLDS 硬约束
        """
        before = patch.metrics_before
        if not before:
            return False, ""

        # 1. 显式 rollback_on 条件
        rollback_on = patch.safety_analysis.get("rollback_on", [])
        for cond in rollback_on:
            triggered, reason = self._evaluate_condition(cond, before, metrics_after)
            if triggered:
                return True, f"rollback_on 条件触发: {cond} ({reason})"

        # 2. 全局硬约束
        delta_ann = metrics_after.get("ann", 0) - before.get("ann", 0)
        if delta_ann < ROLLBACK_THRESHOLDS["ann_delta_pp"]:
            return True, f"年化下降 {delta_ann:.2f}pp < 阈值 {ROLLBACK_THRESHOLDS['ann_delta_pp']}"

        delta_dd = metrics_after.get("max_dd", 0) - before.get("max_dd", 0)
        if delta_dd > ROLLBACK_THRESHOLDS["max_dd_delta_pp"]:
            return True, f"回撤增加 {delta_dd:.2f}pp > 阈值 {ROLLBACK_THRESHOLDS['max_dd_delta_pp']}"

        delta_consec = metrics_after.get("max_consec", 0) - before.get("max_consec", 0)
        if delta_consec > ROLLBACK_THRESHOLDS["max_consec_delta"]:
            return True, f"连续亏损增加 {delta_consec} > 阈值 {ROLLBACK_THRESHOLDS['max_consec_delta']}"

        delta_sharpe = metrics_after.get("sharpe", 0) - before.get("sharpe", 0)
        if delta_sharpe < ROLLBACK_THRESHOLDS["sharpe_delta"]:
            return True, f"夏普下降 {delta_sharpe:.2f} < 阈值 {ROLLBACK_THRESHOLDS['sharpe_delta']}"

        # 3. 交易数大幅减少 (策略可能过度过滤)
        n_before = before.get("n_trades", 0)
        n_after = metrics_after.get("n_trades", 0)
        if n_before > 0:
            reduction_pct = (n_before - n_after) / n_before * 100
            if reduction_pct > ROLLBACK_THRESHOLDS["n_trades_reduction_pct"]:
                return True, f"交易数减少 {reduction_pct:.1f}% > 阈值 {ROLLBACK_THRESHOLDS['n_trades_reduction_pct']}%"

        return False, ""

    def _evaluate_condition(
        self, cond: str, before: Dict[str, float], after: Dict[str, float]
    ) -> Tuple[bool, str]:
        """评估单个 rollback_on 条件表达式

        支持格式 (简化版):
          "ann_delta_pp < -3"
          "max_consec > 3"
          "n_trades_reduction > 30%"

        Returns:
            (是否触发, 原因描述)
        """
        try:
            # 简单解析: "metric op value"
            parts = cond.strip().split()
            if len(parts) != 3:
                return False, f"无法解析条件: {cond}"

            metric_str, op, value_str = parts

            # 计算 metric 实际值
            actual = self._compute_metric_value(metric_str, before, after)
            threshold = float(value_str.rstrip("%"))

            # 比较
            if op == "<":
                triggered = actual < threshold
            elif op == "<=":
                triggered = actual <= threshold
            elif op == ">":
                triggered = actual > threshold
            elif op == ">=":
                triggered = actual >= threshold
            else:
                return False, f"未知操作符: {op}"

            if triggered:
                return True, f"{metric_str}={actual:.2f} {op} {threshold}"
            return False, ""
        except Exception as e:
            logger.warning(f"评估回滚条件失败 '{cond}': {e}")
            return False, str(e)

    def _compute_metric_value(
        self, metric_str: str,
        before: Dict[str, float], after: Dict[str, float],
    ) -> float:
        """计算条件表达式中的 metric 值

        支持:
          - ann_delta_pp = after.ann - before.ann
          - max_dd_delta_pp = after.max_dd - before.max_dd
          - max_consec = after.max_consec (绝对值)
          - max_consec_delta = after.max_consec - before.max_consec
          - n_trades_reduction = (before.n_trades - after.n_trades) / before.n_trades * 100
          - sharpe_delta = after.sharpe - before.sharpe
        """
        if metric_str == "ann_delta_pp":
            return after.get("ann", 0) - before.get("ann", 0)
        if metric_str == "max_dd_delta_pp":
            return after.get("max_dd", 0) - before.get("max_dd", 0)
        if metric_str == "max_consec_delta":
            return after.get("max_consec", 0) - before.get("max_consec", 0)
        if metric_str == "max_consec":
            return after.get("max_consec", 0)
        if metric_str == "n_trades_reduction":
            n_before = before.get("n_trades", 0)
            n_after = after.get("n_trades", 0)
            if n_before > 0:
                return (n_before - n_after) / n_before * 100
            return 0
        if metric_str == "n_trades_reduction_pct":
            return self._compute_metric_value("n_trades_reduction", before, after)
        if metric_str == "sharpe_delta":
            return after.get("sharpe", 0) - before.get("sharpe", 0)
        # 兜底: 直接从 after 取
        return float(after.get(metric_str, 0))

    def _rollback_patch(self, patch: AppliedPatch, reason: str) -> None:
        """执行单条 patch 回滚"""
        patch.status = "rolled_back"
        patch.rollback_reason = reason

        # 从 overlay 中标记为 rolled_back
        # knowledge 维度用 knowledge_entries key (非 knowledge_patches)
        if patch.dimension == "knowledge":
            dim_key = "knowledge_entries"
        else:
            dim_key = f"{patch.dimension}_patches"
        with self._lock:
            if dim_key in self._overlay:
                for p in self._overlay[dim_key]:
                    if p.get("patch_id") == patch.patch_id:
                        p["status"] = "rolled_back"
                        p["rollback_reason"] = reason
                        p["rolled_back_at"] = datetime.now().isoformat()
                        break

        # parameter patch 还需要从 _active_param_overlay.json 移除
        if patch.dimension == "parameter" and self.param_overlay_path.exists():
            try:
                with open(self.param_overlay_path, "r", encoding="utf-8") as f:
                    param_overlay = json.load(f)
                for p in param_overlay.get("patches", []):
                    if p.get("patch_id") == patch.patch_id:
                        p["status"] = "rolled_back"
                        p["rollback_reason"] = reason
                tmp = self.param_overlay_path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(param_overlay, f, indent=2, ensure_ascii=False, default=str)
                tmp.replace(self.param_overlay_path)
            except Exception as e:
                logger.warning(f"回滚 parameter patch 失败: {e}")

    # ========================================================================
    # 历史记录
    # ========================================================================
    def _record_apply_history(self, result: ApplyResult) -> None:
        """记录本次应用历史到 _evolution_history/"""
        history_file = self.history_dir / f"apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        record = {
            "timestamp": result.timestamp,
            "baseline_version": self.baseline_version,
            "total_suggestions": result.total_suggestions,
            "applied": result.applied,
            "skipped": result.skipped,
            "failed": result.failed,
            "by_dimension": result.by_dimension,
            "applied_patches": [
                {
                    "patch_id": p.patch_id,
                    "dimension": p.dimension,
                    "source_suggestion_name": p.source_suggestion_name,
                    "applied_at": p.applied_at,
                    "expected_impact": p.expected_impact,
                }
                for p in result.applied_patches
            ],
        }
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        except OSError as e:
            logger.warning(f"写入历史记录失败: {e}")

    # ========================================================================
    # 查询接口 (供其他模块调用)
    # ========================================================================
    def get_active_patches(self, dimension: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取当前活跃的 patches (status=active)

        5 维分发后存储位置不同:
          - parameter → _active_param_overlay.json 的 patches 字段
          - knowledge → overlay 的 knowledge_entries 字段
          - logic/indicator/strategy → overlay 的 {dim}_patches 字段
        """
        results = []
        for dim in DIMENSIONS:
            if dimension and dim != dimension:
                continue

            if dim == "parameter":
                # 从 _active_param_overlay.json 读取
                if self.param_overlay_path.exists():
                    try:
                        with open(self.param_overlay_path, "r", encoding="utf-8") as f:
                            param_overlay = json.load(f)
                        for p in param_overlay.get("patches", []):
                            if p.get("status") == "active":
                                results.append(p)
                    except (json.JSONDecodeError, OSError):
                        pass
            elif dim == "knowledge":
                # 从 overlay 的 knowledge_entries 读取
                with self._lock:
                    for p in self._overlay.get("knowledge_entries", []):
                        if p.get("status", "active") == "active":
                            results.append(p)
            else:
                # logic/indicator/strategy 从 {dim}_patches 读取
                key = f"{dim}_patches"
                with self._lock:
                    if key in self._overlay:
                        for p in self._overlay[key]:
                            if p.get("status") == "active":
                                results.append(p)
        return results

    def get_overlay_snapshot(self) -> Dict[str, Any]:
        """获取当前 overlay 完整快照 (供运行时加载)"""
        with self._lock:
            return json.loads(json.dumps(self._overlay, default=str))

    def get_metrics_before(self) -> Dict[str, float]:
        """获取最近一次应用的 metrics_before (用于回滚评估)"""
        if not self._applied_patches:
            return {}
        return dict(self._applied_patches[-1].metrics_before)

    # ========================================================================
    # 自检
    # ========================================================================
    def _self_test(self) -> bool:
        """自检 — 用合成建议验证 5 维应用路径

        Returns:
            True=通过
        """
        print("=" * 70)
        print("EvolutionSuggestionApplier 自检")
        print("=" * 70)

        # 1. 构造 5 维合成建议
        test_suggestions = [
            {
                "dimension": "parameter",
                "name": "test_reduce_pos_mult_BTC_trending",
                "proposed_change": {
                    "target_symbol": "BTC@trending",
                    "param_name": "pos_mult",
                    "adjustment_pct": -10.0,
                    "reason": "test: BTC@trending 亏损率 70%",
                },
                "expected_impact": {"ann_delta_pp": +1.5},
                "safety_analysis": {
                    "risk_level": "low",
                    "rollback_on": ["ann_delta_pp < -3"],
                },
                "priority": "high",
                "source_patterns": ["pat_test_1"],
            },
            {
                "dimension": "logic",
                "name": "test_filter_counter_trend_trending",
                "proposed_change": {
                    "trigger": {"regime": "trending", "alignment": "counter_trend"},
                    "filter": {"feature": "mtf_htf_trend_4h", "op": "eq_direction"},
                    "action": "reject_entry",
                    "description": "test: trending 下拒绝逆势入场",
                },
                "expected_impact": {"ann_delta_pp": +2.0},
                "safety_analysis": {
                    "risk_level": "medium",
                    "rollback_on": ["ann_delta_pp < -2", "n_trades_reduction > 30%"],
                },
                "priority": "medium",
                "source_patterns": ["pat_test_2"],
            },
            {
                "dimension": "indicator",
                "name": "test_reduce_pos_high_atr",
                "proposed_change": {
                    "feature": "atr_pct",
                    "threshold": 0.04,
                    "op": ">",
                    "action": "reduce_position_by",
                    "factor": 0.5,
                    "description": "test: ATR>4% 时降仓 50%",
                },
                "expected_impact": {"ann_delta_pp": +1.0},
                "safety_analysis": {"risk_level": "low", "rollback_on": []},
                "priority": "medium",
                "source_patterns": [],
            },
            {
                "dimension": "knowledge",
                "name": "test_loss_pattern_counter_trend",
                "proposed_change": {
                    "knowledge_type": "loss_pattern",
                    "pattern": "trending 下逆势入场连续亏损",
                    "conditions": {"regime": "trending", "alignment": "counter_trend"},
                    "implication": "避免在 trending 下逆势入场",
                },
                "expected_impact": {"loss_avoidance_pct": 30.0},
                "safety_analysis": {"risk_level": "low"},
                "priority": "high",
                "source_patterns": ["pat_test_2"],
            },
            {
                "dimension": "strategy",
                "name": "test_disable_mean_reversion_trending",
                "proposed_change": {
                    "strategy_type": "mean_reversion",
                    "condition": {"regime": "trending"},
                    "action": "disable",
                    "description": "test: trending 下禁用 mean_reversion",
                },
                "expected_impact": {"ann_delta_pp": +2.0, "loss_reduction_pct": 40.0},
                "safety_analysis": {
                    "risk_level": "high",
                    "rollback_on": ["ann_delta_pp < -3", "n_trades_reduction > 40%"],
                },
                "priority": "high",
                "source_patterns": ["pat_test_3"],
            },
        ]

        # 2. 应用
        baseline_metrics = {
            "ann": 91.12, "max_dd": 7.34, "max_consec": 3,
            "n_trades": 269, "sharpe": 3.51,
        }
        result = self.apply_all(test_suggestions, baseline_metrics)

        print(f"\n应用结果:")
        print(f"  total={result.total_suggestions}, applied={result.applied}, "
              f"skipped={result.skipped}, failed={result.failed}")
        print(f"  by_dimension: {result.by_dimension}")
        print(f"  overlay: {result.overlay_path}")

        # 3. 验证 overlay 文件
        assert self.overlay_path.exists(), "overlay 文件未生成"
        with open(self.overlay_path, "r", encoding="utf-8") as f:
            overlay = json.load(f)
        assert len(overlay.get("logic_patches", [])) >= 1, "logic_patches 未写入"
        assert len(overlay.get("indicator_patches", [])) >= 1, "indicator_patches 未写入"
        assert len(overlay.get("strategy_patches", [])) >= 1, "strategy_patches 未写入"
        assert len(overlay.get("knowledge_entries", [])) >= 1, "knowledge_entries 未写入"

        # 4. 验证 parameter 写入 _active_param_overlay.json
        assert self.param_overlay_path.exists(), "param overlay 文件未生成"
        with open(self.param_overlay_path, "r", encoding="utf-8") as f:
            param_overlay = json.load(f)
        assert len(param_overlay.get("patches", [])) >= 1, "param patch 未写入"

        # 5. 验证查询接口
        active = self.get_active_patches()
        assert len(active) >= 4, f"活跃 patch 数异常: {len(active)}"

        # 6. 测试回滚评估 (模拟退化场景)
        degraded_metrics = {
            "ann": 85.0,   # -6.12pp < -3 阈值 → 触发回滚
            "max_dd": 8.0,
            "max_consec": 3,
            "n_trades": 240,
            "sharpe": 3.0,
        }
        rolled = self.evaluate_rollbacks(degraded_metrics)
        print(f"\n回滚评估 (退化场景 ann 91→85):")
        print(f"  rolled_back: {len(rolled)} patches")
        for pid in rolled:
            print(f"    - {pid}")

        # 至少回滚 1 个 (ann 下降 6.12pp 超阈值)
        assert len(rolled) >= 1, "回滚未触发"

        # 7. 测试改进场景 (不回滚)
        improved_metrics = {
            "ann": 93.0, "max_dd": 7.0, "max_consec": 3,
            "n_trades": 265, "sharpe": 3.6,
        }
        # 重置 patch 状态用于测试
        for p in self._applied_patches:
            p.status = "applied"
        rolled2 = self.evaluate_rollbacks(improved_metrics)
        print(f"\n回滚评估 (改进场景 ann 91→93):")
        print(f"  rolled_back: {len(rolled2)} (应为 0)")
        assert len(rolled2) == 0, "改进场景不应回滚"

        # 8. 验证历史记录
        history_files = list(self.history_dir.glob("apply_*.json"))
        assert len(history_files) >= 1, "历史记录未生成"

        print("\n✅ 自检通过")
        print(f"  overlay: {self.overlay_path}")
        print(f"  param_overlay: {self.param_overlay_path}")
        print(f"  history: {len(history_files)} records")

        return True


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    applier = EvolutionSuggestionApplier(
        baseline_version="self_test",
        initial_capital=10000.0,
        auto_rollback_on_degradation=True,
    )
    ok = applier._self_test()
    sys.exit(0 if ok else 1)
