"""基因进化操作函数群 (Phase 7.11 从 gene_codec.py 提取)

包含:
  - _mutate_value: 基因值变异
  - _apply_err_lessons_to_gene: 错误教训应用
  - crossover: 基因交叉
  - gene_similarity_from_vectors: 向量相似度
  - gene_similarity: 基因相似度
"""
from __future__ import annotations

import copy
import random
import uuid
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# 从 gene_codec 导入需要的类
from .gene_codec import (
    AgentGene,
    GeneField,
    GENE_SCHEMA,
)


def _mutate_value(value: Any, field: GeneField) -> Any:
    """对单个基因值进行变异"""
    if field.field_type == "choice":
        if random.random() < field.mutation_rate:
            return random.choice(field.value_range)
        return value
    elif field.field_type == "float":
        if random.random() < field.mutation_rate:
            delta = random.gauss(0, field.mutation_std * (field.value_range[1] - field.value_range[0]))
            new_val = value + delta
            return max(field.value_range[0], min(field.value_range[1], new_val))
        return value
    elif field.field_type == "int":
        if random.random() < field.mutation_rate:
            # v2.18 P1a-rollback: 回退到v2.17.2原始全值域重采样
            # 原因：v2.18 P1a邻近漂移(25%/50%)均导致GT max和wf_ratio显著下降
            #   - 25%步长: GT max 125.5→81.9 (-34.7%), wf_ratio 0.96→0.84
            #   - 50%步长: GT max 125.5→64.6 (-48.5%), wf_ratio 0.96→0.73
            # 教训：原全值域重采样虽然"过激"，但其探索能力是v2.17.2表现的关键
            #       邻近漂移破坏了这种探索能力，违反"标准只升不降"铁律
            low, high = field.value_range
            return random.randint(low, high)
        return value
    elif field.field_type == "bool":
        if random.random() < field.mutation_rate:
            return not value
        return value
    elif field.field_type == "multi_choice":
        if random.random() < field.mutation_rate:
            # 随机替换、添加或删除一个工具
            action = random.choice(["add", "remove", "replace"])
            if action == "add" and len(value) < len(field.value_range):
                # P4#13: 将 value 转为 set，使 `s not in value` 从 O(n) 降为 O(1)
                value_set = set(value)
                candidates = [s for s in field.value_range if s not in value_set]
                if candidates:
                    value = value + [random.choice(candidates)]
            elif action == "remove" and len(value) > 1:
                value = value.copy()
                value.remove(random.choice(value))
            elif action == "replace" and value:
                # P4#13: 将 value 转为 set，使 `s not in value` 从 O(n) 降为 O(1)
                value_set = set(value)
                candidates = [s for s in field.value_range if s not in value_set]
                if candidates:
                    value = value.copy()
                    value[random.randint(0, len(value) - 1)] = random.choice(candidates)
        return value
    return value



def _apply_err_lessons_to_gene(gene: AgentGene, err_lessons: Optional[List[Dict]]) -> AgentGene:
    """v4.0 Phase 6 Task #29.2: ERR教训基因级应用 (内部辅助函数)

    在 crossover / generate_random_gene / generate_seed_based_gene 中复用。
    BLOCK级: 强制 clamp 到安全范围 (基于 related_module)
    WARN级: 由 mutation_rate_boost 机制覆盖, 此处不处理

    应用教训链:
      ERR-106 (v576 LINK trail=1.5退化): trail收紧效果symbol-dependent → 收紧 trailing_distance
      ERR-098 (v479b 模拟-实盘差异): 仓位过大 → 收紧 base_risk
      ERR-093 (consec_loss≤3物理极限): 收紧 stop_loss

    Args:
        gene: 待修改的 AgentGene (原地修改)
        err_lessons: ERR教训列表 (None 或空列表 = 不修改)

    Returns:
        修改后的 gene (原地修改, 返回引用方便链式调用)
    """
    if not err_lessons:
        return gene
    for lesson in err_lessons:
        try:
            severity = lesson.get("severity", "")
            related = lesson.get("related_module", "")
            err_id = lesson.get("err_id", "")
            if severity == "BLOCK":
                # Phase 14.16r B6: 死因教训优先用 err_id 精准匹配
                if err_id == "DEATH_HOLD_TO_DEATH":
                    # 持仓到死 → 缩短 max_hold_hours 到 (12, 24)
                    gene.exit_risk_max_hold_hours = max(12, min(24, gene.exit_risk_max_hold_hours))
                elif err_id == "DEATH_CONSECUTIVE_SL":
                    # 连续止损死亡 → 收紧 stop_loss_pct 到 (0.015, 0.020)
                    gene.exit_risk_stop_loss_pct = max(0.015, min(0.020, gene.exit_risk_stop_loss_pct))
                elif "exit_risk" in related or "trailing" in related:
                    # ERR-106: trail敏感 → 收紧 trailing_distance
                    gene.exit_risk_trailing_distance_pct = min(
                        gene.exit_risk_trailing_distance_pct, 0.015)
                elif "position_sizing" in related or "kelly" in related:
                    # ERR-098: 仓位过大 → 收紧 base_risk
                    gene.position_sizing_base_risk_per_trade = min(
                        gene.position_sizing_base_risk_per_trade, 0.02)
                elif "consec" in related or "loss" in related:
                    # ERR-093: consec_loss≤3是物理极限 → 收紧 stop_loss
                    gene.exit_risk_stop_loss_pct = min(
                        gene.exit_risk_stop_loss_pct, 0.025)
                # ERR-109 (MTF方向对齐): 无对应基因字段, 通过 frontier_enhancement 处理
            # WARN级: DEATH_ZERO_TRADE 由 mutation_rate_boost 机制覆盖, 此处不处理
        except Exception:
            # 教训应用失败不应阻断进化, 静默跳过
            continue
    return gene



def crossover(
    parent_a: AgentGene,
    parent_b: AgentGene,
    generation: int,
    mutation_rate_boost: float = 0.0,
    intraday_mode: bool = False,
    err_lessons: Optional[List[Dict]] = None,
) -> AgentGene:
    """基因交叉 + 变异：从两个父代生成子代

    交叉策略：
      1. 每个模块独立交叉（uniform crossover）
      2. 每个字段独立变异
      3. 精英保留：如果父代A的适应度显著高于B，子代更偏向A

    Args:
        parent_a: 父代A（通常适应度更高）
        parent_b: 父代B
        generation: 子代的代数
        mutation_rate_boost: 变异率加成（种群多样性低时提高）
        intraday_mode: v2.35 日内交易模式 — 变异后强制 clamp 到日内硬约束
            来源: 500轮进化诊断 — 默认范围(1,720)过宽,变异破坏初始约束
        err_lessons: v4.0 Phase 6 ERR教训列表 (闭环核心)
            None=不注入教训(向后兼容); List[Dict]=每条教训含 severity/related_module/err_id
            BLOCK级: 强制 clamp 到安全范围 (基于 related_module)
            WARN级: 由 mutation_rate_boost 机制覆盖, 此处仅记录

    Returns:
        子代AgentGene
    """
    child = AgentGene(
        agent_id=f"agent-gen{generation}-{uuid.uuid4().hex[:8]}",
        generation=generation,
        parent_ids=[parent_a.agent_id, parent_b.agent_id],
    )

    # 根据适应度差异决定偏向
    if parent_a.fitness_score > 0 and parent_b.fitness_score > 0:
        a_weight = parent_a.fitness_score / (parent_a.fitness_score + parent_b.fitness_score)
    else:
        a_weight = 0.5

    for module_name, fields in GENE_SCHEMA.items():
        for field in fields:
            attr_name = f"{module_name}_{field.name}"

            # 选择从哪个父代继承（加权随机）
            if random.random() < a_weight:
                inherited = getattr(parent_a, attr_name)
            else:
                inherited = getattr(parent_b, attr_name)

            # 对multi_choice字段做深拷贝避免引用共享
            if field.field_type == "multi_choice":
                inherited = copy.deepcopy(inherited)

            # 变异
            boosted_rate = min(field.mutation_rate + mutation_rate_boost, 0.5)
            field_copy = GeneField(
                field.name, field.field_type, field.value_range,
                boosted_rate, field.mutation_std
            )
            mutated = _mutate_value(inherited, field_copy)

            setattr(child, attr_name, mutated)

    # Phase E: 沙盘模式下过滤非现货/永续策略
    # 根因: GENE_SCHEMA 的 worldview.primary 字段在 crossover 中被交叉/变异,
    #   可能选中 SANDBOX_UNSUPPORTED_SEEDS 中的策略（期权/链上/MEV/跨链）
    # 修复: 如果子代 worldview_primary 落入 unsupported，替换为随机支持的种子
    import os as _os
    if _os.environ.get("HERMES_SANDBOX_MODE", "0") == "1":
        from .strategy_registry import StrategyRegistry as _SR
        _unsupported = _SR.SANDBOX_UNSUPPORTED_SEEDS
        if child.worldview_primary in _unsupported:
            from .gene_codec import WORLDVIEW_SEEDS as _WS
            _available = [s for s in _WS.keys() if s not in _unsupported]
            if _available:
                child.worldview_primary = random.choice(_available)

    # v2.35 日内交易模式 clamp — 防止变异破坏日内硬约束
    # 根因: _mutate_value 用 GENE_SCHEMA 原范围(如 max_hold_hours 1-720)重采样
    #   会突破 generate_intraday_gene Step 4 的安全网,导致进化后代违反日内硬上限
    # 修复: 子代生成后强制 clamp,与 generate_intraday_gene Step 4 完全一致
    # v503 Fix 23: 同步 Fix 20/21/23 的参数 (TP 3-5%, max_hold 12-24h, trailing 0.5-1%)
    #   原v3.10.21的剥头皮约束(TP≤2.5%, max_hold≤8h)已被Fix 20废弃,但此处的clamp未同步
    #   这会导致进化后基因退化回剥头皮参数,是Fix 21遗漏的第三层(第一层GENE_SCHEMA,第二层INTRADAY_GENE_OVERRIDES,第三层安全网×2)
    if intraday_mode:
        if child.exit_risk_max_hold_hours > 24:
            child.exit_risk_max_hold_hours = random.randint(12, 24)  # v503 Fix 23: 3-8→12-24 (同步Fix 20)
        if child.exit_risk_take_profit_pct > 0.025:
            child.exit_risk_take_profit_pct = random.uniform(0.012, 0.025)  # v503 Fix 30: 0.03-0.05→0.012-0.025 (同步Fix 30, P60 TP 100%不可达)
        if child.exit_risk_stop_loss_pct > 0.03:
            child.exit_risk_stop_loss_pct = random.uniform(0.015, 0.025)  # v503 Fix 23: 0.02-0.03→0.015-0.025 (同步Fix 20)
        # v503 Fix 23: 新增 trailing_distance clamp (P50 trailing_stop 0%触发率)
        if child.exit_risk_trailing_distance_pct > 0.01:
            child.exit_risk_trailing_distance_pct = random.uniform(0.005, 0.01)  # v503 Fix 23: 0.01-0.02→0.005-0.01
        if child.exit_risk_max_drawdown_pct > 0.30:
            child.exit_risk_max_drawdown_pct = random.uniform(0.05, 0.30)
        if child.position_sizing_base_risk_per_trade > 0.05:
            child.position_sizing_base_risk_per_trade = random.uniform(0.01, 0.05)

    # v4.0 Phase 6 Task #29.2: ERR教训基因级应用 (闭环核心)
    # 用户铁律: "迭代经验和教训库使用起来，而不是和复盘一样，在那儿摆看！"
    # 之前: apply_lesson_to_decision 从未被调用, ERR教训仅做日志输出, 闭环断裂
    # 现在: crossover 内部应用 BLOCK 级教训, 强制 clamp 到安全范围
    # 应用教训链:
    #   ERR-106 (v576 LINK trail=1.5退化): trail收紧效果symbol-dependent → 收紧 trailing_distance
    #   ERR-098 (v479b 模拟-实盘差异): 仓位过大 → 收紧 base_risk
    #   ERR-109 (v518 MTF方向对齐被推翻): 不做方向对齐 → 降低方向过滤权重 (无对应字段, pass)
    #   ERR-100 (Kelly增强): kelly_fraction 0.25→0.667 → 已在 frontier_enhancement 处理
    if err_lessons:
        for lesson in err_lessons:
            try:
                severity = lesson.get("severity", "")
                related = lesson.get("related_module", "")
                err_id = lesson.get("err_id", "unknown")

                if severity == "BLOCK":
                    # BLOCK级: 强制 clamp 到安全范围 (基于 related_module)
                    if "exit_risk" in related or "trailing" in related:
                        # ERR-106: trail敏感 → 收紧 trailing_distance
                        child.exit_risk_trailing_distance_pct = min(
                            child.exit_risk_trailing_distance_pct, 0.015)
                    elif "position_sizing" in related or "kelly" in related:
                        # ERR-098: 仓位过大 → 收紧 base_risk
                        child.position_sizing_base_risk_per_trade = min(
                            child.position_sizing_base_risk_per_trade, 0.02)
                    elif "mtf" in related or "trend" in related:
                        # ERR-109: 不做方向对齐 → 降低方向过滤权重
                        # (AgentGene 无方向过滤字段, 通过 frontier_enhancement 处理)
                        pass
                    elif "consec" in related or "loss" in related:
                        # ERR-093: consec_loss≤3是物理极限 → 收紧 stop_loss
                        child.exit_risk_stop_loss_pct = min(
                            child.exit_risk_stop_loss_pct, 0.025)
                # WARN级: 由 mutation_rate_boost 机制覆盖, 此处不重复处理
            except Exception:
                # 教训应用失败不应阻断进化, 静默跳过
                continue

    return child



def gene_similarity_from_vectors(va: Dict[str, float], vb: Dict[str, float]) -> float:
    """P4#1: 从预计算的 style_vector 计算相似度（避免 O(n²) 循环中重复调用 compute_style_vector）。

    与 gene_similarity 行为完全一致，但接受已计算好的向量。
    用法:
        vectors = [g.compute_style_vector() for g in population]
        for i in range(n):
            for j in range(i+1, n):
                sim = gene_similarity_from_vectors(vectors[i], vectors[j])
    """
    # 余弦相似度
    dot = sum(va[k] * vb[k] for k in va)
    norm_a = sum(v * v for v in va.values()) ** 0.5
    norm_b = sum(v * v for v in vb.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return min(1.0, max(0.0, dot / (norm_a * norm_b)))



def gene_similarity(a: AgentGene, b: AgentGene) -> float:
    """计算两个基因的相似度（0-1，1=完全相同）

    用于：
      - 种群多样性评估
      - 避免近亲繁殖
      - 聚类分析

    注意：在 O(n²) 循环中请改用 gene_similarity_from_vectors + 预计算向量，
    避免每个基因的 compute_style_vector 被重复计算 n-1 次。
    """
    va = a.compute_style_vector()
    vb = b.compute_style_vector()
    return gene_similarity_from_vectors(va, vb)



