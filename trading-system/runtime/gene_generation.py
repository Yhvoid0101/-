"""基因生成函数群 (Phase 7.11 从 gene_codec.py 提取)

包含:
  - _ensure_leading_signals: 确保前导信号
  - _random_choice_field: 随机选择字段
  - generate_random_gene: 生成随机基因
  - _trades_per_day_to_aggressiveness: 交易频次转激进度
  - generate_seed_based_gene: 基于种子生成基因
  - generate_intraday_gene: 生成日内基因
"""
from __future__ import annotations

import os
import random
import uuid
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 从 gene_codec 导入需要的类和常量
from .gene_codec import (
    AgentGene,
    GeneField,
    GENE_SCHEMA,
    WORLDVIEW_SEEDS,
    LEADING_SIGNALS_SET,
    _WORLDVIEW_LEADING_CONSTRAINTS,
    _HOLDING_PERIOD_TO_HOURS,
    _TRADES_PER_DAY_TO_AGGRO,
    INTRADAY_GENE_OVERRIDES,
    INTRADAY_WORLDVIEW_WEIGHTS,
)


def _ensure_leading_signals(
    signals: List[str],
    required_leading: Optional[List[str]] = None,
    min_leading: int = 2,
) -> List[str]:
    """确保信号池含足够 LEADING 指标 (v503 P3-2 根治94%沉默)

    Args:
        signals: 原始随机选择的信号列表
        required_leading: 该世界观必备的 LEADING 指标 (来自 _WORLDVIEW_LEADING_CONSTRAINTS)
        min_leading: 最少 LEADING 指标数 (默认2, multi_strategy 等可降至1)

    Returns:
        修正后的信号列表 (含 ≥min_leading 个 LEADING 指标)
    """
    if required_leading is None:
        required_leading = []

    # 添加必备 LEADING 指标 (去重)
    for sig in required_leading:
        if sig not in signals:
            signals.append(sig)

    # 统计当前 LEADING 指标数
    leading_count = sum(1 for s in signals if s in LEADING_SIGNALS_SET)

    # 若 LEADING 仍不足, 从 LEADING 池补充
    if leading_count < min_leading:
        all_leading = sorted(LEADING_SIGNALS_SET)  # 确定性顺序
        for sig in all_leading:
            if sig not in signals:
                signals.append(sig)
                leading_count += 1
                if leading_count >= min_leading:
                    break

    return signals



def _random_choice_field(field: GeneField) -> Any:
    """根据字段定义生成随机值"""
    if field.field_type == "choice":
        return random.choice(field.value_range)
    elif field.field_type == "float":
        low, high = field.value_range
        return random.uniform(low, high)
    elif field.field_type == "int":
        low, high = field.value_range
        return random.randint(low, high)
    elif field.field_type == "bool":
        return random.choice(field.value_range)
    elif field.field_type == "multi_choice":
        # 随机选择1到N/2个工具
        n = random.randint(1, max(1, len(field.value_range) // 2))
        return random.sample(field.value_range, n)
    return None



def generate_random_gene(
    generation: int = 0,
    agent_id: Optional[str] = None,
    err_lessons: Optional[List[Dict]] = None,
) -> AgentGene:
    """生成一个完全随机的子Agent基因

    v4.0 Phase 6 Task #29.2: 新增 err_lessons 参数
      - IPOP restart / 随机注入路径也注入ERR教训, 闭环无死角
      - 教训应用通过内部辅助函数 _apply_err_lessons_to_gene 完成
      - BLOCK级: clamp 到安全范围 (ERR-106 trailing / ERR-098 kelly / ERR-093 stop_loss)
      - WARN级: 由 mutation_rate_boost 机制覆盖, 此处不处理

    v2.15.2: 回退基因初始化改革，只保留Novelty bonus改革
    来源：v2.15/v2.15.1实验结论——基因初始化改革导致GT-Score max下降

    实验数据：
      - v2.14.1（无改革）: GT max=101.9, mean=0.6, trades=4725
      - v2.15（激进改革）: GT max=69.6, mean=2.1, trades=6718
      - v2.15.1（减弱改革）: GT max=62.1, mean=1.4, trades=6601

    结论：
      - 基因初始化改革提升了trades和mean，但导致GT max下降
      - 原因：所有Agent都倾向于相似交易行为，降低了最优Agent的得分
      - Novelty bonus改革是成功的（mean提升），不需要基因初始化改革

    v2.15.2决策：
      - 回退基因初始化改革（恢复完全随机生成）
      - 保留Novelty bonus改革（在behavior_diversity.py中）
      - 让Novelty bonus独立发挥作用

    Args:
        generation: 代数
        agent_id: 可选的agent ID，不传则自动生成

    Returns:
        一个随机初始化的AgentGene
    """
    gene = AgentGene(
        agent_id=agent_id or f"agent-{uuid.uuid4().hex[:8]}",
        generation=generation,
    )

    for module_name, fields in GENE_SCHEMA.items():
        for field in fields:
            attr_name = f"{module_name}_{field.name}"
            if hasattr(gene, attr_name):
                setattr(gene, attr_name, _random_choice_field(field))

    # Phase E: 沙盘模式下过滤非现货/永续策略
    # 根因: GENE_SCHEMA 循环通过 _random_choice_field 随机赋值 worldview_primary,
    #   可能选中 SANDBOX_UNSUPPORTED_SEEDS 中的策略（期权/链上/MEV/跨链）
    # 修复: 如果选中了不支持的种子，替换为随机支持的种子
    _sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
    if _sandbox_mode:
        from .strategy_registry import StrategyRegistry
        _unsupported = StrategyRegistry.SANDBOX_UNSUPPORTED_SEEDS
        if gene.worldview_primary in _unsupported:
            _available = [s for s in WORLDVIEW_SEEDS.keys() if s not in _unsupported]
            if _available:
                gene.worldview_primary = random.choice(_available)

    # v503 P3-2 Fix 1b: 语义约束 primary_signals — 根治94%种群沉默
    # 根因: 完全随机的 primary_signals 有 ~30% 概率 0 个 LEADING 指标
    #   → 命中 agent_decision_engine.py L1610 (independent_count==0) 硬过滤
    #   → 永不交易 → gt_score=-5.0 → 进化淘汰 → 新一代仍沉默 (循环沉默)
    # 修复: 根据已设置的 worldview_primary, 强制 primary_signals 含 ≥2 LEADING 指标
    # 注: 此时 worldview_primary 已通过上面的循环随机赋值
    wv_primary = getattr(gene, "worldview_primary", "multi_strategy")
    required_leading = _WORLDVIEW_LEADING_CONSTRAINTS.get(wv_primary, [])
    # multi_strategy 等保留探索空间的世界观仅需 1 个 LEADING, 其余 ≥2
    min_leading = 1 if wv_primary in (
        "multi_strategy", "funding_rate_arb", "perp_market_making",
        "funding_rate_predictor",
    ) else 2
    fixed_signals = _ensure_leading_signals(
        list(gene.entry_logic_primary_signals),
        required_leading=required_leading,
        min_leading=min_leading,
    )
    gene.entry_logic_primary_signals = fixed_signals

    # v4.0 Phase 6 Task #29.2: ERR教训基因级应用 (IPOP restart / 随机注入路径)
    # 确保所有基因生成路径都注入教训, 闭环无死角 (核心铁律 0.3: 无死角注入)
    from .gene_evolution import _apply_err_lessons_to_gene
    gene = _apply_err_lessons_to_gene(gene, err_lessons)

    return gene



def _trades_per_day_to_aggressiveness(tpd: int) -> float:
    """将每日最大交易次数映射为入激进度"""
    for threshold, (lo, hi) in _TRADES_PER_DAY_TO_AGGRO:
        if tpd <= threshold:
            return random.uniform(lo, hi)
    return random.uniform(0.7, 1.0)



def generate_seed_based_gene(
    generation: int = 0,
    agent_id: Optional[str] = None,
    seed_name: Optional[str] = None,
    err_lessons: Optional[List[Dict]] = None,
) -> AgentGene:
    """从种子库生成基因 (v2.18 P1c)

    与generate_random_gene不同，本函数优先使用WORLDVIEW_SEEDS中的元数据
    来约束基因相关字段，使得注入的个体能保留种子策略的本质特性。

    设计原则：
      - worldview_primary 强制设为指定种子（保证策略多样性）
      - exit_risk.max_hold_hours 根据 holding_period 调整
      - entry_logic.entry_aggressiveness 根据 max_trades_per_day 调整
      - 其他字段保持随机（保证进化探索空间）
      - 不直接映射 key_params（避免破坏AgentGene与strategy class的解耦）

    Args:
        generation: 代数
        agent_id: 可选agent ID
        seed_name: 指定种子名，None则随机选一个

    Returns:
        一个种子驱动的AgentGene
    """
    # 先生成完全随机基因
    # v4.0 Phase 6 Task #29.2: 传递 err_lessons 给底层 generate_random_gene
    # 注: generate_random_gene 内部会首次应用教训, generate_seed_based_gene
    #     在强制覆盖种子字段后(return前)会再次应用, 确保幂等性和无死角闭环
    gene = generate_random_gene(
        generation=generation, agent_id=agent_id, err_lessons=err_lessons,
    )

    # 选择种子
    if seed_name is None or seed_name not in WORLDVIEW_SEEDS:
        # Phase E: 沙盘模式下过滤非现货/永续策略（期权/链上/MEV/跨链）
        _sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
        if _sandbox_mode:
            from .strategy_registry import StrategyRegistry
            _unsupported = StrategyRegistry.SANDBOX_UNSUPPORTED_SEEDS
            _available = [s for s in WORLDVIEW_SEEDS.keys() if s not in _unsupported]
        else:
            _available = list(WORLDVIEW_SEEDS.keys())
        seed_name = random.choice(_available)
    seed = WORLDVIEW_SEEDS[seed_name]

    # 强制设置世界观
    gene.worldview_primary = seed_name

    # 根据 holding_period 调整 max_hold_hours
    holding = seed.get("holding_period", "adaptive")
    if holding in _HOLDING_PERIOD_TO_HOURS:
        lo, hi = _HOLDING_PERIOD_TO_HOURS[holding]
        # 在该holding period的合理范围内随机
        gene.exit_risk_max_hold_hours = random.randint(lo, hi) if hi > lo else lo

    # 根据 max_trades_per_day 调整 entry_aggressiveness
    tpd = seed.get("max_trades_per_day", 5)
    gene.entry_logic_entry_aggressiveness = _trades_per_day_to_aggressiveness(tpd)

    # v503 P3-2 Fix 1c: 世界观语义约束 primary_signals
    # 根因: generate_random_gene() 已应用 Fix 1b, 但世界观是随机选的,
    #   此处强制覆盖为 seed_name 后, primary_signals 可能与新世界观不匹配
    #   (例如: 随机生成时 worldview=trend_following 配了 willr_signal,
    #          种子强制改成 mean_reversion, willr_signal 仍合适但可能缺少 mfi_divergence)
    # 修复: 根据 seed_name 对应的 _WORLDVIEW_LEADING_CONSTRAINTS 重新约束
    required_leading = _WORLDVIEW_LEADING_CONSTRAINTS.get(seed_name, [])
    min_leading = 1 if seed_name in (
        "multi_strategy", "funding_rate_arb", "perp_market_making",
        "funding_rate_predictor",
    ) else 2
    gene.entry_logic_primary_signals = _ensure_leading_signals(
        list(gene.entry_logic_primary_signals),
        required_leading=required_leading,
        min_leading=min_leading,
    )

    # v4.0 Phase 6 Task #29.2: ERR教训基因级应用 (幂等再应用)
    # 原因: 中间步骤强制覆盖了 worldview/max_hold_hours/entry_aggressiveness/primary_signals,
    #       虽然这些字段与ERR clamp字段(trailing/kelly/stop_loss)不冲突, 但为保持
    #       幂等性和"所有路径无死角注入"的闭环原则, return前再次应用教训
    from .gene_evolution import _apply_err_lessons_to_gene
    gene = _apply_err_lessons_to_gene(gene, err_lessons)

    return gene



def generate_intraday_gene(
    generation: int = 0,
    agent_id: Optional[str] = None,
    seed_name: Optional[str] = None,
) -> AgentGene:
    """生成日内交易场景的子Agent基因 (v2.35)

    与 generate_random_gene 的区别:
      1. 应用 INTRADAY_GENE_OVERRIDES 收紧参数范围(日内硬约束)
      2. 应用 INTRADAY_WORLDVIEW_WEIGHTS 加权选择高胜率世界观
      3. 强制 max_hold_hours<=24, take_profit_pct<=0.06

    Args:
        generation: 进化代数
        agent_id: 可选 ID
        seed_name: 指定世界观, None则按权重随机选择

    Returns:
        AgentGene — 符合日内交易硬约束的基因
    """
    # Step 1: 生成随机基因作为基础
    gene = generate_random_gene(generation=generation, agent_id=agent_id)

    # Step 2: 应用日内硬约束覆盖
    for module_name, overrides in INTRADAY_GENE_OVERRIDES.items():
        for field_name, new_range in overrides.items():
            attr_name = f"{module_name}_{field_name}"
            if not hasattr(gene, attr_name):
                continue
            # 在 GENE_SCHEMA 中查找字段类型
            field_def = None
            for f in GENE_SCHEMA.get(module_name, []):
                if f.name == field_name:
                    field_def = f
                    break
            if field_def is None:
                continue
            # 按字段类型在新范围内生成
            if field_def.field_type == "float":
                low, high = new_range
                setattr(gene, attr_name, random.uniform(low, high))
            elif field_def.field_type == "int":
                low, high = new_range
                setattr(gene, attr_name, random.randint(int(low), int(high)))

    # Step 3: 加权选择高胜率世界观
    if seed_name is not None and seed_name in WORLDVIEW_SEEDS:
        gene.worldview_primary = seed_name
    else:
        # Phase E: 沙盘模式下过滤非现货/永续策略
        _sandbox_mode = os.environ.get("HERMES_SANDBOX_MODE", "0") == "1"
        if _sandbox_mode:
            from .strategy_registry import StrategyRegistry
            _unsupported = StrategyRegistry.SANDBOX_UNSUPPORTED_SEEDS
            _filtered = [(n, w) for n, w in INTRADAY_WORLDVIEW_WEIGHTS if n not in _unsupported]
            if not _filtered:
                _filtered = INTRADAY_WORLDVIEW_WEIGHTS
        else:
            _filtered = INTRADAY_WORLDVIEW_WEIGHTS
        weights = [w for _, w in _filtered]
        names = [n for n, _ in _filtered]
        gene.worldview_primary = random.choices(names, weights=weights, k=1)[0]

    # v503 P3-2 Fix 1d: 日内基因世界观覆盖后重新应用语义约束
    # 根因: generate_random_gene() 的 Fix 1b 使用了随机 worldview,
    #   此处覆盖为高胜率世界观后, primary_signals 可能与新世界观不匹配
    # 修复: 根据新 worldview 重新约束 primary_signals
    required_leading = _WORLDVIEW_LEADING_CONSTRAINTS.get(gene.worldview_primary, [])
    min_leading = 1 if gene.worldview_primary in (
        "multi_strategy", "funding_rate_arb", "perp_market_making",
        "funding_rate_predictor",
    ) else 2
    gene.entry_logic_primary_signals = _ensure_leading_signals(
        list(gene.entry_logic_primary_signals),
        required_leading=required_leading,
        min_leading=min_leading,
    )

    # Step 4: 安全网 — 强制日内硬约束(防止 GENE_SCHEMA 覆盖)
    # 这是最关键的兜底,确保任何路径下生成的基因都符合日内硬上限
    # v2.38 修复 (INTRADAY-GUARD): 放宽 max_drawdown + base_risk clamp
    #   根因: 参数评估显示这两个参数正相关 (大回撤容忍+高仓位盈利)
    #         原 clamp (0.12/0.02) 限制了盈利策略的进化空间
    #   修复: max_drawdown 0.12→0.30, base_risk 0.02→0.05
    # v3.14 升级 (ME): clamp范围与v3.13 GENE_SCHEMA一致, 避免覆盖优化
    # v3.10.21 升级 (LOSS-HUNTER): 配合止盈止损比反转, 收紧clamp上限
    # v503 Fix 21: 安全网与 Fix 20 趋势跟随设计对齐
    #   根因 (P48 诊断): Fix 20 在 INTRADAY_GENE_OVERRIDES 设置 TP=4-6%, max_hold=12-24h,
    #     但本安全网仍用旧版剥头皮约束 (TP≤2.5%, max_hold≤8h),
    #     在 Step 2 之后再次覆盖, 导致 Fix 20 基因参数完全失效。
    #   P48 实测: gene_tp_pct=1.68-2.48% (应 4-6%), max_hold=5.4 bars (应 12-24h)
    #   SLTP_DIAG 确认: reg_adj.tp_multiplier=1.0 (认知层未压缩), 基因本身 TP 就不对
    #   深层逻辑 (用户铁律: "理解底层逻辑, 而非表面指标"):
    #     1. 安全网应作为"极端值兜底", 不应覆盖 INTRADAY_GENE_OVERRIDES 的设计意图
    #     2. 趋势跟随需要 TP=4-6% 让利润奔跑, max_hold=12-24h 让趋势发展
    #     3. 旧版剥头皮约束 (TP≤2.5%, max_hold≤8h) 与趋势策略 DNA 完全冲突
    #   修复: 安全网上限与 INTRADAY_GENE_OVERRIDES 一致
    #     - max_hold: 8→24 (允许 12-24h 趋势发展)
    #     - TP: 0.025→0.06 (允许 4-6% 趋势止盈)
    #     - SL: 保持 0.03 上限 (1.5-2.5% 截断亏损)
    #   铁律: "杜绝模拟牛逼,实盘亏钱" — 安全网覆盖基因设计 = 策略 DNA 被篡改 = 模拟必亏
    if gene.exit_risk_max_hold_hours > 24:
        gene.exit_risk_max_hold_hours = random.randint(12, 24)  # v503 Fix 21: 3-8→12-24 趋势跟随
    if gene.exit_risk_take_profit_pct > 0.025:
        gene.exit_risk_take_profit_pct = random.uniform(0.012, 0.025)  # v503 Fix 30: 0.03-0.05→0.012-0.025 (P60 TP 100%不可达)
    if gene.exit_risk_stop_loss_pct > 0.03:
        gene.exit_risk_stop_loss_pct = random.uniform(0.015, 0.025)  # v3.14: 0.01→0.015, 0.03→0.025
    # v503 Fix 23: trailing_distance 安全网 (P50 trailing_stop 0%触发率)
    if gene.exit_risk_trailing_distance_pct > 0.01:
        gene.exit_risk_trailing_distance_pct = random.uniform(0.005, 0.01)  # v503 Fix 23: 0.01-0.02→0.005-0.01
    if gene.exit_risk_max_drawdown_pct > 0.30:
        gene.exit_risk_max_drawdown_pct = random.uniform(0.05, 0.30)
    if gene.position_sizing_base_risk_per_trade > 0.05:
        gene.position_sizing_base_risk_per_trade = random.uniform(0.01, 0.05)

    return gene



