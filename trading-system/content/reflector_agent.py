#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReflectorAgent — Reflexion Self-Reflector 复盘闭环增强层 (Phase 4)
================================================================================
用途: 实现 Reflexion (arXiv:2303.11366) 的 Self-Reflector 角色, 在每代进化后
      反思成败根因, 生成变异偏好 (软偏好), 触发因子挖掘, 形成"写入+检索+应用+验证"
      真闭环 (用户铁律 4).

生命周期 (Reflexion 4 角色):
  1. Actor (EvolutionLoop) 执行进化
  2. Evaluator (v328_adapter) 评估过拟合
  3. Self-Reflector (本类) 反思成败根因
  4. update_memory (RuntimeKnowledgeBase) 入库教训

增强而非替代 (不重复造轮子):
  - AutoReviewEngine (auto_review_system.py): 已有复盘报告生成
  - ReviewFeedbackLoop (review_feedback_loop.py): 已有 ERR 入库
  - RuntimeKnowledgeBase (runtime_knowledge_base.py): 已有 5 库记忆 + BM25 检索
  - 本类专注: 跨代 stats 对比 + 变异偏好生成 + 特征归因 + 因子挖掘触发

依赖注入 (已存在实例, 不新建):
  - knowledge_base: RuntimeKnowledgeBase (Phase 1 已集成 L1076)
  - feature_exposure: FeatureExposure (Phase 3 已集成 L1243)
  - alpha_mining: AlphaMiningSystem (L831 已存在)

集成点 (evolution_loop.py):
  - 顶部软加载: from .reflector_agent import ReflectorAgent
  - __init__ 实例化: self.reflector = ReflectorAgent(knowledge_base=..., ...)
  - 进化后反思: self.reflector.reflect_generation(...) (在 on_generation_end 后)
  - Loop 结束反思: self.reflector.reflect_loop_end(...) (在 check_admission 后)

设计原则:
  - 铁律 2: PopulationManager 保持无记忆纯粹性 — 本类不修改 PopulationManager 内部状态,
            仅返回 mutation_bias 软偏好供 PopulationManager 软参考
  - 铁律 4: 真闭环 = 写入 (reflection_history) + 检索 (knowledge_base.retrieve_lessons)
            + 应用 (mutation_bias) + 验证 (下一代 stats 对比)
  - 守卫调用模式: 所有外部依赖调用用 if self.xxx is not None: try/except 守卫
================================================================================
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReflectorAgent:
    """Self-Reflector (Reflexion 范式) — 复盘闭环增强层

    生命周期:
      1. Actor (EvolutionLoop) 执行进化
      2. Evaluator (v328_adapter) 评估过拟合
      3. Self-Reflector (本类) 反思成败根因
      4. update_memory (RuntimeKnowledgeBase) 入库教训

    增强而非替代:
      - AutoReviewEngine: 已有复盘报告生成 (不重复)
      - ReviewFeedbackLoop: 已有 ERR 入库 (不重复)
      - RuntimeKnowledgeBase: 已有 5 库记忆 (不重复)
      - 本类专注: 跨代对比 + 变异偏好 + 因子挖掘触发
    """

    # 退化检测阈值: 5% 退化触发反思
    REGRESSION_THRESHOLD = 0.05
    # 连续 3 代退化触发因子挖掘
    ALPHA_MINING_TRIGGER_COUNT = 3

    def __init__(self, knowledge_base=None, feature_exposure=None,
                 alpha_mining=None, enabled: bool = True):
        """初始化 Self-Reflector

        Args:
            knowledge_base: RuntimeKnowledgeBase 实例 (Phase 1 已注入)
            feature_exposure: FeatureExposure 实例 (Phase 3 已注入)
            alpha_mining: AlphaMiningSystem 实例 (L831 已存在)
            enabled: 是否启用
        """
        self.enabled = enabled
        self.knowledge_base = knowledge_base
        self.feature_exposure = feature_exposure
        self.alpha_mining = alpha_mining
        self.reflection_history: List[Dict] = []
        self._prev_stats: Optional[Dict] = None
        self._consecutive_regression_count: int = 0
        logger.debug("ReflectorAgent 初始化 (enabled=%s)", enabled)

    def reflect_generation(self, generation: int, stats: Dict,
                           overfitted_agents: set, hall_of_fame: List,
                           prev_stats: Optional[Dict] = None) -> Dict:
        """反思单代进化成败 (Reflexion 第3步)

        5 步分析:
          1. 对比 prev_stats vs stats: 哪些变好/变差
          2. 分析过拟合 Agent: 为何过拟合 (特征暴露+市场状态)
          3. 分析 Hall of Fame: 为何稳定 (可复制模式)
          4. 生成变异偏好: 强化好的方向, 规避差的方向 (软偏好)
          5. 若连续 3 代退化, 触发 alpha_mining 因子挖掘

        Args:
            generation: 当前代数
            stats: 当前代统计指标 (diversity_score/best_fitness/avg_fitness/sharpe/win_rate)
            overfitted_agents: 过拟合 Agent ID 集合
            hall_of_fame: 名人堂 Agent 列表
            prev_stats: 上一代统计指标 (None 时跳过对比)

        Returns:
            {
                'generation': int,
                'improvements': List[str],
                'regressions': List[str],
                'overfitting_analysis': Dict,
                'hof_analysis': Dict,
                'mutation_bias': Dict,
                'triggered_alpha_mining': bool,
                'consecutive_regression_count': int,
            }
        """
        if not self.enabled:
            return self._empty_reflection(generation)

        # Phase H 修复: PopulationStats (dataclass) → Dict (字段名映射到 _compare_stats 期望的 key)
        # 根因: evolution_loop.py 传入 stats 是 PopulationStats 对象, _compare_stats 做 `key in prev` 报 TypeError
        # 映射: diversity→diversity_score, max_gt_score→best_fitness, mean_gt_score→avg_fitness,
        #       mean_sharpe→sharpe, mean_win_rate→win_rate
        def _normalize_stats(s):
            if s is None or isinstance(s, dict):
                return s
            # PopulationStats dataclass: 用 dataclasses.asdict 或属性访问
            try:
                from dataclasses import asdict
                d = asdict(s)
            except Exception:
                # 兜底: 手动提取已知字段
                d = {
                    "generation": getattr(s, "generation", 0),
                    "diversity": getattr(s, "diversity", 0.0),
                    "max_gt_score": getattr(s, "max_gt_score", 0.0),
                    "mean_gt_score": getattr(s, "mean_gt_score", 0.0),
                    "mean_sharpe": getattr(s, "mean_sharpe", 0.0),
                    "mean_win_rate": getattr(s, "mean_win_rate", 0.0),
                }
            # 字段名映射到 _compare_stats 期望的 key
            return {
                "diversity_score": d.get("diversity", 0.0),
                "best_fitness": d.get("max_gt_score", 0.0),
                "avg_fitness": d.get("mean_gt_score", 0.0),
                "sharpe": d.get("mean_sharpe", 0.0),
                "win_rate": d.get("mean_win_rate", 0.0),
                # 保留原始字段供其他消费者使用
                "generation": d.get("generation", 0),
                "population_size": d.get("population_size", 0),
                "mean_drawdown": d.get("mean_drawdown", 0.0),
                "elite_count": d.get("elite_count", 0),
                "eliminated_count": d.get("eliminated_count", 0),
                "new_born_count": d.get("new_born_count", 0),
                "pareto_front_size": d.get("pareto_front_size", 0),
                "overfit_detected_count": d.get("overfit_detected_count", 0),
                "mean_wf_ratio": d.get("mean_wf_ratio", 0.0),
            }

        stats = _normalize_stats(stats)
        prev_stats = _normalize_stats(prev_stats)

        improvements, regressions = [], []

        # Step 1: 跨代对比
        if prev_stats is not None:
            improvements, regressions = self._compare_stats(prev_stats, stats)

        # Step 2: 过拟合分析 (消费 FeatureExposure 特征归因)
        overfitting_analysis = self._analyze_overfitting(overfitted_agents)

        # Step 3: Hall of Fame 稳定性分析
        hof_analysis = self._analyze_hof_stability(hall_of_fame)

        # Step 4: 生成变异偏好
        mutation_bias = self._generate_mutation_bias(
            improvements, regressions, overfitting_analysis, hof_analysis
        )

        # Step 5: 触发 alpha_mining (连续退化)
        triggered_alpha_mining = False
        if regressions:
            self._consecutive_regression_count += 1
        else:
            self._consecutive_regression_count = 0
        if self._consecutive_regression_count >= self.ALPHA_MINING_TRIGGER_COUNT:
            triggered_alpha_mining = self._trigger_alpha_mining()

        # 缓存反思结果
        reflection = {
            "generation": generation,
            "improvements": improvements,
            "regressions": regressions,
            "overfitting_analysis": overfitting_analysis,
            "hof_analysis": hof_analysis,
            "mutation_bias": mutation_bias,
            "triggered_alpha_mining": triggered_alpha_mining,
            "consecutive_regression_count": self._consecutive_regression_count,
        }
        self.reflection_history.append(reflection)
        # 缓存当前 stats 供下一代对比
        self._prev_stats = stats
        return reflection

    def _compare_stats(self, prev: Dict, cur: Dict) -> Tuple[List[str], List[str]]:
        """对比前后代 stats, 返回 (改进项, 退化项)

        比较维度: diversity_score / best_fitness / avg_fitness / sharpe / win_rate
        阈值: REGRESSION_THRESHOLD (5%) — 超过 5% 变化才记录
        """
        improvements, regressions = [], []
        for key in ["diversity_score", "best_fitness", "avg_fitness", "sharpe", "win_rate"]:
            if key in prev and key in cur:
                prev_v, cur_v = prev[key], cur[key]
                if cur_v > prev_v * (1 + self.REGRESSION_THRESHOLD):
                    improvements.append(f"{key}: {prev_v:.4f} → {cur_v:.4f}")
                elif cur_v < prev_v * (1 - self.REGRESSION_THRESHOLD):
                    regressions.append(f"{key}: {prev_v:.4f} → {cur_v:.4f}")
        return improvements, regressions

    def _analyze_overfitting(self, overfitted_agents: set) -> Dict:
        """分析过拟合 Agent (消费 FeatureExposure 特征归因)

        真闭环 (铁律 4): 消费 feature_exposure.get_last_report() 进行特征归因,
        识别过拟合 Agent 共同的特征模式 (供下一代变异规避)
        """
        if not overfitted_agents:
            return {"count": 0, "common_features": {}}
        # 消费 feature_exposure.get_last_report() 进行特征归因
        last_report = None
        if self.feature_exposure is not None:
            try:
                last_report = self.feature_exposure.get_last_report()
            except Exception as e:
                logger.debug("FeatureExposure get_last_report 失败: %s", e)

        # 聚合过拟合 Agent 的共同特征模式 (基于 last_report 的特征暴露分析)
        # FeatureExposure 报告结构: {features, regime, is_extreme, kelly_mult}
        common_features: Dict[str, Any] = {}
        if last_report and isinstance(last_report, dict):
            features = last_report.get("features", {})
            is_extreme = last_report.get("is_extreme", False)
            regime = last_report.get("regime", "unknown")
            # 识别极端特征 (过拟合 Agent 常见模式: 极端参数组合)
            # ERR-110: 特征暴露分析非单特征过滤, 聚合整体模式供变异参考
            extreme_features = {}
            for feat_name, feat_val in features.items():
                if not isinstance(feat_val, (int, float)):
                    continue
                # 识别极端值: RSI>70或<30, ATR>0.04, ADX>40 等已知极端阈值
                is_feat_extreme = False
                feat_lower = feat_name.lower()
                if feat_lower == "rsi" and (feat_val > 70 or feat_val < 30):
                    is_feat_extreme = True
                elif feat_lower == "atr_pct" and feat_val > 0.04:
                    is_feat_extreme = True
                elif feat_lower == "adx" and feat_val > 40:
                    is_feat_extreme = True
                if is_feat_extreme:
                    extreme_features[feat_name] = {
                        "value": round(float(feat_val), 4),
                        "threshold": "extreme",
                    }
            common_features = {
                "extreme_features": extreme_features,
                "regime_context": regime,
                "is_extreme_overall": is_extreme,
                "overfitted_count": len(overfitted_agents),
            }
            logger.debug(
                "ReflectorAgent: 过拟合分析 — %d 个极端特征, regime=%s, is_extreme=%s",
                len(extreme_features), regime, is_extreme,
            )
        return {
            "count": len(overfitted_agents),
            "last_feature_report": last_report,
            "common_features": common_features,
        }

    def _analyze_hof_stability(self, hall_of_fame: List) -> Dict:
        """分析 Hall of Fame 稳定性 (识别可复制模式)

        提取稳定策略的共同特征:
          - 高 consecutive_passes (连续通过代数) = 稳定性强
          - 低 sharpe_history 方差 = 收益一致性
          - 高 dsr/psr = 风险调整后收益稳健
        """
        if not hall_of_fame:
            return {"count": 0, "stable_patterns": []}

        # 提取稳定策略的共同模式
        stable_patterns: List[Dict[str, Any]] = []
        try:
            # 计算HoF统计
            sharpes = [h.get("sharpe", 0.0) for h in hall_of_fame if isinstance(h, dict)]
            consec_passes = [h.get("consecutive_passes", 0) for h in hall_of_fame if isinstance(h, dict)]
            dsrs = [h.get("dsr", 0.0) for h in hall_of_fame if isinstance(h, dict)]

            # 识别高稳定策略 (consecutive_passes >= 2 且 sharpe > 中位数)
            if sharpes:
                import statistics as _stats
                sharpe_median = _stats.median(sharpes) if len(sharpes) > 1 else sharpes[0]
                high_stability = [
                    h for h in hall_of_fame
                    if isinstance(h, dict)
                    and h.get("consecutive_passes", 0) >= 2
                    and h.get("sharpe", 0.0) >= sharpe_median
                ]
                # 提取稳定模式的共同特征
                if high_stability:
                    avg_sharpe = sum(h.get("sharpe", 0.0) for h in high_stability) / len(high_stability)
                    avg_consec = sum(h.get("consecutive_passes", 0) for h in high_stability) / len(high_stability)
                    avg_dsr = sum(h.get("dsr", 0.0) for h in high_stability) / len(high_stability)
                    # 计算 sharpe_history 一致性 (变异系数 = std/mean, 越低越稳定)
                    consistency_scores = []
                    for h in high_stability:
                        hist = h.get("sharpe_history", [])
                        if len(hist) >= 2:
                            mean_h = sum(hist) / len(hist)
                            if mean_h > 0:
                                var_h = sum((x - mean_h) ** 2 for x in hist) / len(hist)
                                std_h = var_h ** 0.5
                                cv = std_h / mean_h  # 变异系数
                                consistency_scores.append(cv)
                    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else None

                    stable_patterns.append({
                        "pattern_type": "high_stability",
                        "n_strategies": len(high_stability),
                        "avg_sharpe": round(avg_sharpe, 3),
                        "avg_consecutive_passes": round(avg_consec, 2),
                        "avg_dsr": round(avg_dsr, 3),
                        "avg_sharpe_cv": round(avg_consistency, 4) if avg_consistency is not None else None,
                        "replicable": True,  # 标记为可复制模式
                    })
                    logger.debug(
                        "ReflectorAgent: HoF稳定性 — %d 高稳定策略, avg_sharpe=%.2f, avg_consec=%.1f",
                        len(high_stability), avg_sharpe, avg_consec,
                    )
        except Exception as e:
            logger.debug("ReflectorAgent HoF稳定性分析失败: %s", e)

        return {
            "count": len(hall_of_fame),
            "stable_patterns": stable_patterns,
        }

    def _generate_mutation_bias(self, improvements, regressions,
                                  overfitting_analysis, hof_analysis) -> Dict:
        """生成变异偏好 (软偏好, 不强制)

        基于反思结果, 指导下一代变异方向:
          - strengthen: 强化改进方向
          - avoid: 规避退化/过拟合方向
          - explore: 探索新因子方向 (连续退化时)

        注: 这是软偏好, PopulationManager 可软参考, 不强制执行 (铁律 2)
        """
        bias = {"strengthen": [], "avoid": [], "explore": []}
        if improvements:
            bias["strengthen"].append("强化改进方向")
        if regressions:
            bias["avoid"].append("规避退化方向")
        if overfitting_analysis.get("count", 0) > 0:
            bias["avoid"].append("规避过拟合模式")
        if self._consecutive_regression_count >= self.ALPHA_MINING_TRIGGER_COUNT:
            bias["explore"].append("探索新因子方向")
        return bias

    def _trigger_alpha_mining(self) -> bool:
        """触发 alpha_mining 因子挖掘 (连续退化时)

        Phase K 修复: ReflectorAgent 无市场数据, 不能直接调用 discover_factors()
        - discover_factors(data, forward_returns) 需要两个必填参数
        - 改为设置 _alpha_mining_pending 标志, evolution_loop 在有数据时触发
        - 记录退化事件供 evolution_loop 消费
        """
        if self.alpha_mining is None:
            return False
        try:
            # Phase K: 标记需要因子挖掘, evolution_loop 在有数据时调用 discover_factors()
            self._alpha_mining_pending = True
            self._alpha_mining_pending_since = self._alpha_mining_pending_since if hasattr(self, '_alpha_mining_pending_since') and self._alpha_mining_pending_since else None
            logger.info(
                "[ReflectorAgent] alpha_mining 标记为 pending (连续%d代退化, 待 evolution_loop 有数据时触发 discover_factors)",
                self._consecutive_regression_count,
            )
            return True
        except Exception as e:
            logger.warning("[ReflectorAgent] alpha_mining 标记失败: %s", e)
            return False

    def reflect_loop_end(self, round_results: List) -> Dict:
        """Loop 结束反思总结

        整体进化趋势 + 教训应用效果 + 下一代建议

        Returns:
            {
                'summary': str,
                'evolution_trend': str,  # 'improving'/'stable'/'degrading'
                'best_strategy_root_cause': str,
                'worst_strategy_root_cause': str,
                'lesson_application_effect': Dict,
                'next_evolution_suggestions': List[str],
            }
        """
        if not self.enabled or not self.reflection_history:
            return self._empty_loop_reflection()

        # 整体进化趋势
        n_generations = len(self.reflection_history)
        total_improvements = sum(len(r.get("improvements", [])) for r in self.reflection_history)
        total_regressions = sum(len(r.get("regressions", [])) for r in self.reflection_history)
        if total_improvements > total_regressions * 1.5:
            trend = "improving"
        elif total_regressions > total_improvements * 1.5:
            trend = "degrading"
        else:
            trend = "stable"

        # 教训应用效果 (从 knowledge_base 检索 — 真闭环: 检索环节)
        lesson_effect = {"retrieval_count": 0, "applied_count": 0}
        if self.knowledge_base is not None:
            try:
                # 检索最近的教训
                lessons = self.knowledge_base.retrieve_lessons(
                    event={"type": "loop_end", "trend": trend},
                    top_k=5,
                )
                lesson_effect["retrieval_count"] = len(lessons) if lessons else 0
            except Exception as e:
                logger.debug("knowledge_base.retrieve_lessons 失败: %s", e)

        # 分析最佳/最差策略根因 (基于 round_results 数据)
        best_cause, worst_cause = self._analyze_round_results(round_results)

        summary = f"Loop 结束: {n_generations} 代进化, trend={trend}, improvements={total_improvements}, regressions={total_regressions}"

        return {
            "summary": summary,
            "evolution_trend": trend,
            "best_strategy_root_cause": best_cause,
            "worst_strategy_root_cause": worst_cause,
            "lesson_application_effect": lesson_effect,
            "next_evolution_suggestions": self._generate_next_suggestions(trend),
        }

    def _analyze_round_results(self, round_results: List) -> Tuple[str, str]:
        """分析 round_results 找出最佳/最差策略根因

        基于 EvolutionRoundResult 的 trades_executed/liquidations/warnings/elite_signals
        综合评分, 识别最佳和最差轮次的特征模式.

        评分公式: score = trades*1.0 + n_elite*2.0 - liquidations*5.0 - n_warnings*1.0
          - 高交易量 = 策略活跃 (正)
          - 多精英信号 = 策略质量高 (正, 权重2)
          - 高清算 = 风险失控 (负, 权重5)
          - 多警告 = 反过拟合警报 (负)
        """
        if not round_results:
            return ("N/A (无round_results数据)", "N/A (无round_results数据)")

        best_round = None
        worst_round = None
        best_score = float('-inf')
        worst_score = float('inf')

        for r in round_results:
            # 安全提取属性 (兼容 dataclass 对象和 dict 两种形式)
            def _get(key, default=0):
                if hasattr(r, key):
                    return getattr(r, key, default)
                elif isinstance(r, dict):
                    return r.get(key, default)
                return default

            trades = _get("trades_executed", 0)
            liquidations = _get("liquidations", 0)
            warnings_list = _get("warnings", [])
            n_warnings = len(warnings_list) if isinstance(warnings_list, (list, tuple)) else 0
            elite_signals = _get("elite_signals", [])
            n_elite = len(elite_signals) if isinstance(elite_signals, (list, tuple)) else 0
            round_num = _get("round_number", "?")
            gen = _get("generation", "?")

            # 综合评分: 交易活跃+精英信号多-清算少-警告少
            score = float(trades) * 1.0 + float(n_elite) * 2.0 - float(liquidations) * 5.0 - float(n_warnings) * 1.0

            if score > best_score:
                best_score = score
                best_round = (round_num, gen, trades, n_elite, liquidations, n_warnings)
            if score < worst_score:
                worst_score = score
                worst_round = (round_num, gen, trades, n_elite, liquidations, n_warnings)

        best_desc = "N/A"
        worst_desc = "N/A"
        if best_round is not None:
            rn, gn, tr, el, lq, wn = best_round
            # 根因描述: 基于数据特征推断成功因素
            success_factors = []
            if tr > 0:
                success_factors.append(f"trades={tr}活跃")
            if el > 0:
                success_factors.append(f"elite={el}高质量信号")
            if lq == 0:
                success_factors.append("零清算")
            if wn == 0:
                success_factors.append("无过拟合警告")
            cause_str = ", ".join(success_factors) if success_factors else "score最高"
            best_desc = f"Round{rn}(Gen{gn}): {cause_str} (score={best_score:.1f})"

        if worst_round is not None:
            rn, gn, tr, el, lq, wn = worst_round
            # 根因描述: 基于数据特征推断失败因素
            failure_factors = []
            if lq > 0:
                failure_factors.append(f"清算={lq}风险失控")
            if wn > 0:
                failure_factors.append(f"警告={wn}过拟合警报")
            if tr == 0:
                failure_factors.append("零交易(策略不活跃)")
            if el == 0:
                failure_factors.append("无精英信号")
            cause_str = ", ".join(failure_factors) if failure_factors else "score最低"
            worst_desc = f"Round{rn}(Gen{gn}): {cause_str} (score={worst_score:.1f})"

        return (best_desc, worst_desc)

    def _generate_next_suggestions(self, trend: str) -> List[str]:
        """生成下一代进化建议 (基于整体趋势)"""
        suggestions = []
        if trend == "degrading":
            suggestions.append("考虑重置 PopulationManager (退化严重)")
            suggestions.append("扩大变异率 (探索新方向)")
        elif trend == "improving":
            suggestions.append("减小变异率 (强化收敛)")
            suggestions.append("提升精英比例 (保留优势)")
        else:
            suggestions.append("保持当前参数 (稳定进化)")
        if self._consecutive_regression_count >= self.ALPHA_MINING_TRIGGER_COUNT:
            suggestions.append("已触发 alpha_mining, 关注新因子")
        return suggestions

    def _empty_reflection(self, generation: int) -> Dict:
        """空反思 (disabled 时)"""
        return {
            "generation": generation,
            "improvements": [],
            "regressions": [],
            "overfitting_analysis": {"count": 0, "common_features": {}},
            "mutation_bias": {"strengthen": [], "avoid": [], "explore": []},
            "triggered_alpha_mining": False,
            "consecutive_regression_count": 0,
        }

    def _empty_loop_reflection(self) -> Dict:
        """空 Loop 反思 (disabled 时)"""
        return {
            "summary": "ReflectorAgent disabled",
            "evolution_trend": "unknown",
            "best_strategy_root_cause": "N/A",
            "worst_strategy_root_cause": "N/A",
            "lesson_application_effect": {"retrieval_count": 0, "applied_count": 0},
            "next_evolution_suggestions": [],
        }
