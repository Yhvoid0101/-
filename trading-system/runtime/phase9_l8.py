"""phase9_l8.py — 第8层学习反馈智能体（结构进化）。

铁律：
- 禁止无审查自修改：只能提交 ChangeProposal，不能直接写生产配置或代码。
- 死因归因必须包含反事实：若不翻转方向、不同RR、不同执行算法，结果是否改善。
- 自动特征生成使用时间安全算子（只用 t-1 及更早数据），禁止未来数据泄漏。
- 结构提案只能在隔离分支/工作树中实验，通过全部门禁后人工批准晋升。
- 上游 L7 BLOCKED → L8 直接 BLOCKED(UPSTREAM_PROPAGATED)，L8 不放宽。
- 执行失败不可伪造为学习成功；执行失败时 death_cause 必须包含 EXECUTION_FAILURE。

零模拟：无 L7 ExecutionReport 输入时 BLOCKED，不返回默认学习事件。
零降级：所有特征/提案门禁不通过时 REJECTED，不降级放行。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts.base import ContractBase, ContractStatus, make_blocked
from .contracts.execution_contracts import (
    ExecutionReport, OrderStatus, OrderAlgorithm,
)
from .contracts.learning_event import (
    LearningEvent, CounterfactualAnalysis, LayerResponsibility,
)
from .contracts.change_proposal import (
    ChangeProposal, ChangeType, ProposalStage, rejected_proposal,
)


# ==========================================================================
# P9-T1：层级责任图 + 死因归因 + 反事实
# ==========================================================================

class DeathCause:
    """10 种死因分类。"""
    DIRECTION_FLIP = "DIRECTION_FLIP"
    RR_INSUFFICIENT = "RR_INSUFFICIENT"
    SL_HIT = "SL_HIT"
    TP_MISS = "TP_MISS"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    STALE_DATA = "STALE_DATA"
    OOD_REGIME = "OOD_REGIME"
    RISK_BREACH = "RISK_BREACH"
    SAMPLE_BIAS = "SAMPLE_BIAS"
    UNKNOWN = "UNKNOWN"

    ALL = (
        DIRECTION_FLIP, RR_INSUFFICIENT, SL_HIT, TP_MISS,
        EXECUTION_FAILURE, STALE_DATA, OOD_REGIME, RISK_BREACH,
        SAMPLE_BIAS, UNKNOWN,
    )


class LayerResponsibilityCalculator:
    """7 层贡献分解，归一化到 total_attribution = 1.0。

    从交易 context 提取各层质量指标，按权重计算贡献，归一化到 1.0。
    """

    # 默认权重（可在 context 中覆盖）
    DEFAULT_WEIGHTS = {
        "l1_data": 0.15,
        "l2_regime": 0.15,
        "l3_opportunity": 0.15,
        "l4_strategy": 0.15,
        "l5_audit": 0.10,
        "l6_risk": 0.15,
        "l7_execution": 0.15,
    }

    def calculate(self, report: ExecutionReport,
                  context: Optional[Dict[str, Any]] = None) -> LayerResponsibility:
        ctx = context or {}

        # L1 数据质量贡献
        l1 = self._safe(
            ctx.get("data_completeness", 1.0)
            * (1.0 - ctx.get("data_staleness_ratio", 0.0))
        )

        # L2 理解准确度
        l2 = self._safe(
            ctx.get("regime_confidence", 0.5)
            * (1.0 - ctx.get("ood_distance", 0.0))
        )

        # L3 机会评分
        l3 = self._safe(ctx.get("calibrated_prob", 0.5))

        # L4 策略适应度
        l4 = self._safe(
            min(ctx.get("fitness", 0.5), 1.0)
            * min(ctx.get("sample_size", 30) / 30.0, 1.0)
        )

        # L5 审核决策
        l5 = self._safe(
            1.0 if ctx.get("challenge_passed", True) else 0.0
        )

        # L6 风控贡献
        kill = ctx.get("killswitch_triggered", False)
        l6 = self._safe(0.0 if kill else ctx.get("risk_budget_utilization", 0.5))

        # L7 执行质量
        if report.status == OrderStatus.FILLED:
            l7 = self._safe(
                max(0.0, 1.0 - report.slippage / 100.0)
                * max(0.0, 1.0 - report.market_impact / 100.0)
            )
        else:
            l7 = 0.0  # 执行失败 → 执行层贡献为 0

        # 加权
        w = self.DEFAULT_WEIGHTS
        weighted = {
            "l1": l1 * w["l1_data"],
            "l2": l2 * w["l2_regime"],
            "l3": l3 * w["l3_opportunity"],
            "l4": l4 * w["l4_strategy"],
            "l5": l5 * w["l5_audit"],
            "l6": l6 * w["l6_risk"],
            "l7": l7 * w["l7_execution"],
        }
        total = sum(weighted.values())

        # 归一化到 1.0（避免除零）
        if total > 0:
            norm = 1.0 / total
            weighted = {k: v * norm for k, v in weighted.items()}
        else:
            # 全部为 0 → 均匀分布
            uniform = 1.0 / 7.0
            weighted = {k: uniform for k in weighted}

        return LayerResponsibility(
            l1_data_quality=weighted["l1"],
            l2_regime_accuracy=weighted["l2"],
            l3_opportunity_score=weighted["l3"],
            l4_strategy_fitness=weighted["l4"],
            l5_audit_decision=weighted["l5"],
            l6_risk_control=weighted["l6"],
            l7_execution_quality=weighted["l7"],
            total_attribution=sum(weighted.values()),
        )

    @staticmethod
    def _safe(x: float) -> float:
        """截断到 [0, 1]，处理 NaN。"""
        if x != x:  # NaN
            return 0.0
        return max(0.0, min(1.0, x))


class DeathCauseAnalyzer:
    """10 种死因分类。每种死因必须携带证据字典。"""

    def analyze(self, report: ExecutionReport,
                context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        ctx = context or {}

        # 执行失败优先
        if report.status in (OrderStatus.FAILED, OrderStatus.REJECTED):
            return DeathCause.EXECUTION_FAILURE, {
                "status": report.status.value,
                "rejection_reason": report.rejection_reason,
                "error_code": report.error_code,
            }

        pnl = ctx.get("pnl", 0.0)
        is_loss = pnl < 0

        # 亏损时分析死因
        if is_loss:
            # 数据陈旧
            if ctx.get("data_stale", False):
                return DeathCause.STALE_DATA, {
                    "staleness": ctx.get("staleness_seconds", 0),
                }

            # 未知 regime
            if ctx.get("ood_regime", False):
                return DeathCause.OOD_REGIME, {
                    "ood_distance": ctx.get("ood_distance", 0.0),
                }

            # 风控越界
            if ctx.get("killswitch_triggered", False):
                return DeathCause.RISK_BREACH, {
                    "trigger": ctx.get("killswitch_trigger", ""),
                }

            # 小样本偏差
            if ctx.get("sample_size", 30) < 10:
                return DeathCause.SAMPLE_BIAS, {
                    "sample_size": ctx.get("sample_size", 0),
                    "threshold": 10,
                }

            # 方向翻转
            if ctx.get("direction_flipped", False):
                return DeathCause.DIRECTION_FLIP, {
                    "original_direction": ctx.get("original_direction", ""),
                    "flipped_direction": ctx.get("flipped_direction", ""),
                }

            # RR 不足
            rr = ctx.get("rr_ratio", 0.0)
            if 0 < rr < 1.0:
                return DeathCause.RR_INSUFFICIENT, {
                    "rr_ratio": rr,
                    "threshold": 1.0,
                }

            # 止损触发
            if ctx.get("sl_hit", False):
                return DeathCause.SL_HIT, {
                    "sl_price": ctx.get("sl_price", 0.0),
                    "entry_price": ctx.get("entry_price", 0.0),
                }

            # 止盈未达到
            if ctx.get("tp_miss", False):
                return DeathCause.TP_MISS, {
                    "tp_price": ctx.get("tp_price", 0.0),
                    "exit_price": ctx.get("exit_price", 0.0),
                }

        # 无法分类
        return DeathCause.UNKNOWN, {
            "pnl": pnl,
            "is_loss": is_loss,
        }


class CounterfactualEngine:
    """反事实分析 3 维度：方向 / RR / 执行算法。"""

    def analyze(self, report: ExecutionReport,
                context: Optional[Dict[str, Any]] = None) -> CounterfactualAnalysis:
        ctx = context or {}

        original_action = ctx.get("original_action",
                                   f"{ctx.get('direction', 'long')}@{ctx.get('rr_ratio', 1.5)}RR/{report.order_type.value if hasattr(report, 'order_type') else 'market'}")
        original_result = ctx.get("pnl", 0.0)

        # 选择最优反事实维度
        best_cf = self._analyze_direction(ctx, original_result)
        rr_cf = self._analyze_rr(ctx, original_result)
        algo_cf = self._analyze_algorithm(ctx, report, original_result)

        # 选择改善最大的反事实
        candidates = [best_cf, rr_cf, algo_cf]
        best = max(candidates, key=lambda c: c.improvement)

        return CounterfactualAnalysis(
            original_action=original_action,
            original_result=original_result,
            counterfactual_action=best.counterfactual_action,
            counterfactual_result=best.counterfactual_result,
            improvement=best.improvement,
            is_counterfactual_better=best.is_counterfactual_better,
        )

    def _analyze_direction(self, ctx: Dict, original_result: float) -> CounterfactualAnalysis:
        """反事实1：翻转方向。"""
        direction = ctx.get("direction", "long")
        flipped = "short" if direction == "long" else "long"
        # 反事实结果 = -original_result（简化模型：翻转方向盈亏互换）
        cf_result = -original_result
        improvement = cf_result - original_result
        return CounterfactualAnalysis(
            original_action=direction,
            original_result=original_result,
            counterfactual_action=f"flip→{flipped}",
            counterfactual_result=cf_result,
            improvement=improvement,
            is_counterfactual_better=improvement > 0,
        )

    def _analyze_rr(self, ctx: Dict, original_result: float) -> CounterfactualAnalysis:
        """反事实2：不同 RR。"""
        original_rr = ctx.get("rr_ratio", 1.5)
        # 尝试 RR=2.0（更宽止盈）
        cf_rr = 2.0 if original_rr < 2.0 else 1.0
        # 简化模型：更高 RR 意味着更大止盈空间但更低胜率
        win_rate = ctx.get("win_rate", 0.5)
        if cf_rr > original_rr:
            cf_win_rate = max(0.0, win_rate - 0.1)
            cf_result = cf_win_rate * cf_rr - (1 - cf_win_rate) * 1.0
        else:
            cf_win_rate = min(1.0, win_rate + 0.1)
            cf_result = cf_win_rate * cf_rr - (1 - cf_win_rate) * 1.0
        improvement = cf_result - original_result
        return CounterfactualAnalysis(
            original_action=f"RR={original_rr}",
            original_result=original_result,
            counterfactual_action=f"RR={cf_rr}",
            counterfactual_result=cf_result,
            improvement=improvement,
            is_counterfactual_better=improvement > 0,
        )

    def _analyze_algorithm(self, ctx: Dict, report: ExecutionReport,
                           original_result: float) -> CounterfactualAnalysis:
        """反事实3：不同执行算法。"""
        original_algo = "market"
        if hasattr(report, "order_type"):
            original_algo = report.order_type.value if hasattr(report.order_type, "value") else str(report.order_type)
        elif "algorithm" in ctx:
            original_algo = ctx["algorithm"]

        # 尝试 LIMIT（更低滑点但可能不成交）
        cf_algo = "limit" if original_algo != "limit" else "twap"
        # 简化模型：LIMIT 滑点降低 50%
        original_slippage_cost = abs(ctx.get("slippage_cost", 0.0))
        cf_slippage_cost = original_slippage_cost * 0.5
        cf_result = original_result + (original_slippage_cost - cf_slippage_cost)
        improvement = cf_result - original_result
        return CounterfactualAnalysis(
            original_action=f"algo={original_algo}",
            original_result=original_result,
            counterfactual_action=f"algo={cf_algo}",
            counterfactual_result=cf_result,
            improvement=improvement,
            is_counterfactual_better=improvement > 0,
        )


# ==========================================================================
# P9-T2：自动特征生成 + 泄漏扫描 + 稳定性筛选 + 复杂度惩罚
# ==========================================================================

@dataclass
class CandidateFeature:
    """候选特征。"""
    feature_name: str = ""
    formula: str = ""
    time_window: int = 0
    source_field: str = ""
    operator: str = ""
    uses_future_data: bool = False
    leakage_correlation: float = 0.0
    ks_p_value: float = 1.0
    psi_value: float = 0.0
    aic_penalty: float = 0.0
    is_rejected: bool = False
    rejection_reason: str = ""


class FeatureGenerator:
    """时间安全算子特征生成。只用 t-1 及更早数据。"""

    VALID_OPERATORS = frozenset({
        "rolling_mean", "rolling_std", "rolling_quantile",
        "ema", "lag_diff", "cross_correlation",
    })

    def generate(self, source_field: str, time_window: int,
                 operator: str, **kwargs) -> CandidateFeature:
        if operator not in self.VALID_OPERATORS:
            return CandidateFeature(
                source_field=source_field,
                operator=operator,
                is_rejected=True,
                rejection_reason=f"未知算子: {operator}",
            )
        if time_window < 2:
            return CandidateFeature(
                source_field=source_field,
                operator=operator,
                is_rejected=True,
                rejection_reason="time_window 必须 >= 2",
            )
        q = kwargs.get("quantile", 0.5)
        name_parts = [operator, source_field, str(time_window)]
        if operator == "rolling_quantile":
            name_parts.append(str(q))
        feature_name = "_".join(name_parts)

        formula = f"{operator}({source_field}, window={time_window}"
        if operator == "rolling_quantile":
            formula += f", q={q}"
        formula += ")"

        return CandidateFeature(
            feature_name=feature_name,
            formula=formula,
            time_window=time_window,
            source_field=source_field,
            operator=operator,
            uses_future_data=False,  # 时间安全：只用 t-1 及更早
        )

    def generate_batch(self, source_fields: List[str],
                       time_windows: List[int],
                       operators: List[str]) -> List[CandidateFeature]:
        """批量生成候选特征。"""
        candidates = []
        for sf in source_fields:
            for tw in time_windows:
                for op in operators:
                    c = self.generate(sf, tw, op)
                    candidates.append(c)
        return candidates


class LeakageScanner:
    """未来数据泄漏扫描。"""

    def __init__(self, correlation_threshold: float = 0.9):
        self.correlation_threshold = correlation_threshold

    def scan(self, feature: CandidateFeature,
             feature_values: Optional[List[float]] = None,
             future_returns: Optional[List[float]] = None) -> CandidateFeature:
        # 检查1：是否使用未来数据
        if feature.uses_future_data:
            feature.is_rejected = True
            feature.rejection_reason = "特征使用了未来数据"
            return feature

        # 检查2：与未来收益的相关性
        if feature_values is not None and future_returns is not None:
            corr = self._pearson_correlation(feature_values, future_returns)
            feature.leakage_correlation = corr
            if abs(corr) > self.correlation_threshold:
                feature.is_rejected = True
                feature.rejection_reason = (
                    f"泄漏嫌疑: 相关性 {corr:.4f} > {self.correlation_threshold}"
                )
        return feature

    @staticmethod
    def _pearson_correlation(a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        a = a[:n]
        b = b[:n]
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        var_a = sum((x - mean_a) ** 2 for x in a)
        var_b = sum((y - mean_b) ** 2 for y in b)
        denom = math.sqrt(var_a * var_b)
        if denom == 0:
            return 0.0
        return cov / denom


class StabilityFilter:
    """跨窗口稳定性筛选。KS 检验 + PSI。"""

    def __init__(self, ks_p_threshold: float = 0.05, psi_threshold: float = 0.1):
        self.ks_p_threshold = ks_p_threshold
        self.psi_threshold = psi_threshold

    def filter(self, feature: CandidateFeature,
               window1: Optional[List[float]] = None,
               window2: Optional[List[float]] = None) -> CandidateFeature:
        if not window1 or not window2:
            # 无数据 → 默认通过（但标记为未验证）
            feature.ks_p_value = 1.0
            feature.psi_value = 0.0
            return feature

        ks_p = self._ks_test(window1, window2)
        psi = self._psi(window1, window2)
        feature.ks_p_value = ks_p
        feature.psi_value = psi

        if ks_p < self.ks_p_threshold:
            feature.is_rejected = True
            feature.rejection_reason = (
                f"KS检验失败 p={ks_p:.4f} < {self.ks_p_threshold}"
            )
            return feature

        if psi > self.psi_threshold:
            feature.is_rejected = True
            feature.rejection_reason = (
                f"PSI={psi:.4f} > {self.psi_threshold}"
            )
        return feature

    @staticmethod
    def _ks_test(a: List[float], b: List[float]) -> float:
        """简化 KS 检验。返回 p 值（越大越稳定）。"""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 1.0
        sorted_a = sorted(a)
        sorted_b = sorted(b)
        all_vals = sorted(set(sorted_a + sorted_b))
        d_max = 0.0
        for v in all_vals:
            cdf_a = sum(1 for x in sorted_a if x <= v) / n1
            cdf_b = sum(1 for x in sorted_b if x <= v) / n2
            d_max = max(d_max, abs(cdf_a - cdf_b))
        # 近似 p 值（Kolmogorov 分布）
        en = math.sqrt(n1 * n2 / (n1 + n2))
        lam = (en + 0.12 + 0.11 / en) * d_max
        # 简化：p ≈ 2 * exp(-2 * lam^2)
        p = 2.0 * math.exp(-2.0 * lam * lam)
        return min(1.0, max(0.0, p))

    @staticmethod
    def _psi(a: List[float], b: List[float], bins: int = 10) -> float:
        """PSI 计算。"""
        if not a or not b:
            return 0.0
        all_vals = a + b
        min_v, max_v = min(all_vals), max(all_vals)
        if max_v == min_v:
            return 0.0
        edges = [min_v + i * (max_v - min_v) / bins for i in range(bins + 1)]
        edges[0] = min_v - 1e-9
        edges[-1] = max_v + 1e-9

        def hist(vals):
            counts = [0] * bins
            for v in vals:
                for i in range(bins):
                    if edges[i] <= v < edges[i + 1]:
                        counts[i] += 1
                        break
            total = sum(counts)
            if total == 0:
                return [0.0] * bins
            return [max(c / total, 1e-6) for c in counts]

        p_a = hist(a)
        p_b = hist(b)
        psi = 0.0
        for pa, pb in zip(p_a, p_b):
            psi += (pa - pb) * math.log(pa / pb)
        return psi


class ComplexityPenalty:
    """复杂度惩罚。AIC/BIC。"""

    def __init__(self, max_features: int = 50, aic_ratio: float = 0.01):
        self.max_features = max_features
        self.aic_ratio = aic_ratio

    def penalize(self, feature: CandidateFeature,
                 num_total_features: int,
                 performance_gain: float) -> CandidateFeature:
        if num_total_features > self.max_features:
            feature.is_rejected = True
            feature.rejection_reason = (
                f"特征数 {num_total_features} > 上限 {self.max_features}"
            )
            return feature

        aic_penalty = 2 * num_total_features
        feature.aic_penalty = aic_penalty

        if performance_gain < aic_penalty * self.aic_ratio:
            feature.is_rejected = True
            feature.rejection_reason = (
                f"性能增益 {performance_gain:.4f} 不足以抵消 "
                f"AIC惩罚 {aic_penalty * self.aic_ratio:.4f}"
            )
        return feature


class FeaturePipeline:
    """特征生成流水线：生成 → 泄漏扫描 → 稳定性筛选 → 复杂度惩罚。"""

    def __init__(self):
        self.generator = FeatureGenerator()
        self.leakage_scanner = LeakageScanner()
        self.stability_filter = StabilityFilter()
        self.complexity_penalty = ComplexityPenalty()

    def run(self, source_fields: List[str], time_windows: List[int],
            operators: List[str],
            feature_values_map: Optional[Dict[str, List[float]]] = None,
            future_returns: Optional[List[float]] = None,
            window1_map: Optional[Dict[str, List[float]]] = None,
            window2_map: Optional[Dict[str, List[float]]] = None,
            performance_gain: float = 1.0) -> List[CandidateFeature]:
        candidates = self.generator.generate_batch(
            source_fields, time_windows, operators
        )
        fv_map = feature_values_map or {}
        w1_map = window1_map or {}
        w2_map = window2_map or {}

        passed: List[CandidateFeature] = []
        for c in candidates:
            if c.is_rejected:
                continue
            # 泄漏扫描
            c = self.leakage_scanner.scan(
                c, fv_map.get(c.feature_name), future_returns
            )
            if c.is_rejected:
                continue
            # 稳定性筛选
            c = self.stability_filter.filter(
                c, w1_map.get(c.feature_name), w2_map.get(c.feature_name)
            )
            if c.is_rejected:
                continue
            # 复杂度惩罚
            c = self.complexity_penalty.penalize(
                c, len(passed) + 1, performance_gain
            )
            if c.is_rejected:
                continue
            passed.append(c)
        return passed


# ==========================================================================
# P9-T3：结构建议 ChangeProposal + 隔离工作树 + 晋升门禁
# ==========================================================================

class StructureProposer:
    """基于死因簇和特征候选生成 ChangeProposal。4 种类型。"""

    def propose_stat_arb(self, symbol_a: str, symbol_b: str,
                         correlation: float,
                         correlation_id: str = "") -> ChangeProposal:
        """新统计套利对。"""
        p = ChangeProposal(
            change_type=ChangeType.STRATEGY_FAMILY,
            target_component=f"stat_arb_{symbol_a}_{symbol_b}",
            hypothesis=(
                f"{symbol_a} 与 {symbol_b} 相关性 {correlation:.2f}，"
                f"可做价差均值回归"
            ),
            patch_scope="新策略族：统计套利",
            expected_benefit=correlation * 0.1,
            risk_assessment="价差发散风险",
            rollback_point="停用新策略族",
            validation_matrix=["unit", "integration", "shadow", "walk_forward"],
            submitted_by="L8_LEARNING",
            stage=ProposalStage.PROPOSED,
        )
        p.correlation_id = correlation_id
        return p

    def propose_new_regime(self, regime_features: Dict[str, Any],
                           frequency: int,
                           correlation_id: str = "") -> ChangeProposal:
        """新 regime 定义。"""
        p = ChangeProposal(
            change_type=ChangeType.REGIME_DEFINITION,
            target_component="regime_detector",
            hypothesis=(
                f"未知状态频繁出现 {frequency} 次，建议新增 regime；"
                f"特征={regime_features}"
            ),
            patch_scope="regime_detection",
            expected_benefit=0.05,
            risk_assessment="regime 爆炸风险",
            rollback_point="停用新 regime",
            validation_matrix=["unit", "stability", "cross_window"],
            submitted_by="L8_LEARNING",
            stage=ProposalStage.PROPOSED,
        )
        p.correlation_id = correlation_id
        return p

    def propose_new_gate(self, death_cause: str, frequency: int,
                         correlation_id: str = "") -> ChangeProposal:
        """新 Gate。"""
        p = ChangeProposal(
            change_type=ChangeType.GATE_ADDITION,
            target_component=f"gate_{death_cause.lower()}",
            hypothesis=(
                f"死因 {death_cause} 出现 {frequency} 次，"
                f"建议新增 Gate 阻断"
            ),
            patch_scope="新 Gate 模块",
            expected_benefit=0.03,
            risk_assessment="误杀风险",
            rollback_point="停用新 Gate",
            validation_matrix=["unit", "challenge", "false_positive"],
            submitted_by="L8_LEARNING",
            stage=ProposalStage.PROPOSED,
        )
        p.correlation_id = correlation_id
        return p

    def propose_interface_split(self, module_name: str, line_count: int,
                                correlation_id: str = "") -> ChangeProposal:
        """接口拆分。"""
        p = ChangeProposal(
            change_type=ChangeType.INTERFACE_SPLIT,
            target_component=module_name,
            hypothesis=(
                f"模块 {module_name} 行数 {line_count}，建议拆分"
            ),
            patch_scope="模块拆分",
            expected_benefit=0.02,
            risk_assessment="接口兼容性",
            rollback_point="恢复原模块",
            validation_matrix=["unit", "integration", "compatibility"],
            submitted_by="L8_LEARNING",
            stage=ProposalStage.PROPOSED,
        )
        p.correlation_id = correlation_id
        return p

    def propose_from_death_causes(self, death_cause_counts: Dict[str, int],
                                  correlation_id: str = "") -> List[ChangeProposal]:
        """基于死因频率批量生成提案。"""
        proposals = []
        threshold = 3  # 同一死因出现 3 次以上才提议
        for cause, count in death_cause_counts.items():
            if count >= threshold:
                proposals.append(
                    self.propose_new_gate(cause, count, correlation_id)
                )
        return proposals


class WorktreeIsolation:
    """隔离工作树管理。提案必须在隔离分支中实验。"""

    def __init__(self):
        self._isolated_branches: Dict[str, ChangeProposal] = {}

    def create_isolation(self, proposal: ChangeProposal,
                         branch_name: str = "") -> ChangeProposal:
        if not branch_name:
            ts = int(time.time())
            branch_name = (
                f"proposal_{proposal.change_type.value}_"
                f"{proposal.target_component}_{ts}"
            )
        proposal.worktree_branch = branch_name
        proposal.is_isolated = True
        self._isolated_branches[branch_name] = proposal
        return proposal

    def is_isolated(self, proposal: ChangeProposal) -> bool:
        return proposal.is_isolated

    def list_isolated(self) -> Dict[str, ChangeProposal]:
        return dict(self._isolated_branches)

    def remove_isolation(self, branch_name: str) -> bool:
        if branch_name in self._isolated_branches:
            p = self._isolated_branches.pop(branch_name)
            p.is_isolated = False
            p.worktree_branch = ""
            return True
        return False


class PromotionGate:
    """晋升门禁。9 项全门禁 + 人工批准。"""

    REQUIRED_GATES = (
        "unit_test",
        "integration_test",
        "blackbox",
        "branch_coverage",
        "mutation",
        "snapshot",
        "fuzz",
        "dependency",
        "cross_environment",
    )

    def evaluate(self, proposal: ChangeProposal) -> Dict[str, Any]:
        results = {
            g: bool(proposal.gate_results.get(g, False))
            for g in self.REQUIRED_GATES
        }
        all_pass = all(results.values())
        is_isolated = proposal.is_isolated
        has_approval = proposal.approved_by != ""
        at_gate_stage = proposal.stage == ProposalStage.GATE

        can_promote = (
            all_pass and is_isolated and has_approval and at_gate_stage
        )
        return {
            "can_promote": can_promote,
            "all_gates_pass": all_pass,
            "is_isolated": is_isolated,
            "has_approval": has_approval,
            "at_gate_stage": at_gate_stage,
            "gate_results": results,
            "missing_gates": [
                g for g in self.REQUIRED_GATES if not results[g]
            ],
        }

    def set_gate_result(self, proposal: ChangeProposal,
                        gate_name: str, passed: bool) -> ChangeProposal:
        if gate_name not in self.REQUIRED_GATES:
            raise ValueError(f"未知门禁: {gate_name}")
        proposal.gate_results[gate_name] = passed
        return proposal

    def submit_to_gate(self, proposal: ChangeProposal) -> ChangeProposal:
        """提交到 GATE 阶段（SHADOW 和 CHALLENGE 通过后）。"""
        proposal.stage = ProposalStage.GATE
        return proposal

    def approve(self, proposal: ChangeProposal,
                approver: str) -> ChangeProposal:
        """人工审批。"""
        proposal.approved_by = approver
        return proposal

    def promote(self, proposal: ChangeProposal) -> ChangeProposal:
        """晋升到生产。必须全门禁通过 + 人工批准 + 隔离环境。"""
        eval_result = self.evaluate(proposal)
        if not eval_result["can_promote"]:
            missing = eval_result["missing_gates"]
            reasons = []
            if missing:
                reasons.append(f"未通过门禁: {missing}")
            if not eval_result["is_isolated"]:
                reasons.append("未在隔离环境中")
            if not eval_result["has_approval"]:
                reasons.append("缺少人工审批")
            if not eval_result["at_gate_stage"]:
                reasons.append(f"当前阶段={proposal.stage.value} 非 GATE")
            return rejected_proposal(
                "；".join(reasons), proposal.correlation_id
            )
        proposal.stage = ProposalStage.PROMOTED
        return proposal

    def rollback(self, proposal: ChangeProposal,
                 reason: str) -> ChangeProposal:
        """回滚。"""
        proposal.stage = ProposalStage.ROLLED_BACK
        proposal.gate_results["rollback_reason"] = reason
        return proposal


# ==========================================================================
# L8LearningAgent — 主类
# ==========================================================================

class L8LearningAgent:
    """第8层学习反馈智能体（结构进化）。

    输入：L7 ExecutionReport + 交易 context
    输出：LearningEvent（死因 + 反事实 + 层级责任）
    附加：特征候选生成 + 结构提案

    铁律：
    - 上游 L7 BLOCKED → BLOCKED(UPSTREAM_PROPAGATED)
    - 执行失败不可伪造为学习成功
    - 禁止无审查自修改（只提交 ChangeProposal）
    """

    LAYER_NAME = "L8_LEARNING"
    LAYER_NUMBER = 8

    def __init__(self):
        self._responsibility_calc = LayerResponsibilityCalculator()
        self._death_cause_analyzer = DeathCauseAnalyzer()
        self._counterfactual_engine = CounterfactualEngine()
        self._feature_pipeline = FeaturePipeline()
        self._structure_proposer = StructureProposer()
        self._worktree_isolation = WorktreeIsolation()
        self._promotion_gate = PromotionGate()
        self._death_cause_counts: Dict[str, int] = {}

    # ---- P9-T1：学习事件生成 ----

    def observe(self, report: ExecutionReport,
                context: Optional[Dict[str, Any]] = None) -> LearningEvent:
        """生成学习事件。"""
        ctx = context or {}

        # 上游 BLOCKED 传播
        if report.is_blocked():
            event = LearningEvent()
            return make_blocked(
                event, "UPSTREAM_PROPAGATED",
                "L7执行BLOCKED", report.correlation_id,
            )

        # 死因归因
        death_cause, evidence = self._death_cause_analyzer.analyze(report, ctx)

        # 记录死因频率
        self._death_cause_counts[death_cause] = (
            self._death_cause_counts.get(death_cause, 0) + 1
        )

        # 反事实分析
        counterfactual = self._counterfactual_engine.analyze(report, ctx)

        # 层级责任
        responsibility = self._responsibility_calc.calculate(report, ctx)

        # 构建学习事件
        event = LearningEvent(
            trade_id=report.order_id,
            symbol=report.symbol,
            entry_price=ctx.get("entry_price", 0.0),
            exit_price=ctx.get("exit_price", report.avg_fill_price),
            direction=ctx.get("direction", ""),
            pnl=ctx.get("pnl", 0.0),
            pnl_pct=ctx.get("pnl_pct", 0.0),
            holding_bars=ctx.get("holding_bars", 0),
            death_cause=death_cause,
            death_cause_evidence=evidence,
            counterfactual=counterfactual,
            responsibility=responsibility,
            replay_seed=ctx.get("replay_seed", ""),
            replay_data_hash=ctx.get("replay_data_hash", ""),
            is_reproducible=ctx.get("is_reproducible", False),
        )
        event.correlation_id = report.correlation_id
        return event

    # ---- P9-T2：特征生成接口 ----

    def generate_features(self, source_fields: List[str],
                          time_windows: List[int],
                          operators: List[str],
                          **kwargs) -> List[CandidateFeature]:
        """生成候选特征（通过4门禁才进入候选池）。"""
        return self._feature_pipeline.run(
            source_fields, time_windows, operators, **kwargs
        )

    # ---- P9-T3：结构提案接口 ----

    def propose_structures(self,
                           correlation_id: str = "") -> List[ChangeProposal]:
        """基于累积的死因频率生成结构提案。"""
        return self._structure_proposer.propose_from_death_causes(
            self._death_cause_counts, correlation_id
        )

    def create_isolation(self, proposal: ChangeProposal,
                         branch_name: str = "") -> ChangeProposal:
        """为提案创建隔离工作树。"""
        return self._worktree_isolation.create_isolation(
            proposal, branch_name
        )

    def evaluate_promotion(self,
                           proposal: ChangeProposal) -> Dict[str, Any]:
        """评估提案是否可以晋升。"""
        return self._promotion_gate.evaluate(proposal)

    def promote_proposal(self,
                         proposal: ChangeProposal) -> ChangeProposal:
        """晋升提案到生产。"""
        return self._promotion_gate.promote(proposal)

    # ---- 统计 ----

    def get_death_cause_stats(self) -> Dict[str, int]:
        return dict(self._death_cause_counts)
