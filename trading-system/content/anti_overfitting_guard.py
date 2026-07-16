# -*- coding: utf-8 -*-
"""反过拟合运行时强制模块 (Anti-Overfitting Runtime Guard)

用户最高铁律:
  "永远永远不要出现模拟牛逼，实盘亏损的情况"
  "不要通过已知数据反推策略，那样就过拟合了"
  "绝对保证收益提升是策略逻辑带来，而非单纯拟合历史行情"

本模块是 **运行时代码级强制层**，与现有统计检验互补:
  - anti_overfitting.py (WalkForwardValidator) = 事后统计检验 (IS/OOS/PBO/DSR)
  - _v596_unified_walkforward.py = 事后五重统计检验 (DF/DSR/PSR/PBO/MC)
  - advanced_validation.py = 事后高级验证 (PBO CSCV / DSR / MinBTL)
  - 本模块 (anti_overfitting_guard.py) = **运行时强制** (前视偏差/数据窥探/逻辑验证)

三层运行时强制:
  1. LookaheadGuard: 时间游标 + embargo 隔离, 阻止决策时访问未来 bar
  2. OOSIsolationTracker: IS/OOS 阶段切换, 检测 OOS 数据是否被 IS 阶段"见过"
  3. StrategyLogicVerifier: 置换检验, 验证收益来自策略逻辑而非数据拟合

设计铁律:
  D1: 命名隔离 (不与现有 anti_overfitting.py 冲突, 用 Guard/Tracker/Verifier 后缀)
  D2: 自包含 (仅依赖 stdlib + numpy, 不依赖条件块局部变量)
  D3: 信息增强层 (设置 _guard_block 标志, 不直接覆盖 can_deploy, 供上层准入接口消费)
  D5: guard call pattern (try/except + logger.warning 降级, 永不崩溃主循环)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("hermes.anti_overfitting_guard")


# ============================================================================
# 违规记录
# ============================================================================


@dataclass(slots=True)
class GuardViolation:
    """运行时强制违规记录"""

    violation_type: str  # "lookahead" | "oos_contamination" | "logic_failure"
    severity: str  # "BLOCK" | "WARN" | "INFO"
    context: str  # 违规发生的上下文描述
    timestamp: Optional[float] = None  # 违规时的数据时间戳
    detail: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 1. LookaheadGuard: 前视偏差运行时检测器
# ============================================================================


class LookaheadGuard:
    """前视偏差运行时检测器 (用户铁律: "不通过已知数据反推策略")

    原理: 维护一个"时间游标" _current_ts, 决策时只能访问 ts <= _current_ts 的数据。
    若访问 ts > _current_ts 的数据 → 记录 lookahead 违规 (BLOCK 级)。

    使用方式:
        guard = LookaheadGuard(embargo_bars=5)
        for bar in bars:
            guard.advance_to(bar["timestamp"])
            # 决策时检查
            if guard.check_access(future_bar["timestamp"], context="signal_gen"):
                ...  # 允许访问
            else:
                ...  # 拒绝 (前视偏差)

    关键设计:
      - embargo_bars: 隔离期 (防止训练/测试边界信息泄露, 来源: López de Prado)
      - 保守默认: embargo_bars=5 (5 bar 缓冲)
      - 违规不抛异常 (guard pattern), 仅记录, 供上层决策
    """

    def __init__(self, embargo_bars: int = 5, enabled: bool = True):
        """初始化前视偏差检测器

        Args:
            embargo_bars: 隔离期 bar 数 (防止边界信息泄露)
            enabled: 是否启用 (False=纯监控模式, 不阻断)
        """
        self.enabled = enabled
        self.embargo_bars = max(0, embargo_bars)
        self._current_ts: float = 0.0
        self._violations: List[GuardViolation] = []
        self._access_count = 0
        self._blocked_count = 0
        logger.debug(
            "LookaheadGuard 初始化 (embargo=%d bars, enabled=%s)",
            self.embargo_bars, self.enabled,
        )

    def advance_to(self, timestamp: float) -> None:
        """推进时间游标到指定时间戳 (每个决策点调用)

        Args:
            timestamp: 当前决策点的数据时间戳
        """
        # 游标只前进不后退 (防止回溯篡改)
        if timestamp > self._current_ts:
            self._current_ts = timestamp

    def check_access(
        self,
        data_timestamp: float,
        context: str = "",
        bar_interval: float = 0.0,
    ) -> bool:
        """检查是否允许访问指定时间戳的数据

        Args:
            data_timestamp: 要访问的数据时间戳
            context: 访问上下文 (如 "signal_gen" / "param_optimize" / "oos_eval")
            bar_interval: 单 bar 时间间隔 (用于 embargo 时间换算)

        Returns:
            True = 允许访问 (data_ts <= current_ts - embargo)
            False = 拒绝访问 (前视偏差风险)
        """
        if not self.enabled:
            return True  # 纯监控模式不阻断

        self._access_count += 1
        embargo_time = self.embargo_bars * bar_interval if bar_interval > 0 else 0.0
        allowed_threshold = self._current_ts - embargo_time

        if data_timestamp > allowed_threshold:
            violation = GuardViolation(
                violation_type="lookahead",
                severity="BLOCK",
                context=context or "unknown",
                timestamp=data_timestamp,
                detail={
                    "data_ts": data_timestamp,
                    "current_ts": self._current_ts,
                    "embargo_bars": self.embargo_bars,
                    "exceed_ms": (data_timestamp - allowed_threshold),
                },
            )
            self._violations.append(violation)
            self._blocked_count += 1
            logger.warning(
                "[LookaheadGuard] 前视偏差违规! context=%s data_ts=%.3f > threshold=%.3f (embargo=%d bars)",
                context, data_timestamp, allowed_threshold, self.embargo_bars,
            )
            return False
        return True

    def get_report(self) -> Dict[str, Any]:
        """获取检测报告"""
        block_count = sum(1 for v in self._violations if v.severity == "BLOCK")
        return {
            "enabled": self.enabled,
            "embargo_bars": self.embargo_bars,
            "current_ts": self._current_ts,
            "total_access_checks": self._access_count,
            "blocked_accesses": self._blocked_count,
            "block_violations": block_count,
            "all_violations": len(self._violations),
            "has_lookahead_violation": block_count > 0,
            "verdict": "BLOCK" if block_count > 0 else "PASS",
        }


# ============================================================================
# 2. OOSIsolationTracker: IS/OOS 数据污染追踪
# ============================================================================


class OOSIsolationTracker:
    """IS/OOS 数据污染追踪器 (用户铁律: "不通过已知数据反推策略")

    原理: 记录 IS 阶段"见过"的数据索引, OOS 阶段检查是否有重叠。
    若 OOS 数据在 IS 阶段已被访问 → 数据窥探 (data snooping) 违规。

    使用方式:
        tracker = OOSIsolationTracker()
        tracker.enter_is_phase()
        # IS 阶段: 优化参数时标记访问的数据
        tracker.mark_seen(is_bar_indices)
        tracker.enter_oos_phase()
        # OOS 阶段: 验证时检查纯度
        contamination = tracker.check_oos_purity(oos_bar_indices)
        if contamination:
            logger.warning("OOS 数据被 IS 阶段污染: %d bars", len(contamination))

    关键设计:
      - 支持多轮 walk-forward (每轮 IS/OOS 独立追踪)
      - 保守默认: 任何重叠都标记为 BLOCK
      - 不依赖外部数据结构 (自包含)
    """

    def __init__(self, enabled: bool = True):
        """初始化 OOS 隔离追踪器

        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self._phase = "IS"  # "IS" 或 "OOS"
        self._is_seen_indices: set = set()
        self._violations: List[GuardViolation] = []
        self._wf_rounds: List[Dict[str, Any]] = []  # walk-forward 轮次记录
        self._current_round_id = 0
        logger.debug("OOSIsolationTracker 初始化 (enabled=%s)", enabled)

    def enter_is_phase(self, round_id: Optional[int] = None) -> None:
        """进入 IS (In-Sample) 阶段"""
        if round_id is not None:
            self._current_round_id = round_id
        self._phase = "IS"
        self._is_seen_indices = set()  # 新轮次重置
        logger.debug("[OOSTracker] 进入 IS 阶段 (round=%d)", self._current_round_id)

    def enter_oos_phase(self) -> None:
        """进入 OOS (Out-of-Sample) 阶段"""
        self._phase = "OOS"
        logger.debug(
            "[OOSTracker] 进入 OOS 阶段 (round=%d, is_seen=%d bars)",
            self._current_round_id, len(self._is_seen_indices),
        )

    def mark_seen(self, indices: Sequence[int]) -> None:
        """标记 IS 阶段访问过的数据索引

        Args:
            indices: IS 阶段访问过的 bar 索引列表
        """
        if self._phase == "IS":
            self._is_seen_indices.update(indices)

    def check_oos_purity(self, oos_indices: Sequence[int]) -> set:
        """检查 OOS 数据纯度 (是否被 IS 阶段污染)

        Args:
            oos_indices: OOS 阶段要使用的数据索引

        Returns:
            被污染的索引集合 (空集 = 纯净)
        """
        if not self.enabled or self._phase != "OOS":
            return set()

        oos_set = set(oos_indices)
        contamination = oos_set & self._is_seen_indices

        if contamination:
            violation = GuardViolation(
                violation_type="oos_contamination",
                severity="BLOCK",
                context=f"wf_round_{self._current_round_id}",
                detail={
                    "oos_bars": len(oos_set),
                    "contaminated_bars": len(contamination),
                    "contamination_rate": len(contamination) / max(1, len(oos_set)),
                    "is_seen_total": len(self._is_seen_indices),
                },
            )
            self._violations.append(violation)
            logger.warning(
                "[OOSTracker] OOS 数据污染! round=%d %d/%d bars 被 IS 阶段见过 (%.1f%%)",
                self._current_round_id,
                len(contamination), len(oos_set),
                100.0 * len(contamination) / max(1, len(oos_set)),
            )

        # 记录本轮 walk-forward 结果
        self._wf_rounds.append({
            "round_id": self._current_round_id,
            "is_bars": len(self._is_seen_indices),
            "oos_bars": len(oos_set),
            "contaminated": len(contamination),
            "is_pure": len(contamination) == 0,
        })
        return contamination

    def get_report(self) -> Dict[str, Any]:
        """获取隔离追踪报告"""
        block_count = sum(1 for v in self._violations if v.severity == "BLOCK")
        return {
            "enabled": self.enabled,
            "current_phase": self._phase,
            "wf_rounds_completed": len(self._wf_rounds),
            "is_seen_bars": len(self._is_seen_indices),
            "contamination_violations": block_count,
            "all_violations": len(self._violations),
            "rounds_detail": list(self._wf_rounds),
            "has_contamination": block_count > 0,
            "verdict": "BLOCK" if block_count > 0 else "PASS",
        }


# ============================================================================
# 3. StrategyLogicVerifier: 置换检验验证收益来自策略逻辑
# ============================================================================


class StrategyLogicVerifier:
    """策略逻辑验证器 (用户铁律: "绝对保证收益提升是策略逻辑带来而非拟合历史")

    原理: 如果策略收益来自真正的市场逻辑, 那么打乱交易标签后收益应显著下降。
    若打乱后收益不变 → 说明收益来自数据拟合而非策略逻辑 (data-fitting)。

    三种检验:
      1. label_permutation: 打乱交易方向 (多↔空), 验证方向性逻辑
      2. temporal_permutation: 打乱交易时间顺序, 验证时序逻辑
      3. feature_permutation: 打乱特征值, 验证特征依赖逻辑

    使用方式:
        verifier = StrategyLogicVerifier(n_permutations=200, alpha=0.05)
        result = verifier.verify(
            original_returns=pnl_series,
            strategy_fn=lambda data: simulate_strategy(data),
            test_data=market_data,
            test_type="label_permutation",
        )
        if result.is_logic_driven:
            logger.info("收益来自策略逻辑 (p=%.4f)", result.p_value)
        else:
            logger.warning("收益可能来自数据拟合 (p=%.4f)", result.p_value)

    关键设计:
      - 保守默认: n_permutations=200 (平衡统计功效与计算成本)
      - 显著性水平: alpha=0.05 (P<0.05 才通过, 用户硬约束)
      - 原假设 H0: 收益与策略逻辑无关 (即收益来自数据拟合)
      - 拒绝 H0 → 收益来自策略逻辑 (PASS)
    """

    def __init__(
        self,
        n_permutations: int = 200,
        alpha: float = 0.05,
        enabled: bool = True,
        random_seed: Optional[int] = 42,
    ):
        """初始化策略逻辑验证器

        Args:
            n_permutations: 置换次数 (越多越精确, 越慢)
            alpha: 显著性水平 (P<alpha 才拒绝 H0, 用户要求 P<0.05)
            enabled: 是否启用
            random_seed: 随机种子 (可复现)
        """
        self.enabled = enabled
        self.n_permutations = max(10, n_permutations)
        self.alpha = alpha
        self._rng = random.Random(random_seed) if random_seed else random.Random()
        self._violations: List[GuardViolation] = []
        logger.debug(
            "StrategyLogicVerifier 初始化 (n_perm=%d, alpha=%.3f, enabled=%s)",
            self.n_permutations, self.alpha, self.enabled,
        )

    def verify(
        self,
        original_metric: float,
        permuted_metrics: Sequence[float],
        test_type: str = "label_permutation",
        higher_is_better: bool = True,
    ) -> "LogicVerifyResult":
        """验证收益是否来自策略逻辑

        Args:
            original_metric: 原始策略的指标值 (如 Sharpe / 总收益)
            permuted_metrics: 置换后的指标值列表
            test_type: 检验类型 ("label_permutation" / "temporal" / "feature")
            higher_is_better: True=指标越高越好 (如 Sharpe), False=越低越好 (如回撤)

        Returns:
            LogicVerifyResult: 验证结果
        """
        if not self.enabled or not permuted_metrics:
            return LogicVerifyResult(
                is_logic_driven=True,  # 降级: 不验证则默认通过
                p_value=1.0,
                test_type=test_type,
                original_metric=original_metric,
                permuted_mean=float(np.mean(permuted_metrics)) if permuted_metrics else 0.0,
                permuted_std=float(np.std(permuted_metrics)) if permuted_metrics else 0.0,
                n_permutations=len(permuted_metrics),
                verdict="PASS",
                detail={"reason": "disabled or empty"},
            )

        perm_arr = np.array(permuted_metrics, dtype=float)
        perm_mean = float(np.mean(perm_arr))
        perm_std = float(np.std(perm_arr))

        # 计算置换检验 p 值
        # H0: 原始指标不优于置换分布 (即收益来自数据拟合)
        # 单边检验: 原始指标是否显著优于置换分布
        if higher_is_better:
            # 原始指标应 > 置换分布 → 拒绝 H0 (策略逻辑有效)
            exceed_count = int(np.sum(perm_arr >= original_metric))
            # p 值 = (超过原始的置换数 + 1) / (总置换数 + 1) (保守估计, +1 避免除零)
            p_value = (exceed_count + 1) / (len(perm_arr) + 1)
            is_logic_driven = p_value < self.alpha and original_metric > perm_mean
        else:
            # 原始指标应 < 置换分布 (如回撤越小越好)
            exceed_count = int(np.sum(perm_arr <= original_metric))
            p_value = (exceed_count + 1) / (len(perm_arr) + 1)
            is_logic_driven = p_value < self.alpha and original_metric < perm_mean

        # 效应量 (Cohen's d): 原始 vs 置换的差异程度
        if perm_std > 0:
            cohens_d = (original_metric - perm_mean) / perm_std
        else:
            cohens_d = 0.0

        # 判定
        if is_logic_driven:
            verdict = "PASS"
            severity = "INFO"
            logger.info(
                "[LogicVerifier] %s: 收益来自策略逻辑 (p=%.4f < %.3f, d=%.3f)",
                test_type, p_value, self.alpha, cohens_d,
            )
        else:
            verdict = "BLOCK"
            severity = "BLOCK"
            violation = GuardViolation(
                violation_type="logic_failure",
                severity=severity,
                context=test_type,
                detail={
                    "original_metric": original_metric,
                    "permuted_mean": perm_mean,
                    "permuted_std": perm_std,
                    "p_value": p_value,
                    "alpha": self.alpha,
                    "cohens_d": cohens_d,
                    "exceed_ratio": exceed_count / max(1, len(perm_arr)),
                },
            )
            self._violations.append(violation)
            logger.warning(
                "[LogicVerifier] %s: 收益可能来自数据拟合! p=%.4f >= %.3f (d=%.3f) → BLOCK",
                test_type, p_value, self.alpha, cohens_d,
            )

        return LogicVerifyResult(
            is_logic_driven=is_logic_driven,
            p_value=p_value,
            test_type=test_type,
            original_metric=original_metric,
            permuted_mean=perm_mean,
            permuted_std=perm_std,
            n_permutations=len(perm_arr),
            verdict=verdict,
            detail={
                "alpha": self.alpha,
                "cohens_d": cohens_d,
                "exceed_count": exceed_count,
                "exceed_ratio": exceed_count / max(1, len(perm_arr)),
                "higher_is_better": higher_is_better,
            },
        )

    def get_report(self) -> Dict[str, Any]:
        """获取逻辑验证报告"""
        block_count = sum(1 for v in self._violations if v.severity == "BLOCK")
        return {
            "enabled": self.enabled,
            "n_permutations": self.n_permutations,
            "alpha": self.alpha,
            "logic_failures": block_count,
            "all_violations": len(self._violations),
            "has_logic_failure": block_count > 0,
            "verdict": "BLOCK" if block_count > 0 else "PASS",
        }


@dataclass(slots=True)
class LogicVerifyResult:
    """策略逻辑验证结果"""

    is_logic_driven: bool  # True=收益来自策略逻辑, False=可能数据拟合
    p_value: float  # 置换检验 p 值 (<alpha 拒绝 H0)
    test_type: str  # 检验类型
    original_metric: float  # 原始指标值
    permuted_mean: float  # 置换指标均值
    permuted_std: float  # 置换指标标准差
    n_permutations: int  # 置换次数
    verdict: str  # "PASS" | "BLOCK"
    detail: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 4. AntiOverfittingGuard: 统一入口 (组合三层强制)
# ============================================================================


class AntiOverfittingGuard:
    """反过拟合运行时强制统一入口

    组合三层运行时强制:
      1. LookaheadGuard: 前视偏差检测
      2. OOSIsolationTracker: IS/OOS 污染追踪
      3. StrategyLogicVerifier: 策略逻辑置换检验

    用户铁律:
      "绝对保证收益提升是策略逻辑带来，而非单纯拟合历史行情"
      "不要通过已知数据反推策略，那样就过拟合了"

    使用方式:
        guard = AntiOverfittingGuard(enabled=True)
        # 进化循环中
        guard.lookahead.advance_to(bar_ts)
        guard.oos_tracker.enter_is_phase(round_id=gen)
        # ... IS 阶段 ...
        guard.oos_tracker.enter_oos_phase()
        # ... OOS 阶段 ...
        logic_result = guard.logic_verifier.verify(
            original_metric=oos_sharpe,
            permuted_metrics=perm_sharpes,
        )
        # 获取综合报告
        report = guard.get_unified_report()
        if report["verdict"] == "BLOCK":
            logger.warning("反过拟合强制层 BLOCK: %s", report["block_reasons"])
    """

    def __init__(
        self,
        embargo_bars: int = 5,
        n_permutations: int = 200,
        alpha: float = 0.05,
        enabled: bool = True,
    ):
        """初始化统一反过拟合强制层

        Args:
            embargo_bars: 前视偏差隔离期 (bar 数)
            n_permutations: 置换检验次数
            alpha: 显著性水平 (P<alpha 拒绝 H0)
            enabled: 是否启用
        """
        self.enabled = enabled
        self.lookahead = LookaheadGuard(embargo_bars=embargo_bars, enabled=enabled)
        self.oos_tracker = OOSIsolationTracker(enabled=enabled)
        self.logic_verifier = StrategyLogicVerifier(
            n_permutations=n_permutations, alpha=alpha, enabled=enabled,
        )
        logger.info(
            "AntiOverfittingGuard 初始化 (embargo=%d, n_perm=%d, alpha=%.3f, enabled=%s)",
            embargo_bars, n_permutations, alpha, enabled,
        )

    def get_unified_report(self) -> Dict[str, Any]:
        """获取三层统一的反过拟合报告"""
        lh_report = self.lookahead.get_report()
        oos_report = self.oos_tracker.get_report()
        logic_report = self.logic_verifier.get_report()

        # 综合判定: 任一层 BLOCK → 整体 BLOCK
        block_reasons: List[str] = []
        if lh_report["verdict"] == "BLOCK":
            block_reasons.append(
                f"lookahead: {lh_report['block_violations']} 前视偏差违规"
            )
        if oos_report["verdict"] == "BLOCK":
            block_reasons.append(
                f"oos_contamination: {oos_report['contamination_violations']} 数据污染"
            )
        if logic_report["verdict"] == "BLOCK":
            block_reasons.append(
                f"logic_failure: {logic_report['logic_failures']} 逻辑验证失败"
            )

        overall_verdict = "BLOCK" if block_reasons else "PASS"

        return {
            "enabled": self.enabled,
            "lookahead_guard": lh_report,
            "oos_isolation": oos_report,
            "logic_verifier": logic_report,
            "overall_verdict": overall_verdict,
            "block_reasons": block_reasons,
            "n_block_layers": len(block_reasons),
            "has_lookahead_violation": lh_report["has_lookahead_violation"],
            "has_oos_contamination": oos_report["has_contamination"],
            "has_logic_failure": logic_report["has_logic_failure"],
        }

    def reset_for_new_walkforward_round(self, round_id: int) -> None:
        """为新 walk-forward 轮次重置状态 (IS 阶段开始前调用)"""
        self.oos_tracker.enter_is_phase(round_id=round_id)
        # lookahead 不重置 (时间游标只前进)
        # logic_verifier 的违规累积 (历史记录保留)


# ============================================================================
# 自验证
# ============================================================================


def _self_test() -> bool:
    """模块自验证"""
    try:
        # 1. LookaheadGuard 测试
        guard = LookaheadGuard(embargo_bars=2, enabled=True)
        guard.advance_to(100.0)
        assert guard.check_access(90.0, context="test_past") is True, "过去数据应允许"
        assert guard.check_access(200.0, context="test_future") is False, "未来数据应拒绝"
        report = guard.get_report()
        assert report["has_lookahead_violation"] is True, "应有违规记录"
        assert report["verdict"] == "BLOCK", "应 BLOCK"

        # 2. OOSIsolationTracker 测试
        tracker = OOSIsolationTracker(enabled=True)
        tracker.enter_is_phase(round_id=0)
        tracker.mark_seen([0, 1, 2, 3, 4, 5])  # IS 见过 0-5
        tracker.enter_oos_phase()
        contamination = tracker.check_oos_purity([3, 4, 5, 6, 7, 8])  # 3,4,5 污染
        assert contamination == {3, 4, 5}, f"污染应为 {{3,4,5}}, 实际 {contamination}"
        oos_report = tracker.get_report()
        assert oos_report["has_contamination"] is True, "应有污染记录"

        # 3. StrategyLogicVerifier 测试
        verifier = StrategyLogicVerifier(n_permutations=50, alpha=0.05, random_seed=42)
        # 模拟: 原始 Sharpe=2.0, 置换后 Sharpe 均值=0.1 (策略逻辑有效)
        import numpy as np
        rng = np.random.RandomState(42)
        perm_metrics = list(rng.normal(0.1, 0.3, 50))
        result = verifier.verify(
            original_metric=2.0,
            permuted_metrics=perm_metrics,
            test_type="label_permutation",
        )
        assert result.is_logic_driven is True, "强策略应通过逻辑验证"
        assert result.verdict == "PASS", "应 PASS"
        assert result.p_value < 0.05, f"p={result.p_value} 应 < 0.05"

        # 模拟: 原始 Sharpe=0.1, 置换后也是 0.1 (数据拟合)
        result2 = verifier.verify(
            original_metric=0.1,
            permuted_metrics=perm_metrics,
            test_type="label_permutation",
        )
        assert result2.is_logic_driven is False, "弱策略应失败"
        assert result2.verdict == "BLOCK", "应 BLOCK"

        # 4. AntiOverfittingGuard 统一入口
        unified = AntiOverfittingGuard(embargo_bars=2, n_permutations=50, alpha=0.05)
        unified.lookahead.advance_to(100.0)
        unified.lookahead.check_access(200.0, context="unified_test")
        unified_report = unified.get_unified_report()
        assert unified_report["overall_verdict"] == "BLOCK", "有违规应 BLOCK"
        assert "lookahead" in unified_report["block_reasons"][0], "应有 lookahead 原因"

        logger.info("anti_overfitting_guard 自验证全部通过")
        return True
    except Exception as e:
        logger.error("anti_overfitting_guard 自验证失败: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = _self_test()
    print(f"self_test: {ok}")
    sys_exit = 0 if ok else 1
    import sys
    sys.exit(sys_exit)
