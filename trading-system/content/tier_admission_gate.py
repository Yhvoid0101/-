#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TierAdmissionGate — 分阶测试准入看门人 (运行时实体, v594 真集成 Phase 2)
================================================================================
用途: 把 v594 从单次评估报告升级为持续监控的准入看门人, 解决"连续3月GO"时间维度阻塞项

设计原则:
  - 复用 _v594_staged_admission_rules.py 的阈值定义和 evaluate_tier 函数
  - 去除顶部 sys.stdout.reconfigure + print + 模块级执行副作用
  - 新增 monthly_go 时间序列 (JSONL 持久化), 解决 Tier 2 阻塞项
  - 实盘禁令保持激活: can_deploy=False 时强制标记 round_results 警告
  - 守卫调用模式: 任何异常不影响主循环

三阶准入规则:
  Tier 1 — 模拟回测 (Simulation Backtest, $0)
    硬约束 (任一未达=BLOCK):
      * ann_sim ≥ 30%
      * max_dd ≤ 15%
      * win_rate ≥ 55%
      * sharpe ≥ 1.5
      * mlp (月亏概率) ≤ 5%
      * gap (模拟-实盘差异率) ≤ 15%
      * 资金容量压力测试通过 (1x/2x/5x/10x全pass)
      * n_symbols ≥ 10
      * years_span ≥ 1.5 (样本充分性)

  Tier 2 — 迷你小额实盘 (Mini Live, $1k-$5k)
    硬约束 (任一未达=BLOCK):
      * Tier 1 全部通过
      * 连续3月模拟GO (时间稳定性) ← 本模块新增 monthly_go 追踪
      * 实盘 mlp ≤ 5%
      * 实盘 gap ≤ 15%
      * 实盘 single_loss ≤ 2% (单次最大亏损)
      * 实盘 consec_loss ≤ 3 (连续亏损次数)
      * 资金上限 $5,000

  Tier 3 — 标准实盘 (Standard Live, $10k+)
    硬约束 (任一未达=BLOCK):
      * Tier 2 运行 ≥ 3月 (时间充分性)
      * 实盘 ann_live ≥ 30%
      * 实盘 max_dd ≤ 15%
      * 实盘 win_rate ≥ 55%
      * 实盘 sharpe ≥ 1.5
      * 实盘 mlp ≤ 5%
      * 月度系统性亏损概率 ≤ 5%

集成点:
  - evolution_loop.py 顶部: 软加载 try/except + None fallback
  - evolution_loop.py __init__: self.tier_gate = TierAdmissionGate(storage_dir=..., enabled=True)
  - evolution_loop.py loop_end: self.tier_gate.check_admission(metrics) → can_deploy=False 时标记 round_results
================================================================================
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义 — 复用 _v594_staged_admission_rules.py (避免 import 副作用, 独立复制)
# ============================================================

# Tier 1: 模拟回测准入阈值
# v598 Phase R2.1: 同步 v700 P2 反过拟合 5 项硬约束 (与 staged_admission_gate 一致)
# 用户铁律: "永远永远不要出现模拟牛逼，实盘亏损的情况"
# 用户铁律: "不要通过已知数据反推策略，那样就过拟合了"
TIER1_THRESHOLDS = {
    "ann_sim_pct": {"min": 30.0, "desc": "年化收益率(模拟)≥30%"},
    "max_dd_pct": {"max": 15.0, "desc": "最大回撤≤15%"},
    "win_rate_pct": {"min": 55.0, "desc": "胜率≥55%"},
    "sharpe": {"min": 1.5, "desc": "夏普比率≥1.5"},
    "mlp_pct": {"max": 5.0, "desc": "月度亏损概率≤5%"},
    "gap_pct": {"max": 15.0, "desc": "模拟-实盘差异率≤15%"},
    "n_symbols": {"min": 10, "desc": "覆盖交易对≥10"},
    "years_span": {"min": 1.5, "desc": "样本时间跨度≥1.5年"},
    "stress_test_pass": {"eq": True, "desc": "资金容量压力测试通过(1x/2x/5x/10x)"},
    # === v700 P2 新增: 反过拟合硬约束 (与 staged_admission_gate L77-84 对齐) ===
    # 来源: Bailey & López de Prado CSCV方法 + QuantProof 2025 + AlgoXpert 2026
    # 任一未达标 → Tier 1 BLOCK → 禁止实盘部署
    "pbo": {"max": 0.5, "desc": "PBO过拟合概率≤0.5 (Bailey CSCV, ≥0.5=过拟合)"},
    "monte_carlo_pvalue": {"max": 0.05, "desc": "蒙特卡洛置换检验p≤0.05 (95%置信策略有效)"},
    "wf_sharpe_decay": {"min": 0.5, "desc": "Walk-Forward Sharpe衰减≥0.5 (OOS Sharpe≥IS×0.5)"},
    "parameter_stability": {"min": 0.5, "desc": "参数稳定性≥0.5 (平台区域, 非尖峰脆弱)"},
    "oos_is_ratio": {"min": 0.3, "desc": "OOS/IS收益比≥0.3 (OOS收益不低于IS×30%)"},
}

# Tier 2: 迷你小额实盘准入阈值
TIER2_THRESHOLDS = {
    "tier1_pass": {"eq": True, "desc": "Tier 1全部通过"},
    "consecutive_sim_go_months": {"min": 3, "desc": "连续3月模拟GO"},
    "live_mlp_pct": {"max": 5.0, "desc": "实盘月度亏损概率≤5%"},
    "live_gap_pct": {"max": 15.0, "desc": "实盘差异率≤15%"},
    "live_single_loss_pct": {"max": 2.0, "desc": "实盘单次最大亏损≤2%本金"},
    "live_consec_loss": {"max": 3, "desc": "实盘连续亏损次数≤3"},
    "capital_max_usd": {"max": 5000, "desc": "资金上限$5,000"},
}

# Tier 3: 标准实盘准入阈值
TIER3_THRESHOLDS = {
    "tier2_run_months": {"min": 3, "desc": "Tier 2运行≥3月"},
    "live_ann_pct": {"min": 30.0, "desc": "实盘年化≥30%"},
    "live_max_dd_pct": {"max": 15.0, "desc": "实盘最大回撤≤15%"},
    "live_win_rate_pct": {"min": 55.0, "desc": "实盘胜率≥55%"},
    "live_sharpe": {"min": 1.5, "desc": "实盘夏普≥1.5"},
    "live_mlp_pct": {"max": 5.0, "desc": "实盘月度亏损≤5%"},
    "systemic_loss_prob_pct": {"max": 5.0, "desc": "系统性亏损概率≤5%"},
}

# 回滚触发条件
ROLLBACK_TRIGGERS = {
    "tier3_to_tier2": [
        {"condition": "任一KPI退化>20%", "action": "回滚Tier 2", "severity": "WARN"},
        {"condition": "连续2月亏损", "action": "回滚Tier 2 + 资金缩减50%", "severity": "BLOCK"},
        {"condition": "单次亏损>2%本金", "action": "暂停 + 人工审核", "severity": "BLOCK"},
        {"condition": "系统性亏损信号", "action": "立即停止 + 全量回滚", "severity": "BLOCK"},
    ],
    "tier2_to_tier1": [
        {"condition": "实盘mlp>5%", "action": "回滚Tier 1", "severity": "BLOCK"},
        {"condition": "实盘gap>15%", "action": "回滚Tier 1 + 重校准成本模型", "severity": "BLOCK"},
        {"condition": "单次亏损>2%本金", "action": "暂停 + 人工审核", "severity": "BLOCK"},
    ],
}


def _now_iso() -> str:
    """ISO 格式时间戳 (避免 utcnow deprecation)"""
    return datetime.now(timezone.utc).isoformat()


def evaluate_threshold(value: Any, threshold: Dict) -> Tuple[bool, str]:
    """评估单条阈值 (复用 v594 L136, 无业务逻辑变化)

    Args:
        value: 待评估的值 (数值或布尔)
        threshold: 阈值定义, 含 min/max/eq + desc

    Returns:
        (passed, detail): 通过标志 + 人类可读详情
    """
    desc = threshold.get("desc", "")
    if value is None:
        return False, f"❌ {desc}: 缺失 (BLOCK)"
    if "min" in threshold:
        passed = value >= threshold["min"]
        return passed, f"{desc}: {value:.4f} {'≥' if passed else '<'} {threshold['min']}"
    if "max" in threshold:
        passed = value <= threshold["max"]
        return passed, f"{desc}: {value:.4f} {'≤' if passed else '>'} {threshold['max']}"
    if "eq" in threshold:
        passed = value == threshold["eq"]
        return passed, f"{desc}: {value} {'==' if passed else '!='} {threshold['eq']}"
    return True, "无阈值"


def evaluate_tier(metrics: Dict, thresholds: Dict) -> Dict:
    """评估单个 tier 的所有阈值 (复用 v594 L153, 无业务逻辑变化)

    Args:
        metrics: 指标字典 {key: value}
        thresholds: 阈值字典 {key: {min/max/eq, desc}}

    Returns:
        {passed: bool, details: List[str], n_pass: int, n_total: int}
    """
    details = []
    n_pass = 0
    n_total = len(thresholds)
    for key, threshold in thresholds.items():
        value = metrics.get(key, None)
        passed, detail = evaluate_threshold(value, threshold)
        if passed:
            n_pass += 1
            details.append(f"✅ {detail}")
        else:
            details.append(f"❌ {detail} (BLOCK)")
    return {
        "passed": n_pass == n_total,
        "details": details,
        "n_pass": n_pass,
        "n_total": n_total,
    }


# ============================================================
# TierAdmissionGate — 分阶测试准入看门人 (运行时实体)
# ============================================================

class TierAdmissionGate:
    """分阶测试准入看门人 (运行时实体)

    复用 _v594_staged_admission_rules.py 的阈值定义和 evaluate_tier 函数,
    去除顶部 sys.stdout.reconfigure + print + JSON 加载副作用.

    新增能力:
      - record_monthly_go(): 记录每月GO/NOGO到时间序列 (解决"连续3月GO"阻塞项)
      - get_consecutive_go_months(): 获取当前连续GO月数
      - check_admission(): 综合准入检查 (返回 blockers + can_deploy)
      - check_rollback(): 回滚触发检查

    实盘禁令保持激活:
      - can_deploy=False 时, 调用方应强制标记 round_results 警告
      - 不自动放行, 仅监控和记录
    """

    def __init__(self, storage_dir: Optional[Path] = None, enabled: bool = True):
        """初始化分阶准入看门人

        Args:
            storage_dir: 持久化存储目录 (用于 monthly_go_history.jsonl)
            enabled: 是否启用 (False 时所有方法返回安全默认值)
        """
        self.enabled = enabled
        self.storage_dir = storage_dir
        # 阈值常量 (类属性, 避免每次实例化复制)
        self.TIER1_THRESHOLDS = TIER1_THRESHOLDS
        self.TIER2_THRESHOLDS = TIER2_THRESHOLDS
        self.TIER3_THRESHOLDS = TIER3_THRESHOLDS
        self.ROLLBACK_TRIGGERS = ROLLBACK_TRIGGERS
        # monthly_go 时间序列 (解决 Tier 2 阻塞项: 连续3月GO)
        # 每条记录: {month: "YYYY-MM", go: bool, metrics: Dict, timestamp: ISO}
        self._monthly_go_history: List[Dict] = []
        # tier2 运行月数追踪 (解决 Tier 3 阻塞项: Tier 2 运行≥3月)
        self._tier2_start_month: Optional[str] = None
        # 加载历史记录
        self._load_history()

    # ============================================================
    # Tier 评估方法 (复用 v594 evaluate_tier)
    # ============================================================

    def evaluate_tier1(self, metrics: Dict) -> Dict:
        """评估 Tier 1 (模拟回测) — 复用 v594 evaluate_tier

        Args:
            metrics: 模拟回测指标, 需含:
                ann_sim_pct, max_dd_pct, win_rate_pct, sharpe,
                mlp_pct, gap_pct, n_symbols, years_span, stress_test_pass

        Returns:
            {passed: bool, details: List[str], n_pass: int, n_total: int}
        """
        return evaluate_tier(metrics, self.TIER1_THRESHOLDS)

    def evaluate_tier2(self, metrics: Dict) -> Dict:
        """评估 Tier 2 (迷你实盘 $1k-$5k)

        Args:
            metrics: 实盘指标, 需含:
                tier1_pass (bool), consecutive_sim_go_months (int, 从本类追踪),
                live_mlp_pct, live_gap_pct, live_single_loss_pct,
                live_consec_loss, capital_max_usd

        Returns:
            {passed: bool, details: List[str], n_pass: int, n_total: int}
        """
        # 自动填充 consecutive_sim_go_months (从 monthly_go_history 追踪)
        if "consecutive_sim_go_months" not in metrics:
            metrics = dict(metrics)  # 浅拷贝避免修改原 dict
            metrics["consecutive_sim_go_months"] = self.get_consecutive_go_months()
        return evaluate_tier(metrics, self.TIER2_THRESHOLDS)

    def evaluate_tier3(self, metrics: Dict) -> Dict:
        """评估 Tier 3 (标准实盘 $10k+)

        Args:
            metrics: 标准实盘指标, 需含:
                tier2_run_months (int), live_ann_pct, live_max_dd_pct,
                live_win_rate_pct, live_sharpe, live_mlp_pct, systemic_loss_prob_pct

        Returns:
            {passed: bool, details: List[str], n_pass: int, n_total: int}
        """
        return evaluate_tier(metrics, self.TIER3_THRESHOLDS)

    # ============================================================
    # monthly_go 时间序列 (新增能力, 解决 Tier 2 阻塞项)
    # ============================================================

    def record_monthly_go(self, month: str, go: bool, metrics: Optional[Dict] = None) -> None:
        """记录每月GO/NOGO到时间序列 (解决 Tier 2 阻塞项: 连续3月GO)

        Args:
            month: 月份字符串 "YYYY-MM" (如 "2026-07")
            go: 该月是否GO (True=通过所有KPI, False=有未达标项)
            metrics: 该月的完整KPI指标 (可选, 用于审计追溯)
        """
        if not self.enabled:
            return
        # 去重: 同 month 替换
        existing_idx = None
        for i, rec in enumerate(self._monthly_go_history):
            if rec.get("month") == month:
                existing_idx = i
                break
        record = {
            "month": month,
            "go": go,
            "metrics": metrics or {},
            "timestamp": _now_iso(),
        }
        if existing_idx is not None:
            self._monthly_go_history[existing_idx] = record
        else:
            self._monthly_go_history.append(record)
        # 按 month 排序 (确保时间序列顺序)
        self._monthly_go_history.sort(key=lambda r: r.get("month", ""))
        # 持久化
        self._save_history()
        logger.info(
            "[TierAdmissionGate] record_monthly_go: month=%s, go=%s, consecutive_go=%d",
            month, go, self.get_consecutive_go_months(),
        )

    def get_consecutive_go_months(self) -> int:
        """获取当前连续GO月数 (从最近一个月往前数)

        Returns:
            连续GO月数 (0=最近一个月NOGO或无记录)
        """
        if not self._monthly_go_history:
            return 0
        # 按月份倒序, 从最近一个开始数连续GO
        sorted_history = sorted(
            self._monthly_go_history,
            key=lambda r: r.get("month", ""),
            reverse=True,
        )
        count = 0
        for rec in sorted_history:
            if rec.get("go", False):
                count += 1
            else:
                break  # 遇到第一个NOGO就停止
        return count

    def get_tier2_run_months(self) -> int:
        """获取 Tier 2 已运行月数 (解决 Tier 3 阻塞项: Tier 2 运行≥3月)

        Returns:
            Tier 2 已运行月数 (0=未启动, 3+=可申请 Tier 3)
        """
        if self._tier2_start_month is None:
            return 0
        # 计算从 tier2_start_month 到当前的总月数
        try:
            start_year, start_mon = map(int, self._tier2_start_month.split("-"))
            now = datetime.now(timezone.utc)
            return (now.year - start_year) * 12 + (now.month - start_mon) + 1
        except (ValueError, AttributeError):
            return 0

    def set_tier2_start(self, month: str) -> None:
        """设置 Tier 2 启动月份 (当 Tier 2 准入通过时调用)

        Args:
            month: Tier 2 启动月份 "YYYY-MM"
        """
        self._tier2_start_month = month
        self._save_history()
        logger.info("[TierAdmissionGate] Tier 2 启动月份设置: %s", month)

    # ============================================================
    # 综合准入检查
    # ============================================================

    def check_admission(self, current_metrics: Dict) -> Dict:
        """综合准入检查: 返回 {current_tier, next_tier, blockers, can_deploy}

        Args:
            current_metrics: 当前所有可用指标 (模拟+实盘混合)
                必须含 Tier 1 指标, 可选含 Tier 2/3 实盘指标

        Returns:
            {
                "current_tier": str,          # "Tier 0"/"Tier 1"/"Tier 2"/"Tier 3"
                "next_tier": Optional[str],   # 下一可申请Tier
                "can_deploy": bool,           # 是否允许实盘部署 (Tier 2+)
                "tier1_result": Dict,         # Tier 1 评估详情
                "tier2_result": Dict,         # Tier 2 评估详情 (含实盘指标时)
                "tier3_result": Dict,         # Tier 3 评估详情 (含实盘指标时)
                "blockers_for_next_tier": List[str],  # 晋升阻塞项
                "consecutive_go_months": int, # 连续GO月数
                "tier2_run_months": int,      # Tier 2 运行月数
            }
        """
        if not self.enabled:
            return {
                "current_tier": "Disabled",
                "next_tier": None,
                "can_deploy": False,
                "blockers_for_next_tier": ["TierAdmissionGate未启用"],
                "consecutive_go_months": 0,
                "tier2_run_months": 0,
            }

        # 评估 Tier 1
        # v598 Phase R2.2: 同步 14 项 TIER1_THRESHOLDS (基础 9 + v700 P2 反过拟合 5)
        # 修复 BUG: 之前只提取 9 项, 反过拟合 5 项变 None → evaluate_threshold 触发 BLOCK
        # 导致即使所有 KPI 达标, current_tier 也错误返回 "Tier 0"
        tier1_metrics = {
            # === 基础 9 项 ===
            "ann_sim_pct": current_metrics.get("ann_sim_pct"),
            "max_dd_pct": current_metrics.get("max_dd_pct"),
            "win_rate_pct": current_metrics.get("win_rate_pct"),
            "sharpe": current_metrics.get("sharpe"),
            "mlp_pct": current_metrics.get("mlp_pct"),
            "gap_pct": current_metrics.get("gap_pct"),
            "n_symbols": current_metrics.get("n_symbols"),
            "years_span": current_metrics.get("years_span"),
            "stress_test_pass": current_metrics.get("stress_test_pass"),
            # === v700 P2 反过拟合 5 项 (R2.2 同步) ===
            "pbo": current_metrics.get("pbo"),
            "monte_carlo_pvalue": current_metrics.get("monte_carlo_pvalue"),
            "wf_sharpe_decay": current_metrics.get("wf_sharpe_decay"),
            "parameter_stability": current_metrics.get("parameter_stability"),
            "oos_is_ratio": current_metrics.get("oos_is_ratio"),
        }
        tier1_result = self.evaluate_tier1(tier1_metrics)

        # 评估 Tier 2 (需要实盘指标, 若缺失则自动 BLOCK)
        tier2_metrics = {
            "tier1_pass": tier1_result["passed"],
            "consecutive_sim_go_months": self.get_consecutive_go_months(),
            "live_mlp_pct": current_metrics.get("live_mlp_pct"),
            "live_gap_pct": current_metrics.get("live_gap_pct"),
            "live_single_loss_pct": current_metrics.get("live_single_loss_pct"),
            "live_consec_loss": current_metrics.get("live_consec_loss"),
            "capital_max_usd": current_metrics.get("capital_max_usd", 5000),
        }
        tier2_result = self.evaluate_tier2(tier2_metrics)

        # 评估 Tier 3 (需要标准实盘指标)
        tier3_metrics = {
            "tier2_run_months": self.get_tier2_run_months(),
            "live_ann_pct": current_metrics.get("live_ann_pct"),
            "live_max_dd_pct": current_metrics.get("live_max_dd_pct"),
            "live_win_rate_pct": current_metrics.get("live_win_rate_pct"),
            "live_sharpe": current_metrics.get("live_sharpe"),
            "live_mlp_pct": current_metrics.get("live_mlp_pct"),
            "systemic_loss_prob_pct": current_metrics.get("systemic_loss_prob_pct"),
        }
        tier3_result = self.evaluate_tier3(tier3_metrics)

        # 确定当前所处 Tier + 下一 Tier + 阻塞项
        if tier3_result["passed"]:
            current_tier = "Tier 3"
            next_tier = None
            can_deploy = True
            blockers_for_next = []
        elif tier2_result["passed"]:
            current_tier = "Tier 2"
            next_tier = "Tier 3"
            can_deploy = True  # Tier 2 允许迷你实盘
            blockers_for_next = self._extract_blockers(tier3_result)
        elif tier1_result["passed"]:
            current_tier = "Tier 1"
            next_tier = "Tier 2"
            can_deploy = False  # Tier 1 禁止实盘部署
            blockers_for_next = self._extract_blockers(tier2_result)
        else:
            current_tier = "Tier 0"
            next_tier = "Tier 1"
            can_deploy = False
            blockers_for_next = self._extract_blockers(tier1_result)

        return {
            "current_tier": current_tier,
            "next_tier": next_tier,
            "can_deploy": can_deploy,
            "tier1_result": tier1_result,
            "tier2_result": tier2_result,
            "tier3_result": tier3_result,
            "blockers_for_next_tier": blockers_for_next,
            "consecutive_go_months": self.get_consecutive_go_months(),
            "tier2_run_months": self.get_tier2_run_months(),
        }

    def check_rollback(self, current_metrics: Dict) -> Dict:
        """回滚触发检查: 检测是否需要从当前 Tier 回滚到上一 Tier

        Args:
            current_metrics: 当前实盘KPI指标

        Returns:
            {
                "triggered": bool,
                "rollback_type": Optional[str],  # "tier3_to_tier2"/"tier2_to_tier1"/None
                "triggered_conditions": List[str],
                "action": str,
            }
        """
        if not self.enabled:
            return {"triggered": False, "rollback_type": None,
                    "triggered_conditions": [], "action": "未启用"}

        triggered_conditions = []

        # 检查 Tier 3 → Tier 2 回滚触发条件
        if current_metrics.get("live_ann_pct") is not None:
            # KPI 退化检查 (简化版: 实盘 ann < 模拟 ann * 0.8)
            sim_ann = current_metrics.get("ann_sim_pct", 0.0)
            live_ann = current_metrics.get("live_ann_pct", 0.0)
            if sim_ann > 0 and live_ann < sim_ann * 0.8:
                triggered_conditions.append("任一KPI退化>20%")
            # 连续2月亏损检查
            if len(self._monthly_go_history) >= 2:
                recent_two = sorted(self._monthly_go_history,
                                   key=lambda r: r.get("month", ""),
                                   reverse=True)[:2]
                if all(not r.get("go", True) for r in recent_two):
                    triggered_conditions.append("连续2月亏损")
            # 单次亏损>2%本金
            if current_metrics.get("live_single_loss_pct", 0.0) > 2.0:
                triggered_conditions.append("单次亏损>2%本金")
            # 系统性亏损信号
            if current_metrics.get("systemic_loss_prob_pct", 0.0) > 5.0:
                triggered_conditions.append("系统性亏损信号")

        if triggered_conditions:
            return {
                "triggered": True,
                "rollback_type": "tier3_to_tier2",
                "triggered_conditions": triggered_conditions,
                "action": "回滚Tier 2 + 资金缩减50% + 人工审核",
            }

        # 检查 Tier 2 → Tier 1 回滚触发条件
        if current_metrics.get("live_mlp_pct", 0.0) > 5.0:
            triggered_conditions.append("实盘mlp>5%")
        if current_metrics.get("live_gap_pct", 0.0) > 15.0:
            triggered_conditions.append("实盘gap>15%")
        if current_metrics.get("live_single_loss_pct", 0.0) > 2.0:
            triggered_conditions.append("单次亏损>2%本金")

        if triggered_conditions:
            return {
                "triggered": True,
                "rollback_type": "tier2_to_tier1",
                "triggered_conditions": triggered_conditions,
                "action": "回滚Tier 1 + 重校准成本模型",
            }

        return {
            "triggered": False,
            "rollback_type": None,
            "triggered_conditions": [],
            "action": "无回滚触发",
        }

    # ============================================================
    # 持久化 (JSONL)
    # ============================================================

    def _load_history(self) -> None:
        """从磁盘加载 monthly_go 历史 (JSONL 格式)"""
        if self.storage_dir is None:
            return
        history_path = self.storage_dir / "monthly_go_history.jsonl"
        if not history_path.exists():
            return
        try:
            with history_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._monthly_go_history.append(rec)
                    except json.JSONDecodeError:
                        continue
            # 加载 tier2_start_month
            tier2_path = self.storage_dir / "tier2_state.json"
            if tier2_path.exists():
                tier2_state = json.loads(tier2_path.read_text(encoding="utf-8"))
                self._tier2_start_month = tier2_state.get("tier2_start_month")
            logger.info(
                "[TierAdmissionGate] 加载 %d 条 monthly_go 历史, tier2_start=%s",
                len(self._monthly_go_history), self._tier2_start_month,
            )
        except Exception as e:
            logger.warning("[TierAdmissionGate] 加载历史失败: %s", e)

    def _save_history(self) -> None:
        """持久化 monthly_go 历史到磁盘 (JSONL 格式)"""
        if self.storage_dir is None:
            return
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            history_path = self.storage_dir / "monthly_go_history.jsonl"
            with history_path.open("w", encoding="utf-8") as f:
                for rec in self._monthly_go_history:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # 持久化 tier2_start_month
            if self._tier2_start_month is not None:
                tier2_path = self.storage_dir / "tier2_state.json"
                tier2_path.write_text(
                    json.dumps({"tier2_start_month": self._tier2_start_month},
                              ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as e:
            logger.warning("[TierAdmissionGate] 持久化失败: %s", e)

    # ============================================================
    # 辅助方法
    # ============================================================

    @staticmethod
    def _extract_blockers(tier_result: Dict) -> List[str]:
        """从 tier_result 提取阻塞项 (失败项列表)"""
        return [d for d in tier_result.get("details", []) if "❌" in d]

    def get_summary(self) -> Dict:
        """获取准入状态摘要 (供日志和报告使用)"""
        return {
            "consecutive_go_months": self.get_consecutive_go_months(),
            "tier2_run_months": self.get_tier2_run_months(),
            "total_months_recorded": len(self._monthly_go_history),
            "tier2_start_month": self._tier2_start_month,
            "enabled": self.enabled,
        }


# ============================================================
# 模块自检入口
# ============================================================

def _self_test() -> bool:
    """自检: 验证 TierAdmissionGate 核心功能可用 (storage_dir=None 避免磁盘 I/O)

    v598 Phase R2.2: 升级为 6 测试用例 + 14 项 passing_metrics
      - 同步 v700 P2 反过拟合 5 项硬约束 (pbo/mc_pvalue/wf_decay/param_stab/oos_is_ratio)
      - 新增测试5: pbo>0.5 过拟合拦截验证 (用户铁律: 反过拟合必须真实生效)
      - 新增测试6: wf_sharpe_decay<0.5 OOS Sharpe 严重衰减拦截验证
      - 新增断言: n_total==14 (确保 TIER1_THRESHOLDS 与 R2.1 升级一致)
    """
    try:
        gate = TierAdmissionGate(storage_dir=None, enabled=True)

        # 测试1: Tier 1 全通过场景 (14/14) → check_admission 返回 Tier 1, 禁止实盘部署
        # v598 Phase R2.2: passing_metrics 升级 9→14 项 (含 v700 P2 反过拟合 5 项)
        passing_metrics = {
            # === 基础 9 项 ===
            "ann_sim_pct": 45.0,
            "max_dd_pct": 10.0,
            "win_rate_pct": 60.0,
            "sharpe": 2.0,
            "mlp_pct": 3.0,
            "gap_pct": 8.0,
            "n_symbols": 15,
            "years_span": 2.0,
            "stress_test_pass": True,
            # === v700 P2 反过拟合 5 项 (R2.2 新增) ===
            "pbo": 0.3,                  # ≤0.5, PBO过拟合概率 (Bailey CSCV)
            "monte_carlo_pvalue": 0.01,  # ≤0.05, 蒙特卡洛置换检验
            "wf_sharpe_decay": 0.8,      # ≥0.5, Walk-Forward Sharpe 衰减
            "parameter_stability": 0.7,  # ≥0.5, 参数稳定性 (平台区域)
            "oos_is_ratio": 0.6,         # ≥0.3, OOS/IS 收益比
        }
        result = gate.check_admission(passing_metrics)
        assert result["current_tier"] == "Tier 1", f"应=Tier 1, 实际={result['current_tier']}"
        assert result["can_deploy"] is False  # Tier 1 禁止实盘
        assert result["tier1_result"]["passed"] is True
        # v598 Phase R2.2: 显式断言 n_total=14 (与 R2.1 TIER1_THRESHOLDS 升级一致)
        assert result["tier1_result"]["n_total"] == 14, (
            f"n_total 应=14, 实际={result['tier1_result']['n_total']} "
            f"(R2.1 TIER1_THRESHOLDS 升级可能未生效)"
        )
        assert result["tier1_result"]["n_pass"] == result["tier1_result"]["n_total"]

        # 测试2: Tier 1 不通过 (sharpe < 1.5) → current_tier="Tier 0", 有阻塞项
        failing_metrics = dict(passing_metrics, sharpe=1.0)
        result_fail = gate.check_admission(failing_metrics)
        assert result_fail["current_tier"] == "Tier 0"
        assert result_fail["can_deploy"] is False
        assert len(result_fail["blockers_for_next_tier"]) > 0
        assert result_fail["tier1_result"]["passed"] is False

        # 测试3: monthly_go 时间序列记录与连续GO月数查询
        assert gate.get_consecutive_go_months() == 0  # 初始为0
        gate.record_monthly_go("2026-06", True, {"sharpe": 2.0})
        gate.record_monthly_go("2026-07", True, {"sharpe": 2.1})
        assert gate.get_consecutive_go_months() == 2
        gate.record_monthly_go("2026-08", False)  # NOGO 中断连续
        assert gate.get_consecutive_go_months() == 0

        # 测试4: disabled 场景 — 所有方法返回安全默认值
        gate_off = TierAdmissionGate(storage_dir=None, enabled=False)
        result_off = gate_off.check_admission(passing_metrics)
        assert result_off["current_tier"] == "Disabled"
        assert result_off["can_deploy"] is False

        # 测试5 (R2.2 新增): pbo > 0.5 过拟合 → 应 BLOCK (用户铁律: 反过拟合真实生效)
        # 场景: 策略所有基础 KPI 达标, 但 PBO=0.6 表明策略在历史数据上"看起来好"是过拟合产物
        # 期望: tier1_result.passed=False, current_tier="Tier 0", 部署禁令激活
        overfit_metrics = dict(passing_metrics, pbo=0.6)  # 0.6 > 0.5 阈值
        result_overfit = gate.check_admission(overfit_metrics)
        assert result_overfit["tier1_result"]["passed"] is False, (
            "pbo=0.6 > 0.5 应 BLOCK Tier 1 (反过拟合硬约束失效!)"
        )
        assert result_overfit["current_tier"] == "Tier 0", (
            f"过拟合策略应处 Tier 0, 实际={result_overfit['current_tier']}"
        )
        assert result_overfit["can_deploy"] is False, "过拟合策略禁止实盘部署"
        # 验证阻塞项确实包含 pbo
        blockers_text = " ".join(result_overfit.get("blockers_for_next_tier", []))
        assert "pbo" in " ".join(result_overfit["tier1_result"]["details"]) or "PBO" in blockers_text or "pbo" in blockers_text, (
            "阻塞项未提及 pbo — 反过拟合拦截可能未在 details 中正确标注"
        )

        # 测试6 (R2.2 新增): wf_sharpe_decay < 0.5 → 应 BLOCK
        # 场景: 策略 IS Sharpe=2.0 但 OOS Sharpe 衰减到 0.4×IS=0.8 (低于 0.5 阈值)
        # 期望: tier1_result.passed=False, 表明 OOS 严重退化
        wf_decay_metrics = dict(passing_metrics, wf_sharpe_decay=0.4)  # 0.4 < 0.5 阈值
        result_wf = gate.check_admission(wf_decay_metrics)
        assert result_wf["tier1_result"]["passed"] is False, (
            "wf_sharpe_decay=0.4 < 0.5 应 BLOCK Tier 1 (OOS 退化检测失效!)"
        )
        assert result_wf["can_deploy"] is False, "OOS 严重衰减策略禁止实盘部署"
        assert result_wf["tier1_result"]["n_pass"] == 13, (
            f"应 13/14 通过 (仅 wf_sharpe_decay 失败), 实际={result_wf['tier1_result']['n_pass']}"
        )

        logger.info("TierAdmissionGate 自检全部通过 (6 测试用例 / 14 项阈值) ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
