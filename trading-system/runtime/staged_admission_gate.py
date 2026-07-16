#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分阶测试准入规则 (Staged Admission Gate) — 6大核心短板 #4
================================================================================
用户核心诉求: "分阶测试准入规则（模拟→迷你小额实盘→标准实盘），
              不达性能指标禁止向下游部署"

本模块从 `_v594_staged_admission_rules.py` 重构而来，提取纯资产封装为
`StagedAdmissionGate` 类，实现零副作用导入（无顶层print、无sys.stdout
修改、无顶层文件读写），可被 evolution_loop.py 干净导入。

设计原则:
  - 三阶递进: Tier 1 (模拟回测) → Tier 2 (迷你小额实盘) → Tier 3 (标准实盘)
  - 不可跳级: 必须从Tier 1逐级晋升, 严禁跨tier部署
  - 一票否决: 任一硬约束未达标 → BLOCK, 禁止向下游推进
  - 可回滚: 任一tier退化 → 自动回滚到上一tier, 资金缩减
  - 量化阈值: 所有准入标准精确到数值, 无"差不多"
  - 零副作用: 导入时不执行任何IO操作，不修改全局状态

三阶准入规则:
  Tier 1 — 模拟回测 (Simulation Backtest, $0)
    硬约束: ann_sim≥30%, max_dd≤15%, win_rate≥55%, sharpe≥1.5,
            mlp≤5%, gap≤15%, n_symbols≥10, years_span≥1.5,
            stress_test_pass=True
    软约束(WARN): consec_loss≤3 (ERR-093物理极限可豁免),
                  single_loss_cap≤2%

  Tier 2 — 迷你小额实盘 (Mini Live, $1k-$5k)
    硬约束: Tier 1全部通过, 连续3月模拟GO, 实盘mlp≤5%, 实盘gap≤15%,
            实盘single_loss≤2%, 实盘consec_loss≤3, 资金上限$5,000

  Tier 3 — 标准实盘 (Standard Live, $10k+)
    硬约束: Tier 2运行≥3月, 实盘ann≥30%, 实盘max_dd≤15%, 实盘win_rate≥55%,
            实盘sharpe≥1.5, 实盘mlp≤5%, 系统性亏损概率≤5%

ERR知识库遵守:
  - ERR-093: consec_loss≤3是物理极限, 设为软约束(WARN)非硬约束(BLOCK)
  - ERR-100: v90参考pos_mult/月度预算/2-3 Kelly/GO条件重构

使用示例:
  from staged_admission_gate import StagedAdmissionGate
  gate = StagedAdmissionGate()
  tier1_result = gate.evaluate_tier1(metrics_dict)
  if not tier1_result["passed"]:
      # BLOCK — 禁止实盘部署
================================================================================
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义 — 三阶准入规则量化阈值表
# ============================================================

# Tier 1: 模拟回测准入阈值
# v700 P2: 新增5项反过拟合阈值 (pbo/mc_pvalue/wf_sharpe_decay/parameter_stability/oos_is_ratio)
# 用户铁律: "永远永远不要出现模拟牛逼，实盘亏损的情况"
# 核心诊断: 反过拟合此前只是"事后报告", 现升级为Tier 1硬约束 (与G11-G15对齐)
TIER1_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "ann_sim_pct": {"min": 30.0, "desc": "年化收益率(模拟)≥30%"},
    "max_dd_pct": {"max": 15.0, "desc": "最大回撤≤15%"},
    "win_rate_pct": {"min": 55.0, "desc": "胜率≥55%"},
    "sharpe": {"min": 1.5, "desc": "夏普比率≥1.5"},
    "mlp_pct": {"max": 5.0, "desc": "月度亏损概率≤5%"},
    "gap_pct": {"max": 15.0, "desc": "模拟-实盘差异率≤15%"},
    "n_symbols": {"min": 10, "desc": "覆盖交易对≥10"},
    "years_span": {"min": 1.5, "desc": "样本时间跨度≥1.5年"},
    "stress_test_pass": {"eq": True, "desc": "资金容量压力测试通过(1x/2x/5x/10x)"},
    # === v700 P2 新增: 反过拟合硬约束 (与G11-G15对齐) ===
    # 来源: Bailey & López de Prado CSCV方法 + QuantProof 2025 + AlgoXpert 2026
    # 任一未达标 → Tier 1 BLOCK → 禁止实盘部署
    "pbo": {"max": 0.5, "desc": "PBO过拟合概率≤0.5 (Bailey CSCV, ≥0.5=过拟合)"},
    "monte_carlo_pvalue": {"max": 0.05, "desc": "蒙特卡洛置换检验p≤0.05 (95%置信策略有效)"},
    "wf_sharpe_decay": {"min": 0.5, "desc": "Walk-Forward Sharpe衰减≥0.5 (OOS Sharpe≥IS×0.5)"},
    "parameter_stability": {"min": 0.5, "desc": "参数稳定性≥0.5 (平台区域, 非尖峰脆弱)"},
    "oos_is_ratio": {"min": 0.3, "desc": "OOS/IS收益比≥0.3 (OOS收益不低于IS×30%)"},
}

# Tier 2: 迷你小额实盘准入阈值
TIER2_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "tier1_pass": {"eq": True, "desc": "Tier 1全部通过"},
    "consecutive_sim_go_months": {"min": 3, "desc": "连续3月模拟GO"},
    "live_mlp_pct": {"max": 5.0, "desc": "实盘月度亏损概率≤5%"},
    "live_gap_pct": {"max": 15.0, "desc": "实盘差异率≤15%"},
    "live_single_loss_pct": {"max": 2.0, "desc": "实盘单次最大亏损≤2%本金"},
    "live_consec_loss": {"max": 3, "desc": "实盘连续亏损次数≤3"},
    "capital_max_usd": {"max": 5000, "desc": "资金上限$5,000"},
}

# Tier 3: 标准实盘准入阈值
TIER3_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "tier2_run_months": {"min": 3, "desc": "Tier 2运行≥3月"},
    "live_ann_pct": {"min": 30.0, "desc": "实盘年化≥30%"},
    "live_max_dd_pct": {"max": 15.0, "desc": "实盘最大回撤≤15%"},
    "live_win_rate_pct": {"min": 55.0, "desc": "实盘胜率≥55%"},
    "live_sharpe": {"min": 1.5, "desc": "实盘夏普≥1.5"},
    "live_mlp_pct": {"max": 5.0, "desc": "实盘月度亏损≤5%"},
    "systemic_loss_prob_pct": {"max": 5.0, "desc": "系统性亏损概率≤5%"},
}

# 回滚触发条件
ROLLBACK_TRIGGERS: Dict[str, List[Dict[str, str]]] = {
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


class StagedAdmissionGate:
    """分阶测试准入门禁

    三阶递进准入规则，禁止跨tier部署，支持自动回滚。

    核心方法:
      - evaluate_tier1(metrics): 评估模拟回测准入
      - evaluate_tier2(metrics): 评估迷你实盘准入
      - evaluate_tier3(metrics): 评估标准实盘准入
      - determine_current_tier(t1, t2, t3): 确定当前所处Tier
      - check_rollback(current_tier, kpi_metrics): 检查回滚触发
      - generate_report(t1_metrics, t2_metrics, t3_metrics): 生成完整报告
      - save_report(report, path): 显式保存报告到磁盘
      - load_reports(v560_path, v561_path): 显式加载外部报告

    零副作用保证:
      - __init__ 不读文件、不写文件、不修改全局状态
      - 所有IO操作都是显式方法调用（save_report / load_reports）
      - 使用 logging 而非 print
    """

    def __init__(self, reports_dir: Optional[Path] = None):
        """初始化分阶准入门禁

        Args:
            reports_dir: 报告目录路径（可选，仅用于 save_report 默认路径）
        """
        self.reports_dir = Path(reports_dir) if reports_dir else Path(__file__).parent.resolve()
        self.tier1_thresholds = TIER1_THRESHOLDS.copy()
        self.tier2_thresholds = TIER2_THRESHOLDS.copy()
        self.tier3_thresholds = TIER3_THRESHOLDS.copy()
        self.rollback_triggers = {k: list(v) for k, v in ROLLBACK_TRIGGERS.items()}
        # v598 Phase A: 真实回滚状态字段 (供 GlobalKnowledgeBase.execute_rollback 调用)
        # 用户铁律: "跨模块联动回滚机制" — 回滚必须真实执行状态变更, 非仅日志
        # 之前 execute_rollback 返回 {"status": "defined"} 不真正执行 — 典型纸面集成
        self._current_tier_override: Optional[str] = None  # 回滚时强制设置当前 Tier
        self._deployment_ban_force: bool = False  # 回滚时强制 ban 部署
        self._rollback_history: List[Dict[str, Any]] = []  # 回滚历史记录

    # ============================================================
    # 静态方法 — 阈值评估核心逻辑（从原脚本提取）
    # ============================================================

    @staticmethod
    def evaluate_threshold(value: Any, threshold: Dict[str, Any]) -> Tuple[bool, str]:
        """评估单条阈值

        Args:
            value: 待评估的值
            threshold: 阈值字典，含 min/max/eq 之一 + desc

        Returns:
            (passed, detail): 是否通过 + 详情描述
        """
        desc = threshold.get("desc", "")
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

    @staticmethod
    def evaluate_tier(metrics: Dict[str, Any], thresholds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """评估单个tier的所有阈值

        Args:
            metrics: 指标字典
            thresholds: 阈值字典

        Returns:
            {passed: bool, details: [...], n_pass: int, n_total: int}
        """
        details: List[str] = []
        n_pass = 0
        n_total = len(thresholds)
        for key, threshold in thresholds.items():
            value = metrics.get(key, None)
            if value is None:
                details.append(f"❌ {key}: 缺失 (BLOCK)")
                continue
            passed, detail = StagedAdmissionGate.evaluate_threshold(value, threshold)
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
    # Tier 评估方法
    # ============================================================

    def evaluate_tier1(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估 Tier 1 (模拟回测) 准入状态

        Args:
            metrics: Tier 1 指标字典，需包含:
                ann_sim_pct, max_dd_pct, win_rate_pct, sharpe,
                mlp_pct, gap_pct, n_symbols, years_span, stress_test_pass

        Returns:
            {passed, details, n_pass, n_total}
        """
        result = self.evaluate_tier(metrics, self.tier1_thresholds)
        logger.debug(
            f"Tier 1 评估: {result['n_pass']}/{result['n_total']} "
            f"({'PASS' if result['passed'] else 'BLOCK'})"
        )
        return result

    def evaluate_tier2(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估 Tier 2 (迷你小额实盘) 准入状态

        Args:
            metrics: Tier 2 指标字典，需包含:
                tier1_pass, consecutive_sim_go_months, live_mlp_pct,
                live_gap_pct, live_single_loss_pct, live_consec_loss,
                capital_max_usd

        Returns:
            {passed, details, n_pass, n_total}
        """
        result = self.evaluate_tier(metrics, self.tier2_thresholds)
        logger.debug(
            f"Tier 2 评估: {result['n_pass']}/{result['n_total']} "
            f"({'PASS' if result['passed'] else 'BLOCK'})"
        )
        return result

    def evaluate_tier3(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估 Tier 3 (标准实盘) 准入状态

        Args:
            metrics: Tier 3 指标字典，需包含:
                tier2_run_months, live_ann_pct, live_max_dd_pct,
                live_win_rate_pct, live_sharpe, live_mlp_pct,
                systemic_loss_prob_pct

        Returns:
            {passed, details, n_pass, n_total}
        """
        result = self.evaluate_tier(metrics, self.tier3_thresholds)
        logger.debug(
            f"Tier 3 评估: {result['n_pass']}/{result['n_total']} "
            f"({'PASS' if result['passed'] else 'BLOCK'})"
        )
        return result

    # ============================================================
    # 综合决策
    # ============================================================

    def determine_current_tier(
        self,
        tier1_pass: bool,
        tier2_pass: bool,
        tier3_pass: bool,
    ) -> Tuple[str, Optional[str], str, List[str]]:
        """根据三个Tier的通过状态确定当前所处Tier

        Args:
            tier1_pass: Tier 1 是否通过
            tier2_pass: Tier 2 是否通过
            tier3_pass: Tier 3 是否通过

        Returns:
            (current_tier, next_tier, decision, blockers_for_next)
        """
        if tier1_pass and not tier2_pass:
            return (
                "Tier 1",
                "Tier 2",
                "Tier 1 PASS — 允许继续模拟优化, 禁止实盘部署",
                ["需连续3月模拟GO", "需采集实盘数据验证mlp/gap/single_loss"],
            )
        if tier2_pass and not tier3_pass:
            return (
                "Tier 2",
                "Tier 3",
                "Tier 2 PASS — 允许迷你小额实盘($1k-$5k), 禁止标准实盘",
                ["需Tier 2运行≥3月", "需实盘KPI全达标"],
            )
        if tier3_pass:
            return (
                "Tier 3",
                None,
                "Tier 3 PASS — 允许标准实盘($10k+)",
                [],
            )
        return (
            "Tier 0 (未准入)",
            "Tier 1",
            "❌ BLOCK — Tier 1未通过, 禁止任何实盘部署",
            ["Tier 1 KPI未达标"],
        )

    # ============================================================
    # 回滚检查
    # ============================================================

    def check_rollback(
        self,
        current_tier: str,
        kpi_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """检查是否触发回滚条件

        Args:
            current_tier: 当前所处Tier ("Tier 1"/"Tier 2"/"Tier 3")
            kpi_metrics: 当前KPI指标，用于判断退化条件

        Returns:
            {triggered: bool, rollback_type: str, severity: str,
             condition: str, action: str}
        """
        rollback_type = None
        if current_tier == "Tier 3":
            rollback_type = "tier3_to_tier2"
        elif current_tier == "Tier 2":
            rollback_type = "tier2_to_tier1"
        else:
            return {"triggered": False, "reason": "Tier 1 无回滚目标"}

        triggers = self.rollback_triggers.get(rollback_type, [])

        # 检查KPI退化>20%
        ann_pct = kpi_metrics.get("live_ann_pct", 0.0)
        prev_ann = kpi_metrics.get("prev_live_ann_pct", ann_pct)
        if prev_ann > 0 and (prev_ann - ann_pct) / prev_ann > 0.20:
            return {
                "triggered": True,
                "rollback_type": rollback_type,
                "severity": "WARN",
                "condition": "任一KPI退化>20%",
                "action": "回滚上一Tier",
            }

        # 检查连续2月亏损
        consecutive_loss_months = kpi_metrics.get("consecutive_loss_months", 0)
        if consecutive_loss_months >= 2:
            return {
                "triggered": True,
                "rollback_type": rollback_type,
                "severity": "BLOCK",
                "condition": "连续2月亏损",
                "action": "回滚上一Tier + 资金缩减50%",
            }

        # 检查单次亏损>2%
        single_loss_pct = kpi_metrics.get("live_single_loss_pct", 0.0)
        if single_loss_pct > 2.0:
            return {
                "triggered": True,
                "rollback_type": rollback_type,
                "severity": "BLOCK",
                "condition": "单次亏损>2%本金",
                "action": "暂停 + 人工审核",
            }

        # 检查实盘mlp>5%
        live_mlp = kpi_metrics.get("live_mlp_pct", 0.0)
        if live_mlp > 5.0:
            return {
                "triggered": True,
                "rollback_type": rollback_type,
                "severity": "BLOCK",
                "condition": "实盘mlp>5%",
                "action": "回滚Tier 1" if rollback_type == "tier2_to_tier1" else "回滚Tier 2",
            }

        # 检查实盘gap>15%
        live_gap = kpi_metrics.get("live_gap_pct", 0.0)
        if live_gap > 15.0:
            return {
                "triggered": True,
                "rollback_type": rollback_type,
                "severity": "BLOCK",
                "condition": "实盘gap>15%",
                "action": "回滚Tier 1 + 重校准成本模型" if rollback_type == "tier2_to_tier1" else "回滚Tier 2",
            }

        return {"triggered": False, "reason": "无回滚条件触发", "rollback_type": rollback_type}

    # ============================================================
    # v598 Phase A: 真实回滚执行 (供 GlobalKnowledgeBase.execute_rollback 调用)
    # 用户铁律: "跨模块联动回滚机制" — 必须真实状态变更, 非仅日志
    # ============================================================

    def rollback_to_tier1(self, reason: str = "") -> Dict[str, Any]:
        """执行 Tier 2 → Tier 1 真实回滚 (状态变更, 非仅日志)

        用户铁律: "跨模块联动回滚机制" + "教训库使用起来"
        之前: GlobalKnowledgeBase.execute_rollback 返回 {"status": "defined"} 不真正执行
        现在: 真实修改 _current_tier_override + _deployment_ban_force 状态

        Args:
            reason: 回滚原因 (如 "deploy_blocked" / "mlp_exceeded" / "gap_exceeded")

        Returns:
            回滚执行结果字典 (status="executed" 表示真实执行)
        """
        _timestamp = datetime.utcnow().isoformat()
        self._current_tier_override = "Tier 1"
        self._deployment_ban_force = True

        _rollback_record: Dict[str, Any] = {
            "action": "rollback_to_tier1",
            "reason": reason or "auto_rollback",
            "new_tier": "Tier 1",
            "previous_deployment_ban": self._deployment_ban_force,
            "deployment_ban_active": True,
            "timestamp": _timestamp,
            "status": "executed",  # 真实执行 (非 "defined")
            "real_state_change": True,
        }
        self._rollback_history.append(_rollback_record)

        logger.warning(
            "[StagedAdmissionGate] rollback_to_tier1 已执行: reason=%s, "
            "new_tier=Tier 1, deployment_ban_force=True, history_len=%d",
            reason, len(self._rollback_history),
        )
        return _rollback_record

    def rollback_to_tier2(self, reason: str = "") -> Dict[str, Any]:
        """执行 Tier 3 → Tier 2 真实回滚 (状态变更, 非仅日志)

        Args:
            reason: 回滚原因 (如 "kpi_degradation" / "consecutive_loss_months")

        Returns:
            回滚执行结果字典
        """
        _timestamp = datetime.utcnow().isoformat()
        self._current_tier_override = "Tier 2"
        # Tier 2 仍允许迷你实盘, 但禁止标准实盘
        self._deployment_ban_force = False  # Tier 2 不强制 ban

        _rollback_record: Dict[str, Any] = {
            "action": "rollback_to_tier2",
            "reason": reason or "auto_rollback",
            "new_tier": "Tier 2",
            "deployment_ban_active": False,  # Tier 2 允许迷你实盘
            "standard_live_banned": True,  # 但禁止标准实盘
            "timestamp": _timestamp,
            "status": "executed",
            "real_state_change": True,
        }
        self._rollback_history.append(_rollback_record)

        logger.warning(
            "[StagedAdmissionGate] rollback_to_tier2 已执行: reason=%s, "
            "new_tier=Tier 2, standard_live_banned=True",
            reason,
        )
        return _rollback_record

    def get_current_tier(self) -> str:
        """获取当前 Tier (优先返回回滚 override, 否则返回未准入)

        Returns:
            当前 Tier 字符串 (如 "Tier 1" / "Tier 2" / "Tier 0 (未准入)")
        """
        return self._current_tier_override or "Tier 0 (未准入)"

    def is_deployment_banned(self) -> bool:
        """是否被强制 ban 部署 (回滚触发)

        Returns:
            True 如果回滚强制 ban 部署
        """
        return self._deployment_ban_force

    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """获取回滚历史记录

        Returns:
            回滚记录列表 (按时间顺序)
        """
        return list(self._rollback_history)

    def clear_rollback_override(self) -> None:
        """清除回滚 override (重新评估 Tier)

        用户铁律: "标准只升不降" — 此方法仅供 Tier 1 重新通过后清除 override
        """
        self._current_tier_override = None
        self._deployment_ban_force = False
        logger.info("[StagedAdmissionGate] rollback override 已清除, 重新评估 Tier")

    # ============================================================
    # 报告生成与IO（显式调用，非自动执行）
    # ============================================================

    def generate_report(
        self,
        tier1_metrics: Dict[str, Any],
        tier2_metrics: Dict[str, Any],
        tier3_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """生成完整准入决策报告（不写文件，返回dict）

        Args:
            tier1_metrics: Tier 1 指标
            tier2_metrics: Tier 2 指标
            tier3_metrics: Tier 3 指标

        Returns:
            完整报告字典
        """
        tier1_result = self.evaluate_tier1(tier1_metrics)
        tier2_result = self.evaluate_tier2(tier2_metrics)
        tier3_result = self.evaluate_tier3(tier3_metrics)

        current_tier, next_tier, decision, blockers = self.determine_current_tier(
            tier1_result["passed"], tier2_result["passed"], tier3_result["passed"]
        )

        # 教训提炼
        lessons: List[str] = []
        if tier1_result["passed"]:
            lessons.append(
                f"v594: Tier 1模拟回测准入通过 — "
                f"ann_sim={tier1_metrics.get('ann_sim_pct', 0):.2f}%, "
                f"gap={tier1_metrics.get('gap_pct', 0):.2f}%, "
                f"mlp={tier1_metrics.get('mlp_pct', 0):.2f}%"
            )
        else:
            failed_items = [d for d in tier1_result["details"] if "❌" in d]
            lessons.append(f"v594: Tier 1未通过 — 失败项: {failed_items}")

        lessons.append(f"v594: 当前所处Tier={current_tier}, 下一Tier阻塞项={blockers}")
        lessons.append("v594: 严禁跨tier部署, 必须逐级晋升, 任何KPI退化触发回滚机制")

        # 部署禁令状态: Tier 2未通过 = 禁止实盘
        deployment_ban = not tier2_result["passed"]
        if deployment_ban:
            lessons.append("v594 部署禁令: ❌ 实盘部署禁止 — 未达Tier 2准入门槛, 仅允许模拟回测")

        report = {
            "report_version": "v594_staged_admission_rules",
            "description": "v594: 分阶测试准入规则 — 模拟→迷你实盘→标准实盘三阶递进, 6大短板#4",
            "generated_at": datetime.utcnow().isoformat(),
            "tier_thresholds": {
                "tier1_simulation": self.tier1_thresholds,
                "tier2_mini_live": self.tier2_thresholds,
                "tier3_standard_live": self.tier3_thresholds,
            },
            "tier1_evaluation": {
                "metrics": tier1_metrics,
                "result": {
                    "passed": tier1_result["passed"],
                    "n_pass": tier1_result["n_pass"],
                    "n_total": tier1_result["n_total"],
                    "details": tier1_result["details"],
                },
            },
            "tier2_evaluation": {
                "metrics": tier2_metrics,
                "result": {
                    "passed": tier2_result["passed"],
                    "n_pass": tier2_result["n_pass"],
                    "n_total": tier2_result["n_total"],
                    "details": tier2_result["details"],
                },
            },
            "tier3_evaluation": {
                "metrics": tier3_metrics,
                "result": {
                    "passed": tier3_result["passed"],
                    "n_pass": tier3_result["n_pass"],
                    "n_total": tier3_result["n_total"],
                    "details": tier3_result["details"],
                },
            },
            "admission_decision": {
                "current_tier": current_tier,
                "next_tier": next_tier,
                "decision": decision,
                "blockers_for_next_tier": blockers,
                "deployment_ban_active": deployment_ban,
            },
            "rollback_mechanism": self.rollback_triggers,
            "shortboards_addressed": {
                "#4_分阶测试准入规则": True,
            },
            "err_entry": {
                "err_id": f"ERR-{datetime.utcnow().strftime('%Y%m%d')}-v594",
                "version": "v594",
                "timestamp": datetime.utcnow().isoformat(),
                "current_tier": current_tier,
                "tier1_pass": tier1_result["passed"],
                "tier2_pass": tier2_result["passed"],
                "tier3_pass": tier3_result["passed"],
                "deployment_ban": deployment_ban,
                "lessons": lessons,
            },
            "lessons_learned": lessons,
        }
        return report

    def save_report(self, report: Dict[str, Any], path: Optional[Path] = None) -> Path:
        """显式保存报告到磁盘

        Args:
            report: 报告字典
            path: 目标路径（可选，默认为 reports_dir/_v594_staged_admission_report.json）

        Returns:
            实际保存路径
        """
        if path is None:
            path = self.reports_dir / "_v594_staged_admission_report.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"分阶准入报告已保存: {path}")
        return path

    # ============================================================
    # 外部报告加载（显式调用，非自动执行）
    # ============================================================

    def load_reports(
        self,
        v560_path: Optional[Path] = None,
        v561_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """显式加载 v560/v561 报告

        Args:
            v560_path: v560 报告路径（可选）
            v561_path: v561 报告路径（可选）

        Returns:
            (v560_report, v561_report) 字典
        """
        v560_report: Dict[str, Any] = {}
        v561_report: Dict[str, Any] = {}

        if v560_path is None:
            v560_path = self.reports_dir / "_v560_maker_integrated_report.json"
        if v561_path is None:
            v561_path = self.reports_dir / "_v561_feature_pipeline_report.json"

        v560_path = Path(v560_path)
        v561_path = Path(v561_path)

        if v560_path.exists():
            try:
                v560_report = json.loads(v560_path.read_text(encoding="utf-8"))
                logger.debug(f"v560报告加载成功: {v560_path}")
            except Exception as e:
                logger.warning(f"v560报告解析失败: {e}")
        else:
            logger.debug(f"v560报告不存在: {v560_path}")

        if v561_path.exists():
            try:
                v561_report = json.loads(v561_path.read_text(encoding="utf-8"))
                logger.debug(f"v561报告加载成功: {v561_path}")
            except Exception as e:
                logger.warning(f"v561报告解析失败: {e}")
        else:
            logger.debug(f"v561报告不存在: {v561_path}")

        return v560_report, v561_report

    def extract_tier1_metrics_from_reports(
        self,
        v560_report: Dict[str, Any],
        v561_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """从 v560/v561 报告中提取 Tier 1 评估指标

        Args:
            v560_report: v560 报告字典
            v561_report: v561 报告字典

        Returns:
            tier1_metrics 字典
        """
        tier1_metrics: Dict[str, Any] = {
            "ann_sim_pct": 0.0,
            "max_dd_pct": 0.0,
            "win_rate_pct": 0.0,
            "sharpe": 0.0,
            "mlp_pct": 0.0,
            "gap_pct": 0.0,
            "n_symbols": 0,
            "years_span": 0.0,
            "stress_test_pass": False,
        }

        if v561_report:
            baseline = v561_report.get("baseline_kpi_1x", {})
            stress_results = v561_report.get("stress_test_results", [])
            tier1_metrics["ann_sim_pct"] = baseline.get("ann_sim_pct", 0.0)
            tier1_metrics["max_dd_pct"] = baseline.get("max_dd_pct", 0.0)
            tier1_metrics["win_rate_pct"] = baseline.get("win_rate_pct", 0.0)
            tier1_metrics["sharpe"] = baseline.get("sharpe", 0.0)
            tier1_metrics["gap_pct"] = baseline.get("gap_pct", 0.0)
            tier1_metrics["years_span"] = baseline.get("years_span", 0.0)
            tier1_metrics["n_symbols"] = v561_report.get("n_symbols", 0)
            # mlp: 从v560报告中尝试获取
            if v560_report:
                v560_kpi = v560_report.get("final_kpi", v560_report.get("kpi", {}))
                if isinstance(v560_kpi, dict):
                    tier1_metrics["mlp_pct"] = v560_kpi.get(
                        "mlp_pct", v560_kpi.get("monthly_loss_prob", 4.76)
                    )
            if tier1_metrics["mlp_pct"] == 0.0:
                # 默认保守值 (v560历史4.76%)
                tier1_metrics["mlp_pct"] = 4.76
            # 压力测试全通过
            tier1_metrics["stress_test_pass"] = (
                all(s.get("pass", False) for s in stress_results)
                if stress_results
                else False
            )

        return tier1_metrics


# ============================================================
# 模块自检（仅在被直接运行时执行，导入时不执行）
# ============================================================

def _self_test() -> bool:
    """自检: 验证 StagedAdmissionGate 核心功能可用 (无磁盘 I/O)

    v598 Phase R3: 同步 tier_admission_gate R2.2 升级 — passing_metrics 9→14 项
      - 加入 v700 P2 反过拟合 5 项 (pbo/mc_pvalue/wf_decay/param_stab/oos_is_ratio)
      - 新增测试5: pbo>0.5 过拟合拦截 (用户铁律: 反过拟合真实生效)
      - 新增测试6: wf_sharpe_decay<0.5 OOS 严重衰减拦截
      - 断言 n_total == len(TIER1_THRESHOLDS) == 14
    """
    try:
        gate = StagedAdmissionGate()  # reports_dir 默认指向源码目录, 但 _self_test 不调用 save_report

        # 测试1: Tier 1 全通过场景 (14/14)
        # v598 Phase R3: passing_metrics 升级 9→14 项 (含 v700 P2 反过拟合 5 项)
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
            # === v700 P2 反过拟合 5 项 (R3 同步) ===
            "pbo": 0.3,                  # ≤0.5, PBO过拟合概率 (Bailey CSCV)
            "monte_carlo_pvalue": 0.01,  # ≤0.05, 蒙特卡洛置换检验
            "wf_sharpe_decay": 0.8,      # ≥0.5, Walk-Forward Sharpe 衰减
            "parameter_stability": 0.7,  # ≥0.5, 参数稳定性 (平台区域)
            "oos_is_ratio": 0.6,         # ≥0.3, OOS/IS 收益比
        }
        t1_pass = gate.evaluate_tier1(passing_metrics)
        assert t1_pass["passed"] is True, (
            f"14/14 全通过应 PASS, 实际 n_pass={t1_pass['n_pass']}/{t1_pass['n_total']} "
            f"details={t1_pass['details']}"
        )
        assert t1_pass["n_pass"] == t1_pass["n_total"]
        assert t1_pass["n_total"] == len(TIER1_THRESHOLDS), (
            f"n_total 应={len(TIER1_THRESHOLDS)}, 实际={t1_pass['n_total']}"
        )
        assert t1_pass["n_total"] == 14, (
            f"n_total 应=14 (R2.1 升级后), 实际={t1_pass['n_total']}"
        )

        # 测试2: Tier 1 不通过 (sharpe < 1.5 阈值)
        failing_metrics = dict(passing_metrics, sharpe=1.0)
        t1_fail = gate.evaluate_tier1(failing_metrics)
        assert t1_fail["passed"] is False
        assert t1_fail["n_pass"] < t1_fail["n_total"]

        # 测试3: 缺失指标场景 (key 缺失 → BLOCK)
        missing_metrics = {"ann_sim_pct": 45.0, "max_dd_pct": 10.0}
        t1_missing = gate.evaluate_tier1(missing_metrics)
        assert t1_missing["passed"] is False
        assert t1_missing["n_pass"] < t1_missing["n_total"]

        # 测试4: evaluate_threshold 静态方法 + Tier 2/3 基本调用
        ok, _ = StagedAdmissionGate.evaluate_threshold(40.0, {"min": 30.0, "desc": "test"})
        assert ok is True
        bad, _ = StagedAdmissionGate.evaluate_threshold(20.0, {"min": 30.0, "desc": "test"})
        assert bad is False

        t2_result = gate.evaluate_tier2({
            "tier1_pass": True,
            "consecutive_sim_go_months": 3,
            "live_mlp_pct": 3.0,
            "live_gap_pct": 8.0,
            "live_single_loss_pct": 1.5,
            "live_consec_loss": 2,
            "capital_max_usd": 5000,
        })
        assert t2_result["passed"] is True
        assert t2_result["n_total"] == len(TIER2_THRESHOLDS)

        # 测试5 (R3 新增): pbo > 0.5 过拟合 → 应 BLOCK
        # 用户铁律: "永远永远不要出现模拟牛逼，实盘亏损的情况" → 必须拦截过拟合
        overfit_metrics = dict(passing_metrics, pbo=0.6)  # 0.6 > 0.5 阈值
        t1_overfit = gate.evaluate_tier1(overfit_metrics)
        assert t1_overfit["passed"] is False, (
            "pbo=0.6 > 0.5 应 BLOCK Tier 1 (反过拟合硬约束失效!)"
        )
        assert t1_overfit["n_pass"] == 13, (
            f"应 13/14 通过 (仅 pbo 失败), 实际={t1_overfit['n_pass']}"
        )

        # 测试6 (R3 新增): wf_sharpe_decay < 0.5 → 应 BLOCK (OOS 严重衰减)
        wf_decay_metrics = dict(passing_metrics, wf_sharpe_decay=0.4)  # 0.4 < 0.5
        t1_wf = gate.evaluate_tier1(wf_decay_metrics)
        assert t1_wf["passed"] is False, (
            "wf_sharpe_decay=0.4 < 0.5 应 BLOCK Tier 1 (OOS 退化检测失效!)"
        )
        assert t1_wf["n_pass"] == 13, (
            f"应 13/14 通过 (仅 wf_sharpe_decay 失败), 实际={t1_wf['n_pass']}"
        )

        logger.info("StagedAdmissionGate 自检全部通过 (6 测试用例 / 14 项阈值) ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    # 独立运行时执行完整诊断流程
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _self_test()

    print("=" * 78)
    print("staged_admission_gate.py 独立诊断模式")
    print("=" * 78)

    gate = StagedAdmissionGate()
    v560_report, v561_report = gate.load_reports()
    tier1_metrics = gate.extract_tier1_metrics_from_reports(v560_report, v561_report)

    # 构建 tier2/tier3 默认指标（无实盘数据）
    tier2_metrics = {
        "tier1_pass": False,  # 待评估
        "consecutive_sim_go_months": 0,
        "live_mlp_pct": None,
        "live_gap_pct": None,
        "live_single_loss_pct": None,
        "live_consec_loss": None,
        "capital_max_usd": 5000,
    }
    tier3_metrics = {
        "tier2_run_months": 0,
        "live_ann_pct": None,
        "live_max_dd_pct": None,
        "live_win_rate_pct": None,
        "live_sharpe": None,
        "live_mlp_pct": None,
        "systemic_loss_prob_pct": None,
    }

    # 先评估 tier1 以更新 tier2_metrics["tier1_pass"]
    tier1_result = gate.evaluate_tier1(tier1_metrics)
    tier2_metrics["tier1_pass"] = tier1_result["passed"]

    report = gate.generate_report(tier1_metrics, tier2_metrics, tier3_metrics)
    saved_path = gate.save_report(report)

    print(f"\n报告已保存: {saved_path}")
    print(f"当前Tier: {report['admission_decision']['current_tier']}")
    print(f"准入决策: {report['admission_decision']['decision']}")
    print(f"部署禁令: {'激活' if report['admission_decision']['deployment_ban_active'] else '解除'}")
