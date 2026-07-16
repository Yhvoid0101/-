"""Tier 2 迷你实盘准备清单（仅准备不部署）.

模块用途:
    封装 v97 Tier 2 迷你实盘准入条件的自动化验证逻辑，与
    `_v97_tier2_admission_checklist.md`（人工阅读文档）互补：
    - MD 文档: 人工审阅 7 章 359 行清单
    - 本模块: 自动化逐项验证 + 生成结构化报告

设计原则:
    1. 逐项验证非综合判定（ERR-20260701-v93bug 教训:
       不能只看 all_pass=true 就声称成功，必须逐项验证每项准入）
    2. deployment_status 必须准确反映当前 BLOCKED 状态
       （非 "PREPARED" 误导用户，违反"未达指标不得部署"铁律）
    3. 仅准备账户配置，不实际下单/部署（用户铁律"仅准备不部署"）
    4. 实盘禁令标记 deployment_ban_active 字段
    5. 数据源: 三个 JSON 文件（go_nogo / v999 / tier2_state），
       无硬编码指标值

参数说明:
    sandbox_dir: Path — sandbox_trading 根目录路径

返回结构:
    各 verify_* 方法返回 dict:
    {
        "name": str,           # 准入项名称
        "passed": bool,        # 是否通过
        "value": Any,           # 当前实际值
        "target": str,          # 目标阈值描述
        "detail": str,          # 详细说明
        "sub_checks": list,     # 子项明细（可选）
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Tier2PreparationChecklist:
    """Tier 2 迷你实盘准备清单（仅准备不部署）.

    与 `_v97_tier2_admission_checklist.md` 第 1 章 12 项准入一一对应:
        准入 1:  Tier 1 12/12 通过
        准入 2:  极端行情 5/5 真通过（用 G9 stress_test_pass 近似）
        准入 3-10: G1-G8 KPI（ann/max_dd/win_rate/sharpe/mlp/gap/n_symbols/years）
        准入 11: v999 复盘闭环完整
        准入 12: 3 月滚动窗口 consecutive_go_months >= 3

    关键约束:
        - deployment_status 仅在 12 项全通过时为 "PREPARED_NOT_DEPLOYED"
        - 任一项未通过 → "BLOCKED"（非 PREPARED，避免误导）
        - prepare_account_config 不实际下单，仅返回配置 dict
    """

    # 12 项准入阈值（与 _v97_tier2_admission_checklist.md 第 1 章一致）
    TIER1_REQUIRED_N_PASS = 12
    CONSECUTIVE_GO_TARGET = 3
    STAGE1_CAPITAL_USD = 1000.0
    STAGE1_POSITION_CAP_PCT = 1.0  # 本金 1%
    STAGE1_SYMBOLS = ["BTC_USDT"]

    def __init__(self, sandbox_dir: Path | str) -> None:
        """初始化准备清单.

        Args:
            sandbox_dir: sandbox_trading 根目录路径
        """
        self.sandbox_dir = Path(sandbox_dir)
        self.go_nogo_path = self.sandbox_dir / "_v97_go_nogo_judgment.json"
        self.v999_path = self.sandbox_dir / "_v999_auto_review_report.json"
        self.tier2_state_path = (
            self.sandbox_dir / "tier2_oos_data" / "tier2_state.json"
        )

    # ------------------------------------------------------------------
    # 内部辅助: 安全读取 JSON
    # ------------------------------------------------------------------
    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """安全加载 JSON 文件（不存在或损坏返回 None）.

        Args:
            path: JSON 文件路径

        Returns:
            解析后的 dict，失败返回 None
        """
        if not path.exists():
            logger.warning("JSON 文件不存在: %s", path)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.error("JSON 解析失败 %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # 准入 1 + 准入 3-10: Tier 1 12/12 + G1-G10
    # ------------------------------------------------------------------
    def verify_tier1_passed(self) -> Dict[str, Any]:
        """验证 v97 Tier 1 12/12 通过（准入项 1 + G1-G10）.

        读取 `_v97_go_nogo_judgment.json`:
            - tier1_pass == True
            - tier1_n_pass == 12
            - go_nogo_checks 中 G1-G10 全部 passed=True

        Returns:
            验证结果 dict（含 sub_checks 明细）
        """
        go_nogo = self._load_json(self.go_nogo_path)
        if go_nogo is None:
            return {
                "name": "Tier 1 12/12 通过",
                "passed": False,
                "value": None,
                "target": "tier1_pass=True, tier1_n_pass=12",
                "detail": f"无法加载 {self.go_nogo_path.name}",
                "sub_checks": [],
            }

        tier1_pass = bool(go_nogo.get("tier1_pass", False))
        tier1_n_pass = int(go_nogo.get("tier1_n_pass", 0))
        go_checks = go_nogo.get("go_nogo_checks", {})
        summary = go_nogo.get("summary", {})

        # 逐项验证 G1-G10（避免综合判定掩盖单项失败，ERR-v93bug 教训）
        sub_checks: List[Dict[str, Any]] = []
        g_all_passed = True

        # L7 审查 WARN-B 修复: 显式校验 G1-G10 全部存在（避免缺失项静默通过）
        REQUIRED_G_KEYS = {
            "G1_ann_sim_30pct", "G2_max_dd_15pct", "G3_win_rate_55pct",
            "G4_sharpe_1_5", "G5_mlp_5pct", "G6_gap_15pct",
            "G7_n_symbols_10", "G8_years_1_5", "G9_stress_test_pass",
            "G10_df_0_5",
        }
        missing_g_keys = REQUIRED_G_KEYS - set(go_checks.keys())
        if missing_g_keys:
            sub_checks.append({
                "name": "MISSING_G_CHECKS",
                "passed": False,
                "value": sorted(missing_g_keys),
                "target": "G1-G10 all present",
                "detail": f"缺失检查项: {sorted(missing_g_keys)}",
            })
            g_all_passed = False

        for g_key, g_data in go_checks.items():
            g_passed = bool(g_data.get("passed", False))
            g_value = g_data.get("value")
            g_target = g_data.get("target", "")
            g_detail = g_data.get("detail", "")
            sub_checks.append({
                "name": g_key,
                "passed": g_passed,
                "value": g_value,
                "target": g_target,
                "detail": g_detail,
            })
            if not g_passed:
                g_all_passed = False

        # 综合判定: tier1_pass + n_pass=12 + G1-G10 全过
        passed = tier1_pass and (tier1_n_pass >= self.TIER1_REQUIRED_N_PASS) and g_all_passed

        # 准入项 2（极端行情 5/5）用 G9 stress_test_pass 近似验证
        stress_check = go_checks.get("G9_stress_test_pass", {})
        stress_passed = bool(stress_check.get("passed", False))

        return {
            "name": "Tier 1 12/12 通过",
            "passed": passed,
            "value": {
                "tier1_pass": tier1_pass,
                "tier1_n_pass": tier1_n_pass,
                "g_all_passed": g_all_passed,
                "extreme_market_approx": stress_passed,
            },
            "target": "tier1_pass=True, tier1_n_pass=12, G1-G10 all passed",
            "detail": (
                f"tier1_pass={tier1_pass}, n_pass={tier1_n_pass}/"
                f"{self.TIER1_REQUIRED_N_PASS}, G1-G10={'all_pass' if g_all_passed else 'FAIL'}; "
                f"准入项2(极端5/5)用 G9 stress_test_pass={stress_passed} 近似, "
                f"建议运行 _v97_extreme_market_test.py 确认 5/5"
            ),
            "sub_checks": sub_checks,
            "summary_metrics": {
                "ann": summary.get("ann"),
                "max_dd": summary.get("max_dd"),
                "win_rate": summary.get("win_rate"),
                "sharpe": summary.get("sharpe"),
                "mlp": summary.get("mlp"),
                "gap": summary.get("gap"),
                "n_symbols": summary.get("n_symbols"),
                "years_span": summary.get("years_span"),
                "stress_pass": summary.get("stress_pass"),
                "df": summary.get("df"),
            },
        }

    # ------------------------------------------------------------------
    # 准入 11: v999 复盘闭环完整
    # ------------------------------------------------------------------
    def verify_v999_closed_loop(self) -> Dict[str, Any]:
        """验证 v999 复盘闭环完整（准入项 11）.

        读取 `_v999_auto_review_report.json`:
            - closed_loop_verification.closed_loop_complete == True
            - closed_loop_verification.new_patterns_count >= 1

        Returns:
            验证结果 dict
        """
        v999 = self._load_json(self.v999_path)
        if v999 is None:
            return {
                "name": "v999 复盘闭环完整",
                "passed": False,
                "value": None,
                "target": "closed_loop_complete=True",
                "detail": f"无法加载 {self.v999_path.name}",
                "sub_checks": [],
            }

        cl = v999.get("closed_loop_verification", {})
        closed_loop_complete = bool(cl.get("closed_loop_complete", False))
        new_patterns_count = int(cl.get("new_patterns_count", 0))
        lessons_applied = int(cl.get("lessons_applied_count", 0))
        lessons_total = int(cl.get("lessons_total_count", 0))

        # 4 项子闭环逐项验证（review→kb→iteration→review 链路完整）
        sub_checks = [
            {
                "name": "review_to_kb_linked",
                "passed": bool(cl.get("review_to_kb_linked", False)),
                "detail": "复盘→经验库链路",
            },
            {
                "name": "kb_to_iteration_linked",
                "passed": bool(cl.get("kb_to_iteration_linked", False)),
                "detail": "经验库→迭代链路",
            },
            {
                "name": "iteration_to_review_linked",
                "passed": bool(cl.get("iteration_to_review_linked", False)),
                "detail": "迭代→复盘链路",
            },
            {
                "name": "recommendations_cite_lessons",
                "passed": bool(cl.get("recommendations_cite_lessons", False)),
                "detail": "建议引用教训",
            },
        ]
        all_linked = all(sc["passed"] for sc in sub_checks)

        # 闭环完整 = closed_loop_complete + 4 项子链路 + 至少 1 个新模式
        passed = closed_loop_complete and all_linked and new_patterns_count >= 1

        return {
            "name": "v999 复盘闭环完整",
            "passed": passed,
            "value": {
                "closed_loop_complete": closed_loop_complete,
                "new_patterns_count": new_patterns_count,
                "lessons_applied": lessons_applied,
                "lessons_total": lessons_total,
                "all_sub_links_passed": all_linked,
            },
            "target": "closed_loop_complete=True, new_patterns_count>=1, 4 sub-links all True",
            "detail": (
                f"closed_loop={closed_loop_complete}, "
                f"new_patterns={new_patterns_count}, "
                f"lessons={lessons_applied}/{lessons_total} applied, "
                f"sub_links={'all_pass' if all_linked else 'FAIL'}"
            ),
            "sub_checks": sub_checks,
        }

    # ------------------------------------------------------------------
    # 准入 12: 3 月滚动窗口
    # ------------------------------------------------------------------
    def verify_3month_rolling_window(self) -> Dict[str, Any]:
        """验证 3 月滚动窗口 consecutive_go_months >= 3（准入项 12）.

        读取 `tier2_oos_data/tier2_state.json`:
            - consecutive_go_months >= 3
            - tier2_passed == True

        Returns:
            验证结果 dict
        """
        tier2 = self._load_json(self.tier2_state_path)
        if tier2 is None:
            return {
                "name": "3 月滚动窗口模拟验证",
                "passed": False,
                "value": None,
                "target": f"consecutive_go_months>={self.CONSECUTIVE_GO_TARGET}",
                "detail": f"无法加载 {self.tier2_state_path.name}",
                "sub_checks": [],
            }

        consec_go = int(tier2.get("consecutive_go_months", 0))
        tier2_passed = bool(tier2.get("tier2_passed", False))
        collection_days = float(tier2.get("collection_days", 0))
        total_trades = int(tier2.get("total_trades", 0))
        total_wins = int(tier2.get("total_wins", 0))
        total_losses = int(tier2.get("total_losses", 0))
        total_live_pnl = float(tier2.get("total_live_pnl_usd", 0))

        # 逐项验证 Tier 2 子评估（避免综合判定掩盖，ERR-v93bug 教训）
        tier2_eval = tier2.get("tier2_evaluation", {})
        eval_checks = tier2_eval.get("checks", [])
        sub_checks: List[Dict[str, Any]] = []
        for chk in eval_checks:
            sub_checks.append({
                "name": chk.get("name", ""),
                "passed": bool(chk.get("passed", False)),
                "detail": chk.get("detail", ""),
            })

        # L7 审查 WARN-A 修复: eval_checks 全通过作为额外约束
        # (防 tier2_passed 综合字段与 eval_checks 逐项结果不一致)
        eval_all_passed = (
            all(bool(chk.get("passed", False)) for chk in eval_checks)
            if eval_checks else False
        )

        # 准入项 12 通过条件: consecutive_go_months >= 3 AND tier2_passed AND eval_checks 全通过
        passed = (
            consec_go >= self.CONSECUTIVE_GO_TARGET
            and tier2_passed
            and eval_all_passed
        )

        return {
            "name": "3 月滚动窗口模拟验证",
            "passed": passed,
            "value": {
                "consecutive_go_months": consec_go,
                "tier2_passed": tier2_passed,
                "collection_days": collection_days,
                "total_trades": total_trades,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "total_live_pnl_usd": total_live_pnl,
                "n_eval_pass": tier2_eval.get("n_pass", 0),
                "n_eval_total": tier2_eval.get("n_total", 0),
            },
            "target": (
                f"consecutive_go_months>={self.CONSECUTIVE_GO_TARGET}, "
                f"tier2_passed=True"
            ),
            "detail": (
                f"consec_go={consec_go}/{self.CONSECUTIVE_GO_TARGET}, "
                f"tier2_passed={tier2_passed}, "
                f"days={collection_days:.2f} (target>=90), "
                f"trades={total_trades} (target>=10), "
                f"wins/losses={total_wins}/{total_losses}, "
                f"live_pnl={total_live_pnl:.2f} USD"
            ),
            "sub_checks": sub_checks,
        }

    # ------------------------------------------------------------------
    # 准备账户配置（不实际部署）
    # ------------------------------------------------------------------
    def prepare_account_config(
        self, capital_usd: float = STAGE1_CAPITAL_USD
    ) -> Dict[str, Any]:
        """准备 Stage 1 账户配置（不实际下单）.

        关键约束:
            - 仅在 12 项准入全通过时 deployment_status="PREPARED_NOT_DEPLOYED"
            - 任一项未通过 → "BLOCKED"（避免误导用户）
            - deployment_ban_active 反映实盘禁令状态
            - 不实际下单，仅返回配置 dict 待用户审批

        Args:
            capital_usd: 账户本金（默认 $1000，Stage 1 标准）

        Returns:
            账户配置 dict
        """
        # 先运行完整准入验证
        report = self.generate_preparation_report()
        all_passed = report["all_passed"]
        n_pass = report["n_pass"]
        n_total = report["n_total"]

        # 准确反映当前状态（非 PREPARED 误导）
        if all_passed:
            deployment_status = "PREPARED_NOT_DEPLOYED"
            deployment_ban_active = False
            stage_ready = "Stage 1 准备就绪，等待用户审批"
        else:
            deployment_status = "BLOCKED"
            deployment_ban_active = True
            failed_items = [
                chk["name"] for chk in report["checks"] if not chk["passed"]
            ]
            stage_ready = (
                f"准入 {n_pass}/{n_total} 通过，"
                f"未通过: {failed_items} — 禁止部署"
            )

        config = {
            "config_version": "v97_tier2_stage1",
            "stage": "Stage 1",
            "capital_usd": capital_usd,
            "position_cap_pct": self.STAGE1_POSITION_CAP_PCT,
            "position_cap_usd": capital_usd * self.STAGE1_POSITION_CAP_PCT / 100.0,
            "symbols": self.STAGE1_SYMBOLS,
            "min_holding_bars": 1,
            "max_holding_bars": 72,
            "deployment_status": deployment_status,
            "deployment_ban_active": deployment_ban_active,
            "admission_status": f"{n_pass}/{n_total} passed",
            "stage_ready_note": stage_ready,
            "user_approval_required": True,
            "live_trading_enabled": False,  # 关键: 实盘交易关闭
            "warning": (
                "本配置仅准备不部署。用户铁律: "
                "'在未达成性能指标前不得推进实盘部署工作'"
            ),
        }
        return config

    # ------------------------------------------------------------------
    # 综合准备报告（12 项准入逐项验证）
    # ------------------------------------------------------------------
    def generate_preparation_report(self) -> Dict[str, Any]:
        """生成 Tier 2 准备综合报告（12 项准入逐项验证）.

        整合三个 verify 方法的输出，逐项列出 12 项准入状态。
        综合判定 all_passed=True 仅当全部 12 项通过。

        Returns:
            综合报告 dict
        """
        # 调用三个 verify 方法
        tier1_result = self.verify_tier1_passed()
        v999_result = self.verify_v999_closed_loop()
        rolling_result = self.verify_3month_rolling_window()

        # 构建 12 项准入清单（与 _v97_tier2_admission_checklist.md 第 1 章一致）
        checks: List[Dict[str, Any]] = []

        # 准入 1: Tier 1 12/12
        tier1_pass = tier1_result["passed"]
        checks.append({
            "no": 1,
            "name": "Tier 1 12/12 通过",
            "passed": tier1_pass,
            "value": tier1_result["value"],
            "target": tier1_result["target"],
            "detail": tier1_result["detail"],
        })

        # 准入 2: 极端行情 5/5（用 G9 stress_test_pass 近似）
        summary_metrics = tier1_result.get("summary_metrics", {})
        stress_pass = bool(summary_metrics.get("stress_pass", False))
        checks.append({
            "no": 2,
            "name": "极端行情 5/5 真通过",
            "passed": stress_pass,
            "value": stress_pass,
            "target": "5/5 (用 G9 stress_test_pass 近似)",
            "detail": (
                f"stress_pass={stress_pass} (G9 近似); "
                f"建议运行 _v97_extreme_market_test.py 确认 5/5"
            ),
        })

        # 准入 3-10: G1-G8 KPI
        g_mapping = [
            (3, "G1_ann_sim_30pct", "年化 ann >= 30%"),
            (4, "G2_max_dd_15pct", "最大回撤 max_dd <= 15%"),
            (5, "G3_win_rate_55pct", "胜率 >= 55%"),
            (6, "G4_sharpe_1_5", "夏普比率 >= 1.5"),
            (7, "G5_mlp_5pct", "月亏率 <= 5%"),
            (8, "G6_gap_15pct", "sim-live gap <= 15%"),
            (9, "G7_n_symbols_10", "交易币种数 >= 10"),
            (10, "G8_years_1_5", "数据跨度 >= 1.5 年"),
        ]
        sub_checks = tier1_result.get("sub_checks", [])
        sc_map = {sc["name"]: sc for sc in sub_checks}
        for no, g_key, desc in g_mapping:
            sc = sc_map.get(g_key, {})
            checks.append({
                "no": no,
                "name": desc,
                "passed": bool(sc.get("passed", False)),
                "value": sc.get("value"),
                "target": sc.get("target", ""),
                "detail": sc.get("detail", ""),
            })

        # 准入 11: v999 闭环
        checks.append({
            "no": 11,
            "name": "v999 复盘闭环完整",
            "passed": v999_result["passed"],
            "value": v999_result["value"],
            "target": v999_result["target"],
            "detail": v999_result["detail"],
        })

        # 准入 12: 3 月滚动窗口
        checks.append({
            "no": 12,
            "name": "3 月滚动窗口模拟验证",
            "passed": rolling_result["passed"],
            "value": rolling_result["value"],
            "target": rolling_result["target"],
            "detail": rolling_result["detail"],
        })

        n_pass = sum(1 for chk in checks if chk["passed"])
        n_total = len(checks)
        all_passed = n_pass == n_total

        # 准确反映当前状态（非 PREPARED 误导）
        if all_passed:
            deployment_status = "PREPARED_NOT_DEPLOYED"
            deployment_ban_active = False
        else:
            deployment_status = "BLOCKED"
            deployment_ban_active = True

        return {
            "report_version": "v97_tier2_preparation_v1.0",
            "all_passed": all_passed,
            "n_pass": n_pass,
            "n_total": n_total,
            "deployment_status": deployment_status,
            "deployment_ban_active": deployment_ban_active,
            "checks": checks,
            "tier1_summary": tier1_result.get("summary_metrics", {}),
            "v999_summary": v999_result.get("value", {}),
            "tier2_summary": rolling_result.get("value", {}),
            "conclusion": (
                f"准入 {n_pass}/{n_total} 通过 — "
                + ("可启动 Tier 2 Stage 1" if all_passed else "BLOCKED 禁止启动")
            ),
            "user_iron_rule": (
                "用户铁律: '在未达成性能指标前不得推进实盘部署工作' — "
                "本报告仅准备不部署，等待用户审批"
            ),
        }


# ----------------------------------------------------------------------
# 模块入口: 命令行运行生成报告
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 默认 sandbox_trading 目录
    default_dir = Path(__file__).parent
    checklist = Tier2PreparationChecklist(default_dir)
    report = checklist.generate_preparation_report()

    print("=" * 60)
    print("v97 Tier 2 迷你实盘准备清单（仅准备不部署）")
    print("=" * 60)
    print(f"准入: {report['n_pass']}/{report['n_total']} 通过")
    print(f"状态: {report['deployment_status']}")
    print(f"实盘禁令: {'激活' if report['deployment_ban_active'] else '未激活'}")
    print()
    for chk in report["checks"]:
        status = "✅" if chk["passed"] else "❌"
        print(f"  {status} 准入{chk['no']:2d}: {chk['name']}")
        print(f"      {chk['detail']}")
    print()
    print(f"结论: {report['conclusion']}")
    print(f"铁律: {report['user_iron_rule']}")

    # 生成账户配置（不实际部署）
    print()
    print("-" * 60)
    print("Stage 1 账户配置（仅准备）")
    print("-" * 60)
    config = checklist.prepare_account_config()
    print(f"  deployment_status: {config['deployment_status']}")
    print(f"  deployment_ban_active: {config['deployment_ban_active']}")
    print(f"  stage_ready_note: {config['stage_ready_note']}")
    print(f"  live_trading_enabled: {config['live_trading_enabled']}")
