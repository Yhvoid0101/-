"""
DeployBlockAdmissionMixin — Phase 7.2 架构解耦
从 evolution_loop.py 提取的部署阻断/分阶准入管理逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖: Host 需提供 self.{_err_kb_block, _recurrence_block,
    _blocking_errs, _blocking_recurrences, _last_staged_admission_report,
    _extreme_test_block, _last_final_metrics, _last_review_result, tier_gate}
  - 零循环依赖: 仅依赖 logger + typing

来源:
  - v92 Phase 4 五重 block 统一准入 (ERR-KB + 复发 + Tier1/2/3 + 极端测试)
  - v614-698 AutoParamApplier 基线指标闭环
  - v704 baseline 硬编码兜底修复
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DeployBlockAdmissionMixin:
    """部署阻断/分阶准入管理 Mixin (Phase 7.2 提取)

    提供:
      - get_deploy_block_state(): 查询当前部署阻断状态
      - clear_blocks_after_mitigation(): 修复后清除阻断
      - get_admission_state(): 统一准入接口 (五重 block)
      - _build_applier_baseline_metrics(): 构建基线指标

    Host 鸭子类型依赖 (在 __init__ 中初始化):
      - self._err_kb_block: bool
      - self._recurrence_block: bool
      - self._blocking_errs: list
      - self._blocking_recurrences: list
      - self._last_staged_admission_report: Optional[Dict]
      - self._extreme_test_block: bool
      - self._last_final_metrics: Dict
      - self._last_review_result: Optional[Dict]
      - self.tier_gate: Optional[Any] (需有 check_admission 方法)
    """

    def get_deploy_block_state(self) -> Dict[str, Any]:
        """返回当前部署阻断状态（供跨 agent 共享 + 外部查询）

        本方法让外部 agent 可主动查询当前 block 状态，无需等待 loop_end。
        解决"无方法查询 block 状态 → 外部 agent 无法主动询问 → 跨 agent 协作断裂"问题。

        Returns:
            Dict 含:
            - blocked: bool — 是否被阻断（ERR-KB 或 recurrence 任一命中即 True）
            - err_kb_block: bool — ERR 知识库命中状态
            - recurrence_block: bool — 复发警报触发状态
            - blocking_errs: List[Dict] — 阻断的 ERR 条目（err_id/title/severity/fix）
            - blocking_recurrences: List[str] — 阻断的复发警报列表
            - block_reasons: List[str] — 人类可读的阻断原因摘要
        """
        blocked = self._err_kb_block or self._recurrence_block
        reasons: List[str] = []
        if self._err_kb_block:
            reasons.append(
                f"ERR_KB_BLOCK: {len(self._blocking_errs)} 条 ERR 命中 — "
                + ", ".join(e["err_id"] for e in self._blocking_errs)
            )
        if self._recurrence_block:
            reasons.append(
                f"RECURRENCE_BLOCK: {len(self._blocking_recurrences)} 条复发警报"
            )
        return {
            "blocked": blocked,
            "err_kb_block": self._err_kb_block,
            "recurrence_block": self._recurrence_block,
            "blocking_errs": list(self._blocking_errs),
            "blocking_recurrences": list(self._blocking_recurrences),
            "block_reasons": reasons,
        }

    def clear_blocks_after_mitigation(self, mitigation_evidence: Dict[str, Any]) -> bool:
        """清除部署阻断状态（当上游 agent/用户提供修复证据时调用）

        本方法不会无条件清除，必须提供结构化修复证据。
        解决"无方法清除 block → 修复后无法解除阻断 → 部署永久卡死"问题。

        Args:
            mitigation_evidence: Dict 含:
                - err_ids_resolved: List[str] — 已修复的 ERR-ID 列表
                - recurrence_resolved: bool — 复发问题是否已修复
                - mitigated_by: str — 修复者（agent 名或 "user"）
                - mitigation_actions: List[str] — 执行的修复动作

        Returns:
            bool — True=成功清除所有 block; False=证据不足或仅部分清除

        Raises:
            ValueError — mitigation_evidence 缺少必需字段时
        """
        if not isinstance(mitigation_evidence, dict):
            raise ValueError("mitigation_evidence 必须是 dict")

        required = ["err_ids_resolved", "recurrence_resolved", "mitigated_by"]
        missing = [f for f in required if f not in mitigation_evidence]
        if missing:
            raise ValueError(f"mitigation_evidence 缺少必需字段: {missing}")

        _err_resolved = mitigation_evidence.get("err_ids_resolved", [])
        _rec_resolved = mitigation_evidence.get("recurrence_resolved", False)
        _by = mitigation_evidence.get("mitigated_by", "unknown")
        _actions = mitigation_evidence.get("mitigation_actions", [])

        _cleared_errs: List[str] = []
        _remaining_errs: list = []
        if self._err_kb_block:
            for err in self._blocking_errs:
                if err["err_id"] in _err_resolved:
                    _cleared_errs.append(err["err_id"])
                else:
                    _remaining_errs.append(err)
            if _remaining_errs:
                logger.warning(
                    "[DeployBlock] 部分清除: %d ERR 已修复, %d 未修复: %s",
                    len(_cleared_errs), len(_remaining_errs),
                    [e["err_id"] for e in _remaining_errs],
                )
                self._blocking_errs = _remaining_errs
                # ERR block 仅在全部 ERR 已修复时才清除
            else:
                self._err_kb_block = False
                self._blocking_errs = []
                logger.info(
                    "[DeployBlock] ERR-KB block 已清除 (mitigated_by=%s, actions=%d, errs=%s)",
                    _by, len(_actions), _cleared_errs,
                )

        if self._recurrence_block and _rec_resolved:
            _n_rec = len(self._blocking_recurrences)
            self._recurrence_block = False
            self._blocking_recurrences = []
            logger.info(
                "[DeployBlock] recurrence block 已清除 (mitigated_by=%s, n_alerts=%d)",
                _by, _n_rec,
            )
        elif self._recurrence_block and not _rec_resolved:
            logger.warning(
                "[DeployBlock] recurrence block 未清除: recurrence_resolved=False"
            )

        _all_cleared = (not self._err_kb_block) and (not self._recurrence_block)
        if _all_cleared:
            logger.info("[DeployBlock] 所有 block 已清除, can_deploy 恢复")
        return _all_cleared

    def get_admission_state(self) -> Dict[str, Any]:
        """统一准入状态接口（消除 tier_gate 与 staged_admission 双重判定）

        v92 Phase 4 核心方法: 整合五重 block 来源为单一权威接口。
        用户铁律: "分阶测试准入规则（模拟→迷你实盘→标准实盘），不达性能指标禁止向下游部署"
                  "永远永远不要出现模拟牛逼，实盘亏损的情况"

        整合的五重 block 来源:
          1. ERR-KB 命中 (self._err_kb_block) — 教训库自动应用
          2. 复发警报 (self._recurrence_block) — 永不二过
          3. staged_admission 的 Tier1/2/3 评估 — 分阶准入
          4. 极端测试退化 (self._extreme_test_block) — Phase 6 极端行情回归
          5. (staged_admission 为权威源, tier_gate 降级为 fallback)

        Returns:
            {
                "can_deploy": bool,            # 统一部署准许（五重 block 全通过才 True）
                "tier": int,                   # 当前达到的 Tier 等级 (0/1/2/3)
                "tier1_pass": bool,            # 模拟回测达标
                "tier2_pass": bool,            # 迷你实盘连续达标
                "tier3_pass": bool,            # 标准实盘验证达标
                "err_kb_block": bool,          # ERR 教训库命中
                "recurrence_block": bool,      # 复发警报活跃
                "extreme_test_block": bool,    # 极端行情测试退化
                "block_reasons": List[str],    # 阻断原因清单（人类可读）
                "staged_admission_report": Optional[Dict],  # 缓存的完整报告
            }
        """
        _block_state = self.get_deploy_block_state()
        _sa_report = getattr(self, "_last_staged_admission_report", None)
        _extreme_block = getattr(self, "_extreme_test_block", False)

        if _sa_report is not None:
            # v4.0 Phase 7 Task #30.1: 修复 key-path Bug
            # 原Bug: 读取顶层 tier1_pass/tier2_pass/tier3_pass/current_tier, 但实际结构为
            #   err_entry.tier1_pass (staged_admission_gate.py:615)
            #   admission_decision.current_tier (staged_admission_gate.py:600, 字符串"Tier 1")
            # 后果: _can_deploy 恒 False, _v4_deployment_ban_active 恒 True (Task #29.3修复失效)
            _err_entry = _sa_report.get("err_entry", {})
            _admission = _sa_report.get("admission_decision", {})
            _tier1_pass = _err_entry.get("tier1_pass", False)
            _tier2_pass = _err_entry.get("tier2_pass", False)
            _tier3_pass = _err_entry.get("tier3_pass", False)
            # current_tier 是字符串 "Tier 0"/"Tier 1"/"Tier 2"/"Tier 3", 转为 int
            _tier_str = _admission.get("current_tier", "Tier 0")
            try:
                _tier = int(str(_tier_str).replace("Tier ", "").strip()) if _tier_str is not None else 0
            except (ValueError, TypeError):
                _tier = 0
        else:
            # 退化: 使用 tier_gate 作为 fallback
            _final_metrics = getattr(self, "_last_final_metrics", None) or {}
            if self.tier_gate is not None and _final_metrics:
                try:
                    _admission = self.tier_gate.check_admission(_final_metrics)
                    _tier1_pass = _admission.get("can_deploy", False)
                    _tier2_pass = False
                    _tier3_pass = False
                    _tier = 1 if _tier1_pass else 0
                except Exception:
                    _tier1_pass = _tier2_pass = _tier3_pass = False
                    _tier = 0
            else:
                _tier1_pass = _tier2_pass = _tier3_pass = False
                _tier = 0

        # v92 Phase 4: 五重 block 统一判定
        _can_deploy = (
            _tier1_pass and _tier2_pass and _tier3_pass
            and not _block_state["blocked"]
            and not _extreme_block
        )

        _reasons = list(_block_state["block_reasons"])
        if not _tier1_pass:
            _reasons.append(f"Tier1 未通过 (current_tier={_tier})")
        if not _tier2_pass:
            _reasons.append("Tier2 未通过 (需迷你实盘连续达标)")
        if not _tier3_pass:
            _reasons.append("Tier3 未通过 (需标准实盘验证)")
        if _extreme_block:
            _reasons.append("ExtremeMarket 测试退化 (extreme_pass_count < 4)")

        return {
            "can_deploy": _can_deploy,
            "tier": _tier,
            "tier1_pass": _tier1_pass,
            "tier2_pass": _tier2_pass,
            "tier3_pass": _tier3_pass,
            "err_kb_block": _block_state["err_kb_block"],
            "recurrence_block": _block_state["recurrence_block"],
            "extreme_test_block": _extreme_block,
            "block_reasons": _reasons,
            "staged_admission_report": _sa_report,
        }

    def _build_applier_baseline_metrics(self) -> Dict[str, Any]:
        """构建 AutoParamApplier.run_iteration 所需的基线指标

        v614-698 闭环下半部分辅助方法 (ERR-20260702-v614-v98)
        优先级:
          1. _last_final_metrics (validate_anti_overfitting 产出的最新指标)
          2. _last_staged_admission_report.metrics (Tier1 评估指标)
          3. _last_review_result (AutoReviewEngine 复盘指标)
          4. 硬编码 v97 已知指标 (最终兜底, 避免空指标导致 run_iteration 失败)

        Returns:
            {ann_sim, max_drawdown, win_rate, sharpe, profit_factor,
             max_consec, n_trades, gap, ...}
        """
        # 1. 优先从 _last_final_metrics 提取
        _fm = getattr(self, "_last_final_metrics", None) or {}
        if _fm and _fm.get("ann", 0) > 0:
            return {
                "ann_sim": float(_fm.get("ann", 0)),
                "max_drawdown": float(_fm.get("max_dd", _fm.get("max_drawdown", 0))),
                "win_rate": float(_fm.get("win_rate", 0)),
                "sharpe": float(_fm.get("sharpe", 0)),
                "profit_factor": float(_fm.get("profit_factor", 1.75)),
                "max_consec": int(_fm.get("max_consec", 0)),
                "n_trades": int(_fm.get("n_trades", 0)),
                "gap": float(_fm.get("gap", 0)),
            }

        # 2. 从 staged_admission_report 提取
        _sa = getattr(self, "_last_staged_admission_report", None) or {}
        _sa_metrics = _sa.get("metrics", {}) if isinstance(_sa, dict) else {}
        if _sa_metrics and _sa_metrics.get("ann", 0) > 0:
            return {
                "ann_sim": float(_sa_metrics.get("ann", 0)),
                "max_drawdown": float(_sa_metrics.get("max_dd", 0)),
                "win_rate": float(_sa_metrics.get("win_rate", 0)),
                "sharpe": float(_sa_metrics.get("sharpe", 0)),
                "profit_factor": float(_sa_metrics.get("profit_factor", 1.75)),
                "max_consec": int(_sa_metrics.get("max_consec", 0)),
                "n_trades": int(_sa_metrics.get("n_trades", 0)),
                "gap": float(_sa_metrics.get("gap", 0)),
            }

        # 3. 从 review_result 提取
        _rr = getattr(self, "_last_review_result", None) or {}
        if isinstance(_rr, dict) and _rr.get("max_drawdown_pct", 0) > 0:
            return {
                "ann_sim": float(_rr.get("ann_sim", 0)),
                "max_drawdown": float(_rr.get("max_drawdown_pct", 0)),
                "win_rate": float(_rr.get("win_rate", 0)),
                "sharpe": float(_rr.get("sharpe", 0)),
                "profit_factor": float(_rr.get("profit_factor", 1.75)),
                "max_consec": int(_rr.get("max_consec", 0)),
                "n_trades": int(_rr.get("n_trades", 0)),
                "gap": float(_rr.get("gap", 0)),
            }

        # 4. v704 修复 (ERR-v704-baseline-hardcode): 硬编码兜底改为全0 + 缺失标记
        # 原硬编码值 (ann_sim=91.12, n_trades=269, gap=1.23) 是 v704 修复前的虚假数据
        # 修复后真实值完全不同 (ann=10.90%, gap=42.12%), 传播虚假数据会误导下游分析
        # 全0表示"基线缺失", 调用方用 .get(key, 0) 已能处理, _v704_baseline_missing 标记可检测
        # 用户铁律: "永远永远不要出现模拟牛逼,实盘亏损的情况" — 传播虚假KPI=模拟牛逼
        return {
            "ann_sim": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "max_consec": 0,
            "n_trades": 0,
            "gap": 0.0,
            "_v704_baseline_missing": True,
        }
