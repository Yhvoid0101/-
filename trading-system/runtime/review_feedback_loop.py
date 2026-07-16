#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复盘-经验库联动闭环 (Review-Experience Feedback Loop)
=====================================================
短板 #1 补齐: 解决"复盘是摆设"问题 — 将复盘推荐建议自动转化为行动

核心痛点:
  - auto_review_system.py 的 _generate_recommendations() 只存报告不回喂
  - 错误知识库 (1500-error-knowledge-base.md) 13 条手动维护, 从不自动入库
  - 复盘发现的模式 (如 ADA 负 alpha, ADX<20 信号亏损) 未跨版本追踪

三层闭环设计:
  Layer A: 复盘报告 → 错误知识库自动入库 (检测新模式 → 格式化入库)
  Layer B: 复盘报告 → 策略参数调整建议 (生成可应用的 param_patch.json)
  Layer C: 跨版本复发追踪 (同一模式 3+ 版本复发 → 升级为 BLOCK)

设计原则 (用户铁律):
  - "复盘不是摆设" — 每条推荐必须产出具体行动
  - "错误库使用起来" — 自动入库, 跨版本匹配
  - "永不二过" — 同一错误第 2 次出现即 WARN, 第 3 次 BLOCK

使用方式:
  from review_feedback_loop import ReviewFeedbackLoop
  loop = ReviewFeedbackLoop()
  actions = loop.process_review_report("path/to/review.json")
  # actions.errors_recorded → 已入库的错误 ID 列表
  # actions.param_patch → 参数调整建议
  # actions.follow_up_tasks → 后续任务
  # actions.recurrence_alerts → 跨版本复发警报

CLI:
  python review_feedback_loop.py --review _review_reports/auto_review_xxx.json
  python review_feedback_loop.py --batch  # 处理所有未消化的复盘报告
  python review_feedback_loop.py --track-recurrence  # 跨版本复发分析
"""
from __future__ import annotations

import re
import json
import hashlib
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

# ============================================================================
# 路径常量
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # 2026/
ERROR_KB_PATH = PROJECT_ROOT / ".trae" / "rules" / "1500-error-knowledge-base.md"
REVIEW_REPORTS_DIR = SCRIPT_DIR / "_review_reports"
FEEDBACK_OUTPUT_DIR = SCRIPT_DIR / "_feedback_actions"
PARAM_PATCH_DIR = SCRIPT_DIR / "_param_patches"
RECURRENCE_DB_PATH = FEEDBACK_OUTPUT_DIR / "recurrence_db.json"

# 复盘报告消化日志 (避免重复处理)
DIGEST_LOG_PATH = FEEDBACK_OUTPUT_DIR / "digested_reports.json"

# ============================================================================
# 日志
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ReviewFeedback] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ReviewFeedbackLoop")

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ErrorEntry:
    """错误知识库条目 (匹配 1500-error-knowledge-base.md 格式)"""
    err_id: str                    # ERR-YYYYMMDD-NNN
    title: str                     # 简短标题
    level: str = "MEDIUM"          # CRITICAL/HIGH/MEDIUM/LOW
    discovered_at: str = ""        # 发现时间
    phenomenon: str = ""           # 现象描述
    root_cause: str = ""           # 根因分析
    fix_direction: str = ""        # 修复方向
    prevention_rule: str = ""      # 预防规则
    pattern: str = ""              # 模式归类
    source_version: str = ""       # 来源版本
    fingerprint: str = ""          # 指纹 hash (用于跨版本去重)

    def to_markdown(self) -> str:
        """渲染为 markdown (匹配 1500-error-knowledge-base.md 现有格式)"""
        return f"""

---

## {self.err_id} — {self.title}

- **等级**: {self.level}
- **发现时间**: {self.discovered_at}
- **现象**: {self.phenomenon}
- **根因**: {self.root_cause}
- **修复方向**: {self.fix_direction}
- **预防规则**: {self.prevention_rule}
- **模式**: {self.pattern}
- **来源版本**: {self.source_version}
- **指纹**: `{self.fingerprint}`
"""

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict (供 GlobalKnowledgeBase.add_knowledge_entry 使用)

        v598 Phase B / Task#89: 复盘-经验库联动闭环
        之前 _append_to_kb 只写本地 markdown 文件, 不写 GlobalKnowledgeBase
        现在同步写入全局知识库, 实现 8 agent 共享教训
        """
        return {
            "err_id": self.err_id,
            "title": self.title,
            "level": self.level,
            "discovered_at": self.discovered_at,
            "phenomenon": self.phenomenon,
            "root_cause": self.root_cause,
            "fix_direction": self.fix_direction,
            "prevention_rule": self.prevention_rule,
            "pattern": self.pattern,
            "source_version": self.source_version,
            "fingerprint": self.fingerprint,
            "category": "review_feedback",  # 供 get_knowledge_entries(category=) 查询
        }


@dataclass
class ParamAdjustment:
    """参数调整建议"""
    target_symbol: str             # 目标币种 ("GLOBAL" 表示全局)
    param_name: str                # 参数名 (如 atr_tp_mult, max_pos)
    current_value: float           # 当前值
    suggested_value: float         # 建议值
    reason: str                    # 调整原因
    confidence: float = 0.5        # 置信度 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)  # 数据证据
    auto_apply: bool = False       # 是否建议自动应用 (高风险调整需人工确认)


@dataclass
class FollowUpTask:
    """后续任务"""
    task_id: str
    title: str
    description: str
    priority: str = "MEDIUM"       # CRITICAL/HIGH/MEDIUM/LOW
    category: str = ""             # backtest/code_change/research/deployment
    estimated_effort: str = ""     # 估算工作量
    dependencies: List[str] = field(default_factory=list)
    source_recommendation: str = ""


@dataclass
class RecurrenceAlert:
    """跨版本复发警报"""
    fingerprint: str
    pattern: str
    occurrences: List[Dict[str, str]] = field(default_factory=list)  # [{version, err_id}]
    severity: str = "WARN"         # WARN (2次) / BLOCK (3+次)
    message: str = ""


@dataclass
class FeedbackActions:
    """复盘反馈行动汇总"""
    version: str = ""
    timestamp: str = ""
    errors_recorded: List[str] = field(default_factory=list)  # 新入库的 ERR-ID
    errors_existing: List[str] = field(default_factory=list)   # 已存在的匹配错误
    param_adjustments: List[ParamAdjustment] = field(default_factory=list)
    follow_up_tasks: List[FollowUpTask] = field(default_factory=list)
    recurrence_alerts: List[RecurrenceAlert] = field(default_factory=list)
    summary: str = ""


# ============================================================================
# 核心引擎
# ============================================================================

class ReviewFeedbackLoop:
    """复盘-经验库联动闭环引擎

    工作流程:
      1. 读取复盘报告 (JSON)
      2. 扫描 KPI 失败项 + 各币种 issues → 提取错误模式
      3. 对每个错误模式生成指纹 (fingerprint) → 查询历史是否已入库
      4. 新错误 → 自动追加到 1500-error-knowledge-base.md
      5. 已存在错误 → 记录复发, 升级严重度 (2次WARN / 3+次BLOCK)
      6. 基于错误模式生成参数调整建议
      7. 基于推荐建议生成后续任务
      8. 输出 FeedbackActions JSON
    """

    # 错误模式检测规则 (pattern_name → 检测条件 → 修复方向模板)
    ERROR_PATTERNS = {
        "single_symbol_negative_ev": {
            "description": "单币种 EV 持续为负",
            "level": "HIGH",
            "detection": lambda sr: sr.get("ev_per_trade_pct", 0) < -0.3 and sr.get("total_trades", 0) >= 10,
            "phenomenon_tpl": "{symbol} EV={ev:.4f}% (持续为负), {trades}笔交易, 总PnL={pnl:.2f}%",
            "root_cause_tpl": "该币种存在结构性负 alpha, 策略信号方向与实际走势相反",
            "fix_direction_tpl": "考虑禁用该币种 (参考 ERR-20260629-049 ADA 案例的负 alpha 禁用模式)",
            "prevention_tpl": "每个币种加入组合前必须独立验证 alpha 符号, 连续3版本为负 alpha → 永久禁用",
            "pattern_label": "负alpha币种未禁用",
        },
        "extreme_consecutive_losses": {
            "description": "极端连续亏损 (超3次)",
            "level": "HIGH",
            "detection": lambda depth: depth.get("max_consecutive_losses", 0) > 3,
            "phenomenon_tpl": "最大连续亏损 {cl} 次 (超阈值3), 月度亏损概率 {mlp:.1f}%",
            "root_cause_tpl": "止损/风控逻辑未能在连续亏损后有效降仓或暂停",
            "fix_direction_tpl": "实施 Layer2 风控: 月内 running PnL 触发阈值后降仓 (参考 v522 ERR-116)",
            "prevention_tpl": "consec_loss ≤ 3 是物理极限, 超过即说明风控层失效, 必须立即修复",
            "pattern_label": "Layer2风控失效",
        },
        "low_winrate_high_pf_paradox": {
            "description": "低胜率高盈亏比悖论 (胜率<45% 但 PF>2)",
            "level": "MEDIUM",
            "detection": lambda agg: agg.get("win_rate", 0) < 45 and agg.get("profit_factor", 0) > 2.0,
            "phenomenon_tpl": "胜率 {wr:.1f}% < 45% 但盈亏比 {pf:.2f} > 2.0, 组合依赖少数大盈利交易",
            "root_cause_tpl": "策略为典型趋势跟踪型, 盈利交易平均收益远大于亏损, 但胜率不稳定",
            "fix_direction_tpl": "在保持盈亏比前提下提升胜率: 入场信号质量改进, ADX/趋势强度过滤",
            "prevention_tpl": "低胜率高盈亏比策略对单笔大亏损敏感, 必须配合单亏≤2%本金硬约束",
            "pattern_label": "胜率-盈亏比失衡",
        },
        "monthly_loss_prob_exceeded": {
            "description": "月度亏损概率超标 (>5%实盘 / >35%模拟)",
            "level": "CRITICAL",
            "detection": lambda agg: agg.get("monthly_loss_prob", 0) > 5.0,
            "phenomenon_tpl": "月度亏损概率 {mlp:.1f}% 超过 5% 实盘阈值",
            "root_cause_tpl": "策略在某些月份集中亏损, 未能在月内实施动态风控",
            "fix_direction_tpl": "实施月内 running PnL 诊断: 月内累计亏损达阈值后降仓/暂停",
            "prevention_tpl": "月度亏损概率 > 5% 禁止实盘部署 (用户硬约束)",
            "pattern_label": "月度亏损概率超标",
        },
        "high_max_drawdown": {
            "description": "最大回撤超标 (>15%)",
            "level": "CRITICAL",
            "detection": lambda agg: agg.get("max_drawdown", 0) > 15.0,
            "phenomenon_tpl": "最大回撤 {dd:.1f}% 超过 15% 阈值",
            "root_cause_tpl": "仓位过重或止损过宽, 单笔/连续亏损累计导致回撤扩大",
            "fix_direction_tpl": "1) 缩减 max_pos 仓位 2) 收紧 ATR 止损倍数 3) 实施 portfolio-level 回撤熔断",
            "prevention_tpl": "最大回撤 > 15% 禁止实盘部署 (用户硬约束)",
            "pattern_label": "回撤超标",
        },
        "adx_weak_signal_loss": {
            "description": "ADX<20 弱趋势信号亏损",
            "level": "HIGH",
            "detection": lambda depth: _check_adx_weak_loss(depth),
            "phenomenon_tpl": "ADX<20 信号 {count}笔 ({pct:.1f}%), 总PnL={pnl:.2f}%, 平均PnL={avg:.3f}%",
            "root_cause_tpl": "策略在弱趋势市场无优势, 信号被噪声主导",
            "fix_direction_tpl": "添加 ADX≥20 过滤器 (参考 v575 ERR-105 ADX分级分析)",
            "prevention_tpl": "弱趋势市场信号必须过滤, ADX<20 信号占比 > 50% 即触发警报",
            "pattern_label": "弱趋势信号未过滤",
        },
    }

    def __init__(
        self,
        error_kb_path: Path = ERROR_KB_PATH,
        output_dir: Path = FEEDBACK_OUTPUT_DIR,
        global_kb: Any = None,  # v598 Phase B / Task#89: 注入 GlobalKnowledgeBase
    ):
        self.error_kb_path = Path(error_kb_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "recurrence_db.json").touch(exist_ok=True)
        self._kb_cache: Optional[str] = None
        self._existing_err_ids: Set[str] = set()
        self._existing_fingerprints: Dict[str, str] = {}  # fingerprint → err_id
        # v598 Phase B / Task#89: 全局知识库引用 (可选, 用于复盘→经验库联动闭环)
        # 用户铁律: "自动化复盘必须与经验库联动形成闭环"
        self.global_kb = global_kb
        self._load_existing_kb()

    # ========================================================================
    # Layer A: 错误知识库自动入库
    # ========================================================================

    def _load_existing_kb(self):
        """加载现有错误知识库, 建立 fingerprint → ERR-ID 索引

        按区块扫描: 每个 ## ERR-XXX 标题与其下的 **指纹**: 字段配对
        """
        if not self.error_kb_path.exists():
            logger.warning(f"错误知识库不存在: {self.error_kb_path}")
            return

        content = self.error_kb_path.read_text(encoding="utf-8")
        self._kb_cache = content

        # 按区块提取: 每个 ERR section 包含标题 + 指纹字段
        # 使用正则匹配整个 section (从 ## ERR- 到下一个 ## 或文件末尾)
        section_pattern = re.compile(
            r"^## (ERR-[A-Za-z0-9\-]+)\s*[—-]\s*(.+?)(?=\n##\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for m in section_pattern.finditer(content):
            err_id = m.group(1)
            title = m.group(2).strip().split("\n")[0]  # 只取标题第一行
            section_body = m.group(2)

            self._existing_err_ids.add(err_id)

            # 1. 基于标题的 fingerprint (兼容旧格式)
            title_fp = self._fingerprint(title)
            if title_fp not in self._existing_fingerprints:
                self._existing_fingerprints[title_fp] = err_id

            # 2. 显式 fingerprint 字段 (新格式, 优先级更高)
            fp_match = re.search(r"\*\*指纹\*\*:\s*`([a-f0-9]{16})`", section_body)
            if fp_match:
                explicit_fp = fp_match.group(1)
                # 显式指纹直接覆盖 (更精确)
                self._existing_fingerprints[explicit_fp] = err_id

        logger.info(f"已加载错误知识库: {len(self._existing_err_ids)} 个 ERR-ID, "
                    f"{len(self._existing_fingerprints)} 个指纹")

    @staticmethod
    def _fingerprint(text: str) -> str:
        """生成 16 字符指纹 (用于跨版本去重)"""
        # 归一化: 小写, 去除币种名/数字/特殊字符
        normalized = re.sub(r"[^a-z\u4e00-\u9fff]", "", text.lower())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]

    def _generate_err_id(self) -> str:
        """生成新的 ERR-ID: ERR-YYYYMMDD-NNN"""
        date_str = datetime.now().strftime("%Y%m%d")
        prefix = f"ERR-{date_str}-"
        # 找今日已存在的最大编号
        existing_nums = [
            int(e.split("-")[-1]) for e in self._existing_err_ids
            if e.startswith(prefix)
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        err_id = f"{prefix}{next_num:03d}"
        self._existing_err_ids.add(err_id)
        return err_id

    def _append_to_kb(self, entry: ErrorEntry) -> bool:
        """追加错误条目到 1500-error-knowledge-base.md

        v598 Phase B / Task#89: 同步写入 GlobalKnowledgeBase, 实现复盘→经验库联动闭环
        用户铁律: "自动化复盘必须与经验库联动形成闭环, 解决复盘与教训库闲置问题"

        Returns:
            True 如果成功写入, False 如果跳过 (已存在)
        """
        # 检查 fingerprint 是否已存在
        if entry.fingerprint and entry.fingerprint in self._existing_fingerprints:
            existing_err_id = self._existing_fingerprints[entry.fingerprint]
            logger.info(f"错误模式已存在 (ERR-ID={existing_err_id}), 跳过入库: {entry.title}")
            return False

        # 追加到本地 KB 末尾
        try:
            with open(self.error_kb_path, "a", encoding="utf-8") as f:
                f.write(entry.to_markdown())
            # 更新缓存
            self._existing_err_ids.add(entry.err_id)
            self._existing_fingerprints[entry.fingerprint] = entry.err_id
            logger.info(f"✅ 已入库(本地): {entry.err_id} — {entry.title}")
        except Exception as e:
            logger.error(f"本地入库失败: {e}")
            return False

        # v598 Phase B / Task#89: 同步写入 GlobalKnowledgeBase (如果注入)
        if self.global_kb is not None:
            try:
                ok = self.global_kb.add_knowledge_entry(entry.to_dict())
                if ok:
                    logger.info(f"✅ 已入库(全局): {entry.err_id} — {entry.title}")
                else:
                    logger.warning(f"全局入库返回 False: {entry.err_id}")
            except Exception as e:
                logger.warning(f"全局入库异常 (不影响本地): {e}")
        return True

    # ========================================================================
    # Layer B: 错误模式检测 (从复盘报告)
    # ========================================================================

    def _detect_errors_from_report(
        self, report: Dict[str, Any], version_label: str = "unknown"
    ) -> List[Tuple[str, ErrorEntry]]:
        """从复盘报告检测错误模式

        Args:
            report: 复盘报告字典
            version_label: 策略版本标签 (如 "v582")

        Returns:
            List of (pattern_name, ErrorEntry) tuples
        """
        detected: List[Tuple[str, ErrorEntry]] = []
        version = version_label

        # 检查 KPI 失败项
        kpi_check = report.get("kpi_check", {})
        agg = report.get("aggregate", {})

        # 模式1: 月度亏损概率超标 (实盘阈值5%, 模拟阈值35%)
        if kpi_check.get("monthly_loss_prob", {}).get("passed") is False:
            mlp = _get_kpi_actual(kpi_check, "monthly_loss_prob", 0.0)
            entry = self._build_error_entry(
                pattern_name="monthly_loss_prob_exceeded",
                version=version_label,
                fmt_args={"mlp": float(mlp)},
                symbol="GLOBAL",
            )
            detected.append(("monthly_loss_prob_exceeded", entry))

        # 模式2: 最大回撤超标 (>15%)
        if kpi_check.get("max_drawdown", {}).get("passed") is False:
            dd = _get_kpi_actual(kpi_check, "max_drawdown", 0.0)
            # 也尝试从 aggregate 取 (兼容字段名 dd_mean_pct)
            if dd == 0.0:
                dd = _get_agg_value(agg, "max_drawdown", "dd_mean_pct", "max_drawdown_pct", default=0.0)
            entry = self._build_error_entry(
                pattern_name="high_max_drawdown",
                version=version_label,
                fmt_args={"dd": float(dd)},
                symbol="GLOBAL",
            )
            detected.append(("high_max_drawdown", entry))

        # 模式3: 低胜率高盈亏比悖论 (胜率<45% 但 PF>2)
        wr = _get_agg_value(agg, "win_rate", "win_rate_mean", default=0.0)
        pf = _get_agg_value(agg, "profit_factor", "pf_mean", default=0.0)
        if wr < 45 and pf > 2.0:
            entry = self._build_error_entry(
                pattern_name="low_winrate_high_pf_paradox",
                version=version_label,
                fmt_args={"wr": float(wr), "pf": float(pf)},
                symbol="GLOBAL",
            )
            detected.append(("low_winrate_high_pf_paradox", entry))

        # 检查各币种
        for symbol, sr in report.get("symbols", {}).items():
            # 模式4: 单币种负 EV
            ev = _parse_numeric(sr.get("ev_per_trade_pct", 0))
            trades_count = int(_parse_numeric(sr.get("total_trades", 0)))
            total_pnl = _parse_numeric(sr.get("total_pnl_pct", 0))
            if ev < -0.3 and trades_count >= 10:
                entry = self._build_error_entry(
                    pattern_name="single_symbol_negative_ev",
                    version=version_label,
                    fmt_args={
                        "symbol": symbol,
                        "ev": float(ev),
                        "trades": trades_count,
                        "pnl": float(total_pnl),
                    },
                    symbol=symbol,
                )
                detected.append(("single_symbol_negative_ev", entry))

        # 模式5: 极端连续亏损 (从聚合或KPI数据)
        max_cl = int(_get_agg_value(agg, "max_consecutive_losses", "max_consec_loss",
                                     default=_get_kpi_actual(kpi_check, "max_consec_loss", 0.0)))
        mlp_val = _get_kpi_actual(kpi_check, "monthly_loss_prob", 0.0)
        if max_cl > 3:
            entry = self._build_error_entry(
                pattern_name="extreme_consecutive_losses",
                version=version_label,
                fmt_args={"cl": max_cl, "mlp": float(mlp_val)},
                symbol="GLOBAL",
            )
            detected.append(("extreme_consecutive_losses", entry))

        # 模式6: ADX 弱趋势信号亏损 (从 enhanced review)
        # 检查 enhanced_review 特有的 ADX 分级数据
        adx_analysis = report.get("adx_analysis") or report.get("adx_breakdown")
        if adx_analysis:
            weak = adx_analysis.get("weak", {}) or adx_analysis.get("moderate", {})
            if weak:
                weak_count = int(weak.get("count", 0))
                weak_pnl = _parse_numeric(weak.get("total_pnl", 0) or weak.get("pnl", 0))
                weak_avg = _parse_numeric(weak.get("avg_pnl", 0))
                total_count = sum(
                    int(v.get("count", 0)) for v in adx_analysis.values()
                    if isinstance(v, dict)
                )
                pct = (weak_count / total_count * 100) if total_count > 0 else 0
                if weak_count > 0 and weak_pnl < 0:
                    entry = self._build_error_entry(
                        pattern_name="adx_weak_signal_loss",
                        version=version_label,
                        fmt_args={
                            "count": weak_count,
                            "pct": float(pct),
                            "pnl": float(weak_pnl),
                            "avg": float(weak_avg),
                        },
                        symbol="GLOBAL",
                    )
                    detected.append(("adx_weak_signal_loss", entry))

        return detected

    def _build_error_entry(
        self,
        pattern_name: str,
        version: str,
        fmt_args: Dict[str, Any],
        symbol: str = "GLOBAL",
    ) -> ErrorEntry:
        """根据模式模板构建 ErrorEntry"""
        tmpl = self.ERROR_PATTERNS.get(pattern_name, {})
        title_core = tmpl.get("description", pattern_name)
        title = f"{title_core}" + (f" [{symbol}]" if symbol != "GLOBAL" else "")

        phenomenon = tmpl.get("phenomenon_tpl", "").format(**fmt_args)
        root_cause = tmpl.get("root_cause_tpl", "")
        fix_direction = tmpl.get("fix_direction_tpl", "")
        prevention = tmpl.get("prevention_tpl", "")
        pattern_label = tmpl.get("pattern_label", pattern_name)

        # fingerprint 基于 pattern + symbol (跨版本去重)
        fp_input = f"{pattern_name}|{symbol}"
        fingerprint = self._fingerprint(fp_input)

        return ErrorEntry(
            err_id=self._generate_err_id(),
            title=title,
            level=tmpl.get("level", "MEDIUM"),
            discovered_at=f"{datetime.now().strftime('%Y-%m-%d')} {version}复盘自动检测",
            phenomenon=phenomenon,
            root_cause=root_cause,
            fix_direction=fix_direction,
            prevention_rule=prevention,
            pattern=pattern_label,
            source_version=version,
            fingerprint=fingerprint,
        )

    # ========================================================================
    # Layer C: 跨版本复发追踪
    # ========================================================================

    def _load_recurrence_db(self) -> Dict[str, Any]:
        """加载复发追踪数据库"""
        try:
            content = RECURRENCE_DB_PATH.read_text(encoding="utf-8")
            if content.strip():
                return json.loads(content)
        except Exception:
            pass
        return {"patterns": {}}

    def _save_recurrence_db(self, db: Dict[str, Any]):
        """保存复发追踪数据库"""
        RECURRENCE_DB_PATH.write_text(
            json.dumps(db, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _update_recurrence(
        self, fingerprint: str, pattern: str, version: str, err_id: str
    ) -> Optional[RecurrenceAlert]:
        """更新复发记录, 返回警报 (如达到阈值)"""
        db = self._load_recurrence_db()
        patterns = db.setdefault("patterns", {})

        record = patterns.setdefault(fingerprint, {
            "pattern": pattern,
            "occurrences": [],
        })

        # 添加本次出现 (去重: 同版本同 fingerprint 只记一次)
        existing_versions = {o.get("version") for o in record["occurrences"]}
        if version not in existing_versions:
            record["occurrences"].append({
                "version": version,
                "err_id": err_id,
                "timestamp": datetime.now().isoformat(),
            })

        self._save_recurrence_db(db)

        # 评估严重度
        count = len(record["occurrences"])
        if count >= 3:
            return RecurrenceAlert(
                fingerprint=fingerprint,
                pattern=pattern,
                occurrences=record["occurrences"],
                severity="BLOCK",
                message=f"模式 '{pattern}' 在 {count} 个版本中复发 → BLOCK 级别, "
                        f"必须深度根因分析 (违反'永不二过'铁律)",
            )
        elif count == 2:
            return RecurrenceAlert(
                fingerprint=fingerprint,
                pattern=pattern,
                occurrences=record["occurrences"],
                severity="WARN",
                message=f"模式 '{pattern}' 第 2 次复发 → WARN, 上次修复可能不完整",
            )
        return None

    # ========================================================================
    # Layer D: 参数调整建议生成
    # ========================================================================

    def _generate_param_adjustments(
        self, report: Dict[str, Any]
    ) -> List[ParamAdjustment]:
        """基于复盘报告生成参数调整建议"""
        adjustments: List[ParamAdjustment] = []
        agg = report.get("aggregate", {})
        kpi = report.get("kpi_check", {})
        version = report.get("version", "unknown")

        # 1. 最大回撤超标 → 缩减仓位
        if kpi.get("max_drawdown", {}).get("passed") is False:
            dd_actual = _get_kpi_actual(kpi, "max_drawdown", 0.0)
            if dd_actual == 0.0:
                dd_actual = _get_agg_value(agg, "max_drawdown", "dd_mean_pct", default=0.0)
            if dd_actual > 15.0:
                # 缩减系数: 目标回撤15% / 实际回撤, 限制在 0.5-1.0
                shrink = max(0.5, min(1.0, 15.0 / max(dd_actual, 0.1)))
                adjustments.append(ParamAdjustment(
                    target_symbol="GLOBAL",
                    param_name="max_pos_multiplier",
                    current_value=1.0,
                    suggested_value=round(shrink, 3),
                    reason=f"最大回撤 {dd_actual:.1f}% 超标, 缩减仓位至 {shrink:.3f}x",
                    confidence=0.7,
                    evidence={"max_drawdown": dd_actual, "target": 15.0},
                    auto_apply=False,  # 高风险调整需人工确认
                ))

        # 2. 单币种负 EV → 禁用建议
        for symbol, sr in report.get("symbols", {}).items():
            ev = _parse_numeric(sr.get("ev_per_trade_pct", 0))
            trades_count = int(_parse_numeric(sr.get("total_trades", 0)))
            if ev < -0.3 and trades_count >= 10:
                adjustments.append(ParamAdjustment(
                    target_symbol=symbol,
                    param_name="enabled",
                    current_value=1.0,
                    suggested_value=0.0,
                    reason=f"{symbol} EV={ev:.4f}% 持续为负, {trades_count}笔交易验证, "
                           f"建议禁用 (参考 ERR-20260629-049)",
                    confidence=0.85,
                    evidence={
                        "ev": ev,
                        "trades": trades_count,
                        "total_pnl": _parse_numeric(sr.get("total_pnl_pct", 0)),
                    },
                    auto_apply=False,
                ))

        # 3. 极端连续亏损 → 收紧 Layer2 风控
        max_cl = int(_get_agg_value(agg, "max_consecutive_losses", "max_consec_loss",
                                     default=_get_kpi_actual(kpi, "max_consec_loss", 0.0)))
        if max_cl > 3:
            adjustments.append(ParamAdjustment(
                target_symbol="GLOBAL",
                param_name="layer2_consec_loss_threshold",
                current_value=3.0,
                suggested_value=2.0,  # 收紧到 2
                reason=f"最大连续亏损 {max_cl} 次, 收紧 Layer2 阈值至 2",
                confidence=0.6,
                evidence={"max_consecutive_losses": max_cl, "target": 3},
                auto_apply=False,
            ))

        # 4. 月度亏损概率超标 → 月内降仓阈值
        if kpi.get("monthly_loss_prob", {}).get("passed") is False:
            mlp = _get_kpi_actual(kpi, "monthly_loss_prob", 0.0)
            adjustments.append(ParamAdjustment(
                target_symbol="GLOBAL",
                param_name="monthly_loss_threshold_pct",
                current_value=2.0,
                suggested_value=1.5,
                reason=f"月度亏损概率 {mlp:.1f}% 超标, 收紧月内降仓阈值",
                confidence=0.65,
                evidence={"monthly_loss_prob": mlp, "target": 5.0},
                auto_apply=False,
            ))

        # 5. 胜率过低 → 入场信号过滤
        wr = _get_agg_value(agg, "win_rate", "win_rate_mean", default=0.0)
        if 0 < wr < 50:
            adjustments.append(ParamAdjustment(
                target_symbol="GLOBAL",
                param_name="adx_filter_threshold",
                current_value=0.0,
                suggested_value=20.0,
                reason=f"胜率 {wr:.1f}% < 50%, 添加 ADX≥20 过滤器剔除弱趋势信号",
                confidence=0.7,
                evidence={"win_rate": wr, "target": 55.0},
                auto_apply=False,
            ))

        # 6. 基于regime的仓位调整 (ERR-20260702-v613: ranging regime弱点)
        # 从交易明细计算per-regime表现, 对弱regime降仓
        regime_adjustments = self._generate_regime_adjustments(report, version)
        adjustments.extend(regime_adjustments)

        # 7. 基于per-symbol×regime矩阵的弱币种禁用 (精细化, 非一刀切)
        symbol_regime_adjustments = self._generate_symbol_regime_adjustments(report, version)
        adjustments.extend(symbol_regime_adjustments)

        return adjustments

    def _generate_regime_adjustments(
        self, report: Dict[str, Any], version: str = "unknown"
    ) -> List[ParamAdjustment]:
        """基于regime表现生成仓位调整建议

        ERR-20260702-v613发现: v97在ranging regime下胜率20%, PnL=-$3686
        用户铁律"多时间框架分析"+"打破瓶颈": 对弱regime降仓而非全面禁用

        逻辑:
          - win_rate < 50% AND total_pnl < 0 → 降仓50%
          - win_rate < 40% AND total_pnl < -1000 → 降仓75%
          - 样本不足(<15笔) → 不生成补丁(避免过拟合)
        """
        adjustments: List[ParamAdjustment] = []

        # 尝试从report中读取regime统计, 或从交易明细重新计算
        regime_stats = report.get("regime_stats", {})

        if not regime_stats:
            # 从交易明细文件计算regime统计
            trades_path = SCRIPT_DIR / "_v97_trades_detail.json"
            if not trades_path.exists():
                # 兼容其他版本
                for tp in SCRIPT_DIR.glob("_v*_trades_detail.json"):
                    trades_path = tp
                    break

            if trades_path.exists():
                try:
                    trades = json.loads(trades_path.read_text(encoding="utf-8"))
                    regime_stats = self._compute_regime_stats(trades)
                except Exception as e:
                    logger.warning(f"计算regime统计失败: {e}")
                    return adjustments

        if not regime_stats:
            return adjustments

        for regime, stats in regime_stats.items():
            n_trades = stats.get("n_trades", 0)
            win_rate = stats.get("win_rate_pct", 0)
            total_pnl = stats.get("total_pnl_usd", 0)

            # 样本不足, 跳过 (避免过拟合, 用户铁律)
            if n_trades < 15:
                continue

            # 弱regime判定: 胜率<50% AND 总PnL为负
            if win_rate < 50 and total_pnl < 0:
                # 降仓幅度: 越弱降越多
                if win_rate < 40 and total_pnl < -1000:
                    shrink = 0.25  # 降仓75%
                    confidence = 0.8
                    reason = (f"{regime} regime表现极差: 胜率{win_rate:.1f}%, "
                              f"总PnL={total_pnl:+.0f}USD, {n_trades}笔, "
                              f"降仓至{shrink:.2f}x (ERR-20260702-v613)")
                elif win_rate < 45:
                    shrink = 0.4  # 降仓60%
                    confidence = 0.75
                    reason = (f"{regime} regime表现较差: 胜率{win_rate:.1f}%, "
                              f"总PnL={total_pnl:+.0f}USD, {n_trades}笔, "
                              f"降仓至{shrink:.2f}x")
                else:
                    shrink = 0.5  # 降仓50%
                    confidence = 0.7
                    reason = (f"{regime} regime表现偏弱: 胜率{win_rate:.1f}%, "
                              f"总PnL={total_pnl:+.0f}USD, {n_trades}笔, "
                              f"降仓至{shrink:.2f}x")

                adjustments.append(ParamAdjustment(
                    target_symbol=f"REGIME:{regime}",
                    param_name="regime_pos_multiplier",
                    current_value=1.0,
                    suggested_value=round(shrink, 3),
                    reason=reason,
                    confidence=confidence,
                    evidence={
                        "regime": regime,
                        "n_trades": n_trades,
                        "win_rate_pct": win_rate,
                        "total_pnl_usd": total_pnl,
                        "profit_factor": stats.get("profit_factor", 0),
                    },
                    auto_apply=True,  # regime降仓风险可控, 可自动应用
                ))
                logger.info(f"生成regime补丁: {regime} → pos_mult={shrink:.2f} "
                           f"(wr={win_rate:.1f}%, pnl={total_pnl:+.0f})")

        return adjustments

    def _generate_symbol_regime_adjustments(
        self, report: Dict[str, Any], version: str = "unknown"
    ) -> List[ParamAdjustment]:
        """基于per-symbol×regime矩阵生成弱币种禁用建议

        ERR-20260702-v613发现: OP_USDT在trending下25%胜率, XRP_USDT在ranging下-142PnL
        用户铁律"亏损币种应禁用": 精细化到symbol×regime级别, 非全局禁用

        逻辑:
          - 某symbol在某regime下: trades≥10 AND win_rate<40% AND pnl<-500 → 禁用该symbol×regime
          - 某symbol在所有regime下都亏损 → 全局禁用 (参考ERR-20260629-049)
        """
        adjustments: List[ParamAdjustment] = []

        # 从交易明细计算symbol×regime矩阵
        trades_path = SCRIPT_DIR / "_v97_trades_detail.json"
        if not trades_path.exists():
            for tp in SCRIPT_DIR.glob("_v*_trades_detail.json"):
                trades_path = tp
                break

        if not trades_path.exists():
            return adjustments

        try:
            trades = json.loads(trades_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"读取交易明细失败: {e}")
            return adjustments

        # 构建 symbol×regime 矩阵
        from collections import defaultdict
        matrix = defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0}))

        for t in trades:
            sym = t.get("symbol", "unknown")
            regime = t.get("regime", "unknown")
            pnl = t.get("live_pnl_usd", 0) or 0
            matrix[sym][regime]["n"] += 1
            matrix[sym][regime]["pnl"] += pnl
            if pnl > 0:
                matrix[sym][regime]["wins"] += 1

        # 检测弱 symbol×regime 组合
        for sym, regimes in matrix.items():
            sym_total_pnl = sum(r["pnl"] for r in regimes.values())
            sym_total_n = sum(r["n"] for r in regimes.values())

            # 全局弱币种: 总PnL<-500 AND trades≥15 → 全局禁用
            if sym_total_pnl < -500 and sym_total_n >= 15:
                adjustments.append(ParamAdjustment(
                    target_symbol=sym,
                    param_name="enabled",
                    current_value=1.0,
                    suggested_value=0.0,
                    reason=f"{sym} 全局表现差: 总PnL={sym_total_pnl:+.0f}USD, "
                           f"{sym_total_n}笔交易, 建议禁用 (ERR-20260629-049模式)",
                    confidence=0.85,
                    evidence={
                        "total_pnl_usd": sym_total_pnl,
                        "total_trades": sym_total_n,
                        "regimes": {r: {"n": d["n"], "pnl": d["pnl"]} for r, d in regimes.items()},
                    },
                    auto_apply=False,  # 全局禁用风险高, 需人工确认
                ))
                continue  # 已建议全局禁用, 不再生成per-regime补丁

            # per-regime 弱点: 某symbol在某regime下 win_rate<40% AND pnl<-300 AND n≥10
            for regime, d in regimes.items():
                if d["n"] >= 10:
                    wr = d["wins"] / d["n"] * 100
                    if wr < 40 and d["pnl"] < -300:
                        adjustments.append(ParamAdjustment(
                            target_symbol=f"{sym}@{regime}",
                            param_name="symbol_regime_enabled",
                            current_value=1.0,
                            suggested_value=0.0,
                            reason=f"{sym}在{regime}下表现差: 胜率{wr:.1f}%, "
                                   f"PnL={d['pnl']:+.0f}USD, {d['n']}笔, "
                                   f"禁用该symbol×regime组合",
                            confidence=0.75,
                            evidence={
                                "symbol": sym,
                                "regime": regime,
                                "n_trades": d["n"],
                                "win_rate_pct": wr,
                                "total_pnl_usd": d["pnl"],
                            },
                            auto_apply=True,  # 精细化禁用风险可控
                        ))

        return adjustments

    @staticmethod
    def _compute_regime_stats(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """从交易列表计算per-regime统计

        Args:
            trades: 交易记录列表

        Returns:
            {regime: {n_trades, win_rate_pct, total_pnl_usd, profit_factor, ...}}
        """
        from collections import defaultdict
        regime_data = defaultdict(lambda: {
            "n_trades": 0, "wins": 0, "losses": 0,
            "total_pnl_usd": 0.0, "win_pnls": [], "loss_pnls": [],
        })

        for t in trades:
            regime = t.get("regime", "unknown")
            pnl = t.get("live_pnl_usd", 0) or 0
            regime_data[regime]["n_trades"] += 1
            regime_data[regime]["total_pnl_usd"] += pnl
            if pnl > 0:
                regime_data[regime]["wins"] += 1
                regime_data[regime]["win_pnls"].append(pnl)
            elif pnl < 0:
                regime_data[regime]["losses"] += 1
                regime_data[regime]["loss_pnls"].append(abs(pnl))

        stats = {}
        for regime, d in regime_data.items():
            n = d["n_trades"]
            wr = d["wins"] / n * 100 if n > 0 else 0
            avg_win = sum(d["win_pnls"]) / len(d["win_pnls"]) if d["win_pnls"] else 0
            avg_loss = sum(d["loss_pnls"]) / len(d["loss_pnls"]) if d["loss_pnls"] else 1
            pf = avg_win / avg_loss if avg_loss > 0 else 0

            stats[regime] = {
                "n_trades": n,
                "wins": d["wins"],
                "losses": d["losses"],
                "win_rate_pct": round(wr, 2),
                "total_pnl_usd": round(d["total_pnl_usd"], 2),
                "profit_factor": round(pf, 2),
                "avg_win_usd": round(avg_win, 2),
                "avg_loss_usd": round(avg_loss, 2),
            }

        return stats

    # ========================================================================
    # Layer E: 后续任务生成
    # ========================================================================

    def _generate_follow_up_tasks(
        self,
        report: Dict[str, Any],
        param_adjustments: List[ParamAdjustment],
        errors_recorded: List[str],
    ) -> List[FollowUpTask]:
        """基于推荐建议生成后续任务"""
        tasks: List[FollowUpTask] = []
        version = report.get("version", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 任务1: 应用参数调整
        if param_adjustments:
            auto_apply_count = sum(1 for a in param_adjustments if a.auto_apply)
            manual_count = len(param_adjustments) - auto_apply_count
            tasks.append(FollowUpTask(
                task_id=f"TASK-{ts}-001",
                title=f"应用 {version} 复盘参数调整建议",
                description=(
                    f"共 {len(param_adjustments)} 项参数调整建议: "
                    f"{auto_apply_count} 项可自动应用, {manual_count} 项需人工确认. "
                    f"详见 _param_patches/param_patch_{version}.json"
                ),
                priority="HIGH",
                category="code_change",
                estimated_effort="2-4h",
                source_recommendation=f"{version} 复盘反馈",
            ))

        # 任务2: 错误库新条目审查
        if errors_recorded:
            tasks.append(FollowUpTask(
                task_id=f"TASK-{ts}-002",
                title=f"审查 {version} 新入库错误 ({len(errors_recorded)} 条)",
                description=(
                    f"新入库错误: {', '.join(errors_recorded)}. "
                    f"请审查根因分析和修复方向是否准确, 必要时补充上下文."
                ),
                priority="MEDIUM",
                category="research",
                estimated_effort="30min",
                source_recommendation=f"{version} 复盘反馈",
            ))

        # 任务3: 基于 KPI 失败项的专项回测
        kpi = report.get("kpi_check", {})
        failed_kpis = [k for k, v in kpi.items() if isinstance(v, dict) and not v.get("passed", True)]
        if failed_kpis:
            tasks.append(FollowUpTask(
                task_id=f"TASK-{ts}-003",
                title=f"针对 {version} 失败 KPI 专项回测",
                description=(
                    f"失败 KPI: {', '.join(failed_kpis)}. "
                    f"需针对每个失败 KPI 设计 A/B 测试, 验证参数调整效果. "
                    f"必须覆盖 ≥10 交易对 + 4 市场条件 (用户硬约束)."
                ),
                priority="HIGH",
                category="backtest",
                estimated_effort="4-8h",
                source_recommendation=f"{version} 复盘反馈",
            ))

        # 任务4: 模拟-实盘差异验证 (如果复盘显示差异过大)
        sim_live_diff = report.get("sim_live_gap") or report.get("sim_live_difference")
        if sim_live_diff and isinstance(sim_live_diff, dict):
            diff_pct = sim_live_diff.get("difference_pct", 0)
            if diff_pct > 15.0:
                tasks.append(FollowUpTask(
                    task_id=f"TASK-{ts}-004",
                    title=f"{version} 模拟-实盘差异 {diff_pct:.1f}% 超标修复",
                    description=(
                        f"差异率 {diff_pct:.1f}% > 15% 阈值, 必须修复. "
                        f"参考 v479b 4因素高保真市场仿真模型: "
                        f"手续费7bps + 点差per-symbol + 滑点Almgren-Chriss + 延迟500ms."
                    ),
                    priority="CRITICAL",
                    category="code_change",
                    estimated_effort="6-12h",
                    source_recommendation=f"{version} 复盘反馈",
                ))

        return tasks

    # ========================================================================
    # 主入口
    # ========================================================================

    def process_review_report(self, report_path: str) -> FeedbackActions:
        """处理一份复盘报告, 生成行动指令

        Args:
            report_path: 复盘报告 JSON 文件路径

        Returns:
            FeedbackActions: 行动汇总
        """
        report_path = Path(report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"复盘报告不存在: {report_path}")

        logger.info(f"📋 处理复盘报告: {report_path.name}")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # 版本号优先从文件名提取 (如 "auto_review_v582.json" → "v582")
        # 报告内的 version 字段是复盘系统版本 (如 "1.0.0"), 不是策略版本
        fname = report_path.stem
        version_match = re.search(r"(v\d+[a-z]?\d*)", fname, re.IGNORECASE)
        if version_match:
            version = version_match.group(1)
        else:
            version = report.get("version_label") or fname or "unknown"

        actions = FeedbackActions(
            version=version,
            timestamp=datetime.now().isoformat(),
        )

        # ===== Layer A: 错误模式检测 + 自动入库 =====
        detected = self._detect_errors_from_report(report, version_label=version)
        logger.info(f"检测到 {len(detected)} 个错误模式")

        for pattern_name, entry in detected:
            # 尝试入库 (新错误入库, 已存在则跳过)
            is_new = self._append_to_kb(entry)
            if is_new:
                actions.errors_recorded.append(entry.err_id)
            else:
                # 已存在, 找到原 err_id
                existing_err_id = self._existing_fingerprints.get(
                    entry.fingerprint, "unknown"
                )
                actions.errors_existing.append(existing_err_id)

            # 更新复发追踪
            alert = self._update_recurrence(
                fingerprint=entry.fingerprint,
                pattern=pattern_name,
                version=version,
                err_id=entry.err_id if is_new else (
                    self._existing_fingerprints.get(entry.fingerprint, entry.err_id)
                ),
            )
            if alert:
                actions.recurrence_alerts.append(alert)
                if alert.severity == "BLOCK":
                    logger.warning(f"🚫 BLOCK 级复发: {alert.message}")
                elif alert.severity == "WARN":
                    logger.warning(f"⚠️ WARN 级复发: {alert.message}")

        # ===== Layer B: 参数调整建议 =====
        actions.param_adjustments = self._generate_param_adjustments(report)
        logger.info(f"生成 {len(actions.param_adjustments)} 项参数调整建议")

        # 保存参数 patch
        if actions.param_adjustments:
            patch_dir = PARAM_PATCH_DIR
            patch_dir.mkdir(parents=True, exist_ok=True)
            patch_path = patch_dir / f"param_patch_{version}.json"
            patch_data = {
                "version": version,
                "generated_at": actions.timestamp,
                "adjustments": [asdict(a) for a in actions.param_adjustments],
            }
            patch_path.write_text(
                json.dumps(patch_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"💾 参数 patch 已保存: {patch_path}")

        # ===== Layer C: 后续任务 =====
        actions.follow_up_tasks = self._generate_follow_up_tasks(
            report, actions.param_adjustments, actions.errors_recorded
        )
        logger.info(f"生成 {len(actions.follow_up_tasks)} 个后续任务")

        # ===== 汇总 =====
        actions.summary = self._build_summary(actions)
        self._save_actions(actions)
        self._mark_digested(report_path)

        return actions

    def _build_summary(self, actions: FeedbackActions) -> str:
        """构建汇总文本"""
        parts = [f"版本 {actions.version} 复盘反馈汇总:"]
        parts.append(f"  - 新入库错误: {len(actions.errors_recorded)} 条")
        parts.append(f"  - 已存在错误: {len(actions.errors_existing)} 条")
        parts.append(f"  - 参数调整建议: {len(actions.param_adjustments)} 项")
        parts.append(f"  - 后续任务: {len(actions.follow_up_tasks)} 个")
        parts.append(f"  - 复发警报: {len(actions.recurrence_alerts)} 个")
        block_alerts = sum(1 for a in actions.recurrence_alerts if a.severity == "BLOCK")
        warn_alerts = sum(1 for a in actions.recurrence_alerts if a.severity == "WARN")
        if block_alerts:
            parts.append(f"  🚫 BLOCK 级复发: {block_alerts} 个 (必须深度根因分析)")
        if warn_alerts:
            parts.append(f"  ⚠️ WARN 级复发: {warn_alerts} 个 (上次修复可能不完整)")
        return "\n".join(parts)

    def _save_actions(self, actions: FeedbackActions):
        """保存行动汇总为 JSON"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"feedback_actions_{actions.version}_{ts}.json"
        out_path.write_text(
            json.dumps(asdict(actions), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"💾 反馈行动已保存: {out_path}")

    def _mark_digested(self, report_path: Path):
        """标记复盘报告已消化 (避免重复处理)

        防御性处理: 历史文件可能因手动编辑或旧版本写入而格式不一致
        (如 ``["digested"]`` 这样的 list 而非 ``{"digested": [...]}`` dict),
        本方法在加载时统一归一化为标准 dict 结构, 保证"复盘不是摆设"
        闭环不被阻断。
        """
        try:
            loaded = json.loads(DIGEST_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            loaded = None

        # 归一化: 无论加载到 list/dict/其它, 都统一为 {"digested": [...]}
        if isinstance(loaded, dict):
            entries = loaded.get("digested", [])
            if not isinstance(entries, list):
                entries = []
        elif isinstance(loaded, list):
            # 旧格式/损坏格式: 整个文件是 list, 视作已消化条目列表
            entries = [
                e for e in loaded if isinstance(e, dict) and "path" in e
            ]
        else:
            entries = []

        digested = {"digested": entries}
        digested["digested"].append({
            "path": str(report_path),
            "timestamp": datetime.now().isoformat(),
        })
        DIGEST_LOG_PATH.write_text(
            json.dumps(digested, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ========================================================================
    # 批量处理
    # ========================================================================

    def process_all_undigested(self) -> List[FeedbackActions]:
        """处理所有未消化的复盘报告"""
        review_dir = REVIEW_REPORTS_DIR
        if not review_dir.exists():
            logger.warning(f"复盘报告目录不存在: {review_dir}")
            return []

        # 加载已消化列表 (防御性归一化, 与 _mark_digested 保持一致)
        try:
            loaded = json.loads(DIGEST_LOG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                entries = loaded.get("digested", [])
                if not isinstance(entries, list):
                    entries = []
            elif isinstance(loaded, list):
                entries = [e for e in loaded if isinstance(e, dict) and "path" in e]
            else:
                entries = []
            digested_paths = {d["path"] for d in entries}
        except Exception:
            digested_paths = set()

        # 找所有未消化的 JSON 报告
        undigested = []
        for jp in sorted(review_dir.glob("*.json")):
            if str(jp) not in digested_paths:
                # 跳过非复盘报告 (multi_agent_state, evolution_report 等)
                name = jp.stem.lower()
                if any(skip in name for skip in ("multi_agent", "evolution_report")):
                    continue
                undigested.append(jp)

        logger.info(f"发现 {len(undigested)} 个未消化复盘报告")
        results = []
        for jp in undigested:
            try:
                actions = self.process_review_report(str(jp))
                results.append(actions)
            except Exception as e:
                logger.error(f"处理 {jp.name} 失败: {e}")
        return results

    def get_recurrence_report(self) -> Dict[str, Any]:
        """生成跨版本复发分析报告"""
        db = self._load_recurrence_db()
        patterns = db.get("patterns", {})

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_patterns": len(patterns),
            "block_level": [],
            "warn_level": [],
            "single_occurrence": [],
            "summary": "",
        }

        for fp, record in patterns.items():
            count = len(record.get("occurrences", []))
            entry = {
                "fingerprint": fp,
                "pattern": record.get("pattern", ""),
                "occurrences": record.get("occurrences", []),
                "count": count,
            }
            if count >= 3:
                report["block_level"].append(entry)
            elif count == 2:
                report["warn_level"].append(entry)
            else:
                report["single_occurrence"].append(entry)

        report["summary"] = (
            f"共追踪 {len(patterns)} 个模式: "
            f"{len(report['block_level'])} BLOCK, "
            f"{len(report['warn_level'])} WARN, "
            f"{len(report['single_occurrence'])} 单次"
        )
        return report


# ============================================================================
# 辅助函数
# ============================================================================

def _check_adx_weak_loss(depth: Dict[str, Any]) -> bool:
    """检查 ADX<20 弱趋势信号是否亏损"""
    adx = depth.get("adx_analysis") or depth.get("adx_breakdown")
    if not adx:
        return False
    weak = adx.get("weak", {}) or adx.get("moderate", {})
    if not weak:
        return False
    return weak.get("count", 0) > 0 and (
        weak.get("total_pnl", 0) or weak.get("pnl", 0)
    ) < 0


def _parse_numeric(value: Any, default: float = 0.0) -> float:
    """从可能是字符串/数字的值中提取浮点数

    处理: "5.9%" → 5.9, "≤15.0%" → 15.0, "-1.678" → -1.678, 30.5 → 30.5
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 提取数字部分 (含负号和小数点)
        import re as _re
        m = _re.search(r"-?\d+\.?\d*", value.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
    return default


def _get_kpi_actual(kpi_check: Dict, key: str, default: float = 0.0) -> float:
    """从 KPI 检查字典中提取 actual 数值 (兼容字符串格式)"""
    entry = kpi_check.get(key, {})
    if isinstance(entry, dict):
        return _parse_numeric(entry.get("actual"), default)
    return _parse_numeric(entry, default)


def _get_agg_value(agg: Dict, *keys: str, default: float = 0.0) -> float:
    """从聚合字典中按多个候选键名取值

    例: _get_agg_value(agg, 'max_drawdown', 'dd_mean_pct', 'max_drawdown_pct')
    依次尝试每个键, 返回第一个存在的数值
    """
    for k in keys:
        if k in agg:
            return _parse_numeric(agg[k], default)
    return default


# ============================================================================
# 自检入口 (单元测试)
# ============================================================================

def _self_test() -> bool:
    """自检: 验证复盘-经验库联动闭环核心功能可用

    测试覆盖:
      1. process_review_report: mock 复盘报告 → FeedbackActions 输出非空
      2. process_all_undigested: 批量处理多份报告 → 结果列表非空
      3. 指纹去重: 重复处理同一报告 → 识别为已存在错误
    所有文件 IO 隔离在临时目录中, 不污染真实错误知识库/复发库/消化日志。
    """
    import tempfile
    from unittest import mock

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # 隔离所有模块级路径常量, 避免污染真实文件
            tmp_recurrence_db = tmp / "recurrence_db.json"
            tmp_digest_log = tmp / "digested_reports.json"
            tmp_param_patch_dir = tmp / "param_patches"
            tmp_param_patch_dir.mkdir(exist_ok=True)
            tmp_error_kb = tmp / "error_kb.md"
            tmp_output_dir = tmp / "feedback_actions"
            tmp_review_dir = tmp / "review_reports"
            tmp_review_dir.mkdir(exist_ok=True)

            # mock 复盘报告: 含失败 KPI (月度亏损概率/最大回撤) + 负 EV 币种 + 低胜率高盈亏比悖论
            def _build_mock_report(version_label: str) -> dict:
                return {
                    "version_label": version_label,
                    "kpi_check": {
                        "monthly_loss_prob": {"passed": False, "actual": 8.5},
                        "max_drawdown": {"passed": False, "actual": 18.3},
                    },
                    "aggregate": {
                        "win_rate": 40.0,
                        "profit_factor": 2.4,
                        "max_drawdown": 18.3,
                        "monthly_loss_prob": 8.5,
                        "max_consecutive_losses": 5,
                    },
                    "symbols": {
                        "ADA-USDT": {
                            "ev_per_trade_pct": -0.6,
                            "total_trades": 15,
                            "total_pnl_pct": -9.0,
                        },
                    },
                }

            report_path = tmp_review_dir / "auto_review_v_selftest1.json"
            report_path.write_text(
                json.dumps(_build_mock_report("v_selftest1"), ensure_ascii=False),
                encoding="utf-8",
            )

            # ===== 测试1: process_review_report 单报告处理 =====
            with mock.patch("review_feedback_loop.RECURRENCE_DB_PATH", tmp_recurrence_db), \
                 mock.patch("review_feedback_loop.DIGEST_LOG_PATH", tmp_digest_log), \
                 mock.patch("review_feedback_loop.PARAM_PATCH_DIR", tmp_param_patch_dir):
                loop = ReviewFeedbackLoop(
                    error_kb_path=tmp_error_kb,
                    output_dir=tmp_output_dir,
                )
                actions = loop.process_review_report(str(report_path))

            assert actions.version == "v_selftest1", f"版本标签错误: {actions.version}"
            assert actions.timestamp != "", "timestamp 为空"
            assert actions.summary != "", "summary 为空"
            # 新 KB 为空, 所有检测到的模式都应入库
            assert len(actions.errors_recorded) >= 1, \
                f"errors_recorded 为空 (期望至少1条新错误): {actions}"
            # 应生成参数调整建议或后续任务
            assert (len(actions.param_adjustments) + len(actions.follow_up_tasks)) >= 1, \
                "未生成任何参数调整建议或后续任务"

            # ===== 测试2: process_all_undigested 批量处理 =====
            report_path_2 = tmp_review_dir / "auto_review_v_selftest2.json"
            report_path_2.write_text(
                json.dumps(_build_mock_report("v_selftest2"), ensure_ascii=False),
                encoding="utf-8",
            )
            with mock.patch("review_feedback_loop.RECURRENCE_DB_PATH", tmp / "recurrence_db2.json"), \
                 mock.patch("review_feedback_loop.DIGEST_LOG_PATH", tmp / "digested2.json"), \
                 mock.patch("review_feedback_loop.PARAM_PATCH_DIR", tmp / "param_patches2"), \
                 mock.patch("review_feedback_loop.REVIEW_REPORTS_DIR", tmp_review_dir), \
                 mock.patch("review_feedback_loop.FEEDBACK_OUTPUT_DIR", tmp / "feedback2"):
                (tmp / "param_patches2").mkdir(exist_ok=True)
                (tmp / "feedback2").mkdir(exist_ok=True)
                loop2 = ReviewFeedbackLoop(
                    error_kb_path=tmp / "error_kb2.md",
                    output_dir=tmp / "feedback2",
                )
                results = loop2.process_all_undigested()
            assert isinstance(results, list), "process_all_undigested 应返回列表"
            assert len(results) >= 1, f"批量处理结果为空 (期望>=1): {len(results)}"

            # ===== 测试3: 指纹去重 — 重复处理同一报告, 错误应识别为已存在 =====
            # 用测试1写入的 tmp_error_kb 重新构造 loop, fingerprint 应命中
            with mock.patch("review_feedback_loop.RECURRENCE_DB_PATH", tmp / "recurrence_db3.json"), \
                 mock.patch("review_feedback_loop.DIGEST_LOG_PATH", tmp / "digested3.json"), \
                 mock.patch("review_feedback_loop.PARAM_PATCH_DIR", tmp / "param_patches3"):
                (tmp / "param_patches3").mkdir(exist_ok=True)
                loop3 = ReviewFeedbackLoop(
                    error_kb_path=tmp_error_kb,  # 复用测试1写入的 KB
                    output_dir=tmp / "feedback3",
                )
                # KB 中应有测试1入库的错误
                assert len(loop3._existing_err_ids) >= 1, \
                    "已加载 KB 的 ERR-ID 索引为空 (期望>=1)"
                assert len(loop3._existing_fingerprints) >= 1, \
                    "已加载 KB 的指纹索引为空 (期望>=1)"
                # 重复处理同一报告 (用新文件名避免被消化日志跳过)
                dup_report = tmp / "auto_review_v_selftest_dup.json"
                dup_report.write_text(
                    json.dumps(_build_mock_report("v_selftest1"), ensure_ascii=False),
                    encoding="utf-8",
                )
                actions_dup = loop3.process_review_report(str(dup_report))
            # fingerprint 命中 → 错误应归入 errors_existing 而非 errors_recorded
            assert len(actions_dup.errors_existing) >= 1, \
                f"指纹去重失败: errors_existing 为空 (期望>=1): {actions_dup}"

            logger.info("ReviewFeedbackLoop 自检全部通过 ✓")
            return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="复盘-经验库联动闭环引擎"
    )
    parser.add_argument(
        "--review", type=str,
        help="处理指定的复盘报告 JSON",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="批量处理所有未消化的复盘报告",
    )
    parser.add_argument(
        "--track-recurrence", action="store_true",
        help="生成跨版本复发分析报告",
    )
    parser.add_argument(
        "--kb-path", type=str, default=str(ERROR_KB_PATH),
        help="错误知识库路径 (默认 1500-error-knowledge-base.md)",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="运行模块自检 (_self_test), 不执行 CLI 业务逻辑",
    )
    args = parser.parse_args()

    if args.self_test:
        ok = _self_test()
        print(f"_self_test: {ok}")
        return

    loop = ReviewFeedbackLoop(error_kb_path=Path(args.kb_path))

    if args.review:
        actions = loop.process_review_report(args.review)
        print(f"\n{'='*70}")
        print(actions.summary)
        print(f"{'='*70}")
        if actions.recurrence_alerts:
            print("\n🚨 复发警报:")
            for alert in actions.recurrence_alerts:
                print(f"  [{alert.severity}] {alert.message}")

    elif args.batch:
        results = loop.process_all_undigested()
        print(f"\n{'='*70}")
        print(f"批量处理完成: {len(results)} 份报告")
        for actions in results:
            print(f"\n--- {actions.version} ---")
            print(actions.summary)

    elif args.track_recurrence:
        report = loop.get_recurrence_report()
        print(f"\n{'='*70}")
        print("📊 跨版本复发分析报告")
        print(f"{'='*70}")
        print(report["summary"])
        if report["block_level"]:
            print(f"\n🚫 BLOCK 级复发 ({len(report['block_level'])} 个):")
            for entry in report["block_level"]:
                print(f"  - {entry['pattern']}: {entry['count']} 次")
                for occ in entry["occurrences"]:
                    print(f"    · {occ['version']} ({occ['err_id']})")
        if report["warn_level"]:
            print(f"\n⚠️ WARN 级复发 ({len(report['warn_level'])} 个):")
            for entry in report["warn_level"]:
                print(f"  - {entry['pattern']}: {entry['count']} 次")
                for occ in entry["occurrences"]:
                    print(f"    · {occ['version']} ({occ['err_id']})")
        # 保存完整报告
        out_path = FEEDBACK_OUTPUT_DIR / f"recurrence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n💾 完整报告: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
