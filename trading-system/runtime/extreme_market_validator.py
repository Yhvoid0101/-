# -*- coding: utf-8 -*-
"""ExtremeMarketValidator — 5 层极端行情验证器 (v92 Phase 6)

将 _v91_comprehensive_validation.py 的 Layer A-E 脚本式代码封装为可重用类,
供 evolution_loop.py 在每次进化后自动调用回归测试。

用户核心诉求: "市场状态自动分类 + 极端行情专项测试集，强化全行情适应性，
打破策略行情局限瓶颈"

5 层测试:
  Layer A: ATR 分位极端档位 (top5/bot5 胜率 ≥ 50%)
  Layer B: 单笔最大盈亏极端事件 (单亏 ≤ 2%)
  Layer C: 连续亏损序列极端事件 (max ≤ 3)
  Layer D: 月度极端事件 (亏损月占比 ≤ 5%)
  Layer E: 滚动窗口压力测试 (90天/180天达标率 ≥ 80%)

判定标准:
  extreme_pass_count ≥ 4: PASS (与 v91 基线一致)
  extreme_pass_count < 4: BLOCK (退化, 阻断 can_deploy)

来源: _v91_comprehensive_validation.py line 148-392 脚本式代码 → 提取为类
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtremeMarketReport:
    """5 层极端行情测试报告"""
    layer_a_pass: bool = False       # ATR 分位极端
    layer_b_pass: bool = False       # 单笔极端
    layer_c_pass: bool = False       # 连续亏损
    layer_d_pass: bool = False       # 月度极端
    layer_e_90_pass: bool = False    # 90 天滚动窗口
    layer_e_180_pass: bool = False   # 180 天滚动窗口
    layer_e_90_rate: float = 0.0
    layer_e_180_rate: float = 0.0
    extreme_pass_count: int = 0      # A+B+C+D+E_90 计数
    regression_pass: bool = False    # 是否未退化 (≥ 4/5)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的 dict（供 report 写入）"""
        return {
            "layer_a": {"pass": self.layer_a_pass},
            "layer_b": {"pass": self.layer_b_pass},
            "layer_c": {"pass": self.layer_c_pass},
            "layer_d": {"pass": self.layer_d_pass},
            "layer_e_90": {"pass": self.layer_e_90_pass, "rate": self.layer_e_90_rate},
            "layer_e_180": {"pass": self.layer_e_180_pass, "rate": self.layer_e_180_rate},
            "extreme_pass_count": self.extreme_pass_count,
            "regression_pass": self.regression_pass,
            "details": self.details,
        }


class ExtremeMarketValidator:
    """5 层极端行情验证器

    v92 Phase 6: 将脚本式极端行情测试封装为可重用类,
    供 evolution_loop.py 在每次进化后自动调用回归测试。

    使用方式:
        validator = ExtremeMarketValidator()
        report = validator.validate(trades, baseline_pass_count=4)
        if not report.regression_pass:
            # BLOCK can_deploy — 退化告警
    """

    def __init__(self, baseline_pass_count: int = 4):
        """初始化

        Args:
            baseline_pass_count: 基线 pass 数阈值 (v91 基线=4/5),
                                 低于此值判退化并阻断部署
        """
        self.baseline_pass_count = baseline_pass_count

    def validate(
        self,
        trades: List[Dict[str, Any]],
        run_backtest_fn: Optional[Any] = None,
        records: Optional[List] = None,
    ) -> ExtremeMarketReport:
        """执行 5 层极端行情测试

        Args:
            trades: 已完成的交易列表 (含 pnl_pct, atr_pct, entry_bar 等)
            run_backtest_fn: 滚动窗口测试用的回测函数 (Layer E 需要, 可选)
            records: 原始 K 线数据 (Layer E 需要, 可选)

        Returns:
            ExtremeMarketReport 含 5 层测试结果 + 汇总判定
        """
        report = ExtremeMarketReport()

        if not trades:
            logger.warning("[ExtremeMarket] trades 为空, 跳过验证")
            report.details["error"] = "trades_empty"
            return report

        # Layer A: ATR 分位极端档位
        try:
            report.layer_a_pass = self._validate_layer_a(trades)
            report.details["layer_a"] = "PASS" if report.layer_a_pass else "FAIL"
        except Exception as e:
            logger.debug("[ExtremeMarket] Layer A 失败(非致命): %s", e)
            report.details["layer_a_error"] = str(e)

        # Layer B: 单笔最大盈亏极端事件
        try:
            report.layer_b_pass = self._validate_layer_b(trades)
            report.details["layer_b"] = "PASS" if report.layer_b_pass else "FAIL"
        except Exception as e:
            logger.debug("[ExtremeMarket] Layer B 失败(非致命): %s", e)
            report.details["layer_b_error"] = str(e)

        # Layer C: 连续亏损序列极端事件
        try:
            report.layer_c_pass = self._validate_layer_c(trades)
            report.details["layer_c"] = "PASS" if report.layer_c_pass else "FAIL"
        except Exception as e:
            logger.debug("[ExtremeMarket] Layer C 失败(非致命): %s", e)
            report.details["layer_c_error"] = str(e)

        # Layer D: 月度极端事件
        try:
            report.layer_d_pass = self._validate_layer_d(trades)
            report.details["layer_d"] = "PASS" if report.layer_d_pass else "FAIL"
        except Exception as e:
            logger.debug("[ExtremeMarket] Layer D 失败(非致命): %s", e)
            report.details["layer_d_error"] = str(e)

        # Layer E: 滚动窗口压力测试 (可选, 需 run_backtest_fn)
        if run_backtest_fn is not None and records is not None:
            try:
                _e90_pass, _e90_rate = self._validate_layer_e(
                    trades, records, run_backtest_fn, window_bars=90
                )
                report.layer_e_90_pass = _e90_pass
                report.layer_e_90_rate = _e90_rate
                _e180_pass, _e180_rate = self._validate_layer_e(
                    trades, records, run_backtest_fn, window_bars=180
                )
                report.layer_e_180_pass = _e180_pass
                report.layer_e_180_rate = _e180_rate
                report.details["layer_e_90"] = (
                    f"PASS(rate={_e90_rate:.2%})" if _e90_pass
                    else f"FAIL(rate={_e90_rate:.2%})"
                )
                report.details["layer_e_180"] = (
                    f"PASS(rate={_e180_rate:.2%})" if _e180_pass
                    else f"FAIL(rate={_e180_rate:.2%})"
                )
            except Exception as e:
                logger.debug("[ExtremeMarket] Layer E 失败(非致命): %s", e)
                report.details["layer_e_error"] = str(e)
        else:
            report.details["layer_e"] = "skipped (run_backtest_fn/records 未提供)"

        # 汇总判定 (A+B+C+D+E_90, 与 _v91_comprehensive_validation 一致)
        report.extreme_pass_count = sum([
            report.layer_a_pass,
            report.layer_b_pass,
            report.layer_c_pass,
            report.layer_d_pass,
            report.layer_e_90_pass,
        ])
        report.regression_pass = report.extreme_pass_count >= self.baseline_pass_count

        logger.info(
            "[ExtremeMarket] 5层测试完成: %d/5 通过 (基线≥%d) — %s",
            report.extreme_pass_count,
            self.baseline_pass_count,
            "PASS" if report.regression_pass else "BLOCK",
        )
        return report

    def _validate_layer_a(self, trades: List[Dict[str, Any]]) -> bool:
        """Layer A: ATR 分位极端档位 (top5/bot5 胜率 ≥ 50%)

        判定: ATR 最高的 5% 交易和最低的 5% 交易, 胜率均需 ≥ 50%
        目的: 确保策略在极端波动行情下不退化
        """
        _atr_values = [
            t.get("atr_pct", 0) for t in trades
            if t.get("atr_pct", 0) > 0
        ]
        if len(_atr_values) < 20:
            return True  # 数据不足, 不阻断

        _sorted = sorted(_atr_values)
        _top5_threshold = _sorted[int(len(_sorted) * 0.95)]
        _bot5_threshold = _sorted[int(len(_sorted) * 0.05)]
        _top5_trades = [t for t in trades if t.get("atr_pct", 0) >= _top5_threshold]
        _bot5_trades = [t for t in trades if t.get("atr_pct", 0) <= _bot5_threshold]

        _top5_win = (
            sum(1 for t in _top5_trades if t.get("pnl_pct", 0) > 0)
            / max(len(_top5_trades), 1)
        )
        _bot5_win = (
            sum(1 for t in _bot5_trades if t.get("pnl_pct", 0) > 0)
            / max(len(_bot5_trades), 1)
        )
        return _top5_win >= 0.50 and _bot5_win >= 0.50

    def _validate_layer_b(self, trades: List[Dict[str, Any]]) -> bool:
        """Layer B: 单笔最大亏损 ≤ 2%

        判定: 所有交易中最大单笔亏损(百分比) ≤ 2%
        目的: 确保单次交易最大亏损不超过本金 2% (实盘铁律)
        """
        _losses = [
            abs(t.get("pnl_pct", 0)) for t in trades
            if t.get("pnl_pct", 0) < 0
        ]
        if not _losses:
            return True
        _max_loss = max(_losses)
        return _max_loss <= 2.0

    def _validate_layer_c(self, trades: List[Dict[str, Any]]) -> bool:
        """Layer C: 连续亏损序列 max ≤ 3

        判定: 最长连续亏损交易数 ≤ 3
        目的: 确保连续亏损次数 ≤ 3次 (实盘铁律)
        """
        _max_consec = 0
        _current = 0
        for t in trades:
            if t.get("pnl_pct", 0) < 0:
                _current += 1
                _max_consec = max(_max_consec, _current)
            else:
                _current = 0
        return _max_consec <= 3

    def _validate_layer_d(self, trades: List[Dict[str, Any]]) -> bool:
        """Layer D: 月度亏损月占比 ≤ 5%

        判定: 亏损月份数 / 总月份数 ≤ 5%
        目的: 确保月度亏损概率 ≤ 5% (实盘铁律)
        """
        # 简化: 按 entry_bar // 30 分组 (假设 bar=日)
        _monthly_pnl: Dict[int, float] = {}
        for t in trades:
            _month = t.get("entry_bar", 0) // 30
            _monthly_pnl[_month] = _monthly_pnl.get(_month, 0) + t.get("pnl_pct", 0)
        if not _monthly_pnl:
            return True
        _loss_months = sum(1 for v in _monthly_pnl.values() if v < 0)
        _loss_ratio = _loss_months / len(_monthly_pnl)
        return _loss_ratio <= 0.05

    def _validate_layer_e(
        self,
        trades: List[Dict[str, Any]],
        records: List,
        run_backtest_fn: Any,
        window_bars: int = 90,
    ) -> Tuple[bool, float]:
        """Layer E: 滚动窗口压力测试

        判定: 滚动窗口达标率 ≥ 80%
        目的: 确保策略在不同时间窗口下稳定盈利

        Args:
            window_bars: 窗口大小 (90 或 180)

        Returns:
            (pass: bool, rate: float) — 达标率是否 ≥ 80% + 实际达标率
        """
        if not records or len(records) < window_bars:
            return (False, 0.0)
        _step = max(window_bars // 3, 1)
        _total_windows = 0
        _pass_windows = 0
        for _start in range(0, len(records) - window_bars, _step):
            _total_windows += 1
            try:
                _result = run_backtest_fn(
                    records, start_bar=_start, end_bar=_start + window_bars
                )
                _kpi = _result.get("kpi", {}) if isinstance(_result, dict) else {}
                _pass = (
                    _kpi.get("ann_live", 0) > 0
                    and _kpi.get("max_dd", 100) < 50
                )
                if _pass:
                    _pass_windows += 1
            except Exception:
                pass
        _rate = _pass_windows / max(_total_windows, 1)
        return (_rate >= 0.80, _rate)


# ============================================================================
# __main__ 守卫自检 (遵循 maker_cost_model.py 范本)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("extreme_market_validator.py — L6 功能自检")
    print("=" * 70)

    n_pass = 0
    n_total = 0

    # 测试1: ExtremeMarketReport 默认值
    n_total += 1
    report = ExtremeMarketReport()
    assert report.layer_a_pass is False
    assert report.extreme_pass_count == 0
    assert report.regression_pass is False
    assert report.to_dict()["extreme_pass_count"] == 0
    print("✅ 测试1: ExtremeMarketReport 默认值正确")
    n_pass += 1

    # 测试2: ExtremeMarketValidator 实例化
    n_total += 1
    validator = ExtremeMarketValidator(baseline_pass_count=4)
    assert validator.baseline_pass_count == 4
    print("✅ 测试2: ExtremeMarketValidator 实例化正确")
    n_pass += 1

    # 测试3: 空 trades 验证
    n_total += 1
    empty_report = validator.validate([])
    assert empty_report.details.get("error") == "trades_empty"
    assert empty_report.extreme_pass_count == 0
    print("✅ 测试3: 空 trades 验证正确")
    n_pass += 1

    # 测试4: Layer B 单亏 > 2% 触发 FAIL
    n_total += 1
    bad_trades = [
        {"pnl_pct": -3.0, "atr_pct": 0.02, "entry_bar": 1},  # 单亏 3% > 2%
        {"pnl_pct": 1.0, "atr_pct": 0.02, "entry_bar": 2},
        {"pnl_pct": 0.5, "atr_pct": 0.02, "entry_bar": 3},
    ]
    b_report = validator.validate(bad_trades)
    assert b_report.layer_b_pass is False, "单亏3%应触发Layer B FAIL"
    assert b_report.regression_pass is False, "退化应阻断"
    print(f"✅ 测试4: Layer B 单亏>2%触发FAIL (count={b_report.extreme_pass_count}/5)")
    n_pass += 1

    # 测试5: Layer C 连续亏损 > 3 触发 FAIL
    n_total += 1
    consec_loss_trades = [
        {"pnl_pct": -0.1, "atr_pct": 0.02, "entry_bar": 1},
        {"pnl_pct": -0.1, "atr_pct": 0.02, "entry_bar": 2},
        {"pnl_pct": -0.1, "atr_pct": 0.02, "entry_bar": 3},
        {"pnl_pct": -0.1, "atr_pct": 0.02, "entry_bar": 4},  # 连续4亏 > 3
        {"pnl_pct": 1.0, "atr_pct": 0.02, "entry_bar": 5},
    ]
    c_report = validator.validate(consec_loss_trades)
    assert c_report.layer_c_pass is False, "连续4亏应触发Layer C FAIL"
    print(f"✅ 测试5: Layer C 连续亏损>3触发FAIL (count={c_report.extreme_pass_count}/5)")
    n_pass += 1

    # 测试6: 正常 trades 应通过 Layer B/C
    n_total += 1
    good_trades = [
        {"pnl_pct": 1.0, "atr_pct": 0.02, "entry_bar": i} for i in range(30)
    ]
    # 加入少量小亏损 (单亏<2%, 连亏<3)
    good_trades[5] = {"pnl_pct": -0.5, "atr_pct": 0.02, "entry_bar": 5}
    good_trades[6] = {"pnl_pct": -0.5, "atr_pct": 0.02, "entry_bar": 6}
    good_trades[7] = {"pnl_pct": 1.0, "atr_pct": 0.02, "entry_bar": 7}
    g_report = validator.validate(good_trades)
    assert g_report.layer_b_pass is True, "单亏<2%应通过Layer B"
    assert g_report.layer_c_pass is True, "连亏<3应通过Layer C"
    print(f"✅ 测试6: 正常trades通过Layer B/C (count={g_report.extreme_pass_count}/5)")
    n_pass += 1

    print("\n" + "=" * 70)
    print(f"自检结果: {n_pass}/{n_total} 通过")
    print("=" * 70)
    if n_pass == n_total:
        print("🎉 全部通过 — ExtremeMarketValidator 功能可用")
    else:
        print("❌ 部分失败 — 需检查")
