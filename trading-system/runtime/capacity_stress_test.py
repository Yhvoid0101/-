# -*- coding: utf-8 -*-
"""资金容量压力测试模块 (Capital Capacity Stress Test)

用户最高铁律:
  "永远永远不要出现模拟牛逼，实盘亏损的情况"
  "模拟/实盘差异全成本量化模型 + 资金容量压力测试, 根治模拟好看实盘亏"

当前问题:
  evolution_loop.py L4822-4823 的 stress_test_pass 默认 True (危险桩):
    if metrics["stress_test_pass"] is None:
        metrics["stress_test_pass"] = True
  即"未运行压力测试却默认通过" — 这是"模拟牛逼实盘亏"的直接原因之一。

本模块实现:
  1. 多资金级别压力测试 ($1k / $10k / $100k / $1M / $10M)
  2. Almgren-Chriss 平方根市场冲击模型: MI = k × √(order_size / ADV)
  3. 流动性枯竭检测: order_size > 10% bar_volume → 警告
  4. 崩溃资金计算: 策略停止盈利的资金阈值
  5. 严格默认: 未测试 → BLOCK (不再默认 True)

设计铁律:
  D1: 命名隔离 (CapacityStressTester, 不与 sim_live_gap_model 冲突)
  D2: 自包含 (仅依赖 stdlib + numpy, 输入 trades + capital_levels)
  D3: 信息增强层 (设置 stress_test_pass + breakdown_capital, 供准入接口消费)
  D5: guard pattern (try/except 降级, 永不崩溃主循环)
  D6: 不通过已知数据反推策略 (用户铁律, 本模块只做容量分析不做策略优化)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("hermes.capacity_stress_test")


# ============================================================================
# 常量与阈值
# ============================================================================

# 默认测试资金级别 (用户要求: $1k 迷你 → $10k+ 标准 → $1M+ 大资金)
DEFAULT_CAPITAL_LEVELS: Tuple[float, ...] = (
    1_000.0,    # Tier 2 迷你实盘 ($1k-$5k)
    5_000.0,    # Tier 2 上限
    10_000.0,   # Tier 3 标准实盘下限
    50_000.0,   # 中等资金
    100_000.0,  # 大资金
    1_000_000.0,  # 机构级 (压力测试)
)

# Almgren-Chriss 市场冲击系数 (来源: Almgren & Chriss 2000)
# k=0.1 为流动性好的市场, k=0.3 为中等流动性
DEFAULT_IMPACT_COEFFICIENT = 0.1

# 流动性枯竭阈值: 单笔订单 > bar 成交量的 10% → 流动性警告
LIQUIDITY_DEPLETION_THRESHOLD = 0.10

# 崩溃资金判定: 净 PnL <= 0 时的最小资金级别
BREAKDOWN_NET_PNL_THRESHOLD = 0.0


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(slots=True)
class CapitalLevelResult:
    """单个资金级别的压力测试结果"""

    capital: float  # 测试资金量
    n_trades: int  # 交易笔数
    gross_pnl: float  # 毛收益 (无摩擦)
    total_friction_cost: float  # 总摩擦成本 (冲击+滑点+流动性)
    market_impact_cost: float  # 市场冲击成本
    slippage_cost: float  # 滑点成本 (基础)
    liquidity_warning_count: int  # 流动性警告次数
    net_pnl: float  # 净收益 (毛 - 摩擦)
    net_roi: float  # 净收益率 (net_pnl / capital)
    max_drawdown: float  # 最大回撤 (含摩擦)
    is_profitable: bool  # 是否盈利
    avg_order_size_pct_adv: float  # 平均订单大小占 ADV 百分比
    worst_trade_friction: float  # 最大单笔摩擦


@dataclass(slots=True)
class CapacityStressReport:
    """资金容量压力测试综合报告"""

    tested_levels: List[CapitalLevelResult]
    breakdown_capital: Optional[float]  # 崩溃资金 (None=所有级别都盈利)
    safe_capital_max: float  # 安全资金上限 (净 ROI>0 的最大资金)
    overall_verdict: str  # "PASS" | "WARN" | "BLOCK"
    n_liquidity_warnings: int  # 总流动性警告
    n_profitable_levels: int  # 盈利级别数
    n_total_levels: int  # 总测试级别数
    detail: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# 资金容量压力测试器
# ============================================================================


class CapacityStressTester:
    """资金容量压力测试器 (用户铁律: 根治模拟好看实盘亏)

    在多个资金级别下模拟策略表现, 检测:
      1. 市场冲击: 大订单推动价格不利 (Almgren-Chriss 平方根)
      2. 流动性枯竭: 订单超过 bar 成交量 10% → 警告
      3. 崩溃资金: 策略停止盈利的资金阈值

    使用方式:
        tester = CapacityStressTester(impact_coefficient=0.1)
        report = tester.run_stress_test(
            trades=trade_list,  # [{pnl, volume, price, ...}, ...]
            base_capital=1000.0,
            capital_levels=(1000, 10000, 100000, 1000000),
        )
        if report.overall_verdict == "BLOCK":
            logger.warning("资金容量压力测试失败: 崩溃资金=$%s", report.breakdown_capital)
    """

    def __init__(
        self,
        impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
        liquidity_threshold: float = LIQUIDITY_DEPLETION_THRESHOLD,
        enabled: bool = True,
    ):
        """初始化资金容量压力测试器

        Args:
            impact_coefficient: Almgren-Chriss 市场冲击系数 k (默认 0.1)
            liquidity_threshold: 流动性枯竭阈值 (订单/bar_volume, 默认 10%)
            enabled: 是否启用
        """
        self.enabled = enabled
        self.impact_coefficient = max(0.0, impact_coefficient)
        self.liquidity_threshold = max(0.01, liquidity_threshold)
        logger.debug(
            "CapacityStressTester 初始化 (k=%.3f, liquidity_thresh=%.2f%%, enabled=%s)",
            self.impact_coefficient, self.liquidity_threshold * 100, self.enabled,
        )

    def _compute_market_impact(
        self,
        order_size_usd: float,
        bar_volume_usd: float,
    ) -> float:
        """计算市场冲击成本 (Almgren-Chriss 平方根模型)

        MI_cost = k × √(order_size / ADV) × order_size

        Args:
            order_size_usd: 订单大小 (美元)
            bar_volume_usd: bar 成交量 (美元, 作为 ADV 代理)

        Returns:
            市场冲击成本 (美元)
        """
        if bar_volume_usd <= 0 or order_size_usd <= 0:
            return 0.0
        # 防止 order_size >> bar_volume 时冲击无限大 (限幅)
        participation_rate = min(order_size_usd / bar_volume_usd, 1.0)
        impact_factor = self.impact_coefficient * math.sqrt(participation_rate)
        return impact_factor * order_size_usd

    def _check_liquidity_depletion(
        self,
        order_size_usd: float,
        bar_volume_usd: float,
    ) -> bool:
        """检查流动性枯竭 (订单 > bar 成交量阈值)

        Returns:
            True = 流动性枯竭警告
        """
        if bar_volume_usd <= 0:
            return True  # 无成交量数据 → 保守警告
        return (order_size_usd / bar_volume_usd) > self.liquidity_threshold

    def run_stress_test(
        self,
        trades: Sequence[Dict[str, Any]],
        base_capital: float = 1000.0,
        capital_levels: Optional[Sequence[float]] = None,
        base_slippage_bps: float = 5.0,
    ) -> CapacityStressReport:
        """运行多资金级别压力测试

        Args:
            trades: 交易列表, 每笔含:
                - pnl: 毛盈亏 (美元, base_capital 下的)
                - volume: 交易量 (合约数或美元)
                - price: 成交价格
                - bar_volume: 所在 bar 的成交量 (可选, 缺失则用 volume 代理)
            base_capital: 基准资金 (trades 的 pnl 基于此资金)
            capital_levels: 测试资金级别 (None=用默认)
            base_slippage_bps: 基础滑点 (bps, 默认 5bps)

        Returns:
            CapacityStressReport: 综合压力测试报告
        """
        if not self.enabled:
            return self._empty_report("disabled")

        if not trades:
            return self._empty_report("no_trades")

        levels = list(capital_levels) if capital_levels else list(DEFAULT_CAPITAL_LEVELS)
        levels = sorted(set(max(100.0, lv) for lv in levels))  # 去重+排序+最小$100

        results: List[CapitalLevelResult] = []
        total_liquidity_warnings = 0

        for capital in levels:
            level_result = self._test_single_level(
                trades=trades,
                base_capital=base_capital,
                target_capital=capital,
                base_slippage_bps=base_slippage_bps,
            )
            results.append(level_result)
            total_liquidity_warnings += level_result.liquidity_warning_count

        # 计算崩溃资金: 第一个 net_pnl <= 0 的级别
        breakdown_capital: Optional[float] = None
        for r in results:
            if r.net_pnl <= BREAKDOWN_NET_PNL_THRESHOLD:
                breakdown_capital = r.capital
                break

        # 安全资金上限: 最后一个盈利级别
        safe_capital_max = 0.0
        for r in results:
            if r.is_profitable:
                safe_capital_max = r.capital

        n_profitable = sum(1 for r in results if r.is_profitable)

        # 综合判定
        if breakdown_capital is not None:
            verdict = "BLOCK"
        elif total_liquidity_warnings > len(trades) * 0.1:
            # 超过 10% 交易触发流动性警告
            verdict = "WARN"
        else:
            verdict = "PASS"

        recommendations = self._generate_recommendations(
            results, breakdown_capital, safe_capital_max,
        )

        report = CapacityStressReport(
            tested_levels=results,
            breakdown_capital=breakdown_capital,
            safe_capital_max=safe_capital_max,
            overall_verdict=verdict,
            n_liquidity_warnings=total_liquidity_warnings,
            n_profitable_levels=n_profitable,
            n_total_levels=len(results),
            detail={
                "impact_coefficient": self.impact_coefficient,
                "liquidity_threshold": self.liquidity_threshold,
                "base_capital": base_capital,
                "base_slippage_bps": base_slippage_bps,
            },
            recommendations=recommendations,
        )

        logger.info(
            "[CapacityStress] verdict=%s breakdown=$%s safe_max=$%s profitable=%d/%d liq_warn=%d",
            verdict,
            f"{breakdown_capital:,.0f}" if breakdown_capital else "None",
            f"{safe_capital_max:,.0f}",
            n_profitable, len(results), total_liquidity_warnings,
        )
        return report

    def _test_single_level(
        self,
        trades: Sequence[Dict[str, Any]],
        base_capital: float,
        target_capital: float,
        base_slippage_bps: float,
    ) -> CapitalLevelResult:
        """测试单个资金级别

        核心逻辑:
          - 资金放大倍数 = target_capital / base_capital
          - 订单大小随资金线性放大
          - 市场冲击随订单大小平方根放大 (Almgren-Chriss)
          - 毛收益随资金线性放大
          - 净收益 = 毛收益 - 摩擦成本
        """
        if base_capital <= 0:
            scale = 1.0
        else:
            scale = target_capital / base_capital

        gross_pnl = 0.0
        total_friction = 0.0
        total_impact = 0.0
        total_slippage = 0.0
        liq_warnings = 0
        worst_friction = 0.0
        order_size_pct_adv_list: List[float] = []
        net_pnl_series: List[float] = []

        for t in trades:
            # 毛收益按资金比例缩放
            trade_pnl = float(t.get("pnl", 0.0)) * scale
            gross_pnl += trade_pnl

            # 订单大小 (美元, 按资金缩放)
            trade_volume = float(t.get("volume", 0.0))
            trade_price = float(t.get("price", 0.0))
            order_size_usd = trade_volume * trade_price * scale

            # bar 成交量 (美元) — bar_volume 通常是合约数, 转换为 USD
            # 约定: bar_volume 字段为合约数 (如 BTC 数量), 需乘以 price 得到 USD 成交额
            bar_volume_raw = float(t.get("bar_volume", 0.0))
            if bar_volume_raw > 0 and trade_price > 0:
                bar_volume_usd = bar_volume_raw * trade_price
            elif bar_volume_raw > 1000:
                # bar_volume 已经是 USD (大数值), 直接使用
                bar_volume_usd = bar_volume_raw
            else:
                # 无成交量数据 → 保守假设 order_size 的 10 倍
                bar_volume_usd = order_size_usd * 10.0

            # 市场冲击 (Almgren-Chriss)
            impact_cost = self._compute_market_impact(order_size_usd, bar_volume_usd)

            # 基础滑点 (bps, 按订单大小)
            slippage_cost = order_size_usd * (base_slippage_bps / 10000.0)

            # 流动性枯竭检测
            if self._check_liquidity_depletion(order_size_usd, bar_volume_usd):
                liq_warnings += 1

            # 订单占 ADV 百分比
            if bar_volume_usd > 0:
                order_size_pct_adv_list.append(order_size_usd / bar_volume_usd)

            total_impact += impact_cost
            total_slippage += slippage_cost
            friction = impact_cost + slippage_cost
            total_friction += friction
            worst_friction = max(worst_friction, friction)

            net_pnl_series.append(trade_pnl - friction)

        net_pnl = gross_pnl - total_friction
        net_roi = net_pnl / target_capital if target_capital > 0 else 0.0

        # 最大回撤 (含摩擦)
        max_dd = self._compute_max_drawdown(net_pnl_series)

        avg_pct_adv = (
            float(np.mean(order_size_pct_adv_list)) if order_size_pct_adv_list else 0.0
        )

        return CapitalLevelResult(
            capital=target_capital,
            n_trades=len(trades),
            gross_pnl=gross_pnl,
            total_friction_cost=total_friction,
            market_impact_cost=total_impact,
            slippage_cost=total_slippage,
            liquidity_warning_count=liq_warnings,
            net_pnl=net_pnl,
            net_roi=net_roi,
            max_drawdown=max_dd,
            is_profitable=net_pnl > 0,
            avg_order_size_pct_adv=avg_pct_adv,
            worst_trade_friction=worst_friction,
        )

    @staticmethod
    def _compute_max_drawdown(pnl_series: List[float]) -> float:
        """计算最大回撤 (基于净 PnL 序列)"""
        if not pnl_series:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_series:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _generate_recommendations(
        self,
        results: List[CapitalLevelResult],
        breakdown_capital: Optional[float],
        safe_capital_max: float,
    ) -> List[str]:
        """生成资金容量建议"""
        recs: List[str] = []
        if breakdown_capital is not None:
            recs.append(
                f"策略在 ${breakdown_capital:,.0f} 资金级别崩溃 (净 PnL<=0), "
                f"建议最大资金 ${safe_capital_max:,.0f}"
            )
        # 流动性警告占比
        total_liq_warn = sum(r.liquidity_warning_count for r in results)
        total_trades = sum(r.n_trades for r in results) if results else 1
        if total_liq_warn > total_trades * 0.1:
            recs.append(
                f"流动性警告 {total_liq_warn}/{total_trades} ({100*total_liq_warn/total_trades:.1f}%), "
                f"建议减少单笔订单大小或拆单"
            )
        # 市场冲击占比
        for r in results:
            if r.gross_pnl > 0:
                impact_ratio = r.market_impact_cost / r.gross_pnl
                if impact_ratio > 0.3:
                    recs.append(
                        f"${r.capital:,.0f} 级别市场冲击占毛收益 {100*impact_ratio:.1f}%, "
                        f"过高, 建议限制资金或改进执行算法"
                    )
                    break
        if not recs:
            recs.append(f"策略在所有测试级别盈利, 安全资金上限 ${safe_capital_max:,.0f}")
        return recs

    def _empty_report(self, reason: str) -> CapacityStressReport:
        """空报告 (降级模式)"""
        return CapacityStressReport(
            tested_levels=[],
            breakdown_capital=None,
            safe_capital_max=0.0,
            overall_verdict="BLOCK",  # 降级时保守 BLOCK (不再默认 True)
            n_liquidity_warnings=0,
            n_profitable_levels=0,
            n_total_levels=0,
            detail={"reason": reason, "conservative_default": "BLOCK (不再默认 True)"},
            recommendations=[f"压力测试未运行 ({reason}), 保守 BLOCK"],
        )


# ============================================================================
# 自验证
# ============================================================================


def _self_test() -> bool:
    """模块自验证"""
    try:
        tester = CapacityStressTester(impact_coefficient=0.1, enabled=True)

        # 模拟交易: 小资金下盈利, 大资金下因市场冲击而亏损
        # 现实策略: 每笔毛 pnl=$50 (base_capital=$1000, 5%每笔), 小订单(0.01 BTC=$500), 高流动性 bar(100 BTC=$5M)
        mock_trades = [
            {"pnl": 50.0, "volume": 0.01, "price": 50000.0, "bar_volume": 100.0}
            for _ in range(100)
        ]

        report = tester.run_stress_test(
            trades=mock_trades,
            base_capital=1000.0,
            capital_levels=(1000, 10000, 100000, 1000000, 10000000),
            base_slippage_bps=5.0,
        )

        # 小资金 ($1k) 应盈利
        assert report.tested_levels[0].capital == 1000.0
        assert report.tested_levels[0].is_profitable, (
            f"$1k 应盈利, net_pnl={report.tested_levels[0].net_pnl}"
        )
        logger.info(
            "$%s: gross=$%.2f friction=$%.2f net=$%.2f ROI=%.2f%%",
            f"{report.tested_levels[0].capital:,.0f}",
            report.tested_levels[0].gross_pnl,
            report.tested_levels[0].total_friction_cost,
            report.tested_levels[0].net_pnl,
            report.tested_levels[0].net_roi * 100,
        )

        # 最大资金级别应因市场冲击而亏损或至少摩擦显著
        big_level = report.tested_levels[-1]
        logger.info(
            "$%s: gross=$%.2f friction=$%.2f net=$%.2f ROI=%.2f%% liq_warn=%d",
            f"{big_level.capital:,.0f}",
            big_level.gross_pnl, big_level.total_friction_cost,
            big_level.net_pnl, big_level.net_roi * 100,
            big_level.liquidity_warning_count,
        )
        # 最大资金时订单远超 bar_volume → 流动性警告
        assert big_level.liquidity_warning_count > 0, f"${big_level.capital:,.0f} 应触发流动性警告"

        # 市场冲击应随资金增大
        impact_1k = report.tested_levels[0].market_impact_cost
        impact_1m = big_level.market_impact_cost
        assert impact_1m > impact_1k, f"市场冲击应随资金增大: $1k={impact_1k} $1M={impact_1m}"

        # 空交易降级
        empty_report = tester.run_stress_test(trades=[], base_capital=1000.0)
        assert empty_report.overall_verdict == "BLOCK", "空交易应保守 BLOCK"
        assert "conservative_default" in empty_report.detail, "应有保守默认说明"

        # disabled 模式
        tester_disabled = CapacityStressTester(enabled=False)
        disabled_report = tester_disabled.run_stress_test(trades=mock_trades)
        assert disabled_report.overall_verdict == "BLOCK", "disabled 应保守 BLOCK"

        logger.info("capacity_stress_test 自验证全部通过")
        return True
    except Exception as e:
        logger.error("capacity_stress_test 自验证失败: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = _self_test()
    print(f"self_test: {ok}")
    import sys
    sys.exit(0 if ok else 1)
