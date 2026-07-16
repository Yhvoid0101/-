"""执行现实性与基准对比 Mixin (Phase 7.10 提取)。

承载:
  - _evaluate_execution_realism_penalty: P0核心,ProductionOMS重评精英策略真实成本
  - _check_staged_admission: 分阶准入门控 (部署禁令)
  - _get_recent_bars_for_benchmark: 基准策略K线获取
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from .staged_admission_gate import StagedAdmissionGate  # noqa: F401
    _STAGED_ADMISSION_AVAILABLE = True
except ImportError:
    _STAGED_ADMISSION_AVAILABLE = False


class ExecutionBenchmarkMixin:
    """执行现实性惩罚 + 分阶准入 + 基准K线获取 Mixin。

    Host 鸭子类型依赖:
      - self.population: 需有 population/_last_scores/_last_trades_by_agent/generation
      - self.production_oms: ProductionOMS 实例或 None
      - self.staged_admission_gate: StagedAdmissionGate 实例或 None
      - self.engine.markets: Dict[str, market]
      - self.data_pipeline: 需有 get_recent_bars() 方法 (可选)
    """

    def _evaluate_execution_realism_penalty(self) -> None:
        """每N轮用ProductionOMS重新评估精英策略的执行现实性惩罚

        对前20%精英agent，用ProductionOMS.submit_order重新模拟其交易，
        计算 execution_realism_penalty = (sandbox_pnl - production_pnl) / max(abs(sandbox_pnl), 1.0)
        将penalty写入FitnessScore并重新计算gt_score。

        来源: ProductionOMS模拟的真实成交成本(滑点/部分成交/拒单)回灌到进化fitness
        铁律: "一定不要出现模拟牛逼，实盘亏钱！"
        """
        from .production_oms import OrderSide, OrderType

        scores = getattr(self.population, "_last_scores", {})
        trades_map = getattr(self.population, "_last_trades_by_agent", {})
        if not scores or not trades_map:
            return

        all_agents = list(self.population.population)
        if not all_agents:
            return

        # 前20%精英agent (按fitness_score降序)
        elite_count = max(1, len(all_agents) // 5)
        elite_agents = sorted(all_agents, key=lambda a: a.fitness_score, reverse=True)[:elite_count]

        oms = self.production_oms
        if oms is None:
            return

        penalized = 0
        for agent in elite_agents:
            trades = trades_map.get(agent.agent_id, [])
            if not trades:
                continue

            sandbox_pnl = sum(t.get("pnl", 0.0) for t in trades)
            if abs(sandbox_pnl) < 1e-8:
                continue

            # 用ProductionOMS重新模拟每笔交易
            production_pnl = 0.0
            for t in trades:
                trade_pnl = t.get("pnl", 0.0)
                entry_price = t.get("entry_price", 0.0)
                exit_price = t.get("exit_price", 0.0)
                side_str = str(t.get("side", "buy")).lower()

                if entry_price <= 0 or abs(exit_price - entry_price) < 1e-10:
                    # 无法推断quantity，假设无滑点影响
                    production_pnl += trade_pnl
                    continue

                # 从pnl和价格差推断quantity
                # pnl = (exit_price - entry_price) * quantity (long)
                # pnl = (entry_price - exit_price) * quantity (short)
                quantity = abs(trade_pnl / (exit_price - entry_price))
                if quantity <= 0:
                    production_pnl += trade_pnl
                    continue

                side = OrderSide.BUY if side_str in ("buy", "long", "1") else OrderSide.SELL
                try:
                    fill_result = oms.submit_order(
                        symbol=t.get("symbol", "BTC/USDT"),
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        current_price=entry_price,
                    )
                except Exception:
                    fill_result = None

                if fill_result is not None and fill_result.success:
                    fill_ratio = fill_result.fill_quantity / quantity if quantity > 0 else 0.0
                    slippage_cost = abs(fill_result.fill_price - entry_price) * fill_result.fill_quantity
                    production_pnl += trade_pnl * fill_ratio - slippage_cost
                else:
                    # 拒单或未成交: 该笔交易无收益
                    production_pnl += 0.0

            # execution_realism_penalty = (sandbox_pnl - production_pnl) / max(abs(sandbox_pnl), 1.0)
            penalty = (sandbox_pnl - production_pnl) / max(abs(sandbox_pnl), 1.0)
            penalty = max(0.0, min(1.0, penalty))

            # 写入FitnessScore并重新计算gt_score
            fitness = scores.get(agent.agent_id)
            if fitness is not None:
                fitness.execution_realism_penalty = penalty
                new_gt = fitness.compute_gt_score()
                agent.fitness_score = new_gt
                penalized += 1

        if penalized > 0:
            logger.info(
                "execution_realism_penalty: gen=%d, 评估 %d 个精英agent, "
                "惩罚已写入FitnessScore并重算gt_score",
                self.population.generation, penalized,
            )


    def _check_staged_admission(self, kpi_metrics: Dict) -> Dict:
        """v4.0 Phase 3: 分阶准入门控 — 部署禁令检查

        用户核心诉求: "在未达成性能指标前不得推进实盘部署工作"
        此方法是部署禁令的代码级实现，Tier 1 未通过 → ban_active=True。

        Args:
            kpi_metrics: 含 ann_sim_pct/max_dd_pct/win_rate_pct/sharpe/mlp_pct/gap_pct/
                         n_symbols/years_span/stress_test_pass 的 dict

        Returns:
            {ban_active: bool, tier1_pass: bool, n_pass: int, n_total: int, details: [...]}
        """
        if not _STAGED_ADMISSION_AVAILABLE or self.staged_admission_gate is None:
            return {
                "ban_active": True,
                "reason": "staged_admission未集成或gate未实例化",
                "tier1_pass": False,
                "n_pass": 0,
                "n_total": 14,  # v700 P2: 9性能 + 5反过拟合
                "details": [],
            }
        try:
            tier1_result = self.staged_admission_gate.evaluate_tier1(kpi_metrics)
            return {
                "ban_active": not tier1_result.get("passed", False),
                "tier1_pass": tier1_result.get("passed", False),
                "n_pass": tier1_result.get("n_pass", 0),
                "n_total": tier1_result.get("n_total", 14),  # v700 P2: 9性能 + 5反过拟合
                "details": tier1_result.get("details", []),
            }
        except Exception as e:
            logger.warning("v4 _check_staged_admission 评估失败(非致命): %s", e)
            return {
                "ban_active": True,
                "reason": f"评估异常: {e}",
                "tier1_pass": False,
                "n_pass": 0,
                "n_total": 14,  # v700 P2: 9性能 + 5反过拟合
                "details": [],
            }


    def _get_recent_bars_for_benchmark(self, n_bars: int = 500) -> List[Any]:
        """获取最近的市场K线数据用于基准策略回测

        从沙盘引擎的历史数据中提取最近的K线，转换为基准策略所需的格式。
        """
        # 沙盘引擎维护了每个symbol的市场数据历史
        # 我们取第一个symbol的数据
        if not hasattr(self.engine, 'markets') or not self.engine.markets:
            return []

        bars = []
        for symbol, market in self.engine.markets.items():
            # 尝试从市场对象获取历史K线
            history = getattr(market, 'history', None) or getattr(market, 'bars', None)
            if history:
                # 取最近n_bars根
                recent = history[-n_bars:] if len(history) > n_bars else history
                bars = list(recent)
                break

        # 如果没有历史数据，尝试从数据管道获取
        if not bars and hasattr(self.data_pipeline, 'get_recent_bars'):
            try:
                bars = self.data_pipeline.get_recent_bars(n_bars)
            except Exception as e:
                # Phase 7J-10 反温室修复 MEDIUM #17: 基准对比数据获取失败可见化
                # 之前: pass, MultiStrategyBenchmark 无数据可对比, 反温室基准失效
                # 现在: warning (反温室: 基准失效=无法判断策略是否真优于基准)
                logger.warning(
                    "get_recent_bars 失败 (反温室基准失效!): %s",
                    str(e)[:100],
                )

        return bars


