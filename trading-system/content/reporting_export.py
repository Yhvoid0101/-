"""
ReportingExportMixin — Phase 7.7 架构解耦
从 evolution_loop.py 提取的报告导出逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖:
      get_summary: 30+ self 属性 (全只读访问, 覆盖engine/population/production/crypto/frontier等)
      export_results: self.data_dir, self.round_results, self.get_summary()
      _compute_final_metrics_from_round_results: self.round_results, self._last_overfitting_validation
  - 属性分散保留: 所有 self 属性在 Host __init__ 中初始化, Mixin 仅访问
  - 零循环依赖: 仅依赖 logger + typing + os + json

来源:
  - get_summary: 全系统状态聚合 (engine/population/production/crypto/frontier/chanlun等)
  - export_results: 导出进化结果到JSON
  - _compute_final_metrics_from_round_results: 从round_results提取9项KPI指标
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ReportingExportMixin:
    """报告导出 Mixin (Phase 7.7 提取)

    Host 鸭子类型依赖 (属性分散在 Host __init__, 全只读):
      - self.engine: 需有 get_stats() 方法
      - self.population: 需有 .history/.generation/.population 属性
      - self.production: 需有 .kill_switch/.audit_logger/.drawdown_monitor
      - self.round_results: List[EvolutionRoundResult]
      - self.data_dir: str (结果导出路径)
      - self._last_overfitting_validation: Optional[Dict]
      - 30+ 其他系统状态属性 (crypto/frontier/chanlun/regime_detector等)
    """

    def _compute_final_metrics_from_round_results(self) -> Dict:
        """从 round_results 提取最终 KPI 指标供 TierAdmissionGate 评估 (Phase 2 新增)

        聚合 ann_sim_pct/max_dd_pct/win_rate_pct/sharpe/mlp_pct/gap_pct/n_symbols/years_span
        若数据不足返回部分指标 (None 值由 evaluate_tier 安全降级为 BLOCK)

        Returns:
            Dict: 含 Tier 1 所需的 9 项指标 (部分可能为 None)
        """
        metrics: Dict[str, any] = {
            "ann_sim_pct": None,
            "max_dd_pct": None,
            "win_rate_pct": None,
            "sharpe": None,
            "mlp_pct": None,
            "gap_pct": None,
            "n_symbols": None,
            "years_span": None,
            "stress_test_pass": None,
        }
        try:
            # 优先复用 _last_overfitting_validation 结果 (若含 KPI)
            _ov = getattr(self, "_last_overfitting_validation", None)
            if isinstance(_ov, dict):
                for _k in list(metrics.keys()):
                    if _k in _ov and _ov[_k] is not None:
                        metrics[_k] = _ov[_k]
            # 从 round_results 聚合每轮 KPI (若 round_results 非空)
            if self.round_results:
                _n = len(self.round_results)
                _wr_sum = 0.0
                _wr_cnt = 0
                _dd_max = 0.0
                _ann_sum = 0.0
                _ann_cnt = 0
                for _r in self.round_results:
                    _r_dict = _r if isinstance(_r, dict) else getattr(_r, "__dict__", {})
                    if not isinstance(_r_dict, dict):
                        continue
                    # 胜率聚合
                    if "win_rate_pct" in _r_dict and _r_dict["win_rate_pct"] is not None:
                        _wr_sum += float(_r_dict["win_rate_pct"])
                        _wr_cnt += 1
                    # 最大回撤聚合 (取最大值)
                    if "max_dd_pct" in _r_dict and _r_dict["max_dd_pct"] is not None:
                        _dd_max = max(_dd_max, float(_r_dict["max_dd_pct"]))
                    # 年化收益聚合 (取平均)
                    if "ann_sim_pct" in _r_dict and _r_dict["ann_sim_pct"] is not None:
                        _ann_sum += float(_r_dict["ann_sim_pct"])
                        _ann_cnt += 1
                if _wr_cnt > 0 and metrics["win_rate_pct"] is None:
                    metrics["win_rate_pct"] = _wr_sum / _wr_cnt
                if _dd_max > 0 and metrics["max_dd_pct"] is None:
                    metrics["max_dd_pct"] = _dd_max
                if _ann_cnt > 0 and metrics["ann_sim_pct"] is None:
                    metrics["ann_sim_pct"] = _ann_sum / _ann_cnt
            # n_symbols 从配置获取
            if metrics["n_symbols"] is None:
                _symbols = getattr(self, "symbols", None) or getattr(self, "_symbols", None)
                if _symbols is not None:
                    metrics["n_symbols"] = len(_symbols) if not isinstance(_symbols, (str, bytes)) else 1
            # years_span 从 bars_per_symbol 估算 (保守: 1000 bars ≈ 0.5 年)
            if metrics["years_span"] is None:
                _bps = getattr(self, "bars_per_symbol", 0) or 0
                if _bps > 0:
                    metrics["years_span"] = max(0.1, _bps / 2000.0)
            # stress_test_pass 默认 False (BLOCK) — 用户铁律: "永远不出现模拟牛逼实盘亏损"
            # 不再默认 True (桩修复); 集成 capacity_stress_tester 实测
            if metrics["stress_test_pass"] is None:
                metrics["stress_test_pass"] = False  # 保守 BLOCK: 未运行压力测试 = 不通过
            # 集成 capacity_stress_tester (若可用) — 根治 stress_test_pass=True 桩
            if self.capacity_stress_tester is not None and not metrics["stress_test_pass"]:
                try:
                    # 从 _last_generation_trades 提取已平仓交易 (BLOCK-6: 不用跨代累积)
                    _all_trades_src = (
                        getattr(self, "_last_generation_trades", None)
                        or self.engine.get_all_trades()
                    )
                    _stress_trades = []
                    for _aid_s, _t_list_s in (_all_trades_src or {}).items():
                        for _t_s in _t_list_s:
                            if isinstance(_t_s, dict):
                                _status_s = _t_s.get("status", "")
                                _pnl_s = float(_t_s.get("pnl", 0))
                                _price_s = float(_t_s.get("entry_price", 0.0))
                                _qty_s = float(_t_s.get("quantity", _t_s.get("size", 0.0)))
                            else:
                                _status_s = getattr(_t_s, "status", "")
                                _pnl_s = float(getattr(_t_s, "pnl", 0))
                                _price_s = float(getattr(_t_s, "entry_price", 0.0))
                                _qty_s = float(getattr(_t_s, "quantity", getattr(_t_s, "size", 0.0)))
                            if _status_s in ("closed", "liquidated") and _price_s > 0:
                                _stress_trades.append({
                                    "pnl": _pnl_s,
                                    "volume": _qty_s,
                                    "price": _price_s,
                                    # bar_volume 缺失, capacity_stress_tester 用 volume 代理
                                })
                    if len(_stress_trades) >= 5:
                        _cap_base = float(getattr(self, "initial_capital", 10000.0))
                        _cap_report = self.capacity_stress_tester.run_stress_test(
                            trades=_stress_trades,
                            base_capital=_cap_base,
                        )
                        metrics["stress_test_pass"] = (_cap_report.overall_verdict != "BLOCK")
                        metrics["capacity_breakdown_capital"] = _cap_report.breakdown_capital
                        metrics["capacity_safe_capital_max"] = _cap_report.safe_capital_max
                        metrics["capacity_verdict"] = _cap_report.overall_verdict
                        # 缓存报告供外部 (validate_anti_overfitting / SKILL) 访问
                        self._last_capacity_stress_report = _cap_report
                    else:
                        metrics["capacity_skip_reason"] = f"trades<5 ({len(_stress_trades)})"
                except Exception as _cap_err:
                    logger.warning("资金容量压力测试失败: %s", _cap_err)
                    metrics["stress_test_pass"] = False  # 保守 BLOCK
                    metrics["capacity_error"] = str(_cap_err)
        except Exception:
            pass  # 安全降级: 返回部分 None, evaluate_tier 会 BLOCK
        return metrics


    def get_summary(self) -> Dict[str, Any]:
        """获取进化摘要"""
        # Phase 7.7: 从 Host 模块动态获取模块级常量 (避免循环导入)
        import sys as _sys
        _host_mod = _sys.modules.get(self.__class__.__module__)
        _STAGED_ADMISSION_AVAILABLE = getattr(_host_mod, "_STAGED_ADMISSION_AVAILABLE", False)
        _GLOBAL_KB_AVAILABLE = getattr(_host_mod, "_GLOBAL_KB_AVAILABLE", False)
        _ERR_KB_AVAILABLE = getattr(_host_mod, "_ERR_KB_AVAILABLE", False)
        _MAKER_COST_AVAILABLE = getattr(_host_mod, "_MAKER_COST_AVAILABLE", False)
        _FEATURE_EXTRACTORS_AVAILABLE = getattr(_host_mod, "_FEATURE_EXTRACTORS_AVAILABLE", False)
        _PAIR_TRADING_AVAILABLE = getattr(_host_mod, "_PAIR_TRADING_AVAILABLE", False)
        _MTF_RESONANCE_AVAILABLE = getattr(_host_mod, "_MTF_RESONANCE_AVAILABLE", False)
        _EXTREME_MARKET_TEST_AVAILABLE = getattr(_host_mod, "_EXTREME_MARKET_TEST_AVAILABLE", False)
        _AUTO_REVIEW_AVAILABLE = getattr(_host_mod, "_AUTO_REVIEW_AVAILABLE", False)
        _AUTO_PARAM_APPLIER_AVAILABLE = getattr(_host_mod, "_AUTO_PARAM_APPLIER_AVAILABLE", False)
        _REGIME_CLASSIFIER_AVAILABLE = getattr(_host_mod, "_REGIME_CLASSIFIER_AVAILABLE", False)

        engine_stats = self.engine.get_stats()
        pop_stats = self.population.history[-1] if self.population.history else None

        # 生产级强化系统状态
        ks = self.production.kill_switch
        production_status = {
            "kill_switch_state": ks.state.value,
            "kill_switch_trigger_count": ks.trigger_count,
            "kill_switch_active": self._kill_switch_active,
            "audit_log_entries": len(self.production.audit_logger._entries),
            "drawdown_alerts": len(self.production.drawdown_monitor._alerts),
        }

        # 加密货币日内交易系统状态
        crypto_status = {
            "vwap_symbols": len(self.crypto_intraday.vwap._sessions),
            "funding_symbols": len(self.crypto_intraday.funding._states),
            "cascade_clusters": sum(len(v) for v in self.crypto_intraday.cascade._clusters.values()),
            "oi_tracked_symbols": len(self.crypto_intraday.oi_tracker._oi_history),
            "current_session": self.crypto_intraday.session.get_session(),
        }

        # 日内交易专属评估指标（R10新增）
        intraday_metrics = self._compute_intraday_metrics()

        # 前沿增强系统状态
        frontier_status = self.frontier.get_status()
        frontier_status["vpin_blocked"] = self._vpin_toxic_blocked
        frontier_status["vpin_block_reason"] = self._vpin_block_reason
        # R12 Phase 4: GNN系统性风险门控状态
        frontier_status["gnn_systemic_risk_blocked"] = self._gnn_systemic_risk_blocked
        if self._gnn_last_signal:
            frontier_status["gnn_signal_type"] = self._gnn_last_signal.get("signal_type", "unknown")
            frontier_status["gnn_systemic_risk_index"] = self._gnn_last_signal.get("systemic_risk_index", 0.0)

        # 缠论+裸K价格行为系统状态
        support, resistance = self.chanlun_pa.get_support_resistance()
        chanlun_status = {
            "support": support,
            "resistance": resistance,
            "current_range_pct": ((resistance - support) / support * 100) if support > 0 else 0,
        }

        # 市场微观结构增强系统状态
        microstructure_status = self.microstructure.get_summary()

        # 市场状态检测系统状态
        regime_status = self.regime_detector.get_summary()

        # 组合优化系统状态
        portfolio_opt_status = self.portfolio_optimizer.get_summary()

        # ML因子挖掘系统状态
        alpha_mining_status = self.alpha_mining.get_state()

        # 深度学习时序预测系统状态
        dl_forecast_status = self.dl_forecast.get_state()

        # 智能执行算法系统状态
        execution_algo_status = self.execution_algo.get_state()

        # 多智能体协作系统状态
        multi_agent_status = self.multi_agent.get_state()

        # v2.3 海龟交易法则系统状态
        turtle_status = self.turtle_system.get_state()

        # v2.3 反过拟合验证状态
        overfitting_status = getattr(self, '_last_overfitting_validation', {"status": "not_run"})

        # B5修复: 持久化验证结果 (DF/gap_factor/predicted_live_sharpe/R2 等)
        # 之前这些结果计算后不写入 JSON, 导致跨版本追踪无数据基础
        validation_persisted = getattr(self, '_last_validation_result', None)

        # Phase 5.1 + v700 P1: 15项 GO/NO-GO 最终判定 (用户铁律: "未达指标禁止部署" + "永不模拟牛逼实盘亏钱")
        # 从 validation_persisted + _last_overfitting_validation 提取指标, 计算15项 GO/NO-GO
        # G1-G10: 性能指标 | G11-G15: 反过拟合指标 (v700 P1新增 BLOCK 条件)
        _go_nogo_checks: Dict[str, bool] = {}
        _go_count = 0
        _can_deploy_live = False
        try:
            _vp = validation_persisted or {}
            _sa = _vp.get("staged_admission", {}) if isinstance(_vp, dict) else {}
            _t1m = _sa.get("tier1_metrics", {}) if isinstance(_sa, dict) else {}
            _r2v = _vp.get("r2_verdict", {}) if isinstance(_vp, dict) else {}

            _g1_ann = float(_t1m.get("ann_sim_pct", _r2v.get("sim_annual_return_pct", 0.0)))
            _g2_dd = float(_t1m.get("max_dd_pct", 100.0))
            _g3_wr = float(_t1m.get("win_rate_pct", 0.0))
            _g4_sh = float(_t1m.get("sharpe", 0.0))
            _g5_mlp = float(_t1m.get("mlp_pct", 100.0))
            _g6_gap = float(_t1m.get("gap_pct", _r2v.get("relative_diff_pct", 100.0)))
            _g7_sym = int(_t1m.get("n_symbols", 0))
            _g8_yrs = float(_t1m.get("years_span", 0.0))
            _g9_str = bool(_t1m.get("stress_test_pass", False))
            _g10_df = float(_vp.get("df_score", 0.0))

            # v700 P1: G11-G15 反过拟合 GO/NO-GO BLOCK 条件
            # 用户铁律: "永远永远不要出现模拟牛逼，实盘亏损的情况"
            # 核心诊断: 反过拟合模块此前只是"事后报告"(logger.warning), 不是"准入门槛"
            # 现升级为 BLOCK 条件 — 任一反过拟合指标不达标 → 禁止实盘部署
            # 数据来源: self._last_overfitting_validation (validate_anti_overfitting() 的返回结构)
            # 保守默认: 未运行验证时全部默认 BLOCK (符合"无验证不交付"铁律)
            _ov = overfitting_status if isinstance(overfitting_status, dict) else {}
            _ov_ran = _ov.get("status", "ran") != "not_run" and bool(_ov)
            _ov_details = _ov.get("details", {}) if isinstance(_ov.get("details", {}), dict) else {}
            _ov_wf = _ov_details.get("walk_forward", {}) if isinstance(_ov_details.get("walk_forward", {}), dict) else {}
            _ov_ps = _ov_details.get("parameter_sensitivity", {}) if isinstance(_ov_details.get("parameter_sensitivity", {}), dict) else {}

            # 默认值设计: 未运行 → 严重过拟合(BLOCK), 已运行 → 取真实值
            # G11 PBO: Bailey & López de Prado CSCV方法, PBO≥0.5=过拟合(BLOCK), <0.2=稳健
            _g11_pbo = float(_ov.get("pbo", 1.0)) if _ov_ran else 1.0
            # G12 蒙特卡洛置换检验 p值: p<0.05=95%置信策略有效, ≥0.10=无效(BLOCK)
            _g12_mc_p = float(_ov.get("monte_carlo_pvalue", 1.0)) if _ov_ran else 1.0
            # G13 Walk-Forward Sharpe衰减: OOS_Sharpe/IS_Sharpe, ≥0.5=通过, <0.3=过拟合
            _g13_wf_decay = float(_ov_wf.get("sharpe_decay", 0.0)) if _ov_ran else 0.0
            # G14 参数稳定性: 0-1评分, >0.5=平台区域(稳定), ≤0.5=尖峰(脆弱)
            _g14_pstab = float(_ov_ps.get("stability_score", 0.0)) if _ov_ran else 0.0
            # G15 OOS/IS收益比: >0.3=可接受, ≤0.2=严重过拟合(BLOCK)
            _g15_oos_is = float(_ov.get("oos_is_ratio", 0.0)) if _ov_ran else 0.0

            _go_nogo_checks = {
                "G1_ann_sim_30pct": _g1_ann >= 30.0,
                "G2_max_dd_15pct": _g2_dd <= 15.0,
                "G3_win_rate_55pct": _g3_wr >= 55.0,
                "G4_sharpe_1_5": _g4_sh >= 1.5,
                "G5_mlp_5pct": _g5_mlp <= 5.0,
                "G6_gap_15pct": _g6_gap <= 15.0,
                "G7_n_symbols_10": _g7_sym >= 10,
                "G8_years_1_5": _g8_yrs >= 1.5,
                "G9_stress_test_pass": _g9_str,
                "G10_df_0_5": _g10_df >= 0.5,
                # v700 P1: 反过拟合 BLOCK 条件 (G11-G15)
                # 任一未通过 → 禁止实盘部署, 直击"模拟牛逼实盘亏钱"根因
                "G11_pbo_0_5": _g11_pbo < 0.5,                    # PBO<0.5 (非过拟合)
                "G12_mc_pvalue_0_05": _g12_mc_p < 0.05,           # MC p<0.05 (95%置信有效)
                "G13_wf_sharpe_decay_0_5": _g13_wf_decay >= 0.5,  # OOS Sharpe≥IS×0.5
                "G14_param_stability_0_5": _g14_pstab > 0.5,      # 参数在平台区域
                "G15_oos_is_ratio_0_3": _g15_oos_is > 0.3,        # OOS收益>IS×30%
            }
            _go_count = sum(1 for v in _go_nogo_checks.values() if v)
            _can_deploy_live = _go_count == 15  # v700 P1: 全部15项通过才允许实盘 (原10项+G11-G15反过拟合)
        except Exception as _gn_err:
            logger.debug("GO/NO-GO 计算异常(降级): %s", _gn_err)

        # v4.0 Phase 6 Task #29.3 缺陷2修复: v4_phase3_status 的 deployment_ban_active 动态化
        # 从统一准入接口获取真实部署禁令状态 (替代原硬编码 True)
        # 用户铁律: "在未达成性能指标前不得推进实盘部署工作"
        try:
            _v4_deployment_ban_active = not self.get_admission_state().get("can_deploy", False)
        except Exception as _da_err:
            logger.debug("admission_state 获取异常(降级为ban): %s", _da_err)
            _v4_deployment_ban_active = True  # 异常时保守激活禁令

        return {
            "total_rounds": self._round_number,
            "total_generations": self.population.generation,
            "population_size": len(self.population.population),
            "engine": engine_stats,
            "latest_population": {
                "mean_gt_score": pop_stats.mean_gt_score if pop_stats else 0,
                "max_gt_score": pop_stats.max_gt_score if pop_stats else 0,
                "diversity": pop_stats.diversity if pop_stats else 0,
                "worldview_distribution": pop_stats.worldview_distribution if pop_stats else {},
            },
            "top_5_elite": self.get_elite_signals(5),
            "risk_status": self.risk_control.get_status()["global"],
            "production_hardening": production_status,
            "crypto_intraday": crypto_status,
            "intraday_metrics": intraday_metrics,
            "frontier_enhancement": frontier_status,
            "chanlun_priceaction": chanlun_status,
            "microstructure": microstructure_status,
            "regime_detection": regime_status,
            "portfolio_optimization": portfolio_opt_status,
            "alpha_mining": alpha_mining_status,
            "dl_forecast": dl_forecast_status,
            "execution_algo": execution_algo_status,
            "multi_agent": multi_agent_status,
            "turtle_trading": turtle_status,
            "anti_overfitting": overfitting_status,
            # B5修复新增: 验证结果持久化 (供跨版本追踪+GO/NO-GO判定使用)
            "validation": validation_persisted or {
                "r2_verdict": None,
                "sim_live_gap_results": {},
                "sim_live_gap_avg_bps": 0.0,
                "paper_trading": None,
                "final_verdict": "NOT_RUN",
                "can_deploy": False,
            },
            # v4.0 Phase 3: 6大短板补齐状态 (L6 功能可用性暴露)
            "v4_phase3_status": {
                "modules_loaded": sum([
                    _STAGED_ADMISSION_AVAILABLE, _GLOBAL_KB_AVAILABLE,
                    _ERR_KB_AVAILABLE, _MAKER_COST_AVAILABLE, _FEATURE_EXTRACTORS_AVAILABLE,
                    _PAIR_TRADING_AVAILABLE, _MTF_RESONANCE_AVAILABLE,
                    _EXTREME_MARKET_TEST_AVAILABLE,
                ]),
                "staged_admission": _STAGED_ADMISSION_AVAILABLE,
                "global_knowledge_base": _GLOBAL_KB_AVAILABLE,
                "err_knowledge_base": _ERR_KB_AVAILABLE,
                "maker_cost_model": _MAKER_COST_AVAILABLE,
                "feature_extractors": _FEATURE_EXTRACTORS_AVAILABLE,
                "pair_trading": _PAIR_TRADING_AVAILABLE,  # Phase 3.3: 风险分散组件
                "mtf_resonance": _MTF_RESONANCE_AVAILABLE,  # Phase 4 Task #27.2: MTF共振
                "extreme_market_test": _EXTREME_MARKET_TEST_AVAILABLE,  # Phase 4 Task #27.3: 极端测试
                # v4.0 Phase 6 Task #29.3 缺陷2修复: deployment_ban_active 动态化
                # (原硬编码 True 永不更新, 即使 Tier 2 通过仍报告 ban_active=True)
                # 现从统一准入接口 get_admission_state() 获取真实部署禁令状态
                "deployment_ban_active": _v4_deployment_ban_active,
            },
            # Phase 3.3: 配对交易风险分散组件状态 (ERR-099应用)
            # 非收益组件(年化2.4%), 价值在于consec_loss改善35倍
            "pair_trading": {
                "available": _PAIR_TRADING_AVAILABLE,
                "enabled": self.pair_trading_strategy is not None,
                "role": "risk_diversification",  # 非收益组件
                "err_reference": "ERR-099",
            },
            # Phase 4.1: 自动复盘引擎状态 (S1 补齐)
            # 用户铁律: "复盘不是摆设" + "复盘是给进化迭代提供真实数据支持的"
            "auto_review": {
                "available": _AUTO_REVIEW_AVAILABLE,
                "enabled": self.auto_review is not None,
                "last_result": self._last_review_result,
                "feedback_loop_enabled": self.review_feedback is not None,
                "role": "evolution_data_support",  # 复盘驱动下一代进化
            },
            # v614-698 闭环下半部分状态 (ERR-20260702-v614-v98) — 用户铁律核心诉求
            # "整个复盘没有作用于整个ai量化技能包" 的根治: 复盘→补丁→应用→overlay→下一代消费
            "review_iteration_loop": {
                "regime_classifier_available": _REGIME_CLASSIFIER_AVAILABLE,
                "auto_param_applier_available": _AUTO_PARAM_APPLIER_AVAILABLE,
                "applier_enabled": self.auto_param_applier is not None,
                "last_overlay": self._last_overlay,
                "last_iteration_record": self._last_iteration_record,
                "last_regime_distribution": getattr(self, "_last_regime_distribution", None),
                "overlay_adjustments_count": (
                    len(self._last_overlay.get("adjustments", []))
                    if self._last_overlay else 0
                ),
                "role": "closed_loop_lower_half",  # 闭环下半部分: 应用补丁+生成overlay
            },
            # Phase 4.3: 全局知识库 + 故障传播状态 (S5 补齐)
            # 用户铁律: "八智能体统一全局经验知识库、任务优先级调度、跨模块联动回滚机制"
            "knowledge_base": {
                "available": _GLOBAL_KB_AVAILABLE,
                "enabled": self._global_kb is not None,
                "agents_registered": len(self._global_kb.list_agents()) if self._global_kb is not None else 0,
                "active_faults": self._global_kb.get_active_faults() if self._global_kb is not None else [],
                "last_rollback": self._last_rollback,
                "role": "cross_module_fault_propagation",  # 跨模块联动回滚
            },
            # Phase 5.1 + v700 P1: 15项 GO/NO-GO 最终判定 (用户铁律: "未达指标禁止部署" + "永不模拟牛逼实盘亏钱")
            # 全部15项通过才允许实盘, 任一未通过 = 实盘部署禁令激活
            # G1-G10: 性能指标 (ann/dd/wr/sharpe/mlp/gap/symbols/years/stress/df)
            # G11-G15: 反过拟合指标 (pbo/mc_pvalue/wf_sharpe_decay/param_stability/oos_is_ratio)
            "go_nogo": {
                "checks": _go_nogo_checks,
                "go_count": _go_count,
                "n_total": 15,  # v700 P1: 10 → 15
                "can_deploy_live": _can_deploy_live,
                "deployment_ban_active": not _can_deploy_live,  # 实盘部署禁令
                "anti_overfitting_gates": {  # v700 P1: 反过拟合BLOCK条件独立暴露
                    "G11_pbo_0_5": _go_nogo_checks.get("G11_pbo_0_5", False),
                    "G12_mc_pvalue_0_05": _go_nogo_checks.get("G12_mc_pvalue_0_05", False),
                    "G13_wf_sharpe_decay_0_5": _go_nogo_checks.get("G13_wf_sharpe_decay_0_5", False),
                    "G14_param_stability_0_5": _go_nogo_checks.get("G14_param_stability_0_5", False),
                    "G15_oos_is_ratio_0_3": _go_nogo_checks.get("G15_oos_is_ratio_0_3", False),
                },
            },
        }


    def export_results(self, filepath: str = "") -> str:
        """导出进化结果"""
        if not filepath:
            filepath = os.path.join(self.data_dir, "evolution_results.json")

        summary = self.get_summary()
        summary["round_results"] = [
            {
                "round": rr.round_number,
                "generation": rr.generation,
                "trades": rr.trades_executed,
                "volume": rr.total_volume,
                "fees": rr.total_fees,
                "liquidations": rr.liquidations,
                "mean_gt_score": rr.population_stats.mean_gt_score if rr.population_stats else 0,
                "max_gt_score": rr.population_stats.max_gt_score if rr.population_stats else 0,
                "diversity": rr.population_stats.diversity if rr.population_stats else 0,
                "elite": rr.elite_signals,
            }
            for rr in self.round_results
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        return filepath

