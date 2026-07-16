"""Agent auxiliary mixin (Phase 7.14b extracted from agent_decision_engine.py).

Contains 10 auxiliary methods: feature extraction, evolution filter, trade notification,
failure detection, v91 adjustment, decision latency tracking, prewarm.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger("hermes.agent_decision_engine")


class AgentAuxiliaryMixin:
    """Mixin providing auxiliary methods for AgentDecisionEngine.

    Extracted in Phase 7.14b to reduce agent_decision_engine.py size.
    All methods use duck-typing (self attribute access) - zero circular deps.
    """

    def _extract_feature_from_data_packet(
        self, data_packet: Dict[str, Any], feature_name: str
    ) -> Any:
        """v699: 从 data_packet 递归查找指定 feature 字段

        归因引擎产出的 logic_patches 的 filter.feature 可能引用任何特征字段
        (如 mtf_htf_trend_4h / atr_pct / rsi_14 等). decide() 的 data_packet
        是嵌套字典, 此方法递归查找.

        查找顺序:
          1. data_packet 顶层
          2. data_packet['indicators'] (指标字典)
          3. data_packet['regime'] (市场状态字典)
          4. 递归遍历所有子字典

        Returns:
            feature 值 (找不到返回 None)
        """
        if not isinstance(data_packet, dict):
            return None

        # 1. 顶层
        if feature_name in data_packet:
            return data_packet[feature_name]

        # 2. 常见子字典
        for _sub_key in ("indicators", "regime", "market", "chanlun_pa", "microstructure"):
            _sub = data_packet.get(_sub_key)
            if isinstance(_sub, dict) and feature_name in _sub:
                return _sub[feature_name]

        # 3. 递归遍历 (深度限制 2 层, 避免性能问题)
        def _recurse(d: Dict[str, Any], depth: int) -> Any:
            if depth > 2:
                return None
            for v in d.values():
                if isinstance(v, dict):
                    if feature_name in v:
                        return v[feature_name]
                    r = _recurse(v, depth + 1)
                    if r is not None:
                        return r
            return None

        return _recurse(data_packet, 0)

    def _eval_evolution_filter(
        self, val: Any, op: str, filter_dict: Dict[str, Any]
    ) -> bool:
        """v699: 评估 logic_patch 的 filter 操作符

        支持的操作符:
          - "eq_direction": 值方向等于 (val > 0 = long, val < 0 = short)
          - "gt" / ">": val > threshold
          - "lt" / "<": val < threshold
          - "ge" / ">=": val >= threshold
          - "le" / "<=": val <= threshold
          - "eq" / "==": val == threshold

        Args:
            val: feature 实际值
            op: 操作符
            filter_dict: filter 字典 (含 threshold 等字段)

        Returns:
            True=匹配, False=不匹配
        """
        try:
            threshold = filter_dict.get("threshold")
            if op == "eq_direction":
                # 方向匹配: val 和 threshold 同号
                if threshold is None:
                    return True
                return (val > 0 and threshold > 0) or (val < 0 and threshold < 0)
            if op in ("gt", ">"):
                return float(val) > float(threshold)
            if op in ("lt", "<"):
                return float(val) < float(threshold)
            if op in ("ge", ">="):
                return float(val) >= float(threshold)
            if op in ("le", "<="):
                return float(val) <= float(threshold)
            if op in ("eq", "=="):
                return val == threshold
            # 未知操作符 → 不匹配 (防御式)
            return False
        except (TypeError, ValueError):
            return False

    def notify_trade_closed(self, trade: Any) -> Optional[Dict[str, Any]]:
        """v598 Phase J: 通知决策引擎一笔 trade 已关闭 (自动联动 FailureModeMatcher)

        由 EvolutionLoop.loop_end() 在迭代 closed trades 时调用.
        内部转发给 FailureModeMatcher.on_trade_closed(pnl, regime) 检测连续亏损.

        闭环:
          1. loop_end 遍历 closed trades, 调用 engine.notify_trade_closed(trade)
          2. matcher.on_trade_closed 累积滑动窗口 + 检测同 regime 连续亏损 ≥3 笔
          3. 命中时返回 FailureAlert dict (供 loop_end 写入 round_results.warnings)
          4. decide() 通过 matcher.detect_failure(current_regime) 查询活跃警报

        ERR-110 应用边界: 仅产出警报供观察/记录, 不直接 block 交易

        Args:
            trade: 已关闭的 trade (dict 或对象, 需含 pnl + regime 字段)

        Returns:
            若触发警报则返回 FailureAlert.to_dict(), 否则返回 None
        """
        # 懒加载 FailureModeMatcher
        if self._failure_mode_matcher is None:
            if not _RFU_AVAILABLE:
                self._failure_mode_matcher = False
            else:
                try:
                    from .regime_failure_understanding import FailureModeMatcher
                    self._failure_mode_matcher = FailureModeMatcher(
                        consec_loss_threshold=3,  # 用户硬约束 max_consec=3
                        window_size=20,
                    )
                except Exception as e:
                    logger.warning(
                        "FailureModeMatcher 实例化失败 (Phase J 降级): %s", e
                    )
                    self._failure_mode_matcher = False

        if self._failure_mode_matcher is False:
            return None

        # 从 trade 提取 pnl + regime (兼容 dict 和对象)
        try:
            if isinstance(trade, dict):
                pnl = float(trade.get("pnl", 0.0))
                regime = str(trade.get("regime", "") or trade.get("market_regime", ""))
            else:
                pnl = float(getattr(trade, "pnl", 0.0))
                regime = str(
                    getattr(trade, "regime", "") or getattr(trade, "market_regime", "")
                )
        except Exception as e:
            logger.debug("notify_trade_closed 字段提取失败: %s", e)
            return None

        try:
            alert = self._failure_mode_matcher.on_trade_closed(pnl, regime)
            return alert.to_dict() if alert is not None else None
        except Exception as e:
            logger.debug("FailureModeMatcher.on_trade_closed 异常: %s", e)
            return None

    def detect_active_failure_alert(self, current_regime: str) -> Optional[Dict[str, Any]]:
        """v598 Phase J: 在 decide() 查询当前 regime 下是否有活跃警报

        Args:
            current_regime: 当前 regime (原始命名, 内部归一化)

        Returns:
            若有活跃警报则返回 FailureAlert.to_dict(), 否则 None
        """
        if self._failure_mode_matcher is False or self._failure_mode_matcher is None:
            return None
        try:
            alert = self._failure_mode_matcher.detect_failure(current_regime)
            return alert.to_dict() if alert is not None else None
        except Exception:
            return None

    def get_failure_matcher_stats(self) -> Dict[str, Any]:
        """v598 Phase J: 获取 FailureModeMatcher 统计 (供 round_results 诊断)"""
        if self._failure_mode_matcher is False or self._failure_mode_matcher is None:
            return {"available": False}
        try:
            stats = self._failure_mode_matcher.get_stats()
            stats["available"] = True
            stats["alert_history"] = self._failure_mode_matcher.get_alert_history()
            return stats
        except Exception as e:
            return {"available": False, "error": str(e)[:80]}

    def _compute_v91_adjustment(
        self,
        atr_val: float,
        close_price: float,
        indicators: Dict[str, Any],
    ) -> Tuple[float, str, float]:
        """v598 Phase 2: 计算 v91 量价双维仓位调整

        将 v91 的 8状态市场分类 + Kelly全局缩放应用到运行时仓位.
        纯调整函数, 不修改任何状态, 线程安全.

        Args:
            atr_val: ATR(14) 值
            close_price: 当前收盘价
            indicators: 技术指标字典 (含 vp_vol_ratio_5)

        Returns:
            (position_multiplier, market_state, market_factor):
                - position_multiplier: 仓位乘数 = market_factor × kelly_scaler
                - market_state: 8状态字符串 (如 "trend/high_vol")
                - market_factor: 8状态因子值

        主动应用教训链:
            ERR-20260701-v91: 量价双维+Kelly ann=65.30% → 运行时集成
            ERR-20260701-v88fp: 状态分类有效 → 用8状态精细化
            ERR-100 (v534 v90): 2/3 Kelly有效 → 应用portfolio Kelly
            用户铁律: "量价分析+凯利公式融会贯通" + "不通过已知数据反推策略"
        """
        if not _V91_RUNTIME_AVAILABLE:
            return 1.0, "unavailable", 1.0

        # 计算 atr_pct (与离线 v89/v90/v91 一致: ATR/close)
        if close_price <= 0 or atr_val <= 0:
            return 1.0, "no_data", 1.0
        atr_pct = atr_val / close_price

        # 获取 vp_vol_ratio_5 (v598 Phase 2 新增的运行时指标)
        vol_ratio = indicators.get("vp_vol_ratio_5", 1.0)
        if not isinstance(vol_ratio, (int, float)) or vol_ratio != vol_ratio or vol_ratio <= 0:
            vol_ratio = 1.0  # 缺失值降级

        # 获取 kelly_scaler (从 portfolio 级累计器)
        kelly_scaler = 1.0
        if self._kelly_scaler_accumulator is not None:
            try:
                kelly_scaler = self._kelly_scaler_accumulator.get_scaler()
            except Exception:
                kelly_scaler = 1.0

        # 计算 v91 仓位调整三元组
        # L7修复: 传入 self._v91_factors 覆盖默认因子 (支持运行时config注入)
        position_multiplier, market_state, market_factor = compute_v91_position_adjustment(
            atr_pct=atr_pct,
            vol_ratio=vol_ratio,
            kelly_scaler=kelly_scaler,
            factors=self._v91_factors,  # None 时自动回退到 V91_DEFAULT_FACTORS
        )

        # 统计 8 状态分布 (每1000次输出一次诊断)
        self._v91_adjustment_count += 1
        self._v91_market_state_stats[market_state] = (
            self._v91_market_state_stats.get(market_state, 0) + 1
        )
        if self._v91_adjustment_count % 1000 == 1:
            logger.info(
                "v598 V91 ADJUST: agent=%s state=%s factor=%.2f scaler=%.2f mult=%.3f "
                "atr_pct=%.4f vol_ratio=%.2f kelly=%s",
                getattr(self.gene, 'agent_id', '?'),
                market_state, market_factor, kelly_scaler, position_multiplier,
                atr_pct, vol_ratio,
                "acc" if self._kelly_scaler_accumulator else "default",
            )

        return position_multiplier, market_state, market_factor

    def _record_decision_latency(self, start: float) -> None:
        """Phase 7J-3: 记录 decide() 耗时 (正常决策路径)

        R19-1 WARN-1修复: 现在记录所有路径(包括 early return None).
        原 Phase 7J-3 设计假设早退<1ms非瓶颈, R20-1实测确认该假设正确:
          - 早退路径仅做字典查找+条件判断, 不含网络I/O或重计算
          - 实测: 早退avg=0.001ms, 完整决策avg=0.016ms, max=0.052ms
          - 记录早退耗时的价值: 完整耗时画像 + 未来回归检测(若某天
            早退耗时突增>1ms, 说明数据获取逻辑被改坏, 可立即定位)
        超过 100ms 时输出 WARNING (业界 HFT 基准: signal generation <10ms).
        """
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._decision_latency_ms.append(latency_ms)
        if latency_ms > 100.0:
            logger.warning(
                "decide() 耗时 %.1fms (超过 100ms 阈值, 影响订单执行速度)",
                latency_ms,
            )

    def _early_return(self, start: float) -> None:
        """R19-1 WARN-1修复: 早退路径统一记录耗时

        替代原 `return None`, 确保所有退出路径都被耗时监控覆盖.
        与 _record_decision_latency 分离是为了: 早退路径未来可加额外埋点
        (如分类统计 GNN/VPIN/Regime 各自的早退频次, 用于优化数据获取顺序).
        """
        self._record_decision_latency(start)
        return None

    def get_decision_latency_report(self) -> Dict[str, Any]:
        """Phase 7J-3: 获取 decide() 决策耗时报告

        Returns:
            dict 含 count/avg_ms/p50_ms/p95_ms/p99_ms/min_ms/max_ms
            业界基准(2026): HFT signal generation <10ms, 散户策略 <100ms
        """
        if not self._decision_latency_ms:
            return {"count": 0}
        arr = sorted(self._decision_latency_ms)
        n = len(arr)
        return {
            "count": n,
            "avg_ms": round(sum(arr) / n, 2),
            "p50_ms": round(arr[n // 2], 2),
            "p95_ms": round(arr[int(n * 0.95)] if n >= 20 else arr[-1], 2),
            "p99_ms": round(arr[int(n * 0.99)] if n >= 100 else arr[-1], 2),
            "min_ms": round(arr[0], 2),
            "max_ms": round(arr[-1], 2),
        }

    def prewarm_strategy(self, symbol: str = "BTC-USDT") -> bool:
        """R19-5 WARN-2修复: 策略预热,避免首次decide()被策略加载阻塞

        问题: 原 Phase 1 设计采用懒加载,首次 decide() 调用时才加载策略实例
              策略加载耗时 50-200ms (含 StrategyRegistry.load + deps 注入)
              首次决策延迟 → 首笔交易信号过期 → 错过最佳入场点
              实盘后果: 首笔交易滑点增大,模拟vs实盘差异扩大

        修复: 在 set_engines() 之后主动调用 prewarm_strategy(symbol)
              提前完成策略加载,首次 decide() 直接使用已加载实例

        Args:
            symbol: 默认交易对(用于deps注入)

        Returns:
            True=预热成功, False=预热失败(将降级到懒加载)
        """
        if self._strategy_registry is None or self._strategy_loaded:
            return self._strategy_loaded
        try:
            deps: Dict[str, Any] = {"symbol": symbol or "BTC-USDT"}
            if self._spot_engine is not None:
                deps["spot_engine"] = self._spot_engine
            if self._perp_engine is not None:
                deps["perp_engine"] = self._perp_engine
            self._strategy_instance = self._strategy_registry.load(
                self.gene.worldview_primary, self.gene,
                deps=deps,
            )
            self._strategy_loaded = True
            logger.debug(
                "策略预热成功 (worldview=%s, symbol=%s)",
                self.gene.worldview_primary, symbol,
            )
            return True
        except Exception as prewarm_err:
            # 预热失败不阻塞,首次decide()时会再次尝试加载
            logger.warning(
                "策略预热失败, 降级到懒加载 (worldview=%s): %s",
                self.gene.worldview_primary, str(prewarm_err)[:150],
            )
            return False

