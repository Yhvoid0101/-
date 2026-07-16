"""Agent setters mixin (Phase 7.14a extracted from agent_decision_engine.py).

Contains 9 setter methods for dependency injection into AgentDecisionEngine.
All methods are pure attribute setters with documentation - zero decision logic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.agent_decision_engine")


class AgentSettersMixin:
    """Mixin providing setter methods for AgentDecisionEngine.

    Extracted in Phase 7.14a to reduce agent_decision_engine.py size.
    All methods use duck-typing (self attribute assignment) - zero circular deps.
    """

    def set_feature_fusion(self, pipeline: Any) -> None:
        """Phase 3.1: 注入 FeatureFusionPipeline 实例 (短板#2 补齐)

        由 EvolutionLoop._init_decision_engines 调用, 将共享的
        FeatureFusionPipeline 注入到每个 Agent 的决策引擎.
        decide() 中用其提取4维融合特征, 作为 signal_score 的增强因子.

        Args:
            pipeline: FeatureFusionPipeline 实例 (None=禁用特征融合)
        """
        self._feature_fusion = pipeline

    def set_regime_detector(self, detector: Any) -> None:
        """Phase 3.2: 注入 RegimeDetectionSystem 实例 (短板#6 补齐)

        由 EvolutionLoop._init_decision_engines 调用, 将共享的
        RegimeDetectionSystem 注入到每个 Agent 的决策引擎.
        decide() 中只读 get_regime() 获取当前市场状态 (HMM 3状态 +
        贝叶斯变点 + GARCH波动率), 根据均值回归策略特性调整仓位倍数.

        ERR-109教训应用: 策略本质是均值回归
          - sideways(震荡) = 均值回归最优状态 → 加仓1.2
          - bull+LOW vol(低波动牛市) → 适度加仓1.1
          - bull+HIGH vol(高波动趋势) = 均值回归劣势 → 减仓0.6
          - bear(熊市) → 减仓0.6
        关键原则: 不过滤逆势trades, 仅调整仓位倍数 (ERR-109 v518证伪)

        Args:
            detector: RegimeDetectionSystem 实例 (None=禁用regime驱动)
        """
        self._regime_detector = detector

    def set_kelly_scaler_accumulator(self, accumulator: Any) -> None:
        """v598 Phase 2: 注入 portfolio级 KellyScalerAccumulator

        由 EvolutionLoop 调用, 将共享的 KellyScalerAccumulator 注入到每个 Agent.
        累计器从运行时实际交易PnL计算Kelly fraction, 避免从离线数据反推.

        Args:
            accumulator: KellyScalerAccumulator 实例 (portfolio级共享)
        """
        self._kelly_scaler_accumulator = accumulator

    def set_v91_factors(self, factors: Optional[Dict[str, float]]) -> None:
        """v598 Phase 2 L7修复: 注入自定义 v91 8状态因子 (可选)

        由 EvolutionLoop 调用, 允许从 config 或基因编码覆盖默认 V91_DEFAULT_FACTORS.
        传入 None 则恢复使用 V91_DEFAULT_FACTORS 基线.
        这是"可覆盖路径"的实现, 使设计原则落地 (避免"声称可覆盖但实际无路径").

        用户铁律: "不通过已知数据反推策略" → 默认因子仅作基线, 可被运行时config覆盖.
        过拟合防护: 因子范围限制在 [0.5, 2.0] (由 get_market_factor_v91 内部保证).

        Args:
            factors: 8状态因子字典 (如 {"trend/high_vol": 1.44, ...}), None=用默认
        """
        self._v91_factors = factors

    def set_hold_bars_adapter(self, adapter: Any) -> None:
        """v598 Phase D: 注入 HoldBarsAdapter 实例 (S2 composite_score 矛盾统一治理)

        由 EvolutionLoop._init_decision_engines 调用, 将共享的 HoldBarsAdapter
        注入到每个 Agent 的决策引擎. decide() 中可调用 adapter.compute_signal_factor()
        获取基于运行时 hold_bars 分布的 signal_score 增强因子 (0.9-1.1 微调).

        ERR-20260701-v88fp: composite_score 与 PnL 负相关 (d=-0.240) → 用 hold_bars (d=+0.461) 替代
        ERR-109: 仅增强非过滤 (策略本质是均值回归, 逆势trades实际盈利更高)
        ERR-110: 仅信号增强非加权决策 (0/10特征通过 P<0.05, 但融合管道信号增强价值≠加权)
        用户铁律: 不通过已知数据反推策略 → 基于运行时分布动态计算, 非历史阈值

        闭环:
          1. EvolutionLoop.loop_end() 调用 adapter.observe(symbol, hold_bars, pnl)
             从已平仓 trades 累积 (hold_bars, pnl) 观察分布
          2. decide() 调用 adapter.compute_signal_factor(symbol, current_hold_bars)
             基于累积分布计算 signal_score 增强因子 ∈ [0.9, 1.1]
          3. 样本不足 (<_MIN_SAMPLES_FOR_ADJUSTMENT=10) 时返回 1.0 (中性, 无影响)

        注: decide() 实际集成留待 Phase L (v97 动态持仓) 完整落地.
            Phase D 仅完成基础设施 (set/inject/observe 路径打通).

        Args:
            adapter: HoldBarsAdapter 实例 (None=禁用 hold_bars 信号增强)
        """
        self._hold_bars_adapter = adapter

    def set_evolution_overlay(
        self,
        logic_patches: Optional[List[Dict[str, Any]]] = None,
        strategy_patches: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """v699: 注入 5维进化建议 overlay (logic_patches + strategy_patches)

        由 EvolutionLoop._init_decision_engines 调用, 将归因引擎产出+应用器
        写入 _active_evolution_overlay.json 的 patches 注入到每个 Agent.

        闭环 (用户铁律"复盘作用于整个AI量化技能包"):
          1. loop_end PerTradeAttributionEngine 归因 → EvolutionSuggestionApplier 应用
          2. _init_decision_engines 加载 overlay → set_evolution_overlay 注入
          3. decide() 消费 patches:
             - logic_patches: trigger(regime+alignment) 匹配 → reject_entry / reduce_position
             - strategy_patches: condition(regime) 匹配 → disable / enable strategy_type
          4. 回测验证 → evaluate_rollbacks(退化则自动回滚)

        安全设计 (防御式集成, 不破坏主决策流程):
          - patch 匹配失败时静默跳过 (不抛异常)
          - 仅 active 状态的 patch 生效 (rolled_back 的被跳过)
          - reject_entry 走 _early_return (保留耗时监控)
          - reduce_position 走 signal_score 乘法衰减

        Args:
            logic_patches: logic 维度 patch 列表 (None=清空)
            strategy_patches: strategy 维度 patch 列表 (None=清空)
        """
        self._logic_patches = logic_patches or []
        self._strategy_patches = strategy_patches or []
        logger.debug(
            "set_evolution_overlay: logic=%d, strategy=%d",
            len(self._logic_patches), len(self._strategy_patches),
        )

    def set_engines(
        self,
        spot_engine: Any = None,
        perp_engine: Any = None,
    ) -> None:
        """注入沙盘引擎引用（Phase 6A 修复 deps 注入缺陷）

        在 EvolutionLoop._init_decision_engines 中调用，让需要 engine 依赖的
        策略类（funding_rate_arb/basis_arbitrage 等）能真正实例化，
        而非静默 fallback 到 GeneDrivenStrategy。

        Args:
            spot_engine: SandboxMatchingEngine 实例（提供现货撮合/持仓/价格查询）
            perp_engine: 永续引擎实例（可选，沙盘模式可能未启用）
        """
        self._spot_engine = spot_engine
        self._perp_engine = perp_engine

    def set_real_leverage_mode(self, enabled: bool) -> None:
        """Phase 7J-6 反温室修复: 设置实盘杠杆模式

        沙盘进化时保持 False (杠杆=1x 避免爆仓, 让策略专注于方向判断);
        实盘部署时设为 True (基于 gene 风险参数计算真实杠杆).

        来源: 反温室铁律 — "沙盘杠杆1x进化出的策略, 在实盘5x杠杆下会因
        仓位不足而收益微小, 或因杠杆放大而爆仓, 二者都是反温室风险"

        Args:
            enabled: True=实盘模式 (基于 gene 计算), False=沙盘模式 (固定 1x)
        """
        self._use_real_leverage = enabled

    def set_bar_index(self, idx: int) -> None:
        """更新当前bar索引（由主循环每tick调用）"""
        self._current_bar_index = idx

