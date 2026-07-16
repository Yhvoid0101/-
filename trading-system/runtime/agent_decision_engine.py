# -*- coding: utf-8 -*-
"""
子Agent决策引擎 (Agent Decision Engine) — 基于基因的确定性交易决策

从 evolution_loop.py 拆分而来（Phase 4 重构），职责：
  - 基于基因参数和当前市场数据做交易决策
  - 不依赖AI模型，完全基于基因参数和规则
  - 决策过程：
    1. 检查世界观是否允许在当前市场状态交易
    2. 计算各信号工具的得分
    3. 如果得分超过置信度阈值，生成入场信号
    4. 根据基因的风控参数设置止损止盈

依赖：
  - AgentGene（基因类型）
  - StrategyRegistry（Phase 1 集成，懒加载）
  - numpy/math（数值计算）
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


from .gene_codec import AgentGene

# v598 Phase 2: v91 量价双维8状态 + Kelly全局缩放 (短板#2 深化运行时集成)
# 来源: _v91_volume_kelly_optimize.py 离线突破 ann=65.30%, CI下限33.98%
# 集成点: decide() 中 regime_adjustment 之后, 将 market_factor * kelly_scaler 乘入仓位
# 设计原则: 只移植PRINCIPLE, 不硬编码v91网格搜索最优因子; Kelly从运行时PnL累计
# 用户铁律: "凯利公式+量价分析融会贯通" + "不要通过已知数据反推策略"
try:
    from .v91_runtime_adapter import (
        classify_market_state_v91,
        get_market_factor_v91,
        compute_v91_position_adjustment,
    )
    _V91_RUNTIME_AVAILABLE = True
except ImportError:
    _V91_RUNTIME_AVAILABLE = False
    def classify_market_state_v91(atr_pct, vol_ratio): return "low_vol/low_vol"
    def get_market_factor_v91(atr_pct, vol_ratio, factors=None): return 1.0
    def compute_v91_position_adjustment(atr_pct, vol_ratio, kelly_scaler, factors=None):
        return 1.0, "low_vol/low_vol", 1.0

# v598 Phase 5: regime_failure_understanding 顶层相对导入 (修复盘点风险点)
# 原缺陷: L1824-1843 用裸路径 + sys.path 注入 + try/except 静默降级(logger.debug)
#   → 导入失败时运行时完全无感知, 6大短板#6 功能静默失效
# 修复: 顶层相对导入 + warning 级日志 + 保留兜底降级
# 来源教训: ERR-20260702-v598p4-deadlock (L7重构未审计所有调用路径)
try:
    from .regime_failure_understanding import (
        RegimeFailureUnderstanding,
        StrategyParadigm,
        FailureMode,
    )
    _RFU_AVAILABLE = True
except ImportError as _rfu_err:
    _RFU_AVAILABLE = False
    RegimeFailureUnderstanding = None  # type: ignore
    StrategyParadigm = None  # type: ignore
    FailureMode = None  # type: ignore
    logger.warning("regime_failure_understanding 不可用 (6大短板#6 功能降级): %s", _rfu_err)

# Phase 14.16r: 市场上下文信号 (资金费率/OI/情绪) — 非侵入式, 失败不阻塞
try:
    from .market_context_signals import generate_market_context_signals
    _MARKET_CONTEXT_AVAILABLE = True
except ImportError:
    _MARKET_CONTEXT_AVAILABLE = False

# v3.21 领先指标集合 (修复根因2: 滞后指标虚假共振)
# 来源: DIRECTION-CHECK智能体诊断 — RSI/MACD/EMA/BB滞后导致反向选择
# 领先指标基于价量关系, 提前反映市场结构变化, 是独立证据
LEADING_SIGNALS: set = {
    "price_momentum_roc",  # ROC价格动量 (无平滑, 直接度量)
    "mfi_divergence",      # MFI资金流量 (价量综合)
    "willr_signal",        # Williams %R (响应快的超买超卖)
    "volume_anomaly",      # 成交量异常 (资金行为先行)
}

logger = logging.getLogger("hermes.evolution_loop")

# v598 Phase L: v97 自适应持仓时间集成 (ERR-109: hold_bars 是 consequence 不是 cause)
# 来源: _v97_adaptive_hold.py compute_adaptive_hold(atr_pct, vol_ratio, adx) -> (hold, regime)
# 集成点: decide() 返回字典新增 suggested_hold_bars 字段 (仅输出, 不修改出场逻辑)
# 设计原则: 市场状态是 CAUSE → 决定适合的持有时间 → hold_bars 是 CONSEQUENCE
# 用户铁律: "凯利+价格行为+量价+多周期融会贯通" + ERR-109 (仅建议非强制出场)
try:
    from ._v97_adaptive_hold import compute_adaptive_hold as _compute_adaptive_hold
    _ADAPTIVE_HOLD_AVAILABLE = True
except ImportError as _ah_err:
    _ADAPTIVE_HOLD_AVAILABLE = False
    def _compute_adaptive_hold(atr_pct: float, vol_ratio: float, adx: float = 0.0) -> Tuple[int, str]:
        # 降级: 返回基线 48 bars (BASE_HOLD_BARS), regime 标记降级
        return 48, "degraded"
    logger.warning("_v97_adaptive_hold 不可用 (Phase L 功能降级): %s", _ah_err)

# Phase 7.14a: Setters extracted to agent_setters.py (Mixin pattern)
from .agent_setters import AgentSettersMixin

# Phase 7.14b: Auxiliary methods extracted to agent_auxiliary.py (Mixin pattern)
from .agent_auxiliary import AgentAuxiliaryMixin

# Phase 7.14c: Regime analysis methods extracted to agent_regime.py (Mixin pattern)
from .agent_regime import AgentRegimeMixin

# Phase 7.14d: Signal computation methods extracted to agent_signal.py (Mixin pattern)
from .agent_signal import AgentSignalMixin


class AgentDecisionEngine(AgentSettersMixin, AgentAuxiliaryMixin, AgentRegimeMixin, AgentSignalMixin):
    """子Agent决策引擎 — 基于基因的确定性交易决策

    不依赖AI模型，完全基于基因参数和当前市场数据做决策。
    决策过程：
      1. 检查世界观是否允许在当前市场状态交易
      2. 计算各信号工具的得分
      3. 如果得分超过置信度阈值，生成入场信号
      4. 根据基因的风控参数设置止损止盈
    """

    def __init__(self, gene: AgentGene):
        self.gene = gene
        self._prev_mm_fused_prob = 0.5
        # R16: 交易频率控制 — 记录每对的上次交易bar索引
        self._last_trade_bar: Dict[str, int] = {}
        self._current_bar_index: int = 0
        # Phase 1 集成：激活35个死策略 + 12个抽象世界观
        # 通过 StrategyRegistry 动态加载 gene.worldview_primary 对应的策略
        # 策略信号作为第10个数据源注入决策流程（与9个sensor并列）
        self._strategy_signal = None  # 缓存最近一次策略信号
        self._strategy_instance = None
        try:
            from .strategy_registry import get_registry, MarketContext
            self._strategy_registry = get_registry()
            self._strategy_market_ctx_type = MarketContext
            # 延迟加载策略实例（首次 decide 时加载，避免 __init__ 阶段依赖未就绪）
            self._strategy_loaded = False
        except Exception:
            self._strategy_registry = None
            self._strategy_market_ctx_type = None
            self._strategy_loaded = True  # 标记为已加载（失败），不再尝试

        # Phase 6A: engine 引用持有 — 修复 deps 注入缺陷
        # 之前只传 {"symbol": symbol} 导致 funding_rate_arb/basis_arbitrage 等
        # 需要 engine 依赖的策略类静默 fallback 到 GeneDrivenStrategy（空壳交付）
        # 来源：核心铁律"禁止空壳交付"+ 子代理集成状态分析报告
        self._spot_engine: Optional[Any] = None
        self._perp_engine: Optional[Any] = None
        # Phase 7J-3: 决策耗时监控 (来源: AWS tick-to-trade Stage 3 "Signal Generation")
        # 之前完全无耗时测量, 无法定位 decide() 热点
        self._decision_latency_ms: deque = deque(maxlen=1000)
        # Phase 7J-6 反温室修复: 实盘杠杆模式标志
        # 之前 _compute_leverage() 固定返回 1 (沙盘避免爆仓)
        # 但实盘部署后杠杆 1x 会大幅降低资金利用率, 进化出的策略未经历过真实杠杆约束
        # 修复: 沙盘模式 use_real_leverage=False (保留 1x), 实盘模式 True (基于 gene 计算)
        # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 沙盘与实盘杠杆差异 = 反温室风险
        self._use_real_leverage: bool = False

        # v447 认知层升级: RegimeFailureUnderstanding (从"过滤"到"理解")
        # 来源: INDICATOR_DEEP_LOGIC.md 第五章4类失败模式 + 用户明确要求
        #   "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
        # 集成位置:
        #   RegimeDetectionSystem (识别层) → RegimeFailureUnderstanding (理解层) → 策略选择 (融合层)
        # 懒加载: 避免循环依赖, 仅在首次 regime_data 到达时初始化
        self._regime_failure_understander: Optional[Any] = None
        self._last_failure_report: Optional[Dict[str, Any]] = None

        # v596 B3: alpha_mining 三档门控监控
        # 来源: 计划要求 alpha_mining 从0.15确认增强升级为主信号门控
        # gate_status: rejected(弱因子<0.3) / weak_gate(0.3-0.6×0.7) / strong_gate(≥0.6×1.0+0.15增强)
        self.alpha_gate_stats: Dict[str, int] = {"rejected": 0, "weak_gate": 0, "strong_gate": 0, "no_data": 0, "untrained_passthrough": 0, "weak_rejected": 0, "direction_mismatch": 0, "strong_direction_mismatch": 0}
        # IC/RankIC/ICIR 因子评估历史 (每周评估, 淘汰 IC<0.03 的因子)
        self.factor_ic_history: List[float] = []
        self._factor_evaluation_bar: int = 0  # 上次因子评估的bar索引

        # v598 Phase 2: v91 量价双维8状态 + Kelly全局缩放 (短板#2 深化)
        # 来源: _v91_volume_kelly_optimize.py 离线突破 ann=65.30%
        # 集成模式: KellyScalerAccumulator 由 EvolutionLoop 注入 (portfolio级共享)
        # 决策时: atr_pct + vp_vol_ratio_5 → 8状态分类 → market_factor × kelly_scaler
        # 设计: 只移植PRINCIPLE不硬编码因子; Kelly从运行时PnL累计避免过拟合
        # L7修复: 增加 _v91_factors 注入路径, EvolutionLoop可从config覆盖默认因子
        self._kelly_scaler_accumulator: Optional[Any] = None
        self._v91_factors: Optional[Dict[str, float]] = None  # None=用V91_DEFAULT_FACTORS
        self._v91_market_state_stats: Dict[str, int] = {}  # 8状态分布统计
        self._v91_adjustment_count: int = 0  # v91调整应用次数

        # Phase 3.1 集成: FeatureFusionPipeline 注入 (短板#2 补齐)
        # 来源: 用户要求"凯利公式、价格行为、量价、多周期结构标准化特征融合管道"
        # 设计: 特征融合仅作为信号增强(0.7-1.3因子), 不直接决策 (遵守ERR-110教训:
        #       0/10特征有预测力, 但特征组合的信号增强价值不等于加权决策)
        # 注入: 由 EvolutionLoop._init_decision_engines() 调用 set_feature_fusion()
        self._feature_fusion: Optional[Any] = None

        # v598 Phase D: HoldBarsAdapter 注入 (composite_score 矛盾统一治理)
        # ERR-20260701-v88fp: composite_score 与 PnL 负相关 (d=-0.240) → 用 hold_bars (d=+0.461) 替代
        # ERR-109: 仅增强非过滤; ERR-110: 仅信号增强非加权决策
        # 注入: 由 EvolutionLoop._init_decision_engines() 调用 set_hold_bars_adapter()
        # 闭环: loop_end observe() → decide() compute_signal_factor()
        self._hold_bars_adapter: Optional[Any] = None

        # v598 Phase J: FailureModeMatcher — 自动联动 regime + 连续亏损 → FailureMode
        # 用户铁律: "深入理解策略在特定市场状态下失败的根本原因" + "形成闭环"
        # 闭环: loop_end notify_trade_closed() → decide() detect_failure() → 警报写入 round_results.warnings
        # ERR-110 边界: 仅产出 failure_alert 供 observe/记录, 不直接 block 交易 (不修改入场/出场条件)
        # 懒加载: 仅在首次 trade 关闭或 decide() 调用时初始化 (避免循环依赖)
        self._failure_mode_matcher: Optional[Any] = None

        # Phase 3.2 集成: RegimeDetectionSystem 注入 (短板#6 补齐)
        # 来源: 用户要求"市场状态分类驱动策略选择"
        # ERR-109教训: 策略本质是均值回归, trend状态下表现差($722),
        #              quiet/sideways状态下好($11985) → 需regime-aware仓位调整
        # 设计原则: 不过滤逆势trades(只调整仓位倍数), 遵守ERR-109
        # 注入: 由 EvolutionLoop._init_decision_engines() 调用 set_regime_detector()
        # 注意: 只读get_regime(), 不调用update() — EvolutionLoop已每周期更新
        self._regime_detector: Optional[Any] = None

        # v699 逐笔归因→5维进化建议 overlay 注入 (用户铁律"复盘作用于整个AI量化技能包")
        # 来源: PerTradeAttributionEngine 产出 → EvolutionSuggestionApplier 应用到 overlay
        # 设计: logic_patches(reject_entry/reduce_position) + strategy_patches(disable/enable)
        # 闭环: loop_end 归因→应用overlay → 下一代 decide() 消费patches → 回测验证→回滚/保留
        # 安全: patch 匹配失败时静默跳过(防御式), 不破坏主决策流程
        # 注入: 由 EvolutionLoop._init_decision_engines() 调用 set_evolution_overlay()
        self._logic_patches: List[Dict[str, Any]] = []
        self._strategy_patches: List[Dict[str, Any]] = []

        # Phase E Batch 1: ModuleIntegrator 接入 — 激活4个深度学习模块
        # 来源：module_integrator.py v4.0 Phase 2 (Diffusion/Transformer/MARL/GNN)
        # 设计：作为辅助确认信号，与 StrategyRegistry 信号同等地位
        #   - 贝叶斯加权融合4模块信号
        #   - 融合信号同向时温和增强（0.10），反向时不改变方向
        #   - 模块降级时用增强版代理信号（非空壳）
        # 铁律14合规：零遗漏 — 已实现模块必须接入主干
        self._module_integrator: Optional[Any] = None
        self._module_integrator_initialized: bool = False

        # Phase E Batch 2: Gate 模块接入 — 11个决策门控模块
        # 来源：fear_greed/institution_retail/gex/sopr/macro_event/
        #       long_short_ratio/liquidation/stablecoin_premium/
        #       smart_money/order_book/urpd_realized_price
        # 设计：每个 Gate 都有 apply_to_decision(decision, ...) 统一接口
        #   - Gate 通过 multiplier 修改 decision["quantity"]
        #   - 多 Gate 链式调用，每个 Gate 看到前一个 Gate 的修改
        #   - quantity <= 0 → 阻断交易；quantity 变化映射为 signal_score 调整
        # 铁律14合规：零遗漏 — 11个已实现 Gate 全部接入主干
        self._gates: Dict[str, Any] = {}
        self._gate_trigger_stats: Dict[str, int] = {}
        # Phase E Batch 2: 急切初始化11个 Gate（不依赖 decide() 运行时数据）
        import importlib as _gate_imp
        _gate_specs = [
            ("fear_greed", "FearGreedGate", "fear_greed_gate"),
            ("institution_retail", "InstitutionRetailGate", "institution_retail_gate"),
            ("gex", "GEXGate", "gex_gate"),
            ("sopr", "SOPRGate", "sopr_gate"),
            ("macro_event", "MacroEventGate", "macro_event_gate"),
            ("long_short_ratio", "LongShortRatioGate", "long_short_ratio"),
            ("liquidation", "LiquidationGate", "liquidation_gate"),
            ("stablecoin_premium", "StablecoinPremiumGate", "stablecoin_premium_gate"),
            ("smart_money", "SmartMoneyGate", "smart_money_gate"),
            ("order_book", "OrderBookGate", "order_book_gate"),
            ("urpd", "URPDGate", "urpd_realized_price"),
        ]
        for _gk, _gcn, _gmn in _gate_specs:
            try:
                _mod = _gate_imp.import_module(f".{_gmn}", package=__package__)
                _cls = getattr(_mod, _gcn)
                if _gk == "smart_money":
                    self._gates[_gk] = _cls(enabled=True, allow_synthetic=False)
                else:
                    self._gates[_gk] = _cls(enabled=True)
                self._gate_trigger_stats[_gk] = 0
            except Exception as _gie:
                logger.warning("Gate[%s]初始化失败: %s", _gk, str(_gie)[:150])

    def decide(
        self, data_packet: Dict[str, Any], balance: float, equity: float,
        frontier: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """根据当前市场数据包做出交易决策

        前沿增强集成：
          - VPIN毒性检测：高毒性市场不交易
          - Dynamic Kelly仓位：贝叶斯更新+分数Kelly
          - ATR优化止损：1:2.5风险回报比+移动止损

        Args:
            data_packet: 数据管道输出的完整数据包
            balance: 当前余额
            equity: 当前权益
            frontier: FrontierEnhancementSystem实例（可选，用于Kelly+ATR止损）

        Returns:
            None = 不交易
            {"action": "long"/"short", "quantity": float, "leverage": int, "stop_loss": float, "take_profit": float}
        """
        # Phase 7J-3: 决策耗时测量 (来源: AWS tick-to-trade Stage 3 "Signal Generation")
        start = time.perf_counter()
        market = data_packet.get("market")
        # R18-4 BLOCK-4修复: 定义close变量(原代码第332-335行引用close但从未定义)
        # 后果: 策略信号增强(35个策略)因NameError永远不生效,进化基于错误信号源
        close = getattr(market, "close", 0.0) if market else 0.0
        indicators = data_packet.get("indicators", {})
        timesfm = data_packet.get("timesfm_prediction", {})
        kronos = data_packet.get("kronos_range", {})
        crypto_intraday = data_packet.get("crypto_intraday", {})
        vpin_data = data_packet.get("vpin", {})
        # v503 Fix 27: 修复 ATR 数据管道 Bug — atr_val 永远=0 导致 Fix 26 从不执行
        # 根因 (P55 诊断): data_packet 中没有 "atr" 键, ATR 实际在 indicators["atr_14"]
        #   - evolution_loop.py L1378 正确读取: tick.get("indicators", {}).get("atr_14", 0.0)
        #   - 但 agent_decision_engine.py L250 错误读取: data_packet.get("atr", 0.0) → 永远 0.0
        # 影响: Fix 26 (L971-979 ATR 自适应 TP/SL) 从不执行 + frontier ATR 失效 + volatility=0
        # 修复: 从 indicators["atr_14"] 读取, 与 data_pipeline._compute_indicators L1603 一致
        # 铁律: "理解底层逻辑, 而非表面指标" — Fix 26 代码正确但数据管道断裂 = 治标不治本
        atr_val = indicators.get("atr_14", 0.0) if indicators else 0.0
        chanlun_pa = data_packet.get("chanlun_pa", {})
        microstructure_data = data_packet.get("microstructure", {})
        regime_data = data_packet.get("regime", {})
        alpha_mining_data = data_packet.get("alpha_mining", {})
        dl_forecast_data = data_packet.get("dl_forecast", {})
        multi_agent_data = data_packet.get("multi_agent", {})
        turtle_data = data_packet.get("turtle", {})
        # 多模型集成信号（LightGBM + Prophet + Qwen 融合）
        # 传感器地位：与TimesFM/Kronos平等，子Agent可选择使用
        multi_model_data = data_packet.get("multi_model", {})
        # v2.25: GNN跨资产关系 + CrossSectional横截面动量信号
        # 来源: openreview.net 2026 GNN + Antonacci Dual Momentum
        # 用途: 1)系统性风险时不交易 2)横截面动量方向过滤
        gnn_data = data_packet.get("gnn_cross_asset", {})
        cross_sectional_data = data_packet.get("cross_sectional_momentum", {})

        if not market:
            return self._early_return(start)

        # v3.27 临时诊断: 在decide最早位置记录(每2000次打印1次)
        if not hasattr(self, '_early_diag_counter'):
            self._early_diag_counter = 0
        self._early_diag_counter += 1
        if self._early_diag_counter % 50 == 1:
            logger.warning(
                "v327 EARLY DIAG: agent=%s symbol=%s market_close=%.2f gnn=%s vpin=%s regime=%s cs=%s dl=%s",
                getattr(self.gene, 'agent_id', '?'),
                data_packet.get("symbol", "?"),
                close,
                "Y" if gnn_data else "N",
                "Y" if vpin_data else "N",
                regime_data.get("risk_level", "?") if regime_data else "N",
                "Y" if cross_sectional_data else "N",
                "Y" if dl_forecast_data else "N",
            )

        symbol = data_packet.get("symbol", "")
        # R16: 最小交易间隔(已禁用) — 由进化自行决定频率
        # v503 P3-2 Fix 3f: 移除时间倒流检查 (根治92%沉默的关键BUG)
        # 根因: evolution_loop.py:1347 每轮调用 set_bar_index(tick_idx),
        #   tick_idx 从 0 开始, 但 _last_trade_bar 跨轮次保留.
        #   第二轮 tick 0 时, _current_bar_index=0 < last_bar=5(上轮交易bar),
        #   触发 return, 导致每轮前 N 个 tick 全部沉默 (70% 拦截率).
        # 修复: 完全移除此检查 (注释本就说"禁用", 但代码未真正禁用)
        # 用户明确要求: "杜绝妥协性处理方式" — 这是 BUG, 必须移除

        # v2.25: GNN系统性风险硬门控 — SRI≥0.8或传染概率≥0.7时不交易
        # 来源: 2008金融危机+2020 COVID实证,相关性收敛=恐慌=必亏
        if gnn_data:
            gnn_sri = gnn_data.get("systemic_risk_index", 0.0)
            gnn_opp = gnn_data.get("best_opportunity") or {}
            gnn_contagion = gnn_opp.get("contagion_probability", 0.0)
            if gnn_sri >= 0.8 or gnn_contagion >= 0.7:
                return self._early_return(start)

        # v2.25: CrossSectional横截面动量方向过滤
        # 如果当前symbol在short_assets中,不做多;如果在long_assets中,不做空
        # 来源: Antonacci Dual Momentum — 截面排名Top做多,Bottom做空
        cs_long_assets = cross_sectional_data.get("long_assets", []) if cross_sectional_data else []
        cs_short_assets = cross_sectional_data.get("short_assets", []) if cross_sectional_data else []
        cs_crash_prob = cross_sectional_data.get("momentum_crash_probability", 0.0) if cross_sectional_data else 0.0
        # 动量崩溃概率≥0.8时不交易（来源: Daniel Moskowitz Momentum Crash研究）
        if cs_crash_prob >= 0.8:
            return self._early_return(start)

        # 前沿增强：VPIN毒性检测（双重保险，主循环已过滤，这里再确认）
        if vpin_data and not vpin_data.get("can_trade", True):
            # v3.27 临时诊断: VPIN阻止交易时打印
            if not hasattr(self, '_vpin_diag_counter'):
                self._vpin_diag_counter = 0
            self._vpin_diag_counter += 1
            if self._vpin_diag_counter % 200 == 1:
                logger.warning(
                    "v327 VPIN BLOCK: agent=%s vpin_data=%s",
                    getattr(self.gene, 'agent_id', '?'),
                    {k: v for k, v in vpin_data.items() if k != 'history'},
                )
            return self._early_return(start)

        # 市场状态检测：Regime检测风险等级
        # 来源：Hamilton 1989 HMM + Adams-MacKay 2007 变点检测
        #
        # v503 P3-2 Fix 6a+6b: EXTREME 硬过滤→衰减 + confidence 字段名修正
        # 根因1 (Fix 6b): regime_data 字典实际字段名是 "market_confidence",
        #   但原代码读取 "confidence" → 永远返回默认值 1.0 (≥0.8)
        #   → 所有 EXTREME 都被判定为"高置信度" → L341 硬过滤 100% 触发
        #   → v503 P3-4 诊断: L341 拦截 1399/1399 次 _early_return (82.7% of decide)
        # 根因2 (Fix 6a): 即使 confidence 正确, 82.7% 拦截率 = 进化算法无法工作
        #   用户要求: "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
        #   + "策略要有深度, 要理解底层逻辑, 而非表面指标" + "杜绝妥协性处理方式"
        # 修复:
        #   - Fix 6b: 读取 "market_confidence" (正确字段名)
        #   - Fix 6a: EXTREME 不直接弃权, 而是按 confidence 分级衰减
        #     confidence≥0.8: 衰减至20% | 0.5-0.8: 衰减至40% | <0.5: 衰减至60%
        # 进化算法学习"在 EXTREME 市场如何降低仓位/选择反向信号"
        regime_extreme_penalty = 1.0
        if regime_data:
            risk_level = regime_data.get("risk_level", "LOW")
            if risk_level == "EXTREME":
                regime_conf = regime_data.get("market_confidence", 1.0)  # Fix 6b: 正确字段名
                if regime_conf >= 0.8:
                    regime_extreme_penalty = 0.2  # 极高风险: 衰减至20%
                elif regime_conf >= 0.5:
                    regime_extreme_penalty = 0.4  # 高风险: 衰减至40%
                else:
                    regime_extreme_penalty = 0.6  # 中风险: 衰减至60%

        # v447 认知层升级: 从"硬过滤"到"理解后切换"
        # 用户明确要求: "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
        # 当 regime 失败模式被识别时, 不直接弃权, 而是:
        #   1. 切换策略范式 (trend_following → mean_reversion)
        #   2. 调整 SL/TP/仓位系数 (基于失败模式知识库)
        #   3. 仅在"完全无有效策略"时才弃权
        # 来源: INDICATOR_DEEP_LOGIC.md 第五章4类失败模式 + 第六章维度独立性
        regime_adjustment = self._apply_regime_failure_understanding(
            regime_data, indicators, start,
        )
        if regime_adjustment is None:
            # 认知层建议完全弃权 (无任何有效策略范式)
            return self._early_return(start)

        # 深度学习时序预测：方向预测置信度过低时不交易
        # 来源：MSTFNet 2026 + TFT Google 2021
        # v503 P3-2 Fix 3b: DL置信度硬过滤→衰减 (根治92%沉默)
        # 用户明确要求: "策略要有深度, 要理解底层逻辑, 而非表面指标"
        # DL 三源完全不一致时, 不直接弃权, 而是降低信号置信度
        # 进化算法可以学习"在DL不确定时如何依赖其他信号源"
        dl_penalty = 1.0
        if dl_forecast_data:
            dl_confidence = dl_forecast_data.get("confidence", 1.0)
            if dl_confidence < 0.2:
                dl_penalty = 0.5  # DL 不确定: 信号置信度衰减至50%

        # 模块2：市场状态判断
        market_state = self._assess_market_state(market, indicators)
        # v3.27 临时诊断: 在市场状态检查点记录
        if not hasattr(self, '_ms_diag_counter'):
            self._ms_diag_counter = 0
        self._ms_diag_counter += 1
        if self._ms_diag_counter % 1000 == 1:
            logger.warning(
                "v327 DIAG market_state: agent=%s suitable=%s regime=%s trend=%.3f vol_regime=%s close=%.2f",
                getattr(self.gene, 'agent_id', '?'),
                market_state["suitable_for_trading"],
                market_state["regime"],
                market_state["trend_strength"],
                market_state["volatility_regime"],
                close,
            )
        # v503 P3-2 Fix 3a: 市场状态硬过滤→衰减 (根治92%沉默)
        # 用户明确要求: "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
        # 不利市场状态下不直接弃权, 而是大幅降低信号置信度
        # 让进化算法学习"在不利市场如何降低仓位/选择更保守的信号组合"
        market_state_penalty = 1.0
        if not market_state["suitable_for_trading"]:
            market_state_penalty = 0.3  # 不利市场: 信号置信度衰减至30%

        # v3.31 根因修复 CRITICAL: 趋势过滤 — 下跌趋势中不做多
        # 根因 (audit.log v3.30 实测): 36笔交易中大部分在BTC下跌趋势中做多=逆势
        #   - mfi_divergence:long 16笔全亏 (MFI<25时做多=超卖反弹=逆势)
        #   - 时间止损后胜率仅3.7% (6bars后价格继续下跌)
        #   - 价格波动平均1.88%, 但exit < entry 占94.4% (系统性下跌)
        # 来源: 趋势跟随原则 "the trend is your friend" (Ed Seykota)
        #       + 双移动均线系统 (Donchian/Turtle) — 价格在均线下方=下跌趋势
        #       + 量化交易基本原则 "don't fight the tape"
        # 铁律: "不要根据现有的历史数据去为了数字好看去过拟合, 而是真正的找到根因"
        #
        # v503 P3-2 Fix 3d: 趋势过滤硬过滤→衰减 (根治92%沉默)
        # 用户明确要求: "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
        # 下跌趋势中不直接弃权, 而是大幅降低信号置信度
        # 进化算法可以学习"在下跌趋势中如何做空或降低仓位"
        sma_50 = indicators.get("sma_50", close)
        if close < sma_50:
            market_state_penalty *= 0.3  # 下跌趋势: 额外衰减至30%

        # 模块3：入场信号计算（含加密货币日内交易信号 + 缠论裸K信号）
        # v2.44 RC-007: 传入 market_state 让 regime 参与方向决策 (废除纯 scores 比较)
        signal_score, signal_direction, signal_count = self._compute_entry_signals(
            market, indicators, timesfm, kronos, crypto_intraday, chanlun_pa,
            market_state=market_state,
        )

        # v503 P3-2 Fix 4: signal_score=0 时设置 floor 并根据趋势选择方向
        # 根因: _compute_entry_signals 内部有 7 个 return 0.0 硬过滤 (信号冲突/弱信号)
        #   49.2% 的调用返回 0.0, 55.1% 的 direction="neutral", 0% 做空信号
        #   导致 92%+ 种群沉默, 进化算法无法工作
        # 修复: signal_score=0 时, 不直接跳过, 而是设置 floor 并根据趋势选择方向
        # 用户要求: "策略要有深度, 要理解底层逻辑, 而非表面指标"
        #   当前先用 floor 让进化算法开始工作 (后续迭代修改 _compute_entry_signals 内部)
        # v503 Fix 16: 基于市场状态的 fallback 方向 (根治 13.3% 胜率 + Short 0% 胜率)
        # 根因诊断 (P42 诊断: 前30样本 83% 是 short, Short 胜率 0%):
        #   Fix 13 的 fallback 让 mean_reversion 族在 close>sma_50 时盲目 short
        #   但 BTC 整体上涨趋势中 short = 必亏 (Short 胜率 0%)
        #   "均值回归"≠盲目逆势, 而是在超买/超卖时逆势 (用户铁律: "理解底层逻辑")
        # 修复原理 (用户铁律: "策略要有深度, 要理解底层逻辑, 而非表面指标"):
        #   1. 均值回归族: 只在真正超买(RSI>70 或 close>Bollinger上轨)时 short,
        #      超卖(RSI<30 或 close<Bollinger下轨)时 long, 否则顺势
        #      → 避免 BTC 上涨趋势中盲目 short 全亏
        #   2. 趋势跟随族: 始终顺势 (close>sma → long)
        #   3. contrarian_bias > 0.5 时翻转方向 (保持种群多样性, 进化需要选择压力)
        #   4. 超买超卖信号优先于 contrarian_bias (不翻转真正的均值回归信号)
        # 数学验证:
        #   BTC 上涨趋势中, RSI>70 约 15% 时间 → mean_reversion 族 85% 时间顺势 long
        #   contrarian_bias > 0.5 的 agent 约 50% → 提供多样性 (部分 short)
        #   预期: 60-70% agent long (顺势), 30-40% agent short (多样性) → 胜率 50%+
        # 来源: 均值回归策略本质 (Connor's RSI-2) + 进化算法多样性原理
        # 铁律: "杜绝模拟牛逼,实盘亏钱" — 盲目逆势 = 必亏 = 无进化价值
        if signal_score == 0.0 or signal_direction == "neutral":
            signal_score = 0.1  # floor: 给进化算法一个非零起点

            # v503 Fix 17: 优先使用 crypto_intraday/chanlun_pa 信号方向 (根治空 indicators → 全 short)
            # 根因 (P44 诊断): 34% 调用 indicators dict 为空, Fix 16 的
            #   sma_50_trend = indicators.get("sma_50", close) → 返回 close (默认值)
            #   trend_up = close > close = False → 所有 agent 选 short → 69.7% short
            #   → Short 胜率 0% (BTC 上涨趋势中 short = 必亏)
            # 修复原理 (用户铁律: "策略要有深度, 要理解底层逻辑, 而非表面指标"):
            #   1. indicators 可用时: 使用趋势+超买超卖逻辑 (原 Fix 16)
            #   2. indicators 缺失时: 从 crypto_intraday/chanlun_pa 提取方向
            #      (这些信号源永远可用 — P44: 0% 为空, 是实际市场信号, 非默认值)
            #   3. 无任何信号时: 默认 trend_up=True (BTC 长期上涨趋势, 非 False)
            # 数学验证: BTC 2010-2026 年化回报 ~150%, 长期上涨 → 默认 long 优于 short
            has_indicators = bool(indicators) and "sma_50" in indicators

            if has_indicators:
                # indicators 可用: 使用趋势+超买超卖逻辑 (原 Fix 16 逻辑)
                sma_50_trend = indicators.get("sma_50", close)
                trend_up = close > sma_50_trend
                rsi = indicators.get("rsi_14", 50.0)
                bollinger_upper = indicators.get("bollinger_upper", close * 1.02)
                bollinger_lower = indicators.get("bollinger_lower", close * 0.98)
                overbought = rsi > 70.0 or close > bollinger_upper
                oversold = rsi < 30.0 or close < bollinger_lower

                # v14.40 Fix 2: 多周期趋势确认 — 根治 ATOM/UNI 下跌市场仍95%做多
                # 根因 (diag_direction.py): close>sma_50 在下跌中继仍True (相对历史均值高)
                #   - ATOM 95.6% long 但市场下跌, long -5.63% / short +1.07%
                #   - UNI 92.9% long 但市场下跌, long -5.63% / short +7.07%
                # 修复原理 (铁律: "理解底层逻辑"):
                #   1. 单均线(sma_50)无法识别趋势反转, 需多周期(sma_200)确认
                #   2. 长期下跌市场 close<sma_200, 短期反弹 close>sma_50 是"死猫跳"
                #   3. 真趋势 = close>sma_50 AND close>sma_200 (双均线确认)
                #   4. 配合 ROC(rate of change) 短期动量确认
                # 数学验证:
                #   ATOM 下跌: close<sma_200, ROC<0 → trend_up=False → 95.6%long转为short
                #   UNI 下跌: close<sma_200, ROC<0 → trend_up=False → 92.9%long转为short
                #   INJ 上涨: close>sma_200, ROC>0 → trend_up=True 保持long (+10.12%)
                # 来源: 双均线趋势过滤 (Golden Cross/Death Cross) + 动量确认
                # 铁律: "杜绝模拟牛逼,实盘亏钱" — 下跌市场做多 = 必亏 = 无进化价值
                # v14.42 Fix 5: 使用ema_200 (data_pipeline.py实际生成的指标)
                # 根因: Fix 2引用sma_200但indicators只有ema_200 → Fix 2未生效
                sma_200 = indicators.get("ema_200", sma_50_trend)  # ema_200已存在
                sma_20 = indicators.get("sma_20", sma_50_trend)
                # 短期ROC: 5根K线动量 (如果有close历史) 或用close vs sma_20代替
                roc_short = (close - sma_20) / max(sma_20, 1e-9) if sma_20 > 0 else 0.0
                # 多周期趋势确认: 双均线 + 短期动量
                # 强上涨: close>sma_50 AND close>sma_200 AND roc>0
                # 强下跌: close<sma_50 AND close<sma_200 AND roc<0
                # 混合: 维持原 trend_up (让均值回归/反转逻辑生效)
                if close < sma_200 and roc_short < -0.005:
                    # 长期下跌 + 短期下跌 = 强下跌趋势, 强制 trend_up=False
                    trend_up = False
                elif close > sma_200 and roc_short > 0.005:
                    # 长期上涨 + 短期上涨 = 强上涨趋势, 强制 trend_up=True
                    trend_up = True
                # else: 混合信号, 保持原 trend_up (close vs sma_50)

            else:
                # indicators 缺失: Phase 14.16 修复 — 用实际价格趋势替代默认True
                # 根因: 2019年BTC从3848跌到129(-96.65%), 默认trend_up=True→全long→全亏
                # 金融原理: 趋势判断必须基于实际价格数据, 不能用硬编码假设
                # 方法1: close vs open (当根K线方向, 短期趋势)
                # 方法2: close vs (high+low)/2 (价格在区间上半/下半, 中期趋势)
                # 方法3: 保留crypto_intraday/chanlun_pa信号覆盖
                market_open = getattr(market, 'open', close) if market else close
                market_high = getattr(market, 'high', close) if market else close
                market_low = getattr(market, 'low', close) if market else close
                range_mid = (market_high + market_low) / 2.0 if (market_high + market_low) > 0 else close

                # 短期趋势: close vs open + close vs range_mid (双重确认)
                # close>open 且 close>range_mid → 强势上涨 → trend_up=True
                # close<open 且 close<range_mid → 强势下跌 → trend_up=False
                # 混合信号 → 默认中性偏多(close>open优先, 因为加密长期上涨)
                if close > market_open and close > range_mid:
                    trend_up = True   # 双重看涨
                elif close < market_open and close < range_mid:
                    trend_up = False  # 双重看跌
                else:
                    trend_up = (close >= market_open)  # 混合: 用close vs open

                overbought = False
                oversold = False

                # 从 crypto_intraday 提取信号方向 (永远可用, P44: 0% 为空)
                if crypto_intraday:
                    vwap_sig = crypto_intraday.get("vwap_signal", {}).get("signal", "neutral")
                    funding_sig = crypto_intraday.get("funding_signal", {}).get("signal", "neutral")
                    oi_div = crypto_intraday.get("oi_divergence", {}).get("divergence", "none")
                    if "long" in vwap_sig or vwap_sig == "vwap_support" or \
                       "long" in funding_sig or oi_div == "bullish":
                        trend_up = True
                    elif "short" in vwap_sig or vwap_sig == "vwap_resistance" or \
                         "short" in funding_sig or oi_div == "bearish":
                        trend_up = False

                # chanlun_pa 强信号覆盖 (永远可用, P44: 0% 为空)
                if chanlun_pa and chanlun_pa.get("signal") in ("long", "short"):
                    pa_strength = chanlun_pa.get("strength", 0)
                    if pa_strength >= 0.5:  # 强信号才覆盖
                        trend_up = (chanlun_pa["signal"] == "long")

            # 根据基因世界观决定方向解读
            worldview = getattr(self.gene, 'worldview_primary', 'trend_following')
            contrarian_bias = getattr(self.gene, 'worldview_contrarian_bias', 0.0)

            MEAN_REVERSION_WORLDVIEWS = {
                "mean_reversion", "stat_arb_pairs", "funding_rate_predictor",
            }
            if worldview in MEAN_REVERSION_WORLDVIEWS:
                # 均值回归族: 优先超买超卖信号, 非超买超卖时顺势
                if overbought:
                    signal_direction = "short"  # 超买 → short (真正的均值回归)
                elif oversold:
                    signal_direction = "long"   # 超卖 → long (真正的均值回归)
                else:
                    # 非超买超卖: 顺势 (避免盲目逆势亏损)
                    signal_direction = "long" if trend_up else "short"
            else:
                # 趋势跟随族: 顺势交易 (close>sma → long)
                signal_direction = "long" if trend_up else "short"

            # contrarian_bias 翻转方向 (增加种群多样性)
            # 但超买超卖信号不翻转 (让真正的均值回归信号生效)
            if contrarian_bias > 0.5 and not overbought and not oversold:
                signal_direction = "short" if signal_direction == "long" else "long"

            signal_count = max(signal_count, 1)

            # Phase F 修复: 当方向被覆盖时, 更新 _last_signal_source 以反映实际决策依据
            # 根因: 无信号触发时 _last_signal_source="none" → has_auditable_signal_source 拒绝
            #   信号触发但方向被覆盖时 _last_signal_source 保留原方向 → direction-conflict 拒绝
            # 修复: 用覆盖后的方向更新 signal_source, 确保审计可追溯 + 方向一致
            _current_source = getattr(self, "_last_signal_source", "none")
            if _current_source == "none" or not _current_source:
                # 无信号触发, 使用趋势默认方向
                self._last_signal_source = f"trend_default:{signal_direction}"
            else:
                # 信号触发但方向被覆盖 (均值回归/反向偏好) — 用覆盖方向替换
                self._last_signal_source = f"override:{signal_direction}"

        # v447 认知层应用: 根据 regime_adjustment 调整信号得分
        # 当 regime 失败模式被识别时, 减弱原策略信号 (而非直接弃权)
        # 这实现了用户要求的"理解后切换"而非"硬过滤"
        if regime_adjustment and regime_adjustment.get("signal_score_multiplier", 1.0) < 1.0:
            signal_score *= regime_adjustment["signal_score_multiplier"]

        # Phase 14.23: regime 方向翻转 — 根治多样性不足 (全 long, 0 short)
        # 根因:
        #   1. BTC 整体上涨 -> close > sma_50 -> trend_up=True -> fallback 全 long
        #   2. regime 识别 counter_trend_short 失败模式, 但只降低 signal_score, 不翻转方向
        #   3. market_state_penalty 在 close < sma_50 时 *= 0.3, 但翻转后 short in downtrend 是顺势
        # 修复: 当 regime 建议做空范式 (counter_trend_short / *_short) 且当前方向为 long 时:
        #   - 翻转 signal_direction (long -> short)
        #   - 重置 market_state_penalty = 1.0 (short in downtrend 是顺势, 不应惩罚)
        #   - 重置 dl_penalty = 1.0 (给翻转后的信号新鲜起点)
        # 数学验证:
        #   - counter_trend_short = regime=downtrend + 原方向=long = 逆势
        #   - 翻转后 short in downtrend = 顺势 (trend-following)
        #   - 顺势交易不应被 market_state_penalty 惩罚
        if regime_adjustment and signal_direction == "long":
            _p1423_paradigm = regime_adjustment.get("strategy_paradigm", "")
            _p1423_failure = regime_adjustment.get("failure_mode", "")
            # 检测需要翻转的场景: regime 建议做空范式 或 失败模式为 counter_trend
            _p1423_needs_flip = (
                "short" in _p1423_paradigm
                or "counter_trend" in _p1423_failure
            )
            if _p1423_needs_flip:
                signal_direction = "short"
                market_state_penalty = 1.0  # short in downtrend 是顺势, 重置惩罚
                dl_penalty = 1.0  # 给翻转后的信号新鲜起点
                try:
                    self._last_signal_source = f"regime_flip:{_p1423_paradigm}"
                except Exception:
                    pass
                try:
                    logger.info(
                        "[Phase14.23] regime_flip: short <- long "
                        f"(paradigm={_p1423_paradigm}, failure={_p1423_failure})"
                    )
                except Exception:
                    pass

        # v503 P3-2 Fix 3: 统一应用市场状态 + DL 置信度 penalty
        # 之前是硬过滤直接 return, 现在是衰减让进化算法学习
        # 用户明确要求: "策略要有深度, 要理解底层逻辑, 而非表面指标"
        # 衰减后的 signal_score 仍需通过 production_threshold 检查 (Fix 3c 会处理)
        signal_score *= market_state_penalty * dl_penalty * regime_extreme_penalty

        # 市场微观结构增强：Hawkes爆发权重 + 订单簿不平衡确认
        # 来源：Hawkes 1971 + quant-flow 2026 + Kyle 1985
        if microstructure_data:
            ms_state = microstructure_data
            hawkes_weight = ms_state.get("hawkes_signal_weight", 1.0)
            imb_direction = ms_state.get("imbalance_direction", "neutral")
            imb_strength = ms_state.get("imbalance_strength", 0.0)

            # Hawkes爆发权重调整信号得分
            signal_score = signal_score * hawkes_weight

            # 订单簿不平衡方向确认/矛盾
            if imb_direction != "neutral":
                if (imb_direction == "bullish" and signal_direction == "long") or \
                   (imb_direction == "bearish" and signal_direction == "short"):
                    # 方向一致，增强信号
                    signal_score *= (1.0 + imb_strength * 0.3)
                else:
                    # 方向矛盾，减弱信号
                    signal_score *= (1.0 - imb_strength * 0.2)

        # 深度学习时序预测增强：信号增强因子 + 方向确认
        # 来源：MSTFNet 2026 (56.3%方向准确率) + TFT Google 2021
        if dl_forecast_data and dl_forecast_data.get("should_enhance", False):
            enhancement_factor = dl_forecast_data.get("enhancement_factor", 1.0)
            dl_direction = dl_forecast_data.get("direction_label", "neutral")
            dl_confidence = dl_forecast_data.get("confidence", 0.0)

            # 信号增强因子调整信号得分
            signal_score = signal_score * enhancement_factor

            # 方向确认：深度学习方向与信号方向一致时额外增强
            if dl_direction != "neutral" and dl_confidence > 0.6:
                if (dl_direction == "bullish" and signal_direction == "long") or \
                   (dl_direction == "bearish" and signal_direction == "short"):
                    signal_score *= (1.0 + dl_confidence * 0.1)

        # ML因子挖掘增强：三档门控升级 (v596 B3)
        # 来源：QuantaAlpha 2026 + AlphaPROBE 2026 (遗传规划因子挖掘)
        # v596 B3变更: 从"确认信号仅增强"升级为"主信号门控"
        #   - alpha_confidence < 0.3  → 拒绝信号 (gate closed, signal_score=0)
        #   - 0.3 ≤ confidence < 0.6  → 弱门控 (方向一致×0.7, 不一致拒绝)
        #   - alpha_confidence ≥ 0.6  → 强门控 (方向一致×1.0+0.15增强, 不一致拒绝)
        # 设计原则: 弱因子不应入场, 中等因子降仓, 强因子增强(保留原0.15增强)
        if alpha_mining_data:
            alpha_direction = alpha_mining_data.get("direction", "neutral")
            alpha_confidence = alpha_mining_data.get("confidence", 0.0)

            # 根治(2026-07-16): alpha_mining gate 不再归零 signal_score
            # 原bug: alpha_mining gate 把 signal_score 归零 → score=0.0000 → 进化系统空转
            # 根因: alpha_mining 未训练时 conf=0+neutral, 真弱因子 conf<0.3
            #   原逻辑: 未训练*=0.5, 真弱因子=0.0, 方向不一致=0.0
            #   结果: 大量策略 score=0 → 无法交易 → 进化空转
            # 修复: alpha_mining 只衰减不归零, 让进化算法有非零起点
            # 铁律15: 发现问题立马根治 — 进化空转是致命问题
            if alpha_confidence < 0.3:
                if alpha_confidence == 0.0 and alpha_direction == "neutral":
                    # alpha_mining 未训练完成: 衰减0.5(不归零)
                    signal_score *= 0.5
                    self.alpha_gate_stats["untrained_passthrough"] += 1
                else:
                    # 真弱因子: 衰减0.3(不归零,让进化有选择压力)
                    signal_score *= 0.3
                    self.alpha_gate_stats["weak_rejected"] += 1
            elif alpha_confidence < 0.6:
                # 中等因子: 弱门控
                if alpha_direction == signal_direction and alpha_direction != "neutral":
                    signal_score *= 0.7
                    self.alpha_gate_stats["weak_gate"] += 1
                else:
                    # 方向不一致: 衰减0.3(不归零)
                    signal_score *= 0.3
                    self.alpha_gate_stats["direction_mismatch"] += 1
            else:
                # 强因子: 强门控
                if alpha_direction == signal_direction and alpha_direction != "neutral":
                    signal_score *= (1.0 + alpha_confidence * 0.15)
                    self.alpha_gate_stats["strong_gate"] += 1
                else:
                    # 方向不一致: 衰减0.3(不归零)
                    signal_score *= 0.3
                    self.alpha_gate_stats["strong_direction_mismatch"] += 1

            # v596 B3: IC/RankIC/ICIR 因子淘汰机制钩子 (每周评估一次)
            # 评估周期: bars_per_week = 168 (与rb.reset_weekly一致)
            bars_per_week = 168
            if self._current_bar_index - self._factor_evaluation_bar >= bars_per_week:
                factor_ic = alpha_mining_data.get("ic", 0.0)
                self.factor_ic_history.append(factor_ic)
                self._factor_evaluation_bar = self._current_bar_index
                # 淘汰 IC<0.03 的因子 (通过降低未来 alpha_confidence 权重实现)
                if factor_ic < 0.03 and len(self.factor_ic_history) > 3:
                    logger.warning(
                        "alpha_mining因子IC=%.4f < 0.03阈值, 建议淘汰 (历史IC均值=%.4f)",
                        factor_ic, sum(self.factor_ic_history) / len(self.factor_ic_history),
                    )
        else:
            self.alpha_gate_stats["no_data"] += 1

        # 根治(2026-07-16): alpha gate 后重新应用 floor
        # 原bug: alpha gate 可能把 score 衰减到极低, 但 floor 修复在 alpha gate 之前
        # 修复: 在 alpha gate 后, 如果 score < 0.05, 重新设置 floor=0.05
        # 这样确保进化算法始终有非零起点, 不会空转
        if signal_score < 0.05 and signal_score > 0:
            signal_score = 0.05  # post-alpha floor

        # v596 C2: RegimeStrategyMatrix 集成 — 市场状态驱动的策略权重调整
        # 来源: 计划要求激活 regime_ensemble + RegimeStrategyMatrix
        # 设计原则:
        #   - RANGING: 均值回归策略权重1.2, 趋势策略0.5
        #   - TRENDING_UP/DOWN: 趋势策略1.3, 均值回归0.4
        #   - VOLATILE: 全面减仓0.6
        #   - CRISIS: 清仓0.0
        #   - ACCUMULATION: 突破策略1.1
        # 权重应用到 signal_score, 影响后续所有增强逻辑
        try:
            from ._v596_regime_namespace import normalize_regime, get_regime_weight
            raw_regime = regime_data.get("regime", "unknown") if regime_data else "unknown"
            regime_enum = normalize_regime(raw_regime)
            # 使用 gene.worldview_primary 作为策略类型, 默认 "all"
            strategy_type = getattr(self.gene, 'worldview_primary', 'all')
            regime_weight = get_regime_weight(regime_enum, strategy_type)
            signal_score *= regime_weight
            # CRISIS 或权重=0 时直接拒绝信号
            if regime_weight == 0.0:
                signal_score = 0.0
        except Exception as regime_err:
            # regime 权重应用失败不应阻断决策, 降级为不调整
            logger.debug("RegimeStrategyMatrix 应用失败, 降级为权重1.0: %s", str(regime_err)[:100])

        # v699 逐笔归因→5维进化建议 overlay 消费 (用户铁律"复盘作用于整个AI量化技能包")
        # 来源: EvolutionSuggestionApplier 写入 _active_evolution_overlay.json
        #       → EvolutionLoop._init_decision_engines 调用 set_evolution_overlay 注入
        # 闭环: 归因→建议→应用overlay→decide()消费→回测验证→回滚/保留
        # 安全: 防御式集成, patch匹配失败/字段缺失时静默跳过, 不破坏主决策流程
        if self._logic_patches or self._strategy_patches:
            try:
                _raw_regime = regime_data.get("regime", "unknown") if regime_data else "unknown"
                _strategy_type = getattr(self.gene, 'worldview_primary', 'all')

                # 1. strategy_patches: condition(regime) 匹配 → disable strategy_type
                for _sp in self._strategy_patches:
                    if _sp.get("status", "active") != "active":
                        continue
                    _cond = _sp.get("condition", {})
                    if _cond.get("regime") and _cond["regime"] == _raw_regime:
                        if _sp.get("action") == "disable" and _sp.get("strategy_type") == _strategy_type:
                            logger.info(
                                "[EvolutionOverlay] strategy disable: %s@%s → reject_entry",
                                _strategy_type, _raw_regime,
                            )
                            return self._early_return(start)

                # 2. logic_patches: trigger(regime) + filter(feature) → reject_entry/reduce_position
                for _lp in self._logic_patches:
                    if _lp.get("status", "active") != "active":
                        continue
                    _trigger = _lp.get("trigger", {})
                    # trigger.regime 匹配 (若指定)
                    if _trigger.get("regime") and _trigger["regime"] != _raw_regime:
                        continue
                    # trigger.alignment 暂不支持 (decide() 无 alignment 字段, 防御式跳过)
                    if _trigger.get("alignment"):
                        continue
                    # filter.feature 匹配 (若指定, 从 data_packet 递归查找)
                    _filter = _lp.get("filter", {})
                    _feat_name = _filter.get("feature")
                    if _feat_name:
                        _feat_val = self._extract_feature_from_data_packet(data_packet, _feat_name)
                        if _feat_val is None:
                            continue  # 字段不存在, 跳过该 patch (防御式)
                        _op = _filter.get("op", "")
                        if not self._eval_evolution_filter(_feat_val, _op, _filter):
                            continue
                    # trigger + filter 都匹配 → 执行 action
                    _action = _lp.get("action", "")
                    if _action == "reject_entry":
                        logger.info(
                            "[EvolutionOverlay] logic reject_entry: trigger=%s filter=%s",
                            _trigger, _filter,
                        )
                        return self._early_return(start)
                    elif _action == "reduce_position":
                        _factor = float(_lp.get("factor", 0.5))
                        signal_score *= _factor
                        logger.info(
                            "[EvolutionOverlay] logic reduce_position: factor=%.2f → signal_score=%.3f",
                            _factor, signal_score,
                        )
            except Exception as _overlay_err:
                logger.debug("[EvolutionOverlay] overlay 消费异常(降级跳过): %s", str(_overlay_err)[:100])

        # Phase 1 集成：35个具体策略 + 12个抽象世界观激活
        # 来源：StrategyRegistry 动态加载 gene.worldview_primary 对应的策略类
        # 设计原则：作为辅助确认信号（与alpha_mining同等地位），
        #   - 同向时温和增强（0.15，与alpha_mining一致）
        #   - 反向时不改变方向（避免策略信号主导决策）
        #   - 策略加载失败/无信号时静默跳过（不影响主流程）
        if self._strategy_registry is not None and not self._strategy_loaded:
            try:
                # Phase 6A: 构造完整 deps — 修复 funding_rate_arb/basis_arbitrage
                # 之前仅传 {"symbol": symbol}，需要 spot_engine/perp_engine 的策略类
                # 静默 fallback 到 GeneDrivenStrategy（违反"禁止空壳交付"铁律）
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
            except Exception as strat_load_err:
                # Phase 7J-5 反温室修复: 策略加载失败必须记录 (之前静默吞没)
                # 铁律: "异常情况处理预案完善" + "一定不要出现模拟牛逼，实盘亏钱！"
                # 实盘后果: 策略加载失败后静默fallback到默认策略, 进化系统以为真实策略已加载
                logger.warning(
                    "策略加载失败, 降级到默认策略 (worldview=%s): %s",
                    self.gene.worldview_primary, str(strat_load_err)[:150],
                )
                self._strategy_loaded = True  # 失败后不再尝试

        if self._strategy_instance is not None and self._strategy_market_ctx_type is not None:
            try:
                # 构造 MarketContext
                price_history = data_packet.get("price_history", [])
                ctx = self._strategy_market_ctx_type(
                    symbol=symbol,
                    timestamp=getattr(market, "timestamp", 0.0),
                    open=getattr(market, "open", close),
                    high=getattr(market, "high", close),
                    low=getattr(market, "low", close),
                    close=close,
                    volume=getattr(market, "volume", 0.0),
                    atr=atr_val,
                    period_return=getattr(market, "period_return", 0.0),
                    price_change=getattr(market, "price_change", 0.0),
                    price_history=list(price_history) if price_history else [],
                    raw_packet=data_packet,
                )
                strategy_signal = self._strategy_instance.update(ctx)
                self._strategy_signal = strategy_signal

                # 策略信号增强（Phase 6B: regime-aware 动态权重）
                # 之前固定 0.15 增强系数，不区分 regime；
                # 升级为根据市场状态动态调整（参考 2026 Regime-Aware LightGBM 最佳实践）
                # 设计原则：
                #   - 趋势市场（bull/bear）：趋势策略更有效，增强系数↑
                #   - 震荡市场（sideways）：趋势策略效果差，增强系数↓
                #   - 风险等级越高，增强系数越低（避免假突破）
                #   - 状态转换期不确定性高，再乘 0.5 衰减
                # 来源：
                #   - MDPI 2026 Regime-Aware LightGBM (滚动HMM检测regime + 条件预测)
                #   - QUANTA 2026 Regime-Filtered Hybrid
                #   - 经典：趋势策略在趋势市场有效，在震荡市场失效
                if strategy_signal.is_actionable:
                    strat_dir = "long" if strategy_signal.direction > 0 else "short"
                    strat_conf = strategy_signal.confidence
                    if strat_conf >= 0.3:
                        if strat_dir == signal_direction:
                            # 计算 regime-aware 增强系数
                            enhancement_coef = self._compute_regime_aware_coef(
                                regime_data, default=0.15,
                            )
                            signal_score *= (1.0 + strat_conf * enhancement_coef)
                        # 反向时不改变方向，仅记录（避免策略主导）
            except Exception as strat_exec_err:
                # Phase 7J-5 反温室修复: 策略执行失败必须记录 (之前 except: pass 静默吞没)
                # 铁律: "异常情况处理预案完善" — 实盘策略异常时无法被发现, Agent继续基于错误信号下单
                logger.warning(
                    "策略执行异常, 跳过策略信号增强: %s",
                    str(strat_exec_err)[:150],
                )

        # Phase E Batch 1: ModuleIntegrator 信号融合
        # 来源：module_integrator.py — 4个深度学习模块贝叶斯加权融合
        # 设计原则（与 StrategyRegistry 一致）：
        #   - 同向时温和增强（0.10，略低于策略信号的0.15）
        #   - 反向时不改变方向（避免深度学习信号主导决策）
        #   - 模块降级时用增强版代理信号（非空壳）
        #   - 融合失败时静默跳过（不影响主流程）
        if not self._module_integrator_initialized:
            try:
                from .module_integrator import ModuleIntegrator
                self._module_integrator = ModuleIntegrator(symbol=symbol or "BTC-USDT")
            except Exception as mi_err:
                logger.warning("ModuleIntegrator初始化失败: %s", str(mi_err)[:150])
                self._module_integrator = None
            self._module_integrator_initialized = True

        if self._module_integrator is not None:
            try:
                import numpy as np
                price_history = data_packet.get("price_history", [])
                if price_history and len(price_history) > 20:
                    prices = np.array(price_history, dtype=float)
                    returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0.0])
                    volumes = np.ones(len(returns))  # 如果没有成交量数据，用1.0占位
                    vol_data = data_packet.get("volume_history", [])
                    if vol_data and len(vol_data) >= len(returns):
                        volumes = np.array(vol_data[-len(returns):], dtype=float)
                    t = len(returns) - 1
                    mi_signals = self._module_integrator.get_all_signals(returns, volumes, t)
                    fused_pos, fused_conf = self._module_integrator.bayesian_fuse(mi_signals)

                    if fused_pos != 0 and fused_conf >= 0.3:
                        mi_dir = "long" if fused_pos > 0 else "short"
                        if mi_dir == signal_direction:
                            # 同向温和增强（0.10系数，低于策略信号的0.15）
                            signal_score *= (1.0 + fused_conf * 0.10)
                            logger.debug(
                                "ModuleIntegrator增强: dir=%s conf=%.2f score=%.3f",
                                mi_dir, fused_conf, signal_score,
                            )
            except Exception as mi_exec_err:
                logger.warning("ModuleIntegrator执行异常: %s", str(mi_exec_err)[:150])

        # Phase E Batch 2: Gate 决策门控信号
        # 来源：11个 Gate 模块，统一 apply_to_decision(decision, ...) 接口
        # 设计原则：
        #   - 构造临时 decision dict（action/quantity/price/symbol）
        #   - 依次通过每个 Gate 的 apply_to_decision()
        #   - Gate 通过 multiplier 修改 quantity（链式调用）
        #   - quantity <= 0 → 阻断交易（返回 None）
        #   - 最终 quantity 变化映射为 signal_score 调整
        # 铁律14合规：零遗漏 — 11个已实现 Gate 全部接入主干
        if self._gates:
            _gate_symbol = symbol or "BTC-USDT"
            _current_price = (
                data_packet.get("price", 0.0)
                or data_packet.get("current_price", 0.0)
                or data_packet.get("last_price", 0.0)
            )
            _gate_decision = {
                "action": signal_direction if signal_direction else "flat",
                "quantity": 1.0,
                "price": _current_price,
                "symbol": _gate_symbol,
                "confidence": signal_score,
            }
            import time as _time_mod
            _current_ts = _time_mod.time()
            _vol = data_packet.get("volume", 0.0) or data_packet.get("current_volume", 0.0)
            _bids = data_packet.get("bids")
            _asks = data_packet.get("asks")
            _ob = data_packet.get("order_book")

            for _gk, _gobj in self._gates.items():
                try:
                    if _gk == "fear_greed":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol, index_value=None)
                    elif _gk == "institution_retail":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol)
                    elif _gk == "gex":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, _current_price, agent_id="", symbol=_gate_symbol)
                    elif _gk == "sopr":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol)
                    elif _gk == "macro_event":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, _current_ts, agent_id="", symbol=_gate_symbol)
                    elif _gk == "long_short_ratio":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol)
                    elif _gk == "liquidation":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, _current_price, agent_id="", symbol=_gate_symbol)
                    elif _gk == "stablecoin_premium":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol, usdt_price=None)
                    elif _gk == "smart_money":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol, price=_current_price, volume=_vol)
                    elif _gk == "order_book":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol, order_book=_ob, bids=_bids, asks=_asks)
                    elif _gk == "urpd":
                        _gate_decision = _gobj.apply_to_decision(
                            _gate_decision, symbol=_gate_symbol, market_price=_current_price)

                    _new_qty = _gate_decision.get("quantity", 1.0)
                    if _new_qty <= 0:
                        self._gate_trigger_stats[_gk] = self._gate_trigger_stats.get(_gk, 0) + 1
                        logger.debug("Gate[%s] BLOCK (qty<=0): dir=%s", _gk, signal_direction)
                        return None
                except Exception as _gee:
                    logger.warning("Gate[%s]执行异常: %s", _gk, str(_gee)[:150])

            _final_qty = _gate_decision.get("quantity", 1.0)
            if _final_qty != 1.0 and _final_qty > 0:
                _gate_factor = _final_qty
                signal_score *= _gate_factor
                if _gate_factor < 0.5:
                    self._gate_trigger_stats["__reduce__"] = self._gate_trigger_stats.get("__reduce__", 0) + 1
                    logger.debug("Gate聚合 REDUCE: factor=%.3f score=%.3f", _gate_factor, signal_score)
                elif _gate_factor > 1.0:
                    self._gate_trigger_stats["__boost__"] = self._gate_trigger_stats.get("__boost__", 0) + 1
                    logger.debug("Gate聚合 BOOST: factor=%.3f score=%.3f", _gate_factor, signal_score)

        # 多智能体协作增强：群体智慧确认 + 拥挤交易反转
        # v2.3 反群体思维修复：共识过高时减弱信号（拥挤交易反转）
        # 原问题：共识高→增强信号→下轮共识更高→死亡螺旋，进化出"跟风基因"
        # 修复方案：
        #   - 中等共识（0.3-0.7）：群体方向一致时温和增强（群体智慧）
        #   - 极高共识（>0.7）：拥挤交易，反而减弱信号（反转风险）
        # 来源：
        #   - 群体智慧：TradingAgents 2026 (+9.3K stars)
        #   - 拥挤交易反转：Stein 2009 + Conrad et al. 2014 实证
        #   - 真实交易中"所有人都看多"=没有更多买盘=见顶信号
        if multi_agent_data:
            ma_direction_label = multi_agent_data.get("direction_label", "neutral")
            ma_confidence = multi_agent_data.get("confidence", 0.0)
            ma_consensus_level = multi_agent_data.get("consensus_level", "low")
            ma_signal_strength = multi_agent_data.get("signal_strength", 0.0)
            crowded_warning = multi_agent_data.get("crowded_warning", 0.0)

            # 拥挤交易反转：极高共识时减弱信号
            # 这是反群体思维的核心——防止"跟风基因"被进化选择
            if crowded_warning > 0.0:
                if (ma_direction_label == "bullish" and signal_direction == "long") or \
                   (ma_direction_label == "bearish" and signal_direction == "short"):
                    # 群体方向一致但过度拥挤，减弱信号（反转风险）
                    signal_score *= (1.0 - crowded_warning * 0.3)
            # 中等共识：群体方向一致时温和增强（群体智慧）
            elif ma_consensus_level == "high" and ma_confidence > 0.5 and ma_signal_strength < 0.7:
                if (ma_direction_label == "bullish" and signal_direction == "long") or \
                   (ma_direction_label == "bearish" and signal_direction == "short"):
                    # 群体方向一致且未拥挤，温和增强信号
                    signal_score *= (1.0 + ma_signal_strength * 0.1)

        # 多模型集成信号增强：LightGBM + Prophet + Qwen 融合信号
        # 来源：theneuralbase.com 2026 + QuantMind 2026 + QUANTA 2026
        # A/B回测验证通过的设计（ab_v17.log）：
        #   - 收益率: -7.32% → 0.32% (↑7.64%)
        #   - 胜率: 50% → 100% (↑50%)
        #   - 最大回撤: 12.58% → 0.01% (↓12.57%)
        #   - Sharpe: -0.6194 → 3.2772 (↑389%)
        # 核心设计：multi_model作为严格质量过滤器，不产生新交易
        #   1. 趋势一致性过滤：与趋势相反的交易直接拒绝
        #   2. 趋势一致性增强：与趋势一致时增强信号
        #   3. 动量确认门禁：动量与信号方向相反时拒绝
        #   4. 趋势强度门禁：趋势太弱时过滤弱信号
        #   5. 模型一致性低时减弱
        if multi_model_data:
            ensemble = multi_model_data.get("ensemble", {})
            if ensemble:
                mm_direction = ensemble.get("direction", "neutral")
                mm_confidence = ensemble.get("confidence", 0.0)
                mm_agreement = ensemble.get("model_agreement", 0.0)
                mm_fused_prob = ensemble.get("fused_probability", 0.5)

                # 计算趋势偏置和动量
                mm_trend_bias = (mm_fused_prob - 0.5) * 2  # -1 to 1
                mm_trend_strength = abs(mm_trend_bias)
                mm_momentum = mm_fused_prob - self._prev_mm_fused_prob
                self._prev_mm_fused_prob = mm_fused_prob

                # 策略1: 趋势一致性过滤 (v2.44 RC-001 根治: 废除压制, 消除99.8%做多根因)
                # 原代码: signal_score *= 0.6 (压制反向信号, 导致99.8%做多)
                # v3.11修复(0.3→0.6)只是缓解, 未根治
                # v2.44根治: 不压制反向信号, 完全依赖动量确认门禁(策略3)过滤
                # 原则: 趋势可能反转, 做空应该被允许; 动量确认是正确的过滤机制
                # 来源: 用户核心诉求"消除99.8%做多" + v2.44架构根因分析
                trend_conflict = False
                if mm_confidence > 0.6:  # 模型不确定时不标记冲突
                    if mm_trend_bias > 0.15 and signal_direction == "short":
                        # 多头趋势中做空 → 标记冲突但不压制 (依赖动量确认门禁过滤)
                        trend_conflict = True
                    elif mm_trend_bias < -0.15 and signal_direction == "long":
                        # 空头趋势中做多 → 标记冲突但不压制 (依赖动量确认门禁过滤)
                        trend_conflict = True

                # 策略2: 趋势一致性增强 - 与趋势一致时增强信号
                if not trend_conflict:
                    if mm_trend_bias > 0.15 and signal_direction == "long":
                        # 多头趋势中做多 → 增强
                        enhancement = mm_trend_bias * 0.6
                        signal_score *= (1.0 + enhancement)
                    elif mm_trend_bias < -0.15 and signal_direction == "short":
                        # 空头趋势中做空 → 增强
                        enhancement = abs(mm_trend_bias) * 0.6
                        signal_score *= (1.0 + enhancement)

                # 策略3: 动量确认门禁 - 动量与信号方向相反时拒绝交易
                if not trend_conflict and abs(mm_momentum) > 0.002:
                    if signal_direction == "long" and mm_momentum < -0.002:
                        # 做多但动量下降 → 拒绝
                        signal_score *= 0.4
                    elif signal_direction == "short" and mm_momentum > 0.002:
                        # 做空但动量上升 → 拒绝
                        signal_score *= 0.4
                    elif signal_direction == "long" and mm_momentum > 0.002:
                        # 做多且动量上升 → 额外增强
                        signal_score *= (1.0 + min(0.3, mm_momentum * 5.0))
                    elif signal_direction == "short" and mm_momentum < -0.002:
                        # 做空且动量下降 → 额外增强
                        signal_score *= (1.0 + min(0.3, abs(mm_momentum) * 5.0))

                # 策略4: 趋势强度门禁 - 趋势太弱时不交易
                if mm_trend_strength < 0.1 and abs(signal_score) < 0.25:
                    signal_score *= 0.5

                # 策略5: 模型一致性低时过滤
                if mm_agreement < 0.4 and mm_direction != "neutral":
                    signal_score *= 0.7

        # v2.3 海龟交易法则确认（实盘验证基准）
        # 来源：Richard Dennis 1980s实盘验证 + AdTurtle 62.71%年化
        # 设计原则：
        #   - 海龟同向入场 → 增强信号（实盘验证确认）
        #   - 海龟exit信号 → 减弱信号（逆向思考，趋势可能结束）
        #   - 海龟胜率仅30-35%，但盈亏比>2:1，作为趋势确认而非方向预测
        if turtle_data:
            turtle_action = turtle_data.get("action", "hold")
            turtle_confidence = turtle_data.get("confidence", 0.0)

            if turtle_action == signal_direction:
                # 海龟同向入场，增强信号（实盘验证确认）
                signal_score *= (1.0 + turtle_confidence * 0.15)
            elif turtle_action in ("exit_long", "exit_short"):
                # 海龟发出出场信号，减弱入场信号（逆向思考）
                # 趋势可能结束，不宜入场
                signal_score *= (1.0 - turtle_confidence * 0.2)
            elif turtle_action in ("long", "short") and turtle_action != signal_direction:
                # 海龟反向入场，减弱信号（方向矛盾）
                signal_score *= (1.0 - turtle_confidence * 0.1)

        # ============== v701 P0-1: composite_score 信号增强下线 (应用 ERR-v88fp) ==============
        # 用户铁律: "复盘不是摆设"、"错误库使用起来"、"很多指标没有融会贯通"
        # 教训 ERR-v88fp: composite_score 与 PnL 负相关 (Cohen's d=-0.240)
        #   - 0/10 子特征通过 P<0.05 + |Cohen's d|>0.1 双重过滤 (ERR-110-v561)
        #   - composite_score 作为 4 维融合(MTF/VP/PA/CVD)唯一输出口, 但被证伪
        #   - 这是"指标未融会贯通"的最致命根因 — 5大维度被压缩到 ±7.5% 弱增强范围
        # v701 修复策略:
        #   1. 完全下线 composite_score 的 0.7-1.3 信号增强 (不乘进 signal_score)
        #   2. 保留 _ff_fv 提取用于诊断 + 后续 v701 P1-3 (FVG 接线) + P3-7 (CVD 直连)
        #   3. 信号增强职责完全交给 hold_bars_adapter (ERR-v88fp 正面应用 d=+0.461)
        # v701 后续: FVG/CVD 信号绕过 composite_score 直达决策, 真正"融会贯通"
        # 历史: Phase 3.1 集成 (短板#2) → v701 P0-1 下线 (应用教训, 打破瓶颈)
        if self._feature_fusion is not None and market is not None:
            try:
                _ff_history = getattr(market, 'history', None) or getattr(market, 'bars', None)
                if _ff_history and len(_ff_history) >= 50:
                    _ff_klines = list(_ff_history[-200:])
                    _ff_fv = self._feature_fusion.extract(
                        klines_1h=_ff_klines,
                        symbol=symbol or "unknown",
                    )
                    _ff_composite = float(_ff_fv.composite_score)  # -1 ~ +1
                    # v701 P0-1: 不再乘进 signal_score (ERR-v88fp 应用)
                    # 保留诊断记录用于监控 composite_score 是否在未来数据上恢复预测力
                    if abs(_ff_composite) > 0.3 and self._diag_counter % 500 == 1:
                        logger.info(
                            "v701 FUSION_DIAG: agent=%s composite=%.3f (诊断 only, 不影响 signal_score, ERR-v88fp)",
                            getattr(self.gene, 'agent_id', '?'),
                            _ff_composite,
                        )
                    # v701 P1-3 钩子: 把 _ff_fv 暴露给后续 FVG/CVD 直连逻辑
                    # (P1-3 FVG 接线 + P3-7 CVD 直连 将在 v701 后续阶段使用)
                    self._last_ff_fv = _ff_fv
            except Exception as _ff_err:
                # 降级跳过: 特征融合失败不影响主决策流程 (ERR-110: 特征无预测力, 失败可接受)
                logger.debug("特征融合降级跳过: %s", _ff_err)
        # ============== v701 P0-1 END (composite_score 增强下线) ==============

        # ============== v701 P3-7: CVD 背离 + FVG 信号直连决策 (绕过 composite_score) ==============
        # 用户铁律: "很多指标技术和知识, 你全部没有融会贯通, 信手拈来, 很局限"
        #                "不要通过已知数据反推策略" (避免过拟合)
        # 教训应用:
        #   - ERR-v88fp: composite_score 被证伪 (Cohen's d=-0.240), 不能作为信号唯一输出口
        #   - ERR-v98-cvd (教训库): CVD 是市场微观结构最强信号 (CryptoQuant 73% BTC反转前出现)
        #   - v701 P1-3: FVG 已接线到 fv.fvg_signal, 现在让决策引擎消费它
        # v701 设计 (与 hold_bars_adapter 一致的独立信号增强模式, 绕过被证伪的 composite):
        #   1. CVD 背离 = 风险减仓信号 (反转预警, 减弱同向入场信号)
        #      - BEARISH 背离 + 做多 → signal_score *= 0.80-0.94 (预警反转, 减仓)
        #      - BULLISH 背离 + 做空 → signal_score *= 0.80-0.94 (预警反转, 减仓)
        #   2. FVG 信号 = 入场信号增强 (机构成本区+流动性磁铁)
        #      - FVG 同向 → signal_score *= 1.03-1.10 (机构成本支撑, 增强入场)
        #      - FVG 反向 → signal_score *= 0.70-0.91 (机构反向, 减弱入场)
        #   3. 强度调制: cvd_confidence / fvg_confidence 自适应调整影响幅度
        #   4. 软增强/减弱 (0.70-1.10), 不硬过滤 (避免过拟合, 让进化算法学习最优组合)
        # 非过拟合依据:
        #   - CVD = 实际交易数据派生 (主动买卖盘累计差), 非参数拟合
        #   - FVG = K 线几何信号 (3-K线模式 + 位移确认), 非参数拟合
        #   - 73% CVD 背离反转率 / 75% FVG 回访填充率 = 业界实证 (CryptoQuant 2026, ICT 实证)
        # 历史: P0-1 暴露 _last_ff_fv 钩子 → P3-7 直连决策 (打破 composite_score 瓶颈)
        if getattr(self, '_last_ff_fv', None) is not None:
            try:
                _ff_fv = self._last_ff_fv
                _pre_p37 = signal_score

                # === CVD 背离风险减仓 (反转预警) ===
                # cvd_divergence: 0=无, 1=BULLISH(底背离, 反转向上), -1=BEARISH(顶背离, 反转向下)
                _cvd_div = int(getattr(_ff_fv, 'cvd_divergence', 0))
                _cvd_conf = float(getattr(_ff_fv, 'cvd_confidence', 0.0))
                if _cvd_div != 0 and _cvd_conf > 0.3:
                    # 强度调制: confidence 0.3-1.0 → 减弱因子 0.94-0.80
                    _cvd_strength = max(0.3, min(1.0, _cvd_conf))
                    _cvd_factor = 1.0 - 0.20 * _cvd_strength  # 0.80-0.94
                    if _cvd_div == -1 and signal_direction == "long":
                        # BEARISH 背离 + 做多 → 减仓信号 (73% 反转概率, CryptoQuant 2026)
                        signal_score *= _cvd_factor
                    elif _cvd_div == 1 and signal_direction == "short":
                        # BULLISH 背离 + 做空 → 减仓信号 (73% 反转概率, CryptoQuant 2026)
                        signal_score *= _cvd_factor

                # === FVG 信号入场增强/减弱 (机构成本区+流动性磁铁) ===
                _fvg_sig = float(getattr(_ff_fv, 'fvg_signal', 0.0))
                _fvg_conf = float(getattr(_ff_fv, 'fvg_confidence', 0.0))
                if abs(_fvg_sig) >= 0.3 and _fvg_conf > 0.3:
                    # 强度调制: confidence 0.3-1.0
                    _fvg_strength = max(0.3, min(1.0, _fvg_conf))
                    if (_fvg_sig > 0 and signal_direction == "long") or \
                       (_fvg_sig < 0 and signal_direction == "short"):
                        # FVG 同向 → 增强 (1.03-1.10, ICT 75% FVG 回访填充率)
                        signal_score *= 1.0 + 0.10 * _fvg_strength
                    elif (_fvg_sig > 0 and signal_direction == "short") or \
                         (_fvg_sig < 0 and signal_direction == "long"):
                        # FVG 反向 (方向矛盾) → 减弱 (0.70-0.91, 机构反向操作)
                        signal_score *= 1.0 - 0.30 * _fvg_strength

                # 诊断日志 (低频, 避免日志爆炸)
                _p37_delta = signal_score - _pre_p37
                if abs(_p37_delta) > 0.005 and getattr(self, '_diag_counter', 0) % 500 == 1:
                    logger.warning(
                        "v701 P3-7: agent=%s sym=%s dir=%s cvd_div=%d cvd_conf=%.2f fvg_sig=%.2f fvg_conf=%.2f score=%.4f→%.4f (Δ=%+.4f)",
                        getattr(self.gene, 'agent_id', '?'),
                        symbol, signal_direction, _cvd_div, _cvd_conf,
                        _fvg_sig, _fvg_conf, _pre_p37, signal_score, _p37_delta,
                    )
            except Exception as _p37_err:
                # 降级跳过: CVD/FVG 直连失败不影响主决策 (软增强/减弱, 失败等同中性 1.0)
                logger.debug("v701 P3-7 CVD/FVG 直连降级跳过: %s", _p37_err)
        # ============== v701 P3-7 END (CVD/FVG 直连决策) ==============

        # ============== v702 P3: Order Block + Liquidity Sweep 信号直连决策 (绕过 composite_score) ==============
        # 用户铁律: "很多指标技术和知识, 你全部没有融会贯通, 信手拈来, 很局限"
        #                "不要通过已知数据反推策略" (避免过拟合)
        # 教训应用:
        #   - ERR-v88fp: composite_score 被证伪 (Cohen's d=-0.240), 不能作为信号唯一输出口
        #   - v702 P1/P2: OB + LS 已接线到 fv.order_block_signal / fv.liquidity_sweep_signal
        #   - ICT SMC 2026: OB = 机构成本区 (75% OB 回调会被尊重); LS = 机构猎杀散户止损后反向建仓
        # v702 设计 (与 v701 P3-7 CVD/FVG 一致的独立信号增强模式):
        #   1. Order Block 信号 = 入场信号增强 (机构成本支撑)
        #      - OB 同向 (bullish OB + long / bearish OB + short) → 1.03-1.10 (机构成本支撑, 增强入场)
        #      - OB 反向 → 0.70-0.91 (机构反向操作, 减弱入场)
        #   2. Liquidity Sweep 信号 = 入场信号增强 (机构猎杀后同向建仓)
        #      - LS 同向 (sell_side_sweep→+1 + long / buy_side_sweep→-1 + short) → 1.03-1.10 (机构同向)
        #      - LS 反向 → 0.70-0.91 (机构反向)
        #   3. 强度调制: order_block_confidence / liquidity_sweep_confidence 自适应调整影响幅度
        #   4. 软增强/减弱 (0.70-1.10), 不硬过滤 (避免过拟合, 让进化算法学习最优组合)
        #   5. 多信号自然乘法叠加: OB+LS 同向 → 1.10*1.10=1.21 (confluence), OB+LS 反向 → 0.70*0.70=0.49
        # 非过拟合依据:
        #   - OB = K 线几何+波段结构信号 (机构建仓最后一根反向K线), 非参数拟合
        #   - LS = K 线几何+流动性池结构信号 (假突破+回归确认), 非参数拟合
        #   - 75% OB 回调尊重率 / 73% LS 反转率 = ICT 2026 实证 (与 CVD 73% 反转率一致)
        # 历史: P0-1 暴露 _last_ff_fv 钩子 → P3-7 (CVD/FVG) → P3 (OB/LS) 完成 SMC 四大支柱端到端贯通
        if not hasattr(self, '_v702_p3_diag_counter'):
            self._v702_p3_diag_counter = 0
        self._v702_p3_diag_counter += 1
        if getattr(self, '_last_ff_fv', None) is not None:
            try:
                _ff_fv = self._last_ff_fv
                _pre_p3 = signal_score

                # === Order Block 入场增强/减弱 (机构成本区) ===
                _ob_sig = float(getattr(_ff_fv, 'order_block_signal', 0.0))
                _ob_conf = float(getattr(_ff_fv, 'order_block_confidence', 0.0))
                if abs(_ob_sig) >= 0.3 and _ob_conf > 0.3:
                    # 强度调制: confidence 0.3-1.0
                    _ob_strength = max(0.3, min(1.0, _ob_conf))
                    if (_ob_sig > 0 and signal_direction == "long") or \
                       (_ob_sig < 0 and signal_direction == "short"):
                        # OB 同向 → 增强 (1.03-1.10, ICT 75% OB 回调尊重率)
                        signal_score *= 1.0 + 0.10 * _ob_strength
                    elif (_ob_sig > 0 and signal_direction == "short") or \
                         (_ob_sig < 0 and signal_direction == "long"):
                        # OB 反向 (方向矛盾) → 减弱 (0.70-0.91, 机构反向操作)
                        signal_score *= 1.0 - 0.30 * _ob_strength

                # === Liquidity Sweep 入场增强/减弱 (机构猎杀后同向建仓) ===
                _ls_sig = float(getattr(_ff_fv, 'liquidity_sweep_signal', 0.0))
                _ls_conf = float(getattr(_ff_fv, 'liquidity_sweep_confidence', 0.0))
                if abs(_ls_sig) >= 0.3 and _ls_conf > 0.3:
                    # 强度调制: confidence 0.3-1.0
                    _ls_strength = max(0.3, min(1.0, _ls_conf))
                    if (_ls_sig > 0 and signal_direction == "long") or \
                       (_ls_sig < 0 and signal_direction == "short"):
                        # LS 同向 → 增强 (1.03-1.10, 机构猎杀散户止损后同向建仓)
                        signal_score *= 1.0 + 0.10 * _ls_strength
                    elif (_ls_sig > 0 and signal_direction == "short") or \
                         (_ls_sig < 0 and signal_direction == "long"):
                        # LS 反向 → 减弱 (0.70-0.91, 机构反向操作)
                        signal_score *= 1.0 - 0.30 * _ls_strength

                # 诊断日志 (低频, 每 500 次输出一次, 避免 日志爆炸)
                _p3_delta = signal_score - _pre_p3
                if abs(_p3_delta) > 0.005 and self._v702_p3_diag_counter % 500 == 1:
                    logger.warning(
                        "v702 P3: agent=%s sym=%s dir=%s ob_sig=%.2f ob_conf=%.2f ls_sig=%.2f ls_conf=%.2f score=%.4f→%.4f (Δ=%+.4f)",
                        getattr(self.gene, 'agent_id', '?'),
                        symbol, signal_direction,
                        _ob_sig, _ob_conf, _ls_sig, _ls_conf,
                        _pre_p3, signal_score, _p3_delta,
                    )
            except Exception as _p3_err:
                # 降级跳过: OB/LS 直连失败不影响主决策 (软增强/减弱, 失败等同中性 1.0)
                logger.debug("v702 P3 OB/LS 直连降级跳过: %s", _p3_err)
        # ============== v702 P3 END (OB/LS 直连决策) ==============

        # ============== v703 P2: BOS/CHoCH 信号直连决策 (绕过 composite_score) ==============
        # 用户铁律: "很多指标技术和知识, 你全部没有融会贯通, 信手拈来, 很局限"
        #                "不要通过已知数据反推策略" (避免过拟合)
        # 教训应用:
        #   - ERR-v88fp: composite_score 被证伪 (Cohen's d=-0.240), 不能作为信号唯一输出口
        #   - v701 P0-2: BOS/CHoCH 在 scores 路径保留 0.4 权重 (减半保留, 等完整 SMC 体系)
        #   - v702 完成: OB+LS 已直连决策, SMC 四大支柱接线完整, 现在补齐 BOS/CHoCH 直连
        #   - ICT SMC 2026: BOS=结构突破(趋势延续), CHoCH=性质转换(趋势反转)
        # v703 设计 (与 v701 P3-7 CVD/FVG + v702 P3 OB/LS 一致的独立信号增强模式):
        #   1. BOS/CHoCH 信号 = 入场信号增强 (市场结构确认)
        #      - 看涨结构 (bos_bull/choch_bull) + long → 同向增强 1.03-1.10
        #      - 看跌结构 (bos_bear/choch_bear) + short → 同向增强 1.03-1.10
        #      - 反向 → 减弱 0.70-0.91 (结构不支持)
        #   2. 强度调制: CHoCH (性质转换, 趋势反转) > BOS (结构突破, 趋势延续)
        #      - choch_*: strength=1.0 (反转信号, ICT 2026 实证 65% 反转率)
        #      - bos_*: strength=0.7 (延续信号, ICT 2026 实证 55% 延续率)
        #   3. 软增强/减弱 (0.70-1.10), 不硬过滤 (避免过拟合, 让进化算法学习最优组合)
        #   4. 多信号自然乘法叠加: BOS/CHoCH + OB + LS + FVG + CVD 五大信号独立直达决策
        # 非过拟合依据:
        #   - BOS/CHoCH = K 线几何+波段结构信号 (swing 高低点突破), 非参数拟合
        #   - 65% CHoCH 反转率 / 55% BOS 延续率 = ICT 2026 实证 (与 CVD 73% / OB 75% / LS 73% 同源)
        # 历史: P0-1 暴露 _last_ff_fv 钩子 → P3-7 (CVD/FVG) → P3 (OB/LS) → P2 (BOS/CHoCH) SMC 四大支柱完整直连
        if not hasattr(self, '_v703_p2_diag_counter'):
            self._v703_p2_diag_counter = 0
        self._v703_p2_diag_counter += 1
        if getattr(self, '_last_ff_fv', None) is not None:
            try:
                _ff_fv = self._last_ff_fv
                _pre_p2 = signal_score

                # === BOS/CHoCH 入场增强/减弱 (市场结构确认) ===
                # pa_bos_choch 来自 chanlun_priceaction.MarketStructureAnalyzer
                # 取值: "bos_bull"/"bos_bear"/"choch_bull"/"choch_bear"/"none"/""
                _bc_str = str(getattr(_ff_fv, 'pa_bos_choch', '') or '').lower().strip()
                if _bc_str and _bc_str != "none":
                    # 信号方向解析
                    _bc_is_bull = "bull" in _bc_str
                    _bc_is_bear = "bear" in _bc_str
                    # 强度调制: CHoCH (反转) > BOS (延续)
                    _bc_strength = 1.0 if "choch" in _bc_str else (0.7 if "bos" in _bc_str else 0.5)
                    if _bc_is_bull and signal_direction == "long":
                        # 看涨结构 + 做多 → 同向增强 (1.03-1.10, ICT 65% CHoCH 反转率 / 55% BOS 延续率)
                        signal_score *= 1.0 + 0.10 * _bc_strength
                    elif _bc_is_bear and signal_direction == "short":
                        # 看跌结构 + 做空 → 同向增强 (1.03-1.10)
                        signal_score *= 1.0 + 0.10 * _bc_strength
                    elif (_bc_is_bull and signal_direction == "short") or \
                         (_bc_is_bear and signal_direction == "long"):
                        # 结构反向 → 减弱 (0.70-0.91, 结构不支持)
                        signal_score *= 1.0 - 0.30 * _bc_strength

                # 诊断日志 (低频, 每 500 次输出一次, 避免日志爆炸)
                _p2_delta = signal_score - _pre_p2
                if abs(_p2_delta) > 0.005 and self._v703_p2_diag_counter % 500 == 1:
                    logger.warning(
                        "v703 P2: agent=%s sym=%s dir=%s bos_choch=%s score=%.4f→%.4f (Δ=%+.4f)",
                        getattr(self.gene, 'agent_id', '?'),
                        symbol, signal_direction, _bc_str,
                        _pre_p2, signal_score, _p2_delta,
                    )
            except Exception as _p2_err:
                # 降级跳过: BOS/CHoCH 直连失败不影响主决策 (软增强/减弱, 失败等同中性 1.0)
                logger.debug("v703 P2 BOS/CHoCH 直连降级跳过: %s", _p2_err)
        # ============== v703 P2 END (BOS/CHoCH 直连决策) ==============

        # ============== v99 Phase 8 Task #31.2: hold_bars 信号增强集成 (Phase L 落地) ==============
        # ERR-v88fp: hold_bars 是 PnL 最强预测因子 (Cohen's d=+0.461~+0.6292, 三样本稳定)
        # ERR-109: 仅增强非过滤 (hold_bars 是 consequence 不是 cause, 不直接用作决策)
        # ERR-110: 仅信号增强非加权决策 (0.9-1.1 微调, 不改变信号方向)
        # 设计: 基于运行时累积的 (hold_bars, pnl) 分布, 给当前信号一个增强因子
        #   - 样本不足 (<min_samples) → 返回 1.0 中性, 不影响信号
        #   - 样本充足 → current_hold_bars 在盈利侧 → 增强 (>1.0)
        #               current_hold_bars 在亏损侧 → 减弱 (<1.0)
        # 开仓决策时 current_hold_bars=0 (无持仓), adapter 基于 hold_bars=0 在分布中的位置决定
        # 异常时降级中性 1.0 (不影响主决策流程)
        if self._hold_bars_adapter is not None:
            try:
                _hb_signal = self._hold_bars_adapter.compute_signal_factor(
                    symbol=symbol or "unknown",
                    current_hold_bars=0,  # 开仓决策时无持仓, 传 0 让 adapter 基于分布决定
                )
                # 信号增强: signal_score *= _hb_signal (范围 0.9-1.1)
                if isinstance(_hb_signal, (int, float)) and 0.9 <= _hb_signal <= 1.1:
                    _pre_hb = signal_score
                    signal_score = signal_score * _hb_signal
                    if abs(_hb_signal - 1.0) > 0.01 and self._diag_counter % 500 == 1:
                        logger.warning(
                            "v99 HOLD_BARS: agent=%s symbol=%s factor=%.4f score=%.4f→%.4f",
                            getattr(self.gene, 'agent_id', '?'),
                            symbol, _hb_signal, _pre_hb, signal_score,
                        )
            except Exception as _hb_err:
                # 降级中性: hold_bars 信号增强失败不影响主决策 (ERR-110: 信号无预测力时失败可接受)
                logger.debug("hold_bars 信号增强降级跳过: %s", _hb_err)
        # ============== v99 Phase 8 Task #31.2 END ==============

        # 检查是否达到置信度阈值
        # Phase 7J-5 反温室修复: 删除沙盘阈值放宽 (之前0.5倍=2倍放宽)
        # 铁律: "一定不要出现模拟牛逼，实盘亏钱！"
        # 沙盘放宽阈值→进化出依赖宽松阈值的策略→实盘恢复严格阈值后交易骤降90%+
        # 现在使用生产级阈值 (1.0倍, 不放宽), 沙盘进化必须在严格阈值下仍可盈利
        # 来源: 反温室审视报告 CRITICAL #2 + 用户铁律
        production_threshold = self.gene.worldview_confidence_threshold
        # v3.27 临时诊断: 记录信号得分详情(只每100次打印一次避免日志爆炸)
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = 0
        self._diag_counter += 1
        if self._diag_counter % 500 == 1:
            logger.warning(
                "v327 DIAG signal: agent=%s score=%.4f threshold=%.4f count=%d/%d direction=%s primary=%s",
                getattr(self.gene, 'agent_id', '?'),
                signal_score, production_threshold,
                signal_count, self.gene.entry_logic_min_signal_confluence,
                signal_direction, self.gene.entry_logic_primary_signals[:3],
            )
        # Phase 14.22: 极弱信号硬地板 — signal_score < 10% threshold 直接弃权
        # 根因: v503 P3-2 Fix 3c 把硬过滤改成软衰减(根治92%沉默), 但副作用是
        #   signal_score=0.0036 (0.7% of threshold=0.5) 这种噪声级信号仍然开仓
        #   → OpportunityScorer BOOST 放大仓位 → 噪声交易亏损
        # 修复: 保留软衰减给 10%-100% 范围的近阈值信号(保持92%沉默修复),
        #   但对 <10% 的噪声信号硬弃权 (return neutral)
        # 数学验证: threshold=0.5, hard_floor=0.05
        #   signal_score=0.0036 < 0.05 → REJECT (噪声)
        #   signal_score=0.3 (60% of threshold) → 软衰减 (保持修复)
        #   signal_score=0.5 → 通过 (无衰减)
        _HARD_FLOOR_RATIO_P1422 = 0.10
        if signal_score < production_threshold * _HARD_FLOOR_RATIO_P1422:
            return self._early_return(start)

        # v503 P3-2 Fix 3c: 信号得分阈值硬过滤→衰减 (根治92%沉默)
        # 用户明确要求: "不只是强制confluence, 而是理解各指标的底层逻辑"
        # 信号得分低于阈值时, 不直接弃权, 而是按比例衰减
        # 进化算法可以选择"低阈值高仓位"或"高阈值低仓位"两种策略
        # score_ratio: 0~1, 得分越接近阈值, 衰减越小
        # 衰减后: 0.4~1.0 线性, 保留最低 0.4 的基础置信度
        if signal_score < production_threshold:
            score_ratio = signal_score / max(production_threshold, 0.01)
            signal_score *= 0.4 + 0.6 * score_ratio  # 0.4-1.0 线性衰减

        # 信号确认检查
        # B1修复（v2.5深度修复）：min_signal_confluence是整数（2-3），表示"需要多少个信号确认"
        # 之前错误地用 min_signal_confluence * 0.3 作为signal_score阈值（=0.6-0.9），
        # 导致signal_score几乎永远无法超过，Agent无法交易，GT-Score=0
        # 正确语义：检查触发的信号数量是否达到min_signal_confluence
        #
        # v503 P3-2 Fix 3e: confluence 硬过滤→衰减 (根治92%沉默)
        # 用户明确要求: "不只是强制confluence, 而是理解各指标的底层逻辑"
        # 信号数量不足时, 不直接弃权, 而是降低信号置信度
        # 进化算法可以学习"低confluence但高signal_score"的策略
        if signal_count < self.gene.entry_logic_min_signal_confluence:
            count_ratio = signal_count / max(self.gene.entry_logic_min_signal_confluence, 1)
            signal_score *= 0.4 + 0.6 * count_ratio  # 0.4-1.0 线性衰减

        # v503 Fix 31: neutral 信号安全网 — 防止 L1049 if/else 把 neutral 当 short 处理
        # 注: Fix 31 v1/v2/v3 的 RSI 过滤已回滚 (P63/P64/P65 三版本均恶化表现)
        #   根因: indicators 为空(34%)时 RSI 默认 50.0 不触发过滤 + 过滤后剩余空头质量更差
        # 当前: signal_direction 不会被主动设为 neutral, 此检查仅作安全网
        if signal_direction == "neutral":
            return self._early_return(start)

        # Phase 14.16: 移除 v503 Fix 32b blanket short ban
        # 根因: Fix 32b 临时禁止所有空头, 但从未移除
        #   - 2019年BTC从3848跌到129(-96.65%), 全long=必亏 (胜率18.6%)
        #   - Phase 14.16 已修复 fallback 方向逻辑 (close<sma_50 → short)
        #   - 空头信号质量已改进, 满足 Fix 32b 的移除条件
        # 金融原理: 暴跌市场做空是唯一盈利方向, 禁止做空=自杀
        # 进化算法需要short交易来学习何时做空最有利

        # v503 Fix 33b: 高波动山寨币入场 per-symbol 过滤 — SOL 趋势确认 + DOGE 均值回归
        # 根因 (P67 诊断): SOL/DOGE 多头胜率 39-43% << BTC 86.2% / ETH 62.2% / BNB 58.9%
        #   - SOL: 137 trades, WR=43.1%, R:R=0.79 → break-even R:R=1.32 → 必亏
        #   - DOGE: 146 trades, WR=39.0%, R:R=0.81 → break-even R:R=1.56 → 必亏
        #
        # P68b 关键发现 (ema_12>ema_26 过滤):
        #   - SOL 改善: WR 43.1%→49.2% (+6.1pp), EV -0.3282%→-0.2889% ✅
        #   - DOGE 恶化: WR 39.0%→28.6% (-10.4pp), EV -0.4870%→-0.5914% ❌
        #   - 根因: SOL 有趋势跟随特性 (ema_12>ema_26 确认趋势有效)
        #           DOGE 是均值回归特性 (ema_12<=ema_26 = 抄底机会, 过滤掉反而保留亏损追涨)
        #   - 反推: DOGE ema_12<=ema_26 的 62 笔交易胜率 = (39%×146 - 28.6%×84)/62 = 53.1%
        #           vs ema_12>ema_26 的 84 笔交易胜率 = 28.6%
        #           → 抄底(53.1%) >> 追涨(28.6%), DOGE 应该买跌而非追涨
        #
        # 底层逻辑 (用户铁律: "理解底层逻辑, 而非表面指标"):
        #   1. 不同币种有不同的市场微观结构: 大盘币趋势持久, 山寨币分化
        #   2. SOL 介于趋势/回归之间, 趋势确认有帮助 (WR +6.1pp)
        #   3. DOGE 强均值回归, 应在短期下行时买入 (抄底), 而非追涨
        #   4. 一刀切过滤是表面指标思维, per-symbol 策略才是底层逻辑
        #
        # 修复原理:
        #   - SOL: ema_12 > ema_26 (趋势确认, 只在短期趋势向上时做多)
        #   - DOGE: ema_12 <= ema_26 (均值回归, 只在短期趋势向下时抄底)
        #   - BTC/ETH/BNB: 无额外过滤 (已盈利)
        #   - indicators 缺失时不过滤 (避免0交易)
        # 数学验证:
        #   SOL 预期: WR 49.2% (P68b实测), R:R 0.78, EV=-0.2889% (改善中, 需进一步优化)
        #   DOGE 预期: WR 53.1% (反推), R:R 0.81, EV=0.531×0.81-0.469×1.0=-0.039% (大幅改善)
        #   整体预期: EV mean 从 0.0002% 提升至 ~0.05%+
        # 来源: 趋势跟随 vs 均值回归双范式 (Connor's RSI-2 + Donchian Channel)
        # 铁律: "理解底层逻辑" — 不同币种不同策略, 而非一刀切
        _symbol = data_packet.get("symbol", "")
        if signal_direction == "long" and _symbol:
            _ema_12 = indicators.get("ema_12", 0.0) if indicators else 0.0
            _ema_26 = indicators.get("ema_26", 0.0) if indicators else 0.0
            if _ema_12 > 0 and _ema_26 > 0:
                if _symbol == "SOL-USDT" and _ema_12 <= _ema_26:
                    # SOL 趋势跟随: 短期趋势向下 → 不追多
                    return self._early_return(start)
                elif _symbol == "DOGE-USDT" and _ema_12 > _ema_26:
                    # DOGE 均值回归: 短期趋势向上 → 不追涨 (等回调抄底)
                    return self._early_return(start)
                # v503 Fix 36c: SOL ADX 趋势强度过滤 — 反转方向 (弱趋势入场)
                # 根因 (P73 反直觉发现): SOL 强趋势(ADX>25) WR=36.2% << 弱趋势(ADX<25) WR=57.8%
                #   - ADX 是滞后指标, ADX>25 时趋势已成熟接近衰竭 (买在顶部)
                #   - ADX<25 是早期趋势, 有空间运行 (买在起步)
                #   - SOL ema_12>ema_26 + ADX<25 = 早期上升趋势 = 最佳入场点
                # 底层逻辑 (用户铁律: "理解底层逻辑, 而非表面指标"):
                #   1. 趋势跟随不是"追强趋势", 而是"在趋势早期入场"
                #   2. ADX>25 = 趋势已成熟 = 接近反转 = 买在顶部 = 必亏
                #   3. ADX<25 = 趋势刚形成 = 有运行空间 = 买在起步 = 盈利
                #   4. 这是"买预期, 卖事实"的量化体现
                # 数学验证 (P73 反推):
                #   ADX<25 的 71 笔交易: WR=57.8%, R:R=0.74
                #   EV = 0.578×0.74 - 0.422×1.0 = 0.428 - 0.422 = +0.006 (转正!)
                #   vs ADX>25 的 47 笔: WR=36.2%, EV=-0.7147% (必亏)
                # 来源: ADX 滞后性 + 趋势生命周期理论 (Wyckoff)
                # 铁律: "理解底层逻辑" — 趋势跟随=早期入场, 非追涨杀跌
                if _symbol == "SOL-USDT" and _ema_12 > _ema_26:
                    _adx = indicators.get("adx_14", 0.0) if indicators else 0.0
                    if _adx > 0 and _adx >= 25.0:
                        # SOL 强趋势 (ADX≥25) → 趋势已成熟, 不追多 (等回调或新趋势)
                        return self._early_return(start)

        # 模块5：仓位计算（前沿增强：Dynamic Kelly优先）
        # Phase 7J-8 反温室修复 HIGH #6: close=0 不交易 (防0价格异常订单)
        # 之前: close=0 时 Kelly 异常被 except 捕获, 降级到基因原始仓位仍用 close=0 → 异常订单
        # 现在: close=0 直接返回 None (反温室: 0价格=数据异常=不交易)
        # 铁律: "一定不要出现模拟牛逼，实盘亏钱！" — 0价格建仓=实盘必亏
        if not market.close or market.close <= 0:
            logger.warning("market.close=%s <= 0, 跳过交易 (反温室: 0价格不交易)", market.close)
            return self._early_return(start)

        if frontier is not None and atr_val > 0:
            # 前沿增强：使用Dynamic Adaptive Kelly计算仓位
            # 来源：CARL AI Labs 2025 — 贝叶斯更新+分数Kelly+VaR限制
            # Phase 7J-4: Kelly/ATR 异常保护 (P0-2 缺口修复)
            # 之前: 无 try/except, frontier.kelly.compute_position_size 异常会崩溃
            #       整个 decide(), 导致单 Agent 故障冒泡到主循环, 整轮进化中断
            # 现在: 异常时降级到基因原始仓位, 记录 warning, 保证 decide() 健壮性
            # 来源: matrixtrak 2026 "classify failure before you act" + 调研报告 P0-2
            try:
                entry_price = market.close
                # 先计算ATR止损价（用于Kelly的风险评估）
                sl_tp = frontier.stop_loss.compute_stop_loss_take_profit(
                    entry_price, atr_val, signal_direction,
                )
                # v3.30 根因修复 CRITICAL (协调者根因分析 + LOSS-HUNTER 联合诊断):
                # 根因: frontier 使用 atr*1.5 作为止损距离, atr*3.75 作为止盈距离
                #   在高波动期(BTC崩盘时ATR可达14000+), 止损距离=21000 (27%!)
                #   → 止损永远不会触发(2.4%触发率) + 止盈永远不会触发(0%触发率)
                #   → 97.6%交易走到时间止损 → 在不利价格平仓 → 76%亏损
                # 实测证据 (audit.log v3.29):
                #   - 平均止损距离 7.78% (远超 gene.exit_risk_stop_loss_pct=2.77%)
                #   - 平均止盈距离 ~20% (远超 gene.exit_risk_take_profit_pct=1.66%)
                #   - 止损触发率 2.4%, 止盈触发率 0.0% (两者都失效)
                #   - 价格波动平均 0.97% (TP/SL距离远超实际波动)
                # 修复原则 (不过拟合, 是架构性修复):
                #   1. 止损距离 = min(atr*乘数, gene.exit_risk_stop_loss_pct*entry)
                #   2. 止盈距离 = 止损距离 * max(2.5, gene_RR) (保持R:R)
                #   3. 但止盈距离不超过 entry*5% (5%已可触发, 不需要更远)
                # 来源: 风险管理基本原则 "stop loss must be achievable"
                #       + QuantStart 2025 "ATR stops need volatility caps"
                gene_sl_pct = self.gene.exit_risk_stop_loss_pct
                gene_tp_pct = self.gene.exit_risk_take_profit_pct
                # v447 认知层应用: 根据 regime_adjustment 调整 SL/TP
                # 当 regime 失败模式被识别时, 使用失败模式知识库推荐的 SL/TP 倍数
                # 例如: ranging 市场 SL 收紧到 80%, TP 缩小到 60% (区间内空间有限)
                #       volatile 市场 SL 放宽到 150% (避免被假突破扫损)
                if regime_adjustment:
                    gene_sl_pct *= regime_adjustment.get("sl_multiplier", 1.0)
                    # v503 Fix 35: ERR-045 修复 — tp_multiplier 地板 0.8
                    # 根因 (ERR-045): regime_adjustment.tp_multiplier (ranging=0.6) 在 ATR cap 之前应用
                    #   → TP 被压缩至 gene_tp×0.6, 远低于 ATR cap (2.0×ATR)
                    #   → R:R 被压缩 (SOL: 预期R:R=0.80, 实测0.68)
                    #   → 盈亏平衡R:R无法达到 → 必亏
                    # 修复: tp_multiplier 地板 = 0.8 (允许适度压缩, 但不过度)
                    #   - ranging: 0.6→0.8 (仍压缩, 但保留 R:R 可行性)
                    #   - volatile: 1.2→1.2 (不变, 本就>0.8)
                    #   - trend: 1.0→1.0 (不变)
                    # 数学验证: SOL gene_tp=2.5%, tp_mult=0.8 → 2.0%, ATR cap=1.66% → final=1.66%
                    #   R:R = 1.66/2.074 = 0.80 (vs 原压缩后 0.68)
                    # 铁律: "理解底层逻辑" — regime压缩TP有其道理(区间内空间小),
                    #        但过度压缩(0.6)使R:R不可行 = 模拟必亏
                    _tp_mult = regime_adjustment.get("tp_multiplier", 1.0)
                    _effective_tp_mult = max(_tp_mult, 0.8)  # Fix 35: tp_multiplier 地板 0.8
                    gene_tp_pct *= _effective_tp_mult

                    # Phase 14.24: RR 保持 — 当 TP 被压缩但 SL 未同步压缩时, 同步压缩 SL
                    # 根因: regime tp_multiplier=0.8 但 sl_multiplier=1.0
                    #   → RR 从 0.8 恶化到 0.64 → 盈亏平衡胜率从 56% 升到 61% → 必亏
                    # 修复: sl_multiplier 同步降到 tp_multiplier 水平, 保持基因原始 RR
                    # 数学验证:
                    #   原始: sl=5%, tp=4%, RR=0.8
                    #   修复后: sl=5%*(0.8/1.0)=4%, tp=3.2%, RR=0.8 (保持)
                    _effective_sl_mult = regime_adjustment.get("sl_multiplier", 1.0)
                    if _effective_tp_mult < _effective_sl_mult:
                        _p1424_sl_ratio = _effective_tp_mult / _effective_sl_mult
                        gene_sl_pct *= _p1424_sl_ratio
                        try:
                            logger.info(
                                f"[Phase14.24] RR_preserve: sl_mult {_effective_sl_mult:.2f}->"
                                f"{_effective_tp_mult:.2f} (ratio={_p1424_sl_ratio:.3f}), "
                                f"sl_pct={gene_sl_pct:.4f} tp_pct={gene_tp_pct:.4f} RR={gene_tp_pct/max(gene_sl_pct,1e-9):.2f}"
                            )
                        except Exception:
                            pass
                # v503 Fix 8/22: 打破止盈死锁 — MIN_STOP_PCT 3.0%→1.5% + R:R 上限 3.0
                # 根因诊断 (P3-12 _v503_p40_trade_diag.py):
                #   - BTC 1h 波动率 0.766%, 持仓 3-8h, 期望波动 1.3-2.2%
                #   - MIN_STOP_PCT=3% → R:R≥1.5 → 止盈≥4.5% >> 波动 0.766%
                #   - 止盈触发率 0.0%, 87.3% 时间止损, 在不利价格平仓
                #   - 空头胜率 0%, 多头胜率 19.4%
                # 修复原理 (用户要求"理解底层逻辑, 而非表面指标"):
                #   1. MIN_STOP_PCT 降到 1.5% (与基因范围下限一致, 适配 BTC 1h 波动率)
                #   2. 移除 R:R≥1.5 下限强制 (让基因参数直接生效, 进化算法学习最优 R:R)
                #   3. Fix 8 原上限 R:R≤2.0, Fix 22 提升到 R:R≤3.0 (适配趋势跟随 TP 4-6%)
                # 铁律: "杜绝模拟牛逼,实盘亏钱" — 止盈不可达 = 模拟必亏
                MIN_STOP_PCT = 0.015  # v503 Fix 8: 3.0%→1.5% (适配 BTC 1h 波动率 0.766%)
                if gene_sl_pct < MIN_STOP_PCT:
                    gene_sl_pct = MIN_STOP_PCT
                # v503 Fix 8/22: 移除 R:R≥1.5 下限, R:R 上限 3.0 (原 2.0, Fix 22 提升)
                # 基因 tp_pct 范围 (0.04, 0.06) [Fix 20 趋势跟随], 直接生效
                effective_tp_pct = gene_tp_pct
                # v503 Fix 14: 移除 R:R≥1.0 下限 — 允许剥头皮策略 (TP<SL)
                # 根因诊断 (Fix 12 后 R:R=1.01, TP 触发率仍 0%):
                #   - INTRADAY_GENE_OVERRIDES 设计: SL=(0.02,0.03), TP=(0.015,0.025)
                #   - 设计意图: TP<SL = 高胜率剥头皮 (R:R=0.5-0.83, 胜率50-60%)
                #   - 但代码强制 TP≥SL (R:R≥1.0) → 破坏基因设计 → TP=SL=2.2%
                #   - TP=2.2% 超过 BTC 中位波动 0.569% → TP 触发率 0%
                # 修复原理 (用户铁律: "理解底层逻辑, 而非表面指标"):
                #   1. 剥头皮策略数学上有效: R:R=0.75, 胜率=65% → EV=+0.175%/笔
                #   2. TP<SL 时 TP 更容易触发 (1.5-2.5% < 2-3%), 提高胜率
                #   3. 进化算法学习最优 R:R, 不应人为限制 R:R≥1.0
                #   4. 仅保留 R:R≤3.0 上限 (Fix 22: 原 2.0, 适配趋势跟随 TP 4-6%)
                # 来源:scalping 策略原理 + INTRADAY_GENE_OVERRIDES 设计注释
                # 铁律: "杜绝模拟牛逼,实盘亏钱" — 强制 R:R≥1.0 = TP 不可达 = 必亏
                # ATR-based 距离 (仅作参考和极端波动保护)
                atr_sl_distance = abs(entry_price - sl_tp["stop_loss"])
                atr_tp_distance = abs(sl_tp["take_profit"] - entry_price)
                # gene-based 距离 (进化算法学到的最优SL/TP)
                gene_sl_distance = entry_price * gene_sl_pct
                gene_tp_distance = entry_price * effective_tp_pct
                # v503 Fix 12: 移除 ATR 覆盖 — 基因参数直接生效
                # 根因诊断 (P3-12 + Fix 10/11 后仍 TP=0%, win=0%):
                #   - min(atr_sl, gene_sl) 永远选 ATR (0.655%) 而非 gene (1.5-2.5%)
                #   - ATR×1.5=0.437% 远低于手续费门槛 2.0% (0.4% 来回 = SL 的 66%)
                #   - TP=1.205% (2×SL) 超过 BTC 中位波动 0.569% → TP 触发率 0%
                #   - 基因进化结果被完全忽略 — min() 让进化算法失效
                # 修复原理 (用户铁律: "理解底层逻辑, 而非表面指标"):
                #   1. 基因参数是进化算法学到的最优 SL/TP, 应该直接生效
                #   2. ATR 只是市场测量, 不应覆盖进化结果
                #   3. 保留 ATR 作为极端波动上限保护 (ATR×8), 防止基因在闪崩时
                #      设置过紧止损被瞬间扫掉 (但不在正常波动时收紧止损)
                # 来源: 风险管理 "let evolution work, but cap extremes"
                # 铁律: "杜绝模拟牛逼,实盘亏钱" — ATR 覆盖 = 数学上必亏 = 模拟必亏
                MAX_SL_ATR_MULT = 8.0  # 极端波动上限: SL ≤ ATR×8 (BTC 1h ATR≈0.437% → 上限3.5%)
                atr_extreme_cap = atr_sl_distance * MAX_SL_ATR_MULT if atr_sl_distance > 0 else gene_sl_distance
                final_sl_distance = min(gene_sl_distance, atr_extreme_cap)
                # TP 跟随 SL (保持 R:R), 不用 ATR 覆盖
                final_tp_distance = gene_tp_distance
                # v503 Fix 26: ATR 自适应 TP/SL — 按波动性确保 TP 可达 + SL 不被噪声扫出
                # 根因 (P53c 诊断): 固定百分比 TP/SL 在不同波动性币种上表现不一致
                #   P53 多币种验证: 跨币种 EV mean=-0.0145%, stdev=1.1552% (极度不一致)
                #   P53c ATR 分析: 所有币种 ATR/TP < 0.3 → TP 全部太远, 24h 触发率<10%
                #     BTC (ATR 0.676%): TP 4% = 5.9× ATR → P53 TP 触发率 0%
                #     BNB (ATR 0.660%): TP 4% = 6.1× ATR → P53 TP 触发率 51.6% (强趋势才达)
                #     DOGE (ATR 0.908%): TP 4% = 4.4× ATR → P53 TP 触发率 0%
                # 修复原理 (用户铁律: "理解底层逻辑, 而非表面指标"):
                #   1. ATR = 市场呼吸频率, TP/SL 应基于呼吸频率而非固定值
                #   2. 24h 预期波动 = ATR × √24 = 4.9× ATR, TP ≤ 5× ATR 确保可达
                #   3. SL ≥ 1.5× ATR 防止被正常波动扫出 (给策略呼吸空间)
                #   4. 基因进化仍控制 R:R 比例, ATR 只提供波动性自适应
                # 数学验证:
                #   TP = 5× ATR: 24h 触发需 5/4.9 = 1.02σ → ~15% 触发概率 (vs 原 <10%)
                #   SL = 1.5× ATR: 正常波动扫出需 -1.5σ → ~7% 误触发 (可接受)
                #   R:R = 1.5/2.0 = 0.75, 盈亏平衡胜率 = 57% (日内策略 55%+ 可达)
                #   跨币种一致性: TP/SL 都按 ATR 倍数, 高/低波动币种行为一致
                #   铁律: "杜绝模拟牛逼,实盘亏钱" — 固定 TP/SL 跨币种不一致 = 模拟必亏
                # v503 Fix 30: TP/SL 可达性重校准 — P60 诊断 TP 100% 不可达 (0/121)
                #   根因: TP=3-4% 但持仓期 MAE=0-1.7%, TP 永远不可达
                #         SL=1.5% 但 MFE=1.5-3%, SL 必被盘中插针扫出
                #   修复: TP cap 5×ATR→1.5×ATR (让 TP 在 24h 内可达)
                #         SL floor 1.5×ATR→2.0×ATR (让 SL 存活盘中插针)
                #   数学验证 (DOGE ATR≈1.92%):
                #     TP cap = 1.5×1.92% = 2.88% (vs MAE 0-1.7%, 仍偏紧但 5×ATR=9.6% 必不可达)
                #     SL floor = 2.0×1.92% = 3.84% (vs MFE 1.5-3%, 应能存活多数插针)
                #     R:R = 2.88/3.84 = 0.75, 盈亏平衡胜率 = 57% (目标 55% 可达)
                #   来源: P60 Short Trade 诊断 (121 笔空头 0% 胜率, TP 0% 可达)
                #   铁律: "理解底层逻辑" — TP 必须 ≤ 持仓期最大有利偏移才可达
                if atr_val > 0 and entry_price > 0:
                    # v503 Fix 34: Per-symbol SL/TP 调整 — SOL 放宽SL + DOGE 收紧TP
                    # 根因 (P68c 诊断, 用户铁律: "理解底层逻辑, 而非表面指标"):
                    #   SOL: SL率49.2%过高 → SL=2.0×ATR太紧, 被盘中插针扫出
                    #        118 trades, WR=49.2%, SL=49.2%, R:R=0.78, EV=-0.2889%
                    #        盈亏平衡R:R=(1-0.492)/0.492=1.03, 实际0.78 < 1.03 → 必亏
                    #        修复: SL 2.0→2.5×ATR (给SOL更多呼吸空间, 降低SL触发率)
                    #
                    #   DOGE: maxHold=36.7%过高 → TP=2.0×ATR太远, 24h内不可达
                    #         79 trades, WR=44.3%, TP=39.2%, maxHold=36.7%, R:R=0.88
                    #         盈亏平衡R:R=(1-0.443)/0.443=1.26, 实际0.88 < 1.26 → 必亏
                    #         修复: TP 2.0→1.5×ATR (让TP在24h内可达, 降低maxHold)
                    #
                    # 底层逻辑:
                    #   1. SL/TP 要匹配币种波动特性, 不是越紧越好也不是越松越好
                    #   2. SOL 波动大, 需更宽SL避免被插针扫出 (SL率49.2%=一半交易被扫)
                    #   3. DOGE 波动小但TP太远, 24h内价格触达TP概率低 (maxHold=36.7%)
                    #   4. per-symbol SL/TP = 风险参数匹配币种微观结构
                    #
                    # 数学验证:
                    #   SOL (SL=2.5, TP=2.0): R:R=0.80, 预期SL率49%→35%, WR→57.8%✅, EV=+0.040
                    #   DOGE (TP=1.5, SL=2.0): R:R=0.75, 预期TP率39%→55%, WR→69.6%✅, EV=+0.218
                    #   R:R cap (L1075) = final_sl×1.0, 对SOL: TP=2.0<2.5=cap, 不再截断
                    # 来源: ATR自适应波动性理论 + per-symbol微观结构匹配
                    # 铁律: "理解底层逻辑" — SL/TP匹配币种波动特性, 而非一刀切
                    # 根治(2026-07-16): SLTP ATR floor/cap 根治 — 50轮923笔交易胜率仅10.4%
                    # 根因诊断 (audit.log 50轮真实数据):
                    #   - 67.7%交易是explicit_stop_loss (止损过紧)
                    #   - BTC entry=107533, atr_val~256(0.24%), gene_sl=1992(1.85%)
                    #   - min_sl_by_atr=256x2.0=512(0.48%) 把gene_sl从1.85%缩到0.48%
                    #   - BTC 1h盘中插针1-2%, 0.48%止损几乎必触 -> 67.7%止损率
                    #   - 同时TP cap把gene_tp缩到512, SL同步缩小到364, 再被floor拉到512
                    #   - 最终SL=TP=512, R:R=1.0 而非gene设计的1.85%/2.6% R:R=1.41
                    # 修复原理 (铁律14: 零纸面通过, 真实数据驱动):
                    #   1. MIN_SL_ATR_MULT 2.0->4.0: SL floor=4xATR=1.0%(BTC), 存活多数插针
                    #   2. MAX_TP_ATR_MULT 2.0->8.0: TP cap=8xATR=2.0%, 让gene TP可达
                    #   3. 移除TP cap对SL的同步缩小: 保持gene原始R:R, 不破坏策略DNA
                    #   4. 保留SL floor对TP的同步放大: R:R保持, SL放宽时TP同步放宽
                    # 数学验证:
                    #   BTC atr_val=256: SL floor=4x256=1024(0.95%), TP cap=8x256=2048(1.9%)
                    #   gene_sl=1992(1.85%) > 1024 -> 不触发floor, gene直接生效
                    #   gene_tp=2801(2.6%) > 2048 -> TP被cap到2048, SL保持1992, R:R=1.03
                    #   原R:R=1.41, cap后R:R=1.03 — 仍优于原0.48%/0.67% R:R=1.0
                    # 铁律: "理解底层逻辑" — ATR是市场呼吸测量, 不应覆盖进化算法学到的最优SL/TP
                    if _symbol == "SOL-USDT":
                        MAX_TP_ATR_MULT = 8.0   # 根治: 2.0->8.0 (让gene TP可达, 不截断)
                        MIN_SL_ATR_MULT = 4.0   # 根治: 2.5->4.0 (SL floor=4xATR, 存活插针)
                    elif _symbol == "DOGE-USDT":
                        MAX_TP_ATR_MULT = 6.0   # 根治: 1.5->6.0 (DOGE ATR大, 6x足够)
                        MIN_SL_ATR_MULT = 4.0   # 根治: 2.0->4.0 (SL floor放宽)
                    else:
                        MAX_TP_ATR_MULT = 8.0   # 根治: 2.0->8.0 (默认, 让gene TP可达)
                        MIN_SL_ATR_MULT = 4.0   # 根治: 2.0->4.0 (默认, SL存活插针)
                    max_tp_by_atr = atr_val * MAX_TP_ATR_MULT
                    min_sl_by_atr = atr_val * MIN_SL_ATR_MULT
                    if final_tp_distance > max_tp_by_atr:
                        # 根治(2026-07-16): TP被cap截断时, 不再缩小SL (保持gene R:R)
                        # 原问题: TP cap截断后SL按比例缩小, 导致SL过紧被扫出
                        # 修复: 只截断TP, SL保持gene值, 让进化算法学习最优R:R
                        final_tp_distance = max_tp_by_atr
                    if final_sl_distance < min_sl_by_atr:
                        # 保留: SL被ATR floor拉高时, TP按相同比例放大, 保持gene R:R
                        _sl_floor_ratio = min_sl_by_atr / max(final_sl_distance, 1e-9)
                        final_sl_distance = min_sl_by_atr
                        if _sl_floor_ratio > 1.0:
                            final_tp_distance = final_tp_distance * _sl_floor_ratio
                # v503 Fix 20: R:R 上限从 2.0 提升到 3.0 — 支持趋势跟随策略
                # 根因 (P47 诊断): Fix 8 的 R:R=2.0 上限是为剥头皮设计 (TP 0.8-1.5%),
                #   但 Fix 20 已将基因切换为趋势跟随 (TP 4-6%, SL 1.5-2.5%, R:R=2.5-4.0)
                #   R:R=2.0 上限会截断趋势策略的 TP: 6%→3%, R:R 从 4.0→2.0
                #   → 数学上又回到剥头皮场景, 趋势信号低胜率(30-40%) + 截断TP = 必亏
                # 深层逻辑 (用户铁律: "理解底层逻辑, 而非表面指标"):
                #   1. R:R 上限必须匹配策略类型, 否则策略基因被代码静默覆盖
                #   2. 趋势跟随本质 = 低胜率 + 高 R:R, 强制 R:R≤2 = 破坏策略 DNA
                #   3. Fix 14 已删除 R:R≥1.0 下限允许剥头皮, Fix 20 提升上限允许趋势跟随
                # 数学验证:
                #   新 R:R 上限 = 3.0, 趋势策略 R:R=2.5-4.0, 上限截断 R:R>3.0 的极端情况
                #   8h 波动 = 0.766% × √8 = 2.17%, TP 中位数 5% (24h 强趋势可达 3.75%)
                #   R:R=1.5 时盈亏平衡胜率 = 1/(1+1.5) = 40% << 日内策略 55%+ → 正期望
                # v503 Fix 30: R:R cap 3.0→1.5 — 匹配 TP/SL 可达性重校准
                #   根因: Fix 30 将 TP cap 从 5×ATR 降到 1.5×ATR, R:R cap=3.0 不再可达
                #   修复: R:R cap=1.5, 匹配新 TP/SL 比例 (TP=1.5×ATR, SL=2.0×ATR, R:R=0.75)
                #   铁律: "理解底层逻辑" — R:R cap 必须 ≤ TP_cap/SL_floor 才有意义
                max_tp_distance = final_sl_distance * 2.5  # v14.42 Fix 8: R:R cap 2.0→2.5 (提升盈利空间, 趋势策略TP可达2×SL)
                if final_tp_distance > max_tp_distance:
                    final_tp_distance = max_tp_distance
                # v14.40 Fix 1: SL绝对百分比上限 — 根治 OP-USDT SL=33.14% 问题
                # 根因 (v14_39 audit.log诊断): MAX_SL_ATR_MULT=8.0 对高波动币种失效
                #   - OP-USDT ATR≈8.3%, 4xATR=33.14% → 单笔亏损最多33% (爆仓级)
                #   - 4xATR floor 在低波动币种(BTC ATR 0.437%)合理, 高波动币种灾难
                # 修复原理 (铁律: "理解底层逻辑"):
                #   1. ATR是相对测量, 高波动币种ATR本身就大, 4xATR放大不合理
                #   2. SL应限制单笔最大亏损比例, 而非无脑按ATR倍数
                #   3. 10%是单笔交易最大可接受亏损 (3x杠杆下=30%账户回撤上限)
                # 数学验证:
                #   OP-USDT: SL=33.14% → cap到10% (仓位=10/33=0.303x原仓位, 亏损=10%)
                #   BTC: SL=1.75% < 10% → 不变
                #   DOGE: SL=7.68% < 10% → 不变
                #   SOL: SL=2.5xATR≈5% < 10% → 不变
                # 铁律: "杜绝模拟牛逼,实盘亏钱" — 33%单笔亏损 = 实盘爆仓 = 模拟必亏
                MAX_SL_PCT = 0.10  # 单笔最大亏损10% (硬上限, 不可被ATR放大突破)
                if entry_price > 0:
                    _max_sl_distance_abs = entry_price * MAX_SL_PCT
                    if final_sl_distance > _max_sl_distance_abs:
                        # SL被cap到10%时, TP按比例缩小保持R:R (避免破坏策略DNA)
                        _sl_cap_ratio = _max_sl_distance_abs / max(final_sl_distance, 1e-9)
                        final_sl_distance = _max_sl_distance_abs
                        final_tp_distance = final_tp_distance * _sl_cap_ratio
                        try:
                            logger.info(
                                f"[v14.40] SL_ABS_CAP: {getattr(self,'_symbol','?')} "
                                f"SL {final_sl_distance/_sl_cap_ratio:.4f}->{final_sl_distance:.4f} "
                                f"({MAX_SL_PCT*100:.1f}%% cap), TP scaled to {final_tp_distance:.4f}"
                            )
                        except Exception:
                            pass
                # v503 Fix 14: 移除 R:R≥1.0 下限 — 允许剥头皮策略 (TP<SL)
                # 原代码: if final_tp_distance < final_sl_distance * 1.0: final_tp_distance = final_sl_distance * 1.0
                # 已删除: 该下限强制 R:R≥1.0, 破坏 INTRADAY_GENE_OVERRIDES 的剥头皮设计
                # v503 Fix 20: 已切换为趋势跟随设计 (TP 4-6% > SL 1.5-2.5%), R:R≥1.6 自然满足
                # 应用最终 SL/TP
                if signal_direction == "long":
                    stop_loss = entry_price - final_sl_distance
                    take_profit = entry_price + final_tp_distance
                else:
                    stop_loss = entry_price + final_sl_distance
                    take_profit = entry_price - final_tp_distance
                # v503 Fix 10 诊断: 打印前5次 SL/TP 计算详情
                if not hasattr(self, '_v503_sltp_diag'):
                    self._v503_sltp_diag = 0
                self._v503_sltp_diag += 1
                if self._v503_sltp_diag <= 5:
                    logger.warning(
                        "v503 SLTP_DIAG #%d dir=%s entry=%.2f SL=%.2f TP=%.2f | "
                        "atr_sl=%.4f atr_tp=%.4f gene_sl=%.4f gene_tp=%.4f | "
                        "final_sl=%.4f final_tp=%.4f RR=%.2f | "
                        "gene_sl_pct=%.4f gene_tp_pct=%.4f eff_tp_pct=%.4f reg_adj=%s",
                        self._v503_sltp_diag, signal_direction, entry_price,
                        stop_loss, take_profit,
                        atr_sl_distance, atr_tp_distance, gene_sl_distance, gene_tp_distance,
                        final_sl_distance, final_tp_distance,
                        final_tp_distance / final_sl_distance if final_sl_distance > 0 else 0,
                        self.gene.exit_risk_stop_loss_pct, self.gene.exit_risk_take_profit_pct,
                        effective_tp_pct, regime_adjustment,
                    )
                # 重新计算 Kelly (使用 capped stop_loss)
                current_volatility = atr_val / entry_price if entry_price > 0 else 0.02
                kelly_result = frontier.kelly.compute_position_size(
                    equity, entry_price, stop_loss, current_volatility,
                )
                quantity = kelly_result["position_size"]
            except Exception as kelly_err:
                logger.warning(
                    "Kelly/ATR 计算异常, 降级到基因原始仓位: %s",
                    str(kelly_err)[:150],
                )
                quantity = self._compute_position_size(balance, equity, market.close, atr=atr_val)  # Phase 14.13: 传入ATR
                stop_loss, take_profit = self._compute_exit_points(
                    market.close, signal_direction,
                    current_price=market.close,
                    atr=atr_val,  # v503 Fix 27: 使用修复后的 atr_val (原 data_packet.get("atr", 0) 永远=0)
                    current_pnl_pct=0.0,  # 开仓时无浮动盈亏
                )
        else:
            # 回退：使用基因原始仓位计算
            quantity = self._compute_position_size(balance, equity, market.close, atr=atr_val)  # Phase 14.13: 传入ATR
            stop_loss, take_profit = self._compute_exit_points(
                market.close, signal_direction,
                current_price=market.close,
                atr=atr_val,  # v503 Fix 27: 使用修复后的 atr_val (原 data_packet.get("atr", 0) 永远=0)
                current_pnl_pct=0.0,
            )

        if quantity <= 0:
            return self._early_return(start)

        # v14.41 Fix 4: 高波动币种仓位降低 + regime_flip仓位降低
        # 根因 (diag_lightning.py 闪电止损103笔0%胜率-7.728%):
        #   - OP-USDT 32笔闪电止损 avg=-12.18% (ATR≈8.3%, 1bar波动10%+)
        #   - UNI-USDT 19笔闪电止损 avg=-10.43%
        #   - regime_flip:counter_trend_short 28笔闪电止损 (翻转时机差)
        #   - 中位SL=9.69% (高波动币种SL接近10%上限)
        # 修复原理 (铁律: "理解底层逻辑"):
        #   1. ATR/price > 5% = 高波动币种, 1bar内可能触发SL → 仓位×0.5
        #   2. regime_flip后的交易 = 翻转信号, 时机不确定 → 仓位×0.7
        #   3. 仓位降低 = 即使触发SL, 亏损绝对值减半 (风控)
        # 数学验证:
        #   OP ATR=8.3%: 仓位×0.5, SL=10%, 实际亏损=5% (vs 10%)
        #   regime_flip short: 仓位×0.7, SL=10%, 实际亏损=7% (vs 10%)
        #   组合: OP+regime_flip: 仓位×0.35, SL=10%, 实际亏损=3.5%
        # 铁律: "杜绝模拟牛逼,实盘亏钱" — 高波动币种全仓 = 必亏 = 无进化价值
        _v1441_vol_mult = 1.0
        _v1441_entry = market.close if market else 0.0
        if _v1441_entry > 0 and atr_val > 0:
            _atr_pct = atr_val / _v1441_entry
            if _atr_pct > 0.08:
                # 极高波动 (ATR>8%): 仓位×0.4 (OP/UNI级别)
                _v1441_vol_mult *= 0.4
            elif _atr_pct > 0.05:
                # 高波动 (ATR>5%): 仓位×0.6
                _v1441_vol_mult *= 0.6
            elif _atr_pct > 0.03:
                # 中高波动 (ATR>3%): 仓位×0.8
                _v1441_vol_mult *= 0.8

        # regime_flip后的交易仓位降低 (翻转信号时机不确定)
        _v1441_signal_src = getattr(self, "_last_signal_source", "") or ""
        if "regime_flip" in _v1441_signal_src:
            _v1441_vol_mult *= 0.7  # 翻转信号仓位×0.7

        if _v1441_vol_mult < 1.0:
            quantity = quantity * _v1441_vol_mult
            try:
                logger.info(
                    f"[v14.41] VOL_POSITION: {getattr(self,'_symbol','?')} "
                    f"atr_pct={_atr_pct*100:.2f}% pos_mult={_v1441_vol_mult:.2f} "
                    f"regime_flip={'Y' if 'regime_flip' in _v1441_signal_src else 'N'}"
                )
            except Exception:
                pass

        # v14.42 Fix 6: ADA-USDT per-symbol仓位调整 (新热点)
        # 根因 (v14.41 ADA 37笔闪电止损): ADA ATR较高, 通用仓位调整不够
        # 修复: ADA额外仓位×0.7 (叠加在通用调整之上)
        _v1442_sym = getattr(self, "_symbol", "") or ""
        if "ADA" in _v1442_sym and _v1441_vol_mult < 1.0:
            quantity = quantity * 0.7  # ADA额外×0.7
            try:
                logger.info(f"[v14.42] ADA_EXTRA_POSITION: {_v1442_sym} extra×0.7")
            except Exception:
                pass

        # 市场状态检测：Regime仓位调整
        # 来源：Hamilton 1989 HMM + GARCH波动率状态
        # 牛市加仓，熊市减仓，状态转换期减仓
        if regime_data:
            regime_adj = regime_data.get("position_adjustment", 1.0)
            quantity = quantity * regime_adj

        # v447 认知层应用: 根据 regime_adjustment 调整仓位
        # 当 regime 失败模式被识别时, 使用失败模式知识库推荐的仓位系数
        # 例如: downtrend 逆势做多时仓位系数 0.5 (大幅减仓)
        #       volatile 假突破时仓位系数 0.6
        if regime_adjustment:
            quantity = quantity * regime_adjustment.get("position_size_multiplier", 1.0)

        # v598 Phase 2: v91 量价双维8状态 + Kelly全局缩放 (短板#2 深化运行时集成)
        # 来源: _v91_volume_kelly_optimize.py 离线突破 ann=65.30%, CI下限33.98%
        # 集成逻辑: atr_pct + vp_vol_ratio_5 → 8状态分类 → market_factor × kelly_scaler
        # 设计原则:
        #   1. 只移植PRINCIPLE(8状态+Kelly), 不硬编码v91网格搜索"最优因子"
        #   2. Kelly scaler 从运行时实际PnL累计计算(避免过拟合)
        #   3. 不硬性过滤交易(ERR-109 v518证伪), 只调整仓位大小
        #   4. 安全边界: 最终仓位乘数限制在[0.3, 3.0]
        # 主动应用教训:
        #   ERR-20260701-v91: 量价双维+Kelly → 运行时集成
        #   ERR-20260701-v88fp: 状态分类有效 → 用8状态精细化
        #   ERR-100 (v534 v90): 2/3 Kelly → 应用portfolio Kelly
        #   用户铁律: "凯利公式+量价分析融会贯通" + "不通过已知数据反推策略"
        try:
            _v91_mult, _v91_state, _v91_factor = self._compute_v91_adjustment(
                atr_val=atr_val,
                close_price=market.close,
                indicators=indicators,
            )
            if _v91_mult != 1.0:
                quantity = quantity * _v91_mult
        except Exception as _v91_err:
            logger.debug("v91 adjustment failed (safe fallback): %s", str(_v91_err)[:80])

        # ============== Phase 3.2 集成: 市场状态分类驱动仓位调整 (短板#6) ==============
        # 用户要求: "市场状态分类驱动策略选择"
        # ERR-109教训: 策略本质是均值回归, trend状态下表现差($722),
        #              quiet/sideways状态下好($11985) → 需regime-aware仓位调整
        # 设计原则: 不过滤逆势trades(只调整仓位倍数), 遵守ERR-109 v518证伪
        # 来源: RegimeDetectionSystem (HMM 3状态bull/sideways/bear + GARCH波动率)
        # 注入: 由 EvolutionLoop._init_decision_engines() 调用 set_regime_detector()
        # 注意: 只读get_regime(), 不调用update() — EvolutionLoop已每周期更新
        if self._regime_detector is not None:
            try:
                _rd_state = self._regime_detector.get_regime()
                _rd_market_regime = _rd_state.get("market_regime", "sideways")
                _rd_vol_regime = _rd_state.get("volatility_state", {}).get(
                    "volatility_regime", "NORMAL"
                )
                _rd_risk_level = _rd_state.get("risk_level", "LOW")

                # 均值回归策略的regime-aware仓位倍数 (ERR-109应用)
                # sideways(震荡): 均值回归最优状态 → 加仓1.2
                # bull+LOW vol(低波动牛市): 适度加仓1.1
                # bull+HIGH vol(高波动趋势): 均值回归劣势 → 减仓0.6
                # bear(熊市): 减仓0.6
                # 其他: 基准1.0
                if _rd_market_regime == "sideways":
                    _rd_position_mult = 1.2
                elif _rd_market_regime == "bull" and _rd_vol_regime == "LOW":
                    _rd_position_mult = 1.1
                elif _rd_market_regime == "bull" and _rd_vol_regime == "HIGH":
                    _rd_position_mult = 0.6
                elif _rd_market_regime == "bear":
                    _rd_position_mult = 0.6
                else:
                    _rd_position_mult = 1.0

                # EXTREME risk额外减仓 (风控优先, 覆盖策略偏好)
                if _rd_risk_level == "EXTREME":
                    _rd_position_mult *= 0.5

                # 安全边界: 限制在[0.3, 2.0] (与v91一致的安全范围)
                _rd_position_mult = max(0.3, min(2.0, _rd_position_mult))

                quantity = quantity * _rd_position_mult

                if self._diag_counter % 500 == 1:
                    logger.warning(
                        "v32 REGIME: agent=%s market=%s vol=%s risk=%s mult=%.2f qty=%.6f",
                        getattr(self.gene, 'agent_id', '?'),
                        _rd_market_regime, _rd_vol_regime, _rd_risk_level,
                        _rd_position_mult, quantity,
                    )
            except Exception as _rd_err:
                # 降级跳过: regime检测失败不影响主决策流程
                logger.debug("市场状态检测降级跳过: %s", str(_rd_err)[:80])
        # ============== Phase 3.2 集成 END ==============

        if quantity <= 0:
            return self._early_return(start)

        # v2.37 修复 BUG-C (INTRADAY-GUARD): round 截断后再次检查 quantity
        # 根因: 高价 symbol (如 SOL=235万) 时 quantity 极小 (如 0.0000212),
        #       round(quantity, 4) 截断为 0.0, 但 line 632 的检查在 round 之前无法拦截
        #       → quantity=0 订单仍被 engine 接受 → agent 无实际持仓但亏钱
        # 修复: round 后再次检查, 若为 0 则不交易 (避免空订单)
        rounded_quantity = round(quantity, 4)
        if rounded_quantity <= 0:
            # 数量过小无法交易 (价格过高或仓位过小), 跳过
            return self._early_return(start)

        # v2.25: CrossSectional横截面动量方向过滤
        # 如果信号做多但CrossSectional建议做空该symbol,则不交易
        # 如果信号做空但CrossSectional建议做多该symbol,则不交易
        # 来源: Antonacci Dual Momentum — 截面排名是最强的动量信号之一
        if cross_sectional_data and cs_long_assets and cs_short_assets:
            if signal_direction == "long" and symbol in cs_short_assets:
                return self._early_return(start)  # 截面动量显示该symbol应做空,不做多
            if signal_direction == "short" and symbol in cs_long_assets:
                return self._early_return(start)  # 截面动量显示该symbol应做多,不做空

        # R16: 记录交易bar, 控制频率
        self._last_trade_bar[symbol] = self._current_bar_index

        # v3.32n 修复#3: R:R 最终校验 (防御性编程)
        # 根因: v3.32m 实测发现10/14笔交易 R:R<1.5 (最低0.909), 说明
        #       line 734-735 和 line 1763-1765 的 R:R>=1.5 强制被某条路径绕过
        # 修复: 在 decide() 返回前再次校验, 保证最终返回的 R:R>=1.5
        # 反过拟合: 这是标准风险管理实践, R:R<1.5=负期望系统
        # v3.32n 修复BUG: 原代码 take_profit = entry_px + target_tp_dist 错误
        #   target_tp_dist是比例值(0.045), entry_px是价格值(157463)
        #   直接相加导致TP≈entry, R:R校验反而制造TP<entry的交易
        #   正确: take_profit = entry_px * (1 + target_tp_dist)
        # v3.32o 修复BUG: R:R校验基于 market.close, 但实际成交价 entry_price 因滑点
        #   而偏离 market.close (多头 entry>close, 空头 entry<close)
        #   导致校验时 R:R>=1.5 但基于实际 entry_price 的 R:R<1.5
        #   实测: 8/9笔交易 R:R<1.5 (平均1.45), 校验形同虚设
        # 修复: 使用 market.ask(多头)/market.bid(空头) 作为预估 entry_price
        #   这更接近实际成交价, 使 R:R 校验生效
        # v3.32o 修复#2: market.ask 仍不够 — sandbox_engine 实际成交价 =
        #   market.ask × (1 + 滑点 + 延迟 + K线跳变) (sandbox_engine.py:447,465-479)
        #   实测: 82/86笔交易 R:R<1.5 (95.3%), 即使 spread=0.003 仍无效
        # v3.32o 修复#3 (SLIPPAGE_BUFFER=0.005): 仍 109/145 (75.2%) R:R<1.5
        #   滑点诊断 (_slippage_diag.py): 实际滑点 avg=0.67%, p95=1.43%, max=1.99%
        #   当前 0.5% 缓冲只覆盖~50%交易, 导致校验失效
        # 修复: SLIPPAGE_BUFFER 从 0.005 → 0.015 (1.5%) 覆盖 p95 滑点
        #   来源: 实证数据 (_slippage_diag.py p95=1.43%) + 真实市场研究
        #   - Binance BTC合约: taker费0.1% + 市场冲击0.3-0.8% + 延迟0.1-0.3% + 跳变0.2-0.5%
        #   - 总成本典型 0.7-1.7%, p95=1.43% 是统计合理上限
        # 反过拟合: 基于实测数据校准, 不是为数字好看调整; 真实市场滑点确实如此
        # 铁律: "杜绝模拟牛逼,实盘亏钱" — 模拟必须包含真实滑点成本
        if stop_loss > 0 and take_profit > 0 and market.close > 0:
            # v503 Fix 10: 移除 SLIPPAGE_BUFFER — 统一 R:R 测量参考点
            # 根因诊断 (P3-15 _v503_p40_trade_diag.py 第三轮验证):
            #   SL/TP 在 L784 基于 entry_price=market.close 计算:
            #     stop_loss = market.close ± final_sl_distance
            #     take_profit = market.close ± final_tp_distance
            #   但 RR_FLOOR/RR_CAP 使用 entry_px=market.ask*(1+0.015) 测量 R:R:
            #     参考点偏移 1.5% → sl_dist 从 0.59% 膨胀到 2.12%
            #     RR_FLOOR 将 tp_dist 提升到膨胀后的 sl_dist → TP 偏移 2.12%
            #     从 market.close 测量 TP 偏移 ≈ 3.5% >> BTC 1h 波动率 0.533%
            #     → TP 触发率 0.0%, 87% 时间止损, 胜率 9.3%
            #   对空头更严重: entry_px=bid*(1-0.015) 低于 market.close,
            #     当 final_tp_distance < 1.5% 时, TP 被推到 entry_px 上方
            #     (空头 TP 应在下方) → 方向错误, 必亏
            # 修复原理 (用户要求"理解底层逻辑, 而非表面指标"):
            #   1. entry_px = market.close (与 SL/TP 计算参考点一致)
            #   2. R:R 测量基于一致参考点, 不再被 SLIPPAGE_BUFFER 膨胀
            #   3. RR_FLOOR 提升 tp_dist 时, 基于真实 sl_dist (0.59%) 而非膨胀值 (2.12%)
            #   4. TP 距离从 3.5% 降到 0.72% (1.35x 波动率, 2-3h 内可达)
            # 滑点风险处理:
            #   - sandbox_engine 撮合模型已包含真实滑点 (avg=0.67%, p95=1.43%)
            #   - 仓位计算的 2% 硬上限 (L1014) 使用 stop_pct, 不依赖 entry_px
            #   - SLIPPAGE_BUFFER 在 R:R 校验中是重复计算, 移除不影响风控
            # 铁律: "杜绝模拟牛逼,实盘亏钱" — 参考点不一致 = 模拟必亏
            entry_px = market.close
            sl_dist = abs(entry_px - stop_loss) / entry_px
            tp_dist = abs(take_profit - entry_px) / entry_px
            # v503 Fix 14: 移除 R:R≥1.0 下限 — 允许剥头皮策略 (TP<SL)
            # 原代码: if tp_dist < sl_dist * 1.0: 强制 TP=SL (R:R=1.0)
            # 已删除: 该下限破坏 INTRADAY_GENE_OVERRIDES 的剥头皮设计
            # 基因设计 TP=(0.015,0.025) < SL=(0.02,0.03), R:R=0.5-0.83, 靠高胜率补偿
            # 数学验证: R:R=0.75, 胜率=65% → EV=+0.175%/笔 (扣费后仍盈利)
            # v503 Fix 22: R:R 上限从 2.0 提升到 3.0 — 与 Fix 20 (L967) 同步
            # 根因 (P49 诊断): Fix 20 将 L967 的 cap 从 2.0 提升到 3.0, 但本处 (L1131)
            #   仍保留 Fix 8 时代的 cap=2.0, 在 L967 之后再次截断 TP:
            #   - L967 (Fix 20): cap=3.0, 允许 R:R=2.56 通过 → SLTP_DIAG 输出 R:R=2.56 ✅
            #   - L1131 (Fix 8): cap=2.0, 将 R:R=2.56 砍回 2.0 → SLTP_RECALC 输出 R:R=2.00 ❌
            #   - 实测: 2728.53 = 1364.27 × 2.00 (被 cap), 应为 3495.30 = 1364.27 × 2.56
            # 深层逻辑 (用户铁律: "理解底层逻辑, 而非表面指标"):
            #   1. 双重 R:R cap 阈值不一致 = 静默覆盖策略基因 = 模拟必亏
            #   2. Fix 20 已证明 R:R=3.0 适配趋势跟随 (TP 4-6%, 盈亏平衡胜率 25% < 30-40%)
            #   3. 保留 cap=2.0 等于 Fix 20 完全失效, 趋势策略 TP 被截断 6%→3%
            # 数学验证:
            #   R:R=3.0 盈亏平衡胜率 = 1/(1+3.0) = 25% << 趋势策略 30-40% → 正期望
            #   BTC 24h 波动 = 0.766% × √24 = 3.75%, TP 中位数 5% (强趋势可达)
            # 铁律: "杜绝模拟牛逼,实盘亏钱" — 双重 cap 不一致 = 模拟必亏
            # v503 Fix 30: R:R cap 3.0→1.5 — 匹配 TP/SL 可达性重校准 (TP cap 1.5×ATR)
            #   根因: P60 诊断 TP 100% 不可达, R:R=3.0 意味着 TP=3×SL, 必然不可达
            #   修复: R:R≤1.5, 与 L1015 和 _compute_exit_points L2435 同步
            rr_ratio = (tp_dist / sl_dist) if sl_dist > 0 else 0
            if sl_dist > 0 and tp_dist > sl_dist * 2.0 * (1.0 + 1e-9):
                # Phase 12: R:R cap 1.0→2.0 (与 decide() L1567 同步, 提升盈利空间)
                # v503 Fix 30b: R:R > 1.0: 强制降低到 1.0 (与 decide() L1015 同步)
                target_tp_dist = sl_dist * 2.0
                old_tp = take_profit
                if signal_direction == "long":
                    take_profit = entry_px * (1 + target_tp_dist)
                elif signal_direction == "short":
                    take_profit = entry_px * (1 - target_tp_dist)
                logger.warning(
                    "Phase 12 RR_CAP: entry_px=%.2f SL=%.2f TP=%.2f→%.2f sl_d=%.4f tp_d=%.4f→%.4f rr=%.2f→2.00",
                    entry_px, stop_loss, old_tp, take_profit,
                    sl_dist, tp_dist, target_tp_dist, rr_ratio,
                )

        # 杠杆
        leverage = self._compute_leverage()

        # v3.40: 单笔最大亏损硬上限 — 限制单次亏损≤2%本金
        # 根因: v3.39沙盘显示单次最大亏损-23.50 USDT (pnl_pct=-5.36%)
        #   ETH交易仓位过大, 止损滑点导致实际亏损远超预期
        #   用户硬性要求: "单次交易最大亏损控制在本金的2%以内"
        # 修复: 计算最大允许仓位, 确保即使止损滑点50%, 亏损也不超过2%本金
        # 来源: 实盘风险管理铁律 — 单笔风险≤2%是专业交易底线
        # 铁律: "杜绝模拟牛逼,实盘亏钱" — 沙盘必须模拟2%硬上限
        # v3.40b: 移除50%滑点缓冲(过保守导致交易量骤降), 保留2%硬上限
        try:
            capital = equity if equity > 0 else balance
            if capital > 0 and entry_px > 0:
                MAX_LOSS_PCT = 0.02  # 单笔最大亏损=2%本金
                # v3.40b: 直接用stop_pct, 不加滑点缓冲(过保守)
                # 滑点由sandbox_engine的滑点模型处理, 仓位计算不需要额外缓冲
                effective_stop_dist = stop_pct
                max_loss_amount = capital * MAX_LOSS_PCT
                max_allowed_qty = max_loss_amount / (entry_px * effective_stop_dist)
                if rounded_quantity > max_allowed_qty:
                    old_notional = rounded_quantity * entry_px
                    rounded_quantity = max_allowed_qty
                    new_notional = rounded_quantity * entry_px
                    logger.warning(
                        "v3.40 POSITION_CAP: notional %.2f→%.2f "
                        "capital=%.2f stop=%.3f max_loss=%.2f",
                        old_notional, new_notional,
                        capital, stop_pct, max_loss_amount,
                    )
        except Exception as e:
            logger.debug("v3.40 position cap failed: %s", e)

        # Phase 7J-3: 记录正常决策路径耗时
        self._record_decision_latency(start)

        # v503 Fix 11: 计算 sl_distance/tp_distance 供成交后基于实际fill price重算SL/TP
        # 根因: decide()基于market.close计算SL/TP绝对价格, 但撮合引擎成交价含滑点
        #   trade.entry_price = market.ask*(1+滑点) 或 market.bid*(1-滑点) ≠ market.close
        #   导致 SL/TP 与 entry_price 不匹配, R:R从2.0反转为0.62
        # 修复: 传递距离而非绝对价格, 让 evolution_loop 基于实际成交价重算
        _v503_ref_px = market.close if market.close > 0 else 0.0
        _v503_sl_dist = abs(_v503_ref_px - stop_loss) if stop_loss > 0 and _v503_ref_px > 0 else 0.0
        _v503_tp_dist = abs(take_profit - _v503_ref_px) if take_profit > 0 and _v503_ref_px > 0 else 0.0

        # v598 Phase L: v97 自适应持仓时间建议 (ERR-109: hold_bars 是 consequence 不是 cause)
        # 设计依据:
        #   - 市场状态是 CAUSE → 决定适合的持有时间
        #   - hold_bars 是 CONSEQUENCE (好入场+适合状态→长持有→高PnL)
        #   - 本字段仅作建议, 不修改入场/出场条件 (ERR-109 应用边界)
        #   - ERR-v88fp: hold_bars d=+0.461 是最强预测因子, 但仅增强非过滤
        _suggested_hold_bars = 48  # 默认基线 (BASE_HOLD_BARS)
        _hold_regime = "default"
        try:
            _ah_atr_pct = (atr_val / close) if close > 0 else 0.0
            _ah_vol = indicators.get("vp_vol_ratio_5", 1.0) if indicators else 1.0
            _ah_adx = indicators.get("adx_14", 0.0) if indicators else 0.0
            _suggested_hold_bars, _hold_regime = _compute_adaptive_hold(
                _ah_atr_pct, _ah_vol, _ah_adx,
            )
        except Exception as _ah_ex:
            logger.debug("Phase L adaptive_hold 降级 (基线 48): %s", _ah_ex)
            _suggested_hold_bars = 48
            _hold_regime = "degraded"

        # Phase 14.16r: 市场上下文信号增强 (非侵入式,失败不阻塞)
        if _MARKET_CONTEXT_AVAILABLE and hasattr(self, '_current_market_context') and self._current_market_context:
            try:
                _mc_signals = generate_market_context_signals(self._current_market_context, self.gene if hasattr(self, 'gene') else None)
                _mc_total = sum(_mc_signals.values())
                if abs(_mc_total) > 0.01:
                    logger.debug("Phase 14.16r market context signals: %s total=%.3f", _mc_signals, _mc_total)
                    # 市场上下文信号作为最终调整(权重0.2,避免主导)
                    _mc_adjustment = _mc_total * 0.2
                    # 仅调整信号强度,不改变方向(除非信号极强)
                    if _mc_adjustment < -0.5 and signal_direction == "long":
                        signal_direction = "neutral"  # 极强反向信号,转为neutral
                    elif _mc_adjustment > 0.5 and signal_direction == "short":
                        signal_direction = "neutral"
            except Exception as _e:
                logger.debug("Phase 14.16r market context failed (non-fatal): %s", _e)

        return {
            "action": signal_direction,
            "quantity": rounded_quantity,  # v2.37: 使用预先检查过的 rounded_quantity
            "leverage": leverage,
            # v503 Fix 28: 移除 round(x, 2) 灾难性精度损失 — 低价币种 SL/TP 被 round 到 entry
            # 根因 (P57 诊断): round(x, 2) 精度=0.01, 对 DOGE ($0.16) = 价格的 6.2%!
            #   - DOGE entry=0.16012, gene SL=1.63% → stop_loss=0.15751 → round(,2)=0.16 ≈ entry
            #   - SL 距离从 1.63% → 0.075% (0.09×ATR), 82.5% 止损触发, 4.9% 胜率
            #   - sl_distance=0.0026 → round(,2)=0.0 → evolution_loop 走 fallback 用 rounded 绝对价格
            #   - Fix 26 (ATR 自适应) 代码正确执行, 但 round 在最后摧毁了所有精度
            # 修复: 8 位小数 (Satoshi 精度, 加密货币交易所标准)
            #   - BTC $100k: 精度=$0.00000001 = 0.00000001% (远超需求)
            #   - DOGE $0.16: 精度=$0.00000001 = 0.00000625% (远超需求)
            # 来源: Binance/OKX API 文档 — 价格精度统一 8 位小数
            # 铁律: "理解底层逻辑, 而非表面指标" — round(,2) 是 BTC 时代的遗留, 不适配低价币种
            # 铁律: "杜绝模拟牛逼,实盘亏钱" — round 摧毁 SL/TP = 模拟必亏
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            # v503 Fix 11: SL/TP 距离 (基于 market.close 计算)
            # evolution_loop 成交后用 trade.entry_price ± distance 重算绝对价格
            "sl_distance": round(_v503_sl_dist, 8),
            "tp_distance": round(_v503_tp_dist, 8),
            "signal_score": round(signal_score, 3),
            "market_state": market_state["regime"],
            "confidence": signal_score,
            # v3.23 修复: 添加 signal_source 字段到 decision 字典
            # 根因: evolution_loop.py:1776 通过 decision.get("signal_source", "") 获取
            # 但原 decide() 不返回此字段, 导致 audit.log 中 signal_source 始终为空
            "signal_source": getattr(self, "_last_signal_source", "none"),
            # v598 Phase L: v97 自适应持仓建议 (ERR-109: 仅建议非强制出场)
            # evolution_loop 可读取此字段用于风险报告/审计, 但不得用作过滤条件
            "suggested_hold_bars": int(_suggested_hold_bars),
            "hold_regime": str(_hold_regime),
            # v699 Phase O (Task #42): 持久化决策时特征上下文
            # 根因: PerTradeAttributionEngine 需要 features 字典做 5 维归因, 但 v97 detail
            #       无 features → confidence=0.104, 98.4% unexplained → 归因"空转"
            # 修复: decide() 返回 features 子集, evolution_loop 写入 trade.features,
            #       loop_end 持久化时保留 → 归因引擎获得 rich data_quality
            # 字段构成: indicators 数值子集 + 决策上下文 + data_packet 高级特征
            # 铁律: "复盘不是摆设" — 无特征 = 无归因 = 无迭代闭环
            "features": {
                **{k: float(v) for k, v in (indicators or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)},
                "signal_score": float(signal_score),
                "signal_direction": str(signal_direction) if signal_direction else "neutral",
                "market_state": (market_state.get("regime", "") if isinstance(market_state, dict) else ""),
                "atr_abs": float(atr_val),
                "atr_pct": (float(atr_val / close) if close > 0 else 0.0),
                "leverage": int(leverage),
                "chanlun_pa_pattern": str((chanlun_pa or {}).get("pattern", "")),
                "chanlun_pa_strength": float((chanlun_pa or {}).get("strength", 0.0)),
                "regime_label": str((regime_data or {}).get("regime", "")),
            },
        }

    def set_market_context(self, context):
        """Phase 14.16r: 设置市场上下文(资金费率/OI/情绪等)"""
        self._current_market_context = context
