"""
InitDecisionMixin — Phase 7.8 架构解耦
从 evolution_loop.py 提取的初始化与决策引擎逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 属性分散保留: 所有 self 属性在 Host __init__ 中初始化, Mixin 仅访问

来源:
  - initialize: 系统初始化 (数据管道+种群+风控+决策引擎+策略注册表)
  - _init_decision_engines: 为每个Agent初始化决策引擎,注入引擎/特征/策略依赖
"""
from __future__ import annotations

import logging
from .gene_codec import AgentGene
from .agent_decision_engine import AgentDecisionEngine

logger = logging.getLogger(__name__)


class InitDecisionMixin:
    """初始化与决策引擎 Mixin (Phase 7.8 提取)

    Host 鸭子类型依赖 (属性分散在 Host __init__):
      - self.pipeline/symbols/evolution_monitor/population/risk_control/engine
      - self._decision_engines/feature_fusion/hold_bars_adapter/regime_detector
      - self.kelly_scaler_acc/evolution_applier/main_symbol/initial_capital/population_size
    """

    def initialize(
        self,
        bars_per_symbol: int = 1000,
        use_mock: bool = False,
        seed_count: int = 0,
        use_intraday: bool = False,
    ) -> None:
        """初始化整个系统

        Phase 7J-5 反温室修复: use_mock 默认改为 False
        之前: use_mock=True (默认使用合成数据, 沙盘进化完全脱离真实市场)
        现在: use_mock=False (默认使用真实数据, 合成数据仅作为降级方案)
        铁律: "一定不要出现模拟牛逼，实盘亏钱！" + "确保所有功能与策略均基于真实市场数据进行检验"
        来源: 反温室审视报告 CRITICAL #3

        Args:
            bars_per_symbol: 每个交易对的历史K线数
            use_mock: 是否使用模拟数据 (默认False, 仅在真实数据不可用时降级)
            seed_count: 种子策略数
            use_intraday: v2.35 是否使用日内交易场景初始化
                False=沿用 initialize_population (全部随机 + 种子注入)
                True=使用 initialize_intraday_population (日内硬约束 + 高胜率世界观配比)
                来源: 500轮进化诊断 — 97% agent 零交易根因
        """
        if use_mock:
            logger.warning(
                "  ⚠️ use_mock=True: 使用合成数据, 进化结果可能与实盘不一致! "
                "(铁律: 一定不要出现模拟牛逼，实盘亏钱！)"
            )
        logger.info("Initializing Evolution Loop (use_mock=%s)...", use_mock)

        # 1. 初始化数据管道
        self.pipeline.initialize(self.symbols, bars_per_symbol, use_mock)
        logger.info("  Data pipeline: %d symbols, %d bars each", len(self.symbols), bars_per_symbol)

        # R12优化: P2-1检查点自动恢复(消除空壳交付)
        # 在initialize_population之前尝试恢复,避免覆盖可恢复的进度
        restored = False
        if self.evolution_monitor:
            try:
                ckpt_data = self.evolution_monitor.try_restore()
                if ckpt_data:
                    # 恢复种群状态
                    self.population.generation = ckpt_data.get("generation", 0)
                    restored_agents = []
                    for agent_data in ckpt_data.get("agents", []):
                        try:
                            gene_data = agent_data.get("gene", {})
                            if isinstance(gene_data, str):
                                try:
                                    import ast
                                    gene_data = ast.literal_eval(gene_data)
                                except (SyntaxError, ValueError):
                                    gene_data = {}
                            gene = AgentGene.from_dict(gene_data) if isinstance(gene_data, dict) else AgentGene()
                            gene.agent_id = agent_data.get("agent_id", gene.agent_id)
                            gene.generation = agent_data.get("generation", gene.generation)
                            score = ckpt_data.get("scores", {}).get(gene.agent_id, {})
                            gene.fitness_score = score.get("gt_score", gene.fitness_score)
                            gene.sharpe_ratio = score.get("sharpe_ratio", gene.sharpe_ratio)
                            gene.win_rate = score.get("win_rate", gene.win_rate)
                            gene.total_trades = score.get("total_trades", gene.total_trades)
                            gene.max_drawdown_realized = score.get("max_drawdown", gene.max_drawdown_realized)
                            restored_agents.append(gene)
                        except Exception as e:
                            # Phase 7J-10 反温室修复 MEDIUM #18: 检查点恢复异常可见化
                            # 之前: continue 静默, 单个 Agent 反序列化失败导致恢复的种群不完整且无告警
                            # 现在: warning + continue (反温室: 种群不完整=进化结果偏差)
                            logger.warning(
                                "P2-1 检查点恢复 Agent 异常 (种群可能不完整!): %s, data=%s",
                                str(e)[:100], str(agent_data)[:100],
                            )
                            continue
                    if restored_agents:
                        self.population.population = restored_agents
                        restored = True
                        logger.info(
                            "  P2-1: 从检查点恢复 gen=%d agents=%d",
                            self.population.generation, len(restored_agents),
                        )
            except Exception as e:
                logger.warning("P2-1: 检查点恢复失败: %s", e)

        # 2. 初始化种群(若未从检查点恢复)
        if not restored:
            if use_intraday:
                # v2.35: 日内交易场景初始化 (高胜率世界观配比 + 日内硬约束)
                self.population.initialize_intraday_population()
                logger.info("  Population (INTRADAY): %d agents", len(self.population.population))
            else:
                self.population.initialize_population(seed_count=seed_count)
                logger.info("  Population: %d agents", len(self.population.population))
        else:
            logger.info("  Population: %d agents (restored from checkpoint, skipped init)",
                        len(self.population.population))

        # 3. 初始化决策引擎
        self._init_decision_engines()

        # 4. 初始化风控（B3修复：沙盘使用生产级阈值，不放宽）
        # 之前的问题：max_drawdown=50%、max_consecutive_losses=10、max_daily_loss=inf
        # 这些放宽的阈值导致温室效应——沙盘生存的策略在实盘会被风控强平
        # 修复：沙盘阈值=生产阈值，确保沙盘生存的策略也能在实盘生存
        self.risk_control.set_global_limits(
            max_daily_loss=self.initial_capital * self.population_size * 0.05,  # 生产级：5%日亏损
            max_exposure=self.initial_capital * self.population_size * 0.5,
            volatility_threshold=0.08,  # 生产级：8%波动率阈值
            black_swan_threshold=0.10,  # 生产级：10%黑天鹅阈值
        )
        # Phase 10.3: 风控参数根据 _production_mode 自适应
        # 根因: 生产级参数(3次连续亏损/1小时冷却)导致沙盘288次decide只4笔交易
        #   - circuit_cooldown_hours=1 → 沙盘189秒跑完, 1小时冷却=熔断后永不恢复
        #   - max_consecutive_losses=3 → 进化初期随机策略容易连续亏损3次
        # 设计: 沙盘模式(进化探索期)放宽让进化有数据, 生产模式保持生产级(无温室)
        if not self._production_mode:
            self.risk_control.set_agent_limits(
                max_consecutive_losses=8,       # 沙盘: 3→8 允许更多探索
                max_drawdown_pct=0.50,          # Phase 10.3b: 0.25→0.50 沙盘进化探索期
                max_daily_loss_pct=0.25,        # 沙盘: 0.05→0.25 允许更大日亏
                circuit_cooldown_hours=0.01,    # 沙盘: 1→0.01(36秒) 快速恢复
            )
            # Phase 10.3: 设置 sandbox_mode 让 L2策略组熔断也放宽
            self.risk_control._sandbox_mode = True
            logger.info("  Phase 10.3: sandbox risk params (consec=8, dd=25%%, daily=25%%, cooldown=36s)")
        else:
            self.risk_control.set_agent_limits(
                max_consecutive_losses=3,  # 生产级：3次连续亏损
                max_drawdown_pct=0.20,  # 生产级：20%最大回撤
                max_daily_loss_pct=0.05,  # 生产级：5%日亏损
                circuit_cooldown_hours=1,  # 生产级：1小时冷却
            )
            logger.info("  Risk control: configured (production-grade thresholds, no greenhouse)")

        # v2.31修复: 沙盘模式心跳超时调整 — 防止EMERGENCY SHUTDOWN误触发
        # 原bug: risk_control._heartbeat_timeout=30.0硬编码,且heartbeat()从未被调用
        #   → 程序启动30秒后所有pre_trade_check触发HEARTBEAT_TIMEOUT
        #   → Round 30+所有交易被阻止(checked=0),进化500轮但只有前30轮有交易
        # 修复: 沙盘模式设为3600秒(与production_hardening.HEARTBEAT_TIMEOUT_SANDBOX一致)
        # 来源: production_hardening.py第92行HEARTBEAT_TIMEOUT_SANDBOX=3600.0
        # 铁律: 沙盘非实时,30秒超时是生产环境实时交易的标准,不适用于沙盘
        #
        # Phase 7J-6 反温室修复: 增加 production_mode 标志
        # 来源: 反温室铁律 — "沙盘心跳 3600s 放宽, 实盘 30s 严格, 进化出的策略
        # 从未经历过 30s 心跳约束, 实盘部署后可能因心跳超时被误触发 Kill Switch"
        # 修复:
        #   - production_mode=False (默认, 沙盘进化): 3600s (保留原行为)
        #   - production_mode=True (实盘部署): 30s (生产级)
        #   - 最后 N 代 (final_generations_no_reset>0): 也切换到 30s
        #     让精英策略经历过短心跳约束, 进化出能在 30s 心跳下生存的策略
        # v2.35修复: _production_mode必须在if块外定义,确保始终存在
        # 原bug(v2.31): 只在hasattr(risk_control,'_heartbeat_timeout')块内定义
        #   如果risk_control没有_heartbeat_timeout属性,_production_mode未定义
        #   → _apply_final_generations_heartbeat()访问时AttributeError
        # 注: __init__中已默认False,这里仅在risk_control支持时更新(保持向后兼容)
        if hasattr(self.risk_control, '_heartbeat_timeout'):
            # 默认沙盘模式: 3600s
            self.risk_control._heartbeat_timeout = 3600.0
            logger.info("  v2.31: risk_control heartbeat timeout = 3600s (sandbox mode, was 30s)")
            logger.info("  Phase 7J-6: production_mode=False (sandbox), 可通过 set_production_mode(True) 切换到 30s")

        # Phase 6A: 策略注册表健康检查 — 让加载失败可见（消除空壳交付）
        # 之前策略加载失败被静默 fallback 到 GeneDrivenStrategy，进化过程无法察觉
        # 现在 initialize() 末尾调用 preload_all()，列出所有策略类的可加载状态
        # 任何加载失败的策略类都会被显式记录到日志，便于诊断
        try:
            from .strategy_registry import get_registry
            registry = get_registry()
            preload_result = registry.preload_all()
            total = len(preload_result)
            ok_count = sum(1 for v in preload_result.values() if v)
            fail_count = total - ok_count
            logger.info(
                "  Strategy registry preload: %d/%d OK, %d failed",
                ok_count, total, fail_count,
            )
            if fail_count > 0:
                failed_names = [name for name, ok in preload_result.items() if not ok]
                logger.warning(
                    "  Strategy preload failures: %s", ", ".join(failed_names),
                )
                # Phase 6A 封装修复：通过公开接口 record_load_failure 记录失败
                # （避免直接访问 registry._load_failures 私有属性）
                for name in failed_names:
                    registry.record_load_failure(name, "preload_all: 类加载失败")
        except Exception as e:
            logger.warning("  Strategy registry preload skipped: %s", e)

        logger.info("Evolution Loop initialized. Ready.")


    def _init_decision_engines(self) -> None:
        """初始化所有Agent的决策引擎

        Phase 6A: 注入 spot_engine 引用，修复 deps 注入缺陷
        之前仅传 {"symbol": symbol}，导致 funding_rate_arb/basis_arbitrage 等
        需要 engine 依赖的策略类静默 fallback 到 GeneDrivenStrategy（空壳交付）。
        现在通过 set_engines() 注入 SandboxMatchingEngine，让这些策略类
        真正实例化并参与决策。

        R19-5 WARN-2修复: 在set_engines()后调用prewarm_strategy()预热策略
        之前: 首次 decide() 调用时才加载策略(懒加载), 首次决策延迟50-200ms
        现在: 初始化时主动加载策略, 首次 decide() 直接使用已加载实例
        实盘后果: 首笔交易信号过期 → 滑点增大 → 模拟vs实盘差异扩大
        """
        self._decision_engines.clear()
        # R19-5: 获取默认symbol用于策略预热(从主symbol派生)
        prewarm_symbol = getattr(self, "main_symbol", "BTC-USDT")
        for agent in self.population.population:
            engine = AgentDecisionEngine(agent)
            # 注入沙盘引擎引用（spot_engine）；沙盘模式不区分 spot/perp，共用同一引擎
            engine.set_engines(spot_engine=self.engine, perp_engine=self.engine)  # P3: 沙盘模式spot/perp共用同一引擎
            # Phase 3.1: 注入 FeatureFusionPipeline (短板#2 补齐)
            # v598 Phase D: decide() 中曾用 composite_score 作为 signal_score 增强因子 (0.7-1.3)
            #   但 ERR-20260701-v88fp 证实 composite_score 与 PnL 负相关 (d=-0.240)
            #   矛盾状态: loop_end 已移除 composite_score 决策逻辑 (v92 Phase 2), decide() 仍引用
            #   Phase D 解决: 注入 HoldBarsAdapter (hold_bars d=+0.461 替代 composite_score)
            #   decide() 实际集成留待 Phase L (v97 动态持仓) 完整落地
            # ERR-110: 特征无预测力但融合管道的信号增强价值 ≠ 加权决策
            if self.feature_fusion is not None:
                engine.set_feature_fusion(self.feature_fusion)
            # v598 Phase D: 注入 HoldBarsAdapter (composite_score 矛盾统一治理)
            # ERR-20260701-v88fp: composite_score 与 PnL 负相关 (d=-0.240) → 用 hold_bars (d=+0.461) 替代
            # ERR-109: 仅增强非过滤 (策略本质是均值回归, 逆势trades实际盈利更高)
            # ERR-110: 仅信号增强非加权决策 (0/10特征通过 P<0.05, 但融合管道信号增强价值≠加权)
            if self.hold_bars_adapter is not None:
                engine.set_hold_bars_adapter(self.hold_bars_adapter)
            # Phase 3.2: 注入 RegimeDetectionSystem (短板#6 补齐)
            # decide() 中只读 get_regime() 获取HMM/GARCH/变点综合状态,
            # 根据均值回归策略特性调整仓位倍数 (ERR-109: sideways优势, trend劣势)
            # 关键: 不过滤逆势trades, 仅调整仓位倍数
            if self.regime_detector is not None:
                engine.set_regime_detector(self.regime_detector)
            # v598 Phase 2: 注入 portfolio级 KellyScalerAccumulator (v91量价双维+Kelly缩放)
            # 来源: _v91_volume_kelly_optimize.py 离线突破 ann=65.30%
            # 共享累计器: 所有Agent共享同一Kelly scaler, 基于portfolio历史PnL计算
            if self.kelly_scaler_acc is not None:
                engine.set_kelly_scaler_accumulator(self.kelly_scaler_acc)
            # L7修复: 注入 v91_factors 覆盖路径 (支持从config或基因编码覆盖默认因子)
            # 设计原则落地: "可通过config覆盖" 不再是纸面声明, 现在有实际注入路径
            engine.set_v91_factors(getattr(self, "_v91_factors_config", None))
            # v699: 注入 5维进化建议 overlay (logic_patches + strategy_patches)
            # 闭环: loop_end 归因→应用overlay → _init_decision_engines 注入 → decide() 消费
            # 安全: evolution_applier 不可用时跳过, decide() 中 patches 为空不影响决策
            if self.evolution_applier is not None:
                try:
                    _logic_patches = self.evolution_applier.get_active_patches(dimension="logic")
                    _strategy_patches = self.evolution_applier.get_active_patches(dimension="strategy")
                    engine.set_evolution_overlay(
                        logic_patches=_logic_patches,
                        strategy_patches=_strategy_patches,
                    )
                except Exception as _e:
                    logger.debug("[EvolutionOverlay] 注入失败(降级跳过): %s", _e)

            # v699: 注入 indicator_patches 到共享 FeatureFusionPipeline
            # 闭环: 归因引擎产出 indicator 建议 → applier 写入 overlay → 此处注入 → _fuse_features 消费
            # 设计: pipeline 是单例(self.feature_fusion), 所有 agent 共享, 仅需注入一次
            # 安全: applier/pipeline 任一不可用时跳过, _fuse_features 中 _indicator_patches=[] 不影响融合
            if self.evolution_applier is not None and self.feature_fusion is not None:
                try:
                    _indicator_patches = self.evolution_applier.get_active_patches(dimension="indicator")
                    self.feature_fusion.set_indicator_patches(patches=_indicator_patches)
                except Exception as _e:
                    logger.debug("[IndicatorOverlay] 注入失败(降级跳过): %s", _e)
            # R19-5 WARN-2: 策略预热,避免首次decide()被策略加载阻塞
            engine.prewarm_strategy(symbol=prewarm_symbol)
            self._decision_engines[agent.agent_id] = engine

        # Phase 2 P1: L2策略组级熔断接入 — 按 worldview 分组
        if self.risk_control is not None:
            worldview_groups = {}
            for agent in self.population.population:
                wv = getattr(agent, 'worldview_primary', 'unknown')
                if wv not in worldview_groups:
                    worldview_groups[wv] = []
                worldview_groups[wv].append(agent.agent_id)
            for group_name, agent_ids in worldview_groups.items():
                self.risk_control.register_strategy_group(group_name, agent_ids)
            logger.info("L2策略组级熔断已注册: %s", {k: len(v) for k, v in worldview_groups.items()})


