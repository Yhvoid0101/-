# -*- coding: utf-8 -*-
"""v706 Forward-test Framework 配置模块

用户铁律:
  "在未达成性能指标前不得推进实盘部署工作"
  "永远永远不要出现模拟牛逼, 实盘亏损的情况"
  "phased testing admission rules (simulation → mini small live trading → standard live trading)"

本模块集中管理 v706 Forward-test Framework 的所有配置:
  1. SIM_THRESHOLDS     - 模拟阶段阈值 (年化≥30%, 回撤≤15%, 胜率≥55%, 夏普≥1.5)
  2. LIVE_THRESHOLDS    - 实盘阶段阈值 (mlp≤5%, 单亏≤2%, 连亏≤3次)
  3. SIM_TO_LIVE_BRIDGE - SIM→LIVE 转换规则 (连续3月GO, kelly死锁已消除)
  4. ALERT_TRIGGERS     - 告警触发条件 (kelly_zero_ratio, gap, consec_loss等)
  5. V704_BASELINE      - v704 3年回测冻结基线 (平均年化16.8%, 退化18.0%)
  6. V706_DATA_DIR      - 独立数据目录配置

来源:
  - 用户铁律: "模拟环境年化收益率≥30%, 最大回撤≤15%, 胜率≥55%, 夏普比率≥1.5"
  - staged_admission_gate.py TIER2_THRESHOLDS
  - _v704_FINAL_DELIVERY.md (3年回测冻结基线)
  - tier2_alert_channel.py ALERT_LEVELS (4级告警)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

# ============================================================================
# 路径配置 (独立于 tier2_oos_data, 避免状态污染)
# ============================================================================

HERE = Path(__file__).parent.resolve()
V706_DATA_DIR = HERE / "v706_oos_data"
V706_STATE_FILE = V706_DATA_DIR / "v706_state.json"
V706_KLINES_DIR = V706_DATA_DIR / "klines"
V706_TRADES_DIR = V706_DATA_DIR / "trades"
V706_REPORTS_DIR = V706_DATA_DIR / "reports"
V706_ALERTS_FILE = V706_DATA_DIR / "v706_alerts.json"

# 初始资金 (与 Tier2 一致, $5k 迷你实盘上限)
V706_CAPITAL_USD = 5000.0
V706_INITIAL_CAPITAL_USD = 10000.0  # 回测基线资金

# ============================================================================
# 1. 模拟阶段阈值 (SIM_THRESHOLDS)
# ============================================================================

SIM_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ann_pct": {"min": 30.0, "desc": "年化收益率≥30%"},
    "max_dd_pct": {"max": 15.0, "desc": "最大回撤≤15%"},
    "win_rate_pct": {"min": 55.0, "desc": "胜率≥55%"},
    "sharpe": {"min": 1.5, "desc": "夏普比率≥1.5"},
    "mlp_pct": {"max": 5.0, "desc": "月度亏损概率≤5%"},
    "gap_pct": {"max": 15.0, "desc": "sim/live差异率≤15%"},
    "consec_loss": {"max": 3, "desc": "连续亏损次数≤3"},
    "single_loss_pct": {"max": 2.0, "desc": "单次最大亏损≤2%本金"},
}

# ============================================================================
# 2. 实盘阶段阈值 (LIVE_THRESHOLDS) — 来自 staged_admission_gate.py TIER2_THRESHOLDS
# ============================================================================

LIVE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "consecutive_sim_go_months": {"min": 3, "desc": "连续3月模拟GO"},
    "live_mlp_pct": {"max": 5.0, "desc": "实盘月度亏损概率≤5%"},
    "live_gap_pct": {"max": 15.0, "desc": "实盘差异率≤15%"},
    "live_single_loss_pct": {"max": 2.0, "desc": "单次最大亏损≤2%本金"},
    "live_consec_loss": {"max": 3, "desc": "连续亏损次数≤3"},
    "capital_max_usd": {"max": 5000, "desc": "迷你实盘上限$5k"},
}

# ============================================================================
# 3. SIM → LIVE 转换规则 (SIM_TO_LIVE_BRIDGE)
# ============================================================================

SIM_TO_LIVE_BRIDGE: Dict[str, Any] = {
    # 连续GO月数要求
    "consecutive_sim_go_months": 3,
    # kelly死锁消除验证: kelly=0占比≤10%才允许进入live
    "kelly_zero_max_ratio": 0.10,
    # regime识别可靠性: regime误判率≤30%
    "regime_mismatch_max": 0.30,
    # 统计显著性: OOS最少30笔交易
    "min_oos_trades": 30,
    # bear_signal验证: v704 Fix2 至少触发1次
    "bear_signal_min_count": 1,
    # gap安全边际: 实际gap需远小于15%红线
    "gap_safety_margin_pct": 10.0,
}

# ============================================================================
# 4. 告警触发条件 (ALERT_TRIGGERS)
# ============================================================================

ALERT_TRIGGERS: Dict[str, Any] = {
    # kelly死锁风险 (v704 Fix1 验证)
    "kelly_zero_ratio_warn": 0.05,        # >5% WARN
    "kelly_zero_ratio_critical": 0.15,    # >15% CRITICAL
    # gap过大 (sim/live差异)
    "gap_pct_warn": 10.0,                  # >10% WARN
    "gap_pct_critical": 13.0,             # >13% CRITICAL (接近15%红线)
    # 连续亏损
    "consec_loss_warn": 2,                 # 连亏2次 WARN
    "consec_loss_critical": 3,            # 连亏3次 CRITICAL
    # 单次亏损
    "single_loss_warn_pct": 1.5,           # 单亏>1.5% WARN
    "single_loss_critical_pct": 2.0,       # 单亏>2% CRITICAL
    # 策略退化
    "regime_degradation_pct": 30.0,        # 退化>30% WARN
    # bear_signal静默 (v704 Fix2 验证)
    "bear_signal_zero_count_days": 7,      # 7天0个bear信号 WARN
    # regime分布异常
    "regime_single_state_max_ratio": 0.70,  # 单regime占比>70% WARN
    # v704 vs v98 对比
    "v704_inferior_delta_usd": -50.0,      # v704 live_pnl比v98低$50 WARN
}

# ============================================================================
# 5. v704 3年回测冻结基线 (V704_BASELINE)
# ============================================================================

V704_BASELINE: Dict[str, Any] = {
    "version": "v704",
    "source": "_v704_FINAL_DELIVERY.md",
    "test_period": "3年 (2022-07 to 2025-07)",
    "n_symbols": 10,
    "symbols": [
        "ADA_USDT", "AVAX_USDT", "BNB_USDT", "BTC_USDT", "DOGE_USDT",
        "DOT_USDT", "ETH_USDT", "LINK_USDT", "LTC_USDT", "SOL_USDT",
    ],
    # 性能指标 (3年平均)
    "avg_ann_pct": 16.8,           # 平均年化16.8% (v534原版0.39%, +43倍)
    "avg_sharpe": 3.22,            # 平均夏普3.22
    "avg_pf": 1.92,                # 平均盈亏比1.92
    "avg_win_rate_pct": 58.5,      # 平均胜率58.5%
    # 过拟合检测
    "degradation_pct": 18.0,        # 前1.5年→后1.5年退化18.0% (<50%阈值)
    "kelly_zero_ratio_lt": 0.01,    # kelly=0占比<1% (v534原版75%+)
    # 三个修复
    "fixes": {
        "fix1_kelly_adaptive": "KellyCriterionSizer 市场状态自适应",
        "fix2_bear_short_signal": "BearTrendShortSignal 熊市做空",
        "fix3_regime_namespace": "6状态 regime 命名空间修复",
    },
    # 多窗口稳定性
    "windows": {
        "front_1_5y_ann": 20.4,      # 前1.5年年化20.4%
        "back_1_5y_ann": 15.3,       # 后1.5年年化15.3%
        "full_3y_ann": 16.8,         # 3年完整年化16.8%
    },
}

# ============================================================================
# 6. v706 风险管理参数
# ============================================================================

V706_RISK_PARAMS: Dict[str, Any] = {
    # 单笔最大风险 (2%本金)
    "max_risk_pct": 0.020,
    # 止损最小距离 (0.8%, 防止过紧被噪音触发)
    "min_risk_pct": 0.008,
    # bear_signal 仓位减半 (熊市做空风险更高)
    "bear_signal_position_scale": 0.5,
    # bear_signal 止损距离上限 (5%, 防止异常大止损)
    "bear_signal_max_stop_pct": 0.05,
    # bear_signal 强制 regime 置信度
    "bear_signal_regime_confidence": 0.9,
    # bear_signal 强制 regime 名称
    "bear_signal_regime_override": "trending_down",
    # 亏损冷却期 (4小时, v99 ERR-20260703-back-to-back-bug)
    "loss_cooldown_hours": 4,
    # 连续亏损上限 (v97 冻结参数)
    "max_consec_loss": 3,
    # 单次最大亏损%本金 (v97 冻结参数)
    "max_single_loss_pct": 2.0,
    # 4h K线聚合窗口 (经典海龟20日突破)
    "bear_signal_lookback": 20,
    # 4h趋势分类回看窗口
    "classify_4h_trend_lookback": 50,
    # 预热K线数 (EMA200所需)
    "warmup_klines_n": 300,
    # ATR周期
    "atr_period": 14,
    # TP/SL ATR倍数 (v97 冻结参数)
    "tp_atr_mult": 2.0,
    "sl_atr_mult": 1.5,
    # ATR自适应持仓周期阈值
    "atr_high_volatility": 0.025,    # ATR% > 2.5% → 24h max_hold
    "atr_medium_volatility": 0.015,  # ATR% > 1.5% → 48h max_hold
    "atr_low_volatility_max_hold": 72,   # ATR% ≤ 1.5% → 72h
    # v99 Task #93 修复3: Volatility Targeting (Rob Carver方法, Man Group/AHL 12年实盘)
    # 目标波动率1.5% (与atr_medium_volatility对齐, 非回测优化)
    # 公式: vol_target_mult = min(target_vol / current_vol, 1.5), 下限0.2
    # 高波动时自动减仓(atr=5%→mult=0.3), 低波动时自动加仓(atr=0.5%→mult=1.5)
    "target_vol_pct": 0.015,
    "vol_target_mult_max": 1.5,      # 上限1.5x (避免低波动时过度加仓)
    "vol_target_mult_min": 0.2,      # 下限0.2x (避免高波动时清仓, 保留探针仓位)
    # v99 Task #96: RSI超卖区域做空减仓 (金融理论: Wilder RSI超卖反弹风险)
    # 根因: Task#94发现BTC做空亏损根因是RSI=32.15超卖区域做空→反弹风险高
    # 理论依据: Wilder RSI经典(1978) RSI<30超卖, Connors RSI-2(2014)超卖不做空
    # 设计: RSI<35(超卖边界)做空时kelly_mult×0.5(减仓50%)
    #   - 不完全禁止(保留趋势跟踪可能性, 只是减仓)
    #   - RSI<30的bear_signal路径完全禁止(在_build_bear_signal中处理)
    "rsi_oversold_short_penalty": 0.5,        # RSI<35做空减仓系数
    "rsi_oversold_threshold": 35.0,           # 超卖边界阈值(v97路径减仓)
    "rsi_oversold_bear_block": 30.0,          # 超卖禁止阈值(bear_signal路径禁止)
    # v99 Task #97: 强趋势RSI超买做空过滤 (Task#96对称优化, 金融理论: 强趋势中超买持续)
    # 根因: OOS数据发现ETH 7月2日04:00做空 RSI=73.07(超买!)+ADX=52.47(强趋势)仍然亏损$-0.71
    #   - 经典RSI理论(RSI>70应该做空盈利)在强趋势中失效
    #   - 强趋势中RSI可以长期处于超买区域(趋势延续信号, 非反转信号)
    # 理论依据 (非回测反推, 符合用户铁律):
    #   - Wilder RSI经典(1978): RSI>70超买, 但在强趋势中可持续
    #   - Wilder ADX(1978): ADX>25强趋势, ADX>40极强趋势
    #   - Connors RSI-2(2014): 强趋势中超买/超卖持续, 反向交易失效
    #   - Rob Carver(Man Group/AHL 12年实盘): RSI反转策略在ADX>25时失效
    # 设计: 与Task#96 RSI超卖过滤对称
    #   - v97路径: ADX>25+RSI>70做空减仓50% (保留趋势跟踪可能性)
    #   - bear_signal路径: ADX>25+RSI>75做空完全禁止 (风险更高)
    "rsi_overbought_short_penalty": 0.5,      # 强趋势+RSI超买做空减仓系数
    "rsi_overbought_threshold": 70.0,         # 超买边界阈值(v97路径减仓)
    "rsi_overbought_bear_block": 75.0,        # 深超买禁止阈值(bear_signal路径禁止)
    "adx_strong_trend_threshold": 25.0,       # 强趋势阈值(Wilder 1978标准)
    # v99 P4k: 动态TP让利润奔跑 (强趋势+顺方向时扩大TP, 不改SL)
    # 理论依据 (非回测反推, 符合用户铁律):
    #   - Jesse Livermore "How to Trade in Stocks" (1940): "Cut your losses and let your profits run"
    #   - Wilder ADX (1978): ADX>25强趋势, 趋势延续概率高, 应让利润奔跑
    #   - Turtle Trading (1980s, Richard Dennis): 强趋势中trailing stop让利润奔跑
    #   - Rob Carver "Leveraged Trading" (Man Group/AHL 12年实盘): 趋势跟踪策略在强趋势中应扩大TP
    # 设计: 强趋势(ADX>=25)+顺方向(regime与direction一致)时, tp_atr_mult从2.0提升到3.0
    #   - 只扩大TP不收紧SL (ERR-20260704-p4i-breakeven-fail教训: 收紧止损被噪音止损)
    #   - 与Task#97对称: Task#97在强趋势中减少做空, P4k在强趋势中让做多利润奔跑
    #   - 顺方向定义: direction=+1 + regime=trending_up / direction=-1 + regime=trending_down
    "tp_atr_mult_base": 2.0,                  # 基础TP乘数 (v97默认值, 不修改原tp_atr_mult)
    "tp_atr_mult_strong_trend": 3.0,          # 强趋势TP乘数 (让利润奔跑)
    "p4k_strong_trend_adx_threshold": 25.0,   # P4k强趋势ADX阈值 (与adx_strong_trend_threshold对齐)
}

# ============================================================================
# 7. v706 OOS 配置
# ============================================================================

V706_OOS_CONFIG: Dict[str, Any] = {
    # OOS 开始日期 (与 Tier2 一致)
    "oos_start_date": "2026-07-01",
    # v706 支持的币种 (与 v97 一致, 15个币种)
    "symbols": [
        "BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT",
        "ARB_USDT", "ATOM_USDT", "AVAX_USDT", "DOT_USDT", "INJ_USDT",
        "LINK_USDT", "LTC_USDT", "OP_USDT", "TRX_USDT", "UNI_USDT",
    ],
    # 禁用币种 (与 v97 一致)
    "disabled_symbols": {"NEAR_USDT"},
    # Gate.io API 配置 (复用 tier2)
    "gateio_base_url": "https://api.gateio.ws/api/v4",
    "gateio_perp_kline_url": "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
    # 数据获取超时
    "fetch_timeout": 10.0,
    # 数据获取重试次数
    "fetch_retry": 3,
}


def ensure_v706_directories() -> None:
    """确保 v706 数据目录存在 (Layer 1 存在性验证)

    在 V706ForwardTestRunner.__init__ 中调用, 确保所有数据目录已创建.
    """
    for d in [V706_DATA_DIR, V706_KLINES_DIR, V706_TRADES_DIR, V706_REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_v706_config_summary() -> Dict[str, Any]:
    """获取 v706 配置摘要 (用于状态报告)"""
    return {
        "version": "v706_forward_test",
        "capital_usd": V706_CAPITAL_USD,
        "n_symbols": len(V706_OOS_CONFIG["symbols"]),
        "oos_start_date": V706_OOS_CONFIG["oos_start_date"],
        "v704_baseline_ann": V704_BASELINE["avg_ann_pct"],
        "sim_thresholds_count": len(SIM_THRESHOLDS),
        "live_thresholds_count": len(LIVE_THRESHOLDS),
        "alert_triggers_count": len(ALERT_TRIGGERS),
    }
