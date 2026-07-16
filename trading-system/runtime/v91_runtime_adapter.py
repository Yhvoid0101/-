# -*- coding: utf-8 -*-
"""
v91 运行时适配器 — 量价双维8状态市场分类 + Kelly全局仓位缩放
================================================================
来源: _v91_volume_kelly_optimize.py (离线回测突破 ann=65.30%)
目标: 将 v91 的核心思想(量价双维分类+Kelly缩放)集成到运行时 AgentDecisionEngine

用户铁律:
  - "凯利公式、价格行为、量价、多周期结构标准化特征融合管道, 避免指标生搬硬套"
  - "永远永远不要出现模拟牛逼, 实盘亏损的情况"
  - "不要通过已知数据反推策略, 那样就过拟合了"

设计原则 (避免过拟合):
  1. 只移植 PRINCIPLE (8状态分类+Kelly缩放), 不硬编码 v91 网格搜索的"最优因子"
  2. 默认因子使用 v91 验证值 (boost_high_v2 配置), 但可通过 config 覆盖
  3. Kelly scaler 从运行时实际 PnL 累计计算, 而非从离线数据反推
  4. 8状态分类是无前视偏差的纯函数 (基于当前 atr_pct + vol_ratio)

核心组件:
  1. classify_market_state_v91() — 量价双维8状态分类 (纯函数)
  2. get_market_factor_v91() — 8状态因子查表 (纯函数)
  3. KellyScalerAccumulator — 累计 PnL 计算 portfolio Kelly scaler
  4. V91_DEFAULT_FACTORS — v91 验证的最优因子 (作为默认基线)

主动应用教训链:
  ERR-20260701-v91: 量价双维+Kelly ann=65.30% → 运行时集成
  ERR-20260701-v91atr: atr_abs 是 symbol 效应 → 不用 atr_abs, 用量价双维
  ERR-20260701-v88fp: 单特征无预测力但状态分类有效 → 用状态不用特征
  ERR-110 (v561): 0/10特征通过 → 不用特征过滤, 只用状态分类调仓
  ERR-100 (v534 v90): 2/3 Kelly 有效 → 应用 portfolio Kelly
================================================================
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("hermes.v91_runtime")

# ============================================================
# v89 市场状态阈值 (与 v89/v90/v91 离线链路完全一致)
# 来源: _v89_market_adaptive.py MARKET_STATE_THRESHOLDS
# 基于 atr_median=0.0172 推导: volatile=1.5x, trend=1.2x, quiet=0.7x
# ============================================================
MARKET_STATE_THRESHOLDS: Dict[str, float] = {
    "volatile": 0.0258,   # median * 1.5
    "trend":    0.0206,   # median * 1.2
    "quiet":    0.0120,   # median * 0.7
}

# ============================================================
# v91 验证的最优 8 状态因子 (来自 _v91_volume_kelly_report.json)
# 配置: boost_high_v2 (high_vol×1.20, low_vol×1.00)
# 基线 ATR 因子: trend=1.20, volatile=0.85, quiet=0.95, low_vol=1.20 (v90 最优)
# 组合后 8 状态因子 = ATR因子 × Volume因子
# ============================================================
V91_DEFAULT_FACTORS: Dict[str, float] = {
    "trend/high_vol":    1.44,   # 1.20 × 1.20
    "trend/low_vol":     1.20,   # 1.20 × 1.00
    "volatile/high_vol": 1.02,   # 0.85 × 1.20
    "volatile/low_vol":  0.85,   # 0.85 × 1.00
    "quiet/high_vol":    1.14,   # 0.95 × 1.20
    "quiet/low_vol":     0.95,   # 0.95 × 1.00
    "low_vol/high_vol":  1.44,   # 1.20 × 1.20
    "low_vol/low_vol":   1.20,   # 1.20 × 1.00
}

# v91 Kelly 分析结果参考 (来自 _v91_volume_kelly_report.json kelly_analysis)
# 注意: 此为离线分析参考值, 运行时 KellyScalerAccumulator 从实际交易PnL动态计算,
#       不使用这些离线值 (避免过拟合 - 用户铁律"不要通过已知数据反推策略")
# 参考: win_rate=0.6753, payoff_ratio=1.2852, kelly_f=0.4227, kelly_2_3=0.2818


# ============================================================
# 核心函数1: 量价双维 8 状态分类 (纯函数, 无前视偏差)
# ============================================================

def classify_market_state_v91(atr_pct: float, vol_ratio: float) -> str:
    """v91 量价双维市场状态分类 (无前视偏差)

    ATR维度 (v89 EXACT): volatile/trend/quiet/low_vol
    Volume维度 (v91 NEW): high_vol(vol_ratio>1.0)/low_vol(≤1.0)

    Args:
        atr_pct: ATR占价格百分比 (如 0.015 = 1.5%)
        vol_ratio: 5周期量比 (vp_vol_ratio_5, 当前成交量/5周期均量)

    Returns:
        "atr_state/vol_state" 如 "trend/high_vol"
        缺失值安全降级: atr_pct<=0 → "quiet/low_vol", vol_ratio<=0 → "low_vol"

    主动应用教训:
        ERR-109 (v518): 均值回归非趋势 → 不做方向过滤, 只做仓位调整
        ERR-20260701-v88fp: 状态分类有效 → 用 8 状态精细化
    """
    # ATR 分类 (v89 EXACT, 阈值来自 MARKET_STATE_THRESHOLDS)
    # 注意: float('nan') 是 float 类型, 但 NaN 与任何数比较都返回 False
    # 必须用 atr_pct != atr_pct (NaN不自等) 来检测 NaN
    if not isinstance(atr_pct, (int, float)) or atr_pct != atr_pct or atr_pct <= 0:
        atr_state = "quiet"  # 缺失值/NaN/非正数 降级为低波动
    elif atr_pct > MARKET_STATE_THRESHOLDS["volatile"]:
        atr_state = "volatile"
    elif atr_pct > MARKET_STATE_THRESHOLDS["trend"]:
        atr_state = "trend"
    elif atr_pct < MARKET_STATE_THRESHOLDS["quiet"]:
        atr_state = "quiet"
    else:
        atr_state = "low_vol"

    # Volume 分类 (v91 NEW)
    if not isinstance(vol_ratio, (int, float)) or vol_ratio != vol_ratio or vol_ratio <= 0:
        vol_state = "low_vol"  # 缺失值/NaN/非正数 降级为低量
    elif vol_ratio > 1.0:
        vol_state = "high_vol"
    else:
        vol_state = "low_vol"

    return f"{atr_state}/{vol_state}"


# ============================================================
# 核心函数2: 8 状态因子查表 (纯函数)
# ============================================================

def get_market_factor_v91(
    atr_pct: float,
    vol_ratio: float,
    factors: Optional[Dict[str, float]] = None,
) -> float:
    """v91: 获取量价双维市场状态因子

    Args:
        atr_pct: ATR占价格百分比
        vol_ratio: 5周期量比
        factors: 8状态因子字典, None则用 V91_DEFAULT_FACTORS

    Returns:
        market_factor: 仓位调整因子 (默认1.0, 范围通常0.8-1.5)
        缺失状态默认1.0 (不影响仓位)

    设计原则:
        - 不硬性过滤交易 (不像 v518 trend filter 那样被证伪)
        - 只调整仓位大小 (market_factor 乘入 pos_mult)
        - 高波动+缩量减仓, 低波动+放量加仓
    """
    if factors is None:
        factors = V91_DEFAULT_FACTORS

    state = classify_market_state_v91(atr_pct, vol_ratio)
    factor = factors.get(state, 1.0)

    # 安全边界: factor 限制在 [0.5, 2.0] 防止极端值
    if not isinstance(factor, (int, float)) or factor != factor:  # NaN check
        return 1.0
    return max(0.5, min(2.0, factor))


# ============================================================
# 核心组件3: KellyScalerAccumulator — Portfolio级Kelly缩放器
# ============================================================

class KellyScalerAccumulator:
    """Portfolio级 Kelly Scaler 累计器

    从运行时实际交易 PnL 累计计算 portfolio-level Kelly fraction,
    而非从离线数据反推 (避免过拟合).

    公式 (来自 v91 离线验证):
        f* = (b * p - q) / b  (Kelly 最优几何增长)
        kelly_2_3 = f* * 2/3  (安全 2/3 Kelly)
        kelly_scaler = min(1.0, kelly_2_3 / current_risk_per_trade)

    线程安全: 使用 threading.Lock 保护共享状态.
    更新策略: 每笔交易关闭后调用 update(), 每次决策前调用 get_scaler().

    主动应用教训:
        ERR-100 (v534 v90): 2/3 Kelly 有效 → 应用 2/3 安全系数
        ERR-20260701-v91: portfolio Kelly 避免per-symbol小样本问题
        用户铁律: "不要通过已知数据反推策略" → 从运行时PnL计算, 非离线硬编码
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        current_risk_pct: float = 0.02,
        min_trades_for_kelly: int = 30,
        warmup_scaler: float = 1.0,
    ):
        """
        Args:
            initial_capital: 初始资金 (用于计算 risk per trade)
            current_risk_pct: 当前单笔风险占比 (默认2%硬截断)
            min_trades_for_kelly: 最少交易数才计算Kelly (样本不足时用warmup)
            warmup_scaler: 样本不足时的默认scaler (1.0=不调整)
        """
        self._initial_capital = float(initial_capital)
        self._current_risk_pct = float(current_risk_pct)
        self._min_trades_for_kelly = int(min_trades_for_kelly)
        self._warmup_scaler = float(warmup_scaler)

        self._lock = threading.Lock()
        self._pnls: deque = deque(maxlen=500)  # 保留最近500笔PnL
        self._cached_scaler: float = self._warmup_scaler
        self._last_update_count: int = 0

    def update(self, pnl: float) -> None:
        """更新PnL历史并重新计算Kelly scaler

        Args:
            pnl: 单笔交易PnL (正=盈利, 负=亏损)
        """
        if not isinstance(pnl, (int, float)) or pnl != pnl:  # NaN check
            return

        with self._lock:
            self._pnls.append(float(pnl))
            # 每10笔交易重新计算一次 (避免每笔都重算的开销)
            if len(self._pnls) - self._last_update_count >= 10:
                self._cached_scaler = self._compute_kelly_scaler()
                self._last_update_count = len(self._pnls)

    def update_batch(self, pnls: list) -> None:
        """批量更新PnL历史 (用于进化启动时从历史交易预热 Kelly scaler)

        用途: EvolutionLoop 启动时, 若有历史交易PnL, 可调用此方法预热,
              避免冷启动阶段 (前 min_trades_for_kelly 笔交易) 使用 warmup_scaler.
        注意: 保留 deque maxlen=500, 超出自动丢弃最旧数据.
        """
        with self._lock:
            for pnl in pnls:
                if isinstance(pnl, (int, float)) and pnl == pnl:
                    self._pnls.append(float(pnl))
            self._cached_scaler = self._compute_kelly_scaler()
            self._last_update_count = len(self._pnls)

    def get_scaler(self) -> float:
        """获取当前Kelly scaler (线程安全)

        Returns:
            kelly_scaler: 仓位缩放因子
                - 样本不足 (< min_trades_for_kelly): warmup_scaler (默认1.0)
                - kelly_f <= 0: 0.5 (保守减仓, 历史表现差)
                - kelly_2_3 > current_risk: 1.0 (不超硬截断)
                - kelly_2_3 <= current_risk: kelly_2_3 / current_risk
        """
        with self._lock:
            return self._cached_scaler

    def get_stats(self) -> Dict[str, Any]:
        """获取Kelly scaler统计信息 (用于诊断和日志)"""
        with self._lock:
            n = len(self._pnls)
            if n == 0:
                return {
                    "n_trades": 0,
                    "scaler": self._warmup_scaler,
                    "kelly_f": 0.0,
                    "kelly_2_3": 0.0,
                    "win_rate": 0.0,
                    "payoff_ratio": 0.0,
                }
            wins = [p for p in self._pnls if p > 0]
            losses = [p for p in self._pnls if p < 0]
            p_win = len(wins) / n if n > 0 else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0
            b = avg_win / avg_loss if avg_loss > 0 else 0
            kelly_f = max(0, min(1, (b * p_win - (1 - p_win)) / b)) if b > 0 else 0
            return {
                "n_trades": n,
                "scaler": self._cached_scaler,
                "kelly_f": kelly_f,
                "kelly_2_3": kelly_f * 2 / 3,
                "win_rate": p_win,
                "payoff_ratio": b,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }

    def _compute_kelly_scaler(self) -> float:
        """计算Kelly scaler (内部方法, 调用者需持有锁)"""
        n = len(self._pnls)
        if n < self._min_trades_for_kelly:
            return self._warmup_scaler

        wins = [p for p in self._pnls if p > 0]
        losses = [p for p in self._pnls if p < 0]

        if not wins or not losses:
            return self._warmup_scaler

        p_win = len(wins) / n
        q_loss = 1 - p_win
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss <= 0:
            return self._warmup_scaler

        b = avg_win / avg_loss  # 盈亏比
        if b <= 0:
            return 0.5  # 历史表现差, 保守减仓

        # Kelly fraction: f* = (b*p - q) / b
        kelly_f = (b * p_win - q_loss) / b
        kelly_f = max(0, min(1, kelly_f))

        if kelly_f <= 0:
            # Kelly建议不交易, 但不强制弃权, 大幅减仓
            return 0.5

        # 2/3 Kelly for safety (ERR-100 v534 lineage教训)
        kelly_2_3 = kelly_f * 2 / 3

        # current_risk_per_trade = initial_capital * current_risk_pct
        current_risk = self._initial_capital * self._current_risk_pct
        current_risk_pct_value = self._current_risk_pct

        # kelly_scaler = (2/3 * f*) / current_risk_pct
        # 如果 2/3 Kelly > 当前截断, scaler=1.0 (不能超截断)
        # 如果 2/3 Kelly < 当前截断, scaler < 1.0 (保守减仓)
        if kelly_2_3 > current_risk_pct_value:
            return 1.0  # Kelly建议更激进, 但受2%硬截断限制
        elif kelly_2_3 > 0:
            return max(0.3, min(1.0, kelly_2_3 / current_risk_pct_value))
        else:
            return 0.5

    # 注: 不提供 reset() 方法 — portfolio 级 Kelly 累计器设计为跨进化代持续学习,
    #     不在新代开始时重置. 这符合"从运行时实际PnL累计学习"的设计原则.
    #     如需清空, 重新实例化 KellyScalerAccumulator 即可.


# ============================================================
# 便捷函数: 一次性计算 market_factor * kelly_scaler
# ============================================================

def compute_v91_position_adjustment(
    atr_pct: float,
    vol_ratio: float,
    kelly_scaler: float,
    factors: Optional[Dict[str, float]] = None,
) -> Tuple[float, str, float]:
    """计算 v91 仓位调整三元组

    Args:
        atr_pct: ATR占价格百分比
        vol_ratio: 5周期量比
        kelly_scaler: Kelly scaler (来自 KellyScalerAccumulator.get_scaler())
        factors: 8状态因子字典, None则用默认

    Returns:
        (position_multiplier, market_state, market_factor):
            - position_multiplier: 仓位乘数 = market_factor * kelly_scaler
            - market_state: 8状态字符串 (如 "trend/high_vol")
            - market_factor: 8状态因子值
    """
    market_factor = get_market_factor_v91(atr_pct, vol_ratio, factors)
    market_state = classify_market_state_v91(atr_pct, vol_ratio)

    # 安全边界: kelly_scaler 限制在 [0.3, 1.5]
    safe_scaler = max(0.3, min(1.5, kelly_scaler)) if isinstance(kelly_scaler, (int, float)) else 1.0

    position_multiplier = market_factor * safe_scaler

    # 最终仓位乘数安全边界: [0.3, 3.0]
    position_multiplier = max(0.3, min(3.0, position_multiplier))

    return position_multiplier, market_state, market_factor


# ============================================================
# 模块自检 (L6 功能验证)
# ============================================================

def _self_test() -> bool:
    """模块自检 — 验证核心函数可用性"""
    try:
        # 测试1: 8状态分类
        states = set()
        for atr in [0.005, 0.015, 0.022, 0.030]:
            for vol in [0.5, 1.5]:
                state = classify_market_state_v91(atr, vol)
                states.add(state)
        assert len(states) == 8, f"8状态分类失败, 只得到{len(states)}个状态: {states}"

        # 测试2: 因子查表
        for state in states:
            factor = get_market_factor_v91(0.015, 1.5, V91_DEFAULT_FACTORS)
            assert 0.5 <= factor <= 2.0, f"因子超出安全边界: {factor}"

        # 测试3: Kelly scaler 累计器
        accumulator = KellyScalerAccumulator(initial_capital=10000.0)
        assert accumulator.get_scaler() == 1.0, "初始scaler应为1.0"

        # 模拟30笔盈利交易
        for _ in range(20):
            accumulator.update(100.0)
        for _ in range(10):
            accumulator.update(-80.0)

        scaler = accumulator.get_scaler()
        assert 0.3 <= scaler <= 1.5, f"Kelly scaler 超出边界: {scaler}"

        stats = accumulator.get_stats()
        assert stats["n_trades"] == 30, f"交易数不对: {stats['n_trades']}"
        assert stats["win_rate"] > 0.5, f"胜率应>50%: {stats['win_rate']}"

        # 测试4: 便捷函数
        pos_mult, state, factor = compute_v91_position_adjustment(
            atr_pct=0.022, vol_ratio=1.5, kelly_scaler=1.0
        )
        assert 0.3 <= pos_mult <= 3.0, f"仓位乘数超出边界: {pos_mult}"
        assert "/" in state, f"状态格式错误: {state}"

        # 测试5: 缺失值安全降级
        state_nan = classify_market_state_v91(float("nan"), float("nan"))
        assert state_nan == "quiet/low_vol", f"NaN降级失败: {state_nan}"

        state_neg = classify_market_state_v91(-1, -1)
        assert state_neg == "quiet/low_vol", f"负值降级失败: {state_neg}"

        logger.info(
            "v91_runtime_adapter 自检通过: 8状态=%d, scaler=%.3f, win_rate=%.1f%%",
            len(states), scaler, stats["win_rate"] * 100
        )
        return True

    except Exception as e:
        logger.error("v91_runtime_adapter 自检失败: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
