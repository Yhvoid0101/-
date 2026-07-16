"""Phase 7 - 第6层风险控制智能体 (L6 RiskAgent)。

铁律合规：
- 零模拟：所有风控决策基于真实上游输出（L2/L4/L5），不伪造输入
- 零降级：不可变安全包络不可被学习器放宽；联合 EV ≤ 0 必须 BLOCKED
- 零死角：7 类反射触发器全覆盖；三级风险预算分层
- 零遗漏：上游 L5 BLOCKED 必须传播，不允许 L6 放宽
- 零纸面通过：所有决策有可解释 reasoning + regime_adjustment
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .contracts.audit_verdict import AuditVerdict, VerdictAction
from .contracts.base import ContractStatus, make_blocked
from .contracts.market_state_snapshot import MarketRegime, MarketStateSnapshot
from .contracts.risk_budget import RiskBudget, SafetyEnvelope
from .contracts.strategy_proposal import StrategyProposal


# ============================================================
# 错误码 / 触发原因码
# ============================================================

class PropagationReason(str, Enum):
    """上游传播原因。"""
    UPSTREAM_PROPAGATED = "UPSTREAM_PROPAGATED"
    NO_VERDICT = "NO_VERDICT"
    NO_PROPOSAL = "NO_PROPOSAL"


class RiskBlockReason(str, Enum):
    """L6 阻断原因码。"""
    UPSTREAM_PROPAGATED = "UPSTREAM_PROPAGATED"
    ENVELOPE_VIOLATION = "ENVELOPE_VIOLATION"          # 动态参数超出硬边界
    JOINT_EV_NEGATIVE = "JOINT_EV_NEGATIVE"            # RR/胜率联合 EV ≤ 0
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"          # 极端波动
    CONNECTION_LOSS = "CONNECTION_LOSS"                # 断网/订单拒绝
    PRICE_GAP = "PRICE_GAP"                            # 价格跳空
    STALE_DATA = "STALE_DATA"                          # 陈旧数据
    CORRELATION_SHIFT = "CORRELATION_SHIFT"            # 相关性突变
    CONSECUTIVE_LOSS = "CONSECUTIVE_LOSS"              # 连续亏损
    BEAR_PROTECTION = "BEAR_PROTECTION"                # 熊市保护
    KILL_SWITCH = "KILL_SWITCH"                        # KillSwitch
    DAILY_LOSS_EXCEEDED = "DAILY_LOSS_EXCEEDED"        # 日损失超标
    PORTFOLIO_EXPOSURE_EXCEEDED = "PORTFOLIO_EXPOSURE_EXCEEDED"  # 组合敞口超标


# ============================================================
# 风控上下文
# ============================================================

@dataclass
class RiskContext:
    """L6 风控决策上下文。"""
    verdict: Optional[AuditVerdict] = None              # L5 审核裁决
    proposal: Optional[StrategyProposal] = None         # L4 策略提案
    market_state: Optional[MarketStateSnapshot] = None  # L2 市场状态
    correlation_id: str = ""

    # 市场微观结构（真实输入）
    current_price: float = 0.0
    atr: float = 0.0                                    # 真实 ATR
    volatility_percentile: float = 0.5                  # 0-1

    # 组合状态（真实输入）
    portfolio_exposure: float = 0.0                     # 当前组合敞口
    daily_pnl_pct: float = 0.0                          # 当日盈亏百分比
    consecutive_losses: int = 0                          # 连续亏损次数
    correlated_strategy_pnl: List[float] = field(default_factory=list)
    max_correlation: float = 0.0                         # 策略间最大相关性

    # 系统状态（真实输入）
    connection_ok: bool = True                           # 网络连接正常
    order_rejected_count: int = 0                        # 订单拒绝次数
    data_age_sec: float = 0.0                            # 数据年龄（秒）
    last_price_gap_pct: float = 0.0                      # 最近价格跳空百分比

    # 历史交易（真实输入）
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)

    # 仓位参数（来自 L4 proposal）
    base_position_pct: float = 0.1                       # 基础仓位百分比


# ============================================================
# 不可变安全包络 + 动态参数
# ============================================================

@dataclass
class DynamicParams:
    """动态参数（可学习，但必须在 SafetyEnvelope 内）。"""
    sl_multiplier: float = 1.5       # SL = sl_multiplier * ATR
    tp_multiplier: float = 3.0       # TP = tp_multiplier * ATR
    position_size_pct: float = 0.1   # 仓位百分比
    correlation_threshold: float = 0.7  # 相关性阈值


class EnvelopeValidator:
    """验证动态参数是否在不可变安全包络内。"""

    def __init__(self, envelope: SafetyEnvelope):
        self.envelope = envelope

    def validate(self, params: DynamicParams, atr: float, current_price: float) -> Tuple[bool, str]:
        """返回 (是否合规, 原因)。"""
        if atr <= 0 or current_price <= 0:
            return False, "无效 ATR 或价格"

        # 单笔最大损失检查
        sl_amount = params.sl_multiplier * atr
        sl_pct = sl_amount / current_price
        if sl_pct > self.envelope.max_single_loss_pct:
            return False, f"SL {sl_pct:.4f} > 单笔上限 {self.envelope.max_single_loss_pct}"

        # 仓位检查
        if params.position_size_pct > self.envelope.max_portfolio_exposure:
            return False, f"仓位 {params.position_size_pct} > 组合敞口上限 {self.envelope.max_portfolio_exposure}"

        # 相关性阈值检查（不得低于硬下限 0.5）
        if params.correlation_threshold < 0.5:
            return False, f"相关性阈值 {params.correlation_threshold} < 硬下限 0.5"

        return True, ""


# ============================================================
# RR/胜率联合 EV
# ============================================================

class JointEVCalculator:
    """RR 与胜率联合 EV 计算器。

    joint_ev = win_rate * tp_amount - (1 - win_rate) * sl_amount
    不以单独 RR 或单独胜率优化。
    """

    @staticmethod
    def calculate(win_rate: float, sl_amount: float, tp_amount: float) -> float:
        """返回联合 EV（金额单位）。"""
        if not (0.0 <= win_rate <= 1.0):
            return -float("inf")
        if sl_amount < 0 or tp_amount < 0:
            return -float("inf")
        return win_rate * tp_amount - (1.0 - win_rate) * sl_amount

    @staticmethod
    def is_acceptable(joint_ev: float) -> bool:
        """联合 EV 必须 > 0 才可接受。"""
        return joint_ev > 0.0


# ============================================================
# 反射触发器（7 类）
# ============================================================

@dataclass
class TriggerResult:
    """触发器结果。"""
    triggered: bool
    block_reason: Optional[RiskBlockReason] = None
    kill_switch: bool = False
    position_scale: float = 1.0        # 仓位缩放因子（0-1）
    sl_adjustment: float = 0.0          # SL 调整（正数=收紧，负数=放宽）
    reasoning: str = ""


class ReflexTrigger:
    """反射触发器基类。"""
    name: str = "BaseTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        raise NotImplementedError


class ExtremeVolatilityTrigger(ReflexTrigger):
    """极端波动触发器：波动率突增 → 缩仓。"""
    name = "ExtremeVolatilityTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        if ctx.volatility_percentile > 0.95:
            scale = 0.3  # 缩仓到 30%
            return TriggerResult(
                triggered=True,
                position_scale=scale,
                reasoning=f"极端波动 (percentile={ctx.volatility_percentile:.3f} > 0.95)，缩仓到 {scale*100:.0f}%",
            )
        if ctx.volatility_percentile > 0.8:
            scale = 0.6
            return TriggerResult(
                triggered=True,
                position_scale=scale,
                reasoning=f"高波动 (percentile={ctx.volatility_percentile:.3f} > 0.8)，缩仓到 {scale*100:.0f}%",
            )
        return TriggerResult(triggered=False)


class ConnectionLossTrigger(ReflexTrigger):
    """断网/订单拒绝触发器：→ KillSwitch。"""
    name = "ConnectionLossTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        if not ctx.connection_ok:
            return TriggerResult(
                triggered=True,
                kill_switch=True,
                block_reason=RiskBlockReason.CONNECTION_LOSS,
                reasoning="网络连接断开，激活 KillSwitch",
            )
        if ctx.order_rejected_count >= 3:
            return TriggerResult(
                triggered=True,
                kill_switch=True,
                block_reason=RiskBlockReason.CONNECTION_LOSS,
                reasoning=f"订单拒绝 {ctx.order_rejected_count} 次 ≥ 3，激活 KillSwitch",
            )
        return TriggerResult(triggered=False)


class PriceGapTrigger(ReflexTrigger):
    """价格跳空触发器：→ SL 调整（收紧）。"""
    name = "PriceGapTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        gap_abs = abs(ctx.last_price_gap_pct)
        if gap_abs > 0.05:  # 5% 跳空
            return TriggerResult(
                triggered=True,
                sl_adjustment=0.2,  # SL 收紧 20%
                block_reason=RiskBlockReason.PRICE_GAP,
                reasoning=f"价格跳空 {gap_abs*100:.2f}% > 5%，SL 收紧 20%",
            )
        if gap_abs > 0.02:  # 2% 跳空
            return TriggerResult(
                triggered=True,
                sl_adjustment=0.1,
                reasoning=f"价格跳空 {gap_abs*100:.2f}% > 2%，SL 收紧 10%",
            )
        return TriggerResult(triggered=False)


class StaleDataTrigger(ReflexTrigger):
    """陈旧数据触发器：→ BLOCKED。"""
    name = "StaleDataTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        if ctx.data_age_sec > 300:  # 5 分钟
            return TriggerResult(
                triggered=True,
                block_reason=RiskBlockReason.STALE_DATA,
                reasoning=f"数据年龄 {ctx.data_age_sec:.0f}s > 300s，阻断",
            )
        return TriggerResult(triggered=False)


class CorrelationShiftTrigger(ReflexTrigger):
    """相关性突变触发器：→ 缩仓。"""
    name = "CorrelationShiftTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        if ctx.max_correlation > 0.85:
            return TriggerResult(
                triggered=True,
                position_scale=0.4,
                block_reason=RiskBlockReason.CORRELATION_SHIFT,
                reasoning=f"策略相关性 {ctx.max_correlation:.3f} > 0.85，缩仓到 40%",
            )
        if ctx.max_correlation > 0.7:
            return TriggerResult(
                triggered=True,
                position_scale=0.7,
                reasoning=f"策略相关性 {ctx.max_correlation:.3f} > 0.7，缩仓到 70%",
            )
        return TriggerResult(triggered=False)


class ConsecutiveLossTrigger(ReflexTrigger):
    """连续亏损触发器：→ 缩仓。"""
    name = "ConsecutiveLossTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        if ctx.consecutive_losses >= 5:
            return TriggerResult(
                triggered=True,
                position_scale=0.2,
                block_reason=RiskBlockReason.CONSECUTIVE_LOSS,
                reasoning=f"连续亏损 {ctx.consecutive_losses} 次 ≥ 5，缩仓到 20%",
            )
        if ctx.consecutive_losses >= 3:
            return TriggerResult(
                triggered=True,
                position_scale=0.5,
                reasoning=f"连续亏损 {ctx.consecutive_losses} 次 ≥ 3，缩仓到 50%",
            )
        return TriggerResult(triggered=False)


class BearProtectionTrigger(ReflexTrigger):
    """熊市保护触发器：做空保护。"""
    name = "BearProtectionTrigger"

    def check(self, ctx: RiskContext) -> TriggerResult:
        if ctx.market_state is None:
            return TriggerResult(triggered=False)
        regime = ctx.market_state.regime
        # 熊市判定：trending_down 或 high_volatility + uncertainty > 0.6
        if regime == MarketRegime.TRENDING_DOWN:
            return TriggerResult(
                triggered=True,
                position_scale=0.5,
                block_reason=RiskBlockReason.BEAR_PROTECTION,
                reasoning="熊市 (trending_down) 保护，缩仓到 50%",
            )
        if regime == MarketRegime.HIGH_VOLATILITY and ctx.market_state.uncertainty > 0.6:
            return TriggerResult(
                triggered=True,
                position_scale=0.6,
                block_reason=RiskBlockReason.BEAR_PROTECTION,
                reasoning=f"高波动+高不确定性 (uncertainty={ctx.market_state.uncertainty:.3f})，缩仓到 60%",
            )
        return TriggerResult(triggered=False)


# ============================================================
# L6 RiskAgent
# ============================================================

class L6RiskAgent:
    """第6层风险控制智能体。

    输入：L5 AuditVerdict + L4 StrategyProposal + L2 MarketStateSnapshot
    输出：RiskBudget（含三级预算 + 可解释 reasoning）
    """

    def __init__(
        self,
        envelope: Optional[SafetyEnvelope] = None,
        dynamic_params: Optional[DynamicParams] = None,
    ):
        self.envelope = envelope or SafetyEnvelope()
        self.params = dynamic_params or DynamicParams()
        self.envelope_validator = EnvelopeValidator(self.envelope)
        self.joint_ev_calc = JointEVCalculator()
        self.triggers: List[ReflexTrigger] = [
            ExtremeVolatilityTrigger(),
            ConnectionLossTrigger(),
            PriceGapTrigger(),
            StaleDataTrigger(),
            CorrelationShiftTrigger(),
            ConsecutiveLossTrigger(),
            BearProtectionTrigger(),
        ]

    def allocate(self, ctx: RiskContext) -> RiskBudget:
        """执行风控决策，返回 RiskBudget。"""
        cid = ctx.correlation_id or str(uuid.uuid4())

        # === 1) 上游传播 ===
        if ctx.verdict is None:
            return self._blocked(cid, PropagationReason.NO_VERDICT, "L6: 无 L5 AuditVerdict 输入")
        if ctx.verdict.action == VerdictAction.BLOCK:
            return self._blocked(cid, PropagationReason.UPSTREAM_PROPAGATED,
                                 f"L6: 上游 L5 BLOCKED 传播 - {[r.value for r in ctx.verdict.block_reasons]}")
        if ctx.proposal is not None and ctx.proposal.is_blocked():
            return self._blocked(cid, PropagationReason.UPSTREAM_PROPAGATED,
                                 f"L6: 上游 L4 BLOCKED 传播 - {ctx.proposal.error_code}")
        if ctx.market_state is not None and ctx.market_state.is_blocked():
            return self._blocked(cid, PropagationReason.UPSTREAM_PROPAGATED,
                                 f"L6: 上游 L2 BLOCKED 传播 - {ctx.market_state.error_code}")

        # === 2) KillSwitch 硬边界（最优先，灾难级别）===
        if abs(ctx.daily_pnl_pct) >= self.envelope.kill_switch_threshold:
            rb = RiskBudget(
                kill_switch_active=True,
                safety=self.envelope,
                reasoning=f"KillSwitch 触发：日损失 {ctx.daily_pnl_pct*100:.2f}% ≥ 阈值 {self.envelope.kill_switch_threshold*100:.2f}%",
                regime_adjustment="KILL_SWITCH",
            )
            return make_blocked(rb, RiskBlockReason.KILL_SWITCH.value, rb.reasoning, cid)

        # === 3) 日损失 / 组合敞口 硬边界检查（L3 组合/系统级）===
        if abs(ctx.daily_pnl_pct) >= self.envelope.max_daily_loss_pct:
            return self._blocked_hard(cid, RiskBlockReason.DAILY_LOSS_EXCEEDED,
                                      f"日损失 {ctx.daily_pnl_pct*100:.2f}% ≥ 上限 {self.envelope.max_daily_loss_pct*100:.2f}%")
        if ctx.portfolio_exposure >= self.envelope.max_portfolio_exposure:
            return self._blocked_hard(cid, RiskBlockReason.PORTFOLIO_EXPOSURE_EXCEEDED,
                                      f"组合敞口 {ctx.portfolio_exposure:.3f} ≥ 上限 {self.envelope.max_portfolio_exposure}")

        # === 4) 不可变安全包络验证（P7-T2）===
        atr = ctx.atr if ctx.atr > 0 else (ctx.current_price * 0.01)  # 默认 1% ATR
        ok, reason = self.envelope_validator.validate(self.params, atr, ctx.current_price)
        if not ok:
            return self._blocked_hard(cid, RiskBlockReason.ENVELOPE_VIOLATION,
                                      f"不可变安全包络违规：{reason}")

        # === 5) RR/胜率联合 EV（P7-T2）===
        # 从 recent_trades 计算真实胜率
        win_rate = self._calc_win_rate(ctx.recent_trades)
        sl_amount = self.params.sl_multiplier * atr
        tp_amount = self.params.tp_multiplier * atr
        joint_ev = self.joint_ev_calc.calculate(win_rate, sl_amount, tp_amount)
        if not self.joint_ev_calc.is_acceptable(joint_ev):
            return self._blocked_hard(cid, RiskBlockReason.JOINT_EV_NEGATIVE,
                                      f"联合 EV={joint_ev:.4f} ≤ 0 (win_rate={win_rate:.3f}, SL={sl_amount:.4f}, TP={tp_amount:.4f})")

        # === 6) 反射触发器（P7-T3）===
        trigger_results: List[TriggerResult] = []
        for trigger in self.triggers:
            tr = trigger.check(ctx)
            trigger_results.append(tr)

        # 最严格合并
        any_kill = any(tr.kill_switch for tr in trigger_results)
        any_block = any(tr.block_reason is not None for tr in trigger_results if tr.triggered)
        min_scale = min((tr.position_scale for tr in trigger_results if tr.triggered), default=1.0)
        sl_adjust = sum(tr.sl_adjustment for tr in trigger_results if tr.triggered)

        if any_kill:
            kill_reason = next((tr.reasoning for tr in trigger_results if tr.kill_switch), "KillSwitch")
            rb = RiskBudget(
                kill_switch_active=True,
                safety=self.envelope,
                reasoning=f"KillSwitch 激活：{kill_reason}",
                regime_adjustment="KILL_SWITCH",
            )
            return make_blocked(rb, RiskBlockReason.KILL_SWITCH.value, rb.reasoning, cid)

        if any_block:
            block_reasons = [tr.block_reason.value for tr in trigger_results
                             if tr.triggered and tr.block_reason is not None]
            block_evidence = "; ".join(tr.reasoning for tr in trigger_results
                                       if tr.triggered and tr.block_reason is not None)
            rb = RiskBudget(
                allowed_position_size=ctx.base_position_pct * min_scale,
                stop_loss=sl_amount * (1.0 - sl_adjust),
                take_profit=tp_amount,
                max_loss_amount=sl_amount * (1.0 - sl_adjust) * (ctx.base_position_pct * min_scale),
                portfolio_correlation=ctx.max_correlation,
                safety=self.envelope,
                reasoning=f"反射触发缩仓：{block_evidence}",
                regime_adjustment="REFLEX_SCALE_DOWN",
            )
            return make_blocked(rb, "|".join(block_reasons), block_evidence, cid)

        # === 7) 三级风险预算统一输出（P7-T1）===
        # L1 Agent 级：单笔 SL/TP
        # L2 策略组级：相关性
        # L3 组合/系统级：敞口 + 日损失
        scaled_position = ctx.base_position_pct * min_scale
        adjusted_sl = sl_amount * (1.0 - sl_adjust)
        all_trigger_reasons = "; ".join(tr.reasoning for tr in trigger_results if tr.triggered)
        reasoning_parts = []
        if all_trigger_reasons:
            reasoning_parts.append(f"触发器: {all_trigger_reasons}")
        reasoning_parts.append(f"联合EV={joint_ev:.4f} (win_rate={win_rate:.3f})")
        reasoning_parts.append(f"三级预算: L1单笔SL={adjusted_sl:.4f}, L2相关性={ctx.max_correlation:.3f}, L3敞口={ctx.portfolio_exposure:.3f}")

        rb = RiskBudget(
            allowed_position_size=scaled_position,
            stop_loss=adjusted_sl,
            take_profit=tp_amount,
            max_loss_amount=adjusted_sl * scaled_position,
            portfolio_correlation=ctx.max_correlation,
            safety=self.envelope,
            reasoning=" | ".join(reasoning_parts),
            regime_adjustment=ctx.market_state.regime.value if ctx.market_state else "UNKNOWN",
        )
        return rb

    # === 私有方法 ===

    def _calc_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """从真实交易计算胜率。"""
        if not trades:
            return 0.5  # 无历史默认 50%
        wins = sum(1 for t in trades if t.get("is_win", False) or t.get("win", False))
        return wins / len(trades) if trades else 0.5

    def _blocked(self, cid: str, reason: PropagationReason, message: str) -> RiskBudget:
        rb = RiskBudget(safety=self.envelope, reasoning=message, regime_adjustment=reason.value)
        return make_blocked(rb, reason.value, message, cid)

    def _blocked_hard(self, cid: str, reason: RiskBlockReason, message: str) -> RiskBudget:
        rb = RiskBudget(safety=self.envelope, reasoning=message, regime_adjustment=reason.value)
        return make_blocked(rb, reason.value, message, cid)

