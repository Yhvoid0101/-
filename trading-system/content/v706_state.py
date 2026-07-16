# -*- coding: utf-8 -*-
"""v706 V706State — 前向验证状态持久化数据结构
================================================================================
用户铁律:
  "在未达成性能指标前不得推进实盘部署工作"
  "永远永远不要出现模拟牛逼, 实盘亏损的情况"

本模块定义 V706State dataclass, 用于 v706 Forward-test Framework 的状态持久化:
  - 月度指标累积 (monthly_metrics)
  - 全局交易统计 (total_trades/wins/losses/pnl)
  - v704 独有字段 (kelly_trade_history/regime_distribution/bear_signal_count)
  - 冷却期持久化 (per_symbol_cooldown, 修复 ERR-20260703-cooldown-persistence)
  - SIM→LIVE 桥接评估结果 (tier2_evaluation)

设计原则:
  1. 独立于 tier2_state.json (D4: 避免污染 v97/v98 基线状态)
  2. JSON 序列化兼容 (所有字段可被 json.dumps 序列化)
  3. 向前兼容 (load_state 容忍缺失字段, 用默认值填充)
  4. 双写冷却期 (signal_gen 运行时 + state 持久化, 跨日不丢失)

来源:
  - tier2_forward_walk_validator.py:237 — Tier2State (参考模式)
  - v706_config.py:V706_STATE_FILE — 状态文件路径
  - ERR-20260703-cooldown-persistence — 冷却期持久化修复
================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from v706_config import V706_STATE_FILE

logger = logging.getLogger("hermes.v706.state")


@dataclass
class V706State:
    """v706 前向验证状态 (持久化)

    用户铁律: "在未达成性能指标前不得推进实盘部署工作"
    本状态记录所有 OOS 期累积数据, 用于 SIM→LIVE 桥接评估.

    字段分组:
      1. 元数据 (version/created_at/last_updated)
      2. 收集状态 (oos_start_date/collection_days/last_collection_ts)
      3. 月度指标累积 (monthly_metrics/consecutive_go_months)
      4. 全局交易统计 (total_trades/wins/losses/pnl/max_*)
      5. v704 独有字段 (kelly_trade_history/regime_distribution/bear_signal_count)
      6. Tier 2 评估 (tier2_passed/tier2_evaluation)
      7. v704 冻结基线 (v704_baseline, 用于对比)
      8. 冷却期持久化 (per_symbol_cooldown, 修复 ERR-20260703-cooldown-persistence)
      9. 重置历史 (reset_history, 追踪 state 重置操作)
    """

    # ===== 1. 元数据 =====
    version: str = "v706_forward_test"
    created_at: str = ""
    last_updated: str = ""

    # ===== 2. 收集状态 =====
    oos_start_date: str = "2026-07-01"
    collection_days: int = 0
    last_collection_ts: float = 0.0

    # ===== 3. 月度指标累积 =====
    monthly_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consecutive_go_months: int = 0

    # ===== 4. 全局交易统计 =====
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_sim_pnl_usd: float = 0.0
    total_live_pnl_usd: float = 0.0
    total_cost_usd: float = 0.0
    max_single_loss_pct: float = 0.0
    max_consec_loss: int = 0

    # ===== 5. v704 独有字段 =====
    # kelly trade history (kelly_sizer.trade_history 的持久化副本)
    v704_kelly_trade_history: List[Dict[str, Any]] = field(default_factory=list)
    # regime 分布统计 (V706RegimeClassifier.get_distribution 的持久化副本)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    # bear_signal 触发次数 (v704 Fix2 验证)
    bear_signal_count: int = 0
    # kelly=0 占比 (v704 Fix1 死锁检测)
    kelly_zero_count: int = 0
    kelly_zero_ratio: float = 0.0

    # ===== 6. Tier 2 评估 =====
    tier2_passed: bool = False
    tier2_evaluation: Dict[str, Any] = field(default_factory=dict)

    # ===== 7. v704 冻结基线 (用于对比, 来自 V704_BASELINE) =====
    v704_baseline: Dict[str, Any] = field(default_factory=dict)

    # ===== 8. 冷却期持久化 (修复 ERR-20260703-cooldown-persistence) =====
    # 根因: per_symbol_cooldown 原仅在 signal_gen 上, 不被 save_state 持久化
    # 跨日 collect_daily_data 时冷却期丢失, 导致跨日背靠背开仓
    # 修复: 双写 signal_gen(运行时) + state(持久化), initialize 时回灌
    per_symbol_cooldown: Dict[str, float] = field(default_factory=dict)

    # ===== 9. v97 状态快照 (委托 v97 追踪, 但持久化在 state) =====
    v97_current_consec_loss: int = 0
    v97_max_consec_seen: int = 0
    v97_max_single_loss_seen: float = 0.0
    v97_per_symbol_consec: Dict[str, int] = field(default_factory=dict)

    # ===== 10. 重置历史 (追踪 state 重置操作) =====
    reset_history: List[Dict[str, Any]] = field(default_factory=list)

    # ========================================================================
    # 持久化方法
    # ========================================================================

    def save_state(self, path: Optional[Path] = None) -> None:
        """保存状态到 JSON 文件

        Args:
            path: 目标路径 (默认 V706_STATE_FILE)
        """
        target_path = path or V706_STATE_FILE
        self.last_updated = datetime.now(timezone.utc).isoformat()

        # 确保目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = asdict(self)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False, default=str)
        logger.info("v706 状态已保存: %s", target_path)

    @classmethod
    def load_state(cls, path: Optional[Path] = None) -> "V706State":
        """从 JSON 文件加载状态

        向前兼容: 容忍缺失字段, 用默认值填充.
        防御性修复 (ERR-20260703-tier2-state-reset): 若文件不存在返回默认状态.

        Args:
            path: 源文件路径 (默认 V706_STATE_FILE)

        Returns:
            V706State 实例 (从文件加载或默认)
        """
        target_path = path or V706_STATE_FILE
        if not target_path.exists():
            logger.info("v706 状态文件不存在, 使用默认状态: %s", target_path)
            return cls()

        try:
            state_dict = json.loads(target_path.read_text(encoding="utf-8"))
            # 仅保留 dataclass 已知字段 (向前兼容)
            known_fields = set(cls.__dataclass_fields__.keys())
            filtered = {k: v for k, v in state_dict.items() if k in known_fields}
            state = cls(**filtered)
            logger.info("v706 状态已加载: %s (trades=%d, days=%d)",
                        target_path, state.total_trades, state.collection_days)
            return state
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("v706 状态加载失败 (%s), 使用默认状态", str(e)[:80])
            return cls()

    # ========================================================================
    # 状态更新辅助方法 (供 V706ForwardTestRunner 调用)
    # ========================================================================

    def update_kelly_zero_ratio(self) -> None:
        """更新 kelly_zero_ratio (基于 kelly_zero_count / total_trades)

        v704 Fix1 验证: kelly=0 占比应 < 10% (SIM_TO_LIVE_BRIDGE.kelly_zero_max_ratio)
        """
        if self.total_trades > 0:
            self.kelly_zero_ratio = self.kelly_zero_count / self.total_trades
        else:
            self.kelly_zero_ratio = 0.0

    def record_trade(
        self,
        pnl_pct: float,
        pnl_usd: float,
        cost_usd: float,
        is_bear_signal: bool = False,
        kelly_mult: float = 0.0,
    ) -> None:
        """记录一笔交易 (更新全局统计)

        Args:
            pnl_pct: 盈亏百分比 (正=盈利, 负=亏损)
            pnl_usd: 盈亏USD (sim_pnl)
            cost_usd: 成本USD
            is_bear_signal: 是否为 bear_signal 路径
            kelly_mult: 该交易的 kelly_mult (用于死锁检测)
        """
        self.total_trades += 1
        if pnl_pct >= 0:
            self.total_wins += 1
        else:
            self.total_losses += 1
        self.total_sim_pnl_usd += pnl_usd
        self.total_live_pnl_usd += pnl_usd - cost_usd
        self.total_cost_usd += cost_usd

        # 单亏追踪
        if pnl_pct < 0:
            loss_pct = abs(pnl_pct) * 100
            if loss_pct > self.max_single_loss_pct:
                self.max_single_loss_pct = loss_pct

        # bear_signal 统计
        if is_bear_signal:
            self.bear_signal_count += 1

        # kelly 死锁检测
        if kelly_mult <= 0:
            self.kelly_zero_count += 1
        self.update_kelly_zero_ratio()

    def update_from_signal_gen(self, signal_gen: Any) -> None:
        """从 V704SignalGenerator 同步状态 (供 save_state 前调用)

        将 signal_gen 运行时状态同步到 state 持久化字段:
          - v97 consec_loss/per_symbol_consec/max_*
          - per_symbol_cooldown
          - kelly_trade_history
          - regime_distribution

        Args:
            signal_gen: V704SignalGenerator 实例
        """
        # v97 状态快照
        self.v97_current_consec_loss = getattr(
            signal_gen.v97, "current_consec_loss", 0)
        self.v97_max_consec_seen = getattr(
            signal_gen.v97, "max_consec_seen", 0)
        self.v97_max_single_loss_seen = getattr(
            signal_gen.v97, "max_single_loss_seen", 0.0)
        self.v97_per_symbol_consec = dict(
            getattr(signal_gen.v97, "per_symbol_consec", {}))

        # 冷却期 (双写: signal_gen + state)
        self.per_symbol_cooldown = dict(
            getattr(signal_gen.v97, "per_symbol_cooldown", {}))

        # kelly trade history
        self.v704_kelly_trade_history = list(
            getattr(signal_gen.kelly_sizer, "trade_history", []))

        # regime 分布
        # 防御性修复 ERR-20260704-v706-regime-distribution-restore-gap:
        #   restore_to_signal_gen 未恢复 regime_distribution 到 signal_gen,
        #   导致 cmd_can_deploy (调用 _load_runner + can_deploy_live + save_state)
        #   流程中, update_from_signal_gen 从空的新 signal_gen 读取 regime_distribution
        #   (total=0), 覆盖了 state 中已有的非空 regime_distribution.
        #   修复: 只在新 distribution 非空 (total > 0) 时才覆盖.
        new_regime_dist = dict(
            getattr(signal_gen.regime_classifier, "get_distribution", lambda: {})())
        new_total = sum(v for k, v in new_regime_dist.items()
                        if k not in ("total", "atr_overrides"))
        existing_total = sum(v for k, v in self.regime_distribution.items()
                             if k not in ("total", "atr_overrides"))
        if new_total > 0 or existing_total == 0:
            # 新 distribution 非空, 或 existing 也空 → 安全覆盖
            self.regime_distribution = new_regime_dist
        else:
            # 新 distribution 空 (signal_gen 未运行策略), 但 existing 非空 → 保留 existing
            # 防止 update_from_signal_gen 用空值覆盖已有的累计统计
            logger.debug(
                "保留已有 regime_distribution (total=%d), "
                "signal_gen 当前 distribution 为空", existing_total)

    def restore_to_signal_gen(self, signal_gen: Any) -> None:
        """将持久化状态回灌到 V704SignalGenerator (供 load_state 后调用)

        恢复运行时状态:
          - v97 per_symbol_cooldown (跨日不丢失)
          - v97 per_symbol_consec
          - kelly_sizer.trade_history

        Args:
            signal_gen: V704SignalGenerator 实例
        """
        # 回灌冷却期
        signal_gen.v97.per_symbol_cooldown = dict(self.per_symbol_cooldown)

        # 回灌 per_symbol_consec
        signal_gen.v97.per_symbol_consec = dict(self.v97_per_symbol_consec)

        # 回灌 consec_loss (可选, 因为可能跨日已重置)
        signal_gen.v97.current_consec_loss = self.v97_current_consec_loss
        signal_gen.v97.max_consec_seen = self.v97_max_consec_seen
        signal_gen.v97.max_single_loss_seen = self.v97_max_single_loss_seen

        # 回灌 kelly trade_history
        signal_gen.kelly_sizer.trade_history = list(self.v704_kelly_trade_history)

        logger.info(
            "状态已回灌到 signal_gen: cooldown=%d syms, kelly_history=%d trades",
            len(signal_gen.v97.per_symbol_cooldown),
            len(signal_gen.kelly_sizer.trade_history))

    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要 (用于状态报告)"""
        return {
            "version": self.version,
            "collection_days": self.collection_days,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": (self.total_wins / self.total_trades * 100
                         if self.total_trades > 0 else 0.0),
            "total_sim_pnl_usd": round(self.total_sim_pnl_usd, 2),
            "total_live_pnl_usd": round(self.total_live_pnl_usd, 2),
            "total_cost_usd": round(self.total_cost_usd, 2),
            "max_single_loss_pct": round(self.max_single_loss_pct, 2),
            "max_consec_loss": self.max_consec_loss,
            "consecutive_go_months": self.consecutive_go_months,
            "bear_signal_count": self.bear_signal_count,
            "kelly_zero_ratio": round(self.kelly_zero_ratio, 4),
            "tier2_passed": self.tier2_passed,
        }


# ============================================================================
# 自验证 (模块加载时自动运行)
# ============================================================================


def _self_validate() -> bool:
    """自验证: 确保 V706State 可实例化且 save/load 正常工作"""
    try:
        # 验证实例化
        state = V706State()
        if state.version != "v706_forward_test":
            logger.error("V706State.version 异常: %s", state.version)
            return False

        # 验证 record_trade
        state.record_trade(
            pnl_pct=0.02, pnl_usd=20.0, cost_usd=1.0,
            is_bear_signal=False, kelly_mult=0.5)
        if state.total_trades != 1 or state.total_wins != 1:
            logger.error("record_trade 异常: trades=%d, wins=%d",
                         state.total_trades, state.total_wins)
            return False

        # 验证 kelly_zero_ratio
        state.record_trade(
            pnl_pct=-0.01, pnl_usd=-10.0, cost_usd=0.5,
            is_bear_signal=True, kelly_mult=0.0)
        if state.kelly_zero_count != 1:
            logger.error("kelly_zero_count 异常: %d", state.kelly_zero_count)
            return False
        if state.bear_signal_count != 1:
            logger.error("bear_signal_count 异常: %d", state.bear_signal_count)
            return False
        state.update_kelly_zero_ratio()
        if abs(state.kelly_zero_ratio - 0.5) > 0.01:
            logger.error("kelly_zero_ratio 异常: %f", state.kelly_zero_ratio)
            return False

        # 验证 get_status_summary
        summary = state.get_status_summary()
        if "win_rate" not in summary:
            logger.error("get_status_summary 缺少 win_rate")
            return False

        return True
    except Exception as e:
        logger.error("V706State 自验证失败: %s", str(e)[:100])
        return False


assert _self_validate(), "V706State 自验证失败"


# ============================================================================
# 快速自测 (python v706_state.py)
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("v706 V706State — 自测")
    print("=" * 70)

    # 测试1: 实例化
    state = V706State()
    print(f"\n[Test 1] 实例化: version={state.version}")
    assert state.version == "v706_forward_test"
    assert state.total_trades == 0
    print("  ✅ PASS")

    # 测试2: record_trade
    state.record_trade(0.02, 20.0, 1.0, is_bear_signal=False, kelly_mult=0.5)
    print(f"\n[Test 2] record_trade(盈利): trades={state.total_trades}, "
          f"wins={state.total_wins}, losses={state.total_losses}")
    assert state.total_trades == 1
    assert state.total_wins == 1
    assert state.total_losses == 0
    print("  ✅ PASS")

    # 测试3: record_trade (亏损 + bear_signal + kelly=0)
    state.record_trade(-0.015, -15.0, 0.8, is_bear_signal=True, kelly_mult=0.0)
    print(f"\n[Test 3] record_trade(亏损+bear+kelly=0): trades={state.total_trades}, "
          f"losses={state.total_losses}, bear={state.bear_signal_count}, "
          f"kelly_zero={state.kelly_zero_count}")
    assert state.total_trades == 2
    assert state.total_losses == 1
    assert state.bear_signal_count == 1
    assert state.kelly_zero_count == 1
    print("  ✅ PASS")

    # 测试4: max_single_loss_pct
    print(f"\n[Test 4] max_single_loss_pct: {state.max_single_loss_pct}")
    assert state.max_single_loss_pct == 1.5  # abs(-0.015) * 100
    print("  ✅ PASS")

    # 测试5: kelly_zero_ratio
    print(f"\n[Test 5] kelly_zero_ratio: {state.kelly_zero_ratio}")
    assert abs(state.kelly_zero_ratio - 0.5) < 0.01  # 1/2
    print("  ✅ PASS")

    # 测试6: get_status_summary
    summary = state.get_status_summary()
    print(f"\n[Test 6] get_status_summary: {summary}")
    assert summary["total_trades"] == 2
    assert summary["win_rate"] == 50.0  # 1 win / 2 trades
    assert summary["bear_signal_count"] == 1
    print("  ✅ PASS")

    # 测试7: save_state + load_state (往返测试)
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        from pathlib import Path
        state.save_state(Path(tmp_path))
        loaded = V706State.load_state(Path(tmp_path))
        print(f"\n[Test 7] save+load: trades={loaded.total_trades}, "
              f"wins={loaded.total_wins}, bear={loaded.bear_signal_count}")
        assert loaded.total_trades == 2
        assert loaded.total_wins == 1
        assert loaded.bear_signal_count == 1
        assert loaded.kelly_zero_count == 1
        print("  ✅ PASS")
    finally:
        os.unlink(tmp_path)

    # 测试8: load_state 不存在的文件
    loaded = V706State.load_state(Path("/nonexistent/path/state.json"))
    print(f"\n[Test 8] load_state(不存在): trades={loaded.total_trades}")
    assert loaded.total_trades == 0
    print("  ✅ PASS")

    print("\n" + "=" * 70)
    print("✅ v706 V706State 自测全部通过 (8/8)")
    print("=" * 70)
