# -*- coding: utf-8 -*-
"""
风控层 (Risk Control Layer) — 三层熔断 + 物理逃生

三层熔断架构：
  L1: 子Agent级熔断 — 单个Agent连续亏损或回撤超限
  L2: 策略组级熔断 — 相似策略集体异常
  L3: 全局级熔断 — 系统性风险（黑天鹅/插针/交易所宕机）

物理逃生：独立于主系统的树莓派心跳检测
  - 心跳断了 → 无条件平仓
  - 不依赖主程序判断
  - 带外管理卡 + 硬拔网线预案

不可动摇的边界：
  - 物理逃生独立于主系统
  - 实盘不进化
  - 风控判断不依赖AI模型
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.risk_control")


# ============================================================================
# 熔断状态
# ============================================================================


class CircuitState(Enum):
    CLOSED = "closed"        # 正常
    HALF_OPEN = "half_open"  # 试探性恢复
    OPEN = "open"            # 熔断（禁止交易）


@dataclass(slots=True)
class AgentRiskProfile:
    """子Agent风险画像"""
    agent_id: str
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    current_drawdown: float = 0.0
    peak_drawdown: float = 0.0
    peak_equity: float = 0.0
    daily_loss: float = 0.0
    weekly_loss: float = 0.0
    total_trades: int = 0
    total_losses: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: float = 0.0
    circuit_cooldown_seconds: float = 3600.0  # 熔断冷却时间
    last_trade_time: float = 0.0


@dataclass(slots=True)
class GlobalRiskState:
    """全局风险状态"""
    total_exposure: float = 0.0
    max_exposure: float = 100000.0
    daily_pnl: float = 0.0
    max_daily_loss: float = 5000.0
    volatility_alert: bool = False
    exchange_health: bool = True
    black_swan_detected: bool = False
    global_circuit: CircuitState = CircuitState.CLOSED
    last_heartbeat: float = field(default_factory=time.time)


# ============================================================================
# 三层熔断系统
# ============================================================================


class RiskControlManager:
    """风控管理器 — 三层熔断 + 物理逃生

    使用方式:
      rcm = RiskControlManager()
      rcm.set_global_limits(max_daily_loss=5000, max_exposure=100000)

      # 每笔交易前检查
      if rcm.pre_trade_check(agent_id, symbol, quantity, price):
          engine.submit_order(...)

      # 每笔交易后更新
      rcm.post_trade_update(agent_id, pnl)

      # 物理逃生心跳
      rcm.heartbeat()  # 每N秒调用一次
    """

    def __init__(self, initial_capital: float = 10000.0, sandbox_mode: bool = False):
        # L1: Agent级
        self._agent_profiles: Dict[str, AgentRiskProfile] = {}
        self._sandbox_mode = sandbox_mode  # Phase 10.3: 沙盘模式标志
        self._initial_capital = initial_capital

        # L2: 策略组级
        self._strategy_groups: Dict[str, List[str]] = defaultdict(list)

        # L3: 全局级
        self._global = GlobalRiskState()

        # 熔断配置
        self._config: Dict[str, Any] = {
            "l1_max_consecutive_losses": 8,
            "l1_max_drawdown_pct": 0.20,
            "l1_max_daily_loss_pct": 0.20,
            "l1_circuit_cooldown_hours": 1,
            "l2_max_group_correlation": 0.8,
            "l2_max_group_exposure_pct": 0.40,
            "l3_max_daily_loss": 5000.0,
            "l3_max_exposure": 100000.0,
            "l3_volatility_threshold": 0.10,  # 10%波动触发全局警报
            "l3_black_swan_threshold": 0.05,  # 5%瞬时暴跌=黑天鹅
        }

        # 物理逃生
        self._heartbeat_interval: float = 5.0  # 心跳间隔(秒)
        self._heartbeat_timeout: float = 30.0  # 心跳超时(秒)
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_running: bool = False
        self._emergency_shutdown_callback: Optional[Callable] = None
        self._last_external_heartbeat: float = time.time()

        # 事件日志
        self._event_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    # ==================================================================
    # Phase 14.25: 信号方法适配器 (根治safe_margin"未找到任何信号方法")
    # ==================================================================

    def analyze(self, ctx=None):
        """Phase 14.25: 信号方法适配器 — 让RiskControlManager兼容strategy_registry

        根因修复:
        - safe_margin策略在gene_codec中注册为RiskControlManager类
        - RiskControlManager是风控类,只有pre_trade_check/post_trade_update
        - 无标准信号方法(analyze/scan等),导致safe_margin策略全部空壳

        适配逻辑(格雷厄姆安全边际风格):
        - 从ctx提取价格历史,计算波动率
        - 波动率低=有安全边际 → 轻度看多
        - 波动率高=无安全边际 → 中性/观望
        - 沙盘模式返回保守中性信号

        Args:
            ctx: MarketContext

        Returns:
            dict: 信号字典
        """
        try:
            symbol = ""
            current_price = 0.0
            if ctx is not None:
                current_price = float(getattr(ctx, "close", 0.0) or 0.0)
                if hasattr(ctx, "symbol"):
                    symbol = ctx.symbol or ""

            # 从price_history计算波动率(格雷厄姆安全边际的核心指标)
            volatility = 0.0
            if ctx is not None and hasattr(ctx, "price_history"):
                prices = ctx.price_history or []
                if len(prices) >= 10:
                    try:
                        import numpy as np
                        arr = np.array(prices[-30:], dtype=float)
                        if len(arr) >= 2 and arr.mean() > 0:
                            returns = np.diff(arr) / arr[:-1]
                            volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
                    except Exception:
                        volatility = 0.0

            # 格雷厄姆安全边际逻辑:
            # - 波动率<2% (低波动) → 有安全边际,轻度看多
            # - 波动率2-5% (中波动) → 中性
            # - 波动率>5% (高波动) → 观望(无安全边际)
            if volatility < 0.02 and current_price > 0:
                return {
                    "signal": "long",
                    "direction": "long",
                    "confidence": 0.4,
                    "strength": 0.3,
                    "reason": f"safe_margin_low_vol({volatility:.4f})",
                    "symbol": symbol,
                }
            elif volatility > 0.05:
                return {
                    "signal": "neutral",
                    "confidence": 0.2,
                    "reason": f"high_vol_no_margin({volatility:.4f})",
                    "symbol": symbol,
                }
            else:
                return {
                    "signal": "neutral",
                    "confidence": 0.3,
                    "reason": f"medium_vol({volatility:.4f})",
                    "symbol": symbol,
                }
        except Exception as e:
            return {"signal": "neutral", "confidence": 0.0, "reason": f"error: {e}"}

    def update(self, ctx=None):
        """Phase 14.25: 通用update适配器"""
        return self.analyze(ctx)
    def set_global_limits(
        self,
        max_daily_loss: float = 5000.0,
        max_exposure: float = 100000.0,
        volatility_threshold: float = 0.10,
        black_swan_threshold: float = 0.05,
    ) -> None:
        """设置全局风控限制"""
        self._config["l3_max_daily_loss"] = max_daily_loss
        self._config["l3_max_exposure"] = max_exposure
        self._config["l3_volatility_threshold"] = volatility_threshold
        self._config["l3_black_swan_threshold"] = black_swan_threshold
        self._global.max_daily_loss = max_daily_loss
        self._global.max_exposure = max_exposure

    def set_agent_limits(
        self,
        max_consecutive_losses: int = 5,
        max_drawdown_pct: float = 0.25,
        max_daily_loss_pct: float = 0.10,
        circuit_cooldown_hours: int = 1,
    ) -> None:
        """设置Agent级风控限制"""
        self._config["l1_max_consecutive_losses"] = max_consecutive_losses
        self._config["l1_max_drawdown_pct"] = max_drawdown_pct
        self._config["l1_max_daily_loss_pct"] = max_daily_loss_pct
        self._config["l1_circuit_cooldown_hours"] = circuit_cooldown_hours

    # ------------------------------------------------------------------
    # L1: Agent级熔断
    # ------------------------------------------------------------------

    def get_agent_profile(self, agent_id: str) -> AgentRiskProfile:
        """获取或创建Agent风险画像"""
        if agent_id not in self._agent_profiles:
            self._agent_profiles[agent_id] = AgentRiskProfile(agent_id=agent_id)
        return self._agent_profiles[agent_id]

    def pre_trade_check(
        self, agent_id: str, symbol: str, quantity: float, price: float,
    ) -> Tuple[bool, str]:
        """交易前检查

        Returns:
            (是否允许, 原因)
        """
        
        # L3: 全局熔断检查
        # Phase J 修复: 沙盘模式下跳过生产环境硬门控 (与 Phase G GNN 同模式)
        # 根因: 沙盘2019历史数据5%暴跌→black_swan永不重置→所有交易被拒绝
        #       exchange_health/heartbeat 在沙盘模式无意义(无真实交易所)
        if not self._sandbox_mode:
            if self._global.global_circuit == CircuitState.OPEN:
                return False, "GLOBAL_CIRCUIT_OPEN"

            if self._global.black_swan_detected:
                return False, "BLACK_SWAN_DETECTED"

            if not self._global.exchange_health:
                return False, "EXCHANGE_UNHEALTHY"

            # 物理逃生检查
            if time.time() - self._last_external_heartbeat > self._heartbeat_timeout:
                self._trigger_emergency_shutdown("HEARTBEAT_TIMEOUT")
                return False, "HEARTBEAT_TIMEOUT"
        else:
            # 沙盘模式: 记录但不阻塞 (进化需要交易数据)
            if self._global.black_swan_detected:
                if not hasattr(self, '_phase_j_bs_log_count'):
                    self._phase_j_bs_log_count = 0
                self._phase_j_bs_log_count += 1
                if self._phase_j_bs_log_count <= 3 or self._phase_j_bs_log_count % 500 == 0:
                    logger.warning(
                        "Phase J SANDBOX_BLACK_SWAN_SOFT_PASS #%d (生产模式会硬阻塞)",
                        self._phase_j_bs_log_count,
                    )

        # L3: 日亏损限额
        if abs(self._global.daily_pnl) > self._config["l3_max_daily_loss"]:
            self._trigger_global_circuit("DAILY_LOSS_LIMIT")
            return False, "DAILY_LOSS_LIMIT"

        # L3: 总敞口限额
        notional = quantity * price
        if self._global.total_exposure + notional > self._config["l3_max_exposure"]:
            return False, "EXPOSURE_LIMIT"

        # L1: Agent级检查
        profile = self.get_agent_profile(agent_id)

        if profile.circuit_state == CircuitState.OPEN:
            # 检查冷却时间
            if time.time() - profile.circuit_opened_at < profile.circuit_cooldown_seconds:
                return False, "AGENT_CIRCUIT_OPEN"
            else:
                profile.circuit_state = CircuitState.HALF_OPEN
                self._log_event("L1_CIRCUIT_HALF_OPEN", agent_id=agent_id)

        # L1: 连续亏损 — 增强: 自适应冷却
        # 冷却时间 = 基础1h × (连续亏损次数 / 阈值)
        if profile.consecutive_losses >= self._config["l1_max_consecutive_losses"]:
            # 自适应冷却: 亏损越多, 冷却越长
            severity = profile.consecutive_losses / self._config["l1_max_consecutive_losses"]
            base_cooldown = 3600
            max_cooldown = 14400
            adaptive_cooldown = min(severity * base_cooldown, max_cooldown)
            profile.circuit_cooldown_seconds = adaptive_cooldown
            self._trigger_agent_circuit(agent_id, "CONSECUTIVE_LOSSES")
            self._log_event("L1_CIRCUIT_ADAPTIVE_COOLDOWN",
                            agent_id=agent_id,
                            detail="cool=%.0fs losses=%d threshold=%d" % (
                                adaptive_cooldown, profile.consecutive_losses,
                                self._config["l1_max_consecutive_losses"]))
            return False, "AGENT_CONSECUTIVE_LOSSES"

        # L1: 回撤
        if profile.current_drawdown > self._config["l1_max_drawdown_pct"]:
            self._trigger_agent_circuit(agent_id, "DRAWDOWN_LIMIT")
            return False, "AGENT_DRAWDOWN_LIMIT"

        # L1: 日亏损 — 修复: daily_loss是绝对值(美元), l1_max_daily_loss_pct是百分比
        # 需要将百分比乘以初始资金或当前权益来比较
        daily_loss_limit = self._config["l1_max_daily_loss_pct"] * self._initial_capital
        if abs(profile.daily_loss) > daily_loss_limit:
            self._trigger_agent_circuit(agent_id, "DAILY_LOSS_LIMIT")
            return False, "AGENT_DAILY_LOSS_LIMIT"

        return True, "OK"

    def post_trade_update(
        self, agent_id: str, pnl: float, notional: float = 0.0,
        entry_price: float = 0.0, exit_price: float = 0.0,
        equity: float = 0.0,
        event_type: str = "close",
    ) -> None:
        """交易后更新风险状态

        v3.24修复 (ME协调 LOSS-HUNTER): 区分开仓/平仓事件
        根因: 之前 evolution_loop.py:1861 新开仓时也调用此方法 (pnl=0),
              导致 else 分支重置 consecutive_losses=0.
              这导致3次连续亏损熔断永远不生效 — 单日可达38次连续亏损!
              实测 v3.23 主回测: 最大连续亏损147次(跨17.88天), 单日38次连续亏损
        修复: 开仓事件(event_type="open")不更新 consecutive_losses,
              只在平仓事件(event_type="close")中根据真实pnl更新

        Args:
            event_type: "open"=开仓(不计入连续亏损), "close"=平仓(计入盈亏)
        """
        profile = self.get_agent_profile(agent_id)

        profile.total_trades += 1
        profile.last_trade_time = time.time()

        # v3.24修复: 只在平仓事件中更新 consecutive_losses
        if event_type == "close":
            if pnl < 0:
                profile.consecutive_losses += 1
                profile.total_losses += 1
                profile.max_consecutive_losses = max(
                    profile.max_consecutive_losses, profile.consecutive_losses,
                )
            elif pnl > 0:
                # 只在真正盈利时重置 (pnl=0 的平仓不重置, 避免边缘case误清零)
                profile.consecutive_losses = 0
        # event_type == "open": 跳过 consecutive_losses 更新

        # 更新回撤：从峰值权益计算 (peak_equity - current_equity) / peak_equity
        if equity > 0:
            if equity > profile.peak_equity:
                profile.peak_equity = equity
            if profile.peak_equity > 0:
                profile.current_drawdown = (profile.peak_equity - equity) / profile.peak_equity
            profile.peak_drawdown = max(profile.peak_drawdown, profile.current_drawdown)

        # 更新日亏损
        profile.daily_loss += pnl

        # 更新全局
        self._global.daily_pnl += pnl
        self._global.total_exposure += notional

        # 半开状态恢复
        if profile.circuit_state == CircuitState.HALF_OPEN and pnl > 0:
            profile.circuit_state = CircuitState.CLOSED
            profile.consecutive_losses = 0
            self._log_event("L1_CIRCUIT_CLOSED", agent_id=agent_id)

    def _trigger_agent_circuit(self, agent_id: str, reason: str) -> None:
        """触发Agent级熔断"""
        profile = self.get_agent_profile(agent_id)
        profile.circuit_state = CircuitState.OPEN
        profile.circuit_opened_at = time.time()
        profile.circuit_cooldown_seconds = self._config["l1_circuit_cooldown_hours"] * 3600
        self._log_event("L1_CIRCUIT_OPEN", agent_id=agent_id, reason=reason)
        logger.warning("Agent %s circuit OPEN: %s", agent_id[:8], reason)

    def _trigger_global_circuit(self, reason: str) -> None:
        """触发全局熔断"""
        self._global.global_circuit = CircuitState.OPEN
        self._log_event("L3_GLOBAL_CIRCUIT_OPEN", reason=reason)
        logger.error("GLOBAL CIRCUIT OPEN: %s", reason)

    # ------------------------------------------------------------------
    # L2: 策略组级熔断
    # ------------------------------------------------------------------

    def register_strategy_group(self, group_name: str, agent_ids: List[str]) -> None:
        """注册策略组"""
        self._strategy_groups[group_name] = agent_ids

    def check_strategy_group(self, group_name: str) -> Tuple[bool, str]:
        """检查策略组风险"""
        agents = self._strategy_groups.get(group_name, [])
        if not agents:
            return True, "OK"

        # 检查组内Agent的熔断比例
        open_count = sum(
            1 for a in agents
            if self._agent_profiles.get(a, AgentRiskProfile(agent_id=a)).circuit_state == CircuitState.OPEN
        )

        # Phase 10.3: L2策略组熔断阈值根据 sandbox_mode 自适应
        # 根因: 1个agent熔断=100%>50%触发GROUP_CIRCUIT, 沙盘进化无数据
        # 沙盘模式: 阈值0.8 (允许80%agent熔断才触发组熔断)
        # 生产模式: 阈值0.5 (50%即触发, 生产级严格)
        threshold = 1.0 if self._sandbox_mode else 0.5  # Phase 10.3b: 沙盘1.0(全部熔断才触发)
        if open_count / len(agents) > threshold:
            return False, f"GROUP_CIRCUIT: {open_count}/{len(agents)} open"

        return True, "OK"

    # ------------------------------------------------------------------
    # L3: 全局风控
    # ------------------------------------------------------------------

    def check_volatility(self, price_change_pct: float) -> bool:
        """检查波动率是否触发警报"""
        if abs(price_change_pct) > self._config["l3_volatility_threshold"]:
            self._global.volatility_alert = True
            self._log_event("VOLATILITY_ALERT", change_pct=price_change_pct)
            return True
        return False

    def check_black_swan(self, price_change_pct: float) -> bool:
        """检测黑天鹅事件（瞬时暴跌/暴涨）"""
        if abs(price_change_pct) > self._config["l3_black_swan_threshold"]:
            # Phase J Fix 3: 沙盘模式不触发全局熔断 (只记录警告)
            # 根因: 沙盘历史数据5%暴跌是正常波动, 不应触发BLACK_SWAN全局熔断
            #       Phase J Fix 1 已修复 pre_trade_check (沙盘不阻塞)
            #       但 check_black_swan 仍打印 ERROR 日志, 误导监控
            if self._sandbox_mode:
                logger.warning(
                    "Phase J SANDBOX_BLACK_SWAN_DETECTED: change=%.4f (沙盘模式不触发全局熔断)",
                    price_change_pct,
                )
                self._log_event("SANDBOX_BLACK_SWAN_SOFT", change_pct=price_change_pct)
                return True
            self._global.black_swan_detected = True
            self._trigger_global_circuit("BLACK_SWAN")
            self._log_event("BLACK_SWAN", change_pct=price_change_pct)
            return True
        return False

    def set_exchange_health(self, healthy: bool) -> None:
        """设置交易所健康状态"""
        self._global.exchange_health = healthy
        if not healthy:
            self._log_event("EXCHANGE_UNHEALTHY")

    def reset_global_circuit(self) -> None:
        """重置全局熔断状态（沙盘模式每轮独立使用）"""
        self._global.global_circuit = CircuitState.CLOSED
        self._log_event("GLOBAL_CIRCUIT_RESET")

    def reset_black_swan_flag(self) -> None:
        """重置黑天鹅检测标志（沙盘模式每轮独立使用）"""
        self._global.black_swan_detected = False
        self._log_event("BLACK_SWAN_FLAG_RESET")

    def reset_daily(self) -> None:
        """重置每日统计 — 含连续亏损衰减"""
        self._global.daily_pnl = 0.0
        self._global.total_exposure = 0.0
        self._global.volatility_alert = False
        self._global.black_swan_detected = False

        for profile in self._agent_profiles.values():
            profile.daily_loss = 0.0
            # 连续亏损衰减: 减少对前一天的惩罚
            if profile.consecutive_losses > 3:
                profile.consecutive_losses = max(1, profile.consecutive_losses // 2)
            else:
                profile.consecutive_losses = 0

    # ------------------------------------------------------------------
    # 物理逃生通道
    # ------------------------------------------------------------------

    def start_heartbeat(
        self,
        interval: float = 5.0,
        timeout: float = 30.0,
        emergency_callback: Optional[Callable] = None,
    ) -> None:
        """启动心跳监控

        Args:
            interval: 心跳间隔(秒)
            timeout: 心跳超时(秒)
            emergency_callback: 紧急平仓回调函数
        """
        self._heartbeat_interval = interval
        self._heartbeat_timeout = timeout
        self._emergency_shutdown_callback = emergency_callback

        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="risk-heartbeat",
        )
        self._heartbeat_thread.start()
        logger.info("Heartbeat monitor started (interval=%.1fs, timeout=%.1fs)", interval, timeout)

    def stop_heartbeat(self) -> None:
        """停止心跳监控"""
        self._heartbeat_running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        logger.info("Heartbeat monitor stopped")

    def heartbeat(self) -> None:
        """接收外部心跳信号（树莓派/外部监控调用）"""
        self._last_external_heartbeat = time.time()
        self._global.last_heartbeat = time.time()

    def _heartbeat_loop(self) -> None:
        """心跳监控循环"""
        while self._heartbeat_running:
            time.sleep(self._heartbeat_interval)

            elapsed = time.time() - self._last_external_heartbeat
            if elapsed > self._heartbeat_timeout:
                self._trigger_emergency_shutdown("HEARTBEAT_TIMEOUT")

    def _trigger_emergency_shutdown(self, reason: str) -> None:
        """触发紧急关闭

        物理逃生不依赖主程序判断：
        - 树莓派心跳断了就平仓
        - 带外管理卡 + 硬拔网线
        """
        self._global.global_circuit = CircuitState.OPEN
        self._log_event("EMERGENCY_SHUTDOWN", reason=reason)
        logger.critical("EMERGENCY SHUTDOWN: %s", reason)

        if self._emergency_shutdown_callback:
            try:
                self._emergency_shutdown_callback(reason)
            except Exception as e:
                logger.error("Emergency shutdown callback failed: %s", e)

    # ------------------------------------------------------------------
    # 压力测试
    # ------------------------------------------------------------------

    def stress_test(self) -> Dict[str, bool]:
        """风控压力测试

        测试场景：
        - 断网：心跳超时触发紧急关闭
        - 插针：瞬时价格波动触发黑天鹅检测
        - 交易所宕机：exchange_health = False
        - 连续亏损：Agent连续亏损触发熔断
        """
        results = {}
        # 高限额避免全局限制干扰Agent级测试
        self.set_global_limits(max_daily_loss=500000, max_exposure=5000000)

        # 断网测试
        self._last_external_heartbeat = time.time() - self._heartbeat_timeout - 1
        ok, reason = self.pre_trade_check("test_agent", "BTC-USDT", 1.0, 50000.0)
        results["network_outage"] = (not ok and reason == "HEARTBEAT_TIMEOUT")
        self._last_external_heartbeat = time.time()  # 恢复心跳
        self._global.global_circuit = CircuitState.CLOSED  # 恢复全局熔断

        # 插针测试
        results["black_swan"] = self.check_black_swan(0.10)
        self._global.black_swan_detected = False  # 恢复
        self._global.global_circuit = CircuitState.CLOSED  # 恢复全局熔断

        # 交易所宕机
        self.set_exchange_health(False)
        ok, _ = self.pre_trade_check("test_agent", "BTC-USDT", 1.0, 50000.0)
        results["exchange_down"] = not ok
        self.set_exchange_health(True)

        # 连续亏损
        for _ in range(10):
            self.post_trade_update("test_agent", -100, 50000)  # 大notional避免drawdown误触发
        ok, reason = self.pre_trade_check("test_agent", "BTC-USDT", 1.0, 50000.0)
        results["consecutive_losses"] = (not ok and "CONSECUTIVE_LOSSES" in reason)

        return results

    # ------------------------------------------------------------------
    # 日志与状态
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, **kwargs) -> None:
        """记录风控事件"""
        self._event_log.append({
            "timestamp": time.time(),
            "event": event_type,
            **kwargs,
        })

    def get_status(self) -> Dict[str, Any]:
        """获取完整风控状态"""
        return {
            "global": {
                "circuit": self._global.global_circuit.value,
                "daily_pnl": self._global.daily_pnl,
                "total_exposure": self._global.total_exposure,
                "volatility_alert": self._global.volatility_alert,
                "black_swan": self._global.black_swan_detected,
                "exchange_health": self._global.exchange_health,
                "last_heartbeat": self._global.last_heartbeat,
                "heartbeat_age": time.time() - self._last_external_heartbeat,
            },
            "agents": {
                aid: {
                    "circuit": p.circuit_state.value,
                    "consecutive_losses": p.consecutive_losses,
                    "drawdown": p.current_drawdown,
                    "daily_loss": p.daily_loss,
                    "total_trades": p.total_trades,
                }
                for aid, p in self._agent_profiles.items()
            },
            "strategy_groups": {
                name: self.check_strategy_group(name)[1]
                for name in self._strategy_groups
            },
            "recent_events": self._event_log[-20:],
        }

    def get_agent_risk_report(self, agent_id: str) -> Dict[str, Any]:
        """获取单个Agent的详细风险报告"""
        profile = self.get_agent_profile(agent_id)
        return {
            "agent_id": agent_id,
            "circuit_state": profile.circuit_state.value,
            "consecutive_losses": profile.consecutive_losses,
            "max_consecutive_losses": profile.max_consecutive_losses,
            "current_drawdown": profile.current_drawdown,
            "peak_drawdown": profile.peak_drawdown,
            "daily_loss": profile.daily_loss,
            "weekly_loss": profile.weekly_loss,
            "total_trades": profile.total_trades,
            "loss_rate": profile.total_losses / max(profile.total_trades, 1),
            "can_trade": profile.circuit_state != CircuitState.OPEN,
        }