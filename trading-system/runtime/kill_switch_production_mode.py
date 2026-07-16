"""
KillSwitchProductionModeMixin — Phase 7.5 架构解耦
从 evolution_loop.py 提取的 Kill Switch 与生产模式切换逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖:
      _check_kill_switch_active: self._kill_switch_active, self.production.kill_switch
      _is_in_final_generations: self._final_generations_no_reset, self.population.max_generations, self.population.generation
      set_production_mode: self._production_mode, self.risk_control._heartbeat_timeout
      _apply_final_generations_heartbeat: self._production_mode, self._is_in_final_generations(),
                                          self.risk_control._heartbeat_timeout, self.population.generation/max_generations
  - 属性分散保留: 所有 self 属性在 Host __init__ 中初始化, Mixin 仅访问
  - 零循环依赖: 仅依赖 logger + typing

来源:
  - Phase 7J-6 反温室修复: Kill Switch 触发后不重置 + 最后N代切换生产级心跳
  - 反温室铁律: "沙盘每轮重置 Kill Switch = 策略从未学习过触发后生存策略"
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class KillSwitchProductionModeMixin:
    """Kill Switch 与生产模式切换 Mixin (Phase 7.5 提取)

    Host 鸭子类型依赖 (属性分散在 Host __init__):
      - self._kill_switch_active: bool
      - self._final_generations_no_reset: int
      - self._production_mode: bool
      - self.production: 需有 .kill_switch.check_heartbeat() 方法
      - self.population: 需有 .max_generations (int) 和 .generation (int) 属性
      - self.risk_control: 需有 ._heartbeat_timeout (float) 属性
    """

    def _check_kill_switch_active(self) -> bool:
        """检查Kill Switch是否处于激活状态，激活时阻止新开仓"""
        if self._kill_switch_active:
            return True
        # 检查心跳超时
        if self.production.kill_switch.check_heartbeat():
            self._kill_switch_active = True
            return True
        return False

    def _is_in_final_generations(self) -> bool:
        """Phase 7J-6 反温室修复: 判断当前是否在最后 N 代

        最后 N 代不重置 Kill Switch, 让精英策略经历过"触发后停止"约束.
        来源: 反温室铁律 — "沙盘每轮重置 Kill Switch = 策略从未学习过
        Kill Switch 触发后的生存策略, 实盘部署后一旦触发会立即停止交易"

        Returns:
            True = 当前在最后 N 代 (不重置 Kill Switch)
            False = 当前不在最后 N 代 (按原逻辑重置)

        判定逻辑:
          - max_generations=0 (未设置上限) → 返回 False (无法判断"最后 N 代")
          - 当前 generation >= max_generations - final_generations_no_reset → True
        """
        if self._final_generations_no_reset <= 0:
            return False
        if self.population.max_generations <= 0:
            return False
        threshold = self.population.max_generations - self._final_generations_no_reset
        return self.population.generation >= threshold

    def set_production_mode(self, enabled: bool) -> None:
        """Phase 7J-6 反温室修复: 切换生产级心跳超时

        沙盘进化时保持 False (心跳 3600s, 避免误触发);
        实盘部署时设为 True (心跳 30s, 生产级严格).

        来源: 反温室铁律 — "沙盘心跳 3600s 放宽, 实盘 30s 严格, 进化出的策略
        从未经历过 30s 心跳约束, 实盘部署后可能因心跳超时被误触发 Kill Switch"

        Args:
            enabled: True=实盘模式 (30s 心跳), False=沙盘模式 (3600s 心跳)
        """
        self._production_mode = enabled
        if hasattr(self.risk_control, '_heartbeat_timeout'):
            if enabled:
                self.risk_control._heartbeat_timeout = 30.0
                logger.warning(
                    "Phase 7J-6: production_mode=True, heartbeat timeout=30s (生产级严格) — "
                    "确保心跳在每 30s 内被调用, 否则触发 EMERGENCY SHUTDOWN"
                )
            else:
                self.risk_control._heartbeat_timeout = 3600.0
                logger.info("Phase 7J-6: production_mode=False, heartbeat timeout=3600s (sandbox)")

    def _apply_final_generations_heartbeat(self) -> None:
        """Phase 7J-6 反温室修复: 最后 N 代切换到生产级心跳 (30s)

        在每轮开始时调用, 如果当前在最后 N 代, 则切换到 30s 心跳,
        让精英策略经历过短心跳约束.

        来源: 反温室铁律 — "进化最后阶段应模拟实盘环境, 让策略适应 30s 心跳"
        """
        if not self._production_mode and self._is_in_final_generations():
            if hasattr(self.risk_control, '_heartbeat_timeout'):
                if self.risk_control._heartbeat_timeout != 30.0:
                    self.risk_control._heartbeat_timeout = 30.0
                    logger.info(
                        "Phase 7J-6: final generations (gen=%d/%d), heartbeat=30s (生产级) — "
                        "anti-greenhouse: 让精英策略适应实盘心跳约束",
                        self.population.generation, self.population.max_generations,
                    )
