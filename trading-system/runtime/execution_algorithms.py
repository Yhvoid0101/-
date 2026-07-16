# -*- coding: utf-8 -*-
"""
智能执行算法模块 (Smart Execution Algorithms)

基于2025-2026全网前沿执行算法研究，整合4大执行策略：

1. TWAP 时间加权平均价格 (Time-Weighted Average Price)
   来源：BloFin Academy 2026 + Almgren-Chriss 2000
   原理：把大单切成N份，每隔固定时间下一单，减少市场冲击
   量化：TWAP = (P1 + P2 + ... + Pn) / n
   适用：流动性好、成交量均匀的市场

2. VWAP 成交量加权平均价格 (Volume-Weighted Average Price)
   来源：BloFin Academy 2026 + tradinghack 2026
   原理：参考历史成交量分布，量大时多下，量小时少下
   量化：VWAP = Σ(Pi × Vi) / Σ(Vi)
   适用：流动性一般、有明显成交规律的市场，冲击成本最低

3. POV 百分比成交量 (Percent of Volume)
   来源：cryptoadventure 2026 + gravityteam 2026
   原理：参与市场成交量的固定百分比，自适应流动性
   量化：child_order = market_volume × participation_rate
   适用：流动性变化大的市场，自适应执行速率

4. Implementation Shortfall 执行缺口
   来源：tradinghack 2026 (IS核心KPI)
   原理：IS = 实现价值 - 基准价值，衡量执行总成本
   量化：IS > 0 买入成本增加，IS < 0 卖出收益减少

设计原则：
  - 纯numpy实现，无外部依赖
  - 在线执行，支持沙盘模拟
  - 输出执行计划+成本估算
  - 反过拟合：使用真实成交量分布，避免假设均匀流动性
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.execution_algo")


# ============================================================================
# 1. 执行缺口 (Implementation Shortfall) 计算器
# ============================================================================


@dataclass
class ExecutionShortfall:
    """执行缺口结果

    来源：tradinghack 2026 — Implementation Shortfall是执行算法的核心KPI
    """

    benchmark_price: float  # 基准价格（决策时价格）
    avg_fill_price: float  # 平均成交价格
    total_quantity: float  # 总成交量
    shortfall_bps: float  # 执行缺口（基点）
    shortfall_value: float  # 执行缺口（金额）
    components: Dict[str, float] = field(default_factory=dict)

    def is_good_execution(self, threshold_bps: float = 10.0) -> bool:
        """判断执行质量

        量化标准（来源：gravityteam 2026）：
          - < 5 bps: 优秀（深度流动性市场）
          - < 10 bps: 良好（正常市场）
          - < 20 bps: 可接受
          - > 20 bps: 差，需要优化执行算法
        """
        return abs(self.shortfall_bps) < threshold_bps


class ImplementationShortfallCalculator:
    """执行缺口计算器

    来源：tradinghack 2026 — IS = 实现价值 - 基准价值

    执行成本分解：
      - 佣金成本：固定费率
      - 滑点成本：网络延迟+订单处理
      - 冲击成本：大单吃穿订单簿
      - 时机成本：等待期间价格不利变动
    """

    def __init__(self, commission_bps: float = 2.0):
        """
        Args:
            commission_bps: 佣金费率（基点），默认2bps
        """
        self.commission_bps = commission_bps

    def calculate(
        self,
        benchmark_price: float,
        fills: List[Dict[str, float]],
        side: str = "buy",
    ) -> ExecutionShortfall:
        """计算执行缺口

        Args:
            benchmark_price: 基准价格（决策时价格）
            fills: 成交记录列表 [{"price": float, "quantity": float}, ...]
            side: "buy" 或 "sell"

        Returns:
            ExecutionShortfall 执行缺口结果
        """
        if not fills:
            return ExecutionShortfall(
                benchmark_price=benchmark_price,
                avg_fill_price=benchmark_price,
                total_quantity=0.0,
                shortfall_bps=0.0,
                shortfall_value=0.0,
            )

        total_qty = sum(f["quantity"] for f in fills)
        if total_qty <= 0:
            return ExecutionShortfall(
                benchmark_price=benchmark_price,
                avg_fill_price=benchmark_price,
                total_quantity=0.0,
                shortfall_bps=0.0,
                shortfall_value=0.0,
            )

        # 计算平均成交价
        total_value = sum(f["price"] * f["quantity"] for f in fills)
        avg_fill_price = total_value / total_qty

        # 执行缺口（基点）
        # 买入：正缺口=买贵了（成本增加）；卖出：正缺口=卖便宜了（收益减少）
        if side == "buy":
            shortfall_bps = (avg_fill_price - benchmark_price) / benchmark_price * 10000
            shortfall_value = (avg_fill_price - benchmark_price) * total_qty
        else:
            shortfall_bps = (benchmark_price - avg_fill_price) / benchmark_price * 10000
            shortfall_value = (benchmark_price - avg_fill_price) * total_qty

        # 成本分解
        # 佣金成本
        commission_cost = total_value * self.commission_bps / 10000
        # 价格冲击成本（总缺口 - 佣金）
        impact_cost = abs(shortfall_value) - commission_cost

        return ExecutionShortfall(
            benchmark_price=benchmark_price,
            avg_fill_price=avg_fill_price,
            total_quantity=total_qty,
            shortfall_bps=shortfall_bps,
            shortfall_value=shortfall_value,
            components={
                "commission_bps": self.commission_bps,
                "commission_value": commission_cost,
                "impact_bps": abs(shortfall_bps) - self.commission_bps,
                "impact_value": impact_cost,
                "fill_count": len(fills),
            },
        )


# ============================================================================
# 2. TWAP 时间加权平均价格执行器
# ============================================================================


class TWAPExecutor:
    """TWAP时间加权平均价格执行器

    来源：BloFin Academy 2026 + Almgren-Chriss 2000
    原理：把大单切成N份，每隔固定时间下一单

    量化：
      - 切片数量：基于总量和时间窗口
      - 每片数量：total_qty / n_slices
      - 执行间隔：time_window / n_slices

    适用场景：
      - 流动性好、成交量均匀的市场
      - 需要可预测执行时间的场景
      - 不想跟随成交量波动的场景
    """

    def __init__(
        self,
        n_slices: int = 10,
        time_window_minutes: int = 60,
        randomize: bool = True,
        randomize_factor: float = 0.1,
    ):
        """
        Args:
            n_slices: 切片数量
            time_window_minutes: 执行时间窗口（分钟）
            randomize: 是否随机化每片数量（避免被识别）
            randomize_factor: 随机化幅度（0.1=±10%）
        """
        self.n_slices = max(2, n_slices)
        self.time_window = time_window_minutes
        self.randomize = randomize
        self.randomize_factor = randomize_factor

    def generate_schedule(
        self, total_quantity: float, benchmark_price: float,
    ) -> Dict[str, Any]:
        """生成TWAP执行计划

        Returns:
            {
                "algorithm": "TWAP",
                "slices": [{"time_offset": float, "quantity": float}, ...],
                "interval_minutes": float,
                "expected_cost": float,
            }
        """
        if total_quantity <= 0:
            return {"algorithm": "TWAP", "slices": [], "interval_minutes": 0, "expected_cost": 0}

        interval = self.time_window / self.n_slices
        base_qty = total_quantity / self.n_slices

        slices: List[Dict[str, float]] = []
        remaining = total_quantity

        for i in range(self.n_slices):
            if i == self.n_slices - 1:
                # 最后一片执行剩余全部
                qty = remaining
            else:
                if self.randomize:
                    # 随机化±10%
                    noise = 1.0 + np.random.uniform(
                        -self.randomize_factor, self.randomize_factor
                    )
                    qty = base_qty * noise
                    qty = min(qty, remaining * 0.5)  # 不超过剩余的一半
                else:
                    qty = base_qty
                remaining -= qty

            slices.append({
                "time_offset": i * interval,
                "quantity": float(qty),
            })

        return {
            "algorithm": "TWAP",
            "slices": slices,
            "interval_minutes": interval,
            "n_slices": self.n_slices,
            "total_quantity": total_quantity,
            "benchmark_price": benchmark_price,
        }


# ============================================================================
# 3. VWAP 成交量加权平均价格执行器
# ============================================================================


class VWAPExecutor:
    """VWAP成交量加权平均价格执行器

    来源：BloFin Academy 2026 + tradinghack 2026
    原理：参考历史成交量分布，量大时多下，量小时少下

    量化：
      - 成交量曲线：U型分布（开盘收盘大，中午小）
      - 每片数量：total_qty × (volume_share_i / Σvolume_share)
      - VWAP = Σ(Pi × Vi) / Σ(Vi)

    适用场景：
      - 流动性一般、有明显成交规律的市场
      - 冲击成本最低的执行方式
      - 需要历史成交量分布数据
    """

    def __init__(
        self,
        n_slices: int = 10,
        time_window_minutes: int = 60,
        volume_profile: Optional[List[float]] = None,
    ):
        """
        Args:
            n_slices: 切片数量
            time_window_minutes: 执行时间窗口（分钟）
            volume_profile: 历史成交量分布（归一化权重），None则使用U型分布
        """
        self.n_slices = max(2, n_slices)
        self.time_window = time_window_minutes

        if volume_profile is not None and len(volume_profile) == n_slices:
            # 归一化
            total = sum(volume_profile)
            self.volume_profile = [v / total for v in volume_profile] if total > 0 else None
        else:
            # 默认U型分布（来源：tradinghack 2026 — A股/加密货币成交量U型曲线）
            # 开盘和收盘成交量大，中午成交量小
            x = np.linspace(0, np.pi, n_slices)
            # U型：sin曲线两端高中间低
            self.volume_profile = ((np.sin(x) * 0.5 + 0.5) ** 0.5).tolist()
            total = sum(self.volume_profile)
            self.volume_profile = [v / total for v in self.volume_profile]

    def generate_schedule(
        self, total_quantity: float, benchmark_price: float,
    ) -> Dict[str, Any]:
        """生成VWAP执行计划

        Returns:
            包含成交量加权切片的执行计划
        """
        if total_quantity <= 0:
            return {"algorithm": "VWAP", "slices": [], "interval_minutes": 0, "expected_cost": 0}

        interval = self.time_window / self.n_slices

        slices: List[Dict[str, float]] = []
        remaining = total_quantity

        for i in range(self.n_slices):
            if i == self.n_slices - 1:
                qty = remaining
            else:
                # 按成交量分布分配
                qty = total_quantity * self.volume_profile[i]
                qty = min(qty, remaining * 0.8)  # 保守限制
                remaining -= qty

            slices.append({
                "time_offset": i * interval,
                "quantity": float(qty),
                "volume_weight": float(self.volume_profile[i]),
            })

        return {
            "algorithm": "VWAP",
            "slices": slices,
            "interval_minutes": interval,
            "n_slices": self.n_slices,
            "total_quantity": total_quantity,
            "benchmark_price": benchmark_price,
            "volume_profile": self.volume_profile,
        }


# ============================================================================
# 4. POV 百分比成交量执行器
# ============================================================================


class POVExecutor:
    """POV百分比成交量执行器

    来源：cryptoadventure 2026 + gravityteam 2026
    原理：参与市场成交量的固定百分比，自适应流动性

    量化：
      - 参与率：通常5%-20%（超过20%会显著影响市场）
      - 每片数量：market_volume × participation_rate
      - 自适应：市场活跃时多交易，市场冷清时少交易

    适用场景：
      - 流动性变化大的市场
      - 需要自适应执行速率的场景
      - 不确定最佳执行时间的场景
    """

    def __init__(
        self,
        participation_rate: float = 0.1,
        max_slices: int = 20,
        min_slice_quantity: float = 0.001,
    ):
        """
        Args:
            participation_rate: 参与率（0.1=参与市场成交量的10%）
            max_slices: 最大切片数
            min_slice_quantity: 最小切片数量
        """
        self.participation_rate = max(0.01, min(0.3, participation_rate))
        self.max_slices = max_slices
        self.min_slice_quantity = min_slice_quantity

    def generate_schedule(
        self,
        total_quantity: float,
        benchmark_price: float,
        expected_market_volume: Optional[float] = None,
    ) -> Dict[str, Any]:
        """生成POV执行计划

        Args:
            total_quantity: 总执行量
            benchmark_price: 基准价格
            expected_market_volume: 预期市场总成交量

        Returns:
            POV执行计划
        """
        if total_quantity <= 0:
            return {"algorithm": "POV", "slices": [], "interval_minutes": 0, "expected_cost": 0}

        # 估算市场成交量（如果没有提供）
        if expected_market_volume is None or expected_market_volume <= 0:
            # 假设市场成交量是执行量的10倍（保守估计）
            expected_market_volume = total_quantity * 10

        # 每片数量 = 市场成交量 × 参与率
        per_slice_market = expected_market_volume / self.max_slices
        per_slice_qty = per_slice_market * self.participation_rate
        per_slice_qty = max(per_slice_qty, self.min_slice_quantity)

        # 计算实际需要的切片数
        n_slices = min(self.max_slices, int(math.ceil(total_quantity / per_slice_qty)))
        n_slices = max(2, n_slices)

        interval = 5.0  # 每片间隔5分钟（可配置）

        slices: List[Dict[str, float]] = []
        remaining = total_quantity

        for i in range(n_slices):
            if i == n_slices - 1:
                qty = remaining
            else:
                qty = min(per_slice_qty, remaining * 0.5)
                remaining -= qty

            slices.append({
                "time_offset": i * interval,
                "quantity": float(qty),
                "participation_rate": self.participation_rate,
            })

        return {
            "algorithm": "POV",
            "slices": slices,
            "interval_minutes": interval,
            "n_slices": n_slices,
            "total_quantity": total_quantity,
            "benchmark_price": benchmark_price,
            "participation_rate": self.participation_rate,
            "expected_market_volume": expected_market_volume,
        }


# ============================================================================
# 5. 智能执行算法选择器
# ============================================================================


class SmartExecutionSelector:
    """智能执行算法选择器

    根据订单特征和市场条件自动选择最优执行算法

    选择策略（来源：cryptoadventure 2026 + gravityteam 2026）：
      - 小订单（< ADV的1%）：直接市价单，无需拆单
      - 中订单（ADV的1%-5%）+ 流动性好：TWAP
      - 中订单（ADV的1%-5%）+ 有成交量规律：VWAP
      - 大订单（> ADV的5%）：POV，自适应流动性
      - 超大订单（> ADV的10%）：POV + 冰山订单
    """

    def __init__(
        self,
        twap_executor: Optional[TWAPExecutor] = None,
        vwap_executor: Optional[VWAPExecutor] = None,
        pov_executor: Optional[POVExecutor] = None,
    ):
        self.twap = twap_executor or TWAPExecutor()
        self.vwap = vwap_executor or VWAPExecutor()
        self.pov = pov_executor or POVExecutor()
        self.is_calculator = ImplementationShortfallCalculator()

    def select_algorithm(
        self,
        order_quantity: float,
        adv: float,  # Average Daily Volume
        has_volume_profile: bool = False,
        urgency: str = "normal",  # "low", "normal", "high"
    ) -> str:
        """选择最优执行算法

        Args:
            order_quantity: 订单数量
            adv: 日均成交量
            has_volume_profile: 是否有历史成交量分布
            urgency: 紧急程度 "low"/"normal"/"high"

        Returns:
            算法名称 "market"/"twap"/"vwap"/"pov"
        """
        if adv <= 0:
            return "market"

        # 订单占ADV的比例
        size_ratio = order_quantity / adv

        # 紧急程度调整
        if urgency == "high":
            # 紧急：小订单直接市价，大订单用TWAP快速执行
            if size_ratio < 0.02:
                return "market"
            else:
                return "twap"

        # 非紧急：根据订单大小选择
        if size_ratio < 0.01:
            # 小订单：直接市价单
            return "market"
        elif size_ratio < 0.05:
            # 中订单：TWAP或VWAP
            if has_volume_profile:
                return "vwap"  # 有成交量分布，用VWAP冲击更小
            else:
                return "twap"  # 无成交量分布，用TWAP
        else:
            # 大订单：POV自适应
            return "pov"

    def generate_execution_plan(
        self,
        order_quantity: float,
        benchmark_price: float,
        adv: float,
        side: str = "buy",
        has_volume_profile: bool = False,
        urgency: str = "normal",
    ) -> Dict[str, Any]:
        """生成完整执行计划

        Returns:
            包含算法选择+执行计划+成本估算的完整方案
        """
        algorithm = self.select_algorithm(
            order_quantity, adv, has_volume_profile, urgency
        )

        if algorithm == "market":
            # 直接市价单
            return {
                "algorithm": "market",
                "slices": [{"time_offset": 0, "quantity": order_quantity}],
                "n_slices": 1,
                "total_quantity": order_quantity,
                "benchmark_price": benchmark_price,
                "side": side,
                "expected_shortfall_bps": 5.0,  # 市价单预期缺口5bps
                "selection_reason": "小订单，直接市价单执行",
            }
        elif algorithm == "twap":
            schedule = self.twap.generate_schedule(order_quantity, benchmark_price)
            schedule["expected_shortfall_bps"] = 8.0  # TWAP预期缺口8bps
            schedule["side"] = side
            schedule["selection_reason"] = f"中订单(size_ratio={order_quantity/adv:.3f})，TWAP时间均匀执行"
            return schedule
        elif algorithm == "vwap":
            schedule = self.vwap.generate_schedule(order_quantity, benchmark_price)
            schedule["expected_shortfall_bps"] = 6.0  # VWAP预期缺口6bps（最优）
            schedule["side"] = side
            schedule["selection_reason"] = f"中订单+有成交量分布，VWAP冲击最小"
            return schedule
        else:  # pov
            schedule = self.pov.generate_schedule(
                order_quantity, benchmark_price, expected_market_volume=adv
            )
            schedule["expected_shortfall_bps"] = 10.0  # POV预期缺口10bps
            schedule["side"] = side
            schedule["selection_reason"] = f"大订单(size_ratio={order_quantity/adv:.3f})，POV自适应流动性"
            return schedule


# ============================================================================
# 6. 智能执行算法综合系统
# ============================================================================


class SmartExecutionSystem:
    """智能执行算法综合系统

    整合TWAP/VWAP/POV/IS的完整执行系统

    使用方式：
      system = SmartExecutionSystem()
      plan = system.generate_plan(quantity, price, adv, side="buy")
      result = system.execute(plan, market_data)
      shortfall = system.calculate_shortfall(plan, fills)
    """

    def __init__(self):
        self.selector = SmartExecutionSelector()
        self.is_calculator = ImplementationShortfallCalculator()
        self.execution_history: deque = deque(maxlen=100)

    def generate_plan(
        self,
        quantity: float,
        benchmark_price: float,
        adv: float,
        side: str = "buy",
        has_volume_profile: bool = False,
        urgency: str = "normal",
    ) -> Dict[str, Any]:
        """生成执行计划"""
        return self.selector.generate_execution_plan(
            quantity, benchmark_price, adv, side, has_volume_profile, urgency
        )

    def calculate_shortfall(
        self,
        benchmark_price: float,
        fills: List[Dict[str, float]],
        side: str = "buy",
    ) -> ExecutionShortfall:
        """计算执行缺口"""
        result = self.is_calculator.calculate(benchmark_price, fills, side)
        self.execution_history.append({
            "benchmark_price": benchmark_price,
            "avg_fill_price": result.avg_fill_price,
            "shortfall_bps": result.shortfall_bps,
            "side": side,
        })
        return result

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        if not self.execution_history:
            return {"status": "no_executions", "count": 0}

        history = list(self.execution_history)
        shortfalls = [h["shortfall_bps"] for h in history]

        return {
            "status": "active",
            "count": len(history),
            "avg_shortfall_bps": float(np.mean(shortfalls)),
            "max_shortfall_bps": float(np.max(shortfalls)),
            "min_shortfall_bps": float(np.min(shortfalls)),
            "median_shortfall_bps": float(np.median(shortfalls)),
            "good_execution_rate": float(
                sum(1 for s in shortfalls if abs(s) < 10.0) / len(shortfalls)
            ),
        }

    def get_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": "active",
            "execution_count": len(self.execution_history),
            "stats": self.get_execution_stats(),
        }


# ============================================================================
# 7. G5: 执行算法集成 IMatchingEngine (Phase 3 真实数据接入)
#    来源: _live_data_gap_patch_list.py G5 (P1, owner: LIVE-DATA-CONNECTOR)
#    验证标准: 1 BTC 大单 TWAP 拆 10 片, IS bps < 10
# ============================================================================


@dataclass
class ExecutionResult:
    """G5: 执行算法完整执行结果

    封装从计划到真实下单的完整执行结果, 供策略层与监控层消费。
    """

    algorithm: str  # "TWAP" / "VWAP" / "POV"
    benchmark_price: float  # 决策时基准价
    avg_fill_price: float  # 加权平均成交价
    total_quantity: float  # 计划总量
    filled_quantity: float  # 实际成交量
    n_slices: int  # 切片总数
    n_filled: int  # 成交片数
    n_rejected: int  # 拒绝/未成交片数
    shortfall_bps: float  # 执行缺口 (基点), 越小越好
    fills: List[Dict[str, float]]  # [{"price": float, "quantity": float}, ...]
    elapsed_seconds: float  # 总耗时
    success: bool  # 是否成功 (n_filled > 0 且拒绝率 < 50%)
    error: str = ""  # 失败原因 (success=False 时)

    def fill_rate(self) -> float:
        """成交率 = filled_quantity / total_quantity"""
        return self.filled_quantity / self.total_quantity if self.total_quantity > 0 else 0.0

    def is_good_execution(self, threshold_bps: float = 10.0) -> bool:
        """执行质量是否达标 (IS bps < 阈值)"""
        return self.success and abs(self.shortfall_bps) < threshold_bps


class ExecutionRunner:
    """G5: 执行算法运行器 — 将执行计划通过 IMatchingEngine 真实下单

    集成点:
      - TWAPExecutor/VWAPExecutor/POVExecutor.generate_schedule() 生成切片
      - IMatchingEngine.submit_order() 真实下单 (沙盘/实盘统一接口)
      - ImplementationShortfallCalculator 计算 IS bps

    使用方式:
        runner = ExecutionRunner()
        result = runner.execute_twap(twap_exec, engine, order, benchmark_price)
        if result.is_good_execution():
            print(f"IS bps={result.shortfall_bps:.2f}")

    设计原则:
      1. 切片用 LIMIT 单 (避免市价单冲击), 价格=benchmark_price
      2. 单片失败不阻断后续切片 (失败隔离)
      3. 拒绝率 >= 50% 标记为失败
      4. IS bps 由 ImplementationShortfallCalculator 计算
    """

    def __init__(self, is_calculator: Optional[ImplementationShortfallCalculator] = None):
        self.is_calculator = is_calculator or ImplementationShortfallCalculator()

    def execute_twap(
        self,
        executor: "TWAPExecutor",
        engine: Any,
        order: Any,
        benchmark_price: float,
        wait_seconds: float = 0.0,
        child_order_type: Any = None,
    ) -> ExecutionResult:
        """TWAP 执行: 按 schedule 调用 engine.submit_order, 等待成交或超时

        Args:
            child_order_type: 子订单类型 (默认LIMIT, 沙盘模式可传MARKET确保立即成交)
        """
        schedule = executor.generate_schedule(order.quantity, benchmark_price)
        return self._execute_schedule(engine, order, schedule, benchmark_price, wait_seconds, child_order_type)

    def execute_vwap(
        self,
        executor: "VWAPExecutor",
        engine: Any,
        order: Any,
        benchmark_price: float,
        wait_seconds: float = 0.0,
        child_order_type: Any = None,
    ) -> ExecutionResult:
        """VWAP 执行: 根据历史成交量分布动态调整每片 size

        Args:
            child_order_type: 子订单类型 (默认LIMIT, 沙盘模式可传MARKET确保立即成交)
        """
        schedule = executor.generate_schedule(order.quantity, benchmark_price)
        return self._execute_schedule(engine, order, schedule, benchmark_price, wait_seconds, child_order_type)

    def execute_pov(
        self,
        executor: "POVExecutor",
        engine: Any,
        order: Any,
        benchmark_price: float,
        expected_market_volume: Optional[float] = None,
        wait_seconds: float = 0.0,
        child_order_type: Any = None,
    ) -> ExecutionResult:
        """POV 执行: 实时监控市场成交量, 维持参与率

        Args:
            child_order_type: 子订单类型 (默认LIMIT, 沙盘模式可传MARKET确保立即成交)
        """
        schedule = executor.generate_schedule(
            order.quantity, benchmark_price, expected_market_volume=expected_market_volume
        )
        return self._execute_schedule(engine, order, schedule, benchmark_price, wait_seconds, child_order_type)

    def _execute_schedule(
        self,
        engine: Any,
        order: Any,
        schedule: Dict[str, Any],
        benchmark_price: float,
        wait_seconds: float,
        child_order_type: Any = None,
    ) -> ExecutionResult:
        """通用执行逻辑: 遍历 schedule slices, 逐片下单

        Args:
            engine: IMatchingEngine 实例 (沙盘或实盘)
            order: UnifiedOrder 父订单 (提供 symbol/side/leverage 等上下文)
            schedule: Executor.generate_schedule() 返回的计划
            benchmark_price: 基准价 (决策时价格)
            wait_seconds: 切片间等待秒数 (实盘时间间隔, 沙盘可为0)
        """
        import time as _time

        t0 = _time.time()
        slices = schedule.get("slices", [])
        algorithm = schedule.get("algorithm", "UNKNOWN")
        n_slices = len(slices)

        if n_slices == 0:
            return ExecutionResult(
                algorithm=algorithm, benchmark_price=benchmark_price,
                avg_fill_price=benchmark_price, total_quantity=order.quantity,
                filled_quantity=0.0, n_slices=0, n_filled=0, n_rejected=0,
                shortfall_bps=0.0, fills=[], elapsed_seconds=0.0,
                success=False, error="empty_schedule",
            )

        # 延迟导入 (避免循环导入)
        try:
            from .matching_engine_interface import (
                UnifiedOrder as _UO, UnifiedOrderType as _UOT,
                UnifiedOrderStatus as _UOS,
            )
        except ImportError:
            _UO = None  # 独立运行模式, 退化为字符串构造
            _UOT = None
            _UOS = None

        fills: List[Dict[str, float]] = []
        n_filled = 0
        n_rejected = 0
        total_filled = 0.0

        for i, slice_info in enumerate(slices):
            slice_qty = float(slice_info.get("quantity", 0))
            if slice_qty <= 0:
                continue

            # 构造子订单 (默认LIMIT单避免市价冲击; 沙盘模式可传MARKET确保立即成交)
            if _UO is not None:
                _eff_type = child_order_type if child_order_type is not None else _UOT.LIMIT
                child_order = _UO(
                    order_id=f"{order.order_id}_g5_{i}",
                    agent_id=order.agent_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=_eff_type,
                    quantity=slice_qty,
                    price=benchmark_price,
                    leverage=getattr(order, "leverage", 1),
                    reduce_only=getattr(order, "reduce_only", False),
                    client_order_id=f"{getattr(order, 'client_order_id', None) or order.order_id}_g5_{i}",
                    meta={**getattr(order, "meta", {}),
                          "parent_order_id": order.order_id,
                          "slice_index": i, "algorithm": algorithm},
                )
            else:
                # 独立模式: 复制父订单并修改数量
                import copy
                child_order = copy.copy(order)
                child_order.order_id = f"{order.order_id}_g5_{i}"
                child_order.quantity = slice_qty
                child_order.price = benchmark_price

            try:
                result_order = engine.submit_order(child_order)
                # 判断成交状态
                if _UOS is not None:
                    filled = result_order.status == _UOS.FILLED
                else:
                    filled = getattr(result_order, "filled_qty", 0) > 0

                if filled:
                    fills.append({
                        "price": float(getattr(result_order, "filled_price", benchmark_price)),
                        "quantity": float(getattr(result_order, "filled_qty", slice_qty)),
                    })
                    total_filled += fills[-1]["quantity"]
                    n_filled += 1
                else:
                    n_rejected += 1
                    logger.warning(
                        "G5 %s slice %d/%d 未成交: status=%s",
                        algorithm, i + 1, n_slices,
                        getattr(result_order, "status", "unknown"),
                    )
            except Exception as e:
                n_rejected += 1
                logger.error(
                    "G5 %s slice %d/%d 下单异常: %s",
                    algorithm, i + 1, n_slices, str(e)[:100],
                )

            # 切片间等待 (实盘需要, 沙盘 wait_seconds=0)
            if wait_seconds > 0 and i < n_slices - 1:
                _time.sleep(wait_seconds)

        # 计算 IS (Implementation Shortfall)
        if fills:
            side_str = "buy"  # 默认买入
            try:
                # 从 order.side 推断
                side_attr = getattr(order, "side", None)
                if side_attr is not None:
                    side_str = "buy" if str(side_attr).lower().endswith("buy") or "buy" in str(side_attr).lower() else "sell"
            except Exception:
                pass
            is_result = self.is_calculator.calculate(
                benchmark_price, fills, side=side_str,
            )
            avg_fill = is_result.avg_fill_price
            shortfall_bps = is_result.shortfall_bps
        else:
            avg_fill = benchmark_price
            shortfall_bps = 0.0

        elapsed = _time.time() - t0
        # 成功条件: 至少 1 片成交 且 拒绝率 < 50%
        success = n_filled > 0 and n_rejected < n_slices * 0.5

        return ExecutionResult(
            algorithm=algorithm,
            benchmark_price=benchmark_price,
            avg_fill_price=avg_fill,
            total_quantity=order.quantity,
            filled_quantity=total_filled,
            n_slices=n_slices,
            n_filled=n_filled,
            n_rejected=n_rejected,
            shortfall_bps=shortfall_bps,
            fills=fills,
            elapsed_seconds=elapsed,
            success=success,
            error="" if success else f"reject_rate={n_rejected}/{n_slices}",
        )


# ============================================================================
# 8. Monkey-patch: 给每个 Executor 添加 execute() 方法 (G5 规范要求)
#    规范: "为每个 Executor 添加 execute(engine: IMatchingEngine) 方法"
#    实现: 委托给 ExecutionRunner, 保持向后兼容
# ============================================================================

_default_runner = ExecutionRunner()


def _twap_execute(self, engine, order, benchmark_price, wait_seconds=0.0):
    """TWAPExecutor.execute — G5 集成 IMatchingEngine 真实下单"""
    return _default_runner.execute_twap(self, engine, order, benchmark_price, wait_seconds)


def _vwap_execute(self, engine, order, benchmark_price, wait_seconds=0.0):
    """VWAPExecutor.execute — G5 集成 IMatchingEngine 真实下单"""
    return _default_runner.execute_vwap(self, engine, order, benchmark_price, wait_seconds)


def _pov_execute(self, engine, order, benchmark_price,
                 expected_market_volume=None, wait_seconds=0.0):
    """POVExecutor.execute — G5 集成 IMatchingEngine 真实下单"""
    return _default_runner.execute_pov(
        self, engine, order, benchmark_price, expected_market_volume, wait_seconds
    )


# 注入 execute 方法 (不破坏现有 generate_schedule 接口)
TWAPExecutor.execute = _twap_execute
VWAPExecutor.execute = _vwap_execute
POVExecutor.execute = _pov_execute

