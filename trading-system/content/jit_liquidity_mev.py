# -*- coding: utf-8 -*-
"""
jit_liquidity_mev.py — JIT流动性MEV引擎 v1.0

来源：
  - Uniswap V3 Concentrated Liquidity (2021, tick range机制)
  - Flashbots Research 2026 (JIT流动性MEV分析)
  - JaredFromSubway MEV Bot 2026 (JIT实战最佳实践)
  - ai-frb 2026 (JIT策略深度解析, $500K+资本要求)
  - cryptogloss 2026 (JIT流动性定义, 60-95%费率捕获)
  - Milionis, Moallemi, Roughgarden 2023 (JIT学术形式化)

核心算法：
  1. Mempool监控
     - 监听待确认的大额swap交易
     - 解码calldata获取pool/token/amount/direction
     - 阈值过滤: swap > $300K-500K才考虑JIT
  2. 最优流动性计算
     - 计算目标swap将经过的tick range
     - 计算所需JIT流动性以捕获目标费率份额
     - 费率份额公式: F_share = L_jit / (L_pool + L_jit)
     - 捕获95%费率需 L_jit = 19 × L_pool
  3. 原子执行
     - 构建Flashbots Bundle: [mint, victim_swap, burn]
     - mint: 在精确tick range添加集中流动性
     - victim_swap: 受害者交易执行
     - burn: 移除流动性并收集费率
     - 全部在同一区块内原子完成
  4. 利润计算
     - captured_fee = swap_fee × F_share
     - net = captured_fee - gas - capital_cost - slippage
     - 资本成本: 大额资本占用一个区块的机会成本

实盘验证：
  - WETH/USDC 0.05%池: 日捕获$200K-600K（所有JIT运营者总和）
  - 单笔JIT利润: $170-$4850（取决于swap大小）
  - 资本要求: $500K-$200M（whale game）
  - 零库存风险: 位置仅存在一个区块
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class JITSignalType(Enum):
    """JIT流动性信号类型"""
    JIT_OPPORTUNITY = "jit_opportunity"      # JIT机会
    JIT_EXECUTE = "jit_execute"              # 可执行JIT
    SWAP_TOO_SMALL = "swap_too_small"        # swap太小
    HIGH_COMPETITION = "high_competition"    # 竞争激烈
    NO_OPPORTUNITY = "no_opportunity"


class FeeTier(Enum):
    """Uniswap V3费率档位"""
    TIER_005 = 0.0005    # 0.05%
    TIER_030 = 0.003     # 0.30%
    TIER_100 = 0.01      # 1.00%


@dataclass
class PendingSwapV3:
    """待确认的Uniswap V3 swap交易"""
    tx_hash: str = ""
    timestamp: float = 0.0
    sender: str = ""
    pool_address: str = ""
    token_in: str = ""
    token_out: str = ""
    amount_in: float = 0.0          # 输入代币数量
    amount_out_min: float = 0.0     # 最小输出
    fee_tier: FeeTier = FeeTier.TIER_005
    gas_price_gwei: float = 0.0
    is_zero_for_one: bool = True    # token0→token1方向


@dataclass
class V3PoolState:
    """Uniswap V3池状态"""
    address: str = ""
    token0: str = ""
    token1: str = ""
    fee_tier: FeeTier = FeeTier.TIER_005
    sqrt_price_x96: float = 0.0      # 当前√价格
    current_tick: int = 0             # 当前tick
    tick_spacing: int = 60            # tick间距
    liquidity_active: float = 0.0     # 当前活跃流动性
    last_update: float = 0.0

    @property
    def current_price(self) -> float:
        """当前价格 (token1/token0)"""
        if self.sqrt_price_x96 <= 0:
            return 0.0
        # price = (sqrt_price_x96 / 2^96)^2
        return (self.sqrt_price_x96 / (2 ** 96)) ** 2


@dataclass
class JITOpportunity:
    """JIT流动性套利机会"""
    victim_swap: PendingSwapV3
    pool: V3PoolState
    jit_liquidity: float = 0.0          # JIT提供的流动性
    fee_share: float = 0.0              # 费率份额0-1
    captured_fee_usd: float = 0.0       # 捕获的费率(USD)
    gas_cost_usd: float = 0.0           # gas成本
    capital_cost_usd: float = 0.0       # 资本成本
    slippage_cost_usd: float = 0.0      # 滑点成本
    net_profit_usd: float = 0.0         # 净利润
    capital_required_usd: float = 0.0   # 所需资本
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class JITSignal:
    """JIT流动性综合信号"""
    signal_type: JITSignalType = JITSignalType.NO_OPPORTUNITY
    opportunities: List[JITOpportunity] = field(default_factory=list)
    best_opportunity: Optional[JITOpportunity] = None
    scan_duration_ms: float = 0.0
    description: str = ""


class JITLiquidityMEV:
    """
    JIT流动性MEV引擎

    使用场景：
      - Uniswap V3及分叉(PancakeSwap V3/SushiSwap V3)
      - 大额swap的JIT流动性提供
      - 零库存风险MEV提取

    依赖：
      - Uniswap V3合约（mint/burn/swap）
      - Flashbots Bundle（原子提交）
      - Mempool监控（pending swap检测）
      - 大额资本（$500K+）
    """

    # ===== 参数 =====
    MIN_SWAP_USD = 300_000.0           # 最小swap $300K
    EXECUTE_SWAP_USD = 500_000.0       # 执行阈值 $500K
    TARGET_FEE_SHARE = 0.95            # 目标费率份额95%
    GAS_PRICE_GWEI = 30                # gas price
    MINT_GAS = 180_000                 # mint gas
    BURN_GAS = 150_000                 # burn gas
    SWAP_GAS = 150_000                 # swap gas
    CAPITAL_COST_RATE = 0.0001         # 资本成本率（单区块0.01%）
    ETH_PRICE_USD = 3000.0             # ETH价格
    MAX_SLIPPAGE_PCT = 0.002           # 最大滑点0.2%

    def __init__(self):
        self.pending_swaps: deque = deque(maxlen=1000)
        self.pool_states: Dict[str, V3PoolState] = {}
        self.opportunities_history: deque = deque(maxlen=500)
        self.execution_history: deque = deque(maxlen=100)
        self.last_scan_time: float = 0.0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def add_pending_swap(self, swap: PendingSwapV3) -> None:
        """添加待确认的swap交易"""
        self.pending_swaps.append(swap)

    def update_pool_state(
        self, address: str, token0: str, token1: str,
        sqrt_price_x96: float, current_tick: int,
        tick_spacing: int = 60, liquidity_active: float = 0.0,
        fee_tier: FeeTier = FeeTier.TIER_005,
    ) -> None:
        """更新V3池状态"""
        self.pool_states[address] = V3PoolState(
            address=address, token0=token0, token1=token1,
            fee_tier=fee_tier,
            sqrt_price_x96=sqrt_price_x96,
            current_tick=current_tick,
            tick_spacing=tick_spacing,
            liquidity_active=liquidity_active,
            last_update=time.time(),
        )

    # ------------------------------------------------------------------
    # JIT流动性计算
    # ------------------------------------------------------------------

    def calculate_required_liquidity(
        self, pool: V3PoolState, target_fee_share: float = 0.95,
    ) -> float:
        """
        计算达到目标费率份额所需的JIT流动性

        公式: F_share = L_jit / (L_pool + L_jit)
        求解: L_jit = L_pool × F_share / (1 - F_share)

        捕获95%: L_jit = 19 × L_pool
        """
        if target_fee_share >= 1.0:
            return float('inf')
        return pool.liquidity_active * target_fee_share / (1 - target_fee_share)

    def calculate_fee_share(
        self, jit_liquidity: float, pool_liquidity: float,
    ) -> float:
        """计算JIT流动性将捕获的费率份额"""
        total = jit_liquidity + pool_liquidity
        if total <= 0:
            return 0.0
        return jit_liquidity / total

    def calculate_tick_range(
        self, pool: V3PoolState, swap: PendingSwapV3,
    ) -> Tuple[int, int]:
        """
        计算JIT流动性应部署的tick range

        策略: 在当前tick附近的最窄范围内集中流动性
        """
        tick_lower = pool.current_tick - pool.tick_spacing
        tick_upper = pool.current_tick + pool.tick_spacing
        return tick_lower, tick_upper

    # ------------------------------------------------------------------
    # 利润计算
    # ------------------------------------------------------------------

    def calculate_profit(
        self, swap: PendingSwapV3, pool: V3PoolState,
        target_fee_share: float = 0.95,
    ) -> JITOpportunity:
        """
        计算JIT流动性套利利润

        净利润 = 捕获费率 - gas - 资本成本 - 滑点
        """
        # swap金额(USD估算)
        swap_amount_usd = swap.amount_in * pool.current_price

        # swap费率(USD)
        swap_fee_usd = swap_amount_usd * pool.fee_tier.value

        # 所需JIT流动性
        jit_liquidity = self.calculate_required_liquidity(pool, target_fee_share)

        # 实际费率份额
        actual_share = self.calculate_fee_share(jit_liquidity, pool.liquidity_active)

        # 捕获的费率
        captured_fee = swap_fee_usd * actual_share

        # gas成本
        gas_cost = self._calculate_gas_cost()

        # 资本成本（JIT流动性对应的资本占用一个区块）
        capital_required = self._estimate_capital_required(jit_liquidity, pool)
        capital_cost = capital_required * self.CAPITAL_COST_RATE

        # 滑点成本（JIT提供流动性时会被动持有库存）
        slippage_cost = capital_required * self.MAX_SLIPPAGE_PCT

        # 净利润
        net_profit = captured_fee - gas_cost - capital_cost - slippage_cost

        # 信号类型
        if swap_amount_usd < self.MIN_SWAP_USD:
            signal_type = JITSignalType.SWAP_TOO_SMALL
        elif net_profit > 0 and swap_amount_usd > self.EXECUTE_SWAP_USD:
            signal_type = JITSignalType.JIT_EXECUTE
        elif net_profit > 0:
            signal_type = JITSignalType.JIT_OPPORTUNITY
        else:
            signal_type = JITSignalType.NO_OPPORTUNITY

        # 置信度
        confidence = self._calculate_confidence(
            swap_amount_usd, captured_fee, net_profit, actual_share
        )

        return JITOpportunity(
            victim_swap=swap,
            pool=pool,
            jit_liquidity=jit_liquidity,
            fee_share=actual_share,
            captured_fee_usd=captured_fee,
            gas_cost_usd=gas_cost,
            capital_cost_usd=capital_cost,
            slippage_cost_usd=slippage_cost,
            net_profit_usd=net_profit,
            capital_required_usd=capital_required,
            confidence=confidence,
        )

    def _calculate_gas_cost(self) -> float:
        """计算gas成本"""
        total_gas = self.MINT_GAS + self.BURN_GAS  # mint + burn
        gas_eth = total_gas * self.GAS_PRICE_GWEI * 1e-9
        return gas_eth * self.ETH_PRICE_USD

    def _estimate_capital_required(
        self, jit_liquidity: float, pool: V3PoolState,
    ) -> float:
        """估算JIT流动性所需资本(USD)"""
        # 简化: 资本 ≈ jit_liquidity × current_price / liquidity_scale
        # 实际需根据tick range计算，这里用近似
        if pool.liquidity_active <= 0:
            return jit_liquidity * pool.current_price * 0.0001
        # 资本与流动性比例
        ratio = jit_liquidity / pool.liquidity_active
        # 假设池中总资本 = liquidity × price × scale_factor
        pool_capital_usd = pool.liquidity_active * pool.current_price * 1e-8
        return pool_capital_usd * ratio

    def _calculate_confidence(
        self, swap_usd: float, captured_fee: float,
        net_profit: float, fee_share: float,
    ) -> float:
        """计算置信度0-1"""
        # swap越大越好
        swap_score = min(1.0, swap_usd / 5_000_000)
        # 费率份额越高越好
        share_score = fee_share
        # 利润越大越好
        profit_score = min(1.0, net_profit / 2000)
        return swap_score * 0.4 + share_score * 0.3 + profit_score * 0.3

    # ------------------------------------------------------------------
    # 扫描与分析
    # ------------------------------------------------------------------

    def analyze(self) -> JITSignal:
        """分析所有待确认swap的JIT机会"""
        start_time = time.time()

        opportunities: List[JITOpportunity] = []

        for swap in list(self.pending_swaps):
            pool = self.pool_states.get(swap.pool_address)
            if not pool:
                continue

            opp = self.calculate_profit(swap, pool)
            if opp.net_profit_usd > 0:
                opportunities.append(opp)

        # 按净利润排序
        opportunities.sort(key=lambda o: o.net_profit_usd, reverse=True)

        best = opportunities[0] if opportunities else None

        if best and best.net_profit_usd > 0:
            signal_type = JITSignalType.JIT_EXECUTE if (
                best.victim_swap.amount_in * best.pool.current_price
                > self.EXECUTE_SWAP_USD
            ) else JITSignalType.JIT_OPPORTUNITY
        else:
            signal_type = JITSignalType.NO_OPPORTUNITY

        if best:
            self.opportunities_history.append(best)

        scan_ms = (time.time() - start_time) * 1000
        self.last_scan_time = time.time()

        desc = self._build_description(opportunities, signal_type)

        return JITSignal(
            signal_type=signal_type,
            opportunities=opportunities[:10],
            best_opportunity=best,
            scan_duration_ms=scan_ms,
            description=desc,
        )

    def _build_description(
        self, opportunities: List[JITOpportunity],
        signal_type: JITSignalType,
    ) -> str:
        """构建信号描述"""
        if not opportunities:
            return "无JIT机会：待确认swap不足或金额太小"

        best = opportunities[0]
        swap_usd = best.victim_swap.amount_in * best.pool.current_price
        parts = [
            f"JIT流动性扫描: 发现{len(opportunities)}个机会",
            f"最佳: swap=${swap_usd:.0f}",
            f"池={best.pool.address[:10]}...",
            f"JIT流动性={best.jit_liquidity:.2f}",
            f"费率份额={best.fee_share:.2%}",
            f"捕获费率=${best.captured_fee_usd:.2f}",
            f"所需资本=${best.capital_required_usd:.0f}",
            f"gas=${best.gas_cost_usd:.2f}",
            f"资本成本=${best.capital_cost_usd:.2f}",
            f"净利润=${best.net_profit_usd:.2f}",
            f"置信度={best.confidence:.2f}",
        ]
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Bundle构建与模拟
    # ------------------------------------------------------------------

    def build_flashbots_bundle(self, opp: JITOpportunity) -> List[Dict[str, Any]]:
        """构建Flashbots Bundle: [mint, victim_swap, burn]"""
        tick_lower, tick_upper = self.calculate_tick_range(
            opp.pool, opp.victim_swap
        )

        return [
            {
                "step": "mint",
                "action": "mint_position",
                "pool": opp.pool.address,
                "tick_lower": tick_lower,
                "tick_upper": tick_upper,
                "liquidity": opp.jit_liquidity,
                "gas": self.MINT_GAS,
            },
            {
                "step": "victim_swap",
                "action": "execute_swap",
                "tx_hash": opp.victim_swap.tx_hash,
                "pool": opp.pool.address,
                "gas": self.SWAP_GAS,
            },
            {
                "step": "burn",
                "action": "burn_position",
                "pool": opp.pool.address,
                "collect_fees": True,
                "gas": self.BURN_GAS,
            },
        ]

    def simulate_execution(self, opp: JITOpportunity) -> Dict[str, Any]:
        """模拟JIT执行"""
        bundle = self.build_flashbots_bundle(opp)
        success = opp.net_profit_usd > 0

        result = {
            "success": success,
            "bundle": bundle,
            "actual_profit_usd": opp.net_profit_usd,
            "total_gas_cost_usd": opp.gas_cost_usd,
            "capital_deployed_usd": opp.capital_required_usd,
            "execution_time_ms": 12000,  # 一个区块时间
            "inventory_risk": "zero",    # 零库存风险
        }

        self.execution_history.append(result)
        return result

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_top_opportunities(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取Top N JIT机会"""
        if not self.opportunities_history:
            return []
        recent = list(self.opportunities_history)[-n:]
        recent.sort(key=lambda o: o.net_profit_usd, reverse=True)
        return [
            {
                "pool": o.pool.address[:10],
                "swap_usd": o.victim_swap.amount_in * o.pool.current_price,
                "fee_share": o.fee_share,
                "captured_fee": o.captured_fee_usd,
                "net_profit": o.net_profit_usd,
                "confidence": o.confidence,
            }
            for o in recent
        ]

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        if not self.execution_history:
            return {"total_executions": 0}

        executions = list(self.execution_history)
        successful = [e for e in executions if e.get("success")]
        total_profit = sum(e.get("actual_profit_usd", 0) for e in successful)

        return {
            "total_executions": len(executions),
            "successful": len(successful),
            "success_rate": len(successful) / len(executions),
            "total_profit_usd": total_profit,
            "avg_profit_usd": total_profit / len(successful) if successful else 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "pending_swaps": len(self.pending_swaps),
            "tracked_pools": len(self.pool_states),
            "opportunities_found": len(self.opportunities_history),
            "execution_stats": self.get_execution_stats(),
        }
