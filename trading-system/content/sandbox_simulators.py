# -*- coding: utf-8 -*-
"""
sandbox_simulators.py — Phase 6C+6D 沙盘模拟器集合

为 11 个孤立策略模块提供数据源的统一模拟器模块。每个模拟器接收沙盘当前
价格/状态，生成对应策略所需的结构化数据（IV/AMM池/跨链桥/DEX报价/巨鲸
交易流/期权链），使原本依赖外部 API 的策略模块能在纯沙盘环境中运行。

设计原则：
  1. 单文件包含所有模拟器类，避免文件膨胀
  2. 每个模拟器提供 ``tick(prices, timestamp)`` 方法接受沙盘当前价格
  3. 模拟器内部维护状态，每次 tick 生成新数据
  4. 纯 Python + numpy 实现，不依赖外部库（除了 numpy）
  5. 所有随机数使用 ``random.Random(seed)`` 实例，确保沙盘可复现
  6. 每个模拟器内部维护 ``deque(maxlen=100)`` 历史用于指标计算

支持的策略模块与模拟器映射：
  - volatility_arbitrage.py / vrp_harvesting.py        → SimulatedIVFeed
  - mev_sandwich_arbitrage.py / jit_liquidity_mev.py   → SimulatedAMMPool
  - cross_chain_arbitrage.py                            → SimulatedBridge
  - cex_dex_flash_arbitrage.py                          → SimulatedDEX
  - on_chain_whale_tracker.py                           → SimulatedWhaleFlow
  - volatility_surface_arb.py / vanna_volga_arbitrage.py → SimulatedOptionsMarket

算法来源：
  - GARCH(1,1): Engle 1982 / Bollerslev 1986（IV 路径生成）
  - CPMM 常数乘积: Uniswap V2 / Curve（AMM 池与 DEX 报价）
  - 泊松过程: Kingman 1993（pending swap / 巨鲸交易流）
  - BSM 期权定价: Black-Scholes 1973
  - SVI 微笑曲线: Gatheral 2004（波动率曲面参数化）
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# ============================================================================
# 模块级辅助函数
# ============================================================================


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """基于 ``random.Random`` 实例的泊松采样（Knuth 算法）

    ``random.Random`` 没有内置 ``poisson`` 方法（numpy 才有），这里用
    Knuth 算法实现，确保所有随机数都来自可种子化的 ``random.Random``
    实例，沙盘可复现。

    对于 ``lam`` 较大（>30）的情况，使用正态近似以保证性能。

    Args:
        rng: ``random.Random`` 实例
        lam: 泊松到达率（期望事件数）

    Returns:
        采样得到的事件数（非负整数）
    """
    if lam <= 0.0:
        return 0
    # 大 lambda 用正态近似（保证性能，lam>30 时误差可忽略）
    if lam > 30.0:
        # Box-Muller 生成标准正态
        u1 = rng.random() or 1e-12
        u2 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return max(0, int(round(z * math.sqrt(lam) + lam)))
    # Knuth 算法（小 lambda 精确）
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


# ============================================================================
# 1. SimulatedIVFeed — 隐含波动率数据源
# ============================================================================


class SimulatedIVFeed:
    """隐含波动率（IV）模拟数据源

    为 ``volatility_arbitrage.py`` 和 ``vrp_harvesting.py`` 提供隐含波动率
    (IV)、实现波动率 (RV)、波动率的波动率 (vvol) 三项数据。

    算法：
      - IV 路径：GARCH(1,1) 简化版生成时变 IV，捕捉波动率聚集
      - RV：由价格历史滚动计算对数收益率标准差（年化）
      - vvol：RV 序列的滚动标准差，衡量波动率自身的不稳定性

    典型用法：
      >>> feed = SimulatedIVFeed("BTC-USDT", initial_iv=0.6)
      >>> data = feed.tick(50000.0, time.time())
      >>> data["iv"], data["rv"], data["vvol"]
    """

    # GARCH(1,1) 参数（年化 IV 序列）
    _OMEGA: float = 0.00002   # 长期方差率
    _ALPHA: float = 0.08      # 新息冲击系数
    _BETA: float = 0.88       # 波动率持续性
    _ANNUALIZE: float = math.sqrt(365 * 24)  # 小时频率年化因子

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        initial_iv: float = 0.6,
        seed: int = 42,
    ) -> None:
        """初始化 IV 数据源

        Args:
            symbol: 标的符号，仅用于标识
            initial_iv: 初始隐含波动率（年化小数，0.6 = 60%）
            seed: 随机种子，确保沙盘可复现
        """
        self.symbol: str = symbol
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)

        # 当前 IV 状态（方差形式，年化）
        self._iv: float = max(initial_iv, 1e-4)
        self._iv_var: float = self._iv ** 2

        # 价格与 RV 历史
        self._price_history: deque = deque(maxlen=100)
        self._rv_history: deque = deque(maxlen=100)
        self._current_rv: float = self._iv
        self._current_vvol: float = 0.0
        self._last_price: Optional[float] = None

    def tick(self, price: float, timestamp: float) -> Dict[str, float]:
        """推进一个 tick，生成新的 IV/RV/vvol 数据

        Args:
            price: 标的当前价格
            timestamp: 当前时间戳（秒）

        Returns:
            包含 ``iv``/``rv``/``vvol`` 三个字段的字典（均为年化小数）
        """
        # ---- 1. GARCH(1,1) 推进 IV 方差 ----
        # 新息：以当前 IV 为尺度的标准正态扰动
        shock = self._rng.gauss(0.0, 1.0)
        eps = shock * math.sqrt(max(self._iv_var, 1e-8))
        self._iv_var = (
            self._OMEGA
            + self._ALPHA * (eps ** 2)
            + self._BETA * self._iv_var
        )
        # 防止方差爆炸或归零
        self._iv_var = max(min(self._iv_var, 4.0), 1e-8)
        self._iv = math.sqrt(self._iv_var)

        # ---- 2. RV 滚动计算 ----
        self._price_history.append(price)
        if len(self._price_history) >= 2:
            returns: List[float] = []
            prices = list(self._price_history)
            for i in range(1, len(prices)):
                if prices[i - 1] > 0:
                    returns.append(math.log(prices[i] / prices[i - 1]))
            if len(returns) >= 2:
                # 年化 RV
                self._current_rv = float(np.std(returns, ddof=1)) * self._ANNUALIZE
                self._rv_history.append(self._current_rv)

        # ---- 3. vvol = RV 序列的滚动标准差 ----
        if len(self._rv_history) >= 2:
            self._current_vvol = float(np.std(list(self._rv_history), ddof=1))

        self._last_price = price
        return {
            "iv": float(self._iv),
            "rv": float(self._current_rv),
            "vvol": float(self._current_vvol),
        }


# ============================================================================
# 2. SimulatedAMMPool — AMM 池 + pending swap 数据源
# ============================================================================


class SimulatedAMMPool:
    """链上 AMM 池与 pending swap 模拟器

    为 ``mev_sandwich_arbitrage.py`` 和 ``jit_liquidity_mev.py`` 提供常数
    乘积做市商 (CPMM) 池状态以及基于泊松过程的 pending swap 流。

    算法：
      - CPMM 模型：reserve_a * reserve_b = k（常数乘积不变量）
      - 池价格根据外部沙盘价格锚定，缓慢再平衡储备量
      - pending swap：泊松过程生成，lambda 与价格波动正相关

    典型用法：
      >>> pool = SimulatedAMMPool("BTC-USDT", 100.0, 5_000_000.0)
      >>> state = pool.tick(50000.0, time.time())
      >>> state["pool_state"]["reserve_a"], state["pending_swaps"]
    """

    _FEE_TIER: float = 0.003  # 0.3% 手续费
    _SWAP_LAMBDA_BASE: float = 2.0  # 每 tick 基础泊松到达率
    _ADDRESSES: Tuple[str, ...] = (
        "0x Whale A", "0x Whale B", "0x Retail 1", "0x Retail 2",
        "0x Arb Bot", "0x MEV Bot", "0x Random 1", "0x Random 2",
    )

    def __init__(
        self,
        base_symbol: str = "BTC-USDT",
        pool_reserve_a: float = 100.0,
        pool_reserve_b: float = 5_000_000.0,
        seed: int = 42,
    ) -> None:
        """初始化 AMM 池

        Args:
            base_symbol: 池基础交易对符号（如 "BTC-USDT"）
            pool_reserve_a: token_a（基础币，如 BTC）初始储备量
            pool_reserve_b: token_b（报价币，如 USDT）初始储备量
            seed: 随机种子
        """
        self.base_symbol: str = base_symbol
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)

        self._reserve_a: float = float(pool_reserve_a)
        self._reserve_b: float = float(pool_reserve_b)
        self._k: float = self._reserve_a * self._reserve_b
        self._last_price: float = (
            self._reserve_b / self._reserve_a if self._reserve_a > 0 else 0.0
        )
        self._price_history: deque = deque(maxlen=100)

    def tick(self, price: float, timestamp: float) -> Dict[str, Any]:
        """推进一个 tick，返回池状态和本 tick 生成的 pending swap 列表

        Args:
            price: 当前沙盘价格（token_b/token_a 报价）
            timestamp: 当前时间戳（秒）

        Returns:
            ``{"pool_state": {...}, "pending_swaps": [...]}``
            pool_state 字段：reserve_a/reserve_b/price/fee_tier/timestamp
            pending_swaps 中每个 swap：
            token_in/token_out/amount_in/sender/timestamp
        """
        # ---- 1. 池价格根据外部价格缓慢再平衡 ----
        # 当外部价格偏离池价时，套利者会调整储备，向 k 不变的方向移动
        if price > 0 and self._last_price > 0:
            # 价格冲击：根据外部价格调整储备比例（保持 k 不变）
            price_ratio = price / self._last_price
            # 调整 a/b 储备使池价向外部价格靠拢（部分再平衡）
            adjust = 0.5  # 50% 套利再平衡
            target_a = math.sqrt(self._k / price) if price > 0 else self._reserve_a
            target_b = self._k / target_a if target_a > 0 else self._reserve_b
            self._reserve_a = self._reserve_a * (1 - adjust) + target_a * adjust
            self._reserve_b = self._reserve_b * (1 - adjust) + target_b * adjust
        self._last_price = price
        self._price_history.append(price)

        # ---- 2. 泊松过程生成 pending swap ----
        # 到达率随价格波动增加
        volatility_factor = 1.0
        if len(self._price_history) >= 2:
            recent = list(self._price_history)[-10:]
            if len(recent) >= 2 and recent[-2] > 0:
                ret = abs(math.log(recent[-1] / recent[-2]))
                volatility_factor = 1.0 + min(ret * 50.0, 3.0)
        lam = self._SWAP_LAMBDA_BASE * volatility_factor
        n_swaps = _poisson_sample(self._rng, lam)

        pending_swaps: List[Dict[str, Any]] = []
        for _ in range(n_swaps):
            # 随机方向：50% a→b, 50% b→a
            if self._rng.random() < 0.5:
                token_in, token_out = "a", "b"
                # amount_in 以 token_a 计，最大不超过储备的 1%
                max_in = self._reserve_a * 0.01
                amount_in = self._rng.uniform(0.001, max(max_in, 0.001))
            else:
                token_in, token_out = "b", "a"
                max_in = self._reserve_b * 0.01
                amount_in = self._rng.uniform(1.0, max(max_in, 1.0))
            sender = self._rng.choice(self._ADDRESSES)
            pending_swaps.append({
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": float(amount_in),
                "sender": sender,
                "timestamp": float(timestamp),
            })

        pool_price = (
            self._reserve_b / self._reserve_a if self._reserve_a > 0 else 0.0
        )
        return {
            "pool_state": {
                "reserve_a": float(self._reserve_a),
                "reserve_b": float(self._reserve_b),
                "price": float(pool_price),
                "fee_tier": self._FEE_TIER,
                "timestamp": float(timestamp),
            },
            "pending_swaps": pending_swaps,
        }


# ============================================================================
# 3. SimulatedBridge — 跨链桥状态数据源
# ============================================================================


class SimulatedBridge:
    """跨链桥状态模拟器

    为 ``cross_chain_arbitrage.py`` 提供多条跨链桥的延迟、桥费、流动性
    状态，以及同一资产在不同链上的价格快照。

    算法：
      - 6 个默认桥：Across / Stargate / Hop / Synapse / Celer / LayerZero
      - 桥延迟 5-30 分钟（每 tick 随机扰动 ±10%）
      - 桥费 0.1%-0.5%（每 tick 随机扰动 ±5%）
      - 流动性 USD：初始值 + 随机漂移
      - 状态：normal / congested / degraded 三态，按概率切换

    典型用法：
      >>> bridge = SimulatedBridge(chains=["ethereum", "arbitrum", "optimism"])
      >>> state = bridge.tick({"ethereum": {"BTC": 50000.0}}, time.time())
    """

    _DEFAULT_CHAINS: List[str] = [
        "ethereum", "arbitrum", "optimism", "base", "polygon", "bsc",
    ]
    _DEFAULT_BRIDGES: List[Dict[str, Any]] = [
        {"name": "Across",    "delay_minutes": 5.0,  "fee_pct": 0.10, "liquidity_usd": 50_000_000.0},
        {"name": "Stargate",  "delay_minutes": 10.0, "fee_pct": 0.15, "liquidity_usd": 80_000_000.0},
        {"name": "Hop",       "delay_minutes": 15.0, "fee_pct": 0.20, "liquidity_usd": 30_000_000.0},
        {"name": "Synapse",   "delay_minutes": 20.0, "fee_pct": 0.25, "liquidity_usd": 40_000_000.0},
        {"name": "Celer",     "delay_minutes": 25.0, "fee_pct": 0.30, "liquidity_usd": 25_000_000.0},
        {"name": "LayerZero", "delay_minutes": 30.0, "fee_pct": 0.50, "liquidity_usd": 60_000_000.0},
    ]

    def __init__(
        self,
        chains: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        """初始化跨链桥模拟器

        Args:
            chains: 支持的链列表，默认 6 条主流链
            seed: 随机种子
        """
        self.chains: List[str] = list(chains) if chains else list(self._DEFAULT_CHAINS)
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)

        # 为每对相邻链初始化桥状态（避免组合爆炸，仅相邻链互通）
        self._bridge_states: List[Dict[str, Any]] = []
        for bridge_tpl in self._DEFAULT_BRIDGES:
            for i in range(len(self.chains) - 1):
                self._bridge_states.append({
                    "name": bridge_tpl["name"],
                    "src_chain": self.chains[i],
                    "dst_chain": self.chains[i + 1],
                    "delay_minutes": float(bridge_tpl["delay_minutes"]),
                    "fee_pct": float(bridge_tpl["fee_pct"]),
                    "liquidity_usd": float(bridge_tpl["liquidity_usd"]),
                    "status": "normal",
                })

    def tick(
        self,
        prices_by_chain: Dict[str, Dict[str, float]],
        timestamp: float,
    ) -> Dict[str, Any]:
        """推进一个 tick，返回桥状态和多链价格快照

        Args:
            prices_by_chain: 每条链上每种资产的价格
                ``{chain: {asset: price}}``
            timestamp: 当前时间戳（秒）

        Returns:
            ``{"bridges": [...], "prices_by_chain": {...}}``
            bridges 字段：name/src_chain/dst_chain/delay_minutes/
            fee_pct/liquidity_usd/status
        """
        # ---- 桥状态随机扰动 + 状态切换 ----
        for b in self._bridge_states:
            # 延迟 ±10% 扰动，钳制在 [5, 30]
            b["delay_minutes"] = float(
                max(5.0, min(30.0, b["delay_minutes"] * (1.0 + self._rng.uniform(-0.1, 0.1))))
            )
            # 桥费 ±5% 扰动，钳制在 [0.05, 0.60]
            b["fee_pct"] = float(
                max(0.05, min(0.60, b["fee_pct"] * (1.0 + self._rng.uniform(-0.05, 0.05))))
            )
            # 流动性 ±2% 漂移
            b["liquidity_usd"] = float(
                max(1_000_000.0, b["liquidity_usd"] * (1.0 + self._rng.uniform(-0.02, 0.02)))
            )
            # 状态切换：5% 概率 congested, 3% degraded, 其余 normal
            r = self._rng.random()
            if r < 0.03:
                b["status"] = "degraded"
            elif r < 0.08:
                b["status"] = "congested"
            else:
                b["status"] = "normal"

        return {
            "bridges": [dict(b) for b in self._bridge_states],
            "prices_by_chain": dict(prices_by_chain),
        }


# ============================================================================
# 4. SimulatedDEX — DEX 报价数据源
# ============================================================================


class SimulatedDEX:
    """去中心化交易所 (DEX) 报价模拟器

    为 ``cex_dex_flash_arbitrage.py`` 提供含滑点和手续费的 DEX 报价。

    算法：
      - CPMM 报价：amount_out = reserve_out * amount_in * (1-fee) /
        (reserve_in + amount_in * (1-fee))
      - 0.3% 默认手续费
      - 滑点 = (中间价 - 实际成交价) / 中间价
      - 流动性 USD = 2 * reserve_in * price（简化估计）

    典型用法：
      >>> dex = SimulatedDEX()
      >>> quote = dex.tick("BTC", 50000.0, time.time())
      >>> quote["dex_price"], quote["slippage_pct"]
    """

    _DEFAULT_FEE: float = 0.003  # 0.3%
    _DEFAULT_TRADE_SIZE: float = 1.0  # 默认查询交易量（基础币单位）

    def __init__(
        self,
        pool_reserves: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
        seed: int = 42,
    ) -> None:
        """初始化 DEX 报价器

        Args:
            pool_reserves: 自定义池储备 {(token_a, token_b): (reserve_a, reserve_b)}
                若为 None 则使用默认 BTC/ETH 池
            seed: 随机种子
        """
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)

        if pool_reserves is None:
            # 默认池：BTC/USDT, ETH/USDT
            self._pools: Dict[Tuple[str, str], Tuple[float, float]] = {
                ("BTC", "USDT"): (100.0, 5_000_000.0),
                ("ETH", "USDT"): (1000.0, 3_000_000.0),
            }
        else:
            self._pools: Dict[Tuple[str, str], Tuple[float, float]] = dict(pool_reserves)

        # 价格历史用于储备再平衡
        self._price_history: deque = deque(maxlen=100)

    def tick(
        self, token: str, price: float, timestamp: float
    ) -> Dict[str, Any]:
        """推进一个 tick，返回该 token 的 DEX 报价

        Args:
            token: 基础代币符号（如 "BTC"）
            price: 当前沙盘价格（USDT 计价）
            timestamp: 当前时间戳（秒）

        Returns:
            ``{"dex_price": float, "slippage_pct": float,
               "fee_pct": float, "liquidity_usd": float}``
        """
        # 查找包含该 token 的池
        pool_key: Optional[Tuple[str, str]] = None
        for key in self._pools:
            if token in key:
                pool_key = key
                break

        if pool_key is None:
            # 自动创建默认池（token 储备 1000，USDT 储备 = 1000*price）
            pool_key = (token, "USDT")
            self._pools[pool_key] = (1000.0, 1000.0 * max(price, 1.0))

        reserve_in, reserve_out = self._pools[pool_key]
        # token_in = token（基础币），token_out = 报价币（USDT）
        # 中间价 = reserve_out / reserve_in
        mid_price = reserve_out / reserve_in if reserve_in > 0 else price

        # ---- 储备再平衡（向外部价格靠拢 50%）----
        if price > 0 and mid_price > 0:
            target_in = math.sqrt((reserve_in * reserve_out) / price) if price > 0 else reserve_in
            target_out = target_in * price
            adjust = 0.5
            reserve_in = reserve_in * (1 - adjust) + target_in * adjust
            reserve_out = reserve_out * (1 - adjust) + target_out * adjust
            self._pools[pool_key] = (reserve_in, reserve_out)
            mid_price = reserve_out / reserve_in if reserve_in > 0 else price

        # ---- CPMM 报价（买入基础币规模 _DEFAULT_TRADE_SIZE）----
        amount_in = self._DEFAULT_TRADE_SIZE
        amount_in_after_fee = amount_in * (1 - self._DEFAULT_FEE)
        if reserve_in + amount_in_after_fee > 0:
            amount_out = (
                reserve_out * amount_in_after_fee
                / (reserve_in + amount_in_after_fee)
            )
        else:
            amount_out = 0.0

        # 实际成交价（每单位基础币换得的报价币）
        dex_price = amount_out / amount_in if amount_in > 0 else mid_price

        # ---- 滑点 = (mid_price - dex_price) / mid_price ----
        if mid_price > 0:
            slippage_pct = abs(mid_price - dex_price) / mid_price * 100.0
        else:
            slippage_pct = 0.0

        # ---- 流动性 USD = 2 * reserve_in * mid_price ----
        liquidity_usd = 2.0 * reserve_in * mid_price

        self._price_history.append(price)
        return {
            "dex_price": float(dex_price),
            "slippage_pct": float(slippage_pct),
            "fee_pct": float(self._DEFAULT_FEE * 100.0),
            "liquidity_usd": float(liquidity_usd),
        }


# ============================================================================
# 5. SimulatedWhaleFlow — 巨鲸交易流数据源
# ============================================================================


class SimulatedWhaleFlow:
    """链上巨鲸交易流模拟器

    为 ``on_chain_whale_tracker.py`` 提供大额链上交易列表。

    算法：
      - 泊松过程生成交易数量，lambda 与价格波动正相关
      - 高波动期巨鲸活跃度提升
      - 每笔交易随机分类：交易所流入/流出/普通转账
      - 交易金额服从对数正态分布（模拟财富幂律）

    典型用法：
      >>> flow = SimulatedWhaleFlow("BTC", large_tx_threshold=1000.0)
      >>> txs = flow.tick(50000.0, time.time())
      >>> txs[0]["value_usd"], txs[0]["is_exchange_inflow"]
    """

    _BASE_LAMBDA: float = 0.5  # 每 tick 基础到达率
    _WALLETS: Tuple[str, ...] = (
        "0xwhale_alpha", "0xwhale_beta", "0xwhale_gamma", "0xwhale_delta",
        "0xexchange_binance", "0xexchange_okx", "0xexchange_coinbase",
        "0xexchange_bybit", "0xwhale_epsilon", "0xwhale_zeta",
    )

    def __init__(
        self,
        target_symbol: str = "BTC",
        large_tx_threshold: float = 1000.0,
        seed: int = 42,
    ) -> None:
        """初始化巨鲸交易流

        Args:
            target_symbol: 目标资产符号
            large_tx_threshold: 大额交易阈值（以基础币计）
            seed: 随机种子
        """
        self.target_symbol: str = target_symbol
        self.large_tx_threshold: float = float(large_tx_threshold)
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)
        self._price_history: deque = deque(maxlen=100)
        self._tx_counter: int = 0

    def _gen_tx_hash(self) -> str:
        """生成模拟交易哈希"""
        self._tx_counter += 1
        return f"0x{self._tx_counter:064x}"

    def tick(self, price: float, timestamp: float) -> List[Dict]:
        """推进一个 tick，返回本 tick 产生的巨鲸交易列表

        Args:
            price: 当前沙盘价格
            timestamp: 当前时间戳（秒）

        Returns:
            交易字典列表，每笔含：
            hash/from/to/amount/value_usd/timestamp/
            is_exchange_inflow/is_exchange_outflow
        """
        self._price_history.append(price)

        # ---- 计算近期波动率，决定巨鲸活跃度 ----
        vol_factor = 1.0
        if len(self._price_history) >= 2:
            recent = list(self._price_history)[-10:]
            if len(recent) >= 2 and recent[-2] > 0:
                ret = abs(math.log(recent[-1] / recent[-2]))
                vol_factor = 1.0 + min(ret * 100.0, 4.0)

        lam = self._BASE_LAMBDA * vol_factor
        n_tx = _poisson_sample(self._rng, lam)

        transactions: List[Dict] = []
        for _ in range(n_tx):
            # 交易金额：对数正态分布（财富幂律），保证 > large_tx_threshold
            # ln(amount) ~ Normal(ln(threshold), 1.0)
            log_amount = math.log(max(self.large_tx_threshold, 1.0)) + self._rng.gauss(0.0, 1.0)
            amount = math.exp(log_amount)
            value_usd = amount * price

            # 随机选 from/to
            from_addr = self._rng.choice(self._WALLETS)
            to_addr = self._rng.choice(self._WALLETS)
            while to_addr == from_addr:
                to_addr = self._rng.choice(self._WALLETS)

            # 判断交易所流入/流出
            is_inflow = from_addr.startswith("0xwhale") and to_addr.startswith("0xexchange")
            is_outflow = from_addr.startswith("0xexchange") and to_addr.startswith("0xwhale")

            transactions.append({
                "hash": self._gen_tx_hash(),
                "from": from_addr,
                "to": to_addr,
                "amount": float(amount),
                "value_usd": float(value_usd),
                "timestamp": float(timestamp),
                "is_exchange_inflow": bool(is_inflow),
                "is_exchange_outflow": bool(is_outflow),
            })

        return transactions


# ============================================================================
# 6. SimulatedOptionsMarket — 期权链报价数据源
# ============================================================================


class SimulatedOptionsMarket:
    """期权链报价模拟器

    为 ``volatility_surface_arb.py`` 和 ``vanna_volga_arbitrage.py`` 提供
    完整期权链（多 expiry + 多 strike）以及 25Δ RR / 25Δ BF / ATM vol。

    算法：
      - BSM (Black-Scholes-Merton) 期权定价
      - SVI 微笑曲线生成每个 strike 的 IV：
        w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
        其中 k = log(strike/spot)，w = iv^2
      - Greeks：delta/gamma 通过 BSM 解析公式计算
      - 25Δ RR = IV(25Δ Call) - IV(25Δ Put)
      - 25Δ BF = (IV(25Δ Call) + IV(25Δ Put))/2 - IV(ATM)

    典型用法：
      >>> mkt = SimulatedOptionsMarket("BTC")
      >>> chain = mkt.tick(50000.0, time.time())
      >>> chain["atm_vol"], chain["rr_25delta"], chain["options"][0]["iv"]
    """

    # 期权链配置
    _EXPIRIES: Tuple[float, ...] = (7.0, 30.0, 90.0, 180.0)  # 到期天数
    _STRIKE_MULTIPLIERS: Tuple[float, ...] = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)
    _RISK_FREE_RATE: float = 0.02  # 无风险利率

    # SVI 默认参数（BTC 期权市场校准值）
    _SVI_A: float = 0.04   # 总方差水平
    _SVI_B: float = 0.15   # 斜率
    _SVI_RHO: float = -0.2  # 偏斜（负 = put skew）
    _SVI_M: float = 0.0    # 中心
    _SVI_SIGMA: float = 0.1  # 曲率

    def __init__(
        self,
        underlying: str = "BTC",
        seed: int = 42,
    ) -> None:
        """初始化期权市场模拟器

        Args:
            underlying: 标的资产符号
            seed: 随机种子
        """
        self.underlying: str = underlying
        self.seed: int = seed
        self._rng: random.Random = random.Random(seed)
        self._spot_history: deque = deque(maxlen=100)
        self._last_spot: float = 0.0

    def _svi_iv(self, strike: float, spot: float) -> float:
        """SVI 微笑曲线计算单个 strike 的 IV

        Args:
            strike: 行权价
            spot: 标的现价

        Returns:
            该 strike 的隐含波动率（年化小数）
        """
        if spot <= 0 or strike <= 0:
            return 0.5
        k = math.log(strike / spot)
        inner = math.sqrt((k - self._SVI_M) ** 2 + self._SVI_SIGMA ** 2)
        w = self._SVI_A + self._SVI_B * (
            self._SVI_RHO * (k - self._SVI_M) + inner
        )
        # 防止负方差
        w = max(w, 0.01)
        return math.sqrt(w)

    def _bsm_price_and_greeks(
        self, spot: float, strike: float, expiry_days: float,
        iv: float, option_type: str,
    ) -> Tuple[float, float, float]:
        """BSM 期权定价 + delta + gamma

        Args:
            spot: 标的现价
            strike: 行权价
            expiry_days: 到期天数
            iv: 隐含波动率（年化）
            option_type: "call" 或 "put"

        Returns:
            (price, delta, gamma)
        """
        T = max(expiry_days / 365.0, 1e-6)
        if iv <= 0:
            iv = 1e-4
        sigma_sqrt_t = iv * math.sqrt(T)
        if sigma_sqrt_t <= 0:
            sigma_sqrt_t = 1e-6

        d1 = (
            math.log(spot / strike)
            + (self._RISK_FREE_RATE + 0.5 * iv ** 2) * T
        ) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        # 标准正态 CDF
        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

        # gamma 公式（call/put 相同）
        gamma = (
            math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
        ) / (spot * sigma_sqrt_t)

        if option_type == "call":
            price = spot * _ncdf(d1) - strike * math.exp(
                -self._RISK_FREE_RATE * T
            ) * _ncdf(d2)
            delta = _ncdf(d1)
        else:  # put
            price = strike * math.exp(
                -self._RISK_FREE_RATE * T
            ) * _ncdf(-d2) - spot * _ncdf(-d1)
            delta = _ncdf(d1) - 1.0

        # 防止负价格（数值误差）
        price = max(price, 0.0)
        return float(price), float(delta), float(gamma)

    def _find_delta_strike(
        self, spot: float, expiry_days: float, target_delta: float,
        option_type: str,
    ) -> Tuple[float, float]:
        """二分法查找给定 delta 对应的 strike 和 IV

        Args:
            spot: 标的现价
            expiry_days: 到期天数
            target_delta: 目标 delta 绝对值（如 0.25）
            option_type: "call" 或 "put"

        Returns:
            (strike, iv)
        """
        lo = spot * 0.5
        hi = spot * 2.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            iv = self._svi_iv(mid, spot)
            _, delta, _ = self._bsm_price_and_greeks(
                spot, mid, expiry_days, iv, option_type
            )
            if option_type == "call":
                # call delta 随 strike 减小而增大
                if abs(delta) > target_delta:
                    lo = mid
                else:
                    hi = mid
            else:
                # put delta（负值）绝对值随 strike 增大而增大
                if abs(delta) > target_delta:
                    hi = mid
                else:
                    lo = mid
        strike = (lo + hi) / 2.0
        iv = self._svi_iv(strike, spot)
        return float(strike), float(iv)

    def tick(self, spot: float, timestamp: float) -> Dict[str, Any]:
        """推进一个 tick，生成完整期权链和波动率微笑指标

        Args:
            spot: 标的现价
            timestamp: 当前时间戳（秒）

        Returns:
            ``{"options": [...], "atm_vol": float, "rr_25delta": float,
               "bf_25delta": float, "spot": float}``
            options 字段：expiry_days/strike/type/iv/price/delta/gamma
        """
        self._spot_history.append(spot)
        self._last_spot = spot
        if spot <= 0:
            return {
                "options": [],
                "atm_vol": 0.0,
                "rr_25delta": 0.0,
                "bf_25delta": 0.0,
                "spot": float(spot),
            }

        # ---- 生成完整期权链 ----
        options: List[Dict[str, Any]] = []
        # 为 SVI 参数加入轻微随机扰动，模拟市场动态
        a = self._SVI_A * (1.0 + self._rng.uniform(-0.02, 0.02))
        b = self._SVI_B * (1.0 + self._rng.uniform(-0.05, 0.05))
        rho = self._SVI_RHO + self._rng.uniform(-0.02, 0.02)
        rho = max(min(rho, 0.95), -0.95)

        for expiry in self._EXPIRIES:
            for mult in self._STRIKE_MULTIPLIERS:
                strike = spot * mult
                # 用扰动后的 SVI 参数计算 IV
                k = math.log(strike / spot)
                inner = math.sqrt(k ** 2 + self._SVI_SIGMA ** 2)
                w = a + b * (rho * k + inner)
                w = max(w, 0.01)
                iv = math.sqrt(w)

                for opt_type in ("call", "put"):
                    price, delta, gamma = self._bsm_price_and_greeks(
                        spot, strike, expiry, iv, opt_type
                    )
                    options.append({
                        "expiry_days": float(expiry),
                        "strike": float(strike),
                        "type": opt_type,
                        "iv": float(iv),
                        "price": float(price),
                        "delta": float(delta),
                        "gamma": float(gamma),
                    })

        # ---- 计算 ATM vol / 25Δ RR / 25Δ BF ----
        # 使用 30 天到期日作为基准
        atm_strike = spot
        atm_vol = self._svi_iv(atm_strike, spot)

        # 25Δ call/put strike（30 天到期）
        call25_strike, iv_25c = self._find_delta_strike(
            spot, 30.0, 0.25, "call"
        )
        put25_strike, iv_25p = self._find_delta_strike(
            spot, 30.0, 0.25, "put"
        )

        rr_25delta = iv_25c - iv_25p
        bf_25delta = (iv_25c + iv_25p) / 2.0 - atm_vol

        return {
            "options": options,
            "atm_vol": float(atm_vol),
            "rr_25delta": float(rr_25delta),
            "bf_25delta": float(bf_25delta),
            "spot": float(spot),
        }


# ============================================================================
# 公开 API
# ============================================================================

__all__ = [
    "SimulatedIVFeed",
    "SimulatedAMMPool",
    "SimulatedBridge",
    "SimulatedDEX",
    "SimulatedWhaleFlow",
    "SimulatedOptionsMarket",
]
