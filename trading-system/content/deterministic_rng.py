# -*- coding: utf-8 -*-
"""
确定性随机数生成器 (Deterministic RNG) — 进化交易系统的可复现性基石

借鉴 NautilusTrader 的"单线程确定性+双纳秒时间戳"设计哲学：
  - 同一基因编码 + 同一历史数据 → 完全相同的交易结果
  - 双时间戳（ts_event + ts_init）支持生产异常重放
  - 可种子化随机源，每个Agent独立RNG，互不干扰

核心铁律：进化交易系统的策略对比必须具备可复现性。
  random.gauss/random.random 破坏了可复现性，是致命缺陷。
  本模块提供可种子化的随机源，替代模块级 random 调用。

来源：NautilusTrader "Why NautilusTrader Exists" 设计理念博客
      "Parity is built into the architecture rather than relying on convention."
"""

from __future__ import annotations

import hashlib
import random as _random
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# ============================================================================
# 双时间戳模型（借鉴 NautilusTrader）
# ============================================================================


@dataclass(slots=True, frozen=True)
class DualTimestamp:
    """双纳秒时间戳 — 锚定状态转换到一致的可审计时间线

    借鉴 NautilusTrader 的双时间戳设计：
      - ts_event: 事件在领域内发生的时间（K线时间）
      - ts_init:  系统创建该事件对象的时间（引擎处理时刻）

    意义：所有状态转换锚定到一致、可审计的时间线。
          生产异常时可在研究中重放场景，让调试变得可处理。
    """

    ts_event: int  # 事件发生时间（纳秒）
    ts_init: int   # 系统处理时间（纳秒）

    @classmethod
    def from_market_time(
        cls,
        market_timestamp: float,
        processing_offset_ns: int = 0,
    ) -> "DualTimestamp":
        """从市场时间戳创建双时间戳

        Args:
            market_timestamp: 市场时间戳（秒，浮点）
            processing_offset_ns: 处理偏移（纳秒），模拟引擎处理延迟

        Returns:
            DualTimestamp 实例
        """
        ts_event = int(market_timestamp * 1_000_000_000)
        ts_init = ts_event + processing_offset_ns
        return cls(ts_event=ts_event, ts_init=ts_init)

    @classmethod
    def now(cls) -> "DualTimestamp":
        """当前时间的双时间戳（实盘模式使用）"""
        now_ns = time.time_ns()
        return cls(ts_event=now_ns, ts_init=now_ns)

    def to_dict(self) -> Dict[str, int]:
        """转换为字典（用于序列化）"""
        return {"ts_event": self.ts_event, "ts_init": self.ts_init}

    @property
    def event_seconds(self) -> float:
        """事件时间（秒）"""
        return self.ts_event / 1_000_000_000

    @property
    def init_seconds(self) -> float:
        """处理时间（秒）"""
        return self.ts_init / 1_000_000_000

    @property
    def processing_latency_ns(self) -> int:
        """处理延迟（纳秒）= ts_init - ts_event"""
        return self.ts_init - self.ts_event


# ============================================================================
# 确定性随机数生成器
# ============================================================================


class DeterministicRNG:
    """可种子化的确定性随机数生成器

    核心设计：
      - 每个 Agent 拥有独立的 RNG 实例（基于 agent_id + generation 派生种子）
      - 同一种子 + 同一调用序列 → 完全相同的随机数序列
      - 替代模块级 random.gauss/random.random，保证可复现性

    使用方式：
      # 方式1：基于 agent_id 派生种子（推荐，每个Agent独立）
      rng = DeterministicRNG.from_agent_id("agent-gen0-001", generation=0)

      # 方式2：固定种子（全局复现）
      rng = DeterministicRNG(seed=42)

      # 替代 random.gauss / random.random
      value = rng.gauss(0, 1)
      value = rng.random()
      value = rng.uniform(0, 1)
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 随机种子。同一种子保证完全相同的随机数序列。
        """
        self._seed = int(seed)
        self._rng = _random.Random(self._seed)
        # 记录调用次数，用于调试和验证
        self._call_count: int = 0

    @classmethod
    def from_agent_id(
        cls,
        agent_id: str,
        generation: int = 0,
        extra_salt: str = "",
    ) -> "DeterministicRNG":
        """基于 agent_id 派生种子，保证每个 Agent 独立且可复现

        Args:
            agent_id: Agent 唯一标识
            generation: 进化代数
            extra_salt: 额外盐值（用于同一Agent不同场景的RNG隔离）

        Returns:
            DeterministicRNG 实例

        设计：
          - 同一 agent_id + generation + salt → 同一种子 → 同一随机序列
          - 不同 agent_id → 不同种子 → 独立随机序列
          - 保证进化对比的可复现性
        """
        key = f"{agent_id}|gen{generation}|{extra_salt}"
        # 使用 SHA256 派生确定性种子
        hash_bytes = hashlib.sha256(key.encode("utf-8")).digest()
        # 取前8字节作为种子（64位整数）
        seed = int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
        return cls(seed=seed)

    @classmethod
    def from_symbol(
        cls,
        symbol: str,
        generation: int = 0,
        bar_timestamp: float = 0.0,
    ) -> "DeterministicRNG":
        """基于交易对+K线时间派生种子，保证市场数据生成可复现

        Args:
            symbol: 交易对
            generation: 进化代数
            bar_timestamp: K线时间戳

        Returns:
            DeterministicRNG 实例
        """
        key = f"{symbol}|gen{generation}|bar{bar_timestamp}"
        hash_bytes = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
        return cls(seed=seed)

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """高斯随机数（替代 random.gauss）

        Args:
            mu: 均值
            sigma: 标准差

        Returns:
            高斯随机数
        """
        self._call_count += 1
        return self._rng.gauss(mu, sigma)

    def random(self) -> float:
        """[0, 1) 均匀随机数（替代 random.random）"""
        self._call_count += 1
        return self._rng.random()

    def uniform(self, a: float, b: float) -> float:
        """[a, b] 均匀随机数（替代 random.uniform）"""
        self._call_count += 1
        return self._rng.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        """[a, b] 整数随机数（替代 random.randint）"""
        self._call_count += 1
        return self._rng.randint(a, b)

    def choice(self, seq):
        """随机选择（替代 random.choice）"""
        self._call_count += 1
        return self._rng.choice(seq)

    def sample(self, population, k: int):
        """随机抽样（替代 random.sample）"""
        self._call_count += 1
        return self._rng.sample(population, k)

    def shuffle(self, x) -> None:
        """原地洗牌（替代 random.shuffle）"""
        self._call_count += 1
        self._rng.shuffle(x)

    def reseed(self, seed: int) -> None:
        """重置种子（替代 seed() 方法，避免与 seed property 冲突）

        Args:
            seed: 新种子
        """
        self._seed = int(seed)
        self._rng = _random.Random(self._seed)
        self._call_count = 0

    def reset(self) -> None:
        """重置到初始状态（用原种子重新初始化）"""
        self._rng = _random.Random(self._seed)
        self._call_count = 0

    @property
    def seed(self) -> int:
        """当前种子（只读属性）"""
        return self._seed

    @property
    def call_count(self) -> int:
        """累计调用次数（用于调试和验证可复现性）"""
        return self._call_count

    def get_state(self) -> tuple:
        """获取内部状态（用于检查点）"""
        return self._rng.getstate()

    def set_state(self, state: tuple) -> None:
        """恢复内部状态（用于检查点恢复）"""
        self._rng.setstate(state)


# ============================================================================
# 全局确定性 RNG 管理器
# ============================================================================


class RNGManager:
    """全局 RNG 管理器 — 为每个 Agent 管理独立的确定性 RNG

    使用方式：
      manager = RNGManager(default_seed=42)

      # 为 Agent 获取/创建 RNG
      rng = manager.get_agent_rng("agent-gen0-001", generation=0)

      # 为市场数据生成获取 RNG
      market_rng = manager.get_market_rng("BTC-USDT", generation=0)

      # 全局 RNG（用于非 Agent 特定的随机操作）
      global_rng = manager.global_rng
    """

    def __init__(self, default_seed: int = 42):
        """
        Args:
            default_seed: 默认种子（用于全局 RNG 和未指定种子的场景）
        """
        self._default_seed = int(default_seed)
        self._global_rng = DeterministicRNG(seed=default_seed)
        self._agent_rngs: Dict[str, DeterministicRNG] = {}
        self._market_rngs: Dict[str, DeterministicRNG] = {}

    def get_agent_rng(
        self,
        agent_id: str,
        generation: int = 0,
        extra_salt: str = "",
    ) -> DeterministicRNG:
        """获取或创建 Agent 专属 RNG

        Args:
            agent_id: Agent 唯一标识
            generation: 进化代数
            extra_salt: 额外盐值

        Returns:
            该 Agent 的确定性 RNG（同一参数总是返回同一实例）
        """
        key = f"{agent_id}|gen{generation}|{extra_salt}"
        if key not in self._agent_rngs:
            self._agent_rngs[key] = DeterministicRNG.from_agent_id(
                agent_id, generation, extra_salt
            )
        return self._agent_rngs[key]

    def get_market_rng(
        self,
        symbol: str,
        generation: int = 0,
        bar_timestamp: float = 0.0,
    ) -> DeterministicRNG:
        """获取或创建市场数据专属 RNG

        Args:
            symbol: 交易对
            generation: 进化代数
            bar_timestamp: K线时间戳

        Returns:
            市场数据的确定性 RNG
        """
        key = f"{symbol}|gen{generation}|bar{bar_timestamp}"
        if key not in self._market_rngs:
            self._market_rngs[key] = DeterministicRNG.from_symbol(
                symbol, generation, bar_timestamp
            )
        return self._market_rngs[key]

    @property
    def global_rng(self) -> DeterministicRNG:
        """全局 RNG（用于非 Agent 特定的随机操作）"""
        return self._global_rng

    def reset_all(self) -> None:
        """重置所有 RNG（用于新一轮进化）"""
        self._global_rng = DeterministicRNG(seed=self._default_seed)
        self._agent_rngs.clear()
        self._market_rngs.clear()

    def reseed(self, seed: int) -> None:
        """重置全局种子并清除所有缓存 RNG

        Args:
            seed: 新的默认种子
        """
        self._default_seed = seed
        self._global_rng = DeterministicRNG(seed=seed)
        self._agent_rngs.clear()
        self._market_rngs.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取 RNG 管理统计"""
        return {
            "default_seed": self._default_seed,
            "global_call_count": self._global_rng.call_count,
            "agent_rng_count": len(self._agent_rngs),
            "market_rng_count": len(self._market_rngs),
            "total_call_count": (
                self._global_rng.call_count
                + sum(rng.call_count for rng in self._agent_rngs.values())
                + sum(rng.call_count for rng in self._market_rngs.values())
            ),
        }


# ============================================================================
# 全局单例（向后兼容）
# ============================================================================

# 全局 RNG 管理器单例
# 使用方式：from .deterministic_rng import global_rng_manager
#           rng = global_rng_manager.global_rng
#           value = rng.gauss(0, 1)
global_rng_manager = RNGManager(default_seed=42)


# ============================================================================
# 验证工具
# ============================================================================


def verify_determinism(
    agent_id: str = "test-agent",
    generation: int = 0,
    n_calls: int = 100,
) -> Tuple[bool, str]:
    """验证确定性：同一参数两次生成完全相同的随机序列

    Args:
        agent_id: 测试用的 Agent ID
        generation: 测试用的代数
        n_calls: 测试调用次数

    Returns:
        (is_deterministic, message)
    """
    # 第一次生成
    rng1 = DeterministicRNG.from_agent_id(agent_id, generation)
    seq1 = [rng1.gauss(0, 1) for _ in range(n_calls)]

    # 第二次生成（相同参数）
    rng2 = DeterministicRNG.from_agent_id(agent_id, generation)
    seq2 = [rng2.gauss(0, 1) for _ in range(n_calls)]

    # 验证完全相同
    if seq1 == seq2:
        return True, f"✅ 确定性验证通过：{n_calls}次调用完全一致"
    else:
        diff_count = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return False, f"❌ 确定性验证失败：{diff_count}/{n_calls} 次调用不一致"


def verify_agent_isolation(
    agent_a: str = "agent-A",
    agent_b: str = "agent-B",
    generation: int = 0,
    n_calls: int = 100,
) -> Tuple[bool, str]:
    """验证 Agent 隔离性：不同 Agent 的 RNG 序列不同

    Args:
        agent_a: Agent A 的 ID
        agent_b: Agent B 的 ID
        generation: 代数
        n_calls: 测试调用次数

    Returns:
        (is_isolated, message)
    """
    rng_a = DeterministicRNG.from_agent_id(agent_a, generation)
    rng_b = DeterministicRNG.from_agent_id(agent_b, generation)

    seq_a = [rng_a.gauss(0, 1) for _ in range(n_calls)]
    seq_b = [rng_b.gauss(0, 1) for _ in range(n_calls)]

    # 验证不同（允许极少数巧合相同）
    diff_count = sum(1 for a, b in zip(seq_a, seq_b) if a != b)
    if diff_count > n_calls * 0.9:  # 90%以上不同即通过
        return True, f"✅ Agent 隔离性验证通过：{diff_count}/{n_calls} 次调用不同"
    else:
        return False, f"❌ Agent 隔离性验证失败：仅 {diff_count}/{n_calls} 次调用不同"


if __name__ == "__main__":
    # 自验证
    print("=" * 60)
    print("DeterministicRNG 自验证")
    print("=" * 60)

    # 验证确定性
    ok, msg = verify_determinism()
    print(msg)

    # 验证 Agent 隔离性
    ok, msg = verify_agent_isolation()
    print(msg)

    # 验证双时间戳
    ts = DualTimestamp.from_market_time(1719500000.0, processing_offset_ns=1000)
    print(f"✅ 双时间戳: ts_event={ts.ts_event}, ts_init={ts.ts_init}, latency={ts.processing_latency_ns}ns")

    # 验证 RNG 管理器
    manager = RNGManager(default_seed=42)
    rng1 = manager.get_agent_rng("agent-001", generation=0)
    rng2 = manager.get_agent_rng("agent-001", generation=0)
    assert rng1 is rng2, "同一参数应返回同一实例"
    print(f"✅ RNG 管理器: 同一参数返回同一实例")

    print("=" * 60)
    print("所有验证通过")
