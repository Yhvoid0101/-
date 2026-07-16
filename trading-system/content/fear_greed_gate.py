#!/usr/bin/env python3
"""恐慌贪婪指数门控模块 — Layer 1 市场数据层

恐慌贪婪指数（Fear & Greed Index）是市场情绪的反向指标：
  - 极度恐慌 (<20)：市场恐慌，资金抛售 → 反向看涨，做多 BOOST
  - 恐慌 (20-40)：市场偏恐慌 → 做多轻度 BOOST
  - 中性 (40-60)：市场情绪中性 → ALLOW
  - 贪婪 (60-80)：市场偏贪婪 → 做空轻度 BOOST
  - 极度贪婪 (>=80)：市场极度贪婪，反向看跌 → 做空 BOOST

门控逻辑（反向指标）:
  - 极度恐慌 + 做多 → BOOST ×1.25
  - 极度恐慌 + 做空 → REDUCE ×0.5
  - 恐慌 + 做多 → BOOST ×1.15
  - 恐慌 + 做空 → REDUCE ×0.7
  - 极度贪婪 + 做多 → REDUCE ×0.5
  - 极度贪婪 + 做空 → BOOST ×1.25
  - 贪婪 + 做多 → REDUCE ×0.7
  - 贪婪 + 做空 → BOOST ×1.15
  - 中性 → ALLOW

数据来源:
  - API: https://api.alternative.me/fng/?limit=1
  - 缓存: data/auto_fetched/fear_greed/latest.json
  - 降级: 50.0 (中性, has_data=False)

用法:
    from sandbox_trading.fear_greed_gate import FearGreedGate
    gate = FearGreedGate(enabled=True)
    decision = gate.apply_to_decision(decision, symbol="BTC-USDT")
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.fear_greed_gate")


# ============================================================================
# 枚举与数据结构
# ============================================================================

class FGLevel(Enum):
    """恐慌贪婪指数等级"""
    EXTREME_FEAR = "extreme_fear"    # 极度恐慌 (<20)
    FEAR = "fear"                    # 恐慌 (20-40)
    NEUTRAL = "neutral"              # 中性 (40-60)
    GREED = "greed"                  # 贪婪 (60-80)
    EXTREME_GREED = "extreme_greed"  # 极度贪婪 (>=80)


class FGAction(Enum):
    """恐慌贪婪门控动作"""
    ALLOW = "allow"
    BOOST = "boost"
    REDUCE = "reduce"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class FGAnalysisResult:
    """恐慌贪婪指数分析结果"""
    index_value: float = 50.0        # 恐慌贪婪指数值 (0-100)
    level: FGLevel = FGLevel.NEUTRAL
    timestamp: float = field(default_factory=time.time)
    has_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_value": self.index_value,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "has_data": self.has_data,
        }


# ============================================================================
# 恐慌贪婪指数门控主类
# ============================================================================

class FearGreedGate:
    """恐慌贪婪指数门控

    工作流程:
      1. 获取恐慌贪婪指数 (API → 缓存 → 降级)
      2. 分类情绪等级
      3. 基于反向指标逻辑调整策略决策

    门控规则 (反向指标):
      1. 极度恐慌 → 做多 BOOST / 做空 REDUCE
      2. 恐慌 → 做多 BOOST / 做空 REDUCE
      3. 中性 → ALLOW
      4. 贪婪 → 做多 REDUCE / 做空 BOOST
      5. 极度贪婪 → 做多 REDUCE / 做空 BOOST
    """

    # 等级阈值
    EXTREME_FEAR_THRESHOLD = 20      # 极度恐慌阈值
    FEAR_THRESHOLD = 40              # 恐慌阈值
    GREED_THRESHOLD = 60             # 贪婪阈值
    EXTREME_GREED_THRESHOLD = 80     # 极度贪婪阈值

    # 调整系数
    BOOST_MULTIPLIER = 1.15
    REDUCE_MULTIPLIER = 0.7
    EXTREME_BOOST_MULTIPLIER = 1.25
    EXTREME_REDUCE_MULTIPLIER = 0.5

    # API配置
    API_URL = "https://api.alternative.me/fng/?limit=1"
    API_TIMEOUT = 5  # 秒

    # API 缓存 TTL (秒)
    _API_SUCCESS_TTL = 3600   # 成功缓存有效期 1 小时
    _API_FAIL_TTL = 300       # 失败缓存期 5 分钟 (避免频繁重试)

    # 缓存路径 (相对于模块文件)
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_PATH = os.path.join(
        _MODULE_DIR, "data", "auto_fetched", "fear_greed", "latest.json"
    )

    def __init__(self, enabled: bool = True):
        """初始化恐慌贪婪指数门控

        Args:
            enabled: 是否启用
        """
        self.enabled = enabled
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis: Optional[FGAnalysisResult] = None
        self._index_history: list[float] = []

        # API 缓存 (优化: 避免频繁调用 API 导致超时)
        self._api_success_cache = None  # (timestamp, value) 或 None
        self._api_fail_until = 0        # 失败缓存到期时间戳
        self._api_success_ttl = self._API_SUCCESS_TTL
        self._api_fail_ttl = self._API_FAIL_TTL

        logger.info(
            "FearGreedGate initialized: enabled=%s "
            "extreme_fear_thresh=%d fear_thresh=%d "
            "greed_thresh=%d extreme_greed_thresh=%d",
            self.enabled,
            self.EXTREME_FEAR_THRESHOLD, self.FEAR_THRESHOLD,
            self.GREED_THRESHOLD, self.EXTREME_GREED_THRESHOLD,
        )

    # ----------------------------------------------------------------
    # 数据获取
    # ----------------------------------------------------------------

    def fetch_from_api(self) -> Optional[Dict[str, Any]]:
        """从 alternative.me API 获取恐慌贪婪指数

        Returns:
            解析后的数据字典, 失败返回 None
            格式: {"value": 25, "classification": "Extreme Fear", "timestamp": ...}
        """
        try:
            req = urllib.request.Request(
                self.API_URL,
                headers={"User-Agent": "Hermes-FearGreedGate/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.API_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            items = data.get("data", [])
            if not items:
                logger.warning("FearGreed API returned empty data")
                return None
            item = items[0]
            value_str = item.get("value", "")
            value = float(value_str)
            result = {
                "value": value,
                "classification": item.get("value_classification", ""),
                "timestamp": int(item.get("timestamp", time.time())),
                "fetched_at": time.time(),
            }
            logger.info(
                "FearGreed API success: value=%.1f classification=%s",
                value, result["classification"],
            )
            return result
        except urllib.error.URLError as e:
            logger.warning("FearGreed API URLError: %s", e)
        except TimeoutError:
            logger.warning("FearGreed API timeout (%ds)", self.API_TIMEOUT)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("FearGreed API parse error: %s", e)
        except Exception as e:
            logger.warning("FearGreed API unknown error: %s", e)
        return None

    def load_from_cache(self) -> Optional[Dict[str, Any]]:
        """从本地缓存加载恐慌贪婪指数

        Returns:
            缓存数据字典, 失败返回 None
        """
        try:
            if not os.path.exists(self.CACHE_PATH):
                return None
            with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            value = data.get("value")
            if value is None:
                return None
            result = {
                "value": float(value),
                "classification": data.get("classification", ""),
                "timestamp": data.get("timestamp", time.time()),
                "fetched_at": data.get("fetched_at", time.time()),
            }
            logger.debug(
                "FearGreed cache loaded: value=%.1f path=%s",
                result["value"], self.CACHE_PATH,
            )
            return result
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("FearGreed cache load error: %s", e)
        return None

    def _save_to_cache(self, data: Dict[str, Any]) -> None:
        """保存数据到本地缓存"""
        try:
            cache_dir = os.path.dirname(self.CACHE_PATH)
            os.makedirs(cache_dir, exist_ok=True)
            with open(self.CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("FearGreed cache saved: %s", self.CACHE_PATH)
        except OSError as e:
            logger.warning("FearGreed cache save error: %s", e)

    def get_current_index(self) -> tuple[float, bool]:
        """获取当前恐慌贪婪指数（带API缓存优化）

        优先级: API(带成功/失败缓存) → 文件缓存 → 降级(50.0, has_data=False)

        Returns:
            (index_value, has_data)
        """
        import time as _time
        _now = _time.time()

        # 1. 检查成功缓存（有效期内直接复用）
        if self._api_success_cache is not None:
            _cache_ts, _cache_val = self._api_success_cache
            if _now - _cache_ts < self._api_success_ttl:
                    return _cache_val, True

        # 2. 检查失败缓存（失败缓存期内跳过API）
        if _now < self._api_fail_until:
            # 跳过API，直接尝试文件缓存
            pass
        elif os.environ.get("HERMES_SANDBOX_MODE", "0") == "1":
            # 沙盘模式: 不发起实时网络请求, 直接使用文件缓存
            # 沙盘使用历史数据回放, 实时API调用破坏可重放性和性能
            pass
        else:
            # 3. 尝试API
            api_data = self.fetch_from_api()
            if api_data is not None:
                # 成功: 保存到文件缓存 + 内存缓存
                self._save_to_cache(api_data)
                self._api_success_cache = (_now, api_data["value"])
                return api_data["value"], True
            else:
                # 失败: 设置失败缓存
                self._api_fail_until = _now + self._api_fail_ttl
                logger.debug(
                    "FearGreed API failed, skip for %ds",
                    int(self._api_fail_ttl),
                )

        # 4. 尝试文件缓存
        cache_data = self.load_from_cache()
        if cache_data is not None:
            return cache_data["value"], True

        # 5. 降级: 返回中性值50
        return 50.0, False

    # ----------------------------------------------------------------
    # 情绪分析
    # ----------------------------------------------------------------

    def _classify_level(self, index_value: float) -> FGLevel:
        """根据指数值分类情绪等级

        Args:
            index_value: 恐慌贪婪指数 (0-100)

        Returns:
            FGLevel 枚举值
        """
        if index_value < self.EXTREME_FEAR_THRESHOLD:
            return FGLevel.EXTREME_FEAR
        elif index_value < self.FEAR_THRESHOLD:
            return FGLevel.FEAR
        elif index_value < self.GREED_THRESHOLD:
            return FGLevel.NEUTRAL
        elif index_value < self.EXTREME_GREED_THRESHOLD:
            return FGLevel.GREED
        else:
            return FGLevel.EXTREME_GREED

    def analyze(self, index_value: Optional[float] = None) -> FGAnalysisResult:
        """分析当前恐慌贪婪指数

        Args:
            index_value: 可选, 直接传入指数值 (用于测试/注入)
                         None则自动获取

        Returns:
            FGAnalysisResult
        """
        result = FGAnalysisResult()

        if index_value is not None:
            # 直接传入值 (测试/注入模式)
            value = float(index_value)
            has_data = True
        else:
            # 自动获取
            value, has_data = self.get_current_index()

        result.index_value = value
        result.has_data = has_data
        result.level = self._classify_level(value)
        result.timestamp = time.time()

        # 更新历史
        self._index_history.append(value)
        if len(self._index_history) > 100:
            self._index_history.pop(0)

        self._last_analysis = result
        return result

    # ----------------------------------------------------------------
    # 决策门控
    # ----------------------------------------------------------------

    def apply_to_decision(
        self,
        decision: Dict[str, Any],
        symbol: str = "",
        index_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """对策略决策应用门控

        Args:
            decision: 策略决策字典
                包含: action("long"/"short"/"flat"), quantity, price 等
            symbol: 交易对
            index_value: 可选, 直接传入恐慌贪婪指数 (用于测试)
                         None则自动获取

        Returns:
            调整后的决策字典, 添加 fg_gate_action, fg_gate_multiplier,
            fg_gate_reason 字段
        """
        if not self.enabled:
            return decision

        self._total_calls += 1

        # 分析恐慌贪婪指数
        analysis = self.analyze(index_value=index_value)

        if not analysis.has_data:
            self._no_data_count += 1
            return decision

        # 应用门控规则
        action, multiplier, reason = self._evaluate_rules(analysis, decision)

        if action in (FGAction.ALLOW, FGAction.NEUTRAL):
            return decision

        # 调整仓位
        new_decision = dict(decision)
        orig_qty = new_decision.get("quantity", 0.0)
        new_qty = orig_qty * multiplier
        if new_qty < 0:
            new_qty = 0.0
        new_decision["quantity"] = new_qty
        new_decision["fg_gate_action"] = action.value
        new_decision["fg_gate_multiplier"] = multiplier
        new_decision["fg_gate_reason"] = reason
        new_decision["fg_index_value"] = analysis.index_value
        new_decision["fg_level"] = analysis.level.value

        self._total_adjusted += 1
        if action == FGAction.BOOST:
            self._boost_count += 1
        else:
            self._reduce_count += 1

        logger.debug(
            "FGGate %s symbol=%s qty=%.6f→%.6f index=%.1f level=%s reason=%s",
            action.value.upper(), symbol, orig_qty, new_qty,
            analysis.index_value, analysis.level.value, reason,
        )

        return new_decision

    def _evaluate_rules(
        self,
        analysis: FGAnalysisResult,
        decision: Dict[str, Any],
    ) -> tuple[FGAction, float, str]:
        """根据分析结果和决策方向，应用门控规则

        恐慌贪婪指数是反向指标:
          - 恐慌时做多 (逆向入场)
          - 贪婪时做空 (逆向入场)

        Returns:
            (action, multiplier, reason)
        """
        action_str = str(decision.get("action", "")).lower()
        is_long = "buy" in action_str or "long" in action_str
        is_short = "sell" in action_str or "short" in action_str
        level = analysis.level
        idx = analysis.index_value

        # 规则1: 极度恐慌 → 做多 BOOST / 做空 REDUCE
        if level == FGLevel.EXTREME_FEAR:
            if is_long:
                return (
                    FGAction.BOOST,
                    self.EXTREME_BOOST_MULTIPLIER,
                    f"extreme_fear_long_boost idx={idx:.1f}",
                )
            if is_short:
                return (
                    FGAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_fear_short_reduce idx={idx:.1f}",
                )

        # 规则2: 恐慌 → 做多 BOOST / 做空 REDUCE
        if level == FGLevel.FEAR:
            if is_long:
                return (
                    FGAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"fear_long_boost idx={idx:.1f}",
                )
            if is_short:
                return (
                    FGAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"fear_short_reduce idx={idx:.1f}",
                )

        # 规则3: 极度贪婪 → 做多 REDUCE / 做空 BOOST
        if level == FGLevel.EXTREME_GREED:
            if is_long:
                return (
                    FGAction.REDUCE,
                    self.EXTREME_REDUCE_MULTIPLIER,
                    f"extreme_greed_long_reduce idx={idx:.1f}",
                )
            if is_short:
                return (
                    FGAction.BOOST,
                    self.EXTREME_BOOST_MULTIPLIER,
                    f"extreme_greed_short_boost idx={idx:.1f}",
                )

        # 规则4: 贪婪 → 做多 REDUCE / 做空 BOOST
        if level == FGLevel.GREED:
            if is_long:
                return (
                    FGAction.REDUCE,
                    self.REDUCE_MULTIPLIER,
                    f"greed_long_reduce idx={idx:.1f}",
                )
            if is_short:
                return (
                    FGAction.BOOST,
                    self.BOOST_MULTIPLIER,
                    f"greed_short_boost idx={idx:.1f}",
                )

        # 规则5: 中性 → ALLOW
        return FGAction.ALLOW, 1.0, "neutral"

    # ----------------------------------------------------------------
    # 合成数据生成 (沙盘测试用)
    # ----------------------------------------------------------------

    def _generate_synthetic_index(self) -> float:
        """生成合成恐慌贪婪指数 (用于沙盘测试)

        基于时间戳生成0-100之间的波动值
        """
        ts = time.time()
        # 用sin函数生成周期性波动 (24小时周期)
        wave = math.sin(ts / 86400 * 2 * math.pi) * 30  # 振幅30
        noise = ((ts * 1000) % 1000) / 1000.0 * 10 - 5  # ±5噪声
        index = 50 + wave + noise
        return max(0.0, min(100.0, index))

    # ----------------------------------------------------------------
    # 统计与辅助
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "total_calls": self._total_calls,
            "total_adjusted": self._total_adjusted,
            "boost_count": self._boost_count,
            "reduce_count": self._reduce_count,
            "no_data_count": self._no_data_count,
            "adjust_rate": (
                self._total_adjusted / self._total_calls
                if self._total_calls > 0
                else 0.0
            ),
            "last_analysis": (
                self._last_analysis.to_dict() if self._last_analysis else None
            ),
            "index_history_len": len(self._index_history),
        }

    def reset(self):
        """重置统计"""
        self._total_calls = 0
        self._total_adjusted = 0
        self._boost_count = 0
        self._reduce_count = 0
        self._no_data_count = 0
        self._last_analysis = None
        self._index_history.clear()
        logger.info("FearGreedGate stats reset")

    def round_complete(self):
        """周期完成回调

        在交易轮次结束时调用, 可用于:
          - 记录本轮统计
          - 清理临时状态
          - 触发数据刷新
        """
        stats = self.get_stats()
        logger.info(
            "FearGreedGate round complete: calls=%d adjusted=%d "
            "boost=%d reduce=%d no_data=%d adjust_rate=%.2f",
            stats["total_calls"],
            stats["total_adjusted"],
            stats["boost_count"],
            stats["reduce_count"],
            stats["no_data_count"],
            stats["adjust_rate"],
        )
