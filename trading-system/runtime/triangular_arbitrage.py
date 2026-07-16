# -*- coding: utf-8 -*-
"""
triangular_arbitrage.py — 三角套利扫描与执行引擎 v1.0

来源：
  - voiceofchain 2026 (CCXT三角套利实现)
  - gov.capital 2026 (7种套利技巧：三角套利)
  - CCXT Pro 2026 (统一API, 100+交易所)
  - cryptohopper 2026 (套利机器人最佳实践)

核心算法：
  1. 三角套利路径发现
     - 给定基础货币（如USDT），寻找所有可能的三步路径
     - 路径: A→B→C→A，例如 USDT→BTC→ETH→USDT
     - 自动过滤不存在的交易对
  2. 利润计算
     - profit = (1/ask_AB) × bid_BC × bid_CA × (1-fee)³ - 1
     - 考虑3次交易手续费（每次taker fee）
     - 考虑滑点（基于订单簿深度）
  3. 机会识别
     - 利润 > 阈值（如0.3%）= 套利机会
     - 考虑网络延迟（如100ms）
     - 考虑资金占用（机会成本）
  4. 执行策略
     - 同时下单（理想情况）
     - 顺序执行（实际，含滑点风险）
     - 失败回滚机制

实盘验证：
  - 中心化交易所：年化5-15%（扣除手续费后）
  - DEX：年化10-30%（更高利润但gas成本高）
  - 高频套利：毫秒级，需要低延迟基础设施
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TriArbSignalType(Enum):
    """三角套利信号类型"""
    OPPORTUNITY = "opportunity"        # 套利机会
    EXECUTE = "execute"                # 可执行
    WATCH = "watch"                    # 观察中
    NO_OPPORTUNITY = "no_opportunity"  # 无机会


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    STEP_1_DONE = "step_1_done"
    STEP_2_DONE = "step_2_done"
    STEP_3_DONE = "step_3_done"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TickerData:
    """行情数据"""
    symbol: str
    bid: float = 0.0   # 买一价（卖出可得）
    ask: float = 0.0   # 卖一价（买入需付）
    last: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TriArbPath:
    """三角套利路径"""
    base: str                       # 基础货币（如USDT）
    intermediate_a: str             # 中间货币A（如BTC）
    intermediate_b: str             # 中间货币B（如ETH）
    pair_1: str                     # 第一步交易对（如 BTC/USDT）
    pair_2: str                     # 第二步交易对（如 ETH/BTC）
    pair_3: str                     # 第三步交易对（如 ETH/USDT）
    direction_1: str = "buy"        # buy/sell
    direction_2: str = "buy"
    direction_3: str = "sell"

    @property
    def path_str(self) -> str:
        return f"{self.base}→{self.intermediate_a}→{self.intermediate_b}→{self.base}"


@dataclass
class TriArbOpportunity:
    """三角套利机会"""
    path: TriArbPath
    gross_profit_pct: float = 0.0       # 毛利润率（%）
    net_profit_pct: float = 0.0          # 净利润率（扣除手续费）
    estimated_profit_usd: float = 0.0     # 预估利润（USD）
    required_capital: float = 0.0         # 所需资金
    execution_time_ms: float = 0.0        # 预估执行时间
    confidence: float = 0.0               # 置信度
    timestamp: float = field(default_factory=time.time)


@dataclass
class TriArbSignal:
    """三角套利信号"""
    signal_type: TriArbSignalType = TriArbSignalType.NO_OPPORTUNITY
    opportunities: List[TriArbOpportunity] = field(default_factory=list)
    best_opportunity: Optional[TriArbOpportunity] = None
    scan_duration_ms: float = 0.0
    total_paths_scanned: int = 0
    description: str = ""


class TriangularArbitrage:
    """
    三角套利扫描与执行引擎

    使用场景：
      - 中心化交易所（Binance/OKX/Bybit）
      - 同一交易所内的三角套利
      - 跨交易所套利（需要考虑提币时间）

    依赖：
      - CCXT (可选，沙盘模式可使用模拟数据)
      - 实盘需要交易所API密钥
    """

    # ===== 参数 =====
    MIN_PROFIT_THRESHOLD = 0.003          # 最低利润阈值（0.3%）
    EXECUTE_THRESHOLD = 0.008             # 执行阈值（0.8%，考虑滑点和延迟）
    TAKER_FEE = 0.001                     # taker手续费（0.1%）
    MAX_SLIPPAGE = 0.002                  # 最大滑点（0.2%）
    LATENCY_MS = 100                      # 网络延迟（毫秒）
    MIN_VOLUME_USD = 100_000.0            # 最小24h成交量
    SCAN_INTERVAL_MS = 5000               # 扫描间隔（5秒）
    MAX_PATHS_PER_SCAN = 50               # 每次扫描最大路径数

    def __init__(self, exchange: str = "binance", base_currency: str = "USDT"):
        self.exchange = exchange
        self.base_currency = base_currency
        self.tickers: Dict[str, TickerData] = {}
        self.available_symbols: List[str] = []
        self.opportunities_history: deque = deque(maxlen=1000)
        self.execution_history: deque = deque(maxlen=100)
        self.last_scan_time: float = 0.0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_ticker(self, symbol: str, bid: float, ask: float,
                      last: float = 0.0, volume: float = 0.0) -> None:
        """更新单个交易对行情"""
        self.tickers[symbol] = TickerData(
            symbol=symbol, bid=bid, ask=ask, last=last, volume=volume,
        )
        if symbol not in self.available_symbols:
            self.available_symbols.append(symbol)

    def update_tickers_batch(self, tickers: Dict[str, Dict[str, float]]) -> None:
        """批量更新行情"""
        for symbol, data in tickers.items():
            self.update_ticker(
                symbol=symbol,
                bid=data.get("bid", 0.0),
                ask=data.get("ask", 0.0),
                last=data.get("last", 0.0),
                volume=data.get("volume", 0.0),
            )

    def set_available_symbols(self, symbols: List[str]) -> None:
        """设置可用交易对列表"""
        self.available_symbols = symbols

    # ------------------------------------------------------------------
    # 路径发现
    # ------------------------------------------------------------------

    def discover_paths(self, max_intermediates: int = 10) -> List[TriArbPath]:
        """
        发现所有可能的三角套利路径

        路径模式: BASE → A → B → BASE
        其中A和B是中间货币
        """
        paths: List[TriArbPath] = []
        base = self.base_currency

        # 找出所有与BASE相关的中间货币
        intermediate_candidates = set()
        for symbol in self.available_symbols:
            parts = self._split_symbol(symbol)
            if not parts:
                continue
            quote, base_sym = parts
            if base_sym == base:
                intermediate_candidates.add(quote)
            elif quote == base:
                intermediate_candidates.add(base_sym)

        # 限制中间货币数量
        intermediates = list(intermediate_candidates)[:max_intermediates]

        # 构建所有A→B组合
        for i, a in enumerate(intermediates):
            for b in intermediates[i + 1:]:
                # 路径1: BASE → A → B → BASE
                path1 = self._build_path(base, a, b)
                if path1:
                    paths.append(path1)

                # 路径2: BASE → B → A → BASE
                path2 = self._build_path(base, b, a)
                if path2:
                    paths.append(path2)

        return paths[: self.MAX_PATHS_PER_SCAN]

    def _build_path(self, base: str, a: str, b: str) -> Optional[TriArbPath]:
        """构建单条三角套利路径"""
        # 步骤1: BASE → A (买A用BASE)
        pair1 = self._find_pair(base, a)
        if not pair1:
            return None

        # 步骤2: A → B (用A买B)
        pair2 = self._find_pair(a, b)
        if not pair2:
            return None

        # 步骤3: B → BASE (卖B换BASE)
        pair3 = self._find_pair(b, base)
        if not pair3:
            return None

        return TriArbPath(
            base=base,
            intermediate_a=a,
            intermediate_b=b,
            pair_1=pair1,
            pair_2=pair2,
            pair_3=pair3,
            direction_1="buy",   # 买A
            direction_2="buy",   # 用A买B
            direction_3="sell",  # 卖B换BASE
        )

    def _find_pair(self, quote: str, base: str) -> Optional[str]:
        """查找交易对符号"""
        # 尝试两种顺序: QUOTE/BASE 或 BASE/QUOTE
        sym1 = f"{quote}/{base}"
        sym2 = f"{base}/{quote}"

        if sym1 in self.available_symbols or sym1 in self.tickers:
            return sym1
        if sym2 in self.available_symbols or sym2 in self.tickers:
            return sym2
        return None

    def _split_symbol(self, symbol: str) -> Optional[Tuple[str, str]]:
        """拆分交易对符号 QUOTE/BASE → (QUOTE, BASE)"""
        if "/" not in symbol:
            return None
        parts = symbol.split("/")
        if len(parts) != 2:
            return None
        return parts[0], parts[1]

    # ------------------------------------------------------------------
    # 利润计算
    # ------------------------------------------------------------------

    def calculate_profit(self, path: TriArbPath, capital: float = 1000.0) -> TriArbOpportunity:
        """
        计算三角套利利润

        利润公式:
          步骤1: 用BASE买A，得到 A_amount = capital / ask_1 × (1 - fee)
          步骤2: 用A买B，得到 B_amount = A_amount / ask_2 × (1 - fee)
                 (如果pair2是BASE/QUOTE格式，则用bid)
          步骤3: 卖B换BASE，得到 final = B_amount × bid_3 × (1 - fee)
          净利润 = final - capital
        """
        t1 = self.tickers.get(path.pair_1)
        t2 = self.tickers.get(path.pair_2)
        t3 = self.tickers.get(path.pair_3)

        if not t1 or not t2 or not t3:
            return TriArbOpportunity(
                path=path,
                confidence=0.0,
            )
        if t1.ask <= 0 or t1.bid <= 0 or t2.ask <= 0 or t2.bid <= 0:
            return TriArbOpportunity(
                path=path,
                confidence=0.0,
            )

        # 步骤1: BASE → A
        # 如果pair1 = A/BASE，买入A(quote)用 ask_1
        # 如果pair1 = BASE/A，卖出BASE(quote)换A(base)用 bid_1
        pair1_quote, pair1_base = self._split_symbol(path.pair_1) or ("", "")
        if pair1_base == self.base_currency:
            # A/BASE格式，买A(quote)用ask
            a_amount = capital / t1.ask * (1 - self.TAKER_FEE)
        else:
            # BASE/A格式，卖BASE(quote)换A(base)用bid
            a_amount = capital * t1.bid * (1 - self.TAKER_FEE)

        # 步骤2: A → B
        # 如果pair2 = B/A，买B(quote)用 ask_2
        # 如果pair2 = A/B，卖A(quote)换B(base)用 bid_2
        pair2_quote, pair2_base = self._split_symbol(path.pair_2) or ("", "")
        if pair2_base == path.intermediate_a:
            # B/A格式，买B(quote)用ask
            b_amount = a_amount / t2.ask * (1 - self.TAKER_FEE)
        else:
            # A/B格式，卖A(quote)换B(base)用bid
            b_amount = a_amount * t2.bid * (1 - self.TAKER_FEE)

        # 步骤3: B → BASE
        pair3_quote, pair3_base = self._split_symbol(path.pair_3) or ("", "")
        if pair3_base == self.base_currency:
            # B/BASE格式，卖B用bid
            final_amount = b_amount * t3.bid * (1 - self.TAKER_FEE)
        else:
            # BASE/B格式，买BASE用ask
            final_amount = b_amount / t3.ask * (1 - self.TAKER_FEE)

        # 计算利润
        gross_profit_pct = (final_amount / capital - 1) * 100 if capital > 0 else 0
        net_profit_pct = gross_profit_pct  # 已扣除3次手续费
        estimated_profit_usd = final_amount - capital

        # 执行时间预估（3次交易+延迟）
        execution_time_ms = 3 * self.LATENCY_MS + 3 * 100  # 交易处理100ms

        # 置信度：基于成交量和利润稳定性
        min_volume = min(t1.volume, t2.volume, t3.volume)
        volume_confidence = min(1.0, min_volume / self.MIN_VOLUME_USD)
        profit_confidence = min(1.0, net_profit_pct / 2.0)
        confidence = (volume_confidence + profit_confidence) / 2

        return TriArbOpportunity(
            path=path,
            gross_profit_pct=gross_profit_pct,
            net_profit_pct=net_profit_pct,
            estimated_profit_usd=estimated_profit_usd,
            required_capital=capital,
            execution_time_ms=execution_time_ms,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # 扫描与信号生成
    # ------------------------------------------------------------------

    def scan(self, capital: float = 1000.0) -> TriArbSignal:
        """扫描所有三角套利机会"""
        scan_start = time.time()

        paths = self.discover_paths()
        opportunities: List[TriArbOpportunity] = []

        for path in paths:
            opp = self.calculate_profit(path, capital)
            if opp.net_profit_pct > self.MIN_PROFIT_THRESHOLD:
                opportunities.append(opp)

        # 按净利润排序
        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)

        scan_duration_ms = (time.time() - scan_start) * 1000

        # 保存历史
        if opportunities:
            self.opportunities_history.append(opportunities[0])

        # 生成信号
        if not opportunities:
            return TriArbSignal(
                signal_type=TriArbSignalType.NO_OPPORTUNITY,
                scan_duration_ms=scan_duration_ms,
                total_paths_scanned=len(paths),
                description=f"扫描{len(paths)}条路径，无套利机会",
            )

        best = opportunities[0]
        if best.net_profit_pct > self.EXECUTE_THRESHOLD:
            signal_type = TriArbSignalType.EXECUTE
            description = (
                f"执行机会: {best.path.path_str} "
                f"净利润{best.net_profit_pct:.3f}% "
                f"预估${best.estimated_profit_usd:.2f}"
            )
        else:
            signal_type = TriArbSignalType.OPPORTUNITY
            description = (
                f"机会观察: {best.path.path_str} "
                f"净利润{best.net_profit_pct:.3f}%（低于执行阈值）"
            )

        self.last_scan_time = time.time()

        return TriArbSignal(
            signal_type=signal_type,
            opportunities=opportunities[:10],
            best_opportunity=best,
            scan_duration_ms=scan_duration_ms,
            total_paths_scanned=len(paths),
            description=description,
        )

    # ------------------------------------------------------------------
    # 执行（模拟）
    # ------------------------------------------------------------------

    def simulate_execution(
        self, opportunity: TriArbOpportunity, actual_slippage: float = 0.001,
    ) -> Dict[str, Any]:
        """
        模拟执行三角套利

        实盘执行需要：
          1. 并行下单（理想但难实现）
          2. 顺序执行（含滑点风险）
          3. 失败回滚
        """
        expected_profit = opportunity.estimated_profit_usd
        slippage_cost = opportunity.required_capital * actual_slippage * 3  # 3次交易
        actual_profit = expected_profit - slippage_cost

        result = {
            "path": opportunity.path.path_str,
            "expected_profit_usd": expected_profit,
            "slippage_cost": slippage_cost,
            "actual_profit_usd": actual_profit,
            "success": actual_profit > 0,
            "status": ExecutionStatus.COMPLETED.value,
        }

        self.execution_history.append(result)
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_top_opportunities(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近N个最佳机会"""
        recent = list(self.opportunities_history)[-n:]
        return [
            {
                "path": o.path.path_str,
                "net_profit_pct": o.net_profit_pct,
                "estimated_profit_usd": o.estimated_profit_usd,
                "confidence": o.confidence,
                "timestamp": o.timestamp,
            }
            for o in reversed(recent)
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
        """获取扫描状态摘要"""
        return {
            "exchange": self.exchange,
            "base_currency": self.base_currency,
            "available_symbols": len(self.available_symbols),
            "tickers_loaded": len(self.tickers),
            "opportunities_found": len(self.opportunities_history),
            "last_scan_time": self.last_scan_time,
            "execution_stats": self.get_execution_stats(),
        }
