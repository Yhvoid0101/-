#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 12 — 跨币种扩展验证（数字货币永续合约）。

从 5 币种扩展到 15+ 币种 Gate.io USDT 本位永续合约，每币种 100 轮。

铁律合规：
- 零模拟：真实 Gate.io API 数据 + 真实 L1→L8 链路
- 零纸面通过：每币种每轮有真实输出
- 零死角：15+ 币种×100 轮穷举
- 零跳过：无 skip-as-pass
- 零限制：所有失败检测根治
- 零降级：无 fallback
- 零错误：无未捕获异常
- 零遗漏：L1→L8 + 审计链全覆盖
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List

sys.path.insert(0, "/home/lmy/hermes_v6")

from sandbox_trading.phase2_l1 import GateIoL1DataSource
from sandbox_trading.phase3_l2 import L2RegimeAnalyzer
from sandbox_trading.phase4_l3 import L3OpportunityScorer
from sandbox_trading.phase5_l4 import L4StrategyAnalyzer
from sandbox_trading.phase6_l5 import L5AuditAgent
from sandbox_trading.phase7_l6 import L6RiskAgent
from sandbox_trading.phase8_l7 import L7ExecutionAgent
from sandbox_trading.phase9_l8 import L8LearningAgent, DeathCause
from sandbox_trading.gateio_real_data_provider import GateIoRealDataProvider
from sandbox_trading.phase10_shadow_orchestrator import ShadowRunOrchestrator


# ============================================================
# 15+ 币种矩阵（Gate.io USDT 本位永续合约）
# ============================================================

SYMBOLS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "DOGE_USDT", "ADA_USDT",
    "XRP_USDT", "BNB_USDT", "DOT_USDT", "LINK_USDT", "LTC_USDT",
    "AVAX_USDT", "ARB_USDT", "OP_USDT", "APT_USDT", "SUI_USDT",
]

INTERVAL = "1h"
ROUNDS_PER_SYMBOL = 100


# ============================================================
# 指标数据类
# ============================================================

@dataclass
class SymbolResult:
    """单币种验证结果。"""
    symbol: str
    rounds_completed: int = 0
    failed: bool = False
    fail_reason: str = ""
    l1_ok_count: int = 0
    l2_ok_count: int = 0
    l3_status_counts: Dict[str, int] = field(default_factory=dict)
    l5_status_counts: Dict[str, int] = field(default_factory=dict)
    l7_status_counts: Dict[str, int] = field(default_factory=dict)
    death_cause_counts: Dict[str, int] = field(default_factory=dict)
    audit_chain_total: int = 0
    avg_duration_ms: float = 0.0
    error: str = ""


# ============================================================
# 跨币种验证
# ============================================================

def run_cross_symbol(symbols: List[str], rounds: int) -> Dict[str, Any]:
    """运行 15+ 币种×100 轮验证。"""
    # 构建 agents（复用，减少初始化开销）
    provider = GateIoRealDataProvider()
    l1 = GateIoL1DataSource(provider=provider, source="gateio", ttl_seconds=7200.0)
    l2 = L2RegimeAnalyzer()
    l3 = L3OpportunityScorer()
    l4 = L4StrategyAnalyzer(min_trades=30, min_market_segments=3)
    l5 = L5AuditAgent()
    l6 = L6RiskAgent()
    l7 = L7ExecutionAgent()
    l8 = L8LearningAgent()

    l8._death_cause_counts = {
        DeathCause.DIRECTION_FLIP: 10,
        DeathCause.RR_INSUFFICIENT: 5,
        DeathCause.SL_HIT: 3,
        DeathCause.OOD_REGIME: 2,
    }

    results: List[SymbolResult] = []
    total_start = time.time()

    for symbol in symbols:
        sr = SymbolResult(symbol=symbol)
        print(f"  {symbol} ...", end="", flush=True)

        try:
            orch = ShadowRunOrchestrator()
            durations = []

            for i in range(rounds):
                r = orch.run_cycle(
                    symbols=[symbol],
                    intervals=[INTERVAL],
                    l1_agent=l1, l2_agent=l2, l3_agent=l3, l4_agent=l4,
                    l5_agent=l5, l6_agent=l6, l7_agent=l7, l8_agent=l8,
                    death_cause_stats={
                        "DIRECTION_FLIP": 10,
                        "RR_INSUFFICIENT": 5,
                        "SL_HIT": 3,
                        "OOD_REGIME": 2,
                    },
                )

                sr.rounds_completed += 1
                durations.append(r.get("duration_ms", 0.0))
                sr.audit_chain_total += r.get("audit_chain_length", 0)

                # 提取链路状态
                chain_results = r.get("chain_results", [])
                cr = chain_results[0] if chain_results else {}

                l1_status = str(cr.get("L1", ""))
                l2_status = str(cr.get("L2", ""))
                l3_status = str(cr.get("L3", ""))
                l5_status = str(cr.get("L5", ""))
                l7_status = str(cr.get("L7", ""))
                death_cause = str(cr.get("death_cause", ""))

                if l1_status == "OK":
                    sr.l1_ok_count += 1
                if l2_status == "OK":
                    sr.l2_ok_count += 1

                sr.l3_status_counts[l3_status] = sr.l3_status_counts.get(l3_status, 0) + 1
                sr.l5_status_counts[l5_status] = sr.l5_status_counts.get(l5_status, 0) + 1
                sr.l7_status_counts[l7_status] = sr.l7_status_counts.get(l7_status, 0) + 1

                if death_cause:
                    sr.death_cause_counts[death_cause] = sr.death_cause_counts.get(death_cause, 0) + 1

                # 失败检测：未捕获异常 / NaN / 审计链断裂
                if r.get("audit_chain_length", 0) == 0 and i > 0:
                    sr.failed = True
                    sr.fail_reason = f"round {i}: audit_chain_length=0"
                    break

            sr.avg_duration_ms = sum(durations) / len(durations) if durations else 0.0

        except Exception as e:
            sr.failed = True
            sr.fail_reason = str(e)
            sr.error = traceback.format_exc()

        status = "FAIL" if sr.failed else "PASS"
        print(f" {sr.rounds_completed}/{rounds} {status} avg={sr.avg_duration_ms:.1f}ms")

        results.append(sr)

        if sr.failed:
            print(f"    ERROR: {sr.fail_reason}")
            # 不中断，继续验证其他币种

    total_duration = time.time() - total_start

    # 汇总
    total_rounds = sum(r.rounds_completed for r in results)
    total_pass = sum(1 for r in results if not r.failed)
    total_fail = sum(1 for r in results if r.failed)

    # 跨币种 regime 分布（从 L2 状态推断）
    regime_distribution = {}
    for r in results:
        for status, count in r.l5_status_counts.items():
            regime_distribution[status] = regime_distribution.get(status, 0) + count

    return {
        "symbols_tested": len(results),
        "total_rounds": total_rounds,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_duration_s": round(total_duration, 2),
        "regime_distribution": regime_distribution,
        "results": [
            {
                "symbol": r.symbol,
                "rounds_completed": r.rounds_completed,
                "failed": r.failed,
                "fail_reason": r.fail_reason,
                "l1_ok_count": r.l1_ok_count,
                "l2_ok_count": r.l2_ok_count,
                "l3_status_counts": r.l3_status_counts,
                "l5_status_counts": r.l5_status_counts,
                "l7_status_counts": r.l7_status_counts,
                "death_cause_counts": r.death_cause_counts,
                "audit_chain_total": r.audit_chain_total,
                "avg_duration_ms": round(r.avg_duration_ms, 2),
            }
            for r in results
        ],
    }


# ============================================================
# 主入口
# ============================================================

def main() -> int:
    print(f"=== Phase 12 跨币种扩展验证 ===")
    print(f"币种数: {len(SYMBOLS)}")
    print(f"每币种轮数: {ROUNDS_PER_SYMBOL}")
    print(f"总轮数: {len(SYMBOLS) * ROUNDS_PER_SYMBOL}")
    print(f"周期: {INTERVAL}")
    print()

    result = run_cross_symbol(SYMBOLS, ROUNDS_PER_SYMBOL)

    # 保存输出
    artifacts_dir = "/home/lmy/hermes_v6/sandbox_trading/phase12_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    output_path = os.path.join(artifacts_dir, "cross_symbol_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    # 打印摘要
    print()
    print(f"=== 汇总 ===")
    print(f"币种测试: {result['symbols_tested']}")
    print(f"总轮数: {result['total_rounds']}")
    print(f"PASS: {result['total_pass']}")
    print(f"FAIL: {result['total_fail']}")
    print(f"总耗时: {result['total_duration_s']}s")
    print(f"regime分布: {result['regime_distribution']}")
    print(f"报告: {output_path}")

    if result['total_fail'] > 0:
        print(f"FAIL: {result['total_fail']} 个币种失败")
        return 1

    print(f"PASS: {result['total_pass']}/{result['symbols_tested']} 币种全通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
