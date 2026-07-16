#!/usr/bin/env python3
"""Phase 14.16i-v2: population=30 重新验证 -4.5回退根因修复

根因确认:
  - behavior_diversity.py L406: 零交易agent gt_score = -5.0 + norm_novelty*0.5 ∈ [-5.0, -4.5]
  - population=5时5个agent全零交易 → 整代max_gt=-4.5 (47/100代)
  - 14.16d基线 population=30 → 0/100代-4.5回退

修复方案:
  - population: 5 → 30 (与14.16d基线一致)
  - 预期: -4.5回退代次大幅减少 (接近0/100)
  - 同时追踪4指标: max_gt_score / diversity / mean_sharpe / max_total_pnl
"""
import sys
import os
import time
import json
from pathlib import Path

SANDBOX = "/home/lmy/hermes_v6/sandbox_trading"
PARENT = "/home/lmy/hermes_v6"
sys.path.insert(0, PARENT)
sys.path.insert(0, SANDBOX)

os.chdir(SANDBOX)

print("=" * 70)
print("Phase 14.16i-v2: population=30 验证 (-4.5回退根因修复)")
print("=" * 70)
print(f"配置: 2币种(BTC,ETH), population=30, 1000轮, evolve_every=10 (100代)")
print(f"预期: -4.5回退代次大幅减少 (14.16d基线=0/100, 14.16i-v1=47/100)")
print()

# 导入
from sandbox_trading.evolution_loop import EvolutionLoop

# 运行
start_time = time.time()
loop = EvolutionLoop(
    symbols=["BTCUSDT", "ETHUSDT"],
    population_size=30,  # 关键修复: 5 → 30
    initial_capital=100000.0,
)
loop.initialize()

results = loop.run(rounds=1000, evolve_every=10, ticks_per_round=24)
elapsed = time.time() - start_time

print(f"\n{'=' * 70}")
print(f"Phase 14.16i-v2 运行完成")
print(f"{'=' * 70}")
print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
print(f"总轮数: {len(results)}")
print(f"总交易: {sum(r.trades_executed for r in results)}")
if hasattr(results[0], 'liquidations'):
    print(f"总爆仓: {sum(r.liquidations for r in results)}")

# 保存结果摘要
result_summary = {
    "phase": "14.16i-v2",
    "config": {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "population_size": 30,
        "rounds": 1000,
        "evolve_every": 10,
        "ticks_per_round": 24,
    },
    "elapsed_seconds": elapsed,
    "total_rounds": len(results),
    "total_trades": sum(r.trades_executed for r in results),
    "total_liquidations": sum(r.liquidations for r in results) if hasattr(results[0], 'liquidations') else 0,
}

result_path = Path("/home/lmy/hermes_v6/data/sandbox_trading/p1416i_v2_result.json")
result_path.write_text(json.dumps(result_summary, indent=2, ensure_ascii=False))
print(f"\n结果已保存: {result_path}")
