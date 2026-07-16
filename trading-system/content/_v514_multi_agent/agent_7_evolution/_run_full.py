#!/usr/bin/env python3
"""运行完整 NSGA-II (pop=50, gen=20, 3标的) + 分析 + 报告"""
import sys
import time
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
SANDBOX = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SANDBOX))

from _v515_ensemble_strategy import load_klines, calc_indicators, TARGETS
from _v515_evolution_engine import NSGA2Engine
from _v515_pareto_analyzer import ParetoAnalyzer

# 3 标的 (BTC/ETH/SOL 流动性最好, 代表性最强)
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
pop_size = 50
generations = 20

print("=" * 70)
print("v515 NSGA-II 完整运行 (Agent-7)")
print("=" * 70)
print(f"标的: {symbols}")
print(f"种群: {pop_size}, 代数: {generations}")
print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

t0 = time.time()

# 加载数据
all_data = {}
for sym in symbols:
    bars = load_klines(sym)
    if not bars:
        print(f"{sym}: FAIL, 跳过")
        continue
    inds = calc_indicators(bars)
    all_data[sym] = (bars, inds)
    print(f"{sym}: {len(bars)} bars")

if not all_data:
    print("[ERROR] 无数据")
    sys.exit(1)

# 运行 NSGA-II
engine = NSGA2Engine(
    all_data=all_data,
    pop_size=pop_size,
    generations=generations,
    inject_memory=True,
)
front = engine.run(verbose=True)

# 保存帕累托前沿 JSON
result = {
    "version": "v515-nsga2",
    "timestamp": time.time(),
    "pop_size": pop_size,
    "generations": generations,
    "symbols": symbols,
    "objectives": ["sharpe", "win_rate", "n_trades", "monthly_loss", "profit_loss_ratio"],
    "objective_directions": {
        "sharpe": True, "win_rate": True, "n_trades": True,
        "monthly_loss": False, "profit_loss_ratio": True,
    },
    "pareto_front_size": len(front),
    "pareto_front": [
        {
            "genes": ind.genes,
            "objectives": {
                "sharpe": ind.sharpe,
                "win_rate": ind.win_rate,
                "n_trades": ind.n_trades,
                "monthly_loss": ind.monthly_loss,
                "profit_loss_ratio": ind.profit_loss_ratio,
            },
            "kpi": {
                "kpi_pass": ind.kpi_pass,
                "max_dd": ind.max_dd,
                "annual_return": ind.annual_return,
                "consec_loss": ind.consec_loss,
                "win_rate_ci_lo": ind.win_rate_ci_lo,
                "is_sharpe": ind.is_sharpe,
                "df": ind.df,
            },
            "fitness_scalar": ind.fitness_scalar,
        }
        for ind in front
    ],
    "history": engine.history,
    "targets": TARGETS,
}
out_json = HERE / "_v515_pareto_front.json"
out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

# 生成分析报告
analyzer = ParetoAnalyzer()
analyzer.load(out_json)
report = analyzer.analyze()

# 打印推荐解
print("\n" + "=" * 70)
print("帕累托前沿推荐解")
print("=" * 70)
print(f"\n前沿大小: {report.front_size}")
print(f"理想点: {report.ideal_point}")
print(f"Nadir点: {report.nadir_point}")
print(f"收敛状态: {report.convergence.get('status')}")
print(f"\n推荐解:")
for name, sol in report.recommendations.items():
    o = sol.objectives
    print(f"  {name:25s}: 夏普={o.get('sharpe',0):.3f}, 胜率={o.get('win_rate',0)*100:.1f}%, "
          f"交易数={o.get('n_trades',0):.1f}, 月亏={o.get('monthly_loss',0)*100:.1f}%, "
          f"盈亏比={o.get('profit_loss_ratio',0):.2f}, KPI={sol.kpi.get('kpi_pass',0)}/7")

# ASCII 可视化
print("\n=== ASCII 可视化 ===")
analyzer.print_ascii_scatter("sharpe", "win_rate")
print()
analyzer.print_parallel_coordinates()

# 保存 Markdown 报告
report_path = HERE / "_v515_evolution_report.md"
analyzer.save_markdown_report(report_path)

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"完成! 总耗时: {elapsed/60:.1f}分钟")
print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"帕累托前沿 JSON: {out_json}")
print(f"分析报告 MD: {report_path}")
print(f"进化记忆: {HERE / '_v515_evolution_memory.json'}")
print(f"{'=' * 70}")
