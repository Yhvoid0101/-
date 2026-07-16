#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes 沙盘进化交易系统 — 入口脚本

用法:
  python run_sandbox.py --rounds 100 --population 50
  python run_sandbox.py --rounds 200 --population 100 --evolve-every 5
  python run_sandbox.py --help
"""

import argparse
import logging
import os
import sys
import time

# 确保hermes_v6在path中
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from sandbox_trading.evolution_loop import EvolutionLoop

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sandbox")


def main():
    # Phase E: 激活沙盘模式 — 让 StrategyRegistry.load() 跳过 SANDBOX_UNSUPPORTED_SEEDS
    # 10个非现货/永续策略（期权4+链上2+MEV2+跨链2）在沙盘模式下不加载
    os.environ["HERMES_SANDBOX_MODE"] = "1"

    parser = argparse.ArgumentParser(
        description="Hermes 沙盘进化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_sandbox.py --rounds 100 --population 50
  python run_sandbox.py --rounds 500 --population 200 --evolve-every 5
  python run_sandbox.py --symbols BTC-USDT,ETH-USDT,SOL-USDT --rounds 200
  python run_sandbox.py --no-mock --capital 50000
        """,
    )

    parser.add_argument("--rounds", type=int, default=500,
                        help="沙盘运行轮数 (默认: 100)")
    parser.add_argument("--population", type=int, default=50,
                        help="子Agent种群规模 (默认: 50)")
    parser.add_argument("--evolve-every", type=int, default=10,
                        help="每N轮进化一次 (默认: 10)")
    parser.add_argument("--symbols", type=str, default="BTC-USDT,ETH-USDT",
                        help="交易对，逗号分隔 (默认: BTC-USDT,ETH-USDT)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="每个Agent的初始资金 (默认: 10000)")
    parser.add_argument("--seed-count", type=int, default=0,
                        help="种子策略数量 (默认: 0=全部随机)")
    parser.add_argument("--bars", type=int, default=4000,
                        help="每个交易对的历史K线数 (默认: 4000, v3.10.16: 覆盖2个完整牛熊周期, 防止单边做多进化偏向)")
    parser.add_argument("--no-mock", action="store_true",
                        help="(已弃用 v502) 不使用模拟数据, 现默认即为真实数据")
    parser.add_argument("--mock", action="store_true",
                        help="v502 反温室工程化: 强制使用模拟(合成)数据 (默认关闭, "
                             "默认使用 data/real_klines/ 真实K线, 杜绝数据温室)")
    parser.add_argument("--ticks-per-round", type=int, default=48,
                        help="每轮推进多少根K线，模拟一天的交易 (默认: 24)")
    parser.add_argument("--output", type=str, default="",
                        help="输出目录")
    parser.add_argument("--quiet", action="store_true",
                        help="减少日志输出")
    parser.add_argument("--intraday", action="store_true",
                        help="v2.35: 使用日内交易场景初始化 (高胜率世界观+日内硬约束)")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    symbols = [s.strip() for s in args.symbols.split(",")]
    # v502 反温室工程化: 默认使用真实数据 (杜绝"模拟牛逼实盘亏钱")
    # v501 根因: 沙盘合成数据 BTC=124238 vs 真实 BTC=74632, 偏差+66.47%
    # 修复: 默认 use_mock=False, 仅当显式传 --mock 时才用合成数据
    use_mock = args.mock  # 默认 False (真实数据), --mock 开启合成数据

    data_dir = args.output or os.path.join(
        os.path.dirname(__file__), "..", "data", "sandbox_trading",
    )

    print("=" * 60)
    print("  Hermes 沙盘进化交易系统")
    print("=" * 60)
    print(f"  种群规模:    {args.population} 个子Agent")
    print(f"  沙盘轮数:    {args.rounds}")
    print(f"  进化周期:    每 {args.evolve_every} 轮")
    print(f"  交易对:      {', '.join(symbols)}")
    print(f"  初始资金:    ${args.capital:,.0f} / Agent")
    print(f"  数据源:      {'模拟数据' if use_mock else '外部数据源'}")
    print(f"  K线/轮:      {args.ticks_per_round}")
    print(f"  日内场景:    {'启用(高胜率世界观+日内硬约束)' if args.intraday else '未启用'}")
    print(f"  输出目录:    {data_dir}")
    print("=" * 60)
    print()

    # 初始化
    t0 = time.time()
    # v2.6: 计算最大进化代数，用于分阶段变异率
    max_generations = max(1, args.rounds // args.evolve_every)
    loop = EvolutionLoop(
        population_size=args.population,
        initial_capital=args.capital,
        symbols=symbols,
        data_dir=data_dir,
        max_generations=max_generations,
        use_mock=use_mock,
    )

    loop.initialize(
        bars_per_symbol=args.bars,
        use_mock=use_mock,
        seed_count=args.seed_count,
        use_intraday=args.intraday,
    )

    init_time = time.time() - t0
    print(f"初始化完成 (耗时 {init_time:.1f}s)")
    print()

    # 运行
    print(f"开始沙盘进化 ({args.rounds} 轮)...")
    print()

    t1 = time.time()
    results = loop.run(
        rounds=args.rounds,
        evolve_every=args.evolve_every,
        ticks_per_round=args.ticks_per_round,
    )
    run_time = time.time() - t1

    print()
    print("=" * 60)
    print("  进化完成!")
    print("=" * 60)
    print(f"  总耗时:      {run_time:.1f}s")
    print(f"  总轮数:      {args.rounds}")
    print(f"  总代数:      {loop.population.generation}")
    print(f"  总交易:      {loop.engine.stats['total_trades']}")
    print(f"  总交易量:    ${loop.engine.stats['total_volume']:,.0f}")
    print(f"  总手续费:    ${loop.engine.stats['total_fees']:,.2f}")
    print(f"  爆仓次数:    {loop.engine.stats['liquidations']}")
    print()

    # 精英信号
    elite = loop.get_elite_signals(5)
    print("--- 精英前5 ---")
    print(f"{'排名':<4} {'Agent ID':<18} {'GT-Score':<10} {'夏普':<8} {'胜率':<8} {'世界观':<16} {'标签':<14}")
    print("-" * 80)
    for i, e in enumerate(elite):
        print(f"{i+1:<4} {e['agent_id']:<18} {e['gt_score']:<10.1f} {e['sharpe']:<8.2f} {e['win_rate']:<8.1%} {e['worldview']:<16} {e['style_label']:<14}")

    print()
    print("--- 世界观分布 ---")
    if loop.population.history:
        last_stats = loop.population.history[-1]
        for wv, count in sorted(last_stats.worldview_distribution.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"  {wv:<20} {count:>3} {bar}")

    # 导出结果
    result_file = loop.export_results()
    print(f"\n结果已导出: {result_file}")

    print("\n沙盘进化完成。")
    print("提示: 精英策略需要再经过沙盘200轮验证 + 模拟盘2周才能进入实盘。")


if __name__ == "__main__":
    main()