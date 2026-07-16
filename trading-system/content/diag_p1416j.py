#!/usr/bin/env python3
"""
Phase 14.16j.1 零交易窗口诊断
分析 42 个 -4.5 回退代次的:
1. walk-forward 窗口数据质量 (BTC/ETH 价格/波动率)
2. 该代的 population 组成 (worldview 分布)
3. audit.log 中该代的交易/决策日志
"""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/home/lmy/hermes_v6/data/sandbox_trading")
STATS_FILE = DATA_DIR / "evolution_stats.jsonl"
AUDIT_LOG = DATA_DIR / "audit.log"
POP_DIR = DATA_DIR  # population_gen*.jsonl

print("=" * 80)
print("Phase 14.16j.1 零交易窗口诊断")
print("=" * 80)

# Step 1: 找出所有 -4.5 回退代次
regression_gens = []
all_gens = []
with open(STATS_FILE) as f:
    for line in f:
        d = json.loads(line)
        all_gens.append(d)
        max_gt = d.get("max_gt_score", 0)
        if max_gt <= -4.4:
            regression_gens.append(d)

print(f"\n[1] -4.5 回退统计")
print(f"  总代数: {len(all_gens)}")
print(f"  -4.5 回退代数: {len(regression_gens)}")
print(f"  回退率: {len(regression_gens)/len(all_gens)*100:.1f}%")

# 回退代次列表
reg_gen_ids = [d["generation"] for d in regression_gens]
print(f"  回退代次: {reg_gen_ids[:20]}{'...' if len(reg_gen_ids) > 20 else ''}")

# 检查回退代次是否连续
if len(reg_gen_ids) > 1:
    gaps = []
    for i in range(1, len(reg_gen_ids)):
        gap = reg_gen_ids[i] - reg_gen_ids[i-1]
        gaps.append(gap)
    print(f"  代次间隔: min={min(gaps)}, max={max(gaps)}, 唯一值={sorted(set(gaps))[:10]}")

# Step 2: 分析回退代次的 population 组成
print(f"\n[2] 回退代次的 population 组成分析")
# 读取回退代次的 population 文件
worldview_counter = Counter()
total_agents = 0
sample_pop_file = None
for gen_id in reg_gen_ids[:10]:  # 只看前10个回退代次
    pop_file = POP_DIR / f"population_gen{gen_id:04d}.jsonl"
    if pop_file.exists():
        if sample_pop_file is None:
            sample_pop_file = pop_file
        with open(pop_file) as pf:
            for line in pf:
                agent = json.loads(line)
                wv = agent.get("worldview_primary", "unknown")
                worldview_counter[wv] += 1
                total_agents += 1

print(f"  样本: 前10个回退代次, {total_agents} agents")
print(f"  世界观分布 (top 10):")
for wv, cnt in worldview_counter.most_common(10):
    print(f"    {wv}: {cnt} ({cnt/total_agents*100:.1f}%)")

# Step 3: 分析一个回退代次的 agent 详情
print(f"\n[3] 样本回退代次 agent 详情 (gen {reg_gen_ids[0]})")
if sample_pop_file:
    with open(sample_pop_file) as pf:
        agents = [json.loads(line) for line in pf]
    print(f"  文件: {sample_pop_file.name}")
    print(f"  Agents 数: {len(agents)}")
    for i, a in enumerate(agents[:5]):
        print(f"\n  Agent {i}:")
        print(f"    id: {a.get('agent_id', 'N/A')}")
        print(f"    worldview: {a.get('worldview_primary', 'N/A')}")
        print(f"    fitness_score: {a.get('fitness_score', 'N/A')}")
        print(f"    total_trades: {a.get('total_trades', 'N/A')}")
        print(f"    sharpe: {a.get('sharpe', 'N/A')}")
        print(f"    win_rate: {a.get('win_rate', 'N/A')}")
        # 找所有数值字段
        numeric_keys = [k for k in a.keys() if isinstance(a[k], (int, float))]
        print(f"    numeric_fields: {numeric_keys[:15]}")

# Step 4: 分析 audit.log 中回退代次的日志
print(f"\n[4] audit.log 分析回退代次日志")
# audit.log 28MB, 只读取部分
# 找出每个回退代次的关键事件: DECIDE, TRADE, PRETRADE, CIRCUIT
if AUDIT_LOG.exists():
    # 只统计关键事件
    event_counter = Counter()
    sample_lines = []
    target_gen_strs = [f"gen={gid}" for gid in reg_gen_ids[:5]]
    target_round_strs = [f"round={gid*10}" for gid in reg_gen_ids[:5]]

    with open(AUDIT_LOG, "r", errors="ignore") as f:
        for line_num, line in enumerate(f):
            if line_num > 500000:  # 限制读取
                break
            # 检查是否包含回退代次的标记
            for marker in target_round_strs:
                if marker in line:
                    # 统计事件类型
                    if "DECIDE" in line:
                        event_counter["DECIDE"] += 1
                    elif "TRADE" in line or "submit" in line:
                        event_counter["TRADE"] += 1
                    elif "PRETRADE" in line or "pre_trade" in line:
                        event_counter["PRETRADE"] += 1
                    elif "CIRCUIT" in line or "circuit" in line:
                        event_counter["CIRCUIT"] += 1
                    elif "COOLDOWN" in line or "cooldown" in line:
                        event_counter["COOLDOWN"] += 1
                    elif "REJECT" in line or "reject" in line:
                        event_counter["REJECT"] += 1
                    elif "BLOCK" in line:
                        event_counter["BLOCK"] += 1
                    if len(sample_lines) < 20:
                        sample_lines.append(line.strip()[:200])
                    break

    print(f"  事件统计 (前5个回退代次):")
    for ev, cnt in event_counter.most_common():
        print(f"    {ev}: {cnt}")
    print(f"  样本日志 (前10行):")
    for s in sample_lines[:10]:
        print(f"    {s}")

# Step 5: 对比回退代次 vs 正常代次的 diversity 和 worldview
print(f"\n[5] 回退代次 vs 正常代次对比")
normal_gens = [d for d in all_gens if d["generation"] not in reg_gen_ids]
if normal_gens:
    reg_div = sum(d.get("diversity", 0) for d in regression_gens) / len(regression_gens)
    norm_div = sum(d.get("diversity", 0) for d in normal_gens) / len(normal_gens)
    reg_pop = sum(d.get("population_size", 0) for d in regression_gens) / len(regression_gens)
    norm_pop = sum(d.get("population_size", 0) for d in normal_gens) / len(normal_gens)
    reg_cells = sum(d.get("feature_map_cells", 0) for d in regression_gens) / len(regression_gens)
    norm_cells = sum(d.get("feature_map_cells", 0) for d in normal_gens) / len(normal_gens)
    print(f"  {'指标':<20} {'回退代次':<15} {'正常代次':<15}")
    print(f"  {'diversity':<20} {reg_div:<15.4f} {norm_div:<15.4f}")
    print(f"  {'population_size':<20} {reg_pop:<15.1f} {norm_pop:<15.1f}")
    print(f"  {'feature_map_cells':<20} {reg_cells:<15.1f} {norm_cells:<15.1f}")

# Step 6: 回退代次的 worldview 分布
print(f"\n[6] 回退代次的 worldview 分布 (vs 正常)")
reg_wv = Counter()
norm_wv = Counter()
for d in regression_gens:
    for wv, cnt in d.get("worldview_distribution", {}).items():
        reg_wv[wv] += cnt
for d in normal_gens:
    for wv, cnt in d.get("worldview_distribution", {}).items():
        norm_wv[wv] += cnt

print(f"  {'worldview':<35} {'回退':<10} {'正常':<10}")
all_wvs = set(reg_wv.keys()) | set(norm_wv.keys())
for wv in sorted(all_wvs, key=lambda x: -(reg_wv[x] + norm_wv[x])):
    print(f"  {wv[:33]:<35} {reg_wv[wv]:<10} {norm_wv[wv]:<10}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
