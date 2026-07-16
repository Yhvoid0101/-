#!/usr/bin/env python3
"""
Phase 14.16j.1-v2 精准诊断
问题: evolution_stats.jsonl 显示 max_gt_score=-4.5, 但 population_gen*.jsonl 的 fitness_score=-0.075
矛盾: population_gen*.jsonl 保存的是进化后下一代, 不是运行数据

精准诊断:
1. 列出所有 -4.5 回退代次的完整 evolution_stats 记录
2. 对比相邻代次 (回退代次的前一代 vs 后一代)
3. 分析 audit.log 找出真实的零交易证据
4. 检查 behavior_diversity.py 的 -4.5 注入逻辑触发条件
"""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/home/lmy/hermes_v6/data/sandbox_trading")
STATS_FILE = DATA_DIR / "evolution_stats.jsonl"
AUDIT_LOG = DATA_DIR / "audit.log"

print("=" * 80)
print("Phase 14.16j.1-v2 精准诊断 -4.5 回退真实原因")
print("=" * 80)

# 读取所有 stats
all_gens = []
with open(STATS_FILE) as f:
    for line in f:
        all_gens.append(json.loads(line))

print(f"\n总代数: {len(all_gens)}")

# Step 1: 找出所有 -4.5 回退代次的完整记录
print(f"\n[1] -4.5 回退代次完整记录")
regression_gens = []
for d in all_gens:
    if d.get("max_gt_score", 0) <= -4.4:
        regression_gens.append(d)

print(f"回退代数: {len(regression_gens)}")
print(f"\n前 15 个回退代次:")
print(f"{'gen':<6} {'max_gt':<12} {'mean_gt':<12} {'pop':<6} {'diversity':<10} {'cells':<6}")
for d in regression_gens[:15]:
    print(f"{d['generation']:<6} {d['max_gt_score']:<12.4f} {d['mean_gt_score']:<12.4f} "
          f"{d['population_size']:<6} {d.get('diversity',0):<10.4f} {d.get('feature_map_cells',0):<6}")

# Step 2: 对比相邻代次
print(f"\n[2] 相邻代次对比 (回退前 vs 回退 vs 回退后)")
gen_map = {d["generation"]: d for d in all_gens}
for d in regression_gens[:10]:
    g = d["generation"]
    prev_d = gen_map.get(g - 1)
    next_d = gen_map.get(g + 1)
    print(f"\n  Gen {g}:")
    if prev_d:
        print(f"    前 Gen {g-1}: max_gt={prev_d['max_gt_score']:.4f}, mean_gt={prev_d['mean_gt_score']:.4f}, pop={prev_d['population_size']}")
    print(f"    本 Gen {g}:  max_gt={d['max_gt_score']:.4f}, mean_gt={d['mean_gt_score']:.4f}, pop={d['population_size']}, diversity={d.get('diversity',0):.4f}")
    if next_d:
        print(f"    后 Gen {g+1}: max_gt={next_d['max_gt_score']:.4f}, mean_gt={next_d['mean_gt_score']:.4f}, pop={next_d['population_size']}")

# Step 3: 统计回退代次的 max_gt_score 分布
print(f"\n[3] 回退代次 max_gt_score 分布")
score_dist = Counter()
for d in regression_gens:
    score = round(d["max_gt_score"], 2)
    score_dist[score] += 1
for score, cnt in sorted(score_dist.items()):
    print(f"  {score}: {cnt}")

# Step 4: 检查回退代次是否有共同的 timestamp 模式
print(f"\n[4] 回退代次 timestamp 模式 (是否周期性)")
for d in regression_gens[:10]:
    ts = d.get("timestamp", 0)
    import time
    ts_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "N/A"
    print(f"  Gen {d['generation']}: ts={ts_str}")

# Step 5: 分析 audit.log 找 -4.5 或零交易证据
print(f"\n[5] audit.log 搜索 -4.5 / 零交易 / zero_trade / no_trade 证据")
patterns = ["-4.5", "-4.50", "zero_trade", "no_trade", "total_trades=0", "trades=0", "NO_TRADES"]
found_lines = {p: [] for p in patterns}

with open(AUDIT_LOG, "r", errors="ignore") as f:
    for line_num, line in enumerate(f):
        if line_num > 1000000:
            break
        for p in patterns:
            if p in line:
                if len(found_lines[p]) < 5:
                    found_lines[p].append((line_num, line.strip()[:250]))
                break

for p, lines in found_lines.items():
    if lines:
        print(f"\n  Pattern '{p}' 找到 {len(lines)} 行样本:")
        for ln, content in lines[:3]:
            print(f"    L{ln}: {content}")
    else:
        print(f"  Pattern '{p}': 未找到")

# Step 6: 检查 behavior_diversity.py 触发条件
print(f"\n[6] behavior_diversity.py -4.5 注入逻辑检查")
BD_FILE = Path("/home/lmy/hermes_v6/sandbox_trading/behavior_diversity.py")
if BD_FILE.exists():
    with open(BD_FILE) as f:
        bd_content = f.read()
    # 找到 -5.0 + exploration_bonus 的上下文
    idx = bd_content.find("exploration_bonus = norm_novelty * 0.5")
    if idx >= 0:
        start = max(0, idx - 800)
        end = min(len(bd_content), idx + 500)
        context = bd_content[start:end]
        print(f"  -4.5 注入逻辑上下文:")
        for line in context.split("\n"):
            print(f"    {line}")
else:
    print(f"  文件不存在: {BD_FILE}")

# Step 7: 检查 population_fitness.py 的零交易硬惩罚
print(f"\n[7] population_fitness.py 零交易硬惩罚 (-5.0) 检查")
PF_FILE = Path("/home/lmy/hermes_v6/sandbox_trading/population_fitness.py")
if PF_FILE.exists():
    with open(PF_FILE) as f:
        pf_content = f.read()
    # 找 -5.0 的位置
    idx = pf_content.find("-5.0")
    while idx >= 0 and idx < len(pf_content):
        start = max(0, idx - 200)
        end = min(len(pf_content), idx + 100)
        context = pf_content[start:end].replace("\n", " | ")
        print(f"  位置 {idx}: ...{context}...")
        idx = pf_content.find("-5.0", idx + 1)
        if idx > 50000:  # 限制
            break

print("\n" + "=" * 80)
print("精准诊断完成")
print("=" * 80)
