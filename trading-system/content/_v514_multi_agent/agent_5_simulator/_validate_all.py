#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent-5 L1-L6 全量验证脚本
执行八层验证体系中的 L1/L2/L3/L4/L6
"""
import os
import sys
import json
import py_compile
import re
from pathlib import Path

HERE = Path(__file__).parent.resolve()
DELIVERABLES = [
    "_v515_high_fidelity_simulator.py",
    "_v515_sim_vs_backtest_compare.py",
    "status.json",
]
GENERATED = [
    "_v515_hifi_sim_result.json",
    "_v515_sim_vs_backtest_compare.json",
    "_v515_sim_vs_backtest_compare.md",
]

V515_PATH = HERE.parent.parent / "_v515_ensemble_strategy.py"
WS_RESULT = HERE.parent / "agent_1_pipeline" / "_v514_ws_test_result.json"

PLACEHOLDER_PATTERNS = [
    r"\bTODO\b", r"\bFIXME\b", r"\bTBD\b", r"\bXXX\b",
    r"占位符", r"待定", r"待补充", r"placeholder",
]

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ===== L1: 存在性验证 =====
def l1_existence():
    section("L1: 存在性验证")
    all_ok = True
    for f in DELIVERABLES + GENERATED:
        p = HERE / f
        if p.exists():
            size = p.stat().st_size
            if size < 100:
                print(f"  FAIL: {f} 存在但过小 ({size} bytes)")
                all_ok = False
            else:
                print(f"  PASS: {f} ({size} bytes)")
        else:
            print(f"  FAIL: {f} 不存在")
            all_ok = False

    # 检查依赖文件
    print(f"\n  依赖文件:")
    for name, p in [("v515策略", V515_PATH), ("WS延迟数据", WS_RESULT)]:
        if p.exists():
            print(f"  PASS: {name} -> {p.name} ({p.stat().st_size} bytes)")
        else:
            print(f"  FAIL: {name} 不存在: {p}")
            all_ok = False
    return all_ok

# ===== L2: 格式验证 =====
def l2_format():
    section("L2: 格式验证")
    all_ok = True
    for f in DELIVERABLES:
        p = HERE / f
        if not p.exists():
            print(f"  FAIL: {f} 不存在")
            all_ok = False
            continue
        if f.endswith(".py"):
            try:
                py_compile.compile(str(p), doraise=True)
                print(f"  PASS: {f} Python 语法有效")
            except py_compile.PyCompileError as e:
                print(f"  FAIL: {f} 语法错误: {e}")
                all_ok = False
        elif f.endswith(".json"):
            try:
                json.loads(p.read_text(encoding="utf-8"))
                print(f"  PASS: {f} JSON 有效")
            except Exception as e:
                print(f"  FAIL: {f} JSON 无效: {e}")
                all_ok = False
    return all_ok

# ===== L3: 交叉引用验证 =====
def l3_cross_reference():
    section("L3: 交叉引用验证")
    all_ok = True

    # 检查 simulator 是否正确引用 v515
    sim_path = HERE / "_v515_high_fidelity_simulator.py"
    sim_content = sim_path.read_text(encoding="utf-8")
    if "_v515_ensemble_strategy.py" in sim_content and V515_PATH.exists():
        print(f"  PASS: simulator 引用 v515 策略路径正确")
    else:
        print(f"  FAIL: simulator v515 引用问题")
        all_ok = False

    # 检查 WS 延迟数据引用
    if "_v514_ws_test_result.json" in sim_content and WS_RESULT.exists():
        print(f"  PASS: simulator 引用 Agent-1 WS 数据正确")
    else:
        print(f"  FAIL: WS 数据引用问题")
        all_ok = False

    # 检查 compare 引用 simulator 结果
    cmp_path = HERE / "_v515_sim_vs_backtest_compare.py"
    cmp_content = cmp_path.read_text(encoding="utf-8")
    if "_v515_hifi_sim_result.json" in cmp_content and (HERE / "_v515_hifi_sim_result.json").exists():
        print(f"  PASS: compare 引用 simulator 结果正确")
    else:
        print(f"  FAIL: compare 引用 simulator 结果问题")
        all_ok = False

    # 检查 status.json 中的 deliverables 路径
    status = json.loads((HERE / "status.json").read_text(encoding="utf-8"))
    for d in status.get("deliverables", []):
        # deliverables 是相对路径 (相对于 _v514_multi_agent)
        full = HERE.parent.parent / d
        if full.exists():
            print(f"  PASS: status.json deliverable 存在: {d}")
        else:
            print(f"  FAIL: status.json deliverable 不存在: {d}")
            all_ok = False

    return all_ok

# ===== L4: 内容完整性 =====
def l4_content():
    section("L4: 内容完整性 (无占位符)")
    all_ok = True
    for f in DELIVERABLES:
        p = HERE / f
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8")
        found = []
        for pattern in PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                found.append((pattern, len(matches)))
        if found:
            print(f"  WARN: {f} 发现占位符:")
            for pat, cnt in found:
                print(f"    {pat}: {cnt} 处")
            all_ok = False
        else:
            print(f"  PASS: {f} 无占位符")
    return all_ok

# ===== L6: 功能可用性 =====
def l6_functional():
    section("L6: 功能可用性")
    all_ok = True

    # 检查仿真结果
    sim_result = HERE / "_v515_hifi_sim_result.json"
    if sim_result.exists():
        try:
            data = json.loads(sim_result.read_text(encoding="utf-8"))
            symbols = data.get("symbols", {})
            ok_count = sum(1 for v in symbols.values() if v.get("status") == "ok")
            print(f"  PASS: 仿真结果有效 ({ok_count}/{len(symbols)} 符号成功)")
            if ok_count > 0:
                # 检查是否有 ideal 和 hifi metrics
                first_ok = next((v for v in symbols.values() if v.get("status") == "ok"), None)
                if first_ok and "ideal_metrics" in first_ok and "hifi_metrics" in first_ok:
                    print(f"  PASS: 包含 ideal_metrics 和 hifi_metrics")
                else:
                    print(f"  FAIL: 缺少 ideal/hifi metrics")
                    all_ok = False
                # 检查 factor_isolation
                if "factor_isolation" in first_ok.get("hifi_metrics", {}):
                    print(f"  PASS: 包含 factor_isolation (因素隔离实验)")
                else:
                    print(f"  FAIL: 缺少 factor_isolation")
                    all_ok = False
            else:
                print(f"  FAIL: 无成功符号")
                all_ok = False
        except Exception as e:
            print(f"  FAIL: 仿真结果解析失败: {e}")
            all_ok = False
    else:
        print(f"  FAIL: 仿真结果不存在")
        all_ok = False

    # 检查对比报告
    cmp_json = HERE / "_v515_sim_vs_backtest_compare.json"
    if cmp_json.exists():
        try:
            data = json.loads(cmp_json.read_text(encoding="utf-8"))
            summary = data.get("summary", {})
            avg_diff = summary.get("avg_diff_rate", 0)
            print(f"  PASS: 对比报告有效 (平均差异率 {avg_diff*100:.2f}%)")
            if "per_symbol" in data and "suggestions" in data:
                print(f"  PASS: 包含 per_symbol 和 suggestions")
            else:
                print(f"  FAIL: 缺少 per_symbol 或 suggestions")
                all_ok = False
        except Exception as e:
            print(f"  FAIL: 对比报告解析失败: {e}")
            all_ok = False
    else:
        print(f"  FAIL: 对比报告不存在")
        all_ok = False

    cmp_md = HERE / "_v515_sim_vs_backtest_compare.md"
    if cmp_md.exists():
        md = cmp_md.read_text(encoding="utf-8")
        if len(md) > 500 and "# v515" in md:
            print(f"  PASS: Markdown 报告有效 ({len(md)} chars)")
        else:
            print(f"  FAIL: Markdown 报告内容不足")
            all_ok = False
    else:
        print(f"  FAIL: Markdown 报告不存在")
        all_ok = False

    return all_ok

# ===== 主函数 =====
def main():
    print("=" * 60)
    print("Agent-5 全量验证 (L1-L6)")
    print("=" * 60)

    results = {
        "L1": l1_existence(),
        "L2": l2_format(),
        "L3": l3_cross_reference(),
        "L4": l4_content(),
        "L6": l6_functional(),
    }

    section("验证汇总")
    all_pass = True
    for layer, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {layer}: {status}")
        if not ok:
            all_pass = False

    print(f"\n  综合评定: {'PASS (可交付)' if all_pass else 'FAIL (需修复)'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
