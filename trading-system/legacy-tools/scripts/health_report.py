#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.quality_gate import QualityGateEngine


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    engine = QualityGateEngine(project_root=root)

    print("=" * 60)
    print("Hermes 代码健康仪表盘")
    print("=" * 60)

    health = engine.health_report()
    print(f"总文件数: {health['total_files']}")
    print(f"总行数: {health['total_lines']}")
    print(f"\nTop 20 最大文件:")
    for entry in health["top_20_largest"]:
        print(f"  {entry['lines']:>6d} 行  {entry['file']}")

    print(f"\n--- 质量门禁快速扫描 ---")
    dead = engine.scan_dead_code()
    cplx = engine.scan_complexity()
    dup = engine.scan_duplication()
    brk = engine.scan_broken_imports()

    print(f"  函数总数: {dead['total_functions']}")
    print(f"  空壳函数: {dead['stub_count']}")
    print(f"  未调用函数: {dead['uncalled_count']}")
    print(f"  复杂度违规: {cplx['violation_count']}")
    print(f"  重复代码块: {dup['duplicate_count']}")
    print(f"  依赖断裂: {brk['issue_count']}")

    contract = engine.evaluate_quality_contract()
    print(f"\n  质量分数: {contract.score}/100")
    print(f"  通过: {'YES' if contract.passed else 'NO'}")

    gate = engine.release_gate()
    print(f"  发布决策: {gate['decision']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
