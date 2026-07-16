#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.quality_gate import QualityGateEngine


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    dry_run = "--apply" not in sys.argv

    engine = QualityGateEngine(project_root=root)

    print("=" * 60)
    print(f"自动修复 Agent ({'DRY RUN' if dry_run else 'APPLY MODE'})")
    print("=" * 60)

    result = engine.auto_fix_unused_imports(dry_run=dry_run)
    print(f"可修复文件数: {result['fix_count']}")
    for fix in result["fixes"][:20]:
        print(f"  {fix['file']}: 移除 {fix['unused_names']} (行 {fix['removed_lines']})")

    if not dry_run and result["fix_count"] > 0:
        print(f"\n已应用 {result['fix_count']} 处修复")
    elif dry_run and result["fix_count"] > 0:
        print(f"\n使用 --apply 参数实际执行修复")

    print("=" * 60)


if __name__ == "__main__":
    main()
