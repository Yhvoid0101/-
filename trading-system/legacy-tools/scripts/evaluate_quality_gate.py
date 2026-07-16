#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.quality_gate import QualityGateEngine


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    rules_path = os.path.join(root, "quality-gate.yml")

    rules = None
    if os.path.exists(rules_path):
        try:
            import yaml
            with open(rules_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            rules = cfg.get("rules", {})
        except Exception:
            pass

    engine = QualityGateEngine(project_root=root)
    report = engine.evaluate_quality_contract(rules=rules)

    print("=" * 60)
    print("质量合同评估报告")
    print("=" * 60)
    print(f"质量分数: {report.score}/100")
    print(f"通过: {'YES' if report.passed else 'NO'}")
    print(f"文件数: {report.file_count}")
    print(f"\n维度详情:")
    for k, v in report.dimensions.items():
        print(f"  {k}: {v}")

    if report.issues:
        print(f"\nTop 问题 ({len(report.issues)}):")
        for issue in report.issues[:15]:
            print(f"  [{issue['level']}:{issue['type']}] {issue['detail']}")

    gate = engine.release_gate()
    print(f"\n发布决策: {gate['decision']} (分数 {gate['score']} >= {gate['min_score']})")
    print("=" * 60)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
