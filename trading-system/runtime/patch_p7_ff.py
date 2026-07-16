#!/usr/bin/env python3
"""Phase 7.4 FeatureFusionAnalysisMixin 提取补丁"""
import ast

FILE = "evolution_loop.py"

with open(FILE, "r", encoding="utf-8") as f:
    source = f.read()
    lines = source.splitlines(keepends=True)

# 1. 添加导入
import_line = "from .feature_fusion_analysis import FeatureFusionAnalysisMixin\n"
mc_idx = None
for i, line in enumerate(lines):
    if "from .metrics_computation import" in line:
        mc_idx = i
        break
assert mc_idx is not None, "未找到 metrics_computation 导入行"
ff_exists = any("from .feature_fusion_analysis import" in line for line in lines)
if not ff_exists:
    lines.insert(mc_idx + 1, import_line)
    print(f"[1] 插入导入行 at L{mc_idx + 2}")
else:
    print("[1] 导入行已存在, 跳过")

# 2. 修改 class 定义
class_found = False
for i, line in enumerate(lines):
    if "class EvolutionLoop(HallOfFameMixin, DeployBlockAdmissionMixin, MetricsComputationMixin):" in line:
        lines[i] = line.replace(
            "class EvolutionLoop(HallOfFameMixin, DeployBlockAdmissionMixin, MetricsComputationMixin):",
            "class EvolutionLoop(HallOfFameMixin, DeployBlockAdmissionMixin, MetricsComputationMixin, FeatureFusionAnalysisMixin):"
        )
        print(f"[2] 修改 class 定义 at L{i + 1}")
        class_found = True
        break
assert class_found, "未找到 class EvolutionLoop 定义行"

# 3. 用 AST 找到2个方法的精确行范围
new_source = "".join(lines)
tree = ast.parse(new_source)

evo_class = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "EvolutionLoop":
        evo_class = node
        break
assert evo_class is not None, "未找到 EvolutionLoop 类"

methods_to_remove = {
    "_run_feature_fusion_analysis",
    "_synthesize_4h_klines",
}

remove_ranges = []
for item in evo_class.body:
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if item.name in methods_to_remove:
            remove_ranges.append((item.lineno, item.end_lineno, item.name))

assert len(remove_ranges) == 2, f"期望找到2个方法, 实际找到 {len(remove_ranges)}: {remove_ranges}"

# 按起始行降序排序
remove_ranges.sort(key=lambda x: x[0], reverse=True)

for start, end, name in remove_ranges:
    del lines[start - 1: end]
    if start - 1 < len(lines) and lines[start - 1].strip() == "":
        del lines[start - 1]
    print(f"[3] 删除 {name} at L{start}-L{end}")

with open(FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"[DONE] 总行数: {len(lines)}")
