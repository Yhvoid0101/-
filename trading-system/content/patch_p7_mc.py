#!/usr/bin/env python3
"""Phase 7.3 MetricsComputationMixin 提取补丁

改动:
  1. 在 deploy_block_admission 导入后添加 metrics_computation 导入
  2. class EvolutionLoop 添加 MetricsComputationMixin 继承
  3. 用 AST 精确定位并删除3个方法
"""
import ast

FILE = "evolution_loop.py"

with open(FILE, "r", encoding="utf-8") as f:
    source = f.read()
    lines = source.splitlines(keepends=True)

# 1. 添加导入
import_line = "from .metrics_computation import MetricsComputationMixin\n"
dba_idx = None
for i, line in enumerate(lines):
    if "from .deploy_block_admission import" in line:
        dba_idx = i
        break
assert dba_idx is not None, "未找到 deploy_block_admission 导入行"
mc_exists = any("from .metrics_computation import" in line for line in lines)
if not mc_exists:
    lines.insert(dba_idx + 1, import_line)
    print(f"[1] 插入导入行 at L{dba_idx + 2}")
else:
    print("[1] 导入行已存在, 跳过")

# 2. 修改 class 定义
class_found = False
for i, line in enumerate(lines):
    if "class EvolutionLoop(HallOfFameMixin, DeployBlockAdmissionMixin):" in line:
        lines[i] = line.replace(
            "class EvolutionLoop(HallOfFameMixin, DeployBlockAdmissionMixin):",
            "class EvolutionLoop(HallOfFameMixin, DeployBlockAdmissionMixin, MetricsComputationMixin):"
        )
        print(f"[2] 修改 class 定义 at L{i + 1}")
        class_found = True
        break
assert class_found, "未找到 class EvolutionLoop 定义行"

# 3. 用 AST 找到3个方法的精确行范围
new_source = "".join(lines)
tree = ast.parse(new_source)

evo_class = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "EvolutionLoop":
        evo_class = node
        break
assert evo_class is not None, "未找到 EvolutionLoop 类"

methods_to_remove = {
    "_serialize_capacity_stress_report",
    "_compute_metrics_from_pnls",
    "_compute_intraday_metrics",
}

remove_ranges = []
for item in evo_class.body:
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if item.name in methods_to_remove:
            remove_ranges.append((item.lineno, item.end_lineno, item.name))

assert len(remove_ranges) == 3, f"期望找到3个方法, 实际找到 {len(remove_ranges)}: {remove_ranges}"

# 按起始行降序排序 (从后往前删除, 避免行号偏移)
remove_ranges.sort(key=lambda x: x[0], reverse=True)

for start, end, name in remove_ranges:
    # 删除方法本身的行 (1-based [start, end] → 0-based [start-1, end))
    del lines[start - 1: end]
    # 删除方法后的一个空行 (如果有, 保持代码间距)
    if start - 1 < len(lines) and lines[start - 1].strip() == "":
        del lines[start - 1]
    print(f"[3] 删除 {name} at L{start}-L{end}")

# 写回
with open(FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"[DONE] 总行数: {len(lines)}")
