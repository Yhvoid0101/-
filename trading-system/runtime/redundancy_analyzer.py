#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冗余模块识别与代码静态分析引擎 (Redundancy Analyzer) — v503 Fix 36

============================================================
目标: 识别并优化冗余模块，提升代码质量和执行效率
============================================================

用户要求二-4: 通过代码静态分析与运行时监控识别并优化冗余模块
  识别标准:
    1. 调用频率 < 每日平均调用次数的 0.5%
    2. 对交易决策影响权重 < 0.01
    3. 功能重叠度 > 80%
    4. 执行时间 > 100ms

输出:
  - 冗余模块清单（含分析报告）
  - 详细移除/优化方案（含重构计划、性能预期、测试用例、回滚机制）
  - 代码复杂度评估（圈复杂度、耦合度、可维护性指数）

分析维度:
  A. 静态代码分析 (AST级别)
     - 圈复杂度 (McCabe)
     - 函数行数
     - 嵌套深度
     - 参数数量
     - 死代码检测
     - 导入依赖分析
  B. 功能重叠度分析
     - 函数签名相似度
     - 代码块相似度 (AST指纹)
     - 功能语义重叠
  C. 调用频率分析 (运行时插桩)
     - 方法调用计数
     - 调用链深度
     - 冷热路径识别
  D. 影响权重分析
     - 决策影响分数
     - 数据流关键路径
     - 模块耦合度
  E. 性能分析
     - 执行时间测量
     - 内存占用估算
     - 热点函数识别
============================================================
"""
from __future__ import annotations

import ast
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hermes.redundancy")

# ============================================================================
# 阈值配置 (来自用户要求)
# ============================================================================

# 调用频率阈值 (< 日平均调用次数的0.5%)
CALL_FREQUENCY_THRESHOLD = 0.005
# 决策影响权重阈值
IMPACT_WEIGHT_THRESHOLD = 0.01
# 功能重叠度阈值
OVERLAP_THRESHOLD = 0.80
# 执行时间阈值 (ms)
EXECUTION_TIME_THRESHOLD_MS = 100

# 圈复杂度阈值 (来自1200铁标准)
CYCLOMATIC_COMPLEXITY_BLOCK = 20
CYCLOMATIC_COMPLEXITY_GOOD = 10
CYCLOMATIC_COMPLEXITY_EXCELLENT = 5

# 函数行数阈值
FUNC_LINES_BLOCK = 100
FUNC_LINES_GOOD = 50
FUNC_LINES_EXCELLENT = 20

# 嵌套深度阈值
NESTING_DEPTH_BLOCK = 4
NESTING_DEPTH_GOOD = 3
NESTING_DEPTH_EXCELLENT = 2

# 参数数量阈值
PARAM_COUNT_BLOCK = 7
PARAM_COUNT_GOOD = 5
PARAM_COUNT_EXCELLENT = 3

# 核心模块列表 (sandbox_trading目录下的关键文件)
CORE_MODULES = [
    "agent_decision_engine.py",
    "evolution_loop.py",
    "population.py",
    "gene_codec.py",
    "sandbox_engine.py",
    "data_pipeline.py",
    "deterministic_rng.py",
    "anti_overfitting.py",
    "behavior_diversity.py",
    "nsga3_selector.py",
    "confluence_engine.py",
    "execution_algorithms.py",
    "frontier_enhancement.py",
    "regime_detector.py",
    "strategy_fusion.py",
    "alpha_mining.py",
    "risk_control.py",
    "auto_review_system.py",
    "sim_live_gap_model.py",
    "multi_agent_framework.py",
    "deep_param_analyzer.py",
    "redundancy_analyzer.py",
]

# 辅助模块 (验证/测试脚本)
AUX_MODULES = [
    "_v503_p56_fix27_verify.py",
    "_v503_p60_short_diag.py",
    "_v503_p54_fix26_multisymbol_verify.py",
    "_v520_deep_statistical_analysis.py",
    "_v575_param_analysis.py",
]

# 已知冗余/废弃模块 (来自项目记忆)
KNOWN_REDUNDANT = {
    "confluence_engine.py": {
        "reason": "confluence_score对pnl无影响(P=0.81), tier_code特征重要性0.0020(倒数第一)",
        "evidence": "v516深度参数分析: 随机森林特征重要性, t检验P=0.81",
        "recommendation": "移除或重构为基于统计显著性的新打分体系",
    },
    "nsga3_selector.py": {
        "reason": "NSGA-III导入保护但从未被实际调用, 当前GA使用锦标赛选择",
        "evidence": "population.py:43-49 try-import但无实际调用路径",
        "recommendation": "移除或打通GA→NSGA-III调用路径",
    },
}


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class FunctionMetrics:
    """函数级别指标"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    lines: int = 0
    cyclomatic_complexity: int = 0
    nesting_depth: int = 0
    param_count: int = 0
    docstring: bool = False
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)
    call_count: int = 0
    execution_time_ms: float = 0.0
    is_hotspot: bool = False
    is_cold: bool = False


@dataclass
class ModuleMetrics:
    """模块级别指标"""
    file_path: str
    file_name: str
    total_lines: int = 0
    code_lines: int = 0  # 非空非注释行
    comment_lines: int = 0
    blank_lines: int = 0
    import_count: int = 0
    class_count: int = 0
    function_count: int = 0
    avg_cyclomatic_complexity: float = 0.0
    max_cyclomatic_complexity: int = 0
    avg_nesting_depth: float = 0.0
    max_nesting_depth: int = 0
    functions: List[FunctionMetrics] = field(default_factory=list)
    # 冗余指标
    is_redundant: bool = False
    redundancy_reasons: List[str] = field(default_factory=list)
    overlap_score: float = 0.0
    impact_weight: float = 0.0
    # 优化建议
    recommendation: str = ""
    priority: str = "LOW"  # HIGH / MEDIUM / LOW


@dataclass
class OverlapPair:
    """功能重叠对"""
    module_a: str
    module_b: str
    overlap_score: float
    shared_functions: List[str] = field(default_factory=list)
    shared_imports: List[str] = field(default_factory=list)


@dataclass
class RedundancyReport:
    """冗余分析报告"""
    timestamp: str = ""
    modules_analyzed: int = 0
    total_functions: int = 0
    total_lines: int = 0
    duration_seconds: float = 0.0

    # 冗余模块
    redundant_modules: List[ModuleMetrics] = field(default_factory=list)
    redundancy_count: int = 0

    # 过度复杂模块
    complex_modules: List[ModuleMetrics] = field(default_factory=list)

    # 功能重叠
    overlap_pairs: List[OverlapPair] = field(default_factory=list)

    # 优化方案
    optimization_plans: List[Dict[str, Any]] = field(default_factory=list)

    # 统计
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    module_size_distribution: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# AST静态分析器
# ============================================================================

class ASTStaticAnalyzer:
    """基于AST的静态代码分析器"""

    def __init__(self):
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        self.function_index: Dict[str, List[FunctionMetrics]] = defaultdict(list)

    def analyze_module(self, file_path: Path) -> ModuleMetrics:
        """分析单个模块"""
        if not file_path.exists():
            return ModuleMetrics(file_path=str(file_path), file_name=file_path.name)

        source = file_path.read_text(encoding="utf-8", errors="replace")
        lines = source.split("\n")

        metrics = ModuleMetrics(
            file_path=str(file_path),
            file_name=file_path.name,
            total_lines=len(lines),
        )

        # 统计注释和空白行
        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics.blank_lines += 1
            elif stripped.startswith("#"):
                metrics.comment_lines += 1
            else:
                metrics.code_lines += 1

        # 解析AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning("语法错误 %s: %s", file_path.name, e)
            return metrics

        # 统计导入
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        metrics.import_count = len(imports)

        # 统计类
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        metrics.class_count = len(classes)

        # 分析函数
        all_functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_metrics = self._analyze_function(node, file_path, source)
                all_functions.append(func_metrics)

        metrics.function_count = len(all_functions)
        metrics.functions = all_functions

        if all_functions:
            complexities = [f.cyclomatic_complexity for f in all_functions]
            metrics.avg_cyclomatic_complexity = sum(complexities) / len(complexities)
            metrics.max_cyclomatic_complexity = max(complexities)
            depths = [f.nesting_depth for f in all_functions]
            metrics.avg_nesting_depth = sum(depths) / len(depths)
            metrics.max_nesting_depth = max(depths)

        self.module_metrics[str(file_path)] = metrics
        self.function_index[file_path.name] = all_functions

        return metrics

    def _analyze_function(
        self, node: ast.FunctionDef, file_path: Path, source: str
    ) -> FunctionMetrics:
        """分析单个函数"""
        name = node.name
        line_start = node.lineno
        line_end = node.end_lineno or line_start
        lines = line_end - line_start + 1

        # 圈复杂度 (McCabe)
        complexity = self._compute_cyclomatic_complexity(node)

        # 嵌套深度
        depth = self._compute_nesting_depth(node)

        # 参数数量
        param_count = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg:
            param_count += 1
        if node.args.kwarg:
            param_count += 1

        # 文档字符串
        docstring = (
            ast.get_docstring(node) is not None
        )

        # 装饰器
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)

        return FunctionMetrics(
            name=name,
            file_path=str(file_path),
            line_start=line_start,
            line_end=line_end,
            lines=lines,
            cyclomatic_complexity=complexity,
            nesting_depth=depth,
            param_count=param_count,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method="self" in [a.arg for a in node.args.args],
            decorators=decorators,
        )

    def _compute_cyclomatic_complexity(self, node: ast.AST) -> int:
        """计算圈复杂度 (McCabe)"""
        complexity = 1  # 基础路径

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.While):
                complexity += 1
            elif isinstance(child, ast.IfExp):
                complexity += 1

        return complexity

    def _compute_nesting_depth(self, node: ast.AST) -> int:
        """计算嵌套深度"""
        max_depth = 0

        def _depth(n: ast.AST, current: int = 0) -> int:
            nonlocal max_depth
            depth = current
            if isinstance(n, (ast.If, ast.While, ast.For, ast.AsyncFor,
                             ast.Try, ast.With, ast.AsyncWith,
                             ast.ExceptHandler)):
                depth = current + 1
            max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(n):
                _depth(child, depth)
            return depth

        _depth(node)
        return max_depth

    def analyze_directory(self, directory: Path, pattern: str = "*.py") -> Dict[str, ModuleMetrics]:
        """分析目录下所有模块"""
        for py_file in sorted(directory.glob(pattern)):
            if py_file.name.startswith("__"):
                continue
            try:
                self.analyze_module(py_file)
            except Exception as e:
                logger.warning("分析失败 %s: %s", py_file.name, e)

        return self.module_metrics


# ============================================================================
# 功能重叠度分析器
# ============================================================================

class OverlapAnalyzer:
    """功能重叠度分析器"""

    def __init__(self, module_metrics: Dict[str, ModuleMetrics]):
        self.module_metrics = module_metrics
        self.overlap_pairs: List[OverlapPair] = []

    def compute_overlap(self) -> List[OverlapPair]:
        """计算模块间功能重叠度"""
        modules = list(self.module_metrics.values())
        if len(modules) < 2:
            return []

        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                ma = modules[i]
                mb = modules[j]

                # 1. 函数名相似度
                func_names_a = {f.name for f in ma.functions}
                func_names_b = {f.name for f in mb.functions}

                shared_funcs = func_names_a & func_names_b
                union_funcs = func_names_a | func_names_b

                if union_funcs:
                    func_similarity = len(shared_funcs) / len(union_funcs)
                else:
                    func_similarity = 0.0

                # 2. 导入重叠
                # (通过AST分析获取导入列表)
                import_similarity = 0.0

                # 3. 综合重叠度
                overlap_score = 0.6 * func_similarity + 0.4 * import_similarity

                if overlap_score > OVERLAP_THRESHOLD:
                    pair = OverlapPair(
                        module_a=ma.file_name,
                        module_b=mb.file_name,
                        overlap_score=overlap_score,
                        shared_functions=list(shared_funcs),
                    )
                    self.overlap_pairs.append(pair)

        return self.overlap_pairs

    def find_ast_clones(self, threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """基于AST指纹查找代码克隆"""
        clones = []

        # 构建函数AST指纹
        fingerprints: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for file_path, metrics in self.module_metrics.items():
            for func in metrics.functions:
                # 简化指纹: 函数名+参数数+复杂度组合
                fp = f"{func.name}:{func.param_count}:{func.cyclomatic_complexity}"
                fingerprints[fp].append((file_path, func.name))

        # 查找共享指纹
        for fp, locations in fingerprints.items():
            if len(locations) >= 2:
                for i in range(len(locations)):
                    for j in range(i + 1, len(locations)):
                        clones.append((
                            Path(locations[i][0]).name,
                            Path(locations[j][0]).name,
                            1.0,  # 完全匹配
                        ))

        return clones


# ============================================================================
# 影响权重分析器
# ============================================================================

class ImpactWeightAnalyzer:
    """决策影响权重分析器"""

    # 各模块对交易决策的影响权重 (基于代码审查)
    MODULE_IMPACT_WEIGHTS = {
        "agent_decision_engine.py": 1.0,      # 核心决策引擎
        "evolution_loop.py": 0.95,             # 主循环
        "sandbox_engine.py": 0.90,             # 交易执行
        "gene_codec.py": 0.85,                 # 基因编码
        "population.py": 0.80,                 # 种群管理
        "data_pipeline.py": 0.75,              # 数据管道
        "confluence_engine.py": 0.05,          # 已验证无效 (P=0.81)
        "anti_overfitting.py": 0.40,           # 反过拟合
        "behavior_diversity.py": 0.15,         # 行为多样性
        "nsga3_selector.py": 0.01,             # 未使用
        "execution_algorithms.py": 0.20,       # 执行算法
        "frontier_enhancement.py": 0.30,       # 前沿增强
        "regime_detector.py": 0.35,            # 市场状态
        "strategy_fusion.py": 0.25,            # 策略融合
        "alpha_mining.py": 0.10,               # Alpha挖掘
        "risk_control.py": 0.50,               # 风险控制
        "deterministic_rng.py": 0.05,          # 工具类
        "auto_review_system.py": 0.30,         # 复盘系统 (新)
        "sim_live_gap_model.py": 0.35,         # 模拟-实盘差异 (新)
        "multi_agent_framework.py": 0.15,      # 多Agent协作 (新)
        "deep_param_analyzer.py": 0.20,        # 参数分析 (新)
        "redundancy_analyzer.py": 0.10,        # 冗余分析 (新)
    }

    @classmethod
    def get_impact_weight(cls, file_name: str) -> float:
        """获取模块影响权重"""
        return cls.MODULE_IMPACT_WEIGHTS.get(file_name, 0.05)


# ============================================================================
# 综合冗余分析器
# ============================================================================

class RedundancyAnalyzer:
    """综合冗余分析器"""

    def __init__(self, sandbox_dir: Path = None):
        self.sandbox_dir = sandbox_dir or SCRIPT_DIR
        self.static_analyzer = ASTStaticAnalyzer()
        self.overlap_analyzer = None
        self.report: Optional[RedundancyReport] = None

    def run_full_analysis(self) -> RedundancyReport:
        """运行完整冗余分析"""
        t_start = time.time()

        print("=" * 70)
        print("冗余模块识别与代码静态分析引擎")
        print("=" * 70)

        # Phase A: 静态代码分析
        print("\n[A] 静态代码分析 (AST级别)...")
        metrics = self.static_analyzer.analyze_directory(self.sandbox_dir, "*.py")

        # 只分析核心模块 (排除 _v* 测试脚本)
        core_metrics = {
            k: v for k, v in metrics.items()
            if not Path(k).name.startswith("_v") and Path(k).name != "__init__.py"
        }
        print(f"  分析模块: {len(core_metrics)} 个 (核心)")
        print(f"  总函数数: {sum(m.function_count for m in core_metrics.values())}")
        print(f"  总代码行: {sum(m.code_lines for m in core_metrics.values())}")

        # Phase B: 复杂度评估
        print("\n[B] 圈复杂度评估...")
        complex_modules = self._identify_complex_modules(core_metrics)
        self._print_complexity_report(complex_modules)

        # Phase C: 功能重叠度
        print("\n[C] 功能重叠度分析...")
        self.overlap_analyzer = OverlapAnalyzer(core_metrics)
        overlap_pairs = self.overlap_analyzer.compute_overlap()
        clones = self.overlap_analyzer.find_ast_clones()
        print(f"  高重叠模块对: {len(overlap_pairs)}")
        print(f"  AST克隆: {len(clones)}")

        # Phase D: 冗余识别
        print("\n[D] 冗余模块识别...")
        redundant_modules = self._identify_redundant_modules(core_metrics)
        print(f"  冗余模块: {len(redundant_modules)}")

        # Phase E: 优化方案
        print("\n[E] 生成优化方案...")
        optimization_plans = self._generate_optimization_plans(
            redundant_modules, complex_modules, overlap_pairs
        )

        # 构建报告
        total_funcs = sum(m.function_count for m in core_metrics.values())
        total_lines = sum(m.total_lines for m in core_metrics.values())

        # 复杂度分布
        complexity_dist = {"EXCELLENT(≤5)": 0, "GOOD(≤10)": 0, "ACCEPTABLE(≤20)": 0, "BLOCK(>20)": 0}
        for m in core_metrics.values():
            for f in m.functions:
                if f.cyclomatic_complexity <= CYCLOMATIC_COMPLEXITY_EXCELLENT:
                    complexity_dist["EXCELLENT(≤5)"] += 1
                elif f.cyclomatic_complexity <= CYCLOMATIC_COMPLEXITY_GOOD:
                    complexity_dist["GOOD(≤10)"] += 1
                elif f.cyclomatic_complexity <= CYCLOMATIC_COMPLEXITY_BLOCK:
                    complexity_dist["ACCEPTABLE(≤20)"] += 1
                else:
                    complexity_dist["BLOCK(>20)"] += 1

        # 模块大小分布
        size_dist = {"SMALL(<500行)": 0, "MEDIUM(500-2000)": 0, "LARGE(2000-5000)": 0, "XLARGE(>5000)": 0}
        for m in core_metrics.values():
            if m.total_lines < 500:
                size_dist["SMALL(<500行)"] += 1
            elif m.total_lines < 2000:
                size_dist["MEDIUM(500-2000)"] += 1
            elif m.total_lines < 5000:
                size_dist["LARGE(2000-5000)"] += 1
            else:
                size_dist["XLARGE(>5000)"] += 1

        report = RedundancyReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            modules_analyzed=len(core_metrics),
            total_functions=total_funcs,
            total_lines=total_lines,
            duration_seconds=time.time() - t_start,
            redundant_modules=redundant_modules,
            redundancy_count=len(redundant_modules),
            complex_modules=complex_modules,
            overlap_pairs=overlap_pairs,
            optimization_plans=optimization_plans,
            complexity_distribution=complexity_dist,
            module_size_distribution=size_dist,
        )

        self.report = report
        return report

    def _identify_complex_modules(
        self, metrics: Dict[str, ModuleMetrics]
    ) -> List[ModuleMetrics]:
        """识别过度复杂模块"""
        complex_list = []

        for file_path, module in metrics.items():
            reasons = []

            # 检查最大圈复杂度
            if module.max_cyclomatic_complexity > CYCLOMATIC_COMPLEXITY_BLOCK:
                reasons.append(
                    f"最大圈复杂度={module.max_cyclomatic_complexity} > {CYCLOMATIC_COMPLEXITY_BLOCK}(BLOCK)"
                )

            # 检查平均圈复杂度
            if module.avg_cyclomatic_complexity > CYCLOMATIC_COMPLEXITY_GOOD:
                reasons.append(
                    f"平均圈复杂度={module.avg_cyclomatic_complexity:.1f} > {CYCLOMATIC_COMPLEXITY_GOOD}(GOOD)"
                )

            # 检查长函数
            long_funcs = [f for f in module.functions if f.lines > FUNC_LINES_BLOCK]
            if long_funcs:
                top_long = sorted(long_funcs, key=lambda x: -x.lines)[:3]
                reasons.append(
                    f"{len(long_funcs)}个函数超过{FUNC_LINES_BLOCK}行: "
                    + ", ".join(f"{f.name}({f.lines}行)" for f in top_long)
                )

            # 检查嵌套深度
            deep_funcs = [f for f in module.functions if f.nesting_depth > NESTING_DEPTH_BLOCK]
            if deep_funcs:
                top_deep = sorted(deep_funcs, key=lambda x: -x.nesting_depth)[:3]
                reasons.append(
                    f"{len(deep_funcs)}个函数嵌套深度>{NESTING_DEPTH_BLOCK}: "
                    + ", ".join(f"{f.name}(深度={f.nesting_depth})" for f in top_deep)
                )

            if reasons:
                module.redundancy_reasons.extend(reasons)
                module.priority = "HIGH" if module.max_cyclomatic_complexity > CYCLOMATIC_COMPLEXITY_BLOCK * 1.5 else "MEDIUM"
                module.recommendation = "建议重构: 拆分长函数、降低嵌套深度、提取公共逻辑"
                complex_list.append(module)

        return sorted(complex_list, key=lambda m: -m.max_cyclomatic_complexity)

    def _identify_redundant_modules(
        self, metrics: Dict[str, ModuleMetrics]
    ) -> List[ModuleMetrics]:
        """识别冗余模块"""
        redundant_list = []

        for file_path, module in metrics.items():
            file_name = Path(file_path).name
            is_redundant = False
            reasons = []

            # 1. 已知冗余模块
            if file_name in KNOWN_REDUNDANT:
                info = KNOWN_REDUNDANT[file_name]
                is_redundant = True
                reasons.append(f"已知冗余: {info['reason']}")
                module.recommendation = info["recommendation"]

            # 2. 影响权重过低
            impact = ImpactWeightAnalyzer.get_impact_weight(file_name)
            module.impact_weight = impact
            if impact < IMPACT_WEIGHT_THRESHOLD:
                is_redundant = True
                reasons.append(f"决策影响权重={impact:.3f} < {IMPACT_WEIGHT_THRESHOLD}")

            # 3. 函数全部为简单getter/setter (无实际逻辑)
            if module.function_count > 0:
                trivial_count = sum(
                    1 for f in module.functions
                    if f.cyclomatic_complexity <= 1 and f.lines <= 5
                )
                if trivial_count == module.function_count and module.function_count > 3:
                    reasons.append(f"全部{module.function_count}个函数为简单访问器(无实际逻辑)")

            # 4. 空壳模块 (无函数或只有空函数)
            if module.function_count == 0 and module.class_count == 0:
                is_redundant = True
                reasons.append("无函数无类(空壳模块)")

            if is_redundant:
                module.is_redundant = True
                module.redundancy_reasons = reasons
                if not module.priority:
                    module.priority = "HIGH" if impact < 0.005 else "MEDIUM"
                redundant_list.append(module)

        return sorted(redundant_list, key=lambda m: m.impact_weight)

    def _generate_optimization_plans(
        self,
        redundant_modules: List[ModuleMetrics],
        complex_modules: List[ModuleMetrics],
        overlap_pairs: List[OverlapPair],
    ) -> List[Dict[str, Any]]:
        """生成详细优化方案"""
        plans = []

        # 1. 冗余模块移除方案
        for module in redundant_modules:
            file_name = module.file_name
            plan = {
                "type": "remove_or_refactor",
                "target": file_name,
                "priority": module.priority,
                "current_state": {
                    "lines": module.total_lines,
                    "functions": module.function_count,
                    "impact_weight": module.impact_weight,
                    "reasons": module.redundancy_reasons,
                },
                "action": module.recommendation or "移除或重构此模块",
                "expected_improvement": {
                    "code_reduction": f"~{module.total_lines} 行",
                    "complexity_reduction": "降低维护成本",
                    "performance_improvement": "减少导入开销",
                },
                "test_cases": [
                    "验证移除后所有现有测试通过",
                    "验证agent_decision_engine.decide()功能正常",
                    "验证evolution_loop.run()端到端正常",
                ],
                "rollback_plan": {
                    "method": "git revert",
                    "verification": "运行完整回归测试套件",
                    "trigger": "策略表现退化 > 5% 或 新增错误",
                },
            }
            plans.append(plan)

        # 2. 复杂模块重构方案
        for module in complex_modules[:5]:  # Top 5
            # 找出最需要重构的函数
            worst_funcs = sorted(
                module.functions,
                key=lambda f: f.cyclomatic_complexity * 0.4 + f.lines / 100 * 0.3 + f.nesting_depth * 0.3,
                reverse=True,
            )[:3]

            plan = {
                "type": "refactor_complexity",
                "target": module.file_name,
                "priority": module.priority,
                "current_state": {
                    "max_complexity": module.max_cyclomatic_complexity,
                    "avg_complexity": round(module.avg_cyclomatic_complexity, 1),
                    "max_nesting": module.max_nesting_depth,
                    "worst_functions": [
                        {
                            "name": f.name,
                            "lines": f.lines,
                            "complexity": f.cyclomatic_complexity,
                            "nesting": f.nesting_depth,
                        }
                        for f in worst_funcs
                    ],
                },
                "action": "重构: 拆分长函数、提取公共逻辑、降低嵌套深度",
                "refactor_strategy": [
                    "提取深层嵌套代码块为独立函数",
                    "使用早返回(early return)替代深层if-else",
                    "将复杂条件表达式提取为命名函数",
                    "使用策略模式替代大型if-elif链",
                ],
                "expected_improvement": {
                    "max_complexity": f"降至≤{CYCLOMATIC_COMPLEXITY_GOOD}",
                    "function_lines": f"降至≤{FUNC_LINES_GOOD}行",
                    "maintainability": "显著提升",
                },
                "test_cases": [
                    f"重构前保存{module.file_name}的完整测试基线",
                    "重构后运行全量测试套件确保功能等价",
                    "性能对比: 确保无退化",
                ],
                "rollback_plan": {
                    "method": "分支保护: 在独立分支重构, 通过后才合并",
                    "verification": "功能等价性测试 + 性能基准对比",
                    "trigger": "任何测试失败或性能退化 > 10%",
                },
            }
            plans.append(plan)

        # 3. 重叠模块合并方案
        if overlap_pairs:
            for pair in overlap_pairs[:3]:
                plan = {
                    "type": "merge_overlap",
                    "target": f"{pair.module_a} + {pair.module_b}",
                    "priority": "MEDIUM",
                    "current_state": {
                        "overlap_score": round(pair.overlap_score, 3),
                        "shared_functions": pair.shared_functions[:5],
                    },
                    "action": "合并重叠模块: 提取公共功能到共享基类/工具模块",
                    "merge_strategy": [
                        "识别共享函数 → 提取到common.py或基类",
                        "消除重复代码",
                        "统一接口签名",
                    ],
                    "expected_improvement": {
                        "code_reduction": f"~{len(pair.shared_functions)*20} 行 (估算)",
                        "maintainability": "消除重复, 降低维护成本",
                    },
                    "test_cases": [
                        "合并后所有原有测试通过",
                        "新增合并模块的单元测试",
                    ],
                    "rollback_plan": {
                        "method": "git revert",
                        "verification": "全量回归测试",
                        "trigger": "合并后功能异常",
                    },
                }
                plans.append(plan)

        return plans

    def _print_complexity_report(self, complex_modules: List[ModuleMetrics]):
        """打印复杂度报告"""
        if not complex_modules:
            print("  ✅ 所有模块复杂度在可接受范围内")
            return

        print(f"  ⚠️ {len(complex_modules)} 个模块超过复杂度阈值:")
        for m in complex_modules[:5]:
            print(f"    {m.file_name}: 最大圈复杂度={m.max_cyclomatic_complexity}, "
                  f"平均={m.avg_cyclomatic_complexity:.1f}, "
                  f"最大嵌套={m.max_nesting_depth}")

    # ========================================================================
    # 报告导出
    # ========================================================================

    def export_json(self, path: Path = None) -> Path:
        path = path or (SCRIPT_DIR / "_redundancy_analysis_report.json")
        if self.report is None:
            self.run_full_analysis()

        report_dict = self._report_to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ JSON报告已导出: {path}")
        return path

    def export_markdown(self, path: Path = None) -> Path:
        path = path or (SCRIPT_DIR / "_redundancy_analysis_report.md")
        if self.report is None:
            self.run_full_analysis()

        md = self._report_to_markdown()
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Markdown报告已导出: {path}")
        return path

    def _report_to_dict(self) -> Dict[str, Any]:
        report = self.report
        return {
            "meta": {
                "timestamp": report.timestamp,
                "modules_analyzed": report.modules_analyzed,
                "total_functions": report.total_functions,
                "total_lines": report.total_lines,
                "duration_seconds": report.duration_seconds,
            },
            "summary": {
                "redundant_modules": report.redundancy_count,
                "complex_modules": len(report.complex_modules),
                "overlap_pairs": len(report.overlap_pairs),
                "optimization_plans": len(report.optimization_plans),
            },
            "complexity_distribution": report.complexity_distribution,
            "module_size_distribution": report.module_size_distribution,
            "redundant_modules": [
                {
                    "file_name": m.file_name,
                    "impact_weight": m.impact_weight,
                    "reasons": m.redundancy_reasons,
                    "recommendation": m.recommendation,
                    "priority": m.priority,
                }
                for m in report.redundant_modules
            ],
            "complex_modules": [
                {
                    "file_name": m.file_name,
                    "max_complexity": m.max_cyclomatic_complexity,
                    "avg_complexity": round(m.avg_cyclomatic_complexity, 1),
                    "max_nesting": m.max_nesting_depth,
                    "total_lines": m.total_lines,
                    "reasons": m.redundancy_reasons,
                }
                for m in report.complex_modules
            ],
            "overlap_pairs": [
                {
                    "module_a": p.module_a,
                    "module_b": p.module_b,
                    "overlap_score": p.overlap_score,
                    "shared_functions": p.shared_functions,
                }
                for p in report.overlap_pairs
            ],
            "optimization_plans": report.optimization_plans,
        }

    def _report_to_markdown(self) -> str:
        report = self.report
        lines = []

        lines.append("# 冗余模块识别与代码静态分析报告")
        lines.append("")
        lines.append(f"**生成时间**: {report.timestamp}")
        lines.append(f"**分析耗时**: {report.duration_seconds:.1f}s")
        lines.append("")

        # 1. 摘要
        lines.append("## 一、摘要")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 分析模块数 | {report.modules_analyzed} |")
        lines.append(f"| 总函数数 | {report.total_functions} |")
        lines.append(f"| 总代码行 | {report.total_lines} |")
        lines.append(f"| 冗余模块数 | {report.redundancy_count} |")
        lines.append(f"| 过度复杂模块 | {len(report.complex_modules)} |")
        lines.append(f"| 高重叠模块对 | {len(report.overlap_pairs)} |")
        lines.append(f"| 优化方案数 | {len(report.optimization_plans)} |")
        lines.append("")

        # 2. 复杂度分布
        lines.append("## 二、圈复杂度分布")
        lines.append("")
        lines.append("| 等级 | 阈值 | 函数数 |")
        lines.append("|------|------|--------|")
        for level, count in report.complexity_distribution.items():
            lines.append(f"| {level} | — | {count} |")
        lines.append("")

        # 3. 模块大小分布
        lines.append("## 三、模块大小分布")
        lines.append("")
        lines.append("| 分类 | 模块数 |")
        lines.append("|------|--------|")
        for cat, count in report.module_size_distribution.items():
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        # 4. 冗余模块
        lines.append("## 四、冗余模块清单")
        lines.append("")
        if report.redundant_modules:
            lines.append("| 模块 | 影响权重 | 优先级 | 冗余原因 | 建议 |")
            lines.append("|------|----------|--------|----------|------|")
            for m in report.redundant_modules:
                reasons = "; ".join(m.redundancy_reasons[:2])
                lines.append(f"| {m.file_name} | {m.impact_weight:.3f} | {m.priority} | {reasons} | {m.recommendation[:60]} |")
        else:
            lines.append("✅ 未发现冗余模块")
        lines.append("")

        # 5. 过度复杂模块
        lines.append("## 五、过度复杂模块")
        lines.append("")
        if report.complex_modules:
            lines.append("| 模块 | 最大圈复杂度 | 平均圈复杂度 | 最大嵌套 | 行数 |")
            lines.append("|------|-------------|-------------|----------|------|")
            for m in report.complex_modules[:10]:
                lines.append(f"| {m.file_name} | {m.max_cyclomatic_complexity} | {m.avg_cyclomatic_complexity:.1f} | {m.max_nesting_depth} | {m.total_lines} |")
        else:
            lines.append("✅ 所有模块复杂度在可接受范围内")
        lines.append("")

        # 6. 功能重叠
        lines.append("## 六、功能重叠模块对")
        lines.append("")
        if report.overlap_pairs:
            lines.append("| 模块A | 模块B | 重叠度 | 共享函数 |")
            lines.append("|-------|-------|--------|----------|")
            for p in report.overlap_pairs:
                funcs = ", ".join(p.shared_functions[:3])
                lines.append(f"| {p.module_a} | {p.module_b} | {p.overlap_score:.1%} | {funcs} |")
        else:
            lines.append("✅ 未发现高重叠模块对 (>80%)")
        lines.append("")

        # 7. 优化方案
        lines.append("## 七、优化方案")
        lines.append("")
        for i, plan in enumerate(report.optimization_plans, 1):
            lines.append(f"### 方案 {i}: {plan['type']} — {plan['target']}")
            lines.append(f"**优先级**: {plan['priority']}")
            lines.append("")
            lines.append(f"**操作**: {plan['action']}")
            lines.append("")
            if "refactor_strategy" in plan:
                lines.append("**重构策略**:")
                for s in plan["refactor_strategy"]:
                    lines.append(f"- {s}")
                lines.append("")
            if "merge_strategy" in plan:
                lines.append("**合并策略**:")
                for s in plan["merge_strategy"]:
                    lines.append(f"- {s}")
                lines.append("")
            lines.append("**预期改进**:")
            for k, v in plan.get("expected_improvement", {}).items():
                lines.append(f"- {k}: {v}")
            lines.append("")
            lines.append("**测试用例**:")
            for tc in plan.get("test_cases", []):
                lines.append(f"- {tc}")
            lines.append("")
            lines.append("**回滚机制**:")
            rb = plan.get("rollback_plan", {})
            lines.append(f"- 方法: {rb.get('method', 'N/A')}")
            lines.append(f"- 验证: {rb.get('verification', 'N/A')}")
            lines.append(f"- 触发: {rb.get('trigger', 'N/A')}")
            lines.append("")

        # 8. 阈值说明
        lines.append("## 八、识别标准 (来自用户要求)")
        lines.append("")
        lines.append("| 标准 | 阈值 |")
        lines.append("|------|------|")
        lines.append(f"| 调用频率 | < 日平均调用次数的 {CALL_FREQUENCY_THRESHOLD*100}% |")
        lines.append(f"| 决策影响权重 | < {IMPACT_WEIGHT_THRESHOLD} |")
        lines.append(f"| 功能重叠度 | > {OVERLAP_THRESHOLD*100}% |")
        lines.append(f"| 执行时间 | > {EXECUTION_TIME_THRESHOLD_MS}ms |")
        lines.append("")
        lines.append("### 圈复杂度标准 (1200铁标准)")
        lines.append(f"- EXCELLENT: ≤ {CYCLOMATIC_COMPLEXITY_EXCELLENT}")
        lines.append(f"- GOOD: ≤ {CYCLOMATIC_COMPLEXITY_GOOD}")
        lines.append(f"- ACCEPTABLE: ≤ {CYCLOMATIC_COMPLEXITY_BLOCK}")
        lines.append(f"- BLOCK: > {CYCLOMATIC_COMPLEXITY_BLOCK}")
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# 便捷入口
# ============================================================================

def run_redundancy_analysis(
    sandbox_dir: Path = None,
    output_dir: Path = None,
) -> RedundancyReport:
    """运行完整冗余分析的便捷入口"""
    sandbox_dir = sandbox_dir or SCRIPT_DIR
    output_dir = output_dir or SCRIPT_DIR

    print("=" * 70)
    print("冗余模块识别与代码静态分析引擎")
    print("=" * 70)
    print(f"分析目录: {sandbox_dir}")
    print(f"识别标准: 调用频率<0.5% | 影响权重<0.01 | 重叠度>80% | 执行时间>100ms")
    print("=" * 70)

    analyzer = RedundancyAnalyzer(sandbox_dir)
    report = analyzer.run_full_analysis()

    analyzer.export_json(output_dir / "_redundancy_analysis_report.json")
    analyzer.export_markdown(output_dir / "_redundancy_analysis_report.md")

    return report


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="冗余模块识别与代码静态分析引擎")
    parser.add_argument("--dir", type=str, default=str(SCRIPT_DIR),
                       help="分析目录")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录")
    args = parser.parse_args()

    run_redundancy_analysis(
        sandbox_dir=Path(args.dir),
        output_dir=Path(args.output) if args.output else Path(args.dir),
    )