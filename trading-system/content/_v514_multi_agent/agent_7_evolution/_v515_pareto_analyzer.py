#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v515 帕累托前沿分析器 — Agent-7 决策支持模块
================================================================
功能:
  1. 加载 NSGA-II 产出的帕累托前沿 (_v515_pareto_front.json)
  2. 多维分析: 目标分布, 权衡曲线, 极端点识别
  3. 可视化: ASCII 散点图 (无需 matplotlib, 终端友好)
  4. 决策支持: 从帕累托前沿中选择"最稳健"的解 (非极端最优)
  5. 进化收敛性诊断: 前沿大小变化, 目标均值变化
  6. 与 v515 单目标 GA 对比: 提升/退化分析
  7. Markdown 报告生成 (中文)

"最稳健解" 选择策略 (4 个候选, 取综合最优):
  - knee_point:    帕累托前沿拐点 (距极端点最远)
  - balanced:      所有目标归一化后均值最高
  - max_kpi:       KPI 通过数最多 (用户7关卡)
  - max_sharpe:    夏普最高 (激进)
  - min_monthly_loss: 月亏最低 (保守)
  - 推荐: balanced (兼顾所有目标, 非极端)
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

HERE = Path(__file__).parent.resolve()


# ============================================================
# 1. 数据结构
# ============================================================
@dataclass
class ParetoSolution:
    """单个帕累托前沿解"""
    genes: Dict[str, float]
    objectives: Dict[str, float]      # 5 个原始目标
    kpi: Dict[str, float]             # 辅助 KPI
    fitness_scalar: float
    # 分析时填充
    norm_objectives: Dict[str, float] = field(default_factory=dict)
    distance_to_ideal: float = 0.0
    distance_to_nadir: float = 0.0
    knee_score: float = 0.0


@dataclass
class AnalysisReport:
    """帕累托前沿分析报告"""
    front_size: int
    ideal_point: Dict[str, float]        # 各目标的最优值
    nadir_point: Dict[str, float]        # 各目标的最差值 (在前沿内)
    extremes: Dict[str, Dict[str, float]]  # 各目标极值解
    recommendations: Dict[str, ParetoSolution]  # 各种推荐策略
    convergence: Dict[str, Any]          # 收敛性分析
    comparison_v515: Dict[str, Any]      # 与 v515 对比


# ============================================================
# 2. 帕累托分析器
# ============================================================
class ParetoAnalyzer:
    """
    帕累托前沿分析器

    用法:
      analyzer = ParetoAnalyzer()
      analyzer.load("_v515_pareto_front.json")
      report = analyzer.analyze()
      analyzer.print_ascii_scatter("sharpe", "win_rate")
      best = analyzer.recommend_balanced()
      analyzer.save_markdown_report()
    """

    # 目标方向: True=maximize, False=minimize
    OBJECTIVE_DIRECTIONS = {
        "sharpe": True,
        "win_rate": True,
        "n_trades": True,
        "monthly_loss": False,
        "profit_loss_ratio": True,
    }

    def __init__(self):
        self.solutions: List[ParetoSolution] = []
        self.history: List[Dict[str, Any]] = []
        self.targets: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}

    # ---------- 加载 ----------

    def load(self, path: Path) -> None:
        """加载 NSGA-II 产出的帕累托前沿 JSON"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.metadata = {
            "version": data.get("version"),
            "timestamp": data.get("timestamp"),
            "pop_size": data.get("pop_size"),
            "generations": data.get("generations"),
        }
        self.history = data.get("history", [])
        self.targets = data.get("targets", {})
        self.solutions = []
        for entry in data.get("pareto_front", []):
            self.solutions.append(ParetoSolution(
                genes=entry["genes"],
                objectives=entry["objectives"],
                kpi=entry.get("kpi", {}),
                fitness_scalar=entry.get("fitness_scalar", 0.0),
            ))

    def load_from_engine(self, engine) -> None:
        """直接从 NSGA2Engine 实例加载 (无需写 JSON)"""
        from _v515_evolution_engine import NSGAIndividual, OBJECTIVE_DIRECTIONS  # type: ignore
        self.metadata = {
            "version": "v515-nsga2",
            "timestamp": time.time(),
            "pop_size": engine.pop_size,
            "generations": engine.generations,
        }
        self.history = engine.history
        self.targets = dict(getattr(engine, "_targets_cache", {}))  # 可能没有
        self.solutions = []
        for ind in engine.pareto_front:
            self.solutions.append(ParetoSolution(
                genes=dict(ind.genes),
                objectives={
                    "sharpe": ind.sharpe,
                    "win_rate": ind.win_rate,
                    "n_trades": ind.n_trades,
                    "monthly_loss": ind.monthly_loss,
                    "profit_loss_ratio": ind.profit_loss_ratio,
                },
                kpi={
                    "kpi_pass": ind.kpi_pass,
                    "max_dd": ind.max_dd,
                    "annual_return": ind.annual_return,
                    "consec_loss": ind.consec_loss,
                    "win_rate_ci_lo": ind.win_rate_ci_lo,
                    "is_sharpe": ind.is_sharpe,
                    "df": ind.df,
                },
                fitness_scalar=ind.fitness_scalar,
            ))

    # ---------- 分析 ----------

    def analyze(self) -> AnalysisReport:
        """执行完整分析: 理想点, 极端点, 推荐解, 收敛性"""
        if not self.solutions:
            raise ValueError("无帕累托前沿解, 请先 load()")

        # 理想点 (各目标最优) / Nadir 点 (前沿内最差)
        ideal_point: Dict[str, float] = {}
        nadir_point: Dict[str, float] = {}
        for obj, is_max in self.OBJECTIVE_DIRECTIONS.items():
            vals = [s.objectives.get(obj, 0.0) for s in self.solutions]
            if not vals:
                continue
            if is_max:
                ideal_point[obj] = max(vals)
                nadir_point[obj] = min(vals)
            else:
                ideal_point[obj] = min(vals)
                nadir_point[obj] = max(vals)

        # 归一化目标到 [0, 1] (1=理想, 0=nadir)
        for s in self.solutions:
            for obj, is_max in self.OBJECTIVE_DIRECTIONS.items():
                v = s.objectives.get(obj, 0.0)
                i = ideal_point[obj]
                n = nadir_point[obj]
                denom = i - n if abs(i - n) > 1e-12 else 1.0
                # max: (v - nadir) / (ideal - nadir)
                # min: (nadir - v) / (nadir - ideal) = (v - nadir) / (ideal - nadir) 仍然
                # 统一: norm = (v - nadir) / (ideal - nadir), 都是 1=理想, 0=nadir
                s.norm_objectives[obj] = (v - n) / denom

        # 距离理想点 (归一化空间, 欧氏距离)
        for s in self.solutions:
            sq_sum = sum((1.0 - s.norm_objectives[obj]) ** 2
                         for obj in self.OBJECTIVE_DIRECTIONS)
            s.distance_to_ideal = math.sqrt(sq_sum)
            sq_sum2 = sum(s.norm_objectives[obj] ** 2
                          for obj in self.OBJECTIVE_DIRECTIONS)
            s.distance_to_nadir = math.sqrt(sq_sum2)

        # Knee score: 距 nadir 远 / 距 ideal 近 = 拐点
        # 公式: knee_score = distance_to_nadir - distance_to_ideal
        # 越大表示越靠近"拐点" (远离 nadir, 但又不极端靠近 ideal)
        for s in self.solutions:
            s.knee_score = s.distance_to_nadir - s.distance_to_ideal

        # 极端点: 各目标最优的解
        extremes: Dict[str, Dict[str, float]] = {}
        for obj, is_max in self.OBJECTIVE_DIRECTIONS.items():
            if is_max:
                best_s = max(self.solutions, key=lambda s: s.objectives.get(obj, 0.0))
            else:
                best_s = min(self.solutions, key=lambda s: s.objectives.get(obj, 0.0))
            extremes[obj] = {
                "value": best_s.objectives.get(obj, 0.0),
                "fitness_scalar": best_s.fitness_scalar,
                "kpi_pass": best_s.kpi.get("kpi_pass", 0),
            }

        # 推荐解
        recommendations: Dict[str, ParetoSolution] = {
            "balanced": self._recommend_balanced(),
            "knee_point": self._recommend_knee(),
            "max_kpi": self._recommend_max_kpi(),
            "max_sharpe": self._recommend_max_objective("sharpe"),
            "min_monthly_loss": self._recommend_min_objective("monthly_loss"),
            "max_profit_loss_ratio": self._recommend_max_objective("profit_loss_ratio"),
        }

        # 收敛性分析
        convergence = self._analyze_convergence()

        # 与 v515 对比 (从 history 取首末代)
        comparison = self._compare_v515()

        return AnalysisReport(
            front_size=len(self.solutions),
            ideal_point=ideal_point,
            nadir_point=nadir_point,
            extremes=extremes,
            recommendations=recommendations,
            convergence=convergence,
            comparison_v515=comparison,
        )

    def _recommend_balanced(self) -> ParetoSolution:
        """最稳健: 归一化目标均值最高 (兼顾所有目标)"""
        return max(self.solutions, key=lambda s: np.mean(list(s.norm_objectives.values())))

    def _recommend_knee(self) -> ParetoSolution:
        """拐点: 距 nadir 远 - 距 ideal 近 (帕累托前沿拐弯处)"""
        return max(self.solutions, key=lambda s: s.knee_score)

    def _recommend_max_kpi(self) -> ParetoSolution:
        """KPI 通过数最多 (用户7关卡)"""
        return max(self.solutions, key=lambda s: s.kpi.get("kpi_pass", 0))

    def _recommend_max_objective(self, obj: str) -> ParetoSolution:
        return max(self.solutions, key=lambda s: s.objectives.get(obj, 0.0))

    def _recommend_min_objective(self, obj: str) -> ParetoSolution:
        return min(self.solutions, key=lambda s: s.objectives.get(obj, 0.0))

    # ---------- 收敛性分析 ----------

    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析进化收敛性"""
        if len(self.history) < 2:
            return {"status": "INSUFFICIENT_HISTORY", "message": "历史数据不足"}

        first = self.history[0]
        last = self.history[-1]

        # 前沿大小变化
        front_sizes = [h.get("front_size", 0) for h in self.history]
        # 目标均值变化
        sharpe_progress = [h.get("front_sharpe", 0) for h in self.history]
        winrate_progress = [h.get("front_winrate", 0) for h in self.history]
        trades_progress = [h.get("front_trades", 0) for h in self.history]
        monthly_loss_progress = [h.get("front_monthly_loss", 1) for h in self.history]
        pf_progress = [h.get("front_pf", 0) for h in self.history]

        # 收敛判定: 最后 3 代前沿均值变化 < 5%
        last_3 = self.history[-3:] if len(self.history) >= 3 else self.history
        if len(last_3) >= 2:
            sharpe_change = abs(last_3[-1]["front_sharpe"] - last_3[0]["front_sharpe"])
            sharpe_mag = abs(last_3[-1]["front_sharpe"]) + 1e-9
            sharpe_rel_change = sharpe_change / sharpe_mag
            converged = sharpe_rel_change < 0.05
        else:
            converged = False
            sharpe_rel_change = 1.0

        # 趋势: 单调上升 / 平稳 / 下降
        def trend(vals):
            if len(vals) < 2:
                return "unknown"
            up = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
            down = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i-1])
            if up > down * 1.5:
                return "improving"
            if down > up * 1.5:
                return "degrading"
            return "stable"

        return {
            "status": "CONVERGED" if converged else "STILL_EVOLVING",
            "converged": converged,
            "last_3_sharpe_rel_change": round(sharpe_rel_change, 4),
            "front_size_change": {
                "first": first.get("front_size", 0),
                "last": last.get("front_size", 0),
                "max": max(front_sizes) if front_sizes else 0,
                "trend": trend(front_sizes),
            },
            "sharpe_progress": {
                "first": first.get("front_sharpe", 0),
                "last": last.get("front_sharpe", 0),
                "max": max(sharpe_progress) if sharpe_progress else 0,
                "trend": trend(sharpe_progress),
            },
            "winrate_progress": {
                "first": first.get("front_winrate", 0),
                "last": last.get("front_winrate", 0),
                "trend": trend(winrate_progress),
            },
            "trades_progress": {
                "first": first.get("front_trades", 0),
                "last": last.get("front_trades", 0),
                "trend": trend(trades_progress),
            },
            "monthly_loss_progress": {
                "first": first.get("front_monthly_loss", 1),
                "last": last.get("front_monthly_loss", 1),
                "trend": trend(monthly_loss_progress),
            },
            "pf_progress": {
                "first": first.get("front_pf", 0),
                "last": last.get("front_pf", 0),
                "trend": trend(pf_progress),
            },
            "generations": len(self.history),
        }

    # ---------- 与 v515 单目标 GA 对比 ----------

    def _compare_v515(self) -> Dict[str, Any]:
        """与 v515 单目标 GA 对比"""
        # v515 基线 (来自 _v515_ensemble_strategy.py 默认参数: pop=20, gen=8)
        v515_baseline = {
            "pop_size": 20,
            "generations": 8,
            "algorithm": "single_objective_GA_weighted_sum",
            "objectives_count": 1,
            "front_size": 1,  # 单目标只保留 1 个最优
        }
        v515_meta = {
            "pop_size": self.metadata.get("pop_size", 50),
            "generations": self.metadata.get("generations", 20),
            "algorithm": "NSGA-II",
            "objectives_count": 5,
            "front_size": len(self.solutions),
        }
        # 提升 (NSGA-II vs v515)
        return {
            "v515_baseline": v515_baseline,
            "nsga2_v515": v515_meta,
            "improvements": {
                "pop_size_x": round(v515_meta["pop_size"] / v515_baseline["pop_size"], 2),
                "generations_x": round(v515_meta["generations"] / v515_baseline["generations"], 2),
                "objectives_x": v515_meta["objectives_count"] / v515_baseline["objectives_count"],
                "front_size_x": v515_meta["front_size"] / v515_baseline["front_size"],
                "total_evaluations_x": round(
                    (v515_meta["pop_size"] * v515_meta["generations"]) /
                    (v515_baseline["pop_size"] * v515_baseline["generations"]), 2
                ),
            },
        }

    # ---------- 可视化 ----------

    def print_ascii_scatter(self, x_obj: str, y_obj: str, width: int = 60, height: int = 20) -> str:
        """
        ASCII 散点图: 在终端显示 2 目标散点
        适用于无 matplotlib 环境
        """
        if not self.solutions:
            return "(无数据)"

        xs = [s.objectives.get(x_obj, 0.0) for s in self.solutions]
        ys = [s.objectives.get(y_obj, 0.0) for s in self.solutions]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max - x_min < 1e-9:
            x_max = x_min + 1.0
        if y_max - y_min < 1e-9:
            y_max = y_min + 1.0

        # 网格
        grid = [[" " for _ in range(width)] for _ in range(height)]
        # 标记点
        for x, y in zip(xs, ys):
            col = int((x - x_min) / (x_max - x_min) * (width - 1))
            row = int((y_max - y) / (y_max - y_min) * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = "*"

        # 标记 balanced 推荐
        balanced = self._recommend_balanced()
        bx = balanced.objectives.get(x_obj, 0.0)
        by = balanced.objectives.get(y_obj, 0.0)
        bcol = int((bx - x_min) / (x_max - x_min) * (width - 1))
        brow = int((y_max - by) / (y_max - y_min) * (height - 1))
        bcol = max(0, min(width - 1, bcol))
        brow = max(0, min(height - 1, brow))
        grid[brow][bcol] = "B"

        # 输出
        lines = []
        x_dir = "↑" if self.OBJECTIVE_DIRECTIONS.get(x_obj, True) else "↓"
        y_dir = "↑" if self.OBJECTIVE_DIRECTIONS.get(y_obj, True) else "↓"
        title = f"帕累托前沿: {y_obj}({y_dir}) vs {x_obj}({x_dir})  [B=balanced推荐]"
        lines.append(title)
        lines.append(f"y:{y_max:.3f} +" + "-" * width + "+")
        for row in grid:
            lines.append("         |" + "".join(row) + "|")
        lines.append(f"y:{y_min:.3f} +" + "-" * width + "+")
        lines.append(f"          {'^':<{width}}")
        lines.append(f"         x:{x_min:.3f}".ljust(width + 15) + f"x:{x_max:.3f}")

        text = "\n".join(lines)
        print(text)
        return text

    def print_parallel_coordinates(self) -> str:
        """
        平行坐标图 (ASCII): 5 个目标并排显示, 每条线代表一个解
        适合多目标帕累托前沿可视化
        """
        if not self.solutions:
            return "(无数据)"

        objectives = list(self.OBJECTIVE_DIRECTIONS.keys())
        n_obj = len(objectives)
        width = 50
        height = 20

        # 用归一化目标 (1=理想, 0=nadir)
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # 每个解画一条线
        for s in self.solutions:
            prev_col = None
            prev_row = None
            for i, obj in enumerate(objectives):
                col = int(i * (width - 1) / (n_obj - 1)) if n_obj > 1 else 0
                norm = s.norm_objectives.get(obj, 0.5)
                row = int((1.0 - norm) * (height - 1))
                col = max(0, min(width - 1, col))
                row = max(0, min(height - 1, row))
                if prev_col is not None:
                    # 画线段 (简易, 用 - 或 |)
                    if row == prev_row:
                        for c in range(min(prev_col, col), max(prev_col, col) + 1):
                            if grid[row][c] == " ":
                                grid[row][c] = "-"
                    else:
                        for r in range(min(prev_row, row), max(prev_row, row) + 1):
                            if grid[r][col] == " ":
                                grid[r][col] = "|"
                if grid[row][col] in (" ", "-", "|"):
                    grid[row][col] = "+"
                prev_col, prev_row = col, row

        # 标记 balanced 推荐
        balanced = self._recommend_balanced()
        prev_col = None
        for i, obj in enumerate(objectives):
            col = int(i * (width - 1) / (n_obj - 1)) if n_obj > 1 else 0
            norm = balanced.norm_objectives.get(obj, 0.5)
            row = int((1.0 - norm) * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = "B"

        lines = []
        lines.append("平行坐标图 (5目标, B=balanced推荐, 1=理想)")
        lines.append("1.0(理想) +" + "-" * width + "+")
        for row in grid:
            lines.append("          |" + "".join(row) + "|")
        lines.append("0.0(nadir) +" + "-" * width + "+")
        # 目标标签
        label_row = [" "] * width
        for i, obj in enumerate(objectives):
            col = int(i * (width - 1) / (n_obj - 1)) if n_obj > 1 else 0
            dir_mark = "↑" if self.OBJECTIVE_DIRECTIONS[obj] else "↓"
            label = f"{obj[:6]}{dir_mark}"
            for j, ch in enumerate(label):
                if col + j < width:
                    label_row[col + j] = ch
        lines.append("           " + "".join(label_row))

        text = "\n".join(lines)
        print(text)
        return text

    # ---------- Markdown 报告 ----------

    def save_markdown_report(self, output_path: Optional[Path] = None) -> Path:
        """生成完整中文 Markdown 报告"""
        if output_path is None:
            output_path = HERE / "_v515_evolution_report.md"

        report = self.analyze()
        md = self._build_markdown(report)
        Path(output_path).write_text(md, encoding="utf-8")
        return output_path

    def _build_markdown(self, r: AnalysisReport) -> str:
        """构建 Markdown 报告文本"""
        lines: List[str] = []
        lines.append("# v515 NSGA-II 进化报告 — Agent-7 策略进化引擎")
        lines.append("")
        lines.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.metadata.get('timestamp', time.time())))}")
        lines.append(f"> 算法: NSGA-II (快速非支配排序 + 拥挤距离)")
        lines.append(f"> 种群: {self.metadata.get('pop_size')} × 代数: {self.metadata.get('generations')}")
        lines.append("")

        # 1. 概要
        lines.append("## 1. 进化概要")
        lines.append("")
        lines.append(f"- 帕累托前沿大小: **{r.front_size}** 个非支配解")
        lines.append(f"- 优化目标: 5 个 (夏普↑, 胜率↑, 交易数↑, 月亏↓, 盈亏比↑)")
        lines.append(f"- 总评估次数: ~{self.metadata.get('pop_size', 0) * self.metadata.get('generations', 0)}")
        lines.append("")

        # 2. 理想点与极端点
        lines.append("## 2. 理想点与极端点")
        lines.append("")
        lines.append("| 目标 | 方向 | 理想值 | Nadir值(前沿内最差) | 极端解KPI |")
        lines.append("|------|------|--------|---------------------|-----------|")
        for obj, is_max in self.OBJECTIVE_DIRECTIONS.items():
            dir_str = "↑ maximize" if is_max else "↓ minimize"
            ideal = r.ideal_point.get(obj, 0.0)
            nadir = r.nadir_point.get(obj, 0.0)
            ext = r.extremes.get(obj, {})
            ext_kpi = ext.get("kpi_pass", 0)
            lines.append(
                f"| {obj} | {dir_str} | {ideal:.4f} | {nadir:.4f} | KPI={ext_kpi}/7 |"
            )
        lines.append("")

        # 3. 推荐解
        lines.append("## 3. 推荐解 (决策支持)")
        lines.append("")
        lines.append("### 3.1 各种推荐策略对比")
        lines.append("")
        lines.append("| 策略 | 夏普 | 胜率 | 交易数 | 月亏 | 盈亏比 | KPI | 年化 | 回撤 |")
        lines.append("|------|------|------|--------|------|--------|-----|------|------|")
        for name, sol in r.recommendations.items():
            o = sol.objectives
            k = sol.kpi
            lines.append(
                f"| **{name}** | {o.get('sharpe',0):.3f} | {o.get('win_rate',0)*100:.1f}% | "
                f"{o.get('n_trades',0):.1f} | {o.get('monthly_loss',0)*100:.1f}% | "
                f"{o.get('profit_loss_ratio',0):.2f} | {k.get('kpi_pass',0)}/7 | "
                f"{k.get('annual_return',0)*100:.1f}% | {k.get('max_dd',0)*100:.1f}% |"
            )
        lines.append("")

        # 推荐 balanced 详细基因
        balanced = r.recommendations["balanced"]
        lines.append("### 3.2 最稳健解 (balanced) 基因")
        lines.append("")
        lines.append("选择理由: 归一化后 5 目标均值最高, 兼顾所有目标, 非极端。")
        lines.append("")
        lines.append("```python")
        for k, v in sorted(balanced.genes.items()):
            lines.append(f"  {k:25s}: {v:.4f}")
        lines.append("```")
        lines.append("")

        # 4. 收敛性分析
        lines.append("## 4. 进化收敛性分析")
        lines.append("")
        conv = r.convergence
        lines.append(f"- **状态**: {conv.get('status', 'unknown')}")
        lines.append(f"- **最后3代夏普相对变化**: {conv.get('last_3_sharpe_rel_change', 'N/A')}")
        lines.append(f"- **是否收敛** (变化<5%): {'是 ✅' if conv.get('converged') else '否 (仍在进化)'}")
        lines.append("")
        lines.append("### 4.1 各目标进展趋势")
        lines.append("")
        lines.append("| 目标 | 首代 | 末代 | 趋势 |")
        lines.append("|------|------|------|------|")
        for key in ["sharpe_progress", "winrate_progress", "trades_progress",
                    "monthly_loss_progress", "pf_progress", "front_size_change"]:
            prog = conv.get(key, {})
            first = prog.get("first", "N/A")
            last = prog.get("last", "N/A")
            trend = prog.get("trend", "unknown")
            trend_mark = {
                "improving": "📈 改善",
                "stable": "➡️ 平稳",
                "degrading": "📉 退化",
                "unknown": "? 未知",
            }.get(trend, trend)
            label = key.replace("_progress", "").replace("_change", "")
            if isinstance(first, float):
                first_str = f"{first:.4f}"
            else:
                first_str = str(first)
            if isinstance(last, float):
                last_str = f"{last:.4f}"
            else:
                last_str = str(last)
            lines.append(f"| {label} | {first_str} | {last_str} | {trend_mark} |")
        lines.append("")

        # 5. 与 v515 单目标 GA 对比
        lines.append("## 5. 与 v515 单目标 GA 对比")
        lines.append("")
        comp = r.comparison_v515
        lines.append("| 维度 | v515 单目标GA | v515 NSGA-II (本引擎) | 提升 |")
        lines.append("|------|---------------|----------------------|------|")
        imp = comp["improvements"]
        lines.append(f"| 种群大小 | {comp['v515_baseline']['pop_size']} | {comp['nsga2_v515']['pop_size']} | {imp['pop_size_x']}x |")
        lines.append(f"| 进化代数 | {comp['v515_baseline']['generations']} | {comp['nsga2_v515']['generations']} | {imp['generations_x']}x |")
        lines.append(f"| 目标数 | {comp['v515_baseline']['objectives_count']} | {comp['nsga2_v515']['objectives_count']} | {imp['objectives_x']}x |")
        lines.append(f"| 输出解数 | {comp['v515_baseline']['front_size']} | {comp['nsga2_v515']['front_size']} | {imp['front_size_x']}x |")
        lines.append(f"| 总评估次数 | ~{comp['v515_baseline']['pop_size']*comp['v515_baseline']['generations']} | "
                     f"~{comp['nsga2_v515']['pop_size']*comp['nsga2_v515']['generations']} | {imp['total_evaluations_x']}x |")
        lines.append("")
        lines.append("**关键进化**:")
        lines.append("- v515 单目标 GA: 用加权和把 5 目标压扁为 1 个 fitness, 只输出 1 个最优解")
        lines.append("- NSGA-II: 真帕累托前沿, 保留所有非支配解, 给决策者完整权衡空间")
        lines.append("- 知识库联动: 历史最优基因注入 + 进化记忆持久化, 跨代累积")
        lines.append("")

        # 6. 用户硬指标达成情况
        if self.targets:
            lines.append("## 6. 用户硬指标 (7 关卡) 达成情况")
            lines.append("")
            balanced_kpi = balanced.kpi
            lines.append("以最稳健解 (balanced) 为准:")
            lines.append("")
            lines.append("| 关卡 | 目标 | 实际 | 达成 |")
            lines.append("|------|------|------|------|")
            checks = [
                ("夏普比率", f"≥{self.targets.get('sharpe', 1.5)}",
                 f"{balanced.objectives.get('sharpe',0):.3f}",
                 balanced.objectives.get('sharpe',0) >= self.targets.get('sharpe', 1.5)),
                ("胜率", f"≥{self.targets.get('win_rate',0.55)*100:.0f}%",
                 f"{balanced.objectives.get('win_rate',0)*100:.1f}%",
                 balanced.objectives.get('win_rate',0) >= self.targets.get('win_rate',0.55)),
                ("交易数", f"≥{self.targets.get('trades_min',30)}",
                 f"{balanced.objectives.get('n_trades',0):.1f}",
                 balanced.objectives.get('n_trades',0) >= self.targets.get('trades_min',30)),
                ("月亏概率", f"≤{self.targets.get('monthly_loss_prob',0.05)*100:.0f}%",
                 f"{balanced.objectives.get('monthly_loss',0)*100:.1f}%",
                 balanced.objectives.get('monthly_loss',0) <= self.targets.get('monthly_loss_prob',0.05)),
                ("最大回撤", f"≤{self.targets.get('max_drawdown',0.15)*100:.0f}%",
                 f"{balanced_kpi.get('max_dd',0)*100:.1f}%",
                 balanced_kpi.get('max_dd',0) <= self.targets.get('max_drawdown',0.15)),
                ("年化收益", f"≥{self.targets.get('annual_return',0.30)*100:.0f}%",
                 f"{balanced_kpi.get('annual_return',0)*100:.1f}%",
                 balanced_kpi.get('annual_return',0) >= self.targets.get('annual_return',0.30)),
                ("连续亏损", f"≤{self.targets.get('consecutive_losses',3)}",
                 f"{balanced_kpi.get('consec_loss',0):.0f}",
                 balanced_kpi.get('consec_loss',0) <= self.targets.get('consecutive_losses',3)),
            ]
            for name, target, actual, passed in checks:
                lines.append(f"| {name} | {target} | {actual} | {'✅' if passed else '❌'} |")
            n_pass = sum(1 for c in checks if c[3])
            lines.append("")
            lines.append(f"**达成: {n_pass}/{len(checks)} 关卡**")
            lines.append("")

        # 7. 结论
        lines.append("## 7. 结论与建议")
        lines.append("")
        conv_status = conv.get("status", "")
        if conv.get("converged"):
            lines.append(f"- **进化已收敛** (最后3代变化<5%), 当前解集已接近最优前沿")
            lines.append(f"- 建议: 用 balanced 解部署, 或继续增大种群/代数探索更广前沿")
        else:
            lines.append(f"- **进化尚未收敛** (最后3代变化≥5%), 继续进化可获得更好解")
            lines.append(f"- 建议: 增加代数到 30-50 代, 或增大种群到 100")
        lines.append(f"- 帕累托前沿大小: {r.front_size} (越大越好, 表示目标间有真实权衡)")
        lines.append(f"- 推荐部署解: balanced (夏普={balanced.objectives.get('sharpe',0):.3f}, "
                     f"胜率={balanced.objectives.get('win_rate',0)*100:.1f}%, "
                     f"KPI={balanced.kpi.get('kpi_pass',0)}/7)")
        lines.append("")

        return "\n".join(lines)


# ============================================================
# 3. 主入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="v515 帕累托前沿分析器")
    parser.add_argument("--input", default=str(HERE / "_v515_pareto_front.json"),
                        help="帕累托前沿 JSON 路径")
    parser.add_argument("--report", default=str(HERE / "_v515_evolution_report.md"),
                        help="输出 Markdown 报告路径")
    args = parser.parse_args()

    analyzer = ParetoAnalyzer()
    analyzer.load(Path(args.input))
    print(f"加载 {len(analyzer.solutions)} 个帕累托前沿解")

    report = analyzer.analyze()
    print(f"\n=== 分析结果 ===")
    print(f"前沿大小: {report.front_size}")
    print(f"理想点: {report.ideal_point}")
    print(f"Nadir点: {report.nadir_point}")
    print(f"收敛状态: {report.convergence.get('status')}")
    print(f"\n推荐解:")
    for name, sol in report.recommendations.items():
        o = sol.objectives
        print(f"  {name:25s}: 夏普={o.get('sharpe',0):.3f}, 胜率={o.get('win_rate',0)*100:.1f}%, "
              f"交易数={o.get('n_trades',0):.1f}, 月亏={o.get('monthly_loss',0)*100:.1f}%, "
              f"盈亏比={o.get('profit_loss_ratio',0):.2f}, KPI={sol.kpi.get('kpi_pass',0)}/7")

    print(f"\n=== ASCII 可视化 ===")
    analyzer.print_ascii_scatter("sharpe", "win_rate")
    print()
    analyzer.print_parallel_coordinates()

    report_path = analyzer.save_markdown_report(Path(args.report))
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
