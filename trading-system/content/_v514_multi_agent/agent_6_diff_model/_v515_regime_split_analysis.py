#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v515 状态分割分析 (Agent-6 第二交付)
=====================================
按 4 种市场状态 (TRENDING/RANGING/HIGH_VOL/LOW_VOL) 分割模拟vs实盘差异,
回答用户问题: "哪些状态下差异最大? 哪个因素在每个状态下贡献最多?"

核心输出: 状态 × 因素 影响矩阵 (4×4)
              | spread | slippage | latency | liquidity |
   TRENDING   |   ?    |    ?     |    ?    |     ?     |
   RANGING    |   ?    |    ?     |    ?    |     ?     |
   HIGH_VOL   |   ?    |    ?     |    ?    |     ?     |
   LOW_VOL    |   ?    |    ?     |    ?    |     ?     |

每个单元格 = 该状态下该因素对差异的贡献占比 (%)。

方法:
  1. 对每个状态单独跑 OLS 回归, 取标准化系数
  2. 取所有 (symbol × regime) 单元的平均因素成本
  3. 输出每个状态的差异率、首要瓶颈、改善优先级
  4. 识别"高差异状态" (差异率最高) 和"高风险因素" (跨状态平均权重最高)

依赖: 复用 _v515_diff_quantification.py 的样本生成与回归函数
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

HERE = Path(__file__).parent.resolve()
# 复用主模型的函数与常量
sys.path.insert(0, str(HERE))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "diff_quant", HERE / "_v515_diff_quantification.py"
)
diff_quant = importlib.util.module_from_spec(_spec)
# 必须在 exec_module 前注册到 sys.modules, 否则 @dataclass 装饰器查找模块会失败
sys.modules["diff_quant"] = diff_quant
_spec.loader.exec_module(diff_quant)

from diff_quant import (
    SYMBOLS, SYMBOL_PARAMS, REGIME_FACTORS,
    LIVE_LATENCY_MS, TARGET_DIFF_RATE, RANDOM_SEED,
    load_sim_baseline, generate_samples, ols_regression,
    compute_friction_costs, TradeRecord,
)

RESULT_JSON_PATH = HERE / "_v515_regime_split_result.json"
REPORT_MD_PATH = HERE / "_v515_regime_split_report.md"


# ============================================================
# 1. 按状态分割并回归
# ============================================================
def regime_split_regression(samples: List[TradeRecord]) -> Dict[str, Dict]:
    """对每个 regime 单独跑回归, 返回该 regime 下的因素权重与差异率。"""
    out = {}
    for regime in REGIME_FACTORS:
        sub = [s for s in samples if s.regime == regime]
        if not sub:
            continue
        X = np.array([
            [1.0, s.spread_bps, s.slippage_bps, s.latency_ms, (1.0 - s.liquidity_score)]
            for s in sub
        ])
        y = np.array([s.sim_return_bps - s.live_return_bps for s in sub])
        reg = ols_regression(X, y)

        # 标准化系数 → 权重占比
        feature_std = np.std(X[:, 1:], axis=0)
        standardized_beta = {
            "spread":    reg.alpha_spread * feature_std[0],
            "slippage":  reg.beta_slippage * feature_std[1],
            "latency":   reg.gamma_latency * feature_std[2],
            "liquidity": reg.delta_liquidity * feature_std[3],
        }
        total_abs = sum(abs(v) for v in standardized_beta.values()) or 1.0
        weights_pct = {k: abs(v) / total_abs * 100.0 for k, v in standardized_beta.items()}

        # 该状态下的差异率
        sim_mean = float(np.mean([s.sim_return_bps for s in sub]))
        live_mean = float(np.mean([s.live_return_bps for s in sub]))
        diff_rate = abs(sim_mean - live_mean) / (abs(sim_mean) + 1e-9)

        # 每个因素的平均成本 (bps)
        avg_costs = {
            "spread":    float(np.mean([s.spread_cost_bps for s in sub])),
            "slippage":  float(np.mean([s.slippage_cost_bps for s in sub])),
            "latency":   float(np.mean([s.latency_cost_bps for s in sub])),
            "liquidity": float(np.mean([s.liquidity_cost_bps for s in sub])),
        }

        # 首要瓶颈
        bottleneck = max(weights_pct, key=weights_pct.get)

        out[regime] = {
            "n_samples": len(sub),
            "r_squared": reg.r_squared,
            "weights_pct": weights_pct,
            "standardized_beta": standardized_beta,
            "diff_rate": diff_rate,
            "sim_mean_bps": sim_mean,
            "live_mean_bps": live_mean,
            "diff_bps": sim_mean - live_mean,
            "avg_costs_bps": avg_costs,
            "total_friction_bps": sum(avg_costs.values()),
            "primary_bottleneck": bottleneck,
            "primary_bottleneck_weight": weights_pct[bottleneck],
            "pass_15pct": diff_rate <= TARGET_DIFF_RATE,
        }
    return out


# ============================================================
# 2. 构建 状态×因素 影响矩阵
# ============================================================
def build_regime_factor_matrix(regime_results: Dict[str, Dict]) -> Dict:
    """4×4 矩阵: 行=状态, 列=因素, 单元格=该状态下该因素的权重占比 (%)。"""
    factors = ["spread", "slippage", "latency", "liquidity"]
    matrix = {}
    for regime, res in regime_results.items():
        matrix[regime] = {f: res["weights_pct"][f] for f in factors}

    # 按因素汇总: 每个因素跨状态的平均权重
    factor_avg = {f: float(np.mean([res["weights_pct"][f] for res in regime_results.values()])) for f in factors}
    # 按状态汇总: 每个状态的总差异率
    regime_diff = {r: res["diff_rate"] for r, res in regime_results.items()}

    # 识别"高差异状态" (差异率最高)
    highest_diff_regime = max(regime_diff, key=regime_diff.get)
    # 识别"跨状态高风险因素" (平均权重最高)
    highest_risk_factor = max(factor_avg, key=factor_avg.get)

    return {
        "matrix": matrix,                      # 4×4 权重占比矩阵
        "factor_avg_weight_pct": factor_avg,   # 跨状态平均权重
        "regime_diff_rate": regime_diff,       # 各状态差异率
        "highest_diff_regime": highest_diff_regime,
        "highest_diff_rate": regime_diff[highest_diff_regime],
        "highest_risk_factor": highest_risk_factor,
        "highest_risk_factor_weight": factor_avg[highest_risk_factor],
    }


# ============================================================
# 3. 状态内交易对差异分布
# ============================================================
def regime_symbol_breakdown(samples: List[TradeRecord]) -> List[Dict]:
    """每个 (regime, symbol) 的差异率, 用于识别状态内的 hot spot。"""
    out = []
    for regime in REGIME_FACTORS:
        for symbol in SYMBOLS:
            cell = [s for s in samples if s.regime == regime and s.symbol == symbol]
            if not cell:
                continue
            sim_mean = float(np.mean([s.sim_return_bps for s in cell]))
            live_mean = float(np.mean([s.live_return_bps for s in cell]))
            diff_rate = abs(sim_mean - live_mean) / (abs(sim_mean) + 1e-9)
            out.append({
                "regime": regime,
                "symbol": symbol,
                "sim_bps": sim_mean,
                "live_bps": live_mean,
                "diff_bps": sim_mean - live_mean,
                "diff_rate": diff_rate,
                "pass_15pct": diff_rate <= TARGET_DIFF_RATE,
                "total_friction_bps": float(np.mean([
                    s.spread_cost_bps + s.slippage_cost_bps + s.latency_cost_bps + s.liquidity_cost_bps
                    for s in cell
                ])),
            })
    return out


# ============================================================
# 4. 状态优先级排序 (改善 ROI)
# ============================================================
def prioritize_regimes(regime_results: Dict[str, Dict], matrix: Dict) -> List[Dict]:
    """按"改善 ROI"对状态排序: ROI = 差异率 × 可建模占比 (R²)。
    高 ROI 状态应优先优化, 因为它差异大且可通过建模降低。"""
    priorities = []
    for regime, res in regime_results.items():
        roi = res["diff_rate"] * res["r_squared"]
        priorities.append({
            "regime": regime,
            "diff_rate": res["diff_rate"],
            "r_squared": res["r_squared"],
            "improvement_roi": roi,
            "primary_bottleneck": res["primary_bottleneck"],
            "primary_bottleneck_weight": res["primary_bottleneck_weight"],
            "total_friction_bps": res["total_friction_bps"],
            "pass_15pct": res["pass_15pct"],
            "recommendation": _regime_recommendation(regime, res),
        })
    priorities.sort(key=lambda x: -x["improvement_roi"])
    return priorities


def _regime_recommendation(regime: str, res: Dict) -> str:
    bn = res["primary_bottleneck"]
    if regime == "HIGH_VOL":
        return f"HIGH_VOL 状态差异最大 ({res['diff_rate']*100:.1f}%): 暂停交易或大幅降仓, 因 {bn} 在高波动下放大 {res['primary_bottleneck_weight']/50:.1f}x"
    if regime == "RANGING":
        return f"RANGING 状态: 均值回归策略敏感于 {bn}, 改用限价单减少 {bn} 损失"
    if regime == "TRENDING":
        return f"TRENDING 状态: 趋势延续时 {bn} 影响小, 可保持仓位, 但需注意入场时 {bn} 损失"
    if regime == "LOW_VOL":
        return f"LOW_VOL 状态: 差异最小, {bn} 影响有限, 可正常交易"
    return f"优化 {bn} ({res['primary_bottleneck_weight']:.1f}%)"


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("v515 状态分割分析 (Agent-6 第二交付)")
    print("=" * 70)
    print(f"实盘延迟: {LIVE_LATENCY_MS}ms")
    print(f"覆盖: {len(SYMBOLS)} 交易对 × {len(REGIME_FACTORS)} 行情条件")
    print()

    sim_baseline = load_sim_baseline()
    print(f"[1] 模拟基线: {sim_baseline}")

    print(f"[2] 生成样本 (naive 场景, 全 4 状态)...")
    samples = generate_samples(sim_baseline, n_per_cell=200, seed=RANDOM_SEED, scenario="naive")
    print(f"    总样本数: {len(samples)}")

    print(f"[3] 按状态分割回归...")
    regime_results = regime_split_regression(samples)
    for r, res in regime_results.items():
        print(f"    {r:10s}: diff={res['diff_rate']*100:5.1f}%  R²={res['r_squared']:.3f}  瓶颈={res['primary_bottleneck']:10s} ({res['primary_bottleneck_weight']:4.1f}%)")

    print(f"[4] 构建 状态×因素 影响矩阵...")
    matrix = build_regime_factor_matrix(regime_results)
    print(f"    最高差异状态: {matrix['highest_diff_regime']} ({matrix['highest_diff_rate']*100:.1f}%)")
    print(f"    跨状态高风险因素: {matrix['highest_risk_factor']} ({matrix['highest_risk_factor_weight']:.1f}%)")

    print(f"[5] 状态×交易对细分...")
    breakdown = regime_symbol_breakdown(samples)

    print(f"[6] 状态改善优先级...")
    priorities = prioritize_regimes(regime_results, matrix)
    for p in priorities:
        print(f"    {p['regime']:10s}: ROI={p['improvement_roi']:.3f}  diff={p['diff_rate']*100:5.1f}%")

    # 输出 JSON
    result = {
        "report_type": "v515_regime_split_analysis",
        "agent_id": "agent_6_diff_model",
        "version": "5.15",
        "timestamp": time.time(),
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "live_latency_ms": LIVE_LATENCY_MS,
            "n_symbols": len(SYMBOLS),
            "n_regimes": len(REGIME_FACTORS),
            "n_samples_total": len(samples),
            "target_diff_rate": TARGET_DIFF_RATE,
        },
        "regime_results": regime_results,
        "regime_factor_matrix": matrix,
        "regime_symbol_breakdown": breakdown,
        "priorities": priorities,
        "answers": {
            "highest_diff_regime": matrix["highest_diff_regime"],
            "highest_diff_rate_pct": matrix["highest_diff_rate"] * 100.0,
            "highest_risk_factor": matrix["highest_risk_factor"],
            "highest_risk_factor_weight_pct": matrix["highest_risk_factor_weight"],
            "is_latency_bottleneck_in_high_vol": (
                regime_results.get("HIGH_VOL", {}).get("primary_bottleneck") == "latency"
            ),
            "does_high_vol_dominate": matrix["highest_diff_regime"] == "HIGH_VOL",
        },
    }

    RESULT_JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] 结果写入: {RESULT_JSON_PATH}")

    md = _build_markdown_report(result)
    REPORT_MD_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] 报告写入: {REPORT_MD_PATH}")
    return result


def _build_markdown_report(r: Dict) -> str:
    lines = []
    lines.append("# v515 状态分割分析报告 (Agent-6)\n")
    lines.append(f"> 生成时间: {r['timestamp_human']}  |  Agent-6  |  版本 v515\n")
    lines.append("---\n")

    lines.append("## 一、核心问题回答\n")
    ans = r["answers"]
    lines.append(f"- **差异最大的状态**: `{ans['highest_diff_regime']}` (差异率 {ans['highest_diff_rate_pct']:.1f}%)")
    lines.append(f"- **跨状态高风险因素**: `{ans['highest_risk_factor']}` (平均权重 {ans['highest_risk_factor_weight_pct']:.1f}%)")
    lines.append(f"- **HIGH_VOL 是否主导差异**: {'是' if ans['does_high_vol_dominate'] else '否'}")
    lines.append(f"- **HIGH_VOL 中延迟是否为瓶颈**: {'是' if ans['is_latency_bottleneck_in_high_vol'] else '否'}\n")

    lines.append("## 二、状态 × 因素 影响矩阵 (4×4)\n")
    lines.append("> 单元格 = 该状态下该因素对差异的贡献权重 (%)，行和=100%\n")
    matrix = r["regime_factor_matrix"]["matrix"]
    factors = ["spread", "slippage", "latency", "liquidity"]
    lines.append("| 状态 \\ 因素 | " + " | ".join(factors) + " | 差异率 | R² | 瓶颈 |")
    lines.append("|" + "------|" * (len(factors) + 4))
    for regime, row in matrix.items():
        rr = r["regime_results"][regime]
        bn = rr["primary_bottleneck"]
        mark = "✅" if rr["pass_15pct"] else "❌"
        cells = [f"{row[f]:.1f}%" for f in factors]
        cells.append(f"{mark} {rr['diff_rate']*100:.1f}%")
        cells.append(f"{rr['r_squared']:.3f}")
        cells.append(f"`{bn}`")
        lines.append(f"| **{regime}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### 跨状态平均权重 (识别全局瓶颈)\n")
    favg = r["regime_factor_matrix"]["factor_avg_weight_pct"]
    lines.append("| 因素 | 平均权重 |")
    lines.append("|------|---------|")
    for f in sorted(favg, key=lambda x: -favg[x]):
        lines.append(f"| `{f}` | {favg[f]:.1f}% |")
    lines.append("")

    lines.append("## 三、各状态详细分析\n")
    for regime, res in r["regime_results"].items():
        mark = "✅" if res["pass_15pct"] else "❌"
        lines.append(f"### {regime}  {mark}")
        lines.append(f"- 样本数: {res['n_samples']}")
        lines.append(f"- 模拟均值: {res['sim_mean_bps']:.2f} bps")
        lines.append(f"- 实盘均值: {res['live_mean_bps']:.2f} bps")
        lines.append(f"- 差异 (sim-live): {res['diff_bps']:.2f} bps")
        lines.append(f"- **差异率: {res['diff_rate']*100:.1f}%** (目标 ≤15%)")
        lines.append(f"- R² (系统性可建模占比): {res['r_squared']*100:.1f}%")
        lines.append(f"- 总摩擦成本: {res['total_friction_bps']:.2f} bps")
        lines.append(f"- **首要瓶颈**: `{res['primary_bottleneck']}` ({res['primary_bottleneck_weight']:.1f}%)")
        lines.append(f"- 各因素平均成本 (bps):")
        for f, c in res["avg_costs_bps"].items():
            lines.append(f"  - {f}: {c:.2f} bps")
        lines.append("")

    lines.append("## 四、状态改善优先级 (按 ROI 排序)\n")
    lines.append("> ROI = 差异率 × R²。高 ROI 状态应优先优化 (差异大且可建模降低)。\n")
    lines.append("| 优先级 | 状态 | 差异率 | R² | ROI | 瓶颈 | 建议 |")
    lines.append("|--------|------|--------|----|-----|------|------|")
    for i, p in enumerate(r["priorities"], 1):
        mark = "✅" if p["pass_15pct"] else "❌"
        lines.append(f"| {i} | {p['regime']} | {mark} {p['diff_rate']*100:.1f}% | {p['r_squared']:.3f} | {p['improvement_roi']:.3f} | `{p['primary_bottleneck']}` | {p['recommendation']} |")
    lines.append("")

    lines.append("## 五、状态 × 交易对 细分 (识别 hot spot)\n")
    lines.append("| 状态 | 交易对 | Sim(bps) | Live(bps) | Diff% | Pass |")
    lines.append("|------|--------|----------|-----------|-------|------|")
    for b in r["regime_symbol_breakdown"]:
        mark = "✅" if b["pass_15pct"] else "❌"
        lines.append(f"| {b['regime']} | {b['symbol']} | {b['sim_bps']:.2f} | {b['live_bps']:.2f} | {b['diff_rate']*100:.1f}% | {mark} |")
    lines.append("")

    lines.append("## 六、关键发现与建议\n")
    pri = r["priorities"]
    top = pri[0]
    lines.append(f"1. **优先优化 `{top['regime']}` 状态**: 差异率 {top['diff_rate']*100:.1f}%, ROI {top['improvement_roi']:.3f}, 瓶颈 `{top['primary_bottleneck']}`")
    if len(pri) > 1:
        second = pri[1]
        lines.append(f"2. **次优 `{second['regime']}` 状态**: 差异率 {second['diff_rate']*100:.1f}%, 瓶颈 `{second['primary_bottleneck']}`")
    lines.append(f"3. **全局高风险因素 `{ans['highest_risk_factor']}`**: 跨状态平均权重 {ans['highest_risk_factor_weight_pct']:.1f}%, 应作为基础设施层优化重点")
    if ans["does_high_vol_dominate"]:
        lines.append(f"4. **HIGH_VOL 状态差异最大** ({ans['highest_diff_rate_pct']:.1f}%): 建议在 HIGH_VOL 状态下暂停交易或大幅降仓, 这是降低整体差异率最有效的单一手段")
    lines.append("")
    lines.append("---")
    lines.append("\n*本报告由 Agent-6 (模拟vs实盘差异量化模型 — 状态分割子模块) 自动生成。*")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
