# v515 状态分割分析报告 (Agent-6)

> 生成时间: 2026-06-28 23:59:28  |  Agent-6  |  版本 v515

---

## 一、核心问题回答

- **差异最大的状态**: `LOW_VOL` (差异率 606.6%)
- **跨状态高风险因素**: `spread` (平均权重 73.9%)
- **HIGH_VOL 是否主导差异**: 否
- **HIGH_VOL 中延迟是否为瓶颈**: 否

## 二、状态 × 因素 影响矩阵 (4×4)

> 单元格 = 该状态下该因素对差异的贡献权重 (%)，行和=100%

| 状态 \ 因素 | spread | slippage | latency | liquidity | 差异率 | R² | 瓶颈 |
|------|------|------|------|------|------|------|------|
| **TRENDING** | 69.1% | 17.9% | 0.6% | 12.4% | ❌ 210.6% | 0.724 | `spread` |
| **RANGING** | 78.5% | 12.6% | 2.2% | 6.7% | ❌ 349.7% | 0.735 | `spread` |
| **HIGH_VOL** | 75.3% | 12.2% | 11.1% | 1.4% | ❌ 187.6% | 0.721 | `spread` |
| **LOW_VOL** | 72.7% | 15.2% | 1.2% | 10.9% | ❌ 606.6% | 0.744 | `spread` |

### 跨状态平均权重 (识别全局瓶颈)

| 因素 | 平均权重 |
|------|---------|
| `spread` | 73.9% |
| `slippage` | 14.5% |
| `liquidity` | 7.8% |
| `latency` | 3.8% |

## 三、各状态详细分析

### TRENDING  ❌
- 样本数: 2000
- 模拟均值: 7.87 bps
- 实盘均值: -8.71 bps
- 差异 (sim-live): 16.58 bps
- **差异率: 210.6%** (目标 ≤15%)
- R² (系统性可建模占比): 72.4%
- 总摩擦成本: 16.66 bps
- **首要瓶颈**: `spread` (69.1%)
- 各因素平均成本 (bps):
  - spread: 14.88 bps
  - slippage: 0.79 bps
  - latency: 0.82 bps
  - liquidity: 0.18 bps

### RANGING  ❌
- 样本数: 2000
- 模拟均值: 4.69 bps
- 实盘均值: -11.72 bps
- 差异 (sim-live): 16.41 bps
- **差异率: 349.7%** (目标 ≤15%)
- R² (系统性可建模占比): 73.5%
- 总摩擦成本: 16.41 bps
- **首要瓶颈**: `spread` (78.5%)
- 各因素平均成本 (bps):
  - spread: 14.90 bps
  - slippage: 0.46 bps
  - latency: 0.88 bps
  - liquidity: 0.17 bps

### HIGH_VOL  ❌
- 样本数: 2000
- 模拟均值: 12.22 bps
- 实盘均值: -10.70 bps
- 差异 (sim-live): 22.92 bps
- **差异率: 187.6%** (目标 ≤15%)
- R² (系统性可建模占比): 72.1%
- 总摩擦成本: 22.67 bps
- **首要瓶颈**: `spread` (75.3%)
- 各因素平均成本 (bps):
  - spread: 14.89 bps
  - slippage: 2.28 bps
  - latency: 5.26 bps
  - liquidity: 0.23 bps

### LOW_VOL  ❌
- 样本数: 2000
- 模拟均值: 2.53 bps
- 实盘均值: -12.79 bps
- 差异 (sim-live): 15.32 bps
- **差异率: 606.6%** (目标 ≤15%)
- R² (系统性可建模占比): 74.4%
- 总摩擦成本: 15.46 bps
- **首要瓶颈**: `spread` (72.7%)
- 各因素平均成本 (bps):
  - spread: 14.91 bps
  - slippage: 0.26 bps
  - latency: 0.13 bps
  - liquidity: 0.16 bps

## 四、状态改善优先级 (按 ROI 排序)

> ROI = 差异率 × R²。高 ROI 状态应优先优化 (差异大且可建模降低)。

| 优先级 | 状态 | 差异率 | R² | ROI | 瓶颈 | 建议 |
|--------|------|--------|----|-----|------|------|
| 1 | LOW_VOL | ❌ 606.6% | 0.744 | 4.514 | `spread` | LOW_VOL 状态: 差异最小, spread 影响有限, 可正常交易 |
| 2 | RANGING | ❌ 349.7% | 0.735 | 2.569 | `spread` | RANGING 状态: 均值回归策略敏感于 spread, 改用限价单减少 spread 损失 |
| 3 | TRENDING | ❌ 210.6% | 0.724 | 1.524 | `spread` | TRENDING 状态: 趋势延续时 spread 影响小, 可保持仓位, 但需注意入场时 spread 损失 |
| 4 | HIGH_VOL | ❌ 187.6% | 0.721 | 1.353 | `spread` | HIGH_VOL 状态差异最大 (187.6%): 暂停交易或大幅降仓, 因 spread 在高波动下放大 1.5x |

## 五、状态 × 交易对 细分 (识别 hot spot)

| 状态 | 交易对 | Sim(bps) | Live(bps) | Diff% | Pass |
|------|--------|----------|-----------|-------|------|
| TRENDING | BTCUSDT | 7.50 | 1.46 | 80.5% | ❌ |
| TRENDING | ETHUSDT | 8.13 | 1.14 | 86.0% | ❌ |
| TRENDING | SOLUSDT | 8.30 | -5.09 | 161.3% | ❌ |
| TRENDING | BNBUSDT | 7.73 | -1.90 | 124.6% | ❌ |
| TRENDING | XRPUSDT | 8.82 | -8.03 | 191.0% | ❌ |
| TRENDING | ADAUSDT | 7.44 | -12.19 | 263.9% | ❌ |
| TRENDING | AVAXUSDT | 7.61 | -14.03 | 284.4% | ❌ |
| TRENDING | DOTUSDT | 7.36 | -13.03 | 277.1% | ❌ |
| TRENDING | MATICUSDT | 8.31 | -19.23 | 331.4% | ❌ |
| TRENDING | LINKUSDT | 7.54 | -16.22 | 315.2% | ❌ |
| RANGING | BTCUSDT | 4.80 | -0.66 | 113.7% | ❌ |
| RANGING | ETHUSDT | 4.07 | -2.74 | 167.4% | ❌ |
| RANGING | SOLUSDT | 4.84 | -8.61 | 277.8% | ❌ |
| RANGING | BNBUSDT | 5.16 | -4.49 | 186.9% | ❌ |
| RANGING | XRPUSDT | 4.87 | -11.44 | 334.9% | ❌ |
| RANGING | ADAUSDT | 4.25 | -15.47 | 464.1% | ❌ |
| RANGING | AVAXUSDT | 4.92 | -16.84 | 442.4% | ❌ |
| RANGING | DOTUSDT | 4.33 | -15.36 | 454.8% | ❌ |
| RANGING | MATICUSDT | 4.75 | -22.47 | 572.9% | ❌ |
| RANGING | LINKUSDT | 4.94 | -19.10 | 486.3% | ❌ |
| HIGH_VOL | BTCUSDT | 12.89 | 3.81 | 70.5% | ❌ |
| HIGH_VOL | ETHUSDT | 11.08 | 0.17 | 98.5% | ❌ |
| HIGH_VOL | SOLUSDT | 11.84 | -7.65 | 164.6% | ❌ |
| HIGH_VOL | BNBUSDT | 11.63 | -2.74 | 123.6% | ❌ |
| HIGH_VOL | XRPUSDT | 11.21 | -10.94 | 197.6% | ❌ |
| HIGH_VOL | ADAUSDT | 13.16 | -14.06 | 206.8% | ❌ |
| HIGH_VOL | AVAXUSDT | 12.08 | -18.16 | 250.3% | ❌ |
| HIGH_VOL | DOTUSDT | 12.64 | -14.80 | 217.1% | ❌ |
| HIGH_VOL | MATICUSDT | 12.56 | -24.34 | 293.8% | ❌ |
| HIGH_VOL | LINKUSDT | 13.13 | -18.31 | 239.4% | ❌ |
| LOW_VOL | BTCUSDT | 2.01 | -3.12 | 255.4% | ❌ |
| LOW_VOL | ETHUSDT | 2.81 | -3.32 | 218.1% | ❌ |
| LOW_VOL | SOLUSDT | 2.51 | -9.76 | 488.7% | ❌ |
| LOW_VOL | BNBUSDT | 2.63 | -5.70 | 316.4% | ❌ |
| LOW_VOL | XRPUSDT | 2.28 | -13.30 | 682.7% | ❌ |
| LOW_VOL | ADAUSDT | 2.42 | -16.52 | 782.4% | ❌ |
| LOW_VOL | AVAXUSDT | 2.16 | -18.19 | 942.9% | ❌ |
| LOW_VOL | DOTUSDT | 2.68 | -15.42 | 676.1% | ❌ |
| LOW_VOL | MATICUSDT | 2.74 | -22.99 | 940.5% | ❌ |
| LOW_VOL | LINKUSDT | 3.02 | -19.60 | 749.5% | ❌ |

## 六、关键发现与建议

1. **优先优化 `LOW_VOL` 状态**: 差异率 606.6%, ROI 4.514, 瓶颈 `spread`
2. **次优 `RANGING` 状态**: 差异率 349.7%, 瓶颈 `spread`
3. **全局高风险因素 `spread`**: 跨状态平均权重 73.9%, 应作为基础设施层优化重点

---

*本报告由 Agent-6 (模拟vs实盘差异量化模型 — 状态分割子模块) 自动生成。*