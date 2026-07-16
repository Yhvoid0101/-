# 📊 深度参数分析报告

**生成时间**: 2026-06-30T08:56:12.643449
**数据样本量**: 45 笔交易
**币种数**: 9 个
**置信水平**: 95% | **显著性**: P<0.05

## 1. 样本概览

| 指标 | 值 |
|------|-----|
| 总交易数 | 45 |
| 盈利交易 | 19 (42.2%) |
| 亏损交易 | 26 (57.8%) |
| 币种数 | 9 |

## 2. Pearson相关性分析 (各特征 vs PnL)

| 排名 | 特征 | 相关系数 | P值 | 显著性 | 方向 | 强度 |
|------|------|----------|-----|--------|------|------|
| 1 | atr_rank | +nan | 0.0000 | ✅ P<0.05 | negative | weak |
| 2 | vol_ratio | +nan | 0.0000 | ✅ P<0.05 | negative | weak |
| 3 | bb_width_q | +nan | 0.0000 | ✅ P<0.05 | negative | weak |
| 4 | trend_strength | +nan | 0.0000 | ✅ P<0.05 | negative | weak |
| 5 | adx_hurst_interaction | +0.5561 | 0.0000 | ✅ P<0.05 | positive | strong |
| 6 | bars_held | +0.5335 | 0.0000 | ✅ P<0.05 | positive | strong |
| 7 | adx | +0.4922 | 0.0002 | ✅ P<0.05 | positive | strong |
| 8 | risk_mult | +0.4694 | 0.0005 | ✅ P<0.05 | positive | strong |
| 9 | actual_risk | +0.4694 | 0.0005 | ✅ P<0.05 | positive | strong |
| 10 | hurst | +0.3651 | 0.0101 | ✅ P<0.05 | positive | strong |
| 11 | score | -0.0956 | 0.5289 | ❌ | negative | weak |
| 12 | score_squared | -0.0956 | 0.5289 | ❌ | negative | weak |
| 13 | trend_encoded | +nan | 0.0000 | ✅ P<0.05 | negative | weak |

## 3. 胜率组 vs 亏损组 差异检验 (Welch's t-test)

| 特征 | 盈利组均值 | 亏损组均值 | 差异 | t值 | P值 | 显著性 |
|------|-----------|-----------|------|-----|-----|--------|
| bars_held | 45.5789 | 11.4231 | +34.1559 | +3.119 | 0.0060 | ✅ P<0.05 |
| adx | 26.7832 | 16.7125 | +10.0707 | +3.558 | 0.0017 | ✅ P<0.05 |
| adx_hurst_interaction | 13.1264 | 7.2552 | +5.8712 | +3.994 | 0.0006 | ✅ P<0.05 |
| risk_mult | 0.7706 | 0.3758 | +0.3948 | +3.113 | 0.0045 | ✅ P<0.05 |
| hurst | 0.4908 | 0.4310 | +0.0598 | +2.465 | 0.0137 | ✅ P<0.05 |
| score_squared | 10.1053 | 10.0769 | +0.0283 | +0.036 | 0.9712 | ❌ |
| actual_risk | 0.0139 | 0.0068 | +0.0071 | +3.113 | 0.0045 | ✅ P<0.05 |
| score | 3.1579 | 3.1538 | +0.0040 | +0.036 | 0.9712 | ❌ |
| atr_rank | 0.0000 | 0.0000 | +0.0000 | +0.000 | 1.0000 | ❌ |
| vol_ratio | 0.0000 | 0.0000 | +0.0000 | +0.000 | 1.0000 | ❌ |
| bb_width_q | 0.0000 | 0.0000 | +0.0000 | +0.000 | 1.0000 | ❌ |
| trend_strength | 0.0000 | 0.0000 | +0.0000 | +0.000 | 1.0000 | ❌ |
| trend_encoded | -1.0000 | -1.0000 | +0.0000 | +0.000 | 1.0000 | ❌ |

## 4. 随机森林特征重要性

**方法**: RandomForestRegressor (n=100, max_depth=5)
**R²**: 0.8922
**样本数**: 45 | **特征数**: 13

| 排名 | 特征 | 重要性 | 占比 |
|------|------|--------|------|
| 10 | bars_held | 0.3231 | 32.3% |
| 11 | adx_hurst_interaction | 0.2354 | 23.5% |
| 1 | adx | 0.1925 | 19.2% |
| 2 | hurst | 0.0952 | 9.5% |
| 7 | risk_mult | 0.0703 | 7.0% |
| 8 | actual_risk | 0.0606 | 6.1% |
| 12 | score_squared | 0.0127 | 1.3% |
| 3 | score | 0.0103 | 1.0% |
| 13 | trend_encoded | 0.0000 | 0.0% |
| 9 | trend_strength | 0.0000 | 0.0% |
| 6 | bb_width_q | 0.0000 | 0.0% |
| 5 | vol_ratio | 0.0000 | 0.0% |
| 4 | atr_rank | 0.0000 | 0.0% |

## 5. SHAP值分析

**方法**: SHAP TreeExplainer

| 排名 | 特征 | 平均|SHAP| | 方向 |
|------|------|----------|------|
| 10 | bars_held | 0.003336 | positive |
| 11 | adx_hurst_interaction | 0.001917 | negative |
| 1 | adx | 0.001273 | negative |
| 7 | risk_mult | 0.000724 | negative |
| 8 | actual_risk | 0.000467 | positive |
| 2 | hurst | 0.000432 | positive |
| 3 | score | 0.000195 | positive |
| 12 | score_squared | 0.000190 | positive |
| 13 | trend_encoded | 0.000000 | negative |
| 9 | trend_strength | 0.000000 | negative |
| 6 | bb_width_q | 0.000000 | negative |
| 5 | vol_ratio | 0.000000 | negative |
| 4 | atr_rank | 0.000000 | negative |

## 6. ADX × Hurst 交叉分析

| ADX | Hurst | 交易数 | 胜率 | 平均PnL | 总PnL |
|-----|-------|--------|------|---------|-------|
| good | high | 2 | 50.0% | +0.409% | +0.82% |
| good | low | 2 | 100.0% | +1.141% | +2.28% |
| moderate | high | 2 | 50.0% | +0.498% | +1.00% |
| moderate | low | 9 | 11.1% | -0.220% | -1.98% |
| moderate | neutral | 2 | 0.0% | -0.259% | -0.52% |
| strong | high | 1 | 100.0% | +1.660% | +1.66% |
| strong | low | 7 | 71.4% | +0.730% | +5.11% |
| strong | neutral | 4 | 100.0% | +1.379% | +5.51% |
| weak | high | 2 | 50.0% | +0.368% | +0.74% |
| weak | low | 12 | 16.7% | -0.142% | -1.71% |
| weak | neutral | 2 | 50.0% | +0.743% | +1.49% |

## 7. 关键发现与可操作建议

### 7.1 最重要的特征 (Top 5)
- **bars_held**: 重要性=32.3%
- **adx_hurst_interaction**: 重要性=23.5%
- **adx**: 重要性=19.2%
- **hurst**: 重要性=9.5%
- **risk_mult**: 重要性=7.0%

### 7.2 显著相关特征 (P<0.05)
- **atr_rank**: r=+nan, P=0.0000
- **vol_ratio**: r=+nan, P=0.0000
- **bb_width_q**: r=+nan, P=0.0000
- **trend_strength**: r=+nan, P=0.0000
- **adx_hurst_interaction**: r=+0.5561, P=0.0000

### 7.3 显著区分盈利/亏损的特征 (P<0.05)
- **bars_held**: 盈利组=45.5789 vs 亏损组=11.4231, P=0.0060
- **adx**: 盈利组=26.7832 vs 亏损组=16.7125, P=0.0017
- **adx_hurst_interaction**: 盈利组=13.1264 vs 亏损组=7.2552, P=0.0006
- **risk_mult**: 盈利组=0.7706 vs 亏损组=0.3758, P=0.0045
- **hurst**: 盈利组=0.4908 vs 亏损组=0.4310, P=0.0137

### 7.4 v583设计建议

基于以上分析, v583应优先考虑:

1. **优化bars_held参数**: 基于重要性排名调整
2. **优化adx_hurst_interaction参数**: 基于重要性排名调整
3. **强化ADX过滤**: ADX是最重要的特征, 考虑ADX≥20硬过滤 (实证: 16笔, PnL+15.39%)

---
*报告由深度参数分析系统自动生成 | 2026-06-30 08:56:12*