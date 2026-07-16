# 任务派发: ARC-009 → STRATEGY-EVOLVER

## 派发方
LIVE-DATA-CONNECTOR (Agent 4)

## 接收方
STRATEGY-EVOLVER (Agent 6)

## 派发时间
2026-06-28 02:18:10

## 任务背景
v2.53 差异模型发现: v2.52 classify_regime 阈值过严.
1000根1h + 1000根4h K线数据 (覆盖42-166天) 全部判定为 RANGING,
但实际市场有明显波动 (SOL/XRP 出现 5-10% 价格变化).
用户要求覆盖 4 种行情条件 (TRENDING/RANGING/HIGH_VOL/LOW_VOL),
目前仅覆盖 1 种.

## 任务目标
改进 regime 分类器, 使其在真实数据上能区分 4 种行情条件.

## 具体任务
1. 分析当前 classify_regime 阈值 (来自 _v252_integrated_evolution.py):
   - atr_pct > 3% → volatile (HIGH_VOL)
   - atr_pct < 0.8% → quiet (LOW_VOL)
   - adx > 25 and ema12 > ema26 → uptrend (TRENDING)
   - adx > 25 and ema12 < ema26 → downtrend (TRENDING)
   - else → ranging (RANGING)

2. 阈值问题:
   - 3% ATR 阈值太高 (大多数 K 线 ATR 在 0.8%-3% 之间)
   - ADX 25 阈值偏高 (大多数 K 线 ADX 在 15-25)

3. 改进方案 (任选其一或组合):
   a) 放宽阈值: ATR 1.5% / 0.5%, ADX 20
   b) 多周期投票: 1h regime + 4h regime + 1d regime 多数表决
   c) 分位数分类: 用历史 ATR/ADX 分位数动态确定阈值
   d) HMM 隐马尔可夫: 用 HMM 学习市场状态转移

4. 输出: _v256_multi_timeframe_regime.py + _v256_regime_report.json

## 验收标准
- 真实数据上能识别出 ≥3 种行情条件 (TRENDING + RANGING + HIGH_VOL 或 LOW_VOL)
- 改进后跑 v2.53 差异模型, 各行情条件都有数据点
- 不引入过拟合 (阈值不能针对特定数据集硬编码)

## 协作要求
- 与 ARC-008 任务可并行执行
- 改进后通知 LIVE-DATA-CONNECTOR (Agent 4) 重跑 v2.53
