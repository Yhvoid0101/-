# 任务派发: ARC-008 → STRATEGY-EVOLVER

## 派发方
LIVE-DATA-CONNECTOR (Agent 4)

## 接收方
STRATEGY-EVOLVER (Agent 6)

## 派发时间
2026-06-28 02:18:10

## 任务背景
v2.53 差异模型发现: 当前策略在 RANGING 行情下每笔 edge≈0.1%, 市场成本≈0.08%,
edge/cost 比率仅 1.25×, 导致差异率 71-96% (目标≤15%).
BTC 4h 已达标 (edge=0.198%, cost=0.026%, ratio=7.6×, diff=9.18% ✅),
但其他交易对/周期未达标.

## 任务目标
进化出 edge/cost ≥ 6.67× 的策略 (使差异率≤15%).

## 任务边界 (避免与 LOSS-HUNTER 工作重复)
- LOSS-HUNTER 负责: 亏损根因分析 (ARC-001~006 已完成)
- STRATEGY-EVOLVER 负责: 策略进化优化 (本次 ARC-008)
- 本任务不修改 v2.48-v2.51 模块, 只在 v2.50 MultiStrategyPortfolio 之上进化

## 具体任务
1. 在 v2.52 IntegratedTrader 基础上, 跑 500 轮完整进化 (50代×10agent)
2. 适应度函数 (v2.51) 增加 edge/cost 比率硬约束:
   - 新硬约束: edge/cost_ratio ≥ 6.67 (BLOCK)
   - 计算: ratio = mean(sim_trade_pnl) / total_cost_pct
3. 进化目标:
   - 平均差异率 ≤15% (跨10交易对×4行情条件)
   - 月度亏损概率 ≤5%
   - 最差窗口胜率 ≥40%
   - 最大连续亏损 ≤3
4. 进化后输出: _v255_strategy_edge_evolution.py + _v255_evolution_report.json

## 验收标准 (L1-L8 八层验证)
- L1: 文件存在且非空
- L2: Python 语法正确, 可 import
- L3: 引用 v2.48-v2.51 模块路径正确
- L4: 无占位符, 有实际进化逻辑
- L5: 与 Gate.io Provider 集成可运行
- L6: 实际跑完 500 轮进化, 产出有效 JSON 报告
- L7: 验证报告可独立复现
- L8: 错误知识库记录新发现

## 风险提示
- 不要为达标而调大 SL/TP 让单笔 edge 变大 → 那是过拟合
- 真正修复: 通过策略组合+regime 滤波, 自然产生高 edge 信号
- 用户铁律: "不要根据现有的历史数据去为了数字好看去过拟合, 而是真正的找到根因, 从而根治!"

## 协作要求
- 进度报告格式: 每代结束后输出 fitness, best_gene, KPI 通过率
- 异常情况: 立即通知 INTRADAY-GUARD (Agent 1)
- 完成后: 通知 LIVE-DATA-CONNECTOR (Agent 4) 重跑 v2.53 差异模型验证
