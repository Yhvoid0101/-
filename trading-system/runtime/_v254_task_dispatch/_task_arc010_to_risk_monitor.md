# 任务派发: ARC-010 → RISK-MONITOR

## 派发方
LIVE-DATA-CONNECTOR (Agent 4)

## 接收方
RISK-MONITOR (Agent 5)

## 派发时间
2026-06-28 02:18:10

## 任务背景
v2.53 差异模型关键发现:
- BTC 4h: edge=0.198%, cost=0.026%, diff=9.18% ✅ PASS
- BTC 1h: edge=0.075%, cost=0.026%, diff=23.88% ❌
- SOL 4h: edge=-0.084%, cost=0.070%, diff=85.52% ❌
- AVAX 4h: edge=0.359%, cost=0.110%, diff=45.49% ❌

规律: diff = cost/edge × 100. 达标需 edge/cost ≥ 6.67×.
高周期 (4h) + 高流动性 (BTC) = 大 edge + 小 cost = 低 diff ✅.

## 任务目标
制定实盘配置策略, 优先部署高 edge/cost 比率的交易对+周期组合.

## 具体任务
1. 实盘部署配置 (优先级排序):
   - Tier 1 (立即部署): BTC 4h (已验证 diff=9.18%)
   - Tier 2 (谨慎部署): ETH 4h, BNB 4h (edge>0, diff 30-67%)
   - Tier 3 (暂不部署): SOL/AVAX/ADA/DOT 4h (edge<0 或 diff>80%)
   - Tier 4 (禁止部署): 任何交易对 1h (edge 太小, diff 必然 >20%)

2. 实盘风控规则:
   - 单笔交易最大仓位: 2% (与 v2.48 一致)
   - 单笔最大亏损: 2% (与 v2.51 HARD_CONSTRAINTS 一致)
   - 月度最大亏损: 6% (v2.48 TIER_4 熔断)
   - 连续亏损≥3次: 触发 5 bar 冷静期 (v2.52 ARC-006)

3. 实盘监控指标 (实时上报 INTRADAY-GUARD):
   - 每笔 sim_trade_pnl vs live_trade_pnl (差异率实时计算)
   - 累计 edge/cost 比率 (低于 6.67× 触发 WARN)
   - 月度 PnL 进度 (接近 -6% 触发 BLOCK)

4. 输出: _v257_live_config_optimization.py + _v257_live_config.json

## 验收标准
- 实盘配置 JSON 可被 v2.52 IntegratedTrader 直接加载
- 风控规则与 v2.48/v2.49/v2.51 兼容
- 监控指标可实时计算 (延迟≤1s)

## 协作要求
- 与 STRATEGY-EVOLVER 的 ARC-008/009 任务并行
- 配置变更需通知 INTRADAY-GUARD (Agent 1)
- 实盘上线前需 LIVE-DATA-CONNECTOR (Agent 4) 复核差异模型
