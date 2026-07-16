# Phase 3 验收门禁报告

> 生成时间: 2026-07-15 16:57:34
> 阶段: Phase 3 — 第2层市场理解智能体
> 铁律合规: 零模拟、零降级、零纸面、零死角、零跳过、零限制、零错误、零遗漏

## 验收门禁 4/4 PASS

### 门禁1: 全部 Phase 3 单元测试 — PASS

12项测试全部 PASS：
- T1 L2RegimeAnalyzer: 6项（BLOCKED传播/无数据BLOCKED/有效regime/OOD BLOCKED/不足BLOCKED/无泄漏）
- T2 候选评估: 4项（合法SHADOW/不稳定REJECTED/样本不足REJECTED/无经济意义REJECTED）
- T3 OOD/不确定性: 2项（高不确定性处理/证据特征完整）

### 门禁2: 真实 Gate.io → L2 端到端 — PASS

真实 API + L2 端到端结果:
```json
{
  "BTC_USDT": {
    "l1_status": "OK",
    "l2_status": "OK",
    "regime": "ranging",
    "confidence": 0.5,
    "ood_distance": 0.0,
    "uncertainty": 0.35,
    "is_ood": false,
    "is_stable": false,
    "features": [
      "volatility",
      "trend_strength",
      "volume_profile",
      "change_point_prob"
    ]
  },
  "ETH_USDT": {
    "l1_status": "OK",
    "l2_status": "OK",
    "regime": "ranging",
    "confidence": 0.5,
    "ood_distance": 0.4925011409035853,
    "uncertainty": 0.42365856854594736,
    "is_ood": false,
    "is_stable": false,
    "features": [
      "volatility",
      "trend_strength",
      "volume_profile",
      "change_point_prob"
    ]
  }
}
```

### 门禁3: 无标签泄漏 + OOD 阻断 + BLOCKED 传播 — PASS

- 无标签泄漏: 特征只用当前及历史数据，regime 不因未来数据改变
- OOD 阻断: 极端跳变 → BLOCKED(OOD_UNKNOWN), regime=UNKNOWN, 不强行归类
- BLOCKED 传播: 上游 BLOCKED → L2 BLOCKED(UPSTREAM_BLOCKED)

### 门禁4: 性能基线 — PASS

| 指标 | 值 | 验收标准 | 结果 |
|------|------|------|------|
| L2 平均延迟 | 0.009 ms | < 10ms | PASS |
| L2 P99 延迟 | 0.033 ms | < 50ms | PASS |
| 1000次 L2 analyze 峰值内存 | 1.4 KB | < 10MB | PASS |
| L2 吞吐 | 109290 analyzes/sec | > 100 | PASS |

## Phase 3 产出物

### 新增代码
- `sandbox_trading/phase3_l2.py` — L2 市场理解智能体
  - L2RegimeAnalyzer: 契约化 Regime 分析器（OOD/不确定性/基础四分类）
  - RegimeCandidate: 无监督状态发现候选
  - RegimeCandidateEvaluator: 候选评估器（稳定性/经济意义/样本数/跨窗口/未来行为）
  - CandidateStatus: 候选状态枚举

### 测试脚本
- `_test_phase3_l2.py` — 12项测试 PASS
- `_test_phase3_gate.py` — 综合验收门禁

## 零模拟合规

1. **不强行归类未知状态**: PASS — OOD → BLOCKED(OOD_UNKNOWN), regime=UNKNOWN
2. **无标签泄漏**: PASS — 特征只用当前及历史数据
3. **BLOCKED 传播**: PASS — 上游 BLOCKED 正确传播
4. **候选不直接改生产**: PASS — 候选经评估进入 SHADOW，不直接改 regime 定义
5. **证据特征完整**: PASS — volatility/trend_strength/volume_profile/change_point_prob

## 下一步

Phase 3 验收门禁 4/4 PASS，可进入 Phase 4（第3层机会发现智能体）。
