# Phase 1 性能基线报告

> 生成时间: 2026-07-15 16:00:00
> 测试模式: 零模拟 Shadow（无真实模块，全 BLOCKED 路径）

## 性能指标

| 指标 | 值 | 验收标准 | 结果 |
|------|------|------|------|
| 8层链路平均延迟（BLOCKED） | 0.062 ms | < 10ms | PASS |
| 8层链路 P99 延迟 | 0.161 ms | < 50ms | PASS |
| 事件总线吞吐 | 88556 events/sec | > 1000 | PASS |
| 1000次8层链路峰值内存 | 7.5 KB | < 10MB | PASS |
| 序列化/反序列化 | 17064 ops/sec | > 1000 | PASS |

## 测试矩阵

| 测试 | 内容 | 结果 |
|------|------|------|
| T10-1 | 无模块8层链路 BLOCKED 传播 | PASS |
| T10-2 | L2 NOT_INITIALIZED → 8层传播 | PASS |
| T10-3 | SYNTHETIC 数据8层 BLOCKED 传播 | PASS |
| T10-4 | 事件总线幂等可重放 | PASS |
| T10-5 | 事件日志持久化 | PASS |
| T10-6 | 100事件流 | PASS |
| T11-1 | 8层链路延迟基线 | PASS |
| T11-2 | 事件总线吞吐基线 | PASS |
| T11-3 | 内存基线 | PASS |
| T11-4 | 序列化性能基线 | PASS |

## 零模拟合规说明

- 所有8层 Adapter 无真实模块时返回 BLOCKED(*_NOT_INITIALIZED)
- 不存在返回默认值（NEUTRAL/ALLOW/FILLED）的软降级路径
- BLOCKED 状态从 L1 正确传播到 L8
- 真实数据链路通过测试将在 Phase 2-9 接入真实模块后进行
