# v514 多智能体协作接口规范
## — 8 Agent 协同进化标准协议

> **版本**: v514.1
> **生成时间**: 2026-06-28
> **核心协调**: Agent-8 (主 Agent)

---

## 一、8 智能体任务矩阵 (RACI)

| Agent | 角色 | 任务 | 依赖 | 交付物 |
|-------|------|------|------|--------|
| **Agent-1** | 数据管道 | WebSocket长连接+延迟优化 | 无 | _v514_ws_pipeline.py |
| **Agent-2** | 策略组合 | 动量+均值回归+套利框架 | Agent-1数据 | _v514_multi_strategy.py |
| **Agent-3** | 参数分析 | SHAP+PCA+显著性检验 | v503-v506数据 | _v514_shap_analysis.json |
| **Agent-4** | 冗余识别 | 静态分析+运行时监控 | v503-v511代码 | _v514_redundancy_report.json |
| **Agent-5** | 高保真仿真 | 滑点/流动性/延迟建模 | Agent-1数据 | _v514_hifi_simulator.py |
| **Agent-6** | 差异量化 | 模拟vs实盘差异模型 | Agent-5仿真 | _v514_sim_live_diff.json |
| **Agent-7** | 进化引擎 | 多目标GA+Walk-Forward | Agent-2策略 | _v514_evolution_engine.py |
| **Agent-8** | 核心协调 | 任务分配+进度同步+质量把关 | 全部 | _v514_coordination_report.md |

---

## 二、标准化数据交换协议

### 2.1 状态文件格式 (每个 Agent)

路径: `_v514_multi_agent/agent_{N}_{name}/status.json`

```json
{
  "agent_id": "agent_1_pipeline",
  "role": "数据管道优化",
  "status": "pending|in_progress|blocked|completed",
  "progress_pct": 0,
  "started_at": null,
  "updated_at": null,
  "completed_at": null,
  "deliverables": [],
  "blockers": [],
  "dependencies": [],
  "metrics": {}
}
```

### 2.2 数据交换格式

路径: `_v514_multi_agent/data_exchange/{from}_{to}_{datatype}.json`

```json
{
  "from_agent": "agent_1_pipeline",
  "to_agent": "agent_2_portfolio",
  "data_type": "kline_cache",
  "timestamp": 1782658000,
  "schema_version": "1.0",
  "checksum": "sha256:...",
  "payload": {}
}
```

### 2.3 进度同步协议

- **频率**: 每 5 分钟更新 status.json
- **格式**: JSON, UTF-8, 无 BOM
- **校验**: 写入后立即 json.loads 验证
- **冲突处理**: 最后写入者胜, 但记录冲突日志

### 2.4 问题跟踪

路径: `_v514_multi_agent/issues.log`

格式: `[timestamp] [agent_id] [severity: INFO|WARN|BLOCK] message`

---

## 三、性能指标基线 (用户硬指标)

### 3.1 模拟环境

| 指标 | 目标 | v505基线 | 差距 | 解决方向 |
|------|------|---------|------|---------|
| 年化收益率 | ≥30% | ~15% | -15% | 多策略组合 |
| 最大回撤 | ≤15% | 8.5% | ✅ 已达 | - |
| 胜率 | ≥55% | 41.7% | -13% | 加入均值回归 |
| 夏普比率 | ≥1.5 | 1.186 | -0.3 | 多策略分散 |
| 交易数 | ≥30 | 12 | -18 | 多策略增加频率 |
| 月亏概率 | ≤5% | 25% | -20% | 套利平滑 |

### 3.2 实盘环境

| 指标 | 目标 | v505基线 | 差距 |
|------|------|---------|------|
| 月度亏损概率 | ≤5% | 25% | -20% |
| 单次最大亏损 | ≤2% | ~2.3% | -0.3% |
| 连续亏损次数 | ≤3 | 5 | +2 |
| 延迟 | ≤100ms | 1056ms | -956ms |

### 3.3 延迟指标诚实评估

⚠️ **100ms 延迟目标在国内物理不可达**:
- Binance API 国内被 GFW 封锁 (v508 已证)
- Gate.io REST 1056ms 是国内最佳
- Gate.io WebSocket 预期 200-500ms (待 Agent-1 验证)
- **替代方案**: 让策略对延迟不敏感 (限价单+长持仓)

---

## 四、协作流程

### 4.1 启动顺序

```
T0: Agent-8 建立 framework (本步骤)
T1: 并行启动 Agent-1, Agent-3, Agent-4 (无依赖)
T2: Agent-1 完成后启动 Agent-2, Agent-5
T3: Agent-5 完成后启动 Agent-6
T4: Agent-2 完成后启动 Agent-7
T5: Agent-8 整合所有结果 → v515
```

### 4.2 质量门槛

每个 Agent 交付前必须通过:
- L1: 文件存在 + 非空
- L2: 语法/格式有效
- L3: 交叉引用一致
- L4: 无占位符/空章节
- L6: 功能可用 (实际执行验证)

### 4.3 应急回滚

- 任何 Agent 阻塞 > 30 分钟 → Agent-8 介入
- 任何交付物 L1-L4 失败 → 退回重做
- 任何指标退化 → 记录到错误知识库

---

## 五、用户铁律执行矩阵

| 铁律 | 执行方式 | 验证 |
|------|---------|------|
| 杜绝模拟牛逼实盘亏钱 | v513 CONDITIONAL GO 已撤回 | ✅ |
| 未达指标不部署 | 7关卡全部BLOCKED直到达标 | ✅ |
| 多维度深度分析 | 8 Agent 并行覆盖所有维度 | ✅ |
| 知识储备利用 | Agent-2 整合动量+均值回归+套利 | ✅ |
| 进化体系 | Agent-7 多目标GA+Walk-Forward | ✅ |
| 严苛标准不妥协 | 性能指标硬编码,不放松 | ✅ |
