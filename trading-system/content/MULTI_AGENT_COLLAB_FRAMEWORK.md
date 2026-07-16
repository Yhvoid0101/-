# 多智能体协作框架 — Hermes v6 沙盘交易进化系统

> **文档版本**: v1.0 | **创建日期**: 2026-06-29 | **状态**: 活跃
> **适用范围**: Hermes v6 sandbox-trading 技能包的8智能体并行进化

---

## 一、协作架构总览

### 1.1 八智能体角色定义与责任矩阵

> **对齐声明**: 本表角色顺序与 `multi_agent_framework.py` `AgentRole` 枚举 (L69-77) 严格一致。
> 任何角色增删/重命名必须先修改代码 AgentRole, 再同步本表 (代码是单一真相源)。

| Agent ID | 角色名称 (代码枚举) | 核心职责 | 技术栈 | 输出产物 |
|:---:|---|---|---|---|
| **A1** | 核心协调Agent (COORDINATOR) | 任务分配、进度同步、冲突仲裁、综合判定 | Python, Shell, JSON | 进度报告、决策日志 |
| **A2** | 策略研究Agent (STRATEGY_RESEARCHER) | 策略设计、参数优化、回测验证 | Python, NumPy, pandas | 策略代码、回测报告 |
| **A3** | 风控管理Agent (RISK_MANAGER) | 风险指标监控、回撤分析、单亏控制、阶梯降仓 | Python, statistics | 风控报告、告警 |
| **A4** | 数据工程Agent (DATA_ENGINEER) | 数据采集、清洗、存储、延迟优化 | Python, SQL, Redis | 数据管道、质量报告 |
| **A5** | 执行优化Agent (EXECUTION_OPTIMIZER) | 模拟-实盘差异分析、滑点建模、执行优化、Maker/Taker路由 | Python, econometrics | 差异评估模型、执行优化方案 |
| **A6** | 代码审查Agent (CODE_REVIEWER) | 代码质量、安全扫描、重构建议、AST分析 | ast, bandit, radon | 审查报告 |
| **A7** | 测试验证Agent (TEST_VALIDATOR) | 测试用例、回归验证、性能基准、L1-L8验证 | pytest, benchmark | 测试报告、基准数据 |
| **A8** | 知识管理Agent (KNOWLEDGE_MANAGER) | Obsidian笔记、错误知识库、经验沉淀、复盘闭环 | Markdown, JSON | 知识库条目 |

### 1.2 任务边界划分原则

**铁律: 任务重叠率必须 ≤ 5%**

```
策略设计(A2 STRATEGY_RESEARCHER) ──参数──> 数据工程(A4 DATA_ENGINEER) ──特征──> 策略优化(A2)
    │                                              │
    ├──回测──> 风控管理(A3 RISK_MANAGER) ──风险告警──> 核心协调(A1 COORDINATOR)
    │                                              │
    └──代码──> 代码审查(A6 CODE_REVIEWER) ──质量报告──> 核心协调(A1 COORDINATOR)
                          │
执行优化(A5 EXECUTION_OPTIMIZER) ──差异模型──> 策略优化(A2)
                          │
测试验证(A7 TEST_VALIDATOR) ──回归测试──> 核心协调(A1 COORDINATOR)
                          │
知识管理(A8 KNOWLEDGE_MANAGER) <──经验──> 所有Agent
```

### 1.3 责任矩阵 (RACI)

> **列对齐**: A1-A8 列与 1.1 节角色表严格一致 (代码 AgentRole 顺序)。
> R=Responsible(执行) A=Accountable(负责) C=Consulted(咨询) I=Informed(知情)

| 任务 | A1 协调 | A2 策略 | A3 风控 | A4 数据 | A5 执行 | A6 代码 | A7 测试 | A8 知识 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 策略版本迭代 | A | R | C | C | I | I | I | I |
| 数据质量保障 | A | I | C | R | C | I | I | I |
| 参数影响分析 | A | C | C | R | I | I | I | I |
| 风险指标监控 | A | C | R | I | C | I | I | I |
| 模拟实盘对齐 | A | C | C | I | R | I | I | I |
| 代码质量审查 | A | C | I | I | I | R | I | I |
| 测试回归验证 | A | C | I | I | I | C | R | I |
| 错误知识沉淀 | A | C | C | C | C | C | C | R |

---

## 二、标准化信息共享机制

### 2.1 统一进度报告模板

```json
{
  "agent_id": "A2",
  "agent_role": "策略研究Agent (STRATEGY_RESEARCHER)",
  "report_time": "2026-06-29T15:30:00",
  "current_task": "v55过滤放松+仓位提升",
  "progress_pct": 75,
  "status": "in_progress",
  "blockers": [],
  "next_milestone": "v55运行完成→分析结果→决定v56方向",
  "metrics_snapshot": {
    "version": "v55",
    "annualized_return": "TBD",
    "win_rate": "TBD",
    "max_drawdown": "TBD",
    "sharpe_ratio": "TBD",
    "targets_met": "TBD/7"
  },
  "dependencies": ["A4(DATA_ENGINEER)数据就绪", "A4特征报告"],
  "artifacts": ["_v470_v55_filter_relax_position_boost.py"]
}
```

### 2.2 跨智能体问题跟踪

```json
{
  "issue_id": "ISSUE-20260629-001",
  "raised_by": "A2 (STRATEGY_RESEARCHER)",
  "assigned_to": "A4 (DATA_ENGINEER)",
  "severity": "HIGH",
  "category": "参数分析",
  "description": "1d趋势过滤降权后BTC年化下降,需SHAP分析逆势交易的真实alpha贡献",
  "status": "open",
  "created_at": "2026-06-29T15:30:00",
  "due_at": "2026-06-29T18:00:00",
  "related_errors": ["ERR-050"]
}
```

### 2.3 进度同步频率

| 同步类型 | 频率 | 触发条件 | 机制 |
|---|---|---|---|
| **心跳同步** | 每30分钟 | 定时 | 写入 `_agent_heartbeat.json` |
| **里程碑同步** | 每完成一个版本 | v55/v56...完成 | 写入 `_progress_log.md` |
| **阻塞告警** | 立即 | 遇到阻塞 | 写入 `_blockers.json` + 通知A1 |
| **错误同步** | 立即 | 发现新错误 | 写入 `1500-error-knowledge-base.md` |

---

## 三、跨智能体接口规范

### 3.1 数据交换协议

#### 3.1.1 交易数据格式 (统一规范)

```json
{
  "schema_version": "1.0",
  "trade": {
    "trade_id": "XRPUSDT_20260629_001",
    "symbol": "XRPUSDT",
    "strategy": "A_momentum",
    "direction": "long",
    "entry_bar": 12345,
    "exit_bar": 12400,
    "entry_price": 0.5234,
    "exit_price": 0.5412,
    "raw_pnl": 0.0340,
    "net_pnl": 0.0318,
    "position_scale": 0.45,
    "effective_position_scale": 0.225,
    "trend_1d_downweight": 0.5,
    "is_counter_trend": true,
    "kelly_fraction": 0.330,
    "max_pos_scale": 1.8,
    "friction_cost": 0.0022,
    "metadata": {
      "rsi_at_entry": 62.3,
      "vwap_deviation": 0.0012,
      "mtf_trend": "up",
      "cvd_divergence": -0.23,
      "atr_pct": 0.0089
    }
  }
}
```

#### 3.1.2 回测结果格式

```json
{
  "schema_version": "1.0",
  "backtest_result": {
    "version": "v55",
    "symbol": "XRPUSDT",
    "config_hash": "sha256:...",
    "metrics": {
      "n_trades": 60,
      "annualized_return": 0.1740,
      "sharpe_ratio": 1.82,
      "max_drawdown": 0.0344,
      "win_rate": 0.5167,
      "max_consecutive_losses": 2,
      "max_single_loss": -0.0193,
      "monthly_loss_prob": 0.2273
    },
    "targets_met": ["SHP", "DD", "CL", "SL"],
    "targets_count": 4,
    "strategy_breakdown": { ... },
    "timestamp": "2026-06-29T15:30:00"
  }
}
```

### 3.2 接口契约

| 接口 | 提供方 | 消费方 | 数据格式 | 传输频率 | 校验机制 |
|---|---|---|---|---|---|
| **行情数据** | A4 (DATA_ENGINEER) | A2,A3,A5 | klines JSON | 每bar | 数据完整性≥99.99%, 延迟≤100ms |
| **交易记录** | A2 (STRATEGY_RESEARCHER) | A3,A5,A8 | trade JSON | 每笔成交 | pnl校验: abs(sum-pnl) < 1e-6 |
| **特征值** | A4 (DATA_ENGINEER) | A2 | feature vector | 每bar | SHAP值合理性检查 |
| **风险指标** | A3 (RISK_MANAGER) | A1,A2 | risk JSON | 每日 | 7指标全量校验 |
| **差异模型** | A5 (EXECUTION_OPTIMIZER) | A1,A2 | diff JSON | 每版本 | 差异率≤15%阈值 |
| **测试报告** | A7 (TEST_VALIDATOR) | A1,A2 | test JSON | 每次提交 | L1-L8全量验证 |
| **代码审查** | A6 (CODE_REVIEWER) | A1,A2 | review JSON | 每次提交 | AST+安全扫描 |
| **错误知识** | A8 (KNOWLEDGE_MANAGER) | All | ERR-YYYYMMDD-NNN | 实时 | 3轮独立验证 |

### 3.3 数据校验机制

```python
def validate_trade_data(trade: dict) -> tuple[bool, str]:
    """交易数据校验 — 所有Agent必须遵循"""
    required_fields = ["trade_id", "symbol", "strategy", "direction",
                       "entry_bar", "exit_bar", "raw_pnl", "net_pnl",
                       "position_scale", "kelly_fraction"]
    for field in required_fields:
        if field not in trade:
            return False, f"缺失必需字段: {field}"

    # pnl一致性校验
    expected_net = trade["raw_pnl"] - trade.get("friction_cost", 0)
    if abs(expected_net - trade["net_pnl"]) > 1e-6:
        return False, f"pnl不一致: expected={expected_net}, got={trade['net_pnl']}"

    # position_scale范围校验
    if not (0 <= trade["position_scale"] <= 2.5):
        return False, f"position_scale越界: {trade['position_scale']}"

    return True, "OK"
```

---

## 四、冲突解决机制

### 4.1 冲突类型与处理

| 冲突类型 | 示例 | 解决方 | 解决机制 |
|---|---|---|---|
| **参数冲突** | A2要放松RSI, A3认为风险过高 | A1 | 风险优先: 风控一票否决 |
| **资源冲突** | A2和A4同时需要GPU | A1 | 按任务优先级排队 |
| **结论冲突** | A2说策略有效, A5说实盘会失效 | A1 | 启动A6独立审查 |
| **版本冲突** | 两个Agent同时修改同一文件 | A1 | 最后写入者胜+告警 |

### 4.2 升级路径

```
Agent间协商 (5分钟) → A1仲裁 (10分钟) → 框架规则判定 (自动) → 用户介入 (仅终极)
```

---

## 五、当前会话应用实例

### 5.1 v55迭代的协作映射

| 实际工作 | 对应Agent (代码枚举) | 协作内容 |
|---|---|---|
| v55脚本修改 | A2 (STRATEGY_RESEARCHER) | 实现1d降权+RSI放松+仓位提升 |
| L1-L3验证 | A6 (CODE_REVIEWER) | 语法检查+依赖验证 |
| L4-L8深度验证 | A7 (TEST_VALIDATOR) | 内容质量+功能可用性+元验证 |
| v55运行 | A2 (STRATEGY_RESEARCHER) | 回测执行 |
| ERR-050记录 | A8 (KNOWLEDGE_MANAGER) | 错误知识库更新 |
| 风险指标检查 | A3 (RISK_MANAGER) | 7指标达标判定 |
| 模拟-实盘差异 | A5 (EXECUTION_OPTIMIZER) | 标准摩擦模型应用 |

### 5.2 待委派任务

| 任务 | 委派Agent (代码枚举) | 优先级 | 状态 |
|---|---|---|---|
| v55结果SHAP分析 | A4 (DATA_ENGINEER) | HIGH | 待v55完成 |
| v55代码深度审查 | A6 (CODE_REVIEWER) | MEDIUM | 待v55完成 |
| v55风控独立验证 | A3 (RISK_MANAGER) | HIGH | 待v55完成 |
| v55差异模型更新 | A5 (EXECUTION_OPTIMIZER) | MEDIUM | 待v55完成 |
| v55测试回归 | A7 (TEST_VALIDATOR) | MEDIUM | 待v55完成 |
| v55经验沉淀 | A8 (KNOWLEDGE_MANAGER) | LOW | 待v55完成 |

---

## 六、质量门槛

### 6.1 协作质量指标

| 指标 | 目标 | 测量方法 |
|---|:---:|---|
| 任务重叠率 | ≤5% | RACI矩阵审计 |
| 信息同步延迟 | ≤30分钟 | 心跳时间戳差 |
| 接口数据校验通过率 | ≥99.9% | validate_trade_data |
| 冲突解决时长 | ≤30分钟 | 冲突日志时间差 |
| 错误知识复用率 | ≥80% | ERR-ID引用次数 |

### 6.2 协作失败红线

以下任一情况 → 协作失败, 立即升级:
- ❌ 任务重叠率 >10%
- ❌ 信息同步延迟 >2小时
- ❌ 接口数据校验失败率 >1%
- ❌ 同一错误被2个Agent重复犯
- ❌ 冲突未在1小时内解决
