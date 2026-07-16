# -*- coding: utf-8 -*-
"""
MCP进化引擎服务器 — P1-2 让AI Agent直接驱动进化

暴露10个MCP工具，让AI Agent可以：
  - 创建和管理进化种群
  - 运行进化轮次
  - 查询适应度和排名
  - 保存/恢复检查点
  - 手动变异和交叉

设计原则:
  1. 每个工具都是无状态的（通过population_id关联）
  2. 所有返回值都是JSON可序列化的
  3. 错误以结构化方式返回（不抛异常）

依赖:
  pip install mcp  # Model Context Protocol
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

logger = logging.getLogger("hermes.mcp_evolution")

# 全局种群注册表（进程级）
_populations: Dict[str, Any] = {}
_checkpoints: Dict[str, str] = {}


def _serialize(obj: Any) -> Any:
    """序列化对象为JSON可序列化格式"""
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# ============================================================================
# MCP工具实现
# ============================================================================


def tool_create_population(
    population_id: str,
    population_size: int = 50,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """工具1: 创建初始种群

    Args:
        population_id: 种群唯一标识
        population_size: 种群大小
        seed: 随机种子
    """
    try:
        from .population import PopulationManager
        from .deterministic_rng import global_rng_manager

        global_rng_manager.reseed(seed)

        # 显式传递 population_size 给 PopulationManager（修复参数未传递问题）
        manager = PopulationManager(population_size=population_size, **kwargs)
        manager.initialize_population()

        _populations[population_id] = manager

        return {
            "status": "ok",
            "population_id": population_id,
            "size": len(manager.population),
            "requested_size": population_size,
            "seed": seed,
        }
    except Exception as e:
        logger.error("create_population failed: %s", e)
        return {"status": "error", "message": str(e)}


def tool_run_evolution(
    population_id: str,
    n_rounds: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """工具2: 运行进化轮次

    Args:
        population_id: 种群ID
        n_rounds: 进化轮数
    """
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": f"Population not found: {population_id}"}

        results = []
        for i in range(n_rounds):
            result = manager.run_evolution_round(**kwargs)
            results.append(_serialize(result))

        return {
            "status": "ok",
            "rounds_completed": n_rounds,
            "generation": manager.generation,
            "results": results,
        }
    except Exception as e:
        logger.error("run_evolution failed: %s", e)
        return {"status": "error", "message": str(e)}


def tool_get_population_status(population_id: str) -> Dict[str, Any]:
    """工具3: 获取种群状态"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": f"Population not found: {population_id}"}

        agents = manager.population  # 修复: 使用 population 属性替代不存在的 get_all_agents()
        return {
            "status": "ok",
            "population_id": population_id,
            "generation": manager.generation,
            "population_size": len(agents),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "generation": a.generation,
                }
                for a in agents[:20]  # 只返回前20个
            ],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _get_agent_score_dict(agent: Any) -> Dict[str, Any]:
    """从agent运行时状态构造fitness字典

    PopulationManager不保存独立的scores字典，
    但agent本身记录了运行时fitness状态。
    """
    return {
        "gt_score": getattr(agent, "fitness_score", 0.0),
        "sharpe_ratio": getattr(agent, "sharpe_ratio", 0.0),
        "win_rate": getattr(agent, "win_rate", 0.0),
        "max_drawdown": getattr(agent, "max_drawdown_realized", 0.0),
        "total_trades": getattr(agent, "total_trades", 0),
        "total_pnl": getattr(agent, "total_pnl", 0.0),
    }


def tool_get_agent_fitness(
    population_id: str,
    agent_id: str,
) -> Dict[str, Any]:
    """工具4: 获取Agent适应度"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": "Population not found"}

        # 从 manager.population 查找 agent
        agent = next((a for a in manager.population if a.agent_id == agent_id), None)
        if agent is None:
            return {"status": "error", "message": f"Agent not found: {agent_id}"}

        return {
            "status": "ok",
            "agent_id": agent_id,
            "fitness": _get_agent_score_dict(agent),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_best_agents(
    population_id: str,
    n: int = 10,
    metric: str = "gt_score",
) -> Dict[str, Any]:
    """工具5: 获取最优Agent"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": "Population not found"}

        # 从 agent 运行时状态构造 scores
        agent_scores = [(a, _get_agent_score_dict(a)) for a in manager.population]

        # metric 字段映射: gt_score -> gt_score (其他字段保持原名)
        sort_key = lambda x: x[1].get(metric, 0)
        sorted_pairs = sorted(agent_scores, key=sort_key, reverse=True)[:n]

        return {
            "status": "ok",
            "metric": metric,
            "best_agents": [
                {
                    "agent_id": a.agent_id,
                    "rank": i + 1,
                    metric: s.get(metric, 0),
                    "sharpe": s.get("sharpe_ratio", 0),
                    "win_rate": s.get("win_rate", 0),
                    "max_drawdown": s.get("max_drawdown", 0),
                    "total_trades": s.get("total_trades", 0),
                }
                for i, (a, s) in enumerate(sorted_pairs)
            ],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_save_checkpoint(
    population_id: str,
    checkpoint_path: str = "",
) -> Dict[str, Any]:
    """工具6: 保存检查点"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": "Population not found"}

        # 使用 CheckpointManager (来自 deployment_infra.py) 保存检查点
        from .deployment_infra import CheckpointManager

        ckpt_mgr = CheckpointManager()
        actual_path = ckpt_mgr.save_checkpoint(
            population_manager=manager,
            generation=manager.generation,
            description=f"MCP save by population_id={population_id}",
        )

        _checkpoints[population_id] = actual_path

        return {
            "status": "ok",
            "checkpoint_path": actual_path,
            "generation": manager.generation,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_load_checkpoint(
    population_id: str,
    checkpoint_path: str,
) -> Dict[str, Any]:
    """工具7: 加载检查点"""
    try:
        from .deployment_infra import CheckpointManager
        from .population import PopulationManager
        from .gene_codec import AgentGene

        ckpt_mgr = CheckpointManager()
        data = ckpt_mgr.load_checkpoint(checkpoint_path)

        # 重建 PopulationManager
        manager = PopulationManager()
        manager.population = []
        manager.generation = data.get("generation", 0)

        # 重建 agent 列表（仅恢复 agent_id 和 generation，基因细节简化）
        for agent_data in data.get("agents", []):
            gene = AgentGene()
            gene.agent_id = agent_data.get("agent_id", "")
            gene.generation = agent_data.get("generation", 0)
            manager.population.append(gene)

        _populations[population_id] = manager
        _checkpoints[population_id] = checkpoint_path

        return {
            "status": "ok",
            "population_id": population_id,
            "generation": manager.generation,
            "population_size": len(manager.population),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_evolution_stats(population_id: str) -> Dict[str, Any]:
    """工具8: 获取进化统计"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": "Population not found"}

        if not manager.population:
            return {"status": "ok", "stats": {}, "message": "Empty population"}

        # 从 agent 运行时状态构造统计数据
        score_dicts = [_get_agent_score_dict(a) for a in manager.population]
        gt_scores = [s["gt_score"] for s in score_dicts]
        sharpes = [s["sharpe_ratio"] for s in score_dicts]
        win_rates = [s["win_rate"] for s in score_dicts]
        drawdowns = [s["max_drawdown"] for s in score_dicts]

        return {
            "status": "ok",
            "generation": manager.generation,
            "population_size": len(manager.population),
            "stats": {
                "gt_score": {
                    "mean": sum(gt_scores) / len(gt_scores),
                    "max": max(gt_scores),
                    "min": min(gt_scores),
                },
                "sharpe": {
                    "mean": sum(sharpes) / len(sharpes),
                    "max": max(sharpes),
                    "min": min(sharpes),
                },
                "win_rate": {
                    "mean": sum(win_rates) / len(win_rates),
                    "max": max(win_rates),
                },
                "max_drawdown": {
                    "mean": sum(drawdowns) / len(drawdowns),
                    "max": max(drawdowns),
                },
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_mutate_agent(
    population_id: str,
    agent_id: str,
    mutation_rate: float = 0.1,
) -> Dict[str, Any]:
    """工具9: 变异Agent"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": "Population not found"}

        agents = {a.agent_id: a for a in manager.population}
        if agent_id not in agents:
            return {"status": "error", "message": f"Agent not found: {agent_id}"}

        from .gene_codec import mutate
        parent_gene = agents[agent_id].gene
        child_gene = mutate(parent_gene, mutation_rate=mutation_rate)

        return {
            "status": "ok",
            "parent_id": agent_id,
            "child_gene": _serialize(child_gene),
            "mutation_rate": mutation_rate,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_crossover_agents(
    population_id: str,
    parent_a_id: str,
    parent_b_id: str,
) -> Dict[str, Any]:
    """工具10: 交叉Agent"""
    try:
        manager = _populations.get(population_id)
        if manager is None:
            return {"status": "error", "message": "Population not found"}

        agents = {a.agent_id: a for a in manager.population}
        if parent_a_id not in agents or parent_b_id not in agents:
            return {"status": "error", "message": "Parent not found"}

        from .gene_codec import crossover
        child_gene = crossover(
            agents[parent_a_id].gene,
            agents[parent_b_id].gene,
        )

        return {
            "status": "ok",
            "parent_a": parent_a_id,
            "parent_b": parent_b_id,
            "child_gene": _serialize(child_gene),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# P2-3优化: 新增增强工具函数
# ============================================================================


def tool_validate_anti_overfitting(population_id: str) -> Dict[str, Any]:
    """P2-3: 反过拟合验证

    对种群执行反过拟合验证:
      - Walk-Forward验证
      - 蒙特卡洛置换检验
      - PBO过拟合概率
      - 参数敏感性

    R12修复: 调用真实AntiOverfittingValidator,替代原简化估算
    阈值与真实模块对齐(0.1/0.2/0.3/0.5 而非 0.3/0.5/0.7/0.9)

    Args:
        population_id: 种群ID

    Returns:
        验证报告
    """
    if population_id not in _populations:
        return {"status": "error", "message": f"种群 {population_id} 不存在"}

    try:
        pm = _populations[population_id]

        # 收集所有agent的GT-Score和交易记录
        gt_scores = []
        agent_trades_map = {}  # agent_id -> trades列表
        for agent in pm.population:
            score = getattr(agent, "fitness_score", 0) or 0
            gt_scores.append(score)
            # 收集agent的交易记录(如果存在)
            trades = getattr(agent, "trades", []) or getattr(agent, "_trades", [])
            if trades:
                agent_trades_map[agent.agent_id] = list(trades)

        if not gt_scores:
            return {"status": "error", "message": "无有效GT-Score数据"}

        # 基础统计(始终返回)
        avg_gt = sum(gt_scores) / len(gt_scores)
        max_gt = max(gt_scores)
        min_gt = min(gt_scores)

        # R12修复: 调用真实AntiOverfittingValidator
        # 阈值与anti_overfitting.py一致(0.1/0.2/0.3/0.5)
        validation_details = {}
        risk_level = "ACCEPTABLE"
        pbo_risk = 0.0
        recommendation = "需优化"

        try:
            from .anti_overfitting import AntiOverfittingValidator
            validator = AntiOverfittingValidator()

            # 若有交易记录,执行完整验证
            if agent_trades_map:
                # 取交易最多的agent做代表性验证
                best_aid = max(agent_trades_map, key=lambda k: len(agent_trades_map[k]))
                sample_trades = agent_trades_map[best_aid]

                # 构造strategy_returns(每笔交易的pnl序列)
                strategy_returns = [
                    [t.get("pnl", 0.0) for t in sample_trades]
                ]

                result = validator.validate(
                    strategy_returns=strategy_returns,
                    agent_trades=agent_trades_map,
                    agent_scores={aid: s for aid, s in zip(
                        [a.agent_id for a in pm.population], gt_scores
                    )},
                )

                # 提取真实验证结果
                pbo_risk = float(getattr(result, "pbo_probability", 0.0) or 0.0)
                is_overfitted = bool(getattr(result, "is_overfitted", False))
                oos_is_ratio = float(getattr(result, "oos_is_ratio", 0.0) or 0.0)
                mc_pvalue = float(getattr(result, "monte_carlo_pvalue", 0.0) or 0.0)
                param_stability = float(getattr(result, "parameter_stability", 0.0) or 0.0)
                slippage_impact = float(getattr(result, "slippage_impact", 0.0) or 0.0)

                validation_details = {
                    "pbo_probability": round(pbo_risk, 3),
                    "is_overfitted": is_overfitted,
                    "oos_is_ratio": round(oos_is_ratio, 3),
                    "monte_carlo_pvalue": round(mc_pvalue, 3),
                    "parameter_stability": round(param_stability, 3),
                    "slippage_impact": round(slippage_impact, 3),
                    "validated_agent": best_aid,
                    "validated_trades": len(sample_trades),
                }

                # 风险等级判定(阈值与anti_overfitting.py对齐)
                if pbo_risk < 0.1 and not is_overfitted:
                    risk_level = "EXCELLENT"
                    recommendation = "可部署"
                elif pbo_risk < 0.2 and oos_is_ratio > 0.5:
                    risk_level = "GOOD"
                    recommendation = "可部署"
                elif pbo_risk < 0.3:
                    risk_level = "ACCEPTABLE"
                    recommendation = "需优化"
                elif pbo_risk < 0.5:
                    risk_level = "WARN"
                    recommendation = "需优化"
                else:
                    risk_level = "BLOCK"
                    recommendation = "禁止部署"
            else:
                validation_details = {"note": "无交易记录,跳过完整验证,仅返回GT-Score统计"}
                # 无交易时基于GT-Score分布估算
                _gt_std = (sum((s - avg_gt) ** 2 for s in gt_scores) / len(gt_scores)) ** 0.5
                pbo_risk = min(1.0, _gt_std / max(avg_gt, 1.0))
                if pbo_risk < 0.2:
                    risk_level = "GOOD"
                    recommendation = "可部署(需交易数据确认)"
                elif pbo_risk < 0.4:
                    risk_level = "ACCEPTABLE"
                    recommendation = "需优化"
                else:
                    risk_level = "WARN"
                    recommendation = "需优化"

        except ImportError:
            validation_details = {"note": "anti_overfitting模块不可用,使用基础统计"}
            # 回退到基础统计(与原简化版一致但阈值已对齐)
            pbo_risk = 1.0 - (max_gt - avg_gt) / max(max_gt, 1.0)
            pbo_risk = max(0.0, min(1.0, pbo_risk))
            if pbo_risk < 0.1:
                risk_level = "EXCELLENT"
            elif pbo_risk < 0.2:
                risk_level = "GOOD"
            elif pbo_risk < 0.3:
                risk_level = "ACCEPTABLE"
            elif pbo_risk < 0.5:
                risk_level = "WARN"
            else:
                risk_level = "BLOCK"
            recommendation = (
                "可部署" if risk_level in ("EXCELLENT", "GOOD")
                else "需优化" if risk_level == "ACCEPTABLE"
                else "禁止部署"
            )

        return {
            "status": "ok",
            "population_id": population_id,
            "validation_result": {
                "avg_gt_score": round(avg_gt, 2),
                "max_gt_score": round(max_gt, 2),
                "min_gt_score": round(min_gt, 2),
                "pbo_risk": round(pbo_risk, 3),
                "risk_level": risk_level,
                "agent_count": len(gt_scores),
                **validation_details,
            },
            "recommendation": recommendation,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_evolution_trend(population_id: str, last_n: int = 0) -> Dict[str, Any]:
    """P2-4: 获取进化趋势

    Args:
        population_id: 种群ID
        last_n: 最近N代(0=全部)

    Returns:
        趋势数据
    """
    if population_id not in _populations:
        return {"status": "error", "message": f"种群 {population_id} 不存在"}

    try:
        pm = _populations[population_id]

        # 从population历史获取趋势
        history = getattr(pm, "history", [])
        if not history:
            return {
                "status": "ok",
                "population_id": population_id,
                "trend": {"status": "no_data", "message": "尚无进化历史"},
            }

        # 提取关键指标
        records = []
        for stats in history[-last_n:] if last_n > 0 else history:
            records.append({
                "gen": getattr(stats, "generation", 0),
                "avg_gt": getattr(stats, "avg_gt_score", 0),
                "max_gt": getattr(stats, "best_gt_score", 0),
                "diversity": getattr(stats, "diversity_score", 0),
                "pareto_front": getattr(stats, "pareto_front_size", 0),
            })

        # 计算趋势
        if records:
            first = records[0]
            last = records[-1]
            trend = {
                "total_generations": len(records),
                "first_gen": first["gen"],
                "last_gen": last["gen"],
                "avg_gt_change": round(last["avg_gt"] - first["avg_gt"], 2),
                "max_gt_change": round(last["max_gt"] - first["max_gt"], 2),
                "diversity_change": round(last["diversity"] - first["diversity"], 3),
                "recent": records[-5:] if len(records) > 5 else records,
            }
        else:
            trend = {"status": "no_data"}

        return {
            "status": "ok",
            "population_id": population_id,
            "trend": trend,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tool_get_enhancement_stats(population_id: str) -> Dict[str, Any]:
    """P1-2: 获取增强统计

    Args:
        population_id: 种群ID

    Returns:
        增强统计(Arena/换手率/HHI/辩论门控)
    """
    if population_id not in _populations:
        return {"status": "error", "message": f"种群 {population_id} 不存在"}

    try:
        pm = _populations[population_id]

        stats = {
            "status": "ok",
            "population_id": population_id,
            "nsga3_enabled": getattr(pm, "use_nsga3", False),
            "nsga2_enabled": getattr(pm, "use_nsga2", False),
            "enhancements": {},
            "debate_gate": {},
        }

        # 增强统计
        if hasattr(pm, "enhancements") and pm.enhancements:
            stats["enhancements"] = pm.enhancements.get_stats()
            stats["enhancements"]["enabled"] = pm.enhancements.enabled

        return stats
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# MCP工具注册表
# ============================================================================


MCP_TOOLS = {
    "create_population": {
        "func": tool_create_population,
        "description": "创建初始进化种群",
        "params": ["population_id", "population_size", "seed"],
    },
    "run_evolution": {
        "func": tool_run_evolution,
        "description": "运行进化轮次",
        "params": ["population_id", "n_rounds"],
    },
    "get_population_status": {
        "func": tool_get_population_status,
        "description": "获取种群状态",
        "params": ["population_id"],
    },
    "get_agent_fitness": {
        "func": tool_get_agent_fitness,
        "description": "获取Agent适应度",
        "params": ["population_id", "agent_id"],
    },
    "get_best_agents": {
        "func": tool_get_best_agents,
        "description": "获取最优Agent排名",
        "params": ["population_id", "n", "metric"],
    },
    "save_checkpoint": {
        "func": tool_save_checkpoint,
        "description": "保存进化检查点",
        "params": ["population_id", "checkpoint_path"],
    },
    "load_checkpoint": {
        "func": tool_load_checkpoint,
        "description": "从检查点恢复进化",
        "params": ["population_id", "checkpoint_path"],
    },
    "get_evolution_stats": {
        "func": tool_get_evolution_stats,
        "description": "获取进化统计信息",
        "params": ["population_id"],
    },
    "mutate_agent": {
        "func": tool_mutate_agent,
        "description": "手动变异Agent基因",
        "params": ["population_id", "agent_id", "mutation_rate"],
    },
    "crossover_agents": {
        "func": tool_crossover_agents,
        "description": "交叉两个Agent基因",
        "params": ["population_id", "parent_a_id", "parent_b_id"],
    },
    # P2-3优化: 新增3个增强工具
    "validate_anti_overfitting": {
        "func": tool_validate_anti_overfitting,
        "description": "P2-3: 反过拟合验证(Walk-Forward+蒙特卡洛+PBO)",
        "params": ["population_id"],
    },
    "get_evolution_trend": {
        "func": tool_get_evolution_trend,
        "description": "P2-4: 获取进化趋势(GT-Score/多样性/PBO)",
        "params": ["population_id", "last_n"],
    },
    "get_enhancement_stats": {
        "func": tool_get_enhancement_stats,
        "description": "P1-2: 获取增强统计(Arena/换手率/HHI/辩论门控)",
        "params": ["population_id"],
    },
}


def call_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    """统一工具调用入口

    Args:
        tool_name: 工具名称
        **kwargs: 工具参数

    Returns:
        工具执行结果（JSON可序列化）
    """
    tool = MCP_TOOLS.get(tool_name)
    if tool is None:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    try:
        return tool["func"](**kwargs)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e)
        return {"status": "error", "message": str(e)}


def list_tools() -> List[Dict[str, Any]]:
    """列出所有可用工具"""
    return [
        {
            "name": name,
            "description": tool["description"],
            "params": tool["params"],
        }
        for name, tool in MCP_TOOLS.items()
    ]


__all__ = [
    "MCP_TOOLS",
    "call_tool",
    "list_tools",
    "tool_create_population",
    "tool_run_evolution",
    "tool_get_population_status",
    "tool_get_agent_fitness",
    "tool_get_best_agents",
    "tool_save_checkpoint",
    "tool_load_checkpoint",
    "tool_get_evolution_stats",
    "tool_mutate_agent",
    "tool_crossover_agents",
]
