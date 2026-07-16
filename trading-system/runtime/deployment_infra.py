# -*- coding: utf-8 -*-
"""
部署基础设施 — P3-1 Docker零部署+检查点恢复+决策日志

三大工程化能力:
  1. 检查点恢复: 进化过程的保存和恢复
     - 定期自动保存
     - 崩溃后从最近检查点恢复
  2. 决策日志: 记录每个交易决策的完整上下文
     - 用于事后分析和进化优化
     - 可追溯的决策链
  3. Docker零部署: 一键部署配置
     - Dockerfile + docker-compose
     - 环境变量配置

参考:
  - NautilusTrader: 检查点恢复机制
  - MLflow: 决策追踪和可复现性
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.deployment")


# ============================================================================
# 检查点管理器
# ============================================================================


@dataclass(slots=True)
class CheckpointMetadata:
    """检查点元数据"""
    checkpoint_id: str
    timestamp: float
    generation: int
    population_size: int
    best_gt_score: float
    avg_gt_score: float
    description: str = ""


class CheckpointManager:
    """检查点管理器

    定期保存进化过程的状态，支持崩溃恢复。
    """

    def __init__(self, checkpoint_dir: str = ""):
        # P1修复: 使用跨平台临时目录，不再硬编码 /tmp/
        if not checkpoint_dir:
            import tempfile
            checkpoint_dir = str(Path(tempfile.gettempdir()) / "evolution_checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._auto_save_interval: int = 10  # 每10代自动保存
        self._last_save_generation: int = 0

    def save_checkpoint(
        self,
        population_manager: Any,
        generation: int,
        description: str = "",
    ) -> str:
        """保存检查点

        Args:
            population_manager: 种群管理器
            generation: 当前代数
            description: 描述

        Returns:
            检查点文件路径
        """
        checkpoint_id = f"ckpt_gen{generation}_{int(time.time())}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        # 收集检查点数据
        # P1修复: PopulationManager 没有 get_all_agents() 和 fitness_scores 属性
        # 改为使用 manager.population 属性，并从 agent 运行时状态构造 scores
        agents = []
        if hasattr(population_manager, "population"):
            agents = list(population_manager.population)
        elif hasattr(population_manager, "get_all_agents"):
            agents = population_manager.get_all_agents()

        # 从 agent 运行时状态构造 scores 字典
        scores = {}
        for a in agents:
            aid = getattr(a, "agent_id", "")
            if aid:
                scores[aid] = {
                    "gt_score": getattr(a, "fitness_score", 0.0),
                    "sharpe_ratio": getattr(a, "sharpe_ratio", 0.0),
                    "win_rate": getattr(a, "win_rate", 0.0),
                    "max_drawdown": getattr(a, "max_drawdown_realized", 0.0),
                    "total_trades": getattr(a, "total_trades", 0),
                }

        gt_scores = [s["gt_score"] for s in scores.values()] if scores else []

        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            generation=generation,
            population_size=len(agents),
            best_gt_score=max(gt_scores) if gt_scores else 0,
            avg_gt_score=sum(gt_scores) / len(gt_scores) if gt_scores else 0,
            description=description,
        )

        # 序列化保存
        checkpoint_data = {
            "metadata": asdict(metadata),
            "generation": generation,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "generation": a.generation,
                    "gene": (
                        asdict(a.gene) if hasattr(a, "gene") and is_dataclass(a.gene)
                        else asdict(a) if is_dataclass(a)
                        else str(getattr(a, "gene", ""))
                    ),
                }
                for a in agents
            ],
            "scores": scores,
        }

        temp_path = checkpoint_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, checkpoint_path)

        self._last_save_generation = generation
        logger.info(
            "Checkpoint saved: %s (gen=%d, size=%d, best=%.2f)",
            checkpoint_path, generation, len(agents), metadata.best_gt_score,
        )

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            检查点数据
        """
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            "Checkpoint loaded: %s (gen=%d, agents=%d)",
            checkpoint_path,
            data.get("generation", 0),
            len(data.get("agents", [])),
        )

        return data

    def find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的检查点"""
        # P1修复: 按 mtime 排序而非文件名，避免跨时区/格式变化导致排序错误
        checkpoints = []
        for checkpoint in self.checkpoint_dir.glob("ckpt_*.json"):
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                continue
            try:
                with checkpoint.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or not data.get("agents"):
                    continue
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            checkpoints.append(checkpoint)
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    def should_auto_save(self, current_generation: int) -> bool:
        """检查是否应该自动保存"""
        return current_generation - self._last_save_generation >= self._auto_save_interval

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        result = []
        for ckpt_file in sorted(self.checkpoint_dir.glob("ckpt_*.json")):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                result.append({
                    "file": str(ckpt_file),
                    "checkpoint_id": meta.get("checkpoint_id", ""),
                    "generation": meta.get("generation", 0),
                    "timestamp": meta.get("timestamp", 0),
                    "best_score": meta.get("best_gt_score", 0),
                })
            except Exception as e:
                logger.warning("Failed to read checkpoint %s: %s", ckpt_file, e)
        return result

    def cleanup_old_checkpoints(self, keep: int = 5) -> int:
        """清理旧检查点，只保留最近的N个"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        deleted = 0
        for ckpt in checkpoints[keep:]:
            ckpt.unlink()
            deleted += 1
        if deleted:
            logger.info("Cleaned up %d old checkpoints", deleted)
        return deleted


# ============================================================================
# 决策日志器
# ============================================================================


@dataclass(slots=True)
class DecisionRecord:
    """决策记录"""
    decision_id: str
    timestamp: float
    agent_id: str
    symbol: str
    action: str          # "open"/"close"/"hold"
    direction: str       # "long"/"short"/"neutral"
    quantity: float = 0.0
    price: float = 0.0
    leverage: int = 1
    reason: str = ""
    signals: Dict[str, Any] = field(default_factory=dict)
    market_state: Dict[str, Any] = field(default_factory=dict)
    debate_result: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)  # 事后结果


class DecisionLogger:
    """决策日志器

    记录每个交易决策的完整上下文，用于:
      1. 事后分析决策质量
      2. 进化优化时提供决策依据
      3. 可追溯的决策链
    """

    def __init__(self, log_dir: str = "/tmp/evolution_decisions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_log_file: Optional[Path] = None
        self._decision_counter: int = 0
        self._buffer: List[DecisionRecord] = []
        self._buffer_size: int = 100

    def log_decision(self, record: DecisionRecord) -> None:
        """记录决策"""
        self._decision_counter += 1
        self._buffer.append(record)

        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        """刷新缓冲区到文件"""
        if not self._buffer:
            return

        # 按日期分文件
        today = time.strftime("%Y%m%d", time.localtime())
        log_file = self.log_dir / f"decisions_{today}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        flushed_count = len(self._buffer)
        self._buffer.clear()
        # P1修复#13: 修复日志计数 — 用实际刷新数量替代 buffer_size 阈值
        logger.debug("Flushed %d decisions to %s", flushed_count, log_file)

    def update_outcome(
        self,
        decision_id: str,
        outcome: Dict[str, Any],
    ) -> None:
        """更新决策的事后结果"""
        # 在实际实现中，需要从日志文件中查找并更新
        # 这里简化为记录到单独的结果文件
        result_file = self.log_dir / "outcomes.jsonl"
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "decision_id": decision_id,
                "outcome": outcome,
                "timestamp": time.time(),
            }, ensure_ascii=False) + "\n")

    def get_decision_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取决策历史"""
        results: List[Dict[str, Any]] = []

        for log_file in sorted(self.log_dir.glob("decisions_*.jsonl"), reverse=True):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if agent_id is None or record.get("agent_id") == agent_id:
                            results.append(record)
                            if len(results) >= limit:
                                return results
                    except json.JSONDecodeError:
                        continue

        return results

    def flush(self) -> None:
        """强制刷新"""
        self._flush()


# ============================================================================
# Docker部署配置生成器
# ============================================================================


class DockerDeploymentGenerator:
    """Docker部署配置生成器

    生成Dockerfile和docker-compose.yml，
    实现一键部署。
    """

    DOCKERFILE_TEMPLATE = """# Hermes Evolution Trading System - Docker Image
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV EVOLUTION_MODE=sandbox
ENV CHECKPOINT_DIR=/data/checkpoints
ENV DECISION_LOG_DIR=/data/decisions

# 数据卷
VOLUME ["/data"]

# 暴露MCP端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# 启动命令
CMD ["python", "-m", "sandbox_trading.evolution_loop", "--mode=${EVOLUTION_MODE}"]
"""

    DOCKER_COMPOSE_TEMPLATE = """version: '3.8'

services:
  evolution-engine:
    build: .
    container_name: hermes-evolution
    restart: unless-stopped
    environment:
      - EVOLUTION_MODE=${EVOLUTION_MODE:-sandbox}
      - POPULATION_SIZE=${POPULATION_SIZE:-50}
      - EVOLUTION_ROUNDS=${EVOLUTION_ROUNDS:-100}
      - CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}
    volumes:
      - evolution-data:/data
      - ./config:/app/config:ro
    ports:
      - "8080:8080"
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  # 可选: Redis用于分布式进化
  redis:
    image: redis:7-alpine
    container_name: hermes-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  evolution-data:
  redis-data:
"""

    REQUIREMENTS_TEMPLATE = """# Hermes Evolution Trading System
numpy>=1.24
pandas>=2.0
scipy>=1.10
ccxt>=4.0
mcp>=0.1
torch>=2.0
timesfm>=0.1
requests>=2.28  # P1修复#12: HEALTHCHECK 依赖
"""

    @classmethod
    def generate_deployment_files(cls, output_dir: str = ".") -> Dict[str, str]:
        """生成部署文件

        Args:
            output_dir: 输出目录

        Returns:
            生成的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {
            "Dockerfile": cls.DOCKERFILE_TEMPLATE,
            "docker-compose.yml": cls.DOCKER_COMPOSE_TEMPLATE,
            "requirements.txt": cls.REQUIREMENTS_TEMPLATE,
        }

        result = {}
        for filename, content in files.items():
            filepath = output_path / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            result[filename] = str(filepath)
            logger.info("Generated: %s", filepath)

        return result


__all__ = [
    "CheckpointMetadata",
    "CheckpointManager",
    "DecisionRecord",
    "DecisionLogger",
    "DockerDeploymentGenerator",
]
