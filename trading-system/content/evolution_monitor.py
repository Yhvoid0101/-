# -*- coding: utf-8 -*-
"""
进化监控器 — P2-1/P2-4优化

两大功能:
  1. 检查点自动恢复 (P2-1): EvolutionLoop启动时自动检查并恢复最近检查点
  2. 进化趋势监控 (P2-4): 实时记录GT-Score/多样性/PBO趋势

设计原则: 不破坏现有流程,作为可选增强层
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.evolution_monitor")

try:
    from .deployment_infra import CheckpointManager
    _CHECKPOINT_AVAILABLE = True
except ImportError:
    _CHECKPOINT_AVAILABLE = False


class EvolutionMonitor:
    """进化监控器

    功能:
      1. 检查点自动恢复: initialize()时检查checkpoint_dir,自动加载最新检查点
      2. 进化趋势记录: 每代记录GT-Score/多样性/PBO等关键指标
      3. 异常检测: 多样性骤降/GT-Score停滞/PBO过高时告警

    使用方式:
        monitor = EvolutionMonitor(checkpoint_dir="/tmp/ckpt", enabled=True)
        # 自动恢复
        restored = monitor.try_restore()
        # 记录每代
        monitor.record_generation(gen=5, stats={...})
        # 获取趋势
        trend = monitor.get_trend()
    """

    def __init__(
        self,
        checkpoint_dir: str = "",
        enabled: bool = True,
        history_size: int = 100,
    ):
        """
        Args:
            checkpoint_dir: 检查点目录
            enabled: 是否启用
            history_size: 历史记录保留数量
        """
        self.enabled = enabled
        self.history_size = history_size

        # 检查点管理器
        if enabled and _CHECKPOINT_AVAILABLE:
            self.checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
        else:
            self.checkpoint_mgr = None

        # 进化历史 (deque自动限制大小)
        self._generation_history: deque = deque(maxlen=history_size)
        self._alerts: List[Dict[str, Any]] = []

        # 统计
        self._stats = {
            "total_generations_recorded": 0,
            "checkpoints_restored": 0,
            "alerts_triggered": 0,
        }

    def try_restore(self) -> Optional[Dict[str, Any]]:
        """尝试从最近检查点恢复

        Returns:
            检查点数据(如果恢复成功), None(如果无检查点或未启用)
        """
        if not self.enabled or not self.checkpoint_mgr:
            return None

        try:
            latest = self.checkpoint_mgr.find_latest_checkpoint()
            if not latest:
                logger.info("P2-1: 无可用检查点,从头开始进化")
                return None

            data = self.checkpoint_mgr.load_checkpoint(latest)
            self._stats["checkpoints_restored"] += 1
            logger.info(
                "P2-1: 从检查点恢复成功 gen=%d, agents=%d, file=%s",
                data.get("generation", 0),
                len(data.get("agents", [])),
                Path(latest).name,
            )
            return data
        except Exception as e:
            logger.warning("P2-1: 检查点恢复失败: %s", e)
            return None

    def record_generation(
        self,
        gen: int,
        gt_scores: List[float],
        diversity: float = 0.0,
        pareto_front_size: int = 0,
        population_size: int = 0,
        pbo_risk: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录每代进化指标

        Args:
            gen: 代数
            gt_scores: 所有Agent的GT-Score列表
            diversity: 基因多样性
            pareto_front_size: 帕累托前沿大小
            population_size: 种群大小
            pbo_risk: PBO过拟合概率
            extra: 额外指标
        """
        if not self.enabled:
            return

        # 计算统计
        if gt_scores:
            avg_gt = sum(gt_scores) / len(gt_scores)
            max_gt = max(gt_scores)
            min_gt = min(gt_scores)
        else:
            avg_gt = max_gt = min_gt = 0.0

        record = {
            "gen": gen,
            "timestamp": time.time(),
            "avg_gt": round(avg_gt, 2),
            "max_gt": round(max_gt, 2),
            "min_gt": round(min_gt, 2),
            "diversity": round(diversity, 3),
            "pareto_front_size": pareto_front_size,
            "population_size": population_size,
            "pbo_risk": round(pbo_risk, 3),
        }
        if extra:
            record.update(extra)

        self._generation_history.append(record)
        self._stats["total_generations_recorded"] += 1

        # 异常检测
        self._check_anomalies(record)

    def _check_anomalies(self, record: Dict[str, Any]) -> None:
        """异常检测"""
        gen = record["gen"]
        diversity = record["diversity"]
        pbo = record["pbo_risk"]

        # 多样性骤降告警
        if len(self._generation_history) >= 2:
            prev = self._generation_history[-2]
            prev_div = prev.get("diversity", 1.0)
            if prev_div > 0 and diversity < prev_div * 0.5:
                alert = {
                    "gen": gen,
                    "type": "diversity_drop",
                    "message": f"多样性骤降: {prev_div:.3f}→{diversity:.3f}",
                    "severity": "WARN",
                }
                self._alerts.append(alert)
                self._stats["alerts_triggered"] += 1
                logger.warning("P2-4告警: %s", alert["message"])

        # PBO过高告警
        if pbo > 0.8:
            alert = {
                "gen": gen,
                "type": "high_pbo",
                "message": f"PBO过拟合风险: {pbo:.3f}",
                "severity": "WARN",
            }
            self._alerts.append(alert)
            self._stats["alerts_triggered"] += 1
            logger.warning("P2-4告警: %s", alert["message"])

        # GT-Score停滞告警
        if len(self._generation_history) >= 5:
            recent = list(self._generation_history)[-5:]
            max_gts = [r["max_gt"] for r in recent]
            if max(max_gts) - min(max_gts) < 1.0:  # 5代最大GT-Score变化<1
                alert = {
                    "gen": gen,
                    "type": "stagnation",
                    "message": f"GT-Score停滞5代(max范围: {min(max_gts):.1f}-{max(max_gts):.1f})",
                    "severity": "INFO",
                }
                self._alerts.append(alert)
                self._stats["alerts_triggered"] += 1
                logger.info("P2-4告警: %s", alert["message"])

    def get_trend(self, last_n: int = 0) -> Dict[str, Any]:
        """获取进化趋势

        Args:
            last_n: 最近N代(0=全部)

        Returns:
            趋势数据
        """
        history = list(self._generation_history)
        if last_n > 0:
            history = history[-last_n:]

        if not history:
            return {"status": "no_data"}

        # 计算趋势
        gt_scores = [r["avg_gt"] for r in history]
        diversities = [r["diversity"] for r in history]

        return {
            "total_generations": len(history),
            "first_gen": history[0]["gen"],
            "last_gen": history[-1]["gen"],
            "avg_gt_trend": {
                "first": gt_scores[0],
                "last": gt_scores[-1],
                "change": gt_scores[-1] - gt_scores[0],
                "max": max(gt_scores),
                "min": min(gt_scores),
            },
            "diversity_trend": {
                "first": diversities[0],
                "last": diversities[-1],
                "change": diversities[-1] - diversities[0],
                "min": min(diversities),
            },
            "recent_records": history[-5:] if len(history) > 5 else history,
            "alerts": self._alerts[-10:] if self._alerts else [],
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return dict(self._stats)

    def save_trend(self, filepath: str) -> None:
        """保存趋势数据到文件"""
        trend = self.get_trend()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trend, f, ensure_ascii=False, indent=2, default=str)
        logger.info("P2-4: 趋势数据已保存到 %s", filepath)


__all__ = ["EvolutionMonitor"]
