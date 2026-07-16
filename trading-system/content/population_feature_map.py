"""Population feature map module (Phase 7.12 extracted from population.py)

Contains:
  - FeatureMapCell: feature map cell dataclass
  - FeatureMap: behavior feature map for diversity maintenance
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .gene_codec import AgentGene
from .population_fitness import FitnessScore


@dataclass(slots=True)
class FeatureMapCell:
    """Feature Map中的一个格子"""
    agent_id: str
    gt_score: float
    generation: int


class FeatureMap:
    """行为特征地图 — 按行为特征分格，每格只保留最优

    来自QuantEvolve论文的核心创新：不按基因相似度维护多样性，
    而是按行为特征（风险、收益、换手率、策略类型）分格，
    确保种群在行为空间中均匀分布。

    4个行为维度：
      1. 风险等级：回撤<5%=低, 5-15%=中, >15%=高
      2. 收益特征：夏普>2=高, 0-2=中, <0=负
      3. 交易频率：>50笔=高频, 10-50=中频, <10=低频
      4. 策略类型：趋势/均值回归/突破/动量/其他
    """

    # 分格定义
    RISK_BINS = [(0, 0.05, "low"), (0.05, 0.15, "mid"), (0.15, float("inf"), "high")]
    RETURN_BINS = [(2.0, float("inf"), "high"), (0.0, 2.0, "mid"), (float("-inf"), 0.0, "neg")]
    FREQ_BINS = [(50, float("inf"), "high"), (10, 50, "mid"), (0, 10, "low")]

    def __init__(self):
        self._cells: Dict[str, FeatureMapCell] = {}

    def _get_cell_key(self, score: FitnessScore, gene: AgentGene) -> str:
        """计算Agent所属的行为格子"""
        # 风险等级
        risk_label = "unknown"
        for lo, hi, label in self.RISK_BINS:
            if lo <= score.max_drawdown < hi:
                risk_label = label
                break

        # 收益特征
        return_label = "unknown"
        for lo, hi, label in self.RETURN_BINS:
            if lo <= score.sharpe_ratio < hi:
                return_label = label
                break

        # 交易频率
        freq_label = "unknown"
        for lo, hi, label in self.FREQ_BINS:
            if lo <= score.total_trades < hi:
                freq_label = label
                break

        # 策略类型
        strategy_label = gene.worldview_primary

        return f"{risk_label}|{return_label}|{freq_label}|{strategy_label}"

    def update(self, agent: AgentGene, score: FitnessScore, generation: int) -> bool:
        """更新Feature Map，返回是否被接受（即是否是该格子的最优）

        Returns:
            True if agent was accepted into the map (is best in its cell)
        """
        key = self._get_cell_key(score, agent)
        existing = self._cells.get(key)

        if existing is None or score.gt_score > existing.gt_score:
            self._cells[key] = FeatureMapCell(
                agent_id=agent.agent_id,
                gt_score=score.gt_score,
                generation=generation,
            )
            return True
        return False

    def get_diverse_elite(self, top_n: int = 5) -> List[str]:
        """从不同格子中选取精英，确保行为多样性"""
        sorted_cells = sorted(
            self._cells.values(),
            key=lambda c: c.gt_score,
            reverse=True,
        )
        return [c.agent_id for c in sorted_cells[:top_n]]

    def get_cell_count(self) -> int:
        """返回当前被占据的格子数（越多越多样）"""
        return len(self._cells)

    def get_occupied_cells(self) -> Set[str]:
        """返回所有被占据的格子key"""
        return set(self._cells.keys())
