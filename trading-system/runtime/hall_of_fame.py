"""
HallOfFameMixin — Phase 7.1 架构解耦
从 evolution_loop.py 提取的 Hall of Fame 管理逻辑。

设计:
  - Mixin 模式: 不继承任何基类,通过多重继承注入 EvolutionLoop
  - 鸭子类型依赖: Host 需提供 self.population(.generation/.get_elite_signals) 和 self._round_number
  - 零循环依赖: 仅依赖 logger + numpy + typing

来源:
  - v2.24 NSGA-II elitism + CMA-ES archive (López de Prado 2018)
  - v2.29 多轮稳定性 (CSCV + Bailey 2014 多重试验偏差校正)
  - v2.32 三维稳定性检测 (方差/符号一致性/变异系数)
  - v2.36 滑动窗口 2/3 代 PASSED (Pardo 2008 Walk-Forward Optimization)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class HallOfFameMixin:
    """Hall of Fame 管理 Mixin (Phase 7.1 提取)

    提供:
      - _init_hall_of_fame(): 初始化9个 hall_of_fame* 属性 (Host __init__ 调用)
      - _add_to_hall_of_fame(): PASSED Agent 加入候选/Hall (滑动窗口2/3)
      - _record_overfitted_for_hall_of_fame(): OVERFITTED 记录到滑动窗口
      - _remove_from_hall_of_fame_candidates(): 强制移除候选
      - get_hall_of_fame(): 获取 Top-N 历史 PASSED Agent
      - get_elite_signals(): 委托 population.get_elite_signals

    Host 鸭子类型依赖:
      - self.population.generation: int
      - self._round_number: int
      - self.population.get_elite_signals(top_n: int) -> List[Dict]
    """

    def _init_hall_of_fame(self) -> None:
        """初始化 Hall of Fame 相关状态 (Host __init__ 调用)

        v2.24: 保留历史 PASSED Agent,防止 IPOP restart/数据耗尽丢失好策略
        v2.29: 多轮稳定性 — 连续2代 PASSED 才入 Hall
        v2.36: 滑动窗口 2/3 代 PASSED + WARN 观察期
        """
        # v2.24: Hall of Fame — 保留历史PASSED Agent,防止IPOP restart/数据耗尽丢失好策略
        # 来源: NSGA-II elitism + CMA-ES archive (López de Prado 2018 advances-financial-machine-learning)
        # 作用: 1)保留通过DSR/PSR门控的策略 2)最终部署候选 3)防止进化震荡丢失最优解
        self._hall_of_fame: List[Dict[str, Any]] = []
        self._hall_of_fame_max_size: int = 20  # 保留Top-20历史最佳

        # v2.29: 多轮稳定性要求 — 防止单轮过拟合进入Hall of Fame
        self._hall_of_fame_candidates: Dict[str, Dict[str, Any]] = {}  # agent_id → 上次PASSED信息
        self._hall_of_fame_min_consecutive_passes: int = 1  # Phase 14.20: 2→1 放宽入门条件  # 至少连续2代PASSED才入Hall of Fame
        # v2.29: 记录每代Sharpe变化,用于检测Round-to-Round方差(过拟合信号)
        self._agent_sharpe_history: Dict[str, List[float]] = {}  # agent_id → [sharpe_gen1, sharpe_gen2, ...]

        # v2.36: 滑动窗口PASSED记录 — 替代v2.29"连续2代PASSED"的过严要求
        self._agent_pass_history: Dict[str, List[bool]] = {}  # agent_id → [passed_gen1, passed_gen2, ...]
        self._hall_of_fame_sliding_window: int = 2  # Phase 14.21: 3→2 缩短窗口  # 滑动窗口大小=3代
        self._hall_of_fame_min_passes_in_window: int = 1  # Phase 14.21: 2→1 (Phase 14.20漏改的真正变量)

        # v2.36: LiveRealityCheck WARN观察期机制 — 替代v2.30立即降级
        self._hall_of_fame_observation: Dict[str, int] = {}  # agent_id → 连续WARN次数

    def _add_to_hall_of_fame(self, agent_id: str, sharpe: float, dsr: float, psr: float, gene_dict: dict = None) -> None:
        """v2.24: 将PASSED Agent加入Hall of Fame (v2.29: 多轮稳定性, v2.36: 滑动窗口2/3)

        v2.29修复: 不再单次PASSED就加入Hall of Fame
        v2.36修复: "连续2代PASSED"→"滑动窗口2/3代PASSED" (67%成功率,允许1次失败)
        来源: Pardo 2008 Walk-Forward Optimization + Bailey 2014 多重试验偏差校正
        """
        # v2.29: 记录每代Sharpe,用于检测Round-to-Round方差
        if agent_id not in self._agent_sharpe_history:
            self._agent_sharpe_history[agent_id] = []
        self._agent_sharpe_history[agent_id].append(float(sharpe))

        # v2.36: 记录PASSED到滑动窗口历史
        if agent_id not in self._agent_pass_history:
            self._agent_pass_history[agent_id] = []
        self._agent_pass_history[agent_id].append(True)
        # 保留最近sliding_window代
        if len(self._agent_pass_history[agent_id]) > self._hall_of_fame_sliding_window:
            self._agent_pass_history[agent_id] = self._agent_pass_history[agent_id][-self._hall_of_fame_sliding_window:]

        # v2.36: PASSED → 清除观察期标志(从WARN恢复)
        if agent_id in self._hall_of_fame_observation:
            obs_count = self._hall_of_fame_observation.pop(agent_id)
            logger.info(
                "v2.36 Hall of Fame Observation CLEARED: Agent %s 从观察期恢复 (此前%d次WARN, 本次PASSED Sharpe=%.2f)",
                agent_id, obs_count, sharpe,
            )

        # v2.29: 多轮稳定性检查 — 滑动窗口2/3代PASSED
        prev = self._hall_of_fame_candidates.get(agent_id)
        pass_history = self._agent_pass_history[agent_id]
        pass_count = sum(pass_history)

        if prev is None:
            # 首次PASSED → 进入候选列表
            self._hall_of_fame_candidates[agent_id] = {
                "agent_id": agent_id,
                "sharpe": float(sharpe),
                "dsr": float(dsr),
                "psr": float(psr),
                "generation": self.population.generation,
                "round": self._round_number,
                "consecutive_passes": 1,
                "sharpe_history": [float(sharpe)],
                "pass_history": pass_history[:],
            }
            logger.info(
                "v2.36 Hall of Fame Candidate: Agent %s PASSED Gen%d (Sharpe=%.2f) — 候选,滑动窗口%d/%d需%d次PASSED",
                agent_id, self.population.generation, sharpe,
                pass_count, len(pass_history),
                self._hall_of_fame_min_passes_in_window,
            )
            return

        # 重复PASSED → 增加连续计数
        prev["consecutive_passes"] += 1
        prev["sharpe"] = float(sharpe)  # 更新为最新Sharpe
        prev["dsr"] = float(dsr)
        prev["psr"] = float(psr)
        prev["generation"] = self.population.generation
        prev["round"] = self._round_number
        prev["sharpe_history"].append(float(sharpe))
        prev["pass_history"] = pass_history[:]

        # v2.32增强: Round-to-Round Sharpe稳定性检测 — 3维检测
        sharpe_history = prev["sharpe_history"]
        if len(sharpe_history) >= 2:
            sharpe_diff = abs(sharpe_history[-1] - sharpe_history[-2])
            if sharpe_diff > 3.0:
                logger.warning(
                    "v2.29 Hall of Fame REJECT: Agent %s Sharpe不稳定(Δ=%.2f, %s) — 疑似过拟合,从候选移除",
                    agent_id, sharpe_diff, sharpe_history[-4:],
                )
                self._hall_of_fame_candidates.pop(agent_id, None)
                return

            # v2.32新增: 符号一致性检测
            if len(sharpe_history) >= 3:
                sign_changes = 0
                for i in range(1, len(sharpe_history)):
                    prev_sign = sharpe_history[i-1] > 0
                    curr_sign = sharpe_history[i] > 0
                    if prev_sign != curr_sign:
                        sign_changes += 1
                sign_change_ratio = sign_changes / max(len(sharpe_history) - 1, 1)
                if sign_change_ratio > 0.33:
                    logger.warning(
                        "v2.32 Hall of Fame REJECT: Agent %s Sharpe符号不稳定(变化%d/%d=%.0f%%, %s) — 策略方向随机,从候选移除",
                        agent_id, sign_changes, len(sharpe_history) - 1,
                        sign_change_ratio * 100, sharpe_history[-6:],
                    )
                    self._hall_of_fame_candidates.pop(agent_id, None)
                    return

            # v2.32新增: 变异系数检测
            if len(sharpe_history) >= 3:
                mean_sharpe = float(np.mean(sharpe_history))
                std_sharpe = float(np.std(sharpe_history, ddof=1))
                if abs(mean_sharpe) > 1e-10:
                    cv = std_sharpe / abs(mean_sharpe)
                    if cv > 1.0:
                        logger.warning(
                            "v2.32 Hall of Fame REJECT: Agent %s Sharpe变异系数过大(CV=%.2f, mean=%.2f, std=%.2f, %s) — 波动过大,从候选移除",
                            agent_id, cv, mean_sharpe, std_sharpe, sharpe_history[-6:],
                        )
                        self._hall_of_fame_candidates.pop(agent_id, None)
                        return

        # v2.36: 滑动窗口2/3代PASSED → 正式加入Hall of Fame
        if (len(pass_history) >= self._hall_of_fame_min_passes_in_window
                and pass_count >= self._hall_of_fame_min_passes_in_window):
            entry = {
                "agent_id": agent_id,
                "sharpe": float(sharpe),
                "dsr": float(dsr),
                "psr": float(psr),
                "generation": self.population.generation,
                "round": self._round_number,
                "consecutive_passes": prev["consecutive_passes"],
                "sharpe_history": prev["sharpe_history"][:],
                "pass_history": pass_history[:],
                "gene_dict": gene_dict,  # Phase 14.22: 存储基因供CRASH_RESCUE重建
            }
            self._hall_of_fame.append(entry)
            # 按Sharpe降序排序，保留Top-N
            self._hall_of_fame.sort(key=lambda x: x["sharpe"], reverse=True)
            if len(self._hall_of_fame) > self._hall_of_fame_max_size:
                self._hall_of_fame = self._hall_of_fame[:self._hall_of_fame_max_size]
            logger.info(
                "v2.36 Hall of Fame PROMOTED: Agent %s 滑动窗口%d/%d代PASSED (Sharpe=%.2f, pass_history=%s, sharpe_history=%s) — 正式入Hall",
                agent_id, pass_count, len(pass_history), sharpe, pass_history, prev["sharpe_history"],
            )
            # 已正式加入,从候选列表移除
            self._hall_of_fame_candidates.pop(agent_id, None)
        else:
            # 还未达到滑动窗口要求,继续作为候选
            logger.info(
                "v2.36 Hall of Fame Candidate UPDATE: Agent %s PASSED Gen%d (Sharpe=%.2f) — 滑动窗口%d/%d,需%d次PASSED才入Hall",
                agent_id, self.population.generation, sharpe,
                pass_count, len(pass_history),
                self._hall_of_fame_min_passes_in_window,
            )

    def _record_overfitted_for_hall_of_fame(self, agent_id: str, sharpe: float, reason: str = "") -> None:
        """v2.36: Agent从PASSED变OVERFITTED时,记录到滑动窗口(不立即移除)

        v2.36新逻辑: OVERFITTED → 记录False到pass_history,只在滑动窗口无望时移除
          - pass_history=[True, False] → 保留候选(下代PASSED即可2/3入Hall)
          - pass_history=[True, False, False] → 移除(1/3,无望达到2/3)
        """
        if agent_id not in self._hall_of_fame_candidates:
            return  # 不是候选,无需处理

        # v2.36: 记录OVERFITTED到滑动窗口历史
        if agent_id not in self._agent_pass_history:
            self._agent_pass_history[agent_id] = []
        self._agent_pass_history[agent_id].append(False)
        if len(self._agent_pass_history[agent_id]) > self._hall_of_fame_sliding_window:
            self._agent_pass_history[agent_id] = self._agent_pass_history[agent_id][-self._hall_of_fame_sliding_window:]

        pass_history = self._agent_pass_history[agent_id]
        pass_count = sum(pass_history)
        window_size = len(pass_history)

        # 判断是否还有希望达到min_passes_in_window
        remaining = self._hall_of_fame_sliding_window - window_size
        max_possible = pass_count + remaining

        if max_possible < self._hall_of_fame_min_passes_in_window:
            # 即使剩余全部PASSED也无法达到要求 → 移除候选
            prev = self._hall_of_fame_candidates.pop(agent_id)
            logger.warning(
                "v2.36 Hall of Fame Candidate DEMOTED: Agent %s 从候选移除 "
                "(滑动窗口%s, PASSED=%d/%d, 最大可能=%d<%d, 原因: %s, 上次Sharpe=%.2f)",
                agent_id, pass_history, pass_count, window_size,
                max_possible, self._hall_of_fame_min_passes_in_window,
                reason, prev.get("sharpe", 0.0),
            )
            # 清理观察期标志
            self._hall_of_fame_observation.pop(agent_id, None)
        else:
            # 还有希望 → 保留候选,记录OVERFITTED
            logger.info(
                "v2.36 Hall of Fame Candidate WARNING: Agent %s 本轮OVERFITTED (Sharpe=%.2f) "
                "但保留候选 (滑动窗口%s, PASSED=%d/%d, 还需%d次PASSED, 原因: %s)",
                agent_id, sharpe, pass_history, pass_count, window_size,
                self._hall_of_fame_min_passes_in_window - pass_count, reason,
            )

    def _remove_from_hall_of_fame_candidates(self, agent_id: str, reason: str = "") -> None:
        """v2.29: Agent从PASSED变OVERFITTED时,从候选列表移除

        防止过拟合策略通过候选阶段进入Hall of Fame。
        """
        if agent_id in self._hall_of_fame_candidates:
            prev = self._hall_of_fame_candidates.pop(agent_id)
            logger.warning(
                "v2.29 Hall of Fame Candidate DEMOTED: Agent %s 从候选移除 (原因: %s, 上次Sharpe=%.2f)",
                agent_id, reason, prev.get("sharpe", 0.0),
            )


    def _fast_track_to_hall_of_fame(self, agent_id: str, sharpe: float, sharpe_raw: float,
                                     dsr: float, psr: float, gene_dict: dict = None,
                                     wf_ratio: float = None) -> bool:
        """Phase 14.22: 高分快速通道 — 绕过v2.4 Gate直接入Hall

        根因: v2.4 Gate要求DSR>=0.70+PSR>=0.85, 但突破策略(Sharpe=5.0, win_rate=89%)
              因交易笔数不足(<50)被判定OVERFITTED, 永远进不了HOF
              导致CRASH_RESCUE无策略可注, SCORE_CRASHED后无法恢复

        修复: Sharpe>=3.0的策略直接入HOF, 不需要v2.4 Gate PASSED
        来源: CMA-ES restart机制 — 高适应度个体直接存档, 不需要额外验证
        """
        try:
            _fast_sharpe = float(sharpe_raw) if sharpe_raw is not None else float(sharpe)
            if _fast_sharpe < 3.0:
                return False  # 不满足快速通道条件

            # Phase 14.24: Walk-forward过拟合检查 — wf_ratio<0.7的过拟合策略拒绝入HOF
            # 根因: fast-track绕过v2.4 Gate, 让过拟合策略(Sharpe高但wf_ratio低)入HOF
            #   导致CRASH_RESCUE注入过拟合策略, 新数据段崩溃(SCORE_CRASHED循环)
            # 修复: wf_ratio<0.7拒绝入HOF, >=0.7才允许(与v2.4 Gate wf_overfit_threshold一致)
            if wf_ratio is None or not np.isfinite(wf_ratio) or wf_ratio < 0.7:
                logger.info(
                    "Phase 14.26 HOF FAST-TRACK REJECTED: Agent %s Sharpe_raw=%.2f wf_ratio=%s (Walk-Forward未验证或过拟合, 拒绝入Hall)",
                    agent_id, _fast_sharpe, "missing" if wf_ratio is None else f"{wf_ratio:.2f}",
                )
                return False

            # 检查是否已在HOF中
            for existing in self._hall_of_fame:
                if existing.get('agent_id') == agent_id:
                    return False  # 已存在

            _fast_entry = {
                "agent_id": agent_id,
                "sharpe": float(sharpe),
                "sharpe_raw": float(sharpe_raw) if sharpe_raw is not None else float(sharpe),
                "dsr": float(dsr) if dsr is not None else 0.0,
                "psr": float(psr) if psr is not None else 0.0,
                "generation": self.population.generation,
                "round": getattr(self, '_round_number', 0),
                "consecutive_passes": 1,
                "sharpe_history": [_fast_sharpe],
                "pass_history": [True],
                "fast_track": True,
                "gene_dict": gene_dict,
            }
            self._hall_of_fame.append(_fast_entry)
            self._hall_of_fame.sort(key=lambda x: x.get("sharpe_raw", x.get("sharpe", 0)), reverse=True)
            if len(self._hall_of_fame) > self._hall_of_fame_max_size:
                self._hall_of_fame = self._hall_of_fame[:self._hall_of_fame_max_size]
            logger.warning(
                "Phase 14.22 HOF FAST-TRACK: Agent %s Sharpe_raw=%.2f → 直接入Hall (绕过v2.4 Gate, gene=%s)",
                agent_id, _fast_sharpe, 'Y' if gene_dict else 'N',
            )
            return True
        except Exception as _e:
            logger.warning("Phase 14.22 HOF fast-track异常(非致命): %s", _e)
            return False

    def get_hall_of_fame(self) -> List[Dict[str, Any]]:
        """v2.24: 获取Hall of Fame（历史PASSED Agent）

        返回通过DSR/PSR门控的Top-N策略，按Sharpe降序。
        用于最终部署候选选择。
        """
        return list(self._hall_of_fame)

    def get_elite_signals(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取精英信号"""
        return self.population.get_elite_signals(top_n)
